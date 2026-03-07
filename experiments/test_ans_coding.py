"""
Test ANS (Asymmetric Numeral Systems) coding of BFloat16 weights.

ANS achieves closer to entropy than Huffman, potentially closing the
~0.24% gap between 16-bit Huffman and the entropy lower bound.
"""

import time
from argparse import ArgumentParser

import torch
import numpy as np
import constriction
from transformers import AutoModelForCausalLM, AutoConfig

WEIGHT_TYPES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def extract_weights(model, num_layers):
    groups = {wt: [] for wt in WEIGHT_TYPES}
    for idx in range(num_layers):
        layer = model.model.layers[idx]
        for wt in WEIGHT_TYPES:
            parts = wt.split(".")
            mod = layer
            for p in parts:
                mod = getattr(mod, p)
            groups[wt].append(mod.weight.data.detach().cpu().to(torch.bfloat16))
    return groups


def test_ans_16bit(tensors, label):
    """Test ANS coding of full 16-bit bf16 values."""
    all_w = torch.cat([t.flatten() for t in tensors])
    W = all_w.view(torch.int16).numpy()
    n = len(W)
    original_bytes = n * 2

    # Map int16 values to contiguous range [0, n_unique)
    unique_vals = np.unique(W)
    n_unique = len(unique_vals)
    val_to_idx = {int(v): i for i, v in enumerate(unique_vals)}
    idx_to_val = {i: int(v) for i, v in enumerate(unique_vals)}

    # Map data to indices (vectorized)
    # Create a mapping array: for each possible int16 value -> index
    # int16 range is -32768 to 32767, shift to 0-65535
    mapping = np.zeros(65536, dtype=np.int32)
    for i, v in enumerate(unique_vals):
        mapping[int(v) + 32768] = i
    data_idx = mapping[(W.astype(np.int32) + 32768)].astype(np.int32)

    # Compute probabilities
    _, counts = np.unique(data_idx, return_counts=True)
    probabilities = counts / n

    # Entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))

    # ANS encode using constriction
    probabilities_f32 = probabilities.astype(np.float32)
    model = constriction.stream.model.Categorical(probabilities_f32, perfect=False)

    # Stack-based ANS (rANS)
    t0 = time.time()
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(data_idx, model)
    compressed = encoder.get_compressed()
    encode_time = time.time() - t0

    # ANS compressed size (in 32-bit words)
    ans_bytes = len(compressed) * 4

    # Mapping table overhead (to reconstruct original int16 values)
    mapping_overhead = n_unique * 2  # store each unique int16 value

    # Probability table overhead
    prob_overhead = n_unique * 4  # float32 per symbol

    total_bytes = ans_bytes + mapping_overhead + prob_overhead

    # Decode and verify
    t0 = time.time()
    decoder = constriction.stream.stack.AnsCoder(compressed)
    decoded_idx = decoder.decode(model, n)
    decode_time = time.time() - t0

    # Verify
    lossless = np.array_equal(data_idx, decoded_idx)

    # DFloat11 baseline
    exp = ((all_w.view(torch.int16) >> 7) & 0xFF).to(torch.uint8).numpy()
    from dahuffman import HuffmanCodec
    exp_freq = {int(v): int(c) for v, c in zip(*np.unique(exp, return_counts=True))}
    codec = HuffmanCodec.from_frequencies(exp_freq)
    exp_encoded = codec.encode(exp.tolist())
    df11_bytes = len(exp_encoded) + n

    # Huffman 16-bit baseline
    freq16 = {int(v): int(c) for v, c in zip(*np.unique(W, return_counts=True))}
    huf16 = HuffmanCodec.from_frequencies(freq16)
    table16 = huf16.get_code_table()
    huf16_bits = sum(code_len * freq16.get(sym, 0) for sym, (code_len, _) in table16.items() if sym in freq16)
    huf16_bytes = (huf16_bits + 7) // 8 + n_unique * 4

    ratio_df11 = df11_bytes / original_bytes * 100
    ratio_huf16 = huf16_bytes / original_bytes * 100
    ratio_ans = total_bytes / original_bytes * 100
    ratio_entropy = (entropy * n / 8) / original_bytes * 100

    print(f"\n  {label}  ({n:,} params)")
    print(f"    DFloat11:         {ratio_df11:6.2f}%  ({df11_bytes/1e6:.2f} MB)")
    print(f"    16-bit Huffman:   {ratio_huf16:6.2f}%  ({huf16_bytes/1e6:.2f} MB)  delta={ratio_huf16-ratio_df11:+.2f}%")
    print(f"    16-bit ANS (rANS):{ratio_ans:6.2f}%  ({total_bytes/1e6:.2f} MB)  delta={ratio_ans-ratio_df11:+.2f}%  "
          f"[data={ans_bytes/1e6:.2f}  map={mapping_overhead/1e3:.1f}KB  prob={prob_overhead/1e3:.1f}KB]")
    print(f"    Entropy LB:       {ratio_entropy:6.2f}%  ({entropy:.4f} bpw)")
    print(f"    Lossless: {lossless}  Encode: {encode_time:.1f}s  Decode: {decode_time:.1f}s")
    print(f"    Unique values: {n_unique}  H={entropy:.4f}  huf_bpw={huf16_bits/n:.4f}  ans_bpw={ans_bytes*8/n:.4f}")

    return {
        "df11": df11_bytes,
        "huf16": huf16_bytes,
        "ans": total_bytes,
        "entropy_bytes": int(np.ceil(entropy * n / 8)),
        "original": original_bytes,
        "lossless": lossless,
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    num_layers = config.num_hidden_layers
    print(f"Model: {args.model_name_or_path}  ({num_layers} layers)")
    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16,
    )
    model.eval()
    print("Extracting weights...", flush=True)
    groups = extract_weights(model, num_layers)
    del model

    totals = {"df11": 0, "huf16": 0, "ans": 0, "entropy_bytes": 0, "original": 0}
    all_lossless = True

    for wt in WEIGHT_TYPES:
        r = test_ans_16bit(groups[wt], wt)
        for k in totals:
            totals[k] += r[k]
        if not r["lossless"]:
            all_lossless = False

    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    for name, key in [("DFloat11", "df11"), ("16-bit Huffman", "huf16"),
                       ("16-bit ANS", "ans"), ("Entropy LB", "entropy_bytes")]:
        ratio = totals[key] / totals["original"] * 100
        delta = ratio - totals["df11"] / totals["original"] * 100
        size_mb = totals[key] / 1e6
        marker = " ***BEATS DF11***" if delta < 0 else ""
        print(f"  {name:<20} {ratio:6.2f}%  ({size_mb:.1f} MB)  {delta:+.2f}%{marker}")

    print(f"\n  All lossless: {all_lossless}")


if __name__ == "__main__":
    main()
