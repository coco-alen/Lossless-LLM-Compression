"""
Test full 16-bit Huffman coding of BFloat16 weights.

Key insight: BFloat16 weights in LLMs only use ~5,000-6,500 unique 16-bit values
out of 65,536 possible. This makes direct 16-bit Huffman coding feasible and
potentially better than DFloat11's byte-separated approach.

DFloat11: Huffman(exp_8bit) + raw(sm_8bit) ≈ 10.7 bpw
16-bit Huffman: Huffman(full_bf16_as_int16) ≈ 10.63+ε bpw

The Huffman overhead ε for ~5,900 symbols should be very small.
"""

import time
import struct
from argparse import ArgumentParser

import torch
import numpy as np
from dahuffman import HuffmanCodec
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


def dfloat11_size(tensors):
    all_w = torch.cat([t.flatten() for t in tensors])
    W = all_w.view(torch.int16)
    n = W.numel()
    exp = ((W >> 7) & 0xFF).to(torch.uint8).numpy()

    vals, counts = np.unique(exp, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(vals, counts)}
    codec = HuffmanCodec.from_frequencies(freq)
    encoded = codec.encode(exp.tolist())

    return len(encoded) + n, n * 2  # compressed, original


def huffman_16bit_size(tensors):
    """Full 16-bit Huffman encoding of bf16 weights."""
    all_w = torch.cat([t.flatten() for t in tensors])
    W = all_w.view(torch.int16).numpy()
    n = len(W)

    # Build frequency table for 16-bit values
    vals, counts = np.unique(W, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(vals, counts)}
    n_unique = len(freq)

    # Build Huffman codec
    codec = HuffmanCodec.from_frequencies(freq)

    # Get code table for analysis
    table = codec.get_code_table()
    max_code_len = max(l for _, (l, _) in table.items() if not isinstance(_, type(None)) and _ in freq)

    # Actually encode (this can be slow for large data)
    t0 = time.time()
    encoded = codec.encode(W.tolist())
    encode_time = time.time() - t0

    # Table overhead: need to store the mapping from int16 value -> code
    # For each unique value: 2 bytes (value) + ~2 bytes (code info) = ~4 bytes
    table_overhead = n_unique * 4

    compressed_bytes = len(encoded) + table_overhead
    original_bytes = n * 2

    # Compute entropy for comparison
    probs = counts / n
    entropy = -np.sum(probs * np.log2(probs))

    # Huffman bits per weight
    huf_bits = 0
    for sym, (code_len, _) in table.items():
        if sym in freq:
            huf_bits += code_len * freq[sym]
    huf_bpw = huf_bits / n

    return {
        "compressed_bytes": compressed_bytes,
        "encoded_bytes": len(encoded),
        "table_overhead": table_overhead,
        "original_bytes": original_bytes,
        "n_unique": n_unique,
        "max_code_len": max_code_len,
        "entropy_bpw": entropy,
        "huffman_bpw": huf_bpw,
        "encode_time": encode_time,
    }


def huffman_16bit_per_layer(tensors):
    """Per-layer 16-bit Huffman (each layer gets its own table)."""
    total_encoded = 0
    total_overhead = 0
    total_n = 0

    for t in tensors:
        W = t.contiguous().view(torch.int16).flatten().numpy()
        n = len(W)
        total_n += n

        vals, counts = np.unique(W, return_counts=True)
        freq = {int(v): int(c) for v, c in zip(vals, counts)}

        codec = HuffmanCodec.from_frequencies(freq)
        encoded = codec.encode(W.tolist())
        total_encoded += len(encoded)
        total_overhead += len(freq) * 4

    return {
        "compressed_bytes": total_encoded + total_overhead,
        "encoded_bytes": total_encoded,
        "table_overhead": total_overhead,
        "original_bytes": total_n * 2,
    }


def exp_conditioned_sm_size(tensors):
    """DFloat11+ : Huffman(exp) + per-exp Huffman(sm)."""
    all_w = torch.cat([t.flatten() for t in tensors])
    W = all_w.view(torch.int16)
    n = W.numel()

    exp = ((W >> 7) & 0xFF).to(torch.uint8).numpy()
    sm = ((W >> 8) & 0x80 | (W & 0x7F)).to(torch.uint8).numpy()

    # Encode exponent
    exp_freq = {int(v): int(c) for v, c in zip(*np.unique(exp, return_counts=True))}
    exp_codec = HuffmanCodec.from_frequencies(exp_freq)
    exp_encoded = exp_codec.encode(exp.tolist())
    exp_bytes = len(exp_encoded)

    # Per-exponent mantissa encoding
    unique_exps = np.unique(exp)
    sm_total = 0
    table_overhead = 0

    for ev in unique_exps:
        mask = exp == ev
        sm_sub = sm[mask]
        sm_freq = {int(v): int(c) for v, c in zip(*np.unique(sm_sub, return_counts=True))}
        sm_codec = HuffmanCodec.from_frequencies(sm_freq)
        sm_encoded = sm_codec.encode(sm_sub.tolist())
        sm_total += len(sm_encoded)
        table_overhead += len(sm_freq) * 3

    return {
        "compressed_bytes": exp_bytes + sm_total + table_overhead,
        "exp_bytes": exp_bytes,
        "sm_bytes": sm_total,
        "table_overhead": table_overhead,
        "original_bytes": n * 2,
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

    total_df11 = {"comp": 0, "orig": 0}
    total_16bit = {"comp": 0, "orig": 0}
    total_16bit_pl = {"comp": 0, "orig": 0}
    total_cond = {"comp": 0, "orig": 0}

    for wt in WEIGHT_TYPES:
        tensors = groups[wt]
        n = sum(t.numel() for t in tensors)
        print(f"\n{'='*100}")
        print(f"  {wt}  ({n:,} params)")
        print(f"{'='*100}")

        # DFloat11
        comp, orig = dfloat11_size(tensors)
        ratio = comp / orig * 100
        bpw = ratio / 100 * 16
        print(f"  DFloat11:           {ratio:6.2f}%  ({bpw:.3f} bpw)  comp={comp/1e6:.2f}MB")
        total_df11["comp"] += comp
        total_df11["orig"] += orig

        # 16-bit Huffman
        print(f"  Encoding 16-bit Huffman (may take a while)...", flush=True)
        r16 = huffman_16bit_size(tensors)
        ratio = r16["compressed_bytes"] / r16["original_bytes"] * 100
        bpw = ratio / 100 * 16
        print(f"  16-bit Huffman:     {ratio:6.2f}%  ({bpw:.3f} bpw)  "
              f"unique={r16['n_unique']}  maxcode={r16['max_code_len']}  "
              f"H={r16['entropy_bpw']:.3f}  huf_bpw={r16['huffman_bpw']:.3f}  "
              f"enc={r16['encoded_bytes']/1e6:.2f}MB  tbl={r16['table_overhead']/1e3:.1f}KB  "
              f"time={r16['encode_time']:.1f}s")
        total_16bit["comp"] += r16["compressed_bytes"]
        total_16bit["orig"] += r16["original_bytes"]

        # Per-layer 16-bit Huffman
        print(f"  Encoding per-layer 16-bit Huffman...", flush=True)
        r16pl = huffman_16bit_per_layer(tensors)
        ratio = r16pl["compressed_bytes"] / r16pl["original_bytes"] * 100
        bpw = ratio / 100 * 16
        print(f"  Per-layer 16-bit:   {ratio:6.2f}%  ({bpw:.3f} bpw)  "
              f"enc={r16pl['encoded_bytes']/1e6:.2f}MB  tbl={r16pl['table_overhead']/1e3:.1f}KB")
        total_16bit_pl["comp"] += r16pl["compressed_bytes"]
        total_16bit_pl["orig"] += r16pl["original_bytes"]

        # Exp-conditioned sm
        print(f"  Encoding exp-conditioned...", flush=True)
        rcond = exp_conditioned_sm_size(tensors)
        ratio = rcond["compressed_bytes"] / rcond["original_bytes"] * 100
        bpw = ratio / 100 * 16
        print(f"  Exp-cond Huf(sm):   {ratio:6.2f}%  ({bpw:.3f} bpw)  "
              f"exp={rcond['exp_bytes']/1e6:.2f}MB  sm={rcond['sm_bytes']/1e6:.2f}MB  tbl={rcond['table_overhead']/1e3:.1f}KB")
        total_cond["comp"] += rcond["compressed_bytes"]
        total_cond["orig"] += rcond["original_bytes"]

    # Overall
    print(f"\n\n{'='*100}")
    print("OVERALL SUMMARY")
    print(f"{'='*100}")

    for name, tot in [("DFloat11", total_df11), ("16-bit Huffman", total_16bit),
                       ("Per-layer 16-bit", total_16bit_pl), ("Exp-cond Huf(sm)", total_cond)]:
        ratio = tot["comp"] / tot["orig"] * 100
        bpw = ratio / 100 * 16
        delta = ratio - total_df11["comp"] / total_df11["orig"] * 100
        marker = " ***BEATS DF11***" if delta < 0 else ""
        print(f"  {name:<25} {ratio:6.2f}%  ({bpw:.3f} bpw)  {delta:+.2f}%  "
              f"size={tot['comp']/1e6:.1f}MB{marker}")


if __name__ == "__main__":
    main()
