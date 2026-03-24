"""
Fast validation of 16-bit Huffman: verify losslessness on small samples,
compute ratio from code table analysis (no full encoding for speed).

Usage:
    python -m new_compression.validate_16bit_fast
    python -m new_compression.validate_16bit_fast --model_name_or_path Qwen/Qwen3-1.7B
"""

import time
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


def extract_weight_groups(model, num_layers):
    groups = {wt: [] for wt in WEIGHT_TYPES}
    for idx in range(num_layers):
        layer = model.model.layers[idx]
        for wt in WEIGHT_TYPES:
            parts = wt.split(".")
            mod = layer
            for p in parts:
                mod = getattr(mod, p)
            assert mod.weight.dtype == torch.bfloat16
            groups[wt].append(mod.weight.data.detach().cpu())
    return groups


def huffman_size_from_freq(freq: dict) -> tuple:
    """Return (encoded_bytes, huffman_bpw, n_unique) from freq table."""
    total = sum(freq.values())
    if len(freq) <= 1:
        return (total + 7) // 8, 1.0, len(freq)
    codec = HuffmanCodec.from_frequencies(freq)
    table = codec.get_code_table()
    huf_bits = sum(code_len * freq.get(sym, 0) for sym, (code_len, _) in table.items() if sym in freq)
    return (huf_bits + 7) // 8, huf_bits / total, len(freq)


def verify_lossless_sample(tensors, n_samples=3):
    """Verify losslessness on a few sample tensors using actual encode/decode."""
    from new_compression.codec16 import compress_16bit, decompress_16bit

    for i, t in enumerate(tensors[:n_samples]):
        compressed = compress_16bit(t)
        decompressed = decompress_16bit(compressed)
        if not torch.equal(t, decompressed):
            return False, f"Mismatch at tensor {i}"
    return True, "OK"


def main():
    parser = ArgumentParser("Fast validate 16-bit Huffman compression")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--skip_verify", action="store_true", help="Skip encode/decode verification")
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
    groups = extract_weight_groups(model, num_layers)
    del model

    total_original = 0
    total_df11 = 0
    total_16bit = 0

    print(f"\n{'Weight Type':<23} {'Params':>12} {'DF11':>7} {'16bit':>7} {'Delta':>7} {'Unique':>7} {'Verify':>8}")
    print("-" * 80)

    for wt in WEIGHT_TYPES:
        tensors = groups[wt]
        n = sum(t.numel() for t in tensors)
        original_bytes = n * 2
        total_original += original_bytes

        # Extract all weights for this type
        all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in tensors])
        W = all_w.numpy()

        # DFloat11 baseline
        exp = ((all_w >> 7) & 0xFF).to(torch.uint8).numpy()
        exp_freq = {int(v): int(c) for v, c in zip(*np.unique(exp, return_counts=True))}
        exp_enc_bytes, _, _ = huffman_size_from_freq(exp_freq)
        df11_bytes = exp_enc_bytes + n
        total_df11 += df11_bytes

        # 16-bit Huffman
        freq16 = {int(v): int(c) for v, c in zip(*np.unique(W, return_counts=True))}
        enc16_bytes, huf16_bpw, n_unique = huffman_size_from_freq(freq16)
        table_overhead = n_unique * 4
        total_16bit_bytes = enc16_bytes + table_overhead
        total_16bit += total_16bit_bytes

        ratio_df11 = df11_bytes / original_bytes * 100
        ratio_16bit = total_16bit_bytes / original_bytes * 100
        delta = ratio_16bit - ratio_df11

        # Verify losslessness on first layer only (fast)
        if not args.skip_verify:
            # Verify on single small tensor
            t_small = tensors[0]
            t0 = time.time()
            verified, msg = verify_lossless_sample([t_small], n_samples=1)
            verify_time = time.time() - t0
            status = f"{'PASS' if verified else 'FAIL'} {verify_time:.0f}s"
        else:
            status = "skip"

        print(f"  {wt:<21} {n:>10,} {ratio_df11:>6.2f}% {ratio_16bit:>6.2f}% {delta:>+6.2f}% {n_unique:>6} {status:>8}")

    # Summary
    total_df11_ratio = total_df11 / total_original * 100
    total_16bit_ratio = total_16bit / total_original * 100
    delta_total = total_16bit_ratio - total_df11_ratio
    savings_mb = (total_df11 - total_16bit) / 1e6

    print("-" * 80)
    print(f"  {'TOTAL':<21} {total_original//2:>10,} {total_df11_ratio:>6.2f}% {total_16bit_ratio:>6.2f}% "
          f"{delta_total:>+6.2f}%")
    print(f"\n  Original size:  {total_original/1e6:.1f} MB")
    print(f"  DFloat11 size:  {total_df11/1e6:.1f} MB  ({total_df11_ratio:.2f}%)")
    print(f"  16-bit size:    {total_16bit/1e6:.1f} MB  ({total_16bit_ratio:.2f}%)")
    print(f"  Savings vs DF11: {savings_mb:.1f} MB  ({-delta_total:.2f}% ratio improvement)")


if __name__ == "__main__":
    main()
