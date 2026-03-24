"""
Validate ANS-16bit compression: verify losslessness and measure ratio.

Usage:
    python -m new_compression.validate_ans16
    python -m new_compression.validate_ans16 --model_name_or_path Qwen/Qwen3-8B
"""

import time
from argparse import ArgumentParser

import torch
import numpy as np
from dahuffman import HuffmanCodec
from transformers import AutoModelForCausalLM, AutoConfig

from new_compression.codec_ans16 import (
    compress_weight_group_ans16,
    decompress_weight_group_ans16,
    compute_compressed_size,
    compute_ratio,
)

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


def dfloat11_size(tensors):
    """Compute DFloat11 compressed size."""
    all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in tensors])
    n = all_w.numel()
    exp = ((all_w >> 7) & 0xFF).to(torch.uint8).numpy()
    freq = {int(v): int(c) for v, c in zip(*np.unique(exp, return_counts=True))}
    codec = HuffmanCodec.from_frequencies(freq)
    table = codec.get_code_table()
    huf_bits = sum(code_len * freq.get(sym, 0) for sym, (code_len, _) in table.items() if sym in freq)
    return (huf_bits + 7) // 8 + n


def main():
    parser = ArgumentParser("Validate ANS-16bit compression")
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
    groups = extract_weight_groups(model, num_layers)
    del model

    total_original = 0
    total_df11 = 0
    total_ans16 = 0
    all_passed = True

    print(f"\n{'Weight Type':<23} {'Params':>12} {'DF11':>7} {'ANS16':>7} {'Delta':>7} {'Lossless':>8} {'Enc':>5} {'Dec':>5}")
    print("-" * 90)

    for wt in WEIGHT_TYPES:
        tensors = groups[wt]
        n = sum(t.numel() for t in tensors)
        original_bytes = n * 2
        total_original += original_bytes

        # Compress
        t0 = time.time()
        compressed = compress_weight_group_ans16(tensors)
        enc_time = time.time() - t0

        # Decompress
        t0 = time.time()
        decompressed = decompress_weight_group_ans16(compressed)
        dec_time = time.time() - t0

        # Verify losslessness
        lossless = True
        for orig, dec in zip(tensors, decompressed):
            if not torch.equal(orig, dec):
                lossless = False
                # Find first mismatch
                diff = (orig.view(torch.int16) != dec.view(torch.int16))
                idx = diff.nonzero()[0]
                print(f"    MISMATCH at {idx}: orig={orig.view(torch.int16).flatten()[idx[0]]}, "
                      f"dec={dec.view(torch.int16).flatten()[idx[0]]}")
                break

        if not lossless:
            all_passed = False

        # Sizes
        comp_size = compute_compressed_size(compressed)
        df11_size_bytes = dfloat11_size(tensors)
        ratio_ans = comp_size / original_bytes * 100
        ratio_df11 = df11_size_bytes / original_bytes * 100
        delta = ratio_ans - ratio_df11

        total_ans16 += comp_size
        total_df11 += df11_size_bytes

        status = "PASS" if lossless else "FAIL"
        print(f"  {wt:<21} {n:>10,} {ratio_df11:>6.2f}% {ratio_ans:>6.2f}% {delta:>+6.2f}% "
              f"  {status:>6} {enc_time:>4.1f}s {dec_time:>4.1f}s")

    # Summary
    df11_ratio = total_df11 / total_original * 100
    ans_ratio = total_ans16 / total_original * 100
    delta_total = ans_ratio - df11_ratio
    savings_mb = (total_df11 - total_ans16) / 1e6

    print("-" * 90)
    print(f"  {'TOTAL':<21} {total_original//2:>10,} {df11_ratio:>6.2f}% {ans_ratio:>6.2f}% "
          f"{delta_total:>+6.2f}%  {'ALL OK' if all_passed else 'FAIL':>6}")

    print(f"\n  Original bf16:   {total_original/1e6:>10.1f} MB")
    print(f"  DFloat11:        {total_df11/1e6:>10.1f} MB  ({df11_ratio:.2f}%)")
    print(f"  ANS-16bit:       {total_ans16/1e6:>10.1f} MB  ({ans_ratio:.2f}%)")
    print(f"  Savings vs DF11: {savings_mb:>10.1f} MB  ({-delta_total:.2f}% improvement)")
    print(f"\n  Lossless: {'PASSED' if all_passed else 'FAILED'}")

    if all_passed and delta_total < 0:
        print(f"\n  *** ANS-16bit beats DFloat11 by {-delta_total:.2f}% — LOSSLESS VERIFIED ***")


if __name__ == "__main__":
    main()
