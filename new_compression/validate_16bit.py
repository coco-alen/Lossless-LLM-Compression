"""
Validate 16-bit Huffman compression: verify losslessness and measure ratio.

Usage:
    python -m new_compression.validate_16bit
    python -m new_compression.validate_16bit --model_name_or_path Qwen/Qwen3-1.7B
"""

import time
from argparse import ArgumentParser

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from new_compression.codec16 import (
    compress_weight_group_16bit,
    decompress_weight_group_16bit,
    compute_compressed_size,
    compute_ratio,
    compute_dfloat11_baseline,
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


def main():
    parser = ArgumentParser("Validate 16-bit Huffman compression")
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
    total_16bit = 0
    total_df11 = 0
    all_passed = True

    print(f"\n{'Weight Type':<23} {'Params':>12} {'DF11':>7} {'16-bit':>7} {'Delta':>7} {'Lossless':>8} {'Time':>6}")
    print("-" * 80)

    for wt in WEIGHT_TYPES:
        tensors = groups[wt]
        n = sum(t.numel() for t in tensors)
        original_bytes = n * 2
        total_original += original_bytes

        # Compress with 16-bit Huffman
        t0 = time.time()
        compressed = compress_weight_group_16bit(tensors, per_layer=False)
        compress_time = time.time() - t0

        # Decompress and verify
        t0 = time.time()
        decompressed = decompress_weight_group_16bit(compressed)
        decompress_time = time.time() - t0

        # Verify losslessness
        lossless = True
        for orig, dec in zip(tensors, decompressed):
            if not torch.equal(orig, dec):
                lossless = False
                break

        if not lossless:
            all_passed = False

        # Compute ratios
        comp_size = compute_compressed_size(compressed)
        ratio_16bit = comp_size / original_bytes * 100
        ratio_df11 = compute_dfloat11_baseline(tensors)
        delta = ratio_16bit - ratio_df11

        total_16bit += comp_size
        total_df11 += int(ratio_df11 / 100 * original_bytes)

        status = "PASS" if lossless else "FAIL"
        print(f"  {wt:<21} {n:>10,} {ratio_df11:>6.2f}% {ratio_16bit:>6.2f}% {delta:>+6.2f}% "
              f"{'  '+status:>8} {compress_time+decompress_time:>5.1f}s")

    # Summary
    total_df11_ratio = total_df11 / total_original * 100
    total_16bit_ratio = total_16bit / total_original * 100
    delta_total = total_16bit_ratio - total_df11_ratio
    savings_mb = (total_df11 - total_16bit) / 1e6

    print("-" * 80)
    print(f"  {'TOTAL':<21} {total_original//2:>10,} {total_df11_ratio:>6.2f}% {total_16bit_ratio:>6.2f}% "
          f"{delta_total:>+6.2f}%  {'ALL PASS' if all_passed else 'FAIL':>8}")
    print(f"\n  Original size:  {total_original/1e6:.1f} MB")
    print(f"  DFloat11 size:  {total_df11/1e6:.1f} MB  ({total_df11_ratio:.2f}%)")
    print(f"  16-bit size:    {total_16bit/1e6:.1f} MB  ({total_16bit_ratio:.2f}%)")
    print(f"  Savings vs DF11: {savings_mb:.1f} MB  ({-delta_total:.2f}% ratio improvement)")
    print(f"\n  Lossless verification: {'PASSED' if all_passed else 'FAILED'}")


if __name__ == "__main__":
    main()
