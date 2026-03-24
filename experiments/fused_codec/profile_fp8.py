"""
Profile FP8 weight distributions for palette-based compression.

For each weight matrix, compute:
- p15: mass covered by top 15 FP8 byte values (4-bit with escape)
- p31: mass covered by top 31 FP8 byte values (5-bit with escape)
- Expected bits per weight for 4-bit and 5-bit palette encoding
- Comparison with i.i.d. entropy

This determines whether fused decode+GEMM with palette encoding is viable.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM
from collections import Counter


def profile_fp8_matrix(bf16_tensor: torch.Tensor) -> dict:
    """Profile a single weight matrix cast to FP8."""
    fp8 = bf16_tensor.to(torch.float8_e4m3fn)
    raw = fp8.view(torch.uint8).flatten().numpy()
    n = len(raw)

    counts = Counter(raw.tolist())
    sorted_counts = sorted(counts.values(), reverse=True)
    total = sum(sorted_counts)
    n_unique = len(sorted_counts)

    # Cumulative coverage for top-k values
    cum = np.cumsum(sorted_counts)
    p15 = cum[min(14, len(cum)-1)] / total if len(cum) >= 1 else 0
    p31 = cum[min(30, len(cum)-1)] / total if len(cum) >= 1 else 0
    p63 = cum[min(62, len(cum)-1)] / total if len(cum) >= 1 else 0
    p7 = cum[min(6, len(cum)-1)] / total if len(cum) >= 1 else 0

    # Expected bits per weight for palette encoding
    # 4-bit (15+escape): common values get 4 bits, escapes get 4+8=12 bits
    escape_rate_4 = 1 - p15
    bpw_4bit = 4 + 8 * escape_rate_4  # 4 bits base + 8 bits for each escape

    # 5-bit (31+escape): common get 5 bits, escapes get 5+8=13 bits
    escape_rate_5 = 1 - p31
    bpw_5bit = 5 + 8 * escape_rate_5

    # 3-bit (7+escape): common get 3 bits, escapes get 3+8=11 bits
    escape_rate_3 = 1 - p7
    bpw_3bit = 3 + 8 * escape_rate_3

    # 6-bit (63+escape): common get 6 bits, escapes get 6+8=14 bits
    escape_rate_6 = 1 - p63
    bpw_6bit = 6 + 8 * escape_rate_6

    # Shannon entropy (theoretical limit)
    probs = np.array(sorted_counts) / total
    entropy = -np.sum(probs * np.log2(probs))

    # Compression ratios (relative to 8-bit FP8)
    return {
        'n_params': n,
        'n_unique': n_unique,
        'p7': p7,
        'p15': p15,
        'p31': p31,
        'p63': p63,
        'bpw_3bit': bpw_3bit,
        'bpw_4bit': bpw_4bit,
        'bpw_5bit': bpw_5bit,
        'bpw_6bit': bpw_6bit,
        'entropy': entropy,
        'ratio_3bit': bpw_3bit / 8 * 100,
        'ratio_4bit': bpw_4bit / 8 * 100,
        'ratio_5bit': bpw_5bit / 8 * 100,
        'ratio_6bit': bpw_6bit / 8 * 100,
        'ratio_entropy': entropy / 8 * 100,
    }


def profile_model(model_name: str):
    """Profile all weight matrices in a model."""
    print(f"\n{'='*110}")
    print(f"FP8 Palette Profile: {model_name}")
    print(f"{'='*110}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    print(f"\n{'Layer':<50} {'Params':>10} {'Uniq':>5} {'p7':>6} {'p15':>6} {'p31':>6} "
          f"{'3bit%':>6} {'4bit%':>6} {'5bit%':>6} {'H%':>6} {'Best':>5}")
    print("-" * 110)

    total_params = 0
    weighted_ratios = {k: 0.0 for k in ['3bit', '4bit', '5bit', '6bit', 'entropy']}

    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16 or param.numel() < 1024:
            continue

        stats = profile_fp8_matrix(param.data)
        total_params += stats['n_params']

        for k in weighted_ratios:
            weighted_ratios[k] += stats[f'ratio_{k}'] * stats['n_params']

        # Find best palette size
        best = min(
            ('3bit', stats['ratio_3bit']),
            ('4bit', stats['ratio_4bit']),
            ('5bit', stats['ratio_5bit']),
            ('6bit', stats['ratio_6bit']),
            key=lambda x: x[1]
        )

        if param.numel() > 500_000:
            print(f"  {name:<48} {stats['n_params']:>10,} {stats['n_unique']:>5} "
                  f"{stats['p7']:>5.1%} {stats['p15']:>5.1%} {stats['p31']:>5.1%} "
                  f"{stats['ratio_3bit']:>5.1f}% {stats['ratio_4bit']:>5.1f}% "
                  f"{stats['ratio_5bit']:>5.1f}% {stats['ratio_entropy']:>5.1f}% "
                  f"{best[0]:>5}")

    print(f"\n{'='*110}")
    print(f"MODEL-WIDE AGGREGATE ({total_params:,} params)")
    print(f"{'='*110}")
    for k in ['3bit', '4bit', '5bit', '6bit', 'entropy']:
        ratio = weighted_ratios[k] / total_params
        savings = 100 - ratio
        bpw = ratio / 100 * 8
        print(f"  {k:>8} palette:  {ratio:.2f}%  ({savings:.1f}% savings, {bpw:.3f} bpw)")

    print(f"\n  Dense FP8:       100.00%  (0.0% savings, 8.000 bpw)")
    print(f"  Dense BF16:      200.00%  (baseline, 16.000 bpw)")
    print(f"\n  Decision rule for 4-bit palette:")
    avg_p15 = weighted_ratios['4bit'] / total_params  # proxy
    print(f"  Target: ratio < 75% → {'VIABLE' if weighted_ratios['4bit']/total_params < 75 else 'MARGINAL'}")
    print(f"  For fused GEMM speedup: need ratio < ~80% to benefit from reduced memory traffic")


if __name__ == "__main__":
    profile_model("Qwen/Qwen3-0.6B")
