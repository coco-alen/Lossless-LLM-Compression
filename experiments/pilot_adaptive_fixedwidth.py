#!/usr/bin/env python3
"""
Pilot: Adaptive Fixed-Width Entropy Coding for LLM weights.

Analyzes per-layer exponent distributions to determine if adaptive
code widths (2-bit, 3-bit, 4-bit, 5-bit) would outperform ZipServ's
uniform 3-bit (k=7) approach.

For each width k, we find the best CONSECUTIVE window of k exponents
that covers the most values. Values inside the window cost
(code_bits + 8) bits; values outside cost 16 bits.
"""

import torch
import numpy as np
import math
from transformers import AutoModelForCausalLM
from collections import Counter

MODEL_NAME = "Qwen/Qwen3-0.6B"

# k values and their corresponding code bits
# k=3 -> 2-bit code (values 0,1,2 => ceil(log2(3))=2 bits)
# k=7 -> 3-bit code (ZipServ style)
# k=15 -> 4-bit code
# k=31 -> 5-bit code
K_CONFIGS = [
    (3, 2, "2-bit"),
    (7, 3, "3-bit"),
    (15, 4, "4-bit"),
    (31, 5, "5-bit"),
]


def extract_exponents(tensor: torch.Tensor) -> np.ndarray:
    """Extract BF16 exponents (bits 7-14) from tensor."""
    int16_view = tensor.view(torch.int16).cpu().numpy().ravel()
    exponents = (int16_view >> 7) & 0xFF
    return exponents


def compute_exponent_freq(exponents: np.ndarray):
    """Return frequency array of length 256 for exponents."""
    freq = np.zeros(256, dtype=np.int64)
    vals, counts = np.unique(exponents, return_counts=True)
    for v, c in zip(vals, counts):
        freq[v] = c
    return freq


def best_consecutive_window(freq: np.ndarray, k: int):
    """
    Find the consecutive window of k exponents with maximum coverage.
    Returns (start_exp, coverage_count).
    """
    if k >= 256:
        return 0, freq.sum()
    best_start = 0
    best_sum = freq[:k].sum()
    current_sum = best_sum
    for start in range(1, 256 - k + 1):
        current_sum = current_sum - freq[start - 1] + freq[start + k - 1]
        if current_sum > best_sum:
            best_sum = current_sum
            best_start = start
    return best_start, int(best_sum)


def compute_bpw(total_count: int, in_window: int, code_bits: int) -> float:
    """
    Compute effective bits per weight.
    - In-window values: code_bits + 7 (mantissa) + 1 (sign) = code_bits + 8
    - Out-of-window values: 16 bits (full BF16)
    - Plus we need 1 bit per value to indicate in/out (bitmap overhead)

    Actually, ZipServ uses a flag bit approach. Let's model it more carefully:
    For in-window: 1 (flag=0) + code_bits + 7 (mantissa) + 1 (sign)
    Wait - ZipServ stores base_exponent per layer (metadata, negligible).

    Simpler model matching ZipServ:
    - Each value gets a 1-bit flag: in-window or not
    - In-window: flag(1) + offset(code_bits) + sign_mantissa(8) = 1 + code_bits + 8
    - Out-of-window: flag(1) + full_bf16(16) = 17 bits

    But actually ZipServ likely uses a different scheme. Let me use:
    - In-window values: (code_bits + 8) bits  (code_bits for exponent offset, 8 for sign+mantissa)
    - Out-of-window: 16 bits raw
    - Need to store a bitmap or index to distinguish. A simple bitmap = 1 bit per weight.

    Let's compute two variants:
    1) With 1-bit flag per value (bitmap overhead)
    2) Without flag (assume separate storage like ZipServ overflow buffer with positions)

    For fairness, use the bitmap approach (1 bit per value):
    """
    out_window = total_count - in_window
    # With bitmap: 1 bit flag per value
    total_bits_bitmap = (in_window * (1 + code_bits + 8) +
                         out_window * (1 + 16))
    bpw_bitmap = total_bits_bitmap / total_count

    # Without bitmap (ZipServ-style: store overflow positions separately)
    # Overflow needs position index: ~log2(total_count) bits per overflow value
    # But for simplicity, assume overflow stored as (position, value) pairs
    # Position cost ~= log2(total_count) bits, but this gets complex.
    # Use simpler model: in-window = code_bits+8, out-window = 16, plus 1 bit flag
    return bpw_bitmap


def compute_entropy_16bit(tensor: torch.Tensor) -> float:
    """Compute i.i.d. entropy of full 16-bit BF16 values."""
    int16_view = tensor.view(torch.int16).cpu().numpy().ravel()
    total = len(int16_view)
    _, counts = np.unique(int16_view, return_counts=True)
    probs = counts / total
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def main():
    print(f"Loading {MODEL_NAME} in BF16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    print("Model loaded.\n")

    results = []
    total_params = 0

    for name, param in model.named_parameters():
        if param.numel() == 0:
            continue

        n = param.numel()
        total_params += n
        exponents = extract_exponents(param.data)
        freq = compute_exponent_freq(exponents)
        entropy_16 = compute_entropy_16bit(param.data)

        layer_result = {
            "name": name,
            "num_params": n,
            "entropy_16bit": entropy_16,
        }

        for k, code_bits, label in K_CONFIGS:
            start_exp, in_count = best_consecutive_window(freq, k)
            coverage = in_count / n
            bpw = compute_bpw(n, in_count, code_bits)
            layer_result[f"k{k}_start"] = start_exp
            layer_result[f"k{k}_coverage"] = coverage
            layer_result[f"k{k}_bpw"] = bpw

        results.append(layer_result)

    # Print per-layer table
    print("=" * 180)
    print(f"{'Layer':<55} {'Params':>10} {'Entropy':>8} | "
          f"{'k=3 cov':>8} {'k=3 bpw':>8} | "
          f"{'k=7 cov':>8} {'k=7 bpw':>8} | "
          f"{'k=15 cov':>8} {'k=15 bpw':>9} | "
          f"{'k=31 cov':>8} {'k=31 bpw':>9} | "
          f"{'Best k':>6} {'Best bpw':>9}")
    print("-" * 180)

    model_wide_bits = {k: 0 for k, _, _ in K_CONFIGS}
    model_wide_params = 0
    adaptive_bits = 0

    for r in results:
        n = r["num_params"]
        model_wide_params += n

        # Find best k for this layer
        best_k = None
        best_bpw = 17.0
        for k, code_bits, label in K_CONFIGS:
            bpw = r[f"k{k}_bpw"]
            model_wide_bits[k] += bpw * n
            if bpw < best_bpw:
                best_bpw = bpw
                best_k = k

        adaptive_bits += best_bpw * n

        # Determine best label
        best_label = {3: "k=3", 7: "k=7", 15: "k=15", 31: "k=31"}[best_k]

        print(f"{r['name']:<55} {n:>10,} {r['entropy_16bit']:>8.3f} | "
              f"{r['k3_coverage']:>8.4f} {r['k3_bpw']:>8.3f} | "
              f"{r['k7_coverage']:>8.4f} {r['k7_bpw']:>8.3f} | "
              f"{r['k15_coverage']:>8.4f} {r['k15_bpw']:>9.3f} | "
              f"{r['k31_coverage']:>8.4f} {r['k31_bpw']:>9.3f} | "
              f"{best_label:>6} {best_bpw:>9.3f}")

    # Model-wide summary
    print("\n" + "=" * 100)
    print("MODEL-WIDE SUMMARY")
    print("=" * 100)
    print(f"Total parameters: {model_wide_params:,}")
    print(f"Original size (BF16): {model_wide_params * 16 / 8 / 1024**2:.1f} MB")
    print()

    print(f"{'Method':<35} {'Avg BPW':>10} {'Ratio':>10} {'Size (MB)':>12} {'vs BF16':>10}")
    print("-" * 80)

    for k, code_bits, label in K_CONFIGS:
        avg_bpw = model_wide_bits[k] / model_wide_params
        ratio = avg_bpw / 16.0
        size_mb = model_wide_params * avg_bpw / 8 / 1024**2
        print(f"Uniform {label} (k={k})"
              f"{avg_bpw:>27.3f} {ratio:>9.2%} {size_mb:>12.1f} {'':>10}")

    adaptive_bpw = adaptive_bits / model_wide_params
    adaptive_ratio = adaptive_bpw / 16.0
    adaptive_size = model_wide_params * adaptive_bpw / 8 / 1024**2
    print(f"Adaptive (best k per layer)"
          f"{adaptive_bpw:>20.3f} {adaptive_ratio:>9.2%} {adaptive_size:>12.1f} {'':>10}")

    # ANS-16bit baseline
    ans_ratio = 0.6596
    ans_bpw = 16 * ans_ratio
    ans_size = model_wide_params * ans_bpw / 8 / 1024**2
    print(f"ANS-16bit (baseline)"
          f"{ans_bpw:>27.3f} {ans_ratio:>9.2%} {ans_size:>12.1f} {'':>10}")

    bf16_size = model_wide_params * 16 / 8 / 1024**2
    print(f"Uncompressed BF16"
          f"{16.0:>30.3f} {'100.00%':>10} {bf16_size:>12.1f} {'':>10}")

    # Adaptive k distribution
    print("\n" + "=" * 100)
    print("ADAPTIVE K DISTRIBUTION (which k is chosen per layer)")
    print("=" * 100)
    k_counts = Counter()
    k_param_counts = Counter()
    for r in results:
        best_k = None
        best_bpw = 17.0
        for k, code_bits, label in K_CONFIGS:
            bpw = r[f"k{k}_bpw"]
            if bpw < best_bpw:
                best_bpw = bpw
                best_k = k
        k_counts[best_k] += 1
        k_param_counts[best_k] += r["num_params"]

    for k, code_bits, label in K_CONFIGS:
        cnt = k_counts.get(k, 0)
        pcnt = k_param_counts.get(k, 0)
        print(f"  k={k:<3} ({label}): {cnt:>4} layers, {pcnt:>12,} params "
              f"({pcnt/model_wide_params*100:.1f}%)")

    # Extra analysis: coverage stats
    print("\n" + "=" * 100)
    print("COVERAGE STATISTICS (min/mean/max across layers)")
    print("=" * 100)
    for k, code_bits, label in K_CONFIGS:
        coverages = [r[f"k{k}_coverage"] for r in results]
        bpws = [r[f"k{k}_bpw"] for r in results]
        print(f"  k={k:<3} ({label}): coverage min={min(coverages):.4f} "
              f"mean={np.mean(coverages):.4f} max={max(coverages):.4f} | "
              f"bpw min={min(bpws):.3f} mean={np.mean(bpws):.3f} max={max(bpws):.3f}")

    # Entropy comparison
    print("\n" + "=" * 100)
    print("ENTROPY COMPARISON")
    print("=" * 100)
    entropies = [r["entropy_16bit"] for r in results]
    weighted_entropy = sum(r["entropy_16bit"] * r["num_params"] for r in results) / model_wide_params
    print(f"  Weighted avg 16-bit entropy: {weighted_entropy:.3f} bits")
    print(f"  Min/Max entropy: {min(entropies):.3f} / {max(entropies):.3f}")
    print(f"  ANS-16bit achieves: {ans_bpw:.3f} bpw (gap to entropy: {ans_bpw - weighted_entropy:.3f} bits)")


if __name__ == "__main__":
    main()
