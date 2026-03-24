#!/usr/bin/env python3
"""
Pilot experiment: Entropy analysis of quantized (INT4 / FP8) LLM weights.

Key questions:
- INT4: Do values cluster around a few integers (e.g., 7,8) or are they uniform?
  If non-uniform, entropy coding on top of INT4 could save 15-30%.
- FP8: How compressible are FP8 weights compared to BF16?

Approach:
1. Load a real GPTQ INT4 model (Qwen2.5-7B-Instruct-GPTQ-Int4) safetensors
   and extract packed INT4 values.
2. Also simulate INT4 quantization from a BF16 model for comparison.
3. Cast BF16 weights to FP8 (float8_e4m3fn) and analyze entropy.
"""

import os
import sys
import time
import numpy as np
import torch
from collections import Counter
from pathlib import Path
import math
import json

# ─── Utility functions ────────────────────────────────────────────────────────

def shannon_entropy(counts: np.ndarray) -> float:
    """Compute Shannon entropy in bits from a count array."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def analyze_int4_values(values: np.ndarray, name: str = ""):
    """Analyze INT4 values (0-15 range). Returns dict of stats."""
    counts = np.bincount(values.astype(np.int64), minlength=16)[:16]
    total = counts.sum()
    entropy = shannon_entropy(counts)
    max_entropy = 4.0  # log2(16)
    compression_ratio = entropy / max_entropy

    # Find top values
    probs = counts / total
    sorted_idx = np.argsort(-counts)

    result = {
        'name': name,
        'num_values': total,
        'entropy_bits': entropy,
        'max_entropy': max_entropy,
        'compression_ratio': compression_ratio,
        'savings_pct': (1 - compression_ratio) * 100,
        'top_values': [(int(sorted_idx[i]), float(probs[sorted_idx[i]])) for i in range(min(5, len(sorted_idx)))],
        'counts': counts,
    }
    return result

def analyze_uint8_values(values: np.ndarray, name: str = "", max_bits: float = 8.0):
    """Analyze uint8 values (0-255 range). Returns dict of stats."""
    counts = np.bincount(values.astype(np.int64), minlength=256)[:256]
    total = counts.sum()
    entropy = shannon_entropy(counts)
    num_unique = np.sum(counts > 0)
    compression_ratio = entropy / max_bits

    result = {
        'name': name,
        'num_values': total,
        'num_unique': int(num_unique),
        'entropy_bits': entropy,
        'max_entropy': max_bits,
        'compression_ratio': compression_ratio,
        'savings_pct': (1 - compression_ratio) * 100,
    }
    return result

def print_int4_analysis(stats: dict):
    """Pretty-print INT4 analysis results."""
    print(f"  {stats['name']}")
    print(f"    Values: {stats['num_values']:,}")
    print(f"    Entropy: {stats['entropy_bits']:.4f} bits (max 4.0)")
    print(f"    Compression ratio: {stats['compression_ratio']:.4f} ({stats['savings_pct']:.2f}% savings)")
    print(f"    Top values: {', '.join(f'{v}({p:.3f})' for v,p in stats['top_values'][:5])}")

def print_uint8_analysis(stats: dict):
    """Pretty-print uint8 analysis results."""
    print(f"  {stats['name']}")
    print(f"    Values: {stats['num_values']:,}  Unique: {stats['num_unique']}")
    print(f"    Entropy: {stats['entropy_bits']:.4f} bits (max {stats['max_entropy']:.1f})")
    print(f"    Compression ratio: {stats['compression_ratio']:.4f} ({stats['savings_pct']:.2f}% savings)")


# ─── Part 1: Real GPTQ INT4 model ────────────────────────────────────────────

def analyze_gptq_model():
    """Download and analyze a real GPTQ INT4 model's weight tensors."""
    print("=" * 80)
    print("PART 1: Real GPTQ INT4 Model (Qwen2.5-7B-Instruct-GPTQ-Int4)")
    print("=" * 80)

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    model_id = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"

    # Download the index to find which files contain which tensors
    try:
        index_path = hf_hub_download(model_id, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        print(f"Model has {len(index['weight_map'])} tensors across files")
    except Exception as e:
        print(f"Could not download index: {e}")
        return None

    # Download first shard
    try:
        shard_path = hf_hub_download(model_id, "model-00001-of-00002.safetensors")
        print(f"Downloaded shard: {shard_path}")
    except Exception as e:
        print(f"Could not download shard: {e}")
        return None

    # Analyze GPTQ packed tensors
    # GPTQ stores INT4 weights packed into int32 (8 values per int32)
    # The tensor names typically end with ".qweight"
    all_int4_stats = []
    all_int4_values_global = []

    # Also track qzeros (zero-points) and scales
    qzeros_info = []
    scales_info = []

    with safe_open(shard_path, framework="numpy") as f:
        tensor_names = f.keys()
        print(f"\nTensors in shard 1: {len(list(tensor_names))}")

        # Categorize tensors
        qweight_names = [n for n in f.keys() if 'qweight' in n]
        qzeros_names = [n for n in f.keys() if 'qzeros' in n]
        scales_names = [n for n in f.keys() if 'scales' in n]
        other_names = [n for n in f.keys() if not any(k in n for k in ['qweight', 'qzeros', 'scales', 'g_idx'])]

        print(f"  qweight tensors: {len(qweight_names)}")
        print(f"  qzeros tensors: {len(qzeros_names)}")
        print(f"  scales tensors: {len(scales_names)}")
        print(f"  other tensors: {len(other_names)}")

        # Show first few tensor names and shapes
        print(f"\nSample tensor names and shapes:")
        for name in list(f.keys())[:15]:
            tensor = f.get_tensor(name)
            print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")

        # Unpack INT4 from qweight tensors
        print(f"\n--- INT4 Weight Analysis (from qweight tensors) ---")
        for name in qweight_names[:30]:  # Analyze first 30 layers
            tensor = f.get_tensor(name)  # int32 packed
            # GPTQ packs 8 INT4 values per int32
            # Each 4-bit nibble: (val >> (i*4)) & 0xF
            flat = tensor.flatten().astype(np.int64)
            int4_values = []
            for shift in range(8):
                nibbles = (flat >> (shift * 4)) & 0xF
                int4_values.append(nibbles)
            int4_values = np.concatenate(int4_values)

            stats = analyze_int4_values(int4_values, name)
            all_int4_stats.append(stats)
            all_int4_values_global.append(int4_values)

            # Print for first few
            if len(all_int4_stats) <= 5:
                print_int4_analysis(stats)

        # Analyze zero-points (qzeros) - also packed INT4
        if qzeros_names:
            print(f"\n--- Zero-point Analysis (qzeros) ---")
            for name in qzeros_names[:3]:
                tensor = f.get_tensor(name)
                flat = tensor.flatten().astype(np.int64)
                int4_values = []
                for shift in range(8):
                    nibbles = (flat >> (shift * 4)) & 0xF
                    int4_values.append(nibbles)
                int4_values = np.concatenate(int4_values)
                stats = analyze_int4_values(int4_values, name)
                print_int4_analysis(stats)

        # Analyze scales (usually float16)
        if scales_names:
            print(f"\n--- Scales Analysis ---")
            for name in scales_names[:3]:
                tensor = f.get_tensor(name)
                print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")
                # Analyze as raw bytes for entropy
                raw = tensor.view(np.uint8 if tensor.dtype == np.uint8 else np.uint16)
                if tensor.dtype == np.float16:
                    raw = tensor.view(np.uint16).flatten()
                    counts = np.bincount(raw.astype(np.int64), minlength=65536)
                    ent = shannon_entropy(counts)
                    unique = np.sum(counts > 0)
                    print(f"    Unique uint16 values: {unique}, entropy: {ent:.4f} bits/16 = {ent/16:.4f} ratio")

    # Aggregate INT4 statistics
    if all_int4_values_global:
        print(f"\n{'='*60}")
        print(f"AGGREGATE INT4 STATISTICS ({len(all_int4_stats)} tensors)")
        print(f"{'='*60}")

        all_values = np.concatenate(all_int4_values_global)
        agg_stats = analyze_int4_values(all_values, "ALL qweight tensors combined")
        print_int4_analysis(agg_stats)

        # Distribution histogram
        counts = np.bincount(all_values.astype(np.int64), minlength=16)[:16]
        total = counts.sum()
        print(f"\n  Full INT4 distribution:")
        for i in range(16):
            bar = '#' * int(counts[i] / total * 200)
            print(f"    {i:2d}: {counts[i]/total:.6f} ({counts[i]:>12,}) {bar}")

        # Per-tensor entropy statistics
        entropies = [s['entropy_bits'] for s in all_int4_stats]
        savings = [s['savings_pct'] for s in all_int4_stats]
        print(f"\n  Per-tensor entropy: min={min(entropies):.4f}, max={max(entropies):.4f}, "
              f"mean={np.mean(entropies):.4f}, std={np.std(entropies):.4f}")
        print(f"  Per-tensor savings: min={min(savings):.2f}%, max={max(savings):.2f}%, "
              f"mean={np.mean(savings):.2f}%")

        # Estimate actual compression
        total_int4_bits = len(all_values) * 4
        entropy_bits = len(all_values) * agg_stats['entropy_bits']
        overhead_estimate = len(all_int4_stats) * 16 * 4 * 8  # per-tensor frequency tables
        print(f"\n  Original INT4 size: {total_int4_bits / 8 / 1024 / 1024:.2f} MB")
        print(f"  Entropy-optimal size: {entropy_bits / 8 / 1024 / 1024:.2f} MB")
        print(f"  Potential savings: {(total_int4_bits - entropy_bits) / 8 / 1024 / 1024:.2f} MB")

        return agg_stats

    return None


# ─── Part 2: Simulated INT4 quantization ─────────────────────────────────────

def simulate_int4_quantization():
    """Load a BF16 model, simulate INT4 quantization, analyze entropy."""
    print("\n" + "=" * 80)
    print("PART 2: Simulated INT4 Quantization (Qwen3-0.6B BF16 -> INT4)")
    print("=" * 80)

    from transformers import AutoModelForCausalLM, AutoConfig

    model_name = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_name}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    all_int4_stats = []
    all_int4_values = []

    print(f"\n--- Symmetric Per-Channel INT4 Quantization ---")

    for name, param in model.named_parameters():
        if param.ndim < 2:  # Skip biases, layernorms
            continue

        weight = param.data.float()  # to float32 for quantization

        # Symmetric per-channel quantization to INT4 (range -8 to 7)
        # scale = max(|w|) / 7 per output channel
        abs_max = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
        scale = abs_max / 7.0
        quantized = torch.clamp(torch.round(weight / scale), -8, 7).to(torch.int8)

        # Map to unsigned 0-15 range for analysis
        unsigned = (quantized + 8).numpy().flatten().astype(np.uint8)

        stats = analyze_int4_values(unsigned, name)
        all_int4_stats.append(stats)
        all_int4_values.append(unsigned)

        if len(all_int4_stats) <= 3:
            print_int4_analysis(stats)

    # Aggregate
    print(f"\n{'='*60}")
    print(f"AGGREGATE Simulated INT4 ({len(all_int4_stats)} tensors)")
    print(f"{'='*60}")

    all_values = np.concatenate(all_int4_values)
    agg = analyze_int4_values(all_values, "ALL simulated INT4 combined")
    print_int4_analysis(agg)

    counts = np.bincount(all_values.astype(np.int64), minlength=16)[:16]
    total = counts.sum()
    print(f"\n  Full distribution:")
    for i in range(16):
        bar = '#' * int(counts[i] / total * 200)
        print(f"    {i:2d}: {counts[i]/total:.6f} ({counts[i]:>12,}) {bar}")

    entropies = [s['entropy_bits'] for s in all_int4_stats]
    print(f"\n  Per-tensor entropy: min={min(entropies):.4f}, max={max(entropies):.4f}, "
          f"mean={np.mean(entropies):.4f}")

    del model
    return agg


# ─── Part 3: FP8 analysis ────────────────────────────────────────────────────

def analyze_fp8():
    """Cast BF16 weights to FP8 and analyze entropy."""
    print("\n" + "=" * 80)
    print("PART 3: FP8 (float8_e4m3fn) Entropy Analysis (Qwen3-0.6B)")
    print("=" * 80)

    from transformers import AutoModelForCausalLM

    model_name = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_name}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    all_fp8_stats = []
    all_fp8_values = []
    all_bf16_stats = []

    print(f"\n--- BF16 -> FP8 (e4m3fn) Conversion & Entropy ---")

    for name, param in model.named_parameters():
        if param.ndim < 2:
            continue

        bf16_data = param.data

        # BF16 entropy (full 16-bit)
        bf16_uint16 = bf16_data.view(torch.int16).numpy().astype(np.uint16).flatten()
        bf16_counts = np.bincount(bf16_uint16.astype(np.int64), minlength=65536)
        bf16_ent = shannon_entropy(bf16_counts)
        bf16_unique = np.sum(bf16_counts > 0)

        # BF16 exponent entropy (bits 7-14, 8 bits)
        bf16_exp = ((bf16_uint16 >> 7) & 0xFF)
        exp_counts = np.bincount(bf16_exp.astype(np.int64), minlength=256)
        exp_ent = shannon_entropy(exp_counts)

        # FP8 e4m3fn (4-bit exponent, 3-bit mantissa, 1 sign)
        fp8_data = bf16_data.float().to(torch.float8_e4m3fn)
        fp8_uint8 = fp8_data.view(torch.uint8).numpy().flatten()

        stats = analyze_uint8_values(fp8_uint8, name, max_bits=8.0)
        stats['bf16_entropy'] = bf16_ent
        stats['bf16_unique'] = bf16_unique
        stats['bf16_exp_entropy'] = exp_ent
        all_fp8_stats.append(stats)
        all_fp8_values.append(fp8_uint8)

        bf16_stats = {
            'name': name,
            'entropy_16bit': bf16_ent,
            'unique_16bit': bf16_unique,
            'exp_entropy': exp_ent,
        }
        all_bf16_stats.append(bf16_stats)

        if len(all_fp8_stats) <= 3:
            print_uint8_analysis(stats)
            print(f"    BF16 full entropy: {bf16_ent:.4f}/16 bits, exponent: {exp_ent:.4f}/8 bits")

    # Aggregate FP8
    print(f"\n{'='*60}")
    print(f"AGGREGATE FP8 STATISTICS ({len(all_fp8_stats)} tensors)")
    print(f"{'='*60}")

    all_fp8 = np.concatenate(all_fp8_values)
    agg = analyze_uint8_values(all_fp8, "ALL FP8 combined", max_bits=8.0)
    print_uint8_analysis(agg)

    # FP8 byte distribution (top 20)
    counts = np.bincount(all_fp8.astype(np.int64), minlength=256)
    total = counts.sum()
    sorted_idx = np.argsort(-counts)
    print(f"\n  Top 20 FP8 byte values:")
    for rank, idx in enumerate(sorted_idx[:20]):
        print(f"    0x{idx:02X} ({idx:3d}): {counts[idx]/total:.6f} ({counts[idx]:>12,})")

    # Aggregate BF16 comparison
    bf16_ents = [s['bf16_entropy'] for s in all_fp8_stats]
    exp_ents = [s['bf16_exp_entropy'] for s in all_fp8_stats]
    fp8_ents = [s['entropy_bits'] for s in all_fp8_stats]

    print(f"\n  Comparison (per-tensor mean):")
    print(f"    BF16 full 16-bit entropy:  {np.mean(bf16_ents):.4f} bits (ratio: {np.mean(bf16_ents)/16:.4f})")
    print(f"    BF16 exponent 8-bit ent:   {np.mean(exp_ents):.4f} bits (ratio: {np.mean(exp_ents)/8:.4f})")
    print(f"    FP8 e4m3fn 8-bit entropy:  {np.mean(fp8_ents):.4f} bits (ratio: {np.mean(fp8_ents)/8:.4f})")

    # FP8 sub-field analysis: sign (1 bit), exponent (4 bits), mantissa (3 bits)
    print(f"\n--- FP8 Sub-field Analysis ---")
    fp8_sign = (all_fp8 >> 7) & 0x1
    fp8_exp = (all_fp8 >> 3) & 0xF  # 4-bit exponent
    fp8_mant = all_fp8 & 0x7  # 3-bit mantissa

    sign_counts = np.bincount(fp8_sign, minlength=2)
    sign_ent = shannon_entropy(sign_counts)

    exp_counts = np.bincount(fp8_exp, minlength=16)
    exp_ent = shannon_entropy(exp_counts)

    mant_counts = np.bincount(fp8_mant, minlength=8)
    mant_ent = shannon_entropy(mant_counts)

    print(f"  Sign (1 bit):     entropy={sign_ent:.4f} (max 1.0)")
    print(f"  Exponent (4 bit): entropy={exp_ent:.4f} (max 4.0), ratio={exp_ent/4:.4f}")
    print(f"  Mantissa (3 bit): entropy={mant_ent:.4f} (max 3.0), ratio={mant_ent/3:.4f}")
    print(f"  Total sub-field:  {sign_ent + exp_ent + mant_ent:.4f} bits")
    print(f"  Joint (full byte): {agg['entropy_bits']:.4f} bits")
    print(f"  Correlation gain: {sign_ent + exp_ent + mant_ent - agg['entropy_bits']:.4f} bits")

    # FP8 exponent distribution
    print(f"\n  FP8 exponent distribution:")
    for i in range(16):
        bar = '#' * int(exp_counts[i] / total * 300)
        print(f"    {i:2d}: {exp_counts[i]/total:.6f} {bar}")

    del model
    return agg


# ─── Part 4: Practical size estimates ────────────────────────────────────────

def practical_estimates():
    """Compute practical compression estimates for a 7B model."""
    print("\n" + "=" * 80)
    print("PART 4: Practical Size Estimates for 7B Model")
    print("=" * 80)

    params_7b = 7_000_000_000

    print(f"\nModel: ~7B parameters")
    print(f"\n{'Format':<30} {'Size (GB)':<12} {'Notes'}")
    print("-" * 70)

    # BF16
    bf16_gb = params_7b * 2 / 1e9
    print(f"{'BF16 (raw)':<30} {bf16_gb:<12.2f} baseline")

    # DFloat11 (~69% of BF16, weight-only)
    df11_gb = bf16_gb * 0.69
    print(f"{'DFloat11':<30} {df11_gb:<12.2f} ~69% of BF16")

    # INT4 (raw)
    int4_gb = params_7b * 0.5 / 1e9  # 4 bits = 0.5 bytes
    print(f"{'INT4 GPTQ (raw)':<30} {int4_gb:<12.2f} 4 bits/weight")

    # INT4 + entropy coding (estimated from analysis)
    # Will be filled with actual numbers
    print(f"{'INT4 + entropy (est. 85%)':<30} {int4_gb * 0.85:<12.2f} entropy-coded INT4")
    print(f"{'INT4 + entropy (est. 75%)':<30} {int4_gb * 0.75:<12.2f} entropy-coded INT4")

    # FP8 (raw)
    fp8_gb = params_7b * 1 / 1e9
    print(f"{'FP8 (raw)':<30} {fp8_gb:<12.2f} 8 bits/weight")

    # FP8 + entropy coding
    print(f"{'FP8 + entropy (est. 80%)':<30} {fp8_gb * 0.80:<12.2f} entropy-coded FP8")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Pilot Experiment: Entropy of Quantized LLM Weights")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Part 1: Real GPTQ INT4 model
    gptq_stats = analyze_gptq_model()

    # Part 2: Simulated INT4
    sim_stats = simulate_int4_quantization()

    # Part 3: FP8 analysis
    fp8_stats = analyze_fp8()

    # Part 4: Practical estimates
    practical_estimates()

    # ─── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if gptq_stats:
        print(f"\n  Real GPTQ INT4:")
        print(f"    Entropy: {gptq_stats['entropy_bits']:.4f} / 4.0 bits")
        print(f"    Compression ratio: {gptq_stats['compression_ratio']:.4f}")
        print(f"    Additional savings: {gptq_stats['savings_pct']:.2f}%")

    if sim_stats:
        print(f"\n  Simulated INT4:")
        print(f"    Entropy: {sim_stats['entropy_bits']:.4f} / 4.0 bits")
        print(f"    Compression ratio: {sim_stats['compression_ratio']:.4f}")
        print(f"    Additional savings: {sim_stats['savings_pct']:.2f}%")

    if fp8_stats:
        print(f"\n  FP8 (e4m3fn):")
        print(f"    Entropy: {fp8_stats['entropy_bits']:.4f} / 8.0 bits")
        print(f"    Compression ratio: {fp8_stats['compression_ratio']:.4f}")
        print(f"    Additional savings: {fp8_stats['savings_pct']:.2f}%")

    print(f"\n  Verdict:")
    if gptq_stats and gptq_stats['savings_pct'] > 10:
        print(f"    INT4: PROMISING - {gptq_stats['savings_pct']:.1f}% additional compression possible!")
    elif gptq_stats:
        print(f"    INT4: MARGINAL - only {gptq_stats['savings_pct']:.1f}% additional savings")

    if fp8_stats and fp8_stats['savings_pct'] > 15:
        print(f"    FP8: PROMISING - {fp8_stats['savings_pct']:.1f}% additional compression possible!")
    elif fp8_stats:
        print(f"    FP8: MARGINAL - only {fp8_stats['savings_pct']:.1f}% additional savings")


if __name__ == "__main__":
    main()
