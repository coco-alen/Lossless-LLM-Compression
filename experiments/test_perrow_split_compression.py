"""
Per-row codebook compression with separate exponent and mantissa coding.

For each row of each weight matrix:
  1. Split BF16 values into exponent byte (sign+exp) and mantissa byte
  2. Build a per-row codebook for each
  3. ANS-encode each part with the row-specific distribution

Compare against:
  - ANS-16bit global (current best: ~65.96%)
  - DFloat11 (Huffman on exponent, raw mantissa: ~66.62%)
"""

import torch
import numpy as np
import math
import time
from collections import Counter
from transformers import AutoModelForCausalLM

try:
    import constriction
    HAS_CONSTRICTION = True
except ImportError:
    HAS_CONSTRICTION = False


def entropy_bits(counts, total):
    """Total Shannon entropy in bits (not per-symbol)."""
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= c * math.log2(p)
    return h


def ans_size_bytes(values_np, counts, n):
    """Estimate ANS compressed size: entropy + small ANS overhead."""
    # ANS achieves near-entropy, ~0.01 bpw overhead for large n
    h_bits = entropy_bits(counts, n)
    # ANS state overhead: ~4 bytes per stream
    return h_bits / 8 + 4


def ans_actual_size(values_np, counts, n):
    """Actual ANS compressed size using constriction."""
    if not HAS_CONSTRICTION or n == 0:
        return ans_size_bytes(values_np, counts, n)

    symbols = sorted(counts.keys())
    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    probs = np.array([counts[s] / n for s in symbols], dtype=np.float64)
    probs = (probs / probs.sum()).astype(np.float32)
    mapped = np.array([sym_to_idx[v] for v in values_np], dtype=np.int32)

    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(mapped, constriction.stream.model.Categorical(probs, perfect=False))
    return len(encoder.get_compressed()) * 4  # bytes


def analyze_weight_tensor(name, tensor):
    """Analyze per-row split compression for one weight tensor."""
    bf16 = tensor.to(torch.bfloat16)
    if bf16.dim() == 1:
        bf16 = bf16.unsqueeze(0)  # treat 1D as single row

    rows, cols = bf16.shape
    int16 = bf16.view(torch.int16).numpy().astype(np.int32)

    # Extract exponent byte (bits 15-7: sign + 7 MSBs of exponent)
    # and mantissa byte (bits 6-0: mantissa + exponent LSB)
    hi_bytes = (int16 >> 8) & 0xFF  # sign + upper exponent
    lo_bytes = int16 & 0xFF         # lower exponent + mantissa

    n_total = rows * cols
    original_bytes = n_total * 2

    # === Method 1: ANS-16bit global (baseline) ===
    flat16 = int16.flatten()
    counts_global = Counter(flat16.tolist())
    global_data_bytes = ans_actual_size(flat16, counts_global, n_total)
    global_table_bytes = len(counts_global) * 6  # 2B symbol + 4B prob
    global_total = global_data_bytes + global_table_bytes

    # === Method 2: Per-row split (exponent + mantissa separate codebooks) ===
    perrow_total = 0
    perrow_exp_data = 0
    perrow_mant_data = 0
    perrow_exp_tables = 0
    perrow_mant_tables = 0

    for r in range(rows):
        row_hi = hi_bytes[r]
        row_lo = lo_bytes[r]

        counts_hi = Counter(row_hi.tolist())
        counts_lo = Counter(row_lo.tolist())

        # ANS compressed data
        exp_data = ans_size_bytes(row_hi, counts_hi, cols)
        mant_data = ans_size_bytes(row_lo, counts_lo, cols)

        # Codebook overhead: for each unique value, store value (1B) + probability (2B)
        # Using 2-byte probs since per-row tables are small
        exp_table = len(counts_hi) * 3  # 1B value + 2B prob
        mant_table = len(counts_lo) * 3

        perrow_exp_data += exp_data
        perrow_mant_data += mant_data
        perrow_exp_tables += exp_table
        perrow_mant_tables += mant_table

    perrow_total = perrow_exp_data + perrow_mant_data + perrow_exp_tables + perrow_mant_tables

    # === Method 3: Per-row joint 16-bit ANS ===
    perrow_joint_data = 0
    perrow_joint_tables = 0
    for r in range(rows):
        row16 = int16[r]
        counts_row = Counter(row16.tolist())
        perrow_joint_data += ans_size_bytes(row16, counts_row, cols)
        perrow_joint_tables += len(counts_row) * 6  # 2B symbol + 4B prob
    perrow_joint_total = perrow_joint_data + perrow_joint_tables

    # === Method 4: Global split (one codebook for all exp, one for all mant) ===
    flat_hi = hi_bytes.flatten()
    flat_lo = lo_bytes.flatten()
    counts_hi_global = Counter(flat_hi.tolist())
    counts_lo_global = Counter(flat_lo.tolist())

    global_split_exp_data = ans_actual_size(flat_hi, counts_hi_global, n_total)
    global_split_mant_data = ans_actual_size(flat_lo, counts_lo_global, n_total)
    global_split_exp_table = len(counts_hi_global) * 3
    global_split_mant_table = len(counts_lo_global) * 3
    global_split_total = (global_split_exp_data + global_split_mant_data +
                          global_split_exp_table + global_split_mant_table)

    # === Method 5: Per-row exp codebook + global mantissa codebook ===
    hybrid_total = 0
    hybrid_exp_data = 0
    hybrid_exp_tables = 0
    for r in range(rows):
        row_hi = hi_bytes[r]
        counts_hi = Counter(row_hi.tolist())
        hybrid_exp_data += ans_size_bytes(row_hi, counts_hi, cols)
        hybrid_exp_tables += len(counts_hi) * 3
    hybrid_mant_data = ans_actual_size(flat_lo, counts_lo_global, n_total)
    hybrid_mant_table = len(counts_lo_global) * 3
    hybrid_total = hybrid_exp_data + hybrid_exp_tables + hybrid_mant_data + hybrid_mant_table

    # === Method 6: Per-row exp with per-exp mantissa codebook (shared across rows) ===
    # Group mantissa values by their exponent byte, build one mantissa table per exp value
    perexp_mant_data = 0
    perexp_mant_tables = 0
    exp_mant_groups = {}
    for r in range(rows):
        for j in range(cols):
            exp_val = int(hi_bytes[r, j])
            mant_val = int(lo_bytes[r, j])
            if exp_val not in exp_mant_groups:
                exp_mant_groups[exp_val] = []
            exp_mant_groups[exp_val].append(mant_val)

    for exp_val, mant_vals in exp_mant_groups.items():
        mant_arr = np.array(mant_vals, dtype=np.int32)
        counts_mant = Counter(mant_vals)
        perexp_mant_data += ans_size_bytes(mant_arr, counts_mant, len(mant_vals))
        perexp_mant_tables += len(counts_mant) * 3

    # Exp part: use per-row codebooks
    perexp_exp_data = hybrid_exp_data
    perexp_exp_tables = hybrid_exp_tables
    perexp_total = perexp_exp_data + perexp_exp_tables + perexp_mant_data + perexp_mant_tables

    # Print results
    print(f"\n  {name} ({rows}×{cols} = {n_total:,} values, {original_bytes/1024:.0f} KB)")

    # Per-row stats
    avg_hi_unique = np.mean([len(Counter(hi_bytes[r].tolist())) for r in range(min(rows, 100))])
    avg_lo_unique = np.mean([len(Counter(lo_bytes[r].tolist())) for r in range(min(rows, 100))])
    avg_16_unique = np.mean([len(Counter(int16[r].tolist())) for r in range(min(rows, 100))])
    print(f"    Avg unique per row: exp={avg_hi_unique:.0f}, mant={avg_lo_unique:.0f}, joint={avg_16_unique:.0f}")
    print(f"    Global unique: exp={len(counts_hi_global)}, mant={len(counts_lo_global)}, joint={len(counts_global)}")

    methods = [
        ("ANS-16 global", global_total, global_data_bytes, global_table_bytes),
        ("Global split (exp+mant)", global_split_total, global_split_exp_data + global_split_mant_data, global_split_exp_table + global_split_mant_table),
        ("Per-row joint 16-bit", perrow_joint_total, perrow_joint_data, perrow_joint_tables),
        ("Per-row split (exp+mant)", perrow_total, perrow_exp_data + perrow_mant_data, perrow_exp_tables + perrow_mant_tables),
        ("Per-row exp + global mant", hybrid_total, hybrid_exp_data + hybrid_mant_data, hybrid_exp_tables + hybrid_mant_table),
        ("Per-row exp + per-exp mant", perexp_total, perexp_exp_data + perexp_mant_data, perexp_exp_tables + perexp_mant_tables),
    ]

    print(f"    {'Method':<30s} {'Total':>8s} {'Ratio':>7s} {'Data':>8s} {'Tables':>8s}")
    print(f"    {'-'*65}")
    for mname, total, data, tables in methods:
        ratio = total / original_bytes * 100
        print(f"    {mname:<30s} {total/1024:>7.0f}K {ratio:>6.2f}% {data/1024:>7.0f}K {tables/1024:>7.0f}K")

    return {m[0]: m[1] / original_bytes for m in methods}


def main():
    model_name = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)

    weight_types = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.down_proj.weight",
    ]

    # Test on first layer of each type
    all_results = {}
    for name, param in model.named_parameters():
        for wt in weight_types:
            if name.endswith(wt) and '.0.' in name:  # first layer
                result = analyze_weight_tensor(name, param.data)
                all_results[wt] = result
                break

    # Also test on all layers combined for one weight type
    print(f"\n{'='*80}")
    print("All layers combined (self_attn.q_proj)")
    print(f"{'='*80}")

    all_q = []
    for name, param in model.named_parameters():
        if name.endswith("self_attn.q_proj.weight"):
            all_q.append(param.data.cpu())

    combined = torch.cat(all_q, dim=0)  # Stack rows across layers
    analyze_weight_tensor("q_proj all layers", combined)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY (first layer of each type)")
    print(f"{'='*80}")
    print(f"  {'Weight type':<25s}", end="")
    for method in list(all_results.values())[0].keys():
        short = method[:20]
        print(f" {short:>12s}", end="")
    print()
    for wt, results in all_results.items():
        short_wt = wt.split('.')[-2] + '.' + wt.split('.')[-1]
        print(f"  {short_wt:<25s}", end="")
        for ratio in results.values():
            print(f" {ratio*100:>11.2f}%", end="")
        print()


if __name__ == '__main__':
    main()
