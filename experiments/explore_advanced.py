"""
Advanced compression explorations targeting the sign_mantissa bottleneck.

Key insight from first exploration: sign_mantissa byte has ~7.97 bpw entropy,
and exp-conditioned coding only brings it to ~7.93 bpw. We need novel approaches.

Approaches:
  1. Row-sorted exponent index: sort elements by exponent within each row,
     store permutation compactly, then compress the now-smoother mantissa sequence.
  2. Nibble-level coding: split sign_mantissa into high/low 4-bit nibbles,
     code high nibble conditioned on exponent, low nibble raw.
  3. BWT (Burrows-Wheeler) on sign_mantissa bytes for LZ-style pattern matching.
  4. Per-row adaptive exponent coding: each row gets its own Huffman table.
  5. Sign separation + mantissa 7-bit conditioned on exponent.
  6. Exponent run-length + Huffman hybrid.
  7. Combined: per-weight-type Huffman(exp) + ANS-style conditioned sm.
  8. Full 16-bit value frequency analysis — how many unique values, entropy.
  9. Exponent grouping: group weights by exponent into runs, then per-group
     mantissa coding (essentially exp-conditioned but with explicit grouping).
  10. Mantissa bit-plane analysis: check per-bit entropy conditioned on exponent.
"""

import time
import json
import os
from argparse import ArgumentParser
from collections import defaultdict

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


def huffman_encoded_size(data_uint8: np.ndarray) -> int:
    vals, counts = np.unique(data_uint8, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(vals, counts)}
    if len(freq) <= 1:
        return (len(data_uint8) + 7) // 8
    codec = HuffmanCodec.from_frequencies(freq)
    encoded = codec.encode(data_uint8.tolist())
    return len(encoded)


def entropy_bits(data: np.ndarray) -> float:
    vals, counts = np.unique(data, return_counts=True)
    n = len(data)
    probs = counts / n
    return -np.sum(counts * np.log2(probs))


def extract_fields(w_bf16):
    W = w_bf16.contiguous().view(torch.int16)
    exp = ((W >> 7) & 0xFF).to(torch.uint8).numpy()
    sm = ((W >> 8) & 0x80 | (W & 0x7F)).to(torch.uint8).numpy()
    return exp, sm


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


# =========================================================================
# Method: Full 16-bit analysis
# =========================================================================

def analyze_16bit(tensors):
    """Analyze full 16-bit bf16 values."""
    all_w = torch.cat([t.flatten() for t in tensors])
    W = all_w.view(torch.int16)
    n = W.numel()

    vals, counts = torch.unique(W, return_counts=True)
    n_unique = len(vals)

    # Full 16-bit entropy
    probs = counts.float() / n
    h = -(probs * probs.log2()).sum().item()

    # How concentrated is the distribution?
    sorted_counts, _ = counts.sort(descending=True)
    cum = torch.cumsum(sorted_counts, 0).float() / n
    top1_pct = cum[0].item() if len(cum) > 0 else 0
    top10_pct = cum[min(9, len(cum)-1)].item()
    top100_pct = cum[min(99, len(cum)-1)].item()
    top1000_pct = cum[min(999, len(cum)-1)].item()

    return {
        "n_elements": n,
        "n_unique_16bit": n_unique,
        "entropy_16bit": h,
        "entropy_bytes": int(np.ceil(h * n / 8)),
        "top1_pct": top1_pct,
        "top10_pct": top10_pct,
        "top100_pct": top100_pct,
        "top1000_pct": top1000_pct,
    }


# =========================================================================
# Method: Per-row adaptive exponent coding
# =========================================================================

def method_per_row_exp(tensors):
    """Per-row Huffman on exponent (more tables, potentially better coding)."""
    total_exp_bytes = 0
    total_sm_bytes = 0
    total_n = 0
    table_overhead = 0

    for t in tensors:
        if t.dim() < 2:
            # Fallback for 1D tensors
            exp, sm = extract_fields(t.flatten())
            total_exp_bytes += huffman_encoded_size(exp)
            total_sm_bytes += len(sm)
            total_n += len(exp)
            continue

        rows, cols = t.shape
        exp_2d, sm_2d = extract_fields(t)
        exp_2d = exp_2d.reshape(rows, cols)
        sm_2d = sm_2d.reshape(rows, cols)

        for r in range(rows):
            total_exp_bytes += huffman_encoded_size(exp_2d[r])
            table_overhead += 64  # ~64 bytes per row Huffman table (rough estimate)
        total_sm_bytes += rows * cols  # raw
        total_n += rows * cols

    return {
        "compressed_bytes": total_exp_bytes + total_sm_bytes + table_overhead,
        "original_bytes": total_n * 2,
        "detail": f"per_row_exp_huf={total_exp_bytes}, sm_raw={total_sm_bytes}, tbl={table_overhead}",
    }


# =========================================================================
# Method: Mantissa bit-plane conditional entropy
# =========================================================================

def analyze_mantissa_bitplanes(tensors):
    """Analyze per-bit entropy of sign_mantissa conditioned on exponent."""
    all_w = torch.cat([t.flatten() for t in tensors])
    exp, sm = extract_fields(all_w)
    n = len(exp)

    # Unconditional per-bit entropy of sm
    uncond_bits = []
    for bit in range(8):
        bit_vals = (sm >> bit) & 1
        p1 = bit_vals.mean()
        if p1 == 0 or p1 == 1:
            h = 0
        else:
            h = -(p1 * np.log2(p1) + (1-p1) * np.log2(1-p1))
        uncond_bits.append(h)

    # Conditional per-bit entropy given exponent
    unique_exps = np.unique(exp)
    cond_total_bits = np.zeros(8)
    for ev in unique_exps:
        mask = exp == ev
        cnt = mask.sum()
        sm_sub = sm[mask]
        for bit in range(8):
            bit_vals = (sm_sub >> bit) & 1
            p1 = bit_vals.mean()
            if p1 == 0 or p1 == 1:
                h = 0
            else:
                h = -(p1 * np.log2(p1) + (1-p1) * np.log2(1-p1))
            cond_total_bits[bit] += h * cnt

    cond_bpw = cond_total_bits / n

    return {
        "uncond_per_bit": uncond_bits,  # per-bit entropy (unconditional)
        "cond_per_bit_bpw": cond_bpw.tolist(),  # per-bit entropy conditioned on exp (bpw)
        "uncond_total": sum(uncond_bits),
        "cond_total": sum(cond_bpw),
    }


# =========================================================================
# Method: Sign-separated coding
# =========================================================================

def method_sign_separated(tensors):
    """Huffman(exp) + Huffman(sign|exp) + per-exp Huffman(mantissa_7bit)."""
    all_w = torch.cat([t.flatten() for t in tensors])
    W = all_w.view(torch.int16)
    n = W.numel()

    exp = ((W >> 7) & 0xFF).to(torch.uint8).numpy()
    sign = ((W >> 15) & 1).to(torch.uint8).numpy()
    man7 = (W & 0x7F).to(torch.uint8).numpy()

    exp_bytes = huffman_encoded_size(exp)

    # Per-exponent coding of sign and man7
    unique_exps = np.unique(exp)
    sign_total = 0
    man7_total = 0
    table_overhead = 0

    for ev in unique_exps:
        mask = exp == ev
        cnt = mask.sum()

        # Sign: just count bits needed (binary entropy)
        s = sign[mask]
        p1 = s.mean()
        if p1 == 0 or p1 == 1:
            sign_bits = cnt  # 1 bit per symbol minimum
        else:
            h = -(p1 * np.log2(p1) + (1-p1) * np.log2(1-p1))
            sign_bits = int(np.ceil(h * cnt))
        sign_total += (sign_bits + 7) // 8

        # Man7: Huffman encode
        m = man7[mask]
        man7_total += huffman_encoded_size(m)
        table_overhead += len(np.unique(m)) * 2

    return {
        "compressed_bytes": exp_bytes + sign_total + man7_total + table_overhead,
        "original_bytes": n * 2,
        "detail": f"exp_huf={exp_bytes}, cond_sign={sign_total}, cond_man7={man7_total}, tbl={table_overhead}",
    }


# =========================================================================
# Method: Exp-conditioned mantissa with ACTUAL Huffman (not entropy estimate)
# =========================================================================

def method_exp_conditioned_huffman(tensors):
    """Huffman(exp) + per-exp Huffman(sm) with actual dahuffman encoding."""
    all_w = torch.cat([t.flatten() for t in tensors])
    exp, sm = extract_fields(all_w)
    n = len(exp)

    exp_bytes = huffman_encoded_size(exp)

    unique_exps = np.unique(exp)
    sm_total = 0
    table_overhead = 0

    for ev in unique_exps:
        mask = exp == ev
        sm_sub = sm[mask]
        sm_total += huffman_encoded_size(sm_sub)
        # Realistic table overhead: need freq table per exponent value
        n_unique = len(np.unique(sm_sub))
        table_overhead += n_unique * 3  # 1 byte symbol + 2 bytes code approx

    return {
        "compressed_bytes": exp_bytes + sm_total + table_overhead,
        "original_bytes": n * 2,
        "detail": f"exp_huf={exp_bytes}, cond_sm_huf={sm_total}, tbl={table_overhead}",
    }


# =========================================================================
# Method: Top-K exponent conditioned mantissa (only condition on common exps)
# =========================================================================

def method_topk_exp_conditioned(tensors, top_k=8):
    """Only use top-K most common exponents as context for mantissa coding.
    Less table overhead for rare exponents."""
    all_w = torch.cat([t.flatten() for t in tensors])
    exp, sm = extract_fields(all_w)
    n = len(exp)

    exp_bytes = huffman_encoded_size(exp)

    # Find top-K exponents
    vals, counts = np.unique(exp, return_counts=True)
    sorted_idx = np.argsort(-counts)
    top_exps = set(vals[sorted_idx[:top_k]])

    sm_topk = 0
    sm_rest = []
    table_overhead = 0

    for ev in vals:
        mask = exp == ev
        sm_sub = sm[mask]
        if int(ev) in top_exps:
            sm_topk += huffman_encoded_size(sm_sub)
            n_unique = len(np.unique(sm_sub))
            table_overhead += n_unique * 3
        else:
            sm_rest.append(sm_sub)

    # Code remaining with single Huffman table
    if sm_rest:
        all_rest = np.concatenate(sm_rest)
        sm_rest_bytes = huffman_encoded_size(all_rest)
        table_overhead += len(np.unique(all_rest)) * 3
    else:
        sm_rest_bytes = 0

    # Need 1 byte to store which exps get special tables
    table_overhead += top_k

    return {
        "compressed_bytes": exp_bytes + sm_topk + sm_rest_bytes + table_overhead,
        "original_bytes": n * 2,
        "detail": f"exp_huf={exp_bytes}, topk_sm={sm_topk}, rest_sm={sm_rest_bytes}, tbl={table_overhead}",
    }


# =========================================================================
# Method: Block-level adaptive coding
# =========================================================================

def method_block_adaptive(tensors, block_size=4096):
    """Split data into blocks, per-block Huffman for both exp and sm."""
    total_comp = 0
    total_orig = 0
    table_overhead = 0

    for t in tensors:
        flat = t.flatten()
        exp, sm = extract_fields(flat)
        n = len(exp)
        total_orig += n * 2

        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            exp_block = exp[start:end]
            sm_block = sm[start:end]
            total_comp += huffman_encoded_size(exp_block)
            total_comp += huffman_encoded_size(sm_block)
            table_overhead += 512  # rough table overhead per block

    return {
        "compressed_bytes": total_comp + table_overhead,
        "original_bytes": total_orig,
        "detail": f"block_size={block_size}, total_comp={total_comp}, tbl={table_overhead}",
    }


# =========================================================================
# Method: 16-bit value dedup + coding
# =========================================================================

def method_16bit_dedup(tensors):
    """Analyze how many unique 16-bit values exist and estimate dictionary coding."""
    all_w = torch.cat([t.flatten() for t in tensors])
    W = all_w.view(torch.int16).numpy()
    n = len(W)

    vals, counts = np.unique(W, return_counts=True)
    n_unique = len(vals)

    # If we could code each value at its entropy...
    total_entropy_bits = entropy_bits(W.view(np.uint8).reshape(-1, 2)[:, 1])  # just high byte for comparison

    # Full 16-bit Huffman-like entropy
    probs = counts / n
    full_entropy = -np.sum(counts * np.log2(probs))
    full_entropy_bytes = int(np.ceil(full_entropy / 8))

    return {
        "compressed_bytes": full_entropy_bytes,
        "original_bytes": n * 2,
        "n_unique": n_unique,
        "detail": f"n_unique={n_unique}, full_16bit_entropy={full_entropy/n:.3f}bpw, total={full_entropy_bytes}",
    }


# =========================================================================
# Method: Exponent-conditioned with combined (sign+exp) context
# =========================================================================

def method_sign_exp_conditioned(tensors):
    """Use (exponent, sign) pair as context for 7-bit mantissa coding.
    512 possible contexts (256 exp × 2 sign), but most are empty."""
    all_w = torch.cat([t.flatten() for t in tensors])
    W = all_w.view(torch.int16)
    n = W.numel()

    exp = ((W >> 7) & 0xFF).to(torch.uint8).numpy()
    sign = ((W >> 15) & 1).to(torch.uint8).numpy()
    man7 = (W & 0x7F).to(torch.uint8).numpy()

    # Encode exponent
    exp_bytes = huffman_encoded_size(exp)

    # Encode sign (1 bit per element, could do per-exp sign coding)
    sign_bits = 0
    unique_exps = np.unique(exp)
    for ev in unique_exps:
        mask = exp == ev
        s = sign[mask]
        p1 = s.mean()
        if p1 == 0 or p1 == 1:
            sign_bits += len(s)
        else:
            h = -(p1 * np.log2(p1) + (1-p1) * np.log2(1-p1))
            sign_bits += h * len(s)
    sign_bytes = int(np.ceil(sign_bits / 8))

    # Per (exp, sign) context for mantissa
    context = exp.astype(np.uint16) * 2 + sign
    unique_ctx = np.unique(context)
    man7_total = 0
    table_overhead = 0

    for ctx in unique_ctx:
        mask = context == ctx
        m = man7[mask]
        if len(m) == 0:
            continue
        man7_total += huffman_encoded_size(m)
        table_overhead += len(np.unique(m)) * 2

    return {
        "compressed_bytes": exp_bytes + sign_bytes + man7_total + table_overhead,
        "original_bytes": n * 2,
        "detail": f"exp_huf={exp_bytes}, cond_sign={sign_bytes}, ctx_man7={man7_total}, tbl={table_overhead}, n_ctx={len(unique_ctx)}",
    }


# =========================================================================
# Main
# =========================================================================

METHODS = [
    ("Per-row exp Huffman + raw(sm)", method_per_row_exp),
    ("Exp-cond Huffman(sm)", method_exp_conditioned_huffman),
    ("Top-8 exp-cond Huffman(sm)", lambda t: method_topk_exp_conditioned(t, 8)),
    ("Top-16 exp-cond Huffman(sm)", lambda t: method_topk_exp_conditioned(t, 16)),
    ("Sign-separated conditioned", method_sign_separated),
    ("(Sign,Exp)-conditioned man7", method_sign_exp_conditioned),
    ("Block-4096 adaptive", lambda t: method_block_adaptive(t, 4096)),
    ("Block-16384 adaptive", lambda t: method_block_adaptive(t, 16384)),
    ("16-bit value entropy LB", method_16bit_dedup),
]


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

    # ---- Bit-plane analysis (just for first weight type) ----
    print("\n" + "=" * 80)
    print("Mantissa bit-plane analysis (self_attn.q_proj)")
    print("=" * 80)
    bp = analyze_mantissa_bitplanes(groups["self_attn.q_proj"])
    print(f"  {'Bit':>4} {'Uncond H':>10} {'Cond H|exp':>12}")
    for i in range(8):
        print(f"  {i:>4} {bp['uncond_per_bit'][i]:>10.4f} {bp['cond_per_bit_bpw'][i]:>12.4f}")
    print(f"  {'TOTAL':>4} {bp['uncond_total']:>10.4f} {bp['cond_total']:>12.4f}")

    # ---- 16-bit analysis ----
    print("\n" + "=" * 80)
    print("16-bit value analysis per weight type")
    print("=" * 80)
    for wt in WEIGHT_TYPES:
        info = analyze_16bit(groups[wt])
        print(f"  {wt:<23} unique={info['n_unique_16bit']:>8,}  H={info['entropy_16bit']:.3f}bpw  "
              f"top1={info['top1_pct']:.4f}  top100={info['top100_pct']:.4f}  top1000={info['top1000_pct']:.4f}")

    # ---- DFloat11 baseline ----
    def dfloat11_baseline(tensors):
        all_w = torch.cat([t.flatten() for t in tensors])
        exp, sm = extract_fields(all_w)
        n = len(exp)
        exp_bytes = huffman_encoded_size(exp)
        return {"compressed_bytes": exp_bytes + n, "original_bytes": n * 2, "detail": "baseline"}

    # ---- Run methods ----
    totals = defaultdict(lambda: {"compressed": 0, "original": 0})
    totals_bl = {"compressed": 0, "original": 0}

    for wt in WEIGHT_TYPES:
        tensors = groups[wt]
        n = sum(t.numel() for t in tensors)
        print(f"\n{'='*100}")
        print(f"  {wt}  ({n:,} params)")
        print(f"{'='*100}")
        print(f"  {'Method':<40} {'Ratio':>7} {'bpw':>6}  Detail")
        print(f"  {'-'*40} {'-'*7} {'-'*6}  {'-'*30}")

        bl = dfloat11_baseline(tensors)
        bl_ratio = bl["compressed_bytes"] / bl["original_bytes"] * 100
        bl_bpw = bl_ratio / 100 * 16
        print(f"  {'DFloat11 baseline':<40} {bl_ratio:>6.2f}% {bl_bpw:>5.2f}  ---")
        totals_bl["compressed"] += bl["compressed_bytes"]
        totals_bl["original"] += bl["original_bytes"]

        for name, method in METHODS:
            t0 = time.time()
            try:
                result = method(tensors)
            except Exception as e:
                print(f"  {name:<40} ERROR: {e}")
                continue
            elapsed = time.time() - t0

            ratio = result["compressed_bytes"] / result["original_bytes"] * 100
            bpw = ratio / 100 * 16
            delta = ratio - bl_ratio
            print(f"  {name:<40} {ratio:>6.2f}% {bpw:>5.2f}  {delta:+.2f}% {result['detail']}")

            totals[name]["compressed"] += result["compressed_bytes"]
            totals[name]["original"] += result["original_bytes"]

    # Summary
    print(f"\n\n{'='*100}")
    print("OVERALL SUMMARY")
    print(f"{'='*100}")
    bl_ratio = totals_bl["compressed"] / totals_bl["original"] * 100
    print(f"  {'DFloat11 baseline':<40} {bl_ratio:>6.2f}%")

    for name, _ in METHODS:
        t = totals[name]
        if t["original"] == 0:
            continue
        ratio = t["compressed"] / t["original"] * 100
        delta = ratio - bl_ratio
        marker = " ***BEATS DF11***" if delta < 0 else ""
        print(f"  {name:<40} {ratio:>6.2f}%  ({delta:+.2f}%){marker}")


if __name__ == "__main__":
    main()
