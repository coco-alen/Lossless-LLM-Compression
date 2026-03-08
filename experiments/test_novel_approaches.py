"""
Final batch of novel compression approaches:

1. Grouped symbol coding: Group 16-bit values by top-K bits, code group ID + residual
2. Cross-type shared table: Use one ANS table for all weight types
3. Byte-interleaved ANS: Separate into high/low bytes, ANS each stream
4. Custom bit-split: Split at different bit boundaries
5. Value clustering: Cluster similar values, code cluster ID + offset
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


def iid_entropy(W):
    vals, counts = np.unique(W, return_counts=True)
    p = counts / len(W)
    return -np.sum(p * np.log2(p))


def ans_encode_safe(data, n_possible=65536, shift=32768):
    """ANS encode with safe probability handling. Returns compressed bytes."""
    n = len(data)
    if n == 0:
        return 0

    vals, counts = np.unique(data, return_counts=True)
    if len(vals) == 1:
        return 6  # just store the value

    probs = (counts / n).astype(np.float64)
    probs = np.maximum(probs, 1e-10)
    probs = (probs / probs.sum()).astype(np.float32)
    probs = np.maximum(probs, np.float32(1e-10))
    probs = probs / probs.sum()

    # Use value-based mapping (handles signed and unsigned data)
    val_min = int(vals.min())
    val_max = int(vals.max())
    mapping_size = val_max - val_min + 1
    mapping = np.zeros(mapping_size, dtype=np.int32)
    for i, v in enumerate(vals):
        mapping[int(v) - val_min] = i
    data_idx = mapping[(data.astype(np.int32) - val_min)].astype(np.int32)

    try:
        model = constriction.stream.model.Categorical(probs, perfect=False)
        encoder = constriction.stream.stack.AnsCoder()
        encoder.encode_reverse(data_idx, model)
        compressed = encoder.get_compressed()
        return len(compressed) * 4 + len(vals) * 6
    except ValueError:
        h = -np.sum(probs * np.log2(np.maximum(probs, 1e-30)))
        return int(np.ceil(h * n / 8)) + len(vals) * 6


def ans16_size(W):
    """Standard ANS-16bit size."""
    return ans_encode_safe(W, 65536, 32768)


def bit_split_ans_size(W, split_bit):
    """
    Split 16-bit values at bit position 'split_bit'.
    High part: bits [15:split_bit], Low part: bits [split_bit-1:0]

    Code high part with ANS, low part with ANS.
    """
    high = (W >> split_bit).astype(np.int16)
    low = (W & ((1 << split_bit) - 1)).astype(np.int16)

    n_high = 1 << (16 - split_bit)
    n_low = 1 << split_bit

    high_bytes = ans_encode_safe(high, n_high, n_high // 2)
    low_bytes = ans_encode_safe(low, n_low, n_low // 2)

    return high_bytes + low_bytes


def exp_mantissa_split_ans_size(W_int16):
    """
    BFloat16 natural split: sign(1) + exp(8) + mantissa(7)
    Code [sign|exp](9 bits) with ANS + mantissa(7 bits) with ANS.
    """
    sign_exp = ((W_int16 >> 7) & 0x1FF).astype(np.int16)  # 9 bits: sign + exp
    mantissa = (W_int16 & 0x7F).astype(np.int16)  # 7 bits

    se_bytes = ans_encode_safe(sign_exp, 512, 256)
    m_bytes = ans_encode_safe(mantissa, 128, 64)

    return se_bytes + m_bytes


def conditioned_split_ans_size(W_int16):
    """
    Code exponent with ANS, then per-exponent code sign+mantissa.
    Equivalent to H(exp) + H(sm|exp).
    """
    exp = ((W_int16 >> 7) & 0xFF).astype(np.uint8)
    sm = (((W_int16 >> 8) & 0x80) | (W_int16 & 0x7F)).astype(np.uint8)

    # Exponent ANS
    total = ans_encode_safe(exp, 256, 0)

    # Per-exp SM ANS
    for e in np.unique(exp):
        mask = exp == e
        sm_sub = sm[mask]
        total += ans_encode_safe(sm_sub, 256, 0)

    return total


def grouped_symbol_size(W, group_bits=12):
    """
    Group by top `group_bits` bits, code group ID + residual separately.
    """
    residual_bits = 16 - group_bits
    group_id = (W >> residual_bits).astype(np.int16)
    residual = (W & ((1 << residual_bits) - 1)).astype(np.int16)

    n_groups = 1 << group_bits
    n_residuals = 1 << residual_bits

    group_bytes = ans_encode_safe(group_id, n_groups, n_groups // 2)

    # Per-group residual coding
    residual_bytes = 0
    for g in np.unique(group_id):
        mask = group_id == g
        res_sub = residual[mask]
        residual_bytes += ans_encode_safe(res_sub, n_residuals, n_residuals // 2)

    return group_bytes + residual_bytes


def cross_type_ans_size(all_groups):
    """Combine ALL weight types and use a single ANS table."""
    all_weights = []
    for wt in WEIGHT_TYPES:
        for t in all_groups[wt]:
            all_weights.append(t.contiguous().view(torch.int16).flatten())
    W = torch.cat(all_weights).numpy()
    return ans16_size(W), len(W)


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

    total_original = 0
    method_totals = {}

    print(f"\n{'='*100}")
    print("NOVEL COMPRESSION APPROACHES")
    print(f"{'='*100}")

    for wt in WEIGHT_TYPES:
        tensors = groups[wt]
        all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in tensors])
        W = all_w.numpy()
        n = len(W)
        original = n * 2
        total_original += original

        print(f"\n  {wt}  ({n:,} params)")

        # Baseline: ANS-16bit
        t0 = time.time()
        baseline = ans16_size(W)
        t1 = time.time()
        method_totals.setdefault("ans16", 0)
        method_totals["ans16"] += baseline
        print(f"    ANS-16bit (baseline):     {baseline/original*100:.3f}%  [{t1-t0:.1f}s]")

        # Bit splits at various positions
        for split in [7, 8, 9, 10, 11]:
            t0 = time.time()
            size = bit_split_ans_size(W, split)
            t1 = time.time()
            delta = size/original*100 - baseline/original*100
            key = f"split_{split}"
            method_totals.setdefault(key, 0)
            method_totals[key] += size
            marker = " ***" if delta < -0.01 else ""
            print(f"    Bit-split at {split:>2}:          {size/original*100:.3f}%  delta={delta:+.3f}%{marker}  [{t1-t0:.1f}s]")

        # BFloat16 natural split
        t0 = time.time()
        se_m = exp_mantissa_split_ans_size(W)
        t1 = time.time()
        delta = se_m/original*100 - baseline/original*100
        method_totals.setdefault("se_m_split", 0)
        method_totals["se_m_split"] += se_m
        print(f"    Sign+Exp / Mantissa:      {se_m/original*100:.3f}%  delta={delta:+.3f}%  [{t1-t0:.1f}s]")

        # Conditioned split (exp + per-exp sm)
        t0 = time.time()
        cond = conditioned_split_ans_size(W)
        t1 = time.time()
        delta = cond/original*100 - baseline/original*100
        method_totals.setdefault("cond_split", 0)
        method_totals["cond_split"] += cond
        print(f"    Exp + per-exp SM:         {cond/original*100:.3f}%  delta={delta:+.3f}%  [{t1-t0:.1f}s]")

        # Grouped symbol coding
        for gb in [10, 11, 12, 13, 14]:
            t0 = time.time()
            size = grouped_symbol_size(W, gb)
            t1 = time.time()
            delta = size/original*100 - baseline/original*100
            key = f"grouped_{gb}"
            method_totals.setdefault(key, 0)
            method_totals[key] += size
            marker = " ***" if delta < -0.01 else ""
            print(f"    Grouped {gb}-bit:            {size/original*100:.3f}%  delta={delta:+.3f}%{marker}  [{t1-t0:.1f}s]")

    # Cross-type shared table
    t0 = time.time()
    cross_size, cross_n = cross_type_ans_size(groups)
    t1 = time.time()
    cross_original = cross_n * 2
    method_totals["cross_type"] = cross_size
    delta = cross_size/cross_original*100 - method_totals["ans16"]/total_original*100
    print(f"\n  Cross-type shared table:    {cross_size/cross_original*100:.3f}%  delta={delta:+.3f}%  [{t1-t0:.1f}s]")

    # Summary
    print(f"\n{'='*100}")
    print("OVERALL SUMMARY")
    print(f"{'='*100}")

    baseline_ratio = method_totals["ans16"] / total_original * 100
    for key in sorted(method_totals.keys()):
        if key == "cross_type":
            size = method_totals[key]
            ratio = size / total_original * 100
        else:
            size = method_totals[key]
            ratio = size / total_original * 100
        delta = ratio - baseline_ratio
        marker = " ***" if delta < -0.01 else ""
        print(f"  {key:<25} {ratio:.3f}%  delta={delta:+.3f}%{marker}")

    print(f"\n  ANS-16bit baseline:          {baseline_ratio:.3f}%  ({method_totals['ans16']/1e6:.1f}MB)")
    print(f"  Original bf16:               {total_original/1e6:.1f}MB")


if __name__ == "__main__":
    main()
