"""
Native FP8 ablation using entropy calculations (fast, no compression needed).

Since ANS achieves within <0.1pp of entropy (proven by our casted experiments),
using Shannon entropy as the metric is valid for comparing factorization losses.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter

from safetensors import safe_open
from huggingface_hub import hf_hub_download, list_repo_files


def entropy(data):
    """Shannon entropy in bits per value."""
    _, counts = np.unique(data, return_counts=True)
    probs = counts / len(data)
    return -np.sum(probs * np.log2(probs))


def ablation_on_raw(raw_bytes, fmt='e4m3fn'):
    """Run factorization ablation on FP8 byte array using entropy."""
    n = len(raw_bytes)

    if fmt == 'e4m3fn':
        sign = (raw_bytes >> 7) & 0x01
        exponent = (raw_bytes >> 3) & 0x0F
        mantissa = raw_bytes & 0x07
        sign_mantissa = (sign << 3) | mantissa  # 4-bit
        raw_sm_bits = 4
    elif fmt == 'e5m2':
        sign = (raw_bytes >> 7) & 0x01
        exponent = (raw_bytes >> 2) & 0x1F
        mantissa = raw_bytes & 0x03
        sign_mantissa = (sign << 2) | mantissa  # 3-bit
        raw_sm_bits = 3

    # Entropies
    h_full = entropy(raw_bytes)          # H(full 8-bit value)
    h_exp = entropy(exponent)            # H(exponent)
    h_sm = entropy(sign_mantissa)        # H(sign+mantissa)

    # Method comparison (all in bits-per-value):
    # 1. Full-value ANS: H(X) bpv
    # 2. Separated ANS: H(exp) + H(s+m) bpv
    # 3. Exp ANS + raw s+m: H(exp) + raw_sm_bits bpv
    # 4. ECF8-style (exp Huffman ≈ exp ANS + ~0.05pp overhead, + raw s+m)

    results = {
        'n_values': n,
        'unique_full': len(np.unique(raw_bytes)),
        'unique_exp': len(np.unique(exponent)),
        'unique_sm': len(np.unique(sign_mantissa)),
        'H_full': h_full,
        'H_exp': h_exp,
        'H_sm': h_sm,
        'H_separated': h_exp + h_sm,
        'mutual_info': (h_exp + h_sm) - h_full,
        'methods': {
            'full_value_ans': {
                'bpv': h_full,
                'ratio': h_full / 8 * 100,
                'label': 'Full-value ANS',
            },
            'separated_ans': {
                'bpv': h_exp + h_sm,
                'ratio': (h_exp + h_sm) / 8 * 100,
                'label': 'Separated ANS (exp + s+m)',
            },
            'exp_ans_raw_sm': {
                'bpv': h_exp + raw_sm_bits,
                'ratio': (h_exp + raw_sm_bits) / 8 * 100,
                'label': 'Exp ANS + raw s+m',
            },
            'ecf8_style': {
                'bpv': h_exp + 0.05 + raw_sm_bits,  # Huffman ~0.05 overhead vs ANS
                'ratio': (h_exp + 0.05 + raw_sm_bits) / 8 * 100,
                'label': 'Exp Huffman + raw s+m (ECF8)',
            },
        },
        'exp_distribution': {},
    }

    # Exponent distribution for histogram
    exp_vals, exp_counts = np.unique(exponent, return_counts=True)
    for v, c in zip(exp_vals, exp_counts):
        results['exp_distribution'][int(v)] = int(c)

    return results


def run_model(model_name, fmt='e4m3fn'):
    """Run ablation on all FP8 tensors from a model."""
    print(f"\n{'='*80}")
    print(f"Native FP8 Ablation: {model_name}")
    print(f"{'='*80}")

    files = list_repo_files(model_name)
    st_files = [f for f in files if f.endswith('.safetensors')]

    all_fp8 = []
    per_tensor_type = {}  # group by tensor type (q_proj, k_proj, etc.)

    for st_file in st_files:
        local_path = hf_hub_download(model_name, st_file)
        with safe_open(local_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if tensor.dtype != torch.float8_e4m3fn or tensor.numel() < 1024:
                    continue

                raw = tensor.view(torch.uint8).numpy().flatten().astype(np.int32)
                all_fp8.append(raw)

                # Determine tensor type
                for ttype in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'embed_tokens', 'lm_head']:
                    if ttype in key:
                        if ttype not in per_tensor_type:
                            per_tensor_type[ttype] = []
                        per_tensor_type[ttype].append(raw)
                        break

    if not all_fp8:
        print("  No FP8 tensors!")
        return None

    concatenated = np.concatenate(all_fp8)
    print(f"  Total FP8 values: {len(concatenated):,}")

    # Overall ablation
    overall = ablation_on_raw(concatenated, fmt)

    print(f"\n  {'Method':<40s} {'bpv':>8s} {'Ratio':>8s} {'Gap':>10s}")
    print(f"  {'-'*66}")
    h = overall['H_full']
    for method_key in ['full_value_ans', 'separated_ans', 'exp_ans_raw_sm', 'ecf8_style']:
        m = overall['methods'][method_key]
        gap = m['bpv'] - h
        print(f"  {m['label']:<40s} {m['bpv']:8.3f} {m['ratio']:7.2f}% {gap:+9.3f}")

    print(f"\n  H(full)={h:.3f}, H(exp)={overall['H_exp']:.3f}, H(s+m)={overall['H_sm']:.3f}")
    print(f"  Mutual info I(exp;s+m) = {overall['mutual_info']:.3f} bits")
    print(f"  Unique: {overall['unique_full']}/256 full, {overall['unique_exp']}/16 exp, {overall['unique_sm']}/16 s+m")

    # Per tensor type analysis
    print(f"\n  Per-tensor-type entropy:")
    per_type_results = {}
    for ttype, arrays in sorted(per_tensor_type.items()):
        cat = np.concatenate(arrays)
        r = ablation_on_raw(cat, fmt)
        per_type_results[ttype] = r
        print(f"    {ttype:15s}: H(full)={r['H_full']:.3f}  sep={r['H_separated']:.3f}  "
              f"gap={r['mutual_info']:.3f}  unique={r['unique_full']}")

    return {
        'model': model_name,
        'overall': overall,
        'per_type': {k: {kk: vv for kk, vv in v.items() if kk != 'exp_distribution'} for k, v in per_type_results.items()},
    }


def main():
    results = {}

    models = [
        ("Qwen/Qwen3-0.6B-FP8", 'e4m3fn'),
        ("Qwen/Qwen3-4B-Instruct-2507-FP8", 'e4m3fn'),
        ("RedHatAI/Llama-3.2-1B-Instruct-FP8", 'e4m3fn'),
    ]

    for model_name, fmt in models:
        try:
            r = run_model(model_name, fmt)
            if r:
                results[model_name] = r
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    # Compare with casted results (from prior experiments)
    print(f"\n\n{'='*80}")
    print(f"COMPARISON: Native FP8 vs Casted FP8 Factorization Penalty")
    print(f"{'='*80}")
    print(f"\n  {'Model':<35s} {'Type':<10s} {'H(full)':>8s} {'H(sep)':>8s} {'MI':>6s} {'Sep penalty':>12s}")
    print(f"  {'-'*80}")

    for model_name, r in results.items():
        o = r['overall']
        penalty = o['H_separated'] - o['H_full']
        print(f"  {model_name:<35s} {'native':<10s} {o['H_full']:8.3f} {o['H_separated']:8.3f} "
              f"{o['mutual_info']:6.3f} {penalty:+11.3f}")

    # Known casted results for comparison
    print(f"  {'Qwen3-0.6B (casted)':<35s} {'casted':<10s} {'5.653':>8s} {'5.699':>8s} "
          f"{'0.047':>6s} {'+0.047':>12s}")

    # Save
    output_path = Path(__file__).parent / "native_fp8_ablation_results.json"

    def clean(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Size):
            return list(obj)
        return obj

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=clean)
    print(f"\n  Results saved to {output_path}")


if __name__ == '__main__':
    main()
