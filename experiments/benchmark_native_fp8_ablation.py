"""
Native FP8 ablation: full-value ANS vs factorized codecs on native FP8 checkpoints.

Tests the penalty of factorization on REAL FP8 checkpoints (not casted).

Methods tested:
1. Full-value ANS (treat each byte as one symbol)
2. Separated ANS (exponent + sign+mantissa, each ANS-coded independently)
3. Exponent ANS + raw sign+mantissa (ECF8-style but with ANS instead of Huffman)
4. Exponent Huffman + raw sign+mantissa (ECF8-style exact)

Also generates histogram analysis for the mechanism figure.
"""

import sys
import os
import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
import constriction
from safetensors import safe_open
from huggingface_hub import hf_hub_download, list_repo_files


def entropy(probs):
    """Shannon entropy in bits."""
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def ans_compress_size(symbols, max_symbol=255):
    """Return compressed size in bytes using full-value ANS."""
    symbols = np.ascontiguousarray(symbols.flatten(), dtype=np.int32)
    unique_vals, counts = np.unique(symbols, return_counts=True)
    n = len(symbols)
    probs = np.ascontiguousarray((counts / n).astype(np.float32))

    mapping = np.full(max_symbol + 1, -1, dtype=np.int32)
    for i, v in enumerate(unique_vals):
        mapping[v] = i

    data_idx = np.ascontiguousarray(mapping[symbols], dtype=np.int32)
    model = constriction.stream.model.Categorical(probs, perfect=False)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(data_idx, model)
    compressed = encoder.get_compressed()

    # Size: compressed words (4 bytes each) + probability table
    table_overhead = len(unique_vals) * 4  # 4 bytes per entry
    return len(compressed) * 4 + table_overhead


def huffman_compress_size(symbols):
    """Return compressed size in bytes using Huffman coding."""
    from dahuffman import HuffmanCodec
    freq = Counter(symbols.tolist())
    codec = HuffmanCodec.from_frequencies(freq)
    encoded = codec.encode(symbols.tolist())
    return len(encoded)


def ablation_on_tensor(raw_bytes: np.ndarray, fmt='e4m3fn'):
    """Run ablation on a single FP8 tensor (as uint8 array).

    Returns dict with bits-per-value for each method.
    """
    n = len(raw_bytes)

    if fmt == 'e4m3fn':
        # e4m3fn: sign(1) | exponent(4) | mantissa(3)
        sign = (raw_bytes >> 7) & 0x01
        exponent = (raw_bytes >> 3) & 0x0F  # 4-bit exponent [0, 15]
        mantissa = raw_bytes & 0x07          # 3-bit mantissa [0, 7]
        sign_mantissa = (sign << 3) | mantissa  # 4-bit [0, 15]
        exp_max = 15
        sm_max = 15
    elif fmt == 'e5m2':
        # e5m2: sign(1) | exponent(5) | mantissa(2)
        sign = (raw_bytes >> 7) & 0x01
        exponent = (raw_bytes >> 2) & 0x1F  # 5-bit exponent [0, 31]
        mantissa = raw_bytes & 0x03          # 2-bit mantissa [0, 3]
        sign_mantissa = (sign << 2) | mantissa  # 3-bit [0, 7]
        exp_max = 31
        sm_max = 7
    else:
        raise ValueError(f"Unknown FP8 format: {fmt}")

    results = {}

    # === Method 1: Full-value ANS ===
    full_size = ans_compress_size(raw_bytes.astype(np.int32), max_symbol=255)
    results['full_value_ans'] = {
        'bpv': full_size * 8 / n,
        'ratio': full_size / n * 100,
    }

    # Entropy bound
    _, counts = np.unique(raw_bytes, return_counts=True)
    full_probs = counts / n
    results['entropy'] = {
        'bpv': entropy(full_probs),
        'ratio': entropy(full_probs) / 8 * 100,
    }

    # === Method 2: Separated ANS (exp ANS + sign+mantissa ANS) ===
    exp_size = ans_compress_size(exponent.astype(np.int32), max_symbol=exp_max)
    sm_size = ans_compress_size(sign_mantissa.astype(np.int32), max_symbol=sm_max)
    total_sep = exp_size + sm_size
    results['separated_ans'] = {
        'bpv': total_sep * 8 / n,
        'ratio': total_sep / n * 100,
        'exp_bpv': exp_size * 8 / n,
        'sm_bpv': sm_size * 8 / n,
    }

    # Entropy of separated
    _, exp_counts = np.unique(exponent, return_counts=True)
    _, sm_counts = np.unique(sign_mantissa, return_counts=True)
    exp_ent = entropy(exp_counts / n)
    sm_ent = entropy(sm_counts / n)
    results['separated_entropy'] = {
        'bpv': exp_ent + sm_ent,
        'exp_entropy': exp_ent,
        'sm_entropy': sm_ent,
    }

    # === Method 3: Exp ANS + raw sign+mantissa ===
    if fmt == 'e4m3fn':
        raw_sm_bits = 4  # sign(1) + mantissa(3) = 4 bits
    else:
        raw_sm_bits = 3  # sign(1) + mantissa(2) = 3 bits
    raw_sm_size = n * raw_sm_bits / 8
    total_exp_ans_raw = exp_size + raw_sm_size
    results['exp_ans_raw_sm'] = {
        'bpv': total_exp_ans_raw * 8 / n,
        'ratio': total_exp_ans_raw / n * 100,
    }

    # === Method 4: Exp Huffman + raw sign+mantissa (ECF8-style) ===
    try:
        exp_huff_size = huffman_compress_size(exponent)
        total_ecf8 = exp_huff_size + raw_sm_size
        results['ecf8_style'] = {
            'bpv': total_ecf8 * 8 / n,
            'ratio': total_ecf8 / n * 100,
        }
    except Exception as e:
        results['ecf8_style'] = {'error': str(e)}

    # === Histogram analysis for mechanism figure ===
    results['histogram'] = {
        'unique_full': len(np.unique(raw_bytes)),
        'unique_exp': len(np.unique(exponent)),
        'unique_sm': len(np.unique(sign_mantissa)),
        'exp_distribution': {int(k): int(v) for k, v in zip(*np.unique(exponent, return_counts=True))},
    }

    return results


def run_model_ablation(model_name: str, fmt='e4m3fn'):
    """Run ablation on all FP8 layers of a model."""
    print(f"\n{'='*80}")
    print(f"Native FP8 Ablation: {model_name}")
    print(f"{'='*80}")

    files = list_repo_files(model_name)
    st_files = [f for f in files if f.endswith('.safetensors')]

    all_fp8_bytes = []
    per_layer = []

    for st_file in st_files:
        local_path = hf_hub_download(model_name, st_file)
        with safe_open(local_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if tensor.dtype != torch.float8_e4m3fn or tensor.numel() < 1024:
                    continue

                raw = np.ascontiguousarray(tensor.view(torch.uint8).numpy().flatten(), dtype=np.int32)
                all_fp8_bytes.append(raw)

                if tensor.numel() > 100_000:
                    result = ablation_on_tensor(raw, fmt)
                    per_layer.append({
                        'name': key,
                        'numel': tensor.numel(),
                        'results': result,
                    })

    if not all_fp8_bytes:
        print(f"  No FP8 tensors found!")
        return None

    # Run ablation on concatenated FP8 weights
    concatenated = np.concatenate(all_fp8_bytes)
    print(f"  Total FP8 values: {len(concatenated):,}")
    print(f"  Unique values: {len(np.unique(concatenated))}/256")

    overall = ablation_on_tensor(concatenated, fmt)

    # Print results
    print(f"\n  {'Method':<35s} {'bpv':>8s} {'Ratio':>8s} {'Gap to H(X)':>12s}")
    print(f"  {'-'*63}")

    h = overall['entropy']['bpv']
    for method, label in [
        ('entropy', 'Shannon Entropy H(X)'),
        ('full_value_ans', 'Full-value ANS'),
        ('separated_ans', 'Separated ANS (exp+s+m)'),
        ('exp_ans_raw_sm', 'Exp ANS + raw s+m'),
        ('ecf8_style', 'Exp Huffman + raw s+m (ECF8)'),
    ]:
        r = overall[method]
        if 'error' in r:
            print(f"  {label:<35s} {'ERROR':>8s}")
            continue
        bpv = r['bpv']
        ratio = r.get('ratio', bpv / 8 * 100)
        gap = bpv - h
        print(f"  {label:<35s} {bpv:8.3f} {ratio:7.2f}% {gap:+11.3f}")

    if 'separated_entropy' in overall:
        se = overall['separated_entropy']
        print(f"\n  Separated entropy: H(exp)={se['exp_entropy']:.3f} + H(s+m)={se['sm_entropy']:.3f} = {se['bpv']:.3f}")
        print(f"  Joint entropy H(X) = {h:.3f}")
        print(f"  Mutual information I(exp; s+m) = {se['bpv'] - h:.3f} bits")

    print(f"\n  Histogram: {overall['histogram']['unique_full']} unique values, "
          f"{overall['histogram']['unique_exp']} unique exponents, "
          f"{overall['histogram']['unique_sm']} unique s+m")

    return {
        'model': model_name,
        'n_values': len(concatenated),
        'overall': overall,
        'per_layer': per_layer[:5],  # Sample for size
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
            r = run_model_ablation(model_name, fmt)
            if r:
                results[model_name] = r
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    # Save results
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
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
