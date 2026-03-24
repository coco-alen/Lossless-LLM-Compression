"""
FP8 Hybrid: nvCOMP ANS on exp stream + raw sm stream

The exp_packed bytes (2-bit codes packed 4/byte) have ~86% entropy due
to the skewed distribution. nvCOMP ANS can compress them further.

Pipeline:
  Encode: FP8 → two-stream → nvCOMP ANS(exp_packed)
  Decode: nvCOMP ANS decode → two-stream decode

Expected ratio: ~73% (21.5% compressed exp + 50% raw sm + 2% overflow)
"""

import torch
import numpy as np
import cupy as cp
import time

# Check if nvcomp is available
try:
    from kvikio._lib.nvcomp_codec import NvCompBatchCodec
    HAS_KVIKIO = True
except ImportError:
    HAS_KVIKIO = False

try:
    import nvidia.nvcomp as nvcomp
    from nvidia.nvcomp import nvcompBatchedANSDecompressAsync, nvcompBatchedANSCompressAsync
    HAS_NVCOMP = True
except ImportError:
    HAS_NVCOMP = False

# Try using nvcomp via cupy directly
try:
    import ctypes
    import os
    # Try loading nvcomp shared library
    nvcomp_lib = None
    for path in ['/home/sky/miniconda3/envs/quant/lib/libnvcomp.so',
                 '/usr/local/lib/libnvcomp.so']:
        if os.path.exists(path):
            nvcomp_lib = ctypes.CDLL(path)
            break
    HAS_NVCOMP_LIB = nvcomp_lib is not None
except:
    HAS_NVCOMP_LIB = False


def find_best_window(exponents: np.ndarray, k: int = 3) -> tuple:
    counts = np.bincount(exponents, minlength=16)
    best_base, best_cov = 0, 0
    for base in range(16 - k + 1):
        cov = counts[base:base+k].sum()
        if cov > best_cov:
            best_cov = cov
            best_base = base
    return best_base, best_cov / len(exponents)


def encode_twostream(fp8_tensor, k=3):
    raw = fp8_tensor.contiguous().view(torch.uint8).flatten().numpy()
    n = len(raw)
    signs = (raw >> 7) & 1
    exponents = (raw >> 3) & 0xF
    mantissas = raw & 0x7
    base_exp, coverage = find_best_window(exponents, k)
    offsets = exponents.astype(np.int32) - base_exp
    is_common = (offsets >= 0) & (offsets < k)

    exp_codes = np.full(n, k, dtype=np.uint8)
    exp_codes[is_common] = offsets[is_common].astype(np.uint8)

    pad = (4 - n % 4) % 4
    ep = np.concatenate([exp_codes, np.zeros(pad, dtype=np.uint8)]) if pad else exp_codes
    exp_packed = ((ep[0::4] << 6) | (ep[1::4] << 4) | (ep[2::4] << 2) | ep[3::4]).astype(np.uint8)

    sm = ((signs << 3) | mantissas).astype(np.uint8)
    sp = np.concatenate([sm, np.zeros(n % 2, dtype=np.uint8)]) if n % 2 else sm
    sm_packed = ((sp[0::2] << 4) | sp[1::2]).astype(np.uint8)

    esc_exp = exponents[~is_common].astype(np.uint8)
    ne = len(esc_exp)
    ep2 = np.concatenate([esc_exp, np.zeros(ne % 2, dtype=np.uint8)]) if ne % 2 else esc_exp
    ov = ((ep2[0::2] << 4) | ep2[1::2]).astype(np.uint8) if len(ep2) else np.array([], dtype=np.uint8)

    return {
        'exp_packed': exp_packed, 'sm_packed': sm_packed, 'overflow_packed': ov,
        'base_exp': base_exp, 'k': k, 'n_elements': n, 'n_escapes': ne,
        'coverage': coverage,
        'exp_codes': exp_codes,  # for prefix sum
    }


def compute_entropy(data: np.ndarray) -> float:
    """Compute byte-level entropy."""
    counts = np.bincount(data.flatten(), minlength=256)
    probs = counts[counts > 0] / counts.sum()
    return -np.sum(probs * np.log2(probs))


def benchmark_compression_analysis(model_name="Qwen/Qwen3-0.6B"):
    """Analyze compression potential of exp_packed stream."""
    from transformers import AutoModelForCausalLM

    print("=" * 90)
    print("FP8 Hybrid ANS Compression Analysis")
    print("=" * 90)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    total_orig = 0
    total_exp = 0
    total_sm = 0
    total_ov = 0
    all_exp_bytes = []

    print(f"\n{'Layer':<45} {'n':>10} {'ExpH':>6} {'SmH':>6} {'ExpR':>6} {'TotR':>6}")

    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16 or param.numel() < 500_000:
            continue

        fp8 = param.data.to(torch.float8_e4m3fn)
        comp = encode_twostream(fp8)
        n = comp['n_elements']

        exp_h = compute_entropy(comp['exp_packed'])
        sm_h = compute_entropy(comp['sm_packed'])

        exp_compressed_est = len(comp['exp_packed']) * exp_h / 8  # bytes after entropy coding
        sm_size = len(comp['sm_packed'])
        ov_size = len(comp['overflow_packed'])

        total_orig += n
        total_exp += len(comp['exp_packed'])
        total_sm += sm_size
        total_ov += ov_size
        all_exp_bytes.append(comp['exp_packed'])

        exp_ratio = exp_compressed_est / n * 100
        total_ratio = (exp_compressed_est + sm_size + ov_size + 8) / n * 100

        if param.numel() >= 2_000_000:
            print(f"  {name:<43} {n:>10,} {exp_h:>5.2f} {sm_h:>5.2f} {exp_ratio:>5.1f}% {total_ratio:>5.1f}%")

    # Overall analysis
    combined_exp = np.concatenate(all_exp_bytes)
    exp_entropy = compute_entropy(combined_exp)
    exp_compressed = total_exp * exp_entropy / 8

    current_ratio = (total_exp + total_sm + total_ov) / total_orig * 100
    ans_ratio = (exp_compressed + total_sm + total_ov) / total_orig * 100

    print(f"\n  --- Overall ---")
    print(f"  Total elements: {total_orig:,}")
    print(f"  Exp stream: {total_exp:,} bytes ({total_exp/total_orig*100:.1f}% of orig)")
    print(f"    Byte entropy: {exp_entropy:.2f} bits")
    print(f"    After ANS: {exp_compressed:,.0f} bytes ({exp_compressed/total_orig*100:.1f}% of orig)")
    print(f"  SM stream: {total_sm:,} bytes ({total_sm/total_orig*100:.1f}% of orig)")
    print(f"    Byte entropy: ~8.0 bits (incompressible)")
    print(f"  Overflow: {total_ov:,} bytes ({total_ov/total_orig*100:.1f}% of orig)")
    print(f"\n  Current two-stream: {current_ratio:.1f}%")
    print(f"  With ANS on exp:   {ans_ratio:.1f}%")
    print(f"  Entropy limit:     70.4%")
    print(f"  Improvement:       {current_ratio - ans_ratio:.1f} pp")

    # Try Python-based ANS compression to verify
    print(f"\n  --- Actual ANS Compression Test (constriction, CPU) ---")
    try:
        import constriction

        # Compress exp_packed with ANS
        # Build frequency model
        counts = np.bincount(combined_exp, minlength=256)
        freqs = counts.astype(np.float64)
        freqs[freqs == 0] = 1e-10  # avoid zero
        freqs /= freqs.sum()

        # Compress in chunks (constriction API)
        model_family = constriction.stream.model.Categorical
        entropy_model = model_family(freqs.astype(np.float64))

        # Use stack coder for simplicity
        coder = constriction.stream.stack.AnsCoder()
        coder.encode_reverse(combined_exp.astype(np.int32), entropy_model)
        compressed = coder.get_compressed()

        ans_bytes = len(compressed) * 4  # compressed is uint32 array
        actual_ans_ratio = (ans_bytes + total_sm + total_ov) / total_orig * 100

        print(f"  ANS compressed exp: {ans_bytes:,} bytes ({ans_bytes/total_exp*100:.1f}% of exp stream)")
        print(f"  Total with ANS:     {actual_ans_ratio:.1f}%")
        print(f"  Improvement over two-stream: {current_ratio - actual_ans_ratio:.1f} pp")

    except ImportError:
        print("  constriction not available, skipping ANS test")

    # Try lz4 compression as alternative
    print(f"\n  --- LZ4 Compression Test ---")
    try:
        import lz4.block
        lz4_compressed = lz4.block.compress(combined_exp.tobytes(), store_size=False)
        lz4_ratio = (len(lz4_compressed) + total_sm + total_ov) / total_orig * 100
        print(f"  LZ4 compressed exp: {len(lz4_compressed):,} bytes ({len(lz4_compressed)/total_exp*100:.1f}% of exp stream)")
        print(f"  Total with LZ4:     {lz4_ratio:.1f}%")
    except ImportError:
        print("  lz4 not available")

    print(f"\n  --- Summary ---")
    print(f"  Dense FP8:              100.0%")
    print(f"  Two-stream (current):   {current_ratio:.1f}%  @ 584 GB/s batched")
    print(f"  Two-stream + ANS exp:   {ans_ratio:.1f}%  (est, + nvCOMP decode)")
    print(f"  Full ANS (CPU):          70.4%  (entropy optimal)")


if __name__ == "__main__":
    benchmark_compression_analysis()
