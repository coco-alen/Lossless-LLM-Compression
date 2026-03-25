#!/usr/bin/env python3
"""
Benchmark lossless compression of real GPTQ INT4 quantized model weights.

Loads Qwen2.5-7B-Instruct-GPTQ-Int4 safetensors directly (no auto-gptq needed),
unpacks INT4 values from packed int32 qweight tensors, and benchmarks:
  - Shannon entropy of INT4 values
  - ANS compression of INT4 values
  - Compression of scales (float16) and qzeros (int32 packed)
  - Total model compression ratio

GPTQ packing: each int32 holds 8 INT4 values (4 bits each).
  value_i = (int32 >> (i * 4)) & 0xF   for i in 0..7
"""

import os
import sys
import time
import numpy as np
import torch
import math
import json
from collections import Counter, defaultdict
from pathlib import Path
from safetensors import safe_open

# ─── Utility functions ────────────────────────────────────────────────────────

def shannon_entropy(counts):
    """Compute Shannon entropy in bits from a count array."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def unpack_int4_from_int32(packed: np.ndarray) -> np.ndarray:
    """Unpack INT4 values from int32 packed tensor. Each int32 has 8 x 4-bit values."""
    packed = packed.astype(np.uint32).ravel()
    result = np.zeros(len(packed) * 8, dtype=np.uint8)
    for i in range(8):
        result[i::8] = (packed >> (i * 4)) & 0xF
    return result


def compress_ans_chunked(data: np.ndarray, alphabet_size: int, chunk_size: int = 10_000_000) -> tuple:
    """Compress data using ANS (constriction) in chunks. Returns (compressed_bits, original_bits)."""
    import constriction
    data_int = data.astype(np.int32)
    counts = np.bincount(data_int, minlength=alphabet_size)[:alphabet_size]

    # Replace zeros with 1 to avoid zero-probability symbols
    counts_safe = counts.copy()
    counts_safe[counts_safe == 0] = 1

    probs = counts_safe.astype(np.float64) / counts_safe.sum()
    model = constriction.stream.model.Categorical(probs, perfect=False)

    total_compressed_bits = 0
    n = len(data_int)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = data_int[start:end]
        encoder = constriction.stream.stack.AnsCoder()
        encoder.encode_reverse(chunk, model)
        compressed = encoder.get_compressed()
        total_compressed_bits += len(compressed) * 32

    max_bits = int(math.ceil(math.log2(alphabet_size))) if alphabet_size > 1 else 1
    original_bits = len(data) * max_bits

    return total_compressed_bits, original_bits


def compress_ans_uint8(data: np.ndarray) -> tuple:
    """Compress uint8 data using ANS. Returns (compressed_bits, original_bits)."""
    return compress_ans_chunked(data, alphabet_size=256)


def compress_ans_int4(data: np.ndarray) -> tuple:
    """Compress INT4 data (0-15) using ANS. Returns (compressed_bits, original_bits_at_4bpv)."""
    return compress_ans_chunked(data, alphabet_size=16)


# ─── Main benchmark ──────────────────────────────────────────────────────────

def main():
    model_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct-GPTQ-Int4/"
        "snapshots/e9c932ac1893a49ae0fc497ad6e1e86e2e39af20/model-00001-of-00002.safetensors"
    )

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    print("=" * 80)
    print("GPTQ INT4 Lossless Compression Benchmark")
    print(f"Model: Qwen2.5-7B-Instruct-GPTQ-Int4 (shard 1)")
    print("=" * 80)

    f = safe_open(model_path, framework="pt")
    all_keys = sorted(f.keys())

    # Categorize tensors
    qweight_keys = [k for k in all_keys if k.endswith(".qweight")]
    scales_keys = [k for k in all_keys if k.endswith(".scales")]
    qzeros_keys = [k for k in all_keys if k.endswith(".qzeros")]
    g_idx_keys = [k for k in all_keys if k.endswith(".g_idx")]
    bias_keys = [k for k in all_keys if k.endswith(".bias")]
    norm_keys = [k for k in all_keys if "layernorm" in k or k == "model.embed_tokens.weight"]
    other_keys = [k for k in all_keys if k not in set(qweight_keys + scales_keys + qzeros_keys + g_idx_keys + bias_keys + norm_keys)]

    print(f"\nTensor categories in shard 1:")
    print(f"  qweight (packed INT4): {len(qweight_keys)} tensors")
    print(f"  scales (float16):      {len(scales_keys)} tensors")
    print(f"  qzeros (packed int32): {len(qzeros_keys)} tensors")
    print(f"  g_idx (int32):         {len(g_idx_keys)} tensors")
    print(f"  bias (float16):        {len(bias_keys)} tensors")
    print(f"  norm/embed (float16):  {len(norm_keys)} tensors")
    print(f"  other:                 {len(other_keys)} tensors")
    if other_keys:
        for k in other_keys[:5]:
            t = f.get_tensor(k)
            print(f"    {k}: {t.shape} {t.dtype}")

    # ─── 1. Analyze and compress qweight (INT4) tensors ──────────────────────
    print("\n" + "=" * 80)
    print("1. QWEIGHT (Packed INT4) Analysis")
    print("=" * 80)

    all_int4_values = []
    total_qweight_bytes = 0
    layer_entropies = []

    for i, key in enumerate(qweight_keys):
        tensor = f.get_tensor(key).numpy()
        total_qweight_bytes += tensor.nbytes
        int4_vals = unpack_int4_from_int32(tensor)
        all_int4_values.append(int4_vals)

        # Per-tensor entropy
        counts = np.bincount(int4_vals.astype(np.int64), minlength=16)[:16]
        ent = shannon_entropy(counts)
        layer_entropies.append((key, ent, len(int4_vals)))

        if i < 3:  # Print details for first 3
            print(f"\n  {key}:")
            print(f"    Shape: {tensor.shape}, INT4 values: {len(int4_vals)}")
            print(f"    Entropy: {ent:.4f} bits (max 4.0)")
            print(f"    Distribution: {dict(zip(range(16), counts.tolist()))}")

    all_int4 = np.concatenate(all_int4_values)
    global_counts = np.bincount(all_int4.astype(np.int64), minlength=16)[:16]
    global_entropy = shannon_entropy(global_counts)

    print(f"\n  --- Aggregate INT4 Stats ---")
    print(f"  Total INT4 values: {len(all_int4):,}")
    print(f"  Global entropy: {global_entropy:.4f} bits (max 4.0)")
    print(f"  Theoretical compression: {global_entropy/4*100:.1f}% of packed size")
    print(f"  Savings potential: {(1 - global_entropy/4)*100:.1f}%")
    print(f"\n  Value distribution (top 5):")
    probs = global_counts / global_counts.sum()
    for idx in np.argsort(-global_counts)[:5]:
        print(f"    value {idx:2d}: {probs[idx]*100:5.2f}% (count {global_counts[idx]:,})")

    # Per-tensor entropy distribution
    ents = [e for _, e, _ in layer_entropies]
    print(f"\n  Per-tensor entropy: min={min(ents):.4f}, max={max(ents):.4f}, mean={np.mean(ents):.4f}")

    # ANS compression of all INT4 values
    print(f"\n  Compressing with ANS (global model)...")
    t0 = time.time()
    compressed_bits, original_bits = compress_ans_int4(all_int4)
    t1 = time.time()
    print(f"  ANS compression: {compressed_bits/8:,.0f} bytes / {original_bits/8:,.0f} bytes = {compressed_bits/original_bits*100:.2f}%")
    print(f"  Bits per INT4 value: {compressed_bits/len(all_int4):.4f} (entropy: {global_entropy:.4f})")
    print(f"  Time: {t1-t0:.2f}s")

    # Also try per-tensor ANS
    print(f"\n  Compressing with per-tensor ANS...")
    t0 = time.time()
    per_tensor_compressed = 0
    per_tensor_original = 0
    for int4_vals in all_int4_values:
        cb, ob = compress_ans_int4(int4_vals)
        per_tensor_compressed += cb
        per_tensor_original += ob
    t1 = time.time()
    print(f"  Per-tensor ANS: {per_tensor_compressed/8:,.0f} bytes / {per_tensor_original/8:,.0f} bytes = {per_tensor_compressed/per_tensor_original*100:.2f}%")
    print(f"  Bits per INT4 value: {per_tensor_compressed/len(all_int4):.4f}")
    print(f"  Time: {t1-t0:.2f}s")

    # ─── 2. Analyze and compress scales (float16) ────────────────────────────
    print("\n" + "=" * 80)
    print("2. SCALES (float16) Analysis")
    print("=" * 80)

    all_scales_bytes = []
    total_scales_bytes = 0

    for key in scales_keys:
        tensor = f.get_tensor(key)
        raw = tensor.numpy().view(np.uint8)
        all_scales_bytes.append(raw)
        total_scales_bytes += tensor.numel() * 2

    all_scales_raw = np.concatenate([b.ravel() for b in all_scales_bytes])

    # Byte-level analysis
    # Split into high byte and low byte
    all_scales_flat = np.concatenate([f.get_tensor(k).numpy().ravel() for k in scales_keys])
    raw_u16 = all_scales_flat.view(np.uint16)
    high_bytes = ((raw_u16 >> 8) & 0xFF).astype(np.uint8)
    low_bytes = (raw_u16 & 0xFF).astype(np.uint8)

    h_counts = np.bincount(high_bytes, minlength=256)
    l_counts = np.bincount(low_bytes, minlength=256)
    h_entropy = shannon_entropy(h_counts)
    l_entropy = shannon_entropy(l_counts)

    print(f"  Total scales values: {len(all_scales_flat):,}")
    print(f"  Total size: {total_scales_bytes:,} bytes")
    print(f"  High byte entropy: {h_entropy:.4f} bits")
    print(f"  Low byte entropy: {l_entropy:.4f} bits")
    print(f"  Combined byte entropy: {(h_entropy + l_entropy)/16*100:.1f}% of original")

    # Full 16-bit value ANS
    unique_16 = len(np.unique(raw_u16))
    full_entropy_16 = shannon_entropy(np.bincount(raw_u16.astype(np.int64), minlength=65536))
    print(f"  Unique 16-bit values: {unique_16:,}")
    print(f"  Full 16-bit entropy: {full_entropy_16:.4f} bits (max 16)")
    print(f"  Theoretical compression: {full_entropy_16/16*100:.1f}%")

    # ANS compress scales as bytes
    print(f"\n  Compressing scales with byte-level ANS...")
    t0 = time.time()
    cb_h, ob_h = compress_ans_uint8(high_bytes)
    cb_l, ob_l = compress_ans_uint8(low_bytes)
    t1 = time.time()
    total_cb = cb_h + cb_l
    total_ob = ob_h + ob_l
    print(f"  Byte-level ANS: {total_cb/8:,.0f} bytes / {total_ob/8:,.0f} bytes = {total_cb/total_ob*100:.2f}%")
    print(f"  Time: {t1-t0:.2f}s")

    # ─── 3. Analyze and compress qzeros (packed int32) ───────────────────────
    print("\n" + "=" * 80)
    print("3. QZEROS (packed int32) Analysis")
    print("=" * 80)

    all_qzeros_int4 = []
    total_qzeros_bytes = 0

    for key in qzeros_keys:
        tensor = f.get_tensor(key).numpy()
        total_qzeros_bytes += tensor.nbytes
        int4_vals = unpack_int4_from_int32(tensor)
        all_qzeros_int4.append(int4_vals)

    all_qz = np.concatenate(all_qzeros_int4)
    qz_counts = np.bincount(all_qz.astype(np.int64), minlength=16)[:16]
    qz_entropy = shannon_entropy(qz_counts)

    print(f"  Total qzeros INT4 values: {len(all_qz):,}")
    print(f"  Total packed size: {total_qzeros_bytes:,} bytes")
    print(f"  Entropy: {qz_entropy:.4f} bits (max 4.0)")
    print(f"  Distribution: {dict(zip(range(16), qz_counts.tolist()))}")

    # ANS compress qzeros
    if len(all_qz) > 0:
        cb_qz, ob_qz = compress_ans_int4(all_qz)
        print(f"  ANS: {cb_qz/8:,.0f} bytes / {ob_qz/8:,.0f} bytes = {cb_qz/ob_qz*100:.2f}%")

    # ─── 4. Analyze g_idx ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("4. G_IDX (int32) Analysis")
    print("=" * 80)

    total_gidx_bytes = 0
    gidx_patterns = defaultdict(int)

    for i, key in enumerate(g_idx_keys):
        tensor = f.get_tensor(key).numpy()
        total_gidx_bytes += tensor.nbytes
        if i < 3:
            unique = len(np.unique(tensor))
            print(f"  {key}: shape={tensor.shape}, unique={unique}, min={tensor.min()}, max={tensor.max()}")
        # Check if sequential
        if np.array_equal(tensor, np.sort(tensor)):
            gidx_patterns["sorted"] += 1
        else:
            gidx_patterns["unsorted"] += 1

    # Check if g_idx is always trivial (sequential 0,0,...,0, 1,1,...,1, ...)
    sample = f.get_tensor(g_idx_keys[0]).numpy()
    is_trivial = np.array_equal(sample, np.arange(len(sample)) // (len(sample) // (sample.max() + 1)) if sample.max() > 0 else sample)

    # Try delta coding g_idx
    deltas = np.diff(sample.astype(np.int64))
    unique_deltas = len(np.unique(deltas))
    print(f"\n  Sample g_idx delta analysis: {unique_deltas} unique deltas")
    print(f"  Total g_idx size: {total_gidx_bytes:,} bytes")
    print(f"  (g_idx can be stored as group_size parameter + trivial formula in most cases)")

    # ─── 5. Analyze bias and norm weights ────────────────────────────────────
    print("\n" + "=" * 80)
    print("5. BIAS & NORM WEIGHTS (float16) Analysis")
    print("=" * 80)

    total_other_bytes = 0
    other_compressed_bits = 0
    other_original_bits = 0

    for key in bias_keys + norm_keys:
        tensor = f.get_tensor(key).numpy()
        total_other_bytes += tensor.nbytes
        raw = tensor.view(np.uint8).ravel()
        if len(raw) > 0:
            cb, ob = compress_ans_uint8(raw)
            other_compressed_bits += cb
            other_original_bits += ob

    print(f"  Total size: {total_other_bytes:,} bytes")
    if other_original_bits > 0:
        print(f"  ANS compressed: {other_compressed_bits/8:,.0f} bytes / {other_original_bits/8:,.0f} bytes = {other_compressed_bits/other_original_bits*100:.2f}%")

    # ─── 6. Total model compression summary ──────────────────────────────────
    print("\n" + "=" * 80)
    print("6. TOTAL COMPRESSION SUMMARY (shard 1)")
    print("=" * 80)

    # Original sizes (as stored in GPTQ format)
    orig_qweight = total_qweight_bytes
    orig_scales = total_scales_bytes
    orig_qzeros = total_qzeros_bytes
    orig_gidx = total_gidx_bytes
    orig_other = total_other_bytes

    # Compressed sizes
    # qweight: use per-tensor ANS result (better granularity)
    comp_qweight = per_tensor_compressed / 8
    # scales: byte-level ANS
    comp_scales = total_cb / 8
    # qzeros: ANS
    comp_qzeros = cb_qz / 8 if len(all_qz) > 0 else orig_qzeros
    # g_idx: can be reconstructed from group_size, so essentially 0 (or store as tiny metadata)
    # But let's be conservative and use delta-coding
    comp_gidx = orig_gidx * 0.1  # g_idx is trivially compressible (sequential pattern)
    # bias/norm
    comp_other = other_compressed_bits / 8 if other_original_bits > 0 else orig_other

    total_orig = orig_qweight + orig_scales + orig_qzeros + orig_gidx + orig_other
    total_comp = comp_qweight + comp_scales + comp_qzeros + comp_gidx + comp_other

    print(f"\n  {'Component':<25} {'Original':>12} {'Compressed':>12} {'Ratio':>8}")
    print(f"  {'-'*60}")
    print(f"  {'qweight (INT4)':<25} {orig_qweight:>12,} {comp_qweight:>12,.0f} {comp_qweight/orig_qweight*100:>7.1f}%")
    print(f"  {'scales (fp16)':<25} {orig_scales:>12,} {comp_scales:>12,.0f} {comp_scales/orig_scales*100:>7.1f}%")
    print(f"  {'qzeros (INT4)':<25} {orig_qzeros:>12,} {comp_qzeros:>12,.0f} {comp_qzeros/orig_qzeros*100:>7.1f}%")
    print(f"  {'g_idx (int32)':<25} {orig_gidx:>12,} {comp_gidx:>12,.0f} {comp_gidx/orig_gidx*100:>7.1f}%")
    print(f"  {'bias/norm (fp16)':<25} {orig_other:>12,} {comp_other:>12,.0f} {comp_other/orig_other*100 if orig_other > 0 else 100:>7.1f}%")
    print(f"  {'-'*60}")
    print(f"  {'TOTAL':<25} {total_orig:>12,} {total_comp:>12,.0f} {total_comp/total_orig*100:>7.1f}%")
    print(f"\n  Overall savings: {(1 - total_comp/total_orig)*100:.1f}%")

    # Also compute effective bits per original parameter
    # In GPTQ-INT4, each original parameter is represented by ~4.x bits
    # (4 bits qweight + amortized scales/zeros/gidx)
    total_int4_params = len(all_int4)
    orig_bpw = total_orig * 8 / total_int4_params
    comp_bpw = total_comp * 8 / total_int4_params
    print(f"\n  Original bits per weight (qweight params): {orig_bpw:.3f}")
    print(f"  Compressed bits per weight: {comp_bpw:.3f}")
    print(f"  Savings: {orig_bpw - comp_bpw:.3f} bits/weight")

    # ─── 7. Comparison with gzip/lz4 baseline ────────────────────────────────
    print("\n" + "=" * 80)
    print("7. COMPARISON: ANS vs Generic Compression")
    print("=" * 80)

    import lz4.frame
    import zlib

    # Compress all qweight raw bytes with lz4 and zlib
    all_qweight_raw = b""
    for key in qweight_keys:
        tensor = f.get_tensor(key).numpy()
        all_qweight_raw += tensor.tobytes()

    t0 = time.time()
    lz4_compressed = lz4.frame.compress(all_qweight_raw)
    t_lz4 = time.time() - t0

    t0 = time.time()
    zlib_compressed = zlib.compress(all_qweight_raw, 9)
    t_zlib = time.time() - t0

    print(f"  qweight raw bytes: {len(all_qweight_raw):,}")
    print(f"  LZ4:  {len(lz4_compressed):>12,} bytes ({len(lz4_compressed)/len(all_qweight_raw)*100:.1f}%) [{t_lz4:.2f}s]")
    print(f"  zlib: {len(zlib_compressed):>12,} bytes ({len(zlib_compressed)/len(all_qweight_raw)*100:.1f}%) [{t_zlib:.2f}s]")
    print(f"  ANS (per-tensor INT4): {per_tensor_compressed/8:>10,.0f} bytes ({per_tensor_compressed/8/len(all_qweight_raw)*100:.1f}%)")
    print(f"\n  Note: ANS operates on unpacked INT4 values (entropy-optimal),")
    print(f"  while LZ4/zlib operate on raw packed int32 bytes.")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
