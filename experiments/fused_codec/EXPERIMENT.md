# FP8 Kernel Optimization Experiment Log

## Baseline Results (2026-03-24)

| # | Method | FP8 Ratio | Decode GB/s | Lossless | Notes |
|---|--------|----------|-------------|----------|-------|
| 0 | Dense FP8 | 100.0% | ~4800 (HBM) | N/A | Baseline |
| 1 | nvCOMP ANS | 85.7% | 29-56 | Yes | NVIDIA library, byte-level |
| 2 | Our Huffman (per-layer) | 77.1% | 5-14 | Yes | DFloat11-style, 9 LUT levels |
| 3 | Our Huffman (shared, analytical) | 71.0% | ~10 est | Yes | Not yet implemented as kernel |
| 4 | TCA-TBE k=3 (unpacked) | 116.6% | Fast | No* | Byte-aligned wastes 2 bits/code |
| 5 | Our ANS (CPU) | 70.4% | CPU only | Yes | Entropy optimal, constriction |

*TCA-TBE decode had lossless verification failures (escape prefix sum bug).

## Target
- Ratio ≤ 80% AND decode ≥ 50 GB/s (stretch: ≥ 100 GB/s)
- OR: fused decode+GEMM that beats dense FP8 latency on decode-phase shapes

---

## Experiments

### Exp 1: Packed 6-bit TBE (fp8_packed_tbe.py)
- **Approach**: Pack 4 × 6-bit TCA-TBE codes per 3 bytes (k=3 exponent window)
- **Ratio**: 91.6% — bitmap overhead (12.5%) kills compression
- **Decode GB/s**: 5.9 (full, including CuPy prefix sum)
- **Lossless**: FAIL (escape handling bug, abandoned in favor of two-stream)
- **Why failed**: Bitmap overhead too high; the TBE escape bitmap costs n/8 bytes

### Exp 2: Two-Stream v1 (fp8_twostream.py)
- **Approach**: Separate exponent (2-bit codes, 4/byte = 25%) from sign+mantissa (4-bit, 2/byte = 50%). Escapes via overflow + CuPy prefix sum.
- **Ratio**: 77.0%
- **Decode GB/s**: 4.0 full / 60 kernel-only
- **Lossless**: PASS (after fixing uint64→uint32 dtype bug in escape_prefix)
- **Why kernel-only fast**: CuPy cumsum on n×uint32 array dominates full pipeline

### Exp 3: Two-Stream v2 — Fused prefix sum (fp8_twostream_v2.py)
- **Approach**: Compute escape prefix sum INSIDE decode kernel using warp ballot + block-level scan. Eliminates CuPy cumsum.
- **Ratio**: 78.6% (slightly higher due to block_escape_prefix metadata)
- **Decode GB/s**: 57.5
- **Lossless**: PASS
- **Key change**: Precomputed block_escape_prefix (per-block escape counts) + warp __ballot_sync + __popc for intra-warp prefix

### Exp 4: Two-Stream v3 — 4 elements/thread (fp8_twostream_v3.py)
- **Approach**: Each thread processes 4 elements (1 exp byte + 2 sm bytes → 4 outputs as uint32)
- **Ratio**: 77.4%
- **Decode GB/s**: 63.8 (aggregate), 328 GB/s on embed_tokens (155M)
- **Lossless**: PASS
- **Key change**: Warp shuffle prefix sum, vectorized uint32 writes

### Exp 5: Two-Stream v4 — 8 elements/thread + prealloc (fp8_twostream_v4.py)
- **Approach**: 8 elements per thread, pre-allocated output buffer, CUDA event timing
- **Ratio**: 77.2%
- **Decode GB/s**: 165 (aggregate), 571 GB/s on embed_tokens (155M)
- **Lossless**: PASS
- **Key change**: Pre-allocated output avoids cp.zeros; CUDA events give accurate GPU timing

### Exp 6: Two-Stream v5 — 16 elements/thread + batched decode ⭐ BEST (fp8_twostream_v5.py)
- **Approach**: 16 elements per thread + single kernel launch for ALL layers via per-layer metadata
- **Ratio**: 77.1%
- **Decode GB/s**: 159 (per-layer) / **584** (batched, all 197 layers in 1.02ms)
- **Lossless**: PASS
- **Key change**: Batched kernel with block_to_layer/block_to_local indirection. 4×uint32 vectorized writes.

### Exp 7: Configuration Sweep (fp8_twostream_v6.py)
- **Approach**: Sweep threads {256,512} × elems/thread {8,16,32}
- **Lossless**: FAIL (generic kernel template bug, abandoned)
- **Observation**: Best configs from v5 already near-optimal. 32 elem/thread reduces throughput (register pressure).

## Summary Table

| # | Method | Ratio | Dec GB/s | Lossless | Status |
|---|--------|-------|----------|----------|--------|
| 1 | Packed 6-bit TBE | 91.6% | 5.9 | FAIL | Abandoned |
| 2 | Two-Stream v1 | 77.0% | 4.0/60 | PASS | Superseded |
| 3 | Two-Stream v2 (fused prefix) | 78.6% | 57.5 | PASS | Superseded |
| 4 | Two-Stream v3 (4 elem/th) | 77.4% | 63.8 | PASS | Superseded |
| 5 | Two-Stream v4 (8 elem/th) | 77.2% | 165 | PASS | Superseded |
| **6** | **Two-Stream v5 (16 elem/th, batched)** | **77.1%** | **584** | **PASS** | **⭐ BEST** |
| 7 | Config sweep | — | — | FAIL | Abandoned |
| 8 | Fused decode+GEMM (Triton) | 75% | 0.4-0.6x dense | N/A | Prototype, 2-5x slower |
| 9 | Hybrid ANS on exp stream | **73.4%** | Est. slower | PASS(CPU) | 3.7pp better ratio, nvCOMP unstable |
| 10 | Interleaved block layout | 77.1% | 250 (0.97x v5) | FAIL | Cache benefit nonexistent on H200 |
| 11 | Branchless near-lossless | 75.0% | 466 per-layer | N/A | 1.6x faster, escape handling = 38% of cost |
| 12 | Triton decode | — | — | FAIL | Triton kernel bugs, abandoned |

## Key Insights

1. **Two-stream separation is the winning idea**: Separating FP8 into exponent (2-bit codes) and sign+mantissa (4-bit raw) enables both good compression (77%) and fast fixed-width decode.

2. **Escape handling costs 38% of kernel time** (466 vs 290 GB/s). For near-lossless inference, skipping escapes is a viable option.

3. **Batched decode eliminates kernel launch overhead**: Single kernel for all 197 layers: 584 GB/s vs 290 GB/s per-layer.

4. **The compression-throughput Pareto frontier**:
   - 70.4%: ANS (CPU only, entropy optimal)
   - 73.4%: Two-stream + ANS on exp (hybrid, needs nvCOMP)
   - 75.0%: Branchless near-lossless (466 GB/s per-layer)
   - **77.1%: Two-stream lossless (584 GB/s batched) ← BEST practical**
   - 85.7%: nvCOMP ANS (29-56 GB/s)
   - 100%: Dense FP8

5. **Memory bandwidth utilization**: 584 GB/s = 12% of H200's 4.8 TB/s. The kernel is compute-bound (prefix sum, branches), not bandwidth-bound.
