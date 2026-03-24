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

(Entries will be added by the optimization loop)
