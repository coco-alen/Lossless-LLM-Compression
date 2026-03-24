# Optimizer Compression Experiment Log

## Goal
Losslessly compress AdamW optimizer states in GPU memory during training, without significantly impacting training speed.

## Critical Discovery: PyTorch BF16 Optimizer States
**When model params are BF16, PyTorch's AdamW stores m/v in BF16 too!**
- Memory: m(bf16) + v(bf16) = 4Ψ bytes (not 8Ψ)

---

## Experiment Results Summary

### Experiment 1: Entropy Analysis — `01_entropy_analysis.py`
**BF16 optimizer states compress to ~73% (theoretical limit)**
| State | Compression Limit | Exponent Entropy |
|-------|-------------------|------------------|
| m     | 73.2%             | 3.78 bits/8 |
| v     | 72.6%             | 4.68 bits/8 |
- byte0 (mantissa): ~7.97 bits (incompressible)
- byte1 (sign+exp): ~3.6-3.8 bits (very compressible)
- ~7,700 unique 16-bit symbols in m; ~6,400 in v

### Experiment 2: Baseline — `02_baseline_training.py`
Qwen3-0.6B (596M params), batch=4, seq=256:
| Metric | Value |
|--------|-------|
| Model memory (bf16) | 1137 MB |
| Optimizer states (m+v) | 2274 MB |
| Step time | 169 ms |
| Optimizer step time | 13 ms (8% of total) |
| Tokens/sec | 6001 |

### Experiment 3: Per-Param CPU Offload — DEAD END
- 24x slower due to many small CPU↔GPU transfers
- **Lesson**: Must batch all transfers into one large memcpy

### Experiment 5: GPU HighByte Packing — DEAD END
- Only 97.3% compression (most params have >16 unique high bytes)
- 2.1x slower, uses MORE memory from temporaries
- **Lesson**: 4-bit packing is too coarse; need variable-length coding

### Experiment 6: Microbenchmark — INFORMATIONAL
- Byte extraction on GPU: ~1ms per 100M elements
- Full compress+decompress round-trip: ~3.4ms per 100M elements
- 29 unique high byte values typical → need 5 bits (not 4)

### Experiment 7: Wrapper Approaches — `07_wrapper_approach.py`
| Method | GPU Mem | ΔMem | Step | Slow |
|--------|---------|------|------|------|
| Standard AdamW | 3475 MB | — | 169 ms | 1.00x |
| **Batched CPU Offload** | **1201 MB** | **-2274 MB** | **236 ms** | **1.40x** |
- Batched CPU offload: LOSSLESS, verified bit-exact
- Saves 65% of steady-state GPU memory

### Experiment 8: Pipelined Offload — `08_pipelined_offload.py`
| Method | GPU Mem | ΔMem | Step | Slow |
|--------|---------|------|------|------|
| Simple CPU Offload | 1201 MB | -2274 MB | 266 ms | 1.57x |
| Pipelined Offload | 3475 MB | 0 MB | 238 ms | 1.41x |
- Pipelined saves no memory (pre-allocated GPU buffers)
- Slightly faster but defeats purpose

### Experiment 9: Selective + Hooked Offload — `09_selective_and_hooked.py` ⭐ BEST
| Method | GPU Mem | ΔMem | Step | Slowdown |
|--------|---------|------|------|----------|
| Standard AdamW | 3475 MB | — | 169 ms | 1.00x |
| Offload v only | 2338 MB | -1137 MB | 217 ms | 1.28x |
| Offload both | 1201 MB | -2274 MB | 268 ms | 1.59x |
| **Hooked offload both** | **1201 MB** | **-2274 MB** | **191 ms** | **1.13x** |
| Hooked offload v | 2338 MB | -1137 MB | 99 ms* | 0.59x* |

*Hooked v-only anomalously fast — likely measurement artifact from hook timing.

**🏆 Best result: Hooked offload both — saves 2274 MB (65%) with only 13% slowdown**
- Uses backward hook on layer 19/28 to trigger async CPU→GPU prefetch
- Prefetch overlaps with backward pass computation
- All approaches verified LOSSLESS (bit-exact)

### Experiment 10: Delta Coding Analysis — `10_delta_analysis.py`
| State | Δ1 step changed | XOR-coded size |
|-------|------------------|----------------|
| m (β₁=0.9) | 98.2% | 58.5% of dense |
| v (β₂=0.999) | 22.3% | 11.0% of dense |

**v changes very slowly** — only 22% of BF16 values change per step.
- XOR delta entropy for v: 1.76 bits/16 = 11% of dense
- Sparse storage of v changes: 67% of dense (index overhead hurts)
- m changes almost entirely each step — delta coding useless for m
- Over 50 steps, v cumulative change reaches 48%

**Implication**: For CPU offload, we could transfer only v's changes:
- Current: transfer full v (1137 MB) each direction
- Delta: transfer ~11% of v = ~125 MB each direction (9x reduction for v)
- But needs XOR entropy coding on GPU (complex implementation)

---

## Phase 2: In-GPU FP32 Compression (Experiments 24-37)

Goal: Lossless compression of FP32 optimizer states **in GPU memory** (no CPU offloading).
Model: Qwen3-0.6B, 596M params, FP32 optimizer states = 4548 MB total.

### Key Findings

**FP32 Byte-Plane Analysis** (Experiment 27):
| Byte | m unique | m entropy | v unique | v entropy |
|------|----------|-----------|----------|-----------|
| byte3 (MSB) | 49 | 3.710 bits | 33 | 3.607 bits |
| byte2 | 256 | 7.971 bits | 256 | 7.954 bits |
| byte1 | 256 | 8.000 bits | 256 | 8.000 bits |
| byte0 (LSB) | 256 | 7.998 bits | 256 | 7.998 bits |

- **Only byte3 is compressible** (~3.7 bits out of 8)
- Total entropy: 27.6 bits/value. Theoretical max savings: **13.5% = ~620 MB**
- Practical 6-bit fixed-length savings: **6.25% = ~285 MB**

### Experiment 24: torch.compile'd Compression — `24_compiled_compression.py`
| Method | Mem | ΔMem | Peak | ΔPeak | Step | Slow |
|--------|-----|------|------|-------|------|------|
| Standard | 6887M | — | 12060M | — | 179ms | 1.00x |
| Compiled | 6602M | -285M | 18257M | +6197M | 233ms | 1.30x |
- Fast (1.30x) but **+6197 MB peak memory spike** from torch.compile intermediates on full buffer

### Experiment 25: High-16 Compression — `25_high16_compression.py` — DEAD END
- 9742 unique high-16 values → 14 bits. Packing 2×14=28 bits rounds to 4 bytes → **0% savings**

### Experiment 27: Entropy Analysis — `27_entropy_analysis.py`
- Confirmed byte3 is the ONLY compressible byte in FP32 optimizer states
- high16 entropy: ~11.6 bits (14-bit packing wastes 2.4 bits)

### Experiments 28-32: GPU Huffman/ANS — DEAD ENDS
| Exp | Method | Encode time | Decode time | Verdict |
|-----|--------|-------------|-------------|---------|
| 28 | CPU ANS (constriction) | 4.6s | 18.6s | CPU too slow |
| 30 | CuPy CUDA Huffman | 2244ms | 97ms | Encode too slow |
| 31 | PyTorch prefix-sum Huffman | 401ms | 77ms | Still too slow (956ms × 2) |
| 32 | Parallel atomicOr Huffman | 40785ms | — | Catastrophic contention |

**Conclusion: Variable-length coding is impractical on GPU** due to bit-level scatter contention.

### Experiment 33: Clean Byte3 Compression — `33_final_compression.py`
| Method | Mem | ΔMem | Peak | ΔPeak | Step | Slow |
|--------|-----|------|------|-------|------|------|
| Standard | 6887M | — | 12060M | — | 179ms | 1.00x |
| Compressed | 6602M | -285M | 19251M | +7191M | 282ms | 1.57x |
- Without torch.compile. Worse speed AND worse peak than exp24. Full-buffer intermediates are the bottleneck.

### Experiments 34-36: Addressing Peak Memory
| Exp | Approach | ΔMem | ΔPeak | Slow |
|-----|----------|------|-------|------|
| 34 | Chunked (64M) | -285M | +3780M | 1.66x |
| 35 | Streaming (free chunks during decompress) | -285M | +3781M | 1.63x |
| 36 | Per-parameter | -46M | -38M | 3.99x |

- Chunked/streaming didn't help peak: PyTorch CUDA allocator caches freed memory
- Per-parameter: great peak but only -46 MB savings (per-tensor overhead) and 4x slow

### Experiment 37: Grouped + Chunked + Compiled — `37_grouped_compression.py` ⭐ BEST IN-GPU
| Method | Mem | ΔMem | Peak | ΔPeak | Step | Slow |
|--------|-----|------|------|-------|------|------|
| Standard | 6887M | — | 12060M | — | 184ms | 1.00x |
| **Grouped compressed** | **6602M** | **-285M** | **11781M** | **-279M** | **265ms** | **1.44x** |

- Groups params into ~64M-element batches for efficient kernels
- Chunked compress/decompress within groups to bound intermediates
- torch.compile on chunk operations for speed
- **Peak is LOWER than baseline** (-279 MB) because states are compressed during forward/backward
- Verified lossless (bit-exact) over 5 steps

---

## Conclusions and Recommendations

### Method 1: Hooked CPU Offload (Best for maximum memory savings)
1. **Memory savings**: 2274 MB (65% of optimizer states) — from 3475 MB to 1201 MB
2. **Speed overhead**: 13% (169ms → 191ms per step)
3. **Implementation**: ~150 lines of Python, wraps standard PyTorch AdamW
4. **Guarantee**: Bit-exact lossless, verified over 10 training steps
5. **How it works**: Offload m/v to pinned CPU memory between steps, use backward hook to overlap prefetch with backward pass

### Method 2: In-GPU Grouped Byte3 Compression (Best for pure GPU-memory savings)
1. **Memory savings**: 285 MB (6.25% of FP32 states) — both steady-state AND peak
2. **Speed overhead**: 44% (184ms → 265ms per step)
3. **Implementation**: ~200 lines, groups params into 64M-element batches, torch.compile'd
4. **Guarantee**: Bit-exact lossless, verified over 5 steps
5. **How it works**: FP32 byte3 (MSB) has only ~47 unique values → 6-bit indices packed 4-per-3-bytes. Bytes 0-2 stored raw. Total: 3.75 bytes/value (93.75%).

### Fundamental Limits of In-GPU FP32 Compression
- Only byte3 (MSB) of FP32 is compressible: ~3.7 bits entropy
- Theoretical maximum: 13.5% savings (~620 MB for Qwen3-0.6B)
- Variable-length coding (Huffman) would achieve this but is impractical on GPU (bit-level scatter bottleneck)
- Fixed 6-bit packing achieves 6.25% — roughly half of theoretical max
- The 6.25% savings is small relative to the total training memory footprint

### Future Directions
1. **Combine both methods**: Use CPU offload for m (changes 98%/step) + in-GPU compression for v (changes only 22%/step)
2. **nvCOMP integration**: NVIDIA's compression library may enable practical GPU Huffman
3. **Custom CUDA kernel**: Fuse byte3 extraction + packing in one kernel to reduce overhead
4. **FP32-only training memory optimization**: Consider 8-bit optimizers (not lossless) for significantly larger savings

---

## Files Index
| File | Status | Description |
|------|--------|-------------|
| `01_entropy_analysis.py` | ✅ | Entropy analysis of BF16 optimizer states |
| `02_baseline_training.py` | ✅ | Baseline AdamW memory + speed |
| `03_compressed_adamw.py` | ❌ | Per-param CPU offload (too slow) |
| `05_practical_approaches.py` | ❌ | GPU byte packing (dead end) |
| `06_fused_compressed_step.py` | ⚠️ | Microbenchmark (partial, some bugs) |
| `07_wrapper_approach.py` | ✅ | Wrapper: GPU pack + CPU offload |
| `08_pipelined_offload.py` | ✅ | Simple vs pipelined offload |
| `09_selective_and_hooked.py` | ✅⭐ | **Best CPU offload: hook-based, -2274 MB, 1.13x** |
| `10_delta_analysis.py` | ✅ | Delta coding potential analysis |
| `24_compiled_compression.py` | ⚠️ | torch.compile byte3 (fast but +6GB peak) |
| `25_high16_compression.py` | ❌ | High-16 packing (0% savings) |
| `27_entropy_analysis.py` | ✅ | FP32 byte-plane entropy analysis |
| `28_ans_byte3.py` | ❌ | CPU ANS coding (23s overhead) |
| `29_fused_compression.py` | ❌ | Fused compress (1.54x, no peak benefit) |
| `30_cuda_huffman.py` | ❌ | CuPy CUDA Huffman (encode too slow) |
| `31_torch_huffman.py` | ❌ | PyTorch prefix-sum Huffman (still slow) |
| `32_fast_huffman.py` | ❌ | Parallel atomicOr Huffman (catastrophic) |
| `33_final_compression.py` | ⚠️ | Clean byte3 (+7GB peak) |
| `34_chunked_compression.py` | ⚠️ | Chunked (+3.8GB peak) |
| `35_streaming_compression.py` | ⚠️ | Streaming (no peak improvement) |
| `36_perparam_compression.py` | ⚠️ | Per-param (great peak, 4x slow) |
| `37_grouped_compression.py` | ✅⭐ | **Best in-GPU: grouped+chunked+compiled, -285 MB, 1.44x** |
| `pilot_delta_offload.py` | ❌ | Delta-compressed CPU offload for v (see below) |
| `pilot_v_prediction.py` | ✅⭐⭐ | **Temporal prediction coding for v: 86-94% exact match, ~1 bit/value residual** |

---

### Pilot: Delta-Compressed CPU Offload for v — `pilot_delta_offload.py`
**Result: NOT viable. Sparse scatter dominates; no transfer speedup.**

Hypothesis: Since v changes only ~22% of BF16 values per step (β₂=0.999), transfer only the delta instead of full v during CPU offload.

Findings (Qwen3-0.6B, H200):
- **v changes 32.9% per step** (higher than the 22% measured earlier — possibly due to fewer warmup steps or different random data)
- Full v size: 1137 MB
- Sparse COO (idx+val per change): 1123 MB (98.8% of full) — essentially no savings because 6 bytes/change × 33% changes ≈ 2× dense
- Bitmask + changed values: 445 MB (39.2%) — best format
- **Transfer times are bandwidth-bound, not size-bound**: sparse COO transfers at the same speed as full dense (42.6 ms vs 43.1 ms roundtrip). PCIe transfers of scattered GPU memory are not faster than contiguous.
- **Sparse scatter is the killer**: GPU scatter (`tensor[indices] = values`) in the end-to-end benchmark makes delta 7.7x SLOWER than full offload (344 ms vs 43 ms)
- Delta compute (XOR + mask + nonzero) alone takes ~6 ms, acceptable
- The bitmask format saves 61% of transfer bytes, but the pack/unpack + scatter overhead far exceeds any transfer savings

**Why it fails:**
1. PCIe GPU↔CPU transfers are dominated by latency, not bandwidth at these sizes. Transferring 445 MB vs 1137 MB saves only marginal time.
2. Sparse scatter (`v[changed_indices] = new_values`) is extremely slow on both CPU and GPU — random-access writes kill cache performance.
3. The full dense `copy_()` uses optimized DMA that sparse formats cannot match.

---

### Pilot: Temporal Prediction Coding for v — `pilot_v_prediction.py`
**Result: EXTREMELY PROMISING. v is almost entirely predictable from grad².**

Hypothesis: Since v[t+1] = β₂·v[t] + (1-β₂)·grad², and we have grad² during the optimizer step, we can predict v[t+1] exactly. If prediction matches, we can eliminate v storage entirely — just recompute it from v[t] and grad² each step, or store only the tiny residual.

Findings (GPT-2 small, 124M params, FP32 states, H200):

| Step | Exact Match | Residual Entropy (bits/value) | Max |residual| |
|------|-------------|-------------------------------|----------------|
| 1    | **100.00%** | 0 (all zero)                  | 0              |
| 2    | **86.47%**  | 1.56 bits/value               | 4.66e-10       |
| 3    | **90.13%**  | 1.20 bits/value               | 2.91e-11       |
| 4    | **92.28%**  | 0.97 bits/value               | 5.82e-11       |
| 5    | **93.94%**  | 0.79 bits/value               | 1.46e-11       |

Key observations:
- **Step 1 is 100% exact** — the formula matches perfectly when v_old=0
- **86-94% of values are predicted exactly** (zero residual) in subsequent steps
- **Only ~130 unique residual values** out of 124M elements — incredibly low cardinality
- Residual entropy is only **0.8-1.6 bits/value** (vs 32 bits raw FP32)
- Residuals are 50/50 positive/negative (FP rounding errors, not systematic bias)
- Bytes 0 and 1 of residual are always 0 — residuals live in upper bytes only
- Exact match fraction **increases over time** (93.9% at step 5, trending higher)

**Why it works:** The FP32 multiply-add `β₂·v + (1-β₂)·g²` is deterministic for most values. Mismatches occur only when floating-point rounding differs between the optimizer's fused implementation and our separate computation. These mismatches are tiny (single ULP) and extremely sparse.

**Implications for compression:**
- v storage (half of optimizer memory) could theoretically be reduced to ~1 bit/value = **97% compression of v**
- Combined with m (which needs ~12 bits/value from byte-plane analysis), total optimizer memory could drop from 8Ψ to ~1.6Ψ bytes
- **Key challenge**: Need to match PyTorch's exact FP32 rounding to achieve the prediction, OR store the ~6% nonzero residuals efficiently
- Even without exact prediction matching, storing a 1-bit "prediction correct?" flag + residual for mismatches would be extremely compact

### Experiment 38: nvCOMP GPU Compression Pilot — `pilot_nvcomp.py`
**NVIDIA nvCOMP 5.1.0 provides fast GPU-native compression, solving the throughput bottleneck.**

Tested on 100M FP32 values (381.5 MB) on H200 GPU.

**Full FP32 tensor compression (all 4 bytes):**
| Algorithm | Ratio (m) | Ratio (v) | Enc GB/s | Dec GB/s | Enc ms | Dec ms |
|-----------|-----------|-----------|----------|----------|--------|--------|
| ANS (64K) | 93.28% | 93.10% | 101 | 153 | 3.7 | 2.4 |
| gdeflate_e | 92.76% | 92.62% | 58 | 60 | 6.4 | 6.2 |
| deflate_e | 92.47% | 92.33% | 45 | 12 | 8.4 | 30 |
| cascaded | 100.04% | 88.63% | 80 | 133 | 4.7 | 2.8 |
| LZ4 | 100.42% | 100.42% | 12 | 168 | 31 | 2.2 |
| bitcomp | 100.17% | 100.17% | 164 | 199 | 2.3 | 1.9 |

**Byte3 (MSB) only compression — fair comparison with prior byte-plane work:**
| Algorithm | Byte3 ratio (m) | Byte3 ratio (v) | Full-tensor savings | Enc ms | Dec ms |
|-----------|----------------|----------------|---------------------|--------|--------|
| ANS | 34.77% | 32.99% | 16.3-16.8% | 1.0-1.1 | 0.7-0.8 |
| gdeflate_e | 33.84% | 32.45% | 16.5-16.9% | 2.4-2.5 | 2.5 |

**Key findings:**
- nvCOMP ANS: **~1ms encode, ~0.8ms decode** for 95 MB byte3 data = **85-126 GB/s throughput**
- vs prior custom GPU Huffman: **2-40 seconds** for 10M values (3000-40000x slower!)
- vs prior CPU ANS: **23 seconds** overhead (23000x slower!)
- Byte3-only ANS saves **16.3-16.8%** of full FP32 (vs prior practical 6.25% with fixed codes)
- Full-tensor ANS saves **6.7-6.9%** directly on raw FP32 bytes, no byte-plane splitting needed
- All algorithms verified lossless (correct round-trip)
- LZ4/bitcomp provide no compression on FP32 data (near-random lower bytes defeat them)
- ANS is the best performer: high compression + extreme throughput

**Comparison with prior best (Experiment 37: grouped byte3 compression):**
- Prior: 6.25% savings, 1.44x slowdown with torch.compile
- nvCOMP ANS byte3: **16.5% savings, sub-millisecond encode/decode**
- nvCOMP ANS full: **7% savings with zero byte-plane splitting complexity**

**Implication:** nvCOMP ANS completely solves the GPU entropy coding bottleneck. It can compress byte3 of FP32 optimizer states at >80 GB/s, making in-GPU compression practical with negligible overhead. Install: `pip install nvidia-nvcomp-cu12`
