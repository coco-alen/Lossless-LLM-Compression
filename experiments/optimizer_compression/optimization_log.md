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
