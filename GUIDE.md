# Algorithm Guide & Reproduction Instructions

This document explains the core algorithms, codebase structure, and how to reproduce all results.

---

## 1. Project Overview

This project explores **lossless compression of LLM weights** across multiple precision formats (BF16, FP8, INT4). The key finding is that treating the entire numerical value as a single entropy-coding symbol ("full-value coding") achieves near-entropy-optimal compression, and the gap between this optimum and hardware-friendly factorized codecs is **format- and model-dependent**.

### Headline Results

| Format | Method | Ratio | Decode GB/s | Notes |
|--------|--------|-------|-------------|-------|
| BF16 | Full-value ANS | 66.1% | CPU only | Within 0.1pp of entropy |
| FP8 (e4m3fn) | Full-value ANS | 70.4% | CPU only | Entropy optimal |
| FP8 (e4m3fn) | Two-Stream GPU | 77.1% | 254 per-layer / 584 batched | Fixed-width, lossless |
| INT4 (real GPTQ) | Full-value ANS | 85.0% | CPU only | On real checkpoint |

---

## 2. Core Algorithms

### 2.1 Full-Value ANS Codec (`experiments/new_compression/codec_ans16.py`)

**Idea**: Instead of separating BF16 into exponent (Huffman) + mantissa (raw) like DFloat11, treat the entire 16-bit BF16 value as a single ANS symbol.

**Why it works**: BF16 LLM weights use only ~5,000–7,000 unique 16-bit values. ANS can code these at essentially the Shannon entropy limit.

**Pipeline**:
```
bf16 tensor
  → view as int16, flatten
  → count unique values, build frequency table
  → map to contiguous indices [0, N_unique)
  → ANS encode (constriction library, rANS)
  → store: compressed_words (uint32[]) + symbol_table (int16[]) + probabilities (float32[])
```

**Decode**: reverse the ANS, map indices back to int16 values, view as bf16.

**Key property**: The symbol table + probability overhead is negligible (~1 KB for 6,000 symbols) vs the data (~GB). This is because we use a shared table across all layers of the same weight type.

### 2.2 Multi-Format ANS Codec (`experiments/new_compression/codec_multiformat.py`)

Extends codec_ans16.py to handle FP8 (e4m3fn, e5m2), packed INT4 (uint8), and INT8. The core logic is identical — only the symbol-to-integer mapping changes per format.

For packed INT4: each uint8 byte contains two 4-bit values. The codec unpacks to individual nibbles (0–15), codes them with ANS, then repacks on decode.

### 2.3 FP8 Two-Stream GPU Codec (`experiments/fused_codec/fp8_twostream_v5.py`)

**The production GPU kernel.** Achieves 77.1% ratio at 584 GB/s batched decode.

**Idea**: Separate FP8 (e4m3fn) into two fixed-width streams for GPU-friendly constant-time decoding:

```
FP8 byte: [S | E3 E2 E1 E0 | M2 M1 M0]
              ↓                    ↓
    Exponent stream          Sign+Mantissa stream
    (2-bit codes, 4/byte)    (4-bit values, 2/byte)
```

**Encoding** (CPU, offline):
1. Find best consecutive exponent window of k=3 values (covers ~96% of weights)
2. Map exponents: if in window → 2-bit offset (0,1,2); else → escape code (3)
3. Pack exponent codes: 4 per byte (bits 7-6, 5-4, 3-2, 1-0)
4. Pack sign+mantissa: 2 per byte (high nibble, low nibble)
5. Escape exponents: packed in overflow buffer (2 per byte)
6. Precompute per-block escape prefix sums for GPU decode

**Decoding** (GPU, CUDA kernel):
1. Each thread decodes 16 elements in constant time
2. Read 4 bytes of exp_packed → extract 16 × 2-bit codes
3. Read 8 bytes of sm_packed → extract 16 × 4-bit sign+mantissa values
4. For escapes: warp-cooperative prefix sum via `__shfl_up_sync` + block-level scan
5. Reconstruct: `output[i] = (sign << 7) | (exponent << 3) | mantissa`
6. Write 16 bytes as 4 × uint32 (vectorized, coalesced)

**Batched mode**: Single kernel launch for ALL layers via `block_to_layer` / `block_to_local` indirection arrays. Eliminates per-layer kernel launch overhead (2x speedup: 254→584 GB/s).

**Compression breakdown**:
- Exponent stream: n/4 bytes (25%)
- Sign+mantissa stream: n/2 bytes (50%)
- Overflow: ~4% × n/2 bytes (~2%)
- Metadata: negligible
- **Total: ~77% of dense FP8**

### 2.4 FP8 Ablation (`experiments/fused_codec/fp8_ablation.py`)

Compares five coding strategies on identical FP8 weights:

| Method | Ratio | Gap to Entropy |
|--------|-------|----------------|
| Full-value ANS | 70.66% | +0.000 bpv |
| Separated ANS (exp + s+m) | 71.24% | +0.047 bpv |
| Exp ANS + raw sign+mantissa | 71.55% | +0.071 bpv |
| ECF8-style (Exp Huffman + raw) | 73.38% | +0.218 bpv |
| Byte-wise ANS (1K blocks) | 97.41% | +2.141 bpv |

The 2.72pp gap between full-value ANS and ECF8-style comes from: Huffman rounding loss (0.17 bpv) + joint coding gain (0.05 bpv) + stream separation overhead.

---

## 3. File Structure

```
experiments/
├── new_compression/                # CPU-side entropy codecs
│   ├── codec_ans16.py             # BF16 full-value ANS (best ratio)
│   ├── codec_multiformat.py       # Multi-format ANS (BF16/FP8/INT4)
│   ├── gpu_codec.py               # BF16 GPU Huffman decoder (CuPy)
│   ├── gpu_decompress.py          # nvCOMP GPU ANS wrapper
│   ├── compress_llm.py            # CLI: compress a model
│   ├── validate_llm.py            # CLI: validate compressed model
│   └── sensitivity_compress.py    # Output-preserving compression
│
├── fused_codec/                   # GPU-optimized FP8 codec
│   ├── fp8_twostream_v5.py       # ⭐ BEST: Two-Stream GPU decode (77.1%, 584 GB/s)
│   ├── fp8_twostream_approx.py   # Near-lossless variant (75%, 466 GB/s)
│   ├── fp8_fused_huffman.py       # FP8 Huffman GPU decode (baseline)
│   ├── fp8_fused_gemm.py          # Fused decode+GEMM prototype (Triton)
│   ├── fp8_hybrid_ans.py          # Hybrid ANS on exp stream (73.4%, CPU)
│   ├── benchmark_all.py           # ⭐ Comprehensive benchmark suite
│   ├── benchmark_baselines.py     # Huffman + nvCOMP baselines
│   ├── profile_fp8.py             # FP8 distribution profiling
│   └── EXPERIMENT.md              # Full experiment log (12 experiments)
│
├── benchmark_multiformat.py       # Cross-model BF16/FP8/INT4 benchmark
├── benchmark_llama.py             # Mistral-7B benchmark
├── benchmark_real_int4.py         # Real GPTQ INT4 benchmark
│
├── optimizer_compression/         # Optimizer state compression (secondary)
│   ├── optimization_log.md        # Full experiment log
│   └── ...                        # Best methods + analysis scripts
│
└── optimization_log.md            # Weight compression exploration log
```

---

## 4. Reproducing Results

### Prerequisites

```bash
conda activate quant

# Core dependencies
pip install constriction dahuffman==0.4.2 cupy-cuda12x safetensors transformers

# Optional (for nvCOMP baselines)
pip install nvidia-nvcomp-cu12
```

### 4.1 BF16/FP8/INT4 Compression Ratios (Table 1)

```bash
# Qwen3-0.6B + Qwen3-8B (BF16 + FP8 + simulated INT4)
python experiments/benchmark_multiformat.py

# With larger models (adds Llama/Mistral if available)
FULL_BENCHMARK=1 python experiments/benchmark_multiformat.py

# Mistral-7B specifically
python experiments/benchmark_llama.py

# Real GPTQ INT4 (Qwen2.5-7B-Instruct-GPTQ-Int4)
python experiments/benchmark_real_int4.py
```

Expected output: per-layer and aggregate compression ratios, verified lossless.

### 4.2 FP8 Ablation (Table 2)

```bash
python experiments/fused_codec/fp8_ablation.py
```

Compares 5 coding strategies on identical FP8 weights. Shows exact entropy gap decomposition.

### 4.3 FP8 GPU Decode Throughput (Table 3)

```bash
# Full benchmark: Two-Stream v5 per-layer + batched + baselines
python experiments/fused_codec/benchmark_all.py

# Huffman + nvCOMP baselines separately
python experiments/fused_codec/benchmark_baselines.py

# Just the Two-Stream v5 kernel
python experiments/fused_codec/fp8_twostream_v5.py
```

Key numbers:
- Two-Stream v5 per-layer: ~254 GB/s (idle GPU)
- Two-Stream v5 batched: ~584 GB/s
- Huffman (DFloat11-style): ~7.4 GB/s
- Dense FP8 memcpy: ~443 GB/s

### 4.4 FP8 Distribution Profiling

```bash
python experiments/fused_codec/profile_fp8.py
```

Shows per-layer exponent coverage (p7, p15, p31, p63), entropy, and optimal palette sizes.

### 4.5 Cross-Model FP8 Variability

The Mistral FP8 result (37.35%) can be reproduced via:
```bash
python experiments/benchmark_llama.py
```

Mistral weights are more concentrated (BF16 range ±0.05–0.2), yielding only 25–81 unique FP8 e4m3fn values vs Qwen's ~117.

---

## 5. Key Findings Summary

1. **Full-value ANS is the entropy oracle** for lossless weight compression. It reaches within 0.0–0.6pp of Shannon entropy across BF16, FP8, and INT4.

2. **The penalty of factorized codecs is format-dependent**:
   - BF16: negligible (~0.1pp, DFloat11 is nearly optimal)
   - FP8: meaningful (2.72pp vs ECF8-style Huffman on exponents)
   - INT4: layout-dependent (real GPTQ at 85% vs simulated at 74%)

3. **FP8 compressibility is highly model-dependent**: Mistral-7B compresses to 37.35% in e4m3fn (3.0 bits entropy) vs Qwen's 70.4% (5.65 bits). This is because concentrated weight distributions collapse to fewer unique FP8 values.

4. **FP8 compressibility is format-dependent**: Mistral e4m3fn = 3.0 bits, Mistral e5m2 = 5.5 bits. The 3-bit mantissa of e4m3fn creates more value collisions than e5m2's 2-bit mantissa with wider exponent range.

5. **Two-Stream GPU codec achieves practical throughput**: 77.1% ratio at 254 GB/s per-layer (584 GB/s batched), verified lossless. 34x faster than Huffman at similar ratio.
