# Full-Value Entropy Coding as an Oracle Baseline for Lossless Compression of Low-Precision LLM Weights

## Abstract

Lossless compression of low-precision LLM weights is increasingly important for storage, distribution, and memory efficiency, yet many LLM-specific codecs operate on decomposed fields (e.g., exponents only) rather than the full value distribution. This paper studies full-value entropy coding as the oracle baseline for BF16, FP8, and INT4 weight compression, and quantifies the penalty of practical factorized codecs on the studied value streams. We find that full-value ANS reaches near-entropy-optimal compression: BF16 consistently achieves ~66% across model families, while FP8 compressibility varies strongly by model (Qwen ~70%, Mistral ~37% in e4m3fn) and by format (e4m3fn ~38% vs e5m2 ~69% for Mistral). INT4 value-stream compression ranges from 74% on simulated symmetric quantization to 85% on real GPTQ qweight data, indicating strong dependence on checkpoint layout. We further present a lossless fixed-width two-stream FP8 codec that achieves 77.1% compression with 254 GB/s standalone per-layer decode and 584 GB/s pre-concatenated batched decode on NVIDIA H200, evaluated as a standalone decode microbenchmark rather than an end-to-end inference system. Our results suggest that the central question is not which single codec universally dominates, but how much compressibility exists in each format and how much practical GPU-decodable codecs leave on the table.

---

## 1. Introduction

The deployment of large language models at scale has made weight storage and transmission a practical bottleneck. A single 70B-parameter model occupies 140 GB in BFloat16 or 70 GB in FP8—costs that multiply across model variants, checkpoints, and serving replicas. Lossless compression offers bit-exact model preservation while reducing these costs, and recent work has demonstrated 25–30% size reductions for BFloat16 weights through entropy coding of the exponent field [DFloat11, ZipNN, ECF8].

However, existing lossless codecs share a common design choice: they decompose floating-point values into subfields (sign, exponent, mantissa) and compress each separately, typically focusing on the low-entropy exponent while storing the high-entropy mantissa raw. This factorization is motivated by GPU decode efficiency—exponent-only Huffman codes are short and decodable via compact lookup tables—but it implicitly discards any joint structure between fields.

This paper takes a different perspective. We ask: **how much compression is theoretically available for each format, and how much do practical factorized codecs leave on the table?** To answer this, we establish full-value entropy coding (treating the entire BF16/FP8/INT4 value as a single symbol) as an oracle baseline, then systematically compare it against factorized alternatives.

Our contributions are:

1. **Oracle baseline**: We show that full-value ANS coding reaches within 0.0–0.6 percentage points of the Shannon entropy limit on the studied BF16, FP8, and INT4 value streams from multiple model families.

2. **Format-dependent analysis**: We demonstrate that the penalty of factorized codecs is small for BF16 (~0.1pp) but meaningful for FP8 (2.72pp vs ECF8-style Huffman), and that FP8 compressibility itself varies dramatically by model (Qwen 70% vs Mistral 37%) and by FP8 format (e4m3fn vs e5m2).

3. **Practical GPU codec**: We present a two-stream FP8 codec with fixed-width exponent codes and packed sign+mantissa, achieving 77.1% ratio at 254 GB/s per-layer GPU decode throughput on H200—34× faster than Huffman-based decoding at comparable ratio.

---

## 2. Background and Problem Setup

### 2.1 Weight Precision Formats

**BFloat16 (BF16)**: 1 sign + 8 exponent + 7 mantissa = 16 bits. The de facto standard for LLM training and inference. Exponent entropy is ~2.6 bits (out of 8), making it the primary compression target.

**FP8 e4m3fn**: 1 sign + 4 exponent + 3 mantissa = 8 bits. Increasingly adopted for inference (DeepSeek-V3, Llama 3.x). Smaller exponent and mantissa ranges create different compressibility characteristics.

**FP8 e5m2**: 1 sign + 5 exponent + 2 mantissa = 8 bits. Wider dynamic range, used for gradients and some inference paths.

**INT4**: 4-bit integer, typically from post-training quantization (GPTQ, AWQ). Packed 2 values per byte with per-group scales and zero-points.

### 2.2 Entropy Accounting

We measure compression quality using:
- **Compression ratio**: compressed_size / original_size × 100% (lower is better)
- **Bits per value (bpv)**: average bits to store one weight value
- **Gap to entropy**: difference between achieved bpv and Shannon entropy H(X)

All reported sizes include metadata (symbol tables, probability arrays, block headers). We use shared frequency tables across layers of the same weight type unless otherwise stated.

### 2.3 Related Work

**Exponent-only coding**: DFloat11 [1] Huffman-codes BF16 exponents (~66.6% ratio) with a GPU kernel using hierarchical lookup tables. ZipNN [2] similarly targets exponents for HuggingFace integration.

**Hardware-aware fixed-width**: ZipServ [3] introduces TCA-TBE, a 3-bit fixed-length encoding fused into Tensor Core GEMM, achieving ~70% ratio with 1.22× inference speedup—the first lossless compression to accelerate serving.

**FP8 compression**: ECF8 [4] extends exponent Huffman coding to FP8, reporting 9.8–26.9% memory savings. Float8@2bits [5] uses ANS on FP8 for sub-2-bit lossy compression.

**INT4 compression**: EntroLLM [6] applies Huffman to INT4 quantized weights for edge devices. QStore [7] jointly stores multi-precision checkpoints.

**GPU compression primitives**: DietGPU [8] provides GPU ANS at 250–410 GB/s; nvCOMP offers LZ4/ANS/Snappy codecs.

None of these works systematically compare full-value coding against factorized alternatives across formats, or characterize the model- and format-dependence of FP8 compressibility.

---

## 3. Oracle Baseline: Full-Value Entropy Coding

### 3.1 Method

We treat each weight value as a single symbol in its native representation and apply asymmetric numeral systems (ANS) coding via the constriction library [9]. For BF16, symbols are 16-bit integers (typical alphabet: ~6,000 unique values). For FP8, symbols are 8-bit bytes (~100–120 unique). For INT4, symbols are 4-bit nibbles (16 possible values).

The frequency table is built from the empirical distribution of all weights of the same type (e.g., all q_proj weights across layers). This shared-table approach keeps metadata negligible: ~1 KB for BF16's 6,000 symbols vs ~GB of compressed data.

### 3.2 Main Results

**Table 1a: BF16 full-value ANS (casted value streams)**

| Model | Params | Ratio | Entropy (bpv) | Gap to H(X) |
|-------|--------|-------|---------------|-------------|
| Qwen3-0.6B | 596M | 66.53% | 10.56 | 0.57pp |
| Qwen3-8B | 8.2B | 66.10% | 10.56 | 0.10pp |
| Mistral-7B | 7.2B | 65.73% | 10.51 | 0.03pp |

**Table 1b: FP8 e4m3fn full-value ANS (BF16→FP8 casted value streams)**

| Model | Ratio | Entropy (bpv) | Unique vals | Gap to H(X) |
|-------|-------|---------------|-------------|-------------|
| Qwen3-0.6B | 70.66% | 5.65 | ~117 | 0.00pp |
| Qwen3-8B | 69.09% | 5.53 | ~100 | ~0.0pp |
| Mistral-7B | **37.35%** | **3.05** | **25–81** | ~0.0pp |

**Table 1c: INT4 full-value ANS**

| Model | Type | Ratio | Entropy (bpv) | Gap to H(X) |
|-------|------|-------|---------------|-------------|
| Qwen3-0.6B | Simulated symmetric | 74.06% | 2.97 | 0.24pp |
| Mistral-7B | Simulated symmetric | 70.84% | 2.83 | ~0.04pp |
| Qwen2.5-7B | Real GPTQ qweight | 85.0% | 3.41 | ~0.8pp |

Note: INT4 simulated results use per-channel symmetric quantization to INT4 then ANS on unpacked 4-bit values. GPTQ result is on actual serialized qweight tensors (asymmetric quantization, group size 128). Total real GPTQ checkpoint including scales/zeros: 81.7%.

All results verified bit-exact lossless. BF16 is consistent (~66%) across model families. FP8 shows striking variability: Mistral compresses to 37.4% in e4m3fn versus Qwen's 70.4%. The FP8 results use identical BF16→FP8 casting (torch.to(float8_e4m3fn)) across all models.

### 3.3 FP8 Ablation: Where Does the Gain Come From?

**Table 2: FP8 coding strategy comparison (Qwen3-0.6B, same weights)**

| Method | Ratio | bpv | Gap to H(X) |
|--------|-------|-----|-------------|
| Full-value ANS (ours) | 70.66% | 5.653 | +0.000 |
| Separated ANS (exp + s+m) | 71.24% | 5.699 | +0.047 |
| Exp ANS + raw sign+mantissa | 71.55% | 5.724 | +0.071 |
| ECF8-style (Exp Huffman + raw) | 73.38% | 5.870 | +0.218 |
| Byte-wise ANS (1K blocks) | 97.41% | 7.793 | +2.141 |

The 2.72pp gap between full-value ANS and ECF8-style decomposes into: Huffman integer-length rounding loss (0.17 bpv), exponent–mantissa joint correlation (0.05 bpv), and stream separation/metadata overhead (remainder).

---

## 4. Empirical Analysis of FP8 Compressibility

### 4.1 Model Dependence

FP8 compressibility varies dramatically across model families. Mistral-7B weights occupy a narrow BF16 range (±0.05–0.2), which collapses to only 25–81 unique FP8 e4m3fn values per layer (mean entropy: 3.05 bits). Qwen models have broader weight ranges, yielding ~100–117 unique FP8 values (mean entropy: 5.6 bits).

**Table 3: Per-model FP8 statistics**

| Model | Unique FP8 vals | Mean entropy | Ratio |
|-------|----------------|-------------|-------|
| Qwen3-0.6B | ~117 | 5.61 bits | 70.66% |
| Qwen3-8B | ~100 | 5.53 bits | 69.09% |
| Mistral-7B | 25–81 | 3.05 bits | 37.35% |

This variability is invisible to exponent-only codecs, which see similar exponent distributions regardless of the mantissa concentration. Full-value coding correctly captures it.

### 4.2 Format Dependence

The choice of FP8 format also affects compressibility. For Mistral-7B:

| FP8 Format | Mean entropy | Estimated ratio |
|------------|-------------|----------------|
| e4m3fn | 3.05 bits | ~38% |
| e5m2 | 5.52 bits | ~69% |

The e4m3fn format has 3 mantissa bits, creating more value collisions from concentrated weights than e5m2's 2 mantissa bits with wider exponent range. This interaction between weight distribution and FP8 format design has not been characterized in prior work.

### 4.3 BF16 and INT4

BF16 compression is consistent across models (~66%, gap <0.6pp), confirming that exponent-only codecs like DFloat11 are already near-optimal for this format.

INT4 compression depends on the quantization pipeline: simulated symmetric quantization yields 70–74% (more concentrated around zero-point), while real GPTQ checkpoints yield ~85% due to asymmetric quantization and group structure.

---

## 5. Practical FP8 GPU Codec

### 5.1 Two-Stream Design

To bridge the gap between entropy-optimal compression (CPU-only, ~70%) and GPU-decodable throughput, we design a fixed-width two-stream codec for FP8 weights.

**Encoding** (offline, CPU):
1. Find the best window of k=3 consecutive exponent values covering ~96% of weights
2. Encode exponents as 2-bit offsets (0,1,2 = in window; 3 = escape), packed 4 per byte
3. Pack sign+mantissa as 4-bit values, 2 per byte
4. Store escape exponents in a packed overflow buffer
5. Precompute per-block escape prefix sums

**Decoding** (GPU, CUDA):
- Each thread decodes 16 elements in constant time
- Fixed-width codes enable branch-free common-case decoding
- Escape indices computed via warp-cooperative prefix sum (`__shfl_up_sync`)
- Vectorized 4×uint32 writes for coalesced memory access
- Batched mode: single kernel launch for all model layers

### 5.2 Results

**Table 4: FP8 GPU decode comparison (Qwen3-0.6B, all on H200, our measurements)**

| Method | Ratio | Per-layer GB/s | Batched GB/s | Lossless |
|--------|-------|---------------|-------------|----------|
| Dense FP8 (memcpy) | 100.0% | 443 | — | N/A |
| **Two-Stream v5 (ours)** | **77.1%** | **254** | **584** | **All pass** |
| Huffman (DFloat11-style, ours) | 78.0% | 7.4 | — | Pass |
| nvCOMP byte ANS (prior pilot) | 85.7% | 29–56 | — | Pass |
| Full-value ANS (CPU only) | 70.4% | — | — | Pass |

*Timing protocol*: All GPU numbers measured via CUDA events on idle H200. Per-layer: median of 50 trials per layer, aggregated as total_bytes / total_time across 197 layers. Batched: single kernel launch with pre-concatenated compressed buffers resident on GPU; median of 50 trials. The batched number reflects amortized kernel launch overhead and is not directly comparable to layer-at-a-time inference decode. nvCOMP numbers are from a prior pilot experiment (same H200, nvidia-nvcomp-cu12 v5.1.0) and could not be re-measured due to API instability.

The two-stream codec is 34× faster than Huffman at comparable ratio. In batched mode, it exceeds dense memcpy throughput (584 vs 443 GB/s) because it reads 23% less data from HBM.

### 5.3 Negative Results

**Fused decode+GEMM**: A Triton prototype that decodes compressed FP8 tiles in shared memory before WGMMA achieved only 0.4–0.6× the throughput of dense FP8 GEMM. Decode overhead in shared memory is not free, even with fixed-width codes.

**NVFP4**: FP4 (e2m1) weights have only 8.5% additional compressibility (3.66/4.0 bits entropy) — the per-channel scaling normalizes distributions too uniformly.

---

## 6. Discussion, Limitations, and Conclusion

### Limitations

- **No end-to-end inference speedup**: Our GPU codec is a standalone decode benchmark, not an integrated serving system. The decode throughput is high, but translating this to end-to-end speedup requires fusion with GEMM kernels, which remains an open problem for variable-ratio codecs.
- **Limited real FP8 checkpoints**: Our FP8 analysis uses BF16→FP8 casting, not natively FP8-trained models. The model-dependence finding should be validated on real FP8 checkpoints as they become available.
- **INT4 coverage is narrow**: One real GPTQ checkpoint is sufficient to demonstrate the approach but not to claim comprehensive INT4 coverage across quantizers.

### Conclusion

Full-value entropy coding provides the correct oracle baseline for evaluating lossless compression of low-precision LLM weights. Our analysis reveals that the penalty of practical factorized codecs is format-dependent: negligible for BF16, meaningful for FP8 (2.72pp), and layout-dependent for INT4. The striking model-dependence of FP8 compressibility (37–70% in e4m3fn) and format-dependence (e4m3fn vs e5m2) are new empirical findings that factorized codecs obscure. The two-stream FP8 codec provides a practical decode throughput of 254–584 GB/s at 77.1% ratio, positioning it on the rate–throughput Pareto frontier between entropy-optimal CPU coding and hardware-native dense storage.

---

## References

[1] Zhang et al. "DFloat11: 70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float." NeurIPS 2025.

[2] Hershcovitch et al. "ZipNN: Lossless Compression for AI Models." arXiv:2411.05239, 2024.

[3] Fan et al. "ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression." ASPLOS 2026.

[4] Yang et al. "To Compress or Not? Pushing the Frontier of Lossless GenAI Model Weights Compression with Exponent Concentration." arXiv:2510.02676, 2025.

[5] Putzky et al. "Float8@2bits: Entropy Coding Enables Data-Free Model Compression." arXiv:2601.22787, 2026.

[6] Sanyal et al. "EntroLLM: Entropy Encoded Weight Compression for Efficient Large Language Model Inference on Edge Devices." arXiv:2505.02380, 2025.

[7] Shah et al. "QStore: Quantization-Aware Compressed Model Storage." arXiv:2505.04081, 2025.

[8] Johnson et al. "DietGPU." github.com/facebookresearch/dietgpu.

[9] Bamler. "constriction: Entropy Coders for Research and Production." github.com/bamler-lab/constriction.
