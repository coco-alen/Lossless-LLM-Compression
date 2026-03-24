# Raw Idea Generation: Lossless LLM Weight Compression

**Date**: 2026-03-24
**Direction**: Lossless compression of LLM weights for storage and inference
**Prior work in this project**: ANS-16bit at 65.96% (i.i.d. entropy limit), exhaustive context-dependent coding exploration (all failed)

---

## Landscape Context

| Method | BF16 Ratio | Speed | Venue |
|--------|-----------|-------|-------|
| DFloat11 | ~66.6% | Huffman GPU kernel, 1.56-3.44x GEMM overhead | NeurIPS '25 |
| Our ANS-16bit | 65.96% | CPU only (no GPU kernel) | This project |
| ZipServ TCA-TBE | ~70% | **Fused ZipGEMM, 1.22x speedup** | ASPLOS '26 |
| ZipNN | ~67% | CPU Huffman | Nov 2024 |
| ECF8 | FP8: 9.8-26.9% savings | GPU Huffman kernel | Oct 2025 |

**Fundamental limit**: BF16 i.i.d. entropy = 65.95%. Our ANS-16bit is at 65.96% (gap 0.01%).
**Bigram entropy**: 9.91 bpw (61.91%) — 4% theoretical savings from sequential correlation, but overhead kills all practical approaches.
**Mantissa**: ~8 bits entropy (incompressible).

---

## Generated Ideas (10)

### Idea 1: GPU-Native ANS-16bit Decompression Kernel
**Hypothesis**: Our ANS-16bit codec achieves the best compression ratio (65.96%) but has no GPU implementation. Building a parallel GPU ANS decoder (inspired by DietGPU's interleaved-stream design) would make this practically deployable for inference.
**Minimum experiment**: Implement a CUDA kernel using interleaved rANS streams (4-8 streams per warp). Each stream decodes independently. Benchmark decode throughput vs DFloat11's Huffman kernel.
**Expected outcome**: 100-400 GB/s decode throughput (DietGPU achieves 250-410 GB/s for generic ANS on A100). Combined with 0.66% better compression than DFloat11, this would be state-of-the-art.
**Risk**: MEDIUM — interleaved ANS is proven (DietGPU) but adapting to 16-bit symbols needs careful engineering.
**Effort**: 1-2 weeks
**Contribution**: Method (GPU kernel) + empirical (compression + speed comparison)
**Novelty**: 7/10 — DietGPU does generic ANS, but nobody has done 16-bit-symbol ANS for LLM weights with inference integration.

### Idea 2: Adaptive Fixed-Width Entropy Coding
**Hypothesis**: ZipServ's TCA-TBE uses fixed 3-bit codewords for all layers (top-7 consecutive exponents cover >95%). But different layers have different exponent distributions — some layers have >99% concentration in top-3 exponents (2-bit suffices), others need top-15 (4-bit). An adaptive fixed-width scheme that selects optimal bit-width per layer achieves better compression than TCA-TBE while maintaining constant-time parallel decoding.
**Minimum experiment**: Analyze exponent distributions of all layers in Llama-3-8B, Qwen3-8B, Mistral-7B. Compute optimal fixed-width per layer. Simulate compression ratio.
**Expected outcome**: 1-3% better compression than TCA-TBE (from ~70% to ~67-69%) while maintaining O(1) decode. Closes the gap between TCA-TBE and entropy-optimal.
**Risk**: LOW — analysis-driven, minimal implementation risk.
**Effort**: 2-3 days analysis, 1 week for GPU kernel
**Contribution**: Method (adaptive encoding) + empirical (per-layer analysis)
**Novelty**: 8/10 — ZipServ fixed 3-bit is the only fixed-length approach. Adaptive per-layer is unexplored.

### Idea 3: Lossless Compression of Quantized (INT4/FP8) Model Weights
**Hypothesis**: GPTQ/AWQ INT4 models have only 16 possible values per weight, but the actual distribution is highly non-uniform (some values appear 10-100x more than others). Entropy coding on top of quantization could save an additional 15-30%, stacking with the 4x from quantization for ~5x total compression.
**Minimum experiment**: Download GPTQ INT4 models, analyze per-channel weight distributions, compute entropy, test if ANS or Huffman gives meaningful savings on packed INT4 data.
**Expected outcome**: INT4 weights likely have 2.5-3.5 bits entropy (out of 4), giving 10-25% additional savings. For a 4-bit Llama-3-70B (~35 GB), this saves 3.5-8.75 GB — significant for fitting on smaller GPUs.
**Risk**: LOW — analysis is trivial, high chance of positive signal.
**Effort**: 1-2 days analysis, 1 week for codec
**Contribution**: Empirical finding + method (codec for quantized weights)
**Novelty**: 9/10 — nobody has done lossless entropy coding on top of already-quantized INT4/FP8 weights.

### Idea 4: Cross-Model Delta Compression for Multi-LoRA Serving
**Hypothesis**: Fine-tuned model variants share >90% of weights with the base model. For multi-tenant serving (many LoRA adapters), store base model compressed + XOR delta per adapter. The delta is extremely sparse (only LoRA-modified layers differ), achieving 90-99% reduction per adapter.
**Minimum experiment**: Take a base model + 3-4 LoRA fine-tunes, compute XOR delta, measure entropy of delta, estimate total memory for serving N adapters.
**Expected outcome**: Each adapter adds only 1-5% of base model size (vs 100% for full copies). Enables serving 10-50 adapters in the memory of 2 full models.
**Risk**: LOW — the math is straightforward; the challenge is in the serving integration.
**Effort**: 3-5 days
**Contribution**: System/method (multi-LoRA serving with compression)
**Novelty**: 7/10 — ZipLLM does cross-model delta for storage; live inference with delta is new.

### Idea 5: Tiled Block Compression for Tensor Cores
**Hypothesis**: Weight matrices can be compressed in tiles matching GPU Tensor Core fragment sizes (16×8 for BF16 mma). Within each tile, exponents are locally similar. A shared-exponent + per-element offset format is lossless and enables direct Tensor Core consumption without full decompression.
**Minimum experiment**: Analyze exponent variance within 16×8 tiles across several LLMs. If max-min exponent range within tiles is ≤ 7, a 3-bit offset suffices (saving 5 bits per exponent, ~31% compression).
**Expected outcome**: Tiles likely have 3-8 exponent range, allowing 3-4 bit offsets. Combined with mantissa packing, ~25-30% savings with tile-aligned decode.
**Risk**: MEDIUM — exponent locality within tiles may not be strong enough.
**Effort**: 1 week
**Contribution**: Method (tile-aligned compression format)
**Novelty**: 7/10 — block floating point exists but tile-aligned for Tensor Cores is new.

### Idea 6: Finite-State Entropy (FSE) for Breaking the i.i.d. Barrier
**Hypothesis**: FSE (tANS, used in zstd) uses finite state machines as implicit context models. With a small state (8-16 states), it can capture sequential correlations without explicit per-context probability tables. For BF16 weights with 9.91 bpw bigram entropy vs 10.55 bpw i.i.d., FSE could capture some of this 0.64 bpw gap without the prohibitive table overhead.
**Minimum experiment**: Implement tANS with 8/16/32 states for BF16 weight streams. Compare compression ratio with i.i.d. ANS. Measure if state machines capture any sequential correlation.
**Expected outcome**: Uncertain. FSE might capture 0.1-0.3 bpw of the 0.64 bpw gap, giving 65.0-65.5% ratio (vs 65.96% i.i.d.). But the sequential nature of FSE limits GPU parallelism.
**Risk**: HIGH — FSE is inherently sequential; GPU parallelism requires stream interleaving which may limit context modeling.
**Effort**: 1 week
**Contribution**: Empirical finding (can FSE beat i.i.d. for weights?)
**Novelty**: 8/10 — nobody has tried FSE/tANS for neural network weight compression.

### Idea 7: Unified Multi-Precision Lossless Compression Framework
**Hypothesis**: Models increasingly use mixed precision (BF16 base + FP8 attention + INT4 FFN). No single codec handles all formats. A unified framework with format-aware entropy coding (ANS-16 for BF16, ANS-8 for FP8, ANS-4 for INT4) could handle any model and outperform format-specific solutions.
**Minimum experiment**: Build a safetensors-compatible codec that auto-detects tensor dtype and applies optimal entropy coding. Test on mixed-precision models.
**Expected outcome**: Consistent 20-30% savings across all precision formats, with one tool.
**Risk**: LOW — each component exists; the novelty is the unified framework.
**Effort**: 2 weeks
**Contribution**: System/framework
**Novelty**: 6/10 — engineering contribution, less algorithmic novelty.

### Idea 8: Learned Probability Model for Weight Compression
**Hypothesis**: Instead of using empirical frequency tables (which are optimal for i.i.d.), use a tiny learned neural network to predict the next weight's distribution given context (position, layer, previous weights). Even a 1-layer MLP with <1KB parameters could capture structural patterns that frequency tables miss.
**Minimum experiment**: Train a tiny context model on one layer's weights, use it as a probability source for arithmetic coding. Compare bits-per-weight with i.i.d. entropy.
**Expected outcome**: Could capture 0.1-0.5 bpw of the bigram gap (64.5-65.5% ratio) without large tables. But decoding speed would be slow (model inference per symbol).
**Risk**: HIGH — model must run at decode time, making GPU inference very slow.
**Effort**: 1 week
**Contribution**: Method + empirical (learned compression for weights)
**Novelty**: 9/10 — learned compression for images is huge; applying to NN weights is unexplored.

### Idea 9: Exponent-Mantissa Factored ANS with Shared Tables
**Hypothesis**: Instead of 16-bit ANS (65,536 possible symbols, ~6000 unique), factor into exponent ANS (256 symbols, ~30 unique) + conditional mantissa ANS (256 symbols per exponent). The factored approach uses smaller tables that can be shared across layers, reducing overhead while maintaining near-entropy-optimal coding.
**Minimum experiment**: Compute H(exp) + H(mantissa|exp) vs H(full_16bit). If the factored entropy is close to joint entropy, the factored approach saves table space with minimal compression loss.
**Expected outcome**: From prior analysis, H(exp)+H(sm|exp) = 2.69+7.93 = 10.62 vs H(full) = 10.55. Gap is only 0.07 bpw (0.04%). But tables are much smaller: 30 × 256 × 2 bytes = 15 KB vs 6000 × 4 bytes = 24 KB. Marginal benefit.
**Risk**: LOW — analysis is straightforward.
**Effort**: 1-2 days
**Contribution**: Empirical comparison
**Novelty**: 5/10 — minor variant of existing approaches.

### Idea 10: Streaming Decompression for Speculative Decoding
**Hypothesis**: In speculative decoding, a small draft model generates candidates that a large target model verifies. If the target model's weights are compressed, decompression latency is critical. Streaming decompression (decompress only the layers needed for the current token) could hide decompression behind draft model execution.
**Minimum experiment**: Profile speculative decoding with DFloat11-compressed target model. Measure what fraction of decompression can be overlapped with draft model execution.
**Expected outcome**: 50-80% of decompression time hidden behind draft model, making compressed serving nearly free for speculative decoding setups.
**Risk**: MEDIUM — depends on draft model speed vs decompression speed.
**Effort**: 1 week
**Contribution**: System optimization
**Novelty**: 7/10 — nobody has studied compression + speculative decoding interaction.

---

## Ranking (by Novelty × Impact × Feasibility)

| Rank | Idea | Novelty | Impact | Feasibility | Score |
|------|------|---------|--------|-------------|-------|
| 1 | **Idea 3: Quantized weight lossless compression** | 9 | 8 | 9 | **216** |
| 2 | **Idea 2: Adaptive fixed-width encoding** | 8 | 8 | 8 | **192** |
| 3 | **Idea 1: GPU ANS-16bit kernel** | 7 | 9 | 6 | **189** |
| 4 | Idea 8: Learned probability model | 9 | 7 | 4 | 168 |
| 5 | Idea 6: FSE for breaking i.i.d. | 8 | 7 | 5 | 168 |
| 6 | Idea 5: Tiled block compression | 7 | 7 | 7 | 168 |
| 7 | Idea 4: Cross-model delta serving | 7 | 7 | 7 | 168 |
| 8 | Idea 10: Streaming + speculative | 7 | 6 | 7 | 147 |
| 9 | Idea 7: Unified multi-precision | 6 | 7 | 7 | 147 |
| 10 | Idea 9: Factored ANS | 5 | 5 | 9 | 112 |

**Top 3 for pilot experiments:**
1. **Idea 3** — Lossless compression of quantized INT4/FP8 weights (analysis + entropy measurement)
2. **Idea 2** — Adaptive fixed-width encoding (per-layer exponent analysis)
3. **Idea 5** — Tiled block compression (tile exponent variance analysis)
