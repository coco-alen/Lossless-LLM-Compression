# Paper Plan: Full-Value Entropy Coding for Lossless LLM Weight Compression

**Venue**: NeurIPS 2026 Datasets & Benchmarks
**Target**: 9 pages main + appendix
**Focus**: Expand algorithm descriptions and experimental analysis

## Section Structure

### 1. Introduction (0.8 pages)
- Motivation: storage/distribution bottleneck
- Gap: existing codecs factorize values, miss joint structure
- Research question: how much compression is available, how much is left on the table?
- Contributions (4 items)

### 2. Background and Problem Setup (0.7 pages)
- 2.1 Weight Precision Formats (BF16, FP8, INT4)
- 2.2 Entropy Accounting (ratio, bpv, gap)
- 2.3 Related Work

### 3. Oracle Baseline and Factorization Analysis (2.1 pages)
- 3.1 Formal Compression Task: define exact-reconstruction on value streams, tensor grouping, metrics
- 3.2 Full-Value ANS Oracle: Algorithm 1 (pseudocode), symbolization per format, shared tables
- 3.3 Complexity and Metadata Overhead: O(n) encode/decode, metadata bytes, shared-table ablation
- 3.4 Factorized Baselines: formal definitions of 5 strategies, taxonomy table
- 3.5 Information-Theoretic Gap Decomposition: H(E,SM) identity, stacked bar figure

### 4. Experimental Protocol and Empirical Analysis (3.3 pages)
- 4.1 Experimental Protocol: hardware, software, timing, verification
- 4.2 BF16 Results: cross-model stability, per-weight-type breakdown
- 4.3 Casted FP8: Model and Format Dependence: mechanism, scatter plot, ablation
- 4.4 Native FP8: Calibration Dominates: 5 models/4 families, factorization ablation, MI
- 4.5 INT4: Simulated vs Real Layouts: GPTQ vs AWQ component breakdown
- 4.6 Cross-Format Synthesis: unifying takeaway

### 5. Reference GPU Codec (1.5 pages)
- 5.1 Design Objective: rate-throughput Pareto target
- 5.2 Two-Stream Encoder: Algorithm 2 (pseudocode), bit layout
- 5.3 CUDA Decoder Kernel: Algorithm 3 (pseudo-CUDA), warp prefix sum, batched mode
- 5.4 GPU Results: throughput table, kernel evolution highlights
- 5.5 Negative Systems Results: failure table

### 6. Discussion, Limitations, Conclusion (0.6 pages)

## Algorithms
- Algorithm 1: Full-Value ANS Encode/Decode
- Algorithm 2: Two-Stream FP8 Encoder
- Algorithm 3: Two-Stream FP8 CUDA Decoder

## Figures
- Fig 1: Factorization gap decomposition (stacked bar)
- Fig 2: FP8 compressibility scatter (unique vals vs entropy, color by family)
- Fig 3: Native FP8 per-tensor MI heatmap
- Fig 4: INT4 component breakdown (stacked bars)
- Fig 5: Two-stream codec bit layout diagram
- Fig 6: GPU rate-throughput Pareto frontier

## Tables
- Tab 1: BF16 results (4 models)
- Tab 2: Casted FP8 results (4 models)
- Tab 3: INT4 results (simulated + GPTQ + AWQ)
- Tab 4: Casted FP8 factorization ablation
- Tab 5: Native FP8 entropy + factorization (5 models)
- Tab 6: Native vs casted factorization comparison
- Tab 7: GPU decode throughput comparison
- Tab 8: Negative results summary

## Claims-Evidence Matrix
| Claim | Evidence |
|-------|----------|
| Full-value ANS is oracle baseline | BF16/FP8/INT4 tables, gap <1pp |
| Pipeline determines compressibility | Native 82-84% vs casted 37-70%, GPTQ 85% vs AWQ 92% |
| Factorization penalty is small but nonzero | 0.10-0.16 bpv native, 0.22 bpv casted |
| BF16 stable across families | 4 models, 65.7-66.5% |
| Casted FP8 model-dependent | Qwen 70% vs Mistral 37%, unique vals mechanism |
| Native FP8 calibration fills range | 254/256 values, 5 models, 4 families |
| Two-stream is best rate-throughput | Pareto plot, 77.1% @ 584 GB/s |
| Several extensions don't help | Fused GEMM, NVFP4, context-dep, delta |
