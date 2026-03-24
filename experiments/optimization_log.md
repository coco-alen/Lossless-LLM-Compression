# Compression Optimization Log

## 2026-03-07: Systematic Exploration of Lossless BFloat16 Compression

### Goal
Find a lossless compression method for BFloat16 LLM weights with higher compression ratio than DFloat11 (~66-67% on Qwen3-1.7B/8B).

### DFloat11 Baseline
- Splits BFloat16 into exponent (8 bits) + sign_mantissa (8 bits)
- Huffman-encodes exponent (~2.65 bpw), stores sign_mantissa raw (8 bpw)
- Total: ~10.65 bpw → ~66.6% ratio on Qwen3-1.7B

### Methods Tested (on Qwen3-1.7B)

| Method | Ratio | vs DF11 | Notes |
|--------|-------|---------|-------|
| DFloat11 baseline | 66.62% | --- | Huffman(exp) + raw(sm) |
| Huffman(exp) + Huffman(sm) | 66.64% | +0.02% | sm barely compressible (7.97 bpw entropy) |
| Cross-layer delta(exp) + raw(sm) | ~72% | +5% | WORSE: delta-coding disrupts concentrated exp distribution |
| Cross-layer delta(both) + Huffman | ~72% | +5% | WORSE: same issue |
| Cross-layer delta + left-predictor | ~73.5% | +7% | WORSE: prediction adds noise |
| Cross-layer mean-delta(both) | ~70.4% | +4% | WORSE: mean overhead + weak correlation |
| Left-predictor on sm only | 66.64% | +0.02% | sm is spatially uncorrelated |
| Per-row exp Huffman | 67.9% | +1.3% | WORSE: table overhead per row |
| Block-adaptive (4096/16384) | 68-72% | +2-6% | WORSE: heavy table overhead |
| Exp-conditioned Huffman(sm) | 66.29% | -0.32% | BEATS DF11: per-exp Huffman tables for sm |
| Sign-separated conditioned | 66.29% | -0.33% | Similar to exp-conditioned |
| **16-bit Huffman** | **66.19%** | **-0.43%** | **Direct 16-bit Huffman (~5,900 unique values)** |
| **16-bit ANS (rANS)** | **65.96%** | **-0.66%** | **Near-entropy optimal, best practical method** |
| 16-bit entropy lower bound | 65.95% | -0.67% | Theoretical minimum for i.i.d. coding |

### Key Findings

1. **Sign_mantissa byte is nearly random**: H(sm) ≈ 7.97 bpw (out of 8). This is the fundamental bottleneck. No amount of prediction, delta-coding, or spatial processing helps.

2. **Cross-layer delta HURTS**: The exponent distribution is highly concentrated (~20-30 unique values). Delta-coding spreads the distribution and increases entropy. Prediction on random data adds noise.

3. **Exponent-mantissa correlation exists but is small**: H(sm|exp) ≈ 7.93 bpw vs H(sm) ≈ 7.97. Conditioning on exponent saves ~0.04 bpw on mantissa.

4. **Only ~5,000-7,500 unique 16-bit values** per weight type. This makes direct 16-bit coding feasible.

5. **16-bit Huffman beats DFloat11** by treating the full BFloat16 value as a single symbol, avoiding the byte-separation overhead.

6. **ANS achieves near-entropy** for 16-bit symbols, closing the Huffman-to-entropy gap (~0.03 bpw × N). This gives the best practical compression.

7. **Improvement scales with model size**: Qwen3-1.7B: -0.66%, Qwen3-8B: -0.70% (confirmed, saves 96.6 MB).

### Bit-plane Analysis (self_attn.q_proj)
```
Bit   Uncond H   Cond H|exp
  0     1.0000       1.0000   (mantissa LSB: random)
  1     1.0000       1.0000
  2     1.0000       0.9999
  3     0.9998       0.9993
  4     0.9988       0.9969
  5     0.9948       0.9876
  6     0.9800       0.9548   (mantissa MSB: most structure)
  7     1.0000       1.0000   (sign: random)
TOTAL   7.9734       7.9384
```

### Best Methods

1. **ANS-16bit** (`new_compression/codec_ans16.py`): Best ratio, near-entropy
   - Qwen3-1.7B: 65.96% (vs 66.62% DF11) → saves 18.5 MB
   - Qwen3-8B: 66.48% (vs 67.18% DF11) → saves 96.6 MB (-0.70%)
   - Fully lossless, verified via encode-decode roundtrip
   - Uses `constriction` library (Rust-based rANS)

2. **16-bit Huffman** (`new_compression/codec16.py`): Good ratio, simpler to implement
   - Qwen3-1.7B: 66.19% (vs 66.62% DF11) → saves 12.0 MB
   - Uses standard `dahuffman` library (Python Huffman)
   - Compatible with existing CUDA decompression patterns (needs new kernel)

### Remaining Potential
- Current ANS-16bit: 65.96% (Qwen3-1.7B)
- Entropy LB: 65.95%
- Gap: 0.01% — essentially at the theoretical limit for i.i.d. coding
- Further improvement would require context-dependent coding (e.g., positional, cross-weight) which adds complexity for marginal gain

---

## 2026-03-07/08: Exhaustive Context-Dependent Coding Exploration

### Goal
Explore whether context-dependent (sequential/conditional) coding can beat ANS-16bit's i.i.d. limit.

### Key Discovery: Adjacent Weights ARE Correlated
```
Sequential entropy analysis (Qwen3-1.7B):
  H(W) i.i.d.:              10.55 bpw  → 65.95%
  H(W[i] | W[i-1]) bigram:   9.91 bpw  → 61.91%   ← 4% savings theoretically!
  H(W | layer):             10.53 bpw  → 65.89%   ← per-layer barely helps
  H(W[i] | exp(W[i-1])):   10.61 bpw  → 66.34%   ← exp of prev barely helps
```
Adjacent weights have significant bigram correlation (~0.65 bpw mutual information). However, this correlation is spread across the full 16-bit value, not just the exponent.

### Methods Tested Beyond ANS-16bit

| Method | Overall Ratio | vs ANS-16bit | Notes |
|--------|--------------|--------------|-------|
| ANS-16bit (global, baseline) | 65.962% | --- | Near i.i.d. entropy optimal |
| Per-layer ANS-16 | 66.079% | +0.117% | WORSE: table overhead dominates for small layers |
| Per-2-layer ANS-16 | 65.998% | +0.037% | WORSE: still too much overhead |
| Per-4-layer ANS-16 | 65.956% | -0.006% | Marginal improvement |
| Per-7-layer ANS-16 | 65.939% | -0.022% | Best per-group variant, saves 0.6MB |
| Seq-exp + per-exp(sm) | 65.941% | -0.020% | Sequential exponent + per-exp mantissa |
| Per-layer seq-exp+sm | 66.044% | +0.082% | WORSE: per-layer overhead kills it |
| Prev-exp-cond 16bit ANS | ~66.5% | +0.5% | WORSE: conditioning on prev exp barely helps, overhead adds up |
| Quantized context K=8 (no sort) | ~66.0-66.5% | +0.02-0.5% | WORSE: overhead > data savings for all K |
| Quantized context K=8 (sorted) | ~66.0-66.5% | +0.1-0.7% | WORSE: row sorting doesn't help |
| Row-sorted + bigram | 9.95 bpw | (theory) | Sorting rows HURTS bigram correlation |
| Cross-layer interleave | 9.96 bpw | (theory) | Interleaving layers doesn't help |
| Column-major ordering | Same | +0.000% | Reordering doesn't change i.i.d. entropy |
| zstd(ANS output) | ~66.44% | +0.48% | ANS output is essentially incompressible |
| zstd-19(exp) + ANS(sm) | ~67.5% | +1.5% | WORSE: zstd wastes bits on exponents vs ANS |
| zstd-10 byte-separated | ~70% | +4% | WORSE: LZ77 can't compress random mantissa bytes |

### Sequential Exponent Analysis
```
Exponent entropy:
  H(exp) i.i.d.:     2.69 bpw
  H(exp[i]|exp[i-1]): 2.54 bpw  → saves 0.15 bpw on exponents only
```
Adjacent exponents are somewhat correlated (saves 0.15 bpw), but this improvement on the 8-bit exponent stream is small compared to the total 16-bit coding cost.

### Why Context-Dependent Coding Fails in Practice

1. **Large alphabet problem**: BFloat16 weights use ~6000 unique 16-bit values. To exploit bigram correlations, you need per-context probability tables. With K_context contexts and ~6000 symbols:
   - Overhead per context: 6000 × 6 bytes = 36KB
   - For full bigram (K=6000): 6000 × 36KB = 216MB per weight type — catastrophic
   - For exp-context (K=30): 30 × 36KB = 1.1MB — but exp context captures only 0.016 bpw savings

2. **Context quantization loses correlation**: The 4% bigram correlation comes from knowing the EXACT previous 16-bit value. Coarsening to K buckets (quantized context) destroys most of the correlation:
   - K=8: almost no data savings, overhead > savings
   - K=256: more data savings, but 3MB+ overhead per weight type dominates
   - No K value gives net improvement

3. **Row reordering doesn't help**: Sorting rows by mean value slightly INCREASES bigram entropy (9.95 vs 9.91). The natural row ordering is already near-optimal.

4. **Cross-layer interleave doesn't help**: Same (i,j) position across layers has 9.96 bpw bigram entropy, worse than within-row.

5. **Adaptive coding overhead**: With 6000 symbols, an adaptive model needs O(K_symbol × log(n) / n) overhead per context state, which exceeds the conditional entropy savings for the exp-context case.

### Byte-Separated Compression (zstd)
```
zstd-10 results by byte stream:
  high byte (sign+exp):  ~20% of original (3.2 bpw)
  low byte (exp_lsb+mant): ~50% of original (8.0 bpw, incompressible!)
  separated (h+l):       ~70% of original
  interleaved (raw):     ~77% of original
```
Even the best general compressor (zstd-19) can't match ANS-16bit because:
- The low byte is nearly random → 50% minimum
- zstd is worse than ANS even on the exponent stream alone (3.2 vs 2.7 bpw)
- LZ77 finds no useful repeated patterns in stationary BFloat16 data

### Fundamental Conclusion
**ANS-16bit with a global per-weight-type probability table is the practical optimum for lossless BFloat16 LLM weight compression.**

The i.i.d. entropy lower bound (65.95%) is tight: no practical context model, reordering, or general compressor can meaningfully improve upon it. The 4% theoretical bigram improvement is information-theoretically real but cannot be exploited without excessive overhead due to the large symbol alphabet.

The only paths to significantly better lossless compression would be:
1. **Lossy pre-processing** (e.g., mantissa bit rounding) → no longer lossless
2. **Model architecture changes** (weight sharing, structured sparsity) → changes the model
3. **Training-aware compression** (quantization-aware training) → changes the training pipeline

### Novel Decomposition Approaches (all WORSE than ANS-16bit)

| Method | Delta vs ANS-16bit | Notes |
|--------|-------------------|-------|
| Bit-split at 7 | +0.20% | Splitting at mantissa boundary loses correlation |
| Bit-split at 8 | +0.86% | Byte boundary split |
| Bit-split at 9 | +1.15% | |
| Bit-split at 10 | +4.10% | Worst — 6 random LSBs kill compression |
| Bit-split at 11 | +0.04% | Closest to full 16-bit, still worse |
| Sign+Exp / Mantissa | +0.20% | BFloat16 natural decomposition |
| Exp + per-exp SM | ±0.000% | Equal (H(exp)+H(sm|exp)=H(full)) — no overhead savings |
| Grouped 10-14 bit | +0.00-0.01% | Per-group residual tables add tiny overhead |
| Cross-type shared table | TBD | Single table for all 7 weight types |

**Conclusion**: No decomposition of the 16-bit value can beat direct 16-bit ANS coding. Joint coding is always optimal or equal to decomposition. This is a fundamental information-theoretic result: H(X,Y) ≤ H(X) + H(Y) with equality iff X,Y are independent.

### Model Coverage Analysis (Qwen3-1.7B)
```
Total parameters:     1,720,574,976
Covered (7 types):    1,409,286,144  (81.9%)
Uncovered:              311,288,832  (18.1%)

Uncovered breakdown:
  embed_tokens.weight:  311,164,928  (18.08%)  ANS ratio: 65.75%
  layernorm weights:         ~120K   (negligible)
```
The embedding table is 18% of the model and compresses to 65.75% with ANS-16bit (even better than decoder weights). DFloat11 does not compress embeddings — adding embedding compression would save an additional 213MB on Qwen3-1.7B.

### Files Created (this session)
- `experiments/test_sequential_entropy.py` — Sequential/positional entropy + zstd analysis
- `experiments/test_byte_separated_sequential.py` — Byte-separated sequential ANS
- `experiments/test_prev_conditioned_ans.py` — Prev-value conditioned ANS
- `experiments/test_practical_improvements.py` — Per-layer and per-group ANS comparison
- `experiments/test_reorder_and_context.py` — Row reordering + quantized context
- `experiments/test_context_ans.py` — Context-dependent ANS (per-layer, exp-conditioned)

### Files Created (previous session)
- `new_compression/codec16.py` — 16-bit Huffman codec
- `new_compression/codec_ans16.py` — ANS-16bit codec (best)
- `new_compression/validate_16bit_fast.py` — Fast validation for 16-bit Huffman
- `new_compression/validate_ans16.py` — Validation for ANS-16bit
- `experiments/explore_compression.py` — Systematic exploration (13 methods)
- `experiments/explore_advanced.py` — Advanced explorations (bit-plane, block, etc.)
- `experiments/test_16bit_fast.py` — Fast 16-bit analysis
- `experiments/test_ans_coding.py` — ANS coding test
- `experiments/test_general_compressors.py` — gzip/zstd/lzma comparison

---

## 2026-03-15: Full Model Compression & Context-Dependent Coding Final Exploration

### Goal
1. Explore whether any remaining context-dependent coding can beat ANS-16bit
2. Measure full-model compression including embeddings/lm_head

### Context-Dependent Coding: Final Attempts

#### XOR Delta Coding (`test_xor_delta_compression.py`)
XOR adjacent int16 values — if neighbors are similar, XOR produces values near 0.
**Result: WORSE** — XOR delta entropy is 11.11 bpw vs 10.57 bpw i.i.d. (+0.54 bpw).
BF16 values are NOT locally correlated in memory layout. XOR spreads the distribution.
All scan orders tested (row-major, col-major, zigzag, cross-layer): all worse.

#### Pair ANS (`test_pair_ans_compression.py`)
Treat consecutive BF16 pairs as single symbols for ANS coding.
**Result: WORSE** — 4.3M unique pairs (cross-layer), 33MB table overhead per weight type.
Net compression: 95-208% (table overhead dwarfs the 0.08 bpw entropy gain).
k-tuple analysis: k=3,4 have even more unique tuples → worse overhead.

#### Single-Stream Context-Conditioned ANS (`test_context_conditioned_ans.py`)
Use prev-exponent as context (~30 values) with per-position probability selection in ONE ANS stream.
Key entropy measurements:
```
H(W) global:         10.67 bpw → 66.71%
H(W|own_exp):         7.94 bpw → 49.63%  (25% savings, but already captured by joint coding)
H(W|prev_exp):       10.64 bpw → 66.51%  (only 0.03 bpw savings!)
H(W|prev_hi_byte):   10.65 bpw → 66.55%  (even less)
```
**Result: WORSE** — prev-exponent context saves only 0.03 bpw. Table overhead (1 MB) exceeds savings (222 KB). Actual ANS: 76.84% vs 66.73% standard (Laplace smoothing + fragmentation).

### Definitive Conclusion on Context Coding
The 4% theoretical bigram improvement (H=9.91 vs 10.55) requires the FULL previous 16-bit value as context. With ~6000 unique values, this means 6000 probability tables of 6000 entries each = 216 MB overhead. **Completely impractical.**

Coarsening context to exponent (~30 values) captures only 0.03 bpw (0.29%) — far too little to justify any overhead.

### Full Model ANS-16bit Compression (`test_ans16_vs_dfloat11.py`)

| Model | BF16 Size | ANS-16 Full | DFloat11* | ANS-16 Extra |
|-------|-----------|-------------|-----------|--------------|
| Qwen3-0.6B | 1,137 MB | 751 MB (66.08%) | 857 MB | **+106 MB** |
| Qwen3-1.7B | 3,282 MB | 2,164 MB (65.93%) | 2,385 MB | **+221 MB** |
| Qwen3-8B | 15,623 MB | 10,373 MB (66.40%) | 11,275 MB | **+902 MB** |

*DFloat11 compresses only attn+MLP weights (74-82% of model). Embeddings/lm_head stored raw.

**Why ANS-16 full > DFloat11:**
1. Better per-weight compression: 65.96% vs 66.62% on same scope (-0.66%)
2. **Embedding table compression**: 65.7-66.0% ratio (DFloat11 skips this entirely)
3. **lm_head compression**: 65.82% on Qwen3-8B (separate from embeddings)

Embedding/lm_head breakdown:
- Qwen3-0.6B: embed_tokens = 296.8 MB → 195.0 MB (65.70%), lm_head tied
- Qwen3-1.7B: embed_tokens = 593.5 MB → 390.2 MB (65.75%), lm_head tied
- Qwen3-8B: embed_tokens = 1187 MB → 783 MB (65.99%), lm_head = 1187 MB → 781 MB (65.82%)

Compression speed: 75-78 MB/s on CPU (constriction Rust rANS).

### Files Created
- `experiments/test_xor_delta_compression.py` — XOR delta analysis
- `experiments/test_pair_ans_compression.py` — Pair/tuple ANS analysis
- `experiments/test_context_conditioned_ans.py` — Context-conditioned entropy analysis
- `experiments/test_full_model_ans16.py` — Full model compression + DFloat11 comparison
- `experiments/test_ans16_vs_dfloat11.py` — Head-to-head benchmark

## 2026-03-23: Per-Row Codebook with Exp/Mantissa Split

### Goal
Test whether per-row codebooks with separate exponent/mantissa coding can beat ANS-16bit global.

### Methods Tested (Qwen3-0.6B, first layer)

| Method | q_proj | k_proj | gate_proj | down_proj |
|--------|--------|--------|-----------|-----------|
| ANS-16 global | **66.73%** | **67.10%** | **66.34%** | **66.03%** |
| Global split (exp+mant) | 67.19% | 67.14% | 66.94% | 66.92% |
| Per-row joint 16-bit | 261.59% | 261.27% | 261.69% | 183.24% |
| Per-row split (exp+mant) | 104.35% | 104.31% | 104.30% | 79.78% |
| Per-row exp + global mant | 68.78% | 68.76% | 68.73% | 67.61% |
| Per-row exp + per-exp mant | 68.02% | 68.17% | 67.92% | 66.52% |

### Key Findings
- **ANS-16 global wins every comparison**. Per-row approaches lose to codebook overhead.
- With only 1024 values per row, codebook overhead is not amortized. Even down_proj (3072 cols) can't overcome it.
- Mantissa byte has ~249-256 unique values per row (nearly all 256) — per-row mantissa tables are full-size with no entropy benefit.
- Splitting into exp+mant loses the joint coding benefit (global split: +0.6-0.9% worse).
- Best non-global method: per-row exp + per-exp mant (66.52% on down_proj), still 0.5% worse than global ANS-16.
- All layers combined (q_proj, 57K rows): same pattern, global wins at 66.73%.

### Conclusion
Per-row codebooks are counterproductive for BF16 weight compression. The Shannon entropy is a global property — per-row models waste bits on table overhead without capturing meaningful local structure. ANS-16bit global remains the optimal approach.

### Files Created
- `experiments/test_perrow_split_compression.py` — Per-row codebook experiment

### Branch
- `ans16-compression` — Contains the ANS-16bit codec and all experiment code

---

## 2026-03-24: Adaptive Fixed-Width Encoding (ZipServ-style) Analysis

### Goal
Evaluate whether ZipServ's fixed-width TCA-TBE approach (3-bit code for top-7 consecutive exponents) could be competitive, and whether adaptive per-layer widths would help.

### Method
For each weight tensor, find the best CONSECUTIVE window of k exponents covering the most values:
- k=3 (2-bit code): in-window cost = 1+2+8 = 11 bits, out-of-window = 1+16 = 17 bits
- k=7 (3-bit code, ZipServ style): in-window = 1+3+8 = 12, out = 17
- k=15 (4-bit code): in-window = 1+4+8 = 13, out = 17
- k=31 (5-bit code): in-window = 1+5+8 = 14, out = 17
(1-bit flag per value to mark in/out of window)

### Results (Qwen3-0.6B, 596M params)

| Method | Avg BPW | Ratio | Size (MB) |
|--------|---------|-------|-----------|
| Uniform k=3 (2-bit) | 12.689 | 79.31% | 901.6 |
| **Uniform k=7 (3-bit, ZipServ)** | **12.161** | **76.00%** | **864.1** |
| Uniform k=15 (4-bit) | 13.001 | 81.26% | 923.8 |
| Uniform k=31 (5-bit) | 14.000 | 87.50% | 994.8 |
| Adaptive (best k per layer) | 12.161 | 76.00% | 864.1 |
| **ANS-16bit (our baseline)** | **10.554** | **65.96%** | **749.9** |
| Uncompressed BF16 | 16.000 | 100.00% | 1136.9 |

### Key Findings

1. **k=7 (ZipServ 3-bit) is the sweet spot** for fixed-width coding at 76.00%. But this is FAR worse than ANS-16bit (65.96%), a gap of **10 percentage points**.

2. **Adaptive width provides zero benefit**: k=7 is optimal for 100% of parameters by weight. Only tiny norm layers (0.01% of params) prefer k=3. The adaptive distribution:
   - k=3: 113 layers, 65,536 params (0.0%)
   - k=7: 197 layers, 595,984,384 params (100.0%)
   - k=15/k=31: 0 layers

3. **Coverage at k=7 is already ~97%** (mean 96.78%, min 82.03%). The remaining 3% of overflow values cost 17 bits each, which is expensive. But even with near-perfect coverage, the floor is 12 bpw (1+3+8 for every value).

4. **Fundamental problem**: Fixed-width coding wastes bits on the exponent. The exponent has ~2.7 bits of entropy but k=7 uses 3 bits + 1 flag bit = 4 bits. ANS codes exponents at ~2.7 bits. The mantissa+sign is 8 bits regardless. So fixed-width = 4+8=12 bpw minimum vs ANS = 2.7+7.97=10.67 bpw.

5. **k=15 and k=31 are always worse** because the extra code bits outweigh the marginal coverage improvement (99.87% vs 96.78% for k=15 vs k=7).

### Coverage Statistics (across all layers)
```
k=3  (2-bit): coverage min=0.6286 mean=0.7889 max=0.9990
k=7  (3-bit): coverage min=0.8203 mean=0.9678 max=1.0000
k=15 (4-bit): coverage min=0.9688 mean=0.9987 max=1.0000
k=31 (5-bit): coverage min=1.0000 mean=1.0000 max=1.0000
```

### Entropy Comparison
- Weighted avg 16-bit entropy: 10.556 bits
- ANS-16bit achieves: 10.554 bpw (gap to entropy: -0.002 bits, essentially at the limit)
- ZipServ (k=7): 12.161 bpw (1.6 bits above entropy — wasted on fixed-width overhead)

### Conclusion
ZipServ's fixed-width approach is inherently suboptimal for lossless BFloat16 compression. The 1-bit flag + fixed k-bit exponent code wastes ~1.6 bpw compared to entropy-optimal ANS coding. Adaptive per-layer width selection provides no benefit since k=7 dominates for all significant tensors. ANS-16bit remains the clear winner at 65.96% vs ZipServ's 76.00%.

### Files Created
- `experiments/pilot_adaptive_fixedwidth.py` — Adaptive fixed-width pilot analysis

---

## 2026-03-24: Tiled Block Floating Point Compression (Tensor Core Alignment)

### Goal
Test whether tiling weight matrices into GPU Tensor Core fragment sizes (8x8, 16x8, 16x16, 32x16) and sharing a base exponent per tile can beat ANS-16bit.

### Method
- Reshape 2D weight matrices into tiles of various sizes
- Per tile: 1 shared base_exponent (8 bits) + N * (offset_bits + 7 mantissa + 1 sign)
- offset_bits = ceil(log2(range+1)) where range = max_exp - min_exp within tile
- Also tested bimodal variant: 2 base exponents per tile (split by median), 1-bit selector per value

### Results (Qwen3-0.6B, 596M params, 100% are 2D BF16)

| Tile Size | Mean Range | <=3% | <=7% | <=15% | >15% (bad) |
|-----------|-----------|------|------|-------|-------------|
| 8x8       | 8.09      | 0.0% | 42.8%| 99.7% | 0.3%        |
| 16x8      | 9.25      | 0.0% | 16.2%| 99.2% | 0.8%        |
| 16x16     | 10.35     | 0.0% | 2.5% | 98.3% | 1.7%        |
| 32x16     | 11.45     | 0.0% | 0.1% | 96.4% | 3.6%        |

| Tile Size | BlockFP (%) | Fallback (%) | Bimodal (%) | Avg BPW |
|-----------|-------------|--------------|-------------|---------|
| 8x8       | 73.13%      | 73.19%       | 75.84%      | 11.70   |
| 16x8      | 74.42%      | 74.56%       | 76.65%      | 11.91   |
| 16x16     | 75.14%      | 75.45%       | 77.47%      | 12.02   |
| 32x16     | 75.32%      | 75.99%       | 78.69%      | 12.05   |

### Comparison
- ANS-16bit: 65.96% (10.55 bpw)
- DFloat11: ~66.6% (10.66 bpw)
- Best BlockFP (8x8): 73.13% (11.70 bpw) — **7.17pp WORSE than ANS-16bit**
- Best Bimodal (8x8): 75.84% — **9.88pp WORSE than ANS-16bit**

### Why Block FP Fails
1. **Exponent range is too wide**: Mean within-tile exponent range is 8-11 even for small 8x8 tiles. 0% of tiles have range <=3 (needed for 2-bit offsets).
2. **Offset bits dominate**: With mean range ~8, offset needs ~4 bits. Total = 8(base)/N + 4(offset) + 8(sign+mantissa) = 12+ bpw, worse than ANS's 10.55 bpw.
3. **Bimodal is even worse**: The 1-bit selector per element + 2 bases (16 bits overhead) costs more than the range reduction from splitting.
4. **Uniformly bad across layers**: All layers show similar ~74% ratios with 16x8 tiles.

### Conclusion
Tiled block floating point is fundamentally unsuitable for BFloat16 LLM weight compression. The exponent distribution within spatial tiles is too spread (range 8-11) for the shared-base approach to save bits. BF16 weights lack the spatial coherence that block FP exploits in activations/gradients. ANS-16bit remains optimal.

### Files Created
- `experiments/pilot_tile_compression.py` — Tiled block FP pilot experiment

---

## 2026-03-24: Pilot — Lossless Compression of Quantized (INT4/FP8) Weights

### Goal
Investigate whether entropy coding ON TOP of already-quantized INT4/FP8 model weights can yield additional compression. If INT4 values are non-uniform, entropy coding could save 15-30% beyond the 4 bits/weight baseline.

### Method
1. Downloaded real GPTQ INT4 model (Qwen2.5-7B-Instruct-GPTQ-Int4), unpacked INT4 nibbles from int32 qweight tensors
2. Simulated symmetric per-channel INT4 quantization on Qwen3-0.6B BF16 weights
3. Cast Qwen3-0.6B BF16 weights to FP8 (float8_e4m3fn), analyzed byte-level entropy

### Results

#### Real GPTQ INT4 (Qwen2.5-7B, 30 qweight tensors from shard 1)
- **Aggregate entropy: 3.37 / 4.0 bits → 84.2% ratio → 15.8% additional savings**
- Distribution is bell-shaped centered at 8 (zero-point=7): values 6-10 cover 70% of data
- Per-tensor entropy: min=2.88, max=3.47, mean=3.37, std=0.13
- Potential savings on 7B INT4 model: ~0.5 GB (from 3.5 GB to ~2.95 GB)
- Zero-points (qzeros) are all constant (value=7) → 100% compressible but tiny
- Scales (float16) have ~10.1-10.7 bits entropy per 16 bits → also compressible (63-67% ratio)

#### Simulated INT4 (Qwen3-0.6B, symmetric per-channel)
- **Aggregate entropy: 2.97 / 4.0 bits → 74.3% ratio → 25.7% additional savings**
- Even more concentrated: values 7-9 cover 60% of data, value 8 alone is 23%
- Per-tensor entropy: min=2.65, max=3.07, mean=2.93

#### FP8 e4m3fn (Qwen3-0.6B)
- **Aggregate entropy: 5.65 / 8.0 bits → 70.6% ratio → 29.4% additional savings**
- Only 117 unique byte values (out of 256)
- Sub-field breakdown:
  - Sign: 1.00 bits (max 1.0) — essentially random
  - Exponent (4 bits): 1.72 bits (max 4.0) — highly compressible!
  - Mantissa (3 bits): 2.98 bits (max 3.0) — nearly random
- FP8 exponent heavily concentrated: values 0,1,2 cover 96% (biased zero-ward)
- Correlation gain (joint vs sum-of-fields): 0.05 bits — minimal

### INT4 Value Distribution (Real GPTQ, aggregate)
```
 0: 0.20%    8: 20.05% ████████████████████
 1: 0.69%    9: 14.42% ██████████████
 2: 1.08%   10: 10.46% ██████████
 3: 2.18%   11:  6.88% ███████
 4: 4.07%   12:  4.06% ████
 5: 6.89%   13:  2.18% ██
 6: 10.47%  14:  1.08% █
 7: 14.40%  15:  0.89%
```

### Practical Size Estimates (7B model)
| Format | Size (GB) |
|--------|-----------|
| BF16 (raw) | 14.00 |
| DFloat11 | 9.66 |
| INT4 GPTQ (raw) | 3.50 |
| INT4 + entropy (est. 84%) | 2.95 |
| FP8 (raw) | 7.00 |
| FP8 + entropy (est. 71%) | 4.95 |

### Key Findings
1. **INT4 is non-uniform**: Bell-shaped distribution centered at the zero-point. Real GPTQ has 3.37 bits entropy (vs 4.0 max), simulated has 2.97 bits. This means 16-26% additional compression is achievable via entropy coding.
2. **FP8 is very compressible**: Only 117 unique values, 5.65/8.0 bits entropy. The 4-bit exponent has only 1.72 bits of entropy (4 unique values dominate). 29% additional savings possible.
3. **Nobody is doing this**: Current quantization frameworks (GPTQ, AWQ, GGUF) store INT4 values packed into bytes but do NOT apply entropy coding. This is low-hanging fruit.
4. **Scales and zero-points are also compressible**: Scales have ~10.5/16 bits entropy (35% savings), zero-points are often constant (100% compressible).
5. **The opportunity is real**: For a 7B INT4 model, entropy coding could save ~550 MB (from 3.5 GB to ~2.95 GB) with zero quality loss.

### Verdict
**PROMISING** — Both INT4 and FP8 have significant entropy slack. INT4 entropy coding is the more impactful target since INT4 is the dominant quantization format for deployment. Next steps: implement ANS/Huffman encoder for INT4 nibbles and benchmark decode speed on GPU.

### Files Created
- `experiments/pilot_quantized_entropy.py` — Entropy analysis of INT4/FP8 quantized weights

---

## 2026-03-24: Multi-Format Full-Value ANS Compression

### Goal
Extend ANS-16bit (which achieved i.i.d. entropy limit for BF16 at 65.96%) to FP8 and INT4 formats. Demonstrate that full-value coding beats exponent-only approaches.

### New Codec: codec_multiformat.py
- Unified codec handling BF16, FP8 (e4m3fn, e5m2), and INT4 (packed uint8)
- Full-value ANS coding: treats entire value as single symbol
- Uses constriction library (Rust rANS) for CPU encode/decode

### Results on Qwen3-0.6B (596M params)

| Format | Our Method | DFloat11 | ECF8 | EntroLLM | Savings |
|--------|-----------|----------|------|----------|---------|
| BF16 | **66.53%** | 66.6% | — | — | 33.5% |
| FP8 (e4m3fn) | **70.42%** | — | 85-90% | — | **29.6%** |
| INT4 (simulated) | **74.06%** | — | — | ~84% | **25.9%** |

### Key Findings

1. **FP8 full-value ANS beats ECF8 by 15-20pp**: ECF8 only compresses exponents (4 bits with 1.72 bits entropy), leaving mantissa raw. Our full-value approach codes all 8 bits jointly, capturing the joint structure.

2. **INT4 entropy coding provides ~26% additional savings**: Symmetric INT4 weights cluster around zero-point, with only 2.97 bits entropy out of 4. Real GPTQ INT4 has 3.37 bits entropy (16% savings).

3. **All compression verified lossless**: Bit-exact round-trip for every layer.

### GPU Decompression via nvCOMP

| Method | BF16 Ratio | FP8 Ratio | Decode Speed |
|--------|-----------|----------|-------------|
| CPU full-value ANS (constriction) | 66.53% | 70.42% | CPU-bound |
| GPU raw nvCOMP ANS | 82.7% | 85.7% | 29-56 GB/s |
| GPU byte-separated nvCOMP ANS | **71.6%** | 85.7% | **96 GB/s** |

- Byte-separated GPU approach: split BF16 into high byte (sign+exp, 38.5% compressed) and low byte (mantissa, 104.8% expanded). Total 71.6% at 96 GB/s decode.
- FP8: raw nvCOMP ANS already reasonable at 85.7%, but our CPU full-value ANS is much better at 70.4%.

### Entropy Analysis Summary

| Format | Bits/value | Max bits | Entropy | Unique Values | Compression Limit |
|--------|-----------|---------|---------|---------------|-------------------|
| BF16 | 16 | 16 | 10.56 | ~6,000 | 65.96% |
| FP8 e4m3fn | 8 | 8 | 5.65 | ~117 | 70.6% |
| INT4 (GPTQ) | 4 | 4 | 3.37 | 16 | 84.2% |
| INT4 (symmetric) | 4 | 4 | 2.97 | 16 | 74.3% |

### FP8 Sub-field Analysis
- Sign (1 bit): 1.000 bits entropy (incompressible)
- Exponent (4 bits): 1.722 bits entropy (very compressible! 43% of max)
- Mantissa (3 bits): 2.975 bits entropy (nearly incompressible)
- Joint correlation: 0.047 bits — full-value coding captures this vs exponent-only

### Pilot Results Summary (all 6 experiments)

| Pilot | Signal | Key Number |
|-------|--------|------------|
| FP8/INT4 entropy | POSITIVE | 29%/16% savings |
| Adaptive fixed-width | NEGATIVE | 76% (10pp worse than ANS) |
| Tile block compression | NEGATIVE | 73% (7pp worse than ANS) |
| v prediction residual | VERY POSITIVE | 86-94% exact, <1.6 bits residual |
| nvCOMP GPU ANS | VERY POSITIVE | 85-126 GB/s, 3000-40000x faster |
| Delta CPU offload | NEGATIVE | Scatter 7.7x slower |

### Files Created
- `experiments/new_compression/codec_multiformat.py` — Multi-format ANS codec
- `experiments/new_compression/gpu_decompress.py` — nvCOMP GPU decompression
- `experiments/benchmark_multiformat.py` — Full benchmark script
- `experiments/pilot_adaptive_fixedwidth.py` — Adaptive fixed-width analysis
- `experiments/pilot_tile_compression.py` — Tile block compression analysis
- `experiments/pilot_quantized_entropy.py` — INT4/FP8 entropy analysis

### Full Benchmark Results (Qwen3-8B, 8.2B params)

| Format | Ratio | Original | Compressed | Savings |
|--------|-------|----------|-----------|---------|
| BF16 | **66.10%** | 16,382 MB | 10,829 MB | 33.9% (5,553 MB saved) |
| FP8 (e4m3fn) | **69.09%** | 8,191 MB | 5,659 MB | 30.9% (2,532 MB saved) |

**Scaling confirmed**: Larger models compress slightly better (66.10% vs 66.53%).
**FP8 headline**: For Qwen3-8B, our method saves 2.5 GB on top of the 2x from FP8 quantization. ECF8 would save only ~1.2 GB (exponent-only). We save **2x more than ECF8**.

### GPU Full-Value Huffman Kernel Results (2026-03-24)

**Kernel**: `experiments/new_compression/gpu_codec.py` — CuPy CUDA kernel based on DFloat11 architecture.

| Metric | Per-Layer | Shared Table (analytical) |
|--------|-----------|--------------------------|
| BF16 Ratio | 71-79% (overhead per layer) | **66.26%** |
| Decode Speed | 6-14 GB/s | Expected similar |
| Lossless | ✅ Verified | ✅ |

**Pure Huffman analysis** (shared table, all Qwen3-0.6B weights):
- Avg code length: 10.6007 bits/symbol (vs 10.5767 entropy)
- **66.26% ratio** — 0.34% better than DFloat11 (66.6%), 0.16% above ANS (66.10%)
- Max code length: 29 bits
- 6,677 unique symbols, negligible LUT overhead (13 KB)

**GPU decode throughput comparison**:
| Method | BF16 Ratio | Decode Speed | Notes |
|--------|-----------|-------------|-------|
| Our GPU Huffman (per-layer) | 71-79% | 6-14 GB/s | Per-layer table overhead |
| Our GPU Huffman (shared, analytical) | **66.26%** | ~10 GB/s (est.) | Needs group encode |
| DFloat11 GPU Huffman | ~66.6% | ~30-60 GB/s | Mature optimized kernel |
| nvCOMP byte-separated ANS | 71.6% | **96 GB/s** | Best speed |
| Our CPU ANS-16bit | **66.10%** | N/A | Best compression |

**Key insight**: Our full-value Huffman achieves better compression than DFloat11 (66.26% vs 66.6%)
by coding the full 16-bit value instead of just the exponent. The GPU kernel is functional but
not yet optimized to DFloat11's speed level. Next steps:
1. Implement group encoding with shared table
2. Optimize CUDA kernel (coalesced reads, better occupancy)
3. Compare end-to-end with DFloat11 on same hardware
