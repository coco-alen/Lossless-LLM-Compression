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

### Branch
- `ans16-compression` — Contains the ANS-16bit codec and all experiment code
