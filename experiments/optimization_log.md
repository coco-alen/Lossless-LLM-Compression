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

### Files Created
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
