# Auto Review Log

## Pre-Loop: Codex Evaluations (Retroactive)

### Evaluation 1: Critical Review of Current Work (GPT-5.4 xhigh)

**Score: 4/10** — Reject in current form.

**Critical Weaknesses:**
1. **FP8 comparison is misleading**: ECF8 saves up to 26.9% (not 10-15%). Our entropy math shows full-value coding beats exponent-only by only ~0.9pp for FP8, not 15-20pp. The baseline comparison was wrong.
2. **Novelty is thin**: "Use full-value ANS instead of exponent-only" is the obvious entropy-coding baseline, not a methods contribution.
3. **Systems story too weak**: 6-14 GB/s decode is not competitive with DFloat11 (30-60 GB/s) or ZipServ (fused GEMM).
4. **Experimental scope too narrow**: Only Qwen models, simulated INT4, no real FP8 checkpoints.
5. **Missing key baseline**: Huff-LLM (Feb 2025) does end-to-end lossless inference with hardware-aware Huffman.
6. **Paper is between chairs**: Neither strong methods nor strong systems.

**Recommended Rescue Paths:**
- Fix FP8 comparison immediately
- Decide: systems paper (need fused GEMM) or analysis paper (format-dependent entropy structure)
- Don't force one codec across all formats — show a regime map of when each approach wins

### Evaluation 2: Idea Generation (GPT-5.4 xhigh)

**Top 3 recommended ideas (out of 10 generated):**

1. **Temporal innovation coding for optimizer states** — Store Adam states as innovations (deltas from recurrence formula). Our pilot already showed 86-94% of v values predicted exactly!
2. **Scale-conditioned FP8 entropy coding** — Condition on per-block scale metadata. FP8's 256-symbol alphabet makes context modeling cheap enough to beat i.i.d.
3. **Compressibility-aware channel permutation** — Exploit exact hidden-unit/head permutation symmetries to reorder weights for better block-local entropy without changing model outputs.

**Other notable ideas:**
- Joint coding of INT4 weights + quantization metadata (scales/zero-points)
- Row/column factorization of exponent fields (e_ij = a_i + b_j + r_ij)
- Fused FP8/INT4 decode + GEMM kernel (ZipServ-style for low-precision)

---

## Synthesis: Recommended Direction Pivot

Based on both evaluations, the current "full-value ANS for all formats" approach is **not publishable** as-is. The strongest pivot options are:

### Option A: Optimizer State Compression (Most Novel)
- Our v prediction pilot (86-94% exact, <1.6 bits residual) is genuinely unprecedented
- No published work addresses lossless optimizer state compression during training
- GPT-5.4 independently generated this as its #1 idea
- Combines with: hooked CPU offload (already -2274 MB), nvCOMP GPU ANS (85-126 GB/s)

### Option B: Format-Dependent Entropy Analysis Paper
- Don't claim "full-value beats exponent-only everywhere"
- Instead: systematic entropy decomposition across BF16/FP8/INT4 showing WHEN each codec wins
- Include scale-conditioned FP8 coding (GPT-5.4's idea #3)
- More of an analysis/benchmark paper than a methods paper

### Option C: Systems Paper (Hardest, Highest Impact)
- Fused decode+GEMM for FP8/INT4 (ZipServ-style)
- Requires significant CUDA kernel engineering
- 4-8 weeks of effort
