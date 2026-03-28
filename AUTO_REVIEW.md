# Auto Review Log

**Topic**: Lossless compression for multi-precision LLM weights — full-value entropy coding as oracle baseline
**Started**: 2026-03-28
**Reviewer**: GPT-5.4 via Codex MCP (model_reasoning_effort: xhigh)

---

## Round 1 (2026-03-28)

### Assessment (Summary)
- Score: **5/10** (main track), **6.5/10** (D&B track)
- Verdict: **Almost** for D&B/workshop. **No** for NeurIPS main.
- Key criticisms:
  1. FP8 results use BF16→FP8 casting, not native FP8 checkpoints
  2. Contribution identity unstable (oracle baseline vs GPU codec)
  3. Benchmark breadth too narrow (only Qwen, Mistral, one GPTQ)
  4. GPU batched comparison not apples-to-apples
  5. INT4 story thin (only one real checkpoint)

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Score**: NeurIPS main track: **5/10**. NeurIPS Datasets & Benchmarks: **6.5/10**.

This is a real paper with one genuinely interesting finding: FP8 compressibility appears much more model- and format-dependent than prior exponent-centric codecs would suggest. The BF16 oracle result is solid, the FP8 ablation is useful, and the negative results increase credibility. But for a top venue, the paper is still not evidence-complete.

**Ready?** **Almost** for D&B/workshop. **No** for NeurIPS main.

**Critical Weaknesses**

1. The headline FP8 result still rests on casted FP8, not native FP8 checkpoints.
   Minimum fix: run ablation on real native FP8 checkpoints from at least two families.

2. Contribution identity unstable. Full-value ANS is an oracle baseline, not strong algorithmic novelty.
   Minimum fix: reframe as analysis + benchmark + reference baseline paper.

3. Benchmark breadth too narrow. Core tables cover Qwen, Mistral, one GPTQ only.
   Minimum fix: add one more dense family plus one more real INT4 quantizer.

4. GPU batched comparison not apples-to-apples.
   Minimum fix: matched baselines or stop implying serving-level win.

5. INT4 story thin.
   Minimum fix: evaluate one more real layout (AWQ/HQQ/Marlin).

**Single Highest-Leverage Improvement**: Validate FP8 story on real native FP8 checkpoints.

</details>

### Actions Taken
1. **Native FP8 benchmarks**: Tested 3 native FP8 checkpoints (Qwen3-0.6B-FP8, Qwen3-4B-Instruct-2507-FP8, Llama-3.2-1B-Instruct-FP8)
2. **Additional model coverage**: Added Qwen3-4B (BF16 + FP8 cast)
3. **AWQ INT4 checkpoint**: Benchmarked Qwen2.5-7B-Instruct-AWQ

### Results

#### Native FP8 (KEY NEW FINDING)
| Model | Source | FP8-only Ratio | Unique vals |
|-------|--------|---------------|------------|
| Qwen3-0.6B-FP8 | Official Qwen | **82.92%** | 254 |
| Qwen3-4B-Instruct-2507-FP8 | Official Qwen | **82.52%** | ~254 |
| Llama-3.2-1B-Instruct-FP8 | RedHatAI | **81.74%** | ~254 |

Native FP8 ~82-83% vs casted FP8 ~68-70% (Qwen) or ~37% (Mistral).

#### Additional BF16
| Model | BF16 Ratio | FP8 Cast Ratio |
|-------|-----------|---------------|
| Qwen3-4B | **66.08%** | **68.07%** |

#### AWQ INT4
| Component | Ratio |
|-----------|-------|
| qweight | **91.54%** |
| scales | 68.0% |
| TOTAL | **80.90%** |

### Status
Continuing to Round 2 with reviewer update.

---

## Round 2 (2026-03-28)

### Assessment (Summary)
- Score: **6/10** (main track), **7.5/10** (D&B track)
- Verdict: **Almost** for D&B after rewrite. **No** for main.
- Key criticisms:
  1. Manuscript out of sync with strongest results
  2. FP8 ablation only on casted, need native FP8 ablation
  3. "Quantization pipeline determines compressibility" claim needs mechanism analysis
  4. INT4 still only one architecture
  5. GPU codec section too prominent

### Reviewer Raw Response

<details>
<summary>Click to expand</summary>

Score: NeurIPS main track: 6/10. NeurIPS Datasets & Benchmarks: 7.5/10.

The native-FP8 and AWQ evidence is a real improvement. This is now a substantially stronger empirical paper.

Biggest concern from Round 1 (does FP8 matter on real checkpoints?) is now largely addressed. The answer is interesting and publication-worthy.

Remaining weaknesses:
1. Manuscript out of sync with results (abstract/contributions/tables still tell the casted story)
2. FP8 ablation only on casted Qwen — need native FP8 factorization ablation
3. "Pipeline determines compressibility" claim needs mechanism (histogram/entropy analysis)
4. INT4 still one architecture (both GPTQ and AWQ are Qwen2.5-7B)
5. GPU codec still too prominent

Highest leverage fix: native-FP8 factorized-vs-full-value ablation plus manuscript rewrite.

</details>

### Actions Taken
1. **Native FP8 ablation** (NEW): Ran entropy-based factorization ablation on 3 native FP8 checkpoints
2. **Manuscript rewrite**: Rewrote abstract, contributions, added native FP8 tables, demoted GPU codec
3. **Mechanism analysis**: Native FP8 uses 254/256 values, MI(exp;s+m) < 0.04 bits

### Results

#### Native FP8 Factorization Ablation
| Method | Qwen3-0.6B-FP8 | Qwen3-4B-FP8 | Llama-3.2-1B-FP8 |
|--------|----------------|--------------|------------------|
| Full-value (H) | 6.663 bpv | 6.630 bpv | 6.700 bpv |
| Separated ANS | +0.034 | +0.038 | +0.029 |
| Exp ANS + raw s+m | +0.056 | +0.059 | +0.050 |
| ECF8-style | +0.106 | +0.109 | +0.100 |

Key insight: Native FP8 factorization penalty is SMALL (~0.1pp for ECF8-style).
MI(exp; s+m) ≈ 0.03-0.04 bits — fields are nearly independent.

#### Manuscript Updates
- Abstract rewritten: highlights pipeline-determines-compressibility finding
- Contributions reframed: #1 oracle, #2 pipeline insight, #3 small factorization penalty, #4 reference codec
- Added native FP8 table (Table 6) and factorization comparison (Table 7)
- GPU codec demoted to "Reference GPU Codec: Proof of Feasibility"
- Limitations updated (no longer cites "no native FP8" as limitation)
- Conclusion rewritten around central pipeline finding

### Status
Continuing to Round 3 for re-assessment.

---

## Round 3 (2026-03-28)

### Assessment (Summary)
- Score: **6.5/10** (main track), **8/10** (D&B track)
- Verdict: **Close to submission-ready** for D&B. Borderline for main.
- Key criticisms:
  1. Unit inconsistency (pp vs bpv) in factorization penalty
  2. GPU section still has misleading "exceeds memcpy" claim
  3. INT4 still one architecture
  4. Main-track novelty ceiling (empirical characterization, not new method)

### Reviewer Raw Response

<details>
<summary>Click to expand</summary>

Score: NeurIPS main track: 6.5/10. NeurIPS Datasets & Benchmarks: 8/10.

The substantive scientific concerns are now mostly addressed. The paper now has a coherent central claim, real native-FP8 evidence, native-FP8 factorization ablations, and a much better contribution framing.

For D&B, this is now close to submission-ready. For main track, it is still borderline because the contribution is primarily empirical characterization, not a new method or full system.

Remaining weaknesses:
1. Unit inconsistency (pp vs bpv) in manuscript
2. GPU section still says "exceeds dense memcpy throughput"
3. INT4 still one architecture
4. Main-track novelty ceiling

Minimum fixes: Fix units, delete misleading GPU claim, narrow INT4 claim or add another architecture.

</details>

### Actions Taken
1. **Unit fix**: Changed all factorization penalty numbers to bpv consistently, added ratio conversion in parentheses where helpful
2. **GPU fix**: Removed "exceeds dense memcpy" sentence, replaced with honest microbenchmark characterization
3. **INT4**: Will explicitly narrow claim to Qwen case study

### Status
Score 8/10 for D&B meets the positive threshold. Proceeding to Round 4 for final polish.

---

## Round 4 — Final (2026-03-28)

### Assessment (Summary)
- Score: **6.5/10** (main track), **8.5/10** (D&B track)
- Verdict: **YES, ready for submission to NeurIPS D&B**
- Remaining: Minor text polish only (not blockers)

### Reviewer Raw Response

<details>
<summary>Click to expand</summary>

Score: NeurIPS Datasets & Benchmarks: 8.5/10. NeurIPS main track: 6.5/10.

Yes, this is ready for submission to NeurIPS D&B. The paper now has a clean empirical identity, the strongest prior attack line on FP8 realism is resolved, and the manuscript reflects the updated story.

The core contribution is now credible and useful: full-value coding as the oracle baseline, plus the stronger finding that pipeline choice dominates compressibility. That is a good D&B paper.

What I would still change before submitting (minor polish):
- Fix FP8 unique values text (now done)
- Fix evaluation-object wording to include AWQ (now done)
- Make INT4 scope explicit in table caption (now done)

Those are not blockers.

</details>

### Actions Taken
- Fixed FP8 unique values text to distinguish casted vs native
- Updated evaluation objects wording to include AWQ
- Made INT4 table caption explicitly a "Qwen case study"

### Final Status: COMPLETE ✓

---

## Score Progression
| Round | Main Track | D&B Track | Verdict |
|-------|-----------|-----------|---------|
| 1 | 5/10 | 6.5/10 | Almost (D&B) |
| 2 | 6/10 | 7.5/10 | Almost (D&B) |
| 3 | 6.5/10 | 8/10 | Close to ready (D&B) |
| 4 | 6.5/10 | **8.5/10** | **Ready for submission (D&B)** |

## Method Description

The paper presents a systematic empirical analysis of lossless compression for multi-precision LLM weights (BF16, FP8, INT4). The core method is full-value ANS (asymmetric numeral systems) coding, which treats each weight value as a single symbol in its native representation and achieves near-entropy-optimal compression. This serves as an oracle baseline against which practical factorized codecs (which decompose values into exponent and mantissa fields) are measured.

The key finding is that the quantization pipeline—not just the number format—determines compressibility: native calibrated FP8 checkpoints (~83% ratio) are far less compressible than naive BF16→FP8 casts (37–70%), and GPTQ qweights (85%) differ from AWQ qweights (92%) on the same base model. A reference two-stream FP8 GPU codec demonstrates that 77.1% compression is achievable at 254 GB/s per-layer decode throughput on H200.

---

## Round 5 (2026-03-28, continued loop)

### Assessment (Summary)
- Score: **8.3/10** (D&B track)
- Verdict: Still ready, but identified 17 polish items
- Key issues: BLOCKER: table 3 caption inconsistency; IMPORTANT: approximate placeholders, missing AWQ entropy, no figures, narrow native FP8 coverage, sparse citations

### Reviewer Raw Response

<details>
<summary>Click to expand</summary>

Score: 8.3/10 for D&B. 17 weaknesses identified:
- 2 BLOCKERS: submission format, table 3 caption
- 12 IMPORTANT: FP8/checkpoint quantity blur, contribution overstates Llama, native FP8 breadth, INT4 weak, AWQ entropy missing, table 6 entropy-based, mechanism under-demonstrated, wording too strong, approximate placeholders, no figures, no reproducibility statement, sparse citations
- 3 NICE-TO-HAVE: more casted ablation rows, GPU section length, title

Single highest-leverage fix: add more native FP8 families + one non-Qwen INT4 → reach 9/10.

</details>

### Actions Taken
1. **Added 2 new native FP8 models**: Tencent HunYuan-1.8B-FP8 (static tensor-level) and IBM Granite-3.3-8B-FP8 (per-channel minmax). Now 5 models across 4 families.
2. **Fixed table 3 caption**: Removed "Qwen case study" label since table includes Mistral sim, added descriptive caption.
3. **Filled AWQ entropy**: 7.33 bpv (byte-level), ratio 91.6%.
4. **Replaced approximate placeholders**: Qwen3-4B BF16 entropy 10.58, FP8 cast entropy 5.48, 155 unique values. Removed all ~.
5. **Added reproducibility section**: Exact checkpoint IDs, hardware, library.
6. **Added GPTQ/AWQ citations**.
7. **Fixed contribution #1**: No longer overstates Llama coverage.
8. **Added per-stream note**: Clarified FP8/INT4 ratios are per-stream not full-checkpoint.
9. **Updated contribution #2**: Explicit "five models from four families", "qweight-only on Qwen2.5-7B".
10. **Updated limitations**: Native FP8 now 5 models / 4 families.
11. **Softened factorization wording**: "modest savings" not "negligible".

### New Results
| Model | Family | Quant Tool | H(full) | Ratio | MI | ECF8 gap |
|-------|--------|-----------|---------|-------|----|----------|
| Tencent HY-MT1.5-1.8B-FP8 | HunYuan | compressed-tensors | 6.737 | 84.2% | 0.027 | +0.098 |
| IBM Granite-3.3-8B-FP8 | Granite | compressed-tensors | 6.556 | 82.0% | 0.087 | +0.162 |

### Status
Continuing to Round 6.

---

## Round 6 (2026-03-28)

### Assessment (Summary)
- Score: **8.6/10** (D&B)
- Verdict: Strong D&B submission. One blocker: factorization claim contradicted by Granite (0.162 > 0.11).

### Actions Taken
- Fixed ≤0.11 bpv → 0.10-0.16 bpv everywhere
- Fixed casted FP8 unique value inconsistencies
- Fixed INT4 mixed analysis clarity
- Validated ANS compression on HunYuan (81.4%) and Granite (81.7%)

---

## Round 7 — Final (2026-03-28)

### Assessment (Summary)
- Score: **8.8/10** (D&B)
- Verdict: **SUBMIT. Nothing remains that would prevent submission.**
- Remaining: presentation polish only (figure, template, wording)

### Final Actions
- Softened contribution #3 wording ("limited but nonzero gains")
- Added per-layer range footnote for Mistral unique values

### Score Progression (continued)
| Round | D&B Score | Key Change |
|-------|-----------|------------|
| 1 | 6.5 | Initial — casted FP8 only, narrow coverage |
| 2 | 7.5 | Native FP8 + AWQ + reframing |
| 3 | 8.0 | Native FP8 ablation + manuscript rewrite |
| 4 | 8.5 | Unit fixes + GPU demoted + minor polish |
| 5 | 8.3 | Detailed re-review, 17 issues identified |
| 6 | 8.6 | +2 FP8 families (Tencent, IBM), exact numbers, citations |
| 7 | **8.8** | Granite contradiction fixed, final polish |

### Final Status: COMPLETE ✓
- **Target venue**: NeurIPS 2026 Datasets & Benchmarks
- **Remaining for camera-ready**: NeurIPS template, 1 overview figure, version pins
- **Models tested**: 10 distinct checkpoints across 4 model families
