# Additional Thesis Experiments

## 1. FP8 Exact Results

| Format | Top-8 Coverage | Ratio vs FP8 | Total Ratio vs BF16 | Encode GB/s | Decode GB/s | Escape Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E4M3 | 95.65% | 1.059 | 2.118 | 136.017 | 289.473 | 4.35% |
| E5M2 | 95.69% | 1.214 | 2.428 | 135.828 | 288.132 | 4.31% |

## 2. Serving Compute Feasibility

- Model proxy: `NousResearch/Meta-Llama-3-8B`, seq-len `32768`, layer `0`
- Original KV compute time: `0.087 ms`
- KV compute time with explicit decompression: `0.204 ms`
- KV compute time with projected merged kernel: `0.135 ms`
- Materialization copy removed in projection: `0.070 ms`
- Note: Merged-kernel number is a projection: explicit decompress+compute minus one measured BF16 materialization copy, floored by original compute time.

## 3. BF16 Top-8 vs Top-16 (Exact, With Escapes)

| Variant | Coverage | Ratio | Encode GB/s | Decode GB/s | Escape Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top-8 (3-bit) | 95.62% | 1.364 | 270.135 | 493.775 | 4.38% |
| Top-16 (4-bit) | 99.60% | 1.316 | 316.424 | 1263.722 | 0.40% |

- Better format on this real BF16 tensor: `top16_4bit_exact`

## 4. Cross-Dataset Calibration

- Dataset A: `wikitext-2-raw-v1`
- Dataset B: `openai_humaneval`
- Calibrate on A, Top-16 coverage on A: `99.67%`
- Calibrate on A, Top-16 coverage on B: `99.66%`
- Calibrate on B, Top-16 coverage on B: `99.66%`

## 5. Calibration Granularity

| Scope | Coverage | Projected Ratio | Escape Rate | Codebook Bytes |
| --- | ---: | ---: | ---: | ---: |
| per_tensor | 99.60% | 1.323 | 0.40% | 16 |
| per_token | 99.74% | 1.322 | 0.26% | 524288 |
| per_channel | 99.98% | 1.329 | 0.02% | 65536 |

## Artifacts

- JSON: `/data02/home/yilian2/project/Lossless-LLM-Compression/experiments/splitzip/thesis_additional_experiments.json`
- Markdown: `/data02/home/yilian2/project/Lossless-LLM-Compression/experiments/splitzip/thesis_additional_experiments.md`
