# Additional Thesis Experiments

## 1. FP8 Exact Results

| Format | Scheme | Coverage | Ratio vs FP8 | Total Ratio vs BF16 | Encode GB/s | Decode GB/s | Escape Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E4M3 | top8_exact | top8=95.65% | 1.059 | 2.118 | 134.487 | 288.418 | 4.35% |
| E5M2 | top8_exact | top8=95.69% | 1.214 | 2.428 | 138.868 | 287.211 | 4.31% |
| E5M2 | top16_exact | top16=99.56% | 1.129 | 2.257 | 140.327 | 289.226 | 0.44% |

## 2. FP8 End-to-End Speedup vs Sequence Length

- Model: `Qwen3-32B`, transport: `RoCE 4x200G` (87.000 GB/s)

### e4m3_top8_exact

| Seq Len | Native ms | SplitZip ms | Speedup (x) | Ratio (x) | Encode GB/s | Decode GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 4.120 | 10.484 | 0.393 | 1.013 | 12.903 | 48.574 |
| 2048 | 6.641 | 11.024 | 0.602 | 1.008 | 24.683 | 93.292 |
| 4096 | 11.513 | 11.538 | 0.998 | 0.998 | 60.148 | 206.711 |
| 8192 | 21.578 | 22.764 | 0.948 | 0.982 | 97.145 | 265.887 |
| 16384 | 42.896 | 42.588 | 1.007 | 0.988 | 112.519 | 271.461 |
| 32768 | 83.361 | 80.208 | 1.039 | 1.020 | 124.749 | 281.513 |
| 65536 | 161.831 | 163.251 | 0.991 | 1.004 | 137.800 | 288.761 |

### e5m2_top8_exact

| Seq Len | Native ms | SplitZip ms | Speedup (x) | Ratio (x) | Encode GB/s | Decode GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 4.120 | 8.847 | 0.466 | 0.966 | 15.319 | 56.645 |
| 2048 | 6.641 | 13.303 | 0.499 | 0.977 | 20.406 | 71.955 |
| 4096 | 11.513 | 11.830 | 0.973 | 1.000 | 47.715 | 163.443 |
| 8192 | 21.578 | 20.717 | 1.042 | 1.044 | 92.481 | 266.826 |
| 16384 | 42.896 | 38.934 | 1.102 | 1.100 | 116.595 | 269.478 |
| 32768 | 83.361 | 73.051 | 1.141 | 1.149 | 129.359 | 279.704 |
| 65536 | 161.831 | 143.199 | 1.130 | 1.137 | 138.443 | 284.272 |

### e5m2_top16_exact

| Seq Len | Native ms | SplitZip ms | Speedup (x) | Ratio (x) | Encode GB/s | Decode GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 4.120 | 13.261 | 0.311 | 1.127 | 10.182 | 36.783 |
| 2048 | 6.641 | 11.840 | 0.561 | 1.126 | 22.930 | 97.272 |
| 4096 | 11.513 | 10.349 | 1.112 | 1.129 | 62.076 | 207.236 |
| 8192 | 21.578 | 19.166 | 1.126 | 1.131 | 101.209 | 267.180 |
| 16384 | 42.896 | 38.053 | 1.127 | 1.132 | 121.033 | 273.188 |
| 32768 | 83.361 | 73.467 | 1.135 | 1.134 | 133.413 | 283.042 |
| 65536 | 161.831 | 142.628 | 1.135 | 1.135 | 143.900 | 288.808 |

## 3. Serving Compute Feasibility

- Model proxy: `NousResearch/Meta-Llama-3-8B`, seq-len `32768`, layer `0`
- Original KV compute time: `0.102 ms`
- KV compute time with explicit decompression: `0.255 ms`
- KV compute time with projected merged kernel: `0.185 ms`
- Materialization copy removed in projection: `0.070 ms`
- Note: Merged-kernel number is a projection: explicit decompress+compute minus one measured BF16 materialization copy, floored by original compute time.

## 4. BF16 Top-8 vs Top-16 (Exact, With Escapes)

| Variant | Coverage | Ratio | Encode GB/s | Decode GB/s | Escape Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top-8 (3-bit) | 95.62% | 1.364 | 272.622 | 492.152 | 4.38% |
| Top-16 (4-bit) | 99.60% | 1.316 | 329.935 | 1263.322 | 0.40% |

- Better format on this real BF16 tensor: `top16_4bit_exact`

## 5. Cross-Dataset Calibration

- Dataset A: `wikitext-2-raw-v1/train`
- Calibrate on A, Top-16 coverage on A: `99.67%`

| Dataset B | Domain | Calibrate on A, Eval on B | Calibrate on B, Eval on B |
| --- | --- | ---: | ---: |
| wikitext-2-raw-v1/test | language | 99.65% | 99.65% |
| openai_humaneval/test | code | 99.66% | 99.66% |
| gsm8k/main/test | math | 99.64% | 99.64% |
| cais/mmlu/all/validation | knowledge | 99.63% | 99.63% |
| ptb_text_only/penn_treebank/test | language | 99.55% | 99.58% |

## 6. Calibration Granularity

| Scope | Coverage | Actual Ratio | Projected Ratio | Encode GB/s | Decode GB/s | vs Base Enc | vs Base Dec | Codebook Bytes | Bench Shape |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| per_tensor | 99.60% | 1.307 | 1.323 | 93.894 | 222.218 | 1.000x | 1.000x | 16 | 1024x4096 |
| per_token | 99.74% | 1.305 | 1.322 | 0.082 | 0.173 | 0.001x | 0.001x | 524288 | 1024x4096 |
| per_channel | 99.98% | 1.320 | 1.329 | 0.020 | 0.060 | 0.000x | 0.000x | 65536 | 1024x4096 |

## Artifacts

- JSON: `/data02/home/yilian2/project/Lossless-LLM-Compression/experiments/splitzip/thesis_additional_experiments.json`
- Markdown: `/data02/home/yilian2/project/Lossless-LLM-Compression/experiments/splitzip/thesis_additional_experiments.md`
