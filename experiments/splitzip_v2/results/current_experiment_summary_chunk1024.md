# Current SplitZip v2 Results, Chunk Size 1024

Source: `experiments/splitzip_v2/results/paper_rerun_v2_qwen32_chunk1024.json`

Model/workload: `Qwen/Qwen3-32B`, real BF16 KV activations assembled as `65536 x 4096`.
Throughput is reported in GB/s over the uncompressed native tensor byte count. Error values are standard errors over 10 measurements.

## Baseline Summary for Future Plotting

The rows below combine the existing baseline measurements with the current SplitZip v2 chunk-local result. For a strict apples-to-apples figure, rerun every baseline on the same `Qwen/Qwen3-32B`, `65536 x 4096` tensor.

| Method | Ratio (x) | Encode GB/s | Decode GB/s | Error for Encode | Error for Decode | Source / note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| nvCOMP LZ4 | 1.019 | 13.399 | 137.109 | 0.424 | 8.796 | existing measured baseline |
| ZipNN | 1.515 | 1.150 | 1.650 | N/A | N/A | user-provided reported value |
| DFloat11 | 1.423 | 0.004 | 468.157 | 0.00005 | 2.499 | existing measured baseline |
| ZipServ | 1.373 | 0.046 | 499.500 | N/A | N/A | measured/provided decode update |
| SplitZip v2 | 1.324 | 435.288 | 1763.787 | 2.816 | 25.327 | current chunk-local Top-16, chunk size 1024 |
| Falcon | 0.559 | 8.859 | 14.442 | 0.202 | 0.389 | FP32 codec on BF16 values cast to FP32; BF16-equivalent payload metrics |

## BF16 Ablations

| Experiment | Variant | Coverage | Ratio (x) | Encode GB/s | Decode GB/s | Escape Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Top-k | Top-8 | 92.11% | 1.038 | 440.080 | 710.537 | 7.89% |
| Top-k | Top-16 | 99.84% | 1.324 | 435.288 | 1763.787 | 0.16% |
| Escape metadata | Top-16 + positions | 99.84% | 1.324 | 435.288 | 1763.787 | 0.16% |
| Escape metadata | Top-15 sentinel | 99.73% | 1.331 | 396.004 | 620.754 | 0.27% |
| Calibration | Pre-calibrated | 99.84% | 1.324 | 430.103 | 1745.133 | 0.16% |
| Calibration | Dynamic Top-16 | 99.84% | 1.324 | 80.651 | 1779.077 | 0.16% |

## Calibration Granularity

| Granularity | Coverage | Projected Ratio (x) | Codebook Bytes | Encode GB/s | Decode GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Per tensor | 99.84% | 1.324 | 16 | 435.288 | 1763.787 |
| Per token | 99.90% | 1.324 | 1048576 | N/A | N/A |
| Per channel | 99.90% | 1.327 | 65536 | N/A | N/A |

Per-token and per-channel rows report full-shape coverage and projected size only; the current high-throughput GPU kernel implements the per-tensor codebook.

## Qwen3-32B Breakdown

Transport mode: RoCE `4x200G`.

| Seq Len | Native ms | Encode ms | Transfer ms | Decode ms | SplitZip Total ms | Encode % | Transfer % | Decode % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 56.470 | 8.119 | 42.063 | 2.920 | 53.102 | 15.29 | 79.21 | 5.50 |
| 16384 | 441.418 | 24.134 | 323.473 | 6.223 | 353.829 | 6.82 | 91.42 | 1.76 |
| 65536 | 1749.276 | 79.958 | 1297.603 | 19.393 | 1396.954 | 5.72 | 92.89 | 1.39 |

## FP8 Exact Results

| Format | Scheme | Coverage | Ratio vs FP8 | Ratio vs BF16 | Encode GB/s | Decode GB/s | Escape Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E4M3 | Top-8 | 92.17% | 0.933 | 1.866 | 219.636 +/- 1.529 | 366.852 +/- 2.507 | 7.83% |
| E5M2 | Top-8 | 92.28% | 1.049 | 2.097 | 221.560 +/- 14.562 | 340.803 +/- 0.905 | 7.72% |
| E5M2 | Top-16 | 99.84% | 1.136 | 2.273 | 249.713 +/- 0.752 | 564.858 +/- 6.520 | 0.16% |

## Falcon Result

Falcon source: `/data02/home/yilian2/project/Falcon`.

Compiled adapter:

```bash
experiments/splitzip_v2/results/falcon/falcon_float_bench
```

Run command:

```bash
conda run -n quant python -m experiments.splitzip_v2.falcon_baseline_bench \
  --falcon-root /data02/home/yilian2/project/Falcon \
  --model Qwen/Qwen3-32B \
  --rows 65536 \
  --width 4096 \
  --device cuda:0 \
  --falcon-device 0 \
  --repeats 10 \
  --output experiments/splitzip_v2/results/falcon_baseline_qwen32.json
```

Output: `experiments/splitzip_v2/results/falcon_baseline_qwen32.json`

| Metric | Value |
| --- | ---: |
| Round trip correct | yes |
| Falcon chunk size | 8396800 |
| Compressed bytes | 960049280 |
| Ratio vs FP32 input | 1.118 |
| Encode throughput, FP32 input | 17.718 +/- 0.403 GB/s |
| Decode throughput, FP32 input | 28.884 +/- 0.778 GB/s |
| Ratio vs original BF16 payload | 0.559 |
| Encode throughput, BF16-equivalent | 8.859 +/- 0.202 GB/s |
| Decode throughput, BF16-equivalent | 14.442 +/- 0.389 GB/s |

Important caveat: this Falcon checkout exposes FP32/FP64 codecs, not a BF16-native codec. The adapter casts real BF16 activation values to FP32 and reports both Falcon-native FP32 metrics and BF16-equivalent payload metrics. For the BF16 baseline chart, use the BF16-equivalent row only if you explicitly label it as an FP32-cast Falcon measurement.
