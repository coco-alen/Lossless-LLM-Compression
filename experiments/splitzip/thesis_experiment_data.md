# Thesis Experiment Data

## Notes

- Baseline codec comparison uses the measured `32768 x 4096` real-activation BF16 tensor from `codec_ablation_results_with_dietgpu_nvcomp.json`.
- ZipNN is included from the reported numbers already provided in the project discussion; it was not rerun locally in this dump.
- Transfer sweeps use real BF16 KV blocks extracted from one 1024-token forward per model, then assembled to the target sequence length.
- The transfer line-chart tables below use a pipelined staged-transfer model: native = raw BF16 DMA/network/DMA pipeline, SplitZip = encode + compressed DMA/network/DMA + decode.
- The Qwen3-32B breakdown table is additive sequential accounting so the stacked bars sum cleanly; it is separate from the pipelined wall-clock tables above.

## 1. Baseline Comparison

| Method | Ratio (x) | Encode GB/s | Decode GB/s | Source |
| --- | ---: | ---: | ---: | --- |
| nvCOMP LZ4 | 1.019 | 13.399 | 137.109 | measured |
| ZipNN | 1.515 | 1.150 | 1.650 | reported |
| DFloat11 | 1.423 | 0.004 | 468.157 | measured |
| ZipServ | 1.373 | 0.046 | 1.280 | measured |
| SplitZip | 1.316 | 332.506 | 1232.124 | measured |

## 2. Transfer Time vs Sequence Length

### Llama-3-8B

- Layers: 32, KV heads: 8, head dim: 128, block width: 2048

#### CPU-RDMA

| Seq Len | Native ms | SplitZip ms | Speedup (x) | Ratio (x) | Enc GB/s | Dec GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1.982 | 3.357 | 0.590 | 1.302 | 20.501 | 50.969 |
| 1024 | 3.276 | 4.109 | 0.797 | 1.302 | 33.901 | 89.514 |
| 2048 | 6.056 | 4.695 | 1.290 | 1.313 | 67.274 | 176.201 |
| 4096 | 12.068 | 9.152 | 1.319 | 1.320 | 114.056 | 345.782 |
| 8192 | 24.099 | 18.227 | 1.322 | 1.324 | 203.963 | 834.699 |
| 16384 | 48.177 | 36.353 | 1.325 | 1.326 | 248.322 | 1205.050 |
| 32768 | 96.306 | 72.666 | 1.325 | 1.326 | 303.147 | 1267.309 |
| 65536 | 192.639 | 145.247 | 1.326 | 1.326 | 328.214 | 1323.279 |

#### RoCE 4x200G

| Seq Len | Native ms | SplitZip ms | Speedup (x) | Ratio (x) | Enc GB/s | Dec GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1.962 | 3.342 | 0.587 | 1.302 | 20.501 | 50.969 |
| 1024 | 3.235 | 4.077 | 0.793 | 1.302 | 33.901 | 89.514 |
| 2048 | 5.792 | 4.632 | 1.250 | 1.313 | 67.274 | 176.201 |
| 4096 | 10.840 | 8.407 | 1.289 | 1.320 | 114.056 | 345.782 |
| 8192 | 21.123 | 16.534 | 1.278 | 1.324 | 203.963 | 834.699 |
| 16384 | 41.864 | 31.839 | 1.315 | 1.326 | 248.322 | 1205.050 |
| 32768 | 82.809 | 63.723 | 1.300 | 1.326 | 303.147 | 1267.309 |
| 65536 | 166.296 | 125.573 | 1.324 | 1.326 | 328.214 | 1323.279 |

### Qwen3-30B-A3B

- Layers: 48, KV heads: 4, head dim: 128, block width: 1024

#### CPU-RDMA

| Seq Len | Native ms | SplitZip ms | Speedup (x) | Ratio (x) | Enc GB/s | Dec GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2.263 | 6.510 | 0.348 | 1.176 | 7.837 | 15.248 |
| 1024 | 2.922 | 5.329 | 0.548 | 1.176 | 19.234 | 39.070 |
| 2048 | 4.821 | 5.691 | 0.847 | 1.222 | 36.361 | 83.310 |
| 4096 | 8.911 | 7.121 | 1.251 | 1.253 | 68.947 | 180.899 |
| 8192 | 17.779 | 13.942 | 1.275 | 1.276 | 119.142 | 315.774 |
| 16384 | 35.522 | 27.507 | 1.291 | 1.292 | 207.817 | 829.786 |
| 32768 | 71.023 | 54.472 | 1.304 | 1.304 | 274.168 | 1080.489 |
| 65536 | 141.997 | 108.817 | 1.305 | 1.306 | 309.659 | 1122.749 |

#### RoCE 4x200G

| Seq Len | Native ms | SplitZip ms | Speedup (x) | Ratio (x) | Enc GB/s | Dec GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2.253 | 6.501 | 0.347 | 1.176 | 7.837 | 15.248 |
| 1024 | 2.901 | 5.312 | 0.546 | 1.176 | 19.234 | 39.070 |
| 2048 | 4.780 | 5.658 | 0.845 | 1.222 | 36.361 | 83.310 |
| 4096 | 8.555 | 7.022 | 1.218 | 1.253 | 68.947 | 180.899 |
| 8192 | 16.002 | 12.923 | 1.238 | 1.276 | 119.142 | 315.774 |
| 16384 | 31.179 | 24.648 | 1.265 | 1.292 | 207.817 | 829.786 |
| 32768 | 61.790 | 47.077 | 1.313 | 1.304 | 274.168 | 1080.489 |
| 65536 | 122.211 | 96.027 | 1.273 | 1.306 | 309.659 | 1122.749 |

## 3. Qwen3-32B Transmission Breakdown

Transport mode: RoCE 4x200G

| Seq Len | Native transfer ms | SplitZip encode ms | SplitZip transfer ms | SplitZip decode ms | SplitZip total ms | Encode % | Transfer % | Decode % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 28.197 | 6.523 | 22.186 | 2.613 | 31.322 | 20.825 | 70.832 | 8.343 |
| 8192 | 104.919 | 12.852 | 81.602 | 2.686 | 97.140 | 13.231 | 84.004 | 2.765 |
| 16384 | 208.470 | 15.835 | 160.499 | 3.912 | 180.245 | 8.785 | 89.045 | 2.170 |
| 32768 | 413.854 | 30.016 | 318.385 | 7.043 | 355.445 | 8.445 | 89.574 | 1.982 |
| 65536 | 829.465 | 53.385 | 633.788 | 13.221 | 700.394 | 7.622 | 90.490 | 1.888 |

## Artifacts

- JSON: `/data02/home/yilian2/project/Lossless-LLM-Compression/experiments/splitzip/thesis_experiment_data.json`
- This Markdown: `/data02/home/yilian2/project/Lossless-LLM-Compression/experiments/splitzip/thesis_experiment_data.md`
