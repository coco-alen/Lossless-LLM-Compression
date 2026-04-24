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
| DietGPU | 1.429 | 530.482 | 630.833 | measured |
| nvCOMP LZ4 | 1.019 | 13.399 | 137.109 | measured |
| DFloat11 | 1.423 | 0.004 | 468.157 | measured |
| ZipServ | 1.373 | 0.046 | 1.280 | measured |
| SplitZip | 1.316 | 332.506 | 1232.124 | measured |
| ZipNN | 1.515 | 1.150 | 1.650 | reported |

## 2. Transfer Time vs Sequence Length

### Llama-3-8B

- Layers: 32, KV heads: 8, head dim: 128, block width: 2048

#### CPU-RDMA

| Seq Len | Native ms | SplitZip ms | Speedup (x) | Ratio (x) | Enc GB/s | Dec GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1.972 | 3.351 | 0.588 | 1.302 | 20.541 | 51.213 |
| 1024 | 3.374 | 4.352 | 0.775 | 1.302 | 31.923 | 87.845 |
| 2048 | 6.058 | 4.731 | 1.281 | 1.313 | 68.170 | 178.747 |
| 4096 | 12.067 | 9.150 | 1.319 | 1.320 | 118.924 | 343.142 |
| 8192 | 24.099 | 19.550 | 1.233 | 1.324 | 208.343 | 767.270 |
| 16384 | 48.155 | 36.329 | 1.326 | 1.326 | 273.571 | 1204.187 |
| 32768 | 96.342 | 72.634 | 1.326 | 1.326 | 303.022 | 1270.833 |

#### RoCE 4x200G

| Seq Len | Native ms | SplitZip ms | Speedup (x) | Ratio (x) | Enc GB/s | Dec GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1.951 | 3.335 | 0.585 | 1.302 | 20.541 | 51.213 |
| 1024 | 3.333 | 4.321 | 0.771 | 1.302 | 31.923 | 87.845 |
| 2048 | 5.885 | 4.668 | 1.261 | 1.313 | 68.170 | 178.747 |
| 4096 | 10.844 | 8.359 | 1.297 | 1.320 | 118.924 | 343.142 |
| 8192 | 21.063 | 19.302 | 1.091 | 1.324 | 208.343 | 767.270 |
| 16384 | 41.445 | 31.446 | 1.318 | 1.326 | 273.571 | 1204.187 |
| 32768 | 83.778 | 63.051 | 1.329 | 1.326 | 303.022 | 1270.833 |

### Qwen3-30B-A3B

- Layers: 48, KV heads: 4, head dim: 128, block width: 1024

#### CPU-RDMA

| Seq Len | Native ms | SplitZip ms | Speedup (x) | Ratio (x) | Enc GB/s | Dec GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1.952 | 4.236 | 0.461 | 1.176 | 12.050 | 26.151 |
| 1024 | 2.906 | 5.219 | 0.557 | 1.176 | 19.627 | 41.587 |
| 2048 | 4.968 | 5.600 | 0.887 | 1.222 | 36.978 | 82.950 |
| 4096 | 8.914 | 7.120 | 1.252 | 1.253 | 62.929 | 210.174 |
| 8192 | 17.779 | 13.943 | 1.275 | 1.276 | 112.534 | 394.950 |
| 16384 | 35.521 | 27.501 | 1.292 | 1.292 | 199.440 | 853.379 |
| 32768 | 71.001 | 54.492 | 1.303 | 1.304 | 260.897 | 1043.807 |

#### RoCE 4x200G

| Seq Len | Native ms | SplitZip ms | Speedup (x) | Ratio (x) | Enc GB/s | Dec GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1.942 | 4.227 | 0.459 | 1.176 | 12.050 | 26.151 |
| 1024 | 2.885 | 5.201 | 0.555 | 1.176 | 19.627 | 41.587 |
| 2048 | 4.927 | 5.567 | 0.885 | 1.222 | 36.978 | 82.950 |
| 4096 | 8.693 | 7.000 | 1.242 | 1.253 | 62.929 | 210.174 |
| 8192 | 16.008 | 12.875 | 1.243 | 1.276 | 112.534 | 394.950 |
| 16384 | 31.089 | 24.443 | 1.272 | 1.292 | 199.440 | 853.379 |
| 32768 | 61.167 | 47.600 | 1.285 | 1.304 | 260.897 | 1043.807 |

## 3. Qwen3-32B Transmission Breakdown

Transport mode: RoCE 4x200G

| Seq Len | Native transfer ms | SplitZip encode ms | SplitZip transfer ms | SplitZip decode ms | SplitZip total ms | Encode % | Transfer % | Decode % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 28.373 | 6.316 | 22.127 | 2.486 | 30.930 | 20.421 | 71.540 | 8.039 |
| 8192 | 104.873 | 9.966 | 80.001 | 2.532 | 92.499 | 10.774 | 86.489 | 2.737 |
| 16384 | 207.059 | 15.499 | 157.422 | 3.733 | 176.654 | 8.774 | 89.113 | 2.113 |
| 32768 | 416.165 | 29.082 | 313.783 | 6.917 | 349.782 | 8.314 | 89.708 | 1.977 |

## Artifacts

- JSON: `/data02/home/yilian2/project/Lossless-LLM-Compression/experiments/splitzip/thesis_experiment_data.json`
- This Markdown: `/data02/home/yilian2/project/Lossless-LLM-Compression/experiments/splitzip/thesis_experiment_data.md`
