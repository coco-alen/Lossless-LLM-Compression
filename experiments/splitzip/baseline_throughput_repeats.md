# Baseline Throughput Repeats

- Tensor shape: `32768 x 4096`
- Repeats requested: `10`
- Error values are sample standard deviation and standard error of the mean across repeated benchmark runs.
- ZipNN is reported-only and therefore has no local error bar.

| Method | N | Ratio Mean | Enc Mean GB/s | Enc Std | Enc SEM | Dec Mean GB/s | Dec Std | Dec SEM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DietGPU | 10 | 1.42911 | 343.957 | 168.557 | 53.3025 | 426.664 | 245.756 | 77.7148 |
| nvCOMP LZ4 | 10 | 1.01873 | 6.9861 | 4.11365 | 1.30085 | 57.6151 | 38.9796 | 12.3264 |
| DFloat11 | 10 | 1.42315 | 0.00405942 | 0.000321961 | 0.000101813 | 364.656 | 96.4534 | 30.5013 |
| ZipServ | 10 | 1.37315 | 0.0457328 | 0.00776592 | 0.0024558 | 1138.32 | 180.185 | 56.9795 |
| SplitZip | 10 | 1.31596 | 205.638 | 90.7512 | 28.698 | 898.768 | 345.955 | 109.401 |
| ZipNN | 0 | 1.51515 | 1.15 | - | - | 1.65 | - | - |

## Plot Error Bars

Use `*_std` for one-standard-deviation error bars, or `*_stderr` for standard-error bars.

