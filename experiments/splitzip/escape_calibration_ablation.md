# Escape / Calibration Ablation

- Tensor: real BF16 activation matrix from `Qwen/Qwen2.5-1.5B`, shape `32768 x 4096`
- Throughput is measured on the same tensor for all variants.

## 1. Escape Position vs. Mask/Sentinel

| Variant | Coverage | Escape Rate | Ratio | Encode GB/s | Decode GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| top16_escape_positions | 99.60% | 0.40% | 1.316 | 320.811 | 1255.358 |
| top15_sentinel_mask | 99.47% | 0.53% | 1.329 | 332.892 | 395.012 |

- Faster encode: `top15_sentinel_mask`
- Faster decode: `top16_escape_positions`

## 2. Pre-Calibration vs. Dynamic Top-16

| Variant | Ratio | Escape Rate | Encode GB/s | Decode GB/s | Calibration GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| precalibrated_top16 | 1.316 | 0.40% | 317.944 | 1257.335 | - |
| dynamic_top16 | 1.316 | 0.40% | 63.403 | 1259.979 | 80.736 |

- Encode slowdown from dynamic top16: `5.015x`
- Decode slowdown from dynamic top16: `0.998x`

