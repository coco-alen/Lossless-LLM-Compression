# Baseline Comparison Table

Throughput is in GB/s. Parentheses report negative/positive fluctuation around the central value.

| Method | Ratio (x) | Encode GB/s | Decode GB/s | Source / caveat |
| --- | ---: | ---: | ---: | --- |
| DietGPU | 1.429 | 343.957 (-262.979/+155.757) | 426.664 (-374.507/+204.059) | legacy 10-repeat baseline; Included for completeness; older 32768x4096 Qwen2.5 activation workload. |
| nvCOMP LZ4 | 1.019 | 13.399 (-0.424/+0.424) | 137.109 (-8.796/+8.796) | existing measured baseline |
| nvCOMP Cascaded | 1.000 | 111.828 (-0.824/+0.824) | 155.165 (-5.570/+5.570) | nvCOMP adapter, largest available shape; Shape [65536, 4096]. |
| nvCOMP Bitcomp | 0.998 | 341.529 (-7.081/+7.081) | 147.693 (-4.832/+4.832) | nvCOMP adapter, largest available shape; Shape [65536, 4096]. |
| DFloat11 | 1.423 | 4.00e-03 (-5.00e-05/+5.00e-05) | 468.157 (-2.499/+2.499) | existing measured baseline |
| ZipNN | 1.515 | 1.150 | 1.650 | user-provided reported value; User-provided reported result. |
| ZipServ | 1.236 | 0.054 (-2.11e-04/+2.11e-04) | 499.500 | ZipServ CPU encode bench + provided decode; Public wrapper encode; decode value kept at 499.5 GB/s per current baseline. |
| ZipServ/TCA-TBE | 1.236 | N/A | 1460.912 (-1.399/+1.399) | GPU-resident ZipServ/TCA-TBE boundary; Decode-only GPU-resident path; public encode is CPU-side. |
| Falcon | 0.559 | 8.859 (-1.784/+0.341) | 14.442 (-3.274/+0.916) | current Falcon FP32-cast run; Falcon exposes FP32/FP64 codecs here; BF16 values are cast to FP32 and reported as BF16-equivalent payload metrics. |
| SplitZip | 1.324 | 472.378 (-3.500/+4.234) | 2181.845 (-38.489/+34.355) | current chunk-local Top-16; Chunk size 1024; ratio from real Qwen3-32B activations; throughput from idle preallocated kernel-stage repeats. |
