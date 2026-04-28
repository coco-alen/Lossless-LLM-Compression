# SplitZip v2 Run Summary

## Chunk-Local GPU Codec Breakdown
- Shape elements: 268,435,456; chunk size: 1024
- Ratio: 1.3296x; escapes: 26,168; coverage: 99.9903%
- Encode total: 492.28 GB/s; decode total: 1029.91 GB/s

## Additional Baselines (Llama-3-8B activations)
- Shape (65536, 4096):
  - nvcomp_cascaded: ratio=0.9996x, encode=111.83 GB/s, decode=155.17 GB/s, ok=True
  - nvcomp_bitcomp: ratio=0.9983x, encode=341.53 GB/s, decode=147.69 GB/s, ok=True
  - zipserv_tca_tbe: ratio=1.2355x, encode=n/a GB/s, decode=1460.91 GB/s, ok=True
  - falcon: external_required

## FP8 E5M2 Chunk-Local Top-8
- Global Top-8 coverage: 98.5183%
- Chunk-local Top-8 coverage: 98.7740%
- Estimated ratio global/chunk-local: 1.2790x / 1.2361x

## FP8 E4M3 Chunk-Local Top-8
- Global Top-8 coverage: 98.9974%
- Chunk-local Top-8 coverage: 99.2336%
- Estimated ratio global/chunk-local: 1.1138x / 1.0804x

## Exponent Stability: exponent_stability_qwen32.json
- Qwen/Qwen3-32B: status=ok
  - profiled KV layers=64, non-standard layers=0
  - K Top-16=99.890%, entropy=2.813 b
  - V Top-16=99.810%, entropy=3.318 b

## Exponent Stability: exponent_stability_qwen3_next.json
- Qwen/Qwen3-Next-80B-A3B-Instruct: status=ok
  - profiled KV layers=12, non-standard layers=36
  - K Top-16=99.974%, entropy=2.708 b
  - V Top-16=99.951%, entropy=2.819 b

## Mooncake KV Transfer
- Llama-3-8B:
  - bs1_seq bs=1 seq=131072: native=270.380 ms, compressed_transfer=212.735 ms, ratio=1.3290x, ok=True
  - bs16_seq bs=16 seq=65536: native=2438.714 ms, compressed_transfer=1894.402 ms, ratio=1.3290x, ok=True
  - seq1024_bs bs=256 seq=1024: native=556.427 ms, compressed_transfer=364.217 ms, ratio=1.3290x, ok=True
  - seq32768_bs bs=128 seq=32768: native=9175.726 ms, compressed_transfer=6023.413 ms, ratio=1.3290x, ok=True

- Qwen3-30B-A3B:
  - bs1_seq bs=1 seq=131072: native=132.026 ms, compressed_transfer=106.257 ms, ratio=1.3150x, ok=True
  - bs16_seq bs=16 seq=65536: native=1271.064 ms, compressed_transfer=913.403 ms, ratio=1.3156x, ok=True
  - seq1024_bs bs=256 seq=1024: native=318.752 ms, compressed_transfer=219.845 ms, ratio=1.3152x, ok=True
  - seq32768_bs bs=128 seq=32768: native=4623.140 ms, compressed_transfer=3780.610 ms, ratio=1.3157x, ok=True

