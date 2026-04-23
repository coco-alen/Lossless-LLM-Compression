# SplitZip: Lossless KV Cache Compression for Disaggregated PD Serving

## What This Is

SplitZip is a lossless compression codec for BF16 KV cache tensors, designed to reduce transfer time in disaggregated prefill-decode (PD) LLM serving systems like Mooncake, DistServe, and vLLM.

**Key numbers:**
- **1.301–1.333x** BF16 lossless compression (bitwise exact, zero errors)
- **1.059x / 1.214x** extra lossless compression on native FP8 E4M3 / E5M2
- **1.225x** BF16 end-to-end speedup on Mooncake TCP loopback
- **1.111x** E5M2 end-to-end speedup over raw native E5M2 on Mooncake TCP loopback
- **2.200x** E5M2+SplitZip end-to-end speedup relative to raw BF16 transfer

## The Algorithm

### Core Observation

BF16 values use 8 bits for the exponent, but in KV caches, only **15–16 unique exponent values** ever appear (out of 256 possible). The top 8 values (121–128) cover 88–95% of all elements across Qwen, Llama, and Phi families. This is a numerical property of layer-normalized neural network activations, not model-specific.

### Compression (Encode)

Given an input BF16 tensor of `n` elements (16 bits each):

```
BF16 layout: [sign(1) | exponent(8) | mantissa(7)]
```

1. **Split** each BF16 value into two 8-bit fields:
   - `exponent = (value >> 7) & 0xFF`
   - `sign_mantissa = ((value >> 8) & 0x80) | (value & 0x7F)`

2. **Encode exponents** via a 256-entry LUT:
   - Top 15 most frequent exponents → codes 0–14
   - All others → escape code 15

3. **Pack** pairs of 4-bit codes into bytes: `(code[i] << 4) | code[i+1]`

4. **Collect escapes** (~0.02% of elements): store their positions and raw exponent values in a tiny side stream.

**Output:** `packed_exponents (n/2 bytes) + sign_mantissa (n bytes) + escape_stream (~0.04% overhead)`

**Compression ratio:** `2n / (n/2 + n + ε) ≈ 1.333x`

### Decompression (Decode)

1. **Unpack** nibble bytes → 4-bit codes via shift and mask
2. **Decode** exponents via 16-entry LUT
3. **Recombine** with sign_mantissa: `(sm & 0x80) << 8 | exp << 7 | sm & 0x7F`
4. **Fix escapes**: a tiny kernel patches escaped positions with their correct exponents

The output is **bitwise identical** to the original BF16 tensor.

### Near-Lossless 3-Bit Variant

Uses only 8 codes (3 bits), packing 8 exponents into 3 bytes (24 bits). Ratio: 1.455x. The ~1.25% of elements outside the top-8 are mapped to the nearest valid code. This is a diagnostic/ablation path only; the main contribution and the FP8 results below are lossless/native-exact.

### Native FP8 Lossless Extension

SplitZip can also compress native FP8 KV cache bytes without changing the FP8 representation:

```
E4M3 layout: [sign(1) | exponent(4) | mantissa(3)]
E5M2 layout: [sign(1) | exponent(5) | mantissa(2)]
```

The implemented exact FP8 path keeps sign+mantissa bits verbatim, packs top-8 exponent codes into 3-bit fields, and stores uncommon exponents in a compact block-local escape stream. On Qwen2.5-1.5B tiled KV, the current exact ratios are:

- E4M3 + SplitZip: **1.059x** over native FP8, **2.118x** versus BF16 bytes
- E5M2 + SplitZip: **1.214x** over native FP8, **2.428x** versus BF16 bytes

E4M3 has limited exact gains because its escape rate is high enough to consume most of the nominal 3-bit exponent savings. E5M2 is the stronger native-FP8 target.

### Layer-Pipelined Transfer

In a real PD serving system, KV cache is transferred layer by layer. SplitZip pipelines encode/transfer/decode across layers:

```
Layer 0:  [encode] → [transfer] → [decode]
Layer 1:            [encode] → [transfer] → [decode]
Layer 2:                      [encode] → [transfer] → [decode]
```

Since per-layer encode (0.24 ms) and decode (0.18 ms) are much smaller than per-layer transfer (1–14 ms), the codec overhead is completely hidden.

---

## Codebase Structure

```
experiments/
├── kv_cache_profile.py              # Profile KV exponent entropy across models
├── kv_decomp_latency.py             # Measure per-page decode latency
└── splitzip/
    ├── __init__.py
    ├── lossless_fast.py             # ★ Production lossless codec (4-bit + escape)
    ├── opt_rounds3.py               # ★ 3-bit near-lossless vectorized codec
    ├── kv_codec.py                  # KV codec with PD transfer simulation
    ├── baseline_comparison.py       # Compare vs FP8, LZ4, zstd
    ├── fp8_kv_profile.py            # FP8 KV cache entropy profiling
    ├── fp8_e5m2_top8_compact_bench.py # ★ Exact FP8 E4M3/E5M2 compact codec
    ├── quality_eval.py              # 140-prompt quality evaluation
    ├── pipeline_and_quality.py      # Pipeline simulation + long-context test
    ├── real_kv_eval.py              # ★ Full eval on Qwen3-30B-A3B real KV caches
    ├── mooncake_integration.py      # Real Mooncake Transfer Engine integration
    ├── mooncake_format_latency.py   # ★ BF16/E4M3/E5M2 Mooncake latency summary
    ├── bandwidth_simulation.py      # v1 simulation (.cpu() serialize path)
    ├── bandwidth_simulation_v2.py   # ★ v2 simulation (pinned-memory DMA path)
    ├── e2e_simulation.py            # ★ Wall-clock end-to-end measurement
    └── main_track_push.py           # Long-context + serving-path proxy

paper/                               # LaTeX paper (10 pages, compiled PDF)
dfloat11/                            # Base DFloat11 codebase (dependency)
```

### Key Files

| File | What It Does | When to Use |
|------|-------------|-------------|
| `lossless_fast.py` | Truly lossless 4-bit codec with escape handling. The production implementation. | Main codec for benchmarking and integration. |
| `fp8_e5m2_top8_compact_bench.py` | Native-FP8 exact E4M3/E5M2 top-8 codec with compact block-local escapes. | Reproduce FP8 lossless ratios and kernel timings. |
| `mooncake_format_latency.py` | Measures Mooncake TCP loopback payload latency and composes full encode-transfer-decode latency. | Reproduce BF16/E4M3/E5M2 end-to-end latency table. |
| `opt_rounds3.py` | 3-bit near-lossless codec with fully unrolled vectorized Triton kernels. | Higher compression (1.455x) when 1.25% approximation is acceptable. |
| `kv_codec.py` | Triton split/recombine kernels + PD transfer simulation at various bandwidths. | Pipeline speedup projections for different network tiers. |
| `quality_eval.py` | Runs 110+ prompts on a model, measures text match and logit difference. | Validate that compression does not affect model output. |
| `mooncake_integration.py` | Initializes Mooncake Transfer Engine, runs real compressed transfers. | End-to-end system integration validation. |
| `kv_cache_profile.py` | Profiles exponent entropy, zero fraction, per-layer/per-head statistics. | Verify the exponent concentration on a new model. |

---

## How to Reproduce

### Prerequisites

```bash
conda activate yipin_quant  # or your CUDA-enabled Python environment
pip install torch triton transformers dahuffman safetensors accelerate
pip install lz4 zstandard                    # for baseline_comparison.py
pip install mooncake-transfer-engine          # for Mooncake integration
conda install -c conda-forge rdma-core etcd -y # Mooncake runtime + metadata server
conda install -c conda-forge tectonic -y       # LaTeX build
```

Mooncake needs `libcudart.so.12` on the library path at runtime:

```bash
export LD_LIBRARY_PATH="$(python -c 'import nvidia.cuda_runtime, os; print(os.path.dirname(nvidia.cuda_runtime.__file__))')/lib:$LD_LIBRARY_PATH"
```

Hardware: NVIDIA H200 (or any Ampere/Hopper GPU with BF16 support).

### 1. Profile KV Cache Exponent Distribution

Verify the exponent concentration on any model:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/kv_cache_profile.py \
    --model Qwen/Qwen2.5-7B --device cuda --max_new_tokens 64
```

Expected output: exponent entropy ~2.7–3.0 bits, top-15 coverage ~99.5%, DFloat-style ratio ~1.48x.

### 2. Run Lossless Codec Benchmark

Measure encode/decode throughput and verify correctness:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/splitzip/lossless_fast.py
```

Expected output:
```
  268M: encode≈257 GB/s, decode≈470 GB/s, ratio=1.333x, escapes≈23K (0.016%), PASS ✓
  Pipeline: GPU-Direct→1.331x, CPU-RDMA→1.326x, RoCE4x200→1.321x, RoCE8x400→1.308x
```

### 3. Run 3-Bit Near-Lossless Benchmark

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/splitzip/opt_rounds3.py
```

Expected output: encode 1709 GB/s, decode 1204 GB/s, ratio 1.455x, pipeline 1.45x.

### 4. Compare Against Baselines (FP8, LZ4, zstd)

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.splitzip.baseline_comparison
```

Expected: SplitZip 1.333x lossless at 252 GB/s. LZ4 gets 1.0x (zero compression). FP8 gets 2x but lossy.

### 5. Quality Evaluation (140 Prompts)

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/splitzip/quality_eval.py
```

Expected: 110/110 text match on Qwen2.5-1.5B, zero logit difference. Takes ~10 minutes.

### 6. FP8 KV Cache Profiling

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/splitzip/fp8_kv_profile.py \
    --model Qwen/Qwen2.5-7B --device cuda
```

Expected entropy profile: FP8 E4M3 has a ~1.16–1.19x exponent-coding ceiling; FP8 E5M2 has a ~1.34–1.39x exponent-coding ceiling.

### 7. Exact Native-FP8 Codec Benchmark

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/splitzip/fp8_e5m2_top8_compact_bench.py \
    --fmt e4m3 --model Qwen/Qwen2.5-1.5B --size-mb 134

CUDA_VISIBLE_DEVICES=0 python experiments/splitzip/fp8_e5m2_top8_compact_bench.py \
    --fmt e5m2 --model Qwen/Qwen2.5-1.5B --size-mb 134
```

Expected current exact ratios on tiled Qwen KV:

| Native FP8 | Extra ratio | Total vs BF16 | Exact |
|------------|-------------|---------------|-------|
| E4M3 + SplitZip | 1.059x | 2.118x | PASS |
| E5M2 + SplitZip | 1.214x | 2.428x | PASS |

### 8. Long-Context + Pipeline Simulation

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/splitzip/pipeline_and_quality.py
```

Expected: lossless PASS at 128–2840 tokens, pipeline speedup 1.33x on all tiers.

### 9. Real Mooncake Transfer Engine Integration

Requires etcd running locally. The script starts two TransferEngine endpoints on localhost and measures both raw and compressed transfer times.

```bash
# Terminal 1: Start etcd (needed for Mooncake metadata)
etcd --data-dir /tmp/etcd-splitzip \
     --listen-client-urls http://localhost:2379 \
     --advertise-client-urls http://localhost:2379 &

# Terminal 2: Run integration (needs libcudart.so.12 on LD_LIBRARY_PATH)
export LD_LIBRARY_PATH="$(python -c 'import nvidia.cuda_runtime, os; print(os.path.dirname(nvidia.cuda_runtime.__file__))')/lib:$LD_LIBRARY_PATH"
CUDA_VISIBLE_DEVICES=0 python experiments/splitzip/mooncake_integration.py
```

Notes on the API:
- `write_bytes_to_buffer(addr, data, length)` — 3 args
- `transfer_sync_write(target_hostname, local_buf, remote_buf, length)` — target first
- Mooncake topology discovery will show `Found 0 HCAs` on machines without active RDMA; TCP loopback is used instead.

Expected on a machine without RDMA: Mooncake topology discovery will report 0 HCAs and use TCP. The older integration script includes Python serialization overhead; use the next step for the clean BF16/E4M3/E5M2 latency table.

### 10. BF16/E4M3/E5M2 Mooncake Loopback Latency

This is the current end-to-end latency summary used by the paper. It measures real Mooncake TCP loopback transfer times for the native and compressed payload sizes, then composes full latency with codec encode/decode throughput:

```bash
etcd --data-dir /tmp/etcd-splitzip-latency \
     --listen-client-urls http://localhost:2379 \
     --advertise-client-urls http://localhost:2379 &

conda run -n yipin_quant python experiments/splitzip/mooncake_format_latency.py
```

Expected on the current local TCP loopback setup:

| Format | Raw native | Full path | Speedup vs native | Speedup vs raw BF16 |
|--------|------------|-----------|-------------------|---------------------|
| BF16 + SplitZip | 49.60 ms | 40.49 ms | 1.225x | 1.225x |
| E4M3 + SplitZip | 25.06 ms | 26.96 ms | 0.929x | 1.840x |
| E5M2 + SplitZip | 25.06 ms | 22.55 ms | 1.111x | 2.200x |

Interpretation: E4M3's current exact ratio is too small to overcome codec overhead on fast local loopback, while E5M2 remains positive versus native FP8.

### 11. Real KV Cache Evaluation (Qwen3-30B-A3B)

Loads Qwen3-30B-A3B, generates real KV caches from 8 semantically diverse prompts, and runs the full codec evaluation.

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/splitzip/real_kv_eval.py
```

Expected on H200 (deterministic — identical across runs):
```
  Unique exponents: 38
  Entropy: 3.548 / 8 bits
  Top-8 coverage: 84.79%
  Top-15 coverage: 99.08%
  Overall: 32.6 MB → 25.2 MB
  Actual compression ratio: 1.2935x
  Overall escape rate: 0.9246%
  All layers correct: YES ✓
```

Real MoE KV has higher entropy (3.55 bits vs 2.55 on randn) → ratio 1.294x rather than 1.333x. Results are saved to `experiments/splitzip/real_kv_eval_results.json`.

### 12. Bandwidth Simulation (Pinned-Memory Fast Path)

Measures real GPU codec + real pinned-memory DMA, then simulates network transfer at datacenter-realistic bandwidths (10G–400G+). This is the right benchmark for the integration path — no Python serialization, models what a C++ integration would achieve.

```bash
CUDA_VISIBLE_DEVICES=2 python experiments/splitzip/bandwidth_simulation_v2.py
```

Expected on H200 (PCIe Gen5):
```
PCIe DMA: GPU→CPU 53.6 GB/s, CPU→GPU 53.4 GB/s
Llama-3-70B 64K: encode 1.05 ms, decode 0.18 ms, ratio 1.333x
At 25G Ethernet:   speedup 1.333x (saves 1718 ms / 21.5 GB KV transfer)
At 100G Ethernet:  speedup 1.333x (saves 431 ms)
At 400G RoCE:      speedup 1.333x (saves 110 ms)
```

### 13. End-to-End Wall-Clock Benchmark

Runs the actual full pipeline (real DMA + real kernels + calibrated network delay via `time.sleep`) and measures wall-clock time. Every stage uses `torch.cuda.synchronize()` to ensure DMA completion is included.

```bash
CUDA_VISIBLE_DEVICES=2 python experiments/splitzip/e2e_simulation.py
```

Expected wall-clock results (Llama-3-70B 64K, 268 MB/layer):

| Network | Raw E2E | SplitZip E2E | Speedup |
|---------|---------|--------------|---------|
| 10G Ethernet | 225 ms | 172 ms | 1.31x |
| 25G Ethernet | 96 ms | 75 ms | 1.28x |
| 100G Ethernet | 32 ms | 26 ms | 1.22x |
| 200G RoCE | 21 ms | 18 ms | 1.18x |
| 400G RoCE | 15 ms | 13 ms | 1.16x |

The per-layer breakdown at 100G confirms where time is spent: GPU encode 1.5 ms + DMA↓ 3.9 ms + network 16.1 ms + DMA↑ 4.0 ms + GPU decode 0.3 ms = 25.8 ms total SplitZip vs 31.5 ms raw (DMA 5.0 + net 21.5 + DMA 5.0).

**Note:** Small KV sizes (16.8 MB/layer) only beat raw at ≤25G bandwidths. Large KV (268 MB/layer) always wins up to 400G+.

### 14. Cross-Model Robustness

To verify the exponent pattern on a different model:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/kv_cache_profile.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cuda
```

Expected: same top-8 exponents (121–128), entropy 3.0–3.4 bits.

---

## Results Summary

### Codec Performance (268 MB per-layer, H200)

| Mode | Ratio | Encode | Decode | Errors | Correct |
|------|-------|--------|--------|--------|---------|
| Lossless (4-bit + escape) | 1.333x | 252 GB/s | 1460 GB/s | 0% | PASS |
| Near-lossless (3-bit) | 1.455x | 1709 GB/s | 1204 GB/s | 1.25% | ~PASS |

### End-to-End Wall-Clock Speedup (Llama-3-70B 64K, 268 MB/layer)

Real DMA + real kernels + simulated network delay, full `cuda.synchronize()` on every stage:

| Network | Raw E2E | SplitZip E2E | Speedup |
|---------|---------|--------------|---------|
| 10G Ethernet | 225 ms | 172 ms | 1.31x |
| 25G Ethernet | 96 ms | 75 ms | 1.28x |
| 100G Ethernet | 32 ms | 26 ms | 1.22x |
| 200G RoCE | 21 ms | 18 ms | 1.18x |
| 400G RoCE | 15 ms | 13 ms | 1.16x |

### Real KV Cache on Qwen3-30B-A3B (MoE)

| Metric | Value |
|--------|-------|
| Compression ratio | 1.294x |
| Escape rate | 0.925% |
| Exponent entropy | 3.548 bits |
| Top-15 coverage | 99.08% |
| Lossless | PASS (all 48 layers) |

### Pipeline Speedup (Llama-3-70B, 64K context, 80 layers)

| Network | Bandwidth | Lossless | Near-lossless |
|---------|-----------|----------|---------------|
| GPU-Direct RDMA | 15 GB/s | 1.331x | 1.452x |
| CPU-RDMA | 47 GB/s | 1.328x | 1.451x |
| RoCE 4×200G | 87 GB/s | 1.324x | 1.452x |
| RoCE 8×400G | 190 GB/s | 1.313x | 1.448x |

### Quality Validation

| Test | Result |
|------|--------|
| 110 prompts on Qwen2.5-1.5B | 100% text match, 0 logit diff |
| 30 prompts on TinyLlama 1.1B | 100% text match, 0 logit diff |
| Long-context 128–2840 tokens | All PASS, 0 lossless errors |
| Cross-family (Qwen, Llama, Phi) | Same exponent pattern (121–128) |

---

## How to Integrate Into Your System

SplitZip is a drop-in compression layer between KV generation and transfer. The integration pattern:

```python
from experiments.splitzip.lossless_fast import FastLosslessCodec

codec = FastLosslessCodec(device='cuda')
codec.calibrate(sample_kv_tensor)  # one-time, during model init

# Prefill side: compress before transfer
encoded = codec.encode(kv_tensor)        # → (packed, sm, esc_pos, esc_val, n, n_esc)
# ... serialize and transfer encoded buffers ...

# Decode side: decompress after transfer
kv_restored = codec.decode(*encoded)     # → original BF16 tensor (bitwise exact)
```

For Mooncake integration, see `experiments/splitzip/mooncake_integration.py` for the complete `allocate_managed_buffer → write → transfer_sync_write → read → decode` flow.
