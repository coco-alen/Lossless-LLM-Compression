# SplitZip: Lossless KV Cache Compression for Disaggregated PD Serving

## What This Is

SplitZip is a lossless compression codec for BF16 KV cache tensors, designed to reduce transfer time in disaggregated prefill-decode (PD) LLM serving systems like Mooncake, DistServe, and vLLM.

**Key numbers:**
- **1.333x** lossless compression (bitwise exact, zero errors)
- **252 / 1460 GB/s** encode / decode on H200 GPU
- **1.16–2.27x** real transfer speedup on Mooncake Transfer Engine
- **1.31–1.33x** end-to-end pipeline speedup across all tested bandwidth tiers

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

Uses only 8 codes (3 bits), packing 8 exponents into 3 bytes (24 bits). Ratio: 1.455x. The ~1.25% of elements outside the top-8 are mapped to the nearest valid code. In testing across 140 prompts and 2 model families, this produces **bit-identical text output** with zero logit difference.

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
    ├── quality_eval.py              # 140-prompt quality evaluation
    ├── pipeline_and_quality.py      # Pipeline simulation + long-context test
    ├── mooncake_integration.py      # Real Mooncake Transfer Engine integration
    └── main_track_push.py           # Long-context + serving-path proxy

paper/                               # LaTeX paper (10 pages, compiled PDF)
dfloat11/                            # Base DFloat11 codebase (dependency)
```

### Key Files

| File | What It Does | When to Use |
|------|-------------|-------------|
| `lossless_fast.py` | Truly lossless 4-bit codec with escape handling. The production implementation. | Main codec for benchmarking and integration. |
| `opt_rounds3.py` | 3-bit near-lossless codec with fully unrolled vectorized Triton kernels. | Higher compression (1.455x) when 1.25% approximation is acceptable. |
| `kv_codec.py` | Triton split/recombine kernels + PD transfer simulation at various bandwidths. | Pipeline speedup projections for different network tiers. |
| `quality_eval.py` | Runs 110+ prompts on a model, measures text match and logit difference. | Validate that compression does not affect model output. |
| `mooncake_integration.py` | Initializes Mooncake Transfer Engine, runs real compressed transfers. | End-to-end system integration validation. |
| `kv_cache_profile.py` | Profiles exponent entropy, zero fraction, per-layer/per-head statistics. | Verify the exponent concentration on a new model. |

---

## How to Reproduce

### Prerequisites

```bash
conda activate quant  # or your CUDA-enabled Python environment
pip install torch triton transformers dahuffman safetensors
# For Mooncake integration:
pip install mooncake-transfer-engine
conda install -c conda-forge etcd  # needed for Mooncake metadata
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
  268M: encode=252 GB/s, decode=1460 GB/s, ratio=1.333x, escapes=23K (0.016%), PASS ✓
  Pipeline: GPU-Direct→1.331x, CPU-RDMA→1.328x, RoCE4x200→1.324x
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

Expected: FP8 E4M3 gets additional 1.19x lossless on top of FP8. FP8 E5M2 + SplitZip = 2.77x total.

### 7. Long-Context + Pipeline Simulation

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/splitzip/pipeline_and_quality.py
```

Expected: lossless PASS at 128–2840 tokens, pipeline speedup 1.33x on all tiers.

### 8. Real Mooncake Transfer Engine Integration

Requires etcd running locally:

```bash
# Terminal 1: Start etcd
etcd --listen-client-urls http://localhost:2379 --advertise-client-urls http://localhost:2379

# Terminal 2: Run integration test
LD_LIBRARY_PATH=$(python -c "import nvidia.cuda_runtime; print(nvidia.cuda_runtime.__path__[0])")/lib:$LD_LIBRARY_PATH \
CUDA_VISIBLE_DEVICES=0 python experiments/splitzip/mooncake_integration.py
```

Expected: real Mooncake TCP transfer speedup 1.16–2.27x depending on KV size.

### 9. Cross-Model Robustness

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

### Real Mooncake Transfer Speedup (TCP)

| KV Size | Raw | Compressed | Speedup |
|---------|-----|-----------|---------|
| 512 KB | 1.95 ms | 0.86 ms | 2.27x |
| 2 MB | 1.62 ms | 1.39 ms | 1.16x |
| 4 MB | 2.77 ms | 2.26 ms | 1.22x |
| 16 MB | 10.89 ms | 7.45 ms | 1.46x |

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
