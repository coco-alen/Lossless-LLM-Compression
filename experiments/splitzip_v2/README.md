# SplitZip v2 Experiment Scaffold

This folder contains the code needed to rerun the v2 reviewer experiments once GPUs and Mooncake endpoints are available.  CPU-only tests validate the new chunk-local format now; GPU and serving scripts are written as launchable entry points.

## What Changed

- `codec_cpu.py` implements a lossless reference codec with chunk-local `uint16` escape positions.
- `codec_gpu.py` implements the Triton codec path with two-pass escape compaction: per-chunk count, prefix sum, and per-chunk scatter into disjoint output ranges.
- `gpu_breakdown_bench.py` times encode/decode stages separately: dense nibble encode, escape count, prefix sum, escape scatter, dense decode, and escape fix-up.
- `mooncake_kv_sweep.py` transfers actual KV payloads through Mooncake for Llama-3-8B and Qwen3-30B-A3B over the requested BS/sequence grid.
- `sglang_sweep.py` generates the native and SplitZip SGLang PD-disaggregation launch plan for Qwen3-32B.
- `sglang_integration_notes.md` identifies the sender/receiver hook boundary needed for a correct compressed SGLang path.
- `additional_baseline_bench.py` adds adapters for nvCOMP Cascaded/Bitcomp, ZipServ/TCA-TBE, and a Falcon command hook.
- `exponent_stability.py` records per-layer K/V exponent histograms for Qwen3-32B and Qwen3-Next.
- `fp8_chunk_topk.py` compares FP8 global Top-8 against chunk-local Top-8 coverage and size estimates.
- `orthogonality_analysis.py` composes SplitZip byte reduction with FlowKV/HybridServe/KVPR-style transfer reduction to show additive behavior under an explicit model.

## CPU Sanity Tests

```bash
conda run -n quant python -m pytest tests/splitzip_v2
```

This repository environment may not have `pytest`; the same tests also run with:

```bash
conda run -n quant python -m unittest discover -s tests -p 'test*.py'
```

## One-Shot GPU Queue

```bash
./experiments/splitzip_v2/run_v2_gpu_queue.sh
```

## GPU Codec Breakdown

```bash
conda run -n quant python -m experiments.splitzip_v2.gpu_breakdown_bench \
  --numel $((65536 * 4096)) \
  --chunk-size 1024 \
  --device cuda:0 \
  --output experiments/splitzip_v2/results/chunklocal_gpu_breakdown.json
```

The chunk size is configurable.  Any value up to 65,536 fits the `uint16` local offset format; smaller chunks usually improve per-program occupancy while adding a small count-table overhead.

## Mooncake KV Transfer

```bash
conda run -n quant python -m experiments.splitzip_v2.mooncake_kv_sweep \
  --models Llama-3-8B Qwen3-30B-A3B \
  --device cuda:0 \
  --protocol tcp \
  --mooncake-device cpu \
  --output experiments/splitzip_v2/results/mooncake_kv_sweep.json
```

Use `--native-only` to measure raw Mooncake transfer first.  The compressed transfer path serializes the chunk-local reference payload and verifies exact round trip; GPU codec stage timings should be taken from `gpu_breakdown_bench.py`.

## SGLang Plan

```bash
conda run -n quant python -m experiments.splitzip_v2.sglang_sweep \
  --model Qwen/Qwen3-32B \
  --sglang-root /data02/home/yilian2/project/sglang-dev \
  --transfer-backend mooncake \
  --output experiments/splitzip_v2/results/sglang_sweep_plan.json
```

This writes exact server and benchmark commands for the requested TTFT/throughput grid.  The SplitZip mode is gated by `SPLITZIP_SGLANG_ENABLE=1`, which is the hook to use when wiring the codec into SGLang's Mooncake KV transfer path.

After runs finish:

```bash
conda run -n quant python -m experiments.splitzip_v2.collect_sglang_results \
  --root experiments/splitzip_v2/results/sglang \
  --output experiments/splitzip_v2/results/sglang_summary.json
```

## Reviewer Paragraph

SplitZip v2 stores escape metadata in fixed-size chunks.  The encoder first runs a per-chunk counting kernel that computes the number of uncommon exponents in each chunk.  A prefix sum over these counts gives a disjoint output range for every chunk.  A second per-chunk scatter kernel then writes `uint16` local offsets and raw exponent values into the assigned range.  Because each chunk owns a non-overlapping segment, the implementation avoids global atomic append operations and the associated contention; decode mirrors this layout by launching independent chunk-local fix-up programs.
