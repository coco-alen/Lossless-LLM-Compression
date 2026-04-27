#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY="conda run -n quant python"

$PY -m experiments.splitzip_v2.gpu_breakdown_bench \
  --numel $((65536 * 4096)) \
  --chunk-size 1024 \
  --device cuda:0 \
  --output experiments/splitzip_v2/results/chunklocal_gpu_breakdown.json

$PY -m experiments.splitzip_v2.additional_baseline_bench \
  --model NousResearch/Meta-Llama-3-8B \
  --seq-lens 1024 4096 16384 65536 \
  --hidden-dim 4096 \
  --device cuda:0 \
  --repeats 10 \
  --output experiments/splitzip_v2/results/additional_baselines_llama3_8b.json

$PY -m experiments.splitzip_v2.fp8_chunk_topk \
  --fmt e5m2 \
  --numel $((65536 * 4096)) \
  --chunk-size 256 \
  --output experiments/splitzip_v2/results/fp8_e5m2_chunk_top8.json

$PY -m experiments.splitzip_v2.fp8_chunk_topk \
  --fmt e4m3 \
  --numel $((65536 * 4096)) \
  --chunk-size 256 \
  --output experiments/splitzip_v2/results/fp8_e4m3_chunk_top8.json

$PY -m experiments.splitzip_v2.exponent_stability \
  --models Qwen/Qwen3-32B Qwen/Qwen3-Next-80B-A3B-Instruct \
  --device cuda:0 \
  --prompt-tokens 1024 \
  --output experiments/splitzip_v2/results/exponent_stability_qwen.json

$PY -m experiments.splitzip_v2.mooncake_kv_sweep \
  --models Llama-3-8B Qwen3-30B-A3B \
  --device cuda:0 \
  --protocol tcp \
  --mooncake-device cpu \
  --output experiments/splitzip_v2/results/mooncake_kv_sweep.json

$PY -m experiments.splitzip_v2.sglang_sweep \
  --model Qwen/Qwen3-32B \
  --sglang-root /data02/home/yilian2/project/sglang-dev \
  --transfer-backend mooncake \
  --output experiments/splitzip_v2/results/sglang_sweep_plan.json
