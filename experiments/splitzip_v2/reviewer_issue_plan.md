# Reviewer Issue Coverage Plan

| Reviewer issue | v2 code path | Output |
|---|---|---|
| 32-bit absolute escape positions may overflow or waste metadata | `codec_cpu.py`, `codec_gpu.py` | Chunk-local `uint16` offsets plus per-chunk counts |
| Escape collection pass may cause contention | `codec_gpu.py`, `gpu_breakdown_bench.py` | Count, prefix-sum, scatter pipeline with no global atomic append |
| End-to-end transfer was simulated | `mooncake_kv_sweep.py` | Real Mooncake transfer of actual KV-derived BF16 payloads |
| Missing Cascaded/Bitcomp/ZipServ/TCA-TBE/Falcon baselines | `additional_baseline_bench.py`, `baseline_matrix.py` | nvCOMP adapter, ZipServ/TCA-TBE adapter, Falcon command hook |
| Heterogeneous architecture coverage | `exponent_stability.py` | Qwen3-Next can be profiled for Table 1 |
| No SGLang integration evidence | `sglang_sweep.py` | Qwen3-32B PD-disaggregation launch and benchmark plan |
| FlowKV/HybridServe/KVPR complementarity | `orthogonality_analysis.py` | Explicit additive-speedup model; can be replaced with measured fractions |
| Codebook stability across layer and K/V | `exponent_stability.py` | Per-layer K and V histograms, Top-8/Top-16 coverage, entropy |
| FP8 chunk-local Top-8 | `fp8_chunk_topk.py` | Global vs chunk-local Top-8 coverage and ratio estimates |
| Pathological escape behavior | `gpu_breakdown_bench.py`, `breakdown_bench.py` | Stage-level escape-count and ratio reporting; adversarial tensors can be passed by extending input generation |

The scripts are intentionally separated by concern so final GPU runs can be launched independently and merged into the paper tables afterward.
