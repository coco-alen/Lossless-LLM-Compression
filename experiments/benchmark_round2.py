"""
Round 2 review fixes: benchmark native FP8, Llama BF16, and AWQ INT4.

Addresses reviewer concerns:
1. Native FP8 checkpoints (not just BF16→FP8 cast)
2. Additional model family (Llama)
3. Additional INT4 quantizer (AWQ)

Usage:
    conda activate quant
    python experiments/benchmark_round2.py
"""

import sys
import os
import time
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from new_compression.codec_multiformat import (
    compress, decompress, compressed_size_bytes, original_size_bytes,
    entropy_analysis,
)


def benchmark_native_fp8(model_name: str, results: dict):
    """Benchmark compression on a NATIVE FP8 checkpoint (not casted)."""
    from transformers import AutoModelForCausalLM, AutoConfig
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download, list_repo_files

    print(f"\n{'='*80}")
    print(f"NATIVE FP8 Benchmark: {model_name}")
    print(f"{'='*80}")

    # Download safetensors files to inspect native dtypes
    files = list_repo_files(model_name)
    st_files = [f for f in files if f.endswith('.safetensors')]
    print(f"  Found {len(st_files)} safetensor files")

    total_original = 0
    total_compressed = 0
    total_params = 0
    total_fp8_params = 0
    total_bf16_params = 0
    dtype_counts = {}
    layer_results = []

    for st_file in st_files:
        local_path = hf_hub_download(model_name, st_file)
        with safe_open(local_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                dtype_str = str(tensor.dtype)
                dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + tensor.numel()

                # Only compress weight tensors
                if tensor.numel() < 1024:
                    continue

                if tensor.dtype == torch.float8_e4m3fn:
                    # Native FP8 — this is what we want!
                    comp = compress(tensor)
                    recovered = decompress(comp)
                    assert torch.equal(tensor, recovered), f"Lossless check failed for {key}"

                    orig = original_size_bytes(comp)
                    comp_size = compressed_size_bytes(comp)
                    ratio = comp_size / orig * 100
                    ent = entropy_analysis(tensor)

                    total_original += orig
                    total_compressed += comp_size
                    total_params += tensor.numel()
                    total_fp8_params += tensor.numel()

                    layer_results.append({
                        'name': key,
                        'dtype': 'fp8_e4m3fn',
                        'params': tensor.numel(),
                        'ratio': ratio,
                        'entropy_bpv': ent['entropy_bpv'],
                        'unique': ent['unique_values'],
                    })

                    if tensor.numel() > 1_000_000:
                        print(f"  [FP8] {key:50s}  {tensor.numel():>12,}  {ratio:6.2f}%  "
                              f"H={ent['entropy_bpv']:.3f}  unique={ent['unique_values']}")

                elif tensor.dtype == torch.bfloat16:
                    # BF16 weights in the FP8 checkpoint (e.g., norms, embeddings)
                    comp = compress(tensor)
                    recovered = decompress(comp)
                    assert torch.equal(tensor, recovered), f"Lossless check failed for {key}"

                    orig = original_size_bytes(comp)
                    comp_size = compressed_size_bytes(comp)
                    ratio = comp_size / orig * 100
                    ent = entropy_analysis(tensor)

                    total_original += orig
                    total_compressed += comp_size
                    total_params += tensor.numel()
                    total_bf16_params += tensor.numel()

                    layer_results.append({
                        'name': key,
                        'dtype': 'bf16',
                        'params': tensor.numel(),
                        'ratio': ratio,
                        'entropy_bpv': ent['entropy_bpv'],
                        'unique': ent['unique_values'],
                    })

                    if tensor.numel() > 1_000_000:
                        print(f"  [BF16] {key:50s}  {tensor.numel():>12,}  {ratio:6.2f}%  "
                              f"H={ent['entropy_bpv']:.3f}")

                elif tensor.dtype == torch.float16:
                    # FP16 weights
                    comp = compress(tensor)
                    recovered = decompress(comp)
                    assert torch.equal(tensor, recovered), f"Lossless check failed for {key}"

                    orig = original_size_bytes(comp)
                    comp_size = compressed_size_bytes(comp)

                    total_original += orig
                    total_compressed += comp_size
                    total_params += tensor.numel()
                    total_bf16_params += tensor.numel()

                    ent = entropy_analysis(tensor)
                    layer_results.append({
                        'name': key,
                        'dtype': 'fp16',
                        'params': tensor.numel(),
                        'ratio': comp_size / orig * 100,
                        'entropy_bpv': ent['entropy_bpv'],
                        'unique': ent['unique_values'],
                    })

    print(f"\n  Dtype distribution:")
    for dtype, count in sorted(dtype_counts.items()):
        print(f"    {dtype}: {count:,} params ({count*8/1e9:.2f} GB)")

    if total_original > 0:
        overall_ratio = total_compressed / total_original * 100

        # Also compute FP8-only ratio
        fp8_orig = sum(r['params'] for r in layer_results if r['dtype'] == 'fp8_e4m3fn')
        fp8_comp = sum(r['params'] * r['ratio'] / 100 for r in layer_results if r['dtype'] == 'fp8_e4m3fn')
        fp8_ratio = fp8_comp / fp8_orig * 100 if fp8_orig > 0 else 0

        print(f"\n  TOTAL: {total_params:,} params | {total_original/1e6:.1f} MB → "
              f"{total_compressed/1e6:.1f} MB | {overall_ratio:.2f}%")
        print(f"  FP8 only: {total_fp8_params:,} params | FP8 ratio: {fp8_ratio:.2f}%")
        print(f"  BF16/FP16: {total_bf16_params:,} params")

        results[f'{model_name}_native_fp8'] = {
            'format': 'native_fp8',
            'total_params': total_params,
            'fp8_params': total_fp8_params,
            'bf16_params': total_bf16_params,
            'original_mb': total_original / 1e6,
            'compressed_mb': total_compressed / 1e6,
            'overall_ratio': overall_ratio,
            'fp8_only_ratio': fp8_ratio,
            'dtype_counts': dtype_counts,
            'layers': layer_results,
        }
        return overall_ratio
    return None


def benchmark_bf16_model(model_name: str, results: dict):
    """Benchmark BF16 compression."""
    from transformers import AutoModelForCausalLM

    print(f"\n{'='*80}")
    print(f"BF16 Benchmark: {model_name}")
    print(f"{'='*80}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    total_original = 0
    total_compressed = 0
    total_params = 0

    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16 or param.numel() < 1024:
            continue

        comp = compress(param.data)
        recovered = decompress(comp)
        assert torch.equal(param.data, recovered), f"Lossless check failed for {name}"

        orig = original_size_bytes(comp)
        comp_size = compressed_size_bytes(comp)
        ratio = comp_size / orig * 100
        ent = entropy_analysis(param.data)

        total_original += orig
        total_compressed += comp_size
        total_params += param.numel()

        if param.numel() > 1_000_000:
            print(f"  {name:50s}  {param.numel():>12,}  {ratio:6.2f}%  H={ent['entropy_bpv']:.3f}")

    overall_ratio = total_compressed / total_original * 100
    print(f"\n  TOTAL: {total_params:,} params | {total_original/1e6:.1f} MB → "
          f"{total_compressed/1e6:.1f} MB | {overall_ratio:.2f}%")

    results[f'{model_name}_bf16'] = {
        'format': 'bf16',
        'total_params': total_params,
        'original_mb': total_original / 1e6,
        'compressed_mb': total_compressed / 1e6,
        'ratio': overall_ratio,
    }

    del model
    return overall_ratio


def benchmark_fp8_cast(model_name: str, results: dict):
    """Benchmark FP8 compression via BF16→FP8 cast for comparison."""
    from transformers import AutoModelForCausalLM

    print(f"\n{'='*80}")
    print(f"FP8 CAST Benchmark: {model_name} (BF16→FP8)")
    print(f"{'='*80}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    total_original = 0
    total_compressed = 0
    total_params = 0

    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16 or param.numel() < 1024:
            continue

        fp8_data = param.data.to(torch.float8_e4m3fn)
        comp = compress(fp8_data)
        recovered = decompress(comp)
        assert torch.equal(fp8_data, recovered), f"Lossless check failed for {name}"

        orig = original_size_bytes(comp)
        comp_size = compressed_size_bytes(comp)

        total_original += orig
        total_compressed += comp_size
        total_params += param.numel()

    overall_ratio = total_compressed / total_original * 100
    print(f"\n  TOTAL: {total_params:,} params | {total_original/1e6:.1f} MB → "
          f"{total_compressed/1e6:.1f} MB | {overall_ratio:.2f}%")

    results[f'{model_name}_fp8_cast'] = {
        'format': 'fp8_e4m3fn_cast',
        'total_params': total_params,
        'original_mb': total_original / 1e6,
        'compressed_mb': total_compressed / 1e6,
        'ratio': overall_ratio,
    }

    del model
    return overall_ratio


def benchmark_awq_int4(model_name: str, results: dict):
    """Benchmark AWQ INT4 checkpoint."""
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download, list_repo_files

    print(f"\n{'='*80}")
    print(f"AWQ INT4 Benchmark: {model_name}")
    print(f"{'='*80}")

    files = list_repo_files(model_name)
    st_files = [f for f in files if f.endswith('.safetensors')]
    print(f"  Found {len(st_files)} safetensor files")

    total_by_component = {}  # component -> {original, compressed, params}
    dtype_counts = {}
    layer_results = []

    for st_file in st_files:
        local_path = hf_hub_download(model_name, st_file)
        with safe_open(local_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                dtype_str = str(tensor.dtype)
                dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + tensor.numel()

                if tensor.numel() < 64:
                    continue

                # Determine component type
                if 'qweight' in key:
                    component = 'qweight'
                elif 'qzeros' in key:
                    component = 'qzeros'
                elif 'scales' in key:
                    component = 'scales'
                elif 'g_idx' in key:
                    component = 'g_idx'
                else:
                    component = 'other'

                # Compress based on dtype
                if tensor.dtype in (torch.int32,):
                    # AWQ packs INT4 into int32 — compress as raw bytes
                    raw = tensor.view(torch.uint8)
                    comp = compress(raw, format_hint='uint8')
                    recovered = decompress(comp)
                    assert torch.equal(raw, recovered), f"Lossless check failed for {key}"
                elif tensor.dtype == torch.float16:
                    comp = compress(tensor)
                    recovered = decompress(comp)
                    assert torch.equal(tensor, recovered), f"Lossless check failed for {key}"
                elif tensor.dtype == torch.bfloat16:
                    comp = compress(tensor)
                    recovered = decompress(comp)
                    assert torch.equal(tensor, recovered), f"Lossless check failed for {key}"
                else:
                    # Try as uint8
                    raw = tensor.view(torch.uint8)
                    comp = compress(raw, format_hint='uint8')
                    recovered = decompress(comp)
                    assert torch.equal(raw, recovered), f"Lossless check failed for {key}"

                orig = original_size_bytes(comp)
                comp_size = compressed_size_bytes(comp)
                ratio = comp_size / orig * 100

                if component not in total_by_component:
                    total_by_component[component] = {'original': 0, 'compressed': 0, 'params': 0}
                total_by_component[component]['original'] += orig
                total_by_component[component]['compressed'] += comp_size
                total_by_component[component]['params'] += tensor.numel()

                layer_results.append({
                    'name': key,
                    'component': component,
                    'dtype': dtype_str,
                    'params': tensor.numel(),
                    'ratio': ratio,
                })

                if tensor.numel() > 100_000:
                    print(f"  [{component:8s}] {key:50s}  {tensor.numel():>12,}  {ratio:6.2f}%  {dtype_str}")

    print(f"\n  Dtype distribution:")
    for dtype, count in sorted(dtype_counts.items()):
        print(f"    {dtype}: {count:,}")

    print(f"\n  Per-component breakdown:")
    total_orig = 0
    total_comp = 0
    for comp_name, stats in sorted(total_by_component.items()):
        r = stats['compressed'] / stats['original'] * 100 if stats['original'] > 0 else 0
        print(f"    {comp_name:10s}: {stats['original']/1e6:8.1f} MB → {stats['compressed']/1e6:8.1f} MB  ({r:.1f}%)")
        total_orig += stats['original']
        total_comp += stats['compressed']

    if total_orig > 0:
        overall_ratio = total_comp / total_orig * 100
        print(f"\n  TOTAL: {total_orig/1e6:.1f} MB → {total_comp/1e6:.1f} MB | {overall_ratio:.2f}%")

        # qweight-only ratio
        if 'qweight' in total_by_component:
            qw = total_by_component['qweight']
            qw_ratio = qw['compressed'] / qw['original'] * 100
            print(f"  qweight only: {qw['original']/1e6:.1f} MB → {qw['compressed']/1e6:.1f} MB | {qw_ratio:.2f}%")

        results[f'{model_name}_awq_int4'] = {
            'format': 'awq_int4',
            'original_mb': total_orig / 1e6,
            'compressed_mb': total_comp / 1e6,
            'overall_ratio': overall_ratio,
            'per_component': {k: {
                'original_mb': v['original']/1e6,
                'compressed_mb': v['compressed']/1e6,
                'ratio': v['compressed']/v['original']*100 if v['original'] > 0 else 0,
            } for k, v in total_by_component.items()},
            'dtype_counts': {str(k): v for k, v in dtype_counts.items()},
        }
        return overall_ratio
    return None


def main():
    results = {}

    # === Fix #1: Native FP8 checkpoints ===
    native_fp8_models = [
        "Qwen/Qwen3-0.6B-FP8",
        "RedHatAI/Llama-3.2-1B-Instruct-FP8",  # Different family
    ]

    for model_name in native_fp8_models:
        try:
            benchmark_native_fp8(model_name, results)
        except Exception as e:
            import traceback
            print(f"  ERROR with {model_name}: {e}")
            traceback.print_exc()

    # === Fix #3: Add Llama BF16 + FP8 cast for comparison ===
    llama_models = [
        "meta-llama/Llama-3.2-1B",  # Same base as the FP8 checkpoint
    ]

    for model_name in llama_models:
        try:
            benchmark_bf16_model(model_name, results)
            benchmark_fp8_cast(model_name, results)
        except Exception as e:
            import traceback
            print(f"  ERROR with {model_name}: {e}")
            traceback.print_exc()

    # === Fix #5: AWQ INT4 checkpoint ===
    awq_models = [
        "Qwen/Qwen2.5-7B-Instruct-AWQ",
    ]

    for model_name in awq_models:
        try:
            benchmark_awq_int4(model_name, results)
        except Exception as e:
            import traceback
            print(f"  ERROR with {model_name}: {e}")
            traceback.print_exc()

    # === Summary ===
    print(f"\n\n{'='*100}")
    print(f"{'ROUND 2 REVIEW FIX BENCHMARK RESULTS':^100}")
    print(f"{'='*100}")

    for key, r in sorted(results.items()):
        print(f"\n  {key}:")
        for k2, v2 in r.items():
            if k2 not in ('layers', 'dtype_counts', 'per_component'):
                print(f"    {k2}: {v2}")

    # Save results
    output_path = Path(__file__).parent / "benchmark_round2_results.json"
    with open(output_path, 'w') as f:
        # Convert non-serializable types
        def clean(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        json.dump(results, f, indent=2, default=clean)
    print(f"\n  Results saved to {output_path}")


if __name__ == '__main__':
    main()
