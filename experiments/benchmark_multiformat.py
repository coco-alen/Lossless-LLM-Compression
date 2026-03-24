"""
Benchmark: Multi-format lossless compression across LLM models.

Tests full-value ANS coding on BF16, FP8, and INT4 formats.
Compares with DFloat11 baseline and reports per-layer and aggregate statistics.

Usage:
    conda activate quant
    python experiments/benchmark_multiformat.py
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
    compress, decompress, compress_group, decompress_group,
    compressed_size_bytes, original_size_bytes, compression_ratio,
    entropy_analysis,
)


def benchmark_bf16_model(model_name: str, results: dict):
    """Benchmark BF16 compression on a HuggingFace model."""
    from transformers import AutoModelForCausalLM, AutoConfig

    print(f"\n{'='*80}")
    print(f"BF16 Benchmark: {model_name}")
    print(f"{'='*80}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    total_original = 0
    total_compressed = 0
    total_params = 0
    layer_results = []

    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16:
            continue

        t0 = time.time()
        comp = compress(param.data)
        encode_time = time.time() - t0

        # Verify lossless
        t1 = time.time()
        recovered = decompress(comp)
        decode_time = time.time() - t1

        assert torch.equal(param.data, recovered), f"Lossless check failed for {name}"

        orig = original_size_bytes(comp)
        comp_size = compressed_size_bytes(comp)
        ratio = comp_size / orig * 100

        total_original += orig
        total_compressed += comp_size
        total_params += param.numel()

        # Entropy analysis
        ent = entropy_analysis(param.data)

        layer_results.append({
            'name': name,
            'params': param.numel(),
            'ratio': ratio,
            'entropy_bpv': ent['entropy_bpv'],
            'unique': ent['unique_values'],
            'encode_ms': encode_time * 1000,
            'decode_ms': decode_time * 1000,
        })

        if param.numel() > 1_000_000:
            print(f"  {name:50s}  {param.numel():>12,}  {ratio:6.2f}%  H={ent['entropy_bpv']:.3f}  "
                  f"enc={encode_time*1000:.0f}ms  dec={decode_time*1000:.0f}ms")

    overall_ratio = total_compressed / total_original * 100
    print(f"\n  TOTAL: {total_params:,} params | {total_original/1e6:.1f} MB → "
          f"{total_compressed/1e6:.1f} MB | {overall_ratio:.2f}%")

    results[f'{model_name}_bf16'] = {
        'format': 'bf16',
        'total_params': total_params,
        'original_mb': total_original / 1e6,
        'compressed_mb': total_compressed / 1e6,
        'ratio': overall_ratio,
        'layers': layer_results,
    }

    del model
    torch.cuda.empty_cache()
    return overall_ratio


def benchmark_fp8_model(model_name: str, results: dict):
    """Benchmark FP8 compression (cast BF16 → FP8, then compress)."""
    from transformers import AutoModelForCausalLM

    print(f"\n{'='*80}")
    print(f"FP8 (e4m3fn) Benchmark: {model_name}")
    print(f"{'='*80}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    total_original = 0
    total_compressed = 0
    total_params = 0
    layer_results = []

    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16:
            continue

        # Cast to FP8
        fp8_data = param.data.to(torch.float8_e4m3fn)

        comp = compress(fp8_data)
        recovered = decompress(comp)

        assert torch.equal(fp8_data, recovered), f"Lossless check failed for {name}"

        orig = original_size_bytes(comp)
        comp_size = compressed_size_bytes(comp)
        ratio = comp_size / orig * 100

        total_original += orig
        total_compressed += comp_size
        total_params += param.numel()

        ent = entropy_analysis(fp8_data)

        layer_results.append({
            'name': name,
            'params': param.numel(),
            'ratio': ratio,
            'entropy_bpv': ent['entropy_bpv'],
            'unique': ent['unique_values'],
        })

        if param.numel() > 1_000_000:
            print(f"  {name:50s}  {param.numel():>12,}  {ratio:6.2f}%  H={ent['entropy_bpv']:.3f}")

    overall_ratio = total_compressed / total_original * 100
    print(f"\n  TOTAL: {total_params:,} params | {total_original/1e6:.1f} MB → "
          f"{total_compressed/1e6:.1f} MB | {overall_ratio:.2f}%")

    results[f'{model_name}_fp8'] = {
        'format': 'fp8_e4m3fn',
        'total_params': total_params,
        'original_mb': total_original / 1e6,
        'compressed_mb': total_compressed / 1e6,
        'ratio': overall_ratio,
        'layers': layer_results,
    }

    del model
    torch.cuda.empty_cache()
    return overall_ratio


def benchmark_int4_model(model_name: str, results: dict):
    """Benchmark INT4 compression on a GPTQ model."""
    from transformers import AutoModelForCausalLM

    print(f"\n{'='*80}")
    print(f"INT4 (GPTQ) Benchmark: {model_name}")
    print(f"{'='*80}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="cpu", torch_dtype=torch.float16
        )
    except Exception as e:
        print(f"  Failed to load {model_name}: {e}")
        return None

    total_original = 0
    total_compressed = 0
    total_params = 0
    layer_results = []

    for name, param in model.named_parameters():
        # GPTQ stores quantized weights as uint8 or int32 packed
        if 'qweight' not in name and 'weight' not in name:
            continue
        if param.dtype == torch.uint8:
            # Packed INT4
            comp = compress(param.data, format_hint='uint8')
            recovered = decompress(comp)
            assert torch.equal(param.data, recovered), f"Lossless check failed for {name}"

            orig = original_size_bytes(comp)
            comp_size = compressed_size_bytes(comp)
            ratio = comp_size / orig * 100

            total_original += orig
            total_compressed += comp_size
            total_params += param.numel() * 2  # 2 values per byte

            ent = entropy_analysis(param.data, format_hint='uint8')
            layer_results.append({
                'name': name,
                'params': param.numel() * 2,
                'ratio': ratio,
                'entropy_bpv': ent['entropy_bpv'],
                'unique': ent['unique_values'],
            })

            if param.numel() > 100_000:
                print(f"  {name:50s}  {param.numel()*2:>12,}  {ratio:6.2f}%  H={ent['entropy_bpv']:.3f}")

    if total_original > 0:
        overall_ratio = total_compressed / total_original * 100
        print(f"\n  TOTAL: {total_params:,} params | {total_original/1e6:.1f} MB → "
              f"{total_compressed/1e6:.1f} MB | {overall_ratio:.2f}%")

        results[f'{model_name}_int4'] = {
            'format': 'int4_packed',
            'total_params': total_params,
            'original_mb': total_original / 1e6,
            'compressed_mb': total_compressed / 1e6,
            'ratio': overall_ratio,
            'layers': layer_results,
        }
    else:
        print("  No INT4 packed weights found — trying BF16 simulated INT4...")
        # Fallback: simulate INT4 quantization
        return benchmark_simulated_int4(model_name, results, model)

    del model
    torch.cuda.empty_cache()
    return overall_ratio


def benchmark_simulated_int4(model_name: str, results: dict, model=None):
    """Simulate INT4 quantization and measure entropy coding savings."""
    from transformers import AutoModelForCausalLM

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cpu"
        )

    total_original_int4 = 0
    total_compressed = 0
    total_params = 0

    for name, param in model.named_parameters():
        if param.numel() < 1024:
            continue

        w = param.data.float()

        # Simple symmetric per-channel INT4 quantization
        if w.dim() >= 2:
            scale = w.abs().amax(dim=-1, keepdim=True) / 7.0
            scale = scale.clamp(min=1e-10)
            w_int = torch.clamp(torch.round(w / scale), -8, 7).to(torch.int8)
        else:
            scale = w.abs().max() / 7.0
            scale = max(scale.item(), 1e-10)
            w_int = torch.clamp(torch.round(w / scale), -8, 7).to(torch.int8)

        # Shift to unsigned [0, 15] and pack into uint8
        w_uint = (w_int + 8).to(torch.uint8).flatten()

        # Pack pairs into single bytes
        if len(w_uint) % 2 != 0:
            w_uint = torch.cat([w_uint, torch.zeros(1, dtype=torch.uint8)])
        packed = w_uint[0::2] | (w_uint[1::2] << 4)

        comp = compress(packed, format_hint='uint8')
        orig = original_size_bytes(comp)
        comp_sz = compressed_size_bytes(comp)

        total_original_int4 += orig
        total_compressed += comp_sz
        total_params += param.numel()

    if total_original_int4 > 0:
        ratio = total_compressed / total_original_int4 * 100
        print(f"\n  Simulated INT4 TOTAL: {total_params:,} params | "
              f"{total_original_int4/1e6:.1f} MB → {total_compressed/1e6:.1f} MB | {ratio:.2f}%")

        results[f'{model_name}_sim_int4'] = {
            'format': 'simulated_int4',
            'total_params': total_params,
            'original_mb': total_original_int4 / 1e6,
            'compressed_mb': total_compressed / 1e6,
            'ratio': ratio,
        }
        return ratio

    return None


def print_summary(results: dict):
    """Print a clean comparison table."""
    print(f"\n\n{'='*100}")
    print(f"{'MULTI-FORMAT LOSSLESS COMPRESSION BENCHMARK':^100}")
    print(f"{'='*100}")
    print(f"\n{'Model':<40} {'Format':<12} {'Params':>12} {'Original':>10} {'Compressed':>12} {'Ratio':>8} {'Savings':>8}")
    print(f"{'-'*40} {'-'*12} {'-'*12} {'-'*10} {'-'*12} {'-'*8} {'-'*8}")

    for key, r in sorted(results.items()):
        savings = 100 - r['ratio']
        print(f"{key:<40} {r['format']:<12} {r['total_params']:>12,} "
              f"{r['original_mb']:>9.1f}M {r['compressed_mb']:>11.1f}M "
              f"{r['ratio']:>7.2f}% {savings:>7.1f}%")

    print(f"\n{'='*100}")
    print("Comparison with prior work:")
    print("  DFloat11 (NeurIPS '25):  BF16 → ~66.6%  (exponent-only Huffman)")
    print("  ECF8 (Oct 2025):         FP8  → ~85-90% (exponent-only Huffman)")
    print("  EntroLLM (May 2025):     INT4 → ~84%    (Huffman)")
    print("  ZipServ (ASPLOS '26):    BF16 → ~70%    (fixed 3-bit TCA-TBE, but with fused GEMM speedup)")
    print(f"{'='*100}")


def main():
    # Models to benchmark
    bf16_models = [
        "Qwen/Qwen3-0.6B",
    ]

    # Check if larger models are requested via env var
    if os.environ.get("FULL_BENCHMARK"):
        bf16_models.extend([
            "Qwen/Qwen3-8B",
            "meta-llama/Llama-3.1-8B",
        ])

    results = {}

    # BF16 benchmarks
    for model_name in bf16_models:
        try:
            benchmark_bf16_model(model_name, results)
        except Exception as e:
            print(f"  ERROR: {e}")

    # FP8 benchmarks (cast from BF16)
    for model_name in bf16_models:
        try:
            benchmark_fp8_model(model_name, results)
        except Exception as e:
            print(f"  ERROR: {e}")

    # INT4 benchmarks
    int4_models = []
    if os.environ.get("FULL_BENCHMARK"):
        int4_models.append("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")

    # Always run simulated INT4 on smallest model
    for model_name in bf16_models[:1]:
        try:
            benchmark_simulated_int4(model_name, results)
        except Exception as e:
            print(f"  ERROR: {e}")

    for model_name in int4_models:
        try:
            benchmark_int4_model(model_name, results)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    print_summary(results)

    # Save results
    output_path = Path(__file__).parent / "benchmark_results.json"
    serializable = {}
    for k, v in results.items():
        sv = {kk: vv for kk, vv in v.items() if kk != 'layers'}
        sv['n_layers'] = len(v.get('layers', []))
        serializable[k] = sv

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
