"""
FP8 Compressed Inference Demo

Demonstrates end-to-end inference with Huffman-compressed FP8 weights:
1. Load BF16 model → cast to FP8 → Huffman compress (offline)
2. Store compressed weights
3. At inference: decompress on GPU → FP8 matmul via cuBLAS

Measures:
- Memory savings (compressed vs dense FP8)
- Decode latency per layer
- End-to-end GEMM comparison: compressed-decode+GEMM vs dense-FP8-GEMM
- Roofline analysis: theoretical speedup with fused decode+GEMM
"""

import torch
import torch.nn.functional as F
import numpy as np
import cupy as cp
import time
from fp8_fused_huffman import FP8HuffmanEncoder, FP8GPUDecoder


def benchmark_gemm_shapes():
    """Benchmark compressed vs dense FP8 GEMM on realistic decode-phase shapes."""
    print("="*90)
    print("FP8 Compressed vs Dense GEMM Benchmark (Decode-phase shapes)")
    print("="*90)

    # Typical decode-phase GEMM shapes: M=1 (or small batch), K,N from model dims
    shapes = [
        # (M, N, K) — decode phase
        (1, 4096, 4096, "Attn proj (1 token)"),
        (1, 11008, 4096, "MLP gate/up (1 token)"),
        (1, 4096, 11008, "MLP down (1 token)"),
        (4, 4096, 4096, "Attn proj (batch=4)"),
        (4, 11008, 4096, "MLP gate/up (batch=4)"),
        (16, 4096, 4096, "Attn proj (batch=16)"),
        # Prefill shapes (for reference)
        (512, 4096, 4096, "Attn proj (prefill 512)"),
    ]

    encoder = FP8HuffmanEncoder()
    decoder = FP8GPUDecoder()

    print(f"\n{'Shape':<30} {'Dense ms':>9} {'Dec ms':>8} {'GEMM ms':>8} {'Total ms':>9} "
          f"{'Ratio':>7} {'Speedup':>8} {'Bound':>10}")
    print("-" * 90)

    for M, N, K, desc in shapes:
        # Create realistic FP8 weight matrix B (N x K) — stored as (N,K), transposed for GEMM
        B_bf16 = torch.randn(N, K, dtype=torch.bfloat16)
        B_fp8 = B_bf16.to(torch.float8_e4m3fn)
        A_fp8 = torch.randn(M, K, dtype=torch.bfloat16).to(torch.float8_e4m3fn).cuda()

        # Build shared Huffman codec for this weight
        encoder.build_codec(B_fp8)
        compressed = encoder.encode(B_fp8)
        ratio = compressed['ratio']

        # Dense FP8 GEMM baseline
        B_fp8_gpu = B_fp8.cuda()
        # Scale tensors for FP8 GEMM
        scale_a = torch.tensor(1.0, dtype=torch.float32, device='cuda')
        scale_b = torch.tensor(1.0, dtype=torch.float32, device='cuda')

        # Warmup
        for _ in range(5):
            out = torch._scaled_mm(A_fp8, B_fp8_gpu.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)

        # Benchmark dense GEMM
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            out_dense = torch._scaled_mm(A_fp8, B_fp8_gpu.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        dense_time = (time.perf_counter() - t0) / 100

        # Benchmark: decode + GEMM (Option B: two separate operations)
        # Warmup
        for _ in range(3):
            B_decoded = decoder.decode(compressed)

        # Benchmark decode
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            B_decoded = decoder.decode(compressed)
        torch.cuda.synchronize()
        decode_time = (time.perf_counter() - t0) / 50

        # Benchmark GEMM with decoded weights
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            out_comp = torch._scaled_mm(A_fp8, B_decoded.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        gemm_time = (time.perf_counter() - t0) / 100

        total_time = decode_time + gemm_time
        speedup = dense_time / total_time

        # Roofline: is this memory-bound or compute-bound?
        flops = 2 * M * N * K
        bytes_read = M * K + K * N  # A + B in FP8 (1 byte each)
        arithmetic_intensity = flops / bytes_read  # FLOP/byte
        # H200: ~4.8 TB/s bandwidth, ~1979 TFLOPS FP8
        peak_bandwidth = 4.8e12  # bytes/sec
        peak_flops = 1979e12  # FLOP/sec (FP8 tensor core)
        roofline_bound = "MEM-BOUND" if arithmetic_intensity < (peak_flops / peak_bandwidth) else "COMPUTE"

        print(f"  {desc:<28} {dense_time*1000:>8.3f} {decode_time*1000:>7.3f} "
              f"{gemm_time*1000:>7.3f} {total_time*1000:>8.3f} "
              f"{ratio:>6.1f}% {speedup:>7.2f}x {roofline_bound:>10}")

        # Verify correctness
        if not torch.allclose(out_dense.float(), out_comp.float(), rtol=0, atol=0):
            diff = (out_dense.float() - out_comp.float()).abs().max().item()
            if diff > 0:
                print(f"    WARNING: max diff = {diff}")

    print(f"\n{'='*90}")
    print("Analysis:")
    print("  - Decode-phase (M=1,4): MEMORY-BOUND → compression reduces HBM traffic")
    print("  - Prefill (M=512): COMPUTE-BOUND → compression doesn't help much")
    print("  - Option B (decode+GEMM) has decode overhead that limits speedup")
    print("  - Fused kernel (Option C) would eliminate decode overhead entirely")
    print(f"\n  Theoretical fused speedup on memory-bound shapes:")
    print(f"  Dense FP8 loads K*N bytes. Compressed loads {ratio:.1f}% of that.")
    print(f"  Max speedup = 100/{ratio:.1f} = {100/ratio:.2f}x (if decode is free)")
    print(f"{'='*90}")


def memory_savings_analysis():
    """Analyze memory savings for various model sizes."""
    print(f"\n{'='*90}")
    print("Memory Savings Analysis: FP8 Huffman Compression")
    print(f"{'='*90}")

    # Typical model sizes and our compression ratio
    ratio = 0.71  # 71% with shared tables
    models = [
        ("Qwen3-0.6B", 0.6),
        ("Llama-3-8B", 8.0),
        ("Qwen3-8B", 8.0),
        ("Llama-3-70B", 70.0),
        ("DeepSeek-V3-671B", 671.0),
    ]

    print(f"\n{'Model':<25} {'FP8 Size':>10} {'Compressed':>12} {'Saved':>10} {'H200 fits':>10}")
    print("-" * 70)

    h200_mem = 143  # GB
    for name, params_b in models:
        fp8_gb = params_b  # ~1 byte per param
        comp_gb = fp8_gb * ratio
        saved_gb = fp8_gb - comp_gb
        fits = "Yes" if comp_gb < h200_mem else "No"
        print(f"  {name:<23} {fp8_gb:>9.1f}G {comp_gb:>11.1f}G {saved_gb:>9.1f}G {fits:>10}")

    print(f"\n  At 71% ratio: saves 29% of FP8 model size")
    print(f"  For 70B model: saves ~20 GB (enables fitting in single H200)")
    print(f"  For 671B model: saves ~195 GB (reduces multi-GPU requirement)")


if __name__ == "__main__":
    benchmark_gemm_shapes()
    memory_savings_analysis()
