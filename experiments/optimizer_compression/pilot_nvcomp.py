"""
Pilot experiment: NVIDIA nvCOMP GPU compression for optimizer states.

Tests whether nvCOMP can provide fast GPU-native entropy coding for FP32
optimizer states, overcoming the bottleneck found in prior experiments
(custom GPU Huffman: 2-40 seconds for 10M+ values due to atomicOr contention).

Benchmarks: ANS, LZ4, Bitcomp, GDeflate, Deflate, Cascaded
on 100M FP32 values with realistic optimizer state distributions.
"""

import time
import torch
import numpy as np

import nvidia.nvcomp as nvc


def generate_optimizer_states(n=100_000_000, seed=42):
    """Generate realistic FP32 optimizer state data.

    AdamW stores:
      m (first moment):  similar to gradients, roughly normal, small magnitude
      v (second moment): squared gradients, positive, concentrated exponents
    """
    torch.manual_seed(seed)

    # m: first moment - roughly normal, small values
    m = torch.randn(n, dtype=torch.float32) * 0.001

    # v: second moment - positive, concentrated around small values
    # v = beta2 * v + (1-beta2) * grad^2, so it's EMA of squared gradients
    v = (torch.randn(n, dtype=torch.float32) * 0.001).abs() ** 2 + 1e-10

    return m, v


def benchmark_codec(algorithm, data_bytes, uncomp_chunk_size=65536, warmup=2, repeats=5, **kwargs):
    """Benchmark a single nvCOMP codec on GPU data.

    Returns dict with compression ratio, encode/decode throughput.
    """
    try:
        codec = nvc.Codec(
            algorithm=algorithm,
            uncomp_chunk_size=uncomp_chunk_size,
            **kwargs
        )
    except Exception as e:
        return {"algorithm": algorithm, "error": str(e)}

    input_arr = nvc.as_array(data_bytes)
    input_size = input_arr.buffer_size
    data_size_gb = input_size / (1024**3)

    # Warmup encode
    try:
        for _ in range(warmup):
            encoded = codec.encode(input_arr)
        torch.cuda.synchronize()
    except Exception as e:
        return {"algorithm": algorithm, "error": f"encode failed: {e}"}

    # Benchmark encode
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        encoded = codec.encode(input_arr)
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    encode_time = (t1 - t0) / repeats

    compressed_size = encoded.buffer_size
    ratio = compressed_size / input_size * 100
    encode_throughput = data_size_gb / encode_time

    # Warmup decode
    try:
        for _ in range(warmup):
            decoded = codec.decode(encoded)
        torch.cuda.synchronize()
    except Exception as e:
        return {
            "algorithm": algorithm,
            "ratio": ratio,
            "compressed_MB": compressed_size / (1024**2),
            "original_MB": input_size / (1024**2),
            "encode_throughput_GBps": encode_throughput,
            "error": f"decode failed: {e}",
        }

    # Benchmark decode
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        decoded = codec.decode(encoded)
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    decode_time = (t1 - t0) / repeats
    decode_throughput = data_size_gb / decode_time

    # Verify correctness (view as uint8 to avoid signed/unsigned mismatch)
    decoded_t = torch.from_dlpack(decoded).view(torch.uint8)
    original_t = torch.from_dlpack(input_arr).view(torch.uint8)
    correct = torch.equal(decoded_t, original_t)

    return {
        "algorithm": algorithm,
        "ratio": ratio,
        "compressed_MB": compressed_size / (1024**2),
        "original_MB": input_size / (1024**2),
        "encode_throughput_GBps": encode_throughput,
        "decode_throughput_GBps": decode_throughput,
        "encode_time_ms": encode_time * 1000,
        "decode_time_ms": decode_time * 1000,
        "correct": correct,
    }


def main():
    print("=" * 70)
    print("nvCOMP GPU Compression Benchmark for Optimizer States")
    print("=" * 70)
    print(f"nvCOMP version: {nvc.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Generate data
    print("Generating 100M FP32 optimizer state values...")
    m, v = generate_optimizer_states(n=100_000_000)

    # Test on both m and v states
    for name, tensor in [("m (first moment)", m), ("v (second moment)", v)]:
        print(f"\n{'='*70}")
        print(f"Testing on: {name}")
        print(f"Shape: {tensor.shape}, dtype: {tensor.dtype}")
        data_size_mb = tensor.numel() * tensor.element_size() / (1024**2)
        print(f"Size: {data_size_mb:.1f} MB")

        # Move to GPU and get raw bytes
        gpu_tensor = tensor.cuda().contiguous()
        # View as uint8 for nvCOMP
        gpu_bytes = gpu_tensor.view(torch.uint8)

        print(f"\n{'Algorithm':<15} {'Ratio%':>8} {'Enc GB/s':>10} {'Dec GB/s':>10} {'Enc ms':>10} {'Dec ms':>10} {'OK':>5}")
        print("-" * 70)

        # Test each algorithm
        algorithms = [
            ("lz4", {}),
            ("ans", {}),
            ("bitcomp", {}),
            ("bitcomp_s", {"algorithm": "bitcomp", "algorithm_type": 1}),  # sparse variant
            ("gdeflate", {}),
            ("gdeflate_e", {"algorithm": "gdeflate", "algorithm_type": 0}),  # entropy-only
            ("deflate", {}),
            ("deflate_e", {"algorithm": "deflate", "algorithm_type": 0}),  # entropy-only
            ("cascaded", {}),
        ]

        # Also try different chunk sizes for best performers
        chunk_sizes = [65536, 262144, 1048576]

        results = []
        for label, kwargs in algorithms:
            algo = kwargs.pop("algorithm", label)
            r = benchmark_codec(algo, gpu_bytes, **kwargs)
            r["label"] = label
            results.append(r)

            if "error" in r:
                print(f"{label:<15} {'ERROR':>8} {r['error']}")
            else:
                print(f"{label:<15} {r['ratio']:>7.2f}% {r['encode_throughput_GBps']:>9.2f} {r['decode_throughput_GBps']:>9.2f} {r['encode_time_ms']:>9.1f} {r['decode_time_ms']:>9.1f} {'Y' if r['correct'] else 'N':>5}")

        # Try best algorithms with different chunk sizes
        print(f"\n--- Chunk size sweep (best algorithms) ---")
        print(f"{'Algorithm':<15} {'Chunk':>8} {'Ratio%':>8} {'Enc GB/s':>10} {'Dec GB/s':>10}")
        print("-" * 55)

        for algo_name in ["lz4", "ans", "bitcomp"]:
            for cs in chunk_sizes:
                extra = {}
                r = benchmark_codec(algo_name, gpu_bytes, uncomp_chunk_size=cs, **extra)
                if "error" not in r:
                    print(f"{algo_name:<15} {cs:>8} {r['ratio']:>7.2f}% {r['encode_throughput_GBps']:>9.2f} {r['decode_throughput_GBps']:>9.2f}")
                else:
                    print(f"{algo_name:<15} {cs:>8} ERROR: {r['error']}")

        del gpu_tensor, gpu_bytes

    # Test byte3 (MSB) only - fair comparison with prior byte-plane approach
    print(f"\n{'='*70}")
    print("Byte3 (MSB) only compression - fair comparison with prior work")
    print("=" * 70)
    for name, tensor in [("m (first moment)", m), ("v (second moment)", v)]:
        gpu_tensor = tensor.cuda().contiguous()
        gpu_bytes = gpu_tensor.view(torch.uint8)
        # Extract byte3 (MSB) - every 4th byte starting at offset 3
        byte3 = gpu_bytes[3::4].contiguous()
        byte3_size_mb = byte3.numel() / (1024**2)
        full_size_mb = gpu_tensor.numel() * 4 / (1024**2)
        print(f"\n{name}: byte3 size = {byte3_size_mb:.1f} MB (of {full_size_mb:.1f} MB full)")

        for algo in ["ans", "gdeflate_e"]:
            algo_name = "gdeflate" if "gdeflate" in algo else algo
            extra = {"algorithm_type": 0} if algo == "gdeflate_e" else {}
            r = benchmark_codec(algo_name, byte3, **extra)
            if "error" not in r:
                byte3_compressed_mb = r["compressed_MB"]
                # Total size = compressed byte3 + raw bytes 0-2
                total_compressed = byte3_compressed_mb + byte3_size_mb * 3
                total_ratio = total_compressed / full_size_mb * 100
                byte3_savings_pct = (1 - r["ratio"]/100) * 25  # byte3 is 25% of total
                print(f"  {algo:<12} byte3 ratio: {r['ratio']:.2f}%  "
                      f"full-tensor savings: {byte3_savings_pct:.2f}%  "
                      f"enc: {r['encode_time_ms']:.1f}ms ({r['encode_throughput_GBps']:.1f} GB/s)  "
                      f"dec: {r['decode_time_ms']:.1f}ms ({r['decode_throughput_GBps']:.1f} GB/s)")
            else:
                print(f"  {algo:<12} ERROR: {r.get('error')}")
        del gpu_tensor, gpu_bytes, byte3

    # Comparison summary
    print(f"\n{'='*70}")
    print("COMPARISON WITH PRIOR RESULTS")
    print("=" * 70)
    print("Prior: Custom GPU Huffman on byte3 (MSB) of FP32:")
    print("  - 2-40 seconds for 10M values (atomicOr contention)")
    print("  - Theoretical max savings: 13.5% of FP32 states")
    print("  - Practical (6-bit fixed code): 6.25% savings")
    print()
    print("Prior: CPU ANS coding:")
    print("  - 23 seconds overhead for GPU->CPU transfer + encode/decode")
    print()
    print("Prior: Best practical method (hooked CPU offload):")
    print("  - Saves 65% GPU memory, 13% slowdown")
    print("  - But doesn't actually compress, just offloads to CPU")
    print()
    print("nvCOMP operates on raw bytes of the full FP32 tensor,")
    print("so the compression ratio reflects ALL 4 bytes including")
    print("the ~8-bit-entropy lower bytes (which are incompressible).")
    print("For fair comparison: 100% - ratio = savings on full tensor.")

    del m, v
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
