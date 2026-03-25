"""
nvCOMP ANS benchmark using the C API via ctypes.
Bypasses the broken Python high-level API.
"""

import ctypes
import numpy as np
import cupy as cp
import torch
import struct

LIB_PATH = "/home/sky/miniconda3/envs/quant/lib/python3.12/site-packages/nvidia/libnvcomp/lib64/libnvcomp.so.5"
nvcomp = ctypes.CDLL(LIB_PATH)

# Type aliases
c_size_t = ctypes.c_size_t
c_void_p = ctypes.c_void_p
c_int = ctypes.c_int

# nvcompBatchedANSCompressOpts_t = struct { int unused; } — empty in practice
class ANSCompressOpts(ctypes.Structure):
    _fields_ = []

class ANSDecompressOpts(ctypes.Structure):
    _fields_ = []


def nvcomp_ans_compress(data_gpu: cp.ndarray):
    """Compress a single chunk using nvCOMP batched ANS (batch_size=1)."""
    n = data_gpu.nbytes

    # Get max compressed chunk size
    max_output_size = ctypes.c_size_t()
    opts = ANSCompressOpts()
    ret = nvcomp.nvcompBatchedANSCompressGetMaxOutputChunkSize(
        c_size_t(n), ctypes.byref(opts), ctypes.byref(max_output_size))
    if ret != 0:
        raise RuntimeError(f"GetMaxOutputChunkSize failed: {ret}")

    # Get temp size
    temp_size = ctypes.c_size_t()
    batch_size = 1
    uncomp_sizes = cp.array([n], dtype=cp.uint64)
    ret = nvcomp.nvcompBatchedANSCompressGetTempSizeSync(
        c_size_t(batch_size),
        uncomp_sizes.data.ptr,
        ctypes.byref(opts),
        ctypes.byref(temp_size))
    if ret != 0:
        raise RuntimeError(f"CompressGetTempSize failed: {ret}")

    # Allocate
    comp_buf = cp.empty(max_output_size.value, dtype=cp.uint8)
    temp_buf = cp.empty(temp_size.value, dtype=cp.uint8) if temp_size.value > 0 else cp.empty(1, dtype=cp.uint8)
    comp_sizes = cp.zeros(1, dtype=cp.uint64)

    # Input/output pointer arrays (batch_size=1)
    input_ptrs = cp.array([data_gpu.data.ptr], dtype=cp.uint64)
    output_ptrs = cp.array([comp_buf.data.ptr], dtype=cp.uint64)

    # Compress
    ret = nvcomp.nvcompBatchedANSCompressAsync(
        input_ptrs.data.ptr,
        uncomp_sizes.data.ptr,
        c_size_t(0),  # max_uncompressed_chunk_bytes (0 = use uncomp_sizes)
        c_size_t(batch_size),
        temp_buf.data.ptr,
        c_size_t(temp_size.value),
        output_ptrs.data.ptr,
        comp_sizes.data.ptr,
        ctypes.byref(opts),
        c_void_p(0),  # stream = default
    )
    cp.cuda.Stream.null.synchronize()
    if ret != 0:
        raise RuntimeError(f"CompressAsync failed: {ret}")

    actual_comp_size = int(comp_sizes.get()[0])
    return comp_buf[:actual_comp_size], actual_comp_size


def nvcomp_ans_decompress(comp_buf: cp.ndarray, comp_size: int, orig_size: int):
    """Decompress a single chunk."""
    # Get temp size
    temp_size = ctypes.c_size_t()
    batch_size = 1
    comp_sizes = cp.array([comp_size], dtype=cp.uint64)
    comp_ptrs = cp.array([comp_buf.data.ptr], dtype=cp.uint64)

    ret = nvcomp.nvcompBatchedANSDecompressGetTempSizeSync(
        c_size_t(batch_size),
        comp_ptrs.data.ptr,
        comp_sizes.data.ptr,
        ctypes.byref(temp_size))
    if ret != 0:
        raise RuntimeError(f"DecompressGetTempSize failed: {ret}")

    # Allocate
    output = cp.empty(orig_size, dtype=cp.uint8)
    temp = cp.empty(temp_size.value, dtype=cp.uint8) if temp_size.value > 0 else cp.empty(1, dtype=cp.uint8)
    actual_sizes = cp.zeros(1, dtype=cp.uint64)
    statuses = cp.zeros(1, dtype=cp.int32)

    output_ptrs = cp.array([output.data.ptr], dtype=cp.uint64)
    uncomp_sizes = cp.array([orig_size], dtype=cp.uint64)

    ret = nvcomp.nvcompBatchedANSDecompressAsync(
        comp_ptrs.data.ptr,
        comp_sizes.data.ptr,
        uncomp_sizes.data.ptr,
        actual_sizes.data.ptr,
        c_size_t(batch_size),
        temp.data.ptr,
        c_size_t(temp_size.value),
        output_ptrs.data.ptr,
        statuses.data.ptr,
        c_void_p(0),  # stream
    )
    cp.cuda.Stream.null.synchronize()
    if ret != 0:
        raise RuntimeError(f"DecompressAsync failed: {ret}")

    return output


def main():
    from transformers import AutoModelForCausalLM

    print("=" * 100)
    print("nvCOMP ANS Baseline (C API via ctypes)")
    print("=" * 100)

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16, device_map="cpu")

    start_evt = cp.cuda.Event()
    end_evt = cp.cuda.Event()

    total_orig = 0
    total_comp = 0
    total_us = 0

    print(f"\n{'Layer':<45} {'n':>10} {'Ratio':>7} {'Dec GB/s':>9} {'OK':>4}")

    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16 or param.numel() < 2_000_000:
            continue

        fp8 = param.data.to(torch.float8_e4m3fn)
        n = fp8.numel()
        raw = fp8.view(torch.uint8).flatten().numpy()
        raw_gpu = cp.asarray(raw)

        try:
            comp_buf, comp_size = nvcomp_ans_compress(raw_gpu)
            ratio = comp_size / n * 100
            total_orig += n
            total_comp += comp_size

            # Warmup decompress
            for _ in range(5):
                nvcomp_ans_decompress(comp_buf, comp_size, n)
            cp.cuda.Stream.null.synchronize()

            # Benchmark decompress
            times = []
            for _ in range(30):
                start_evt.record()
                nvcomp_ans_decompress(comp_buf, comp_size, n)
                end_evt.record()
                end_evt.synchronize()
                times.append(cp.cuda.get_elapsed_time(start_evt, end_evt) * 1000)

            times.sort()
            avg = np.mean(times[3:-3])
            gbps = n / 1e9 / (avg / 1e6)
            total_us += avg

            # Verify
            dec = nvcomp_ans_decompress(comp_buf, comp_size, n)
            ok = np.array_equal(raw, dec.get())

            print(f"  {name:<43} {n:>10,} {ratio:>6.1f}% {gbps:>8.1f} {'Y' if ok else 'N':>4}")

            del raw_gpu, comp_buf, dec
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            print(f"  {name:<43} {n:>10,} ERROR: {e}")
            # Continue trying
            del raw_gpu
            cp.get_default_memory_pool().free_all_blocks()

    if total_orig > 0:
        agg_ratio = total_comp / total_orig * 100
        agg_gbps = total_orig / 1e9 / (total_us / 1e6)
        print(f"\n  Aggregate: {agg_ratio:.1f}% @ {agg_gbps:.1f} GB/s")


if __name__ == "__main__":
    main()
