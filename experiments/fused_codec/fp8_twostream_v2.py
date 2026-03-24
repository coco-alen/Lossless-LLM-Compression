"""
FP8 Two-Stream Codec v2: Fused prefix sum + decode

Key optimization: Compute escape prefix sum INSIDE the decode kernel using
warp ballot + block-level scan. Eliminates the separate CuPy cumsum pass
that was bottlenecking at 4 GB/s.

Architecture:
1. Each thread reads its 2-bit exp code from packed stream
2. Warp-level ballot identifies escapes, __popc gives intra-warp prefix
3. Block-level prefix sum of per-warp escape counts (in shared memory)
4. Block-to-block prefix sum via precomputed block_escape_counts array
5. Each thread reconstructs FP8 using the computed escape index

This eliminates:
- The n-element uint32 escape_prefix array (saves 4n bytes)
- The separate CuPy cumsum kernel launch
- The unpacked exp_codes transfer
"""

import torch
import numpy as np
import cupy as cp
import time


def find_best_window(exponents: np.ndarray, k: int = 3) -> tuple:
    counts = np.bincount(exponents, minlength=16)
    best_base, best_cov = 0, 0
    for base in range(16 - k + 1):
        cov = counts[base:base+k].sum()
        if cov > best_cov:
            best_cov = cov
            best_base = base
    return best_base, best_cov / len(exponents)


class FP8TwoStreamEncoderV2:
    def __init__(self, k=3):
        self.k = k

    def encode(self, fp8_tensor: torch.Tensor, base_exp: int = None) -> dict:
        raw = fp8_tensor.contiguous().view(torch.uint8).flatten().numpy()
        n = len(raw)

        signs = (raw >> 7) & 1
        exponents = (raw >> 3) & 0xF
        mantissas = raw & 0x7

        if base_exp is None:
            base_exp, coverage = find_best_window(exponents, self.k)
        else:
            offsets_tmp = exponents.astype(np.int32) - base_exp
            coverage = np.mean((offsets_tmp >= 0) & (offsets_tmp < self.k))

        offsets = exponents.astype(np.int32) - base_exp
        is_common = (offsets >= 0) & (offsets < self.k)
        is_escape = ~is_common

        # Stream 1: exp_codes (2 bits each, packed 4 per byte)
        exp_codes = np.full(n, self.k, dtype=np.uint8)
        exp_codes[is_common] = offsets[is_common].astype(np.uint8)

        pad_exp = (4 - n % 4) % 4
        exp_padded = np.concatenate([exp_codes, np.zeros(pad_exp, dtype=np.uint8)]) if pad_exp > 0 else exp_codes
        exp_packed = (exp_padded[0::4] << 6) | (exp_padded[1::4] << 4) | (exp_padded[2::4] << 2) | exp_padded[3::4]
        exp_packed = exp_packed.astype(np.uint8)

        # Stream 2: sign|mantissa (4 bits each, packed 2 per byte)
        sm = ((signs << 3) | mantissas).astype(np.uint8)
        pad_sm = n % 2
        sm_padded = np.concatenate([sm, np.zeros(pad_sm, dtype=np.uint8)]) if pad_sm > 0 else sm
        sm_packed = (sm_padded[0::2] << 4) | sm_padded[1::2]
        sm_packed = sm_packed.astype(np.uint8)

        # Stream 3: overflow exponents (4 bits each, packed 2 per byte)
        escape_exps = exponents[is_escape].astype(np.uint8)
        n_escapes = len(escape_exps)
        pad_esc = n_escapes % 2
        esc_padded = np.concatenate([escape_exps, np.zeros(pad_esc, dtype=np.uint8)]) if pad_esc > 0 else escape_exps
        if len(esc_padded) > 0:
            overflow_packed = (esc_padded[0::2] << 4) | esc_padded[1::2]
        else:
            overflow_packed = np.array([], dtype=np.uint8)
        overflow_packed = overflow_packed.astype(np.uint8)

        # Precompute per-block escape counts for block-level prefix sum
        # Using threads_per_block = 256
        threads_per_block = 256
        n_blocks = (n + threads_per_block - 1) // threads_per_block
        block_escape_counts = np.zeros(n_blocks + 1, dtype=np.uint32)
        for b in range(n_blocks):
            start = b * threads_per_block
            end = min(start + threads_per_block, n)
            block_escape_counts[b + 1] = block_escape_counts[b] + np.sum(exp_codes[start:end] >= self.k)

        # Size computation
        exp_bytes = len(exp_packed)
        sm_bytes = len(sm_packed)
        overflow_bytes = len(overflow_packed)
        prefix_bytes = len(block_escape_counts) * 4
        header_bytes = 8

        compressed_bytes = header_bytes + exp_bytes + sm_bytes + overflow_bytes + prefix_bytes
        original_bytes = n

        return {
            'exp_packed': exp_packed,
            'sm_packed': sm_packed,
            'overflow_packed': overflow_packed,
            'block_escape_prefix': block_escape_counts,
            'base_exp': int(base_exp),
            'k': self.k,
            'n_elements': n,
            'n_escapes': n_escapes,
            'n_blocks': n_blocks,
            'shape': fp8_tensor.shape,
            'coverage': coverage,
            'original_bytes': original_bytes,
            'compressed_bytes': compressed_bytes,
            'ratio': compressed_bytes / original_bytes * 100,
        }


# Fused prefix sum + decode kernel
FUSED_DECODE_KERNEL = r"""
extern "C"
__global__ void fp8_twostream_fused_decode(
    const unsigned char* __restrict__ exp_packed,
    const unsigned char* __restrict__ sm_packed,
    const unsigned char* __restrict__ overflow_packed,
    const unsigned int*  __restrict__ block_escape_prefix,
    unsigned char*       __restrict__ output,
    const int base_exp,
    const int k,
    const int n
) {
    // Each thread handles one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Read 2-bit exponent code
    int code = 0;
    int is_esc = 0;
    if (idx < n) {
        int exp_byte_idx = idx >> 2;
        int exp_bit_pos  = (3 - (idx & 3)) * 2;
        unsigned char exp_byte = exp_packed[exp_byte_idx];
        code = (exp_byte >> exp_bit_pos) & 0x3;
        is_esc = (code >= k) ? 1 : 0;
    }

    // Warp-level escape counting using ballot
    unsigned int warp_mask = __ballot_sync(0xFFFFFFFF, is_esc);
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    // Intra-warp prefix: count escapes in lanes before me
    unsigned int lanes_before = (1u << lane_id) - 1u;
    int intra_warp_prefix = __popc(warp_mask & lanes_before);
    int warp_escape_count = __popc(warp_mask);

    // Block-level prefix sum of warp escape counts (in shared memory)
    __shared__ unsigned int warp_prefix[33];  // max 256/32 = 8 warps + 1
    int n_warps = blockDim.x >> 5;

    if (lane_id == 0) {
        warp_prefix[warp_id] = warp_escape_count;
    }
    __syncthreads();

    // Exclusive prefix sum of warp counts (single-warp scan, n_warps <= 8)
    if (threadIdx.x == 0) {
        unsigned int sum = 0;
        for (int w = 0; w < n_warps; w++) {
            unsigned int tmp = warp_prefix[w];
            warp_prefix[w] = sum;
            sum += tmp;
        }
        warp_prefix[n_warps] = sum;
    }
    __syncthreads();

    // Compute global escape index
    unsigned int esc_idx = block_escape_prefix[blockIdx.x] + warp_prefix[warp_id] + intra_warp_prefix;

    if (idx >= n) return;

    // Read 4-bit sign|mantissa
    int sm_byte_idx = idx >> 1;
    int sm_nibble   = 1 - (idx & 1);
    unsigned char sm_byte = sm_packed[sm_byte_idx];
    int sm = (sm_byte >> (sm_nibble * 4)) & 0xF;

    int sign = (sm >> 3) & 1;
    int mantissa = sm & 0x7;

    int exponent;
    if (code < k) {
        exponent = base_exp + code;
    } else {
        int ov_byte_idx = esc_idx >> 1;
        int ov_nibble = 1 - (esc_idx & 1);
        exponent = (overflow_packed[ov_byte_idx] >> (ov_nibble * 4)) & 0xF;
    }

    output[idx] = (unsigned char)((sign << 7) | (exponent << 3) | mantissa);
}
"""


class FP8TwoStreamDecoderV2:
    def __init__(self):
        self.kernel = cp.RawKernel(FUSED_DECODE_KERNEL, 'fp8_twostream_fused_decode')

    def decode(self, compressed: dict, exp_gpu=None, sm_gpu=None, overflow_gpu=None, prefix_gpu=None) -> torch.Tensor:
        n = compressed['n_elements']
        shape = compressed['shape']

        if exp_gpu is None:
            exp_gpu = cp.asarray(compressed['exp_packed'])
        if sm_gpu is None:
            sm_gpu = cp.asarray(compressed['sm_packed'])
        if overflow_gpu is None:
            overflow_gpu = cp.asarray(compressed['overflow_packed']) if compressed['n_escapes'] > 0 else cp.zeros(1, dtype=cp.uint8)
        if prefix_gpu is None:
            prefix_gpu = cp.asarray(compressed['block_escape_prefix'])

        output_gpu = cp.zeros(n, dtype=cp.uint8)

        threads = 256
        blocks = compressed['n_blocks']

        self.kernel(
            (blocks,), (threads,),
            (exp_gpu, sm_gpu, overflow_gpu, prefix_gpu,
             output_gpu, compressed['base_exp'], compressed['k'], n)
        )
        cp.cuda.Stream.null.synchronize()

        result = torch.as_tensor(output_gpu, device='cuda').view(torch.float8_e4m3fn).reshape(shape)
        return result.clone()


def benchmark(model_name="Qwen/Qwen3-0.6B"):
    from transformers import AutoModelForCausalLM

    print("=" * 90)
    print("FP8 Two-Stream v2 (Fused Prefix Sum) Benchmark")
    print("=" * 90)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    encoder = FP8TwoStreamEncoderV2(k=3)
    decoder = FP8TwoStreamDecoderV2()

    print(f"\n{'Layer':<50} {'Params':>10} {'Cover':>6} {'Ratio':>7} "
          f"{'Dec ms':>8} {'Dec GB/s':>9} {'OK':>4}")

    total_orig = 0
    total_comp = 0
    total_dec_time = 0
    all_ok = True

    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16 or param.numel() < 500_000:
            continue

        fp8 = param.data.to(torch.float8_e4m3fn)
        comp = encoder.encode(fp8)

        # GPU decode
        recovered = decoder.decode(comp)
        is_ok = torch.equal(fp8.view(torch.uint8).cuda(), recovered.view(torch.uint8))
        if not is_ok:
            all_ok = False

        # Precompute GPU arrays
        exp_gpu = cp.asarray(comp['exp_packed'])
        sm_gpu = cp.asarray(comp['sm_packed'])
        overflow_gpu = cp.asarray(comp['overflow_packed']) if comp['n_escapes'] > 0 else cp.zeros(1, dtype=cp.uint8)
        prefix_gpu = cp.asarray(comp['block_escape_prefix'])

        # Warmup
        for _ in range(5):
            decoder.decode(comp, exp_gpu, sm_gpu, overflow_gpu, prefix_gpu)

        # Benchmark
        times = []
        for _ in range(30):
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            decoder.decode(comp, exp_gpu, sm_gpu, overflow_gpu, prefix_gpu)
            cp.cuda.Stream.null.synchronize()
            times.append(time.perf_counter() - t0)
        avg = np.mean(sorted(times)[:25])  # trim worst 5
        throughput = fp8.numel() / 1e9 / avg

        total_orig += comp['original_bytes']
        total_comp += comp['compressed_bytes']
        total_dec_time += avg

        print(f"  {name:<48} {fp8.numel():>10,} {comp['coverage']:>5.1%} "
              f"{comp['ratio']:>6.1f}% {avg*1000:>7.3f} "
              f"{throughput:>8.1f} "
              f"{'Y' if is_ok else 'N':>4}")

    overall = total_comp / total_orig * 100
    agg = total_orig / 1e9 / total_dec_time

    print(f"\n  Aggregate ratio: {overall:.2f}%")
    print(f"  Aggregate Dec GB/s: {agg:.1f}")
    print(f"  Lossless: {'ALL PASS' if all_ok else 'FAIL'}")
    print(f"\n  Comparison:")
    print(f"    Dense FP8:              100.0%  (baseline)")
    print(f"    Our Two-Stream v2:      {overall:.1f}%  @ {agg:.0f} GB/s")
    print(f"    Our Two-Stream v1:       77.0%  @ 60 GB/s (kernel-only)")
    print(f"    nvCOMP ANS:              85.7%  @ 29-56 GB/s")
    print(f"    Entropy limit:           70.4%")


if __name__ == "__main__":
    benchmark()
