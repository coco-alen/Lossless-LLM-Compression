"""
FP8 Two-Stream Codec v3: 4 elements per thread + vectorized access

Each thread processes 4 consecutive FP8 values:
  - Reads 1 byte from exp_packed (4 × 2-bit codes)
  - Reads 2 bytes from sm_packed (4 × 4-bit sign|mantissa)
  - Writes 4 bytes to output (as uint32)

This naturally aligns with the packed format and reduces thread count 4×.
Warp ballot still handles escape prefix sum.
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


class FP8TwoStreamEncoderV3:
    def __init__(self, k=3):
        self.k = k

    def encode(self, fp8_tensor: torch.Tensor) -> dict:
        raw = fp8_tensor.contiguous().view(torch.uint8).flatten().numpy()
        n = len(raw)

        signs = (raw >> 7) & 1
        exponents = (raw >> 3) & 0xF
        mantissas = raw & 0x7

        base_exp, coverage = find_best_window(exponents, self.k)

        offsets = exponents.astype(np.int32) - base_exp
        is_common = (offsets >= 0) & (offsets < self.k)
        is_escape = ~is_common

        # Stream 1: exp_codes (2 bits each, packed 4 per byte)
        exp_codes = np.full(n, self.k, dtype=np.uint8)
        exp_codes[is_common] = offsets[is_common].astype(np.uint8)

        # Pad to multiple of 4
        pad = (4 - n % 4) % 4
        if pad > 0:
            exp_codes_padded = np.concatenate([exp_codes, np.zeros(pad, dtype=np.uint8)])
        else:
            exp_codes_padded = exp_codes
        exp_packed = (exp_codes_padded[0::4] << 6) | (exp_codes_padded[1::4] << 4) | \
                     (exp_codes_padded[2::4] << 2) | exp_codes_padded[3::4]
        exp_packed = exp_packed.astype(np.uint8)

        # Stream 2: sign|mantissa (4 bits each, packed 2 per byte)
        sm = ((signs << 3) | mantissas).astype(np.uint8)
        if n % 2:
            sm_padded = np.concatenate([sm, np.zeros(1, dtype=np.uint8)])
        else:
            sm_padded = sm
        sm_packed = (sm_padded[0::2] << 4) | sm_padded[1::2]
        sm_packed = sm_packed.astype(np.uint8)

        # Stream 3: overflow exponents
        escape_exps = exponents[is_escape].astype(np.uint8)
        n_escapes = len(escape_exps)
        if n_escapes % 2:
            esc_padded = np.concatenate([escape_exps, np.zeros(1, dtype=np.uint8)])
        else:
            esc_padded = escape_exps
        if len(esc_padded) > 0:
            overflow_packed = (esc_padded[0::2] << 4) | esc_padded[1::2]
        else:
            overflow_packed = np.array([], dtype=np.uint8)
        overflow_packed = overflow_packed.astype(np.uint8)

        # Precompute per-block escape counts
        # Each thread handles 4 elements, so elements_per_block = threads * 4
        threads_per_block = 256
        elems_per_block = threads_per_block * 4
        n_blocks = (n + elems_per_block - 1) // elems_per_block
        block_escape_prefix = np.zeros(n_blocks + 1, dtype=np.uint32)
        for b in range(n_blocks):
            start = b * elems_per_block
            end = min(start + elems_per_block, n)
            block_escape_prefix[b + 1] = block_escape_prefix[b] + np.sum(exp_codes[start:end] >= self.k)

        # Size
        exp_bytes = len(exp_packed)
        sm_bytes = len(sm_packed)
        overflow_bytes = len(overflow_packed)
        prefix_bytes = len(block_escape_prefix) * 4
        compressed_bytes = 8 + exp_bytes + sm_bytes + overflow_bytes + prefix_bytes
        original_bytes = n

        return {
            'exp_packed': exp_packed,
            'sm_packed': sm_packed,
            'overflow_packed': overflow_packed,
            'block_escape_prefix': block_escape_prefix,
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


FUSED_DECODE_V3 = r"""
extern "C"
__global__ void fp8_twostream_v3_decode(
    const unsigned char* __restrict__ exp_packed,
    const unsigned char* __restrict__ sm_packed,
    const unsigned char* __restrict__ overflow_packed,
    const unsigned int*  __restrict__ block_escape_prefix,
    unsigned char*       __restrict__ output,
    const int base_exp,
    const int k,
    const int n
) {
    // Each thread handles 4 consecutive elements
    const int tid = threadIdx.x;
    const int elems_per_block = blockDim.x * 4;
    const int base_idx = blockIdx.x * elems_per_block + tid * 4;

    // Read 1 byte of exp_packed (contains 4 × 2-bit codes)
    unsigned char exp_byte = 0;
    if (base_idx < n) {
        exp_byte = exp_packed[blockIdx.x * blockDim.x + tid];
    }

    // Extract 4 codes
    int code0 = (exp_byte >> 6) & 0x3;
    int code1 = (exp_byte >> 4) & 0x3;
    int code2 = (exp_byte >> 2) & 0x3;
    int code3 = exp_byte & 0x3;

    // Count escapes in this thread's 4 elements
    int my_escapes = 0;
    int esc0 = 0, esc1 = 0, esc2 = 0, esc3 = 0;
    if (base_idx < n)     { esc0 = (code0 >= k); my_escapes += esc0; }
    if (base_idx + 1 < n) { esc1 = (code1 >= k); my_escapes += esc1; }
    if (base_idx + 2 < n) { esc2 = (code2 >= k); my_escapes += esc2; }
    if (base_idx + 3 < n) { esc3 = (code3 >= k); my_escapes += esc3; }

    // Warp-level prefix sum of escape counts
    int lane_id = tid & 31;
    int warp_id = tid >> 5;

    // Inclusive scan within warp using shuffle
    int warp_prefix = my_escapes;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        int tmp = __shfl_up_sync(0xFFFFFFFF, warp_prefix, offset);
        if (lane_id >= offset) warp_prefix += tmp;
    }
    int warp_total = __shfl_sync(0xFFFFFFFF, warp_prefix, 31);
    int warp_excl = warp_prefix - my_escapes;  // exclusive within warp

    // Block-level prefix sum of warp totals
    __shared__ unsigned int warp_sums[33];
    int n_warps = blockDim.x >> 5;

    if (lane_id == 31) {
        warp_sums[warp_id] = warp_total;
    }
    __syncthreads();

    if (tid == 0) {
        unsigned int sum = 0;
        for (int w = 0; w < n_warps; w++) {
            unsigned int tmp = warp_sums[w];
            warp_sums[w] = sum;
            sum += tmp;
        }
    }
    __syncthreads();

    // Global escape index for this thread's first element
    unsigned int global_esc = block_escape_prefix[blockIdx.x] + warp_sums[warp_id] + warp_excl;

    // Read 2 bytes of sm_packed (4 × 4-bit values)
    int sm_base = blockIdx.x * blockDim.x * 2 + tid * 2;
    unsigned char sm_byte0 = 0, sm_byte1 = 0;
    if (sm_base < (n + 1) / 2)     sm_byte0 = sm_packed[sm_base];
    if (sm_base + 1 < (n + 1) / 2) sm_byte1 = sm_packed[sm_base + 1];

    int sm0 = (sm_byte0 >> 4) & 0xF;
    int sm1 = sm_byte0 & 0xF;
    int sm2 = (sm_byte1 >> 4) & 0xF;
    int sm3 = sm_byte1 & 0xF;

    // Decode 4 elements
    unsigned char out0 = 0, out1 = 0, out2 = 0, out3 = 0;
    unsigned int esc_idx = global_esc;

    // Element 0
    if (base_idx < n) {
        int sign = (sm0 >> 3) & 1;
        int mant = sm0 & 0x7;
        int exp;
        if (code0 < k) {
            exp = base_exp + code0;
        } else {
            int ob = esc_idx >> 1;
            int on = 1 - (esc_idx & 1);
            exp = (overflow_packed[ob] >> (on * 4)) & 0xF;
            esc_idx++;
        }
        out0 = (sign << 7) | (exp << 3) | mant;
    }

    // Element 1
    if (base_idx + 1 < n) {
        int sign = (sm1 >> 3) & 1;
        int mant = sm1 & 0x7;
        int exp;
        if (code1 < k) {
            exp = base_exp + code1;
        } else {
            int ob = esc_idx >> 1;
            int on = 1 - (esc_idx & 1);
            exp = (overflow_packed[ob] >> (on * 4)) & 0xF;
            esc_idx++;
        }
        out1 = (sign << 7) | (exp << 3) | mant;
    }

    // Element 2
    if (base_idx + 2 < n) {
        int sign = (sm2 >> 3) & 1;
        int mant = sm2 & 0x7;
        int exp;
        if (code2 < k) {
            exp = base_exp + code2;
        } else {
            int ob = esc_idx >> 1;
            int on = 1 - (esc_idx & 1);
            exp = (overflow_packed[ob] >> (on * 4)) & 0xF;
            esc_idx++;
        }
        out2 = (sign << 7) | (exp << 3) | mant;
    }

    // Element 3
    if (base_idx + 3 < n) {
        int sign = (sm3 >> 3) & 1;
        int mant = sm3 & 0x7;
        int exp;
        if (code3 < k) {
            exp = base_exp + code3;
        } else {
            int ob = esc_idx >> 1;
            int on = 1 - (esc_idx & 1);
            exp = (overflow_packed[ob] >> (on * 4)) & 0xF;
            esc_idx++;
        }
        out3 = (sign << 7) | (exp << 3) | mant;
    }

    // Write 4 bytes (potentially as uint32 for coalesced write)
    if (base_idx + 3 < n) {
        // Write as uint32 for coalesced access
        unsigned int packed_out = ((unsigned int)out0) | ((unsigned int)out1 << 8) |
                                  ((unsigned int)out2 << 16) | ((unsigned int)out3 << 24);
        *reinterpret_cast<unsigned int*>(&output[base_idx]) = packed_out;
    } else {
        if (base_idx < n)     output[base_idx] = out0;
        if (base_idx + 1 < n) output[base_idx + 1] = out1;
        if (base_idx + 2 < n) output[base_idx + 2] = out2;
        if (base_idx + 3 < n) output[base_idx + 3] = out3;
    }
}
"""


class FP8TwoStreamDecoderV3:
    def __init__(self):
        self.kernel = cp.RawKernel(FUSED_DECODE_V3, 'fp8_twostream_v3_decode')

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
    print("FP8 Two-Stream v3 (4 elems/thread + vectorized write) Benchmark")
    print("=" * 90)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    encoder = FP8TwoStreamEncoderV3(k=3)
    decoder = FP8TwoStreamDecoderV3()

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

        recovered = decoder.decode(comp)
        is_ok = torch.equal(fp8.view(torch.uint8).cuda(), recovered.view(torch.uint8))
        if not is_ok:
            all_ok = False

        exp_gpu = cp.asarray(comp['exp_packed'])
        sm_gpu = cp.asarray(comp['sm_packed'])
        overflow_gpu = cp.asarray(comp['overflow_packed']) if comp['n_escapes'] > 0 else cp.zeros(1, dtype=cp.uint8)
        prefix_gpu = cp.asarray(comp['block_escape_prefix'])

        for _ in range(5):
            decoder.decode(comp, exp_gpu, sm_gpu, overflow_gpu, prefix_gpu)

        times = []
        for _ in range(30):
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            decoder.decode(comp, exp_gpu, sm_gpu, overflow_gpu, prefix_gpu)
            cp.cuda.Stream.null.synchronize()
            times.append(time.perf_counter() - t0)
        avg = np.mean(sorted(times)[:25])
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
    print(f"    Our Two-Stream v3:      {overall:.1f}%  @ {agg:.0f} GB/s")
    print(f"    Our Two-Stream v2:       78.6%  @ 57 GB/s")
    print(f"    nvCOMP ANS:              85.7%  @ 29-56 GB/s")
    print(f"    Entropy limit:           70.4%")


if __name__ == "__main__":
    benchmark()
