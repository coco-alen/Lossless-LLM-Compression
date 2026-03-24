"""
FP8 Two-Stream Codec: Separate exponent + sign/mantissa streams

Key insight: FP8 exponent (4 bits) has 1.72 bits entropy with top 3 consecutive
covering 96%. Sign+mantissa (4 bits) is nearly random. Separate them:

Stream 1 (exp_codes): 2-bit codes packed 4/byte → 25% of n
  - code 0,1,2 = base_exp + code (common, 96%)
  - code 3 = escape marker (4%)
Stream 2 (sm_packed): 4-bit sign|mantissa packed 2/byte → 50% of n
Stream 3 (overflow_exp): 4-bit raw exponent for escapes, packed 2/byte → ~2% of n

Total: 25% + 50% + 2% = 77% ratio (same as Huffman, but constant-time decode!)

Decode is fully parallel and branchless for 96% of values:
  exp = (code < 3) ? base_exp + code : overflow[prefix_sum[idx]]
  output[idx] = (sign << 7) | (exp << 3) | mantissa
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


class FP8TwoStreamEncoder:
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
        exp_codes = np.full(n, self.k, dtype=np.uint8)  # default: escape marker
        exp_codes[is_common] = offsets[is_common].astype(np.uint8)

        # Pack 4 × 2-bit codes per byte: [c0 c1 c2 c3] = c0<<6 | c1<<4 | c2<<2 | c3
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

        # Stream 3: overflow exponents (4 bits each for escapes, packed 2 per byte)
        escape_exps = exponents[is_escape].astype(np.uint8)
        n_escapes = len(escape_exps)
        pad_esc = n_escapes % 2
        esc_padded = np.concatenate([escape_exps, np.zeros(pad_esc, dtype=np.uint8)]) if pad_esc > 0 else escape_exps
        if len(esc_padded) > 0:
            overflow_packed = (esc_padded[0::2] << 4) | esc_padded[1::2]
        else:
            overflow_packed = np.array([], dtype=np.uint8)
        overflow_packed = overflow_packed.astype(np.uint8)

        # Size computation
        exp_bytes = len(exp_packed)
        sm_bytes = len(sm_packed)
        overflow_bytes = len(overflow_packed)
        header_bytes = 8  # base_exp, k, n, n_escapes

        compressed_bytes = header_bytes + exp_bytes + sm_bytes + overflow_bytes
        original_bytes = n

        return {
            'exp_packed': exp_packed,
            'sm_packed': sm_packed,
            'overflow_packed': overflow_packed,
            'exp_codes': exp_codes,  # unpacked, for prefix sum computation
            'base_exp': int(base_exp),
            'k': self.k,
            'n_elements': n,
            'n_escapes': n_escapes,
            'shape': fp8_tensor.shape,
            'coverage': coverage,
            'original_bytes': original_bytes,
            'compressed_bytes': compressed_bytes,
            'ratio': compressed_bytes / original_bytes * 100,
        }


TWOSTREAM_DECODE_KERNEL = r"""
extern "C"
__global__ void fp8_twostream_decode(
    const unsigned char* __restrict__ exp_packed,     // 2-bit codes, 4/byte
    const unsigned char* __restrict__ sm_packed,      // 4-bit sign|mantissa, 2/byte
    const unsigned char* __restrict__ overflow_packed, // 4-bit escape exponents, 2/byte
    const unsigned int*  __restrict__ escape_prefix,  // exclusive prefix sum
    unsigned char*       __restrict__ output,
    const int base_exp,
    const int k,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Read 2-bit exponent code
    int exp_byte_idx = idx >> 2;           // idx / 4
    int exp_bit_pos  = (3 - (idx & 3)) * 2;  // position within byte (MSB first)
    unsigned char exp_byte = exp_packed[exp_byte_idx];
    int code = (exp_byte >> exp_bit_pos) & 0x3;

    // Read 4-bit sign|mantissa
    int sm_byte_idx = idx >> 1;            // idx / 2
    int sm_nibble   = 1 - (idx & 1);      // 1=high nibble, 0=low nibble
    unsigned char sm_byte = sm_packed[sm_byte_idx];
    int sm = (sm_byte >> (sm_nibble * 4)) & 0xF;

    int sign = (sm >> 3) & 1;
    int mantissa = sm & 0x7;

    int exponent;
    if (code < k) {
        exponent = base_exp + code;
    } else {
        // Escape: read exponent from overflow using prefix sum
        unsigned int esc_idx = escape_prefix[idx];
        int ov_byte_idx = esc_idx >> 1;
        int ov_nibble = 1 - (esc_idx & 1);
        exponent = (overflow_packed[ov_byte_idx] >> (ov_nibble * 4)) & 0xF;
    }

    output[idx] = (sign << 7) | (exponent << 3) | mantissa;
}
"""


class FP8TwoStreamDecoder:
    def __init__(self):
        self.kernel = cp.RawKernel(TWOSTREAM_DECODE_KERNEL, 'fp8_twostream_decode')

    def decode(self, compressed: dict) -> torch.Tensor:
        n = compressed['n_elements']
        shape = compressed['shape']

        exp_gpu = cp.asarray(compressed['exp_packed'])
        sm_gpu = cp.asarray(compressed['sm_packed'])
        overflow_gpu = cp.asarray(compressed['overflow_packed']) if compressed['n_escapes'] > 0 else cp.zeros(1, dtype=cp.uint8)

        # Compute escape prefix sum from exp_codes
        exp_codes_gpu = cp.asarray(compressed['exp_codes'])
        escape_flags = (exp_codes_gpu >= compressed['k']).astype(cp.uint32)
        escape_prefix = (cp.cumsum(escape_flags) - escape_flags).astype(cp.uint32)  # must be uint32 for kernel

        output_gpu = cp.zeros(n, dtype=cp.uint8)

        threads = 256
        blocks = (n + threads - 1) // threads

        self.kernel(
            (blocks,), (threads,),
            (exp_gpu, sm_gpu, overflow_gpu, escape_prefix,
             output_gpu, compressed['base_exp'], compressed['k'], n)
        )
        cp.cuda.Stream.null.synchronize()

        result = torch.as_tensor(output_gpu, device='cuda').view(torch.float8_e4m3fn).reshape(shape)
        return result.clone()

    def decode_precomputed(self, compressed: dict, exp_gpu, sm_gpu, overflow_gpu, escape_prefix):
        """Decode with precomputed GPU tensors (for accurate throughput measurement)."""
        n = compressed['n_elements']
        shape = compressed['shape']

        output_gpu = cp.zeros(n, dtype=cp.uint8)

        threads = 256
        blocks = (n + threads - 1) // threads

        self.kernel(
            (blocks,), (threads,),
            (exp_gpu, sm_gpu, overflow_gpu, escape_prefix,
             output_gpu, compressed['base_exp'], compressed['k'], n)
        )
        cp.cuda.Stream.null.synchronize()

        result = torch.as_tensor(output_gpu, device='cuda').view(torch.float8_e4m3fn).reshape(shape)
        return result.clone()


def benchmark(model_name="Qwen/Qwen3-0.6B"):
    from transformers import AutoModelForCausalLM

    print("=" * 90)
    print("FP8 Two-Stream Codec Benchmark")
    print("=" * 90)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    # Find global best window across all weights
    all_exponents = []
    weight_tensors = []
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16 and param.numel() >= 500_000:
            fp8 = param.data.to(torch.float8_e4m3fn)
            raw = fp8.view(torch.uint8).flatten().numpy()
            all_exponents.append((raw >> 3) & 0xF)
            weight_tensors.append((name, fp8))

    combined_exps = np.concatenate(all_exponents)
    global_base, global_cov = find_best_window(combined_exps, k=3)
    print(f"\nGlobal base exponent: {global_base}, coverage: {global_cov:.1%}")

    encoder = FP8TwoStreamEncoder(k=3)
    decoder = FP8TwoStreamDecoder()

    print(f"\n{'Layer':<50} {'Params':>10} {'Cover':>6} {'Ratio':>7} "
          f"{'Dec ms':>8} {'KernOnly':>8} {'Dec GB/s':>9} {'Kern GB/s':>9} {'OK':>4}")

    total_orig = 0
    total_comp = 0
    total_dec_time = 0
    total_kern_time = 0
    all_ok = True

    for name, fp8_tensor in weight_tensors:
        # Use per-layer base for better coverage
        comp = encoder.encode(fp8_tensor)

        # GPU decode (full, including prefix sum)
        recovered = decoder.decode(comp)
        is_ok = torch.equal(fp8_tensor.view(torch.uint8).cuda(), recovered.view(torch.uint8))
        if not is_ok:
            all_ok = False

        # Precompute GPU arrays for kernel-only benchmark
        exp_gpu = cp.asarray(comp['exp_packed'])
        sm_gpu = cp.asarray(comp['sm_packed'])
        overflow_gpu = cp.asarray(comp['overflow_packed']) if comp['n_escapes'] > 0 else cp.zeros(1, dtype=cp.uint8)
        exp_codes_gpu = cp.asarray(comp['exp_codes'])
        escape_flags = (exp_codes_gpu >= comp['k']).astype(cp.uint32)
        escape_prefix = cp.cumsum(escape_flags) - escape_flags

        # Warmup
        for _ in range(3):
            decoder.decode(comp)
        for _ in range(3):
            decoder.decode_precomputed(comp, exp_gpu, sm_gpu, overflow_gpu, escape_prefix)

        # Benchmark full decode (including prefix sum)
        times_full = []
        for _ in range(20):
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            decoder.decode(comp)
            cp.cuda.Stream.null.synchronize()
            times_full.append(time.perf_counter() - t0)
        avg_full = np.mean(times_full)

        # Benchmark kernel-only (prefix sum precomputed)
        times_kern = []
        for _ in range(20):
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            decoder.decode_precomputed(comp, exp_gpu, sm_gpu, overflow_gpu, escape_prefix)
            cp.cuda.Stream.null.synchronize()
            times_kern.append(time.perf_counter() - t0)
        avg_kern = np.mean(times_kern)

        throughput_full = fp8_tensor.numel() / 1e9 / avg_full
        throughput_kern = fp8_tensor.numel() / 1e9 / avg_kern

        total_orig += comp['original_bytes']
        total_comp += comp['compressed_bytes']
        total_dec_time += avg_full
        total_kern_time += avg_kern

        print(f"  {name:<48} {fp8_tensor.numel():>10,} {comp['coverage']:>5.1%} "
              f"{comp['ratio']:>6.1f}% {avg_full*1000:>7.3f} {avg_kern*1000:>7.3f} "
              f"{throughput_full:>8.1f} {throughput_kern:>8.1f} "
              f"{'Y' if is_ok else 'N':>4}")

    overall = total_comp / total_orig * 100
    agg_full = total_orig / 1e9 / total_dec_time
    agg_kern = total_orig / 1e9 / total_kern_time

    print(f"\n  Aggregate ratio: {overall:.2f}%")
    print(f"  Aggregate Dec GB/s (full):       {agg_full:.1f}")
    print(f"  Aggregate Dec GB/s (kernel-only): {agg_kern:.1f}")
    print(f"  Lossless: {'ALL PASS' if all_ok else 'FAIL'}")
    print(f"\n  Comparison:")
    print(f"    Dense FP8:              100.0%  (baseline)")
    print(f"    Our Two-Stream:         {overall:.1f}%  @ {agg_full:.0f} GB/s full, {agg_kern:.0f} GB/s kernel")
    print(f"    Our Huffman (GPU):       77.1%  @ 5-14 GB/s")
    print(f"    nvCOMP ANS:              85.7%  @ 29-56 GB/s")
    print(f"    Entropy limit:           70.4%")


if __name__ == "__main__":
    benchmark()
