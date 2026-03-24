"""
FP8 Packed 6-bit TCA-TBE Codec

Packs 4 × 6-bit codes into 3 bytes (24 bits). Common values (96%): 2-bit exponent
offset + 1-bit sign + 3-bit mantissa = 6 bits. Escapes: offset = k marker, raw FP8
in overflow buffer. Escape prefix sum precomputed on GPU.

Expected ratio: ~79% (75% codes + 4% overflow + ~0.5% header/prefix)
Expected decode: high throughput (fixed-width, no variable-length branching)
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


class FP8PackedTBEEncoder:
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

        # 6-bit codes: offset(2) | sign(1) | mantissa(3)
        # Escape marker: offset = k (value 3 for k=3)
        codes_6bit = np.zeros(n, dtype=np.uint8)
        codes_6bit[is_common] = (offsets[is_common].astype(np.uint8) << 4) | (signs[is_common] << 3) | mantissas[is_common]
        codes_6bit[is_escape] = (self.k << 4)  # escape marker: 0x30 for k=3

        # Pack 4 codes per 3 bytes (24 bits = 4 × 6)
        # Pad to multiple of 4
        pad = (4 - n % 4) % 4
        if pad > 0:
            codes_6bit = np.concatenate([codes_6bit, np.zeros(pad, dtype=np.uint8)])
        n_padded = len(codes_6bit)

        # Pack: codes[0] in bits 23-18, codes[1] in 17-12, codes[2] in 11-6, codes[3] in 5-0
        packed = np.zeros(n_padded * 3 // 4, dtype=np.uint8)
        for group in range(n_padded // 4):
            c0, c1, c2, c3 = codes_6bit[group*4:group*4+4]
            # 24-bit value
            val24 = (int(c0 & 0x3F) << 18) | (int(c1 & 0x3F) << 12) | (int(c2 & 0x3F) << 6) | int(c3 & 0x3F)
            packed[group*3]   = (val24 >> 16) & 0xFF
            packed[group*3+1] = (val24 >> 8) & 0xFF
            packed[group*3+2] = val24 & 0xFF

        # Overflow buffer
        overflow = raw[is_escape].astype(np.uint8)
        n_escapes = len(overflow)

        # Escape flag array (for prefix sum computation on GPU)
        escape_flags = is_escape.astype(np.uint8)

        # Size computation
        packed_bytes = len(packed)
        overflow_bytes = n_escapes
        flags_bytes = n  # 1 byte per element for escape flags (could be packed to n/8)
        # Actually let's pack flags as bits
        flags_packed = np.packbits(escape_flags)
        flags_bytes = len(flags_packed)
        header_bytes = 8  # base_exp, k, n, n_escapes

        compressed_bytes = header_bytes + packed_bytes + flags_bytes + overflow_bytes
        original_bytes = n

        return {
            'packed_codes': packed,
            'escape_flags_packed': flags_packed,
            'overflow': overflow,
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


# GPU decode kernel for packed 6-bit TBE
PACKED_TBE_DECODE_KERNEL = r"""
extern "C"
__global__ void fp8_packed_tbe_decode(
    const unsigned char* __restrict__ packed_codes,  // 3 bytes per 4 codes
    const unsigned char* __restrict__ escape_flags,  // packed bits: 1=escape
    const unsigned char* __restrict__ overflow,      // raw FP8 for escapes
    const unsigned int*  __restrict__ escape_prefix, // exclusive prefix sum of escape flags
    unsigned char*       __restrict__ output,
    const int base_exp,
    const int k,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Extract 6-bit code from packed stream
    // 4 codes per 3 bytes: group = idx/4, position = idx%4
    int group = idx >> 2;
    int pos = idx & 3;
    int byte_offset = group * 3;

    // Read 3 bytes of this group
    unsigned int b0 = packed_codes[byte_offset];
    unsigned int b1 = packed_codes[byte_offset + 1];
    unsigned int b2 = packed_codes[byte_offset + 2];
    unsigned int val24 = (b0 << 16) | (b1 << 8) | b2;

    // Extract 6-bit code at position (0=MSB, 3=LSB)
    int shift = (3 - pos) * 6;
    unsigned char code = (val24 >> shift) & 0x3F;

    int offset = (code >> 4) & 0x3;  // top 2 bits of 6-bit code

    if (offset < k) {
        // Common value: reconstruct FP8
        int sign = (code >> 3) & 1;
        int mantissa = code & 0x7;
        int exponent = base_exp + offset;
        output[idx] = (sign << 7) | (exponent << 3) | mantissa;
    } else {
        // Escape: read from overflow using precomputed prefix sum
        unsigned int esc_idx = escape_prefix[idx];
        output[idx] = overflow[esc_idx];
    }
}
"""


class FP8PackedTBEDecoder:
    def __init__(self):
        self.kernel = cp.RawKernel(PACKED_TBE_DECODE_KERNEL, 'fp8_packed_tbe_decode')

    def decode(self, compressed: dict) -> torch.Tensor:
        n = compressed['n_elements']
        shape = compressed['shape']

        packed_gpu = cp.asarray(compressed['packed_codes'])
        flags_packed_gpu = cp.asarray(compressed['escape_flags_packed'])
        overflow_gpu = cp.asarray(compressed['overflow']) if compressed['n_escapes'] > 0 else cp.zeros(1, dtype=cp.uint8)

        # Compute escape prefix sum
        escape_flags = cp.unpackbits(flags_packed_gpu)[:n].astype(cp.uint32)
        escape_prefix = cp.cumsum(escape_flags) - escape_flags  # exclusive

        output_gpu = cp.zeros(n, dtype=cp.uint8)

        threads = 256
        blocks = (n + threads - 1) // threads

        self.kernel(
            (blocks,), (threads,),
            (packed_gpu, flags_packed_gpu, overflow_gpu, escape_prefix,
             output_gpu, compressed['base_exp'], compressed['k'], n)
        )
        cp.cuda.Stream.null.synchronize()

        result = torch.as_tensor(output_gpu, device='cuda').view(torch.float8_e4m3fn).reshape(shape)
        return result.clone()


def benchmark(model_name="Qwen/Qwen3-0.6B"):
    from transformers import AutoModelForCausalLM

    print("=" * 90)
    print("FP8 Packed 6-bit TBE Benchmark")
    print("=" * 90)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    encoder = FP8PackedTBEEncoder(k=3)
    decoder = FP8PackedTBEDecoder()

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

        # Benchmark decode speed (include prefix sum computation)
        for _ in range(3):
            decoder.decode(comp)
        times = []
        for _ in range(20):
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            decoder.decode(comp)
            cp.cuda.Stream.null.synchronize()
            times.append(time.perf_counter() - t0)
        avg = np.mean(times)
        throughput = fp8.numel() / 1e9 / avg

        total_orig += comp['original_bytes']
        total_comp += comp['compressed_bytes']
        total_dec_time += avg

        print(f"  {name:<48} {param.numel():>10,} {comp['coverage']:>5.1%} "
              f"{comp['ratio']:>6.1f}% {avg*1000:>7.3f} {throughput:>8.1f} "
              f"{'Y' if is_ok else 'N':>4}")

    overall = total_comp / total_orig * 100
    print(f"\n  Aggregate ratio: {overall:.2f}%")
    print(f"  Aggregate Dec GB/s: {total_orig / 1e9 / total_dec_time:.1f}")
    print(f"  Lossless: {'ALL PASS' if all_ok else 'FAIL'}")
    print(f"\n  Comparison:")
    print(f"    Dense FP8:              100.0%  (baseline)")
    print(f"    Our Packed TBE (6-bit): {overall:.1f}%  (constant-time, fusable)")
    print(f"    Old TBE (byte-aligned): 116.6%  (wasted 2 bits/code)")
    print(f"    Our Huffman (GPU):       77.1%  (variable-length)")
    print(f"    Entropy limit:           70.4%")


if __name__ == "__main__":
    benchmark()
