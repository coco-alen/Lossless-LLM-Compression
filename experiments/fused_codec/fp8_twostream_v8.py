"""
FP8 Two-Stream v8: Per-WARP escape prefix sums (eliminate __syncthreads)

Instead of block-level prefix sum with 2 __syncthreads barriers, precompute
per-WARP cumulative escape counts. Each warp looks up its warp-level prefix
directly from the precomputed array (no shared memory, no sync).

This eliminates:
- __syncthreads() x2 for block-level scan (major latency source)
- Shared memory for warp_sums

Trade-off: slightly larger metadata (warps_per_block * n_blocks * 4 bytes).
For 256 threads: 8 warps/block. For Qwen3-0.6B 155M embed: ~38K blocks × 8 = 304K entries × 4 = 1.2MB.
For 3M layer: ~768 blocks × 8 = 6K entries × 4 = 24KB. Negligible.
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


class EncoderV8:
    def __init__(self, k=3, tpb=256, ept=16):
        self.k, self.tpb, self.ept = k, tpb, ept
        self.warps_per_block = tpb // 32

    def encode(self, fp8_tensor: torch.Tensor) -> dict:
        raw = fp8_tensor.contiguous().view(torch.uint8).flatten().numpy()
        n = len(raw)
        signs = (raw >> 7) & 1
        exponents = (raw >> 3) & 0xF
        mantissas = raw & 0x7
        base_exp, coverage = find_best_window(exponents, self.k)
        offsets = exponents.astype(np.int32) - base_exp
        is_common = (offsets >= 0) & (offsets < self.k)

        exp_codes = np.full(n, self.k, dtype=np.uint8)
        exp_codes[is_common] = offsets[is_common].astype(np.uint8)

        pad = (4 - n % 4) % 4
        ep = np.concatenate([exp_codes, np.zeros(pad, dtype=np.uint8)]) if pad else exp_codes
        exp_packed = ((ep[0::4] << 6) | (ep[1::4] << 4) | (ep[2::4] << 2) | ep[3::4]).astype(np.uint8)

        sm = ((signs << 3) | mantissas).astype(np.uint8)
        sp = np.concatenate([sm, np.zeros(n % 2, dtype=np.uint8)]) if n % 2 else sm
        sm_packed = ((sp[0::2] << 4) | sp[1::2]).astype(np.uint8)

        esc_exp = exponents[~is_common].astype(np.uint8)
        ne = len(esc_exp)
        ep2 = np.concatenate([esc_exp, np.zeros(ne % 2, dtype=np.uint8)]) if ne % 2 else esc_exp
        ov = ((ep2[0::2] << 4) | ep2[1::2]).astype(np.uint8) if len(ep2) else np.array([], dtype=np.uint8)

        # Per-WARP escape prefix sums
        elems_per_block = self.tpb * self.ept
        elems_per_warp = 32 * self.ept  # 512 elements per warp
        nb = (n + elems_per_block - 1) // elems_per_block
        wpb = self.warps_per_block

        # warp_prefix[block * wpb + warp] = cumulative escapes BEFORE this warp
        warp_prefix = np.zeros(nb * wpb + 1, dtype=np.uint32)
        total_esc = 0
        for b in range(nb):
            for w in range(wpb):
                warp_prefix[b * wpb + w] = total_esc
                s = b * elems_per_block + w * elems_per_warp
                e = min(s + elems_per_warp, n)
                if e > s:
                    total_esc += np.sum(exp_codes[s:e] >= self.k)
        warp_prefix[nb * wpb] = total_esc

        cb = 8 + len(exp_packed) + len(sm_packed) + len(ov) + len(warp_prefix) * 4
        return {
            'exp_packed': exp_packed, 'sm_packed': sm_packed, 'overflow_packed': ov,
            'warp_escape_prefix': warp_prefix,
            'base_exp': int(base_exp), 'k': self.k,
            'n_elements': n, 'n_escapes': ne, 'n_blocks': nb,
            'shape': fp8_tensor.shape, 'coverage': coverage,
            'original_bytes': n, 'compressed_bytes': cb, 'ratio': cb / n * 100,
        }


V8_KERNEL = r"""
extern "C"
__global__ void fp8_v8_decode(
    const unsigned char* __restrict__ exp_packed,
    const unsigned char* __restrict__ sm_packed,
    const unsigned char* __restrict__ overflow_packed,
    const unsigned int*  __restrict__ warp_escape_prefix,  // per-warp prefix
    unsigned char*       __restrict__ output,
    const int base_exp, const int k, const int n,
    const int warps_per_block
) {
    const int tid = threadIdx.x;
    const int EPT = 16;
    const int base_idx = blockIdx.x * blockDim.x * EPT + tid * EPT;

    // Read exp codes
    int exp_off = blockIdx.x * blockDim.x * 4 + tid * 4;
    unsigned int ew = (base_idx < n) ? *reinterpret_cast<const unsigned int*>(&exp_packed[exp_off]) : 0;
    unsigned char e0=ew&0xFF, e1=(ew>>8)&0xFF, e2=(ew>>16)&0xFF, e3=(ew>>24)&0xFF;

    int codes[16];
    codes[0]=(e0>>6)&3; codes[1]=(e0>>4)&3; codes[2]=(e0>>2)&3; codes[3]=e0&3;
    codes[4]=(e1>>6)&3; codes[5]=(e1>>4)&3; codes[6]=(e1>>2)&3; codes[7]=e1&3;
    codes[8]=(e2>>6)&3; codes[9]=(e2>>4)&3; codes[10]=(e2>>2)&3; codes[11]=e2&3;
    codes[12]=(e3>>6)&3; codes[13]=(e3>>4)&3; codes[14]=(e3>>2)&3; codes[15]=e3&3;

    // Count this thread's escapes
    int my_esc = 0;
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        if (base_idx + i < n && codes[i] >= k) my_esc++;
    }

    // Intra-warp exclusive prefix sum (warp shuffle, NO shared memory)
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int wp = my_esc;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int t = __shfl_up_sync(0xFFFFFFFF, wp, o);
        if (lane >= o) wp += t;
    }
    int warp_excl = wp - my_esc;

    // Global escape index: precomputed warp prefix + intra-warp offset
    // NO __syncthreads needed!
    unsigned int ei = warp_escape_prefix[blockIdx.x * warps_per_block + warp_id] + warp_excl;

    // Read sm
    int sm_off = blockIdx.x * blockDim.x * 8 + tid * 8;
    unsigned int sw0 = 0, sw1 = 0;
    if (sm_off < (n + 1) / 2) sw0 = *reinterpret_cast<const unsigned int*>(&sm_packed[sm_off]);
    if (sm_off + 4 < (n + 1) / 2) sw1 = *reinterpret_cast<const unsigned int*>(&sm_packed[sm_off + 4]);

    unsigned char sb[8];
    sb[0]=sw0&0xFF; sb[1]=(sw0>>8)&0xFF; sb[2]=(sw0>>16)&0xFF; sb[3]=(sw0>>24)&0xFF;
    sb[4]=sw1&0xFF; sb[5]=(sw1>>8)&0xFF; sb[6]=(sw1>>16)&0xFF; sb[7]=(sw1>>24)&0xFF;

    int sm[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) { sm[i*2] = (sb[i] >> 4) & 0xF; sm[i*2+1] = sb[i] & 0xF; }

    // Decode
    unsigned char out[16];
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        if (base_idx + i < n) {
            int sign = (sm[i] >> 3) & 1;
            int mant = sm[i] & 7;
            int exp_v;
            if (codes[i] < k) {
                exp_v = base_exp + codes[i];
            } else {
                int ob = ei >> 1, on = 1 - (ei & 1);
                exp_v = (overflow_packed[ob] >> (on * 4)) & 0xF;
                ei++;
            }
            out[i] = (sign << 7) | (exp_v << 3) | mant;
        }
    }

    // Write
    if (base_idx + 15 < n) {
        *reinterpret_cast<unsigned int*>(&output[base_idx])    = out[0]|(out[1]<<8)|(out[2]<<16)|(out[3]<<24);
        *reinterpret_cast<unsigned int*>(&output[base_idx+4])  = out[4]|(out[5]<<8)|(out[6]<<16)|(out[7]<<24);
        *reinterpret_cast<unsigned int*>(&output[base_idx+8])  = out[8]|(out[9]<<8)|(out[10]<<16)|(out[11]<<24);
        *reinterpret_cast<unsigned int*>(&output[base_idx+12]) = out[12]|(out[13]<<8)|(out[14]<<16)|(out[15]<<24);
    } else {
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            if (base_idx + i < n) output[base_idx + i] = out[i];
        }
    }
}
"""


def benchmark(model_name="Qwen/Qwen3-0.6B"):
    from transformers import AutoModelForCausalLM
    # Import v5 for comparison
    import sys; sys.path.insert(0, '.')
    from experiments.fused_codec.fp8_twostream_v5 import FP8TwoStreamEncoderV5

    print("=" * 90)
    print("FP8 Two-Stream v8 (No __syncthreads) vs v5 Benchmark")
    print("=" * 90)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    v5_enc = FP8TwoStreamEncoderV5(k=3, elems_per_thread=16)
    v8_enc = EncoderV8(k=3)
    v8_kernel = cp.RawKernel(V8_KERNEL, 'fp8_v8_decode')

    # Also use v5 single kernel from v5's file
    from experiments.fused_codec.fp8_twostream_v5 import FUSED_DECODE_V5
    v5_kernel = cp.RawKernel(FUSED_DECODE_V5, 'fp8_twostream_v5_decode')

    start_evt = cp.cuda.Event()
    end_evt = cp.cuda.Event()

    print(f"\n{'Layer':<42} {'n':>10} {'v5 GB/s':>9} {'v8 GB/s':>9} {'v8/v5':>7} {'OK':>4}")

    total_n = 0
    v5_total = 0
    v8_total = 0
    all_ok = True

    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16 or param.numel() < 500_000:
            continue

        fp8 = param.data.to(torch.float8_e4m3fn)
        n = fp8.numel()

        # V5 data
        c5 = v5_enc.encode(fp8)
        e5 = cp.asarray(c5['exp_packed'])
        s5 = cp.asarray(c5['sm_packed'])
        o5 = cp.asarray(c5['overflow_packed']) if c5['n_escapes'] > 0 else cp.zeros(1, dtype=cp.uint8)
        p5 = cp.asarray(c5['block_escape_prefix'])
        out5 = cp.empty(n, dtype=cp.uint8)

        # V8 data
        c8 = v8_enc.encode(fp8)
        e8 = cp.asarray(c8['exp_packed'])
        s8 = cp.asarray(c8['sm_packed'])
        o8 = cp.asarray(c8['overflow_packed']) if c8['n_escapes'] > 0 else cp.zeros(1, dtype=cp.uint8)
        w8 = cp.asarray(c8['warp_escape_prefix'])
        out8 = cp.empty(n, dtype=cp.uint8)

        nb = c8['n_blocks']
        wpb = v8_enc.warps_per_block

        # Verify
        v8_kernel((nb,), (256,), (e8, s8, o8, w8, out8, c8['base_exp'], c8['k'], n, wpb))
        v5_kernel((c5['n_blocks'],), (256,), (e5, s5, o5, p5, out5, c5['base_exp'], c5['k'], n))
        cp.cuda.Stream.null.synchronize()

        ok_v8 = torch.equal(fp8.view(torch.uint8).cuda(), torch.as_tensor(out8, device='cuda'))
        ok_v5 = torch.equal(fp8.view(torch.uint8).cuda(), torch.as_tensor(out5, device='cuda'))
        if not (ok_v5 and ok_v8):
            all_ok = False

        # Warmup
        for _ in range(10):
            v5_kernel((c5['n_blocks'],), (256,), (e5, s5, o5, p5, out5, c5['base_exp'], c5['k'], n))
            v8_kernel((nb,), (256,), (e8, s8, o8, w8, out8, c8['base_exp'], c8['k'], n, wpb))
        cp.cuda.Stream.null.synchronize()

        # Benchmark
        tv5 = []
        for _ in range(50):
            start_evt.record()
            v5_kernel((c5['n_blocks'],), (256,), (e5, s5, o5, p5, out5, c5['base_exp'], c5['k'], n))
            end_evt.record()
            end_evt.synchronize()
            tv5.append(cp.cuda.get_elapsed_time(start_evt, end_evt) * 1000)

        tv8 = []
        for _ in range(50):
            start_evt.record()
            v8_kernel((nb,), (256,), (e8, s8, o8, w8, out8, c8['base_exp'], c8['k'], n, wpb))
            end_evt.record()
            end_evt.synchronize()
            tv8.append(cp.cuda.get_elapsed_time(start_evt, end_evt) * 1000)

        a5 = np.mean(sorted(tv5)[:40])
        a8 = np.mean(sorted(tv8)[:40])
        g5 = n / 1e9 / (a5 / 1e6)
        g8 = n / 1e9 / (a8 / 1e6)

        total_n += n
        v5_total += a5
        v8_total += a8

        if n >= 2_000_000:
            print(f"  {name:<40} {n:>10,} {g5:>8.1f} {g8:>8.1f} {a5/a8:>6.2f}x "
                  f"{'Y' if ok_v5 and ok_v8 else 'N':>4}")

    agg5 = total_n / 1e9 / (v5_total / 1e6)
    agg8 = total_n / 1e9 / (v8_total / 1e6)
    print(f"\n  Aggregate v5: {agg5:.1f} GB/s")
    print(f"  Aggregate v8: {agg8:.1f} GB/s  ({agg8/agg5:.2f}x)")
    print(f"  Lossless: {'ALL PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    benchmark()
