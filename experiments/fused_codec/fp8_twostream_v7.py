"""
FP8 Two-Stream v7: Interleaved block layout + per-layer adaptive base_exp

Key changes from v5:
- Interleaved layout: each block's exp + sm data is contiguous (better L2 cache)
- Per-layer optimal base_exp (already in v5, but now we also try global shared)
- 16 elements per thread, 256 threads per block
- Also benchmark: global base_exp (shared across all layers, smaller metadata)

Block data layout (per block of 4096 elements = 256 threads × 16 EPT):
  [exp_bytes: 1024B] [sm_bytes: 2048B] [padding: 0B] = 3072 bytes per block
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


class EncoderV7:
    def __init__(self, k=3, tpb=256, ept=16):
        self.k, self.tpb, self.ept = k, tpb, ept
        self.block_exp_bytes = tpb * ept // 4  # 1024
        self.block_sm_bytes = tpb * ept // 2   # 2048
        self.block_data_bytes = self.block_exp_bytes + self.block_sm_bytes  # 3072

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

        # Pack exp codes
        pad = (4 - n % 4) % 4
        ep = np.concatenate([exp_codes, np.zeros(pad, dtype=np.uint8)]) if pad else exp_codes
        exp_packed = ((ep[0::4] << 6) | (ep[1::4] << 4) | (ep[2::4] << 2) | ep[3::4]).astype(np.uint8)

        # Pack sm
        sm = ((signs << 3) | mantissas).astype(np.uint8)
        sp = np.concatenate([sm, np.zeros(n % 2, dtype=np.uint8)]) if n % 2 else sm
        sm_packed = ((sp[0::2] << 4) | sp[1::2]).astype(np.uint8)

        # Overflow
        esc_exp = exponents[~is_common].astype(np.uint8)
        ne = len(esc_exp)
        ep2 = np.concatenate([esc_exp, np.zeros(ne % 2, dtype=np.uint8)]) if ne % 2 else esc_exp
        ov = ((ep2[0::2] << 4) | ep2[1::2]).astype(np.uint8) if len(ep2) else np.array([], dtype=np.uint8)

        # Block prefix
        epb = self.tpb * self.ept
        nb = (n + epb - 1) // epb
        bp = np.zeros(nb + 1, dtype=np.uint32)
        for b in range(nb):
            s, e = b * epb, min((b+1)*epb, n)
            bp[b+1] = bp[b] + np.sum(exp_codes[s:e] >= self.k)

        # Interleaved layout: [block0_exp | block0_sm | block1_exp | block1_sm | ...]
        interleaved = np.zeros(nb * self.block_data_bytes, dtype=np.uint8)
        for b in range(nb):
            exp_start = b * (n // 4 // nb) if nb > 1 else 0
            exp_end = min(exp_start + self.block_exp_bytes, len(exp_packed))
            sm_start = b * (n // 2 // nb) if nb > 1 else 0
            sm_end = min(sm_start + self.block_sm_bytes, len(sm_packed))

            dest = b * self.block_data_bytes
            exp_chunk = exp_packed[b * self.block_exp_bytes:(b+1) * self.block_exp_bytes]
            sm_chunk = sm_packed[b * self.block_sm_bytes:(b+1) * self.block_sm_bytes]

            interleaved[dest:dest+len(exp_chunk)] = exp_chunk
            interleaved[dest+self.block_exp_bytes:dest+self.block_exp_bytes+len(sm_chunk)] = sm_chunk

        cb = 8 + len(interleaved) + len(ov) + len(bp) * 4
        return {
            'interleaved': interleaved,
            'exp_packed': exp_packed, 'sm_packed': sm_packed,  # keep separate for v5 comparison
            'overflow_packed': ov, 'block_escape_prefix': bp,
            'base_exp': int(base_exp), 'k': self.k,
            'n_elements': n, 'n_escapes': ne, 'n_blocks': nb,
            'shape': fp8_tensor.shape, 'coverage': coverage,
            'original_bytes': n, 'compressed_bytes': cb, 'ratio': cb / n * 100,
        }


INTERLEAVED_KERNEL = r"""
extern "C"
__global__ void fp8_interleaved_decode(
    const unsigned char* __restrict__ interleaved,
    const unsigned char* __restrict__ overflow_packed,
    const unsigned int*  __restrict__ block_escape_prefix,
    unsigned char*       __restrict__ output,
    const int base_exp, const int k, const int n,
    const int block_exp_bytes, const int block_data_bytes
) {
    const int tid = threadIdx.x;
    const int EPT = 16;
    const int epb = blockDim.x * EPT;
    const int base_idx = blockIdx.x * epb + tid * EPT;

    // Interleaved layout: this block's data starts at blockIdx.x * block_data_bytes
    const unsigned char* block_data = &interleaved[blockIdx.x * block_data_bytes];
    const unsigned char* exp_data = block_data;                    // first block_exp_bytes
    const unsigned char* sm_data = block_data + block_exp_bytes;   // next block_sm_bytes

    // Read 4 exp bytes (16 × 2-bit codes) using uint32
    int exp_off = tid * 4;
    unsigned int exp_word = 0;
    if (base_idx < n) {
        exp_word = *reinterpret_cast<const unsigned int*>(&exp_data[exp_off]);
    }

    unsigned char eb0 = exp_word & 0xFF, eb1 = (exp_word >> 8) & 0xFF;
    unsigned char eb2 = (exp_word >> 16) & 0xFF, eb3 = (exp_word >> 24) & 0xFF;

    int codes[16];
    codes[0]  = (eb0 >> 6) & 3; codes[1]  = (eb0 >> 4) & 3;
    codes[2]  = (eb0 >> 2) & 3; codes[3]  = eb0 & 3;
    codes[4]  = (eb1 >> 6) & 3; codes[5]  = (eb1 >> 4) & 3;
    codes[6]  = (eb1 >> 2) & 3; codes[7]  = eb1 & 3;
    codes[8]  = (eb2 >> 6) & 3; codes[9]  = (eb2 >> 4) & 3;
    codes[10] = (eb2 >> 2) & 3; codes[11] = eb2 & 3;
    codes[12] = (eb3 >> 6) & 3; codes[13] = (eb3 >> 4) & 3;
    codes[14] = (eb3 >> 2) & 3; codes[15] = eb3 & 3;

    int my_esc = 0;
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        if (base_idx + i < n && codes[i] >= k) my_esc++;
    }

    int lane = tid & 31, wid = tid >> 5;
    int wp = my_esc;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int t = __shfl_up_sync(0xFFFFFFFF, wp, o);
        if (lane >= o) wp += t;
    }
    int wt = __shfl_sync(0xFFFFFFFF, wp, 31);
    int we = wp - my_esc;

    __shared__ unsigned int ws[33];
    int nw = blockDim.x >> 5;
    if (lane == 31) ws[wid] = wt;
    __syncthreads();
    if (tid == 0) {
        unsigned int s = 0;
        for (int w = 0; w < nw; w++) { unsigned int t = ws[w]; ws[w] = s; s += t; }
    }
    __syncthreads();

    unsigned int ei = block_escape_prefix[blockIdx.x] + ws[wid] + we;

    // Read 8 sm bytes (16 × 4-bit) as 2 × uint32
    int sm_off = tid * 8;
    unsigned int sm_w0 = 0, sm_w1 = 0;
    if (base_idx < n) sm_w0 = *reinterpret_cast<const unsigned int*>(&sm_data[sm_off]);
    if (base_idx + 8 < n) sm_w1 = *reinterpret_cast<const unsigned int*>(&sm_data[sm_off + 4]);

    unsigned char sb[8];
    sb[0] = sm_w0 & 0xFF; sb[1] = (sm_w0 >> 8) & 0xFF;
    sb[2] = (sm_w0 >> 16) & 0xFF; sb[3] = (sm_w0 >> 24) & 0xFF;
    sb[4] = sm_w1 & 0xFF; sb[5] = (sm_w1 >> 8) & 0xFF;
    sb[6] = (sm_w1 >> 16) & 0xFF; sb[7] = (sm_w1 >> 24) & 0xFF;

    int sm[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) { sm[i*2] = (sb[i] >> 4) & 0xF; sm[i*2+1] = sb[i] & 0xF; }

    unsigned char out[16];
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        if (base_idx + i < n) {
            int sign = (sm[i] >> 3) & 1;
            int mant = sm[i] & 0x7;
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

    if (base_idx + 15 < n) {
        unsigned int w0 = out[0]|(out[1]<<8)|(out[2]<<16)|(out[3]<<24);
        unsigned int w1 = out[4]|(out[5]<<8)|(out[6]<<16)|(out[7]<<24);
        unsigned int w2 = out[8]|(out[9]<<8)|(out[10]<<16)|(out[11]<<24);
        unsigned int w3 = out[12]|(out[13]<<8)|(out[14]<<16)|(out[15]<<24);
        *reinterpret_cast<unsigned int*>(&output[base_idx]) = w0;
        *reinterpret_cast<unsigned int*>(&output[base_idx+4]) = w1;
        *reinterpret_cast<unsigned int*>(&output[base_idx+8]) = w2;
        *reinterpret_cast<unsigned int*>(&output[base_idx+12]) = w3;
    } else {
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            if (base_idx + i < n) output[base_idx + i] = out[i];
        }
    }
}
"""

# Also keep v5 kernel for comparison
V5_KERNEL = r"""
extern "C"
__global__ void fp8_v5_decode(
    const unsigned char* __restrict__ exp_packed,
    const unsigned char* __restrict__ sm_packed,
    const unsigned char* __restrict__ overflow_packed,
    const unsigned int*  __restrict__ block_escape_prefix,
    unsigned char*       __restrict__ output,
    const int base_exp, const int k, const int n
) {
    const int tid = threadIdx.x;
    const int EPT = 16;
    const int epb = blockDim.x * EPT;
    const int base_idx = blockIdx.x * epb + tid * EPT;

    int exp_off = blockIdx.x * blockDim.x * 4 + tid * 4;
    unsigned int exp_word = 0;
    if (base_idx < n) exp_word = *reinterpret_cast<const unsigned int*>(&exp_packed[exp_off]);

    unsigned char eb0 = exp_word & 0xFF, eb1 = (exp_word >> 8) & 0xFF;
    unsigned char eb2 = (exp_word >> 16) & 0xFF, eb3 = (exp_word >> 24) & 0xFF;

    int codes[16];
    codes[0]  = (eb0 >> 6) & 3; codes[1]  = (eb0 >> 4) & 3;
    codes[2]  = (eb0 >> 2) & 3; codes[3]  = eb0 & 3;
    codes[4]  = (eb1 >> 6) & 3; codes[5]  = (eb1 >> 4) & 3;
    codes[6]  = (eb1 >> 2) & 3; codes[7]  = eb1 & 3;
    codes[8]  = (eb2 >> 6) & 3; codes[9]  = (eb2 >> 4) & 3;
    codes[10] = (eb2 >> 2) & 3; codes[11] = eb2 & 3;
    codes[12] = (eb3 >> 6) & 3; codes[13] = (eb3 >> 4) & 3;
    codes[14] = (eb3 >> 2) & 3; codes[15] = eb3 & 3;

    int my_esc = 0;
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        if (base_idx + i < n && codes[i] >= k) my_esc++;
    }

    int lane = tid & 31, wid = tid >> 5;
    int wp = my_esc;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int t = __shfl_up_sync(0xFFFFFFFF, wp, o);
        if (lane >= o) wp += t;
    }
    int wt = __shfl_sync(0xFFFFFFFF, wp, 31);
    int we = wp - my_esc;

    __shared__ unsigned int ws[33];
    int nw = blockDim.x >> 5;
    if (lane == 31) ws[wid] = wt;
    __syncthreads();
    if (tid == 0) {
        unsigned int s = 0;
        for (int w = 0; w < nw; w++) { unsigned int t = ws[w]; ws[w] = s; s += t; }
    }
    __syncthreads();

    unsigned int ei = block_escape_prefix[blockIdx.x] + ws[wid] + we;

    int sm_off = blockIdx.x * blockDim.x * 8 + tid * 8;
    unsigned int sm_w0 = 0, sm_w1 = 0;
    if (sm_off < (n + 1) / 2) sm_w0 = *reinterpret_cast<const unsigned int*>(&sm_packed[sm_off]);
    if (sm_off + 4 < (n + 1) / 2) sm_w1 = *reinterpret_cast<const unsigned int*>(&sm_packed[sm_off + 4]);

    unsigned char sb[8];
    sb[0] = sm_w0 & 0xFF; sb[1] = (sm_w0 >> 8) & 0xFF;
    sb[2] = (sm_w0 >> 16) & 0xFF; sb[3] = (sm_w0 >> 24) & 0xFF;
    sb[4] = sm_w1 & 0xFF; sb[5] = (sm_w1 >> 8) & 0xFF;
    sb[6] = (sm_w1 >> 16) & 0xFF; sb[7] = (sm_w1 >> 24) & 0xFF;

    int sm[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) { sm[i*2] = (sb[i] >> 4) & 0xF; sm[i*2+1] = sb[i] & 0xF; }

    unsigned char out[16];
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        if (base_idx + i < n) {
            int sign = (sm[i] >> 3) & 1;
            int mant = sm[i] & 0x7;
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

    if (base_idx + 15 < n) {
        unsigned int w0 = out[0]|(out[1]<<8)|(out[2]<<16)|(out[3]<<24);
        unsigned int w1 = out[4]|(out[5]<<8)|(out[6]<<16)|(out[7]<<24);
        unsigned int w2 = out[8]|(out[9]<<8)|(out[10]<<16)|(out[11]<<24);
        unsigned int w3 = out[12]|(out[13]<<8)|(out[14]<<16)|(out[15]<<24);
        *reinterpret_cast<unsigned int*>(&output[base_idx]) = w0;
        *reinterpret_cast<unsigned int*>(&output[base_idx+4]) = w1;
        *reinterpret_cast<unsigned int*>(&output[base_idx+8]) = w2;
        *reinterpret_cast<unsigned int*>(&output[base_idx+12]) = w3;
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

    print("=" * 90)
    print("FP8 Two-Stream v7: Interleaved vs Separate layout comparison")
    print("=" * 90)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    encoder = EncoderV7(k=3, tpb=256, ept=16)
    interleaved_kernel = cp.RawKernel(INTERLEAVED_KERNEL, 'fp8_interleaved_decode')
    v5_kernel = cp.RawKernel(V5_KERNEL, 'fp8_v5_decode')

    start_evt = cp.cuda.Event()
    end_evt = cp.cuda.Event()

    # Test on embed_tokens (large) and a few representative layers
    test_layers = []
    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16 or param.numel() < 500_000:
            continue
        fp8 = param.data.to(torch.float8_e4m3fn)
        test_layers.append((name, fp8))

    print(f"\n{'Layer':<45} {'n':>10} {'v5 GB/s':>9} {'v7 GB/s':>9} {'Speedup':>8} {'OK':>4}")

    total_orig = 0
    v5_total_us = 0
    v7_total_us = 0
    all_ok = True

    for name, fp8 in test_layers:
        comp = encoder.encode(fp8)
        n = comp['n_elements']

        # GPU arrays
        exp_gpu = cp.asarray(comp['exp_packed'])
        sm_gpu = cp.asarray(comp['sm_packed'])
        il_gpu = cp.asarray(comp['interleaved'])
        ov_gpu = cp.asarray(comp['overflow_packed']) if comp['n_escapes'] > 0 else cp.zeros(1, dtype=cp.uint8)
        pf_gpu = cp.asarray(comp['block_escape_prefix'])
        out_v5 = cp.empty(n, dtype=cp.uint8)
        out_v7 = cp.empty(n, dtype=cp.uint8)

        nb = comp['n_blocks']

        # Verify interleaved
        interleaved_kernel(
            (nb,), (256,),
            (il_gpu, ov_gpu, pf_gpu, out_v7, comp['base_exp'], comp['k'], n,
             encoder.block_exp_bytes, encoder.block_data_bytes)
        )
        cp.cuda.Stream.null.synchronize()
        ok_v7 = torch.equal(fp8.view(torch.uint8).cuda(),
                           torch.as_tensor(out_v7, device='cuda').view(torch.float8_e4m3fn).view(torch.uint8))

        # Verify v5
        v5_kernel(
            (nb,), (256,),
            (exp_gpu, sm_gpu, ov_gpu, pf_gpu, out_v5, comp['base_exp'], comp['k'], n)
        )
        cp.cuda.Stream.null.synchronize()
        ok_v5 = torch.equal(fp8.view(torch.uint8).cuda(),
                           torch.as_tensor(out_v5, device='cuda').view(torch.float8_e4m3fn).view(torch.uint8))

        if not (ok_v5 and ok_v7):
            all_ok = False

        # Warmup
        for _ in range(10):
            v5_kernel((nb,), (256,), (exp_gpu, sm_gpu, ov_gpu, pf_gpu, out_v5, comp['base_exp'], comp['k'], n))
            interleaved_kernel((nb,), (256,), (il_gpu, ov_gpu, pf_gpu, out_v7, comp['base_exp'], comp['k'], n, encoder.block_exp_bytes, encoder.block_data_bytes))
        cp.cuda.Stream.null.synchronize()

        # Benchmark v5
        times_v5 = []
        for _ in range(50):
            start_evt.record()
            v5_kernel((nb,), (256,), (exp_gpu, sm_gpu, ov_gpu, pf_gpu, out_v5, comp['base_exp'], comp['k'], n))
            end_evt.record()
            end_evt.synchronize()
            times_v5.append(cp.cuda.get_elapsed_time(start_evt, end_evt) * 1000)

        # Benchmark interleaved
        times_v7 = []
        for _ in range(50):
            start_evt.record()
            interleaved_kernel((nb,), (256,), (il_gpu, ov_gpu, pf_gpu, out_v7, comp['base_exp'], comp['k'], n, encoder.block_exp_bytes, encoder.block_data_bytes))
            end_evt.record()
            end_evt.synchronize()
            times_v7.append(cp.cuda.get_elapsed_time(start_evt, end_evt) * 1000)

        avg_v5 = np.mean(sorted(times_v5)[:40])
        avg_v7 = np.mean(sorted(times_v7)[:40])
        gbps_v5 = n / 1e9 / (avg_v5 / 1e6)
        gbps_v7 = n / 1e9 / (avg_v7 / 1e6)
        speedup = avg_v5 / avg_v7

        total_orig += n
        v5_total_us += avg_v5
        v7_total_us += avg_v7

        if n >= 2_000_000:
            print(f"  {name:<43} {n:>10,} {gbps_v5:>8.1f} {gbps_v7:>8.1f} {speedup:>7.2f}x "
                  f"{'Y' if ok_v5 and ok_v7 else 'N':>4}")

    agg_v5 = total_orig / 1e9 / (v5_total_us / 1e6)
    agg_v7 = total_orig / 1e9 / (v7_total_us / 1e6)
    print(f"\n  Aggregate v5: {agg_v5:.1f} GB/s")
    print(f"  Aggregate v7: {agg_v7:.1f} GB/s  ({agg_v7/agg_v5:.2f}x)")
    print(f"  Lossless: {'ALL PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    benchmark()
