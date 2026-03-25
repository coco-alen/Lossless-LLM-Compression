"""
Comprehensive benchmark of all FP8 two-stream codec variants.
Runs on idle GPU for accurate results.

Tests:
  1. Huffman baseline (fp8_fused_huffman.py)
  2. TCA-TBE baseline (fp8_tbe.py)
  3. Two-stream v2 (fused prefix sum, 1 elem/thread)
  4. Two-stream v3 (4 elem/thread)
  5. Two-stream v4 (8 elem/thread + prealloc)
  6. Two-stream v5 (16 elem/thread, single-layer + batched)
  7. Branchless near-lossless
  8. memcpy baseline (theoretical ceiling)
"""

import torch
import numpy as np
import cupy as cp
import time
import gc
import sys
sys.path.insert(0, '/home/sky/Lossless-LLM-Compression')


def find_best_window(exponents, k=3):
    counts = np.bincount(exponents, minlength=16)
    best_base, best_cov = 0, 0
    for base in range(16 - k + 1):
        cov = counts[base:base+k].sum()
        if cov > best_cov:
            best_cov = cov
            best_base = base
    return best_base, best_cov / len(exponents)


def encode_twostream(fp8_tensor, k=3, tpb=256, ept=16):
    """Unified encoder for all two-stream variants."""
    raw = fp8_tensor.contiguous().view(torch.uint8).flatten().numpy()
    n = len(raw)
    signs = (raw >> 7) & 1
    exponents = (raw >> 3) & 0xF
    mantissas = raw & 0x7
    base_exp, coverage = find_best_window(exponents, k)
    offsets = exponents.astype(np.int32) - base_exp
    is_common = (offsets >= 0) & (offsets < k)

    exp_codes = np.full(n, k, dtype=np.uint8)
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

    epb = tpb * ept
    nb = (n + epb - 1) // epb
    bp = np.zeros(nb + 1, dtype=np.uint32)
    for b in range(nb):
        s, e = b * epb, min((b + 1) * epb, n)
        bp[b + 1] = bp[b] + np.sum(exp_codes[s:e] >= k)

    # Also compute per-element escape prefix (for v1-style decode)
    esc_flags = (exp_codes >= k).astype(np.uint32)
    esc_prefix = (np.cumsum(esc_flags) - esc_flags).astype(np.uint32)

    cb = 8 + len(exp_packed) + len(sm_packed) + len(ov) + len(bp) * 4
    return {
        'exp_packed': exp_packed, 'sm_packed': sm_packed, 'overflow_packed': ov,
        'block_escape_prefix': bp, 'escape_prefix': esc_prefix,
        'exp_codes': exp_codes,
        'base_exp': int(base_exp), 'k': k,
        'n_elements': n, 'n_escapes': ne, 'n_blocks': nb,
        'shape': fp8_tensor.shape, 'coverage': coverage,
        'original_bytes': n, 'compressed_bytes': cb, 'ratio': cb / n * 100,
    }


# ==================== Kernel Sources ====================

# V2: 1 element per thread, fused block-level prefix sum
V2_KERNEL = r"""
extern "C"
__global__ void v2_decode(
    const unsigned char* __restrict__ exp_packed,
    const unsigned char* __restrict__ sm_packed,
    const unsigned char* __restrict__ overflow_packed,
    const unsigned int*  __restrict__ block_escape_prefix,
    unsigned char*       __restrict__ output,
    const int base_exp, const int k, const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int code = 0, is_esc = 0;
    if (idx < n) {
        int eb = idx >> 2, bp = (3 - (idx & 3)) * 2;
        code = (exp_packed[eb] >> bp) & 0x3;
        is_esc = (code >= k) ? 1 : 0;
    }
    unsigned int wm = __ballot_sync(0xFFFFFFFF, is_esc);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    unsigned int lb = (1u << lane) - 1u;
    int iwp = __popc(wm & lb);
    int wec = __popc(wm);
    __shared__ unsigned int ws[33];
    int nw = blockDim.x >> 5;
    if (lane == 0) ws[wid] = wec;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int s = 0;
        for (int w = 0; w < nw; w++) { unsigned int t = ws[w]; ws[w] = s; s += t; }
    }
    __syncthreads();
    unsigned int ei = block_escape_prefix[blockIdx.x] + ws[wid] + iwp;
    if (idx >= n) return;
    int smbi = idx >> 1, smn = 1 - (idx & 1);
    int sv = (sm_packed[smbi] >> (smn * 4)) & 0xF;
    int sign = (sv >> 3) & 1, mant = sv & 7;
    int exp_v;
    if (code < k) { exp_v = base_exp + code; }
    else { int ob = ei >> 1, on = 1 - (ei & 1); exp_v = (overflow_packed[ob] >> (on * 4)) & 0xF; }
    output[idx] = (sign << 7) | (exp_v << 3) | mant;
}
"""

# V3: 4 elements per thread
V3_KERNEL = r"""
extern "C"
__global__ void v3_decode(
    const unsigned char* __restrict__ exp_packed,
    const unsigned char* __restrict__ sm_packed,
    const unsigned char* __restrict__ overflow_packed,
    const unsigned int*  __restrict__ block_escape_prefix,
    unsigned char*       __restrict__ output,
    const int base_exp, const int k, const int n
) {
    const int tid = threadIdx.x;
    const int EPT = 4;
    const int base_idx = blockIdx.x * blockDim.x * EPT + tid * EPT;
    unsigned char eb = (base_idx < n) ? exp_packed[blockIdx.x * blockDim.x + tid] : 0;
    int c0 = (eb >> 6) & 3, c1 = (eb >> 4) & 3, c2 = (eb >> 2) & 3, c3 = eb & 3;
    int me = 0;
    if (base_idx < n && c0 >= k) me++;
    if (base_idx+1 < n && c1 >= k) me++;
    if (base_idx+2 < n && c2 >= k) me++;
    if (base_idx+3 < n && c3 >= k) me++;
    int lane = tid & 31, wid = tid >> 5, wp = me;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int t = __shfl_up_sync(0xFFFFFFFF, wp, o); if (lane >= o) wp += t; }
    int wt = __shfl_sync(0xFFFFFFFF, wp, 31), we = wp - me;
    __shared__ unsigned int ws[33];
    int nw = blockDim.x >> 5;
    if (lane == 31) ws[wid] = wt;
    __syncthreads();
    if (tid == 0) { unsigned int s = 0; for (int w = 0; w < nw; w++) { unsigned int t = ws[w]; ws[w] = s; s += t; } }
    __syncthreads();
    unsigned int ei = block_escape_prefix[blockIdx.x] + ws[wid] + we;
    int smo = blockIdx.x * blockDim.x * 2 + tid * 2;
    unsigned char sb0 = (smo < (n+1)/2) ? sm_packed[smo] : 0;
    unsigned char sb1 = (smo+1 < (n+1)/2) ? sm_packed[smo+1] : 0;
    int sm0 = (sb0 >> 4) & 0xF, sm1 = sb0 & 0xF, sm2 = (sb1 >> 4) & 0xF, sm3 = sb1 & 0xF;
    int codes[4] = {c0, c1, c2, c3};
    int sms[4] = {sm0, sm1, sm2, sm3};
    unsigned char out[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        if (base_idx + i < n) {
            int s = (sms[i] >> 3) & 1, m = sms[i] & 7, ev;
            if (codes[i] < k) { ev = base_exp + codes[i]; }
            else { int ob = ei >> 1, on = 1 - (ei & 1); ev = (overflow_packed[ob] >> (on * 4)) & 0xF; ei++; }
            out[i] = (s << 7) | (ev << 3) | m;
        }
    }
    if (base_idx + 3 < n) {
        *reinterpret_cast<unsigned int*>(&output[base_idx]) = out[0]|(out[1]<<8)|(out[2]<<16)|(out[3]<<24);
    } else {
        for (int i = 0; i < 4; i++) { if (base_idx + i < n) output[base_idx + i] = out[i]; }
    }
}
"""

# V4: 8 elements per thread
V4_KERNEL = r"""
extern "C"
__global__ void v4_decode(
    const unsigned char* __restrict__ exp_packed,
    const unsigned char* __restrict__ sm_packed,
    const unsigned char* __restrict__ overflow_packed,
    const unsigned int*  __restrict__ block_escape_prefix,
    unsigned char*       __restrict__ output,
    const int base_exp, const int k, const int n
) {
    const int tid = threadIdx.x;
    const int EPT = 8;
    const int base_idx = blockIdx.x * blockDim.x * EPT + tid * EPT;
    int eo = blockIdx.x * blockDim.x * 2 + tid * 2;
    unsigned char e0 = (base_idx < n) ? exp_packed[eo] : 0;
    unsigned char e1 = (base_idx+4 < n) ? exp_packed[eo+1] : 0;
    int codes[8];
    codes[0]=(e0>>6)&3; codes[1]=(e0>>4)&3; codes[2]=(e0>>2)&3; codes[3]=e0&3;
    codes[4]=(e1>>6)&3; codes[5]=(e1>>4)&3; codes[6]=(e1>>2)&3; codes[7]=e1&3;
    int me = 0;
    #pragma unroll
    for (int i = 0; i < EPT; i++) { if (base_idx+i < n && codes[i] >= k) me++; }
    int lane = tid & 31, wid = tid >> 5, wp = me;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int t = __shfl_up_sync(0xFFFFFFFF, wp, o); if (lane >= o) wp += t; }
    int wt = __shfl_sync(0xFFFFFFFF, wp, 31), we = wp - me;
    __shared__ unsigned int ws[33]; int nw = blockDim.x >> 5;
    if (lane == 31) ws[wid] = wt; __syncthreads();
    if (tid == 0) { unsigned int s = 0; for (int w = 0; w < nw; w++) { unsigned int t = ws[w]; ws[w] = s; s += t; } }
    __syncthreads();
    unsigned int ei = block_escape_prefix[blockIdx.x] + ws[wid] + we;
    int so = blockIdx.x * blockDim.x * 4 + tid * 4;
    unsigned char sb[4] = {0,0,0,0};
    if (so < (n+1)/2) sb[0] = sm_packed[so];
    if (so+1 < (n+1)/2) sb[1] = sm_packed[so+1];
    if (so+2 < (n+1)/2) sb[2] = sm_packed[so+2];
    if (so+3 < (n+1)/2) sb[3] = sm_packed[so+3];
    int sm[8];
    for (int i = 0; i < 4; i++) { sm[i*2] = (sb[i] >> 4) & 0xF; sm[i*2+1] = sb[i] & 0xF; }
    unsigned char out[8];
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        if (base_idx+i < n) {
            int s = (sm[i] >> 3) & 1, m = sm[i] & 7, ev;
            if (codes[i] < k) { ev = base_exp + codes[i]; }
            else { int ob = ei >> 1, on = 1-(ei&1); ev = (overflow_packed[ob] >> (on*4)) & 0xF; ei++; }
            out[i] = (s << 7) | (ev << 3) | m;
        }
    }
    if (base_idx+7 < n) {
        *reinterpret_cast<unsigned int*>(&output[base_idx]) = out[0]|(out[1]<<8)|(out[2]<<16)|(out[3]<<24);
        *reinterpret_cast<unsigned int*>(&output[base_idx+4]) = out[4]|(out[5]<<8)|(out[6]<<16)|(out[7]<<24);
    } else {
        for (int i = 0; i < EPT; i++) { if (base_idx+i < n) output[base_idx+i] = out[i]; }
    }
}
"""

# V5: 16 elements per thread with uint32 vectorized loads
V5_KERNEL = r"""
extern "C"
__global__ void v5_decode(
    const unsigned char* __restrict__ exp_packed,
    const unsigned char* __restrict__ sm_packed,
    const unsigned char* __restrict__ overflow_packed,
    const unsigned int*  __restrict__ block_escape_prefix,
    unsigned char*       __restrict__ output,
    const int base_exp, const int k, const int n
) {
    const int tid = threadIdx.x;
    const int EPT = 16;
    const int base_idx = blockIdx.x * blockDim.x * EPT + tid * EPT;

    int eo = blockIdx.x * blockDim.x * 4 + tid * 4;
    unsigned int ew = (base_idx < n) ? *reinterpret_cast<const unsigned int*>(&exp_packed[eo]) : 0;
    unsigned char eb0=ew&0xFF, eb1=(ew>>8)&0xFF, eb2=(ew>>16)&0xFF, eb3=(ew>>24)&0xFF;
    int codes[16];
    codes[0]=(eb0>>6)&3; codes[1]=(eb0>>4)&3; codes[2]=(eb0>>2)&3; codes[3]=eb0&3;
    codes[4]=(eb1>>6)&3; codes[5]=(eb1>>4)&3; codes[6]=(eb1>>2)&3; codes[7]=eb1&3;
    codes[8]=(eb2>>6)&3; codes[9]=(eb2>>4)&3; codes[10]=(eb2>>2)&3; codes[11]=eb2&3;
    codes[12]=(eb3>>6)&3; codes[13]=(eb3>>4)&3; codes[14]=(eb3>>2)&3; codes[15]=eb3&3;

    int me = 0;
    #pragma unroll
    for (int i = 0; i < EPT; i++) { if (base_idx+i < n && codes[i] >= k) me++; }
    int lane = tid & 31, wid = tid >> 5, wp = me;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int t = __shfl_up_sync(0xFFFFFFFF, wp, o); if (lane >= o) wp += t; }
    int wt = __shfl_sync(0xFFFFFFFF, wp, 31), we = wp - me;
    __shared__ unsigned int ws[33]; int nw = blockDim.x >> 5;
    if (lane == 31) ws[wid] = wt; __syncthreads();
    if (tid == 0) { unsigned int s = 0; for (int w = 0; w < nw; w++) { unsigned int t = ws[w]; ws[w] = s; s += t; } }
    __syncthreads();
    unsigned int ei = block_escape_prefix[blockIdx.x] + ws[wid] + we;

    int so = blockIdx.x * blockDim.x * 8 + tid * 8;
    unsigned int sw0 = 0, sw1 = 0;
    if (so < (n+1)/2) sw0 = *reinterpret_cast<const unsigned int*>(&sm_packed[so]);
    if (so+4 < (n+1)/2) sw1 = *reinterpret_cast<const unsigned int*>(&sm_packed[so+4]);
    unsigned char sb[8];
    sb[0]=sw0&0xFF; sb[1]=(sw0>>8)&0xFF; sb[2]=(sw0>>16)&0xFF; sb[3]=(sw0>>24)&0xFF;
    sb[4]=sw1&0xFF; sb[5]=(sw1>>8)&0xFF; sb[6]=(sw1>>16)&0xFF; sb[7]=(sw1>>24)&0xFF;
    int sm[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) { sm[i*2] = (sb[i] >> 4) & 0xF; sm[i*2+1] = sb[i] & 0xF; }

    unsigned char out[16];
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        if (base_idx+i < n) {
            int s = (sm[i] >> 3) & 1, m = sm[i] & 7, ev;
            if (codes[i] < k) { ev = base_exp + codes[i]; }
            else { int ob = ei >> 1, on = 1-(ei&1); ev = (overflow_packed[ob] >> (on*4)) & 0xF; ei++; }
            out[i] = (s << 7) | (ev << 3) | m;
        }
    }
    if (base_idx+15 < n) {
        *reinterpret_cast<unsigned int*>(&output[base_idx])    = out[0]|(out[1]<<8)|(out[2]<<16)|(out[3]<<24);
        *reinterpret_cast<unsigned int*>(&output[base_idx+4])  = out[4]|(out[5]<<8)|(out[6]<<16)|(out[7]<<24);
        *reinterpret_cast<unsigned int*>(&output[base_idx+8])  = out[8]|(out[9]<<8)|(out[10]<<16)|(out[11]<<24);
        *reinterpret_cast<unsigned int*>(&output[base_idx+12]) = out[12]|(out[13]<<8)|(out[14]<<16)|(out[15]<<24);
    } else {
        #pragma unroll
        for (int i = 0; i < EPT; i++) { if (base_idx+i < n) output[base_idx+i] = out[i]; }
    }
}
"""

# Branchless (near-lossless, no escapes)
BRANCHLESS_KERNEL = r"""
extern "C"
__global__ void branchless_decode(
    const unsigned char* __restrict__ exp_packed,
    const unsigned char* __restrict__ sm_packed,
    unsigned char*       __restrict__ output,
    const int base_exp, const int n
) {
    const int tid = threadIdx.x;
    const int EPT = 16;
    const int base_idx = blockIdx.x * blockDim.x * EPT + tid * EPT;
    int eo = blockIdx.x * blockDim.x * 4 + tid * 4;
    unsigned int ew = (base_idx < n) ? *reinterpret_cast<const unsigned int*>(&exp_packed[eo]) : 0;
    int so = blockIdx.x * blockDim.x * 8 + tid * 8;
    unsigned int sw0 = (base_idx < n) ? *reinterpret_cast<const unsigned int*>(&sm_packed[so]) : 0;
    unsigned int sw1 = (base_idx+8 < n) ? *reinterpret_cast<const unsigned int*>(&sm_packed[so+4]) : 0;

    unsigned int ow[4] = {0,0,0,0};
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int bi = i / 4;
        int bp = (3 - (i % 4)) * 2;
        unsigned char eb = (ew >> (bi * 8)) & 0xFF;
        int code = (eb >> bp) & 3;
        unsigned int smw = (i < 8) ? sw0 : sw1;
        int sbi = (i / 2) % 4;
        int sn = 1 - (i & 1);
        unsigned char smb = (smw >> (sbi * 8)) & 0xFF;
        int sv = (smb >> (sn * 4)) & 0xF;
        int s = (sv >> 3) & 1, m = sv & 7;
        int ev = base_exp + code;  // clamp code to 0-2 implicitly
        unsigned char ob = (s << 7) | (ev << 3) | m;
        ow[i / 4] |= ((unsigned int)ob) << ((i % 4) * 8);
    }
    if (base_idx+15 < n) {
        *reinterpret_cast<unsigned int*>(&output[base_idx]) = ow[0];
        *reinterpret_cast<unsigned int*>(&output[base_idx+4]) = ow[1];
        *reinterpret_cast<unsigned int*>(&output[base_idx+8]) = ow[2];
        *reinterpret_cast<unsigned int*>(&output[base_idx+12]) = ow[3];
    } else {
        for (int i = 0; i < 16; i++) {
            if (base_idx+i < n) output[base_idx+i] = (ow[i/4] >> ((i%4)*8)) & 0xFF;
        }
    }
}
"""

# Memcpy kernel (throughput ceiling)
MEMCPY_KERNEL = r"""
extern "C"
__global__ void memcpy_kernel(
    const unsigned char* __restrict__ src,
    unsigned char*       __restrict__ dst,
    const int n
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    if (idx + 15 < n) {
        uint4 val = *reinterpret_cast<const uint4*>(&src[idx]);
        *reinterpret_cast<uint4*>(&dst[idx]) = val;
    } else {
        for (int i = 0; i < 16; i++) {
            if (idx + i < n) dst[idx + i] = src[idx + i];
        }
    }
}
"""


def benchmark_kernel(kernel, args, n, label, start_evt, end_evt, grid, threads=256,
                     n_warmup=20, n_iter=100):
    """Run kernel benchmark and return avg time in us."""
    for _ in range(n_warmup):
        kernel(grid, (threads,), args)
    cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(n_iter):
        start_evt.record()
        kernel(grid, (threads,), args)
        end_evt.record()
        end_evt.synchronize()
        times.append(cp.cuda.get_elapsed_time(start_evt, end_evt) * 1000)  # us

    # Trim top/bottom 10%
    times.sort()
    trim = len(times) // 10
    avg = np.mean(times[trim:-trim]) if trim > 0 else np.mean(times)
    return avg


def main(model_name="Qwen/Qwen3-0.6B"):
    from transformers import AutoModelForCausalLM

    print("=" * 100)
    print("FP8 Two-Stream Codec — Comprehensive Benchmark (Idle GPU)")
    print("=" * 100)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    # Compile all kernels
    k_v2 = cp.RawKernel(V2_KERNEL, 'v2_decode')
    k_v3 = cp.RawKernel(V3_KERNEL, 'v3_decode')
    k_v4 = cp.RawKernel(V4_KERNEL, 'v4_decode')
    k_v5 = cp.RawKernel(V5_KERNEL, 'v5_decode')
    k_bl = cp.RawKernel(BRANCHLESS_KERNEL, 'branchless_decode')
    k_mc = cp.RawKernel(MEMCPY_KERNEL, 'memcpy_kernel')

    start_evt = cp.cuda.Event()
    end_evt = cp.cuda.Event()

    # Collect test layers
    layers = []
    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16 and param.numel() >= 500_000:
            continue
        if param.dtype == torch.bfloat16 and param.numel() >= 500_000:
            fp8 = param.data.to(torch.float8_e4m3fn)
            layers.append((name, fp8))

    # Pre-encode all layers
    print(f"\nEncoding {len(layers)} layers...")
    encoded = []
    total_orig = 0
    total_comp = 0
    for name, fp8 in layers:
        comp = encode_twostream(fp8, k=3, tpb=256, ept=16)
        n = comp['n_elements']

        # GPU arrays
        eg = cp.asarray(comp['exp_packed'])
        sg = cp.asarray(comp['sm_packed'])
        og = cp.asarray(comp['overflow_packed']) if comp['n_escapes'] > 0 else cp.zeros(1, dtype=cp.uint8)
        bp16 = cp.asarray(comp['block_escape_prefix'])

        # Different block prefix for different EPTs
        bpe = {}
        for ept in [1, 4, 8, 16]:
            epb = 256 * ept
            nb = (n + epb - 1) // epb
            bpx = np.zeros(nb + 1, dtype=np.uint32)
            for b in range(nb):
                s, e = b * epb, min((b+1)*epb, n)
                bpx[b+1] = bpx[b] + np.sum(comp['exp_codes'][s:e] >= 3)
            bpe[ept] = (cp.asarray(bpx), nb)

        out_buf = cp.empty(n, dtype=cp.uint8)
        src_buf = cp.asarray(fp8.view(torch.uint8).flatten().numpy())

        total_orig += comp['original_bytes']
        total_comp += comp['compressed_bytes']

        encoded.append({
            'name': name, 'fp8': fp8, 'comp': comp, 'n': n,
            'eg': eg, 'sg': sg, 'og': og, 'bpe': bpe,
            'out': out_buf, 'src': src_buf,
        })

    ratio = total_comp / total_orig * 100
    print(f"Aggregate ratio: {ratio:.2f}%  ({total_orig:,} → {total_comp:,} bytes)")

    # ==================== Per-layer benchmark ====================
    print(f"\n{'='*100}")
    print(f"Per-Layer Throughput (GB/s) — CUDA event timing, 100 iterations")
    print(f"{'='*100}")
    print(f"\n{'Layer':<42} {'n':>10} {'memcpy':>7} {'BL':>7} {'v2':>7} {'v3':>7} {'v4':>7} {'v5':>7} {'OK':>4}")

    agg = {k: 0.0 for k in ['memcpy', 'bl', 'v2', 'v3', 'v4', 'v5']}
    all_ok = True

    for d in encoded:
        n = d['n']
        name = d['name']
        eg, sg, og = d['eg'], d['sg'], d['og']
        out = d['out']
        src = d['src']
        fp8 = d['fp8']
        base_exp = d['comp']['base_exp']
        k = d['comp']['k']

        results = {}

        # memcpy
        nb_mc = (n + 256 * 16 - 1) // (256 * 16)
        t = benchmark_kernel(k_mc, (src, out, n), n, 'memcpy', start_evt, end_evt,
                            (nb_mc,), n_warmup=20, n_iter=100)
        results['memcpy'] = t

        # Branchless
        nb_bl = (n + 256 * 16 - 1) // (256 * 16)
        t = benchmark_kernel(k_bl, (eg, sg, out, base_exp, n), n, 'bl', start_evt, end_evt,
                            (nb_bl,), n_warmup=20, n_iter=100)
        results['bl'] = t

        # V2 (1 elem/thread)
        bp2, nb2 = d['bpe'][1]
        t = benchmark_kernel(k_v2, (eg, sg, og, bp2, out, base_exp, k, n), n, 'v2', start_evt, end_evt,
                            (nb2,), n_warmup=20, n_iter=100)
        results['v2'] = t

        # V3 (4 elem/thread)
        bp3, nb3 = d['bpe'][4]
        t = benchmark_kernel(k_v3, (eg, sg, og, bp3, out, base_exp, k, n), n, 'v3', start_evt, end_evt,
                            (nb3,), n_warmup=20, n_iter=100)
        results['v3'] = t

        # V4 (8 elem/thread)
        bp4, nb4 = d['bpe'][8]
        t = benchmark_kernel(k_v4, (eg, sg, og, bp4, out, base_exp, k, n), n, 'v4', start_evt, end_evt,
                            (nb4,), n_warmup=20, n_iter=100)
        results['v4'] = t

        # V5 (16 elem/thread)
        bp5, nb5 = d['bpe'][16]
        t = benchmark_kernel(k_v5, (eg, sg, og, bp5, out, base_exp, k, n), n, 'v5', start_evt, end_evt,
                            (nb5,), n_warmup=20, n_iter=100)
        results['v5'] = t

        # Verify v5 correctness
        k_v5((nb5,), (256,), (eg, sg, og, bp5, out, base_exp, k, n))
        cp.cuda.Stream.null.synchronize()
        ok = torch.equal(fp8.view(torch.uint8).flatten().cuda(),
                        torch.as_tensor(out, device='cuda').flatten())
        if not ok:
            all_ok = False

        for key, t_us in results.items():
            agg[key] += t_us

        if n >= 2_000_000:
            vals = {k: n/1e9/(v/1e6) for k, v in results.items()}
            print(f"  {name:<40} {n:>10,} {vals['memcpy']:>6.0f} {vals['bl']:>6.0f} "
                  f"{vals['v2']:>6.0f} {vals['v3']:>6.0f} {vals['v4']:>6.0f} {vals['v5']:>6.0f} "
                  f"{'Y' if ok else 'N':>4}")

    print(f"\n  --- Aggregate (all {len(encoded)} layers) ---")
    print(f"  {'Method':<25} {'GB/s':>8} {'Total ms':>10} {'vs memcpy':>10}")
    for key in ['memcpy', 'bl', 'v5', 'v4', 'v3', 'v2']:
        gbps = total_orig / 1e9 / (agg[key] / 1e6)
        ms = agg[key] / 1000
        vs = agg['memcpy'] / agg[key]
        label = {'memcpy': 'memcpy (ceiling)', 'bl': 'Branchless (approx)',
                 'v2': 'v2 (1 elem/th)', 'v3': 'v3 (4 elem/th)',
                 'v4': 'v4 (8 elem/th)', 'v5': 'v5 (16 elem/th)'}[key]
        print(f"  {label:<25} {gbps:>7.1f} {ms:>9.2f} {vs:>9.2f}x")

    print(f"\n  Lossless (v5): {'ALL PASS' if all_ok else 'FAIL'}")
    print(f"  Compression ratio: {ratio:.2f}%")

    # ==================== Batched benchmark (v5 only) ====================
    print(f"\n{'='*100}")
    print(f"Batched Decode (ALL layers in ONE kernel) — v5")
    print(f"{'='*100}")

    from experiments.fused_codec.fp8_twostream_v5 import BATCHED_DECODE_KERNEL
    k_batch = cp.RawKernel(BATCHED_DECODE_KERNEL, 'fp8_twostream_batched_decode')

    # Build batch data
    all_exp = cp.concatenate([d['eg'] for d in encoded])
    all_sm = cp.concatenate([d['sg'] for d in encoded])
    all_ov = cp.concatenate([d['og'] for d in encoded])
    all_pf_list = []
    layer_meta = []
    b2l, b2b = [], []
    out_offs = []
    eo = so = oo = po = outo = 0
    total_blocks = 0

    for i, d in enumerate(encoded):
        c = d['comp']
        n = c['n_elements']
        _, nb = d['bpe'][16]
        layer_meta.extend([eo, so, oo, po, c['base_exp'], n])
        out_offs.append(outo)
        for b in range(nb):
            b2l.append(i)
            b2b.append(b)
        eo += len(c['exp_packed'])
        so += len(c['sm_packed'])
        oo += len(c['overflow_packed'])
        _, nb16 = d['bpe'][16]
        po += len(c['block_escape_prefix'])
        outo += n
        total_blocks += nb
        all_pf_list.append(d['bpe'][16][0])

    all_pf = cp.concatenate(all_pf_list)
    meta_gpu = cp.asarray(np.array(layer_meta, dtype=np.uint32))
    b2l_gpu = cp.asarray(np.array(b2l, dtype=np.uint32))
    b2b_gpu = cp.asarray(np.array(b2b, dtype=np.uint32))
    outoff_gpu = cp.asarray(np.array(out_offs, dtype=np.uint32))
    batch_out = cp.empty(outo, dtype=cp.uint8)

    # Verify
    k_batch((total_blocks,), (256,),
            (all_exp, all_sm, all_ov, all_pf, batch_out,
             meta_gpu, b2l_gpu, b2b_gpu, outoff_gpu, 3))
    cp.cuda.Stream.null.synchronize()

    batch_ok = True
    for i, d in enumerate(encoded):
        n = d['n']
        off = out_offs[i]
        rec = batch_out[off:off+n].get()
        orig = d['fp8'].view(torch.uint8).flatten().numpy()
        if not np.array_equal(orig, rec):
            batch_ok = False
            print(f"  BATCH FAIL: {d['name']}")

    # Benchmark batched
    for _ in range(20):
        k_batch((total_blocks,), (256,),
                (all_exp, all_sm, all_ov, all_pf, batch_out,
                 meta_gpu, b2l_gpu, b2b_gpu, outoff_gpu, 3))
    cp.cuda.Stream.null.synchronize()

    batch_times = []
    for _ in range(100):
        start_evt.record()
        k_batch((total_blocks,), (256,),
                (all_exp, all_sm, all_ov, all_pf, batch_out,
                 meta_gpu, b2l_gpu, b2b_gpu, outoff_gpu, 3))
        end_evt.record()
        end_evt.synchronize()
        batch_times.append(cp.cuda.get_elapsed_time(start_evt, end_evt) * 1000)

    batch_times.sort()
    trim = len(batch_times) // 10
    avg_batch = np.mean(batch_times[trim:-trim])
    batch_gbps = total_orig / 1e9 / (avg_batch / 1e6)

    print(f"\n  Total blocks: {total_blocks}")
    print(f"  Batch decode time: {avg_batch:.1f} us ({avg_batch/1000:.3f} ms)")
    print(f"  Batch throughput: {batch_gbps:.1f} GB/s")
    print(f"  Batch lossless: {'ALL PASS' if batch_ok else 'FAIL'}")

    # ==================== Final summary ====================
    agg_v5_gbps = total_orig / 1e9 / (agg['v5'] / 1e6)
    agg_mc_gbps = total_orig / 1e9 / (agg['memcpy'] / 1e6)

    print(f"\n{'='*100}")
    print(f"FINAL SUMMARY")
    print(f"{'='*100}")
    print(f"  Model: {model_name}")
    print(f"  GPU: NVIDIA H200 (4.8 TB/s HBM)")
    print(f"  Layers: {len(encoded)} (>500K params each)")
    print(f"  Total FP8 bytes: {total_orig:,}")
    print(f"")
    print(f"  Compression ratio: {ratio:.2f}%")
    print(f"  Lossless: {'ALL PASS' if all_ok and batch_ok else 'FAIL'}")
    print(f"")
    print(f"  Throughput (per-layer, aggregate):")
    print(f"    memcpy ceiling:    {agg_mc_gbps:>7.1f} GB/s")
    print(f"    v5 lossless:       {agg_v5_gbps:>7.1f} GB/s  ({agg_v5_gbps/agg_mc_gbps*100:.0f}% of memcpy)")
    print(f"    v5 batched:        {batch_gbps:>7.1f} GB/s  ({batch_gbps/agg_mc_gbps*100:.0f}% of memcpy)")
    print(f"")
    print(f"  External baselines (from prior experiments):")
    print(f"    Huffman (GPU):      5-14 GB/s  @ 77.1%")
    print(f"    nvCOMP ANS (GPU):  29-56 GB/s  @ 85.7%")


if __name__ == "__main__":
    main()
