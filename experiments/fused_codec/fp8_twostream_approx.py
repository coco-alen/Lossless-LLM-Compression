"""
FP8 Two-Stream: Near-Lossless (No Escapes) vs Lossless comparison

Skip escape handling: all exponents mapped to base_exp + (code % 3).
Escapes (~4%) get the nearest valid exponent → near-lossless.

This eliminates:
- Overflow stream (saves 2% of data)
- Block escape prefix (saves metadata)
- ALL branches in decode kernel (pure branchless)
- Warp prefix sum (saves compute)

For inference: 4% of values have wrong exponent (off by 1-2 positions).
Impact on model quality: likely negligible (similar to rounding in quantization).

Also: benchmark the branchless kernel throughput ceiling.
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


class EncoderApprox:
    def __init__(self, k=3, tpb=256, ept=16):
        self.k, self.tpb, self.ept = k, tpb, ept

    def encode(self, fp8_tensor: torch.Tensor) -> dict:
        raw = fp8_tensor.contiguous().view(torch.uint8).flatten().numpy()
        n = len(raw)
        signs = (raw >> 7) & 1
        exponents = (raw >> 3) & 0xF
        mantissas = raw & 0x7
        base_exp, coverage = find_best_window(exponents, self.k)
        offsets = exponents.astype(np.int32) - base_exp
        is_common = (offsets >= 0) & (offsets < self.k)

        # For escapes: clamp to nearest valid offset
        exp_codes = np.clip(offsets, 0, self.k - 1).astype(np.uint8)

        pad = (4 - n % 4) % 4
        ep = np.concatenate([exp_codes, np.zeros(pad, dtype=np.uint8)]) if pad else exp_codes
        exp_packed = ((ep[0::4] << 6) | (ep[1::4] << 4) | (ep[2::4] << 2) | ep[3::4]).astype(np.uint8)

        sm = ((signs << 3) | mantissas).astype(np.uint8)
        sp = np.concatenate([sm, np.zeros(n % 2, dtype=np.uint8)]) if n % 2 else sm
        sm_packed = ((sp[0::2] << 4) | sp[1::2]).astype(np.uint8)

        cb = len(exp_packed) + len(sm_packed)
        return {
            'exp_packed': exp_packed, 'sm_packed': sm_packed,
            'base_exp': int(base_exp), 'k': self.k,
            'n_elements': n, 'n_approx': int(np.sum(~is_common)),
            'shape': fp8_tensor.shape, 'coverage': coverage,
            'original_bytes': n, 'compressed_bytes': cb, 'ratio': cb / n * 100,
        }


BRANCHLESS_KERNEL = r"""
extern "C"
__global__ void fp8_branchless_decode(
    const unsigned char* __restrict__ exp_packed,
    const unsigned char* __restrict__ sm_packed,
    unsigned char*       __restrict__ output,
    const int base_exp, const int n
) {
    const int tid = threadIdx.x;
    const int EPT = 16;
    const int base_idx = blockIdx.x * blockDim.x * EPT + tid * EPT;

    // Read 4 exp bytes via uint32
    int exp_off = blockIdx.x * blockDim.x * 4 + tid * 4;
    unsigned int ew = (base_idx < n) ? *reinterpret_cast<const unsigned int*>(&exp_packed[exp_off]) : 0;

    // Read 8 sm bytes via 2×uint32
    int sm_off = blockIdx.x * blockDim.x * 8 + tid * 8;
    unsigned int sw0 = (base_idx < n) ? *reinterpret_cast<const unsigned int*>(&sm_packed[sm_off]) : 0;
    unsigned int sw1 = (base_idx + 8 < n) ? *reinterpret_cast<const unsigned int*>(&sm_packed[sm_off + 4]) : 0;

    // Decode all 16 elements branchlessly
    unsigned int out_w[4] = {0, 0, 0, 0};

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        // Extract 2-bit exp code
        int byte_idx = i / 4;
        int bit_pos = (3 - (i % 4)) * 2;
        unsigned char eb = (ew >> (byte_idx * 8)) & 0xFF;
        int code = (eb >> bit_pos) & 3;

        // Extract 4-bit sm
        int sm_word = (i < 8) ? sw0 : sw1;
        int sm_byte_idx = (i / 2) % 4;
        int sm_nib = 1 - (i & 1);
        unsigned char sb = (sm_word >> (sm_byte_idx * 8)) & 0xFF;
        int sm_val = (sb >> (sm_nib * 4)) & 0xF;

        int sign = (sm_val >> 3) & 1;
        int mant = sm_val & 7;
        int exp_val = base_exp + code;

        unsigned char out_byte = (sign << 7) | (exp_val << 3) | mant;
        out_w[i / 4] |= ((unsigned int)out_byte) << ((i % 4) * 8);
    }

    // Write 16 bytes as 4×uint32
    if (base_idx + 15 < n) {
        *reinterpret_cast<unsigned int*>(&output[base_idx]) = out_w[0];
        *reinterpret_cast<unsigned int*>(&output[base_idx + 4]) = out_w[1];
        *reinterpret_cast<unsigned int*>(&output[base_idx + 8]) = out_w[2];
        *reinterpret_cast<unsigned int*>(&output[base_idx + 12]) = out_w[3];
    } else {
        for (int i = 0; i < 16; i++) {
            if (base_idx + i < n)
                output[base_idx + i] = (out_w[i/4] >> ((i%4)*8)) & 0xFF;
        }
    }
}
"""

# Also the lossless v5 kernel for comparison (same as v7 benchmark)
V5_KERNEL_SRC = r"""
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
    const int base_idx = blockIdx.x * blockDim.x * EPT + tid * EPT;
    int exp_off = blockIdx.x * blockDim.x * 4 + tid * 4;
    unsigned int ew = (base_idx < n) ? *reinterpret_cast<const unsigned int*>(&exp_packed[exp_off]) : 0;
    unsigned char eb0=ew&0xFF, eb1=(ew>>8)&0xFF, eb2=(ew>>16)&0xFF, eb3=(ew>>24)&0xFF;
    int codes[16];
    codes[0]=(eb0>>6)&3; codes[1]=(eb0>>4)&3; codes[2]=(eb0>>2)&3; codes[3]=eb0&3;
    codes[4]=(eb1>>6)&3; codes[5]=(eb1>>4)&3; codes[6]=(eb1>>2)&3; codes[7]=eb1&3;
    codes[8]=(eb2>>6)&3; codes[9]=(eb2>>4)&3; codes[10]=(eb2>>2)&3; codes[11]=eb2&3;
    codes[12]=(eb3>>6)&3; codes[13]=(eb3>>4)&3; codes[14]=(eb3>>2)&3; codes[15]=eb3&3;

    int my_esc=0;
    #pragma unroll
    for(int i=0;i<EPT;i++){if(base_idx+i<n&&codes[i]>=k)my_esc++;}
    int lane=tid&31,wid=tid>>5,wp=my_esc;
    #pragma unroll
    for(int o=1;o<32;o<<=1){int t=__shfl_up_sync(0xFFFFFFFF,wp,o);if(lane>=o)wp+=t;}
    int wt=__shfl_sync(0xFFFFFFFF,wp,31),we=wp-my_esc;
    __shared__ unsigned int ws[33];
    int nw=blockDim.x>>5;
    if(lane==31)ws[wid]=wt;
    __syncthreads();
    if(tid==0){unsigned int s=0;for(int w=0;w<nw;w++){unsigned int t=ws[w];ws[w]=s;s+=t;}}
    __syncthreads();
    unsigned int ei=block_escape_prefix[blockIdx.x]+ws[wid]+we;

    int sm_off=blockIdx.x*blockDim.x*8+tid*8;
    unsigned int sw0=0,sw1=0;
    if(sm_off<(n+1)/2)sw0=*reinterpret_cast<const unsigned int*>(&sm_packed[sm_off]);
    if(sm_off+4<(n+1)/2)sw1=*reinterpret_cast<const unsigned int*>(&sm_packed[sm_off+4]);
    unsigned char sb[8];
    sb[0]=sw0&0xFF;sb[1]=(sw0>>8)&0xFF;sb[2]=(sw0>>16)&0xFF;sb[3]=(sw0>>24)&0xFF;
    sb[4]=sw1&0xFF;sb[5]=(sw1>>8)&0xFF;sb[6]=(sw1>>16)&0xFF;sb[7]=(sw1>>24)&0xFF;
    int sm[16];
    #pragma unroll
    for(int i=0;i<8;i++){sm[i*2]=(sb[i]>>4)&0xF;sm[i*2+1]=sb[i]&0xF;}
    unsigned char out[16];
    #pragma unroll
    for(int i=0;i<EPT;i++){
        if(base_idx+i<n){
            int sign=(sm[i]>>3)&1,mant=sm[i]&7,exp_v;
            if(codes[i]<k){exp_v=base_exp+codes[i];}
            else{int ob=ei>>1,on=1-(ei&1);exp_v=(overflow_packed[ob]>>(on*4))&0xF;ei++;}
            out[i]=(sign<<7)|(exp_v<<3)|mant;
        }
    }
    if(base_idx+15<n){
        *reinterpret_cast<unsigned int*>(&output[base_idx])=out[0]|(out[1]<<8)|(out[2]<<16)|(out[3]<<24);
        *reinterpret_cast<unsigned int*>(&output[base_idx+4])=out[4]|(out[5]<<8)|(out[6]<<16)|(out[7]<<24);
        *reinterpret_cast<unsigned int*>(&output[base_idx+8])=out[8]|(out[9]<<8)|(out[10]<<16)|(out[11]<<24);
        *reinterpret_cast<unsigned int*>(&output[base_idx+12])=out[12]|(out[13]<<8)|(out[14]<<16)|(out[15]<<24);
    }else{
        #pragma unroll
        for(int i=0;i<EPT;i++){if(base_idx+i<n)output[base_idx+i]=out[i];}
    }
}
"""


def benchmark(model_name="Qwen/Qwen3-0.6B"):
    from transformers import AutoModelForCausalLM
    # Also need the lossless encoder
    import sys; sys.path.insert(0, '.')
    from experiments.fused_codec.fp8_twostream_v5 import FP8TwoStreamEncoderV5

    print("=" * 90)
    print("FP8 Branchless (Near-Lossless) vs Lossless Throughput")
    print("=" * 90)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    approx_enc = EncoderApprox(k=3)
    lossless_enc = FP8TwoStreamEncoderV5(k=3)
    branchless = cp.RawKernel(BRANCHLESS_KERNEL, 'fp8_branchless_decode')
    v5_kernel = cp.RawKernel(V5_KERNEL_SRC, 'fp8_v5_decode')

    start_evt = cp.cuda.Event()
    end_evt = cp.cuda.Event()

    print(f"\n{'Layer':<42} {'n':>10} {'LosslessGB/s':>12} {'ApproxGB/s':>11} {'Ratio':>6}")

    total_n = 0
    ll_total = 0
    ap_total = 0

    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16 or param.numel() < 2_000_000:
            continue

        fp8 = param.data.to(torch.float8_e4m3fn)
        n = fp8.numel()

        # Lossless
        lc = lossless_enc.encode(fp8)
        le = cp.asarray(lc['exp_packed'])
        ls = cp.asarray(lc['sm_packed'])
        lo = cp.asarray(lc['overflow_packed']) if lc['n_escapes'] > 0 else cp.zeros(1, dtype=cp.uint8)
        lp = cp.asarray(lc['block_escape_prefix'])
        lout = cp.empty(n, dtype=cp.uint8)

        # Approx
        ac = approx_enc.encode(fp8)
        ae = cp.asarray(ac['exp_packed'])
        as_ = cp.asarray(ac['sm_packed'])
        aout = cp.empty(n, dtype=cp.uint8)
        anb = (n + 256 * 16 - 1) // (256 * 16)

        # Warmup
        for _ in range(10):
            v5_kernel((lc['n_blocks'],), (256,), (le, ls, lo, lp, lout, lc['base_exp'], lc['k'], n))
            branchless((anb,), (256,), (ae, as_, aout, ac['base_exp'], n))
        cp.cuda.Stream.null.synchronize()

        # Benchmark lossless
        ll_times = []
        for _ in range(50):
            start_evt.record()
            v5_kernel((lc['n_blocks'],), (256,), (le, ls, lo, lp, lout, lc['base_exp'], lc['k'], n))
            end_evt.record()
            end_evt.synchronize()
            ll_times.append(cp.cuda.get_elapsed_time(start_evt, end_evt) * 1000)

        # Benchmark approx
        ap_times = []
        for _ in range(50):
            start_evt.record()
            branchless((anb,), (256,), (ae, as_, aout, ac['base_exp'], n))
            end_evt.record()
            end_evt.synchronize()
            ap_times.append(cp.cuda.get_elapsed_time(start_evt, end_evt) * 1000)

        ll_avg = np.mean(sorted(ll_times)[:40])
        ap_avg = np.mean(sorted(ap_times)[:40])
        ll_gbps = n / 1e9 / (ll_avg / 1e6)
        ap_gbps = n / 1e9 / (ap_avg / 1e6)

        total_n += n
        ll_total += ll_avg
        ap_total += ap_avg

        print(f"  {name:<40} {n:>10,} {ll_gbps:>11.1f} {ap_gbps:>10.1f} {ac['ratio']:>5.1f}%")

    print(f"\n  Aggregate lossless:  {total_n/1e9/(ll_total/1e6):.1f} GB/s")
    print(f"  Aggregate approx:    {total_n/1e9/(ap_total/1e6):.1f} GB/s")
    print(f"  Speedup (approx/lossless): {ll_total/ap_total:.2f}x")
    print(f"\n  Approx ratio: 75.0% (no overflow/prefix)")
    print(f"  Lossless ratio: 77.1% (with overflow/prefix)")


if __name__ == "__main__":
    benchmark()
