"""
FP8 Two-Stream Decode via Triton

Triton version of the lossless two-stream decode kernel.
Since Triton doesn't support warp-level intrinsics directly,
we use a precomputed per-element escape prefix (from CuPy)
and benchmark just the decode kernel throughput.

Compare: CuPy RawKernel (v5) vs Triton vs memcpy baseline.
"""

import torch
import triton
import triton.language as tl
import numpy as np
import cupy as cp
import time
import sys
sys.path.insert(0, '.')
from experiments.fused_codec.fp8_twostream_v5 import FP8TwoStreamEncoderV5


@triton.jit
def _twostream_decode_kernel(
    exp_packed_ptr, sm_packed_ptr, overflow_ptr, escape_prefix_ptr,
    output_ptr,
    base_exp: tl.constexpr, k: tl.constexpr, n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Read 2-bit exp codes from packed bytes
    exp_byte_idx = offsets >> 2
    exp_bit_pos = (3 - (offsets & 3)) * 2
    exp_bytes = tl.load(exp_packed_ptr + exp_byte_idx, mask=mask, other=0).to(tl.uint8)
    codes = (exp_bytes >> exp_bit_pos.to(tl.uint8)) & 0x3

    # Read 4-bit sm from packed bytes
    sm_byte_idx = offsets >> 1
    sm_nibble = (1 - (offsets & 1)).to(tl.uint8)
    sm_bytes = tl.load(sm_packed_ptr + sm_byte_idx, mask=mask, other=0).to(tl.uint8)
    sm_vals = (sm_bytes >> (sm_nibble * 4)) & 0xF

    sign = (sm_vals >> 3) & 1
    mantissa = sm_vals & 0x7

    # Exponent: common or escape
    is_escape = codes >= k
    common_exp = (base_exp + codes).to(tl.uint8)

    # For escapes, read from overflow
    esc_idx = tl.load(escape_prefix_ptr + offsets, mask=mask & is_escape, other=0)
    ov_byte_idx = esc_idx >> 1
    ov_nibble = (1 - (esc_idx & 1)).to(tl.uint8)
    ov_bytes = tl.load(overflow_ptr + ov_byte_idx, mask=mask & is_escape, other=0).to(tl.uint8)
    escape_exp = (ov_bytes >> (ov_nibble * 4)) & 0xF

    exponent = tl.where(is_escape, escape_exp, common_exp)
    result = (sign << 7) | (exponent << 3) | mantissa

    tl.store(output_ptr + offsets, result.to(tl.uint8), mask=mask)


def triton_decode(exp_packed, sm_packed, overflow, escape_prefix, n, base_exp, k=3, BLOCK_SIZE=1024):
    output = torch.empty(n, dtype=torch.uint8, device='cuda')
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _twostream_decode_kernel[grid](
        exp_packed, sm_packed, overflow, escape_prefix,
        output,
        base_exp=base_exp, k=k, n=n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def benchmark(model_name="Qwen/Qwen3-0.6B"):
    from transformers import AutoModelForCausalLM

    print("=" * 90)
    print("FP8 Triton Decode vs CuPy Kernel Benchmark")
    print("=" * 90)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    encoder = FP8TwoStreamEncoderV5(k=3, elems_per_thread=16)

    # V5 CuPy kernel
    V5_KERNEL = r"""
extern "C" __global__ void v5(
    const unsigned char* __restrict__ ep, const unsigned char* __restrict__ sp,
    const unsigned char* __restrict__ ov, const unsigned int* __restrict__ bp,
    unsigned char* __restrict__ out, const int be, const int k, const int n) {
    const int tid=threadIdx.x; const int EPT=16;
    const int bi=blockIdx.x*blockDim.x*EPT+tid*EPT;
    int eo=blockIdx.x*blockDim.x*4+tid*4;
    unsigned int ew=(bi<n)?*reinterpret_cast<const unsigned int*>(&ep[eo]):0;
    unsigned char e0=ew&0xFF,e1=(ew>>8)&0xFF,e2=(ew>>16)&0xFF,e3=(ew>>24)&0xFF;
    int c[16]; c[0]=(e0>>6)&3;c[1]=(e0>>4)&3;c[2]=(e0>>2)&3;c[3]=e0&3;
    c[4]=(e1>>6)&3;c[5]=(e1>>4)&3;c[6]=(e1>>2)&3;c[7]=e1&3;
    c[8]=(e2>>6)&3;c[9]=(e2>>4)&3;c[10]=(e2>>2)&3;c[11]=e2&3;
    c[12]=(e3>>6)&3;c[13]=(e3>>4)&3;c[14]=(e3>>2)&3;c[15]=e3&3;
    int me=0; for(int i=0;i<EPT;i++){if(bi+i<n&&c[i]>=k)me++;}
    int ln=tid&31,wd=tid>>5,wp=me;
    for(int o=1;o<32;o<<=1){int t=__shfl_up_sync(0xFFFFFFFF,wp,o);if(ln>=o)wp+=t;}
    int wt=__shfl_sync(0xFFFFFFFF,wp,31),we=wp-me;
    __shared__ unsigned int ws[33]; int nw=blockDim.x>>5;
    if(ln==31)ws[wd]=wt; __syncthreads();
    if(tid==0){unsigned int s=0;for(int w=0;w<nw;w++){unsigned int t=ws[w];ws[w]=s;s+=t;}} __syncthreads();
    unsigned int ei=bp[blockIdx.x]+ws[wd]+we;
    int so=blockIdx.x*blockDim.x*8+tid*8;
    unsigned int w0=0,w1=0;
    if(so<(n+1)/2)w0=*reinterpret_cast<const unsigned int*>(&sp[so]);
    if(so+4<(n+1)/2)w1=*reinterpret_cast<const unsigned int*>(&sp[so+4]);
    unsigned char sb[8]; sb[0]=w0&0xFF;sb[1]=(w0>>8)&0xFF;sb[2]=(w0>>16)&0xFF;sb[3]=(w0>>24)&0xFF;
    sb[4]=w1&0xFF;sb[5]=(w1>>8)&0xFF;sb[6]=(w1>>16)&0xFF;sb[7]=(w1>>24)&0xFF;
    int sm[16]; for(int i=0;i<8;i++){sm[i*2]=(sb[i]>>4)&0xF;sm[i*2+1]=sb[i]&0xF;}
    unsigned char o2[16];
    for(int i=0;i<EPT;i++){if(bi+i<n){int s=(sm[i]>>3)&1,m=sm[i]&7,ev;
    if(c[i]<k){ev=be+c[i];}else{int ob=ei>>1,on=1-(ei&1);ev=(ov[ob]>>(on*4))&0xF;ei++;}
    o2[i]=(s<<7)|(ev<<3)|m;}}
    if(bi+15<n){*reinterpret_cast<unsigned int*>(&out[bi])=o2[0]|(o2[1]<<8)|(o2[2]<<16)|(o2[3]<<24);
    *reinterpret_cast<unsigned int*>(&out[bi+4])=o2[4]|(o2[5]<<8)|(o2[6]<<16)|(o2[7]<<24);
    *reinterpret_cast<unsigned int*>(&out[bi+8])=o2[8]|(o2[9]<<8)|(o2[10]<<16)|(o2[11]<<24);
    *reinterpret_cast<unsigned int*>(&out[bi+12])=o2[12]|(o2[13]<<8)|(o2[14]<<16)|(o2[15]<<24);}
    else{for(int i=0;i<EPT;i++){if(bi+i<n)out[bi+i]=o2[i];}}
}"""
    v5_kernel = cp.RawKernel(V5_KERNEL, 'v5')

    start_evt = cp.cuda.Event()
    end_evt = cp.cuda.Event()
    start_t = torch.cuda.Event(enable_timing=True)
    end_t = torch.cuda.Event(enable_timing=True)

    print(f"\n{'Layer':<42} {'n':>10} {'CuPy GB/s':>10} {'Triton GB/s':>12} {'Ratio':>7}")

    total_n = 0
    cupy_total = 0
    triton_total = 0

    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16 or param.numel() < 2_000_000:
            continue

        fp8 = param.data.to(torch.float8_e4m3fn)
        comp = encoder.encode(fp8)
        n = comp['n_elements']

        # CuPy arrays
        exp_cp = cp.asarray(comp['exp_packed'])
        sm_cp = cp.asarray(comp['sm_packed'])
        ov_cp = cp.asarray(comp['overflow_packed']) if comp['n_escapes'] > 0 else cp.zeros(1, dtype=cp.uint8)
        pf_cp = cp.asarray(comp['block_escape_prefix'])
        out_cp = cp.empty(n, dtype=cp.uint8)

        # Torch arrays (for Triton)
        exp_t = torch.as_tensor(exp_cp, device='cuda')
        sm_t = torch.as_tensor(sm_cp, device='cuda')
        ov_t = torch.as_tensor(ov_cp, device='cuda')

        # Precompute escape prefix for Triton (needs per-element uint32)
        exp_codes_cp = cp.asarray(comp['exp_codes'] if 'exp_codes' in comp else
                                   np.zeros(n, dtype=np.uint8))
        # For Triton, compute prefix from exp_codes
        # Actually we need exp_codes for the prefix sum. Let me recompute.
        raw = fp8.view(torch.uint8).flatten().numpy()
        exponents = (raw >> 3) & 0xF
        offsets = exponents.astype(np.int32) - comp['base_exp']
        exp_codes = np.where((offsets >= 0) & (offsets < comp['k']),
                            offsets, comp['k']).astype(np.uint8)
        esc_flags = (exp_codes >= comp['k']).astype(np.uint32)
        esc_prefix = (np.cumsum(esc_flags) - esc_flags).astype(np.uint32)
        esc_prefix_t = torch.from_numpy(esc_prefix).cuda()

        nb = comp['n_blocks']

        # Verify
        out_triton = triton_decode(exp_t, sm_t, ov_t, esc_prefix_t, n, comp['base_exp'], comp['k'])
        ok_triton = torch.equal(fp8.view(torch.uint8).cuda(), out_triton)

        v5_kernel((nb,), (256,), (exp_cp, sm_cp, ov_cp, pf_cp, out_cp, comp['base_exp'], comp['k'], n))
        cp.cuda.Stream.null.synchronize()
        ok_cupy = torch.equal(fp8.view(torch.uint8).cuda(),
                             torch.as_tensor(out_cp, device='cuda'))

        if not ok_triton or not ok_cupy:
            print(f"  {name}: CuPy={'Y' if ok_cupy else 'N'} Triton={'Y' if ok_triton else 'N'}")
            continue

        # Warmup
        for _ in range(10):
            v5_kernel((nb,), (256,), (exp_cp, sm_cp, ov_cp, pf_cp, out_cp, comp['base_exp'], comp['k'], n))
            triton_decode(exp_t, sm_t, ov_t, esc_prefix_t, n, comp['base_exp'], comp['k'])
        cp.cuda.Stream.null.synchronize()
        torch.cuda.synchronize()

        # Benchmark CuPy
        times_cp = []
        for _ in range(50):
            start_evt.record()
            v5_kernel((nb,), (256,), (exp_cp, sm_cp, ov_cp, pf_cp, out_cp, comp['base_exp'], comp['k'], n))
            end_evt.record()
            end_evt.synchronize()
            times_cp.append(cp.cuda.get_elapsed_time(start_evt, end_evt) * 1000)

        # Benchmark Triton
        times_tr = []
        for _ in range(50):
            start_t.record()
            triton_decode(exp_t, sm_t, ov_t, esc_prefix_t, n, comp['base_exp'], comp['k'])
            end_t.record()
            end_t.synchronize()
            times_tr.append(start_t.elapsed_time(end_t) * 1000)

        avg_cp = np.mean(sorted(times_cp)[:40])
        avg_tr = np.mean(sorted(times_tr)[:40])
        gbps_cp = n / 1e9 / (avg_cp / 1e6)
        gbps_tr = n / 1e9 / (avg_tr / 1e6)

        total_n += n
        cupy_total += avg_cp
        triton_total += avg_tr

        print(f"  {name:<40} {n:>10,} {gbps_cp:>9.1f} {gbps_tr:>11.1f} {avg_cp/avg_tr:>6.2f}x")

    print(f"\n  Aggregate CuPy:   {total_n/1e9/(cupy_total/1e6):.1f} GB/s")
    print(f"  Aggregate Triton: {total_n/1e9/(triton_total/1e6):.1f} GB/s")


if __name__ == "__main__":
    benchmark()
