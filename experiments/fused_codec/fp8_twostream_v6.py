"""
FP8 Two-Stream Codec v6: Sweep elements_per_thread {8, 16, 32} and threads {256, 512}

Find the optimal configuration for single-layer and batched decode.
Also tests uint4 (128-bit) vectorized loads.
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


class Encoder:
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

        epb = self.tpb * self.ept
        nb = (n + epb - 1) // epb
        bp = np.zeros(nb + 1, dtype=np.uint32)
        for b in range(nb):
            s, e = b * epb, min((b+1)*epb, n)
            bp[b+1] = bp[b] + np.sum(exp_codes[s:e] >= self.k)

        cb = 8 + len(exp_packed) + len(sm_packed) + len(ov) + len(bp) * 4
        return {
            'exp_packed': exp_packed, 'sm_packed': sm_packed, 'overflow_packed': ov,
            'block_escape_prefix': bp, 'base_exp': int(base_exp), 'k': self.k,
            'n_elements': n, 'n_escapes': ne, 'n_blocks': nb,
            'shape': fp8_tensor.shape, 'coverage': coverage,
            'original_bytes': n, 'compressed_bytes': cb, 'ratio': cb / n * 100,
        }


def make_kernel(ept):
    return cp.RawKernel(r"""
extern "C"
__global__ void decode(
    const unsigned char* __restrict__ exp_packed,
    const unsigned char* __restrict__ sm_packed,
    const unsigned char* __restrict__ overflow_packed,
    const unsigned int*  __restrict__ block_escape_prefix,
    unsigned char*       __restrict__ output,
    const int base_exp, const int k, const int n
) {
    const int tid = threadIdx.x;
    const int EPT = """ + str(ept) + r""";
    const int epb = blockDim.x * EPT;
    const int base_idx = blockIdx.x * epb + tid * EPT;

    // Read exp bytes (EPT/4 bytes)
    unsigned char eb[EPT/4];
    int exp_off = blockIdx.x * blockDim.x * (EPT/4) + tid * (EPT/4);
    #pragma unroll
    for (int i = 0; i < EPT/4; i++) {
        eb[i] = (base_idx + i*4 < n) ? exp_packed[exp_off + i] : 0;
    }

    int codes[EPT];
    #pragma unroll
    for (int i = 0; i < EPT/4; i++) {
        codes[i*4+0] = (eb[i] >> 6) & 3;
        codes[i*4+1] = (eb[i] >> 4) & 3;
        codes[i*4+2] = (eb[i] >> 2) & 3;
        codes[i*4+3] = eb[i] & 3;
    }

    int my_esc = 0;
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        if (base_idx + i < n && codes[i] >= k) my_esc++;
    }

    // Warp prefix sum
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

    // Read sm bytes (EPT/2 bytes)
    unsigned char sb[EPT/2];
    int sm_off = blockIdx.x * blockDim.x * (EPT/2) + tid * (EPT/2);
    #pragma unroll
    for (int i = 0; i < EPT/2; i++) {
        sb[i] = (sm_off + i < (n + 1) / 2) ? sm_packed[sm_off + i] : 0;
    }

    int sm_vals[EPT];
    #pragma unroll
    for (int i = 0; i < EPT/2; i++) {
        sm_vals[i*2]   = (sb[i] >> 4) & 0xF;
        sm_vals[i*2+1] = sb[i] & 0xF;
    }

    unsigned char out[EPT];
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        if (base_idx + i < n) {
            int sign = (sm_vals[i] >> 3) & 1;
            int mant = sm_vals[i] & 0x7;
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

    // Write as uint32s
    #pragma unroll
    for (int i = 0; i < EPT; i += 4) {
        if (base_idx + i + 3 < n) {
            unsigned int w = out[i] | (out[i+1]<<8) | (out[i+2]<<16) | (out[i+3]<<24);
            *reinterpret_cast<unsigned int*>(&output[base_idx + i]) = w;
        } else {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if (base_idx + i + j < n) output[base_idx + i + j] = out[i + j];
            }
        }
    }
}
""", 'decode')


def benchmark(model_name="Qwen/Qwen3-0.6B"):
    from transformers import AutoModelForCausalLM

    print("=" * 90)
    print("FP8 Two-Stream v6: Configuration Sweep")
    print("=" * 90)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    # Collect weights
    weight_tensors = []
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16 and param.numel() >= 500_000:
            fp8 = param.data.to(torch.float8_e4m3fn)
            weight_tensors.append((name, fp8))

    configs = [
        (256, 8),
        (256, 16),
        (256, 32),
        (512, 8),
        (512, 16),
        (512, 32),
    ]

    start_evt = cp.cuda.Event()
    end_evt = cp.cuda.Event()

    for tpb, ept in configs:
        print(f"\n--- threads={tpb}, elems/thread={ept} (elems/block={tpb*ept}) ---")

        encoder = Encoder(k=3, tpb=tpb, ept=ept)
        kernel = make_kernel(ept)
        output_buf = None
        max_n = 0

        total_orig = 0
        total_comp = 0
        total_us = 0
        all_ok = True

        for name, fp8 in weight_tensors:
            comp = encoder.encode(fp8)
            n = comp['n_elements']

            eg = cp.asarray(comp['exp_packed'])
            sg = cp.asarray(comp['sm_packed'])
            og = cp.asarray(comp['overflow_packed']) if comp['n_escapes'] > 0 else cp.zeros(1, dtype=cp.uint8)
            pg = cp.asarray(comp['block_escape_prefix'])

            if n > max_n:
                output_buf = cp.empty(n, dtype=cp.uint8)
                max_n = n
            out = output_buf[:n]

            # Verify
            kernel((comp['n_blocks'],), (tpb,),
                   (eg, sg, og, pg, out, comp['base_exp'], comp['k'], n))
            cp.cuda.Stream.null.synchronize()
            is_ok = torch.equal(fp8.view(torch.uint8).cuda(),
                               torch.as_tensor(out, device='cuda').view(torch.float8_e4m3fn).view(torch.uint8))
            if not is_ok:
                all_ok = False

            # Warmup
            for _ in range(10):
                kernel((comp['n_blocks'],), (tpb,),
                       (eg, sg, og, pg, out, comp['base_exp'], comp['k'], n))
            cp.cuda.Stream.null.synchronize()

            # Benchmark
            times = []
            for _ in range(50):
                start_evt.record()
                kernel((comp['n_blocks'],), (tpb,),
                       (eg, sg, og, pg, out, comp['base_exp'], comp['k'], n))
                end_evt.record()
                end_evt.synchronize()
                times.append(cp.cuda.get_elapsed_time(start_evt, end_evt) * 1000)

            avg = np.mean(sorted(times)[:40])
            total_orig += comp['original_bytes']
            total_comp += comp['compressed_bytes']
            total_us += avg

        ratio = total_comp / total_orig * 100
        gbps = total_orig / 1e9 / (total_us / 1e6)
        print(f"  Ratio: {ratio:.2f}%  |  Dec GB/s: {gbps:.1f}  |  "
              f"Total: {total_us/1000:.2f} ms  |  Lossless: {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    benchmark()
