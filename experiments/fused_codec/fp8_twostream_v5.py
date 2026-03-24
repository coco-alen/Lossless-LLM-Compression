"""
FP8 Two-Stream Codec v5: 16 elements/thread + batched multi-layer decode

Key changes from v4:
- 16 elements per thread (4 exp bytes, 8 sm bytes → 16 outputs, written as 4×uint32)
- Batched decode: all layers in a single kernel launch via per-layer metadata
- Further reduces kernel launch overhead for the many small layers
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


class FP8TwoStreamEncoderV5:
    def __init__(self, k=3, threads_per_block=256, elems_per_thread=16):
        self.k = k
        self.tpb = threads_per_block
        self.ept = elems_per_thread

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
        exp_padded = np.concatenate([exp_codes, np.zeros(pad, dtype=np.uint8)]) if pad else exp_codes
        exp_packed = (exp_padded[0::4] << 6) | (exp_padded[1::4] << 4) | \
                     (exp_padded[2::4] << 2) | exp_padded[3::4]
        exp_packed = exp_packed.astype(np.uint8)

        sm = ((signs << 3) | mantissas).astype(np.uint8)
        sm_padded = np.concatenate([sm, np.zeros(n % 2, dtype=np.uint8)]) if n % 2 else sm
        sm_packed = (sm_padded[0::2] << 4) | sm_padded[1::2]
        sm_packed = sm_packed.astype(np.uint8)

        escape_exps = exponents[~is_common].astype(np.uint8)
        n_escapes = len(escape_exps)
        esc_padded = np.concatenate([escape_exps, np.zeros(n_escapes % 2, dtype=np.uint8)]) if n_escapes % 2 else escape_exps
        overflow_packed = ((esc_padded[0::2] << 4) | esc_padded[1::2]).astype(np.uint8) if len(esc_padded) else np.array([], dtype=np.uint8)

        elems_per_block = self.tpb * self.ept
        n_blocks = (n + elems_per_block - 1) // elems_per_block
        block_prefix = np.zeros(n_blocks + 1, dtype=np.uint32)
        for b in range(n_blocks):
            s, e = b * elems_per_block, min((b + 1) * elems_per_block, n)
            block_prefix[b + 1] = block_prefix[b] + np.sum(exp_codes[s:e] >= self.k)

        compressed_bytes = 8 + len(exp_packed) + len(sm_packed) + len(overflow_packed) + len(block_prefix) * 4

        return {
            'exp_packed': exp_packed, 'sm_packed': sm_packed,
            'overflow_packed': overflow_packed, 'block_escape_prefix': block_prefix,
            'base_exp': int(base_exp), 'k': self.k,
            'n_elements': n, 'n_escapes': n_escapes, 'n_blocks': n_blocks,
            'shape': fp8_tensor.shape, 'coverage': coverage,
            'original_bytes': n, 'compressed_bytes': compressed_bytes,
            'ratio': compressed_bytes / n * 100,
        }


FUSED_DECODE_V5 = r"""
extern "C"
__global__ void fp8_twostream_v5_decode(
    const unsigned char* __restrict__ exp_packed,
    const unsigned char* __restrict__ sm_packed,
    const unsigned char* __restrict__ overflow_packed,
    const unsigned int*  __restrict__ block_escape_prefix,
    unsigned char*       __restrict__ output,
    const int base_exp,
    const int k,
    const int n
) {
    const int tid = threadIdx.x;
    const int EPT = 16;
    const int elems_per_block = blockDim.x * EPT;
    const int base_idx = blockIdx.x * elems_per_block + tid * EPT;

    // Read 4 bytes of exp_packed (16 × 2-bit codes)
    int exp_off = blockIdx.x * blockDim.x * 4 + tid * 4;
    unsigned int exp_word = 0;
    if (base_idx < n) {
        // Try to read as uint32 for coalesced access
        if (exp_off + 3 < (n + 3) / 4) {
            exp_word = *reinterpret_cast<const unsigned int*>(&exp_packed[exp_off]);
        } else {
            unsigned char b0 = exp_packed[exp_off];
            unsigned char b1 = (exp_off + 1 < (n + 3) / 4) ? exp_packed[exp_off + 1] : 0;
            unsigned char b2 = (exp_off + 2 < (n + 3) / 4) ? exp_packed[exp_off + 2] : 0;
            unsigned char b3 = (exp_off + 3 < (n + 3) / 4) ? exp_packed[exp_off + 3] : 0;
            exp_word = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
        }
    }

    // Extract 16 codes from 4 bytes (little-endian)
    unsigned char eb0 = exp_word & 0xFF;
    unsigned char eb1 = (exp_word >> 8) & 0xFF;
    unsigned char eb2 = (exp_word >> 16) & 0xFF;
    unsigned char eb3 = (exp_word >> 24) & 0xFF;

    int codes[16];
    codes[0]  = (eb0 >> 6) & 3; codes[1]  = (eb0 >> 4) & 3;
    codes[2]  = (eb0 >> 2) & 3; codes[3]  = eb0 & 3;
    codes[4]  = (eb1 >> 6) & 3; codes[5]  = (eb1 >> 4) & 3;
    codes[6]  = (eb1 >> 2) & 3; codes[7]  = eb1 & 3;
    codes[8]  = (eb2 >> 6) & 3; codes[9]  = (eb2 >> 4) & 3;
    codes[10] = (eb2 >> 2) & 3; codes[11] = eb2 & 3;
    codes[12] = (eb3 >> 6) & 3; codes[13] = (eb3 >> 4) & 3;
    codes[14] = (eb3 >> 2) & 3; codes[15] = eb3 & 3;

    int my_escapes = 0;
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        if (base_idx + i < n && codes[i] >= k) my_escapes++;
    }

    // Warp prefix sum
    int lane_id = tid & 31;
    int warp_id = tid >> 5;
    int warp_prefix = my_escapes;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        int tmp = __shfl_up_sync(0xFFFFFFFF, warp_prefix, off);
        if (lane_id >= off) warp_prefix += tmp;
    }
    int warp_total = __shfl_sync(0xFFFFFFFF, warp_prefix, 31);
    int warp_excl = warp_prefix - my_escapes;

    __shared__ unsigned int warp_sums[33];
    int n_warps = blockDim.x >> 5;
    if (lane_id == 31) warp_sums[warp_id] = warp_total;
    __syncthreads();
    if (tid == 0) {
        unsigned int sum = 0;
        for (int w = 0; w < n_warps; w++) {
            unsigned int t = warp_sums[w]; warp_sums[w] = sum; sum += t;
        }
    }
    __syncthreads();

    unsigned int esc_idx = block_escape_prefix[blockIdx.x] + warp_sums[warp_id] + warp_excl;

    // Read 8 bytes of sm_packed (16 × 4-bit values) as 2 × uint32
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
    for (int i = 0; i < 8; i++) {
        sm[i*2]   = (sb[i] >> 4) & 0xF;
        sm[i*2+1] = sb[i] & 0xF;
    }

    // Decode 16 elements
    unsigned char out[16];
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        if (base_idx + i < n) {
            int sign = (sm[i] >> 3) & 1;
            int mant = sm[i] & 0x7;
            int exp;
            if (codes[i] < k) {
                exp = base_exp + codes[i];
            } else {
                int ob = esc_idx >> 1;
                int on = 1 - (esc_idx & 1);
                exp = (overflow_packed[ob] >> (on * 4)) & 0xF;
                esc_idx++;
            }
            out[i] = (sign << 7) | (exp << 3) | mant;
        }
    }

    // Write 16 bytes as 4 × uint32
    if (base_idx + 15 < n) {
        unsigned int w0 = out[0] | (out[1] << 8) | (out[2] << 16) | (out[3] << 24);
        unsigned int w1 = out[4] | (out[5] << 8) | (out[6] << 16) | (out[7] << 24);
        unsigned int w2 = out[8] | (out[9] << 8) | (out[10] << 16) | (out[11] << 24);
        unsigned int w3 = out[12] | (out[13] << 8) | (out[14] << 16) | (out[15] << 24);
        *reinterpret_cast<unsigned int*>(&output[base_idx]) = w0;
        *reinterpret_cast<unsigned int*>(&output[base_idx + 4]) = w1;
        *reinterpret_cast<unsigned int*>(&output[base_idx + 8]) = w2;
        *reinterpret_cast<unsigned int*>(&output[base_idx + 12]) = w3;
    } else {
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            if (base_idx + i < n) output[base_idx + i] = out[i];
        }
    }
}
"""

# Batched decode: all layers in one kernel launch
BATCHED_DECODE_KERNEL = r"""
extern "C"
__global__ void fp8_twostream_batched_decode(
    const unsigned char* __restrict__ exp_packed,     // concatenated
    const unsigned char* __restrict__ sm_packed,       // concatenated
    const unsigned char* __restrict__ overflow_packed, // concatenated
    const unsigned int*  __restrict__ block_escape_prefix, // concatenated
    unsigned char*       __restrict__ output,          // concatenated
    const unsigned int*  __restrict__ layer_meta,      // [n_layers][6]: exp_off, sm_off, ov_off, prefix_off, base_exp, n_elements
    const unsigned int*  __restrict__ block_to_layer,  // maps global block_id -> layer_id
    const unsigned int*  __restrict__ block_to_local,  // maps global block_id -> local block_id within layer
    const unsigned int*  __restrict__ output_offsets,  // output offset per layer
    const int k
) {
    const int tid = threadIdx.x;
    const int global_block = blockIdx.x;
    const int EPT = 16;

    // Look up which layer and local block this is
    unsigned int layer_id = block_to_layer[global_block];
    unsigned int local_block = block_to_local[global_block];

    unsigned int exp_off_base    = layer_meta[layer_id * 6 + 0];
    unsigned int sm_off_base     = layer_meta[layer_id * 6 + 1];
    unsigned int ov_off_base     = layer_meta[layer_id * 6 + 2];
    unsigned int prefix_off_base = layer_meta[layer_id * 6 + 3];
    int base_exp                 = (int)layer_meta[layer_id * 6 + 4];
    int n                        = (int)layer_meta[layer_id * 6 + 5];
    unsigned int out_off         = output_offsets[layer_id];

    const int elems_per_block = blockDim.x * EPT;
    const int base_idx = local_block * elems_per_block + tid * EPT;

    // Read 4 bytes of exp_packed
    int exp_off = exp_off_base + local_block * blockDim.x * 4 + tid * 4;
    unsigned int exp_word = 0;
    if (base_idx < n) {
        exp_word = *reinterpret_cast<const unsigned int*>(&exp_packed[exp_off]);
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

    int my_escapes = 0;
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        if (base_idx + i < n && codes[i] >= k) my_escapes++;
    }

    int lane_id = tid & 31;
    int warp_id = tid >> 5;
    int warp_prefix = my_escapes;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        int tmp = __shfl_up_sync(0xFFFFFFFF, warp_prefix, off);
        if (lane_id >= off) warp_prefix += tmp;
    }
    int warp_total = __shfl_sync(0xFFFFFFFF, warp_prefix, 31);
    int warp_excl = warp_prefix - my_escapes;

    __shared__ unsigned int warp_sums[33];
    int n_warps = blockDim.x >> 5;
    if (lane_id == 31) warp_sums[warp_id] = warp_total;
    __syncthreads();
    if (tid == 0) {
        unsigned int sum = 0;
        for (int w = 0; w < n_warps; w++) {
            unsigned int t = warp_sums[w]; warp_sums[w] = sum; sum += t;
        }
    }
    __syncthreads();

    unsigned int esc_idx = block_escape_prefix[prefix_off_base + local_block]
                           + warp_sums[warp_id] + warp_excl;

    // Read sm_packed
    int sm_off = sm_off_base + local_block * blockDim.x * 8 + tid * 8;
    unsigned int sm_w0 = 0, sm_w1 = 0;
    if (base_idx < n) sm_w0 = *reinterpret_cast<const unsigned int*>(&sm_packed[sm_off]);
    if (base_idx + 8 < n) sm_w1 = *reinterpret_cast<const unsigned int*>(&sm_packed[sm_off + 4]);

    unsigned char sb[8];
    sb[0] = sm_w0 & 0xFF; sb[1] = (sm_w0 >> 8) & 0xFF;
    sb[2] = (sm_w0 >> 16) & 0xFF; sb[3] = (sm_w0 >> 24) & 0xFF;
    sb[4] = sm_w1 & 0xFF; sb[5] = (sm_w1 >> 8) & 0xFF;
    sb[6] = (sm_w1 >> 16) & 0xFF; sb[7] = (sm_w1 >> 24) & 0xFF;

    int sm[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) { sm[i*2] = (sb[i] >> 4) & 0xF; sm[i*2+1] = sb[i] & 0xF; }

    // Overflow pointer for this layer
    const unsigned char* ov_ptr = &overflow_packed[ov_off_base];

    unsigned char out_arr[16];
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        if (base_idx + i < n) {
            int sign = (sm[i] >> 3) & 1;
            int mant = sm[i] & 0x7;
            int exp_val;
            if (codes[i] < k) {
                exp_val = base_exp + codes[i];
            } else {
                int ob = esc_idx >> 1;
                int on = 1 - (esc_idx & 1);
                exp_val = (ov_ptr[ob] >> (on * 4)) & 0xF;
                esc_idx++;
            }
            out_arr[i] = (sign << 7) | (exp_val << 3) | mant;
        }
    }

    unsigned int out_base = out_off + base_idx;
    if (base_idx + 15 < n) {
        unsigned int w0 = out_arr[0] | (out_arr[1]<<8) | (out_arr[2]<<16) | (out_arr[3]<<24);
        unsigned int w1 = out_arr[4] | (out_arr[5]<<8) | (out_arr[6]<<16) | (out_arr[7]<<24);
        unsigned int w2 = out_arr[8] | (out_arr[9]<<8) | (out_arr[10]<<16) | (out_arr[11]<<24);
        unsigned int w3 = out_arr[12] | (out_arr[13]<<8) | (out_arr[14]<<16) | (out_arr[15]<<24);
        *reinterpret_cast<unsigned int*>(&output[out_base]) = w0;
        *reinterpret_cast<unsigned int*>(&output[out_base+4]) = w1;
        *reinterpret_cast<unsigned int*>(&output[out_base+8]) = w2;
        *reinterpret_cast<unsigned int*>(&output[out_base+12]) = w3;
    } else {
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            if (base_idx + i < n) output[out_base + i] = out_arr[i];
        }
    }
}
"""


class FP8TwoStreamDecoderV5:
    def __init__(self):
        self.single_kernel = cp.RawKernel(FUSED_DECODE_V5, 'fp8_twostream_v5_decode')
        self.batch_kernel = cp.RawKernel(BATCHED_DECODE_KERNEL, 'fp8_twostream_batched_decode')
        self._output_buf = None
        self._max_n = 0

    def _get_output(self, n):
        if n > self._max_n:
            self._output_buf = cp.empty(n, dtype=cp.uint8)
            self._max_n = n
        return self._output_buf[:n]

    def decode_single(self, n, shape, base_exp, k, n_blocks,
                      exp_gpu, sm_gpu, overflow_gpu, prefix_gpu, threads=256):
        output = self._get_output(n)
        self.single_kernel(
            (n_blocks,), (threads,),
            (exp_gpu, sm_gpu, overflow_gpu, prefix_gpu, output, base_exp, k, n)
        )
        return torch.as_tensor(output, device='cuda').view(torch.float8_e4m3fn).reshape(shape)

    def decode_batch(self, layers_data, threads=256):
        """Decode all layers in a single kernel launch."""
        # Concatenate all compressed data
        all_exp = []
        all_sm = []
        all_overflow = []
        all_prefix = []
        layer_meta = []
        block_to_layer = []
        block_to_local = []
        output_offsets = []

        exp_offset = 0
        sm_offset = 0
        ov_offset = 0
        prefix_offset = 0
        out_offset = 0
        total_blocks = 0

        for i, (comp, exp_gpu, sm_gpu, overflow_gpu, prefix_gpu) in enumerate(layers_data):
            n = comp['n_elements']
            n_blocks = comp['n_blocks']

            layer_meta.extend([exp_offset, sm_offset, ov_offset, prefix_offset,
                             comp['base_exp'], n])
            output_offsets.append(out_offset)

            for b in range(n_blocks):
                block_to_layer.append(i)
                block_to_local.append(b)

            exp_offset += len(comp['exp_packed'])
            sm_offset += len(comp['sm_packed'])
            ov_offset += len(comp['overflow_packed'])
            prefix_offset += len(comp['block_escape_prefix'])
            out_offset += n
            total_blocks += n_blocks

            all_exp.append(exp_gpu)
            all_sm.append(sm_gpu)
            all_overflow.append(overflow_gpu)
            all_prefix.append(prefix_gpu)

        # Concatenate on GPU
        cat_exp = cp.concatenate(all_exp)
        cat_sm = cp.concatenate(all_sm)
        cat_overflow = cp.concatenate(all_overflow) if all_overflow else cp.zeros(1, dtype=cp.uint8)
        cat_prefix = cp.concatenate(all_prefix)
        output = cp.empty(out_offset, dtype=cp.uint8)

        meta_gpu = cp.asarray(np.array(layer_meta, dtype=np.uint32))
        b2l_gpu = cp.asarray(np.array(block_to_layer, dtype=np.uint32))
        b2b_gpu = cp.asarray(np.array(block_to_local, dtype=np.uint32))
        out_off_gpu = cp.asarray(np.array(output_offsets, dtype=np.uint32))

        self.batch_kernel(
            (total_blocks,), (threads,),
            (cat_exp, cat_sm, cat_overflow, cat_prefix, output,
             meta_gpu, b2l_gpu, b2b_gpu, out_off_gpu, 3)
        )

        return output, output_offsets, layers_data


def benchmark(model_name="Qwen/Qwen3-0.6B"):
    from transformers import AutoModelForCausalLM

    print("=" * 90)
    print("FP8 Two-Stream v5 (16 elems/thread + batched) Benchmark")
    print("=" * 90)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    encoder = FP8TwoStreamEncoderV5(k=3, elems_per_thread=16)
    decoder = FP8TwoStreamDecoderV5()

    # ============= Per-layer benchmark (single kernel per layer) =============
    print(f"\n--- Per-Layer (single kernel) ---")
    print(f"{'Layer':<50} {'Params':>10} {'Ratio':>7} {'Dec us':>8} {'Dec GB/s':>9} {'OK':>4}")

    total_orig = 0
    total_comp = 0
    all_ok = True
    layers = []

    for name, param in model.named_parameters():
        if param.dtype != torch.bfloat16 or param.numel() < 500_000:
            continue
        fp8 = param.data.to(torch.float8_e4m3fn)
        comp = encoder.encode(fp8)
        exp_gpu = cp.asarray(comp['exp_packed'])
        sm_gpu = cp.asarray(comp['sm_packed'])
        overflow_gpu = cp.asarray(comp['overflow_packed']) if comp['n_escapes'] > 0 else cp.zeros(1, dtype=cp.uint8)
        prefix_gpu = cp.asarray(comp['block_escape_prefix'])
        layers.append((name, fp8, comp, exp_gpu, sm_gpu, overflow_gpu, prefix_gpu))

    # Verify and benchmark
    start_evt = cp.cuda.Event()
    end_evt = cp.cuda.Event()
    total_time_us = 0

    for name, fp8, comp, exp_gpu, sm_gpu, overflow_gpu, prefix_gpu in layers:
        n = comp['n_elements']
        recovered = decoder.decode_single(n, comp['shape'], comp['base_exp'], comp['k'],
                                          comp['n_blocks'], exp_gpu, sm_gpu, overflow_gpu, prefix_gpu)
        cp.cuda.Stream.null.synchronize()
        is_ok = torch.equal(fp8.view(torch.uint8).cuda(), recovered.view(torch.uint8))
        if not is_ok:
            all_ok = False

        for _ in range(10):
            decoder.decode_single(n, comp['shape'], comp['base_exp'], comp['k'],
                                 comp['n_blocks'], exp_gpu, sm_gpu, overflow_gpu, prefix_gpu)
        cp.cuda.Stream.null.synchronize()

        times_us = []
        for _ in range(50):
            start_evt.record()
            decoder.decode_single(n, comp['shape'], comp['base_exp'], comp['k'],
                                 comp['n_blocks'], exp_gpu, sm_gpu, overflow_gpu, prefix_gpu)
            end_evt.record()
            end_evt.synchronize()
            times_us.append(cp.cuda.get_elapsed_time(start_evt, end_evt) * 1000)

        avg_us = np.mean(sorted(times_us)[:40])
        throughput = n / 1e9 / (avg_us / 1e6)
        total_orig += comp['original_bytes']
        total_comp += comp['compressed_bytes']
        total_time_us += avg_us

        if n >= 2_000_000:
            print(f"  {name:<48} {n:>10,} {comp['ratio']:>6.1f}% {avg_us:>7.1f} {throughput:>8.1f} {'Y' if is_ok else 'N':>4}")

    overall = total_comp / total_orig * 100
    agg = total_orig / 1e9 / (total_time_us / 1e6)
    print(f"\n  Aggregate ratio: {overall:.2f}%")
    print(f"  Aggregate Dec GB/s (single): {agg:.1f}")
    print(f"  Total decode time: {total_time_us/1000:.2f} ms ({len(layers)} layers)")
    print(f"  Lossless: {'ALL PASS' if all_ok else 'FAIL'}")

    # ============= Batched benchmark =============
    print(f"\n--- Batched (single kernel for ALL layers) ---")
    batch_data = [(comp, exp_gpu, sm_gpu, overflow_gpu, prefix_gpu)
                  for _, _, comp, exp_gpu, sm_gpu, overflow_gpu, prefix_gpu in layers]

    # Precompute concatenated data (do this once, amortized)
    all_exp = cp.concatenate([d[1] for d in batch_data])
    all_sm = cp.concatenate([d[2] for d in batch_data])
    all_ov_list = [d[3] for d in batch_data]
    all_ov = cp.concatenate(all_ov_list)
    all_pf = cp.concatenate([d[4] for d in batch_data])

    layer_meta = []
    block_to_layer_list = []
    block_to_local_list = []
    output_offsets_list = []
    exp_off = sm_off = ov_off = pf_off = out_off = 0
    total_blocks = 0

    for i, (comp, _, _, _, _) in enumerate(batch_data):
        n = comp['n_elements']
        nb = comp['n_blocks']
        layer_meta.extend([exp_off, sm_off, ov_off, pf_off, comp['base_exp'], n])
        output_offsets_list.append(out_off)
        for b in range(nb):
            block_to_layer_list.append(i)
            block_to_local_list.append(b)
        exp_off += len(comp['exp_packed'])
        sm_off += len(comp['sm_packed'])
        ov_off += len(comp['overflow_packed'])
        pf_off += len(comp['block_escape_prefix'])
        out_off += n
        total_blocks += nb

    meta_gpu = cp.asarray(np.array(layer_meta, dtype=np.uint32))
    b2l_gpu = cp.asarray(np.array(block_to_layer_list, dtype=np.uint32))
    b2b_gpu = cp.asarray(np.array(block_to_local_list, dtype=np.uint32))
    outoff_gpu = cp.asarray(np.array(output_offsets_list, dtype=np.uint32))
    batch_output = cp.empty(out_off, dtype=cp.uint8)

    # Verify batched
    decoder.batch_kernel(
        (total_blocks,), (256,),
        (all_exp, all_sm, all_ov, all_pf, batch_output,
         meta_gpu, b2l_gpu, b2b_gpu, outoff_gpu, 3)
    )
    cp.cuda.Stream.null.synchronize()

    batch_ok = True
    for i, (name, fp8, comp, _, _, _, _) in enumerate(layers):
        n = comp['n_elements']
        off = output_offsets_list[i]
        recovered = batch_output[off:off+n].get()
        original = fp8.view(torch.uint8).flatten().numpy()
        if not np.array_equal(original, recovered):
            batch_ok = False
            print(f"  BATCH FAIL: {name}")

    # Warmup batched
    for _ in range(10):
        decoder.batch_kernel(
            (total_blocks,), (256,),
            (all_exp, all_sm, all_ov, all_pf, batch_output,
             meta_gpu, b2l_gpu, b2b_gpu, outoff_gpu, 3)
        )
    cp.cuda.Stream.null.synchronize()

    # Benchmark batched
    batch_times = []
    for _ in range(50):
        start_evt.record()
        decoder.batch_kernel(
            (total_blocks,), (256,),
            (all_exp, all_sm, all_ov, all_pf, batch_output,
             meta_gpu, b2l_gpu, b2b_gpu, outoff_gpu, 3)
        )
        end_evt.record()
        end_evt.synchronize()
        batch_times.append(cp.cuda.get_elapsed_time(start_evt, end_evt) * 1000)

    avg_batch_us = np.mean(sorted(batch_times)[:40])
    batch_gbps = total_orig / 1e9 / (avg_batch_us / 1e6)

    print(f"\n  Batched decode time: {avg_batch_us:.1f} us ({total_blocks} blocks)")
    print(f"  Batched Dec GB/s: {batch_gbps:.1f}")
    print(f"  Batched Lossless: {'ALL PASS' if batch_ok else 'FAIL'}")

    print(f"\n  Comparison:")
    print(f"    Dense FP8:              100.0%  (baseline)")
    print(f"    Our v5 single:          {overall:.1f}%  @ {agg:.0f} GB/s")
    print(f"    Our v5 batched:         {overall:.1f}%  @ {batch_gbps:.0f} GB/s")
    print(f"    Our v4 (8 elem/th):      77.2%  @ 165 GB/s")
    print(f"    nvCOMP ANS:              85.7%  @ 29-56 GB/s")
    print(f"    Entropy limit:           70.4%")


if __name__ == "__main__":
    benchmark()
