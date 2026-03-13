"""
Experiment 30: GPU Block-Parallel Huffman for byte3 via CuPy

byte3 has 3.7 bits entropy. Fixed 6-bit: 6.25% savings. Huffman: ~13.5% savings.

Approach:
- Build Huffman tree on CPU (fast, ≤256 symbols)
- GPU encode: prefix sum of code lengths per block, scatter bits
- GPU decode: each block starts at known bit offset, sequential table decode
- Block size: 256 values per thread block

Expected: ~3.7 bits/value → ~46% ratio on byte3
Total savings: ~13.5% = ~614 MB (from 4548 MB)
"""

import torch
import time
import gc
import math
import heapq
import numpy as np
from transformers import AutoModelForCausalLM

import cupy as cp


# ============ Huffman Tree Building (CPU) ============

def build_huffman_codes(freq_np):
    """Build canonical Huffman codes from frequency array[256].
    Returns dict: symbol → (code, length).
    """
    # Build tree
    heap = []
    for sym in range(256):
        if freq_np[sym] > 0:
            heapq.heappush(heap, (freq_np[sym], sym, sym))

    if len(heap) <= 1:
        sym = heap[0][2] if heap else 0
        return {sym: (0, 1)}

    nodes = {}
    next_id = 256
    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        nodes[next_id] = (n1, n2)
        heapq.heappush(heap, (f1 + f2, next_id, next_id))
        next_id += 1

    codes = {}
    root = heap[0][2]
    stack = [(root, 0, 0)]
    while stack:
        node, code, length = stack.pop()
        if node < 256:
            codes[node] = (code, max(length, 1))
        else:
            left, right = nodes[node]
            stack.append((left, code << 1, length + 1))
            stack.append((right, (code << 1) | 1, length + 1))

    return codes


def reverse_bits(code, length):
    """Reverse bit order of a code."""
    result = 0
    for i in range(length):
        result |= ((code >> i) & 1) << (length - 1 - i)
    return result


def build_decode_lut(codes, max_len):
    """Build decode LUT for LSB-first bit ordering.
    Index by max_len bits read from bitstream (LSB-first) → (symbol, code_length).
    """
    lut_size = 1 << max_len
    dec_sym = np.zeros(lut_size, dtype=np.uint8)
    dec_len = np.zeros(lut_size, dtype=np.uint8)

    for sym, (code, length) in codes.items():
        # Reverse code bits for LSB-first storage
        rev_code = reverse_bits(code, length)
        padding = max_len - length
        # rev_code occupies low `length` bits, padding in high bits
        for i in range(1 << padding):
            idx = rev_code | (i << length)
            dec_sym[idx] = sym
            dec_len[idx] = length

    return dec_sym, dec_len


# ============ CUDA Kernels ============

# Encode kernel: each thread processes one value
# Phase 1: compute per-value code lengths, do block-level prefix sum
# Phase 2: scatter bits into output

ENCODE_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void huffman_encode(
    const unsigned char* __restrict__ byte3,    // [N] input byte3 values
    const unsigned int* __restrict__ enc_codes,  // [256] huffman codes
    const unsigned char* __restrict__ enc_lens,  // [256] code lengths
    unsigned char* __restrict__ output,          // [out_bytes] packed output
    unsigned int* __restrict__ block_bit_offsets,// [n_blocks+1] cumulative bit offsets
    const int N,
    const int BLOCK_SIZE
) {
    int block_id = blockIdx.x;
    int tid = threadIdx.x;
    int global_id = block_id * BLOCK_SIZE + tid;

    // Shared memory for block prefix sum
    extern __shared__ unsigned int smem[];  // [BLOCK_SIZE]

    unsigned int my_len = 0;
    unsigned int my_code = 0;
    unsigned char my_sym = 0;

    if (global_id < N) {
        my_sym = byte3[global_id];
        my_code = enc_codes[my_sym];
        my_len = enc_lens[my_sym];
    }

    // Inclusive prefix sum of code lengths within block
    smem[tid] = my_len;
    __syncthreads();

    // Simple parallel prefix sum (works for BLOCK_SIZE <= 1024)
    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        unsigned int val = 0;
        if (tid >= stride) {
            val = smem[tid - stride];
        }
        __syncthreads();
        smem[tid] += val;
        __syncthreads();
    }

    // smem[tid] = inclusive prefix sum of lengths
    unsigned int local_bit_end = smem[tid];
    unsigned int local_bit_start = local_bit_end - my_len;

    // Store block total for phase 2
    if (tid == BLOCK_SIZE - 1) {
        block_bit_offsets[block_id + 1] = local_bit_end;
    }

    // Wait for block_bit_offsets to be populated (done in separate kernel)
    // For now, we'll compute global offsets in a second pass
}
''', 'huffman_encode')


# Two-phase approach:
# Phase 1: compute block totals
# Phase 2: prefix sum of block totals (on CPU, it's small)
# Phase 3: scatter bits using global offsets

COMPUTE_BLOCK_TOTALS = cp.RawKernel(r'''
extern "C" __global__
void compute_block_totals(
    const unsigned char* __restrict__ byte3,
    const unsigned char* __restrict__ enc_lens,
    unsigned int* __restrict__ block_totals,  // [n_blocks]
    const int N,
    const int BLOCK_SIZE
) {
    int block_id = blockIdx.x;
    int tid = threadIdx.x;
    int global_id = block_id * BLOCK_SIZE + tid;

    extern __shared__ unsigned int smem[];

    unsigned int my_len = 0;
    if (global_id < N) {
        my_len = enc_lens[byte3[global_id]];
    }

    smem[tid] = my_len;
    __syncthreads();

    // Parallel reduction for block total
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_totals[block_id] = smem[0];
    }
}
''', 'compute_block_totals')


SCATTER_BITS = cp.RawKernel(r'''
extern "C" __global__
void scatter_bits(
    const unsigned char* __restrict__ byte3,
    const unsigned int* __restrict__ enc_codes,
    const unsigned char* __restrict__ enc_lens,
    unsigned char* __restrict__ output,
    const unsigned long long* __restrict__ block_offsets,
    const int N,
    const int BLOCK_SIZE
) {
    // One thread per block, sequential encode (simple & correct)
    int block_id = blockIdx.x;
    if (threadIdx.x != 0) return;

    int start = block_id * BLOCK_SIZE;
    int end = min(start + BLOCK_SIZE, N);
    unsigned long long bit_pos = block_offsets[block_id];

    for (int i = start; i < end; i++) {
        unsigned char sym = byte3[i];
        unsigned int code = enc_codes[sym];
        unsigned int len = enc_lens[sym];

        // Write bits MSB-first
        for (unsigned int b = 0; b < len; b++) {
            unsigned int bit = (code >> (len - 1 - b)) & 1;
            if (bit) {
                unsigned long long abs_bit = bit_pos + b;
                unsigned long long byte_idx = abs_bit / 8;
                unsigned int bit_idx = (unsigned int)(abs_bit % 8);
                atomicOr((unsigned int*)(output + (byte_idx & ~3ull)),
                         1u << (8u * (unsigned int)(byte_idx & 3ull) + bit_idx));
            }
        }
        bit_pos += len;
    }
}
''', 'scatter_bits')


DECODE_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void huffman_decode(
    const unsigned char* __restrict__ input,     // packed bitstream
    unsigned char* __restrict__ output,           // [N] decoded byte3
    const unsigned char* __restrict__ dec_sym,    // [1<<max_len] decode LUT symbols
    const unsigned char* __restrict__ dec_len,    // [1<<max_len] decode LUT lengths
    const unsigned long long* __restrict__ block_offsets, // [n_blocks] bit offsets (uint64)
    const int N,
    const int BLOCK_SIZE,
    const int max_len,
    const long long total_bits
) {
    int block_id = blockIdx.x;
    int start_val = block_id * BLOCK_SIZE;
    int end_val = min(start_val + BLOCK_SIZE, N);

    unsigned long long bit_pos = block_offsets[block_id];
    unsigned int mask = (1u << max_len) - 1;
    unsigned long long total_bytes = (total_bits + 7) / 8;

    for (int i = start_val; i < end_val; i++) {
        unsigned long long byte_idx = bit_pos / 8;
        unsigned int bit_offset = (unsigned int)(bit_pos % 8);

        // Read 4 bytes (handles max_len up to 25 with any bit_offset)
        unsigned int val = 0;
        val = input[byte_idx];
        if (byte_idx + 1 < total_bytes) val |= ((unsigned int)input[byte_idx + 1]) << 8;
        if (byte_idx + 2 < total_bytes) val |= ((unsigned int)input[byte_idx + 2]) << 16;
        if (byte_idx + 3 < total_bytes) val |= ((unsigned int)input[byte_idx + 3]) << 24;

        unsigned int window = (val >> bit_offset) & mask;

        output[i] = dec_sym[window];
        bit_pos += dec_len[window];
    }
}
''', 'huffman_decode')


# ============ High-level compress/decompress ============

def huffman_compress_byte3(byte3_gpu, device):
    """Compress byte3 tensor on GPU using block-parallel Huffman.
    Returns (packed_data, block_offsets, codes_info, total_bits).
    """
    n = byte3_gpu.numel()
    BLOCK_SIZE = 256

    # Build Huffman codes on CPU from GPU histogram
    counts = torch.bincount(byte3_gpu.to(torch.int32), minlength=256).cpu().numpy()
    codes = build_huffman_codes(counts.astype(np.int64))
    max_len = max(l for _, l in codes.values())

    # Build encode tables
    enc_codes_np = np.zeros(256, dtype=np.uint32)
    enc_lens_np = np.zeros(256, dtype=np.uint8)
    for sym, (code, length) in codes.items():
        enc_codes_np[sym] = code
        enc_lens_np[sym] = length

    # Build decode LUT
    dec_sym_np, dec_len_np = build_decode_lut(codes, max_len)

    # Transfer tables to GPU (CuPy)
    byte3_cp = cp.asarray(byte3_gpu.contiguous())
    enc_codes_cp = cp.asarray(enc_codes_np)
    enc_lens_cp = cp.asarray(enc_lens_np)

    n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Phase 1: compute block bit totals
    block_totals_cp = cp.zeros(n_blocks, dtype=cp.uint32)
    COMPUTE_BLOCK_TOTALS(
        (n_blocks,), (BLOCK_SIZE,),
        (byte3_cp, enc_lens_cp, block_totals_cp, np.int32(n), np.int32(BLOCK_SIZE)),
        shared_mem=BLOCK_SIZE * 4
    )

    # Phase 2: prefix sum of block totals (CPU, small array)
    block_totals_np = block_totals_cp.get().astype(np.uint64)
    block_offsets_np = np.zeros(n_blocks, dtype=np.uint64)
    block_offsets_np[1:] = np.cumsum(block_totals_np[:-1])
    total_bits = int(np.sum(block_totals_np))
    block_offsets_cp = cp.asarray(block_offsets_np)

    # Phase 3: scatter bits
    total_bytes = (total_bits + 7) // 8
    # Round up to multiple of 4 for atomicOr alignment
    output_bytes = ((total_bytes + 3) // 4) * 4
    output_cp = cp.zeros(output_bytes, dtype=cp.uint8)

    SCATTER_BITS(
        (n_blocks,), (BLOCK_SIZE,),
        (byte3_cp, enc_codes_cp, enc_lens_cp, output_cp, block_offsets_cp,
         np.int32(n), np.int32(BLOCK_SIZE)),
        shared_mem=BLOCK_SIZE * 4
    )

    # Convert back to torch tensors
    packed_data = torch.as_tensor(output_cp[:total_bytes], device=device)
    block_offsets = torch.as_tensor(block_offsets_cp, device=device).to(torch.int64)

    # Store decode tables
    dec_sym_t = torch.from_numpy(dec_sym_np).to(device)
    dec_len_t = torch.from_numpy(dec_len_np).to(device)

    return packed_data, block_offsets, dec_sym_t, dec_len_t, max_len, total_bits


def huffman_decompress_byte3(packed_data, block_offsets, dec_sym_t, dec_len_t,
                              max_len, total_bits, n, device):
    """Decompress byte3 on GPU using block-parallel decode."""
    BLOCK_SIZE = 256
    n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    packed_cp = cp.asarray(packed_data.contiguous())
    dec_sym_cp = cp.asarray(dec_sym_t.contiguous())
    dec_len_cp = cp.asarray(dec_len_t.contiguous())
    # Convert block_offsets to uint64 for the kernel
    block_off_torch = block_offsets.to(torch.int64).contiguous()
    block_off_cp = cp.asarray(block_off_torch).view(cp.uint64)
    output_cp = cp.zeros(n, dtype=cp.uint8)

    DECODE_KERNEL(
        (n_blocks,), (1,),  # 1 thread per block for sequential decode
        (packed_cp, output_cp, dec_sym_cp, dec_len_cp, block_off_cp,
         np.int32(n), np.int32(BLOCK_SIZE), np.int32(max_len), np.int64(total_bits))
    )

    return torch.as_tensor(output_cp, device=device)


# ============ Full Compression Pipeline ============

def compress_flat(flat_fp32):
    """Compress flat FP32 with Huffman byte3."""
    n = flat_fp32.numel()
    device = flat_fp32.device
    int32 = flat_fp32.view(torch.int32)

    byte3 = ((int32 >> 24) & 0xFF).to(torch.uint8)
    byte012 = int32.view(torch.uint8).reshape(n, 4)[:, :3].contiguous().reshape(-1)

    packed, block_off, dec_sym, dec_len, max_len, total_bits = huffman_compress_byte3(byte3, device)
    del byte3

    # Also need codebook for decode (byte3 values, not indices)
    # Actually Huffman decode directly gives byte3 values, no codebook needed!

    return byte012, packed, block_off, dec_sym, dec_len, max_len, total_bits, n


def decompress_flat(data):
    byte012, packed, block_off, dec_sym, dec_len, max_len, total_bits, n = data
    device = byte012.device

    byte3 = huffman_decompress_byte3(packed, block_off, dec_sym, dec_len,
                                      max_len, total_bits, n, device)

    b = byte012.reshape(n, 3)
    result = (b[:, 0].to(torch.int32) |
              (b[:, 1].to(torch.int32) << 8) |
              (b[:, 2].to(torch.int32) << 16) |
              (byte3.to(torch.int32) << 24))
    return result.view(torch.float32)


# ============ Optimizer Wrapper ============

class HuffmanCompressedAdamW:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._params = None
        self._sizes = None
        self._offsets = None
        self._total_n = 0
        self._flat_m = None
        self._flat_v = None
        self._m_data = None
        self._v_data = None
        self._is_compressed = False
        self._first_step = True

    def _init_params(self):
        self._params = []
        self._sizes = []
        self._offsets = []
        offset = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p in self.optimizer.state and 'exp_avg' in self.optimizer.state[p]:
                    self._params.append(p)
                    self._sizes.append(p.numel())
                    self._offsets.append(offset)
                    offset += p.numel()
        self._total_n = offset

    def _set_views(self, flat, key):
        for p, off, sz in zip(self._params, self._offsets, self._sizes):
            self.optimizer.state[p][key] = flat[off:off+sz].view(p.shape)

    def _initial_gather(self, key):
        device = self._params[0].device
        flat = torch.empty(self._total_n, dtype=torch.float32, device=device)
        for p, off, sz in zip(self._params, self._offsets, self._sizes):
            flat[off:off+sz] = self.optimizer.state[p][key].flatten()
        return flat

    def _compress_states(self):
        if self._first_step:
            self._init_params()
            self._flat_m = self._initial_gather('exp_avg')
            self._flat_v = self._initial_gather('exp_avg_sq')
            self._first_step = False

        device = self._params[0].device

        self._m_data = compress_flat(self._flat_m)
        del self._flat_m; self._flat_m = None

        self._v_data = compress_flat(self._flat_v)
        del self._flat_v; self._flat_v = None

        for p in self._params:
            self.optimizer.state[p]['exp_avg'] = torch.empty(0, dtype=torch.float32, device=device)
            self.optimizer.state[p]['exp_avg_sq'] = torch.empty(0, dtype=torch.float32, device=device)

        self._is_compressed = True

    def _decompress_states(self):
        self._flat_m = decompress_flat(self._m_data)
        self._m_data = None
        self._set_views(self._flat_m, 'exp_avg')

        self._flat_v = decompress_flat(self._v_data)
        self._v_data = None
        self._set_views(self._flat_v, 'exp_avg_sq')

        self._is_compressed = False

    def step(self):
        if self._is_compressed:
            self._decompress_states()
        self.optimizer.step()
        self._compress_states()

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def get_stats(self):
        total_c, total_o = 0, self._total_n * 4 * 2
        for name, data in [('m', self._m_data), ('v', self._v_data)]:
            if data:
                byte012, packed, block_off, dec_sym, dec_len, max_len, total_bits, n = data
                c = byte012.numel() + packed.numel() + block_off.numel() * 4
                avg_bits = total_bits / n
                total_c += c
                print(f"  {name}: avg {avg_bits:.2f} bits/byte3, "
                      f"ratio={c/(n*4)*100:.1f}%, saves {(n*4-c)/1024**2:.0f} MB")
        return {'ratio': total_c / total_o, 'savings_mb': (total_o - total_c) / 1024**2}


# ============ Verify & Benchmark ============

def verify(model_name="Qwen/Qwen3-0.6B"):
    print("--- Verify ---")

    # Small round-trip test
    for test_n in [256, 1024, 10000, 100000]:
        data = torch.randn(test_n, dtype=torch.float32, device='cuda')
        compressed = compress_flat(data)
        restored = decompress_flat(compressed)
        if not torch.all(data == restored):
            diff = (data - restored).abs().max().item()
            print(f"  FAILED n={test_n}, max_diff={diff}")
            # Debug: check which values differ
            mismatch = (data != restored).nonzero(as_tuple=True)[0]
            if len(mismatch) > 0:
                idx = mismatch[0].item()
                orig = data[idx].view(torch.int32).item()
                rest = restored[idx].view(torch.int32).item()
                print(f"  First mismatch at [{idx}]: orig=0x{orig & 0xFFFFFFFF:08X}, restored=0x{rest & 0xFFFFFFFF:08X}")
            return False
        else:
            pass
    print("  ✓ Small round-trip OK")

    # Full optimizer test
    torch.manual_seed(42)
    m1 = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    m1.train()
    o1 = torch.optim.AdamW(m1.parameters(), lr=1e-4, weight_decay=0.01)

    torch.manual_seed(42)
    m2 = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    m2.train()
    inner = torch.optim.AdamW(m2.parameters(), lr=1e-4, weight_decay=0.01)
    o2 = HuffmanCompressedAdamW(inner)

    for s in range(5):
        torch.manual_seed(s + 100)
        ids = torch.randint(100, 10000, (2, 128), device='cuda')
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            m1(input_ids=ids, labels=ids).loss.backward()
        o1.step(); o1.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            m2(input_ids=ids, labels=ids).loss.backward()
        o2.step(); o2.zero_grad()

    max_diff = max((p1.data - p2.data).abs().max().item()
                   for p1, p2 in zip(m1.parameters(), m2.parameters()))
    print(f"  Max diff: {max_diff}" + (" ✓" if max_diff == 0 else " ✗"))

    if max_diff == 0:
        stats = o2.get_stats()
        print(f"  Total: ratio={stats['ratio']*100:.1f}%, savings={stats['savings_mb']:.0f} MB")

    del m1, m2, o1, o2, inner
    gc.collect(); torch.cuda.empty_cache()
    return max_diff == 0


def benchmark(model_name="Qwen/Qwen3-0.6B", batch_size=4, seq_len=256):
    print("\n" + "=" * 80)
    print("GPU Huffman Compressed FP32 Optimizer")
    print("=" * 80)

    results = []
    for name, use_comp in [("Standard FP32 AdamW", False),
                            ("GPU Huffman Byte3", True)]:
        print(f"\n--- {name} ---")
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
        model.train()
        inner = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        opt = HuffmanCompressedAdamW(inner) if use_comp else inner

        for _ in range(10):
            ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                model(input_ids=ids, labels=ids).loss.backward()
            opt.step(); opt.zero_grad()

        gc.collect(); torch.cuda.empty_cache()
        gpu_mem = torch.cuda.memory_allocated() / 1024**2

        if use_comp:
            stats = opt.get_stats()
            print(f"  Total: ratio={stats['ratio']*100:.1f}%, savings={stats['savings_mb']:.0f} MB")

        torch.cuda.synchronize()
        times = []
        for _ in range(40):
            ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                model(input_ids=ids, labels=ids).loss.backward()
            opt.step(); opt.zero_grad()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg = sum(times) / len(times)
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Mem: {gpu_mem:.0f} MB, Peak: {peak:.0f} MB, Step: {avg*1000:.1f} ms")
        results.append({'method': name, 'gpu_mem': gpu_mem, 'peak': peak, 'step_ms': avg*1000})

        del model, inner, opt
        gc.collect(); torch.cuda.empty_cache()

    bl = results[0]
    print(f"\n{'='*60}")
    print(f"{'Method':<25} {'Mem':>7} {'ΔMem':>7} {'Peak':>7} {'Step':>7} {'Slow':>5}")
    print("-" * 58)
    for r in results:
        dm = r['gpu_mem'] - bl['gpu_mem']
        s = r['step_ms'] / bl['step_ms']
        print(f"{r['method']:<25} {r['gpu_mem']:>6.0f}M {dm:>+6.0f}M {r['peak']:>6.0f}M "
              f"{r['step_ms']:>6.1f} {s:>4.2f}x")


if __name__ == '__main__':
    ok = verify()
    if ok:
        benchmark()
