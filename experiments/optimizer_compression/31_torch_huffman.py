"""
Experiment 31: Pure PyTorch Huffman Encoding for byte3

Use PyTorch ops for Huffman encode/decode:
- Encode: prefix sum of code lengths → bit positions, then multi-pass bit scatter
- Decode: block-parallel with sync offsets, implemented as a custom CUDA kernel

Key insight: encode is parallelizable, decode needs sync offsets.
For decode: use CuPy kernel with 1 thread per block (already verified correct
when encode is correct).
"""

import torch
import time
import gc
import math
import heapq
import numpy as np
from transformers import AutoModelForCausalLM

import cupy as cp


def build_huffman_codes(counts_np):
    """Build Huffman codes from frequency counts[256]."""
    heap = []
    for sym in range(256):
        if counts_np[sym] > 0:
            heapq.heappush(heap, (int(counts_np[sym]), sym, sym))

    if len(heap) <= 1:
        sym = heap[0][2] if heap else 0
        return {sym: (0, 1)}, 1

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

    max_len = max(l for _, l in codes.values())
    return codes, max_len


def reverse_bits(code, length):
    result = 0
    for i in range(length):
        result |= ((code >> i) & 1) << (length - 1 - i)
    return result


def build_tables(codes, max_len, device):
    """Build encode and decode tables on device."""
    enc_codes = torch.zeros(256, dtype=torch.int32, device=device)
    enc_lens = torch.zeros(256, dtype=torch.int32, device=device)

    for sym, (code, length) in codes.items():
        enc_codes[sym] = code
        enc_lens[sym] = length

    # Decode LUT (LSB-first ordering for decode)
    lut_size = 1 << max_len
    dec_sym = torch.zeros(lut_size, dtype=torch.uint8, device=device)
    dec_len = torch.zeros(lut_size, dtype=torch.uint8, device=device)

    for sym, (code, length) in codes.items():
        rev = reverse_bits(code, length)
        padding = max_len - length
        for i in range(1 << padding):
            idx = rev | (i << length)
            dec_sym[idx] = sym
            dec_len[idx] = length

    return enc_codes, enc_lens, dec_sym, dec_len


def huffman_encode_torch(byte3: torch.Tensor, enc_codes, enc_lens, max_len):
    """Encode byte3 using Huffman codes with pure PyTorch ops.

    Strategy:
    1. Look up code and length for each value
    2. Prefix sum of lengths → starting bit positions
    3. For each bit position in the code, compute output byte and bit offset
    4. Use scatter_add to set bits in output
    """
    n = byte3.numel()
    device = byte3.device

    # Look up codes and lengths
    idx32 = byte3.to(torch.int32)
    codes = enc_codes[idx32]   # [N] int32
    lens = enc_lens[idx32]     # [N] int32

    # Compute bit positions via cumsum
    bit_starts = torch.cumsum(lens, dim=0) - lens  # [N] exclusive prefix sum
    total_bits = (bit_starts[-1] + lens[-1]).item()
    total_bytes = (total_bits + 7) // 8

    # Compute sync offsets for decode (every BLOCK_SIZE values)
    BLOCK_SIZE = 256
    n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    sync_indices = torch.arange(0, n, BLOCK_SIZE, device=device)
    sync_offsets = bit_starts[sync_indices]  # [n_blocks] int32 → need int64 for large N
    sync_offsets = sync_offsets.to(torch.int64)

    # Output buffer (round up to multiple of 4 for safety)
    out_bytes = ((total_bytes + 3) // 4) * 4
    output = torch.zeros(out_bytes, dtype=torch.uint8, device=device)

    # For each bit position in the codes (max_len iterations):
    # Set the corresponding bit in the output
    for bit_idx in range(max_len):
        # Which values have code length > bit_idx?
        active = lens > bit_idx  # [N] bool
        if not active.any():
            break

        # Get the bit value at position bit_idx (MSB-first)
        # bit_idx=0 is MSB, bit_idx=len-1 is LSB
        # Bit value = (code >> (len - 1 - bit_idx)) & 1
        active_codes = codes[active]
        active_lens = lens[active]
        active_starts = bit_starts[active]

        bit_vals = (active_codes >> (active_lens - 1 - bit_idx)) & 1  # [K]

        # Only process values where bit is 1
        ones = bit_vals == 1
        if not ones.any():
            continue

        abs_bit_pos = active_starts[ones] + bit_idx  # [M] int32/int64
        abs_bit_pos = abs_bit_pos.to(torch.int64)

        byte_pos = abs_bit_pos // 8                    # [M]
        bit_offset = (abs_bit_pos % 8).to(torch.int32) # [M]

        # Group by bit_offset to avoid duplicate byte_pos in scatter_add_
        # (CUDA scatter_add_ with duplicate indices is non-deterministic)
        for bo in range(8):
            bo_mask = bit_offset == bo
            if not bo_mask.any():
                continue
            bo_byte_pos = byte_pos[bo_mask]
            bo_byte_val = torch.full_like(bo_byte_pos, 1 << bo, dtype=torch.uint8)
            output.scatter_add_(0, bo_byte_pos, bo_byte_val)

    return output[:total_bytes], sync_offsets, total_bits


# Decode kernel (CuPy): 1 thread per block, sequential decode
DECODE_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void huffman_decode(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    const unsigned char* __restrict__ dec_sym,
    const unsigned char* __restrict__ dec_len,
    const long long* __restrict__ block_offsets,
    const int N,
    const int BLOCK_SIZE,
    const int max_len,
    const long long total_bytes
) {
    int block_id = blockIdx.x;
    if (threadIdx.x != 0) return;

    int start = block_id * BLOCK_SIZE;
    int end = min(start + BLOCK_SIZE, N);
    long long bit_pos = block_offsets[block_id];
    unsigned int mask = (1u << max_len) - 1;

    for (int i = start; i < end; i++) {
        long long byte_idx = bit_pos >> 3;
        unsigned int bit_offset = (unsigned int)(bit_pos & 7);

        // Read up to 4 bytes (handles max_len up to 25 with any bit_offset)
        unsigned int val = (unsigned int)input[byte_idx];
        if (byte_idx + 1 < total_bytes)
            val |= (unsigned int)input[byte_idx + 1] << 8;
        if (byte_idx + 2 < total_bytes)
            val |= (unsigned int)input[byte_idx + 2] << 16;
        if (byte_idx + 3 < total_bytes)
            val |= (unsigned int)input[byte_idx + 3] << 24;

        unsigned int window = (val >> bit_offset) & mask;
        output[i] = dec_sym[window];
        bit_pos += dec_len[window];
    }
}
''', 'huffman_decode')


def huffman_decode_gpu(packed, sync_offsets, total_bits, n, dec_sym, dec_len, max_len, device):
    """Decode Huffman-coded byte3 on GPU."""
    BLOCK_SIZE = 256
    n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_bytes = (total_bits + 7) // 8

    packed_cp = cp.asarray(packed.contiguous())
    dec_sym_cp = cp.asarray(dec_sym.contiguous())
    dec_len_cp = cp.asarray(dec_len.contiguous())
    offsets_cp = cp.asarray(sync_offsets.contiguous())
    output_cp = cp.zeros(n, dtype=cp.uint8)

    DECODE_KERNEL(
        (n_blocks,), (1,),
        (packed_cp, output_cp, dec_sym_cp, dec_len_cp, offsets_cp,
         np.int32(n), np.int32(BLOCK_SIZE), np.int32(max_len), np.int64(total_bytes))
    )

    return torch.as_tensor(output_cp, device=device)


# ============ Full Pipeline ============

def compress_flat(flat_fp32):
    n = flat_fp32.numel()
    device = flat_fp32.device
    int32 = flat_fp32.view(torch.int32)

    byte3 = ((int32 >> 24) & 0xFF).to(torch.uint8)

    # Build Huffman
    counts = torch.bincount(byte3.to(torch.int32), minlength=256).cpu().numpy()
    codes, max_len = build_huffman_codes(counts)
    enc_codes, enc_lens, dec_sym, dec_len = build_tables(codes, max_len, device)

    # Encode
    packed, sync_offsets, total_bits = huffman_encode_torch(byte3, enc_codes, enc_lens, max_len)
    del byte3

    # byte012
    byte012 = int32.view(torch.uint8).reshape(n, 4)[:, :3].contiguous().reshape(-1)

    return byte012, packed, sync_offsets, total_bits, dec_sym, dec_len, max_len, n


def decompress_flat(data):
    byte012, packed, sync_offsets, total_bits, dec_sym, dec_len, max_len, n = data
    device = byte012.device

    byte3 = huffman_decode_gpu(packed, sync_offsets, total_bits, n, dec_sym, dec_len, max_len, device)

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
                byte012, packed, sync_off, total_bits, *_, n = data
                c = byte012.numel() + packed.numel() + sync_off.numel() * 8
                avg_bits = total_bits / n
                total_c += c
                print(f"  {name}: avg {avg_bits:.2f} bits/byte3, "
                      f"ratio={c/(n*4)*100:.1f}%, saves {(n*4-c)/1024**2:.0f} MB")
        return {'ratio': total_c / total_o, 'savings_mb': (total_o - total_c) / 1024**2}


# ============ Verify ============

def verify(model_name="Qwen/Qwen3-0.6B"):
    print("--- Verify ---")

    for test_n in [256, 1024, 10000, 100000, 1000000, 10000000]:
        data = torch.randn(test_n, dtype=torch.float32, device='cuda')
        compressed = compress_flat(data)
        restored = decompress_flat(compressed)
        match = torch.all(data == restored).item()
        if not match:
            mismatches = (data != restored).sum().item()
            print(f"  n={test_n}: FAIL ({mismatches} mismatches)")
            return False
    print("  ✓ Round-trip OK")

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

    del m1, m2, o1, o2, inner
    gc.collect(); torch.cuda.empty_cache()
    return max_diff == 0


def benchmark(model_name="Qwen/Qwen3-0.6B", batch_size=4, seq_len=256):
    print("\n" + "=" * 80)
    print("PyTorch Huffman Compressed FP32 Optimizer")
    print("=" * 80)

    results = []
    for name, use_comp in [("Standard FP32 AdamW", False),
                            ("Huffman Byte3", True)]:
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
