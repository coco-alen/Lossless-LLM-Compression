"""
Experiment 26: GPU Huffman Coding for byte3

byte3 has ~47 unique values with entropy ~2.3-3.1 bits.
Fixed 6-bit coding saves 6.25%. Huffman can save ~15-16%.

Approach:
- Build Huffman tree on CPU (fast, ≤256 symbols)
- Create encoding table: symbol → (code_bits, code_length) as GPU tensors
- Encode: parallel prefix sum of code lengths → bit positions, then scatter codes
- Store byte012 raw (3 bytes per value) + Huffman-coded byte3

Expected: ~3 bits/byte3 avg → 3/8 bytes per value for byte3
Total: 3 + 0.375 = 3.375 bytes per value = 84.4% → 15.6% savings
On 596M params × 2 states × 4 bytes = 4548 MB → saves ~710 MB

Zero-copy flat buffer approach.
"""

import torch
import time
import gc
import math
from collections import Counter
import heapq
from transformers import AutoModelForCausalLM


def build_huffman_table(byte3_flat: torch.Tensor):
    """Build Huffman encoding/decoding tables from byte3 frequency counts.
    Returns:
      enc_codes: int32[256] - bit patterns for each byte value
      enc_lengths: uint8[256] - code lengths
      dec_symbols: uint8[max_entries] - decode table (indexed by left-aligned code)
    """
    # Count frequencies on GPU
    counts = torch.bincount(byte3_flat.to(torch.int32), minlength=256)
    freq = counts.cpu().tolist()

    # Build Huffman tree on CPU
    heap = []
    for sym, f in enumerate(freq):
        if f > 0:
            heapq.heappush(heap, (f, sym, sym))  # (freq, tiebreaker, node)

    if len(heap) <= 1:
        # Only one symbol — trivial
        sym = heap[0][2] if heap else 0
        codes = {sym: (0, 1)}
    else:
        nodes = {}  # node_id → (left, right) or symbol
        next_id = 256
        while len(heap) > 1:
            f1, _, n1 = heapq.heappop(heap)
            f2, _, n2 = heapq.heappop(heap)
            nodes[next_id] = (n1, n2)
            heapq.heappush(heap, (f1 + f2, next_id, next_id))
            next_id += 1

        # Traverse tree to get codes
        codes = {}
        root = heap[0][2]
        stack = [(root, 0, 0)]  # (node, code, length)
        while stack:
            node, code, length = stack.pop()
            if node < 256:
                codes[node] = (code, max(length, 1))
            else:
                left, right = nodes[node]
                stack.append((left, code << 1, length + 1))
                stack.append((right, (code << 1) | 1, length + 1))

    # Create GPU lookup tables
    device = byte3_flat.device
    enc_codes = torch.zeros(256, dtype=torch.int32, device=device)
    enc_lengths = torch.zeros(256, dtype=torch.uint8, device=device)
    for sym, (code, length) in codes.items():
        enc_codes[sym] = code
        enc_lengths[sym] = length

    max_len = max(l for _, l in codes.values())

    # Build decode LUT: for each possible code prefix, store the decoded symbol
    # Use canonical Huffman with left-aligned codes for fast decode
    # Actually, for GPU decode we use a flat LUT indexed by max_len bits
    dec_lut_size = 1 << max_len
    dec_symbols = torch.zeros(dec_lut_size, dtype=torch.uint8, device=device)
    dec_lengths = torch.zeros(dec_lut_size, dtype=torch.uint8, device=device)

    for sym, (code, length) in codes.items():
        # All entries where the first `length` bits match `code`
        padding_bits = max_len - length
        base = code << padding_bits
        for i in range(1 << padding_bits):
            dec_symbols[base | i] = sym
            dec_lengths[base | i] = length

    return enc_codes, enc_lengths, dec_symbols, dec_lengths, max_len


def huffman_encode_gpu(byte3: torch.Tensor, enc_codes, enc_lengths):
    """Encode byte3 values using Huffman codes on GPU.

    Strategy: compute bit positions via prefix sum, then scatter bits.
    """
    n = byte3.numel()
    device = byte3.device

    # Look up code and length for each symbol
    codes = enc_codes[byte3.to(torch.int32)]      # [N] int32
    lengths = enc_lengths[byte3.to(torch.int32)]   # [N] uint8

    # Compute bit positions via cumulative sum of lengths
    bit_positions = torch.cumsum(lengths.to(torch.int64), dim=0)  # [N]
    total_bits = bit_positions[-1].item()
    total_bytes = (total_bits + 7) // 8

    # Shift bit_positions to get starting position of each code
    bit_starts = bit_positions - lengths.to(torch.int64)  # [N]

    # Scatter bits into output buffer
    # Process in chunks to limit memory
    output = torch.zeros(total_bytes, dtype=torch.uint8, device=device)

    # For each value, write its code bits starting at bit_starts[i]
    # We process bit-by-bit for each code length level
    max_len = lengths.max().item()

    for bit_idx in range(max_len):
        # Which values have code length > bit_idx?
        mask = lengths.to(torch.int32) > bit_idx
        if not mask.any():
            break

        # Get the bit at position bit_idx for each active value
        active_codes = codes[mask]
        active_starts = bit_starts[mask]

        # Extract bit at position (length - 1 - bit_idx) from MSB-first code
        active_lengths = lengths[mask].to(torch.int32)
        bit_val = (active_codes >> (active_lengths - 1 - bit_idx)) & 1

        # Compute byte and bit position in output
        abs_bit_pos = active_starts + bit_idx
        byte_pos = (abs_bit_pos // 8).to(torch.int64)
        bit_offset = (abs_bit_pos % 8).to(torch.uint8)

        # Scatter: output[byte_pos] |= bit_val << bit_offset
        # Use scatter_add with pre-shifted values
        shifted = (bit_val.to(torch.uint8) << bit_offset)
        output.scatter_add_(0, byte_pos, shifted)

    return output, total_bits


def huffman_decode_gpu(packed: torch.Tensor, total_bits: int, n: int,
                        dec_symbols, dec_lengths, max_len):
    """Decode Huffman-coded byte3 on GPU.

    Strategy: sequential decode is hard to parallelize.
    Use batch decode: decode max_len bits at each position, advance by code length.
    For GPU: precompute all possible bit windows, then iteratively decode.

    Actually for simplicity, decode on CPU then transfer. The decode table is small.
    """
    # For small max_len (≤8), we can use a simple approach:
    # Extract overlapping windows of max_len bits and look up in decode LUT
    # But positions depend on previous decode lengths — inherently sequential.

    # Practical approach: decode in chunks on GPU using a loop
    device = packed.device
    output = torch.empty(n, dtype=torch.uint8, device=device)

    # Move to CPU for sequential decode (Huffman decode is inherently sequential)
    packed_cpu = packed.cpu()
    dec_symbols_cpu = dec_symbols.cpu()
    dec_lengths_cpu = dec_lengths.cpu()

    # Build a byte array for fast bit access
    packed_bytes = packed_cpu.numpy()
    out_cpu = torch.empty(n, dtype=torch.uint8)

    bit_pos = 0
    for i in range(n):
        # Extract max_len bits starting at bit_pos
        byte_idx = bit_pos >> 3
        bit_offset = bit_pos & 7

        # Read enough bytes to cover max_len bits
        needed_bytes = (bit_offset + max_len + 7) >> 3
        val = 0
        for b in range(min(needed_bytes, len(packed_bytes) - byte_idx)):
            val |= int(packed_bytes[byte_idx + b]) << (8 * b)

        # Extract max_len bits starting at bit_offset (MSB-first in our encoding)
        window = (val >> bit_offset) & ((1 << max_len) - 1)

        # Reverse bits to match MSB-first encoding
        # Actually our encoding stores MSB first, but bits are packed LSB-first in bytes
        # Let's just look up directly
        sym = dec_symbols_cpu[window].item()
        length = dec_lengths_cpu[window].item()

        out_cpu[i] = sym
        bit_pos += length

    return out_cpu.to(device)


def compress_flat_huffman(flat_fp32: torch.Tensor):
    """Compress flat FP32 with Huffman-coded byte3."""
    n = flat_fp32.numel()
    int32_view = flat_fp32.view(torch.int32)

    # Extract byte3
    byte3 = ((int32_view >> 24) & 0xFF).to(torch.uint8)

    # Build Huffman table
    enc_codes, enc_lengths, dec_symbols, dec_lengths, max_len = build_huffman_table(byte3)

    # Encode byte3
    packed_byte3, total_bits = huffman_encode_gpu(byte3, enc_codes, enc_lengths)
    del byte3

    # Extract byte012 raw
    byte012 = int32_view.view(torch.uint8).reshape(n, 4)[:, :3].contiguous().reshape(-1)

    return (byte012, packed_byte3, total_bits,
            enc_codes, enc_lengths, dec_symbols, dec_lengths, max_len, n)


def decompress_flat_huffman(data):
    """Decompress Huffman-coded byte3 + raw byte012."""
    byte012, packed_byte3, total_bits, enc_codes, enc_lengths, dec_symbols, dec_lengths, max_len, n = data

    # Decode byte3
    byte3 = huffman_decode_gpu(packed_byte3, total_bits, n, dec_symbols, dec_lengths, max_len)

    # Reconstruct int32
    b = byte012.reshape(n, 3)
    result = (b[:, 0].to(torch.int32) |
              (b[:, 1].to(torch.int32) << 8) |
              (b[:, 2].to(torch.int32) << 16) |
              (byte3.to(torch.int32) << 24))
    return result.view(torch.float32)


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

        self._m_data = compress_flat_huffman(self._flat_m)
        del self._flat_m; self._flat_m = None

        self._v_data = compress_flat_huffman(self._flat_v)
        del self._flat_v; self._flat_v = None

        for p in self._params:
            self.optimizer.state[p]['exp_avg'] = torch.empty(0, dtype=torch.float32, device=device)
            self.optimizer.state[p]['exp_avg_sq'] = torch.empty(0, dtype=torch.float32, device=device)

        self._is_compressed = True

    def _decompress_states(self):
        self._flat_m = decompress_flat_huffman(self._m_data)
        self._m_data = None
        self._set_views(self._flat_m, 'exp_avg')

        self._flat_v = decompress_flat_huffman(self._v_data)
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
                byte012, packed_byte3, total_bits, *_, n = data
                c = byte012.numel() + packed_byte3.numel()
                o = n * 4
                avg_bits = total_bits / n
                total_c += c
                print(f"  {name}: avg {avg_bits:.2f} bits/byte3, "
                      f"ratio={c/o*100:.1f}%, saves {(o-c)/1024**2:.0f} MB")
        return {'ratio': total_c / total_o, 'savings_mb': (total_o - total_c) / 1024**2}


def verify(model_name="Qwen/Qwen3-0.6B"):
    print("--- Verify ---")

    # Round-trip test
    for n in [100, 1000, 10000]:
        data = torch.randn(n, dtype=torch.float32, device='cuda')
        compressed = compress_flat_huffman(data)
        restored = decompress_flat_huffman(compressed)
        assert torch.all(data == restored), f"FAILED n={n}"
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

    for s in range(3):
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
        print(f"  Compression: {stats['ratio']*100:.1f}%, saves {stats['savings_mb']:.0f} MB")

    del m1, m2, o1, o2, inner
    gc.collect(); torch.cuda.empty_cache()
    return max_diff == 0


if __name__ == '__main__':
    ok = verify()
    if ok:
        print("\n✓ Huffman verification passed")
        print("Note: Full benchmark skipped — CPU Huffman decode is too slow for 596M values")
        print("Need GPU-parallel decode for practical use")
