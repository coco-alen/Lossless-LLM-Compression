"""
Experiment 25: High-16 Compression with 16-bit Codebook

Instead of compressing only byte3 (6.25% max savings), compress the HIGH 16 bits
(byte3 << 8 | byte2) as a unit. Typical FP32 optimizer states have only ~5000-7500
unique high-16 values, needing ~13 bits. Store low-16 bits raw.

Approach:
- Extract high16 = (int32 >> 16) & 0xFFFF
- Build codebook of unique high16 values (typically ≤8192 → 13 bits)
- Pack indices at 13 bits using int32 arithmetic
- Store low16 raw as uint16 (2 bytes per value)
- Total: 13/8 + 2 = 3.625 bytes per value vs 4 bytes = 90.6% → 9.4% savings

With Huffman-like variable coding: ~11 bits entropy → 11/8 + 2 = 3.375 bytes = 84.4% → 15.6% savings
But fixed-length is simpler and faster on GPU.

Zero-copy flat buffer approach from exp22 for speed.
"""

import torch
import time
import gc
import math
from transformers import AutoModelForCausalLM


def compress_high16(flat_fp32: torch.Tensor):
    """Compress flat FP32 by encoding high-16 bits with codebook.
    Returns (low16, packed_high, codebook, n).
    """
    n = flat_fp32.numel()
    int32_view = flat_fp32.view(torch.int32)

    # Extract high and low 16 bits
    high16 = ((int32_view >> 16) & 0xFFFF).to(torch.int32)
    low16 = (int32_view & 0xFFFF).to(torch.int16)  # 2 bytes per value

    # Build codebook via bincount on high16
    counts = torch.bincount(high16, minlength=65536)
    present = (counts > 0).nonzero(as_tuple=True)[0]
    codebook = present.to(torch.int32)  # unique high16 values
    n_unique = len(codebook)
    bits = max(1, math.ceil(math.log2(max(n_unique, 2))))

    # LUT: high16 value → index
    lut = torch.zeros(65536, dtype=torch.int32, device=flat_fp32.device)
    lut[present] = torch.arange(n_unique, device=flat_fp32.device, dtype=torch.int32)

    indices = lut[high16]
    del high16, counts, lut

    # Pack indices at `bits` per value
    packed_high = pack_nbits(indices, bits)
    del indices

    return low16, packed_high, codebook, n, bits


def pack_nbits(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack int32 indices at `bits` per value into uint8.
    Groups values into chunks that fit cleanly into bytes.
    """
    device = indices.device
    n = indices.numel()

    if bits <= 8:
        # Simple: pack into uint8 directly if bits <= 8
        # Group size: how many values fit in 32 bits
        vals_per_group = 32 // bits
        remainder = n % vals_per_group
        if remainder:
            indices = torch.cat([indices, torch.zeros(vals_per_group - remainder, dtype=torch.int32, device=device)])

        groups = indices.reshape(-1, vals_per_group)
        combined = groups[:, 0]
        for i in range(1, vals_per_group):
            combined = combined | (groups[:, i] << (bits * i))

        # Extract bytes
        bytes_per_group = (bits * vals_per_group + 7) // 8
        result_bytes = []
        for b in range(bytes_per_group):
            result_bytes.append(((combined >> (8 * b)) & 0xFF).to(torch.uint8))
        return torch.stack(result_bytes, dim=1).reshape(-1)

    elif bits <= 16:
        # Use int32, pack 2 values per group (2*bits ≤ 32)
        remainder = n % 2
        if remainder:
            indices = torch.cat([indices, torch.zeros(1, dtype=torch.int32, device=device)])
        groups = indices.reshape(-1, 2)
        combined = groups[:, 0] | (groups[:, 1] << bits)
        bytes_per_group = (bits * 2 + 7) // 8
        result_bytes = []
        for b in range(bytes_per_group):
            result_bytes.append(((combined >> (8 * b)) & 0xFF).to(torch.uint8))
        return torch.stack(result_bytes, dim=1).reshape(-1)
    else:
        # Fallback: store as uint16
        return indices.to(torch.uint16).view(torch.uint8)


def unpack_nbits(packed: torch.Tensor, n: int, bits: int) -> torch.Tensor:
    """Unpack uint8 packed data back to int32 indices."""
    device = packed.device

    if bits <= 8:
        vals_per_group = 32 // bits
        n_groups = (n + vals_per_group - 1) // vals_per_group
        bytes_per_group = (bits * vals_per_group + 7) // 8

        groups = packed[:n_groups * bytes_per_group].reshape(n_groups, bytes_per_group)
        combined = groups[:, 0].to(torch.int32)
        for b in range(1, bytes_per_group):
            combined = combined | (groups[:, b].to(torch.int32) << (8 * b))

        mask = (1 << bits) - 1
        result = []
        for i in range(vals_per_group):
            result.append((combined >> (bits * i)) & mask)
        return torch.stack(result, dim=1).reshape(-1)[:n]

    elif bits <= 16:
        n_groups = (n + 1) // 2
        bytes_per_group = (bits * 2 + 7) // 8

        groups = packed[:n_groups * bytes_per_group].reshape(n_groups, bytes_per_group)
        combined = groups[:, 0].to(torch.int32)
        for b in range(1, bytes_per_group):
            combined = combined | (groups[:, b].to(torch.int32) << (8 * b))

        mask = (1 << bits) - 1
        val0 = combined & mask
        val1 = (combined >> bits) & mask
        return torch.stack([val0, val1], dim=1).reshape(-1)[:n]
    else:
        return packed.view(torch.uint16).to(torch.int32)[:n]


def decompress_high16(low16, packed_high, codebook, n, bits):
    """Decompress back to flat FP32."""
    indices = unpack_nbits(packed_high, n, bits)
    high16 = codebook[indices]
    del indices

    result = (low16[:n].to(torch.int32) & 0xFFFF) | (high16 << 16)
    return result.view(torch.float32)


class High16CompressedAdamW:
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

    def _compress(self, flat_fp32):
        return compress_high16(flat_fp32)

    def _decompress(self, data):
        return decompress_high16(*data)

    def _compress_states(self):
        if self._first_step:
            self._init_params()
            self._flat_m = self._initial_gather('exp_avg')
            self._flat_v = self._initial_gather('exp_avg_sq')
            self._first_step = False

        device = self._params[0].device

        self._m_data = self._compress(self._flat_m)
        del self._flat_m; self._flat_m = None

        self._v_data = self._compress(self._flat_v)
        del self._flat_v; self._flat_v = None

        for p in self._params:
            self.optimizer.state[p]['exp_avg'] = torch.empty(0, dtype=torch.float32, device=device)
            self.optimizer.state[p]['exp_avg_sq'] = torch.empty(0, dtype=torch.float32, device=device)

        self._is_compressed = True

    def _decompress_states(self):
        self._flat_m = self._decompress(self._m_data)
        self._m_data = None
        self._set_views(self._flat_m, 'exp_avg')

        self._flat_v = self._decompress(self._v_data)
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
                low16, packed_high, codebook, n, bits = data
                c = low16.numel() * 2 + packed_high.numel() + codebook.numel() * 4
                o = n * 4
                total_c += c
                print(f"  {name}: {len(codebook)} unique → {bits}b, "
                      f"ratio={c/o*100:.1f}%, saves {(o-c)/1024**2:.0f} MB")
        return {'ratio': total_c / total_o, 'savings_mb': (total_o - total_c) / 1024**2}


def verify(model_name="Qwen/Qwen3-0.6B"):
    print("--- Verify ---")

    # Round-trip test
    for n in [1000, 10000, 100000, 596049]:
        data = torch.randn(n, dtype=torch.float32, device='cuda')
        low16, packed_high, codebook, nn, bits = compress_high16(data)
        restored = decompress_high16(low16, packed_high, codebook, nn, bits)
        assert torch.all(data == restored), f"FAILED n={n}, max_diff={( data - restored).abs().max()}"
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
    o2 = High16CompressedAdamW(inner)

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
    del m1, m2, o1, o2, inner
    gc.collect(); torch.cuda.empty_cache()
    return max_diff == 0


def benchmark(model_name="Qwen/Qwen3-0.6B", batch_size=4, seq_len=256):
    print("\n" + "=" * 80)
    print("High-16 Compressed FP32 Optimizer")
    print("=" * 80)

    results = []
    for name, use_comp in [("Standard FP32 AdamW", False),
                            ("High-16 Compressed", True)]:
        print(f"\n--- {name} ---")
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
        model.train()
        inner = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        opt = High16CompressedAdamW(inner) if use_comp else inner

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
