"""
Experiment 19: Lean Byte3 Compression with Minimal Intermediates

Fixes from memory debug:
1. Use int32 (not int64) for all intermediates → 50% less temp memory
2. Pack 4 values at 6 bits into 3 bytes (covers ≤64 unique byte3)
3. Process m and v sequentially, freeing as we go
4. Single global codebook (no per-param torch.unique)
5. Minimize intermediate tensor count

Expected: ~6% savings (~284 MB for 596M params), <15% slowdown.
"""

import torch
import time
import gc
import math
from transformers import AutoModelForCausalLM


def pack_6bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack uint8 indices (≤63) as 6-bit values: 4 values → 3 bytes.
    Uses int32 intermediates only (not int64)."""
    device = indices.device
    n = indices.numel()
    # Pad to multiple of 4
    remainder = n % 4
    if remainder:
        indices = torch.cat([indices, torch.zeros(4 - remainder, dtype=torch.uint8, device=device)])
    groups = indices.reshape(-1, 4)  # [N/4, 4], uint8
    # Pack: combined(24 bits) = a | (b << 6) | (c << 12) | (d << 18)
    combined = groups[:, 0].to(torch.int32)
    combined = combined | (groups[:, 1].to(torch.int32) << 6)
    combined = combined | (groups[:, 2].to(torch.int32) << 12)
    combined = combined | (groups[:, 3].to(torch.int32) << 18)
    # Extract 3 bytes
    packed = torch.stack([
        (combined & 0xFF).to(torch.uint8),
        ((combined >> 8) & 0xFF).to(torch.uint8),
        ((combined >> 16) & 0xFF).to(torch.uint8),
    ], dim=1).reshape(-1)
    return packed


def unpack_6bit(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Unpack 6-bit values: 3 bytes → 4 values."""
    device = packed.device
    n_groups = (n + 3) // 4
    groups = packed[:n_groups * 3].reshape(n_groups, 3)
    combined = (groups[:, 0].to(torch.int32) |
                (groups[:, 1].to(torch.int32) << 8) |
                (groups[:, 2].to(torch.int32) << 16))
    result = torch.stack([
        (combined & 0x3F).to(torch.uint8),
        ((combined >> 6) & 0x3F).to(torch.uint8),
        ((combined >> 12) & 0x3F).to(torch.uint8),
        ((combined >> 18) & 0x3F).to(torch.uint8),
    ], dim=1).reshape(-1)[:n]
    return result


def pack_5bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack uint8 indices (≤31) as 5-bit values: 8 values → 5 bytes.
    Uses int64 but for groups of 8 (reduced by 8x vs per-element)."""
    device = indices.device
    n = indices.numel()
    remainder = n % 8
    if remainder:
        indices = torch.cat([indices, torch.zeros(8 - remainder, dtype=torch.uint8, device=device)])
    groups = indices.reshape(-1, 8)  # [N/8, 8], uint8
    # Pack into 40-bit value (fits in int64)
    combined = groups[:, 0].to(torch.int64)
    for i in range(1, 8):
        combined = combined | (groups[:, i].to(torch.int64) << (i * 5))
    # Extract 5 bytes
    packed = torch.stack([
        ((combined >> (i * 8)) & 0xFF).to(torch.uint8) for i in range(5)
    ], dim=1).reshape(-1)
    return packed


def unpack_5bit(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Unpack 5-bit values: 5 bytes → 8 values."""
    device = packed.device
    n_groups = (n + 7) // 8
    groups = packed[:n_groups * 5].reshape(n_groups, 5)
    combined = groups[:, 0].to(torch.int64)
    for i in range(1, 5):
        combined = combined | (groups[:, i].to(torch.int64) << (i * 8))
    result = torch.stack([
        ((combined >> (i * 5)) & 0x1F).to(torch.uint8) for i in range(8)
    ], dim=1).reshape(-1)[:n]
    return result


class LeanCompressedAdamW:
    """AdamW with lean byte3 compression. Minimal temporaries."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._params = None
        self._sizes = None
        self._offsets = None
        self._total_n = 0
        self._initialized = False

        # Global codebooks (computed once, reused)
        self._m_codebook = None
        self._v_codebook = None
        self._m_lut = None
        self._v_lut = None
        self._m_bits = 0
        self._v_bits = 0

        # Compressed state
        self._m_byte012 = None
        self._m_packed3 = None
        self._v_byte012 = None
        self._v_packed3 = None
        self._is_compressed = False

    def _init(self):
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
        self._initialized = True

    def _gather(self, key):
        device = self._params[0].device
        flat = torch.empty(self._total_n, dtype=torch.float32, device=device)
        for p, off, sz in zip(self._params, self._offsets, self._sizes):
            flat[off:off+sz] = self.optimizer.state[p][key].flatten()
        return flat

    def _scatter(self, flat, key):
        for p, off, sz in zip(self._params, self._offsets, self._sizes):
            # Use clone() so per-param tensors don't keep the flat tensor alive
            self.optimizer.state[p][key] = flat[off:off+sz].reshape(p.shape).clone()

    def _build_codebook(self, byte3: torch.Tensor):
        """Build codebook and LUT from byte3 values."""
        codebook = torch.unique(byte3).clone()  # clone() to avoid 568MB storage leak from sorted view
        n_unique = len(codebook)
        bits = max(1, math.ceil(math.log2(max(n_unique, 2))))
        lut = torch.zeros(256, dtype=torch.uint8, device=byte3.device)
        lut[codebook.long()] = torch.arange(n_unique, device=byte3.device, dtype=torch.uint8)
        return codebook, lut, bits

    def _compress_tensor(self, flat_fp32, codebook, lut, bits):
        """Compress flat FP32 → (byte012, packed_byte3). Minimal intermediates."""
        n = flat_fp32.numel()
        int32_view = flat_fp32.view(torch.int32)

        # Extract byte3 and map to indices in one step
        byte3_raw = ((int32_view >> 24) & 0xFF).to(torch.uint8)  # N bytes
        indices = lut[byte3_raw.long()]  # N bytes (uses int64 temp internally but freed immediately)
        del byte3_raw

        # Pack byte3 indices
        if bits <= 6:
            packed3 = pack_6bit(indices)
        elif bits <= 5:
            packed3 = pack_5bit(indices)
        else:
            packed3 = indices  # No packing if >6 bits
        del indices

        # Extract lower 3 bytes efficiently using uint8 view
        # View int32 as uint8: [b0, b1, b2, b3, b0, b1, b2, b3, ...]
        uint8_view = int32_view.view(torch.uint8).reshape(n, 4)  # [N, 4], zero-copy view
        byte012 = uint8_view[:, :3].contiguous()  # [N, 3], copies only 3 bytes per element

        return byte012.reshape(-1), packed3, n

    def _decompress_tensor(self, byte012, packed3, codebook, bits, n):
        """Decompress back to flat FP32."""
        # Unpack byte3
        if bits <= 6:
            indices = unpack_6bit(packed3, n)
        elif bits <= 5:
            indices = unpack_5bit(packed3, n)
        else:
            indices = packed3[:n]

        byte3 = codebook[indices.long()]
        del indices

        # Reconstruct from byte012 + byte3
        b012 = byte012.reshape(n, 3)
        result = torch.zeros(n, dtype=torch.int32, device=byte012.device)
        result |= b012[:, 0].to(torch.int32)
        result |= b012[:, 1].to(torch.int32) << 8
        result |= b012[:, 2].to(torch.int32) << 16
        result |= byte3.to(torch.int32) << 24

        return result.view(torch.float32)

    def _compress_states(self):
        if not self._initialized:
            self._init()

        device = self._params[0].device

        # --- Compress m ---
        flat_m = self._gather('exp_avg')
        # Rebuild codebook every step (new byte3 values may appear)
        byte3_m = ((flat_m.view(torch.int32) >> 24) & 0xFF).to(torch.uint8)
        self._m_codebook, self._m_lut, self._m_bits = self._build_codebook(byte3_m)
        del byte3_m
        self._m_byte012, self._m_packed3, _ = self._compress_tensor(
            flat_m, self._m_codebook, self._m_lut, self._m_bits)
        del flat_m

        # Free per-param m
        for p in self._params:
            self.optimizer.state[p]['exp_avg'] = torch.empty(0, dtype=torch.float32, device=device)

        # --- Compress v ---
        flat_v = self._gather('exp_avg_sq')
        byte3_v = ((flat_v.view(torch.int32) >> 24) & 0xFF).to(torch.uint8)
        self._v_codebook, self._v_lut, self._v_bits = self._build_codebook(byte3_v)
        del byte3_v
        self._v_byte012, self._v_packed3, _ = self._compress_tensor(
            flat_v, self._v_codebook, self._v_lut, self._v_bits)
        del flat_v

        # Free per-param v
        for p in self._params:
            self.optimizer.state[p]['exp_avg_sq'] = torch.empty(0, dtype=torch.float32, device=device)

        self._is_compressed = True

    def _decompress_states(self):
        # Decompress m
        flat_m = self._decompress_tensor(
            self._m_byte012, self._m_packed3, self._m_codebook, self._m_bits, self._total_n)
        self._scatter(flat_m, 'exp_avg')
        self._m_byte012 = None
        self._m_packed3 = None
        del flat_m

        # Decompress v
        flat_v = self._decompress_tensor(
            self._v_byte012, self._v_packed3, self._v_codebook, self._v_bits, self._total_n)
        self._scatter(flat_v, 'exp_avg_sq')
        self._v_byte012 = None
        self._v_packed3 = None
        del flat_v

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
        total_c = 0
        total_o = self._total_n * 4 * 2  # m + v, FP32
        for byte012, packed3, cb, bits, label in [
            (self._m_byte012, self._m_packed3, self._m_codebook, self._m_bits, 'm'),
            (self._v_byte012, self._v_packed3, self._v_codebook, self._v_bits, 'v'),
        ]:
            if byte012 is not None:
                c = byte012.numel() + packed3.numel() + cb.numel()
                total_c += c
                o = self._total_n * 4
                print(f"  {label}: {len(cb)} unique → {bits}b, "
                      f"{c/1024**2:.0f} MB compressed, {o/1024**2:.0f} MB original, "
                      f"ratio={c/o*100:.1f}%")
        if total_c > 0:
            return {
                'compressed_bytes': total_c,
                'original_bytes': total_o,
                'ratio': total_c / total_o,
                'savings_mb': (total_o - total_c) / 1024**2,
            }
        return None


def verify_lossless(model_name="Qwen/Qwen3-0.6B"):
    print("--- Losslessness Verification ---")

    # Unit tests
    for n in [1, 3, 4, 100, 10001, 1000000]:
        data = torch.randint(0, 64, (n,), dtype=torch.uint8, device='cuda')
        packed = pack_6bit(data)
        restored = unpack_6bit(packed, n)
        assert torch.all(data == restored), f"6-bit FAILED n={n}"

    for n in [1, 7, 8, 100, 1000001]:
        data = torch.randint(0, 32, (n,), dtype=torch.uint8, device='cuda')
        packed = pack_5bit(data)
        restored = unpack_5bit(packed, n)
        assert torch.all(data == restored), f"5-bit FAILED n={n}"
    print("  ✓ Pack/unpack round-trips OK")

    # Full optimizer test
    torch.manual_seed(42)
    m1 = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    m1.train()
    o1 = torch.optim.AdamW(m1.parameters(), lr=1e-4, weight_decay=0.01)

    torch.manual_seed(42)
    m2 = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    m2.train()
    inner = torch.optim.AdamW(m2.parameters(), lr=1e-4, weight_decay=0.01)
    o2 = LeanCompressedAdamW(inner)

    for step_i in range(5):
        torch.manual_seed(step_i + 100)
        ids = torch.randint(100, 10000, (2, 128), device='cuda')
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            m1(input_ids=ids, labels=ids).loss.backward()
        o1.step(); o1.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            m2(input_ids=ids, labels=ids).loss.backward()
        o2.step(); o2.zero_grad()

    max_diff = max(
        (p1.data - p2.data).abs().max().item()
        for p1, p2 in zip(m1.parameters(), m2.parameters())
    )
    print(f"  Max param diff: {max_diff}" + (" ✓ LOSSLESS" if max_diff == 0 else " ✗ FAILED"))

    del m1, m2, o1, o2, inner
    gc.collect(); torch.cuda.empty_cache()
    return max_diff == 0


def benchmark(model_name="Qwen/Qwen3-0.6B", batch_size=4, seq_len=256):
    print("\n" + "=" * 80)
    print("Lean Byte3 Compressed FP32 Optimizer Benchmark")
    print("=" * 80)

    results = []
    n_warmup, n_measure = 10, 40

    for name, use_compressed in [("Standard FP32 AdamW", False),
                                  ("Lean Byte3 Compressed", True)]:
        print(f"\n--- {name} ---")
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
        model.train()
        inner = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        opt = LeanCompressedAdamW(inner) if use_compressed else inner

        for _ in range(n_warmup):
            ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                model(input_ids=ids, labels=ids).loss.backward()
            opt.step(); opt.zero_grad()

        gc.collect(); torch.cuda.empty_cache()
        gpu_mem = torch.cuda.memory_allocated() / 1024**2

        if use_compressed:
            stats = opt.get_stats()
            if stats:
                print(f"  Total: ratio={stats['ratio']*100:.1f}%, savings={stats['savings_mb']:.0f} MB")

        torch.cuda.synchronize()
        times = []
        for _ in range(n_measure):
            ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                model(input_ids=ids, labels=ids).loss.backward()
            opt.step(); opt.zero_grad()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg_step = sum(times) / len(times)
        peak = torch.cuda.max_memory_allocated() / 1024**2
        tps = batch_size * seq_len / avg_step

        print(f"  Mem: {gpu_mem:.0f} MB, Peak: {peak:.0f} MB, "
              f"Step: {avg_step*1000:.1f} ms, Tok/s: {tps:.0f}")

        results.append({
            'method': name, 'gpu_mem': gpu_mem, 'peak': peak,
            'step_ms': avg_step * 1000, 'tps': tps,
        })

        del model, inner, opt
        gc.collect(); torch.cuda.empty_cache()

    bl = results[0]
    print(f"\n{'='*80}")
    print(f"{'Method':<30} {'Mem':>7} {'ΔMem':>7} {'Peak':>7} {'Step':>7} {'Slow':>5}")
    print("-" * 65)
    for r in results:
        dm = r['gpu_mem'] - bl['gpu_mem']
        s = r['step_ms'] / bl['step_ms']
        print(f"{r['method']:<30} {r['gpu_mem']:>6.0f}M {dm:>+6.0f}M {r['peak']:>6.0f}M "
              f"{r['step_ms']:>6.1f} {s:>4.2f}x")


if __name__ == '__main__':
    ok = verify_lossless()
    if ok:
        benchmark()
