"""
Experiment 17: Flat State Compression

Key insight: Avoid per-param Python loops entirely.
1. Pre-allocate ONE flat tensor for all m states, one for all v states
2. Each param's optimizer state is a VIEW into the flat tensor
3. After optimizer.step(): compress the ENTIRE flat tensor in ONE operation
4. Before next step: decompress in ONE operation, set up views

Compression: byte3 has ~33-47 unique values globally.
Use global codebook + vectorized bitpack on the ENTIRE flat tensor.
Single torch.unique, single LUT lookup, single bitpack call.

Expected: similar savings (~8%) but MUCH faster (no per-param loop).
"""

import torch
import time
import gc
import math
from transformers import AutoModelForCausalLM


def pack_bits_int64(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack uint8 indices at `bits` bits/value using int64. Vectorized."""
    device = indices.device
    n = indices.numel()
    remainder = n % 8
    if remainder:
        indices = torch.cat([indices, torch.zeros(8 - remainder, dtype=torch.uint8, device=device)])
    groups = indices.reshape(-1, 8).to(torch.int64)
    shifts = torch.arange(8, device=device, dtype=torch.int64) * bits
    combined = (groups << shifts.unsqueeze(0)).sum(dim=1)
    byte_shifts = torch.arange(bits, device=device, dtype=torch.int64) * 8
    packed = ((combined.unsqueeze(1) >> byte_shifts.unsqueeze(0)) & 0xFF).to(torch.uint8)
    return packed.reshape(-1)


def unpack_bits_int64(packed: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    """Unpack int64-packed uint8 indices."""
    device = packed.device
    n_groups = ((n + 7) // 8)
    groups = packed[:n_groups * bits].reshape(n_groups, bits).to(torch.int64)
    byte_shifts = torch.arange(bits, device=device, dtype=torch.int64) * 8
    combined = (groups << byte_shifts.unsqueeze(0)).sum(dim=1)
    bit_shifts = torch.arange(8, device=device, dtype=torch.int64) * bits
    mask = (1 << bits) - 1
    result = ((combined.unsqueeze(1) >> bit_shifts.unsqueeze(0)) & mask).to(torch.uint8)
    return result.reshape(-1)[:n]


class FlatStateCompressedAdamW:
    """AdamW with flat state storage and global byte3 compression."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._param_list = []
        self._param_sizes = []
        self._param_offsets = []
        self._total_n = 0
        self._initialized = False

        # Flat state tensors (used during optimizer.step)
        self._flat_m = None
        self._flat_v = None

        # Compressed storage
        self._compressed_m = None  # (byte012_flat, packed_byte3, codebook, bits)
        self._compressed_v = None
        self._is_compressed = False

    def _init_params(self):
        """Build param list and pre-allocate flat tensors."""
        params = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p not in self.optimizer.state:
                    continue
                state = self.optimizer.state[p]
                if 'exp_avg' not in state:
                    continue
                params.append(p)

        self._param_list = params
        self._param_sizes = [p.numel() for p in params]
        self._param_offsets = []
        offset = 0
        for s in self._param_sizes:
            self._param_offsets.append(offset)
            offset += s
        self._total_n = offset
        self._initialized = True

    def _gather_flat(self, key: str) -> torch.Tensor:
        """Gather state into pre-allocated flat tensor."""
        device = self._param_list[0].device
        flat = torch.empty(self._total_n, dtype=torch.float32, device=device)
        for p, offset, size in zip(self._param_list, self._param_offsets, self._param_sizes):
            flat[offset:offset+size] = self.optimizer.state[p][key].flatten()
        return flat

    def _scatter_flat(self, flat: torch.Tensor, key: str):
        """Scatter flat tensor back as views into optimizer states."""
        for p, offset, size in zip(self._param_list, self._param_offsets, self._param_sizes):
            self.optimizer.state[p][key] = flat[offset:offset+size].view(p.shape)

    def _compress_flat(self, flat_fp32: torch.Tensor):
        """Compress a flat FP32 tensor. Returns (byte012, packed_byte3, codebook, bits, n)."""
        n = flat_fp32.numel()
        int32_view = flat_fp32.view(torch.int32)

        # Extract byte planes
        byte0 = (int32_view & 0xFF).to(torch.uint8)
        byte1 = ((int32_view >> 8) & 0xFF).to(torch.uint8)
        byte2 = ((int32_view >> 16) & 0xFF).to(torch.uint8)
        byte3 = ((int32_view >> 24) & 0xFF).to(torch.uint8)

        # Global codebook for byte3
        codebook = torch.unique(byte3)
        n_unique = len(codebook)
        bits = max(1, math.ceil(math.log2(max(n_unique, 2))))

        # LUT: byte3 value → index
        lut = torch.zeros(256, dtype=torch.uint8, device=flat_fp32.device)
        lut[codebook.long()] = torch.arange(n_unique, device=flat_fp32.device, dtype=torch.uint8)
        indices = lut[byte3.long()]

        # Bit-pack indices
        packed_byte3 = pack_bits_int64(indices, bits)

        # Free intermediates
        del int32_view, byte3, indices, lut

        # Store byte012 as interleaved uint8 [N, 3]
        byte012 = torch.stack([byte0, byte1, byte2], dim=1).reshape(-1)
        del byte0, byte1, byte2

        return (byte012, packed_byte3, codebook, bits, n)

    def _decompress_flat(self, compressed) -> torch.Tensor:
        """Decompress back to flat FP32."""
        byte012, packed_byte3, codebook, bits, n = compressed
        device = byte012.device

        # Unpack byte3
        indices = unpack_bits_int64(packed_byte3, bits, n)
        byte3 = codebook[indices.long()]
        del indices

        # De-interleave byte012
        byte012_3 = byte012.reshape(n, 3)
        byte0 = byte012_3[:, 0]
        byte1 = byte012_3[:, 1]
        byte2 = byte012_3[:, 2]

        # Reconstruct int32
        result = (byte0.to(torch.int32) |
                  (byte1.to(torch.int32) << 8) |
                  (byte2.to(torch.int32) << 16) |
                  (byte3.to(torch.int32) << 24))

        return result.view(torch.float32)

    def _compress_states(self):
        if not self._initialized:
            self._init_params()

        # Gather, compress, free originals
        flat_m = self._gather_flat('exp_avg')
        self._compressed_m = self._compress_flat(flat_m)
        del flat_m

        flat_v = self._gather_flat('exp_avg_sq')
        self._compressed_v = self._compress_flat(flat_v)
        del flat_v

        # Free original states
        for p in self._param_list:
            state = self.optimizer.state[p]
            state['exp_avg'] = torch.empty(0, dtype=torch.float32, device=p.device)
            state['exp_avg_sq'] = torch.empty(0, dtype=torch.float32, device=p.device)

        self._is_compressed = True

    def _decompress_states(self):
        flat_m = self._decompress_flat(self._compressed_m)
        self._scatter_flat(flat_m, 'exp_avg')
        self._compressed_m = None
        del flat_m

        flat_v = self._decompress_flat(self._compressed_v)
        self._scatter_flat(flat_v, 'exp_avg_sq')
        self._compressed_v = None
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
        stats = {}
        for name, compressed in [('m', self._compressed_m), ('v', self._compressed_v)]:
            if compressed is None:
                continue
            byte012, packed_byte3, codebook, bits, n = compressed
            c_bytes = byte012.numel() + packed_byte3.numel() + codebook.numel()
            o_bytes = n * 4
            stats[name] = {
                'n_unique': len(codebook),
                'bits': bits,
                'compressed_bytes': c_bytes,
                'original_bytes': o_bytes,
                'ratio': c_bytes / o_bytes,
                'savings_mb': (o_bytes - c_bytes) / 1024**2,
            }
        return stats


def verify_lossless(model_name="Qwen/Qwen3-0.6B"):
    print("--- Losslessness Verification ---")

    print("  Test 1: Full optimizer (5 steps)...")
    torch.manual_seed(42)
    m1 = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    m1.train()
    o1 = torch.optim.AdamW(m1.parameters(), lr=1e-4, weight_decay=0.01)

    torch.manual_seed(42)
    m2 = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    m2.train()
    inner = torch.optim.AdamW(m2.parameters(), lr=1e-4, weight_decay=0.01)
    o2 = FlatStateCompressedAdamW(inner)

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

    if max_diff == 0:
        stats = o2.get_stats()
        for k, s in stats.items():
            print(f"  {k}: {s['n_unique']} unique → {s['bits']}b, "
                  f"ratio={s['ratio']*100:.1f}%, savings={s['savings_mb']:.0f} MB")

    del m1, m2, o1, o2, inner
    gc.collect(); torch.cuda.empty_cache()
    return max_diff == 0


def benchmark(model_name="Qwen/Qwen3-0.6B", batch_size=4, seq_len=256):
    print("\n" + "=" * 80)
    print("Flat State Compressed FP32 Optimizer Benchmark")
    print("=" * 80)

    results = []
    n_warmup, n_measure = 10, 40

    for name, use_compressed in [("Standard FP32 AdamW", False),
                                  ("FlatState Compressed", True)]:
        print(f"\n--- {name} ---")
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
        model.train()

        inner = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        opt = FlatStateCompressedAdamW(inner) if use_compressed else inner

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
                for k, s in stats.items():
                    print(f"  {k}: {s['n_unique']} unique → {s['bits']}b, "
                          f"ratio={s['ratio']*100:.1f}%, savings={s['savings_mb']:.0f} MB")

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
