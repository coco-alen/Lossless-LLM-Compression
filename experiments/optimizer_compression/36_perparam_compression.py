"""
Experiment 36: Per-Parameter Byte3 Compression

Instead of flat buffers, compress/decompress each parameter independently.
This avoids ever having the full flat FP32 + full compressed data simultaneously.

Largest param in Qwen3-0.6B is ~3M elements = 12MB FP32, so intermediates are tiny.
Tradeoff: many small kernel launches (slower) but much better peak memory.
"""

import torch
import time
import gc
from transformers import AutoModelForCausalLM


class CompressedAdamW:
    """Per-parameter lossless FP32 optimizer compression via byte3 6-bit packing."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._params = None
        self._compressed = {}  # param -> {key: (byte012, packed3, codebook, shape)}
        self._is_compressed = False
        self._first_step = True

    def _init_params(self):
        self._params = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p in self.optimizer.state and 'exp_avg' in self.optimizer.state[p]:
                    self._params.append(p)

    @staticmethod
    @torch.no_grad()
    def _compress_tensor(t):
        """Compress FP32 tensor → (byte012, packed3, codebook, shape)."""
        shape = t.shape
        flat = t.flatten()
        n = flat.numel()
        device = flat.device
        int32 = flat.view(torch.int32)

        # byte3 → LUT → indices
        byte3 = ((int32 >> 24) & 0xFF).to(torch.uint8)
        counts = torch.bincount(byte3.to(torch.int32), minlength=256)
        present = (counts > 0).nonzero(as_tuple=True)[0]
        codebook = present.to(torch.uint8)
        lut = torch.zeros(256, dtype=torch.uint8, device=device)
        lut[present] = torch.arange(len(codebook), device=device, dtype=torch.uint8)
        indices = lut[byte3.to(torch.int32)]
        del byte3, counts, lut, present

        # Pack 6-bit (4 indices → 3 bytes)
        pad = (4 - n % 4) % 4
        if pad:
            indices = torch.cat([indices, torch.zeros(pad, dtype=torch.uint8, device=device)])
        groups = indices.reshape(-1, 4)
        combined = (groups[:, 0].to(torch.int32) |
                    (groups[:, 1].to(torch.int32) << 6) |
                    (groups[:, 2].to(torch.int32) << 12) |
                    (groups[:, 3].to(torch.int32) << 18))
        del indices, groups
        packed3 = torch.stack([
            (combined & 0xFF).to(torch.uint8),
            ((combined >> 8) & 0xFF).to(torch.uint8),
            ((combined >> 16) & 0xFF).to(torch.uint8),
        ], dim=1).reshape(-1)
        del combined

        byte012 = int32.view(torch.uint8).reshape(n, 4)[:, :3].contiguous().reshape(-1)

        return byte012, packed3, codebook, shape, n

    @staticmethod
    @torch.no_grad()
    def _decompress_tensor(data):
        """Decompress (byte012, packed3, codebook, shape, n) → FP32 tensor."""
        byte012, packed3, codebook, shape, n = data

        n_groups = (n + 3) // 4
        groups = packed3[:n_groups * 3].reshape(n_groups, 3)
        combined = (groups[:, 0].to(torch.int32) |
                    (groups[:, 1].to(torch.int32) << 8) |
                    (groups[:, 2].to(torch.int32) << 16))
        del groups
        indices = torch.stack([
            (combined & 0x3F).to(torch.uint8),
            ((combined >> 6) & 0x3F).to(torch.uint8),
            ((combined >> 12) & 0x3F).to(torch.uint8),
            ((combined >> 18) & 0x3F).to(torch.uint8),
        ], dim=1).reshape(-1)[:n]
        del combined

        byte3 = codebook[indices.to(torch.int32)]
        del indices

        b = byte012.reshape(n, 3)
        result = (b[:, 0].to(torch.int32) |
                  (b[:, 1].to(torch.int32) << 8) |
                  (b[:, 2].to(torch.int32) << 16) |
                  (byte3.to(torch.int32) << 24))
        return result.view(torch.float32).view(shape)

    def _compress_states(self):
        if self._first_step:
            self._init_params()
            self._first_step = False

        device = self._params[0].device
        for p in self._params:
            state = self.optimizer.state[p]
            # Compress exp_avg
            self._compressed[id(p), 'exp_avg'] = self._compress_tensor(state['exp_avg'])
            state['exp_avg'] = torch.empty(0, dtype=torch.float32, device=device)
            # Compress exp_avg_sq
            self._compressed[id(p), 'exp_avg_sq'] = self._compress_tensor(state['exp_avg_sq'])
            state['exp_avg_sq'] = torch.empty(0, dtype=torch.float32, device=device)

        self._is_compressed = True

    def _decompress_states(self):
        for p in self._params:
            state = self.optimizer.state[p]
            # Decompress exp_avg
            key_m = (id(p), 'exp_avg')
            state['exp_avg'] = self._decompress_tensor(self._compressed[key_m])
            del self._compressed[key_m]
            # Decompress exp_avg_sq
            key_v = (id(p), 'exp_avg_sq')
            state['exp_avg_sq'] = self._decompress_tensor(self._compressed[key_v])
            del self._compressed[key_v]

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
        total_c, total_o = 0, 0
        n_unique_m, n_unique_v = set(), set()
        for (pid, key), data in self._compressed.items():
            b012, p3, cb, shape, n = data
            c = b012.numel() + p3.numel() + cb.numel()
            total_c += c
            total_o += n * 4
            if key == 'exp_avg':
                n_unique_m.update(cb.tolist())
            else:
                n_unique_v.update(cb.tolist())
        print(f"  m: {len(n_unique_m)} unique byte3, v: {len(n_unique_v)} unique byte3")
        print(f"  ratio={total_c/total_o*100:.1f}%, savings={(total_o-total_c)/1024**2:.0f} MB")
        return {'ratio': total_c / total_o, 'savings_mb': (total_o - total_c) / 1024**2}


def verify_lossless(model_name="Qwen/Qwen3-0.6B"):
    print("--- Verify ---")
    torch.manual_seed(42)
    m1 = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    m1.train()
    o1 = torch.optim.AdamW(m1.parameters(), lr=1e-4, weight_decay=0.01)

    torch.manual_seed(42)
    m2 = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    m2.train()
    inner = torch.optim.AdamW(m2.parameters(), lr=1e-4, weight_decay=0.01)
    o2 = CompressedAdamW(inner)

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
    print("Lossless FP32 Optimizer Compression (Per-Param Byte3 6-bit)")
    print("=" * 80)

    results = []
    for name, use_comp in [("Standard FP32 AdamW", False),
                            ("Compressed AdamW", True)]:
        print(f"\n--- {name} ---")
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
        model.train()
        inner = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        opt = CompressedAdamW(inner) if use_comp else inner

        for _ in range(10):
            ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                model(input_ids=ids, labels=ids).loss.backward()
            opt.step(); opt.zero_grad()

        gc.collect(); torch.cuda.empty_cache()
        gpu_mem = torch.cuda.memory_allocated() / 1024**2

        if use_comp:
            stats = opt.get_stats()

        torch.cuda.reset_peak_memory_stats()

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
    print(f"\n{'='*65}")
    print(f"{'Method':<25} {'Mem':>7} {'ΔMem':>7} {'Peak':>7} {'ΔPeak':>7} {'Step':>7} {'Slow':>5}")
    print("-" * 65)
    for r in results:
        dm = r['gpu_mem'] - bl['gpu_mem']
        dp = r['peak'] - bl['peak']
        s = r['step_ms'] / bl['step_ms']
        print(f"{r['method']:<25} {r['gpu_mem']:>6.0f}M {dm:>+6.0f}M {r['peak']:>6.0f}M "
              f"{dp:>+6.0f}M {r['step_ms']:>6.1f} {s:>4.2f}x")


if __name__ == '__main__':
    ok = verify_lossless()
    if ok:
        benchmark()
