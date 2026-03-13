"""
Experiment 7: Wrapper-based Compressed AdamW (Revised)

Wrap PyTorch's native AdamW. Compress/decompress states per-parameter but
use fast GPU byte operations instead of Python-loop overhead.

Key findings from previous experiments:
- Per-parameter CPU offload: 24x slower (too many small transfers)
- GPU byte-plane for BF16: only ~25% savings possible
- Full concatenation: OOM risk

New approach: Per-parameter GPU compression using int16 byte operations.
Process parameters in chunks to avoid OOM. The key is that int16 byte ops
are very fast on GPU (1ms per 100M elements from microbenchmark).

For BF16 optimizer states (2 bytes each):
- We save the low byte raw and pack the high byte (4 bits → 0.5 bytes per element)
- Total: 1.5 bytes per element vs 2 bytes = 75% ratio = 25% savings
- For 596M params: save 596M * 2 states * 0.5 bytes = 596 MB

For CPU offload variant:
- Move to pinned CPU memory in one batch
- Transfer back to GPU only during optimizer step
- Saves 100% of optimizer GPU memory
"""

import torch
import time
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer


class PerParamCompressedAdamW:
    """Wraps PyTorch AdamW, compresses each param's m/v after step.

    Uses int16 byte-plane compression: store low byte raw, pack high byte
    to 4 bits using codebook (for params with ≤16 unique high bytes).
    Falls back to storing both bytes for params with >16 unique high bytes.
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._compressed = {}  # param_id -> {m_low, m_high_packed, ...}
        self._initialized = False

    def _compress_param_state(self, tensor):
        """Compress a single BF16 tensor's high byte."""
        n = tensor.numel()
        int16_view = tensor.contiguous().view(torch.int16).flatten()

        low = (int16_view & 0xFF).to(torch.uint8)
        high = ((int16_view >> 8) & 0xFF).to(torch.uint8)

        unique = torch.unique(high)
        n_unique = len(unique)

        if n_unique <= 16:
            lut = torch.zeros(256, dtype=torch.uint8, device=tensor.device)
            for i, v in enumerate(unique):
                lut[v] = i

            indices = lut[high.long()]
            # Pack 2 per byte
            if n % 2 != 0:
                indices = torch.cat([indices, torch.zeros(1, dtype=torch.uint8, device=tensor.device)])
            packed = (indices[0::2] << 4) | indices[1::2]

            return {'low': low, 'packed': packed, 'codebook': unique, 'n': n,
                    'shape': tensor.shape, 'method': 'pack4'}
        else:
            return {'low': low, 'high': high, 'n': n,
                    'shape': tensor.shape, 'method': 'raw'}

    def _decompress_param_state(self, comp):
        """Decompress back to BF16 tensor."""
        n = comp['n']
        low = comp['low']

        if comp['method'] == 'pack4':
            pk = comp['packed']
            codebook = comp['codebook']
            hi_idx = torch.zeros(len(pk) * 2, dtype=torch.uint8, device=low.device)
            hi_idx[0::2] = (pk >> 4) & 0x0F
            hi_idx[1::2] = pk & 0x0F
            hi_idx = hi_idx[:n]
            high = codebook[hi_idx.long()]
        else:
            high = comp['high']

        recon = (high.to(torch.int16) << 8) | low.to(torch.int16)
        return recon.view(torch.bfloat16).reshape(comp['shape'])

    def _compressed_size(self, comp):
        total = comp['low'].numel()
        if comp['method'] == 'pack4':
            total += comp['packed'].numel() + comp['codebook'].numel()
        else:
            total += comp['high'].numel()
        return total

    def step(self):
        # Decompress all states before step
        for group in self.optimizer.param_groups:
            for p in group['params']:
                pid = id(p)
                if pid in self._compressed:
                    comp = self._compressed[pid]
                    state = self.optimizer.state[p]
                    state['exp_avg'] = self._decompress_param_state(comp['m'])
                    state['exp_avg_sq'] = self._decompress_param_state(comp['v'])
                    del self._compressed[pid]

        # Run normal step
        self.optimizer.step()

        # Compress all states after step
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p not in self.optimizer.state:
                    continue
                state = self.optimizer.state[p]
                if 'exp_avg' not in state:
                    continue

                pid = id(p)
                self._compressed[pid] = {
                    'm': self._compress_param_state(state['exp_avg']),
                    'v': self._compress_param_state(state['exp_avg_sq']),
                }
                # Free original tensors
                state['exp_avg'] = torch.empty(0, device=p.device, dtype=p.dtype)
                state['exp_avg_sq'] = torch.empty(0, device=p.device, dtype=p.dtype)

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def get_stats(self):
        total_comp = 0
        total_orig = 0
        n_pack4 = 0
        n_raw = 0
        for pid, comp in self._compressed.items():
            for key in ['m', 'v']:
                c = comp[key]
                total_comp += self._compressed_size(c)
                total_orig += c['n'] * 2  # bf16 = 2 bytes
                if c['method'] == 'pack4':
                    n_pack4 += 1
                else:
                    n_raw += 1
        return {
            'compressed_bytes': total_comp,
            'original_bytes': total_orig,
            'ratio': total_comp / max(total_orig, 1),
            'n_pack4': n_pack4,
            'n_raw': n_raw,
        }


class CPUOffloadAdamW:
    """Wraps PyTorch AdamW, offloads all states to CPU between steps.

    Uses a single flat buffer for all m states and another for all v states.
    Transfers are batched (one big memcpy instead of per-parameter).
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._cpu_m = None
        self._cpu_v = None
        self._param_info = None  # [(param, offset, numel, shape), ...]
        self._total_elements = 0
        self._offloaded = False

    def _init_offload(self):
        """Build parameter mapping and allocate CPU buffers."""
        self._param_info = []
        offset = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p in self.optimizer.state and 'exp_avg' in self.optimizer.state[p]:
                    n = p.numel()
                    self._param_info.append((p, offset, n, p.shape))
                    offset += n
        self._total_elements = offset
        # Use pinned memory for fast GPU→CPU and CPU→GPU transfers
        self._cpu_m = torch.empty(self._total_elements, dtype=torch.bfloat16,
                                   pin_memory=True)
        self._cpu_v = torch.empty(self._total_elements, dtype=torch.bfloat16,
                                   pin_memory=True)

    def _offload_to_cpu(self):
        """Gather all m/v into CPU buffers."""
        if self._param_info is None:
            self._init_offload()

        for p, offset, n, shape in self._param_info:
            state = self.optimizer.state[p]
            self._cpu_m[offset:offset + n].copy_(state['exp_avg'].flatten(), non_blocking=True)
            self._cpu_v[offset:offset + n].copy_(state['exp_avg_sq'].flatten(), non_blocking=True)
            # Free GPU memory
            state['exp_avg'] = torch.empty(0, device=p.device, dtype=p.dtype)
            state['exp_avg_sq'] = torch.empty(0, device=p.device, dtype=p.dtype)

        torch.cuda.synchronize()
        self._offloaded = True

    def _restore_from_cpu(self):
        """Scatter CPU buffers back to GPU optimizer states."""
        for p, offset, n, shape in self._param_info:
            state = self.optimizer.state[p]
            state['exp_avg'] = self._cpu_m[offset:offset + n].to(
                p.device, non_blocking=True).view(shape)
            state['exp_avg_sq'] = self._cpu_v[offset:offset + n].to(
                p.device, non_blocking=True).view(shape)
        torch.cuda.synchronize()
        self._offloaded = False

    def step(self):
        if self._offloaded:
            self._restore_from_cpu()
        self.optimizer.step()
        self._offload_to_cpu()

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups


def verify_lossless(model_name="Qwen/Qwen3-0.6B"):
    """Verify both wrapper approaches are lossless."""
    print("--- Losslessness Verification ---")

    for wrapper_name, WrapperClass in [("PerParamCompressed", PerParamCompressedAdamW),
                                        ("CPUOffload", CPUOffloadAdamW)]:
        print(f"\n  Testing {wrapper_name}...")
        torch.manual_seed(42)
        model1 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
        model1.train()
        opt1 = torch.optim.AdamW(model1.parameters(), lr=1e-4, weight_decay=0.01)

        torch.manual_seed(42)
        model2 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
        model2.train()
        inner = torch.optim.AdamW(model2.parameters(), lr=1e-4, weight_decay=0.01)
        opt2 = WrapperClass(inner)

        for step in range(10):
            torch.manual_seed(step + 100)
            input_ids = torch.randint(100, 10000, (2, 128), device='cuda')

            model1(input_ids=input_ids, labels=input_ids).loss.backward()
            opt1.step(); opt1.zero_grad()

            model2(input_ids=input_ids, labels=input_ids).loss.backward()
            opt2.step(); opt2.zero_grad()

        max_diff = 0
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            diff = (p1.data.float() - p2.data.float()).abs().max().item()
            max_diff = max(max_diff, diff)

        if max_diff == 0:
            print(f"  {wrapper_name}: VERIFIED lossless!")
        else:
            print(f"  {wrapper_name}: FAILED! max_diff={max_diff}")

        del model1, model2, opt1, opt2, inner
        gc.collect(); torch.cuda.empty_cache()


def benchmark(model_name="Qwen/Qwen3-0.6B", batch_size=4, seq_len=256):
    print("\n" + "=" * 80)
    print("Wrapper-based Compressed AdamW Benchmark")
    print("=" * 80)

    results = []
    n_warmup = 5
    n_measure = 30

    configs = [
        ("Standard AdamW", None),
        ("PerParam GPU Pack", PerParamCompressedAdamW),
        ("Batched CPU Offload", CPUOffloadAdamW),
    ]

    for method_name, WrapperClass in configs:
        print(f"\n--- {method_name} ---")
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
        model.train()
        n_params = sum(p.numel() for p in model.parameters())

        inner_opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        if WrapperClass:
            opt = WrapperClass(inner_opt)
        else:
            opt = inner_opt

        # Warmup
        for _ in range(n_warmup):
            ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
            model(input_ids=ids, labels=ids).loss.backward()
            opt.step(); opt.zero_grad()

        gc.collect(); torch.cuda.empty_cache()
        gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024

        # Measure full step time
        torch.cuda.synchronize()
        times = []
        for _ in range(n_measure):
            ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(input_ids=ids, labels=ids).loss.backward()
            opt.step(); opt.zero_grad()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        # Measure optimizer-only time
        opt_times = []
        for _ in range(20):
            ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
            model(input_ids=ids, labels=ids).loss.backward()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            opt.step()
            torch.cuda.synchronize()
            opt_times.append(time.perf_counter() - t0)
            opt.zero_grad()

        avg_step = sum(times) / len(times)
        avg_opt = sum(opt_times) / len(opt_times)
        tps = batch_size * seq_len / avg_step
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024

        print(f"  GPU mem (steady state): {gpu_mem:.1f} MB")
        print(f"  Peak GPU mem: {peak:.1f} MB")
        print(f"  Step: {avg_step*1000:.2f} ms, Opt: {avg_opt*1000:.2f} ms, Tok/s: {tps:.0f}")

        if WrapperClass == PerParamCompressedAdamW:
            stats = opt.get_stats()
            print(f"  Compression: {stats['ratio']*100:.1f}%, pack4={stats['n_pack4']}, raw={stats['n_raw']}")

        results.append({
            'method': method_name,
            'gpu_mem': gpu_mem,
            'peak_mem': peak,
            'step_ms': avg_step * 1000,
            'opt_ms': avg_opt * 1000,
            'tps': tps,
        })

        del model, inner_opt, opt
        gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    baseline = results[0]
    print(f"{'Method':<25} {'Mem(MB)':>8} {'Peak':>8} {'ΔMem':>8} {'Step':>8} {'Slow':>6} {'Tok/s':>8}")
    print("-" * 74)
    for r in results:
        delta = r['gpu_mem'] - baseline['gpu_mem']
        slow = r['step_ms'] / baseline['step_ms']
        print(f"{r['method']:<25} {r['gpu_mem']:>7.0f} {r['peak_mem']:>7.0f} {delta:>+7.0f} "
              f"{r['step_ms']:>7.1f} {slow:>5.2f}x {r['tps']:>7.0f}")


if __name__ == '__main__':
    verify_lossless()
    benchmark()
