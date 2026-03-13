"""
Experiment 8: Pipelined CPU Offload for AdamW

From Experiment 7: Batched CPU offload saves 2274MB with 1.40x slowdown.
The slowdown comes from ~90ms of CPU↔GPU transfer for m/v states.

Key insight: We can pipeline these transfers with forward/backward:
- After optimizer.step(): start async offload of m/v to CPU (overlaps with next forward)
- Before optimizer.step(): prefetch m/v from CPU during backward pass

Using CUDA streams for async transfers:
1. Default stream: forward/backward/optimizer
2. Transfer stream: async CPU↔GPU copies

Also try: compress on CPU after offload (lz4) to reduce CPU memory footprint.
"""

import torch
import time
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer


class PipelinedCPUOffloadAdamW:
    """AdamW with pipelined CPU offload of optimizer states.

    Uses CUDA streams to overlap CPU↔GPU transfers with computation:
    - After step: async copy m/v to CPU (overlaps with next forward pass)
    - Before step: async copy m/v to GPU (overlaps with backward pass)

    The optimizer states are stored on CPU between steps, saving GPU memory.
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._transfer_stream = torch.cuda.Stream()
        self._param_info = None
        self._cpu_m = None
        self._cpu_v = None
        self._gpu_m = None  # pre-allocated GPU buffer for restoration
        self._gpu_v = None
        self._offloaded = False
        self._prefetch_started = False

    def _init(self):
        """Initialize buffers on first step (after optimizer states exist)."""
        self._param_info = []
        offset = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p in self.optimizer.state and 'exp_avg' in self.optimizer.state[p]:
                    n = p.numel()
                    self._param_info.append((p, offset, n, p.shape))
                    offset += n

        self._total_elements = offset
        dtype = torch.bfloat16  # match optimizer state dtype

        # Pinned CPU buffers for fast async transfer
        self._cpu_m = torch.empty(self._total_elements, dtype=dtype, pin_memory=True)
        self._cpu_v = torch.empty(self._total_elements, dtype=dtype, pin_memory=True)

        # Pre-allocated GPU buffers for restoration (avoids allocation during prefetch)
        self._gpu_m = torch.empty(self._total_elements, dtype=dtype, device='cuda')
        self._gpu_v = torch.empty(self._total_elements, dtype=dtype, device='cuda')

        print(f"  PipelinedOffload: {len(self._param_info)} params, "
              f"{self._total_elements:,} elements, "
              f"{self._total_elements * 2 / 1024/1024:.0f} MB per state")

    def _offload_async(self):
        """Async copy m/v from GPU to CPU. Call after optimizer.step()."""
        with torch.cuda.stream(self._transfer_stream):
            for p, offset, n, shape in self._param_info:
                state = self.optimizer.state[p]
                self._cpu_m[offset:offset+n].copy_(state['exp_avg'].flatten(), non_blocking=True)
                self._cpu_v[offset:offset+n].copy_(state['exp_avg_sq'].flatten(), non_blocking=True)

        # We DON'T free GPU state yet — wait for transfer to complete
        self._offloaded = False  # mark as "offload in progress"

    def _finish_offload(self):
        """Wait for offload to complete, then free GPU state tensors."""
        self._transfer_stream.synchronize()
        for p, offset, n, shape in self._param_info:
            state = self.optimizer.state[p]
            state['exp_avg'] = torch.empty(0, device='cuda', dtype=torch.bfloat16)
            state['exp_avg_sq'] = torch.empty(0, device='cuda', dtype=torch.bfloat16)
        self._offloaded = True

    def _prefetch_async(self):
        """Async copy m/v from CPU to GPU. Call before optimizer.step()."""
        with torch.cuda.stream(self._transfer_stream):
            self._gpu_m.copy_(self._cpu_m, non_blocking=True)
            self._gpu_v.copy_(self._cpu_v, non_blocking=True)
        self._prefetch_started = True

    def _finish_prefetch(self):
        """Wait for prefetch to complete, scatter into optimizer states."""
        self._transfer_stream.synchronize()
        for p, offset, n, shape in self._param_info:
            state = self.optimizer.state[p]
            state['exp_avg'] = self._gpu_m[offset:offset+n].view(shape)
            state['exp_avg_sq'] = self._gpu_v[offset:offset+n].view(shape)
        self._prefetch_started = False
        self._offloaded = False

    def step(self):
        """Run optimizer step with pipelined transfers."""
        if self._param_info is None:
            # First step: just run normally, then init and offload
            self.optimizer.step()
            self._init()
            self._offload_async()
            self._finish_offload()
            return

        if self._offloaded:
            # Need to bring states back
            if not self._prefetch_started:
                self._prefetch_async()
            self._finish_prefetch()

        self.optimizer.step()

        # Start async offload (overlaps with next forward pass)
        self._offload_async()
        self._finish_offload()

    def start_prefetch(self):
        """Manually trigger prefetch (call during backward pass for best overlap)."""
        if self._offloaded and not self._prefetch_started:
            self._prefetch_async()

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups


class SimpleCPUOffloadAdamW:
    """Simple (non-pipelined) CPU offload for comparison.
    Same as Experiment 7's CPUOffloadAdamW."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._param_info = None
        self._cpu_m = None
        self._cpu_v = None
        self._offloaded = False

    def _init(self):
        self._param_info = []
        offset = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p in self.optimizer.state and 'exp_avg' in self.optimizer.state[p]:
                    n = p.numel()
                    self._param_info.append((p, offset, n, p.shape))
                    offset += n
        self._total = offset
        self._cpu_m = torch.empty(self._total, dtype=torch.bfloat16, pin_memory=True)
        self._cpu_v = torch.empty(self._total, dtype=torch.bfloat16, pin_memory=True)

    def _offload(self):
        if self._param_info is None:
            self._init()
        for p, offset, n, shape in self._param_info:
            state = self.optimizer.state[p]
            self._cpu_m[offset:offset+n].copy_(state['exp_avg'].flatten())
            self._cpu_v[offset:offset+n].copy_(state['exp_avg_sq'].flatten())
            state['exp_avg'] = torch.empty(0, device='cuda', dtype=torch.bfloat16)
            state['exp_avg_sq'] = torch.empty(0, device='cuda', dtype=torch.bfloat16)
        self._offloaded = True

    def _restore(self):
        for p, offset, n, shape in self._param_info:
            state = self.optimizer.state[p]
            state['exp_avg'] = self._cpu_m[offset:offset+n].to('cuda').view(shape)
            state['exp_avg_sq'] = self._cpu_v[offset:offset+n].to('cuda').view(shape)
        self._offloaded = False

    def step(self):
        if self._offloaded:
            self._restore()
        self.optimizer.step()
        self._offload()

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups


def verify_lossless(model_name="Qwen/Qwen3-0.6B"):
    """Verify pipelined offload is lossless."""
    print("--- Losslessness Verification ---")

    for name, Cls in [("Simple CPU Offload", SimpleCPUOffloadAdamW),
                       ("Pipelined CPU Offload", PipelinedCPUOffloadAdamW)]:
        print(f"\n  {name}...")

        torch.manual_seed(42)
        model1 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
        model1.train()
        opt1 = torch.optim.AdamW(model1.parameters(), lr=1e-4, weight_decay=0.01)

        torch.manual_seed(42)
        model2 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
        model2.train()
        inner = torch.optim.AdamW(model2.parameters(), lr=1e-4, weight_decay=0.01)
        opt2 = Cls(inner)

        for step in range(10):
            torch.manual_seed(step + 100)
            ids = torch.randint(100, 10000, (2, 128), device='cuda')

            model1(input_ids=ids, labels=ids).loss.backward()
            opt1.step(); opt1.zero_grad()

            model2(input_ids=ids, labels=ids).loss.backward()
            opt2.step(); opt2.zero_grad()

        max_diff = max(
            (p1.data.float() - p2.data.float()).abs().max().item()
            for p1, p2 in zip(model1.parameters(), model2.parameters())
        )

        print(f"  Max diff: {max_diff}" + (" ✓ LOSSLESS" if max_diff == 0 else " ✗ FAILED"))

        del model1, model2, opt1, opt2, inner
        gc.collect(); torch.cuda.empty_cache()


def benchmark(model_name="Qwen/Qwen3-0.6B", batch_size=4, seq_len=256):
    print("\n" + "="*80)
    print("Pipelined CPU Offload Benchmark")
    print("="*80)

    results = []
    n_warmup = 10
    n_measure = 50

    configs = [
        ("Standard AdamW", None),
        ("Simple CPU Offload", SimpleCPUOffloadAdamW),
        ("Pipelined CPU Offload", PipelinedCPUOffloadAdamW),
    ]

    for method_name, Cls in configs:
        print(f"\n--- {method_name} ---")
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
        model.train()

        inner_opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        opt = Cls(inner_opt) if Cls else inner_opt

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

            # For pipelined version, start prefetch during backward
            # (In real training, this would be triggered by a backward hook)
            if isinstance(opt, PipelinedCPUOffloadAdamW):
                opt.start_prefetch()

            opt.step(); opt.zero_grad()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        # Optimizer-only timing
        opt_times = []
        for _ in range(20):
            ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
            model(input_ids=ids, labels=ids).loss.backward()
            if isinstance(opt, PipelinedCPUOffloadAdamW):
                opt.start_prefetch()
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

        print(f"  Steady-state GPU: {gpu_mem:.0f} MB, Peak: {peak:.0f} MB")
        print(f"  Step: {avg_step*1000:.1f} ms, Opt: {avg_opt*1000:.1f} ms, Tok/s: {tps:.0f}")

        results.append({
            'method': method_name,
            'gpu_mem': gpu_mem,
            'peak': peak,
            'step_ms': avg_step * 1000,
            'opt_ms': avg_opt * 1000,
            'tps': tps,
        })

        del model, inner_opt, opt
        gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY (Qwen3-0.6B, batch=4, seq=256)")
    print(f"{'='*80}")
    bl = results[0]
    print(f"{'Method':<28} {'Mem':>6} {'ΔMem':>7} {'Peak':>6} {'Step':>7} {'Slow':>5} {'Tok/s':>7}")
    print("-"*68)
    for r in results:
        dm = r['gpu_mem'] - bl['gpu_mem']
        s = r['step_ms'] / bl['step_ms']
        print(f"{r['method']:<28} {r['gpu_mem']:>5.0f}M {dm:>+6.0f}M {r['peak']:>5.0f}M "
              f"{r['step_ms']:>6.1f} {s:>4.2f}x {r['tps']:>6.0f}")

    # Memory savings analysis
    print(f"\nMemory savings analysis:")
    for r in results[1:]:
        saved = bl['gpu_mem'] - r['gpu_mem']
        pct = saved / bl['gpu_mem'] * 100
        overhead = r['step_ms'] - bl['step_ms']
        print(f"  {r['method']}: saves {saved:.0f} MB ({pct:.0f}%), "
              f"overhead {overhead:.1f} ms/step ({overhead/bl['step_ms']*100:.0f}%)")


if __name__ == '__main__':
    verify_lossless()
    benchmark()
