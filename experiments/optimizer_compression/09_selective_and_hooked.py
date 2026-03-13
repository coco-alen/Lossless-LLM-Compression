"""
Experiment 9: Selective Offload + Backward-Hook Pipelining

Two optimizations to improve on the simple CPU offload:

A) Selective offload: Only offload v (second moment). Keep m on GPU.
   - v changes slowly (β₂=0.999) so transfer cost is "wasted" less
   - Halves transfer volume: 1137 MB instead of 2274 MB
   - Saves 1137 MB GPU memory (50% of original savings but 50% less overhead)

B) Backward-hook prefetch: Register a hook on the last layer's backward pass
   to start CPU→GPU transfer early. This overlaps with backward computation.
   - Backward takes ~120ms; transfer takes ~45ms
   - Could hide most of the prefetch latency

C) Combined: selective offload + backward-hook prefetch.
"""

import torch
import time
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer


class SelectiveOffloadAdamW:
    """Offload only v (second moment) to CPU, keep m on GPU.

    v changes slowly (β₂=0.999) and takes half the optimizer memory.
    This halves transfer volume and overhead.
    """

    def __init__(self, optimizer, offload='v'):
        self.optimizer = optimizer
        self.offload = offload  # 'v', 'm', or 'both'
        self._param_info = None
        self._cpu_bufs = {}
        self._total = 0
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
        dtype = torch.bfloat16

        targets = []
        if self.offload in ('v', 'both'):
            targets.append('v')
        if self.offload in ('m', 'both'):
            targets.append('m')

        for t in targets:
            self._cpu_bufs[t] = torch.empty(self._total, dtype=dtype, pin_memory=True)

    def _do_offload(self):
        if self._param_info is None:
            self._init()

        for p, offset, n, shape in self._param_info:
            state = self.optimizer.state[p]
            if 'v' in self._cpu_bufs:
                self._cpu_bufs['v'][offset:offset+n].copy_(state['exp_avg_sq'].flatten())
                state['exp_avg_sq'] = torch.empty(0, device='cuda', dtype=torch.bfloat16)
            if 'm' in self._cpu_bufs:
                self._cpu_bufs['m'][offset:offset+n].copy_(state['exp_avg'].flatten())
                state['exp_avg'] = torch.empty(0, device='cuda', dtype=torch.bfloat16)
        self._offloaded = True

    def _do_restore(self):
        for p, offset, n, shape in self._param_info:
            state = self.optimizer.state[p]
            if 'v' in self._cpu_bufs:
                state['exp_avg_sq'] = self._cpu_bufs['v'][offset:offset+n].to('cuda').view(shape)
            if 'm' in self._cpu_bufs:
                state['exp_avg'] = self._cpu_bufs['m'][offset:offset+n].to('cuda').view(shape)
        self._offloaded = False

    def step(self):
        if self._offloaded:
            self._do_restore()
        self.optimizer.step()
        self._do_offload()

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups


class HookedOffloadAdamW:
    """CPU offload with backward-hook-based prefetch.

    Registers a hook on a specified module's backward pass to start
    CPU→GPU transfer early, overlapping with backward computation.
    """

    def __init__(self, optimizer, model, offload='both'):
        self.optimizer = optimizer
        self.model = model
        self.offload = offload
        self._param_info = None
        self._cpu_bufs = {}
        self._total = 0
        self._offloaded = False
        self._prefetching = False
        self._transfer_stream = torch.cuda.Stream()

        # Register backward hook on an early layer to trigger prefetch
        # Hook triggers when backward pass reaches this layer
        self._hook_handle = None
        self._register_prefetch_hook()

    def _register_prefetch_hook(self):
        """Register hook on a middle layer of the model."""
        # Find layers
        layers = None
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h

        if layers is not None and len(layers) > 2:
            # Hook on the layer at ~1/3 from the end (backward processes from last to first)
            hook_idx = len(layers) - len(layers) // 3
            target_layer = layers[hook_idx]

            def hook_fn(module, grad_input, grad_output):
                if self._offloaded and not self._prefetching:
                    self._start_prefetch()

            self._hook_handle = target_layer.register_full_backward_hook(hook_fn)
            print(f"  Registered prefetch hook on layer {hook_idx}/{len(layers)}")

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
        dtype = torch.bfloat16

        targets = ['m', 'v'] if self.offload == 'both' else [self.offload]
        for t in targets:
            self._cpu_bufs[t] = torch.empty(self._total, dtype=dtype, pin_memory=True)

    def _do_offload(self):
        """Synchronous offload to CPU."""
        if self._param_info is None:
            self._init()

        for p, offset, n, shape in self._param_info:
            state = self.optimizer.state[p]
            if 'v' in self._cpu_bufs:
                self._cpu_bufs['v'][offset:offset+n].copy_(state['exp_avg_sq'].flatten())
                state['exp_avg_sq'] = torch.empty(0, device='cuda', dtype=torch.bfloat16)
            if 'm' in self._cpu_bufs:
                self._cpu_bufs['m'][offset:offset+n].copy_(state['exp_avg'].flatten())
                state['exp_avg'] = torch.empty(0, device='cuda', dtype=torch.bfloat16)
        self._offloaded = True

    def _start_prefetch(self):
        """Start async CPU→GPU transfer."""
        self._prefetching = True
        # Allocate GPU buffers
        self._gpu_bufs = {}
        with torch.cuda.stream(self._transfer_stream):
            for key in self._cpu_bufs:
                self._gpu_bufs[key] = torch.empty(self._total, dtype=torch.bfloat16, device='cuda')
                self._gpu_bufs[key].copy_(self._cpu_bufs[key], non_blocking=True)

    def _finish_prefetch(self):
        """Wait for prefetch and scatter into optimizer states."""
        self._transfer_stream.synchronize()
        for p, offset, n, shape in self._param_info:
            state = self.optimizer.state[p]
            if 'v' in self._gpu_bufs:
                state['exp_avg_sq'] = self._gpu_bufs['v'][offset:offset+n].view(shape)
            if 'm' in self._gpu_bufs:
                state['exp_avg'] = self._gpu_bufs['m'][offset:offset+n].view(shape)
        self._prefetching = False
        self._offloaded = False
        self._gpu_bufs = {}

    def step(self):
        if self._offloaded:
            if not self._prefetching:
                self._start_prefetch()
            self._finish_prefetch()
        self.optimizer.step()
        self._do_offload()

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups


def verify_lossless(model_name="Qwen/Qwen3-0.6B"):
    """Verify all approaches are lossless."""
    print("--- Losslessness Verification ---")

    configs = [
        ("Selective (v only)", lambda opt, model: SelectiveOffloadAdamW(opt, offload='v')),
        ("Selective (both)", lambda opt, model: SelectiveOffloadAdamW(opt, offload='both')),
        ("Hooked (both)", lambda opt, model: HookedOffloadAdamW(opt, model, offload='both')),
        ("Hooked (v only)", lambda opt, model: HookedOffloadAdamW(opt, model, offload='v')),
    ]

    for name, make_opt in configs:
        print(f"\n  {name}...", end=" ")

        torch.manual_seed(42)
        m1 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
        m1.train()
        o1 = torch.optim.AdamW(m1.parameters(), lr=1e-4, weight_decay=0.01)

        torch.manual_seed(42)
        m2 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
        m2.train()
        inner = torch.optim.AdamW(m2.parameters(), lr=1e-4, weight_decay=0.01)
        o2 = make_opt(inner, m2)

        for step in range(10):
            torch.manual_seed(step + 100)
            ids = torch.randint(100, 10000, (2, 128), device='cuda')

            m1(input_ids=ids, labels=ids).loss.backward()
            o1.step(); o1.zero_grad()

            m2(input_ids=ids, labels=ids).loss.backward()
            o2.step(); o2.zero_grad()

        max_diff = max(
            (p1.data.float() - p2.data.float()).abs().max().item()
            for p1, p2 in zip(m1.parameters(), m2.parameters())
        )
        print("✓ LOSSLESS" if max_diff == 0 else f"✗ FAILED (diff={max_diff})")

        del m1, m2, o1, o2, inner
        gc.collect(); torch.cuda.empty_cache()


def benchmark(model_name="Qwen/Qwen3-0.6B", batch_size=4, seq_len=256):
    print("\n" + "="*80)
    print("Selective + Hooked Offload Benchmark")
    print("="*80)

    results = []
    n_warmup = 10
    n_measure = 50

    configs = [
        ("Standard AdamW", None),
        ("Offload v only", lambda opt, model: SelectiveOffloadAdamW(opt, offload='v')),
        ("Offload both", lambda opt, model: SelectiveOffloadAdamW(opt, offload='both')),
        ("Hooked offload both", lambda opt, model: HookedOffloadAdamW(opt, model, offload='both')),
        ("Hooked offload v", lambda opt, model: HookedOffloadAdamW(opt, model, offload='v')),
    ]

    for method_name, make_opt in configs:
        print(f"\n--- {method_name} ---")
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
        model.train()

        inner_opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        opt = make_opt(inner_opt, model) if make_opt else inner_opt

        for _ in range(n_warmup):
            ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
            model(input_ids=ids, labels=ids).loss.backward()
            opt.step(); opt.zero_grad()

        gc.collect(); torch.cuda.empty_cache()
        gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024

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

        avg_step = sum(times) / len(times)
        tps = batch_size * seq_len / avg_step
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024

        print(f"  Mem: {gpu_mem:.0f} MB, Peak: {peak:.0f} MB, "
              f"Step: {avg_step*1000:.1f} ms, Tok/s: {tps:.0f}")

        results.append({
            'method': method_name,
            'gpu_mem': gpu_mem,
            'peak': peak,
            'step_ms': avg_step * 1000,
            'tps': tps,
        })

        del model, inner_opt, opt
        gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    bl = results[0]
    print(f"{'Method':<28} {'Mem':>6} {'ΔMem':>7} {'Peak':>6} {'Step':>7} {'Slow':>5} {'Tok/s':>7}")
    print("-"*68)
    for r in results:
        dm = r['gpu_mem'] - bl['gpu_mem']
        s = r['step_ms'] / bl['step_ms']
        print(f"{r['method']:<28} {r['gpu_mem']:>5.0f}M {dm:>+6.0f}M {r['peak']:>5.0f}M "
              f"{r['step_ms']:>6.1f} {s:>4.2f}x {r['tps']:>6.0f}")


if __name__ == '__main__':
    verify_lossless()
    benchmark()
