"""
Experiment 2: Baseline AdamW Training

Measure standard AdamW training:
- Memory usage breakdown (model, optimizer states, activations)
- Training speed (tokens/sec, step time)
- This serves as the baseline for all compression experiments.
"""

import torch
import torch.nn as nn
import time
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    return torch.cuda.memory_allocated() / 1024 / 1024


def get_gpu_memory_reserved_mb():
    """Get reserved GPU memory in MB."""
    return torch.cuda.memory_reserved() / 1024 / 1024


def main():
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    print("="*80)
    print("Baseline AdamW Training Measurement")
    print("="*80)

    model_name = "Qwen/Qwen3-0.6B"

    mem_before = get_gpu_memory_mb()
    print(f"\nGPU memory before loading: {mem_before:.1f} MB")

    # Load model in bf16
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = model.cuda()
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"Trainable: {n_trainable:,} ({n_trainable/1e6:.1f}M)")

    mem_after_model = get_gpu_memory_mb()
    print(f"\nGPU memory after model load: {mem_after_model:.1f} MB")
    print(f"Model memory: {mem_after_model - mem_before:.1f} MB")
    print(f"Expected (bf16): {n_params * 2 / 1024 / 1024:.1f} MB")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    mem_after_optim_init = get_gpu_memory_mb()
    print(f"\nGPU memory after optimizer init (no states yet): {mem_after_optim_init:.1f} MB")

    # Warmup step to initialize optimizer states
    input_ids = torch.randint(100, 10000, (4, 256), device='cuda')
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Clear activation memory
    del outputs, loss
    gc.collect()
    torch.cuda.empty_cache()

    mem_after_first_step = get_gpu_memory_mb()
    optimizer_state_mem = mem_after_first_step - mem_after_model
    print(f"\nGPU memory after first optimizer step: {mem_after_first_step:.1f} MB")
    print(f"Optimizer state memory: {optimizer_state_mem:.1f} MB")
    print(f"Expected (3x FP32): {n_trainable * 12 / 1024 / 1024:.1f} MB")

    # Count actual optimizer state sizes
    total_state_elements = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state:
                state = optimizer.state[p]
                if 'exp_avg' in state:
                    total_state_elements += state['exp_avg'].numel()
                if 'exp_avg_sq' in state:
                    total_state_elements += state['exp_avg_sq'].numel()
    print(f"Optimizer state elements (m+v): {total_state_elements:,}")
    print(f"Optimizer state size (m+v, FP32): {total_state_elements * 4 / 1024 / 1024:.1f} MB")

    # Measure training speed
    print(f"\n--- Training Speed Benchmark ---")
    batch_size = 4
    seq_len = 256
    n_warmup = 5
    n_measure = 50

    # Warmup
    for _ in range(n_warmup):
        input_ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()

    # Measure
    step_times = []
    for i in range(n_measure):
        input_ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        step_times.append(t1 - t0)

    avg_time = sum(step_times) / len(step_times)
    tokens_per_step = batch_size * seq_len
    tokens_per_sec = tokens_per_step / avg_time

    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"\nBatch size: {batch_size}, Seq len: {seq_len}")
    print(f"Average step time: {avg_time*1000:.2f} ms")
    print(f"Tokens/sec: {tokens_per_sec:.0f}")
    print(f"Peak GPU memory: {peak_mem:.1f} MB")
    print(f"Step time std: {torch.tensor(step_times).std().item()*1000:.2f} ms")

    # Measure just the optimizer step time
    print(f"\n--- Optimizer Step Time Breakdown ---")

    # Prepare a gradient
    input_ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
    outputs = model(input_ids=input_ids, labels=input_ids)
    outputs.loss.backward()

    optim_times = []
    for _ in range(20):
        # We need fresh gradients each time since step modifies params
        input_ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        optimizer.zero_grad()

        optim_times.append(t1 - t0)

    avg_optim_time = sum(optim_times) / len(optim_times)
    print(f"Average optimizer.step() time: {avg_optim_time*1000:.2f} ms")
    print(f"Optimizer step as % of total: {avg_optim_time/avg_time*100:.1f}%")

    print(f"\n{'='*80}")
    print("BASELINE SUMMARY")
    print(f"{'='*80}")
    print(f"Model: {model_name} ({n_params/1e6:.1f}M params)")
    print(f"Model memory (bf16): {mem_after_model - mem_before:.1f} MB")
    print(f"Optimizer states (m+v, fp32): {total_state_elements * 4 / 1024 / 1024:.1f} MB")
    print(f"Total training memory (peak): {peak_mem:.1f} MB")
    print(f"Step time: {avg_time*1000:.2f} ms")
    print(f"Optimizer step time: {avg_optim_time*1000:.2f} ms")
    print(f"Tokens/sec: {tokens_per_sec:.0f}")


if __name__ == '__main__':
    main()
