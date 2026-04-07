"""Quick measurement of KV page decompression latency vs attention read time."""
import torch, time
import triton, triton.language as tl

@triton.jit
def _recomb(exp_ptr, sm_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    e = tl.load(exp_ptr + offs, mask=m, other=0).to(tl.int16)
    s = tl.load(sm_ptr + offs, mask=m, other=0).to(tl.int16)
    tl.store(out_ptr + offs, ((s & 0x80) << 8) | (e << 7) | (s & 0x7F), mask=m)

device = 'cuda'
configs = [
    ('page_16tok_4kv', 4, 16, 128),
    ('page_64tok_4kv', 4, 64, 128),
    ('page_256tok_4kv', 4, 256, 128),
    ('1k_tokens_4kv', 4, 1024, 128),
    ('4k_tokens_4kv', 4, 4096, 128),
    ('page_16tok_8kv', 8, 16, 128),    # Llama-70B
    ('4k_tokens_8kv', 8, 4096, 128),   # Llama-70B
    ('16k_tokens_8kv', 8, 16384, 128), # Long context
]

print(f"{'Config':<25} {'Size':>10} {'Decomp us':>10} {'BW GB/s':>10} {'Attn read us':>13}")
print("-" * 72)

for name, h, s, d in configs:
    n = h * s * d
    nbytes = n * 2
    exp = torch.randint(0, 255, (n,), dtype=torch.uint8, device=device)
    sm = torch.randint(0, 255, (n,), dtype=torch.uint8, device=device)
    out = torch.empty(n, dtype=torch.int16, device=device)
    BLOCK = 1024
    grid = ((n + BLOCK - 1) // BLOCK,)

    for _ in range(50):
        _recomb[grid](exp, sm, out, n, BLOCK=BLOCK)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 2000
    for _ in range(iters):
        _recomb[grid](exp, sm, out, n, BLOCK=BLOCK)
    torch.cuda.synchronize()
    decomp_us = (time.perf_counter() - t0) / iters * 1e6

    # FlashAttention reads K+V (2x KV bytes) from HBM; H200 HBM BW = 4.8 TB/s
    attn_read_us = nbytes * 2 / 4.8e12 * 1e6

    bw = nbytes / (decomp_us / 1e6) / 1e9
    print(f'{name:<25} {nbytes/1024:>8.1f}KB {decomp_us:>9.2f} {bw:>9.1f} {attn_read_us:>12.3f}')

print()
print("If Decomp <= Attn_read, decompression overhead is hidden by attention HBM access time")
print("H200 HBM bandwidth: 4.8 TB/s")
