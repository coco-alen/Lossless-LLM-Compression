"""
Rounds 21-30: Push 3-bit decode speed + try hybrid approaches.

Current bottleneck: 3-bit decode at 427 GB/s (vs 1603 GB/s 4-bit).
Root cause: inner loop `for i in range(8)` with scalar loads in _dec_3bit_v1.

Strategy:
  R21: Unroll 3-bit decode manually with direct bit extraction
  R22: Vectorized 3-bit decode: load int32, extract all 8 codes at once
  R23: Hybrid: 4-bit for K cache (needs precision), 3-bit for V cache (more tolerant)
  R24: Adaptive per-layer: use 3-bit when top-8 coverage > 99%, else 4-bit
  R25: 3-bit encode optimization: vectorize the inner loop
  R26: Combined: best encode + best decode
  R27-30: Block size and num_warps tuning on best variant
"""

import torch
import triton
import triton.language as tl
import time


def bench_kernel(fn, warmup=30, iters=300):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def pipeline_speedup(enc_gbs, dec_gbs, ratio, bw_gbs=87.0,
                     n_layers=80, per_layer_bytes=268.4e6):
    enc_ms = per_layer_bytes / (enc_gbs * 1e9) * 1000
    dec_ms = per_layer_bytes / (dec_gbs * 1e9) * 1000
    xfer_ms = per_layer_bytes / ratio / (bw_gbs * 1e9) * 1000
    bn = max(enc_ms, xfer_ms, dec_ms)
    return per_layer_bytes * n_layers / (bw_gbs * 1e9) * 1000 / (enc_ms + bn * n_layers + dec_ms)


def build_codebook(exponents, k, device):
    vals, counts = torch.unique(exponents, return_counts=True)
    si = torch.argsort(counts, descending=True)
    enc = torch.full((256,), min(k-1, 255), dtype=torch.uint8, device=device)
    dec = torch.zeros(k, dtype=torch.uint8, device=device)
    top = min(k, vals.numel())
    for i in range(top):
        enc[vals[si[i]].item()] = i
        dec[i] = vals[si[i]].item()
    # Nearest-neighbor for unmapped
    for i in range(top, vals.numel()):
        v = vals[si[i]].item()
        best_d = 999; best_c = 0
        for j in range(top):
            d = abs(v - vals[si[j]].item())
            if d < best_d: best_d = d; best_c = j
        enc[v] = best_c
    return enc, dec


# ============================================================
# R21: Vectorized 3-bit decode — extract all 8 codes from int32 at once
# ============================================================

@triton.jit
def _dec_3bit_vec(pk, sm, dlut, out, n, BLOCK: tl.constexpr):
    """Vectorized 3-bit decode: load 3 bytes as int32, extract 8 codes via shifts."""
    pid = tl.program_id(0)
    goffs = tl.arange(0, BLOCK)
    gidx = pid * BLOCK + goffs
    elem_base = gidx * 8
    valid = elem_base + 7 < n
    pk_base = gidx * 3

    b0 = tl.load(pk + pk_base, mask=valid, other=0).to(tl.int32)
    b1 = tl.load(pk + pk_base + 1, mask=valid, other=0).to(tl.int32)
    b2 = tl.load(pk + pk_base + 2, mask=valid, other=0).to(tl.int32)
    packed = (b0 << 16) | (b1 << 8) | b2

    # Extract 8 3-bit codes: bits [23:21], [20:18], ... [2:0]
    c0 = (packed >> 21) & 7; c1 = (packed >> 18) & 7
    c2 = (packed >> 15) & 7; c3 = (packed >> 12) & 7
    c4 = (packed >> 9) & 7;  c5 = (packed >> 6) & 7
    c6 = (packed >> 3) & 7;  c7 = packed & 7

    # Decode all 8 exponents
    e0 = tl.load(dlut + c0, mask=valid, other=0).to(tl.int16)
    e1 = tl.load(dlut + c1, mask=valid, other=0).to(tl.int16)
    e2 = tl.load(dlut + c2, mask=valid, other=0).to(tl.int16)
    e3 = tl.load(dlut + c3, mask=valid, other=0).to(tl.int16)
    e4 = tl.load(dlut + c4, mask=valid, other=0).to(tl.int16)
    e5 = tl.load(dlut + c5, mask=valid, other=0).to(tl.int16)
    e6 = tl.load(dlut + c6, mask=valid, other=0).to(tl.int16)
    e7 = tl.load(dlut + c7, mask=valid, other=0).to(tl.int16)

    # Load 8 sign_mantissa values and recombine
    s0 = tl.load(sm + elem_base, mask=valid, other=0).to(tl.int16)
    s1 = tl.load(sm + elem_base + 1, mask=valid, other=0).to(tl.int16)
    s2 = tl.load(sm + elem_base + 2, mask=valid, other=0).to(tl.int16)
    s3 = tl.load(sm + elem_base + 3, mask=valid, other=0).to(tl.int16)
    s4 = tl.load(sm + elem_base + 4, mask=valid, other=0).to(tl.int16)
    s5 = tl.load(sm + elem_base + 5, mask=valid, other=0).to(tl.int16)
    s6 = tl.load(sm + elem_base + 6, mask=valid, other=0).to(tl.int16)
    s7 = tl.load(sm + elem_base + 7, mask=valid, other=0).to(tl.int16)

    tl.store(out + elem_base,     ((s0&0x80)<<8)|(e0<<7)|(s0&0x7F), mask=valid)
    tl.store(out + elem_base + 1, ((s1&0x80)<<8)|(e1<<7)|(s1&0x7F), mask=valid)
    tl.store(out + elem_base + 2, ((s2&0x80)<<8)|(e2<<7)|(s2&0x7F), mask=valid)
    tl.store(out + elem_base + 3, ((s3&0x80)<<8)|(e3<<7)|(s3&0x7F), mask=valid)
    tl.store(out + elem_base + 4, ((s4&0x80)<<8)|(e4<<7)|(s4&0x7F), mask=valid)
    tl.store(out + elem_base + 5, ((s5&0x80)<<8)|(e5<<7)|(s5&0x7F), mask=valid)
    tl.store(out + elem_base + 6, ((s6&0x80)<<8)|(e6<<7)|(s6&0x7F), mask=valid)
    tl.store(out + elem_base + 7, ((s7&0x80)<<8)|(e7<<7)|(s7&0x7F), mask=valid)


# ============================================================
# R22: Vectorized 3-bit encode — fully unrolled
# ============================================================

@triton.jit
def _enc_3bit_vec(inp, lut, pk, sm, n, BLOCK: tl.constexpr):
    """Vectorized 3-bit encode: load 8 elements, pack into 3 bytes."""
    pid = tl.program_id(0)
    goffs = tl.arange(0, BLOCK)
    gidx = pid * BLOCK + goffs
    elem_base = gidx * 8
    valid = elem_base + 7 < n
    pk_base = gidx * 3

    # Load 8 BF16 values
    v0 = tl.load(inp + elem_base, mask=valid, other=0).to(tl.int16)
    v1 = tl.load(inp + elem_base+1, mask=valid, other=0).to(tl.int16)
    v2 = tl.load(inp + elem_base+2, mask=valid, other=0).to(tl.int16)
    v3 = tl.load(inp + elem_base+3, mask=valid, other=0).to(tl.int16)
    v4 = tl.load(inp + elem_base+4, mask=valid, other=0).to(tl.int16)
    v5 = tl.load(inp + elem_base+5, mask=valid, other=0).to(tl.int16)
    v6 = tl.load(inp + elem_base+6, mask=valid, other=0).to(tl.int16)
    v7 = tl.load(inp + elem_base+7, mask=valid, other=0).to(tl.int16)

    # Extract exponents and look up 3-bit codes
    c0 = tl.load(lut + ((v0>>7)&0xFF).to(tl.int32), mask=valid, other=7).to(tl.int32)
    c1 = tl.load(lut + ((v1>>7)&0xFF).to(tl.int32), mask=valid, other=7).to(tl.int32)
    c2 = tl.load(lut + ((v2>>7)&0xFF).to(tl.int32), mask=valid, other=7).to(tl.int32)
    c3 = tl.load(lut + ((v3>>7)&0xFF).to(tl.int32), mask=valid, other=7).to(tl.int32)
    c4 = tl.load(lut + ((v4>>7)&0xFF).to(tl.int32), mask=valid, other=7).to(tl.int32)
    c5 = tl.load(lut + ((v5>>7)&0xFF).to(tl.int32), mask=valid, other=7).to(tl.int32)
    c6 = tl.load(lut + ((v6>>7)&0xFF).to(tl.int32), mask=valid, other=7).to(tl.int32)
    c7 = tl.load(lut + ((v7>>7)&0xFF).to(tl.int32), mask=valid, other=7).to(tl.int32)

    # Pack: [c0:3][c1:3][c2:3][c3:3][c4:3][c5:3][c6:3][c7:3] = 24 bits
    packed = (c0<<21)|(c1<<18)|(c2<<15)|(c3<<12)|(c4<<9)|(c5<<6)|(c6<<3)|c7
    tl.store(pk + pk_base,     ((packed>>16)&0xFF).to(tl.uint8), mask=valid)
    tl.store(pk + pk_base + 1, ((packed>>8)&0xFF).to(tl.uint8), mask=valid)
    tl.store(pk + pk_base + 2, (packed&0xFF).to(tl.uint8), mask=valid)

    # Store 8 sign_mantissa values
    tl.store(sm + elem_base,   (((v0>>8)&0x80)|(v0&0x7F)).to(tl.uint8), mask=valid)
    tl.store(sm + elem_base+1, (((v1>>8)&0x80)|(v1&0x7F)).to(tl.uint8), mask=valid)
    tl.store(sm + elem_base+2, (((v2>>8)&0x80)|(v2&0x7F)).to(tl.uint8), mask=valid)
    tl.store(sm + elem_base+3, (((v3>>8)&0x80)|(v3&0x7F)).to(tl.uint8), mask=valid)
    tl.store(sm + elem_base+4, (((v4>>8)&0x80)|(v4&0x7F)).to(tl.uint8), mask=valid)
    tl.store(sm + elem_base+5, (((v5>>8)&0x80)|(v5&0x7F)).to(tl.uint8), mask=valid)
    tl.store(sm + elem_base+6, (((v6>>8)&0x80)|(v6&0x7F)).to(tl.uint8), mask=valid)
    tl.store(sm + elem_base+7, (((v7>>8)&0x80)|(v7&0x7F)).to(tl.uint8), mask=valid)


# ============================================================
# 4-bit wide kernels (for comparison)
# ============================================================

@triton.jit
def _enc_4bit_wide(inp, lut, pk, sm, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    po = tl.arange(0, BLOCK)
    b = pid * BLOCK
    for step in range(4):
        pi = b + step * BLOCK + po
        ei = pi * 2; oi = ei + 1
        em = ei < n; om = oi < n
        v0 = tl.load(inp + ei, mask=em, other=0).to(tl.int16)
        v1 = tl.load(inp + oi, mask=om, other=0).to(tl.int16)
        i0 = tl.load(lut + ((v0>>7)&0xFF).to(tl.int32), mask=em, other=15).to(tl.uint8)
        i1 = tl.load(lut + ((v1>>7)&0xFF).to(tl.int32), mask=om, other=15).to(tl.uint8)
        tl.store(pk + pi, (i0 << 4) | i1, mask=em)
        tl.store(sm + ei, (((v0>>8)&0x80)|(v0&0x7F)).to(tl.uint8), mask=em)
        tl.store(sm + oi, (((v1>>8)&0x80)|(v1&0x7F)).to(tl.uint8), mask=om)

@triton.jit
def _dec_4bit_wide(pk, sm, dlut, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    po = tl.arange(0, BLOCK)
    b = pid * BLOCK
    for step in range(4):
        pi = b + step * BLOCK + po
        ei = pi * 2; oi = ei + 1
        em = ei < n; om = oi < n
        packed = tl.load(pk + pi, mask=em, other=0)
        e0 = tl.load(dlut + ((packed>>4)&0x0F).to(tl.int32), mask=em, other=0).to(tl.int16)
        e1 = tl.load(dlut + (packed&0x0F).to(tl.int32), mask=om, other=0).to(tl.int16)
        s0 = tl.load(sm + ei, mask=em, other=0).to(tl.int16)
        s1 = tl.load(sm + oi, mask=om, other=0).to(tl.int16)
        tl.store(out + ei, ((s0&0x80)<<8)|(e0<<7)|(s0&0x7F), mask=em)
        tl.store(out + oi, ((s1&0x80)<<8)|(e1<<7)|(s1&0x7F), mask=om)


# ============================================================
# Main
# ============================================================

def main():
    device = 'cuda'
    print(f"GPU: {torch.cuda.get_device_name()}")

    results = []

    for size_mb in [64, 268]:
        n = int(size_mb * 1024 * 1024 / 2)
        kv = torch.randn(n, dtype=torch.bfloat16, device=device)
        int16 = kv.view(torch.int16)
        nbytes = n * 2
        exponents = ((int16 >> 7) & 0xFF).to(torch.uint8)

        enc8, dec8 = build_codebook(exponents, 8, device)
        enc16, dec16 = build_codebook(exponents, 16, device)

        print(f"\n{'='*120}")
        print(f"SIZE: {size_mb} MB")
        print(f"{'='*120}")
        print(f"{'Round':<6} {'Name':<42} {'Enc GB/s':>9} {'Dec GB/s':>9} {'Ratio':>7} "
              f"{'Err%':>8} {'87G SP':>8} {'190G SP':>8}")
        print("-" * 105)

        def run(name, enc_fn, dec_fn, ratio, pk_bytes):
            pk = torch.empty(pk_bytes, dtype=torch.uint8, device=device)
            sm = torch.empty(n, dtype=torch.uint8, device=device)
            out = torch.empty(n, dtype=torch.int16, device=device)

            for _ in range(30):
                enc_fn(int16, pk, sm); dec_fn(pk, sm, out)

            et = bench_kernel(lambda: enc_fn(int16, pk, sm))
            enc_fn(int16, pk, sm)
            dt = bench_kernel(lambda: dec_fn(pk, sm, out))
            enc_fn(int16, pk, sm); dec_fn(pk, sm, out)
            mis = (int16 != out).sum().item()

            eg = nbytes/et/1e9; dg = nbytes/dt/1e9
            sp87 = pipeline_speedup(eg, dg, ratio, 87.0)
            sp190 = pipeline_speedup(eg, dg, ratio, 190.0)
            r = {'name': name, 'enc_gbs': eg, 'dec_gbs': dg, 'ratio': ratio,
                 'err': mis/n, 'mis': mis, 'sp87': sp87, 'sp190': sp190, 'size': size_mb}
            results.append(r)
            best = " ★" if sp87 > 1.44 and dg > 600 else ""
            print(f"{name[:4]:<6} {name:<42} {eg:>8.0f} {dg:>8.0f} {ratio:>6.3f}x "
                  f"{mis/n*100:>7.4f}% {sp87:>7.3f}x {sp190:>7.3f}x{best}")

        n3 = n * 3 // 8  # 3-bit packed bytes
        n4 = n // 2      # 4-bit packed bytes

        # ---- 4-bit baselines ----
        B = 256
        g4 = lambda: ((n//2 + B*4 - 1) // (B*4),)
        run("R21: 4-bit wide B=256 top-16",
            lambda i,p,s: _enc_4bit_wide[g4()](i, enc16, p, s, n, BLOCK=B),
            lambda p,s,o: _dec_4bit_wide[g4()](p, s, dec16, o, n, BLOCK=B),
            1.333, n4)

        # ---- 3-bit with loop (old) — skipped, use vectorized instead ----

        # ---- 3-bit vectorized ----
        for BK in [64, 128, 256, 512, 1024]:
            g3 = lambda BK=BK: ((n//8 + BK - 1) // BK,)
            run(f"R23: 3-bit vec enc+dec B={BK}",
                lambda i,p,s,BK=BK: _enc_3bit_vec[g3(BK)](i, enc8, p, s, n, BLOCK=BK),
                lambda p,s,o,BK=BK: _dec_3bit_vec[g3(BK)](p, s, dec8, o, n, BLOCK=BK),
                1.455, n3)

        # ---- 3-bit vec encode + 3-bit vec decode, best BLOCK ----
        # (already covered above)

    # ---- Final Summary ----
    print(f"\n{'='*120}")
    print("FINAL RANKING (sorted by pipeline speedup on 87 GB/s)")
    print("=" * 120)

    # Filter to 268MB results
    r268 = [r for r in results if r['size'] == 268]
    r268.sort(key=lambda r: r['sp87'], reverse=True)

    print(f"{'#':<3} {'Name':<42} {'Enc':>6} {'Dec':>6} {'Ratio':>7} {'Err%':>8} {'87G':>7} {'190G':>7}")
    print("-" * 90)
    for i, r in enumerate(r268):
        m = " ★" if i == 0 else ""
        print(f"{i+1:<3} {r['name']:<42} {r['enc_gbs']:>5.0f} {r['dec_gbs']:>5.0f} "
              f"{r['ratio']:>6.3f}x {r['err']*100:>7.4f}% {r['sp87']:>6.3f}x {r['sp190']:>6.3f}x{m}")

    best = r268[0]
    print(f"\n★ BEST @ 268MB: {best['name']}")
    print(f"  Encode: {best['enc_gbs']:.0f} GB/s, Decode: {best['dec_gbs']:.0f} GB/s")
    print(f"  Ratio: {best['ratio']:.3f}x, Error: {best['err']*100:.4f}%")
    print(f"  Pipeline: 87G→{best['sp87']:.3f}x, 190G→{best['sp190']:.3f}x")


if __name__ == "__main__":
    main()
