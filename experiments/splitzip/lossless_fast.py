"""
Fast Truly Lossless SplitZip Codec
====================================
Addresses reviewer feedback: must be genuinely lossless, not "near-lossless."

Key insight: escape rate is only 0.01-0.02%. That's ~3000-6000 elements
out of 33M per 64MB layer. The escape stream is tiny (<30 KB).

Strategy: Two-kernel encode.
  Kernel 1: Fast 4-bit nibble pack + sign_mantissa (the same 1680 GB/s kernel)
  Kernel 2: Triton kernel that scans the original exponents for values outside
            the top-16 common set, collecting (position, raw_exponent) pairs
            into a small escape buffer.

Escaped elements use an arbitrary dummy 4-bit code in the packed stream. Decode
first reconstructs from the packed code, then _fix_escapes overwrites those
positions with the exact original exponent.
"""

import torch
import triton
import triton.language as tl
import time


# ====== Kernel 1: 4-bit wide encode (same as best from opt_rounds) ======

@triton.jit
def _enc_4bit(inp, lut, pk, sm, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    po = tl.arange(0, BLOCK)
    b = pid * BLOCK * 4  # FIX: each program handles 4*BLOCK pairs
    for step in range(4):
        pi = b + step * BLOCK + po
        ei = pi * 2; oi = ei + 1
        em = ei < n; om = oi < n
        v0 = tl.load(inp + ei, mask=em, other=0).to(tl.int16)
        v1 = tl.load(inp + oi, mask=om, other=0).to(tl.int16)
        i0 = tl.load(lut + ((v0>>7)&0xFF).to(tl.int32), mask=em, other=0).to(tl.uint8)
        i1 = tl.load(lut + ((v1>>7)&0xFF).to(tl.int32), mask=om, other=0).to(tl.uint8)
        tl.store(pk + pi, (i0 << 4) | i1, mask=em)
        tl.store(sm + ei, (((v0>>8)&0x80)|(v0&0x7F)).to(tl.uint8), mask=em)
        tl.store(sm + oi, (((v1>>8)&0x80)|(v1&0x7F)).to(tl.uint8), mask=om)


# ====== Kernel 2: Escape collection ======
# Scans raw exponents. For each exponent outside the top-16 common set, records
# (element_position, raw_exponent).
# Uses atomic counter for output positioning.

@triton.jit
def _collect_escapes(inp, common_lut,
                     esc_pos, esc_val, esc_count,
                     n_pairs, n, BLOCK: tl.constexpr):
    """Collect uncommon exponents into compact arrays."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_pairs

    hi_pos = offs * 2
    lo_pos = offs * 2 + 1

    hi_raw = tl.load(inp + hi_pos, mask=mask, other=0).to(tl.int32)
    lo_mask = mask & (lo_pos < n)
    lo_raw = tl.load(inp + lo_pos, mask=lo_mask, other=0).to(tl.int32)
    hi_exp = (hi_raw >> 7) & 0xFF
    lo_exp = (lo_raw >> 7) & 0xFF

    hi_common = tl.load(common_lut + hi_exp, mask=mask, other=1).to(tl.int32)
    lo_common = tl.load(common_lut + lo_exp, mask=lo_mask, other=1).to(tl.int32)
    hi_esc = (hi_common == 0) & mask
    lo_esc = (lo_common == 0) & lo_mask

    hi_i = hi_esc.to(tl.int32)
    lo_i = lo_esc.to(tl.int32)
    n_hi = tl.sum(hi_i, axis=0)
    n_lo = tl.sum(lo_i, axis=0)
    n_total = n_hi + n_lo
    base_pos = tl.atomic_add(esc_count, n_total)

    hi_rank = tl.cumsum(hi_i, 0) - 1
    lo_rank = n_hi + tl.cumsum(lo_i, 0) - 1

    tl.store(esc_pos + base_pos + hi_rank, hi_pos.to(tl.int32), mask=hi_esc)
    tl.store(esc_val + base_pos + hi_rank, hi_exp.to(tl.uint8), mask=hi_esc)
    tl.store(esc_pos + base_pos + lo_rank, lo_pos.to(tl.int32), mask=lo_esc)
    tl.store(esc_val + base_pos + lo_rank, lo_exp.to(tl.uint8), mask=lo_esc)


# ====== Kernel 3: 4-bit decode (same fast kernel) ======

@triton.jit
def _dec_4bit(pk, sm, dlut, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    po = tl.arange(0, BLOCK)
    b = pid * BLOCK * 4  # FIX: each program handles 4*BLOCK pairs
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


# ====== Kernel 4: Fix escapes in decoded output ======

@triton.jit
def _fix_escapes(esc_pos, esc_val, sm, out, n_esc, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_esc
    pos = tl.load(esc_pos + offs, mask=mask, other=0)
    exp = tl.load(esc_val + offs, mask=mask, other=0).to(tl.int16)
    s = tl.load(sm + pos, mask=mask, other=0).to(tl.int16)
    tl.store(out + pos, ((s & 0x80) << 8) | (exp << 7) | (s & 0x7F), mask=mask)


# ====== Codec class ======

class FastLosslessCodec:
    def __init__(self, device='cuda'):
        self.device = device
        self.enc_lut = None
        self.dec_lut = None
        self.common_lut = None

    def calibrate(self, sample):
        int16 = sample.contiguous().view(torch.int16)
        exponents = ((int16 >> 7) & 0xFF).to(torch.uint8)
        vals, counts = torch.unique(exponents, return_counts=True)
        si = torch.argsort(counts, descending=True)
        self.enc_lut = torch.zeros((256,), dtype=torch.uint8, device=self.device)
        self.dec_lut = torch.zeros(16, dtype=torch.uint8, device=self.device)
        self.common_lut = torch.zeros((256,), dtype=torch.uint8, device=self.device)
        n_common = min(16, vals.numel())
        for i in range(n_common):
            value = vals[si[i]].item()
            self.enc_lut[value] = i
            self.dec_lut[i] = value
            self.common_lut[value] = 1
        cov = counts[si[:n_common]].sum().item() / counts.sum().item()
        return cov

    def encode(self, tensor):
        n = tensor.numel()
        int16 = tensor.contiguous().view(torch.int16)
        n_pairs = n // 2

        pk = torch.empty(n_pairs, dtype=torch.uint8, device=self.device)
        sm = torch.empty(n, dtype=torch.uint8, device=self.device)

        # Kernel 1: Fast 4-bit encode
        B = 256
        g = ((n_pairs + B*4 - 1) // (B*4),)
        _enc_4bit[g](int16, self.enc_lut, pk, sm, n, BLOCK=B)

        # Kernel 2: vectorized escape compaction from original exponents.
        max_esc = max(n // 10, 1024)
        esc_pos_buf = torch.empty(max_esc, dtype=torch.int32, device=self.device)
        esc_val_buf = torch.empty(max_esc, dtype=torch.uint8, device=self.device)
        esc_count = torch.zeros((), dtype=torch.int32, device=self.device)
        C = 128
        cg = ((n_pairs + C - 1) // C,)
        _collect_escapes[cg](int16, self.common_lut,
                             esc_pos_buf, esc_val_buf, esc_count,
                             n_pairs, n, BLOCK=C)
        n_esc = esc_count.item()
        if n_esc > max_esc:
            raise RuntimeError(f"escape buffer too small: {n_esc} > {max_esc}")
        return pk, sm, esc_pos_buf[:n_esc], esc_val_buf[:n_esc], n, n_esc

    def decode(self, pk, sm, esc_pos, esc_val, n, n_esc):
        output = torch.empty(n, dtype=torch.int16, device=self.device)
        n_pairs = n // 2

        # Kernel 3: Fast 4-bit decode
        B = 256
        g = ((n_pairs + B*4 - 1) // (B*4),)
        _dec_4bit[g](pk, sm, self.dec_lut, output, n, BLOCK=B)

        # Kernel 4: Fix escapes
        if n_esc > 0:
            fg = ((n_esc + 255) // 256,)
            _fix_escapes[fg](esc_pos, esc_val, sm, output, n_esc, BLOCK=256)

        return output.view(torch.bfloat16)


def bench(fn, warmup=30, iters=300):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    device = 'cuda'
    print(f"GPU: {torch.cuda.get_device_name()}")

    codec = FastLosslessCodec(device)

    print(f"\n{'='*110}")
    print("FAST TRULY LOSSLESS 4-BIT CODEC")
    print(f"{'='*110}")
    print(f"{'Size':>6} {'Enc ms':>8} {'Dec ms':>8} {'Enc GB/s':>9} {'Dec GB/s':>9} "
          f"{'Ratio':>7} {'Escapes':>10} {'Correct':>8}")
    print("-" * 75)

    # Calibrate ONCE on a large representative sample
    calib = torch.randn(64 * 1024 * 1024 // 2, dtype=torch.bfloat16, device=device)
    cov = codec.calibrate(calib)
    print(f"Codebook coverage: {cov*100:.3f}%\n")

    for size_mb in [4, 16, 64, 268]:
        n = int(size_mb * 1024 * 1024 / 2)
        kv = torch.randn(n, dtype=torch.bfloat16, device=device)
        nbytes = n * 2

        # Warmup
        for _ in range(20):
            r = codec.encode(kv)
            codec.decode(*r)

        # Verify
        r = codec.encode(kv)
        decoded = codec.decode(*r)
        correct = torch.equal(kv.view(torch.int16), decoded.view(torch.int16))
        n_esc = r[5]

        # Benchmark
        et = bench(lambda: codec.encode(kv))
        r = codec.encode(kv)
        dt = bench(lambda: codec.decode(*r))

        comp = r[0].numel() + r[1].numel() + r[2].numel() * 4 + r[3].numel()
        ratio = nbytes / comp

        eg = nbytes / et / 1e9
        dg = nbytes / dt / 1e9

        print(f"{size_mb:>5}M {et*1000:>7.3f} {dt*1000:>7.3f} {eg:>8.0f} {dg:>8.0f} "
              f"{ratio:>6.3f}x {n_esc:>10} {'PASS ✓' if correct else 'FAIL ✗':>8}")

    # Pipeline speedup comparison
    print(f"\n{'='*110}")
    print("PIPELINE SPEEDUP: Lossless vs Near-Lossless (Llama-70B 64K, 80 layers)")
    print(f"{'='*110}")

    n = 268 * 1024 * 1024 // 2  # per-layer
    kv = torch.randn(n, dtype=torch.bfloat16, device=device)
    nbytes = n * 2
    codec.calibrate(kv)

    for _ in range(20):
        codec.encode(kv); codec.decode(*codec.encode(kv))

    et = bench(lambda: codec.encode(kv), iters=200)
    r = codec.encode(kv)
    dt = bench(lambda: codec.decode(*r), iters=200)

    eg = nbytes / et / 1e9
    dg = nbytes / dt / 1e9
    comp = r[0].numel() + r[1].numel() + r[2].numel() * 4 + r[3].numel()
    ratio = nbytes / comp

    print(f"Lossless: enc={eg:.0f} GB/s, dec={dg:.0f} GB/s, ratio={ratio:.3f}x")
    print(f"Escapes per layer: {r[5]} ({r[5]/n*100:.4f}%)")
    print(f"Escape stream overhead: {r[2].numel()*4 + r[3].numel()} bytes ({(r[2].numel()*4 + r[3].numel())/nbytes*100:.4f}%)")

    total_kv = nbytes * 80  # 80 layers
    print(f"\n{'Network':>18} {'Raw ms':>8} {'Lossless pipe':>14} {'Speedup':>8}")
    print("-" * 52)

    for name, bw in [("GPU-Direct(15)", 15), ("CPU-RDMA(47)", 47),
                      ("RoCE4x200(87)", 87), ("RoCE8x400(190)", 190)]:
        raw = total_kv / (bw * 1e9) * 1000
        enc_ms = nbytes / (eg * 1e9) * 1000
        dec_ms = nbytes / (dg * 1e9) * 1000
        xfer_ms = nbytes / ratio / (bw * 1e9) * 1000
        bn = max(enc_ms, xfer_ms, dec_ms)
        pipe = enc_ms + bn * 80 + dec_ms
        sp = raw / pipe
        print(f"{name:>18} {raw:>7.1f} {pipe:>13.1f} {sp:>7.3f}x")


if __name__ == "__main__":
    main()
