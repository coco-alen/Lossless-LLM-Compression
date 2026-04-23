"""
Native FP8 exact top-8 benchmark with compact block-local escapes.

This format uses all eight 3-bit exponent codes for the common FP8 exponents
and stores uncommon exponents in a separate compact stream:

  - E4M3 sign + mantissa: 4 bits/element
  - E5M2 sign + mantissa: 3 bits/element
  - common exponent code: 3 bits/element
  - escape counts: 1 byte per 256-element block
  - E4M3 escape payload: uint8 local offset + 4-bit original exponent
  - E5M2 escape payload: uint8 local offset + packed 5-bit original exponent

The codec is exact with respect to native FP8 bytes.
"""

import argparse
import os
import sys
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from experiments.splitzip.fp8_fixed_codec_bench import (
    FP8FixedCodec,
    load_model_kv_bf16,
    raw_fp8_from_bf16,
)


@triton.jit
def _count_external_fp8(raw, common_lut, counts, n: tl.constexpr,
                        FMT: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    r = tl.load(raw + offs, mask=mask, other=0).to(tl.int32)
    if FMT == 0:
        exp = (r >> 3) & 0x0F
    else:
        exp = (r >> 2) & 0x1F
    common = tl.load(common_lut + exp, mask=mask, other=1).to(tl.int32)
    esc = (common == 0) & mask
    count = tl.sum(esc.to(tl.int32), axis=0)
    tl.store(counts + pid, count)


@triton.jit
def _write_external_fp8(raw, common_lut, starts, local_pos, esc_val,
                        n: tl.constexpr, FMT: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base_elem = pid * BLOCK
    offs = base_elem + tl.arange(0, BLOCK)
    mask = offs < n
    r = tl.load(raw + offs, mask=mask, other=0).to(tl.int32)
    if FMT == 0:
        exp = (r >> 3) & 0x0F
    else:
        exp = (r >> 2) & 0x1F
    common = tl.load(common_lut + exp, mask=mask, other=1).to(tl.int32)
    esc = (common == 0) & mask
    rank = tl.cumsum(esc.to(tl.int32), 0) - 1
    start = tl.load(starts + pid)
    out_idx = start + rank
    local = offs - base_elem
    tl.store(local_pos + out_idx, local.to(tl.uint8), mask=esc)
    tl.store(esc_val + out_idx, exp.to(tl.uint8), mask=esc)


@triton.jit
def _fix_external_fp8(counts, starts, local_pos, esc_val, out,
                      FMT: tl.constexpr, BLOCK_ELEMS: tl.constexpr,
                      BLOCK_ESC: tl.constexpr):
    pid = tl.program_id(0)
    count = tl.load(counts + pid)
    start = tl.load(starts + pid)
    offs = tl.arange(0, BLOCK_ESC)
    mask = offs < count
    idx = start + offs
    local = tl.load(local_pos + idx, mask=mask, other=0).to(tl.int32)
    exp = tl.load(esc_val + idx, mask=mask, other=0).to(tl.int32)
    pos = pid * BLOCK_ELEMS + local
    raw = tl.load(out + pos, mask=mask, other=0).to(tl.int32)
    if FMT == 0:
        fixed = (raw & 0x87) | (exp << 3)
    else:
        fixed = (raw & 0x83) | (exp << 2)
    tl.store(out + pos, fixed.to(tl.uint8), mask=mask)


@triton.jit
def _pack_e4_escape_values(esc_val, esc_val_packed, n_esc,
                           BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    elem0 = offs * 2
    elem1 = elem0 + 1
    mask0 = elem0 < n_esc
    mask1 = elem1 < n_esc
    v0 = tl.load(esc_val + elem0, mask=mask0, other=0).to(tl.int32) & 0x0F
    v1 = tl.load(esc_val + elem1, mask=mask1, other=0).to(tl.int32) & 0x0F
    tl.store(esc_val_packed + offs, ((v0 << 4) | v1).to(tl.uint8), mask=mask0)


@triton.jit
def _fix_external_e4_packed(counts, starts, local_pos, esc_val_packed, out,
                            BLOCK_ELEMS: tl.constexpr, BLOCK_ESC: tl.constexpr):
    pid = tl.program_id(0)
    count = tl.load(counts + pid)
    start = tl.load(starts + pid)
    offs = tl.arange(0, BLOCK_ESC)
    mask = offs < count
    idx = start + offs
    local = tl.load(local_pos + idx, mask=mask, other=0).to(tl.int32)
    packed = tl.load(esc_val_packed + (idx // 2), mask=mask, other=0).to(tl.int32)
    exp = tl.where((idx & 1) == 0, (packed >> 4) & 0x0F, packed & 0x0F)
    pos = pid * BLOCK_ELEMS + local
    raw = tl.load(out + pos, mask=mask, other=0).to(tl.int32)
    tl.store(out + pos, ((raw & 0x87) | (exp << 3)).to(tl.uint8), mask=mask)


@triton.jit
def _pack_e5_escape_values(esc_val, esc_val_packed, n_esc,
                           BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    elem = offs * 8
    mask = elem < n_esc

    v0 = tl.load(esc_val + elem, mask=mask, other=0).to(tl.int32) & 0x1F
    v1 = tl.load(esc_val + elem + 1, mask=elem + 1 < n_esc, other=0).to(tl.int32) & 0x1F
    v2 = tl.load(esc_val + elem + 2, mask=elem + 2 < n_esc, other=0).to(tl.int32) & 0x1F
    v3 = tl.load(esc_val + elem + 3, mask=elem + 3 < n_esc, other=0).to(tl.int32) & 0x1F
    v4 = tl.load(esc_val + elem + 4, mask=elem + 4 < n_esc, other=0).to(tl.int32) & 0x1F
    v5 = tl.load(esc_val + elem + 5, mask=elem + 5 < n_esc, other=0).to(tl.int32) & 0x1F
    v6 = tl.load(esc_val + elem + 6, mask=elem + 6 < n_esc, other=0).to(tl.int32) & 0x1F
    v7 = tl.load(esc_val + elem + 7, mask=elem + 7 < n_esc, other=0).to(tl.int32) & 0x1F

    out = offs * 5
    b0 = v0 | ((v1 & 0x07) << 5)
    b1 = ((v1 >> 3) & 0x03) | (v2 << 2) | ((v3 & 0x01) << 7)
    b2 = ((v3 >> 1) & 0x0F) | ((v4 & 0x0F) << 4)
    b3 = ((v4 >> 4) & 0x01) | (v5 << 1) | ((v6 & 0x03) << 6)
    b4 = ((v6 >> 2) & 0x07) | (v7 << 3)

    tl.store(esc_val_packed + out, b0.to(tl.uint8), mask=mask)
    tl.store(esc_val_packed + out + 1, b1.to(tl.uint8), mask=mask)
    tl.store(esc_val_packed + out + 2, b2.to(tl.uint8), mask=mask)
    tl.store(esc_val_packed + out + 3, b3.to(tl.uint8), mask=mask)
    tl.store(esc_val_packed + out + 4, b4.to(tl.uint8), mask=mask)


@triton.jit
def _fix_external_e5_packed(counts, starts, local_pos, esc_val_packed, out,
                            BLOCK_ELEMS: tl.constexpr, BLOCK_ESC: tl.constexpr):
    pid = tl.program_id(0)
    count = tl.load(counts + pid)
    start = tl.load(starts + pid)
    offs = tl.arange(0, BLOCK_ESC)
    mask = offs < count
    idx = start + offs
    local = tl.load(local_pos + idx, mask=mask, other=0).to(tl.int32)
    bit_pos = idx * 5
    byte_idx = bit_pos // 8
    shift = bit_pos & 7
    b0 = tl.load(esc_val_packed + byte_idx, mask=mask, other=0).to(tl.int32)
    b1 = tl.load(esc_val_packed + byte_idx + 1, mask=mask, other=0).to(tl.int32)
    exp = ((b0 | (b1 << 8)) >> shift) & 0x1F
    pos = pid * BLOCK_ELEMS + local
    raw = tl.load(out + pos, mask=mask, other=0).to(tl.int32)
    tl.store(out + pos, ((raw & 0x83) | (exp << 2)).to(tl.uint8), mask=mask)


def bench(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def next_power_of_2(x):
    return 1 << (max(1, int(x)) - 1).bit_length()


class FP8Top8CompactCodec:
    def __init__(self, fmt, raw, block=128, escape_block=256):
        self.fmt = fmt
        self.fmt_id = 0 if fmt == "e4m3" else 1
        self.raw = raw
        self.n = raw.numel()
        self.device = raw.device
        self.escape_block = escape_block
        self.core = FP8FixedCodec(fmt, raw, lossless=False, strategy="freq", block=block)
        lut_size = 16 if fmt == "e4m3" else 32
        self.common_lut = torch.zeros((lut_size,), dtype=torch.uint8, device=self.device)
        self.common_lut[self.core.dlut.long()] = 1
        self.n_blocks = (self.n + escape_block - 1) // escape_block
        self.counts = torch.empty((self.n_blocks,), dtype=torch.int32, device=self.device)
        self.starts = torch.empty((self.n_blocks,), dtype=torch.int32, device=self.device)
        self.max_esc = max(self.n // 5, 1024)
        self.local_pos = torch.empty((self.max_esc,), dtype=torch.uint8, device=self.device)
        self.esc_val = torch.empty((self.max_esc,), dtype=torch.uint8, device=self.device)
        if fmt == "e4m3":
            packed_size = (self.max_esc + 1) // 2
        else:
            packed_size = (self.max_esc * 5 + 7) // 8 + 1
        self.esc_val_packed = torch.empty((packed_size,), dtype=torch.uint8,
                                          device=self.device)
        self.count_grid = (self.n_blocks,)

    def encode_core(self):
        self.core.encode_core(self.raw)

    def count_escapes(self):
        _count_external_fp8[self.count_grid](
            self.raw, self.common_lut, self.counts, self.n,
            FMT=self.fmt_id, BLOCK=self.escape_block)

    def write_escapes(self):
        _write_external_fp8[self.count_grid](
            self.raw, self.common_lut, self.starts, self.local_pos, self.esc_val,
            self.n, FMT=self.fmt_id, BLOCK=self.escape_block)

    def pack_escapes(self, n_esc):
        if self.fmt == "e4m3":
            grid = (((n_esc + 1) // 2 + 255) // 256,)
            _pack_e4_escape_values[grid](self.esc_val, self.esc_val_packed, n_esc, BLOCK=256)
        else:
            grid = (((n_esc + 7) // 8 + 255) // 256,)
            _pack_e5_escape_values[grid](self.esc_val, self.esc_val_packed, n_esc, BLOCK=256)

    def encode_full(self):
        self.encode_core()
        self.count_escapes()
        offsets = torch.cumsum(self.counts, dim=0)
        self.starts = offsets - self.counts
        n_esc = offsets[-1].item()
        if n_esc > self.max_esc:
            raise RuntimeError(f"escape buffer too small: {n_esc} > {self.max_esc}")
        self.write_escapes()
        self.pack_escapes(n_esc)
        return n_esc

    def decode_full(self, max_count):
        self.core.decode_core()
        block_esc = next_power_of_2(max_count)
        if self.fmt == "e4m3":
            _fix_external_e4_packed[self.count_grid](
                self.counts, self.starts, self.local_pos, self.esc_val_packed,
                self.core.out, BLOCK_ELEMS=self.escape_block, BLOCK_ESC=block_esc)
        else:
            _fix_external_e5_packed[self.count_grid](
                self.counts, self.starts, self.local_pos, self.esc_val_packed,
                self.core.out, BLOCK_ELEMS=self.escape_block, BLOCK_ESC=block_esc)

    def compressed_bytes(self, n_esc):
        if self.fmt == "e4m3":
            escape_bytes = n_esc + ((n_esc + 1) // 2)
        else:
            escape_bytes = n_esc + ((n_esc * 5 + 7) // 8)
        return self.core.code_bytes + self.core.sm_bytes + self.n_blocks + escape_bytes


def make_raw(fmt, size_mb, scale, device):
    n = size_mb * 1024 * 1024
    bf16 = (torch.randn(n, dtype=torch.bfloat16, device=device) * scale).contiguous()
    if fmt == "e4m3":
        return bf16.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    return bf16.to(torch.float8_e5m2).view(torch.uint8).contiguous()


def pipeline_speedup(raw_bytes, comp_bytes, enc_s, dec_s, bw_gbs=87.0, n_layers=80):
    raw_total = raw_bytes * n_layers / (bw_gbs * 1e9)
    xfer = comp_bytes / (bw_gbs * 1e9)
    stage = max(enc_s, xfer, dec_s)
    total = enc_s + stage * n_layers + dec_s
    return raw_total / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmt", choices=["e4m3", "e5m2"], default="e5m2")
    parser.add_argument("--size-mb", type=int, default=134)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--escape-block", type=int, default=256)
    args = parser.parse_args()
    if args.escape_block > 256:
        raise ValueError("--escape-block must be <= 256 because local offsets are stored as uint8")

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name()}")
    if args.model:
        bf16 = load_model_kv_bf16(args.model, args.max_new_tokens, device)
        raw = raw_fp8_from_bf16(args.fmt, bf16, args.size_mb)
    else:
        raw = make_raw(args.fmt, args.size_mb, args.scale, device)

    codec = FP8Top8CompactCodec(args.fmt, raw, block=args.block, escape_block=args.escape_block)
    n_esc = codec.encode_full()
    max_count = codec.counts.max().item()
    codec.decode_full(max_count)
    correct = torch.equal(raw, codec.core.out)
    comp_bytes = codec.compressed_bytes(n_esc)
    ratio = raw.numel() / comp_bytes

    enc_core_s = bench(codec.encode_core, iters=args.iters)
    count_s = bench(codec.count_escapes, iters=args.iters)
    enc_full_s = bench(codec.encode_full, iters=max(20, args.iters // 2))
    n_esc = codec.encode_full()
    max_count = codec.counts.max().item()
    dec_s = bench(lambda: codec.decode_full(max_count), iters=args.iters)

    raw_bytes = raw.numel()
    print()
    print(f"{args.fmt.upper()} top-8 exact compact escapes")
    print(f"  Native FP8 size: {raw_bytes / 1024 / 1024:.1f} MB")
    print(f"  Top-8 coverage: {codec.core.coverage * 100:.3f}%")
    print(f"  Escapes: {n_esc} ({n_esc / raw.numel() * 100:.4f}%), max/block {max_count}")
    print(f"  Ratio vs FP8: {ratio:.3f}x")
    print(f"  Total vs BF16: {ratio * 2:.3f}x")
    print(f"  Correct: {'PASS' if correct else 'FAIL'}")
    print(f"  Encode core: {raw_bytes / enc_core_s / 1e9:.0f} GB/s")
    print(f"  Escape count: {raw_bytes / count_s / 1e9:.0f} GB/s")
    print(f"  Encode full: {raw_bytes / enc_full_s / 1e9:.0f} GB/s")
    print(f"  Decode full: {raw_bytes / dec_s / 1e9:.0f} GB/s")
    print(f"  Speedup vs raw FP8 @87GB/s: "
          f"{pipeline_speedup(raw_bytes, comp_bytes, enc_full_s, dec_s, 87.0):.3f}x")
    print(f"  Speedup vs raw BF16 @87GB/s: "
          f"{pipeline_speedup(raw_bytes * 2, comp_bytes, enc_full_s, dec_s, 87.0):.3f}x")
    print(f"  Speedup vs raw FP8 @190GB/s: "
          f"{pipeline_speedup(raw_bytes, comp_bytes, enc_full_s, dec_s, 190.0):.3f}x")
    print(f"  Speedup vs raw BF16 @190GB/s: "
          f"{pipeline_speedup(raw_bytes * 2, comp_bytes, enc_full_s, dec_s, 190.0):.3f}x")


if __name__ == "__main__":
    main()
