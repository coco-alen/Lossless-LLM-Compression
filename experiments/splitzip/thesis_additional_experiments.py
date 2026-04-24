import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.splitzip.codec_ablation_bench import collect_real_activations
from experiments.splitzip.fp8_e5m2_top8_compact_bench import (
    FP8Top8CompactCodec,
    _count_external_fp8,
    _write_external_fp8,
    _pack_e5_escape_values,
    _fix_external_e5_packed,
)
from experiments.splitzip.lossless_fast import FastLosslessCodec
from experiments.splitzip.opt_rounds3 import _dec_3bit_vec, _enc_3bit_vec
from experiments.splitzip.thesis_experiment_dump import (
    assemble_row_prefix,
    load_model_activation_blocks,
    measure_dma_time,
    simulate_transport,
)


DEFAULT_JSON = ROOT / "experiments" / "splitzip" / "thesis_additional_experiments.json"
DEFAULT_MD = ROOT / "experiments" / "splitzip" / "thesis_additional_experiments.md"

REP_ACTIVATION_MODEL = "Qwen/Qwen2.5-1.5B"
REP_SHAPE = (32768, 4096)
CAL_MODEL = "Qwen/Qwen2.5-1.5B"
SERVING_MODEL = "NousResearch/Meta-Llama-3-8B"
SERVING_SEQ_LEN = 32768
SERVING_LAYER_IDX = 0
FP8_TRANSFER_MODEL = "Qwen3-32B"
FP8_TRANSFER_HF = "Qwen/Qwen3-32B"
FP8_TRANSFER_SEQ_LENS = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
FP8_TRANSFER_MODE = {"name": "RoCE 4x200G", "bandwidth_gbs": 87.0}
CAL_PROMPTS_PER_SET = 32
GRANULARITY_BENCH_ROWS = 1024


def bench_cuda(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def load_representative_bf16_matrix(device):
    mats = collect_real_activations(REP_ACTIVATION_MODEL, [REP_SHAPE[0]], str(device))
    return mats[0]


def bf16_to_raw_fp8(flat_bf16, fmt):
    if fmt == "e4m3":
        raw = flat_bf16.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    else:
        raw = flat_bf16.to(torch.float8_e5m2).view(torch.uint8).contiguous()
    usable = raw.numel() - (raw.numel() % 8)
    return raw[:usable].contiguous()


@triton.jit
def _enc_e5_top16(raw, lut, code_pk, sm_pk, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    elem = offs * 8
    valid = elem + 7 < n

    r0 = tl.load(raw + elem, mask=valid, other=0).to(tl.int32)
    r1 = tl.load(raw + elem + 1, mask=valid, other=0).to(tl.int32)
    r2 = tl.load(raw + elem + 2, mask=valid, other=0).to(tl.int32)
    r3 = tl.load(raw + elem + 3, mask=valid, other=0).to(tl.int32)
    r4 = tl.load(raw + elem + 4, mask=valid, other=0).to(tl.int32)
    r5 = tl.load(raw + elem + 5, mask=valid, other=0).to(tl.int32)
    r6 = tl.load(raw + elem + 6, mask=valid, other=0).to(tl.int32)
    r7 = tl.load(raw + elem + 7, mask=valid, other=0).to(tl.int32)

    e0 = (r0 >> 2) & 0x1F
    e1 = (r1 >> 2) & 0x1F
    e2 = (r2 >> 2) & 0x1F
    e3 = (r3 >> 2) & 0x1F
    e4 = (r4 >> 2) & 0x1F
    e5 = (r5 >> 2) & 0x1F
    e6 = (r6 >> 2) & 0x1F
    e7 = (r7 >> 2) & 0x1F

    s0 = ((r0 >> 5) & 0x04) | (r0 & 0x03)
    s1 = ((r1 >> 5) & 0x04) | (r1 & 0x03)
    s2 = ((r2 >> 5) & 0x04) | (r2 & 0x03)
    s3 = ((r3 >> 5) & 0x04) | (r3 & 0x03)
    s4 = ((r4 >> 5) & 0x04) | (r4 & 0x03)
    s5 = ((r5 >> 5) & 0x04) | (r5 & 0x03)
    s6 = ((r6 >> 5) & 0x04) | (r6 & 0x03)
    s7 = ((r7 >> 5) & 0x04) | (r7 & 0x03)

    sm_packed = (s0 << 21) | (s1 << 18) | (s2 << 15) | (s3 << 12) | \
                (s4 << 9) | (s5 << 6) | (s6 << 3) | s7
    sm_base = offs * 3
    tl.store(sm_pk + sm_base, ((sm_packed >> 16) & 0xFF).to(tl.uint8), mask=valid)
    tl.store(sm_pk + sm_base + 1, ((sm_packed >> 8) & 0xFF).to(tl.uint8), mask=valid)
    tl.store(sm_pk + sm_base + 2, (sm_packed & 0xFF).to(tl.uint8), mask=valid)

    c0 = tl.load(lut + e0, mask=valid, other=0).to(tl.int32)
    c1 = tl.load(lut + e1, mask=valid, other=0).to(tl.int32)
    c2 = tl.load(lut + e2, mask=valid, other=0).to(tl.int32)
    c3 = tl.load(lut + e3, mask=valid, other=0).to(tl.int32)
    c4 = tl.load(lut + e4, mask=valid, other=0).to(tl.int32)
    c5 = tl.load(lut + e5, mask=valid, other=0).to(tl.int32)
    c6 = tl.load(lut + e6, mask=valid, other=0).to(tl.int32)
    c7 = tl.load(lut + e7, mask=valid, other=0).to(tl.int32)

    code_base = offs * 4
    tl.store(code_pk + code_base, ((c0 << 4) | c1).to(tl.uint8), mask=valid)
    tl.store(code_pk + code_base + 1, ((c2 << 4) | c3).to(tl.uint8), mask=valid)
    tl.store(code_pk + code_base + 2, ((c4 << 4) | c5).to(tl.uint8), mask=valid)
    tl.store(code_pk + code_base + 3, ((c6 << 4) | c7).to(tl.uint8), mask=valid)


@triton.jit
def _dec_e5_top16(code_pk, sm_pk, dlut, out, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    elem = offs * 8
    valid = elem + 7 < n

    code_base = offs * 4
    p0 = tl.load(code_pk + code_base, mask=valid, other=0).to(tl.int32)
    p1 = tl.load(code_pk + code_base + 1, mask=valid, other=0).to(tl.int32)
    p2 = tl.load(code_pk + code_base + 2, mask=valid, other=0).to(tl.int32)
    p3 = tl.load(code_pk + code_base + 3, mask=valid, other=0).to(tl.int32)

    c0 = (p0 >> 4) & 0x0F
    c1 = p0 & 0x0F
    c2 = (p1 >> 4) & 0x0F
    c3 = p1 & 0x0F
    c4 = (p2 >> 4) & 0x0F
    c5 = p2 & 0x0F
    c6 = (p3 >> 4) & 0x0F
    c7 = p3 & 0x0F

    e0 = tl.load(dlut + c0, mask=valid, other=0).to(tl.int32)
    e1 = tl.load(dlut + c1, mask=valid, other=0).to(tl.int32)
    e2 = tl.load(dlut + c2, mask=valid, other=0).to(tl.int32)
    e3 = tl.load(dlut + c3, mask=valid, other=0).to(tl.int32)
    e4 = tl.load(dlut + c4, mask=valid, other=0).to(tl.int32)
    e5 = tl.load(dlut + c5, mask=valid, other=0).to(tl.int32)
    e6 = tl.load(dlut + c6, mask=valid, other=0).to(tl.int32)
    e7 = tl.load(dlut + c7, mask=valid, other=0).to(tl.int32)

    sm_base = offs * 3
    sb0 = tl.load(sm_pk + sm_base, mask=valid, other=0).to(tl.int32)
    sb1 = tl.load(sm_pk + sm_base + 1, mask=valid, other=0).to(tl.int32)
    sb2 = tl.load(sm_pk + sm_base + 2, mask=valid, other=0).to(tl.int32)
    sm_packed = (sb0 << 16) | (sb1 << 8) | sb2

    s0 = (sm_packed >> 21) & 0x07
    s1 = (sm_packed >> 18) & 0x07
    s2 = (sm_packed >> 15) & 0x07
    s3 = (sm_packed >> 12) & 0x07
    s4 = (sm_packed >> 9) & 0x07
    s5 = (sm_packed >> 6) & 0x07
    s6 = (sm_packed >> 3) & 0x07
    s7 = sm_packed & 0x07

    o0 = ((s0 & 0x04) << 5) | (e0 << 2) | (s0 & 0x03)
    o1 = ((s1 & 0x04) << 5) | (e1 << 2) | (s1 & 0x03)
    o2 = ((s2 & 0x04) << 5) | (e2 << 2) | (s2 & 0x03)
    o3 = ((s3 & 0x04) << 5) | (e3 << 2) | (s3 & 0x03)
    o4 = ((s4 & 0x04) << 5) | (e4 << 2) | (s4 & 0x03)
    o5 = ((s5 & 0x04) << 5) | (e5 << 2) | (s5 & 0x03)
    o6 = ((s6 & 0x04) << 5) | (e6 << 2) | (s6 & 0x03)
    o7 = ((s7 & 0x04) << 5) | (e7 << 2) | (s7 & 0x03)

    tl.store(out + elem, o0.to(tl.uint8), mask=valid)
    tl.store(out + elem + 1, o1.to(tl.uint8), mask=valid)
    tl.store(out + elem + 2, o2.to(tl.uint8), mask=valid)
    tl.store(out + elem + 3, o3.to(tl.uint8), mask=valid)
    tl.store(out + elem + 4, o4.to(tl.uint8), mask=valid)
    tl.store(out + elem + 5, o5.to(tl.uint8), mask=valid)
    tl.store(out + elem + 6, o6.to(tl.uint8), mask=valid)
    tl.store(out + elem + 7, o7.to(tl.uint8), mask=valid)


def next_power_of_2(x):
    return 1 << (max(1, int(x)) - 1).bit_length()


class FP8E5M2Top16ExactCodec:
    def __init__(self, raw, block=128, escape_block=256):
        self.raw = raw
        self.device = raw.device
        self.n = raw.numel()
        if self.n % 8 != 0:
            raise ValueError("E5M2 top16 benchmark expects numel divisible by 8")
        self.block = block
        self.escape_block = escape_block
        self.groups = self.n // 8
        self.code_bytes = self.groups * 4
        self.sm_bytes = self.groups * 3
        self.code_pk = torch.empty(self.code_bytes, dtype=torch.uint8, device=self.device)
        self.sm_pk = torch.empty(self.sm_bytes, dtype=torch.uint8, device=self.device)
        self.out = torch.empty_like(raw)
        self.n_blocks = (self.n + escape_block - 1) // escape_block
        self.counts = torch.empty(self.n_blocks, dtype=torch.int32, device=self.device)
        self.starts = torch.empty(self.n_blocks, dtype=torch.int32, device=self.device)
        self.max_esc = max(self.n // 5, 1024)
        self.local_pos = torch.empty(self.max_esc, dtype=torch.uint8, device=self.device)
        self.esc_val = torch.empty(self.max_esc, dtype=torch.uint8, device=self.device)
        packed_size = (self.max_esc * 5 + 7) // 8 + 1
        self.esc_val_packed = torch.empty(packed_size, dtype=torch.uint8, device=self.device)
        self.grid = ((self.groups + block - 1) // block,)
        self.count_grid = (self.n_blocks,)
        self.lut, self.dlut, self.common_lut, self.coverage = self.build_codebook()

    def build_codebook(self):
        exponents = ((self.raw >> 2) & 0x1F).to(torch.uint8)
        vals, counts = torch.unique(exponents, return_counts=True)
        order = torch.argsort(counts, descending=True)
        lut = torch.zeros((32,), dtype=torch.uint8, device=self.device)
        dlut = torch.zeros((16,), dtype=torch.uint8, device=self.device)
        common = torch.zeros((32,), dtype=torch.uint8, device=self.device)
        top = min(16, vals.numel())
        for i in range(top):
            v = vals[order[i]].item()
            lut[v] = i
            dlut[i] = v
            common[v] = 1
        coverage = counts[order[:top]].sum().item() / counts.sum().item()
        return lut, dlut, common, coverage

    def encode_core(self):
        _enc_e5_top16[self.grid](self.raw, self.lut, self.code_pk, self.sm_pk, self.n, BLOCK=self.block)

    def count_escapes(self):
        _count_external_fp8[self.count_grid](
            self.raw, self.common_lut, self.counts, self.n, FMT=1, BLOCK=self.escape_block
        )

    def write_escapes(self):
        _write_external_fp8[self.count_grid](
            self.raw, self.common_lut, self.starts, self.local_pos, self.esc_val,
            self.n, FMT=1, BLOCK=self.escape_block
        )

    def pack_escapes(self, n_esc):
        grid = (((n_esc + 7) // 8 + 255) // 256,)
        _pack_e5_escape_values[grid](self.esc_val, self.esc_val_packed, n_esc, BLOCK=256)

    def encode_full(self):
        self.encode_core()
        self.count_escapes()
        offsets = torch.cumsum(self.counts, dim=0)
        self.starts = offsets - self.counts
        n_esc = int(offsets[-1].item()) if offsets.numel() else 0
        if n_esc > self.max_esc:
            raise RuntimeError(f"escape buffer too small: {n_esc} > {self.max_esc}")
        self.write_escapes()
        self.pack_escapes(n_esc)
        return n_esc

    def decode_core(self):
        _dec_e5_top16[self.grid](self.code_pk, self.sm_pk, self.dlut, self.out, self.n, BLOCK=self.block)

    def decode_full(self, max_count):
        self.decode_core()
        block_esc = next_power_of_2(max_count)
        _fix_external_e5_packed[self.count_grid](
            self.counts, self.starts, self.local_pos, self.esc_val_packed,
            self.out, BLOCK_ELEMS=self.escape_block, BLOCK_ESC=block_esc
        )
        return self.out

    def compressed_bytes(self, n_esc):
        escape_bytes = n_esc + ((n_esc * 5 + 7) // 8)
        return self.code_bytes + self.sm_bytes + self.n_blocks + escape_bytes


def run_fp8_exact_results(cpu_bf16, device):
    flat = cpu_bf16.to(device=device, dtype=torch.bfloat16).contiguous().view(-1)
    results = []
    for fmt in ("e4m3", "e5m2"):
        raw = bf16_to_raw_fp8(flat, fmt)
        codec = FP8Top8CompactCodec(fmt, raw)
        n_esc = codec.encode_full()
        max_count = codec.counts.max().item()
        codec.decode_full(max_count)
        correct = torch.equal(raw, codec.core.out)
        if not correct:
            raise RuntimeError(f"FP8 round-trip failed for {fmt}")
        comp_bytes = codec.compressed_bytes(n_esc)
        enc_full_s = bench_cuda(codec.encode_full, warmup=10, iters=60)
        n_esc = codec.encode_full()
        max_count = codec.counts.max().item()
        dec_full_s = bench_cuda(lambda: codec.decode_full(max_count), warmup=10, iters=80)
        raw_bytes = raw.numel()
        results.append(
            {
                "format": fmt,
                "scheme": "top8_exact",
                "coverage_name": "top8",
                "raw_fp8_bytes": int(raw_bytes),
                "compressed_bytes": int(comp_bytes),
                "ratio_vs_fp8": raw_bytes / comp_bytes,
                "ratio_vs_bf16": 2.0 * raw_bytes / comp_bytes,
                "coverage": float(codec.core.coverage),
                "escapes": int(n_esc),
                "escape_rate": float(n_esc / raw.numel()),
                "encode_gbs": raw_bytes / enc_full_s / 1e9,
                "decode_gbs": raw_bytes / dec_full_s / 1e9,
            }
        )
    raw = bf16_to_raw_fp8(flat, "e5m2")
    codec = FP8E5M2Top16ExactCodec(raw)
    n_esc = codec.encode_full()
    max_count = codec.counts.max().item()
    codec.decode_full(max_count)
    correct = torch.equal(raw, codec.out)
    if not correct:
        raise RuntimeError("FP8 E5M2 top16 round-trip failed")
    comp_bytes = codec.compressed_bytes(n_esc)
    enc_full_s = bench_cuda(codec.encode_full, warmup=10, iters=60)
    n_esc = codec.encode_full()
    max_count = codec.counts.max().item()
    dec_full_s = bench_cuda(lambda: codec.decode_full(max_count), warmup=10, iters=80)
    raw_bytes = raw.numel()
    results.append(
        {
            "format": "e5m2",
            "scheme": "top16_exact",
            "coverage_name": "top16",
            "raw_fp8_bytes": int(raw_bytes),
            "compressed_bytes": int(comp_bytes),
            "ratio_vs_fp8": raw_bytes / comp_bytes,
            "ratio_vs_bf16": 2.0 * raw_bytes / comp_bytes,
            "coverage": float(codec.coverage),
            "escapes": int(n_esc),
            "escape_rate": float(n_esc / raw.numel()),
            "encode_gbs": raw_bytes / enc_full_s / 1e9,
            "decode_gbs": raw_bytes / dec_full_s / 1e9,
        }
    )
    return results


@triton.jit
def _count_external_bf16(inp, common_lut, counts, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    raw = tl.load(inp + offs, mask=mask, other=0).to(tl.int32)
    exp = (raw >> 7) & 0xFF
    common = tl.load(common_lut + exp, mask=mask, other=1).to(tl.int32)
    esc = (common == 0) & mask
    count = tl.sum(esc.to(tl.int32), axis=0)
    tl.store(counts + pid, count)


@triton.jit
def _write_external_bf16(inp, common_lut, starts, local_pos, esc_val, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base_elem = pid * BLOCK
    offs = base_elem + tl.arange(0, BLOCK)
    mask = offs < n
    raw = tl.load(inp + offs, mask=mask, other=0).to(tl.int32)
    exp = (raw >> 7) & 0xFF
    common = tl.load(common_lut + exp, mask=mask, other=1).to(tl.int32)
    esc = (common == 0) & mask
    rank = tl.cumsum(esc.to(tl.int32), 0) - 1
    start = tl.load(starts + pid)
    out_idx = start + rank
    local = offs - base_elem
    tl.store(local_pos + out_idx, local.to(tl.uint8), mask=esc)
    tl.store(esc_val + out_idx, exp.to(tl.uint8), mask=esc)


@triton.jit
def _fix_external_bf16(counts, starts, local_pos, esc_val, sm, out, BLOCK_ELEMS: tl.constexpr, BLOCK_ESC: tl.constexpr):
    pid = tl.program_id(0)
    count = tl.load(counts + pid)
    start = tl.load(starts + pid)
    offs = tl.arange(0, BLOCK_ESC)
    mask = offs < count
    idx = start + offs
    local = tl.load(local_pos + idx, mask=mask, other=0).to(tl.int32)
    exp = tl.load(esc_val + idx, mask=mask, other=0).to(tl.int32)
    pos = pid * BLOCK_ELEMS + local
    raw_sm = tl.load(sm + pos, mask=mask, other=0).to(tl.int32)
    fixed = ((raw_sm & 0x80) << 8) | (exp << 7) | (raw_sm & 0x7F)
    tl.store(out + pos, fixed.to(tl.int16), mask=mask)


@triton.jit
def _enc_4bit_compact(inp, lut, pk, sm, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    po = tl.arange(0, BLOCK)
    b = pid * BLOCK * 4
    for step in range(4):
        pi = b + step * BLOCK + po
        ei = pi * 2
        oi = ei + 1
        em = ei < n
        om = oi < n
        v0 = tl.load(inp + ei, mask=em, other=0).to(tl.int16)
        v1 = tl.load(inp + oi, mask=om, other=0).to(tl.int16)
        i0 = tl.load(lut + ((v0 >> 7) & 0xFF).to(tl.int32), mask=em, other=0).to(tl.uint8)
        i1 = tl.load(lut + ((v1 >> 7) & 0xFF).to(tl.int32), mask=om, other=0).to(tl.uint8)
        tl.store(pk + pi, (i0 << 4) | i1, mask=em)
        tl.store(sm + ei, (((v0 >> 8) & 0x80) | (v0 & 0x7F)).to(tl.uint8), mask=em)
        tl.store(sm + oi, (((v1 >> 8) & 0x80) | (v1 & 0x7F)).to(tl.uint8), mask=om)


@triton.jit
def _dec_4bit_compact(pk, sm, dlut, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    po = tl.arange(0, BLOCK)
    b = pid * BLOCK * 4
    for step in range(4):
        pi = b + step * BLOCK + po
        ei = pi * 2
        oi = ei + 1
        em = ei < n
        om = oi < n
        packed = tl.load(pk + pi, mask=em, other=0)
        e0 = tl.load(dlut + ((packed >> 4) & 0x0F).to(tl.int32), mask=em, other=0).to(tl.int16)
        e1 = tl.load(dlut + (packed & 0x0F).to(tl.int32), mask=om, other=0).to(tl.int16)
        s0 = tl.load(sm + ei, mask=em, other=0).to(tl.int16)
        s1 = tl.load(sm + oi, mask=om, other=0).to(tl.int16)
        tl.store(out + ei, ((s0 & 0x80) << 8) | (e0 << 7) | (s0 & 0x7F), mask=em)
        tl.store(out + oi, ((s1 & 0x80) << 8) | (e1 << 7) | (s1 & 0x7F), mask=om)


class BF16Top8ExactCodec:
    def __init__(self, flat_bf16, device, escape_block=256):
        self.device = device
        self.raw = flat_bf16.contiguous().view(torch.int16)
        self.n = self.raw.numel()
        if self.n % 8 != 0:
            raise ValueError("top8 exact benchmark expects numel divisible by 8")
        self.escape_block = escape_block
        self.n_blocks = (self.n + escape_block - 1) // escape_block
        self.block = 128
        self.code_bytes = self.n * 3 // 8
        self.sm_bytes = self.n
        self.enc_lut, self.dec_lut, self.common_lut, self.coverage = self.build_codebook()
        self.code_pk = torch.empty(self.code_bytes, dtype=torch.uint8, device=self.device)
        self.sm = torch.empty(self.n, dtype=torch.uint8, device=self.device)
        self.out = torch.empty(self.n, dtype=torch.int16, device=self.device)
        self.counts = torch.empty(self.n_blocks, dtype=torch.int32, device=self.device)
        self.starts = torch.empty(self.n_blocks, dtype=torch.int32, device=self.device)
        self.max_esc = max(self.n // 4, 1024)
        self.local_pos = torch.empty(self.max_esc, dtype=torch.uint8, device=self.device)
        self.esc_val = torch.empty(self.max_esc, dtype=torch.uint8, device=self.device)

    def build_codebook(self):
        exponents = ((self.raw >> 7) & 0xFF).to(torch.uint8)
        vals, counts = torch.unique(exponents, return_counts=True)
        order = torch.argsort(counts, descending=True)
        enc = torch.zeros((256,), dtype=torch.uint8, device=self.device)
        dec = torch.zeros((8,), dtype=torch.uint8, device=self.device)
        common = torch.zeros((256,), dtype=torch.uint8, device=self.device)
        top = min(8, vals.numel())
        for i in range(top):
            v = vals[order[i]].item()
            enc[v] = i
            dec[i] = v
            common[v] = 1
        coverage = counts[order[:top]].sum().item() / counts.sum().item()
        return enc, dec, common, coverage

    def encode_core(self):
        grid = ((self.n // 8 + self.block - 1) // self.block,)
        _enc_3bit_vec[grid](self.raw, self.enc_lut, self.code_pk, self.sm, self.n, BLOCK=self.block)

    def count_escapes(self):
        _count_external_bf16[(self.n_blocks,)](self.raw, self.common_lut, self.counts, self.n, BLOCK=self.escape_block)

    def write_escapes(self):
        _write_external_bf16[(self.n_blocks,)](
            self.raw, self.common_lut, self.starts, self.local_pos, self.esc_val, self.n, BLOCK=self.escape_block
        )

    def encode_full(self):
        self.encode_core()
        self.count_escapes()
        offsets = torch.cumsum(self.counts, dim=0)
        self.starts = offsets - self.counts
        n_esc = int(offsets[-1].item()) if offsets.numel() else 0
        if n_esc > self.max_esc:
            raise RuntimeError(f"escape buffer too small: {n_esc} > {self.max_esc}")
        self.write_escapes()
        return n_esc

    def decode_core(self):
        grid = ((self.n // 8 + self.block - 1) // self.block,)
        _dec_3bit_vec[grid](self.code_pk, self.sm, self.dec_lut, self.out, self.n, BLOCK=self.block)

    def decode_full(self, max_count):
        self.decode_core()
        block_esc = 1 << (max(1, int(max_count)) - 1).bit_length() if max_count > 0 else 1
        _fix_external_bf16[(self.n_blocks,)](
            self.counts,
            self.starts,
            self.local_pos,
            self.esc_val,
            self.sm,
            self.out,
            BLOCK_ELEMS=self.escape_block,
            BLOCK_ESC=block_esc,
        )
        return self.out

    def compressed_bytes(self, n_esc):
        return self.code_bytes + self.sm_bytes + self.n_blocks + n_esc * 2


def run_bf16_topk_ablation(cpu_bf16, device):
    flat = cpu_bf16.to(device=device, dtype=torch.bfloat16).contiguous().view(-1)

    codec16 = FastLosslessCodec(str(device))
    codec16.calibrate(flat)
    encoded16 = codec16.encode(flat)
    decoded16 = codec16.decode(*encoded16)
    if not torch.equal(flat.view(torch.int16), decoded16.view(torch.int16)):
        raise RuntimeError("top16 lossless round-trip failed")
    comp16 = encoded16[0].numel() + encoded16[1].numel() + encoded16[2].numel() * 4 + encoded16[3].numel()
    enc16_s = bench_cuda(lambda: codec16.encode(flat), warmup=10, iters=60)
    encoded16 = codec16.encode(flat)
    dec16_s = bench_cuda(lambda: codec16.decode(*encoded16), warmup=10, iters=80)
    exp = ((flat.view(torch.int16) >> 7) & 0xFF).to(torch.uint8)
    vals, counts = torch.unique(exp, return_counts=True)
    order = torch.argsort(counts, descending=True)
    cov16 = counts[order[: min(16, vals.numel())]].sum().item() / counts.sum().item()

    codec8 = BF16Top8ExactCodec(flat, device)
    n_esc8 = codec8.encode_full()
    max_count8 = codec8.counts.max().item() if codec8.counts.numel() else 0
    decoded8 = codec8.decode_full(max_count8)
    if not torch.equal(flat.view(torch.int16), decoded8.view(torch.int16)):
        raise RuntimeError("top8 lossless round-trip failed")
    comp8 = codec8.compressed_bytes(n_esc8)
    enc8_s = bench_cuda(codec8.encode_full, warmup=10, iters=40)
    n_esc8 = codec8.encode_full()
    max_count8 = codec8.counts.max().item() if codec8.counts.numel() else 0
    dec8_s = bench_cuda(lambda: codec8.decode_full(max_count8), warmup=10, iters=60)

    raw_bytes = flat.numel() * 2
    return {
        "top8_3bit_exact": {
            "coverage": float(codec8.coverage),
            "escapes": int(n_esc8),
            "escape_rate": float(n_esc8 / flat.numel()),
            "compressed_bytes": int(comp8),
            "ratio": raw_bytes / comp8,
            "encode_gbs": raw_bytes / enc8_s / 1e9,
            "decode_gbs": raw_bytes / dec8_s / 1e9,
        },
        "top16_4bit_exact": {
            "coverage": float(cov16),
            "escapes": int(encoded16[5]),
            "escape_rate": float(encoded16[5] / flat.numel()),
            "compressed_bytes": int(comp16),
            "ratio": raw_bytes / comp16,
            "encode_gbs": raw_bytes / enc16_s / 1e9,
            "decode_gbs": raw_bytes / dec16_s / 1e9,
        },
    }


def load_prompt_texts(dataset_key, count):
    if dataset_key == "wikitext_train":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [x["text"].strip() for x in ds if x["text"].strip()]
    elif dataset_key == "wikitext_test":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [x["text"].strip() for x in ds if x["text"].strip()]
    elif dataset_key == "humaneval":
        ds = load_dataset("openai_humaneval", split="test")
        texts = [x["prompt"].strip() for x in ds if x["prompt"].strip()]
    elif dataset_key == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        texts = [x["question"].strip() for x in ds if x["question"].strip()]
    elif dataset_key == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="validation")
        texts = []
        for x in ds:
            prompt = x["question"].strip()
            if x.get("choices"):
                choice_str = " ".join(f"({chr(65+i)}) {c}" for i, c in enumerate(x["choices"]))
                prompt = f"{prompt} {choice_str}"
            if prompt.strip():
                texts.append(prompt)
    elif dataset_key == "ptb":
        ds = load_dataset("ptb_text_only", "penn_treebank", split="test")
        texts = [x["sentence"].strip() for x in ds if x["sentence"].strip()]
    else:
        raise ValueError(dataset_key)
    return texts[:count]


def load_calibration_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map={"": device.index if isinstance(device, torch.device) else 0},
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def collect_exponents_for_prompts(tokenizer, model, prompts, device, max_length=256):
    exponents = []
    with torch.inference_mode():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(str(device)) for k, v in inputs.items()}
            outputs = model(**inputs, use_cache=True, return_dict=True)
            pkv = outputs.past_key_values
            if hasattr(pkv, "to_legacy_cache"):
                pkv = pkv.to_legacy_cache()
            for key, value in pkv:
                exponents.append(((key.contiguous().view(torch.int16) >> 7) & 0xFF).to(torch.uint8).view(-1).cpu())
                exponents.append(((value.contiguous().view(torch.int16) >> 7) & 0xFF).to(torch.uint8).view(-1).cpu())
    return torch.cat(exponents, dim=0)


def topk_coverage_from_codebook(sample_exp, eval_exp, k=16):
    vals, counts = torch.unique(sample_exp, return_counts=True)
    order = torch.argsort(counts, descending=True)
    top_vals = vals[order[: min(k, vals.numel())]]
    lut = torch.zeros((256,), dtype=torch.uint8)
    lut[top_vals.long()] = 1
    return float(lut[eval_exp.long()].float().mean().item())


def run_cross_dataset_calibration(device):
    dataset_specs = {
        "dataset_a": {"key": "wikitext_train", "label": "wikitext-2-raw-v1/train", "domain": "language"},
        "wikitext_test": {"key": "wikitext_test", "label": "wikitext-2-raw-v1/test", "domain": "language"},
        "humaneval": {"key": "humaneval", "label": "openai_humaneval/test", "domain": "code"},
        "gsm8k": {"key": "gsm8k", "label": "gsm8k/main/test", "domain": "math"},
        "mmlu": {"key": "mmlu", "label": "cais/mmlu/all/validation", "domain": "knowledge"},
        "ptb": {"key": "ptb", "label": "ptb_text_only/penn_treebank/test", "domain": "language"},
    }
    tokenizer, model = load_calibration_model(CAL_MODEL, device)
    exponents = {}
    for name, spec in dataset_specs.items():
        prompts = load_prompt_texts(spec["key"], CAL_PROMPTS_PER_SET)
        exponents[name] = collect_exponents_for_prompts(tokenizer, model, prompts, device)
    del model
    torch.cuda.empty_cache()
    exp_a = exponents["dataset_a"]
    cov_aa = topk_coverage_from_codebook(exp_a, exp_a, 16)
    dataset_b_results = []
    for name, spec in dataset_specs.items():
        if name == "dataset_a":
            continue
        exp_b = exponents[name]
        dataset_b_results.append(
            {
                "name": name,
                "label": spec["label"],
                "domain": spec["domain"],
                "calibrate_A_eval_B_top16": topk_coverage_from_codebook(exp_a, exp_b, 16),
                "calibrate_B_eval_B_top16": topk_coverage_from_codebook(exp_b, exp_b, 16),
            }
        )
    return {
        "model": CAL_MODEL,
        "dataset_a": dataset_specs["dataset_a"]["label"],
        "calibrate_A_eval_A_top16": cov_aa,
        "dataset_b_results": dataset_b_results,
    }


def build_top16_ratio_with_metadata(exp_matrix, grouping):
    rows, cols = exp_matrix.shape
    n = rows * cols
    if grouping == "per_tensor":
        groups = [exp_matrix.reshape(-1)]
    elif grouping == "per_token":
        groups = [exp_matrix[i, :] for i in range(rows)]
    elif grouping == "per_channel":
        groups = [exp_matrix[:, j] for j in range(cols)]
    else:
        raise ValueError(grouping)

    total_esc = 0
    total_common = 0
    for g in groups:
        vals, counts = torch.unique(g, return_counts=True)
        order = torch.argsort(counts, descending=True)
        top_vals = vals[order[: min(16, vals.numel())]]
        lut = torch.zeros((256,), dtype=torch.uint8)
        lut[top_vals.long()] = 1
        common = int(lut[g.long()].sum().item())
        total_common += common
        total_esc += int(g.numel() - common)

    codebook_bytes = len(groups) * 16
    n_blocks = math.ceil(n / 256)
    compressed_bytes = (n // 2) + n + n_blocks + total_esc * 2 + codebook_bytes
    return {
        "coverage": total_common / n,
        "escapes": total_esc,
        "escape_rate": total_esc / n,
        "codebook_bytes": codebook_bytes,
        "projected_ratio": (n * 2) / compressed_bytes,
    }


def run_calibration_granularity(cpu_bf16, device):
    exp = ((cpu_bf16.contiguous().view(torch.int16) >> 7) & 0xFF).to(torch.uint8).view(cpu_bf16.shape[0], cpu_bf16.shape[1])
    ratio_data = {
        "per_tensor": build_top16_ratio_with_metadata(exp, "per_tensor"),
        "per_token": build_top16_ratio_with_metadata(exp, "per_token"),
        "per_channel": build_top16_ratio_with_metadata(exp, "per_channel"),
    }
    bench_cpu = cpu_bf16[:GRANULARITY_BENCH_ROWS].contiguous()
    bench_gpu = bench_cpu.to(device=device, dtype=torch.bfloat16)
    rows, cols = bench_gpu.shape

    def prepare_codecs(grouping):
        codecs = []
        groups = []
        if grouping == "per_tensor":
            groups = [bench_gpu.reshape(-1)]
        elif grouping == "per_token":
            groups = [bench_gpu[i, :].contiguous() for i in range(rows)]
        elif grouping == "per_channel":
            groups = [bench_gpu[:, j].contiguous() for j in range(cols)]
        for g in groups:
            codec = FastLosslessCodec(str(device))
            codec.calibrate(g)
            codecs.append(codec)
        return groups, codecs

    def benchmark_grouping(grouping, groups, codecs):
        encoded = []
        comp_bytes = 0
        for g, c in zip(groups, codecs):
            r = c.encode(g)
            encoded.append((c, r, g))
            comp_bytes += r[0].numel() + r[1].numel() + r[2].numel() * 4 + r[3].numel()
        comp_bytes += len(groups) * 16

        def encode_all():
            for g, c in zip(groups, codecs):
                c.encode(g)

        def decode_all():
            for c, r, _g in encoded:
                c.decode(*r)

        encode_s = bench_cuda(encode_all, warmup=2, iters=5)
        decode_s = bench_cuda(decode_all, warmup=2, iters=5)
        raw_bytes = bench_gpu.numel() * 2
        return {
            "actual_ratio": raw_bytes / comp_bytes,
            "encode_gbs": raw_bytes / encode_s / 1e9,
            "decode_gbs": raw_bytes / decode_s / 1e9,
            "bench_shape": [rows, cols],
        }

    groups_t, codecs_t = prepare_codecs("per_tensor")
    groups_tok, codecs_tok = prepare_codecs("per_token")
    groups_ch, codecs_ch = prepare_codecs("per_channel")
    bench_data = {
        "per_tensor": benchmark_grouping("per_tensor", groups_t, codecs_t),
        "per_token": benchmark_grouping("per_token", groups_tok, codecs_tok),
        "per_channel": benchmark_grouping("per_channel", groups_ch, codecs_ch),
    }
    out = {}
    for name in ("per_tensor", "per_token", "per_channel"):
        out[name] = ratio_data[name] | bench_data[name]
    baseline = out["per_tensor"]
    for name in ("per_tensor", "per_token", "per_channel"):
        out[name]["vs_baseline_encode"] = out[name]["encode_gbs"] / baseline["encode_gbs"]
        out[name]["vs_baseline_decode"] = out[name]["decode_gbs"] / baseline["decode_gbs"]
        out[name]["vs_baseline_ratio"] = out[name]["actual_ratio"] / baseline["actual_ratio"]
    return out


def run_fp8_transfer_sweep(device):
    blocks, meta = load_model_activation_blocks(FP8_TRANSFER_HF, device)
    dma_cache = {}
    results = {
        "model": FP8_TRANSFER_MODEL,
        "hf_name": FP8_TRANSFER_HF,
        "transport_mode": FP8_TRANSFER_MODE["name"],
        "bandwidth_gbs": FP8_TRANSFER_MODE["bandwidth_gbs"],
        "num_hidden_layers": meta["num_hidden_layers"],
        "rows": {
            "e4m3_top8_exact": [],
            "e5m2_top8_exact": [],
            "e5m2_top16_exact": [],
        },
    }

    for seq_len in FP8_TRANSFER_SEQ_LENS:
        cpu_rows = assemble_row_prefix(blocks, seq_len)
        flat_bf16 = cpu_rows.to(device=device, dtype=torch.bfloat16).contiguous().view(-1)
        codec_specs = [
            ("e4m3_top8_exact", "e4m3", FP8Top8CompactCodec),
            ("e5m2_top8_exact", "e5m2", FP8Top8CompactCodec),
            ("e5m2_top16_exact", "e5m2", FP8E5M2Top16ExactCodec),
        ]
        for name, fmt, codec_cls in codec_specs:
            raw = bf16_to_raw_fp8(flat_bf16, fmt)
            codec = codec_cls(fmt, raw) if codec_cls is FP8Top8CompactCodec else codec_cls(raw)
            n_esc = codec.encode_full()
            max_count = codec.counts.max().item()
            if codec_cls is FP8Top8CompactCodec:
                codec.decode_full(max_count)
                decoded = codec.core.out
            else:
                decoded = codec.decode_full(max_count)
            if not torch.equal(raw, decoded):
                raise RuntimeError(f"FP8 transfer round-trip failed for {name} at seq_len={seq_len}")
            comp_bytes = codec.compressed_bytes(n_esc)
            enc_s = bench_cuda(codec.encode_full, warmup=8, iters=30)
            n_esc = codec.encode_full()
            max_count = codec.counts.max().item()
            dec_s = bench_cuda(lambda: codec.decode_full(max_count), warmup=8, iters=40)

            raw_d2h_s = measure_dma_time(raw.numel(), "d2h", device, dma_cache)
            raw_h2d_s = measure_dma_time(raw.numel(), "h2d", device, dma_cache)
            comp_d2h_s = measure_dma_time(comp_bytes, "d2h", device, dma_cache)
            comp_h2d_s = measure_dma_time(comp_bytes, "h2d", device, dma_cache)
            sim = simulate_transport(
                raw_bytes=raw.numel(),
                comp_bytes=comp_bytes,
                enc_s=enc_s,
                dec_s=dec_s,
                n_layers=meta["num_hidden_layers"],
                raw_d2h_s=raw_d2h_s,
                raw_h2d_s=raw_h2d_s,
                comp_d2h_s=comp_d2h_s,
                comp_h2d_s=comp_h2d_s,
                net_gbs=FP8_TRANSFER_MODE["bandwidth_gbs"],
            )
            results["rows"][name].append(
                {
                    "seq_len": seq_len,
                    "native_ms": sim["raw_pipe_s"] * 1000,
                    "splitzip_ms": sim["splitzip_pipe_s"] * 1000,
                    "speedup": sim["speedup"],
                    "ratio": raw.numel() / comp_bytes,
                    "encode_gbs": raw.numel() / enc_s / 1e9,
                    "decode_gbs": raw.numel() / dec_s / 1e9,
                    "escape_rate": n_esc / raw.numel(),
                    "splitzip_breakdown_sequential_ms": {
                        k: v * 1000 for k, v in sim["splitzip_breakdown_sequential_s"].items()
                    },
                }
            )
    return results


def build_serving_kv(rows_cpu, seq_len, device):
    rows = assemble_row_prefix([rows_cpu], seq_len).to(device=device, dtype=torch.bfloat16)
    half = rows.shape[1] // 2
    k = rows[:, :half].contiguous().view(seq_len, 8, 128).permute(1, 0, 2).contiguous()
    v = rows[:, half:].contiguous().view(seq_len, 8, 128).permute(1, 0, 2).contiguous()
    return k, v


def attention_proxy(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(1, 2)) * scale
    probs = torch.softmax(scores.float(), dim=-1).to(torch.bfloat16)
    out = torch.matmul(probs, v)
    return out


def run_serving_compute_proxy(device):
    blocks, meta = load_model_activation_blocks(SERVING_MODEL, device)
    rows_cpu = blocks[SERVING_LAYER_IDX]
    k, v = build_serving_kv(rows_cpu, SERVING_SEQ_LEN, device)
    q = torch.randn((k.shape[0], 1, k.shape[2]), dtype=torch.bfloat16, device=device)

    codec = FastLosslessCodec(str(device))
    codec.calibrate(torch.cat([k.view(-1), v.view(-1)]))
    k_enc = codec.encode(k.view(-1))
    v_enc = codec.encode(v.view(-1))

    _ = attention_proxy(q, k, v)
    original_s = bench_cuda(lambda: attention_proxy(q, k, v), warmup=20, iters=80)

    def decompress_then_compute():
        k_dec = codec.decode(*k_enc).view_as(k)
        v_dec = codec.decode(*v_enc).view_as(v)
        return attention_proxy(q, k_dec, v_dec)

    _ = decompress_then_compute()
    dec_compute_s = bench_cuda(decompress_then_compute, warmup=10, iters=40)

    materialized_k = torch.empty_like(k)
    materialized_v = torch.empty_like(v)

    def materialize_copy():
        materialized_k.copy_(k)
        materialized_v.copy_(v)

    copy_s = bench_cuda(materialize_copy, warmup=20, iters=80)
    merged_proj_s = max(original_s, dec_compute_s - copy_s)

    return {
        "model": SERVING_MODEL,
        "seq_len": SERVING_SEQ_LEN,
        "layer_index": SERVING_LAYER_IDX,
        "heads": int(k.shape[0]),
        "head_dim": int(k.shape[2]),
        "original_kv_compute_ms": original_s * 1000,
        "decompress_then_compute_ms": dec_compute_s * 1000,
        "projected_merged_kernel_ms": merged_proj_s * 1000,
        "materialization_copy_ms": copy_s * 1000,
        "note": "Merged-kernel number is a projection: explicit decompress+compute minus one measured BF16 materialization copy, floored by original compute time.",
    }


def to_pct(x):
    return f"{x * 100:.2f}%"


def fmt(x, digits=3):
    return f"{x:.{digits}f}"


def make_markdown(data):
    lines = []
    lines.append("# Additional Thesis Experiments")
    lines.append("")
    lines.append("## 1. FP8 Exact Results")
    lines.append("")
    lines.append("| Format | Scheme | Coverage | Ratio vs FP8 | Total Ratio vs BF16 | Encode GB/s | Decode GB/s | Escape Rate |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in data["fp8_results"]:
        lines.append(
            f"| {row['format'].upper()} | {row['scheme']} | {row['coverage_name']}={to_pct(row['coverage'])} | {fmt(row['ratio_vs_fp8'])} | "
            f"{fmt(row['ratio_vs_bf16'])} | {fmt(row['encode_gbs'])} | {fmt(row['decode_gbs'])} | {to_pct(row['escape_rate'])} |"
        )
    lines.append("")
    lines.append("## 2. FP8 End-to-End Speedup vs Sequence Length")
    lines.append("")
    lines.append(
        f"- Model: `{data['fp8_transfer']['model']}`, transport: `{data['fp8_transfer']['transport_mode']}` "
        f"({fmt(data['fp8_transfer']['bandwidth_gbs'])} GB/s)"
    )
    lines.append("")
    for name, rows in data["fp8_transfer"]["rows"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Seq Len | Native ms | SplitZip ms | Speedup (x) | Ratio (x) | Encode GB/s | Decode GB/s |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            lines.append(
                f"| {row['seq_len']} | {fmt(row['native_ms'])} | {fmt(row['splitzip_ms'])} | {fmt(row['speedup'])} | "
                f"{fmt(row['ratio'])} | {fmt(row['encode_gbs'])} | {fmt(row['decode_gbs'])} |"
            )
        lines.append("")
    lines.append("## 3. Serving Compute Feasibility")
    lines.append("")
    s = data["serving_compute"]
    lines.append(f"- Model proxy: `{s['model']}`, seq-len `{s['seq_len']}`, layer `{s['layer_index']}`")
    lines.append(f"- Original KV compute time: `{fmt(s['original_kv_compute_ms'])} ms`")
    lines.append(f"- KV compute time with explicit decompression: `{fmt(s['decompress_then_compute_ms'])} ms`")
    lines.append(f"- KV compute time with projected merged kernel: `{fmt(s['projected_merged_kernel_ms'])} ms`")
    lines.append(f"- Materialization copy removed in projection: `{fmt(s['materialization_copy_ms'])} ms`")
    lines.append(f"- Note: {s['note']}")
    lines.append("")
    lines.append("## 4. BF16 Top-8 vs Top-16 (Exact, With Escapes)")
    lines.append("")
    lines.append("| Variant | Coverage | Ratio | Encode GB/s | Decode GB/s | Escape Rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    top8 = data["bf16_topk_ablation"]["top8_3bit_exact"]
    top16 = data["bf16_topk_ablation"]["top16_4bit_exact"]
    lines.append(
        f"| Top-8 (3-bit) | {to_pct(top8['coverage'])} | {fmt(top8['ratio'])} | {fmt(top8['encode_gbs'])} | {fmt(top8['decode_gbs'])} | {to_pct(top8['escape_rate'])} |"
    )
    lines.append(
        f"| Top-16 (4-bit) | {to_pct(top16['coverage'])} | {fmt(top16['ratio'])} | {fmt(top16['encode_gbs'])} | {fmt(top16['decode_gbs'])} | {to_pct(top16['escape_rate'])} |"
    )
    better = data["bf16_topk_ablation"]["better_format"]
    lines.append("")
    lines.append(f"- Better format on this real BF16 tensor: `{better}`")
    lines.append("")
    lines.append("## 5. Cross-Dataset Calibration")
    lines.append("")
    c = data["cross_dataset_calibration"]
    lines.append(f"- Dataset A: `{c['dataset_a']}`")
    lines.append(f"- Calibrate on A, Top-16 coverage on A: `{to_pct(c['calibrate_A_eval_A_top16'])}`")
    lines.append("")
    lines.append("| Dataset B | Domain | Calibrate on A, Eval on B | Calibrate on B, Eval on B |")
    lines.append("| --- | --- | ---: | ---: |")
    for row in c["dataset_b_results"]:
        lines.append(
            f"| {row['label']} | {row['domain']} | {to_pct(row['calibrate_A_eval_B_top16'])} | {to_pct(row['calibrate_B_eval_B_top16'])} |"
        )
    lines.append("")
    lines.append("## 6. Calibration Granularity")
    lines.append("")
    lines.append("| Scope | Coverage | Actual Ratio | Projected Ratio | Encode GB/s | Decode GB/s | vs Base Enc | vs Base Dec | Codebook Bytes | Bench Shape |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for name, row in data["calibration_granularity"].items():
        lines.append(
            f"| {name} | {to_pct(row['coverage'])} | {fmt(row['actual_ratio'])} | {fmt(row['projected_ratio'])} | "
            f"{fmt(row['encode_gbs'])} | {fmt(row['decode_gbs'])} | {fmt(row['vs_baseline_encode'])}x | "
            f"{fmt(row['vs_baseline_decode'])}x | {row['codebook_bytes']} | {row['bench_shape'][0]}x{row['bench_shape'][1]} |"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- JSON: `{data['artifacts']['json']}`")
    lines.append(f"- Markdown: `{data['artifacts']['markdown']}`")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default=str(DEFAULT_JSON))
    parser.add_argument("--md-out", default=str(DEFAULT_MD))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading representative BF16 activations...", flush=True)
    cpu_bf16 = load_representative_bf16_matrix(device)

    print("Running FP8 exact benchmarks...", flush=True)
    fp8_results = run_fp8_exact_results(cpu_bf16, device)

    print("Running BF16 top-8 vs top-16 ablation...", flush=True)
    bf16_topk = run_bf16_topk_ablation(cpu_bf16, device)
    better_format = "top16_4bit_exact"
    if bf16_topk["top8_3bit_exact"]["ratio"] > bf16_topk["top16_4bit_exact"]["ratio"] and \
       bf16_topk["top8_3bit_exact"]["decode_gbs"] >= bf16_topk["top16_4bit_exact"]["decode_gbs"]:
        better_format = "top8_3bit_exact"
    bf16_topk["better_format"] = better_format

    print("Running FP8 transfer sweep...", flush=True)
    fp8_transfer = run_fp8_transfer_sweep(device)

    print("Running cross-dataset calibration study...", flush=True)
    cross_dataset = run_cross_dataset_calibration(device)

    print("Running calibration granularity study...", flush=True)
    granularity = run_calibration_granularity(cpu_bf16, device)

    print("Running serving compute proxy...", flush=True)
    serving = run_serving_compute_proxy(device)

    out = {
        "representative_tensor": {
            "activation_model": REP_ACTIVATION_MODEL,
            "shape": list(REP_SHAPE),
        },
        "fp8_results": fp8_results,
        "fp8_transfer": fp8_transfer,
        "bf16_topk_ablation": bf16_topk,
        "cross_dataset_calibration": cross_dataset,
        "calibration_granularity": granularity,
        "serving_compute": serving,
        "artifacts": {
            "json": str(Path(args.json_out).resolve()),
            "markdown": str(Path(args.md_out).resolve()),
        },
    }

    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    json_path.write_text(json.dumps(out, indent=2))
    md_path.write_text(make_markdown(out))
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
