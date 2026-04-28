from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
import triton
import triton.language as tl
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.splitzip.escape_calibration_ablation import (
    _count_sentinel_escapes,
    _fix_sentinel_escapes,
    _write_sentinel_values,
)
from experiments.splitzip.fp8_e5m2_top8_compact_bench import (
    _count_external_fp8,
    _pack_e4_escape_values,
    _pack_e5_escape_values,
)
from experiments.splitzip.fp8_fixed_codec_bench import FP8FixedCodec
from experiments.splitzip.lossless_fast import _dec_4bit, _enc_4bit
from experiments.splitzip.opt_rounds3 import _dec_3bit_vec, _enc_3bit_vec
from experiments.splitzip.thesis_additional_experiments import _dec_e5_top16, _enc_e5_top16
from experiments.splitzip.thesis_experiment_dump import (
    assemble_row_prefix,
    load_model_activation_blocks,
    measure_dma_time,
    simulate_transport,
)
from experiments.splitzip_v2.benchmark_utils import mean_std, write_json
from experiments.splitzip_v2.codec_gpu import (
    ChunkLocalSplitZipGPU,
    _count_escapes_chunk,
    _fix_escapes_local_linear,
)
from experiments.splitzip_v2.zipserv_encode_bench import assemble_matrix_width


MODEL_QWEN32 = "Qwen/Qwen3-32B"
HIDDEN_DIM = 4096
DEFAULT_SEQ_LEN = 65536
BREAKDOWN_SEQ_LENS = [2048, 16384, 65536]
FP8_SEQ_LENS = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
ROCE_GBS = 87.0


def set_device(device: str) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    return dev


def bench_repeats(fn, repeats: int = 10, warmup: int = 5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def gbs_summary(raw_bytes: int, times):
    return mean_std([raw_bytes / t / 1e9 for t in times])


def fp8_raw_from_bf16(flat_bf16: torch.Tensor, fmt: str) -> torch.Tensor:
    if fmt == "e4m3":
        raw = flat_bf16.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    elif fmt == "e5m2":
        raw = flat_bf16.to(torch.float8_e5m2).view(torch.uint8).contiguous()
    else:
        raise ValueError(fmt)
    return raw[: raw.numel() - (raw.numel() % 8)].contiguous()


@triton.jit
def _write_external_fp8_u16(raw, common_lut, starts, local_pos, esc_val,
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
    tl.store(local_pos + out_idx, local.to(tl.uint16), mask=esc)
    tl.store(esc_val + out_idx, exp.to(tl.uint8), mask=esc)


@triton.jit
def _fix_external_e4_packed_u16(counts, starts, local_pos, esc_val_packed, out,
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
def _fix_external_e5_packed_u16(counts, starts, local_pos, esc_val_packed, out,
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


@triton.jit
def _write_bf16_escapes_chunkid(inp, common_lut, starts, chunk_id, local_pos, esc_val,
                                n: tl.constexpr, CHUNK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * CHUNK
    offs = base + tl.arange(0, CHUNK)
    mask = offs < n
    raw = tl.load(inp + offs, mask=mask, other=0).to(tl.int32)
    exp = (raw >> 7) & 0xFF
    common = tl.load(common_lut + exp, mask=mask, other=1).to(tl.int32)
    esc = (common == 0) & mask
    rank = tl.cumsum(esc.to(tl.int32), 0) - 1
    start = tl.load(starts + pid)
    out = start + rank
    tl.store(chunk_id + out, pid, mask=esc)
    tl.store(local_pos + out, (offs - base).to(tl.uint16), mask=esc)
    tl.store(esc_val + out, exp.to(tl.uint8), mask=esc)


def next_power_of_2(x: int) -> int:
    return 1 << (max(1, int(x)) - 1).bit_length()


class BF16Top8ChunkCodec:
    def __init__(self, flat_bf16: torch.Tensor, chunk_size: int):
        self.raw = flat_bf16.contiguous().view(torch.int16)
        self.device = self.raw.device
        self.n = int(self.raw.numel())
        if self.n % 8:
            raise ValueError("top8 benchmark expects numel divisible by 8")
        self.chunk_size = int(chunk_size)
        self.n_chunks = (self.n + self.chunk_size - 1) // self.chunk_size
        self.code_bytes = self.n * 3 // 8
        self.sm_bytes = self.n
        self.code_pk = torch.empty(self.code_bytes, dtype=torch.uint8, device=self.device)
        self.sm = torch.empty(self.n, dtype=torch.uint8, device=self.device)
        self.out = torch.empty(self.n, dtype=torch.int16, device=self.device)
        self.counts = torch.empty(self.n_chunks, dtype=torch.int32, device=self.device)
        self.starts = torch.empty(self.n_chunks, dtype=torch.int32, device=self.device)
        self.max_esc = max(self.n // 4, 1024)
        self.chunk_id = torch.empty(self.max_esc, dtype=torch.int32, device=self.device)
        self.local_pos = torch.empty(self.max_esc, dtype=torch.uint16, device=self.device)
        self.esc_val = torch.empty(self.max_esc, dtype=torch.uint8, device=self.device)
        self.enc_lut, self.dec_lut, self.common_lut, self.coverage = self._build_codebook()

    def _build_codebook(self):
        exp = ((self.raw >> 7) & 0xFF).to(torch.uint8)
        vals, counts = torch.unique(exp, return_counts=True)
        order = torch.argsort(counts, descending=True)
        enc = torch.zeros(256, dtype=torch.uint8, device=self.device)
        dec = torch.zeros(8, dtype=torch.uint8, device=self.device)
        common = torch.zeros(256, dtype=torch.uint8, device=self.device)
        for code in range(min(8, vals.numel())):
            value = int(vals[order[code]].item())
            enc[value] = code
            dec[code] = value
            common[value] = 1
        coverage = float(counts[order[: min(8, vals.numel())]].sum().item() / counts.sum().item())
        return enc, dec, common, coverage

    def encode_core(self):
        block = 128
        _enc_3bit_vec[((self.n // 8 + block - 1) // block,)](
            self.raw, self.enc_lut, self.code_pk, self.sm, self.n, BLOCK=block
        )

    def encode_full(self):
        self.encode_core()
        _count_escapes_chunk[(self.n_chunks,)](
            self.raw, self.common_lut, self.counts, self.n, CHUNK=self.chunk_size
        )
        offsets = torch.cumsum(self.counts, dim=0)
        self.starts = offsets - self.counts
        n_esc = int(offsets[-1].item()) if offsets.numel() else 0
        if n_esc > self.max_esc:
            raise RuntimeError(f"escape buffer too small: {n_esc} > {self.max_esc}")
        if n_esc:
            _write_bf16_escapes_chunkid[(self.n_chunks,)](
                self.raw, self.common_lut, self.starts, self.chunk_id, self.local_pos, self.esc_val,
                self.n, CHUNK=self.chunk_size
            )
        return n_esc

    def decode_full(self, n_esc: int):
        block = 128
        _dec_3bit_vec[((self.n // 8 + block - 1) // block,)](
            self.code_pk, self.sm, self.dec_lut, self.out, self.n, BLOCK=block
        )
        if n_esc:
            _fix_escapes_local_linear[((n_esc + 127) // 128,)](
                self.chunk_id,
                self.local_pos,
                self.esc_val,
                self.sm,
                self.out,
                n_esc,
                CHUNK=self.chunk_size,
                BLOCK_ESC=128,
            )
        return self.out

    def compressed_bytes(self, n_esc: int):
        return self.code_bytes + self.sm_bytes + n_esc * (4 + 2 + 1)


class Top15SentinelCodec:
    def __init__(self, device: str, scan_block: int):
        self.device = device
        self.scan_block = int(scan_block)
        self.enc_lut = None
        self.dec_lut = None

    def calibrate(self, sample):
        int16 = sample.contiguous().view(torch.int16)
        exponents = ((int16 >> 7) & 0xFF).to(torch.uint8)
        vals, counts = torch.unique(exponents, return_counts=True)
        order = torch.argsort(counts, descending=True)
        self.enc_lut = torch.full((256,), 15, dtype=torch.uint8, device=self.device)
        self.dec_lut = torch.zeros((16,), dtype=torch.uint8, device=self.device)
        top = min(15, vals.numel())
        for i in range(top):
            v = int(vals[order[i]].item())
            self.enc_lut[v] = i
            self.dec_lut[i] = v
        return float(counts[order[:top]].sum().item() / counts.sum().item())

    def encode(self, tensor):
        n = int(tensor.numel())
        int16 = tensor.contiguous().view(torch.int16)
        n_pairs = (n + 1) // 2
        pk = torch.empty(n_pairs, dtype=torch.uint8, device=self.device)
        sm = torch.empty(n, dtype=torch.uint8, device=self.device)
        block = 256
        _enc_4bit[((n_pairs + block * 4 - 1) // (block * 4),)](int16, self.enc_lut, pk, sm, n, BLOCK=block)
        n_blocks = (n_pairs + self.scan_block - 1) // self.scan_block
        counts = torch.empty(n_blocks, dtype=torch.int32, device=self.device)
        _count_sentinel_escapes[(n_blocks,)](pk, counts, n_pairs, n, BLOCK=self.scan_block)
        offsets = torch.cumsum(counts, dim=0)
        starts = offsets - counts
        n_esc = int(offsets[-1].item()) if offsets.numel() else 0
        esc_val = torch.empty(n_esc, dtype=torch.uint8, device=self.device)
        if n_esc:
            _write_sentinel_values[(n_blocks,)](int16, pk, starts, esc_val, n_pairs, n, BLOCK=self.scan_block)
        return pk, sm, starts, esc_val, n, n_esc

    def decode(self, pk, sm, starts, esc_val, n, n_esc):
        out = torch.empty(n, dtype=torch.int16, device=self.device)
        n_pairs = (n + 1) // 2
        block = 256
        _dec_4bit[((n_pairs + block * 4 - 1) // (block * 4),)](pk, sm, self.dec_lut, out, n, BLOCK=block)
        if n_esc:
            n_blocks = (n_pairs + self.scan_block - 1) // self.scan_block
            _fix_sentinel_escapes[(n_blocks,)](pk, starts, esc_val, sm, out, n_pairs, n, BLOCK=self.scan_block)
        return out.view(torch.bfloat16)

    @staticmethod
    def compressed_bytes(encoded):
        pk, sm, _starts, esc_val, _n, _n_esc = encoded
        return pk.numel() + sm.numel() + esc_val.numel()


class FP8Top8ChunkCodecU16:
    def __init__(self, fmt: str, raw: torch.Tensor, chunk_size: int, block: int = 128):
        self.fmt = fmt
        self.fmt_id = 0 if fmt == "e4m3" else 1
        self.raw = raw
        self.n = int(raw.numel())
        self.device = raw.device
        self.chunk_size = int(chunk_size)
        self.core = FP8FixedCodec(fmt, raw, lossless=False, strategy="freq", block=block)
        lut_size = 16 if fmt == "e4m3" else 32
        self.common_lut = torch.zeros(lut_size, dtype=torch.uint8, device=self.device)
        self.common_lut[self.core.dlut.long()] = 1
        self.n_blocks = (self.n + self.chunk_size - 1) // self.chunk_size
        self.counts = torch.empty(self.n_blocks, dtype=torch.int32, device=self.device)
        self.starts = torch.empty(self.n_blocks, dtype=torch.int32, device=self.device)
        self.max_esc = max(self.n // 5, 1024)
        self.local_pos = torch.empty(self.max_esc, dtype=torch.uint16, device=self.device)
        self.esc_val = torch.empty(self.max_esc, dtype=torch.uint8, device=self.device)
        packed_size = ((self.max_esc + 1) // 2) if fmt == "e4m3" else ((self.max_esc * 5 + 7) // 8 + 1)
        self.esc_val_packed = torch.empty(packed_size, dtype=torch.uint8, device=self.device)

    def encode_core(self):
        self.core.encode_core(self.raw)

    def encode_full(self):
        self.encode_core()
        _count_external_fp8[(self.n_blocks,)](
            self.raw, self.common_lut, self.counts, self.n, FMT=self.fmt_id, BLOCK=self.chunk_size
        )
        offsets = torch.cumsum(self.counts, dim=0)
        self.starts = offsets - self.counts
        n_esc = int(offsets[-1].item()) if offsets.numel() else 0
        if n_esc > self.max_esc:
            raise RuntimeError(f"escape buffer too small: {n_esc} > {self.max_esc}")
        if n_esc:
            _write_external_fp8_u16[(self.n_blocks,)](
                self.raw, self.common_lut, self.starts, self.local_pos, self.esc_val,
                self.n, FMT=self.fmt_id, BLOCK=self.chunk_size
            )
            if self.fmt == "e4m3":
                _pack_e4_escape_values[(((n_esc + 1) // 2 + 255) // 256,)](
                    self.esc_val, self.esc_val_packed, n_esc, BLOCK=256
                )
            else:
                _pack_e5_escape_values[(((n_esc + 7) // 8 + 255) // 256,)](
                    self.esc_val, self.esc_val_packed, n_esc, BLOCK=256
                )
        return n_esc

    def decode_full(self, n_esc: int, max_count: int):
        self.core.decode_core()
        if n_esc:
            block_esc = next_power_of_2(max_count)
            if self.fmt == "e4m3":
                _fix_external_e4_packed_u16[(self.n_blocks,)](
                    self.counts, self.starts, self.local_pos, self.esc_val_packed, self.core.out,
                    BLOCK_ELEMS=self.chunk_size, BLOCK_ESC=block_esc
                )
            else:
                _fix_external_e5_packed_u16[(self.n_blocks,)](
                    self.counts, self.starts, self.local_pos, self.esc_val_packed, self.core.out,
                    BLOCK_ELEMS=self.chunk_size, BLOCK_ESC=block_esc
                )
        return self.core.out

    def compressed_bytes(self, n_esc: int):
        esc_val_bytes = ((n_esc + 1) // 2) if self.fmt == "e4m3" else ((n_esc * 5 + 7) // 8)
        return self.core.code_bytes + self.core.sm_bytes + self.n_blocks + n_esc * 2 + esc_val_bytes


class FP8E5M2Top16ChunkCodecU16:
    def __init__(self, raw: torch.Tensor, chunk_size: int, block: int = 128):
        self.raw = raw
        self.device = raw.device
        self.n = int(raw.numel())
        if self.n % 8:
            raise ValueError("E5M2 top16 benchmark expects numel divisible by 8")
        self.chunk_size = int(chunk_size)
        self.block = int(block)
        self.groups = self.n // 8
        self.code_bytes = self.groups * 4
        self.sm_bytes = self.groups * 3
        self.code_pk = torch.empty(self.code_bytes, dtype=torch.uint8, device=self.device)
        self.sm_pk = torch.empty(self.sm_bytes, dtype=torch.uint8, device=self.device)
        self.out = torch.empty_like(raw)
        self.n_blocks = (self.n + self.chunk_size - 1) // self.chunk_size
        self.counts = torch.empty(self.n_blocks, dtype=torch.int32, device=self.device)
        self.starts = torch.empty(self.n_blocks, dtype=torch.int32, device=self.device)
        self.max_esc = max(self.n // 5, 1024)
        self.local_pos = torch.empty(self.max_esc, dtype=torch.uint16, device=self.device)
        self.esc_val = torch.empty(self.max_esc, dtype=torch.uint8, device=self.device)
        self.esc_val_packed = torch.empty((self.max_esc * 5 + 7) // 8 + 1, dtype=torch.uint8, device=self.device)
        self.lut, self.dlut, self.common_lut, self.coverage = self._build_codebook()

    def _build_codebook(self):
        exp = ((self.raw >> 2) & 0x1F).to(torch.uint8)
        vals, counts = torch.unique(exp, return_counts=True)
        order = torch.argsort(counts, descending=True)
        lut = torch.zeros(32, dtype=torch.uint8, device=self.device)
        dlut = torch.zeros(16, dtype=torch.uint8, device=self.device)
        common = torch.zeros(32, dtype=torch.uint8, device=self.device)
        for code in range(min(16, vals.numel())):
            value = int(vals[order[code]].item())
            lut[value] = code
            dlut[code] = value
            common[value] = 1
        coverage = float(counts[order[: min(16, vals.numel())]].sum().item() / counts.sum().item())
        return lut, dlut, common, coverage

    def encode_core(self):
        _enc_e5_top16[((self.groups + self.block - 1) // self.block,)](
            self.raw, self.lut, self.code_pk, self.sm_pk, self.n, BLOCK=self.block
        )

    def encode_full(self):
        self.encode_core()
        _count_external_fp8[(self.n_blocks,)](
            self.raw, self.common_lut, self.counts, self.n, FMT=1, BLOCK=self.chunk_size
        )
        offsets = torch.cumsum(self.counts, dim=0)
        self.starts = offsets - self.counts
        n_esc = int(offsets[-1].item()) if offsets.numel() else 0
        if n_esc > self.max_esc:
            raise RuntimeError(f"escape buffer too small: {n_esc} > {self.max_esc}")
        if n_esc:
            _write_external_fp8_u16[(self.n_blocks,)](
                self.raw, self.common_lut, self.starts, self.local_pos, self.esc_val,
                self.n, FMT=1, BLOCK=self.chunk_size
            )
            _pack_e5_escape_values[(((n_esc + 7) // 8 + 255) // 256,)](
                self.esc_val, self.esc_val_packed, n_esc, BLOCK=256
            )
        return n_esc

    def decode_full(self, n_esc: int, max_count: int):
        _dec_e5_top16[((self.groups + self.block - 1) // self.block,)](
            self.code_pk, self.sm_pk, self.dlut, self.out, self.n, BLOCK=self.block
        )
        if n_esc:
            _fix_external_e5_packed_u16[(self.n_blocks,)](
                self.counts, self.starts, self.local_pos, self.esc_val_packed, self.out,
                BLOCK_ELEMS=self.chunk_size, BLOCK_ESC=next_power_of_2(max_count)
            )
        return self.out

    def compressed_bytes(self, n_esc: int):
        return self.code_bytes + self.sm_bytes + self.n_blocks + n_esc * 2 + ((n_esc * 5 + 7) // 8)


def load_matrix(model: str, device: torch.device, rows: int, width: int):
    blocks, meta = load_model_activation_blocks(model, device)
    return assemble_matrix_width(blocks, rows, width).contiguous(), meta, blocks


def bench_top16_v2(flat: torch.Tensor, chunk_size: int, repeats: int):
    codec = ChunkLocalSplitZipGPU(device=str(flat.device), chunk_size=chunk_size)
    coverage = codec.calibrate(flat)
    encoded = codec.encode(flat)
    decoded = codec.decode(encoded)
    if not torch.equal(flat.view(torch.int16), decoded.view(torch.int16)):
        raise RuntimeError("top16 v2 round-trip failed")
    raw_bytes = flat.numel() * 2
    enc_times = bench_repeats(lambda: codec.encode(flat), repeats=repeats, warmup=5)
    encoded = codec.encode(flat)
    dec_times = bench_repeats(lambda: codec.decode(encoded), repeats=repeats, warmup=5)
    return {
        "coverage": coverage,
        "escapes": int(encoded.n_esc),
        "escape_rate": encoded.n_esc / flat.numel(),
        "compressed_bytes": int(encoded.compressed_bytes),
        "ratio": raw_bytes / encoded.compressed_bytes,
        "encode_gbs": gbs_summary(raw_bytes, enc_times),
        "decode_gbs": gbs_summary(raw_bytes, dec_times),
    }, codec


def bench_bf16_top8(flat: torch.Tensor, chunk_size: int, repeats: int):
    codec = BF16Top8ChunkCodec(flat, chunk_size)
    n_esc = codec.encode_full()
    decoded = codec.decode_full(n_esc)
    if not torch.equal(flat.view(torch.int16), decoded):
        raise RuntimeError("top8 chunk round-trip failed")
    raw_bytes = flat.numel() * 2
    enc_times = bench_repeats(codec.encode_full, repeats=repeats, warmup=5)
    n_esc = codec.encode_full()
    dec_times = bench_repeats(lambda: codec.decode_full(n_esc), repeats=repeats, warmup=5)
    comp = codec.compressed_bytes(n_esc)
    return {
        "coverage": codec.coverage,
        "escapes": int(n_esc),
        "escape_rate": n_esc / flat.numel(),
        "compressed_bytes": int(comp),
        "ratio": raw_bytes / comp,
        "encode_gbs": gbs_summary(raw_bytes, enc_times),
        "decode_gbs": gbs_summary(raw_bytes, dec_times),
    }


def bench_top15_sentinel(flat: torch.Tensor, chunk_size: int, repeats: int):
    codec = Top15SentinelCodec(str(flat.device), scan_block=chunk_size)
    coverage = codec.calibrate(flat)
    encoded = codec.encode(flat)
    decoded = codec.decode(*encoded)
    if not torch.equal(flat.view(torch.int16), decoded.view(torch.int16)):
        raise RuntimeError("top15 sentinel round-trip failed")
    raw_bytes = flat.numel() * 2
    enc_times = bench_repeats(lambda: codec.encode(flat), repeats=repeats, warmup=5)
    encoded = codec.encode(flat)
    dec_times = bench_repeats(lambda: codec.decode(*encoded), repeats=repeats, warmup=5)
    comp = codec.compressed_bytes(encoded)
    return {
        "coverage": coverage,
        "escapes": int(encoded[5]),
        "escape_rate": encoded[5] / flat.numel(),
        "compressed_bytes": int(comp),
        "ratio": raw_bytes / comp,
        "encode_gbs": gbs_summary(raw_bytes, enc_times),
        "decode_gbs": gbs_summary(raw_bytes, dec_times),
    }


def run_dynamic_calibration(flat: torch.Tensor, chunk_size: int, repeats: int):
    pre, _codec = bench_top16_v2(flat, chunk_size, repeats)

    def dynamic_encode():
        codec = ChunkLocalSplitZipGPU(device=str(flat.device), chunk_size=chunk_size)
        codec.calibrate(flat)
        return codec, codec.encode(flat)

    codec, encoded = dynamic_encode()
    decoded = codec.decode(encoded)
    if not torch.equal(flat.view(torch.int16), decoded.view(torch.int16)):
        raise RuntimeError("dynamic top16 round-trip failed")
    raw_bytes = flat.numel() * 2
    dyn_times = []
    for _ in range(2):
        dynamic_encode()
    torch.cuda.synchronize()
    for _ in range(repeats):
        t0 = time.perf_counter()
        codec, encoded = dynamic_encode()
        torch.cuda.synchronize()
        dyn_times.append(time.perf_counter() - t0)
    dec_times = bench_repeats(lambda: codec.decode(encoded), repeats=repeats, warmup=5)
    return {
        "precalibrated": pre,
        "dynamic": {
            "coverage": pre["coverage"],
            "escapes": int(encoded.n_esc),
            "escape_rate": encoded.n_esc / flat.numel(),
            "compressed_bytes": int(encoded.compressed_bytes),
            "ratio": raw_bytes / encoded.compressed_bytes,
            "encode_gbs": gbs_summary(raw_bytes, dyn_times),
            "decode_gbs": gbs_summary(raw_bytes, dec_times),
        },
    }


def topk_coverage_from_codebook(sample_exp, eval_exp, k=16):
    vals, counts = torch.unique(sample_exp, return_counts=True)
    order = torch.argsort(counts, descending=True)
    top_vals = vals[order[: min(k, vals.numel())]]
    lut = torch.zeros((256,), dtype=torch.uint8)
    lut[top_vals.long()] = 1
    return float(lut[eval_exp.long()].float().mean().item())


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
            choices = x.get("choices") or []
            choice_str = " ".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
            texts.append(f"{x['question'].strip()} {choice_str}".strip())
    elif dataset_key == "ptb":
        ds = load_dataset("ptb_text_only", "penn_treebank", split="test")
        texts = [x["sentence"].strip() for x in ds if x["sentence"].strip()]
    else:
        raise ValueError(dataset_key)
    return texts[:count]


def collect_exponents_for_prompts(tokenizer, model, device: torch.device, dataset_key: str, count: int):
    exponents = []
    with torch.inference_mode():
        for prompt in load_prompt_texts(dataset_key, count):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(str(device))
            outputs = model(**inputs, use_cache=True, return_dict=True)
            pkv = outputs.past_key_values
            if hasattr(pkv, "to_legacy_cache"):
                pkv = pkv.to_legacy_cache()
            for key, value in pkv:
                exponents.append(((key.contiguous().view(torch.int16) >> 7) & 0xFF).to(torch.uint8).view(-1).cpu())
                exponents.append(((value.contiguous().view(torch.int16) >> 7) & 0xFF).to(torch.uint8).view(-1).cpu())
    return torch.cat(exponents, dim=0)


def run_cross_dataset(model_name: str, device: torch.device, prompt_count: int):
    specs = [
        ("wikitext_test", "WikiText-2", "Language"),
        ("humaneval", "HumanEval", "Code"),
        ("gsm8k", "GSM8K", "Math"),
        ("mmlu", "MMLU", "Knowledge"),
        ("ptb", "PTB", "Language"),
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map={"": device.index if device.index is not None else 0},
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    try:
        exp_a = collect_exponents_for_prompts(tokenizer, model, device, "wikitext_train", prompt_count)
        rows = []
        for key, label, domain in specs:
            exp_b = collect_exponents_for_prompts(tokenizer, model, device, key, prompt_count)
            rows.append({
                "dataset": label,
                "domain": domain,
                "a_to_b": topk_coverage_from_codebook(exp_a, exp_b, 16),
                "b_to_b": topk_coverage_from_codebook(exp_b, exp_b, 16),
            })
        return {
            "model": model_name,
            "dataset_a": "WikiText-2 train",
            "prompt_count_per_set": prompt_count,
            "a_to_a": topk_coverage_from_codebook(exp_a, exp_a, 16),
            "rows": rows,
        }
    finally:
        del model
        torch.cuda.empty_cache()


def build_granularity_stats(matrix: torch.Tensor, device: torch.device, batch_rows: int = 1024):
    exp = ((matrix.contiguous().view(torch.int16) >> 7) & 0xFF).to(torch.uint8).view(matrix.shape)
    rows, cols = exp.shape

    def finalize(total_common: int, total: int, n_groups: int):
        esc = int(total - total_common)
        codebook_bytes = n_groups * 16
        compressed = (rows * cols) // 2 + rows * cols + esc * 7 + codebook_bytes
        return {
            "coverage": float(total_common / total),
            "escape_rate": float(esc / total),
            "projected_ratio": (rows * cols * 2) / compressed,
            "codebook_bytes": codebook_bytes,
        }

    flat = exp.reshape(-1).to(device=device)
    tensor_hist = torch.bincount(flat.long(), minlength=256)
    tensor_common = int(torch.topk(tensor_hist, min(16, tensor_hist.numel())).values.sum().item())

    token_common = 0
    channel_hist = torch.zeros((cols, 256), dtype=torch.int32, device=device)
    ones_cache = None
    for start in range(0, rows, batch_rows):
        chunk = exp[start:start + batch_rows].to(device=device, non_blocking=False).long()
        if ones_cache is None or tuple(ones_cache.shape) != tuple(chunk.shape):
            ones_cache = torch.ones_like(chunk, dtype=torch.int32)
        token_hist = torch.zeros((chunk.shape[0], 256), dtype=torch.int32, device=device)
        token_hist.scatter_add_(1, chunk, ones_cache)
        token_common += int(torch.topk(token_hist, 16, dim=1).values.sum().item())

        channel_hist.scatter_add_(1, chunk.t().contiguous(), ones_cache.t().contiguous())

    channel_common = int(torch.topk(channel_hist, 16, dim=1).values.sum().item())
    total = rows * cols
    return {
        "per_tensor": finalize(tensor_common, total, 1),
        "per_token": finalize(token_common, total, rows),
        "per_channel": finalize(channel_common, total, cols),
    }


def run_granularity(matrix: torch.Tensor, flat: torch.Tensor, top16_result: dict, chunk_size: int):
    stats = build_granularity_stats(matrix.cpu(), flat.device)
    stats["per_tensor"]["actual_ratio"] = top16_result["ratio"]
    stats["per_tensor"]["encode_gbs"] = top16_result["encode_gbs"]["mean"]
    stats["per_tensor"]["decode_gbs"] = top16_result["decode_gbs"]["mean"]
    stats["per_token"]["actual_ratio"] = None
    stats["per_channel"]["actual_ratio"] = None
    stats["per_token"]["encode_gbs"] = None
    stats["per_token"]["decode_gbs"] = None
    stats["per_channel"]["encode_gbs"] = None
    stats["per_channel"]["decode_gbs"] = None
    stats["note"] = (
        "Throughput is measured for the implemented per-tensor codec. "
        "Per-token and per-channel rows report full-shape coverage/ratio projection; "
        "they require separate per-group codebooks and are not implemented in the current high-throughput kernel."
    )
    stats["chunk_size"] = chunk_size
    return stats


def run_fp8_codec(raw: torch.Tensor, fmt: str, scheme: str, chunk_size: int, repeats: int):
    if scheme == "top8":
        codec = FP8Top8ChunkCodecU16(fmt, raw, chunk_size)
        coverage = codec.core.coverage
    elif scheme == "top16" and fmt == "e5m2":
        codec = FP8E5M2Top16ChunkCodecU16(raw, chunk_size)
        coverage = codec.coverage
    else:
        raise ValueError((fmt, scheme))
    n_esc = codec.encode_full()
    max_count = int(codec.counts.max().item()) if codec.counts.numel() else 0
    decoded = codec.decode_full(n_esc, max_count)
    if not torch.equal(raw, decoded):
        raise RuntimeError(f"FP8 round-trip failed: {fmt} {scheme}")
    raw_bytes = raw.numel()
    enc_times = bench_repeats(codec.encode_full, repeats=repeats, warmup=5)
    n_esc = codec.encode_full()
    max_count = int(codec.counts.max().item()) if codec.counts.numel() else 0
    dec_times = bench_repeats(lambda: codec.decode_full(n_esc, max_count), repeats=repeats, warmup=5)
    comp = codec.compressed_bytes(n_esc)
    return {
        "format": fmt,
        "scheme": scheme,
        "coverage": coverage,
        "escapes": int(n_esc),
        "escape_rate": n_esc / raw.numel(),
        "raw_fp8_bytes": int(raw_bytes),
        "compressed_bytes": int(comp),
        "ratio_vs_fp8": raw_bytes / comp,
        "ratio_vs_bf16": 2.0 * raw_bytes / comp,
        "encode_gbs": gbs_summary(raw_bytes, enc_times),
        "decode_gbs": gbs_summary(raw_bytes, dec_times),
    }


def run_breakdown(blocks, meta, device: torch.device, chunk_size: int, repeats: int):
    dma_cache = {}
    rows = []
    n_layers = int(meta["num_hidden_layers"])
    for seq_len in BREAKDOWN_SEQ_LENS:
        matrix = assemble_matrix_width(blocks, seq_len, HIDDEN_DIM).to(device=device, dtype=torch.bfloat16).contiguous()
        flat = matrix.view(-1)
        result, _ = bench_top16_v2(flat, chunk_size, repeats=max(3, repeats // 2))
        raw_bytes = int(flat.numel() * 2)
        comp_bytes = int(result["compressed_bytes"])
        enc_s = raw_bytes / (result["encode_gbs"]["mean"] * 1e9)
        dec_s = raw_bytes / (result["decode_gbs"]["mean"] * 1e9)
        sim = simulate_transport(
            raw_bytes=raw_bytes,
            comp_bytes=comp_bytes,
            enc_s=enc_s,
            dec_s=dec_s,
            n_layers=n_layers,
            raw_d2h_s=measure_dma_time(raw_bytes, "d2h", device, dma_cache),
            raw_h2d_s=measure_dma_time(raw_bytes, "h2d", device, dma_cache),
            comp_d2h_s=measure_dma_time(comp_bytes, "d2h", device, dma_cache),
            comp_h2d_s=measure_dma_time(comp_bytes, "h2d", device, dma_cache),
            net_gbs=ROCE_GBS,
        )
        bd = sim["splitzip_breakdown_sequential_s"]
        total = bd["total"]
        rows.append({
            "seq_len": seq_len,
            "raw_bytes": raw_bytes,
            "compressed_bytes": comp_bytes,
            "ratio": result["ratio"],
            "native_transfer_ms": sim["native_transfer_sequential_s"] * 1000,
            "splitzip_encode_ms": bd["encode"] * 1000,
            "splitzip_transfer_ms": bd["transfer"] * 1000,
            "splitzip_decode_ms": bd["decode"] * 1000,
            "splitzip_total_ms": total * 1000,
            "encode_pct": bd["encode"] / total * 100,
            "transfer_pct": bd["transfer"] / total * 100,
            "decode_pct": bd["decode"] / total * 100,
        })
    return {"model": MODEL_QWEN32, "transport_mode": "RoCE 4x200G", "rows": rows}


def run_fp8_transfer(blocks, meta, device: torch.device, chunk_size: int, repeats: int):
    dma_cache = {}
    n_layers = int(meta["num_hidden_layers"])
    out = {"model": MODEL_QWEN32, "transport_mode": "RoCE 4x200G", "rows": {
        "e4m3_top8_exact": [],
        "e5m2_top8_exact": [],
        "e5m2_top16_exact": [],
    }}
    specs = [
        ("e4m3_top8_exact", "e4m3", "top8"),
        ("e5m2_top8_exact", "e5m2", "top8"),
        ("e5m2_top16_exact", "e5m2", "top16"),
    ]
    for seq_len in FP8_SEQ_LENS:
        matrix = assemble_matrix_width(blocks, seq_len, HIDDEN_DIM).to(device=device, dtype=torch.bfloat16).contiguous()
        flat = matrix.view(-1)
        for name, fmt, scheme in specs:
            raw = fp8_raw_from_bf16(flat, fmt)
            result = run_fp8_codec(raw, fmt, scheme, chunk_size, repeats=max(3, repeats // 2))
            raw_bytes = int(raw.numel())
            comp_bytes = int(result["compressed_bytes"])
            enc_s = raw_bytes / (result["encode_gbs"]["mean"] * 1e9)
            dec_s = raw_bytes / (result["decode_gbs"]["mean"] * 1e9)
            sim = simulate_transport(
                raw_bytes=raw_bytes,
                comp_bytes=comp_bytes,
                enc_s=enc_s,
                dec_s=dec_s,
                n_layers=n_layers,
                raw_d2h_s=measure_dma_time(raw_bytes, "d2h", device, dma_cache),
                raw_h2d_s=measure_dma_time(raw_bytes, "h2d", device, dma_cache),
                comp_d2h_s=measure_dma_time(comp_bytes, "d2h", device, dma_cache),
                comp_h2d_s=measure_dma_time(comp_bytes, "h2d", device, dma_cache),
                net_gbs=ROCE_GBS,
            )
            out["rows"][name].append({
                "seq_len": seq_len,
                "native_ms": sim["raw_pipe_s"] * 1000,
                "splitzip_ms": sim["splitzip_pipe_s"] * 1000,
                "speedup": sim["speedup"],
                "ratio": result["ratio_vs_fp8"],
                "encode_gbs": result["encode_gbs"]["mean"],
                "decode_gbs": result["decode_gbs"]["mean"],
                "escape_rate": result["escape_rate"],
            })
    return out


def format_mean_pm(summary, digits=1):
    return f"{summary['mean']:.{digits}f}$\\pm${summary['stderr']:.{digits}f}"


def write_fp8_table(path: Path, fp8_results):
    rows = {(r["format"], r["scheme"]): r for r in fp8_results}
    e4 = rows[("e4m3", "top8")]
    e5 = rows[("e5m2", "top8")]
    e516 = rows[("e5m2", "top16")]
    text = f"""\\begin{{figure*}}[t]
    \\centering

    \\begin{{minipage}}[t]{{0.4\\textwidth}}
        \\vspace{{0pt}}
        \\centering
        \\captionof{{table}}{{FP8 codec results.}}
        \\label{{tab:fp8_exact_results}}
        \\footnotesize
        \\setlength{{\\tabcolsep}}{{2.5pt}}
        \\renewcommand{{\\arraystretch}}{{1.12}}
        \\begin{{tabular}}{{@{{}}lccc@{{}}}}
            \\toprule
            \\textbf{{Metric}}
            & \\textbf{{E4M3}}
            & \\textbf{{E5M2}}
            & \\textbf{{E5M2}} \\\\
            & \\textbf{{Top-8}}
            & \\textbf{{Top-8}}
            & \\textbf{{Top-16}} \\\\
            \\midrule
            Coverage
            & {e4['coverage'] * 100:.2f}\\%
            & {e5['coverage'] * 100:.2f}\\%
            & {e516['coverage'] * 100:.2f}\\% \\\\
            Ratio vs. FP8
            & ${e4['ratio_vs_fp8']:.3f}\\times$
            & ${e5['ratio_vs_fp8']:.3f}\\times$
            & ${e516['ratio_vs_fp8']:.3f}\\times$ \\\\
            Ratio vs. BF16
            & ${e4['ratio_vs_bf16']:.3f}\\times$
            & ${e5['ratio_vs_bf16']:.3f}\\times$
            & ${e516['ratio_vs_bf16']:.3f}\\times$ \\\\
            Encode(GB/s)
            & {format_mean_pm(e4['encode_gbs'])}
            & {format_mean_pm(e5['encode_gbs'])}
            & {format_mean_pm(e516['encode_gbs'])} \\\\
            Decode(GB/s)
            & {format_mean_pm(e4['decode_gbs'])}
            & {format_mean_pm(e5['decode_gbs'])}
            & {format_mean_pm(e516['decode_gbs'])} \\\\
            Escape rate
            & {e4['escape_rate'] * 100:.2f}\\%
            & {e5['escape_rate'] * 100:.2f}\\%
            & {e516['escape_rate'] * 100:.2f}\\% \\\\
            \\bottomrule
        \\end{{tabular}}
    \\end{{minipage}}
    \\hfill
    \\begin{{minipage}}[t]{{0.5\\textwidth}}
        \\vspace{{0pt}}
        \\centering
        \\includegraphics[width=\\linewidth]{{lossless-paper/figure/fp8_e2e_speedup_vs_seq_len.pdf}}
        \\vspace{{-1.5em}}
        \\captionof{{figure}}{{
        FP8 end-to-end transfer time on Qwen3-32B under RoCE $4\\times200$G.
        }}
        \\label{{fig:fp8_transfer_time}}
    \\end{{minipage}}
\\vspace{{-2em}}
\\end{{figure*}}
"""
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--rows", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--width", type=int, default=HIDDEN_DIM)
    parser.add_argument("--model", default=MODEL_QWEN32)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--calibration-prompts", type=int, default=4)
    parser.add_argument("--skip-calibration-datasets", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("experiments/splitzip_v2/results/paper_rerun_v2.json"))
    parser.add_argument("--md-output", type=Path, default=Path("experiments/splitzip_v2/results/paper_rerun_v2.md"))
    parser.add_argument("--update-fp8-table", action="store_true")
    args = parser.parse_args()

    device = set_device(args.device)
    print(f"Loading {args.model} activations", flush=True)
    matrix, meta, blocks = load_matrix(args.model, device, args.rows, args.width)
    flat = matrix.to(device=device, dtype=torch.bfloat16).contiguous().view(-1)

    print("Running BF16 Top-16 v2", flush=True)
    top16, _ = bench_top16_v2(flat, args.chunk_size, args.repeats)
    print("Running BF16 Top-8 v2 ablation", flush=True)
    top8 = bench_bf16_top8(flat, args.chunk_size, args.repeats)
    print("Running Top-15 sentinel ablation", flush=True)
    top15 = bench_top15_sentinel(flat, args.chunk_size, args.repeats)
    print("Running dynamic calibration ablation", flush=True)
    dynamic = run_dynamic_calibration(flat, args.chunk_size, max(3, args.repeats // 2))
    print("Running granularity stats", flush=True)
    granularity = run_granularity(matrix, flat, top16, args.chunk_size)

    calibration = None
    if not args.skip_calibration_datasets:
        print("Running cross-dataset calibration", flush=True)
        calibration = run_cross_dataset(args.model, device, args.calibration_prompts)

    print("Running transmission breakdown", flush=True)
    breakdown = run_breakdown(blocks, meta, device, args.chunk_size, args.repeats)

    print("Running FP8 exact codec table", flush=True)
    fp8_results = [
        run_fp8_codec(fp8_raw_from_bf16(flat, "e4m3"), "e4m3", "top8", args.chunk_size, args.repeats),
        run_fp8_codec(fp8_raw_from_bf16(flat, "e5m2"), "e5m2", "top8", args.chunk_size, args.repeats),
        run_fp8_codec(fp8_raw_from_bf16(flat, "e5m2"), "e5m2", "top16", args.chunk_size, args.repeats),
    ]
    print("Running FP8 transfer sweep", flush=True)
    fp8_transfer = run_fp8_transfer(blocks, meta, device, args.chunk_size, args.repeats)

    out = {
        "model": args.model,
        "shape": [args.rows, args.width],
        "chunk_size": args.chunk_size,
        "meta": meta,
        "bf16_topk": {"top8": top8, "top16": top16},
        "escape_metadata": {"top16_positions": top16, "top15_sentinel": top15},
        "precalibration": dynamic,
        "granularity": granularity,
        "calibration": calibration,
        "breakdown": breakdown,
        "fp8_results": fp8_results,
        "fp8_transfer": fp8_transfer,
    }
    write_json(args.output, out)
    args.md_output.write_text(json.dumps(out, indent=2) + "\n")
    if args.update_fp8_table:
        write_fp8_table(Path("lossless-paper/tables/fp8_result.tex"), fp8_results)
    print(f"Wrote {args.output}", flush=True)
    print(f"Wrote {args.md_output}", flush=True)


if __name__ == "__main__":
    main()
