from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

from experiments.splitzip.lossless_fast import _dec_4bit, _enc_4bit


@triton.jit
def _count_escapes_chunk(inp, common_lut, counts, n: tl.constexpr, CHUNK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * CHUNK + tl.arange(0, CHUNK)
    mask = offs < n
    raw = tl.load(inp + offs, mask=mask, other=0).to(tl.int32)
    exp = (raw >> 7) & 0xFF
    common = tl.load(common_lut + exp, mask=mask, other=1).to(tl.int32)
    esc = (common == 0) & mask
    tl.store(counts + pid, tl.sum(esc.to(tl.int32), axis=0))


@triton.jit
def _write_escapes_chunk(inp, common_lut, starts, local_pos, esc_val,
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
    tl.store(local_pos + out, (offs - base).to(tl.uint16), mask=esc)
    tl.store(esc_val + out, exp.to(tl.uint8), mask=esc)


@triton.jit
def _fix_escapes_chunk(counts, starts, local_pos, esc_val, sm, out,
                       CHUNK: tl.constexpr, BLOCK_ESC: tl.constexpr):
    pid = tl.program_id(0)
    count = tl.load(counts + pid)
    start = tl.load(starts + pid)
    offs = tl.arange(0, BLOCK_ESC)
    mask = offs < count
    src = start + offs
    local = tl.load(local_pos + src, mask=mask, other=0).to(tl.int32)
    exp = tl.load(esc_val + src, mask=mask, other=0).to(tl.int32)
    pos = pid * CHUNK + local
    s = tl.load(sm + pos, mask=mask, other=0).to(tl.int32)
    fixed = ((s & 0x80) << 8) | (exp << 7) | (s & 0x7F)
    tl.store(out + pos, fixed.to(tl.int16), mask=mask)


@triton.jit
def _fix_escapes_chunk_blocked(counts, starts, local_pos, esc_val, sm, out,
                               CHUNK: tl.constexpr, BLOCK_ESC: tl.constexpr):
    chunk = tl.program_id(0)
    block = tl.program_id(1)
    count = tl.load(counts + chunk)
    start = tl.load(starts + chunk)
    offs = block * BLOCK_ESC + tl.arange(0, BLOCK_ESC)
    mask = offs < count
    src = start + offs
    local = tl.load(local_pos + src, mask=mask, other=0).to(tl.int32)
    exp = tl.load(esc_val + src, mask=mask, other=0).to(tl.int32)
    pos = chunk * CHUNK + local
    s = tl.load(sm + pos, mask=mask, other=0).to(tl.int32)
    fixed = ((s & 0x80) << 8) | (exp << 7) | (s & 0x7F)
    tl.store(out + pos, fixed.to(tl.int16), mask=mask)


def _next_power_of_2(x: int) -> int:
    return 1 << (max(1, int(x)) - 1).bit_length()


@dataclass
class ChunkLocalGPUEncoded:
    pk: torch.Tensor
    sm: torch.Tensor
    counts: torch.Tensor
    starts: torch.Tensor
    local_pos: torch.Tensor
    esc_val: torch.Tensor
    n: int
    n_esc: int
    chunk_size: int

    @property
    def compressed_bytes(self) -> int:
        return (
            self.pk.numel()
            + self.sm.numel()
            + self.counts.numel() * 4
            + self.local_pos.numel() * 2
            + self.esc_val.numel()
        )


class ChunkLocalSplitZipGPU:
    """Triton chunk-local SplitZip implementation.

    Escape collection is two-pass and lock-free at the global level:
    per-chunk count, prefix sum, per-chunk scatter into disjoint ranges.
    """

    def __init__(self, device: str = "cuda", chunk_size: int = 1024):
        if chunk_size <= 0 or chunk_size > 65536:
            raise ValueError("chunk_size must be in [1, 65536] for uint16 offsets")
        if chunk_size & (chunk_size - 1):
            raise ValueError("chunk_size must be a power of two for Triton blocks")
        self.device = device
        self.chunk_size = int(chunk_size)
        self.enc_lut: Optional[torch.Tensor] = None
        self.dec_lut: Optional[torch.Tensor] = None
        self.common_lut: Optional[torch.Tensor] = None

    def calibrate(self, sample: torch.Tensor) -> float:
        flat = sample.contiguous().view(torch.int16)
        exp = ((flat >> 7) & 0xFF).to(torch.uint8)
        vals, counts = torch.unique(exp, return_counts=True)
        order = torch.argsort(counts, descending=True)
        self.enc_lut = torch.zeros(256, dtype=torch.uint8, device=self.device)
        self.dec_lut = torch.zeros(16, dtype=torch.uint8, device=self.device)
        self.common_lut = torch.zeros(256, dtype=torch.uint8, device=self.device)
        top = min(16, vals.numel())
        for code in range(top):
            value = vals[order[code]].item()
            self.enc_lut[value] = code
            self.dec_lut[code] = value
            self.common_lut[value] = 1
        return float(counts[order[:top]].sum().item() / counts.sum().item())

    def _check_ready(self):
        if self.enc_lut is None or self.dec_lut is None or self.common_lut is None:
            raise RuntimeError("codec must be calibrated")

    def encode(self, tensor: torch.Tensor) -> ChunkLocalGPUEncoded:
        self._check_ready()
        flat = tensor.contiguous().view(torch.int16)
        n = int(flat.numel())
        n_pairs = (n + 1) // 2
        pk = torch.empty(n_pairs, dtype=torch.uint8, device=self.device)
        sm = torch.empty(n, dtype=torch.uint8, device=self.device)

        block = 256
        _enc_4bit[((n_pairs + block * 4 - 1) // (block * 4),)](
            flat, self.enc_lut, pk, sm, n, BLOCK=block
        )

        n_chunks = (n + self.chunk_size - 1) // self.chunk_size
        counts = torch.empty(n_chunks, dtype=torch.int32, device=self.device)
        _count_escapes_chunk[(n_chunks,)](
            flat, self.common_lut, counts, n, CHUNK=self.chunk_size
        )
        offsets = torch.cumsum(counts, dim=0)
        starts = offsets - counts
        n_esc = int(offsets[-1].item()) if offsets.numel() else 0
        local_pos = torch.empty(n_esc, dtype=torch.uint16, device=self.device)
        esc_val = torch.empty(n_esc, dtype=torch.uint8, device=self.device)
        if n_esc:
            _write_escapes_chunk[(n_chunks,)](
                flat, self.common_lut, starts, local_pos, esc_val, n, CHUNK=self.chunk_size
            )
        return ChunkLocalGPUEncoded(pk, sm, counts, starts, local_pos, esc_val, n, n_esc, self.chunk_size)

    def decode(self, encoded: ChunkLocalGPUEncoded) -> torch.Tensor:
        self._check_ready()
        n_pairs = (encoded.n + 1) // 2
        out = torch.empty(encoded.n, dtype=torch.int16, device=self.device)
        block = 256
        _dec_4bit[((n_pairs + block * 4 - 1) // (block * 4),)](
            encoded.pk, encoded.sm, self.dec_lut, out, encoded.n, BLOCK=block
        )
        if encoded.n_esc:
            max_count = int(encoded.counts.max().item())
            block_esc = min(1024, _next_power_of_2(max_count))
            grid = (encoded.counts.numel(), (max_count + block_esc - 1) // block_esc)
            _fix_escapes_chunk_blocked[grid](
                encoded.counts,
                encoded.starts,
                encoded.local_pos,
                encoded.esc_val,
                encoded.sm,
                out,
                CHUNK=encoded.chunk_size,
                BLOCK_ESC=block_esc,
            )
        return out.view(torch.bfloat16)
