from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

from experiments.splitzip.lossless_fast import _dec_4bit, _enc_4bit


@triton.jit
def _dec_4bit_pair32(pk, sm, dlut, out32, n_pairs, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_pairs
    packed = tl.load(pk + offs, mask=mask, other=0).to(tl.int32)
    pos0 = offs * 2
    pos1 = pos0 + 1

    exp0 = tl.load(dlut + ((packed >> 4) & 0x0F), mask=mask, other=0).to(tl.int32)
    exp1 = tl.load(dlut + (packed & 0x0F), mask=mask, other=0).to(tl.int32)
    sm0 = tl.load(sm + pos0, mask=mask, other=0).to(tl.int32)
    sm1 = tl.load(sm + pos1, mask=mask, other=0).to(tl.int32)

    raw0 = ((sm0 & 0x80) << 8) | (exp0 << 7) | (sm0 & 0x7F)
    raw1 = ((sm1 & 0x80) << 8) | (exp1 << 7) | (sm1 & 0x7F)
    pair = (raw0 & 0xFFFF) | ((raw1 & 0xFFFF) << 16)
    tl.store(out32 + offs, pair, mask=mask)


@triton.jit
def _dec_4bit_quad64(pk16, sm32, dlut, out64, n_quads, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_quads
    packed = tl.load(pk16 + offs, mask=mask, other=0).to(tl.int32)
    p0 = packed & 0xFF
    p1 = (packed >> 8) & 0xFF

    sm_pack = tl.load(sm32 + offs, mask=mask, other=0).to(tl.int32)
    sm0 = sm_pack & 0xFF
    sm1 = (sm_pack >> 8) & 0xFF
    sm2 = (sm_pack >> 16) & 0xFF
    sm3 = (sm_pack >> 24) & 0xFF

    exp0 = tl.load(dlut + ((p0 >> 4) & 0x0F), mask=mask, other=0).to(tl.int32)
    exp1 = tl.load(dlut + (p0 & 0x0F), mask=mask, other=0).to(tl.int32)
    exp2 = tl.load(dlut + ((p1 >> 4) & 0x0F), mask=mask, other=0).to(tl.int32)
    exp3 = tl.load(dlut + (p1 & 0x0F), mask=mask, other=0).to(tl.int32)

    raw0 = ((sm0 & 0x80) << 8) | (exp0 << 7) | (sm0 & 0x7F)
    raw1 = ((sm1 & 0x80) << 8) | (exp1 << 7) | (sm1 & 0x7F)
    raw2 = ((sm2 & 0x80) << 8) | (exp2 << 7) | (sm2 & 0x7F)
    raw3 = ((sm3 & 0x80) << 8) | (exp3 << 7) | (sm3 & 0x7F)

    quad = (
        raw0.to(tl.uint64)
        | (raw1.to(tl.uint64) << 16)
        | (raw2.to(tl.uint64) << 32)
        | (raw3.to(tl.uint64) << 48)
    )
    tl.store(out64 + offs, quad, mask=mask)


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
def _write_escapes_chunk(inp, common_lut, starts, chunk_id, local_pos, esc_val,
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


@triton.jit
def _fix_escapes_local_linear(chunk_id, local_pos, esc_val, sm, out,
                              n_esc, CHUNK: tl.constexpr, BLOCK_ESC: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_ESC + tl.arange(0, BLOCK_ESC)
    mask = offs < n_esc
    chunk = tl.load(chunk_id + offs, mask=mask, other=0).to(tl.int32)
    local = tl.load(local_pos + offs, mask=mask, other=0).to(tl.int32)
    exp = tl.load(esc_val + offs, mask=mask, other=0).to(tl.int32)
    pos = chunk * CHUNK + local
    s = tl.load(sm + pos, mask=mask, other=0).to(tl.int32)
    fixed = ((s & 0x80) << 8) | (exp << 7) | (s & 0x7F)
    tl.store(out + pos, fixed.to(tl.int16), mask=mask)

@dataclass
class ChunkLocalGPUEncoded:
    pk: torch.Tensor
    sm: torch.Tensor
    counts: torch.Tensor
    starts: torch.Tensor
    chunk_id: torch.Tensor
    local_pos: torch.Tensor
    esc_val: torch.Tensor
    n: int
    n_esc: int
    chunk_size: int

    @property
    def compressed_bytes(self) -> int:
        # counts/starts are construction scratch data; decode uses compact chunk_id
        # plus local offsets, so they are not part of the transmitted payload.
        return (
            self.pk.numel()
            + self.sm.numel()
            + self.chunk_id.numel() * 4
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
        chunk_id = torch.empty(n_esc, dtype=torch.int32, device=self.device)
        local_pos = torch.empty(n_esc, dtype=torch.uint16, device=self.device)
        esc_val = torch.empty(n_esc, dtype=torch.uint8, device=self.device)
        if n_esc:
            _write_escapes_chunk[(n_chunks,)](
                flat, self.common_lut, starts, chunk_id, local_pos, esc_val, n, CHUNK=self.chunk_size
            )
        return ChunkLocalGPUEncoded(pk, sm, counts, starts, chunk_id, local_pos, esc_val, n, n_esc, self.chunk_size)

    def decode(self, encoded: ChunkLocalGPUEncoded) -> torch.Tensor:
        self._check_ready()
        n_pairs = (encoded.n + 1) // 2
        out = torch.empty(encoded.n, dtype=torch.int16, device=self.device)
        if encoded.n % 4 == 0:
            block = 512
            n_quads = encoded.n // 4
            _dec_4bit_quad64[((n_quads + block - 1) // block,)](
                encoded.pk.view(torch.int16),
                encoded.sm.view(torch.int32),
                self.dec_lut,
                out.view(torch.int64),
                n_quads,
                BLOCK=block,
                num_warps=4,
            )
        elif encoded.n % 2 == 0:
            block = 1024
            _dec_4bit_pair32[((n_pairs + block - 1) // block,)](
                encoded.pk, encoded.sm, self.dec_lut, out.view(torch.int32), n_pairs, BLOCK=block, num_warps=4
            )
        else:
            block = 1024
            _dec_4bit[((n_pairs + block * 4 - 1) // (block * 4),)](
                encoded.pk, encoded.sm, self.dec_lut, out, encoded.n, BLOCK=block, num_warps=4
            )
        if encoded.n_esc:
            block_esc = 128
            _fix_escapes_local_linear[((encoded.n_esc + block_esc - 1) // block_esc,)](
                encoded.chunk_id,
                encoded.local_pos,
                encoded.esc_val,
                encoded.sm,
                out,
                encoded.n_esc,
                CHUNK=encoded.chunk_size,
                BLOCK_ESC=block_esc,
            )
        return out.view(torch.bfloat16)
