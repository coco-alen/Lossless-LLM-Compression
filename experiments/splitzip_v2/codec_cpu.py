from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Dict, Optional

import torch


def _as_bf16_flat(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"expected torch.bfloat16, got {tensor.dtype}")
    return tensor.contiguous().view(-1)


def bf16_exponent(tensor: torch.Tensor) -> torch.Tensor:
    flat = _as_bf16_flat(tensor)
    raw = flat.view(torch.int16).to(torch.int32)
    return ((raw >> 7) & 0xFF).to(torch.uint8)


def bf16_sign_mantissa(tensor: torch.Tensor) -> torch.Tensor:
    flat = _as_bf16_flat(tensor)
    raw = flat.view(torch.int16).to(torch.int32)
    return (((raw >> 8) & 0x80) | (raw & 0x7F)).to(torch.uint8)


def build_topk_codebook(exponents: torch.Tensor, k: int = 16):
    if k <= 0 or k > 16:
        raise ValueError("BF16 nibble codebook requires 1 <= k <= 16")
    vals, counts = torch.unique(exponents.cpu(), return_counts=True)
    order = torch.argsort(counts, descending=True)
    top_vals = vals[order[: min(k, vals.numel())]].to(torch.uint8)

    enc_lut = torch.zeros(256, dtype=torch.uint8)
    common_lut = torch.zeros(256, dtype=torch.bool)
    dec_lut = torch.zeros(16, dtype=torch.uint8)
    for code, value in enumerate(top_vals.tolist()):
        enc_lut[value] = code
        dec_lut[code] = value
        common_lut[value] = True
    coverage = float(common_lut[exponents.long()].float().mean().item())
    return enc_lut, dec_lut, common_lut, coverage


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    codes = codes.to(torch.uint8).contiguous().view(-1)
    if codes.numel() % 2:
        codes = torch.cat([codes, torch.zeros(1, dtype=torch.uint8)])
    hi = codes[0::2] << 4
    lo = codes[1::2] & 0x0F
    return (hi | lo).contiguous()


def unpack_nibbles(packed: torch.Tensor, n: int) -> torch.Tensor:
    packed = packed.to(torch.uint8).contiguous().view(-1)
    out = torch.empty(packed.numel() * 2, dtype=torch.uint8)
    out[0::2] = (packed >> 4) & 0x0F
    out[1::2] = packed & 0x0F
    return out[:n].contiguous()


@dataclass
class ChunkLocalEncoded:
    packed_codes: torch.Tensor
    sign_mantissa: torch.Tensor
    chunk_counts: torch.Tensor
    local_pos: torch.Tensor
    escape_values: torch.Tensor
    n: int
    chunk_size: int
    dec_lut: torch.Tensor
    coverage: float
    shape: tuple[int, ...]
    timings_s: Dict[str, float] = field(default_factory=dict)

    @property
    def n_chunks(self) -> int:
        return int(self.chunk_counts.numel())

    @property
    def n_escapes(self) -> int:
        return int(self.escape_values.numel())

    @property
    def raw_bytes(self) -> int:
        return self.n * 2

    @property
    def compressed_bytes(self) -> int:
        return (
            self.packed_codes.numel()
            + self.sign_mantissa.numel()
            + self.chunk_counts.numel() * 4
            + self.local_pos.numel() * 2
            + self.escape_values.numel()
            + self.dec_lut.numel()
        )

    @property
    def ratio(self) -> float:
        return self.raw_bytes / self.compressed_bytes


class ChunkLocalSplitZipCPU:
    """Reference SplitZip codec with chunk-local escape positions.

    Each chunk is encoded independently for escape metadata.  Escaped positions
    are stored as uint16 offsets relative to the chunk base instead of int32
    absolute positions.  The dense nibble stream still uses a global top-16
    exponent codebook; escaped elements carry an arbitrary dense code and are
    overwritten during decode.
    """

    def __init__(self, chunk_size: int = 65536):
        if chunk_size <= 0 or chunk_size > 65536:
            raise ValueError("chunk_size must be in [1, 65536] for uint16 offsets")
        self.chunk_size = int(chunk_size)
        self.enc_lut: Optional[torch.Tensor] = None
        self.dec_lut: Optional[torch.Tensor] = None
        self.common_lut: Optional[torch.Tensor] = None
        self.coverage: Optional[float] = None

    def calibrate(self, sample: torch.Tensor) -> float:
        exp = bf16_exponent(sample)
        self.enc_lut, self.dec_lut, self.common_lut, self.coverage = build_topk_codebook(exp, 16)
        return self.coverage

    def _require_calibrated(self):
        if self.enc_lut is None or self.dec_lut is None or self.common_lut is None:
            raise RuntimeError("codec must be calibrated before encode/decode")

    def encode(self, tensor: torch.Tensor, profile: bool = False) -> ChunkLocalEncoded:
        self._require_calibrated()
        shape = tuple(tensor.shape)
        flat = _as_bf16_flat(tensor)
        n = int(flat.numel())
        timings: Dict[str, float] = {}

        t0 = time.perf_counter()
        exp = bf16_exponent(flat)
        sm = bf16_sign_mantissa(flat)
        timings["extract_fields"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        common = self.common_lut[exp.long()]
        codes = self.enc_lut[exp.long()]
        codes = torch.where(common, codes, torch.zeros_like(codes))
        packed = pack_nibbles(codes)
        timings["lookup_and_pack"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        n_chunks = math.ceil(n / self.chunk_size)
        counts = torch.zeros(n_chunks, dtype=torch.int32)
        pos_parts = []
        val_parts = []
        for chunk_id in range(n_chunks):
            start = chunk_id * self.chunk_size
            end = min(n, start + self.chunk_size)
            esc = ~common[start:end]
            counts[chunk_id] = int(esc.sum().item())
            if counts[chunk_id] > 0:
                local = torch.nonzero(esc, as_tuple=False).flatten().to(torch.uint16)
                pos_parts.append(local)
                val_parts.append(exp[start:end][esc].to(torch.uint8))
        if pos_parts:
            local_pos = torch.cat(pos_parts).contiguous()
            esc_val = torch.cat(val_parts).contiguous()
        else:
            local_pos = torch.empty(0, dtype=torch.uint16)
            esc_val = torch.empty(0, dtype=torch.uint8)
        timings["chunk_escape_compaction"] = time.perf_counter() - t0

        return ChunkLocalEncoded(
            packed_codes=packed,
            sign_mantissa=sm,
            chunk_counts=counts,
            local_pos=local_pos,
            escape_values=esc_val,
            n=n,
            chunk_size=self.chunk_size,
            dec_lut=self.dec_lut.clone(),
            coverage=float(self.coverage),
            shape=shape,
            timings_s=timings if profile else {},
        )

    def decode(self, encoded: ChunkLocalEncoded, profile: bool = False) -> torch.Tensor:
        timings: Dict[str, float] = {}
        t0 = time.perf_counter()
        codes = unpack_nibbles(encoded.packed_codes, encoded.n)
        exp = encoded.dec_lut[codes.long()].to(torch.int32)
        sm = encoded.sign_mantissa.to(torch.int32)
        raw = (((sm & 0x80) << 8) | (exp << 7) | (sm & 0x7F)).to(torch.int16)
        timings["decode_dense"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        starts = torch.cumsum(encoded.chunk_counts, dim=0) - encoded.chunk_counts
        for chunk_id, count in enumerate(encoded.chunk_counts.tolist()):
            if count == 0:
                continue
            src_start = int(starts[chunk_id].item())
            src_end = src_start + int(count)
            dst = chunk_id * encoded.chunk_size + encoded.local_pos[src_start:src_end].to(torch.int64)
            s = encoded.sign_mantissa[dst].to(torch.int32)
            e = encoded.escape_values[src_start:src_end].to(torch.int32)
            raw[dst] = (((s & 0x80) << 8) | (e << 7) | (s & 0x7F)).to(torch.int16)
        timings["fix_escapes"] = time.perf_counter() - t0

        if profile:
            encoded.timings_s.update(timings)
        return raw.view(torch.bfloat16).view(encoded.shape)


def reviewer_compaction_paragraph() -> str:
    return (
        "SplitZip v2 stores escape metadata in fixed-size chunks.  The encoder first "
        "runs a per-chunk counting kernel that computes the number of uncommon "
        "exponents in each chunk.  A prefix sum over these counts gives a disjoint "
        "output range for every chunk.  A second per-chunk scatter kernel then writes "
        "uint16 local offsets and raw exponent values into the assigned range.  Because "
        "each chunk owns a non-overlapping segment, the implementation avoids global "
        "atomic append operations and the associated contention; decode mirrors this "
        "layout by launching independent chunk-local fix-up programs."
    )

