from __future__ import annotations

import struct
from typing import Iterable

import torch

from experiments.splitzip_v2.codec_cpu import ChunkLocalEncoded


MAGIC = b"SZV2BF16"
HEADER_STRUCT = struct.Struct("<8sQIIQQQQQQ")


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _read_tensor(payload: bytes, offset: int, nbytes: int, dtype: torch.dtype):
    segment = bytearray(payload[offset:offset + nbytes])
    if len(segment) != nbytes:
        raise ValueError("truncated SplitZip payload")
    return torch.frombuffer(segment, dtype=dtype).clone(), offset + nbytes


def serialize_chunklocal(encoded: ChunkLocalEncoded) -> bytes:
    shape = tuple(int(x) for x in encoded.shape)
    shape_blob = struct.pack(f"<{len(shape)}Q", *shape) if shape else b""
    packed = _tensor_bytes(encoded.packed_codes)
    sm = _tensor_bytes(encoded.sign_mantissa)
    counts = _tensor_bytes(encoded.chunk_counts.to(torch.int32))
    local_pos = _tensor_bytes(encoded.local_pos.to(torch.uint16))
    esc_val = _tensor_bytes(encoded.escape_values)
    dec_lut = _tensor_bytes(encoded.dec_lut)

    header = HEADER_STRUCT.pack(
        MAGIC,
        int(encoded.n),
        int(encoded.chunk_size),
        len(shape),
        len(packed),
        len(sm),
        len(counts),
        len(local_pos),
        len(esc_val),
        len(dec_lut),
    )
    return header + shape_blob + packed + sm + counts + local_pos + esc_val + dec_lut


def deserialize_chunklocal(payload: bytes) -> ChunkLocalEncoded:
    if len(payload) < HEADER_STRUCT.size:
        raise ValueError("payload is too short for SplitZip header")
    (
        magic,
        n,
        chunk_size,
        ndim,
        packed_len,
        sm_len,
        counts_len,
        local_pos_len,
        esc_val_len,
        dec_lut_len,
    ) = HEADER_STRUCT.unpack_from(payload, 0)
    if magic != MAGIC:
        raise ValueError(f"bad SplitZip magic: {magic!r}")

    offset = HEADER_STRUCT.size
    shape_bytes = ndim * 8
    if len(payload) < offset + shape_bytes:
        raise ValueError("payload is too short for SplitZip shape")
    shape = struct.unpack_from(f"<{ndim}Q", payload, offset) if ndim else ()
    offset += shape_bytes

    packed, offset = _read_tensor(payload, offset, packed_len, torch.uint8)
    sm, offset = _read_tensor(payload, offset, sm_len, torch.uint8)
    counts, offset = _read_tensor(payload, offset, counts_len, torch.int32)
    local_pos, offset = _read_tensor(payload, offset, local_pos_len, torch.uint16)
    esc_val, offset = _read_tensor(payload, offset, esc_val_len, torch.uint8)
    dec_lut, offset = _read_tensor(payload, offset, dec_lut_len, torch.uint8)
    if offset != len(payload):
        raise ValueError("payload has trailing bytes")

    return ChunkLocalEncoded(
        packed_codes=packed,
        sign_mantissa=sm,
        chunk_counts=counts,
        local_pos=local_pos,
        escape_values=esc_val,
        n=int(n),
        chunk_size=int(chunk_size),
        dec_lut=dec_lut,
        coverage=float("nan"),
        shape=tuple(int(x) for x in shape),
    )


def payload_size_breakdown(encoded: ChunkLocalEncoded) -> dict[str, int]:
    return {
        "header": HEADER_STRUCT.size + 8 * len(encoded.shape),
        "packed_codes": int(encoded.packed_codes.numel()),
        "sign_mantissa": int(encoded.sign_mantissa.numel()),
        "chunk_counts": int(encoded.chunk_counts.numel() * 4),
        "local_pos_uint16": int(encoded.local_pos.numel() * 2),
        "escape_values": int(encoded.escape_values.numel()),
        "decode_lut": int(encoded.dec_lut.numel()),
    }
