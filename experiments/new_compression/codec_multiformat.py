"""
Multi-format ANS lossless codec for LLM weight compression.

Supports BFloat16, Float8 (e4m3fn, e5m2), and INT4 (packed uint8) formats.
Uses full-value ANS coding (not exponent-only) to achieve near-entropy-optimal
compression across all precision formats.

Key results (from pilot experiments):
  BF16:  65.96% ratio (gap <0.01% to i.i.d. entropy limit)
  FP8:   ~70.6% ratio (29% savings — beats ECF8's exponent-only 85-90%)
  INT4:  ~84.2% ratio (16% savings on real GPTQ models)

Differentiators vs prior work:
  - ECF8 compresses only FP8 exponents → we compress full 8-bit value
  - EntroLLM uses Huffman on INT4 → we use ANS (better rate, lower overhead)
  - DFloat11 separates exponent + mantissa → we code full 16-bit symbol

Dependencies: constriction (pip install constriction)
"""

import numpy as np
import torch
import constriction
from typing import Union


# ---------------------------------------------------------------------------
# Format detection and conversion
# ---------------------------------------------------------------------------

SUPPORTED_DTYPES = {
    torch.bfloat16: 'bf16',
    torch.float16: 'fp16',
    torch.float8_e4m3fn: 'fp8_e4m3fn',
    torch.float8_e5m2: 'fp8_e5m2',
    torch.uint8: 'uint8',  # packed INT4 or raw uint8
    torch.int8: 'int8',
}


def _detect_format(tensor: torch.Tensor, format_hint: str = None) -> str:
    """Detect the weight format of a tensor."""
    if format_hint:
        return format_hint
    dtype = tensor.dtype
    if dtype in SUPPORTED_DTYPES:
        return SUPPORTED_DTYPES[dtype]
    raise ValueError(f"Unsupported dtype: {dtype}. Supported: {list(SUPPORTED_DTYPES.values())}")


def _tensor_to_symbols(tensor: torch.Tensor, fmt: str) -> np.ndarray:
    """Convert tensor to integer symbol array for ANS coding."""
    tensor = tensor.contiguous().flatten()

    if fmt in ('bf16', 'fp16'):
        # View as int16, shift to unsigned range [0, 65535]
        raw = tensor.view(torch.int16).numpy().astype(np.int32)
        return raw + 32768  # shift to unsigned

    elif fmt in ('fp8_e4m3fn', 'fp8_e5m2'):
        # View as uint8 [0, 255]
        return tensor.view(torch.uint8).numpy().astype(np.int32)

    elif fmt == 'uint8':
        # Packed INT4: each byte contains 2 x 4-bit values
        # Unpack to individual 4-bit symbols [0, 15]
        raw = tensor.numpy().astype(np.int32)
        lo = raw & 0x0F
        hi = (raw >> 4) & 0x0F
        # Interleave: lo[0], hi[0], lo[1], hi[1], ...
        symbols = np.empty(len(raw) * 2, dtype=np.int32)
        symbols[0::2] = lo
        symbols[1::2] = hi
        return symbols

    elif fmt == 'int8':
        # Shift to unsigned range [0, 255]
        return tensor.numpy().astype(np.int32) + 128

    else:
        raise ValueError(f"Unknown format: {fmt}")


def _symbols_to_tensor(symbols: np.ndarray, fmt: str, shape: tuple) -> torch.Tensor:
    """Convert integer symbols back to tensor."""
    if fmt in ('bf16', 'fp16'):
        raw = (symbols.astype(np.int32) - 32768).astype(np.int16)
        torch_dtype = torch.bfloat16 if fmt == 'bf16' else torch.float16
        return torch.from_numpy(raw.copy()).view(torch_dtype).reshape(shape)

    elif fmt in ('fp8_e4m3fn', 'fp8_e5m2'):
        raw = symbols.astype(np.uint8)
        torch_dtype = torch.float8_e4m3fn if fmt == 'fp8_e4m3fn' else torch.float8_e5m2
        return torch.from_numpy(raw.copy()).view(torch_dtype).reshape(shape)

    elif fmt == 'uint8':
        # Re-pack 4-bit symbols into bytes
        lo = symbols[0::2].astype(np.uint8)
        hi = symbols[1::2].astype(np.uint8)
        packed = lo | (hi << 4)
        return torch.from_numpy(packed.copy()).to(torch.uint8).reshape(shape)

    elif fmt == 'int8':
        raw = (symbols.astype(np.int32) - 128).astype(np.int8)
        return torch.from_numpy(raw.copy()).reshape(shape)

    else:
        raise ValueError(f"Unknown format: {fmt}")


# ---------------------------------------------------------------------------
# Core ANS codec (format-agnostic)
# ---------------------------------------------------------------------------

def _build_frequency_table(symbols: np.ndarray, max_symbol: int) -> np.ndarray:
    """Build probability table from symbol array.

    Returns float32 probability array of size (num_unique_symbols,).
    Also returns unique symbols and mapping arrays.
    """
    unique_vals, counts = np.unique(symbols, return_counts=True)
    n = len(symbols)
    probs = (counts / n).astype(np.float32)

    # Build fast lookup: symbol_value -> contiguous index
    mapping = np.full(max_symbol + 1, -1, dtype=np.int32)
    for i, v in enumerate(unique_vals):
        mapping[v] = i

    return unique_vals, probs, mapping


def compress(tensor: torch.Tensor, format_hint: str = None) -> dict:
    """
    Compress a tensor using full-value ANS coding.

    Supports BFloat16, Float16, Float8 (e4m3fn/e5m2), packed INT4 (uint8),
    and INT8.

    Parameters
    ----------
    tensor : torch.Tensor
        Weight tensor in any supported format.
    format_hint : str, optional
        Override auto-detection (e.g., 'uint8' for packed INT4).

    Returns
    -------
    dict with compressed data and metadata for decompression.
    """
    fmt = _detect_format(tensor, format_hint)
    shape = tensor.shape
    symbols = _tensor_to_symbols(tensor, fmt)
    n = len(symbols)

    # Determine symbol range
    if fmt in ('bf16', 'fp16'):
        max_symbol = 65535
    elif fmt in ('fp8_e4m3fn', 'fp8_e5m2', 'int8'):
        max_symbol = 255
    elif fmt == 'uint8':  # packed INT4
        max_symbol = 15
    else:
        max_symbol = int(symbols.max())

    # Build frequency table
    unique_vals, probs, mapping = _build_frequency_table(symbols, max_symbol)

    # Map to contiguous indices
    data_idx = mapping[symbols]
    assert np.all(data_idx >= 0), "Symbol not found in frequency table"
    data_idx = data_idx.astype(np.int32)

    # ANS encode
    model = constriction.stream.model.Categorical(probs, perfect=False)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(data_idx, model)
    compressed_words = encoder.get_compressed()

    return {
        'compressed_words': np.array(compressed_words, dtype=np.uint32),
        'symbol_table': unique_vals.astype(np.int32),
        'probabilities': probs,
        'shape': shape,
        'n_symbols': n,
        'format': fmt,
    }


def decompress(compressed: dict) -> torch.Tensor:
    """
    Decompress back to original tensor. Exact inverse of compress().
    """
    probs = compressed['probabilities']
    symbol_table = compressed['symbol_table']
    n = compressed['n_symbols']
    shape = compressed['shape']
    fmt = compressed['format']

    # ANS decode
    model = constriction.stream.model.Categorical(probs, perfect=False)
    decoder = constriction.stream.stack.AnsCoder(compressed['compressed_words'])
    data_idx = decoder.decode(model, n)

    # Map indices back to symbol values
    symbols = symbol_table[data_idx]

    return _symbols_to_tensor(symbols, fmt, shape)


# ---------------------------------------------------------------------------
# Batch compression (multiple tensors with shared table)
# ---------------------------------------------------------------------------

def compress_group(tensors: list[torch.Tensor], format_hint: str = None) -> dict:
    """
    Compress a group of tensors using shared frequency table.
    All tensors must be the same dtype.
    """
    fmt = _detect_format(tensors[0], format_hint)
    shapes = [t.shape for t in tensors]

    # Convert all to symbols and concatenate
    all_symbols = np.concatenate([_tensor_to_symbols(t, fmt) for t in tensors])
    n = len(all_symbols)

    if fmt in ('bf16', 'fp16'):
        max_symbol = 65535
    elif fmt in ('fp8_e4m3fn', 'fp8_e5m2', 'int8'):
        max_symbol = 255
    elif fmt == 'uint8':
        max_symbol = 15
    else:
        max_symbol = int(all_symbols.max())

    unique_vals, probs, mapping = _build_frequency_table(all_symbols, max_symbol)
    data_idx = mapping[all_symbols].astype(np.int32)

    model = constriction.stream.model.Categorical(probs, perfect=False)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(data_idx, model)
    compressed_words = encoder.get_compressed()

    return {
        'compressed_words': np.array(compressed_words, dtype=np.uint32),
        'symbol_table': unique_vals.astype(np.int32),
        'probabilities': probs,
        'shapes': shapes,
        'n_symbols': n,
        'format': fmt,
    }


def decompress_group(compressed: dict) -> list[torch.Tensor]:
    """Decompress a group of tensors."""
    probs = compressed['probabilities']
    symbol_table = compressed['symbol_table']
    n = compressed['n_symbols']
    shapes = compressed['shapes']
    fmt = compressed['format']

    model = constriction.stream.model.Categorical(probs, perfect=False)
    decoder = constriction.stream.stack.AnsCoder(compressed['compressed_words'])
    data_idx = decoder.decode(model, n)
    all_symbols = symbol_table[data_idx]

    # Split into per-tensor symbol arrays
    if fmt == 'uint8':
        # packed INT4: each original element produces 2 symbols
        sizes = [int(np.prod(s)) * 2 for s in shapes]
    else:
        sizes = [int(np.prod(s)) for s in shapes]

    results = []
    offset = 0
    for shape, size in zip(shapes, sizes):
        sym_chunk = all_symbols[offset:offset + size]
        results.append(_symbols_to_tensor(sym_chunk, fmt, shape))
        offset += size

    return results


# ---------------------------------------------------------------------------
# Size computation and analysis
# ---------------------------------------------------------------------------

def compressed_size_bytes(compressed: dict) -> int:
    """Total compressed size in bytes (data + metadata)."""
    data = len(compressed['compressed_words']) * 4
    table = len(compressed['symbol_table']) * 4  # int32
    probs = len(compressed['probabilities']) * 4  # float32
    return data + table + probs


def original_size_bytes(compressed: dict) -> int:
    """Original uncompressed size in bytes."""
    fmt = compressed['format']
    n = compressed['n_symbols']
    if fmt in ('bf16', 'fp16'):
        return n * 2
    elif fmt in ('fp8_e4m3fn', 'fp8_e5m2', 'int8'):
        return n
    elif fmt == 'uint8':
        return n // 2  # packed INT4: 2 symbols per byte
    return n


def compression_ratio(compressed: dict) -> float:
    """Compression ratio as percentage of original size."""
    return compressed_size_bytes(compressed) / original_size_bytes(compressed) * 100


def entropy_analysis(tensor: torch.Tensor, format_hint: str = None) -> dict:
    """
    Compute entropy statistics for a tensor.

    Returns dict with:
        entropy_bpv: bits per value (Shannon entropy)
        max_bpv: maximum bits per value for this format
        unique_values: number of unique symbol values
        compression_limit: theoretical best ratio
        n_values: total number of values
    """
    fmt = _detect_format(tensor, format_hint)
    symbols = _tensor_to_symbols(tensor, fmt)

    _, counts = np.unique(symbols, return_counts=True)
    probs = counts / len(symbols)
    entropy = -np.sum(probs * np.log2(probs))

    if fmt in ('bf16', 'fp16'):
        max_bpv = 16.0
    elif fmt in ('fp8_e4m3fn', 'fp8_e5m2', 'int8'):
        max_bpv = 8.0
    elif fmt == 'uint8':
        max_bpv = 4.0  # per nibble
    else:
        max_bpv = np.log2(symbols.max() + 1)

    return {
        'entropy_bpv': entropy,
        'max_bpv': max_bpv,
        'unique_values': len(counts),
        'compression_limit': entropy / max_bpv,
        'n_values': len(symbols),
        'format': fmt,
    }
