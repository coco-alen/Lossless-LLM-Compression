"""
16-bit Huffman Codec for BFloat16 weight compression.

Key insight: BFloat16 LLM weights use only ~5,000-6,500 unique values
out of 65,536 possible. Direct 16-bit Huffman coding achieves ~10.59 bpw
vs DFloat11's ~10.66 bpw — a ~0.43% compression ratio improvement.

This approach avoids DFloat11's byte-separation overhead. Instead of
Huffman(exponent) + raw(sign_mantissa), we Huffman-encode the full 16-bit
value directly, capturing correlations between exponent and mantissa
that byte-separated coding misses.

Pipeline:
    bf16 weights → int16 view → build frequency table → Huffman encode

Decompression:
    Huffman decode → int16 → bf16 view

All operations are lossless (exact bit-level roundtrip).
"""

import numpy as np
import torch
from dahuffman import HuffmanCodec


# ---------------------------------------------------------------------------
# Core codec: 16-bit Huffman
# ---------------------------------------------------------------------------

def build_freq_table(int16_data: np.ndarray) -> dict:
    """Build frequency table from int16 data."""
    vals, counts = np.unique(int16_data, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def compress_16bit(bf16_tensor: torch.Tensor) -> dict:
    """
    Compress a single BFloat16 tensor using 16-bit Huffman coding.

    Parameters
    ----------
    bf16_tensor : torch.Tensor (bfloat16)

    Returns
    -------
    dict with:
        'encoded'    : bytes  — Huffman-encoded data
        'freq_table' : dict   — symbol frequency table (for decoder)
        'shape'      : tuple  — original tensor shape
        'n_elements' : int    — number of elements
    """
    assert bf16_tensor.dtype == torch.bfloat16, f"Expected bfloat16, got {bf16_tensor.dtype}"

    shape = bf16_tensor.shape
    W = bf16_tensor.contiguous().view(torch.int16).flatten().numpy()
    n = len(W)

    freq = build_freq_table(W)
    codec = HuffmanCodec.from_frequencies(freq)
    encoded = codec.encode(W.tolist())

    return {
        'encoded': encoded,
        'freq_table': freq,
        'shape': shape,
        'n_elements': n,
    }


def decompress_16bit(compressed: dict) -> torch.Tensor:
    """
    Decompress back to BFloat16 tensor.
    Exact inverse of compress_16bit.
    """
    freq = compressed['freq_table']
    codec = HuffmanCodec.from_frequencies(freq)
    decoded = codec.decode(compressed['encoded'])

    W = np.array(decoded[:compressed['n_elements']], dtype=np.int16)
    return torch.from_numpy(W.copy()).view(torch.bfloat16).view(compressed['shape'])


# ---------------------------------------------------------------------------
# Weight group compression (multiple layers of same weight type)
# ---------------------------------------------------------------------------

def compress_weight_group_16bit(
    bf16_tensors: list[torch.Tensor],
    per_layer: bool = False,
) -> dict:
    """
    Compress a group of bf16 tensors using 16-bit Huffman.

    Parameters
    ----------
    bf16_tensors : list[Tensor]
        BFloat16 weight tensors (e.g., q_proj across all layers).
    per_layer : bool
        If True, build separate Huffman tables per layer.
        If False, use a single shared table for all layers (better compression
        for large models, slightly worse for small ones).

    Returns
    -------
    dict with compressed data
    """
    shapes = [t.shape for t in bf16_tensors]

    if per_layer:
        # Per-layer: each layer has its own Huffman table
        layer_data = []
        for t in bf16_tensors:
            layer_data.append(compress_16bit(t))
        return {
            'mode': 'per_layer',
            'layer_data': layer_data,
            'shapes': shapes,
        }
    else:
        # Shared table: concatenate all weights, build one table
        all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in bf16_tensors])
        W = all_w.numpy()
        n = len(W)

        freq = build_freq_table(W)
        codec = HuffmanCodec.from_frequencies(freq)
        encoded = codec.encode(W.tolist())

        return {
            'mode': 'shared',
            'encoded': encoded,
            'freq_table': freq,
            'shapes': shapes,
            'n_elements': n,
        }


def decompress_weight_group_16bit(compressed: dict) -> list[torch.Tensor]:
    """
    Decompress a weight group back to list of BFloat16 tensors.
    """
    shapes = compressed['shapes']

    if compressed['mode'] == 'per_layer':
        return [decompress_16bit(ld) for ld in compressed['layer_data']]
    else:
        freq = compressed['freq_table']
        codec = HuffmanCodec.from_frequencies(freq)
        decoded = codec.decode(compressed['encoded'])

        W = np.array(decoded[:compressed['n_elements']], dtype=np.int16)
        all_tensors = torch.from_numpy(W.copy()).view(torch.bfloat16)

        # Split into per-layer tensors
        sizes = [int(np.prod(s)) for s in shapes]
        tensors = list(torch.split(all_tensors, sizes))
        return [t.view(s) for t, s in zip(tensors, shapes)]


# ---------------------------------------------------------------------------
# Compression ratio helpers
# ---------------------------------------------------------------------------

def compute_compressed_size(compressed: dict) -> int:
    """Total compressed size in bytes (data + table overhead)."""
    if compressed['mode'] == 'per_layer':
        total = 0
        for ld in compressed['layer_data']:
            total += len(ld['encoded'])
            total += len(ld['freq_table']) * 4  # table overhead
        return total
    else:
        data_bytes = len(compressed['encoded'])
        table_bytes = len(compressed['freq_table']) * 4  # 2 bytes value + 2 bytes code
        return data_bytes + table_bytes


def compute_ratio(compressed: dict) -> float:
    """Compression ratio as percentage of original."""
    if compressed['mode'] == 'per_layer':
        original = sum(ld['n_elements'] * 2 for ld in compressed['layer_data'])
    else:
        original = compressed['n_elements'] * 2
    return compute_compressed_size(compressed) / original * 100


def compute_dfloat11_baseline(bf16_tensors: list[torch.Tensor]) -> float:
    """Estimate DFloat11 compression ratio for comparison."""
    all_w = torch.cat([t.contiguous().flatten() for t in bf16_tensors])
    W = all_w.view(torch.int16)
    n = W.numel()

    exp = ((W >> 7) & 0xFF).to(torch.uint8).numpy()
    vals, counts = np.unique(exp, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(vals, counts)}

    codec = HuffmanCodec.from_frequencies(freq)
    encoded = codec.encode(exp.tolist())

    df11_bytes = len(encoded) + n  # huffman(exp) + raw(sign_mantissa)
    return df11_bytes / (n * 2) * 100
