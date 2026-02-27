"""
Core codec for Predictive Coding + Cross-Layer Delta Compression.

Pipeline (compression):
    bf16 weights → int16 view → cross-layer delta → predictive coding
    → byte split → Huffman on high byte → store

Pipeline (decompression, exact inverse):
    load → Huffman decode high byte → byte merge → inverse prediction
    → inverse delta → int16 → bf16 view

All arithmetic uses int16 wrapping (two's complement) to guarantee losslessness.
"""

import numpy as np
import torch
from dahuffman import HuffmanCodec
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 1. Cross-layer delta coding
# ---------------------------------------------------------------------------

def cross_layer_delta_encode(weight_list: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Encode a list of same-shape int16 tensors as layer-to-layer deltas.

    weight_list[0] is stored as-is; subsequent entries become
    delta[l] = weight[l] - weight[l-1]  (int16 wrapping).
    """
    deltas = [weight_list[0].clone()]
    for i in range(1, len(weight_list)):
        deltas.append(weight_list[i] - weight_list[i - 1])  # int16 wrapping
    return deltas


def cross_layer_delta_decode(delta_list: list[torch.Tensor]) -> list[torch.Tensor]:
    """Inverse of cross_layer_delta_encode (cumulative sum)."""
    weights = [delta_list[0].clone()]
    for i in range(1, len(delta_list)):
        weights.append(weights[i - 1] + delta_list[i])  # int16 wrapping
    return weights


# ---------------------------------------------------------------------------
# 2. Predictive coding (intra-matrix spatial prediction)
# ---------------------------------------------------------------------------

def predictive_encode(matrix_int16: torch.Tensor) -> torch.Tensor:
    """
    Encode a 2-D int16 matrix using a left-predictor.

    residual[:, 0]  = matrix[:, 0]           (no prediction for first column)
    residual[:, j]  = matrix[:, j] - matrix[:, j-1]   for j > 0

    All arithmetic wraps in int16.
    """
    residual = matrix_int16.clone()
    if matrix_int16.dim() == 2 and matrix_int16.shape[1] > 1:
        residual[:, 1:] = matrix_int16[:, 1:] - matrix_int16[:, :-1]
    elif matrix_int16.dim() == 1 and matrix_int16.shape[0] > 1:
        residual[1:] = matrix_int16[1:] - matrix_int16[:-1]
    return residual


def predictive_decode(residual_int16: torch.Tensor) -> torch.Tensor:
    """
    Inverse of predictive_encode (cumulative sum along last dim).
    """
    if residual_int16.dim() == 2:
        return residual_int16.cumsum(dim=1).to(torch.int16)
    elif residual_int16.dim() == 1:
        return residual_int16.cumsum(dim=0).to(torch.int16)
    return residual_int16.clone()


# ---------------------------------------------------------------------------
# 3. Byte-level Huffman coding
# ---------------------------------------------------------------------------

def huffman_encode_bytes(data_uint8: np.ndarray):
    """
    Huffman-encode a uint8 array.

    Returns
    -------
    encoded_bytes : bytes
        Compressed bitstream (dahuffman format, self-delimiting).
    freq_table : dict[int, int]
        Symbol → frequency mapping (needed for decoding).
    """
    # Build frequency table
    vals, counts = np.unique(data_uint8, return_counts=True)
    freq_table = {int(v): int(c) for v, c in zip(vals, counts)}

    codec = HuffmanCodec.from_frequencies(freq_table)
    encoded_bytes = codec.encode(data_uint8.tolist())
    return encoded_bytes, freq_table


def huffman_decode_bytes(encoded_bytes: bytes, freq_table: dict, n_elements: int) -> np.ndarray:
    """
    Decode Huffman-encoded bytes back to uint8 array.
    """
    codec = HuffmanCodec.from_frequencies(freq_table)
    decoded = codec.decode(encoded_bytes)
    return np.array(decoded[:n_elements], dtype=np.uint8)


# ---------------------------------------------------------------------------
# 4. Byte splitting / merging for int16
# ---------------------------------------------------------------------------

def split_int16_bytes(data_int16: torch.Tensor):
    """
    Split int16 tensor into high byte and low byte (uint8 each).

    For value v (viewed as uint16):
        high_byte = v >> 8
        low_byte  = v & 0xFF
    """
    data_uint8 = data_int16.contiguous().view(torch.uint8)
    # torch stores int16 in little-endian: [low_byte, high_byte]
    low_bytes = data_uint8[0::2].numpy()
    high_bytes = data_uint8[1::2].numpy()
    return high_bytes, low_bytes


def merge_int16_bytes(high_bytes: np.ndarray, low_bytes: np.ndarray) -> torch.Tensor:
    """
    Inverse of split_int16_bytes. Reconstruct int16 from high + low bytes.
    """
    n = len(high_bytes)
    interleaved = np.empty(n * 2, dtype=np.uint8)
    interleaved[0::2] = low_bytes   # little-endian: low first
    interleaved[1::2] = high_bytes
    return torch.from_numpy(interleaved.copy()).view(torch.int16)


# ---------------------------------------------------------------------------
# 5. High-level compress / decompress for a weight group
# ---------------------------------------------------------------------------

def compress_weight_group(
    bf16_tensors: list[torch.Tensor],
    use_cross_layer: bool = True,
    use_prediction: bool = True,
) -> dict:
    """
    Compress a group of bf16 weight tensors (same weight type across layers).

    Parameters
    ----------
    bf16_tensors : list[Tensor]
        List of bfloat16 weight tensors, one per layer.  Can be 2-D (matrix)
        or 1-D (bias / embedding row).
    use_cross_layer : bool
        Apply cross-layer delta encoding before prediction.
    use_prediction : bool
        Apply intra-matrix predictive coding.

    Returns
    -------
    dict with keys:
        'encoded_high'   : bytes   — Huffman-encoded high bytes
        'high_freq'      : dict    — frequency table for high-byte Huffman
        'low_bytes'      : bytes   — raw low bytes
        'shapes'         : list    — original tensor shapes
        'n_elements'     : int     — total element count
        'use_cross_layer': bool
        'use_prediction' : bool
    """
    shapes = [t.shape for t in bf16_tensors]

    # Convert bf16 → int16
    int16_tensors = [t.contiguous().view(torch.int16).flatten() for t in bf16_tensors]

    # Step 1: Cross-layer delta
    if use_cross_layer and len(int16_tensors) > 1:
        int16_tensors = cross_layer_delta_encode(int16_tensors)

    # Step 2: Predictive coding (reshape to 2-D for spatial prediction)
    processed = []
    for tensor, shape in zip(int16_tensors, shapes):
        mat = tensor.view(shape) if len(shape) == 2 else tensor
        if use_prediction:
            mat = predictive_encode(mat)
        processed.append(mat.flatten())

    # Concatenate all residuals
    all_residuals = torch.cat(processed)
    n_elements = all_residuals.numel()

    # Step 3: Byte split
    high_bytes, low_bytes = split_int16_bytes(all_residuals)

    # Step 4: Huffman encode high byte
    encoded_high, high_freq = huffman_encode_bytes(high_bytes)

    return {
        'encoded_high': encoded_high,
        'high_freq': high_freq,
        'low_bytes': low_bytes.tobytes(),
        'shapes': shapes,
        'n_elements': n_elements,
        'use_cross_layer': use_cross_layer,
        'use_prediction': use_prediction,
    }


def decompress_weight_group(compressed: dict) -> list[torch.Tensor]:
    """
    Decompress a weight group back to list of bfloat16 tensors.
    Exact inverse of compress_weight_group.
    """
    shapes = compressed['shapes']
    n_elements = compressed['n_elements']
    use_cross_layer = compressed['use_cross_layer']
    use_prediction = compressed['use_prediction']

    # Step 4 inverse: Huffman decode high byte
    high_bytes = huffman_decode_bytes(
        compressed['encoded_high'], compressed['high_freq'], n_elements,
    )

    # Step 3 inverse: Byte merge
    low_bytes = np.frombuffer(compressed['low_bytes'], dtype=np.uint8)
    all_residuals = merge_int16_bytes(high_bytes, low_bytes)

    # Split back into per-layer tensors
    sizes = [int(np.prod(s)) for s in shapes]
    tensors = list(torch.split(all_residuals, sizes))

    # Step 2 inverse: Inverse prediction
    for i, (tensor, shape) in enumerate(zip(tensors, shapes)):
        mat = tensor.view(shape) if len(shape) == 2 else tensor
        if use_prediction:
            mat = predictive_decode(mat)
        tensors[i] = mat.flatten()

    # Step 1 inverse: Inverse cross-layer delta
    if use_cross_layer and len(tensors) > 1:
        tensors = cross_layer_delta_decode(tensors)

    # Convert int16 → bf16
    bf16_tensors = []
    for tensor, shape in zip(tensors, shapes):
        bf16 = tensor.view(torch.bfloat16).view(shape)
        bf16_tensors.append(bf16)

    return bf16_tensors


# ---------------------------------------------------------------------------
# 6. Compression ratio helpers
# ---------------------------------------------------------------------------

def compute_ratio(compressed: dict, original_bytes: int) -> float:
    """Compute compressed size as percentage of original."""
    compressed_bytes = (
        len(compressed['encoded_high'])
        + len(compressed['low_bytes'])
    )
    return compressed_bytes / original_bytes * 100


def compute_dfloat11_baseline_ratio(bf16_tensors: list[torch.Tensor]) -> float:
    """
    Estimate DFloat11 compression ratio for comparison.
    DFloat11 stores: Huffman(exponent 8-bit) + raw(sign_mantissa 8-bit).
    The sign_mantissa is always 1 byte per element, so ratio >= 50%.
    """
    all_weights = torch.cat([t.contiguous().flatten() for t in bf16_tensors])
    W = all_weights.view(torch.int16)
    exponent_8bits = ((W >> 7) & 0xFF).to(torch.uint8).numpy()

    # Estimate Huffman-compressed exponent size
    encoded_exp, _ = huffman_encode_bytes(exponent_8bits)

    original_bytes = all_weights.numel() * 2  # bf16 = 2 bytes
    dfloat11_bytes = len(encoded_exp) + all_weights.numel()  # huffman_exp + raw_sign_mantissa
    return dfloat11_bytes / original_bytes * 100
