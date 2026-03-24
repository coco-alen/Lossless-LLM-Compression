"""
GPU-accelerated full-value Huffman codec for lossless LLM weight compression.

Architecture based on DFloat11's proven CUDA kernel, but extended to code
the FULL 16-bit BF16 value (or 8-bit FP8) as a single Huffman symbol.

Key differences from DFloat11:
  - DFloat11: Huffman(8-bit exponent) + raw(8-bit sign+mantissa) = ~66.6%
  - Ours:     Huffman(full 16-bit value)                         = ~66.2%
  - For FP8:  Huffman(full 8-bit value)                          = ~70.4%

The decoder CUDA kernel uses DFloat11's two-phase approach:
  Phase 1: Each thread counts decoded symbols (no writes)
  Phase 2: Prefix-sum for positions, then decode + write

Dependencies: cupy-cuda12x, dahuffman==0.4.2, torch
"""

import numpy as np
import torch
import cupy as cp
from dahuffman import HuffmanCodec
from copy import copy
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BYTES_PER_THREAD = 8
THREADS_PER_BLOCK = 512

# ---------------------------------------------------------------------------
# CUDA Kernel Source
# ---------------------------------------------------------------------------

DECODE_KERNEL_SOURCE = r"""
#define SENTINEL 0xF000u

extern "C"
__global__ void decode_fullvalue(
    const unsigned short * __restrict__  luts,      // uint16 LUT entries
    const unsigned char  * __restrict__  codes,
    const unsigned int   * __restrict__  position_offsets,
    const unsigned char  * __restrict__  gaps,
    const unsigned short * __restrict__  lens_table, // code lengths per symbol index
    unsigned short       * __restrict__  outputs,
    const int n_lut_rows,
    const int n_bytes,
    const int n_elements
) {
    // Full-value Huffman decoder with uint16 LUTs for large alphabets.
    // Decoded values are symbol indices [0, N_symbols).
    // Sentinel values >= SENTINEL indicate "follow pointer to next LUT level".

    unsigned char register_buffer[12];
    extern __shared__ volatile unsigned char shared_mem[];
    volatile unsigned int* accumulators = (volatile unsigned int*) shared_mem;
    volatile unsigned short* write_buffer = (volatile unsigned short*) (shared_mem + blockDim.x * 4 + 4);

    const int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int BPT = 8;

    // Load encoded bytes
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        int idx = global_thread_id * BPT + i;
        register_buffer[i] = (idx < n_bytes) ? codes[idx] : 0;
    }
    __syncthreads();

    // Parse gap
    unsigned char buf[12];
    unsigned long long &long_buf = *reinterpret_cast<unsigned long long*>(buf);
    unsigned int &int_buf = *reinterpret_cast<unsigned int*>(buf + 8);
    unsigned short &short_buf = *reinterpret_cast<unsigned short*>(buf + 8);

    buf[8] = gaps[global_thread_id * 5 / 8 + 1];
    buf[9] = gaps[global_thread_id * 5 / 8];
    const unsigned char gap = (short_buf >> (11 - (global_thread_id * 5 % 8))) & 0x1f;

    unsigned int thread_counter = 0;

    // Load first 8 bytes (big-endian)
    buf[0]=register_buffer[7]; buf[1]=register_buffer[6];
    buf[2]=register_buffer[5]; buf[3]=register_buffer[4];
    buf[4]=register_buffer[3]; buf[5]=register_buffer[2];
    buf[6]=register_buffer[1]; buf[7]=register_buffer[0];

    long_buf <<= gap;
    unsigned char free_bits = gap;
    unsigned short decoded;

    // --- Phase 1: Count ---
    #define DECODE_SYMBOL() \
        decoded = luts[(long_buf >> 56)]; \
        if (decoded >= SENTINEL) { \
            decoded = luts[256 * (decoded - SENTINEL) + ((long_buf >> 48) & 0xff)]; \
            if (decoded >= SENTINEL) { \
                decoded = luts[256 * (decoded - SENTINEL) + ((long_buf >> 40) & 0xff)]; \
                if (decoded >= SENTINEL) { \
                    decoded = luts[256 * (decoded - SENTINEL) + ((long_buf >> 32) & 0xff)]; \
                } \
            } \
        }

    while (free_bits < 32) {
        DECODE_SYMBOL();
        thread_counter += 1;
        unsigned char code_len = (unsigned char)lens_table[decoded];
        long_buf <<= code_len;
        free_bits += code_len;
    }

    buf[8]=register_buffer[11]; buf[9]=register_buffer[10];
    buf[10]=register_buffer[9]; buf[11]=register_buffer[8];
    long_buf |= static_cast<unsigned long long>(int_buf) << (free_bits - 32);
    free_bits -= 32;

    while (4 + free_bits / 8 < BPT) {
        DECODE_SYMBOL();
        thread_counter += 1;
        unsigned char code_len = (unsigned char)lens_table[decoded];
        long_buf <<= code_len;
        free_bits += code_len;
    }

    // --- Prefix sum ---
    if (threadIdx.x == 0) {
        accumulators[0] = position_offsets[blockIdx.x] + thread_counter;
    } else {
        accumulators[threadIdx.x] = thread_counter;
    }
    __syncthreads();
    for (int i = 2; i <= (int)blockDim.x; i <<= 1) {
        if (((threadIdx.x + 1) & (i - 1)) == 0)
            accumulators[threadIdx.x] += accumulators[threadIdx.x - (i >> 1)];
        __syncthreads();
    }
    if (threadIdx.x == 0) accumulators[blockDim.x - 1] = 0;
    __syncthreads();
    for (int i = blockDim.x; i >= 2; i >>= 1) {
        if (((threadIdx.x + 1) & (i - 1)) == 0) {
            accumulators[threadIdx.x] += accumulators[threadIdx.x - (i >> 1)];
            accumulators[threadIdx.x - (i >> 1)] =
                accumulators[threadIdx.x] - accumulators[threadIdx.x - (i >> 1)];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        accumulators[0] = position_offsets[blockIdx.x];
        accumulators[blockDim.x] = position_offsets[blockIdx.x + 1];
    }
    __syncthreads();

    unsigned int output_idx = accumulators[threadIdx.x];
    const unsigned int write_offset = accumulators[0];
    const unsigned int end_idx = min(output_idx + thread_counter, (unsigned int)n_elements);

    // --- Phase 2: Decode + write ---
    buf[0]=register_buffer[7]; buf[1]=register_buffer[6];
    buf[2]=register_buffer[5]; buf[3]=register_buffer[4];
    buf[4]=register_buffer[3]; buf[5]=register_buffer[2];
    buf[6]=register_buffer[1]; buf[7]=register_buffer[0];
    long_buf <<= gap;
    free_bits = gap;

    while (free_bits < 32 && output_idx < end_idx) {
        DECODE_SYMBOL();
        write_buffer[output_idx - write_offset] = decoded;
        output_idx += 1;
        unsigned char code_len = (unsigned char)lens_table[decoded];
        long_buf <<= code_len;
        free_bits += code_len;
    }

    buf[8]=register_buffer[11]; buf[9]=register_buffer[10];
    buf[10]=register_buffer[9]; buf[11]=register_buffer[8];
    long_buf |= static_cast<unsigned long long>(int_buf) << (free_bits - 32);
    free_bits -= 32;

    while (output_idx < end_idx) {
        DECODE_SYMBOL();
        write_buffer[output_idx - write_offset] = decoded;
        output_idx += 1;
        unsigned char code_len = (unsigned char)lens_table[decoded];
        long_buf <<= code_len;
        free_bits += code_len;
    }
    __syncthreads();

    #undef DECODE_SYMBOL

    // Coalesced global write
    int block_elems = min(accumulators[blockDim.x] - write_offset,
                          (unsigned int)n_elements - write_offset);
    for (int i = threadIdx.x; i < block_elems; i += blockDim.x) {
        outputs[i + write_offset] = write_buffer[i];
    }
}
"""


# ---------------------------------------------------------------------------
# Encoder: Full-value Huffman with DFloat11-compatible LUT generation
# ---------------------------------------------------------------------------

class FullValueHuffmanEncoder:
    """Huffman encoder for full 16-bit BF16 (or 8-bit FP8) values.

    Generates:
    - Huffman bitstream (encoded data)
    - Hierarchical LUTs for GPU decoding
    - Position offsets and gaps for parallel thread coordination
    """

    def __init__(self, bytes_per_thread=BYTES_PER_THREAD, threads_per_block=THREADS_PER_BLOCK):
        self.bytes_per_thread = bytes_per_thread
        self.threads_per_block = threads_per_block

    def _build_codec(self, values: np.ndarray) -> tuple:
        """Build Huffman codec from value frequencies.

        For full 16-bit values (~6000 unique), we need to ensure max code length <= 32.
        If longer, we boost rare symbol frequencies (like DFloat11's get_32bit_codec).
        """
        unique_vals, counts = np.unique(values, return_counts=True)
        counter = {int(v): int(c) for v, c in zip(unique_vals, counts)}

        codec = HuffmanCodec.from_frequencies(counter)
        table = codec.get_code_table()

        max_len = max(l for _, (l, _) in table.items() if isinstance(_, int))

        # If max code length > 32, boost rare symbols
        if max_len > 32:
            freq_arr = np.array(list(counter.values()))
            min_k = 2
            while max_len > 32:
                min_indices = np.argpartition(freq_arr, min_k)[:min_k]
                min_k += 1
                min_keys = np.array(list(counter.keys()))[min_indices]
                compressed_counter = copy(counter)
                for k in min_keys:
                    compressed_counter[k] = max(compressed_counter[k], 1)
                codec = HuffmanCodec.from_frequencies(compressed_counter)
                table = codec.get_code_table()
                max_len = max(l for _, (l, _) in table.items() if isinstance(_, int))

        return codec, counter, table

    def _build_luts(self, table: dict) -> np.ndarray:
        """Build hierarchical LUTs for GPU decoding.

        Uses uint16 LUT entries (supports up to ~60000 symbols).
        Values >= SENTINEL (0xF000) indicate "follow pointer to next level".
        Last table row stores code lengths (uint16, but values are small).

        For full-value coding, the LUT stores SYMBOL INDICES (0..N-1),
        which map to actual 16-bit values via a separate symbol table.
        """
        SENTINEL = 0xF000  # Values >= this are pointers to next LUT level

        # Collect all prefix boundaries (multiples of 8 bits)
        prefixes = ['']
        for key, (bits, val) in table.items():
            if isinstance(key, int):
                prefix = bin(val)[2:].rjust(bits, "0")[:((bits - 1) // 8 * 8)]
                if prefix not in prefixes:
                    prefixes.append(prefix)
        prefixes.sort(key=len)

        # Build LUT tables (uint16 to support >256 symbols)
        n_tables = len(prefixes)
        luts = np.zeros((n_tables, 256), dtype=np.uint16)

        # Create symbol-to-index mapping
        symbols = sorted([k for k in table.keys() if isinstance(k, int)])
        sym_to_idx = {s: i for i, s in enumerate(symbols)}
        assert len(symbols) < SENTINEL, f"Too many symbols ({len(symbols)}) for sentinel {SENTINEL}"

        for pi, p in enumerate(prefixes):
            bytes_dict = {}
            pl = len(p) // 8
            for key, (bits, val) in table.items():
                if isinstance(key, int):
                    bin_val = bin(val)[2:].rjust(bits, '0')
                    if bin_val.startswith(p):
                        if (bits - 1) // 8 == pl:
                            dict_key = int(bin_val[(pl * 8):].ljust(8, '0'), 2)
                            dict_value = sym_to_idx[key]
                        else:
                            dict_key = int(bin_val[(pl * 8):(pl * 8 + 8)], 2)
                            # Pointer to next level: SENTINEL + index
                            dict_value = SENTINEL + prefixes.index(bin_val[:(pl * 8 + 8)])

                        if dict_key not in bytes_dict:
                            bytes_dict[dict_key] = dict_value

            curr_val = 0
            for i in range(256):
                if i in bytes_dict:
                    curr_val = bytes_dict[i]
                luts[pi, i] = curr_val

        # Add code length table (last row)
        # We need up to N_symbols entries. Pad to multiple of 256 for alignment.
        n_sym = len(symbols)
        lens_size = ((n_sym + 255) // 256) * 256
        lens = np.zeros(lens_size, dtype=np.uint16)
        for key, (bits, val) in table.items():
            if isinstance(key, int):
                idx = sym_to_idx[key]
                lens[idx] = bits

        # Reshape lens to (lens_size//256, 256) and concatenate
        lens_rows = lens.reshape(-1, 256)
        all_luts = np.concatenate((luts, lens_rows), axis=0)

        return all_luts, symbols, n_tables

    def _encode_bitstream(self, values, codec) -> tuple:
        """Encode values into bitstream with gap/position metadata."""
        encoded = []
        gaps = []
        output_positions = []

        buffer = 0
        size = 0
        total_size = 0
        element_count = 0

        for s in values:
            if total_size // (8 * self.bytes_per_thread) + 1 > len(gaps):
                gaps.append(total_size - total_size // (8 * self.bytes_per_thread) * (8 * self.bytes_per_thread))

            if total_size // (8 * self.bytes_per_thread * self.threads_per_block) + 1 > len(output_positions):
                output_positions.append(element_count)

            b, v = codec._table[s]
            buffer = (buffer << b) + v
            size += b
            total_size += b
            element_count += 1

            while size >= 8:
                byte = buffer >> (size - 8)
                encoded.append(byte)
                buffer = buffer - (byte << (size - 8))
                size -= 8

        # Handle trailing bits
        if size > 0:
            if total_size // (8 * self.bytes_per_thread) + 1 > len(gaps):
                gaps.append(total_size - total_size // (8 * self.bytes_per_thread) * (8 * self.bytes_per_thread))
            if total_size // (8 * self.bytes_per_thread * self.threads_per_block) + 1 > len(output_positions):
                output_positions.append(element_count)

            b, v = codec._table[codec._eof]
            buffer = (buffer << b) + v
            size += b
            if size >= 8:
                byte = buffer >> (size - 8)
            else:
                byte = buffer << (8 - size)
            encoded.append(byte)

        output_positions.append(len(values))

        # Pad gaps to full grid size
        blocks_per_grid = int(np.ceil(len(encoded) / (self.threads_per_block * self.bytes_per_thread)))
        total_threads = self.threads_per_block * blocks_per_grid
        gaps.extend([0] * (total_threads - len(gaps)))

        # Pack gaps as 5-bit values
        binary_str_gaps = [format(gap, '05b') for gap in gaps]
        binary_gaps = [int(bit) for binary in binary_str_gaps for bit in binary]

        return (
            np.frombuffer(bytes(encoded), dtype=np.uint8),
            np.packbits(binary_gaps),
            np.array(output_positions, dtype=np.uint32),
        )

    def encode(self, tensor: torch.Tensor) -> dict:
        """Encode a BF16 or FP8 tensor.

        Returns dict with all data needed for GPU decoding.
        """
        shape = tensor.shape
        n_elements = tensor.numel()

        if tensor.dtype == torch.bfloat16:
            values = tensor.contiguous().view(torch.int16).flatten().numpy().tolist()
            fmt = 'bf16'
        elif tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            values = tensor.contiguous().view(torch.uint8).flatten().numpy().tolist()
            fmt = 'fp8'
        else:
            raise ValueError(f"Unsupported dtype: {tensor.dtype}")

        # Build codec
        codec, counter, table = self._build_codec(np.array(values))
        luts, symbols, n_lut_rows = self._build_luts(table)
        n_luts = n_lut_rows  # number of decode LUT rows (excluding lens rows)

        # Encode bitstream
        encoded_bytes, packed_gaps, output_positions = self._encode_bitstream(values, codec)

        # Compute sizes
        original_bytes = n_elements * (2 if fmt == 'bf16' else 1)
        compressed_bytes = (
            len(encoded_bytes) +
            len(packed_gaps) +
            len(output_positions) * 4 +
            luts.size * 2 +  # uint16 LUT entries
            len(symbols) * (2 if fmt == 'bf16' else 1)
        )

        return {
            'encoded_bytes': encoded_bytes,
            'packed_gaps': packed_gaps,
            'output_positions': output_positions,
            'luts': luts,
            'symbols': np.array(symbols),
            'n_luts': n_luts,
            'shape': shape,
            'n_elements': n_elements,
            'format': fmt,
            'original_bytes': original_bytes,
            'compressed_bytes': compressed_bytes,
            'ratio': compressed_bytes / original_bytes * 100,
        }


# ---------------------------------------------------------------------------
# GPU Decoder
# ---------------------------------------------------------------------------

class GPUDecoder:
    """GPU Huffman decoder using CuPy."""

    def __init__(self):
        self.kernel = cp.RawKernel(DECODE_KERNEL_SOURCE, 'decode_fullvalue')

    def decode(self, compressed: dict) -> torch.Tensor:
        """Decode compressed data on GPU. Returns torch tensor on CUDA."""
        n_elements = compressed['n_elements']
        fmt = compressed['format']
        shape = compressed['shape']
        n_lut_rows = compressed['n_luts']  # decode LUT rows only

        luts_all = compressed['luts']  # shape (n_lut_rows + lens_rows, 256), uint16

        # Split LUTs: decode tables and length tables
        decode_luts = luts_all[:n_lut_rows].flatten().astype(np.uint16)
        lens_table = luts_all[n_lut_rows:].flatten().astype(np.uint16)

        # Transfer to GPU
        luts_gpu = cp.asarray(decode_luts)
        lens_gpu = cp.asarray(lens_table)
        codes_gpu = cp.asarray(compressed['encoded_bytes'])
        positions_gpu = cp.asarray(compressed['output_positions'])
        gaps_gpu = cp.asarray(compressed['packed_gaps'])

        # Output buffer (uint16 symbol indices)
        output_gpu = cp.zeros(n_elements, dtype=cp.uint16)

        n_bytes = len(compressed['encoded_bytes'])
        blocks = len(compressed['output_positions']) - 1

        # Shared memory: accumulators (uint32) + write buffer (uint16)
        # Max elements per block ≈ 8*8/10.6 * 512 ≈ 3000 for BF16
        max_elems_per_block = BYTES_PER_THREAD * 8 * THREADS_PER_BLOCK // 8 + THREADS_PER_BLOCK
        shared_bytes = THREADS_PER_BLOCK * 4 + 4 + max_elems_per_block * 2
        shared_bytes = min(shared_bytes, 48 * 1024)

        self.kernel(
            (blocks,), (THREADS_PER_BLOCK,),
            (luts_gpu, codes_gpu, positions_gpu, gaps_gpu, lens_gpu, output_gpu,
             n_lut_rows, n_bytes, n_elements),
            shared_mem=shared_bytes,
        )
        cp.cuda.Stream.null.synchronize()

        # Map indices to actual values via symbol table — ENTIRELY ON GPU
        symbols_gpu = cp.asarray(compressed['symbols'].astype(np.int16 if fmt == 'bf16' else np.uint8))
        # Use GPU gather: output_gpu contains indices, symbols_gpu maps them to values
        values_gpu = symbols_gpu[output_gpu.astype(cp.int32)]

        if fmt == 'bf16':
            # Convert cupy → torch on same GPU (zero-copy via DLPack)
            values_torch = torch.as_tensor(values_gpu, device='cuda').view(torch.bfloat16).reshape(shape)
            return values_torch.clone()  # clone to own memory
        else:
            dtype = torch.float8_e4m3fn if fmt == 'fp8' else torch.float8_e5m2
            values_torch = torch.as_tensor(values_gpu.astype(cp.uint8), device='cuda').view(dtype).reshape(shape)
            return values_torch.clone()


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

class FullValueGPUCodec:
    """Complete GPU codec: CPU encode + GPU decode."""

    def __init__(self):
        self.encoder = FullValueHuffmanEncoder()
        self.decoder = GPUDecoder()

    def compress(self, tensor: torch.Tensor) -> dict:
        """Compress tensor on CPU."""
        return self.encoder.encode(tensor)

    def decompress_gpu(self, compressed: dict) -> torch.Tensor:
        """Decompress on GPU."""
        return self.decoder.decode(compressed)

    def verify_lossless(self, tensor: torch.Tensor) -> bool:
        """Verify round-trip is bit-exact."""
        compressed = self.compress(tensor)
        recovered = self.decompress_gpu(compressed).cpu()
        original = tensor.cpu()
        return torch.equal(original.view(torch.int16), recovered.view(torch.int16))


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(model_name: str = "Qwen/Qwen3-0.6B"):
    """Benchmark the full-value GPU codec."""
    import time
    from transformers import AutoModelForCausalLM

    print(f"\n{'='*80}")
    print(f"Full-Value GPU Huffman Benchmark: {model_name}")
    print(f"{'='*80}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    codec = FullValueGPUCodec()

    # Test on a few representative layers
    test_layers = []
    for name, param in model.named_parameters():
        if param.numel() > 1_000_000 and 'weight' in name:
            test_layers.append((name, param))
            if len(test_layers) >= 5:
                break

    print(f"\nTesting {len(test_layers)} layers...")
    print(f"{'Layer':<50} {'Params':>10} {'Ratio':>8} {'Enc ms':>8} {'Dec ms':>8} {'Lossless':>8}")

    total_orig = 0
    total_comp = 0

    for name, param in test_layers:
        # Encode (CPU)
        t0 = time.perf_counter()
        compressed = codec.compress(param.data)
        enc_time = time.perf_counter() - t0

        # Decode (GPU)
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        recovered = codec.decompress_gpu(compressed)
        cp.cuda.Stream.null.synchronize()
        dec_time = time.perf_counter() - t0

        # Verify
        is_lossless = torch.equal(
            param.data.view(torch.int16),
            recovered.cpu().view(torch.int16)
        )

        ratio = compressed['ratio']
        total_orig += compressed['original_bytes']
        total_comp += compressed['compressed_bytes']

        print(f"  {name:<48} {param.numel():>10,} {ratio:>7.2f}% "
              f"{enc_time*1000:>7.0f} {dec_time*1000:>7.1f} {'✓' if is_lossless else '✗':>8}")

    overall = total_comp / total_orig * 100
    print(f"\n  Aggregate: {overall:.2f}%")
    print(f"\n  Comparison:")
    print(f"    DFloat11 (exponent-only Huffman): ~66.6%")
    print(f"    Our CPU ANS-16bit:                ~66.1%")
    print(f"    Our GPU full-value Huffman:        {overall:.1f}%")


if __name__ == "__main__":
    benchmark()
