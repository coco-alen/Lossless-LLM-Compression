"""
FP8 Fused Huffman Decode + GEMM Kernel

Key insight: FP8 has only 256 possible byte values (~117 unique in practice).
A single-level 256-entry Huffman LUT fits in shared memory.
Average code length ~5.65 bits → 29.6% compression → significant HBM savings.

Architecture:
  1. Offline: Huffman-encode FP8 weights (CPU, ~5.65 bpw avg)
  2. Online: GPU kernel loads compressed bitstream, decodes via LUT in shared memory,
     feeds decoded FP8 bytes to matmul

For the fused kernel, we use Triton for the initial prototype:
  - Load compressed tiles of B matrix
  - Decode in shared memory using the 256-entry LUT
  - Perform standard FP8 matmul

Standalone decode kernel is built first (simpler), then fused version.
"""

import torch
import numpy as np
import time
from collections import Counter
from dahuffman import HuffmanCodec
from copy import copy


# ============================================================================
# Part 1: FP8 Huffman Encoder (CPU, offline)
# ============================================================================

class FP8HuffmanEncoder:
    """Encode FP8 tensors with Huffman coding optimized for GPU decode."""

    def __init__(self):
        self.codec = None
        self.code_table = None
        self.decode_lut = None  # 256-entry for single-level decode

    def build_codec(self, fp8_tensor: torch.Tensor):
        """Build Huffman codec from FP8 weight distribution."""
        raw = fp8_tensor.view(torch.uint8).flatten().numpy()
        counts = Counter(raw.tolist())
        self.codec = HuffmanCodec.from_frequencies(counts)
        self.code_table = self.codec.get_code_table()

        # Ensure max code length <= 24 for efficient GPU decode
        max_len = max(l for k, (l, _) in self.code_table.items() if isinstance(k, int))
        if max_len > 24:
            freq = np.array(list(counts.values()))
            min_k = min(2, len(freq) - 1)
            while max_len > 24 and min_k < len(freq):
                min_indices = np.argpartition(freq, min_k)[:min_k]
                min_k += 1
                comp_counts = dict(counts)
                for idx in min_indices:
                    key = list(counts.keys())[idx]
                    comp_counts[key] = max(comp_counts[key], freq.max() // 100)
                self.codec = HuffmanCodec.from_frequencies(comp_counts)
                self.code_table = self.codec.get_code_table()
                max_len = max(l for k, (l, _) in self.code_table.items() if isinstance(k, int))

        # Build hierarchical LUTs for GPU decode (DFloat11 style)
        self._build_luts()

        # Compute stats
        total = sum(counts.values())
        avg_bits = sum(counts[k] * self.code_table[k][0]
                       for k in counts if k in self.code_table) / total
        return {
            'n_unique': len(counts),
            'max_code_len': max_len,
            'avg_bits': avg_bits,
            'ratio': avg_bits / 8 * 100,
        }

    def _build_luts(self):
        """Build hierarchical LUTs for GPU Huffman decoding.

        For FP8 with ~117 unique values, we typically need 1-2 LUT levels.
        Level 0: 256 entries indexed by first 8 bits of code
        Level 1+: for codes longer than 8 bits
        """
        table = self.code_table
        prefixes = ['']
        for key, (bits, val) in table.items():
            if isinstance(key, int):
                prefix = bin(val)[2:].rjust(bits, "0")[:((bits - 1) // 8 * 8)]
                if prefix not in prefixes:
                    prefixes.append(prefix)
        prefixes.sort(key=len)

        n_tables = len(prefixes)
        luts = np.zeros((n_tables, 256), dtype=np.uint8)

        for pi, p in enumerate(prefixes):
            bytes_dict = {}
            pl = len(p) // 8
            for key, (bits, val) in table.items():
                if isinstance(key, int):
                    bin_val = bin(val)[2:].rjust(bits, '0')
                    if bin_val.startswith(p):
                        if (bits - 1) // 8 == pl:
                            dict_key = int(bin_val[(pl * 8):].ljust(8, '0'), 2)
                            dict_value = key  # actual FP8 byte value
                        else:
                            dict_key = int(bin_val[(pl * 8):(pl * 8 + 8)], 2)
                            dict_value = 256 - prefixes.index(bin_val[:(pl * 8 + 8)])
                        if dict_key not in bytes_dict:
                            bytes_dict[dict_key] = dict_value

            curr_val = 0
            for i in range(256):
                if i in bytes_dict:
                    curr_val = bytes_dict[i]
                luts[pi, i] = curr_val

        # Code length table (last row)
        lens = np.zeros((1, 256), dtype=np.uint8)
        for key, (bits, val) in table.items():
            if isinstance(key, int):
                lens[0, key] = bits

        self.luts = np.concatenate((luts, lens), axis=0)
        self.n_lut_levels = n_tables

    def encode(self, fp8_tensor: torch.Tensor, bytes_per_thread=8, threads_per_block=256):
        """Encode FP8 tensor into Huffman bitstream with GPU decode metadata."""
        raw = fp8_tensor.view(torch.uint8).flatten().numpy()
        n = len(raw)

        encoded = []
        gaps = []
        output_positions = []

        buffer = 0
        size = 0
        total_size = 0
        element_count = 0

        for s in raw:
            s = int(s)
            if total_size // (8 * bytes_per_thread) + 1 > len(gaps):
                gaps.append(total_size - total_size // (8 * bytes_per_thread) * (8 * bytes_per_thread))
            if total_size // (8 * bytes_per_thread * threads_per_block) + 1 > len(output_positions):
                output_positions.append(element_count)

            b, v = self.codec._table[s]
            buffer = (buffer << b) + v
            size += b
            total_size += b
            element_count += 1
            while size >= 8:
                byte = buffer >> (size - 8)
                encoded.append(byte)
                buffer = buffer - (byte << (size - 8))
                size -= 8

        # Trailing bits
        if size > 0:
            if total_size // (8 * bytes_per_thread) + 1 > len(gaps):
                gaps.append(total_size - total_size // (8 * bytes_per_thread) * (8 * bytes_per_thread))
            if total_size // (8 * bytes_per_thread * threads_per_block) + 1 > len(output_positions):
                output_positions.append(element_count)
            b, v = self.codec._table[self.codec._eof]
            buffer = (buffer << b) + v
            size += b
            byte = buffer >> (size - 8) if size >= 8 else buffer << (8 - size)
            encoded.append(byte)

        output_positions.append(n)

        blocks = int(np.ceil(len(encoded) / (threads_per_block * bytes_per_thread)))
        total_threads = threads_per_block * blocks
        gaps.extend([0] * (total_threads - len(gaps)))
        binary_gaps = [int(bit) for gap in gaps for bit in format(gap, '05b')]

        encoded_bytes = np.frombuffer(bytes(encoded), dtype=np.uint8)
        packed_gaps = np.packbits(binary_gaps)
        positions = np.array(output_positions, dtype=np.uint32)

        # Sizes
        original_bytes = n
        compressed_bytes = len(encoded_bytes) + len(packed_gaps) + len(positions) * 4 + self.luts.size

        return {
            'encoded_bytes': encoded_bytes,
            'packed_gaps': packed_gaps,
            'output_positions': positions,
            'luts': self.luts,
            'n_lut_levels': self.n_lut_levels,
            'n_elements': n,
            'shape': fp8_tensor.shape,
            'original_bytes': original_bytes,
            'compressed_bytes': compressed_bytes,
            'ratio': compressed_bytes / original_bytes * 100,
            'bytes_per_thread': bytes_per_thread,
            'threads_per_block': threads_per_block,
        }


# ============================================================================
# Part 2: FP8 GPU Huffman Decode Kernel (CuPy)
# ============================================================================

import cupy as cp

# Reuse DFloat11's proven kernel architecture but for FP8:
# - No sign_mantissa stream (we decode full FP8 bytes directly)
# - Output is uint8 (FP8 bytes) not uint16 (BF16 values)
# - LUT values are actual FP8 byte values (0-255), not symbol indices

FP8_DECODE_KERNEL = r"""
extern "C"
__global__ void fp8_huffman_decode(
    const unsigned char * __restrict__  luts,
    const unsigned char * __restrict__  codes,
    const unsigned int  * __restrict__  position_offsets,
    const unsigned char * __restrict__  gaps,
    unsigned char       * __restrict__  outputs,
    const int n_luts,
    const int n_bytes,
    const int n_elements
) {
    // FP8 Huffman decoder: decodes compressed bitstream to FP8 byte values.
    // Uses DFloat11's proven two-phase approach (count + prefix-sum + decode+write).
    // LUT entries are actual FP8 byte values (0-255).
    // Values >= 240 in LUT indicate "follow pointer to next level".

    unsigned char register_buffer[12];
    extern __shared__ volatile unsigned char shared_mem[];
    volatile unsigned int* accumulators = (volatile unsigned int*) shared_mem;
    // write_buffer is uint8 for FP8 (not uint16 like DFloat11)
    volatile unsigned char* write_buffer = shared_mem + blockDim.x * 4 + 4;

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

    // Load first 8 bytes (big-endian for bitstream)
    buf[0]=register_buffer[7]; buf[1]=register_buffer[6];
    buf[2]=register_buffer[5]; buf[3]=register_buffer[4];
    buf[4]=register_buffer[3]; buf[5]=register_buffer[2];
    buf[6]=register_buffer[1]; buf[7]=register_buffer[0];

    long_buf <<= gap;
    unsigned char free_bits = gap;
    unsigned char decoded;

    // Hierarchical LUT decode macro
    #define DECODE_FP8() \
        decoded = __ldg(&luts[long_buf >> 56]); \
        if (decoded >= 240) { \
            decoded = __ldg(&luts[256 * (256 - decoded) + ((long_buf >> 48) & 0xff)]); \
            if (decoded >= 240) { \
                decoded = __ldg(&luts[256 * (256 - decoded) + ((long_buf >> 40) & 0xff)]); \
            } \
        }

    // --- Phase 1: Count ---
    while (free_bits < 32) {
        DECODE_FP8();
        thread_counter += 1;
        unsigned char code_len = __ldg(&luts[256 * (n_luts - 1) + decoded]);
        long_buf <<= code_len;
        free_bits += code_len;
    }

    buf[8]=register_buffer[11]; buf[9]=register_buffer[10];
    buf[10]=register_buffer[9]; buf[11]=register_buffer[8];
    long_buf |= static_cast<unsigned long long>(int_buf) << (free_bits - 32);
    free_bits -= 32;

    while (4 + free_bits / 8 < BPT) {
        DECODE_FP8();
        thread_counter += 1;
        unsigned char code_len = __ldg(&luts[256 * (n_luts - 1) + decoded]);
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
        DECODE_FP8();
        write_buffer[output_idx - write_offset] = decoded;
        output_idx += 1;
        unsigned char code_len = __ldg(&luts[256 * (n_luts - 1) + decoded]);
        long_buf <<= code_len;
        free_bits += code_len;
    }

    buf[8]=register_buffer[11]; buf[9]=register_buffer[10];
    buf[10]=register_buffer[9]; buf[11]=register_buffer[8];
    long_buf |= static_cast<unsigned long long>(int_buf) << (free_bits - 32);
    free_bits -= 32;

    while (output_idx < end_idx) {
        DECODE_FP8();
        write_buffer[output_idx - write_offset] = decoded;
        output_idx += 1;
        unsigned char code_len = __ldg(&luts[256 * (n_luts - 1) + decoded]);
        long_buf <<= code_len;
        free_bits += code_len;
    }
    __syncthreads();

    #undef DECODE_FP8

    // Coalesced global write (uint8)
    int block_elems = min(accumulators[blockDim.x] - write_offset,
                          (unsigned int)n_elements - write_offset);
    for (int i = threadIdx.x; i < block_elems; i += blockDim.x) {
        outputs[i + write_offset] = write_buffer[i];
    }
}
"""


class FP8GPUDecoder:
    """GPU Huffman decoder for FP8 weights."""

    def __init__(self):
        self.kernel = cp.RawKernel(FP8_DECODE_KERNEL, 'fp8_huffman_decode')

    def decode(self, compressed: dict) -> torch.Tensor:
        """Decode compressed FP8 data on GPU."""
        n_elements = compressed['n_elements']
        shape = compressed['shape']
        threads = compressed['threads_per_block']

        luts_gpu = cp.asarray(compressed['luts'].flatten())
        codes_gpu = cp.asarray(compressed['encoded_bytes'])
        positions_gpu = cp.asarray(compressed['output_positions'])
        gaps_gpu = cp.asarray(compressed['packed_gaps'])
        output_gpu = cp.zeros(n_elements, dtype=cp.uint8)

        n_bytes = len(compressed['encoded_bytes'])
        n_luts = compressed['luts'].shape[0]
        blocks = len(compressed['output_positions']) - 1

        # Shared memory for accumulators + write buffer (uint8)
        max_elems = compressed['bytes_per_thread'] * 8 * threads // 5 + threads
        shared_bytes = threads * 4 + 4 + max_elems
        shared_bytes = min(shared_bytes, 48 * 1024)

        self.kernel(
            (blocks,), (threads,),
            (luts_gpu, codes_gpu, positions_gpu, gaps_gpu, output_gpu,
             n_luts, n_bytes, n_elements),
            shared_mem=shared_bytes,
        )
        cp.cuda.Stream.null.synchronize()

        # Convert cupy uint8 → torch FP8 on GPU
        result = torch.as_tensor(output_gpu, device='cuda').view(torch.float8_e4m3fn).reshape(shape)
        return result.clone()


# ============================================================================
# Part 3: Benchmark
# ============================================================================

def benchmark_fp8_huffman(model_name="Qwen/Qwen3-0.6B"):
    """Benchmark FP8 Huffman encode + GPU decode."""
    from transformers import AutoModelForCausalLM

    print(f"\n{'='*90}")
    print(f"FP8 Fused Huffman Benchmark: {model_name}")
    print(f"{'='*90}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    encoder = FP8HuffmanEncoder()
    decoder = FP8GPUDecoder()

    # Collect all FP8 values for shared codec
    all_fp8 = []
    weight_tensors = []
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16 and param.numel() > 1024:
            fp8 = param.data.to(torch.float8_e4m3fn)
            all_fp8.append(fp8.view(torch.uint8).flatten())
            weight_tensors.append((name, fp8))

    combined_fp8 = torch.cat(all_fp8)
    combined_fp8_tensor = combined_fp8.view(torch.float8_e4m3fn)

    # Build shared codec
    stats = encoder.build_codec(combined_fp8_tensor)
    print(f"\nShared codec: {stats['n_unique']} unique values, "
          f"avg {stats['avg_bits']:.2f} bpw, max code {stats['max_code_len']} bits, "
          f"ratio {stats['ratio']:.1f}%")
    print(f"LUT levels: {encoder.n_lut_levels}")

    # Test on representative layers
    print(f"\n{'Layer':<50} {'Params':>10} {'Ratio':>7} {'Enc ms':>8} {'Dec ms':>8} "
          f"{'Dec GB/s':>9} {'Lossless':>8}")

    total_orig = 0
    total_comp = 0
    total_dec_time = 0

    for name, fp8_tensor in weight_tensors[:10]:  # test first 10 layers
        if fp8_tensor.numel() < 500_000:
            continue

        # Encode
        t0 = time.perf_counter()
        compressed = encoder.encode(fp8_tensor)
        enc_time = time.perf_counter() - t0

        # Decode on GPU
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        recovered = decoder.decode(compressed)
        cp.cuda.Stream.null.synchronize()
        dec_time = time.perf_counter() - t0

        # Verify lossless
        is_ok = torch.equal(
            fp8_tensor.view(torch.uint8),
            recovered.cpu().view(torch.uint8)
        )

        ratio = compressed['ratio']
        total_orig += compressed['original_bytes']
        total_comp += compressed['compressed_bytes']

        # Benchmark decode (warm)
        times = []
        for _ in range(3):
            decoder.decode(compressed)
        for _ in range(10):
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            decoder.decode(compressed)
            cp.cuda.Stream.null.synchronize()
            times.append(time.perf_counter() - t0)
        avg_dec = np.mean(times)
        throughput = fp8_tensor.numel() / 1e9 / avg_dec  # GB/s (1 byte per FP8)

        total_dec_time += avg_dec

        print(f"  {name:<48} {fp8_tensor.numel():>10,} {ratio:>6.1f}% "
              f"{enc_time*1000:>7.0f} {avg_dec*1000:>7.2f} "
              f"{throughput:>8.1f} {'✓' if is_ok else '✗':>8}")

    overall = total_comp / total_orig * 100 if total_orig > 0 else 0
    print(f"\n  Aggregate ratio: {overall:.2f}%")
    print(f"\n  Comparison:")
    print(f"    Dense FP8:                  100.0%  (baseline)")
    print(f"    nvCOMP ANS (raw byte):       85.7%  (29-56 GB/s decode)")
    print(f"    Our FP8 Huffman (GPU):      {overall:.1f}%")
    print(f"    Entropy limit:               70.4%")


if __name__ == "__main__":
    benchmark_fp8_huffman()
