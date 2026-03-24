"""Directly test the GPU kernel with known inputs."""
import torch, numpy as np, cupy as cp
import sys
sys.path.insert(0, '.')
from experiments.fused_codec.fp8_twostream import FP8TwoStreamEncoder, FP8TwoStreamDecoder, TWOSTREAM_DECODE_KERNEL

torch.manual_seed(42)
bf16 = torch.randn(32, dtype=torch.bfloat16)
fp8 = bf16.to(torch.float8_e4m3fn)
raw = fp8.view(torch.uint8).flatten().numpy()

encoder = FP8TwoStreamEncoder(k=3)
comp = encoder.encode(fp8)

# Prepare GPU arrays exactly as decode() does
exp_gpu = cp.asarray(comp['exp_packed'])
sm_gpu = cp.asarray(comp['sm_packed'])
overflow_gpu = cp.asarray(comp['overflow_packed'])

exp_codes_gpu = cp.asarray(comp['exp_codes'])
escape_flags = (exp_codes_gpu >= comp['k']).astype(cp.uint32)
escape_prefix = cp.cumsum(escape_flags) - escape_flags

print(f"exp_gpu dtype: {exp_gpu.dtype}, shape: {exp_gpu.shape}")
print(f"sm_gpu dtype: {sm_gpu.dtype}, shape: {sm_gpu.shape}")
print(f"overflow_gpu dtype: {overflow_gpu.dtype}, shape: {overflow_gpu.shape}")
print(f"escape_prefix dtype: {escape_prefix.dtype}, shape: {escape_prefix.shape}")
print(f"escape_prefix values: {escape_prefix.get()}")

n = comp['n_elements']
output_gpu = cp.zeros(n, dtype=cp.uint8)

kernel = cp.RawKernel(TWOSTREAM_DECODE_KERNEL, 'fp8_twostream_decode')

threads = 256
blocks = (n + threads - 1) // threads

kernel(
    (blocks,), (threads,),
    (exp_gpu, sm_gpu, overflow_gpu, escape_prefix,
     output_gpu, comp['base_exp'], comp['k'], n)
)
cp.cuda.Stream.null.synchronize()

result = output_gpu.get()
print(f"\nOriginal: {raw}")
print(f"GPU out:  {result}")
print(f"Match: {np.array_equal(raw, result)}")

# Detailed comparison at escape positions
escape_positions = np.where(comp['exp_codes'] >= 3)[0]
for pos in escape_positions:
    esc_idx = int(escape_prefix[pos].get())
    print(f"  pos={pos}: orig={raw[pos]}, gpu={result[pos]}, esc_idx={esc_idx}, "
          f"match={'Y' if raw[pos]==result[pos] else 'N'}")

# Use debug kernel that prints values at escape positions
DEBUG_KERNEL = r"""
extern "C"
__global__ void debug_decode(
    const unsigned char* __restrict__ exp_packed,
    const unsigned char* __restrict__ sm_packed,
    const unsigned char* __restrict__ overflow_packed,
    const unsigned int*  __restrict__ escape_prefix,
    unsigned char*       __restrict__ output,
    const int base_exp,
    const int k,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int exp_byte_idx = idx >> 2;
    int exp_bit_pos  = (3 - (idx & 3)) * 2;
    unsigned char exp_byte = exp_packed[exp_byte_idx];
    int code = (exp_byte >> exp_bit_pos) & 0x3;

    int sm_byte_idx = idx >> 1;
    int sm_nibble   = 1 - (idx & 1);
    unsigned char sm_byte = sm_packed[sm_byte_idx];
    int sm = (sm_byte >> (sm_nibble * 4)) & 0xF;

    int sign = (sm >> 3) & 1;
    int mantissa = sm & 0x7;

    int exponent;
    if (code < k) {
        exponent = base_exp + code;
    } else {
        unsigned int esc_idx = escape_prefix[idx];
        int ov_byte_idx = esc_idx >> 1;
        int ov_nibble = 1 - (esc_idx & 1);
        exponent = (overflow_packed[ov_byte_idx] >> (ov_nibble * 4)) & 0xF;
        if (idx < 32) {
            printf("ESCAPE idx=%d code=%d esc_idx=%u ov_byte=%d ov_nib=%d ov_raw=0x%02x exp=%d sign=%d mant=%d out=%d\\n",
                   idx, code, esc_idx, ov_byte_idx, ov_nibble,
                   (int)overflow_packed[ov_byte_idx], exponent, sign, mantissa,
                   (sign << 7) | (exponent << 3) | mantissa);
        }
    }

    output[idx] = (sign << 7) | (exponent << 3) | mantissa;
}
"""

print("\n--- Debug kernel output ---")
debug_kernel = cp.RawKernel(DEBUG_KERNEL, 'debug_decode')
output2 = cp.zeros(n, dtype=cp.uint8)
debug_kernel(
    (blocks,), (threads,),
    (exp_gpu, sm_gpu, overflow_gpu, escape_prefix,
     output2, comp['base_exp'], comp['k'], n)
)
cp.cuda.Stream.null.synchronize()
