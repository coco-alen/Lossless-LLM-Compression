"""Debug two-stream codec correctness."""
import torch
import numpy as np
import cupy as cp
import sys
sys.path.insert(0, '.')
from experiments.fused_codec.fp8_twostream import FP8TwoStreamEncoder, FP8TwoStreamDecoder

# Create a small known FP8 tensor
torch.manual_seed(42)
bf16 = torch.randn(32, dtype=torch.bfloat16)
fp8 = bf16.to(torch.float8_e4m3fn)
raw = fp8.view(torch.uint8).flatten().numpy()

print("Original FP8 bytes:", raw[:32])
print("Signs:    ", (raw >> 7) & 1)
print("Exponents:", (raw >> 3) & 0xF)
print("Mantissas:", raw & 0x7)

encoder = FP8TwoStreamEncoder(k=3)
comp = encoder.encode(fp8)

print(f"\nbase_exp: {comp['base_exp']}, coverage: {comp['coverage']:.1%}, n_escapes: {comp['n_escapes']}")
print(f"exp_codes: {comp['exp_codes'][:32]}")
print(f"exp_packed: {comp['exp_packed'][:8]} (hex: {[hex(x) for x in comp['exp_packed'][:8]]})")
print(f"sm_packed: {comp['sm_packed'][:16]} (hex: {[hex(x) for x in comp['sm_packed'][:16]]})")

# Manual decode
n = comp['n_elements']
exp_codes = comp['exp_codes']
base_exp = comp['base_exp']
signs = (raw >> 7) & 1
mantissas = raw & 0x7
exponents = (raw >> 3) & 0xF

# Check sm packing/unpacking
sm = ((signs << 3) | mantissas).astype(np.uint8)
print(f"\nsm values:  {sm[:32]}")

# Manually unpack sm from sm_packed
sm_packed = comp['sm_packed']
sm_unpacked = np.zeros(n, dtype=np.uint8)
for i in range(n):
    byte_idx = i >> 1
    nibble = 1 - (i & 1)  # 1=high, 0=low
    sm_unpacked[i] = (sm_packed[byte_idx] >> (nibble * 4)) & 0xF
print(f"sm unpack:  {sm_unpacked[:32]}")
print(f"sm match: {np.array_equal(sm, sm_unpacked)}")

# Manually unpack exp codes from exp_packed
exp_packed = comp['exp_packed']
exp_unpacked = np.zeros(n, dtype=np.uint8)
for i in range(n):
    byte_idx = i >> 2
    bit_pos = (3 - (i & 3)) * 2
    exp_unpacked[i] = (exp_packed[byte_idx] >> bit_pos) & 0x3
print(f"\nexp_codes:  {exp_codes[:32]}")
print(f"exp unpack: {exp_unpacked[:32]}")
print(f"exp match: {np.array_equal(exp_codes, exp_unpacked)}")

# Manually reconstruct
reconstructed = np.zeros(n, dtype=np.uint8)
esc_idx = 0
overflow = comp['overflow_packed']
for i in range(n):
    code = exp_unpacked[i]
    s = (sm_unpacked[i] >> 3) & 1
    m = sm_unpacked[i] & 0x7
    if code < 3:
        e = base_exp + code
    else:
        ov_byte = esc_idx >> 1
        ov_nib = 1 - (esc_idx & 1)
        e = (overflow[ov_byte] >> (ov_nib * 4)) & 0xF
        esc_idx += 1
    reconstructed[i] = (s << 7) | (e << 3) | m

print(f"\nOriginal:      {raw[:32]}")
print(f"Reconstructed: {reconstructed[:32]}")
print(f"CPU match: {np.array_equal(raw, reconstructed)}")

# GPU decode
decoder = FP8TwoStreamDecoder()
recovered = decoder.decode(comp)
rec_np = recovered.cpu().view(torch.uint8).flatten().numpy()
print(f"\nGPU decoded:   {rec_np[:32]}")
print(f"GPU match: {np.array_equal(raw, rec_np)}")

# Find first mismatch
mismatches = np.where(raw != rec_np)[0]
if len(mismatches) > 0:
    idx = mismatches[0]
    print(f"\nFirst mismatch at idx {idx}:")
    print(f"  Original: {raw[idx]} (S={raw[idx]>>7} E={(raw[idx]>>3)&0xF} M={raw[idx]&0x7})")
    print(f"  Decoded:  {rec_np[idx]} (S={rec_np[idx]>>7} E={(rec_np[idx]>>3)&0xF} M={rec_np[idx]&0x7})")
    print(f"  exp_code: {exp_codes[idx]}")
    print(f"  base_exp: {base_exp}")
else:
    print("\nPERFECT MATCH!")
