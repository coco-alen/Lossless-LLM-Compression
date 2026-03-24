"""Debug escape prefix sum on GPU."""
import torch, numpy as np, cupy as cp
import sys
sys.path.insert(0, '.')
from experiments.fused_codec.fp8_twostream import FP8TwoStreamEncoder

torch.manual_seed(42)
bf16 = torch.randn(32, dtype=torch.bfloat16)
fp8 = bf16.to(torch.float8_e4m3fn)

encoder = FP8TwoStreamEncoder(k=3)
comp = encoder.encode(fp8)

exp_codes = comp['exp_codes']
print(f"exp_codes: {exp_codes}")

# CPU escape prefix
escape_flags_cpu = (exp_codes >= 3).astype(np.uint32)
prefix_cpu = np.cumsum(escape_flags_cpu) - escape_flags_cpu
print(f"escape_flags (CPU): {escape_flags_cpu}")
print(f"escape_prefix (CPU): {prefix_cpu}")

# GPU escape prefix
exp_codes_gpu = cp.asarray(exp_codes)
escape_flags_gpu = (exp_codes_gpu >= 3).astype(cp.uint32)
prefix_gpu = cp.cumsum(escape_flags_gpu) - escape_flags_gpu
print(f"escape_prefix (GPU): {prefix_gpu.get()}")
print(f"Match: {np.array_equal(prefix_cpu, prefix_gpu.get())}")

# Check overflow packing
overflow = comp['overflow_packed']
print(f"\noverflow_packed: {overflow} (hex: {[hex(x) for x in overflow]})")

# Manually decode escapes
escape_positions = np.where(exp_codes >= 3)[0]
print(f"escape positions: {escape_positions}")
for i, pos in enumerate(escape_positions):
    ov_byte = i >> 1
    ov_nib = 1 - (i & 1)
    exp_val = (overflow[ov_byte] >> (ov_nib * 4)) & 0xF
    raw = fp8.view(torch.uint8).flatten().numpy()
    orig_exp = (raw[pos] >> 3) & 0xF
    print(f"  escape {i} at pos {pos}: overflow_exp={exp_val}, original_exp={orig_exp}, match={exp_val==orig_exp}")
