"""Compute byte-level and value-level entropy for AWQ qweights, BF16, and FP8 models."""
import numpy as np
import torch
from collections import Counter
import math
import glob
import os
from safetensors import safe_open

def shannon_entropy(counts_dict, total):
    """Compute Shannon entropy in bits from a counts dictionary."""
    entropy = 0.0
    for c in counts_dict.values():
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy

# ============================================================
# Task 1: AWQ qweight byte-level entropy
# ============================================================
print("=" * 60)
print("Task 1: AWQ qweight byte-level entropy (Qwen2.5-7B-Instruct-AWQ)")
print("=" * 60)

from huggingface_hub import snapshot_download
awq_path = snapshot_download("Qwen/Qwen2.5-7B-Instruct-AWQ")
print(f"AWQ model path: {awq_path}")

# Find safetensor files
st_files = sorted(glob.glob(os.path.join(awq_path, "*.safetensors")))
print(f"Found {len(st_files)} safetensor files")

all_bytes = []
total_qweight_elements = 0
for sf in st_files:
    with safe_open(sf, framework="pt", device="cpu") as f:
        for key in f.keys():
            if "qweight" in key:
                tensor = f.get_tensor(key)
                total_qweight_elements += tensor.numel()
                # View int32 as bytes
                raw = tensor.contiguous().view(torch.int32).numpy().view(np.uint8)
                all_bytes.append(raw.ravel())

all_bytes = np.concatenate(all_bytes)
print(f"Total qweight int32 elements: {total_qweight_elements}")
print(f"Total bytes: {len(all_bytes)}")

counts = Counter(all_bytes.tolist())
unique_bytes = len(counts)
entropy_bpb = shannon_entropy(counts, len(all_bytes))
print(f"AWQ qweight byte-level entropy: {entropy_bpb:.4f} bits/byte")
print(f"Unique byte values: {unique_bytes}/256")
print(f"Theoretical compression ratio vs raw: {entropy_bpb/8:.4f}")
print()

# ============================================================
# Task 2: Qwen3-4B BF16 entropy (16-bit value level)
# ============================================================
print("=" * 60)
print("Task 2: Qwen3-4B BF16 entropy (per 16-bit value)")
print("=" * 60)

qwen3_path = snapshot_download("Qwen/Qwen3-4B")
print(f"Qwen3-4B path: {qwen3_path}")

st_files = sorted(glob.glob(os.path.join(qwen3_path, "*.safetensors")))
print(f"Found {len(st_files)} safetensor files")

all_int16 = []
total_bf16_elements = 0
for sf in st_files:
    with safe_open(sf, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if tensor.dtype == torch.bfloat16:
                total_bf16_elements += tensor.numel()
                raw = tensor.contiguous().view(torch.int16).numpy().view(np.uint16)
                all_int16.append(raw.ravel())

all_int16 = np.concatenate(all_int16)
print(f"Total BF16 elements: {total_bf16_elements}")

counts16 = Counter(all_int16.tolist())
unique_16bit = len(counts16)
entropy_bp16 = shannon_entropy(counts16, len(all_int16))
print(f"Qwen3-4B BF16 entropy: {entropy_bp16:.4f} bits per 16-bit value")
print(f"Unique 16-bit values: {unique_16bit}/65536")
print(f"Bits per byte equivalent: {entropy_bp16/2:.4f}")
print()

# ============================================================
# Task 3: Qwen3-4B FP8 cast entropy (per 8-bit value)
# ============================================================
print("=" * 60)
print("Task 3: Qwen3-4B FP8 (float8_e4m3fn) cast entropy")
print("=" * 60)

all_fp8_bytes = []
total_fp8_elements = 0
for sf in st_files:
    with safe_open(sf, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if tensor.dtype == torch.bfloat16:
                # Cast to FP8 on CPU - need to go through float first
                fp8 = tensor.to(torch.float32).to(torch.float8_e4m3fn)
                total_fp8_elements += fp8.numel()
                raw = fp8.contiguous().view(torch.uint8).numpy()
                all_fp8_bytes.append(raw.ravel())

all_fp8_bytes = np.concatenate(all_fp8_bytes)
print(f"Total FP8 elements: {total_fp8_elements}")

counts8 = Counter(all_fp8_bytes.tolist())
unique_8bit = len(counts8)
entropy_bp8 = shannon_entropy(counts8, len(all_fp8_bytes))
print(f"Qwen3-4B FP8 entropy: {entropy_bp8:.4f} bits per 8-bit value")
print(f"Unique 8-bit values: {unique_8bit}/256")
print(f"Theoretical compression ratio vs raw FP8: {entropy_bp8/8:.4f}")
print()

# ============================================================
# Summary
# ============================================================
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"AWQ qweight:  {entropy_bpb:.4f} bits/byte,  {unique_bytes} unique byte values")
print(f"Qwen3-4B BF16: {entropy_bp16:.4f} bits/16-bit value, {unique_16bit} unique 16-bit values")
print(f"Qwen3-4B FP8:  {entropy_bp8:.4f} bits/8-bit value,  {unique_8bit} unique 8-bit values")
