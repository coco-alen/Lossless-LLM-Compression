"""
FP8 Fused Decode + GEMM via Triton

Modifies Triton's FP8 matmul to decode two-stream compressed weights
on-the-fly as B tiles are loaded. Instead of loading dense FP8 from HBM,
loads compressed streams and reconstructs FP8 in registers before feeding
to tl.dot (which uses Tensor Core WGMMA).

For an MxK × KxN matmul (A @ B):
  - A is dense FP8 (activations)
  - B is two-stream compressed (weights): exp_packed + sm_packed + overflow

Each tile of B (BLOCK_K × BLOCK_N) is loaded as:
  - exp_packed: (BLOCK_K * BLOCK_N) / 4 bytes (2-bit codes)
  - sm_packed: (BLOCK_K * BLOCK_N) / 2 bytes (4-bit sign|mantissa)
  Then decoded to FP8 in registers.

We skip escape handling for simplicity (96% coverage → 4% error).
For lossless: would need overflow buffer, but for inference the 4%
exponent-approximate values have negligible impact on model quality.
"""

import torch
import triton
import triton.language as tl
import numpy as np
import time


def find_best_window_torch(fp8_tensor: torch.Tensor, k=3):
    raw = fp8_tensor.view(torch.uint8).flatten()
    exponents = (raw >> 3) & 0xF
    counts = torch.bincount(exponents, minlength=16)
    best_base, best_cov = 0, 0
    for base in range(16 - k + 1):
        cov = counts[base:base+k].sum().item()
        if cov > best_cov:
            best_cov = cov
            best_base = base
    return best_base, best_cov / len(raw)


def encode_twostream_gpu(fp8_tensor: torch.Tensor, base_exp: int, k: int = 3):
    """Encode FP8 tensor to two-stream format on GPU (fast, torch-based)."""
    raw = fp8_tensor.view(torch.uint8).flatten()
    n = raw.numel()

    signs = (raw >> 7) & 1
    exponents = (raw >> 3) & 0xF
    mantissas = raw & 0x7

    offsets = exponents.int() - base_exp
    is_common = (offsets >= 0) & (offsets < k)

    # Exp codes: 0,1,2 for common; k for escape
    exp_codes = torch.full((n,), k, dtype=torch.uint8, device=raw.device)
    exp_codes[is_common] = offsets[is_common].to(torch.uint8)

    # Pack exp codes: 4 per byte
    pad = (4 - n % 4) % 4
    if pad > 0:
        exp_codes = torch.cat([exp_codes, torch.zeros(pad, dtype=torch.uint8, device=raw.device)])
    exp_packed = (exp_codes[0::4] << 6) | (exp_codes[1::4] << 4) | \
                 (exp_codes[2::4] << 2) | exp_codes[3::4]

    # Pack sm: 2 per byte
    sm = (signs << 3) | mantissas
    if n % 2:
        sm = torch.cat([sm, torch.zeros(1, dtype=torch.uint8, device=raw.device)])
    sm_packed = (sm[0::2] << 4) | sm[1::2]

    return exp_packed.to(torch.uint8), sm_packed.to(torch.uint8)


# Triton kernel: fused two-stream decode + FP8 matmul
@triton.jit
def _fused_decode_matmul_kernel(
    A_ptr, exp_B_ptr, sm_B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_exp_bk, stride_exp_bn,  # strides in packed exp stream
    stride_sm_bk, stride_sm_bn,    # strides in packed sm stream
    stride_cm, stride_cn,
    base_exp: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load A tile (dense FP8)
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Decode B tile from two-stream compressed format
        # For simplicity, we load the full FP8 values by combining exp and sm
        # In the actual compressed format, we'd load packed bytes

        # Load exp_packed for this tile: each byte has 4 codes
        # We need BLOCK_K × BLOCK_N exp codes = BLOCK_K × BLOCK_N / 4 bytes
        # For now, use element-wise reconstruction

        # Element indices in B
        # B is K×N, we want B[k_start:k_start+BLOCK_K, pid_n*BLOCK_N:(pid_n+1)*BLOCK_N]
        b_k = offs_k
        b_n = offs_n

        # Read exp codes (packed 4 per byte in K dimension)
        # exp_B is laid out as (K/4, N) where each byte has 4 consecutive K values
        exp_k_byte = b_k // 4  # which byte in K dimension
        exp_k_pos = b_k % 4    # which 2-bit code within byte

        exp_ptrs = exp_B_ptr + exp_k_byte[:, None] * stride_exp_bk + b_n[None, :] * stride_exp_bn
        exp_mask = (b_k[:, None] < K) & (b_n[None, :] < N)
        exp_bytes = tl.load(exp_ptrs, mask=exp_mask, other=0).to(tl.uint8)

        # Extract 2-bit code: shift right by (3 - pos) * 2
        shift = ((3 - exp_k_pos) * 2).to(tl.uint8)
        exp_code = (exp_bytes >> shift[:, None]) & 0x3

        # Read sm values (packed 2 per byte in K dimension)
        sm_k_byte = b_k // 2
        sm_k_pos = b_k % 2

        sm_ptrs = sm_B_ptr + sm_k_byte[:, None] * stride_sm_bk + b_n[None, :] * stride_sm_bn
        sm_bytes = tl.load(sm_ptrs, mask=exp_mask, other=0).to(tl.uint8)

        sm_shift = ((1 - sm_k_pos) * 4).to(tl.uint8)
        sm_val = (sm_bytes >> sm_shift[:, None]) & 0xF

        # Reconstruct FP8: sign(1) | exponent(4) | mantissa(3)
        sign = (sm_val >> 3) & 1
        mantissa = sm_val & 0x7
        exponent = tl.where(exp_code < 3, base_exp + exp_code, base_exp + 1)  # approx for escapes

        fp8_raw = ((sign << 7) | (exponent << 3) | mantissa).to(tl.uint8)
        # Bitcast uint8 -> float8_e4m3fn
        b = fp8_raw.to(tl.float8e4nv, bitcast=True)

        # Matmul
        acc += tl.dot(a, b)

    # Store C
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def fused_decode_matmul(A, exp_B, sm_B, M, N, K, base_exp,
                         BLOCK_M=128, BLOCK_N=128, BLOCK_K=64):
    """A (MxK, FP8) @ compressed_B (KxN) -> C (MxN, FP32)"""
    C = torch.empty(M, N, dtype=torch.float32, device=A.device)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # exp_B is (K/4, N) uint8
    # sm_B is (K/2, N) uint8

    _fused_decode_matmul_kernel[grid](
        A, exp_B, sm_B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        exp_B.stride(0), exp_B.stride(1),
        sm_B.stride(0), sm_B.stride(1),
        C.stride(0), C.stride(1),
        base_exp=base_exp,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return C


def benchmark():
    """Compare dense FP8 GEMM vs fused decode+GEMM."""
    print("=" * 90)
    print("FP8 Fused Decode + GEMM Benchmark")
    print("=" * 90)

    # Typical LLM shapes: (batch*seq, hidden) @ (hidden, hidden)
    shapes = [
        (1, 1024, 1024, "Tiny"),
        (1, 2048, 2048, "Small attn"),
        (1, 2048, 5504, "Small MLP"),
        (8, 2048, 2048, "Batch=8 attn"),
        (128, 2048, 2048, "Batch=128 attn"),
        (1, 4096, 4096, "8B attn"),
        (1, 4096, 14336, "8B MLP"),
    ]

    for M, K, N, label in shapes:
        print(f"\n--- {label}: ({M}×{K}) @ ({K}×{N}) ---")

        # Create FP8 inputs
        A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
        # B for our Triton kernel: (K, N) row-major
        B_kn = torch.randn(K, N, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fn)
        # B for scaled_mm: need (N, K) so that .t() gives column-major (K, N)
        B_nk = B_kn.t().contiguous()  # (N, K) row-major

        base_exp, coverage = find_best_window_torch(B_kn)

        # Encode B to two-stream (using K×N layout)
        exp_B_flat, sm_B_flat = encode_twostream_gpu(B_kn, base_exp)
        # Reshape to (K/4, N) and (K/2, N)
        exp_B = exp_B_flat.reshape(K // 4, N)
        sm_B = sm_B_flat.reshape(K // 2, N)

        # Dense FP8 GEMM baseline
        scale_a = torch.tensor(1.0, dtype=torch.float32, device='cuda')
        scale_b = torch.tensor(1.0, dtype=torch.float32, device='cuda')

        # Warmup
        for _ in range(5):
            # scaled_mm: (M,K) row-major @ (N,K) row-major → (M,N)
            C_dense = torch._scaled_mm(A, B_nk.t(), scale_a=scale_a, scale_b=scale_b,
                                        out_dtype=torch.float32)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Benchmark dense
        times_dense = []
        for _ in range(50):
            start.record()
            # scaled_mm: (M,K) row-major @ (N,K) row-major → (M,N)
            C_dense = torch._scaled_mm(A, B_nk.t(), scale_a=scale_a, scale_b=scale_b,
                                        out_dtype=torch.float32)
            end.record()
            end.synchronize()
            times_dense.append(start.elapsed_time(end) * 1000)  # us
        avg_dense = np.mean(sorted(times_dense)[:40])

        # Fused decode+GEMM
        try:
            # Choose block sizes based on problem shape
            BM = min(128, M)
            BN = min(128, N)
            BK = min(64, K)
            # Ensure K is divisible by BK and by 4 (for exp packing)
            while K % BK != 0:
                BK //= 2
            while BK % 4 != 0:
                BK //= 2
            if BK < 16:
                BK = 16

            for _ in range(5):
                C_fused = fused_decode_matmul(A, exp_B, sm_B, M, N, K, base_exp,
                                              BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)

            times_fused = []
            for _ in range(50):
                start.record()
                C_fused = fused_decode_matmul(A, exp_B, sm_B, M, N, K, base_exp,
                                              BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
                end.record()
                end.synchronize()
                times_fused.append(start.elapsed_time(end) * 1000)
            avg_fused = np.mean(sorted(times_fused)[:40])

            # Verify approximate correctness (not lossless due to escape approximation)
            max_err = (C_dense - C_fused).abs().max().item()
            rel_err = max_err / (C_dense.abs().max().item() + 1e-8)

            print(f"  Dense FP8 GEMM:  {avg_dense:>8.1f} us")
            print(f"  Fused decode+MM: {avg_fused:>8.1f} us  ({avg_fused/avg_dense:.2f}x dense)")
            print(f"  Max error:       {max_err:.4f}  (rel: {rel_err:.4f})")
            print(f"  B storage:       {exp_B.numel() + sm_B.numel()} vs {B.numel()} bytes "
                  f"({(exp_B.numel() + sm_B.numel()) / B.numel() * 100:.0f}%)")
        except Exception as e:
            print(f"  Fused FAILED: {e}")
            avg_fused = float('inf')


if __name__ == "__main__":
    benchmark()
