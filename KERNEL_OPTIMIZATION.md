## Project Goal

Build the fastest GPU-decodable lossless compression for FP8 (e4m3fn) LLM weights on NVIDIA H200 (SM 90a, Hopper), achieving **the highest decode throughput (GB/s)** while maintaining compression ratio ≤ 80% of dense FP8.

The ultimate target is a **fused decode+GEMM kernel** that loads compressed FP8 weights from HBM, decompresses on-the-fly in shared memory/registers, and feeds them directly to Tensor Core WGMMA — achieving both storage savings AND inference speedup over dense FP8.

**Primary metric**: Decode throughput (GB/s) — measured as original_bytes / decode_time.
**Secondary metric**: Compression ratio (%) — compressed_size / original_size × 100. Must be ≤ 80%.
**Constraint**: Bit-exact lossless — decoded FP8 bytes must match original exactly.

### Current Best Results

| Method | FP8 Ratio | Decode GB/s | Fusable? | Notes |
|--------|----------|-------------|----------|-------|
| Dense FP8 | 100% | ~4800 (HBM bandwidth) | Baseline | No compression |
| nvCOMP ANS (byte-level) | 85.7% | 29-56 | No (separate kernel) | NVIDIA library |
| Our Huffman (GPU, per-layer) | 77.1% | 5-14 | No (variable-length) | DFloat11-style LUT |
| Our Huffman (shared table, analytical) | 71.0% | ~10 (est.) | No | Best compression |
| Our ANS (CPU only) | 70.4% | N/A | N/A | Entropy optimal |
| TCA-TBE fixed-width (k=3, unpacked) | ~117% | Fast | Yes | Worse than dense (overhead) |

### FP8 Distribution Facts (Qwen3-0.6B, 596M params)
- Only 117 unique FP8 byte values (out of 256 possible)
- Entropy: 5.65 bits / 8 bits = 70.6% compression limit
- Exponent (4 bits): 1.72 bits entropy, top-3 consecutive cover 96%
- Mantissa (3 bits): 2.98 bits entropy (nearly random)
- Sign (1 bit): 1.00 bits entropy (random)
- Joint correlation (full value vs separated fields): 0.047 bits

### Key Challenge
Variable-length codes (Huffman/ANS) achieve the best ratio (~71%) but decode at only 5-14 GB/s — 100x slower than HBM bandwidth (4.8 TB/s). Fixed-width codes decode fast but achieve poor ratio (~79% theoretical, 117% with naive implementation). The gap between these two regimes is the core research problem.

## Experiment

Each experiment runs on the H200 GPU. Use conda env `quant`. You are sharing the GPU with other programs; do not terminate any other processes.

### What you CAN modify:

- `experiments/fused_codec/fp8_fused_huffman.py` — FP8 Huffman encoder/decoder
- `experiments/fused_codec/fp8_tbe.py` — FP8 TCA-TBE encoder/decoder
- `experiments/fused_codec/profile_fp8.py` — FP8 distribution profiling
- Create new files in `experiments/fused_codec/` for new approaches
- The CUDA/CuPy kernel code within these files
- Triton kernels if you choose that path
- Any new encoding format or decode strategy

### What you CANNOT modify:

- Do NOT modify files outside `experiments/fused_codec/`
- Do NOT change the FP8 test data generation (model weights must be real, not synthetic)
- Do NOT disable the lossless verification check
- Do NOT kill other processes on the GPU

### Benchmark command:

The standard benchmark evaluates on Qwen3-0.6B FP8 weights. Each approach must report:
1. Compression ratio (%)
2. Decode throughput (GB/s) — original bytes / decode time
3. Lossless verification (pass/fail)
4. Which layers were tested

For quick iteration, test on embed_tokens (155M params, largest layer) and one attention layer (2M params). For final results, test on all layers ≥ 500K params.

### Logging results:

Record ALL experimental conditions and results in `experiments/fused_codec/EXPERIMENT.md`. Document results **regardless** of whether they improve or deteriorate. Each entry should include:
- Experiment name and approach description
- Key code changes (algorithm, data layout, kernel parameters)
- Compression ratio, decode throughput, lossless status
- Why it worked or didn't
- All run logs go in `experiments/fused_codec/logs/`

### The experiment loop

The experiment runs on a dedicated branch: `git checkout -b fp8-kernel-opt`

LOOP FOREVER:

1. Look at the git state: current branch/commit
2. Modify code with an experimental idea
3. `git commit -m "exp: <description>"`
4. Run the experiment: `python experiments/fused_codec/<script>.py > experiments/fused_codec/logs/<exp_name>.log 2>&1`
5. Read out the results: `grep -E "Aggregate|Dec GB/s|Ratio|Lossless" experiments/fused_codec/logs/<exp_name>.log`
6. If grep output is empty, the run crashed. Run `tail -n 50` on the log to read the stack trace and attempt a fix. If you can't fix after a few attempts, skip it.
7. Record the results in `experiments/fused_codec/EXPERIMENT.md`
8. If decode throughput improved OR compression ratio improved (without breaking the other), keep the commit ("advance the branch")
9. If both metrics are equal or worse, `git reset --hard HEAD~1` to revert
10. Move on to the next idea

**Crashes:** If a run crashes (OOM, CUDA error, bug), fix trivial issues and re-run. If the approach is fundamentally broken, skip it, log "crash" as status, and move on.

**Timeout:** FP8 encoding of all Qwen3-0.6B layers takes ~minutes (Python encoder). Decoding should be <10s total. If a script runs >15 minutes, kill it and log as timeout.

**NEVER STOP:** Once the loop begins, do NOT pause to ask if you should continue. The human may be asleep and expects autonomous progress. If you run out of ideas, re-read the code, study DFloat11's kernel, look at nvCOMP's design, try combining approaches. The loop runs until manually interrupted.

## Ideas to Explore (starting points)

### Encoding Format Ideas:
1. **Packed 6-bit TCA-TBE**: Pack 4 × 6-bit codes per 3 bytes (instead of 1 byte each). k=3 exponent window + sign + mantissa = 6 bits. ~79% ratio, constant-time decode.
2. **Packed 5-bit format**: Top 31 FP8 values get 5-bit codes (covers ~70%), escapes get 5+8=13 bits. Pack 8 codes per 5 bytes.
3. **Two-stream separation**: Store exponents compressed (2-bit offset, covers 96%) and sign+mantissa raw (4 bits packed 2 per byte). Total: 2+4=6 bits → 75%.
4. **Byte-level lookup**: Create a 256→N mapping where N common values get short codes. Escape via flag byte.
5. **Tile-local palettes**: Per-tile (e.g., 128 values) palette of top-K values. Different tiles may have different palettes.
6. **Hybrid Huffman**: Use Huffman only for the exponent (4-bit alphabet, fast LUT), keep sign+mantissa raw.
7. **Run-length on exponents**: Consecutive values often share exponents. RLE the exponent stream.
8. **nvCOMP integration**: Use nvCOMP's ANS/LZ4 but on separated byte streams (high nibble vs low nibble).

### Kernel Optimization Ideas:
9. **Increase BYTES_PER_THREAD**: Currently 8. Try 16, 32 — more work per thread hides latency.
10. **Warp-cooperative decode**: One warp decodes a tile together using ballot/shuffle.
11. **Shared memory LUT caching**: Load the 256-entry Huffman LUT into shared memory once.
12. **Reduce LUT levels**: Current Huffman uses 9 levels. Try canonical Huffman with max code length 12 → 2 levels.
13. **Vectorized memory access**: Use uint4/uint2 loads for coalesced bitstream reads.
14. **Persistent kernel**: Keep the kernel running across multiple weight matrices.
15. **Triton kernel**: Rewrite decode in Triton for auto-tuning of block sizes and memory access.
16. **Double-buffered decode**: Overlap decode of next tile with consumption of current tile.

### Fused Decode+GEMM Ideas:
17. **Triton FP8 matmul + decode prologue**: Modify Triton's FP8 persistent matmul tutorial to add a decode step before loading B tiles.
18. **CUTLASS mainloop modification**: Replace B's global load with compressed load + shared memory decode.
19. **Staged approach**: Dense decode to a small staging buffer in shared memory, then standard WGMMA.

## Architecture Reference

### FP8 e4m3fn Bit Layout
```
Bit 7: Sign (S)
Bits 6-3: Exponent (E3 E2 E1 E0)
Bits 2-0: Mantissa (M2 M1 M0)

Value = (-1)^S × 2^(E-7) × (1 + M/8)    [normal]
Value = (-1)^S × 2^(-6) × (M/8)          [subnormal, E=0]
```

### File Structure
| File | Purpose |
|------|---------|
| `experiments/fused_codec/fp8_fused_huffman.py` | FP8 Huffman encoder + CuPy GPU decoder |
| `experiments/fused_codec/fp8_tbe.py` | FP8 TCA-TBE fixed-width encoder + GPU decoder |
| `experiments/fused_codec/profile_fp8.py` | FP8 distribution profiling across models |
| `experiments/fused_codec/fp8_inference_demo.py` | End-to-end compressed inference benchmark |
| `experiments/fused_codec/EXPERIMENT.md` | Experiment log (create this) |
| `experiments/fused_codec/logs/` | Run logs (create this directory) |

### Key Libraries
- `cupy` — CuPy RawKernel for CUDA kernels
- `triton` — Triton compiler for fused kernels
- `dahuffman==0.4.2` — Huffman codec (for encoder)
- `nvidia.nvcomp` — nvCOMP GPU compression (for comparison)
- `torch._scaled_mm` — PyTorch FP8 scaled GEMM (for baseline)
- `constriction` — ANS coding (for entropy-optimal baseline)

### H200 Hardware Specs
- HBM3e bandwidth: 4.8 TB/s
- FP8 Tensor Core: 1979 TFLOPS
- Shared memory per SM: 228 KB (configurable)
- L2 cache: 50 MB
- SM count: 132
- CUDA cores per SM: 128

### DFloat11 Kernel Architecture (reference)
The existing DFloat11 kernel (`dfloat11/decode.cu`) uses:
1. Each thread loads BYTES_PER_THREAD=8 bytes of Huffman bitstream
2. Phase 1: Count decoded symbols per thread (no writes)
3. Prefix-sum across block to compute output positions
4. Phase 2: Decode again, write to shared memory buffer
5. Coalesced global write from shared memory

Key insight: the two-phase approach avoids global atomics. The prefix-sum tells each thread where to write.

### ZipServ TCA-TBE Architecture (reference, ASPLOS '26)
- Fixed 3-bit codeword per BF16 value (top-7 consecutive exponents)
- Each 8×8 tile has 3 bitmaps: bitmap0, bitmap1, bitmap2 (one per code bit)
- Common values: reconstruct from base_exponent + 3-bit offset + raw sign+mantissa
- Escape values: stored in overflow buffer, located via bitmap scan
- Fused into Tensor Core GEMM via ZipGEMM: decode in register file, feed to mma.sync
