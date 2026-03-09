# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Continuous Improvement Suggestions

It is recommended that you record your thought process and results in `experiments/optimization_log.md` each time you try new method or hyperparameters to prevent repeated experiments in the future. 
If you find any improvements, please be sure to record them. If the improvements were made by adjusting hyperparameters, simply record the changes in a file; if the code/algorithm was modified, be sure to create a new branch for backup.

## Project Overview

DFloat11 is a **lossless compression framework** for LLM and diffusion model weights. It compresses BFloat16 weights ~30% by Huffman-coding the 8-bit exponent, storing the sign+mantissa byte raw. This codebase is a fork of DFloat11. My previous goal is to explore a lossless LLM compression method with a higher compression ratio. I've already done some research, and the code is in the `new_compression` and `experiment` folder, but the results haven't been good.

Now, let's change the target. Currently, there are some lossless compression methods for LLM weights. However, it seems there is no lossless compression method for the optimizer during LLM training. When using Adamw, it is necessary to store two sets of FP32 values ​​with an amount equal to the number of LLM parameters. Wouldn't this have a larger overhead?


## Build & Install

```bash
# Install from source (requires CUDA 12 + PyTorch)
nvcc -O3 -ptx dfloat11/decode.cu -o dfloat11/decode.ptx
pip install .[cuda12]

# Or from PyPI
pip install -U dfloat11[cuda12]
```

Key dependencies: `accelerate`, `dahuffman==0.4.2`, `cupy-cuda12x`, `safetensors`, `transformers`, `huggingface-hub`.

## Running Inference Benchmark

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --model_name_or_path DFloat11/Qwen3-8B-DF11 \
  --prompt "Question: What is a binary tree? Answer:" \
  --num_tokens 512 --batch_size 1
```

Use `--bf16` flag for uncompressed BFloat16 models (comparison baseline).

## Compressing Models

```bash
# LLMs (Llama, Qwen2.5, Qwen3)
python examples/compress_llm/compress_llm.py \
  --model_name_or_path Qwen/Qwen3-8B --save_path ./Qwen3-8B-DF11

# FLUX.1 diffusion models
python examples/compress_flux1/compress_flux.py \
  --model_name_or_path black-forest-labs/FLUX.1-dev --save_path ./FLUX.1-dev-DF11
```

Add `--check_correctness` to verify decompressed weights match originals during compression.

## Architecture

### Core Package: `dfloat11/`

- **`dfloat11.py`** — Main entry point. Contains `DFloat11Model.from_pretrained()` (loads compressed models) and `compress_model()` (compresses BFloat16 models). The loading path: download/resolve model -> init model without weights via `AutoModelForCausalLM.from_config()` -> load compressed safetensors -> register forward pre-hooks for on-the-fly CUDA decompression.
- **`dfloat11_utils.py`** — Huffman codec construction (`get_codec`, `get_32bit_codec`), lookup table generation (`get_luts`), and weight encoding (`encode_weights`). BFloat16 is split into 8-bit exponent (Huffman-coded) and 8-bit sign+mantissa (stored raw).
- **`decode.cu`** — CUDA kernel for GPU decompression. Each thread decodes `BYTES_PER_THREAD=8` bytes of Huffman-encoded exponents using multi-level LUT traversal (values >= 240 indicate continuation to next LUT level). Uses shared memory for prefix-sum output positioning and a write buffer. Compiled to PTX at build time.

### Decompression Flow (inference)

`DFloat11Model.from_pretrained()` registers a **forward pre-hook** on each compressed module. On each forward pass:
1. The hook launches the CUDA `decode` kernel to reconstruct BFloat16 weights from `encoded_exponent` + `sign_mantissa` buffers
2. Reconstructed weights are injected into the module's `.weight` attribute
3. After the matmul, weights are discarded (when using CPU offloading)

`TensorManager` is a static allocator that reuses GPU tensors across hooks to avoid repeated allocation.

### Compression Flow

`compress_model()` iterates modules matching regex patterns in `pattern_dict`. For each matched module:
1. Extract and concatenate BFloat16 weight tensors
2. Build per-module Huffman codec from exponent byte frequencies
3. Encode exponents, generate LUTs, compute `output_positions` and `gaps` for CUDA kernel synchronization
4. Store compressed buffers as registered buffers on the module
5. Save as safetensors + config.json with `dfloat11_config`

The `pattern_dict` maps regex patterns to tuples of sub-module paths (e.g., `r"model\.layers\.\d+"` -> `("self_attn.q_proj", "self_attn.k_proj", ...)`).

### Experimental: `new_compression/`

Research code exploring compression beyond DFloat11's exponent-only Huffman approach:

- **`codec.py`** — Predictive coding (left-predictor residuals) + cross-layer delta encoding (layer-to-layer differences) + byte-level Huffman on the full int16 high byte. Pipeline: `bf16 -> int16 -> cross-layer delta -> predictive coding -> byte split -> Huffman(high byte)`.
- **`compress_llm.py` / `validate_llm.py`** — CLI tools to compress/validate LLMs with the new codec. Run as `python -m new_compression.compress_llm` / `python -m new_compression.validate_llm`.
- **`sensitivity_compress.py`** — Output-preserving compression: binary-searches for mantissa LSBs that can be zeroed without changing final logits, then re-compresses. Run as `python -m new_compression.sensitivity_compress`.

### Analysis Scripts (root)

- `analyze_cross_layer.py` — Generates 3D waterfall plots comparing original vs delta-from-mean weight distributions across layers; prints entropy/std statistics.
- `analyze_dual_group.py` — Additional cross-layer analysis.

## Key Conventions

- All weight compression assumes **BFloat16** input tensors — assertions enforce this.
- The CUDA kernel config uses `threads_per_block=(512,)` and `bytes_per_thread=8` as constants.
- Compressed models store a `dfloat11_config` dict in `config.json` containing `version`, `threads_per_block`, `bytes_per_thread`, and `pattern_dict`.
- Multi-device distribution uses HuggingFace Accelerate's `dispatch_model` with `no_split_module_classes` derived from `pattern_dict` to keep compressed buffers on the same device as their module.
- CPU offloading pins `encoded_exponent` and `sign_mantissa` tensors to host memory and transfers them to GPU on-demand per forward pass.
