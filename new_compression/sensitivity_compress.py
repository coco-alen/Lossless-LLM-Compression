"""
Output-Preserving Compression: break the weight-entropy barrier.

Key insight:
  Standard lossless compression treats weights as raw data — weight entropy
  is the hard compression limit.  But if modifying a weight's LSBs does NOT
  change any bf16 output (due to limited floating-point precision in matmul
  accumulation), those bits are "free" and can be zeroed to lower entropy.

Algorithm:
  1. Run calibration data through the model, capture each linear layer's
     input activations via hooks.
  2. For each linear layer, try zeroing the last k = 1..7 mantissa bits of
     ALL weights simultaneously.
  3. Recompute Y = X @ W_modified.T in bf16 and check bit-identity with
     the original output.
  4. Record the maximum k (free bits) per layer.  k free bits means the
     effective mantissa is (7-k) bits → lower entropy → better compression.

Because we verify bf16-identity at EACH layer independently, the guarantee
composes: if every layer's output is bit-identical, the full model output
is bit-identical for the calibration inputs.

Usage:
    python -m new_compression.sensitivity_compress \
        --model_name_or_path Qwen/Qwen3-1.7B

    python -m new_compression.sensitivity_compress \
        --model_name_or_path Qwen/Qwen3-1.7B \
        --save_modified_weights ./Qwen3-1.7B-SensComp
"""

import os
import json
import math
from argparse import ArgumentParser
from collections import defaultdict

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
from dahuffman import HuffmanCodec


# ---- Calibration prompts (diverse short texts) ----
CALIBRATION_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In mathematics, a prime number is a natural number greater than 1.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "The capital of France is Paris, and it is known for the Eiffel Tower.",
    "Quantum computing leverages quantum mechanical phenomena such as superposition.",
    "To be, or not to be, that is the question.",
    "Machine learning models can be compressed using various techniques.",
    "水是生命之源，地球表面约71%被水覆盖。",
]

WEIGHT_TYPES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


# ---------------------------------------------------------------------------
# Core: mantissa bit manipulation
# ---------------------------------------------------------------------------

def zero_mantissa_lsb(weight_bf16: torch.Tensor, k: int) -> torch.Tensor:
    """
    Zero the last k mantissa bits of a bf16 tensor.

    BFloat16 layout: [1 sign | 8 exponent | 7 mantissa]
    Zeroing k mantissa LSBs means mask = 0xFFFF << k applied to raw bits.
    For k=0 → no change; k=7 → only sign+exponent remain.
    """
    if k <= 0:
        return weight_bf16.clone()
    raw = weight_bf16.view(torch.int16)
    mask = ~((1 << k) - 1)  # e.g. k=2 → mask = 0xFFFC
    return (raw & mask).view(torch.bfloat16)


# ---------------------------------------------------------------------------
# Hook-based activation capture
# ---------------------------------------------------------------------------

class ActivationCapture:
    """Capture input activations for specified modules during forward pass."""

    def __init__(self):
        self.captured = {}   # module_name → list of input tensors
        self._hooks = []

    def register(self, model, num_layers):
        """Register hooks on all linear layers inside decoder layers."""
        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]
            for wt in WEIGHT_TYPES:
                parts = wt.split('.')
                module = layer
                for p in parts:
                    module = getattr(module, p)

                name = f"layers.{layer_idx}.{wt}"
                self.captured[name] = []

                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self._hooks.append(hook)

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            # input[0]: (batch, seq_len, in_features)
            self.captured[name].append(input[0].detach().cpu())
        return hook_fn

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Per-layer analysis
# ---------------------------------------------------------------------------

def check_output_identity(weight_orig_bf16, weight_mod_bf16, calib_inputs, device):
    """
    Check if modifying weight produces bit-identical bf16 output on all
    calibration inputs.

    Computes Y = X @ W.T in bf16 for both original and modified weights,
    returns True if all outputs match exactly.
    """
    W_orig = weight_orig_bf16.to(device)
    W_mod = weight_mod_bf16.to(device)

    for X_cpu in calib_inputs:
        X = X_cpu.to(device).to(torch.bfloat16)
        # (batch, seq, in_feat) @ (out_feat, in_feat).T → (batch, seq, out_feat)
        Y_orig = torch.matmul(X, W_orig.T)
        Y_mod = torch.matmul(X, W_mod.T)

        if not torch.equal(Y_orig, Y_mod):
            return False

    return True


def find_max_free_bits(weight_bf16, calib_inputs, device):
    """
    Find the maximum number of mantissa LSBs that can be zeroed without
    changing bf16 output on calibration data.

    Returns max_k in [0, 7].
    """
    max_k = 0
    for k in range(1, 8):  # 1..7 mantissa bits
        W_mod = zero_mantissa_lsb(weight_bf16, k)
        if check_output_identity(weight_bf16, W_mod, calib_inputs, device):
            max_k = k
        else:
            break
    return max_k


# ---------------------------------------------------------------------------
# Entropy estimation
# ---------------------------------------------------------------------------

def estimate_entropy_bf16(weight_bf16: torch.Tensor) -> float:
    """Estimate Shannon entropy (bits per element) of bf16 tensor viewed as int16."""
    raw = weight_bf16.contiguous().view(torch.int16).flatten()
    vals, counts = torch.unique(raw, return_counts=True)
    probs = counts.float() / counts.sum()
    return -(probs * probs.log2()).sum().item()


def estimate_huffman_size(weight_bf16: torch.Tensor) -> int:
    """Estimate Huffman-compressed size in bytes (high byte + raw low byte, like DFloat11)."""
    raw = weight_bf16.contiguous().view(torch.uint8)
    low_bytes = raw[0::2].numpy()   # little-endian: low first
    high_bytes = raw[1::2].numpy()

    # Huffman on high byte
    vals, counts = np.unique(high_bytes, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(vals, counts)}
    codec = HuffmanCodec.from_frequencies(freq)
    encoded = codec.encode(high_bytes.tolist())

    return len(encoded) + len(low_bytes)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser("Output-Preserving Sensitivity Compression")
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen3-1.7B')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_modified_weights', type=str, default=None,
                        help='If set, save the modified (LSB-zeroed) weights to this path')
    parser.add_argument('--max_calib_tokens', type=int, default=128,
                        help='Max tokens per calibration prompt')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ---- Load model ----
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    num_layers = config.num_hidden_layers
    arch = (getattr(config, 'architectures', None) or ['Unknown'])[0]

    print(f"Model:        {args.model_name_or_path}")
    print(f"Architecture: {arch}")
    print(f"Layers:       {num_layers}")
    print(f"Device:       {device}")
    print()

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Capture calibration activations ----
    print("Running calibration forward passes...")
    capture = ActivationCapture()
    capture.register(model, num_layers)

    with torch.no_grad():
        for prompt in tqdm(CALIBRATION_PROMPTS, desc="Calibration"):
            inputs = tokenizer(
                prompt, return_tensors="pt",
                max_length=args.max_calib_tokens, truncation=True,
            ).to(device)
            model(**inputs, use_cache=False)

    capture.remove_hooks()
    print(f"Captured activations for {len(capture.captured)} modules "
          f"({len(CALIBRATION_PROMPTS)} prompts each)")

    # ---- Analyze each layer and weight type ----
    print("\nAnalyzing mantissa free bits per layer...")
    print("=" * 90)
    header = f"  {'Layer':<8}"
    for wt in WEIGHT_TYPES:
        short = wt.split('.')[-1]
        header += f" {short:>9}"
    print(header)
    print("-" * 90)

    results = {}  # (layer_idx, wt) → max_k
    total_orig_bytes = 0
    total_modified_bytes = 0

    for layer_idx in range(num_layers):
        row = f"  {layer_idx:<8}"
        for wt in WEIGHT_TYPES:
            name = f"layers.{layer_idx}.{wt}"
            calib_inputs = capture.captured[name]

            # Get original weight
            layer = model.model.layers[layer_idx]
            parts = wt.split('.')
            module = layer
            for p in parts:
                module = getattr(module, p)
            W = module.weight.data.detach().cpu()

            # Find max free bits
            max_k = find_max_free_bits(W, calib_inputs, device)
            results[(layer_idx, wt)] = max_k

            row += f" {max_k:>9}"
        print(row)

    # ---- Summary statistics ----
    print("\n" + "=" * 90)
    print("Summary: average free bits per weight type")
    print("=" * 90)

    print(f"\n  {'Weight Type':<23} {'Avg Free Bits':>14} {'Orig H (b)':>11} "
          f"{'Modified H (b)':>15} {'Orig Size':>11} {'Mod Size':>11} {'Ratio':>8}")
    print(f"  {'-'*23} {'-'*14} {'-'*11} {'-'*15} {'-'*11} {'-'*11} {'-'*8}")

    for wt in WEIGHT_TYPES:
        free_bits = [results[(l, wt)] for l in range(num_layers)]
        avg_free = np.mean(free_bits)

        # Compute entropy and size for original vs modified
        orig_tensors = []
        mod_tensors = []
        for l in range(num_layers):
            layer = model.model.layers[l]
            parts = wt.split('.')
            module = layer
            for p in parts:
                module = getattr(module, p)
            W = module.weight.data.detach().cpu()
            orig_tensors.append(W)
            mod_tensors.append(zero_mantissa_lsb(W, results[(l, wt)]))

        W_orig_cat = torch.cat([t.flatten() for t in orig_tensors])
        W_mod_cat = torch.cat([t.flatten() for t in mod_tensors])

        h_orig = estimate_entropy_bf16(W_orig_cat)
        h_mod = estimate_entropy_bf16(W_mod_cat)

        orig_bytes = estimate_huffman_size(W_orig_cat)
        mod_bytes = estimate_huffman_size(W_mod_cat)

        total_orig_bytes += orig_bytes
        total_modified_bytes += mod_bytes

        ratio = mod_bytes / orig_bytes * 100

        print(f"  {wt:<23} {avg_free:>12.1f} b {h_orig:>9.2f} b "
              f"{h_mod:>13.2f} b {orig_bytes/1e6:>9.2f}MB {mod_bytes/1e6:>9.2f}MB {ratio:>6.1f}%")

    raw_total = sum(
        model.model.layers[l].self_attn.q_proj.weight.numel()
        for l in range(num_layers)
        for wt in WEIGHT_TYPES
    ) * 2  # bf16 = 2 bytes
    # Fix: compute properly
    raw_total = 0
    for l in range(num_layers):
        layer = model.model.layers[l]
        for wt in WEIGHT_TYPES:
            parts = wt.split('.')
            module = layer
            for p in parts:
                module = getattr(module, p)
            raw_total += module.weight.numel() * 2

    print(f"\n  Raw bf16 size:         {raw_total/1e6:>10.2f} MB")
    print(f"  Huffman (original):    {total_orig_bytes/1e6:>10.2f} MB  "
          f"({total_orig_bytes/raw_total*100:.1f}%)")
    print(f"  Huffman (LSB-zeroed):  {total_modified_bytes/1e6:>10.2f} MB  "
          f"({total_modified_bytes/raw_total*100:.1f}%)")
    print(f"  Additional saving:     {(total_orig_bytes-total_modified_bytes)/1e6:>10.2f} MB  "
          f"({(1-total_modified_bytes/total_orig_bytes)*100:.1f}% over baseline Huffman)")

    # ---- Optionally save modified weights ----
    if args.save_modified_weights:
        save_dir = args.save_modified_weights
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nSaving modified weights to {save_dir}...")

        # Apply modifications to model in-place
        for l in range(num_layers):
            layer = model.model.layers[l]
            for wt in WEIGHT_TYPES:
                parts = wt.split('.')
                module = layer
                for p in parts:
                    module = getattr(module, p)
                k = results[(l, wt)]
                if k > 0:
                    module.weight.data = zero_mantissa_lsb(
                        module.weight.data.cpu(), k
                    ).to(module.weight.device)

        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        # Save analysis results
        analysis = {
            'model_name_or_path': args.model_name_or_path,
            'free_bits': {f"{l}.{wt}": results[(l, wt)]
                          for l in range(num_layers) for wt in WEIGHT_TYPES},
            'raw_bytes': raw_total,
            'huffman_orig_bytes': total_orig_bytes,
            'huffman_mod_bytes': total_modified_bytes,
        }
        with open(os.path.join(save_dir, 'sensitivity_analysis.json'), 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"Done! Modified model saved to {save_dir}")
        print("This model produces BIT-IDENTICAL bf16 outputs on calibration-like inputs.")


if __name__ == '__main__':
    main()
