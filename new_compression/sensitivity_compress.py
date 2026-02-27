"""
Output-Preserving Compression: break the weight-entropy barrier.

Key insight:
  Standard lossless compression treats weights as raw data — weight entropy
  is the hard compression limit.  But if modifying a weight's mantissa LSBs
  does NOT change the model's FINAL OUTPUT LOGITS, those bits are "free"
  and can be zeroed to lower entropy → better compression.

Why check final logits (not per-layer matmul output)?
  A small change in one layer's matmul result may be absorbed by:
    - Residual connections (adding back the unmodified residual stream)
    - RMSNorm / LayerNorm (re-scaling activations)
    - Subsequent layers (masking small perturbations)
  So per-layer checking is too conservative.  End-to-end logit checking
  finds MORE free bits because it accounts for the full model's tolerance.

Algorithm:
  1. Run calibration data, record reference logits (the ground truth).
  2. Process layers SEQUENTIALLY (layer 0, 1, …, L-1).
     For each layer, for each weight type (q_proj, k_proj, ...):
       - Binary search k ∈ [0, 7]: temporarily zero k mantissa LSBs,
         run FULL model forward pass, check if final logits are bit-identical.
       - Apply the maximum safe k PERMANENTLY before moving to next layer.
     Sequential ordering ensures cumulative effects are captured:
     earlier modifications are present when testing later layers.
  3. Report per-layer free bits, entropy reduction, and compression gain.

Usage:
    python -m new_compression.sensitivity_compress \
        --model_name_or_path Qwen/Qwen3-1.7B

    python -m new_compression.sensitivity_compress \
        --model_name_or_path Qwen/Qwen3-1.7B \
        --save_modified_weights ./Qwen3-1.7B-SensComp
"""

import os
import json
from argparse import ArgumentParser

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
    Zero the last k mantissa bits of a bf16 tensor (in-place friendly).

    BFloat16 layout: [1 sign | 8 exponent | 7 mantissa]
    k=0 → no change; k=7 → only sign+exponent remain.
    """
    if k <= 0:
        return weight_bf16
    raw = weight_bf16.view(torch.int16)
    mask = ~((1 << k) - 1)
    return (raw & mask).view(torch.bfloat16)


# ---------------------------------------------------------------------------
# End-to-end logit checking
# ---------------------------------------------------------------------------

def get_reference_logits(model, calib_token_ids, device):
    """Run full model forward on all calibration inputs, return logits (cpu)."""
    ref = []
    with torch.no_grad():
        for ids in calib_token_ids:
            logits = model(ids.to(device), use_cache=False).logits
            ref.append(logits.cpu())
    return ref


def check_logits_identical(model, calib_token_ids, ref_logits, device):
    """Return True if model's current logits are bit-identical to ref on all inputs."""
    with torch.no_grad():
        for i, ids in enumerate(calib_token_ids):
            logits = model(ids.to(device), use_cache=False).logits
            if not torch.equal(logits.cpu(), ref_logits[i]):
                return False
    return True


# ---------------------------------------------------------------------------
# Weight access helpers
# ---------------------------------------------------------------------------

def get_module(model, layer_idx, wt):
    """Navigate to a specific nn.Linear inside a decoder layer."""
    layer = model.model.layers[layer_idx]
    module = layer
    for p in wt.split('.'):
        module = getattr(module, p)
    return module


# ---------------------------------------------------------------------------
# Binary search: max free bits checked against final logits
# ---------------------------------------------------------------------------

def find_max_free_bits_logit(model, layer_idx, wt,
                             calib_token_ids, ref_logits, device):
    """
    Binary search for max k ∈ [0,7] mantissa LSBs that can be zeroed
    without changing the model's final logits.
    """
    module = get_module(model, layer_idx, wt)
    W_orig = module.weight.data.clone()

    lo, hi = 0, 7
    while lo < hi:
        mid = (lo + hi + 1) // 2
        # temporarily apply modification
        module.weight.data.copy_(zero_mantissa_lsb(W_orig, mid))
        if check_logits_identical(model, calib_token_ids, ref_logits, device):
            lo = mid      # safe, try more aggressive
        else:
            hi = mid - 1  # too aggressive

    # restore original (caller will apply permanently if desired)
    module.weight.data.copy_(W_orig)
    return lo


# ---------------------------------------------------------------------------
# Entropy / compression helpers
# ---------------------------------------------------------------------------

def estimate_entropy_bf16(weight_bf16: torch.Tensor) -> float:
    """Shannon entropy (bits/element) of bf16 viewed as int16."""
    raw = weight_bf16.contiguous().view(torch.int16).flatten()
    _, counts = torch.unique(raw, return_counts=True)
    probs = counts.float() / counts.sum()
    return -(probs * probs.log2()).sum().item()


def estimate_huffman_size(weight_bf16: torch.Tensor) -> int:
    """DFloat11-style estimate: Huffman(high byte) + raw(low byte)."""
    raw = weight_bf16.contiguous().view(torch.uint8)
    low_bytes = raw[0::2].numpy()
    high_bytes = raw[1::2].numpy()

    vals, counts = np.unique(high_bytes, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(vals, counts)}
    codec = HuffmanCodec.from_frequencies(freq)
    encoded = codec.encode(high_bytes.tolist())
    return len(encoded) + len(low_bytes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser("Output-Preserving Compression (logit-level)")
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen3-1.7B')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_modified_weights', type=str, default=None,
                        help='Save the LSB-zeroed model to this path')
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

    # ---- Tokenize calibration prompts ----
    calib_token_ids = []
    for prompt in CALIBRATION_PROMPTS:
        ids = tokenizer(
            prompt, return_tensors="pt",
            max_length=args.max_calib_tokens, truncation=True,
        ).input_ids
        calib_token_ids.append(ids)

    # ---- Compute reference logits ----
    print("Computing reference logits...")
    ref_logits = get_reference_logits(model, calib_token_ids, device)
    print(f"  {len(ref_logits)} prompts, logit shapes: "
          f"{[l.shape for l in ref_logits[:3]]}...")

    # ---- Sequential layer-by-layer analysis ----
    #
    # KEY: process in order.  After finding safe k for layer l, APPLY it
    # permanently, then update reference logits.  This way, when testing
    # layer l+1, all prior modifications are in effect → cumulative effects
    # are fully captured.
    #
    print("\nAnalyzing free mantissa bits (end-to-end logit check)...")
    print("  Binary search: ~3 full forward passes × 7 weight types per layer")
    print("=" * 90)

    header = f"  {'Layer':<8}"
    for wt in WEIGHT_TYPES:
        header += f" {wt.split('.')[-1]:>9}"
    print(header)
    print("-" * 90)

    results = {}  # (layer_idx, wt) → max_k

    for layer_idx in tqdm(range(num_layers), desc="Layers"):
        row = f"  {layer_idx:<8}"

        for wt in WEIGHT_TYPES:
            max_k = find_max_free_bits_logit(
                model, layer_idx, wt,
                calib_token_ids, ref_logits, device,
            )
            results[(layer_idx, wt)] = max_k

            # Apply permanently
            if max_k > 0:
                module = get_module(model, layer_idx, wt)
                module.weight.data.copy_(
                    zero_mantissa_lsb(module.weight.data, max_k)
                )

            row += f" {max_k:>9}"

        # Update reference logits (reflects all modifications up to this layer)
        ref_logits = get_reference_logits(model, calib_token_ids, device)
        tqdm.write(row)

    # ---- Reload original model for comparison ----
    print("\nReloading original model for compression comparison...")
    model_orig = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16,
    ).eval()

    # ---- Summary ----
    print("\n" + "=" * 100)
    print("Summary: free bits & compression gain per weight type")
    print("=" * 100)
    print(f"  {'Weight Type':<23} {'Avg Free':>9} {'Orig H':>9} {'Mod H':>9} "
          f"{'Orig Size':>11} {'Mod Size':>11} {'Saving':>8}")
    print(f"  {'-'*23} {'-'*9} {'-'*9} {'-'*9} {'-'*11} {'-'*11} {'-'*8}")

    total_orig_bytes = 0
    total_mod_bytes = 0
    raw_total = 0

    for wt in WEIGHT_TYPES:
        free_bits = [results[(l, wt)] for l in range(num_layers)]
        avg_free = np.mean(free_bits)

        orig_tensors = []
        mod_tensors = []

        for l in range(num_layers):
            W_orig = get_module(model_orig, l, wt).weight.data.cpu()
            W_mod = get_module(model, l, wt).weight.data.cpu()
            orig_tensors.append(W_orig.flatten())
            mod_tensors.append(W_mod.flatten())
            raw_total += W_orig.numel() * 2

        W_orig_cat = torch.cat(orig_tensors)
        W_mod_cat = torch.cat(mod_tensors)

        h_orig = estimate_entropy_bf16(W_orig_cat)
        h_mod = estimate_entropy_bf16(W_mod_cat)
        orig_bytes = estimate_huffman_size(W_orig_cat)
        mod_bytes = estimate_huffman_size(W_mod_cat)

        total_orig_bytes += orig_bytes
        total_mod_bytes += mod_bytes
        saving = (1 - mod_bytes / orig_bytes) * 100

        print(f"  {wt:<23} {avg_free:>7.1f} b {h_orig:>7.2f} b {h_mod:>7.2f} b "
              f"{orig_bytes/1e6:>9.2f}MB {mod_bytes/1e6:>9.2f}MB {saving:>6.1f}%")

    del model_orig

    print(f"\n  Raw bf16 size:            {raw_total/1e6:>10.2f} MB")
    print(f"  Huffman (original):       {total_orig_bytes/1e6:>10.2f} MB  "
          f"({total_orig_bytes/raw_total*100:.1f}% of raw)")
    print(f"  Huffman (logit-zeroed):   {total_mod_bytes/1e6:>10.2f} MB  "
          f"({total_mod_bytes/raw_total*100:.1f}% of raw)")
    print(f"  Additional saving:        {(total_orig_bytes - total_mod_bytes)/1e6:>10.2f} MB  "
          f"({(1 - total_mod_bytes/total_orig_bytes)*100:.1f}% beyond baseline Huffman)")

    # ---- Optionally save ----
    if args.save_modified_weights:
        save_dir = args.save_modified_weights
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nSaving modified model to {save_dir}...")

        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        analysis = {
            'model_name_or_path': args.model_name_or_path,
            'architecture': arch,
            'num_layers': num_layers,
            'method': 'logit-level LSB zeroing (sequential, cumulative)',
            'num_calib_prompts': len(CALIBRATION_PROMPTS),
            'free_bits': {
                f"layers.{l}.{wt}": results[(l, wt)]
                for l in range(num_layers) for wt in WEIGHT_TYPES
            },
            'raw_bytes': raw_total,
            'huffman_orig_bytes': total_orig_bytes,
            'huffman_mod_bytes': total_mod_bytes,
        }
        with open(os.path.join(save_dir, 'sensitivity_analysis.json'), 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"Done!  Modified model saved to: {save_dir}")
        print("Final logits are BIT-IDENTICAL to original on all calibration inputs.")


if __name__ == '__main__':
    main()
