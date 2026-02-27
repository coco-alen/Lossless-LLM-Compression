"""
Compress an LLM using Predictive Coding + Cross-Layer Delta Compression.

Groups same-type weight matrices across all decoder layers, applies
cross-layer delta encoding followed by spatial predictive coding,
then Huffman-encodes the residuals.

Usage:
    python -m new_compression.compress_llm \
        --model_name_or_path Qwen/Qwen3-1.7B \
        --save_path ./Qwen3-1.7B-NewComp

    python -m new_compression.compress_llm \
        --model_name_or_path meta-llama/Llama-3.1-8B \
        --save_path ./Llama-3.1-8B-NewComp
"""

import json
import os
import pickle
from argparse import ArgumentParser

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from tqdm import tqdm

from codec import (
    compress_weight_group,
    compute_ratio,
    compute_dfloat11_baseline_ratio,
)

# Weight sub-modules inside each decoder layer to compress
WEIGHT_TYPES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def extract_weight_groups(model, num_layers: int) -> dict[str, list[torch.Tensor]]:
    """
    Extract weight matrices grouped by type across layers.

    Returns e.g.:
        {"self_attn.q_proj": [layer0_q.weight, layer1_q.weight, ...], ...}
    """
    groups = {wt: [] for wt in WEIGHT_TYPES}

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        for wt in WEIGHT_TYPES:
            parts = wt.split('.')
            module = layer
            for p in parts:
                module = getattr(module, p)
            assert module.weight.dtype == torch.bfloat16, (
                f"Expected bf16, got {module.weight.dtype} at layer {layer_idx} {wt}"
            )
            groups[wt].append(module.weight.data.detach().cpu())

    return groups


def main():
    parser = ArgumentParser("Compress LLM with Predictive + Cross-Layer Delta Coding")
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--no_cross_layer', action='store_true',
                        help='Disable cross-layer delta encoding')
    parser.add_argument('--no_prediction', action='store_true',
                        help='Disable intra-matrix predictive coding')
    args = parser.parse_args()

    if args.save_path is None:
        name = args.model_name_or_path.rstrip('/').split('/')[-1]
        args.save_path = f'./{name}-NewComp'

    # Load config
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    num_layers = config.num_hidden_layers
    arch = (getattr(config, 'architectures', None) or ['Unknown'])[0]
    print(f"Model:        {args.model_name_or_path}")
    print(f"Architecture: {arch}")
    print(f"Layers:       {num_layers}")
    print(f"Cross-layer:  {not args.no_cross_layer}")
    print(f"Prediction:   {not args.no_prediction}")
    print()

    # Load model
    print("Loading model in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16,
    )
    model.eval()

    # Extract weight groups
    print("Extracting weight groups...")
    groups = extract_weight_groups(model, num_layers)

    # Compress each group
    os.makedirs(args.save_path, exist_ok=True)
    total_original = 0
    total_compressed = 0
    all_compressed = {}

    print("\nCompressing weight groups:")
    print(f"{'Weight Type':<25} {'Original MB':>12} {'Compressed MB':>14} {'Ratio':>8} {'DF11 est.':>10}")
    print("-" * 75)

    for wt in tqdm(WEIGHT_TYPES, desc="Compressing"):
        tensors = groups[wt]
        original_bytes = sum(t.numel() * 2 for t in tensors)

        compressed = compress_weight_group(
            tensors,
            use_cross_layer=not args.no_cross_layer,
            use_prediction=not args.no_prediction,
        )
        all_compressed[wt] = compressed

        ratio = compute_ratio(compressed, original_bytes)
        df11_ratio = compute_dfloat11_baseline_ratio(tensors)

        comp_bytes = len(compressed['encoded_high']) + len(compressed['low_bytes'])
        total_original += original_bytes
        total_compressed += comp_bytes

        print(f"  {wt:<23} {original_bytes/1e6:>10.2f}MB {comp_bytes/1e6:>12.2f}MB "
              f"{ratio:>6.1f}% {df11_ratio:>8.1f}%")

    total_ratio = total_compressed / total_original * 100
    print("-" * 75)
    print(f"  {'TOTAL':<23} {total_original/1e6:>10.2f}MB {total_compressed/1e6:>12.2f}MB "
          f"{total_ratio:>6.1f}%")
    print()

    # Save compressed data
    print(f"Saving to {args.save_path}...")
    with open(os.path.join(args.save_path, 'compressed_weights.pkl'), 'wb') as f:
        pickle.dump(all_compressed, f)

    # Save metadata
    metadata = {
        'model_name_or_path': args.model_name_or_path,
        'architecture': arch,
        'num_layers': num_layers,
        'weight_types': list(WEIGHT_TYPES),
        'use_cross_layer': not args.no_cross_layer,
        'use_prediction': not args.no_prediction,
        'original_bytes': total_original,
        'compressed_bytes': total_compressed,
        'compression_ratio': total_ratio,
    }
    with open(os.path.join(args.save_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Done! Compression ratio: {total_ratio:.1f}% of original size")


if __name__ == '__main__':
    main()
