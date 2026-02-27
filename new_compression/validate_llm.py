"""
Validate that the new compression is lossless by decompressing and comparing
with the original model weights bit-for-bit.

Usage:
    python -m new_compression.validate_llm \
        --model_name_or_path Qwen/Qwen3-1.7B \
        --compressed_path ./Qwen3-1.7B-NewComp
"""

import json
import os
import pickle
from argparse import ArgumentParser

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from .codec import decompress_weight_group

WEIGHT_TYPES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def main():
    parser = ArgumentParser("Validate lossless compression")
    parser.add_argument('--model_name_or_path', type=str, required=True,
                        help='Original bf16 model for comparison')
    parser.add_argument('--compressed_path', type=str, required=True,
                        help='Path to compressed output directory')
    args = parser.parse_args()

    # Load metadata
    with open(os.path.join(args.compressed_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    print(f"Compressed model: {metadata['model_name_or_path']}")
    print(f"Compression ratio: {metadata['compression_ratio']:.1f}%")
    print()

    # Load compressed data
    with open(os.path.join(args.compressed_path, 'compressed_weights.pkl'), 'rb') as f:
        all_compressed = pickle.load(f)

    # Load original model
    print(f"Loading original model: {args.model_name_or_path}")
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    num_layers = config.num_hidden_layers

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16,
    )
    model.eval()

    # Validate each weight group
    print(f"\n{'Weight Type':<25} {'Layers':>7} {'Status':>10}")
    print("-" * 50)

    all_pass = True
    for wt in WEIGHT_TYPES:
        # Get original weights
        originals = []
        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]
            parts = wt.split('.')
            module = layer
            for p in parts:
                module = getattr(module, p)
            originals.append(module.weight.data.detach().cpu())

        # Decompress
        decompressed = decompress_weight_group(all_compressed[wt])

        # Compare
        match = True
        for layer_idx, (orig, decomp) in enumerate(zip(originals, decompressed)):
            if not torch.equal(orig, decomp):
                print(f"  {wt:<23} layer {layer_idx:>3}   FAIL")
                max_diff = (orig.view(torch.int16) - decomp.view(torch.int16)).abs().max().item()
                print(f"    Max int16 diff: {max_diff}")
                match = False
                all_pass = False

        status = "PASS" if match else "FAIL"
        print(f"  {wt:<23} {num_layers:>5}   {status:>8}")

    print("-" * 50)
    if all_pass:
        print("ALL CHECKS PASSED — compression is bit-identical lossless!")
    else:
        print("SOME CHECKS FAILED — see details above.")


if __name__ == '__main__':
    main()
