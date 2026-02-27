"""
Compress LLM models (Qwen3, Qwen2.5, Llama series) using DFloat11 lossless compression.

Supported architectures:
  - Llama:   meta-llama/Llama-3.1-8B, meta-llama/Llama-3.1-8B-Instruct,
             meta-llama/Llama-3.2-3B, meta-llama/Llama-3.2-1B, etc.
  - Qwen2.5: Qwen/Qwen2.5-7B, Qwen/Qwen2.5-7B-Instruct,
             Qwen/Qwen2.5-14B, Qwen/Qwen2.5-3B, etc.
  - Qwen3:   Qwen/Qwen3-8B, Qwen/Qwen3-4B, Qwen/Qwen3-1.7B, etc.

Usage:
  python compress_llm.py --model_name_or_path Qwen/Qwen2.5-7B --save_path ./Qwen2.5-7B-DF11
  python compress_llm.py --model_name_or_path meta-llama/Llama-3.1-8B --save_path ./Llama-3.1-8B-DF11
  python compress_llm.py --model_name_or_path Qwen/Qwen3-8B --save_path ./Qwen3-8B-DF11
"""

from argparse import ArgumentParser

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from dfloat11 import compress_model


# All supported LLM architectures share the same decoder layer structure:
#   model.layers.<N>.self_attn.{q,k,v,o}_proj  (Linear)
#   model.layers.<N>.mlp.{gate,up,down}_proj    (Linear)
#
# The pattern_dict tells compress_model which modules to compress and
# how to locate the weight tensors within each matched module.
LLM_PATTERN_DICT = {
    r"model\.layers\.\d+": (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ),
}

SUPPORTED_ARCHITECTURES = [
    "LlamaForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
]


def main():
    parser = ArgumentParser("Compress LLM models using DFloat11")
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        required=True,
        help='HuggingFace model name or local path (e.g. Qwen/Qwen2.5-7B, meta-llama/Llama-3.1-8B)',
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default=None,
        help='Path to save the compressed model (default: ./<model_name>-DF11)',
    )
    parser.add_argument(
        '--save_single_file',
        action='store_true',
        help='Save the compressed model as a single .safetensors file',
    )
    parser.add_argument(
        '--check_correctness',
        action='store_true',
        help='Verify decompressed weights match originals during compression',
    )
    parser.add_argument(
        '--block_range',
        type=int,
        nargs=2,
        default=(0, 10000),
        help='Range of decoder layers to compress (for parallel compression across CPU cores)',
    )
    args = parser.parse_args()

    # Default save path
    if args.save_path is None:
        model_short_name = args.model_name_or_path.rstrip('/').split('/')[-1]
        args.save_path = f'./{model_short_name}-DF11'

    # Validate architecture
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    arch_list = getattr(config, 'architectures', []) or []
    arch = arch_list[0] if arch_list else 'Unknown'

    if arch not in SUPPORTED_ARCHITECTURES:
        print(f"Warning: architecture '{arch}' has not been tested. "
              f"Supported: {SUPPORTED_ARCHITECTURES}. Proceeding anyway...")

    num_layers = getattr(config, 'num_hidden_layers', '?')
    print(f"Model:        {args.model_name_or_path}")
    print(f"Architecture: {arch}")
    print(f"Layers:       {num_layers}")
    print(f"Save path:    {args.save_path}")
    print(f"Block range:  {args.block_range}")
    print()

    # Load model in bfloat16
    print("Loading model in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )

    # Compress
    print("Starting DFloat11 compression...")
    compress_model(
        model=model,
        pattern_dict=LLM_PATTERN_DICT,
        save_path=args.save_path,
        save_single_file=args.save_single_file,
        check_correctness=args.check_correctness,
        block_range=args.block_range,
    )

    print(f"\nCompression complete! Saved to: {args.save_path}")


if __name__ == '__main__':
    main()
