"""
Validate DFloat11-compressed LLM by comparing outputs with the original bfloat16 model.

This script:
  1. Loads the original bf16 model and generates reference output
  2. Loads the DFloat11 compressed model and generates output
  3. Compares logits to verify lossless compression

Usage:
  python validate_llm.py \
      --original_model Qwen/Qwen2.5-7B \
      --compressed_model ./Qwen2.5-7B-DF11

  python validate_llm.py \
      --original_model meta-llama/Llama-3.1-8B \
      --compressed_model ./Llama-3.1-8B-DF11

  python validate_llm.py \
      --compressed_model ./Qwen3-8B-DF11 \
      --skip_comparison
"""

import time
from argparse import ArgumentParser

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from dfloat11 import DFloat11Model


def run_inference(model, tokenizer, prompt, max_new_tokens, device):
    """Run model.generate and return generated text + logits."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for reproducibility
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = outputs.sequences
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # Stack all per-step logits: (num_new_tokens, vocab_size)
    scores = torch.stack(outputs.scores, dim=0)
    return text, scores


def main():
    parser = ArgumentParser("Validate DFloat11 compressed LLM")
    parser.add_argument(
        '--original_model',
        type=str,
        default=None,
        help='Original bf16 model name/path for comparison (e.g. Qwen/Qwen2.5-7B)',
    )
    parser.add_argument(
        '--compressed_model',
        type=str,
        required=True,
        help='Path to the DFloat11 compressed model',
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='Question: What is a binary tree? Answer:',
        help='Prompt to use for generation',
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=50,
        help='Number of tokens to generate',
    )
    parser.add_argument(
        '--skip_comparison',
        action='store_true',
        help='Skip bf16 comparison, only test that compressed model runs',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    args = parser.parse_args()

    set_seed(args.seed)

    # ---- Load compressed model ----
    print(f"Loading DFloat11 compressed model from: {args.compressed_model}")
    t0 = time.time()
    compressed_model = DFloat11Model.from_pretrained(
        args.compressed_model,
        device_map="auto",
    )
    print(f"Compressed model loaded in {time.time() - t0:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(args.original_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Warm-up
    warmup_inputs = tokenizer("hello", return_tensors="pt").to(compressed_model.device)
    with torch.no_grad():
        compressed_model(**warmup_inputs, use_cache=False)
    del warmup_inputs

    # Generate with compressed model
    print(f"\nGenerating with compressed model (prompt: '{args.prompt}')...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    compressed_text, compressed_scores = run_inference(
        compressed_model, tokenizer, args.prompt, args.max_new_tokens, compressed_model.device,
    )
    torch.cuda.synchronize()
    compressed_time = time.time() - t0

    peak_mem = sum(
        torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
    ) / 1e9

    print(f"  Output: {compressed_text}")
    print(f"  Time:   {compressed_time:.2f}s")
    print(f"  Peak GPU memory: {peak_mem:.2f} GB")

    # Free compressed model
    del compressed_model
    torch.cuda.empty_cache()

    if args.skip_comparison or args.original_model is None:
        print("\nSkipping comparison with original model.")
        print("DFloat11 compressed model runs successfully!")
        return

    # ---- Load original bf16 model ----
    print(f"\nLoading original bf16 model: {args.original_model}")
    t0 = time.time()
    original_model = AutoModelForCausalLM.from_pretrained(
        args.original_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Original model loaded in {time.time() - t0:.1f}s")

    original_tokenizer = AutoTokenizer.from_pretrained(args.original_model)
    if original_tokenizer.pad_token is None:
        original_tokenizer.pad_token = original_tokenizer.eos_token

    # Generate with original model
    print(f"\nGenerating with original bf16 model...")
    t0 = time.time()
    original_text, original_scores = run_inference(
        original_model, original_tokenizer, args.prompt, args.max_new_tokens, original_model.device,
    )
    torch.cuda.synchronize()
    original_time = time.time() - t0

    print(f"  Output: {original_text}")
    print(f"  Time:   {original_time:.2f}s")

    # ---- Compare outputs ----
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    text_match = compressed_text == original_text
    print(f"Text match:   {'PASS' if text_match else 'FAIL'}")

    # Compare logits (move to same device)
    min_len = min(compressed_scores.shape[0], original_scores.shape[0])
    cs = compressed_scores[:min_len].cpu().float()
    os_ = original_scores[:min_len].cpu().float()

    logits_match = torch.equal(cs, os_)
    max_diff = (cs - os_).abs().max().item()
    mean_diff = (cs - os_).abs().mean().item()

    print(f"Logits match: {'PASS' if logits_match else 'FAIL'}")
    print(f"Max logit diff:  {max_diff:.6e}")
    print(f"Mean logit diff: {mean_diff:.6e}")

    if text_match and logits_match:
        print("\nDFloat11 compression is LOSSLESS - outputs are bit-identical!")
    elif text_match:
        print("\nGenerated text matches. Minor logit differences may be due to "
              "floating-point ordering on different devices.")
    else:
        print("\nWARNING: Generated text differs. Check logit diffs above.")


if __name__ == '__main__':
    main()
