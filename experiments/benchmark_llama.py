"""Benchmark multi-format compression on Llama-3.1-8B (or fallback models)."""
import sys
import os

# We need to import codec_multiformat directly, bypassing the broken __init__.py
# Temporarily fix the import by directly importing what benchmark_multiformat needs
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Patch new_compression to avoid broken __init__.py
import types
import importlib.util

nc_module = types.ModuleType('new_compression')
nc_module.__path__ = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'new_compression')]
nc_module.__package__ = 'new_compression'
sys.modules['new_compression'] = nc_module

# Now load codec_multiformat
spec = importlib.util.spec_from_file_location(
    'new_compression.codec_multiformat',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'new_compression', 'codec_multiformat.py')
)
codec_mod = importlib.util.module_from_spec(spec)
sys.modules['new_compression.codec_multiformat'] = codec_mod
spec.loader.exec_module(codec_mod)

from benchmark_multiformat import benchmark_bf16_model, benchmark_fp8_model, benchmark_simulated_int4

results = {}

models_to_try = [
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.2-3B",
    "mistralai/Mistral-7B-v0.3",
]

chosen_model = None
for model_name in models_to_try:
    try:
        print(f"Trying {model_name}...")
        benchmark_bf16_model(model_name, results)
        chosen_model = model_name
        break
    except Exception as e:
        print(f"  Failed: {e}\n  Trying next model...")

if chosen_model is None:
    print("ERROR: All models failed to load.")
    sys.exit(1)

benchmark_fp8_model(chosen_model, results)
benchmark_simulated_int4(chosen_model, results)

# Print summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
for key, r in results.items():
    print(f"{key}: {r['ratio']:.2f}% ({r['original_mb']:.1f}MB -> {r['compressed_mb']:.1f}MB)")
