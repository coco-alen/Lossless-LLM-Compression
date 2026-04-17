
### Real KV Evaluation on Qwen3-30B-A3B + OpenOrca (2026-04-08)

**First fully authentic evaluation:**
- Model: Qwen3-30B-A3B (MoE, 128 experts, 48 layers)
- Prompts: 20 real prompts from OpenOrca dataset (not synthetic)
- KV: 402.7 MB BF16, 4096 tokens
- Transfer: Real Mooncake Transfer Engine TCP

**Results on REAL data:**
- Compression ratio: 1.3010x (lower than 1.333x on randn due to higher entropy)
- Escape rate: 0.745% (vs 0.02% on randn — 37x more escapes)
- Exponent entropy: 3.52 bits (vs 2.55 on randn)
- Lossless: PASS ✓
- Encode: 147 GB/s, Decode: 881 GB/s

**Real Mooncake transfer (median of 20 runs, 402.7 MB):**
- Raw: 146.88 ms
- Compressed: 115.44 ms
- Transfer speedup: 1.272x
- Full pipeline (enc+xfer+dec): 1.238x end-to-end

**Honest update:** The previous 1.333x ratio was overly optimistic because it was
measured on randn data. Real MoE KV caches give 1.301x due to higher exponent
diversity and more escapes. The paper numbers should be updated accordingly.
