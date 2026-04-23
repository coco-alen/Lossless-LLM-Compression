
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

### BF16 + FP8 follow-up optimization (2026-04-21, RTX 6000 Ada, `yipin_quant`)

**BF16 lossless escape collection moved from PyTorch nonzero to vectorized Triton compaction:**
- File: `experiments/splitzip/lossless_fast.py`
- 268 MB synthetic BF16 layer, bit-exact reconstruction
- Before this change in the same environment: encode 121 GB/s, decode 472 GB/s
- After this change: encode 257 GB/s, decode 471 GB/s
- Ratio remains 1.333x, escapes 22,937 / layer (0.0163%)
- Projected Llama-70B 64K pipeline speedup:
  - 87 GB/s: 1.321x
  - 190 GB/s: 1.308x
- Key impact: high-bandwidth BF16 path is no longer encode-limited on this GPU.

**FP8 profiling and practical codec experiments:**
- `fp8_kv_profile.py` on Qwen/Qwen2.5-1.5B:
  - E4M3 entropy profile: 1.191x extra over native FP8, 2.381x total vs BF16
  - E5M2 entropy profile: 1.390x extra over native FP8, 2.779x total vs BF16
- `fp8_fixed_codec_bench.py` near-lossless top-8 diagnostic only:
  - E5M2 synthetic: 1.333x over FP8, 2.667x vs BF16, but changes ~1.18% of FP8 bytes
  - E5M2 tiled Qwen KV: 1.333x over FP8, 2.650x vs BF16, but changes ~4.30% of FP8 bytes
- Earlier no-escape logit probe on Qwen/Qwen2.5-1.5B:
  - E5M2 top-8 frequency code: near-vs-FP8 logit RMSE 3.005, argmax match 2/4
  - E5M2 top-16 frequency code: near-vs-FP8 logit RMSE 2.428, argmax match 1/4
  - Conclusion: no-escape near-lossless FP8 is not yet quality-safe.
- Earlier exact E5M2 top-15 + vectorized escape compaction trial:
  - 134 MB tiled Qwen KV, exact native-FP8 reconstruction
  - Coverage 99.522%, escapes 672,218 (0.4784%)
  - Ratio 1.112x over FP8, 2.225x total vs BF16
  - Encode core 433 GB/s, escape collect 273-274 GB/s with collect block 128
  - Full encode 178-179 GB/s, full decode 405-414 GB/s
  - Projected speedup at 87 GB/s: 1.102x vs raw FP8, 2.203x vs raw BF16
- `fp8_e5m2_top8_compact_bench.py` exact E5M2 top-8 + compact block-local escapes:
  - 134 MB tiled Qwen KV, exact native-FP8 reconstruction
  - Top-8 coverage 95.705%, escapes 6,034,886 (4.2950%), max 125 escapes / 256-element block
  - After packing escaped 5-bit exponent values: ratio 1.214x over FP8, 2.428x total vs BF16
  - Full encode 135 GB/s, full decode 233 GB/s
  - Projected speedup at 87 GB/s: 1.195x vs raw FP8, 2.391x vs raw BF16
  - `--escape-block 128` is slower; `--escape-block > 256` is invalid for uint8 local offsets
- E4M3 mantissa factoring check on Qwen/Qwen2.5-1.5B generated KV, 4 prompts:
  - Average full-byte entropy: 6.837 bits
  - Average exponent entropy: 2.886 bits
  - Average mantissa entropy: 2.978 / 3 bits
  - Average sign+mantissa entropy: 3.978 / 4 bits
  - Average top-8 exponent coverage: 95.628%
  - Initial unpacked-escape top-8 model before compact escape packing: ~1.04x over FP8
  - Ideal exponent-only ratio: 1.162x over FP8
  - Ideal exponent + mantissa + raw sign ratio: 1.166x over FP8
  - Ideal full-byte entropy ceiling: 1.170x over FP8
  - Conclusion: E4M3 mantissa is nearly uniform, so factoring mantissa adds
    only ~0.004x over separate exponent coding and ~0.008x versus the current
    exponent-only entropy path. It is unlikely to pay for GPU-side complexity.
- E4M3 exact top-8 compact escape trial using `fp8_e5m2_top8_compact_bench.py --fmt e4m3`:
  - 134 MB tiled Qwen KV, exact native-FP8 reconstruction
  - Top-8 exponent coverage 95.638%, escapes 6,129,371 (4.3623%), max 126 escapes / 256-element block
  - After packing escaped 4-bit exponent values: ratio 1.059x over FP8, 2.118x total vs BF16
  - Correctness: PASS
  - Conclusion: fixed 3-bit exponent coding is lossless and general, but
    escape overhead consumes most of the nominal 8/7 gain for E4M3.
  - Timing note: later FP8 throughput runs on 2026-04-21 were affected by an
    unrelated `Retrial_Agent.py` process using the GPU heavily; rerun speed
    measurements on an idle GPU before quoting final GB/s.
- Inline variable-length top-7 ratio model on Qwen/Qwen2.5-1.5B generated KV:
  - E4M3: top-7 coverage 93.644%, modeled 7.254 bits/elem, 1.103x over FP8
  - E5M2: top-7 coverage 93.526%, modeled 6.324 bits/elem, 1.265x over FP8
  - This removes explicit escape positions but makes GPU decode sequential
    within each block unless we add extra indexing metadata.
- Canonical Huffman exponent-only lossless target on the same Qwen KV sample:
  - E4M3: exponent entropy 2.877 bits, Huffman average 2.920 bits,
    modeled ratio 1.156x over FP8
  - E5M2: exponent entropy 2.916 bits, Huffman average 2.954 bits,
    modeled ratio 1.344x over FP8
  - This is the strongest lossless FP8 target observed so far, but requires
    an efficient GPU bitstream decoder or block indexing strategy.

**Next target:** lossless FP8 ratio. The compact top-8 E5M2 path improves exact
ratio to 1.214x over FP8, but decode and the count/cumsum/write escape path are
now the speed bottlenecks. A GPU-friendly entropy code is still needed to approach
the 1.39x E5M2 entropy-profile ceiling. For E4M3, practical lossless gains likely
require entropy coding the exponent stream directly; fixed 3-bit + compact escapes
only reaches 1.059x on the tiled Qwen sample, and mantissa coding is low priority
because the mantissa is almost full entropy.

### Mooncake local-loopback format latency (2026-04-21, `yipin_quant`)

Installed Mooncake Transfer Engine 0.3.10.post1 plus `rdma-core` and `etcd`
inside `yipin_quant`, started a local etcd metadata server, and measured real
Mooncake TCP loopback transfer times with `experiments/splitzip/mooncake_format_latency.py`.
No RDMA devices were found, so the transport is TCP.

Measured Mooncake transfer medians:
- 402.7 MB: 49.60 ms (8.12 GB/s)
- 309.5 MB: 38.07 ms (8.13 GB/s)
- 201.3 MB: 25.06 ms (8.03 GB/s)
- 190.1 MB: 24.51 ms (7.76 GB/s)
- 165.9 MB: 20.17 ms (8.22 GB/s)

Full encode-transfer-decode latency using current lossless codec speeds:
- BF16 + SplitZip: raw 49.60 ms, full 40.49 ms, 1.225x vs raw BF16
- E4M3 + SplitZip: raw native FP8 25.06 ms, full 26.96 ms,
  0.929x vs raw E4M3, 1.840x vs raw BF16
- E5M2 + SplitZip: raw native FP8 25.06 ms, full 22.55 ms,
  1.111x vs raw E5M2, 2.200x vs raw BF16

Interpretation: on fast local loopback, E4M3's exact top-8 compact ratio
(1.059x) is too small to overcome encode/decode overhead, while E5M2's exact
ratio (1.214x) still produces a native-FP8 full-path speedup. BF16 remains
positive with the optimized escape compaction path.
