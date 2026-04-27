# Paper Review Result

## Technical Limitations or Concerns

- The compression ratio ceiling for BF16 is inherently modest, around `1.3x` in typical cases. This limits benefits where interconnects are fast, such as NVLink/NVSwitch, or where contexts are short.
- The escape metadata format uses 32-bit absolute positions. In worst-case or elevated-escape regimes, metadata can dominate. Chunk-local or delta-encoded positions might further reduce overhead while remaining GPU-friendly.
- The encode path requires an additional escape-collection pass. Details on guaranteeing lock-free or low-contention compaction at scale are not fully fleshed out.

## Experimental Gaps or Methodological Issues

- The "end-to-end" KV transfer results are composed from component measurements and analytical pipeline models rather than measured on a live multi-node system with NICs and real RDMA/RoCE stacks. This can omit driver, NIC, protocol overheads, and the effects of kernel launch and overlap in real serving pipelines.
- Baseline coverage is limited. nvCOMP Cascaded/Bitcomp or other GPU entropy coders, as well as recent GPU-centric lossless float compressors, are not included. Only CPU-based ZipNN/DFloat11 and GPU LZ4 are compared.
- Evaluation is limited to batch size 1 and two model families for transfer. Sensitivity to batch size, heterogeneous GPUs, and other serving regimes is not shown.
- No integration into production serving frameworks, such as vLLM or SGLang, is demonstrated. Token-level impacts, such as TTFT or steady-state throughput, are not reported in a full PD-disaggregated deployment.

## Clarity or Presentation Issues

- "CPU-RDMA" and the precise transport setup are underspecified. It is unclear whether NICs were present or whether timings are bandwidth-model based.
- Some references appear incomplete, such as "various" placeholders, or have inconsistent years. One DistServe citation is a placeholder in the related work.
- Code is not yet released. The promise of future open-sourcing reduces immediate reproducibility.

## Missing Related Work or Comparisons

- Related GPU-centric lossless float compressors and fused execution works, such as ZipServ, TCA-TBE, and Falcon, are highly relevant to the discussion of fixed-length, SIMT-friendly decoding but are not discussed or compared.
- Systems that mitigate KV transfer via software techniques, such as FlowKV, HybridServe, and KVPR, are mentioned but not empirically combined with SplitZip to demonstrate complementarity.

## Detailed Comments

### Technical Soundness Evaluation

- The bit-level transformations for BF16, including extracting the exponent via `(x >> 7) & 0xff`, packing sign and mantissa into one byte, and reconstructing with shifts/ORs, are correct for the canonical BF16 layout and enable a simple lossless round trip.
- The analytical size model `B_SZ = N(1.5 + 5epsilon)` bytes and the implied ratio `rho = 2 / (1.5 + 5epsilon)` are sound. Given the reported `epsilon ~= 0.3-0.5%` typical escape rate, the ratios align with plots and tables.
- The Top-16 versus Top-15 sentinel ablation convincingly argues that uniform dense decoding plus sparse correction is materially faster on GPUs than interspersed sentinel handling, despite slightly larger metadata.
- The pipeline hide-threshold derivation `B_hide = min(G_enc, G_dec) / rho` is a useful rule of thumb. With reported throughputs, it places the method in a favorable regime for common interconnects. This aligns with the observed long-context benefits and short-context penalties.

### Experimental Evaluation Assessment

- Codec microbenchmarks are compelling. Encode throughput of `332.5 GB/s` and decode throughput of `1232.1 GB/s` are strong results for the stated hardware. Reporting uncompressed-byte throughput is a fair convention here.
- The end-to-end "transfer" evaluation is modeled rather than measured on a real interconnect path. Without actual NIC, kernel, and stack effects, speedups may differ in practice. Transfer consolidation, such as FlowKV-style consolidation, and concurrent workloads could also interact with codec overlap.
- Ablations are thorough and highlight important engineering choices: 3-bit versus 4-bit packing, GPU alignment, sentinel versus explicit positions, decode regularity, calibration granularity, codebook locality versus throughput, and dynamic calibration overheads.
- FP8 results sensibly show smaller gains and the cost of non-byte-aligned 3-bit packing. The E5M2/E4M3 differences are informative.

### Comparison With Related Work

- SplitZip stands apart from lossy KV compression methods such as PackKV, which achieve much larger ratios by accepting quantization error and fusing decompression with compute. SplitZip is strictly lossless and therefore complementary.
- Compared with DFloat11 and ZipNN, which are weights-focused and Huffman-based, SplitZip's fixed-length exponent coding removes parse irregularity and large LUTs, driving much higher GPU throughput on dynamic activation/KV workloads. This is an appropriate design for online paths where encode time matters.
- FlowKV shows that transfer latency can be dramatically reduced by restructuring KV payloads and transfers. SplitZip is orthogonal because it reduces bytes, and a combined evaluation would be valuable in future work.
- GPU lossless float compressors such as Falcon and weight-focused ZipServ demonstrate that GPU-friendly, fixed-structure encodings can outpace entropy-coded pipelines. Citing and contrasting these systems would further contextualize SplitZip's design decisions.

### Broader Impact and Significance

- The method directly targets a practical bottleneck in PD-disaggregated serving and can yield material cost, latency, and energy reductions for long-context workloads without changing model behavior.
- As with most performance improvements, broader availability could lower the barrier to deploying high-context LLMs. This has positive effects, such as efficiency and accessibility, and potential risks, such as misuse or increased overall compute consumption. The paper's brief acknowledgement of this is appropriate.

## Questions for Authors

1. How are escape positions compacted on GPU at scale? Please describe the parallel algorithm, such as warp ballot plus prefix-sum or two-pass counting and scatter, and explain how atomics/contention are avoided for tens of millions of elements.
2. Are escape positions stored as absolute 32-bit indices across the entire tensor? Could a chunked format with 16-bit intra-chunk offsets and a per-chunk base reduce metadata without harming decode regularity?
3. The "end-to-end" transfer results are composed from component timings and bandwidth models. Do you have measurements from a real multi-node setup with RoCE/RDMA NICs and an integrated serving pipeline, such as vLLM or SGLang, to validate that modeled speedups translate to practice?
4. Did you evaluate nvCOMP Cascaded or Bitcomp, or other GPU-oriented entropy coders, as baselines? If not, can you comment on expected performance relative to these for BF16 activations?
5. How stable is the codebook across layers and between K versus V caches? A per-layer histogram could reveal whether certain layers or cache types exhibit heavier tails that increase escape rates.
6. For FP8, have you considered chunk-local Top-8 versus global Top-8 to trade a small per-chunk header for potentially higher coverage without sacrificing decode regularity?
7. What is the behavior on pathological inputs, such as adversarial exponent distributions or many NaN/Inf values, and how does throughput degrade as escape rate grows?
8. Can SplitZip be combined with transfer consolidation, such as FlowKV-like techniques, and/or fused decode with attention kernels to further reduce overhead or avoid additional staging copies?
