# Paper Plan: SplitZip

**Title**: SplitZip: Lossless KV Cache Compression for Disaggregated Prefill-Decode LLM Serving
**Venue**: MLSys / EuroSys
**Pages**: 10-12

## Section Structure

1. Introduction (1.25 pages)
2. Background and Motivation (0.9 pages)
   - 2.1 Disaggregated PD Serving
   - 2.2 Why BF16 KV Caches Are Hard to Compress
   - 2.3 Empirical Exponent Concentration
3. Design Overview (0.7 pages)
4. SplitZip Codec (1.6 pages)
   - 4.1 BF16 Exponent Coding
   - 4.2 Escape Stream
   - 4.3 3-Bit Near-Lossless Variant
   - 4.4 Correctness Properties
5. GPU Implementation and System Integration (1.4 pages)
   - 5.1 Triton Kernels
   - 5.2 Mooncake Integration
   - 5.3 Layer-Pipelined Transfer
6. Evaluation (0.7 pages methodology + 2.6 pages results)
7. Discussion (0.8 pages)
8. Related Work (0.8 pages)
9. Conclusion (0.25 pages)

## Figures
- Fig 1: PD serving architecture + SplitZip insertion point
- Fig 2: BF16 exponent concentration across models
- Fig 3: Codec format schematic
- Fig 4: Layer-pipelined transfer timeline
- Fig 5: Codec microbenchmarks
- Fig 6: Mooncake transfer speedup
- Fig 7: End-to-end pipeline gains
- Fig 8: Correctness over prompts and context lengths
- Table 1: Main results summary
- Table 2: Experimental setup
- Table 3: Cross-family exponent statistics
- Table 4: Baseline comparison
