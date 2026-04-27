We now need to address these issues in the v2 version of the paper. However, given that the GPUs are currently occupied, I suggest you first write and thoroughly test the code to resolve the following issues within a new folder; once the GPUs become available, we can generate all the results simultaneously. Please ensure the code is clean and modularized so that I can easily reproduce the results.

- Regarding 32-bit absolute value encoding: I recommend employing a "chunk-local" encoding approach. This should improve parallel efficiency and prevent potential overflow issues. Additionally, we need to conduct a breakdown analysis of the various operational stages within the encoding and decoding pipelines.

- "The encode path requires an additional escape-collection pass. Details on guaranteeing lock-free or low-contention compaction at scale are not fully fleshed out." I do not fully understand this specific point. Could you please either explain it to me, resolve the issue directly within the new implementation, or draft a brief explanatory paragraph that I can incorporate into the paper?

- The current experiments were indeed conducted via simulation. I recommend we update the experimental results for `llama3-8b` and `qwen3-30b-a3b` by utilizing "Mooncake" to transmit actual KV values. We should conduct experiments with a fixed batch size (BS) of 1 and sequence lengths ranging from 512 to 131,072; experiments with a fixed BS of 16 and sequence lengths ranging from 128 to 65,536; and finally, experiments with a fixed sequence length of 1,024 (varying BS from 1 to 256) and a fixed sequence length of 32,768 (varying BS from 1 to 128).

- Regarding additional baselines: Please incorporate data for Cascaded/Bitcomp, ZipServ, TCA-TBE, and Falcon.

- Regarding heterogeneous architectures: I suggest adding results for `Qwen3-next` to Table 1 to demonstrate that our proposed method remains effective within heterogeneous attention contexts.

- Integrate the current compression mechanism into `sglang`. Run `Qwen3-32b` using the same experimental configurations as mentioned above: fixed BS=1 with sequence lengths from 512 to 131,072; fixed BS=16 with sequence lengths from 128 to 65,536; and fixed sequence lengths of 1,024 (BS 1–256) and 32,768 (BS 1–128). Please report the observed changes in TTFT (Time-to-First-Token) and steady-state throughput. - “Systems that mitigate KV transfer via software techniques—such as FlowKV, HybridServe, and KVPR—are mentioned but not empirically combined with SplitZip to demonstrate complementarity.” Could you attempt a comparative analysis, or integrate our method into their techniques, to demonstrate that our optimization approach is orthogonal and that the performance gains are additive?

- “How stable is the codebook across layers and between K versus V caches? A per-layer histogram could reveal whether certain layers or cache types exhibit heavier tails that increase escape rates.” We suggest conducting an experiment on Qwen3-32b to investigate this.

- “For FP8, have you considered chunk-local Top-8 versus global Top-8 to trade a small per-chunk header for potentially higher coverage without sacrificing decode regularity?” We suggest adding an experiment to explore this.

- Finally, please address the specific issues raised by the reviewer.