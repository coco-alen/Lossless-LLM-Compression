from __future__ import annotations

from dataclasses import dataclass


SEQ_SWEEP_BS1 = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
SEQ_SWEEP_BS16 = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
BS_SWEEP_SEQ1024 = [1, 2, 4, 8, 16, 32, 64, 128, 256]
BS_SWEEP_SEQ32768 = [1, 2, 4, 8, 16, 32, 64, 128]


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    hf_name: str
    family: str


TRANSFER_MODELS = [
    ModelSpec("Llama-3-8B", "NousResearch/Meta-Llama-3-8B", "Llama"),
    ModelSpec("Qwen3-30B-A3B", "Qwen/Qwen3-30B-A3B", "Qwen-MoE"),
]


TABLE1_EXTRA_MODELS = [
    ModelSpec("Qwen3-Next", "Qwen/Qwen3-Next-80B-A3B-Instruct", "Qwen-Next"),
]


SGLANG_MODEL = ModelSpec("Qwen3-32B", "Qwen/Qwen3-32B", "Qwen")


def requested_transfer_grid():
    rows = []
    for seq_len in SEQ_SWEEP_BS1:
        rows.append({"sweep": "bs1_seq", "batch_size": 1, "seq_len": seq_len})
    for seq_len in SEQ_SWEEP_BS16:
        rows.append({"sweep": "bs16_seq", "batch_size": 16, "seq_len": seq_len})
    for bs in BS_SWEEP_SEQ1024:
        rows.append({"sweep": "seq1024_bs", "batch_size": bs, "seq_len": 1024})
    for bs in BS_SWEEP_SEQ32768:
        rows.append({"sweep": "seq32768_bs", "batch_size": bs, "seq_len": 32768})
    return rows

