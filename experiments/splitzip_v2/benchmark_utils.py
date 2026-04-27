from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import statistics
import time
from typing import Dict, Iterable, List


@contextmanager
def timed_stage(name: str, out: Dict[str, float]):
    start = time.perf_counter()
    yield
    out[name] = time.perf_counter() - start


def mean_std(values: Iterable[float]):
    vals = list(values)
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {"mean": mean, "std": std, "stderr": std / (len(vals) ** 0.5)}


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


@dataclass
class StageBreakdown:
    runs: List[Dict[str, float]] = field(default_factory=list)

    def add(self, stages: Dict[str, float]):
        self.runs.append(dict(stages))

    def summary(self):
        keys = sorted({k for row in self.runs for k in row})
        return {key: mean_std(row[key] for row in self.runs if key in row) for key in keys}

