import torch
import unittest

from experiments.splitzip_v2.config import requested_transfer_grid
from experiments.splitzip_v2.fp8_chunk_topk import analyze_raw, synthetic_raw
from experiments.splitzip_v2.orthogonality_analysis import compose


class FP8AndConfigTest(unittest.TestCase):
    def test_requested_grid_contains_v2_extremes(self):
        grid = requested_transfer_grid()
        self.assertIn({"sweep": "bs1_seq", "batch_size": 1, "seq_len": 131072}, grid)
        self.assertIn({"sweep": "bs16_seq", "batch_size": 16, "seq_len": 65536}, grid)
        self.assertIn({"sweep": "seq1024_bs", "batch_size": 256, "seq_len": 1024}, grid)
        self.assertIn({"sweep": "seq32768_bs", "batch_size": 128, "seq_len": 32768}, grid)

    def test_fp8_chunk_local_analysis_reports_coverages(self):
        if not hasattr(torch, "float8_e5m2"):
            raise unittest.SkipTest("installed PyTorch does not expose torch.float8_e5m2")
        raw = synthetic_raw(4096, "e5m2", seed=1)
        result = analyze_raw(raw, "e5m2", chunk_size=256)
        self.assertEqual(result["fmt"], "e5m2")
        self.assertLessEqual(0.0, result["global_top8_coverage"])
        self.assertLessEqual(result["global_top8_coverage"], 1.0)
        self.assertLessEqual(0.0, result["chunk_local_top8_coverage"])
        self.assertLessEqual(result["chunk_local_top8_coverage"], 1.0)
        self.assertGreater(result["chunk_local_top8_ratio_est"], 0.0)

    def test_orthogonality_model_combines_byte_reduction_with_transfer_reduction(self):
        result = compose(
            raw_bytes=2 * 32768 * 4096,
            native_transfer_ms=4.0,
            optimized_transfer_fraction=0.5,
            ratio=1.25,
            encode_gbs=300.0,
            decode_gbs=500.0,
            overlap=True,
        )
        self.assertLess(result["combined_ms"], result["software_only_ms"])
        self.assertGreater(result["combined_speedup_vs_native"], result["software_speedup"])
