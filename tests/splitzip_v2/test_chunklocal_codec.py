import torch
import unittest

from experiments.splitzip_v2.codec_cpu import (
    ChunkLocalSplitZipCPU,
    pack_nibbles,
    reviewer_compaction_paragraph,
    unpack_nibbles,
)
from experiments.splitzip_v2.serialization import deserialize_chunklocal, serialize_chunklocal


def bf16_from_raw(raw_values):
    return torch.tensor(raw_values, dtype=torch.int32).to(torch.uint16).view(torch.int16).view(torch.bfloat16)


class ChunkLocalCodecTest(unittest.TestCase):
    def test_pack_unpack_odd_length(self):
        codes = torch.arange(17, dtype=torch.uint8) & 0x0F
        packed = pack_nibbles(codes)
        self.assertEqual(packed.numel(), 9)
        self.assertTrue(torch.equal(unpack_nibbles(packed, codes.numel()), codes))

    def test_chunklocal_roundtrip_with_escapes_across_chunks(self):
        raw = []
        for i in range(81):
            exp = 120 + (i % 16)
            if i in (0, 7, 17, 33, 64, 80):
                exp = 200 + (i % 3)
            sign = 0x8000 if i % 5 == 0 else 0
            mantissa = i & 0x7F
            raw.append(sign | (exp << 7) | mantissa)
        x = bf16_from_raw(raw).view(9, 9)
        codec = ChunkLocalSplitZipCPU(chunk_size=16)
        codec.calibrate(bf16_from_raw([(120 + (i % 16)) << 7 for i in range(160)]))
        encoded = codec.encode(x, profile=True)
        decoded = codec.decode(encoded, profile=True)
        self.assertEqual(encoded.local_pos.dtype, torch.uint16)
        self.assertEqual(encoded.n_chunks, 6)
        self.assertEqual(encoded.n_escapes, 6)
        self.assertTrue(torch.equal(x.view(torch.int16), decoded.view(torch.int16)))

    def test_serialization_roundtrip_is_lossless(self):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(123)
        x = torch.randn(257, dtype=torch.bfloat16, generator=gen)
        codec = ChunkLocalSplitZipCPU(chunk_size=64)
        codec.calibrate(x[:128])
        payload = serialize_chunklocal(codec.encode(x))
        restored = deserialize_chunklocal(payload)
        decoded = codec.decode(restored)
        self.assertTrue(torch.equal(x.view(torch.int16), decoded.view(torch.int16)))

    def test_reviewer_paragraph_mentions_low_contention_algorithm(self):
        paragraph = reviewer_compaction_paragraph()
        self.assertIn("per-chunk counting", paragraph)
        self.assertIn("prefix sum", paragraph)
        self.assertIn("avoids global atomic", paragraph)
