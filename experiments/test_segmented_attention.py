"""CPU fixture for the actual inference optimization; not a model forward."""

import unittest

import torch

from decision_scorer import segmented_vision_eager
from xtuner.v1.ops.attn_imp import eager_attention


class SegmentTests(unittest.TestCase):
    def test_matches_masked_eager_and_blocks_cross_image_leakage(self):
        generator = torch.Generator().manual_seed(42)
        q, k, v = [torch.randn(1, 2, 20, 8, generator=generator) for _ in range(3)]
        boundaries = torch.tensor([0, 5, 13, 20], dtype=torch.int32)
        kwargs = {"cu_seqlens_q": boundaries, "softmax_scale": 8 ** -0.5, "causal": False}
        reference = eager_attention(q, k, v, **kwargs)["raw_output"]
        actual = segmented_vision_eager(q, k, v, **kwargs)["raw_output"]
        torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
        changed = v.clone()
        changed[:, :, 5:] += 1000
        changed_output = segmented_vision_eager(q, k, changed, **kwargs)["raw_output"]
        torch.testing.assert_close(actual[:, :5], changed_output[:, :5], rtol=0, atol=0)

    def test_refuses_causal_or_dropout_change(self):
        q = torch.zeros(1, 2, 4, 8)
        for kwargs in [{"causal": True}, {"dropout_p": 0.1}]:
            with self.assertRaises(ValueError):
                segmented_vision_eager(q, q, q, torch.tensor([0, 4]), 8 ** -0.5, **kwargs)


if __name__ == "__main__":
    unittest.main()
