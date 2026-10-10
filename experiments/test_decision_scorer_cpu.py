"""CPU control-flow fixtures, not 2B-model parity or CUDA performance evidence."""

import datetime
import hashlib
import inspect
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard, distribute_tensor

import decision_scorer as scorer_module
from xtuner.v1.loss.ce_loss import LMHeadLossContext
from xtuner.v1.model.dense.qwen3vl_text import Qwen3VLTextDense
from xtuner.v1.module.lm_head.lm_head import LMHead

torch.set_num_threads(1)


class EmptyRoPE(nn.Module):
    def forward(self, hidden, positions):
        return None


class TinyText(Qwen3VLTextDense):
    """Run the actual inherited forward with no decoder layers or checkpoint."""

    def __init__(self):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(return_hidden_states=False)
        self.embed_tokens = nn.Identity()
        self.rotary_emb = EmptyRoPE()
        self.layers = nn.ModuleDict()
        self.norm = nn.LayerNorm(8)
        self.lm_head = LMHead(8, 16, bias=False)


def context():
    return SimpleNamespace(
        input_ids=None, inputs_embeds=torch.randn(1, 4, 8),
        position_ids=torch.arange(4).reshape(1, 4),
        cu_seq_lens_q=torch.tensor([0, 4], dtype=torch.int32),
        cu_seq_lens_k=torch.tensor([0, 4], dtype=torch.int32),
        deepstack_visual_embeds=None, visual_pos_masks=None,
    )


class TextHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        dist.init_process_group("gloo", init_method=f"file://{cls.temp.name}/store",
                                rank=0, world_size=1, timeout=datetime.timedelta(seconds=30))
        cls.mesh = DeviceMesh("cpu", [0])

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
        cls.temp.cleanup()

    def setUp(self):
        torch.manual_seed(42)
        self.model, self.ctx = TinyText().eval(), context()
        weight = distribute_tensor(self.model.lm_head.weight.detach(), self.mesh, [Shard(0)])
        self.model.lm_head.weight = nn.Parameter(weight, requires_grad=False)

    def test_none_reproduces_actual_mixed_tensor_dtensor_failure(self):
        with self.assertRaisesRegex(RuntimeError, "mixed torch.Tensor and DTensor"):
            self.model(self.ctx, loss_ctx=None)

    def test_non_none_container_calls_actual_head_without_ce(self):
        calls = []
        handle = self.model.lm_head.register_forward_pre_hook(lambda _m, args: calls.append(args[1]))
        try:
            with patch.object(LMHeadLossContext, "forward", side_effect=AssertionError("CE invoked")):
                result = self.model(self.ctx, loss_ctx={"lm": None})
        finally:
            handle.remove()
        expected = nn.functional.linear(self.model.norm(self.ctx.inputs_embeds),
                                        self.model.lm_head.weight.to_local())
        torch.testing.assert_close(result.logits, expected, rtol=0, atol=0)
        self.assertIsNone(result.loss)
        self.assertEqual(calls, [None])

    def test_decision_position_selection_matches_full_head(self):
        reference = self.model(self.ctx, loss_ctx={"lm": None}).logits[:, 2:3]
        reference_hidden = self.model.norm(self.ctx.inputs_embeds)[:, 2:3]
        def select(_module, _args, output):
            selected = output[:, 2:3]
            torch.testing.assert_close(selected, reference_hidden, rtol=0, atol=0)
            return selected
        handle = self.model.norm.register_forward_hook(select)
        try:
            actual = self.model(self.ctx, loss_ctx={"lm": None}).logits
        finally:
            handle.remove()
        # CPU GEMM changes shape; use the already locked raw-logit tolerance.
        torch.testing.assert_close(actual, reference, rtol=0, atol=0.001)
        print(f"CPU selected/full head raw_abs_error={float((actual - reference).abs().max())}", flush=True)
        self.assertEqual(tuple(actual.shape), (1, 1, 16))


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(model="released", pair_id="p0", mode="decision_position",
                        projection="reference_post_fsdp_fp32")

    def test_failed_or_missing_check_cannot_restore_success(self):
        for checks in [[], [dict(self.row, **{"pass": False})]]:
            self.assertEqual(scorer_module.restored_results([self.row], checks, "released"), {})

    def test_latest_failure_overrides_pass(self):
        checks = [dict(self.row, **{"pass": True}), dict(self.row, **{"pass": False})]
        self.assertEqual(scorer_module.restored_results([self.row], checks, "released"), {})

    def test_models_and_projections_cannot_mix(self):
        for check in [dict(self.row, model="hf800", **{"pass": True}),
                      dict(self.row, projection="native_bf16", **{"pass": True})]:
            self.assertEqual(scorer_module.restored_results([self.row], [check], "released"), {})
        good = dict(self.row, **{"pass": True})
        self.assertEqual(scorer_module.restored_results([self.row], [good], "released"),
                         {("p0", "decision_position"): self.row})
        self.assertEqual(scorer_module.restored_results([self.row], [good], "hf800"), {})

    def test_historical_reference_is_retained(self):
        reference = dict(self.row, mode="reference_ce")
        reference.pop("projection")
        self.assertEqual(scorer_module.restored_results([reference], [], "released"),
                         {("p0", "reference_ce"): reference})


class Qwen3VLVisionAttention(nn.Identity):
    def __init__(self):
        super().__init__()
        self.attn_impl_func = object()


class FakeVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Identity()
        self.attention = Qwen3VLVisionAttention()
        self.norm = nn.LayerNorm(8)
        self.fail_forward = False

    def fast_pos_embed_interpolate(self, hidden):
        return hidden

    def rot_pos_emb(self, hidden):
        return hidden

    def forward(self, hidden):
        hidden = self.fast_pos_embed_interpolate(hidden)
        if self.fail_forward:
            raise RuntimeError("fixture forward failure")
        return self.norm(self.attention(self.patch_embed(self.rot_pos_emb(hidden))))


class FakeVL(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = FakeVision()
        self.multi_modal_projector = nn.Identity()
        self.language_model = TinyText()

    def forward(self, seq_ctx, loss_ctx):
        seq_ctx.inputs_embeds = self.multi_modal_projector(self.vision_tower(seq_ctx.inputs_embeds))
        return self.language_model(seq_ctx, loss_ctx)


class FakeEvent:
    def __init__(self, **kwargs):
        pass

    def record(self):
        pass

    def elapsed_time(self, other):
        return 1.0


class HookCleanupTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.scorer = scorer_module.DecisionScorer.__new__(scorer_module.DecisionScorer)
        self.scorer.model = FakeVL().eval()
        self.scorer.head = self.scorer.model.language_model.lm_head
        self.scorer.freeze = {}
        self.prepared = (context(), {"lm": None},
                         {"decision_position": 2, "symbol_ids": {"no": 3, "yes": 7}}, {})
        vision = self.scorer.model.vision_tower
        self.methods = [vision.fast_pos_embed_interpolate, vision.rot_pos_emb]
        self.attention = vision.attention.attn_impl_func
        for name, value in [("Event", FakeEvent), ("synchronize", None),
                            ("reset_peak_memory_stats", None), ("max_memory_allocated", 0),
                            ("max_memory_reserved", 0)]:
            mock = patch.object(torch.cuda, name, **({"new": value} if name == "Event" else {"return_value": value}))
            mock.start()
            self.addCleanup(mock.stop)

    def assert_clean(self):
        vision = self.scorer.model.vision_tower
        self.assertEqual([vision.fast_pos_embed_interpolate, vision.rot_pos_emb], self.methods)
        self.assertIs(vision.attention.attn_impl_func, self.attention)
        for module in self.scorer.model.modules():
            self.assertFalse(module._forward_pre_hooks)
            self.assertFalse(module._forward_hooks)

    def test_cleanup_after_success(self):
        result = self.scorer.forward(self.prepared, detail_profile=True)
        self.assertEqual(result["projection"], "reference_post_fsdp_fp32")
        self.assertIn("vision_detail_cuda_ms", result["timing"])
        language = self.scorer.model.language_model
        hidden = language.norm(self.prepared[0].inputs_embeds)[0, 2]
        expected = nn.functional.linear(hidden.unsqueeze(0), self.scorer.head.weight)[0]
        self.assertEqual(result["raw_logits"], {"no": float(expected[3]), "yes": float(expected[7])})
        self.assert_clean()

    def test_cleanup_after_forward_failure(self):
        self.scorer.model.vision_tower.fail_forward = True
        with self.assertRaisesRegex(RuntimeError, "fixture forward failure"):
            self.scorer.forward(self.prepared, detail_profile=True, segmented=True)
        self.assert_clean()

    def test_cleanup_on_setup_failure(self):
        with patch.object(torch.cuda, "reset_peak_memory_stats", side_effect=RuntimeError("fixture setup failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture setup failure"):
                self.scorer.forward(self.prepared, segmented=True)
        self.assert_clean()

    def test_cleanup_on_setup_failure_with_detail_profile(self):
        with patch.object(torch.cuda, "reset_peak_memory_stats", side_effect=RuntimeError("fixture setup failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture setup failure"):
                self.scorer.forward(self.prepared, detail_profile=True, segmented=True)
        self.assert_clean()


if __name__ == "__main__":
    if torch.cuda.is_available():
        raise SystemExit("CPU regression must not expose CUDA")
    for cls in [Qwen3VLTextDense, LMHead]:
        source = Path(inspect.getfile(cls))
        print(f"source {source} sha256={hashlib.sha256(source.read_bytes()).hexdigest()}", flush=True)
    print(f"torch={torch.__version__} CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r}", flush=True)
    unittest.main(verbosity=2)
