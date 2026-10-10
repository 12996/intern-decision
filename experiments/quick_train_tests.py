"""CPU-only checks; importing this module does not execute the test suite."""

import copy
import json
import os
from pathlib import Path
import random
import tempfile
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from frozen_visual_cache import FrozenVisualCache, tensor_digest
from quick_validation import metrics


def assert_tree_equal(a, b):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and torch.equal(a.cpu(), b.cpu()), "Tensor state differs"
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and np.array_equal(a, b), "Array state differs"
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), "State keys differ"
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), "Sequence state differs"
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b, "Scalar state differs"


class VisualFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = nn.Linear(4, 4)
        self.multi_modal_projector = nn.Linear(4, 3)
        self.calls = 0
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def get_visual_features(self, pixels, grid, mesh=None):
        self.calls += 1
        value = self.multi_modal_projector(self.vision_tower(pixels))
        return value, [value + 1]


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = VisualFixture()
        self.pixels, self.grid = torch.rand(8, 4), torch.tensor([[1, 2, 4]])
        self.identity = {"pixels_sha256": tensor_digest(self.pixels), "image_grid_thw": self.grid.tolist(),
                         "frame_sha256": ["fixture"] * 8}
        self.cache = FrozenVisualCache(self.temp.name, {"fixture_revision": "one"})

    def forward(self):
        return self.model.get_visual_features(self.pixels, self.grid)

    def test_lossless_hit_and_method_cleanup(self):
        original = self.model.get_visual_features
        with torch.no_grad(), self.cache.bind(self.model, self.identity):
            first = self.forward()
        self.assertFalse(self.cache.last_hit)
        with torch.no_grad(), self.cache.bind(self.model, self.identity):
            second = self.forward()
        self.assertTrue(self.cache.last_hit)
        self.assertEqual(self.model.calls, 1)
        assert_tree_equal(first, second)
        self.assertEqual(original, self.model.get_visual_features)

    def test_train_or_grad_rejected(self):
        with self.assertRaisesRegex(ValueError, "eval/no-grad"):
            with self.cache.bind(self.model, self.identity):
                pass
        self.model.train()
        with torch.no_grad(), self.assertRaises(ValueError):
            with self.cache.bind(self.model, self.identity):
                pass

    def test_mismatch_and_failure_restore_method(self):
        original = self.model.get_visual_features
        wrong = dict(self.identity, pixels_sha256="wrong")
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "input mismatch"):
            with self.cache.bind(self.model, wrong):
                self.forward()
        self.assertEqual(original, self.model.get_visual_features)

    def test_corrupted_cache_rejected(self):
        with torch.no_grad(), self.cache.bind(self.model, self.identity):
            self.forward()
        path = next(Path(self.temp.name).glob("*.pt"))
        payload = torch.load(path, weights_only=True)
        payload["output"][0][0, 0] += 1
        torch.save(payload, path)
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "checksum"):
            with self.cache.bind(self.model, self.identity):
                self.forward()

    def test_provenance_prevents_reuse(self):
        with torch.no_grad(), self.cache.bind(self.model, self.identity):
            self.forward()
        other = FrozenVisualCache(self.temp.name, {"fixture_revision": "two"})
        with torch.no_grad(), other.bind(self.model, self.identity):
            self.forward()
        self.assertFalse(other.last_hit)
        self.assertEqual(self.model.calls, 2)

    def test_read_only_miss_never_calls_vision_or_writes(self):
        cache = FrozenVisualCache(self.temp.name, {"fixture_revision": "one"}, read_only=True)
        original = self.model.get_visual_features
        with torch.no_grad(), self.assertRaisesRegex(FileNotFoundError, "Read-only"):
            with cache.bind(self.model, self.identity):
                self.forward()
        self.assertEqual(self.model.calls, 0)
        self.assertEqual(original, self.model.get_visual_features)
        self.assertEqual(list(Path(self.temp.name).glob("*.pt")), [])

    def test_read_only_valid_hit_is_lossless(self):
        with torch.no_grad(), self.cache.bind(self.model, self.identity):
            first = self.forward()
        cache = FrozenVisualCache(self.temp.name, {"fixture_revision": "one"}, read_only=True)
        with torch.no_grad(), cache.bind(self.model, self.identity):
            second = self.forward()
        assert_tree_equal(first, second)
        self.assertTrue(cache.last_hit)
        self.assertEqual(self.model.calls, 1)


class ProtocolTests(unittest.TestCase):
    def test_manifest_and_confirmation_exclusion(self):
        folder = Path(__file__).parent
        manifest = json.loads((folder / "quick_validation_pairs_20261003.json").read_text())
        protocol = json.loads((folder / "evaluation_protocol_v2_20261003.json").read_text())
        self.assertEqual(len(manifest["pairs"]), 160)
        self.assertEqual(len(manifest["queries"]), 32)
        self.assertFalse({p["video_id"] for p in manifest["pairs"]} & set(protocol["confirmation_videos"]))
        self.assertFalse({q["cap_id"] for q in manifest["queries"]} - set(protocol["development"]))
        for query in manifest["queries"]:
            candidates = [p for p in manifest["pairs"] if p["cap_id"] == query["cap_id"]]
            self.assertEqual(len({p["video_id"] for p in candidates}), 5)
            self.assertEqual(sorted(p["pair_type"] for p in candidates), ["hard", "hard", "positive", "random", "random"])
            self.assertFalse({p["video_id"] for p in candidates if p["label"] == "no"} & set(query["known_positives"]))

    def test_stable_ties_do_not_prefer_positive(self):
        pairs = [{"id": str(i), "label": "yes" if i == 4 else "no", "candidate_order": i} for i in range(5)]
        manifest = {"pairs": pairs, "queries": [{"cap_id": "q", "positive_video": "v", "pair_ids": list(map(str, range(5)))}]}
        scores = [{"pair_id": str(i), "score": 1} for i in range(5)]
        self.assertEqual(metrics(manifest, scores)["macro_mrr5"], 0.2)
        with self.assertRaisesRegex(ValueError, "Missing/duplicate"):
            metrics(manifest, scores[:-1])

    def test_rng_and_per_module_mode_preserved(self):
        from quick_train_hooks import validation_state
        from scaleup_observe import _rng_state
        model = VisualFixture()
        model.train()
        model.vision_tower.eval()
        modes = [m.training for m in model.modules()]
        before = _rng_state()
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with validation_state(model):
                torch.rand(3)
                random.random()
                np.random.rand()
                raise RuntimeError("fixture")
        assert_tree_equal(before, _rng_state())
        self.assertEqual(modes, [m.training for m in model.modules()])

    def test_removal_guard(self):
        from quick_train_hooks import safe_remove
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            for path in [parent / "old-history", parent.parent / "ckpt-step-1"]:
                with self.assertRaises(ValueError):
                    safe_remove(path, parent, "ckpt-step-")

    def test_gpu2_is_never_selected(self):
        from run_quick_training import choose_gpu
        with patch("run_quick_training.subprocess.check_output", return_value="1, 20000, 0\n2, 32000, 0\n5, 27000, 0\n"):
            self.assertEqual(choose_gpu(), (27000, 5))
        with patch("run_quick_training.subprocess.check_output", return_value="1, 20000, 0\n2, 32000, 0\n"):
            with self.assertRaisesRegex(RuntimeError, "no permittedGPU"):
                choose_gpu()

    def test_stop_refuses_reused_pid_identity(self):
        from run_quick_training import stop_owned
        record = {"pid": 123, "start_ticks": 11, "cmdline": "owned fixture"}
        with patch("run_quick_training.identity", return_value={**record, "start_ticks": 12}), \
                patch("run_quick_training.os.kill") as kill, patch("run_quick_training.time.sleep"):
            stop_owned([record])
            kill.assert_not_called()


class OfficialDCPTests(unittest.TestCase):
    def test_official_engine_restores_nonzero_adam_moments(self):
        import torch.distributed as dist
        from xtuner.v1.engine.train_engine import TrainEngine
        with tempfile.TemporaryDirectory() as folder:
            rendezvous = Path(folder) / "rendezvous"
            dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
            try:
                engine = TrainEngine.__new__(TrainEngine)
                engine.model = nn.Linear(4, 3)
                engine.model_cfg = SimpleNamespace(dcp_ignore_frozen_params=False)
                engine.has_freeze_params = False
                engine.optimizer = torch.optim.AdamW(engine.model.parameters(), lr=2e-6, foreach=False)
                engine.model(torch.rand(2, 4)).sum().backward()
                engine.optimizer.step()
                weights = copy.deepcopy(engine.model.state_dict())
                optimizer = copy.deepcopy(engine.optimizer.state_dict())
                checkpoint = Path(folder) / "weights"
                # Official profiling assumes an accelerator. Skip only those
                # counters on this CPU fixture, not any DCP/state operation.
                with patch("xtuner.v1.engine.train_engine.profile_time_and_memory", side_effect=lambda *a, **k: nullcontext()):
                    engine.save_dcp(checkpoint)
                with torch.no_grad():
                    for p in engine.model.parameters():
                        p.add_(1)
                for state in engine.optimizer.state.values():
                    state["exp_avg"].zero_()
                with patch("xtuner.v1.engine.train_engine.profile_time_and_memory", side_effect=lambda *a, **k: nullcontext()):
                    engine.load_dcp(checkpoint, load_states=True, load_args=True)
                assert_tree_equal(weights, engine.model.state_dict())
                assert_tree_equal(optimizer, engine.optimizer.state_dict())
                self.assertTrue(any(s["exp_avg"].abs().sum() > 0 for s in engine.optimizer.state.values()))
            finally:
                dist.destroy_process_group()


class SaveSpaceTests(unittest.TestCase):
    def fixture(self, folder, step):
        return SimpleNamespace(work_dir=Path(folder), cur_step=step, total_step=100,
                               _checkpoint_interval=25, _snapshot_interval=None)

    def test_insufficient_space_keeps_previous_last_even_at_final(self):
        from quick_train_hooks import maybe_save_with_space, GIB
        for step in [25, 50, 100]:
            with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"QUICK_OUT": folder}), \
                    patch("quick_train_hooks.shutil.disk_usage", return_value=SimpleNamespace(free=73 * GIB)):
                original = unittest.mock.Mock()
                self.assertFalse(maybe_save_with_space(self.fixture(folder, step), original))
                original.assert_not_called()
                record = json.loads((Path(folder) / f"last-save-skipped-{step}.json").read_text())
                self.assertTrue(record["previous_last_kept"])

    def test_enough_space_calls_official_save(self):
        from quick_train_hooks import maybe_save_with_space, GIB
        trainer = self.fixture(Path.cwd(), 25)
        with patch("quick_train_hooks.shutil.disk_usage", return_value=SimpleNamespace(free=74 * GIB)):
            original = unittest.mock.Mock(return_value=True)
            self.assertTrue(maybe_save_with_space(trainer, original))
            original.assert_called_once_with(trainer, is_snapshot=False)

    def test_initial_restore_checkpoint_not_optional(self):
        from quick_train_hooks import maybe_save_with_space, GIB
        with patch("quick_train_hooks.shutil.disk_usage", return_value=SimpleNamespace(free=73 * GIB)):
            with self.assertRaisesRegex(RuntimeError, "cannot be skipped"):
                maybe_save_with_space(self.fixture(Path.cwd(), 0), unittest.mock.Mock())

    def test_not_due_or_snapshot_disabled_no_io(self):
        from quick_train_hooks import maybe_save_with_space
        with patch("quick_train_hooks.shutil.disk_usage") as disk:
            self.assertFalse(maybe_save_with_space(self.fixture(Path.cwd(), 24), unittest.mock.Mock()))
            self.assertFalse(maybe_save_with_space(self.fixture(Path.cwd(), 25), unittest.mock.Mock(), is_snapshot=True))
            disk.assert_not_called()


class BestSelectionTests(unittest.TestCase):
    def test_baseline_tie_and_improvement_rolling(self):
        from quick_train_hooks import select_best
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            released, exp = out / "released", out / "new-experiment"
            released.mkdir()
            exp.mkdir()
            (released / "config.json").write_text("{}")
            exports = []
            def export(path, save_dtype):
                self.assertEqual(save_dtype, torch.bfloat16)
                path = Path(path)
                path.mkdir()
                save_file({"fixture": torch.ones(2, dtype=save_dtype)}, path / "model.safetensors")
                (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"fixture": "model.safetensors"}}))
                exports.append(path)
            def adapter(engine, path):
                engine.save_hf(path, save_dtype=torch.bfloat16)
                return {"pass": True}
            trainer = SimpleNamespace(work_dir=out, exp_dir=exp, cur_step=0,
                                      _engine=SimpleNamespace(save_hf=export),
                                      tokenizer=SimpleNamespace(save_pretrained=lambda path: None))
            env = {"QUICK_OUT": str(out), "MODEL_PATH": str(released), "QUICK_MANIFEST_SHA256": "fixture"}
            with patch.dict(os.environ, env), patch("quick_train_hooks.check_disk", return_value=1000), \
                    patch("quick_train_hooks.save_hf_fp32", side_effect=adapter):
                select_best(trainer, {"step": 0, "macro_mrr5": 0.4})
                trainer.cur_step = 50
                select_best(trainer, {"step": 50, "macro_mrr5": 0.4})
                self.assertEqual(exports, [])
                select_best(trainer, {"step": 50, "macro_mrr5": 0.5})
                self.assertTrue(exports[0].is_dir())
                trainer.cur_step = 100
                select_best(trainer, {"step": 100, "macro_mrr5": 0.6})
                self.assertFalse(exports[0].exists())
                self.assertTrue(exports[1].is_dir())
                self.assertTrue(released.is_dir())
                self.assertEqual(json.loads((out / "proxy_best.json").read_text())["step"], 100)

    def test_failed_export_keeps_previous_selection(self):
        from quick_train_hooks import select_best
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            trainer = SimpleNamespace(work_dir=out, exp_dir=out, cur_step=0,
                                      _engine=SimpleNamespace(save_hf=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fixture export failure"))))
            env = {"QUICK_OUT": str(out), "MODEL_PATH": str(out), "QUICK_MANIFEST_SHA256": "fixture"}
            with patch.dict(os.environ, env), patch("quick_train_hooks.check_disk", return_value=1000), \
                    patch("quick_train_hooks.save_hf_fp32", side_effect=RuntimeError("fixture export failure")):
                select_best(trainer, {"step": 0, "macro_mrr5": 0.4})
                trainer.cur_step = 50
                with self.assertRaisesRegex(RuntimeError, "fixture export failure"):
                    select_best(trainer, {"step": 50, "macro_mrr5": 0.5})
                self.assertEqual(trainer._quick_best["step"], 0)
                self.assertEqual(json.loads((out / "proxy_best.json").read_text())["step"], 0)


if __name__ == "__main__":
    if torch.cuda.is_available():
        raise SystemExit("CPU suite must not expose GPUs")
    unittest.main(verbosity=2)
