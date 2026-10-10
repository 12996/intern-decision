"""Actual vendor save/load regression, CPU only; no 2B model allocation."""

from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from safetensors.torch import load_file, save_file
import torch
import torch.distributed as dist
from torch import nn
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.model.base import BaseModel, HFSaveCfg, XTunerBaseModelConfig
from xtuner.v1.model.compose.base import BaseComposeModel

from fp32_hf_export import fp32_save_policy, save_hf_fp32, verify_fp32_export


class TinyModel(BaseModel):
    def __init__(self, source, prefix):
        super().__init__(XTunerBaseModelConfig(compile_cfg=False, hf_save_cfg=HFSaveCfg(
            bucket_size=64, worker_per_rank=1, fp32_keys_pattern=[r"never_matches"])))
        self.prefix = prefix
        self.weight = nn.Parameter(torch.tensor([1.0 + 2e-6, -0.5 - 1e-6], dtype=torch.float32))
        self.fused = nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3) + 2e-6)
        self.register_buffer("counter", torch.tensor(7, dtype=torch.int64))
        self.register_buffer("coefficient", torch.tensor([0.5], dtype=torch.bfloat16))
        self.set_hf(source)
        self._init_load_spec()

    def to_hf_key_list(self, key):
        keys = ["q", "k"] if key == "fused" else [key]
        return [f"{self.prefix}.{k}" for k in keys]


class VendorFP32Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "released"
        self.source.mkdir()
        (self.source / "config.json").write_text(json.dumps({
            "model_type": "fixture", "torch_dtype": "bfloat16", "text_config": {"dtype": "bfloat16"}}))
        dist.init_process_group("gloo", init_method=f"file://{self.root / 'rendezvous'}", rank=0, world_size=1)
        self.addCleanup(dist.destroy_process_group)
        # Only accelerator profiling/cache cleanup/sync are disabled. SavePlan, dtype
        # policy, safetensors serialization, compose index and loader are real.
        profiler = patch("xtuner.v1.model.base.profile_time_and_memory", side_effect=lambda *a, **k: nullcontext())
        profiler.start()
        self.addCleanup(profiler.stop)
        cache_cleanup = patch("xtuner.v1.model.base.DEVICE_MODULE.empty_cache", create=True)
        cache_cleanup.start()
        self.addCleanup(cache_cleanup.stop)
        accelerator_sync = patch("torch.accelerator.synchronize")
        accelerator_sync.start()
        self.addCleanup(accelerator_sync.stop)

    def engine(self, compose=False):
        engine = TrainEngine.__new__(TrainEngine)
        if not compose:
            engine.model = TinyModel(self.source, "text")
        else:
            model = BaseComposeModel.__new__(BaseComposeModel)
            nn.Module.__init__(model)
            model.language_model = TinyModel(self.source, "text")
            model.vision_tower = TinyModel(self.source, "vision")
            model.multi_modal_projector = TinyModel(self.source, "projector")
            engine.model = model
        return engine

    def test_old_interface_fails_and_plain_bf16_loses_small_update(self):
        engine = self.engine()
        with self.assertRaisesRegex(AssertionError, "torch.float32 is not supported"):
            engine.save_hf(str(self.root / "unsupported"), save_dtype=torch.float32)
        path = self.root / "rounded"
        engine.save_hf(str(path), save_dtype=torch.bfloat16)
        index = json.loads((path / "model.safetensors.index.json").read_text())
        weight = load_file(path / index["weight_map"]["text.weight"])["text.weight"]
        self.assertEqual(weight.dtype, torch.bfloat16)
        self.assertEqual(weight[0].item(), 1.0)
        self.assertNotEqual(weight[0].item(), engine.model.weight[0].item())

    def test_composed_export_and_official_reload_keep_every_value(self):
        engine = self.engine(compose=True)
        before = copy.deepcopy(engine.model.state_dict())
        policies = [(p.config.hf_save_cfg, p.config.hf_save_cfg.fp32_keys_pattern)
                    for p in engine.model.children()]
        receipt = save_hf_fp32(engine, self.root / "best")
        self.assertTrue(receipt["all_tensor_values_bitwise_equal"])
        self.assertEqual(receipt["checked_tensors"], 15)
        self.assertEqual(receipt["fp32_parameter_tensors"], 9)
        for cfg, original in policies:
            self.assertIs(cfg.fp32_keys_pattern, original)
        with torch.no_grad():
            for parameter in engine.model.parameters():
                parameter.zero_()
        engine.model.from_hf(self.root / "best", strict=True)
        for name, tensor in before.items():
            current = engine.model.state_dict()[name]
            self.assertEqual(current.dtype, tensor.dtype)
            self.assertTrue(torch.equal(current, tensor), name)
        config = json.loads((self.root / "best/config.json").read_text())
        self.assertEqual(config["torch_dtype"], "float32")
        self.assertEqual(config["text_config"]["dtype"], "float32")

    def test_single_rank_dtensor_export_uses_real_unshard(self):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import distribute_tensor, Shard
        engine = self.engine()
        expected = engine.model.weight.detach().clone()
        mesh = init_device_mesh("cpu", (1,))
        engine.model.weight = nn.Parameter(distribute_tensor(expected, mesh, [Shard(0)]))
        receipt = save_hf_fp32(engine, self.root / "sharded-best")
        self.assertTrue(receipt["pass"])
        index = json.loads((self.root / "sharded-best/model.safetensors.index.json").read_text())
        actual = load_file(self.root / "sharded-best" / index["weight_map"]["text.weight"])["text.weight"]
        self.assertTrue(torch.equal(actual, expected))

    def test_corrupt_tensor_is_rejected(self):
        engine = self.engine()
        path = self.root / "best"
        save_hf_fp32(engine, path)
        index = json.loads((path / "model.safetensors.index.json").read_text())
        shard = path / index["weight_map"]["text.weight"]
        tensors = load_file(shard)
        tensors["text.weight"][0] = 1.0
        save_file(tensors, shard)
        with fp32_save_policy(engine.model), self.assertRaisesRegex(ValueError, "Lossy HF export"):
            verify_fp32_export(engine.model, path)

    def test_policy_restored_after_failed_export_and_no_overwrite(self):
        engine = self.engine()
        original = engine.model.config.hf_save_cfg.fp32_keys_pattern
        with patch.object(engine, "save_hf", side_effect=RuntimeError("writer failure")):
            with self.assertRaisesRegex(RuntimeError, "writer failure"):
                save_hf_fp32(engine, self.root / "failed")
        self.assertIs(engine.model.config.hf_save_cfg.fp32_keys_pattern, original)
        with self.assertRaisesRegex(ValueError, "Refuse overwrite"):
            save_hf_fp32(engine, self.source)

    def test_missing_index_entry_is_rejected(self):
        engine = self.engine()
        path = self.root / "best"
        save_hf_fp32(engine, path)
        index_path = path / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        del index["weight_map"]["text.weight"]
        index_path.write_text(json.dumps(index))
        with fp32_save_policy(engine.model), self.assertRaises(ValueError):
            verify_fp32_export(engine.model, path)


class SaveOrderTests(unittest.TestCase):
    def fixture(self, folder, step=50):
        return SimpleNamespace(work_dir=Path(folder), cur_step=step, total_step=100,
                               _checkpoint_interval=25, _snapshot_interval=None, _async_checkpoint=False,
                               _quick_baseline={"macro_mrr5": 0.4})

    def test_export_failure_cannot_prevent_checkpoint_and_only_one_save(self):
        from quick_train_hooks import GIB, maybe_save_then_validate
        with tempfile.TemporaryDirectory() as folder:
            trainer = self.fixture(folder)
            checkpoint = Path(folder) / "durable-checkpoint"
            order = []
            def save(trainer, is_snapshot):
                checkpoint.write_text("saved")
                order.append("save")
                return True
            def evaluate(trainer, step):
                self.assertTrue(checkpoint.exists())
                order.append("evaluate")
                return {"macro_mrr5": 0.5, "step": step}
            def export(*args):
                order.append("export")
                raise RuntimeError("export failure")
            original = Mock(side_effect=save)
            with patch.dict(os.environ, {"QUICK_OUT": folder}), \
                    patch("quick_train_hooks.shutil.disk_usage", return_value=SimpleNamespace(free=74 * GIB)), \
                    patch("quick_train_hooks.event"), patch("quick_train_hooks.evaluate", side_effect=evaluate), \
                    patch("quick_train_hooks.select_best", side_effect=export):
                with self.assertRaisesRegex(RuntimeError, "export failure"):
                    maybe_save_then_validate(trainer, original)
            self.assertEqual(order, ["save", "evaluate", "export"])
            original.assert_called_once_with(trainer, is_snapshot=False)
            self.assertTrue(checkpoint.exists())

    def test_space_skip_preserves_validation_policy(self):
        from quick_train_hooks import GIB, maybe_save_then_validate
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"QUICK_OUT": folder}), \
                patch("quick_train_hooks.shutil.disk_usage", return_value=SimpleNamespace(free=73 * GIB)), \
                patch("quick_train_hooks.event"), patch("quick_train_hooks.evaluate", return_value={"macro_mrr5": 0.5}) as evaluate, \
                patch("quick_train_hooks.select_best") as select:
            trainer, original = self.fixture(folder), Mock()
            self.assertFalse(maybe_save_then_validate(trainer, original))
            original.assert_not_called()
            evaluate.assert_called_once_with(trainer, 50)
            select.assert_called_once()
            self.assertTrue((Path(folder) / "last-save-skipped-50.json").exists())

    def test_async_checkpoint_is_rejected_before_io(self):
        from quick_train_hooks import maybe_save_then_validate
        trainer = self.fixture(Path.cwd())
        trainer._async_checkpoint = True
        with self.assertRaisesRegex(ValueError, "synchronous"):
            maybe_save_then_validate(trainer, Mock())

    def test_nonvalidation_step_only_saves(self):
        from quick_train_hooks import GIB, maybe_save_then_validate
        with patch("quick_train_hooks.shutil.disk_usage", return_value=SimpleNamespace(free=74 * GIB)), \
                patch("quick_train_hooks.evaluate") as evaluate:
            trainer, original = self.fixture(Path.cwd(), step=25), Mock(return_value=True)
            self.assertTrue(maybe_save_then_validate(trainer, original))
            original.assert_called_once_with(trainer, is_snapshot=False)
            evaluate.assert_not_called()


if __name__ == "__main__":
    if torch.cuda.is_available():
        raise SystemExit("CPU export regression must not expose GPUs")
    unittest.main(verbosity=2)
