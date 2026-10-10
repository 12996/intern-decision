"""CPU regressions for nonzero official DCP restoration and bounded continuation."""

from contextlib import nullcontext
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import MethodType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
from xtuner.v1.engine.train_engine import TrainEngine
from xtuner.v1.train.trainer import Trainer, LoadCheckpointConfig
import xtuner.v1.optim.swap_adamw as vendor_swap

from resume_state import make_reference, check_loaded_state, restore_resume, resume_source
from run_quick_training import smoke_passed
from scaleup_observe import _rng_state, _restore_rng, _offload_swap_moments


class LoaderFixture:
    def __init__(self):
        self.state = {"sampler": {"step": 25, "epoch": 0}, "total_consumed_samples": 25}

    def get_state_dict(self):
        return copy.deepcopy(self.state)

    def load_state_dict(self, state):
        self.state = copy.deepcopy(state)


class NonzeroRestoreTests(unittest.TestCase):
    STEP = 25

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        dist.init_process_group("gloo", init_method=f"file://{self.root / 'rendezvous'}", rank=0, world_size=1)
        self.addCleanup(dist.destroy_process_group)
        profiler = patch("xtuner.v1.engine.train_engine.profile_time_and_memory", side_effect=lambda *a, **k: nullcontext())
        profiler.start()
        self.addCleanup(profiler.stop)
        pin = patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor)
        pin.start()
        self.addCleanup(pin.stop)
        # Run the actual vendor SwapAdamW implementation with CPU-only device
        # bookkeeping, not a substitute optimizer or mocked state restoration.
        utils = ModuleType("xtuner.v1.utils")
        utils.get_device = lambda: "cpu"
        utils.get_torch_device_module = lambda: SimpleNamespace(synchronize=lambda: None, current_device=lambda: 0)
        utils.get_logger = lambda: SimpleNamespace(info=lambda *a, **k: None)
        spec = importlib.util.spec_from_file_location("cpu_resume_swap", vendor_swap.__file__)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"xtuner.v1.utils": utils}):
            spec.loader.exec_module(module)
        model = nn.Module()
        model.language_model = nn.Linear(4, 3)
        model.vision_tower = nn.Linear(2, 2)
        model.multi_modal_projector = nn.Linear(2, 2)
        model.vision_tower.requires_grad_(False)
        model.multi_modal_projector.requires_grad_(False)
        engine = TrainEngine.__new__(TrainEngine)
        engine.model = model
        engine.model_cfg = SimpleNamespace(dcp_ignore_frozen_params=False)
        engine.has_freeze_params = True
        engine.optimizer = module.SwapAdamW(model.language_model.parameters(), lr=2e-6, foreach=False, fused=False)
        scheduler = torch.optim.lr_scheduler.SequentialLR(engine.optimizer,
            [torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda step: step / 3),
             torch.optim.lr_scheduler.CosineAnnealingLR(engine.optimizer, T_max=97, eta_min=2e-7)], milestones=[3])
        for _ in range(self.STEP):
            engine.optimizer.zero_grad()
            model.language_model(torch.ones(2, 4)).sum().backward()
            engine.optimizer.step()
            scheduler.step()
        self.checkpoint = self.root / f"ckpt-step-{self.STEP}"
        engine.save_dcp(self.checkpoint / "weights")
        loader = LoaderFixture()
        loader.state["sampler"]["step"] = self.STEP
        loader.state["total_consumed_samples"] = self.STEP
        torch.save(loader.get_state_dict(), self.checkpoint / "dataloader")
        torch.save(scheduler.state_dict(), self.checkpoint / "lr_scheduler")
        torch.save(_rng_state(), self.checkpoint / "rng-rank0.pt")
        (self.checkpoint / "train_state.json").write_text(json.dumps({
            "cur_step": self.STEP, "cur_epoch": 1, "total_consumed_tokens": 33427, "train_time_offset": 1.0}))
        (self.checkpoint / "trainer_config.json").write_text('{"total_step":100,"seed":42}')
        self.reference = make_reference(self.checkpoint, step=self.STEP)
        trainer = Trainer.__new__(Trainer)
        trainer._engine, trainer._lr_scheduler, trainer._dataloader = engine, scheduler, loader
        trainer._cur_step, trainer._cur_epoch, trainer._total_step = 0, 0, 100
        trainer._load_checkpoint_cfg = LoadCheckpointConfig()
        def load_checkpoint(self):
            Trainer._load_checkpoint(self)
            _offload_swap_moments(self)
            _restore_rng(torch.load(self._load_checkpoint_cfg.checkpoint_path / "rng-rank0.pt",
                                    map_location="cpu", weights_only=False))
        trainer._load_checkpoint = MethodType(load_checkpoint, trainer)
        self.trainer = trainer
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1)
        for state in engine.optimizer.state.values():
            state["exp_avg"].zero_()
            state["exp_avg_sq"].zero_()
            state["step"].zero_()
        loader.state["total_consumed_samples"] = 0

    def restore(self):
        return restore_resume(self.trainer, {"checkpoint": self.checkpoint}, self.reference)

    def test_actual_nonzero_restore_and_next_update_keep_original_schedule(self):
        saved = self.trainer._load_checkpoint_cfg
        proof = self.restore()
        self.assertTrue(proof["all_model_optimizer_entries_exact"])
        self.assertTrue(proof["scheduler_data_rng_exact"])
        self.assertIs(self.trainer._load_checkpoint_cfg, saved)
        self.assertEqual(self.trainer.cur_step, 25)
        self.assertEqual(self.trainer.total_step, 100)
        self.assertEqual(self.trainer._lr_scheduler.state_dict()["_schedulers"][1]["T_max"], 97)
        optimizer = self.trainer._engine.optimizer
        expected_lr = self.trainer._lr_scheduler.get_last_lr()[0]
        self.assertEqual(optimizer.param_groups[0]["lr"], expected_lr)
        parameter = next(self.trainer._engine.model.language_model.parameters())
        before = parameter.detach().clone()
        optimizer.zero_grad()
        self.trainer._engine.model.language_model(torch.ones(2, 4)).sum().backward()
        optimizer.step()
        self.trainer._lr_scheduler.step()
        self.assertFalse(torch.equal(before, parameter))
        self.assertEqual(float(optimizer.state[parameter]["step"]), 26)
        self.assertEqual(self.trainer._lr_scheduler.last_epoch, 26)

    def test_corrupt_restored_parameter_or_moment_is_rejected(self):
        self.restore()
        parameter = next(self.trainer._engine.model.language_model.parameters())
        with torch.no_grad():
            parameter[0, 0] += 1e-6
        with self.assertRaisesRegex(ValueError, "Restored DCP value differs"):
            check_loaded_state(self.trainer, self.reference)
        self.restore()
        self.trainer._engine.optimizer.state[parameter]["exp_avg"].zero_()
        with self.assertRaisesRegex(ValueError, "Restored DCP value differs"):
            check_loaded_state(self.trainer, self.reference)

    def test_changed_small_checkpoint_file_is_rejected_before_load(self):
        (self.checkpoint / "train_state.json").write_text('{"cur_step":0}')
        with self.assertRaisesRegex(ValueError, "changed after CPU preflight"):
            self.restore()

    def test_reference_cannot_be_used_with_reset_horizon(self):
        self.restore()
        self.trainer._total_step = 75
        with self.assertRaisesRegex(ValueError, "step/horizon"):
            check_loaded_state(self.trainer, self.reference)


class ResumeReceiptTests(unittest.TestCase):
    def receipt(self):
        return {"pass": True, "step": 30, "resume_step": 25, "updates_since_restore": 5, "total_step": 100,
                "at": 101.0, "work_dir": "/approved/run", "frozen_visual_unchanged": True,
                "finite_loss_and_gradient": True, "first_five_steps_are_part_of_the100_update_pilot": True,
                "update_proof": {"step": 30, "updates_since_restore": 5, "full_hash_checked": True,
                    "update_detected": True, "before_sha256": "a" * 64, "after_sha256": "b" * 64}}

    def test_absolute30_receipt_passes_and_old_step5_does_not(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "smoke_pass.json"
            path.write_text(json.dumps(self.receipt()))
            self.assertTrue(smoke_passed(folder, 100, Path("/approved/run"), "a" * 64, resume_step=25))
            for changed in [{"step": 5}, {"resume_step": 0}, {"updates_since_restore": 30}, {"total_step": 75}]:
                receipt = self.receipt()
                receipt.update(changed)
                path.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, "continuation forbidden"):
                    smoke_passed(folder, 100, Path("/approved/run"), "a" * 64, resume_step=25)

    def test_unapproved_source_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "approved terminal step25"):
                resume_source(folder, Path(folder) / "wrong")


if __name__ == "__main__":
    if torch.cuda.is_available():
        raise SystemExit("Resume regression must not expose GPUs")
    unittest.main(verbosity=2)
