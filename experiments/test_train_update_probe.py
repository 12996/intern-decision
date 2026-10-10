"""CPU regressions for sparse samples, true no-update and retry storage."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from quick_train_hooks import model_digest
from train_update_probe import LanguageUpdateProbe


def probe_for(module):
    wrapper = SimpleNamespace(language_model=module)
    digest = lambda: model_digest(wrapper, ["language_model"])["language_model"]
    return LanguageUpdateProbe(module, digest, digest())


class ProbeTests(unittest.TestCase):
    def test_embedding_prefix_unchanged_is_not_global_no_update(self):
        module = nn.Module()
        module.embedding = nn.Embedding(2048, 4)
        with torch.no_grad():
            module.embedding.weight.zero_()
        probe = probe_for(module)
        old_prefix = module.embedding.weight.detach().reshape(-1)[:1024].clone()
        with torch.no_grad():
            module.embedding.weight[1500, 3] += 1e-6
        self.assertTrue(torch.equal(old_prefix, module.embedding.weight.detach().reshape(-1)[:1024]))
        result = probe.check(3)
        self.assertEqual(result["sample_max_abs_delta"], 0)
        self.assertTrue(result["full_hash_checked"])
        self.assertTrue(result["update_detected"])

    def test_truly_unchanged_model_fails_the_proof(self):
        result = probe_for(nn.Linear(128, 64)).check(3)
        self.assertTrue(result["full_hash_checked"])
        self.assertFalse(result["update_detected"])

    def test_dense_sample_change_is_detected(self):
        module = nn.Linear(128, 64)
        probe = probe_for(module)
        name = next(iter(probe.regions))
        with torch.no_grad():
            dict(module.named_parameters())[name].reshape(-1)[0] += 1e-5
        result = probe.check(3)
        self.assertTrue(result["update_detected"])
        self.assertFalse(result["full_hash_checked"])
        self.assertIn(name, result["changed_sample_parameters"])

    def test_step5_always_has_whole_state_proof(self):
        module = nn.Linear(128, 64)
        probe = probe_for(module)
        with torch.no_grad():
            module.weight[0, 0] += 1e-5
        result = probe.check(5, verify_full=True)
        self.assertTrue(result["full_hash_checked"])
        self.assertNotEqual(result["before_sha256"], result["after_sha256"])

    def test_warmup_zero_lr_not_prematurely_hash_checked(self):
        result = probe_for(nn.Linear(128, 64)).check(1)
        self.assertFalse(result["update_detected"])
        self.assertFalse(result["full_hash_checked"])

    def test_frozen_model_and_nonfinite_sample_are_rejected(self):
        module = nn.Linear(128, 64)
        module.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "No trainable"):
            probe_for(module)
        module.requires_grad_(True)
        probe = probe_for(module)
        name = next(iter(probe.regions))
        with torch.no_grad():
            dict(module.named_parameters())[name].reshape(-1)[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            probe.check(3)

    def test_reused_checkpoint_has_smaller_start_gate_not_lower_reserve(self):
        from run_quick_training import required_free, GIB
        self.assertEqual(required_free(SimpleNamespace(retry_source=Path("approved"))), 51 * GIB)
        self.assertEqual(required_free(SimpleNamespace(retry_source=None)), int(76.25 * GIB))


class SmokeReceiptTests(unittest.TestCase):
    def fixture(self):
        return {"pass": True, "step": 5, "at": 101.0, "work_dir": "/approved/run",
                "frozen_visual_unchanged": True, "finite_loss_and_gradient": True,
                "first_five_steps_are_part_of_the100_update_pilot": True,
                "update_proof": {"step": 5, "full_hash_checked": True, "update_detected": True,
                                 "before_sha256": "a" * 64, "after_sha256": "b" * 64}}

    def test_missing_receipt_is_not_pass(self):
        from run_quick_training import smoke_passed
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(smoke_passed(folder, 100.0, Path("/approved/run"), "a" * 64))

    def test_exact_complete_receipt_is_pass(self):
        from run_quick_training import smoke_passed
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "smoke_pass.json").write_text(json.dumps(self.fixture()))
            self.assertTrue(smoke_passed(folder, 100.0, Path("/approved/run"), "a" * 64))

    def test_incomplete_wrong_run_late_or_unchanged_receipt_cannot_continue(self):
        from run_quick_training import smoke_passed
        invalid = [{"pass": True}, {**self.fixture(), "step": 4},
                   {**self.fixture(), "work_dir": "/other/run"},
                   {**self.fixture(), "at": 1900.0}, {**self.fixture(), "at": 99.0},
                   {**self.fixture(), "finite_loss_and_gradient": False},
                   {**self.fixture(), "frozen_visual_unchanged": False}]
        for changes in [{"full_hash_checked": False}, {"after_sha256": "a" * 64},
                        {"before_sha256": "c" * 64}, {"after_sha256": None}]:
            receipt = copy.deepcopy(self.fixture())
            receipt["update_proof"].update(changes)
            invalid.append(receipt)
        with tempfile.TemporaryDirectory() as folder:
            for receipt in invalid:
                with self.subTest(receipt=receipt):
                    (Path(folder) / "smoke_pass.json").write_text(json.dumps(receipt))
                    with self.assertRaisesRegex(ValueError, "continuation forbidden"):
                        smoke_passed(folder, 100.0, Path("/approved/run"), "a" * 64)


if __name__ == "__main__":
    unittest.main()
