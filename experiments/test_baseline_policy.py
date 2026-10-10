"""CPU checks for explicit rebaseline policy and additive GPU accounting."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from baseline_policy import baseline_policy, requires_historical_replay
import retry_inputs
from quick_validation import digest


class BaselinePolicyTests(unittest.TestCase):
    def test_default_still_requires_historical_replay(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(requires_historical_replay())

    def test_explicit_rebase_does_not_claim_historical_equivalence(self):
        with patch.dict(os.environ, {"QUICK_BASELINE_POLICY": "fresh_destination"}):
            self.assertEqual(baseline_policy(), "fresh_destination")
            self.assertFalse(requires_historical_replay())

    def test_unknown_policy_is_not_a_bypass(self):
        with patch.dict(os.environ, {"QUICK_BASELINE_POLICY": "ignore_everything"}):
            with self.assertRaises(ValueError):
                requires_historical_replay()

    def fixture(self, root, **overrides):
        attempt = root / "outputs/goal-20261008-retry-v1"
        attempt.mkdir(parents=True)
        ledger = {"result": "fail", "owned_remaining": [], "gpu_hours_charged": 0.04,
                  "prior_gpu_hours": 5.14, "total_gpu_hours_charged": 5.18}
        ledger.update(overrides)
        path = attempt / "training_budget.json"
        path.write_text(json.dumps(ledger))
        fixture = patch.object(retry_inputs, "MIGRATED_ATTEMPT_SHA256", digest(path))
        fixture.start()
        self.addCleanup(fixture.stop)
        return attempt

    def test_previous_gpu_charge_is_added_not_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            attempt = self.fixture(root)
            result = retry_inputs.validate_previous_attempt(root, attempt, 5.14)
            self.assertAlmostEqual(result["prior_gpu_hours"], 5.18)

    def test_live_or_inconsistent_previous_attempt_is_rejected(self):
        for changed in [{"result": "running"}, {"owned_remaining": [{"pid": 1}]},
                        {"total_gpu_hours_charged": 5.14}, {"gpu_hours_charged": -1}]:
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                attempt = self.fixture(root, **changed)
                with self.assertRaises(ValueError):
                    retry_inputs.validate_previous_attempt(root, attempt, 5.14)

    def test_changed_ledger_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            attempt = self.fixture(root)
            (attempt / "training_budget.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "ledger changed"):
                retry_inputs.validate_previous_attempt(root, attempt, 5.14)

    def test_both_failed_attempts_are_charged_in_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            second = root / "outputs/goal-20261008-rebase-v1"
            second.mkdir()
            path = second / "training_budget.json"
            path.write_text(json.dumps({"result": "fail", "owned_remaining": [],
                "prior_gpu_hours": 5.18, "gpu_hours_charged": 0.07, "total_gpu_hours_charged": 5.25}))
            with patch.object(retry_inputs, "REBASE_ATTEMPT_SHA256", digest(path)):
                result = retry_inputs.validate_previous_attempt(root, second, 5.14)
            self.assertAlmostEqual(result["prior_gpu_hours"], 5.25)
            self.assertAlmostEqual(result["preceding_attempt"]["charge_gpu_hours"], 0.04)

    def test_export_failure_is_charged_after_both_earlier_attempts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            second = root / "outputs/goal-20261008-rebase-v1"
            third = root / "outputs/goal-20261008-swapfix-v1"
            second.mkdir()
            third.mkdir()
            path2, path3 = second / "training_budget.json", third / "training_budget.json"
            path2.write_text(json.dumps({"result": "fail", "owned_remaining": [],
                "prior_gpu_hours": 5.18, "gpu_hours_charged": 0.07, "total_gpu_hours_charged": 5.25}))
            path3.write_text(json.dumps({"result": "fail", "owned_remaining": [],
                "prior_gpu_hours": 5.25, "gpu_hours_charged": 0.3663, "total_gpu_hours_charged": 5.6163}))
            with patch.object(retry_inputs, "REBASE_ATTEMPT_SHA256", digest(path2)), \
                    patch.object(retry_inputs, "SWAPFIX_ATTEMPT_SHA256", digest(path3)):
                result = retry_inputs.validate_previous_attempt(root, third, 5.14)
            self.assertAlmostEqual(result["prior_gpu_hours"], 5.6163)
            self.assertAlmostEqual(result["preceding_attempt"]["prior_gpu_hours"], 5.25)


if __name__ == "__main__":
    unittest.main()
