"""Budget and GPU scope regressions; never allocates a GPU."""

import unittest
import json
from pathlib import Path
import tempfile

from run_confirmation import append, budget_seconds, repair_envelope, select_gpu


class SupervisorTests(unittest.TestCase):
    def test_health_event_accepts_existing_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            append(path, {"event": "health", "at": 123.5, "model": "released"})
            event = json.loads(path.read_text())
            self.assertEqual(event["at"], 123.5)
            self.assertEqual(event["event"], "health")

    def test_reserved_or_busy_gpu_never_selected(self):
        csv = "2, 32607, 32000, 0\n7, 32607, 32000, 100\n3, 32607, 31000, 0"
        self.assertEqual(select_gpu(csv)[1], 3)
        with self.assertRaises(RuntimeError):
            select_gpu("2, 32607, 32000, 0\n7, 32607, 1000, 0")

    def test_old_spend_and_ten_hour_limit(self):
        self.assertEqual(budget_seconds(6.138003225922584), 36000)
        self.assertAlmostEqual(budget_seconds(23), 3600)
        for prior in (-1, 24, 25, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                budget_seconds(prior)

    def test_repair_keeps_failure_charge_and_original_deadline(self):
        failed = {"result": "fail", "error": "dict() got multiple values for keyword argument 'at'",
                  "prior_gpu_hours": 6, "gpu_hours_charged": .1, "total_gpu_hours_charged": 6.1, "deadline_epoch": 1000}
        self.assertEqual(repair_envelope(6, failed, 500), (6.1, 1000))
        for changed in (dict(failed, result="running"), dict(failed, gpu_hours_charged=float("nan")),
                        dict(failed, total_gpu_hours_charged=6), dict(failed, deadline_epoch=499)):
            with self.assertRaises(ValueError):
                repair_envelope(6, changed, 500)


if __name__ == "__main__":
    unittest.main()
