"""CPU regressions for incomplete-baseline handling and concurrent reservation."""

import unittest

from selected_eval_protocol import MANIFEST_SHA, partial_inputs, reserved_seconds
from run_selected_evaluation import choose_independent, combined_charge


class SelectedEvalTests(unittest.TestCase):
    def fixture(self):
        manifest = {"rows": [{"cap_id": "q", "positive_rank": 1, "top20": ["a", "b"]}]}
        row = {"pair_id": "q#candidate#a", "cap_id": "q", "video_id": "a", "model": "released",
               "score": 1., "confirmation_manifest_sha256": MANIFEST_SHA, "input": {"hash": "x"}}
        return manifest, row

    def test_partial_input_evidence_does_not_require_or_invent_missing_scores(self):
        manifest, row = self.fixture()
        result = partial_inputs(manifest, [row])
        self.assertEqual(result, {"q#candidate#a": {"hash": "x"}})
        self.assertNotIn("q#candidate#b", result)

    def test_invalid_duplicate_and_wrong_model_partial_evidence_rejected(self):
        manifest, row = self.fixture()
        for rows in ([row, row], [dict(row, model="step50")], [dict(row, score=float("nan"))],
                     [dict(row, pair_id="outside")], [dict(row, confirmation_manifest_sha256="changed")]):
            with self.assertRaises(ValueError):
                partial_inputs(manifest, rows)

    def test_training_full_budget_is_reserved_before_evaluation(self):
        self.assertEqual(reserved_seconds(7.578377276460329, 14400), 7190)
        self.assertAlmostEqual(reserved_seconds(19, 14400), 3590)
        for prior in (float("nan"), float("inf"), -1, 20, 24):
            with self.assertRaises(ValueError):
                reserved_seconds(prior, 14400)
        with self.assertRaises(ValueError):
            reserved_seconds(7, 14401)

    def test_separate_gpu_excludes_training_forbidden_and_busy_devices(self):
        csv = "3, 32607, 32110, 0\n2, 32607, 32110, 0\n1, 32607, 32110, 0\n4, 32607, 1000, 100"
        self.assertEqual(choose_independent(csv, 3)[1], 1)
        with self.assertRaises(RuntimeError):
            choose_independent("3, 32607, 32110, 0\n2, 32607, 32110, 0", 3)

    def test_concurrent_charges_are_additive_not_a_reset_serial_budget(self):
        result = combined_charge(7, {"result": "running", "start_epoch": 100}, .5, 3700)
        self.assertEqual(result["combined_gpu_hours"], 8.5)
        result = combined_charge(7, {"result": "pass", "gpu_hours_charged": 2}, .5, 9999)
        self.assertEqual(result["combined_gpu_hours"], 9.5)
        self.assertTrue(result["training_terminal"])


if __name__ == "__main__":
    unittest.main()
