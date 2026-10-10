"""CPU checks for immutable score reuse and complete paired coverage."""

import unittest

from released_completion_protocol import verify_seed
from selected_eval_protocol import MANIFEST_SHA


class CompletionTests(unittest.TestCase):
    def fixture(self):
        manifest = {"rows": [{"cap_id": "q", "positive_video": "v0", "positive_rank": 1,
                              "top20": [f"v{i}" for i in range(20)]}]}
        selected = [{"pair_id": f"q#candidate#v{i}", "cap_id": "q", "video_id": f"v{i}",
                     "model": "step50", "score": float(i), "input": {"hash": str(i)},
                     "confirmation_manifest_sha256": MANIFEST_SHA} for i in range(20)]
        seed = [dict(row, model="released") for row in selected[:3]]
        return manifest, seed, selected

    def test_prefix_and_all_selected_inputs_validated(self):
        manifest, seed, selected = self.fixture()
        old, all_inputs = verify_seed(manifest, seed, selected)
        self.assertEqual(len(old), 3)
        self.assertEqual(len(all_inputs), 20)

    def test_empty_full_reordered_duplicate_seeds_rejected(self):
        manifest, seed, selected = self.fixture()
        for bad in ([], list(reversed(seed)), seed + seed[:1], [dict(r, model="released") for r in selected]):
            with self.assertRaises(ValueError):
                verify_seed(manifest, bad, selected)

    def test_incomplete_or_relabelled_selected_scores_rejected(self):
        manifest, seed, selected = self.fixture()
        for bad in (selected[:-1], [dict(r, model="released") for r in selected]):
            with self.assertRaises(ValueError):
                verify_seed(manifest, seed, bad)

    def test_reused_input_identity_drift_rejected(self):
        manifest, seed, selected = self.fixture()
        seed[0] = dict(seed[0], input={"hash": "changed"})
        with self.assertRaisesRegex(ValueError, "inputs differ"):
            verify_seed(manifest, seed, selected)


if __name__ == "__main__":
    unittest.main()
