"""CPU tests: grouped holdout is deterministic and cannot split target videos."""

import unittest

from audit_goal_inputs import freeze_groups
from retrieval_metrics import query_result, summarize


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"cap_id": f"v{i}#enc#{j}", "positive_video": f"v{i}"}
                     for i in range(100) for j in range(2)]

    def test_grouped_disjoint_holdout(self):
        split = freeze_groups(self.rows, {"v99"})
        self.assertEqual(len(split["development"]), 64)
        self.assertEqual(len(split["confirmation"]), 136)
        self.assertIn("v99#enc#0", split["development"])
        self.assertIn("v99#enc#1", split["development"])
        self.assertFalse(set(split["development_videos"]) & set(split["confirmation_videos"]))
        self.assertEqual(set(split["development"]) | set(split["confirmation"]), {r["cap_id"] for r in self.rows})

    def test_deterministic_without_rank_results(self):
        split = freeze_groups(self.rows, {"v99"})
        changed = [dict(r, positive_rank=1 if i % 2 else 1000, score=999 - i) for i, r in enumerate(self.rows)]
        self.assertEqual(split, freeze_groups(changed, {"v99"}))

    def test_does_not_split_group_to_hit_target(self):
        rows = [{"cap_id": f"v{i}#enc#{j}", "positive_video": f"v{i}"}
                for i in range(30) for j in range(3)]
        split = freeze_groups(rows, set(), target=64)
        self.assertEqual(len(split["development"]), 66)

    def test_candidate_missing_positive_cannot_improve(self):
        ranking = [f"v{i}" for i in range(1334)]
        row = {"cap_id": "v99#enc#0", "positive_video": "v99", "ranking": ranking, "top20": ranking[:20]}
        result = query_result(row, {video: i for i, video in enumerate(row["top20"])})
        self.assertEqual(result["rank"], 100)
        self.assertEqual(result["mrr20"], 0)
        self.assertEqual(summarize([result])["R@100"], 100)

    def test_reranking_and_percentage_rsum(self):
        ranking = [f"v{i}" for i in range(1334)]
        row = {"cap_id": "v19#enc#0", "positive_video": "v19", "ranking": ranking, "top20": ranking[:20]}
        result = query_result(row, {video: i for i, video in enumerate(row["top20"])})
        self.assertEqual(result["rank"], 1)
        self.assertEqual(summarize([result])["Rsum"], 400)
        self.assertEqual(result["top20"][0], "v19")

    def test_exact_ties_stable_and_nonfinite_rejected(self):
        ranking = [f"v{i}" for i in range(1334)]
        row = {"cap_id": "v19#enc#0", "positive_video": "v19", "ranking": ranking, "top20": ranking[:20]}
        result = query_result(row, dict.fromkeys(row["top20"], 0))
        self.assertEqual(result["top20"], row["top20"])
        self.assertEqual(result["rank"], 20)
        for scores in [dict.fromkeys(row["top20"], float("nan")), {"v0": 1}]:
            with self.assertRaises(ValueError):
                query_result(row, scores)


if __name__ == "__main__":
    unittest.main()
