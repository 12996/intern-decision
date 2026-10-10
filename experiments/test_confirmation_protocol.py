"""CPU regressions for query-denominator preservation and paired inference."""

import unittest

from confirmation_protocol import paired, query_result, replay
from retrieval_metrics import query_result as full_query_result


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.library = [f"v{i}" for i in range(1334)]

    def row(self, rank):
        return {"cap_id": f"v{rank-1}#enc#0", "positive_video": f"v{rank-1}",
                "positive_rank": rank, "top20": self.library[:20]}

    def test_miss_shortcut_is_exact_for_metrics(self):
        scores = {v: i for i, v in enumerate(self.library[:20])}
        for rank in (21, 100, 101, 1334):
            row = self.row(rank)
            compact = query_result(row)
            full = full_query_result(dict(row, ranking=self.library), scores)
            for key in ("rank", "mrr20", "candidate_hit"):
                self.assertEqual(compact[key], full[key])

    def test_hit_matches_full_ranking(self):
        row = self.row(20)
        scores = {v: i for i, v in enumerate(self.library[:20])}
        compact = query_result(row, scores)
        full = full_query_result(dict(row, ranking=self.library), scores)
        self.assertEqual(compact["rank"], full["rank"])
        self.assertEqual(compact["rank"], 1)

    def test_misses_remain_in_denominator(self):
        manifest = {"rows": [self.row(20), self.row(500)]}
        scores = [{"cap_id": self.row(20)["cap_id"], "video_id": v, "score": i}
                  for i, v in enumerate(self.library[:20])]
        result = replay(manifest, scores)
        self.assertEqual(result["metrics"]["queries"], 2)
        self.assertEqual(result["metrics"]["R@1"], 50)
        self.assertEqual(result["metrics"]["Rsum"], 200)
        self.assertEqual(result["metrics"]["MRR@20"], .5)

    def test_ties_and_invalid_scores(self):
        row = self.row(20)
        scores = dict.fromkeys(row["top20"], 0)
        self.assertEqual(query_result(row, scores)["rank"], 20)
        for bad in ({}, dict.fromkeys(row["top20"], float("nan"))):
            with self.assertRaises(ValueError):
                query_result(row, bad)
        with self.assertRaises(ValueError):
            query_result(self.row(21), scores)

    def test_coverage_rejects_duplicates_and_missing(self):
        row = self.row(1)
        manifest = {"rows": [row]}
        scores = [{"cap_id": row["cap_id"], "video_id": v, "score": 0} for v in row["top20"]]
        for bad in (scores[:-1], scores + scores[:1], scores + [{"cap_id": "other", "video_id": "other", "score": 0}]):
            with self.assertRaises(ValueError):
                replay(manifest, bad)

    def test_group_bootstrap_units_and_no_miss_optimism(self):
        row = self.row(20)
        manifest = {"rows": [row, dict(row, cap_id="samevideo#enc#1")]}
        before = replay(manifest)
        scores = [{"cap_id": r["cap_id"], "video_id": v, "score": i}
                  for r in manifest["rows"] for i, v in enumerate(r["top20"])]
        after = replay(manifest, scores)
        result = paired(before, after, draws=100)
        self.assertEqual(result["video_groups"], 1)
        self.assertEqual(result["delta"]["Rsum"], 300)
        self.assertEqual(result["paired_positive_video_bootstrap95"]["Rsum"], [300, 300])
        self.assertEqual(result["paired_positive_video_bootstrap95"]["R@100"], [0, 0])


if __name__ == "__main__":
    unittest.main()
