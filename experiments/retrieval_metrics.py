"""Replay fixed full-library metrics; top20 reranking never alters rank21+."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def query_result(row, scores=None):
    original = row["ranking"]
    candidates = row["top20"]
    if len(original) != len(set(original)) or candidates != original[:20]:
        raise ValueError("Invalid frozen ranking/candidates")
    order = candidates
    unresolved = []
    if scores is not None:
        if set(scores) != set(candidates) or any(not math.isfinite(s) for s in scores.values()):
            raise ValueError("Require all and only 20 finite candidate scores")
        order = sorted(candidates, key=lambda video: -scores[video])
        unresolved = [[a, b] for i, a in enumerate(order) for b in order[i + 1:] if abs(scores[a] - scores[b]) <= 0.002]
    ranking = order + original[20:]
    positive = row["positive_video"]
    before, after = original.index(positive) + 1, ranking.index(positive) + 1
    assert ranking[20:] == original[20:]
    assert (before <= 100) == (after <= 100)
    if before > 20:
        assert before == after
    return {
        "cap_id": row["cap_id"], "positive_video": positive,
        "original_rank": before, "rank": after, "rank_change": before - after,
        "candidate_hit": before <= 20, "mrr20": 1 / after if after <= 20 else 0,
        "top20": order, "numerically_unresolved_pairs": unresolved,
    }


def summarize(rows):
    if not rows:
        raise ValueError("Empty evaluation")
    n = len(rows)
    recalls = {f"R@{k}": 100 * sum(r["rank"] <= k for r in rows) / n for k in [1, 5, 10, 100]}
    return {"queries": n, **recalls, "Rsum": sum(recalls.values()),
            "MRR@20": sum(r["mrr20"] for r in rows) / n,
            "candidate Recall@20": 100 * sum(r["candidate_hit"] for r in rows) / n,
            "improved_queries": sum(r["rank_change"] > 0 for r in rows),
            "worse_queries": sum(r["rank_change"] < 0 for r in rows),
            "unresolved_queries": sum(bool(r["numerically_unresolved_pairs"]) for r in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--split", choices=["development", "confirmation"], default="development")
    parser.add_argument("--selection-lock", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if args.split == "confirmation":
        if not args.selection_lock:
            raise ValueError("Confirmation requires prospective checkpoint/config selection lock")
        lock = json.loads(args.selection_lock.read_text())
        if lock["protocol_sha256"] != hashlib.sha256(args.protocol.read_bytes()).hexdigest():
            raise ValueError("Selection lock protocol hash mismatch")
    if hashlib.sha256(args.rankings.read_bytes()).hexdigest() != protocol["ranking_sha256"]:
        raise ValueError("Ranking hash mismatch")
    ranks = json.loads(args.rankings.read_text())
    selected = set(protocol[args.split])
    rows = [r for r in ranks["rows"] if r["cap_id"] in selected]
    assert len(rows) == len(selected)
    score_map = {}
    timings, memory = [], []
    if args.scores:
        for line in args.scores.read_text().splitlines():
            record = json.loads(line)
            query_id, video = record["cap_id"], record["video_id"]
            if query_id not in selected:
                raise ValueError("Score file includes queries outside requested split")
            group = score_map.setdefault(query_id, {})
            if video in group:
                raise ValueError("Duplicate query/video score")
            group[video] = record["score"]
            if "elapsed_s" in record:
                timings.append(record["elapsed_s"])
            if "peak_reserved_bytes" in record:
                memory.append(record["peak_reserved_bytes"])
        if set(score_map) != selected:
            raise ValueError("Missing query scores")
    results = [query_result(r, score_map[r["cap_id"]] if args.scores else None) for r in rows]
    payload = {"split": args.split, "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
               "ranking_sha256": protocol["ranking_sha256"], "metrics": summarize(results), "queries": results,
               "inference_s_sum": sum(timings) if timings else None,
               "peak_reserved_bytes": max(memory) if memory else None,
               "R@100_preserved_per_query": True,
               "scope": "200 frozen validation queries split by target video; exploratory single-seed, not independent system test"}
    if args.output.exists():
        raise FileExistsError("Refuse overwriting an evaluation result")
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "queries"}))


if __name__ == "__main__":
    main()
