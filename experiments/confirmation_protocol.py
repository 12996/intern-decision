"""Freeze and replay the existing, video-disjoint confirmation evaluation."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(rankings, old, captions, proxy):
    caption_map = dict(line.split(" ", 1) for line in captions.splitlines() if line.strip())
    library = {key.split("#", 1)[0] for key in caption_map}
    selected = set(old["confirmation"])
    if selected & set(old["development"]):
        raise ValueError("Overlapping splits")
    if not .03 <= len(selected) / len(caption_map) <= .05:
        raise ValueError("Not 3-5% of full validation queries")
    proxy_videos = {p["video_id"] for p in proxy["pairs"]}
    proxy_queries = {p["cap_id"] for p in proxy["pairs"]}
    rows = []
    for r in rankings["rows"]:
        if r["cap_id"] not in selected:
            continue
        if r["cap_id"] in proxy_queries or r["positive_video"] in proxy_videos:
            raise ValueError("Confirmation positive exposed to proxy selection")
        if caption_map[r["cap_id"]] != r["query"] or r["positive_video"] != r["cap_id"].split("#", 1)[0]:
            raise ValueError("Caption/target mismatch")
        order = r["ranking"]
        if len(order) != len(library) or set(order) != library or r["top20"] != order[:20]:
            raise ValueError("Invalid full-library ranking")
        rank = order.index(r["positive_video"]) + 1
        if rank != r["positive_rank"]:
            raise ValueError("Original rank drift")
        rows.append({k: r[k] for k in ("cap_id", "query", "positive_video", "positive_rank", "top20")})
        rows[-1]["full_ranking_sha256"] = hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest()
    if {r["cap_id"] for r in rows} != selected or len(rows) != len(selected):
        raise ValueError("Missing/duplicate confirmation rows")
    rows.sort(key=lambda r: r["cap_id"])
    return {"version": "E05-confirmation-v1", "seed": 42, "full_validation_queries": len(caption_map),
        "queries": len(rows), "query_fraction": len(rows) / len(caption_map), "library_videos": len(library),
        "candidates_per_query": 20, "rows": rows,
        "primary_comparison": "proxy-selected step50 versus released, no confirmation checkpoint reselection",
        "secondary_comparison": "predeclared final100 diagnostic; cannot replace selected50 based on confirmation",
        "metrics": ["R@1", "R@5", "R@10", "R@100", "Rsum", "MRR@20", "candidate Recall@20"],
        "metric_exact_miss_shortcut": "When own positive is outside Top20, reranking cannot change any reported metric; retain those queries and original target ranks, omit their model forwards. This is a label-aware evaluation shortcut, not deployed inference latency or complete candidate rankings.",
        "ground_truth": "caption-owning video; same first-stage protocol, no semantic relabeling or injected positives",
        "ties": "stable original order for exact ties; flag pair gaps<=0.002",
        "uncertainty": "10000 bootstrap resamples of positive-video groups, seed42",
        "practical_acceptance": "step50 Rsum >= stronger(first-stage,released)+2pp and R@1 no worse than that same baseline; baseline tie R@1 then first-stage",
        "statistical_support": "primary paired Rsum 95% interval lower bound >0; otherwise report inconclusive or negative, not confirmed benefit",
        "limits": ["Selected from a pre-existing 200-query pool, not a uniform full-validation draw.",
                   "Unseen for this fine-tuning selection, but first-stage checkpoint selected on validation; not an independent system test.",
                   "Single training seed; no multi-seed replication.", "Top20 reranking cannot improve candidate coverage or R@100."]}


def query_result(row, scores=None):
    candidates = row["top20"]
    if len(candidates) != 20 or len(set(candidates)) != 20:
        raise ValueError("Invalid candidates")
    before = row["positive_rank"]
    if before <= 20 and candidates.index(row["positive_video"]) + 1 != before:
        raise ValueError("Invalid original target rank")
    if before > 20 and row["positive_video"] in candidates:
        raise ValueError("Miss contains injected positive")
    after = before
    unresolved = False
    if before <= 20:
        if scores is not None:
            if set(scores) != set(candidates) or any(not math.isfinite(s) for s in scores.values()):
                raise ValueError("Missing/extra/nonfinite scores")
            ranked = sorted(candidates, key=lambda v: -scores[v])
            after = ranked.index(row["positive_video"]) + 1
            unresolved = any(abs(scores[a] - scores[b]) <= .002 for i, a in enumerate(ranked) for b in ranked[i+1:])
    elif scores is not None:
        raise ValueError("Unexpected scores for metric-invariant miss")
    return {"cap_id": row["cap_id"], "positive_video": row["positive_video"], "original_rank": before,
        "rank": after, "rank_change": before - after, "candidate_hit": before <= 20,
        "mrr20": 1 / after if after <= 20 else 0, "numerically_unresolved": unresolved,
        "model_forwards_omitted_metric_invariant": before > 20}


def summarize(rows):
    if not rows:
        raise ValueError("Empty rows")
    recalls = {f"R@{k}": 100 * sum(r["rank"] <= k for r in rows) / len(rows) for k in (1, 5, 10, 100)}
    return {"queries": len(rows), **recalls, "Rsum": sum(recalls.values()),
        "MRR@20": sum(r["mrr20"] for r in rows) / len(rows),
        "candidate Recall@20": 100 * sum(r["candidate_hit"] for r in rows) / len(rows),
        "unresolved_queries": sum(r["numerically_unresolved"] for r in rows)}


def replay(manifest, scores=None):
    groups = defaultdict(dict)
    expected = {(r["cap_id"], v) for r in manifest["rows"] if r["positive_rank"] <= 20 for v in r["top20"]}
    if scores is not None:
        actual = set()
        for s in scores:
            key = s["cap_id"], s["video_id"]
            if key in actual:
                raise ValueError("Duplicate score")
            actual.add(key)
            groups[s["cap_id"]][s["video_id"]] = s["score"]
        if actual != expected:
            raise ValueError("Score coverage mismatch")
    rows = [query_result(r, groups[r["cap_id"]] if scores is not None and r["positive_rank"] <= 20 else None)
            for r in manifest["rows"]]
    return {"metrics": summarize(rows), "rows": rows}


def paired(before, after, draws=10000):
    old = {r["cap_id"]: r for r in before["rows"]}
    new = {r["cap_id"]: r for r in after["rows"]}
    if old.keys() != new.keys():
        raise ValueError("Paired query coverage differs")
    by_video = defaultdict(list)
    for key in sorted(old):
        a, b = old[key], new[key]
        if a["positive_video"] != b["positive_video"]:
            raise ValueError("Paired target drift")
        by_video[a["positive_video"]].append([
            *(int(b["rank"] <= k) - int(a["rank"] <= k) for k in (1, 5, 10, 100)), b["mrr20"] - a["mrr20"]])
    clusters = list(by_video.values())
    rng = random.Random(42)
    sampled = [[] for _ in range(6)]
    for _ in range(draws):
        rows = [r for group in rng.choices(clusters, k=len(clusters)) for r in group]
        values = [sum(r[i] for r in rows) / len(rows) for i in range(5)]
        values = [100 * v for v in values[:4]] + [values[4]] + [100 * sum(values[:4])]
        for bucket, value in zip(sampled, values):
            bucket.append(value)
    intervals = {}
    for name, values in zip(("R@1", "R@5", "R@10", "R@100", "MRR@20", "Rsum"), sampled):
        values.sort()
        intervals[name] = [values[int(draws * .025)], values[min(draws-1, int(draws * .975))]]
    return {"delta": {k: after["metrics"][k] - before["metrics"][k] for k in intervals},
        "paired_positive_video_bootstrap95": intervals, "video_groups": len(clusters),
        "improved": sum(new[k]["rank"] < old[k]["rank"] for k in old),
        "worsened": sum(new[k]["rank"] > old[k]["rank"] for k in old),
        "unchanged": sum(new[k]["rank"] == old[k]["rank"] for k in old)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--old-protocol", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = json.loads(args.old_protocol.read_text())
    if digest(args.rankings) != old["ranking_sha256"]:
        raise ValueError("First-stage ranking source changed")
    result = freeze(json.loads(args.rankings.read_text()), old, args.captions.read_text(), json.loads(args.proxy.read_text()))
    result["sources_sha256"] = {key: digest(getattr(args, key)) for key in ("rankings", "old_protocol", "captions", "proxy")}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"manifest_sha256": digest(args.output), "queries": result["queries"],
        "query_fraction": result["query_fraction"], "hit_queries": sum(r["positive_rank"] <= 20 for r in result["rows"])}))


if __name__ == "__main__":
    main()
