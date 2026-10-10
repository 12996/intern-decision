"""Freeze a small development-only ranking proxy before model scoring."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashed_order(values, prefix):
    return sorted(values, key=lambda value: hashlib.sha256(f"{prefix}|{value}".encode()).hexdigest())


def build_pairs(selection, protocol, manifest, rankings, captions, frames):
    by_id = {row["cap_id"]: row for row in rankings["rows"]}
    manifest_rows = {row["cap_id"]: row for row in manifest["val"]}
    known_by_text = {}
    for line in captions.splitlines():
        if line.strip():
            cap_id, text = line.split(" ", 1)
            known_by_text.setdefault(text, set()).add(cap_id.split("#", 1)[0])
    forbidden = set(protocol["confirmation_videos"])
    pairs, queries = [], []
    for selected in selection["rows"]:
        cap_id = selected["cap_id"]
        if cap_id not in protocol["development"] or cap_id in protocol["confirmation"]:
            raise ValueError("Non-development query")
        row = by_id[cap_id]
        original = manifest_rows[cap_id]
        for field in ["query", "positive_video"]:
            if row[field] != original[field]:
                raise ValueError("Frozen query drift")
        if row["positive_video"] != selected["positive_video"] or row["positive_rank"] != selected["first_stage_rank"]:
            raise ValueError("Frozen selection drift")
        ranking = row["ranking"]
        if len(ranking) != len(set(ranking)):
            raise ValueError("Duplicate library IDs")
        known = known_by_text[row["query"]] | {row["positive_video"]}
        pool = [video for video in ranking if video not in known and video not in forbidden]
        hard = pool[:2]
        random = hashed_order(set(pool) - set(hard), f"quick-val4-neg|42|{cap_id}")[:2]
        if len(hard) != 2 or len(random) != 2:
            raise ValueError("Insufficient candidates")
        roles = [("positive", row["positive_video"]), *[("hard", v) for v in hard], *[("random", v) for v in random]]
        # Pre-score order, rather than positive-first order, breaks exact ties.
        roles.sort(key=lambda item: ranking.index(item[1]))
        ids = []
        for order, (kind, video) in enumerate(roles):
            if video in forbidden:
                raise ValueError("Confirmation positive video used in development")
            hashes = []
            for i in range(8):
                path = frames / video / f"frame_{i:02d}.png"
                if not path.is_file():
                    raise ValueError(f"Missing frame: {path}")
                hashes.append(digest(path))
            pair_id = f"{cap_id}#{kind}#{video}"
            ids.append(pair_id)
            pairs.append({"id": pair_id, "cap_id": cap_id, "query": row["query"],
                          "video_id": video, "pair_type": kind, "label": "yes" if kind == "positive" else "no",
                          "library_rank": ranking.index(video) + 1, "candidate_order": order,
                          "frame_count": 8, "frame_sha256": hashes})
        queries.append({"cap_id": cap_id, "positive_video": row["positive_video"],
                        "known_positives": sorted(known), "pair_ids": ids,
                        "first_stage_rank": row["positive_rank"], "stratum": selected["stratum"]})
    if len(pairs) != 160 or len(queries) != 32 or len({q["positive_video"] for q in queries}) != 32:
        raise ValueError("Expected32queries/160pairs/32positivevideos")
    return {"status": "frozen_before_model_scoring", "seed": 42, "pairs": pairs, "queries": queries,
            "unique_videos": len({p["video_id"] for p in pairs}), "pair_scale_fraction": 0.04,
            "negative_policy": "Highest two known-nonpositive ranks and SHA256-fixed two random negatives; exclude confirmation-positive videos, same-text positives and duplicates",
            "tie_policy": "Stable first-stage order; score differences<=0.002 reported unresolved",
            "selection_metric": "constructed macroMRR@5; NOT full-retrieval Rsum", "gradient_training": False}


def metrics(manifest, results):
    by_id = {r["pair_id"]: r for r in results}
    if len(by_id) != len(results) or set(by_id) != {p["id"] for p in manifest["pairs"]}:
        raise ValueError("Missing/duplicate/extra scores")
    pair_map = {p["id"]: p for p in manifest["pairs"]}
    rows, grouped_ce = [], {}
    for query in manifest["queries"]:
        candidates = [pair_map[key] for key in query["pair_ids"]]
        scores = {p["id"]: by_id[p["id"]]["score"] for p in candidates}
        if any(not math.isfinite(s) for s in scores.values()):
            raise ValueError("Nonfinite score")
        ranked = sorted(candidates, key=lambda p: (-scores[p["id"]], p["candidate_order"]))
        positive = next(p for p in candidates if p["label"] == "yes")
        rank = next(i + 1 for i, p in enumerate(ranked) if p["id"] == positive["id"])
        margins = [scores[positive["id"]] - scores[p["id"]] for p in candidates if p["label"] == "no"]
        rows.append({"cap_id": query["cap_id"], "positive_video": query["positive_video"],
                     "positive_rank": rank, "rr": 1 / rank, "win_fraction": sum(m > 0 for m in margins) / 4,
                     "unresolved_comparisons": sum(abs(m) <= 0.002 for m in margins), "margins": margins})
        for p in candidates:
            result = by_id[p["id"]]
            if "scoring_full_vocab_ce_fp32_diagnostic" in result:
                grouped_ce.setdefault(p["pair_type"], []).append(result["scoring_full_vocab_ce_fp32_diagnostic"])
    return {"macro_mrr5": sum(r["rr"] for r in rows) / len(rows),
            "positive_negative_win_rate": sum(r["win_fraction"] for r in rows) / len(rows),
            "query_count": len(rows), "pair_count": len(results), "rows": rows,
            "grouped_scoring_full_vocab_ce_fp32_diagnostic_only": {k: {"n": len(v), "mean": sum(v) / len(v)} for k, v in grouped_ce.items()},
            "not_full_retrieval": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    if digest(args.protocol) != selection["source_protocol_sha256"]:
        raise ValueError("Protocol hash drift")
    data = args.root / "data/full-20260929"
    manifest_path, ranking_path = data / "manifest.json", data / "val_top20.json"
    manifest = json.loads(manifest_path.read_text())
    captions = Path(manifest["sources"]["val_captions"])
    result = build_pairs(selection, json.loads(args.protocol.read_text()), manifest,
                         json.loads(ranking_path.read_text()), captions.read_text(), args.root / "frames/id_pilot_frames_8")
    result["sources"] = {"selection": digest(args.selection), "protocol": digest(args.protocol),
                         "manifest": digest(manifest_path), "rankings": digest(ranking_path), "captions": digest(captions)}
    if args.output.exists():
        raise FileExistsError("Do not overwrite frozen candidates")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"sha256": digest(args.output), "queries": len(result["queries"]),
                      "pairs": len(result["pairs"]), "unique_videos": result["unique_videos"]}))


if __name__ == "__main__":
    main()
