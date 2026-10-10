"""CPU-only replay of frozen pilot scores; performs no model forward."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean, median

from quick_validation import metrics


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def paired_change(before, after):
    old = {row["cap_id"]: row["rr"] for row in before["rows"]}
    new = {row["cap_id"]: row["rr"] for row in after["rows"]}
    if old.keys() != new.keys():
        raise ValueError("Different query sets")
    deltas = [new[key] - old[key] for key in sorted(old)]
    rng = random.Random(42)
    draws = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(10000))
    return {"mean_rr_delta": mean(deltas), "paired_query_bootstrap95": [draws[250], draws[9749]],
            "improved": sum(d > 0 for d in deltas), "worsened": sum(d < 0 for d in deltas),
            "unchanged": sum(d == 0 for d in deltas), "bootstrap_seed": 42, "bootstrap_draws": 10000}


def describe_scores(manifest, rows):
    results = {row["pair_id"]: row for row in rows}
    groups = {}
    for kind in ("positive", "hard", "random"):
        subset = [results[p["id"]] for p in manifest["pairs"] if p["pair_type"] == kind]
        scores = [r["score"] for r in subset]
        probabilities = [r["p_yes_candidate_softmax_not_calibrated"] for r in subset]
        groups[kind] = {"n": len(subset), "mean_score": mean(scores), "median_score": median(scores),
                        "score_positive_count": sum(s > 0 for s in scores),
                        "p_yes_above_099_count": sum(p > .99 for p in probabilities),
                        "mean_full_vocab_diagnostic_ce": mean(r["scoring_full_vocab_ce_fp32_diagnostic"] for r in subset)}
    wins = {}
    for kind in ("hard", "random"):
        margins = []
        for query in manifest["queries"]:
            pairs = [p for p in manifest["pairs"] if p["cap_id"] == query["cap_id"]]
            positive = next(p for p in pairs if p["pair_type"] == "positive")
            margins.extend(results[positive["id"]]["score"] - results[p["id"]]["score"]
                           for p in pairs if p["pair_type"] == kind)
        wins[kind] = {"comparisons": len(margins), "wins": sum(m > 0 for m in margins),
                      "win_rate": mean(m > 0 for m in margins),
                      "near_ties_abs_le_0002": sum(abs(m) <= .002 for m in margins)}
    lengths = [r["input"]["num_tokens"] for r in rows]
    return {"groups": groups, "positive_negative_by_role": wins,
            "score_input_token_range": [min(lengths), max(lengths)],
            "score_inputs_over_training_limit_2066": sum(n > 2066 for n in lengths)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    manifest_path = root / "quick_validation_pairs_20261003.json"
    manifest = read_json(manifest_path)
    result = {"cpu_only": True, "not_full_retrieval": True, "single_seed": True, "inputs_sha256": {}, "steps": {}}
    result["inputs_sha256"][manifest_path.name] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    computed = {}
    for step, name in [(0, "e04_swapfix_scores0_20261008.jsonl"),
                       (50, "e04_resume_scores50_20261008.jsonl"),
                       (100, "e04_resume_scores100_20261008.jsonl")]:
        path = root / name
        result["inputs_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = read_rows(path)
        computed[step] = metrics(manifest, rows)
        replay = computed[step]
        if step:
            saved = read_json(root / f"e04_resume_metrics{step}_20261008.json")
            for key in ("macro_mrr5", "positive_negative_win_rate", "rows"):
                same = saved[key] == replay[key] if key == "rows" else math.isclose(saved[key], replay[key], rel_tol=0, abs_tol=1e-12)
                if not same:
                    raise ValueError(f"Metric replay mismatch at step {step}: {key}")
        result["steps"][step] = {"mrr5": replay["macro_mrr5"],
                                  "win_rate": replay["positive_negative_win_rate"],
                                  **describe_scores(manifest, rows)}
    result["paired_50_to_100"] = paired_change(computed[50], computed[100])
    result["interpretation_limits"] = [
        "Constructed 32-query / 160-pair development proxy; not 4% of dataset queries.",
        "Bootstrap conditions on this stratified selected set, not the full dataset.",
        "Candidate probabilities are uncalibrated; use unsquashed margins for ranking.",
        "Token counts do not prove train/scoring processor parity or training input truncation.",
        "Known-nonpositive candidates can contain unlabeled semantic positives.",
    ]
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
