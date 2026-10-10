"""Validate immutable partial inputs without treating them as a full baseline."""

import json
import math
from pathlib import Path

from confirmation_protocol import digest, paired, replay


MANIFEST_SHA = "be6f62ea57ff09b282201667d6d83ae4464d301d784d2a8fd7757c7e701c46e9"


def partial_inputs(manifest, rows):
    expected = {f"{r['cap_id']}#candidate#{v}": (r["cap_id"], v)
                for r in manifest["rows"] if r["positive_rank"] <= 20 for v in r["top20"]}
    result = {}
    for row in rows:
        key = row["pair_id"]
        if (key not in expected or key in result or row["model"] != "released"
                or (row["cap_id"], row["video_id"]) != expected[key]
                or not math.isfinite(row["score"]) or row["confirmation_manifest_sha256"] != MANIFEST_SHA):
            raise ValueError("Invalid partial released input evidence")
        result[key] = row["input"]
    return result


def reserved_seconds(prior, training_max_seconds):
    if not math.isfinite(prior) or not 0 <= prior < 24 or not 0 <= training_max_seconds <= 14400:
        raise ValueError("Invalid concurrent budget envelope")
    available = (24 - prior) * 3600 - training_max_seconds
    if available < 1800:
        raise ValueError("Insufficient budget after reserving live training")
    return min(7200, available) - 10


def report_selected(out):
    out = Path(out)
    manifest = json.loads((out / "confirmation_manifest.json").read_text())
    if digest(out / "confirmation_manifest.json") != MANIFEST_SHA:
        raise ValueError("Confirmation manifest changed")
    rows = [json.loads(line) for line in (out / "scores-step50.jsonl").read_text().splitlines()]
    selected, first = replay(manifest, rows), replay(manifest)
    saved = json.loads((out / "metrics-step50.json").read_text())
    if selected["metrics"] != saved["metrics"] or selected["rows"] != saved["rows"]:
        raise ValueError("Selected-model metrics do not replay")
    comparison = paired(first, selected)
    delta = comparison["delta"]
    return {"execution_result": "pass", "selected_checkpoint": "step50", "no_checkpoint_reselection": True,
        "manifest_sha256": MANIFEST_SHA, "metrics": {"first_stage": first["metrics"], "step50": selected["metrics"]},
        "comparison_step50_vs_first_stage": comparison,
        "practical_acceptance_vs_first_stage": delta["Rsum"] >= 2 and delta["R@1"] >= 0,
        "statistical_support_vs_first_stage": comparison["paired_positive_video_bootstrap95"]["Rsum"][0] > 0,
        "primary_step50_vs_released": "unmeasured: released evaluation is incomplete",
        "original_stronger_baseline_acceptance": "unmeasured: do not substitute first_stage for stronger(first_stage,released)",
        "original_improvement_goal_confirmed": False,
        "scope": {key: manifest[key] for key in ("queries", "full_validation_queries", "query_fraction", "library_videos")},
        "limitations": manifest["limits"] + [manifest["metric_exact_miss_shortcut"]],
        "scores_sha256": digest(out / "scores-step50.jsonl")}
