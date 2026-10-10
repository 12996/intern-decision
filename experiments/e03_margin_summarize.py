"""Summarize flushed E03 margin rows. Read-only. No model load."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path


def valid_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise
    return records


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def model_report(rows: list[dict]) -> dict:
    by_query: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row.get("control") != "matched":
            continue
        by_query[row["cap_id"]][row["pair_type"]] = row
    comparisons = []
    for cap_id, parts in by_query.items():
        if not {"positive", "random", "hard"} <= set(parts):
            continue
        positive = parts["positive"]["yes_no_margin_logit_yes_minus_no"]
        for kind in ("random", "hard"):
            negative = parts[kind]["yes_no_margin_logit_yes_minus_no"]
            comparisons.append(
                {
                    "cap_id": cap_id,
                    "against": kind,
                    "positive_margin": positive,
                    "negative_margin": negative,
                    "positive_wins": positive > negative,
                    "tie": positive == negative,
                }
            )
    groups = {}
    matched = [row for row in rows if row.get("control") == "matched"]
    for kind in ("positive", "random", "hard"):
        subset = [row for row in matched if row["pair_type"] == kind]
        groups[kind] = {
            "n": len(subset),
            "mean_margin": mean([row["yes_no_margin_logit_yes_minus_no"] for row in subset]),
            "mean_full_vocabulary_ce": mean([row["full_vocabulary_ce"] for row in subset]),
            "predictions": [row["predicted_label_yes_no"] for row in subset],
        }
    spans = []
    for cap_id, parts in by_query.items():
        if not {"positive", "random", "hard"} <= set(parts):
            continue
        margins = [parts[kind]["yes_no_margin_logit_yes_minus_no"] for kind in ("positive", "random", "hard")]
        spans.append({"cap_id": cap_id, "margin_span": max(margins) - min(margins)})
    reverses = []
    positives = {row["cap_id"]: row for row in matched if row["pair_type"] == "positive"}
    for row in rows:
        if row.get("control") != "reversed_frames":
            continue
        base = positives.get(row["cap_id"])
        if base is None:
            continue
        delta = abs(base["yes_no_margin_logit_yes_minus_no"] - row["yes_no_margin_logit_yes_minus_no"])
        reverses.append({"cap_id": row["cap_id"], "margin_abs_delta": delta, "unchanged": delta < 1e-5})
    margins = [row["yes_no_margin_logit_yes_minus_no"] for row in matched]
    losses = [row["full_vocabulary_ce"] for row in matched]
    finite = all(math.isfinite(value) for value in margins + losses)
    constant = len({round(value, 8) for value in margins}) <= 1
    return {
        "rows": len(matched),
        "wins": sum(1 for item in comparisons if item["positive_wins"]),
        "ties": sum(1 for item in comparisons if item["tie"]),
        "comparisons": comparisons,
        "groups": groups,
        "mean_positive_minus_negative_margin": mean(
            [item["positive_margin"] - item["negative_margin"] for item in comparisons]
        ),
        "image_margin_spans": spans,
        "reverse_controls": reverses,
        "finite": finite,
        "constant_margins": constant,
        "mask_failures": [
            row["id"] for row in matched if row.get("mask_check_passed") is not True
        ],
        "all_predictions_no": all(row["predicted_label_yes_no"] == "no" for row in matched) if matched else None,
    }


def main() -> None:
    out = Path(os.environ.get("OUTPUT", "/home/zhaopu/intern-decision/outputs/e03-check-20261002/summary.json"))
    frozen = json.loads(Path(os.environ["FROZEN_PAIRS"]).read_text(encoding="utf-8"))
    released = [row for row in valid_records(Path(os.environ["RELEASED_PARTIAL"])) if row.get("model") == "released"]
    tuned = [row for row in valid_records(Path(os.environ["HF800_PARTIAL"])) if row.get("model") == "hf800"]
    reports = {"released": model_report(released), "hf800": model_report(tuned)}
    reasons = []
    unclear = []
    flat_models = []
    for name, report in reports.items():
        if report["rows"] != 12 or len(report["comparisons"]) != 8:
            unclear.append(f"{name} scored {report['rows']} matched rows and {len(report['comparisons'])} comparisons")
        if report["rows"] and not report["finite"]:
            reasons.append(f"{name} has a non-finite score")
        if report["rows"] and report["constant_margins"]:
            reasons.append(f"{name} margins are constant")
        if report["mask_failures"]:
            reasons.append(f"{name} mask check failed for {report['mask_failures']}")
        flat = [item for item in report["image_margin_spans"] if item["margin_span"] < 1e-5]
        if report["image_margin_spans"] and len(flat) == len(report["image_margin_spans"]):
            flat_models.append(name)
            if report["reverse_controls"]:
                if all(item["unchanged"] for item in report["reverse_controls"]):
                    reasons.append(f"{name} reversed frames did not change the positive margin")
            else:
                unclear.append(f"{name} matched videos have identical margins and reverse-frame controls were not run")
    released_wins = reports["released"]["wins"]
    tuned_wins = reports["hf800"]["wins"]
    if reports["released"]["rows"] == 12 and reports["hf800"]["rows"] == 12:
        if released_wins >= 6 and tuned_wins <= 4:
            reasons.append(f"score-order degradation released {released_wins}/8 versus hf800 {tuned_wins}/8")
    controls_needed = bool(flat_models) and not reports["released"]["reverse_controls"]
    if reasons:
        status = "block"
    elif unclear:
        status = "unclear"
    else:
        status = "pass"
    payload = {
        "status": status,
        "reasons": reasons,
        "unclear": unclear,
        "controls_needed": controls_needed and status != "pass",
        "query_ids": [item["cap_id"] for item in frozen["queries"]],
        "models": reports,
        "note": "yes/no hard prediction is diagnostic only and is not the ranking gate",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "released_wins": released_wins, "hf800_wins": tuned_wins, "reasons": reasons, "unclear": unclear}))


if __name__ == "__main__":
    main()
