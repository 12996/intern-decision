"""Paired, predeclared confirmation comparison; never reselect checkpoints."""

import argparse
import json
from pathlib import Path

from confirmation_protocol import digest, paired, replay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    manifest = json.loads((out / "confirmation_manifest.json").read_text())
    methods = {"first_stage": replay(manifest)}
    for name in ("released", "step50", "step100"):
        rows = [json.loads(line) for line in (out / f"scores-{name}.jsonl").read_text().splitlines()]
        methods[name] = replay(manifest, rows)
        saved = json.loads((out / f"metrics-{name}.json").read_text())
        if methods[name]["metrics"] != saved["metrics"] or methods[name]["rows"] != saved["rows"]:
            raise ValueError("Saved metrics do not replay")
    baseline_name = max(("first_stage", "released"), key=lambda name:
                        (methods[name]["metrics"]["Rsum"], methods[name]["metrics"]["R@1"], name == "first_stage"))
    comparisons = {"step50_vs_released": paired(methods["released"], methods["step50"]),
                   "step50_vs_first_stage": paired(methods["first_stage"], methods["step50"]),
                   "step100_vs_released_diagnostic": paired(methods["released"], methods["step100"])}
    before, after = methods[baseline_name]["metrics"], methods["step50"]["metrics"]
    practical = after["Rsum"] - before["Rsum"] >= 2 and after["R@1"] >= before["R@1"]
    primary_lower = comparisons["step50_vs_released"]["paired_positive_video_bootstrap95"]["Rsum"][0]
    report = {"execution_result": "pass", "manifest_sha256": digest(out / "confirmation_manifest.json"),
        "metrics": {name: result["metrics"] for name, result in methods.items()}, "comparisons": comparisons,
        "stronger_baseline": baseline_name, "practical_acceptance_met": practical,
        "primary_statistical_support": primary_lower > 0,
        "improvement_goal_confirmed_on_this_subset": practical and primary_lower > 0,
        "selected_checkpoint_unchanged": "step50", "no_confirmation_checkpoint_reselection": True,
        "scope": {key: manifest[key] for key in ("queries", "full_validation_queries", "query_fraction", "library_videos")},
        "limitations": manifest["limits"] + [manifest["metric_exact_miss_shortcut"]],
        "scores_sha256": {name: digest(out / f"scores-{name}.jsonl") for name in ("released", "step50", "step100")}}
    destination = out / "confirmation_report.json"
    with destination.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
