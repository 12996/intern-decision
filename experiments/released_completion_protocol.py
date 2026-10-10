"""Hash-checked partial released scoring and the original primary comparison."""

import json
from pathlib import Path

from confirmation_protocol import digest, paired, replay
from selected_eval_protocol import MANIFEST_SHA, partial_inputs


def verify_seed(manifest, seed, selected):
    replay(manifest, selected)
    selected_map = {r["pair_id"]: r for r in selected}
    expected_ids = [f"{r['cap_id']}#candidate#{v}" for r in manifest["rows"]
                    if r["positive_rank"] <= 20 for v in r["top20"]]
    if [r["pair_id"] for r in seed] != expected_ids[:len(seed)] or not 0 < len(seed) < len(expected_ids):
        raise ValueError("Seed must be a nonempty incomplete scoring prefix")
    old_inputs = partial_inputs(manifest, seed)
    for row in selected:
        if (row["model"] != "step50" or row["confirmation_manifest_sha256"] != MANIFEST_SHA
                or row["pair_id"] != f"{row['cap_id']}#candidate#{row['video_id']}"):
            raise ValueError("Selected comparison inputs changed")
    if any(selected_map[key]["input"] != identity for key, identity in old_inputs.items()):
        raise ValueError("Seed and selected inputs differ")
    return old_inputs, {key: row["input"] for key, row in selected_map.items()}


def report_completed(out):
    out = Path(out)
    if digest(out / "confirmation_manifest.json") != MANIFEST_SHA:
        raise ValueError("Confirmation protocol changed")
    manifest = json.loads((out / "confirmation_manifest.json").read_text())
    first = replay(manifest)
    methods = {"first_stage": first}
    hashes = {}
    for name in ("released", "step50"):
        path = out / f"scores-{name}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        methods[name] = replay(manifest, rows)
        saved = json.loads((out / f"metrics-{name}.json").read_text())
        if methods[name]["metrics"] != saved["metrics"] or methods[name]["rows"] != saved["rows"]:
            raise ValueError("Metric replay differs from saved result")
        hashes[name] = digest(path)
    primary = paired(methods["released"], methods["step50"])
    vs_first = paired(first, methods["step50"])
    stronger = max(("first_stage", "released"), key=lambda name:
        (methods[name]["metrics"]["Rsum"], methods[name]["metrics"]["R@1"], name == "first_stage"))
    before, after = methods[stronger]["metrics"], methods["step50"]["metrics"]
    practical = after["Rsum"] - before["Rsum"] >= 2 and after["R@1"] >= before["R@1"]
    supported = primary["paired_positive_video_bootstrap95"]["Rsum"][0] > 0
    return {"execution_result": "pass", "manifest_sha256": MANIFEST_SHA,
        "metrics": {name: method["metrics"] for name, method in methods.items()},
        "comparisons": {"step50_vs_released": primary, "step50_vs_first_stage": vs_first},
        "stronger_baseline": stronger, "practical_acceptance_met": practical,
        "primary_statistical_support": supported, "improvement_goal_confirmed_on_this_subset": practical and supported,
        "selected_checkpoint_unchanged": "step50", "no_confirmation_checkpoint_reselection": True,
        "step100_diagnostic": "not executed in this bounded baseline completion",
        "scope": {key: manifest[key] for key in ("queries", "full_validation_queries", "query_fraction", "library_videos")},
        "limitations": manifest["limits"] + [manifest["metric_exact_miss_shortcut"]], "scores_sha256": hashes}
