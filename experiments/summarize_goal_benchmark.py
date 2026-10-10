"""CPU-only final first-check summary; failures cannot certify an optimization."""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def records(name):
        return [json.loads(line) for line in (args.directory / name).read_text().splitlines()]
    budget = json.loads((args.directory / "benchmark_budget.json").read_text())
    assert budget["result"] != "running"
    scores = records("benchmark_scores.jsonl")
    checks = records("equivalence.jsonl")
    accepted = [r for r in scores if r["mode"] == "decision_position" and r.get("projection") == "reference_post_fsdp_fp32"]
    matching = [r for r in checks if r["mode"] == "decision_position" and r.get("projection") == "reference_post_fsdp_fp32"]
    assert len(accepted) == len(matching) == 3 and all(r["pass"] for r in matching)
    pair_s = [sum(r["timing"][name] for name in ["read_processor_hash_s", "transfer_sync_s", "forward_sync_s"]) for r in accepted]
    mean_s = statistics.mean(pair_s)
    output = {
        "first_check_result": budget["result"], "correctness_scope": "released, three fixed same-query positive/random/hard pairs; not full retrieval",
        "exact_reference_logit_parity": True, "raw_logit_max_abs_error": max(r["logit_abs_error"] for r in matching),
        "score_max_abs_error": max(r["score_abs_error"] for r in matching),
        "accepted_mode": "decision_position with historical post-FSDP FP32 head projection, no labels or CE",
        "accepted_pair_seconds": pair_s, "mean_pair_seconds": mean_s,
        "mean_vision_cuda_seconds": statistics.mean(r["timing"]["cuda_ms"]["vision"] / 1000 for r in accepted),
        "scoring_peak_reserved_bytes": max(r["peak_reserved_bytes"] for r in accepted),
        "rejected_equivalence_checks": [r for r in checks if not r["pass"]],
        "forward_attempts_including_failures": len(records("forward_attempts.jsonl")),
        "first_check_wall_seconds_including_idle_gaps": budget["elapsed_s"],
        "gpu_hours_charged_including_loads_failures_retries": budget["gpu_hours_charged"],
        "remaining_total_gpu_hours": 24 - budget["gpu_hours_charged"],
        "cost_extrapolations_not_promises": {
            "development_pairs_per_model": 64 * 20,
            "development_scoring_hours_per_model": 64 * 20 * mean_s / 3600,
            "all_200_pairs_per_model": 200 * 20,
            "all_200_scoring_hours_per_model": 200 * 20 * mean_s / 3600,
            "A_B_C_development_gpu_hours_B_and_C_only": 2 * 64 * 20 * mean_s / 3600,
        },
        "efficiency_gate": "fail: even B/C development alone exceeds remaining24h; do not train or silently shrink frozen evaluation",
        "hf800_new_forward_check": "not executed; historical12pair margins remain smoke only",
        "confirmation_results": "unopened", "new_training": "not started",
    }
    if args.output.exists():
        raise FileExistsError("Refuse overwrite")
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
