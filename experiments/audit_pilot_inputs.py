"""Read-only CPU audit of the existing E04 training inputs and consumed rows."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    train_path = root / "outputs/goal-20261003-quick-v1/balanced_train.jsonl"
    val_path = root / "outputs/goal-20261008-resume25-v1/code/quick_validation_pairs_20261003.json"
    train_bytes = train_path.read_bytes()
    train_sha = hashlib.sha256(train_bytes).hexdigest()
    val_sha = hashlib.sha256(val_path.read_bytes()).hexdigest()
    if train_sha != "1b577306c8e15f497648d54e96f64823db4df87cf6bfbf401fae7bbb24ecd531":
        raise ValueError("Training source drift")
    if val_sha != "c226b985145344d7415c8e46f4e601f31fbf6147f669d144d53caa05c61ef355":
        raise ValueError("Validation source drift")
    train = [json.loads(line) for line in train_bytes.splitlines() if line.strip()]
    val = json.loads(val_path.read_text())
    train_queries = {r["id"].rsplit("#", 1)[0] for r in train}
    train_videos = {image.split("/", 1)[0] for r in train for image in r["images"]}
    train_texts = {r["state"]["query"] for r in train}
    logs = [root / "runs/charades-id2b-e04-balanced-20261008-192643/scaleup_steps.jsonl",
            root / "runs/charades-id2b-e04-balanced-20261008-213420/scaleup_steps.jsonl"]
    steps = {}
    hashes = {}
    for index, path in enumerate(logs):
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if (index == 0 and 1 <= row["step"] <= 25) or (index == 1 and 26 <= row["step"] <= 100):
                if row["step"] in steps:
                    raise ValueError("Duplicate optimizer step")
                steps[row["step"]] = row
    if set(steps) != set(range(1, 101)):
        raise ValueError("Missing optimizer steps")
    samples = [s for step in sorted(steps) for s in steps[step]["samples"]]
    print(json.dumps({"cpu_only": True, "train_sha256": train_sha, "val_sha256": val_sha,
        "log_sha256": hashes, "training_rows": len(train),
        "unique_training_pair_ids": len({r["id"] for r in train}),
        "consumed_samples": len(samples), "unique_consumed_pairs": len({s["id"] for s in samples}),
        "consumed_fraction_of_rows": len(samples) / len(train),
        "consumed_roles": dict(Counter(s["pair_type"] for s in samples)),
        "consumed_labels": dict(Counter(s["label"] for s in samples)),
        "consumed_token_range": [min(s["num_tokens"] for s in samples), max(s["num_tokens"] for s in samples)],
        "consumed_samples_over_2066": sum(s["num_tokens"] > 2066 for s in samples),
        "validation_query_id_overlap": len(train_queries & {q["cap_id"] for q in val["queries"]}),
        "validation_candidate_video_overlap": len(train_videos & {p["video_id"] for p in val["pairs"]}),
        "validation_query_text_overlap": len(train_texts & {p["query"] for p in val["pairs"]}),
        "text_overlap_alone_is_not_video_leakage": True,
        "last_total_consumed_samples": steps[100]["data_state"]["ints"]["total_consumed_samples"],
        "last_lr": steps[100]["lr"]}, indent=2))


if __name__ == "__main__":
    main()
