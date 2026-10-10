"""CPU-only audit and prospective grouped evaluation freeze for the goal."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_groups(rows, used_videos, seed=42, target=64):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["positive_video"]].append(row["cap_id"])
    dev_videos = set(groups) & set(used_videos)
    remaining = sorted(set(groups) - dev_videos)
    random.Random(seed).shuffle(remaining)
    count = sum(len(groups[video]) for video in dev_videos)
    for video in remaining:
        if count >= target:
            break
        dev_videos.add(video)
        count += len(groups[video])
    return {
        "development": [r["cap_id"] for r in rows if r["positive_video"] in dev_videos],
        "confirmation": [r["cap_id"] for r in rows if r["positive_video"] not in dev_videos],
        "development_videos": sorted(dev_videos),
        "confirmation_videos": sorted(set(groups) - dev_videos),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--frozen-pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = root / "Intern-Decision"
    sys.path.insert(0, str(repo))
    from src.inputs.schema import compile_row

    run = root / "runs/charades-id2b-e03-20260929-224918"
    log = (run / "log.txt").read_text()
    match = re.search(r"data_path=([^\s]+)", log)
    if not match:
        raise ValueError("Actual loader path absent from original log")
    train_path = Path(match.group(1))
    config = json.loads((run / "training-config.json").read_text())
    if digest(train_path) != config["data_sha256"]:
        raise ValueError("Actual training file no longer matches logged hash")
    data = train_path.parent
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    ranks_path = data / "val_top20.json"
    ranks = json.loads(ranks_path.read_text())
    frozen = json.loads(args.frozen_pairs.read_text())
    for key, path in [("manifest_sha256", manifest_path), ("val_top20_sha256", ranks_path)]:
        if digest(path) != frozen["sources"][key]:
            raise ValueError(f"Diagnostic provenance mismatch: {key}")
    sources = manifest["sources"]
    train_captions = Path(sources["train_captions"])
    val_captions = Path(sources["val_captions"])
    train_videos = {line.split("#")[0] for line in train_captions.read_text().splitlines() if line}
    val_videos = {line.split("#")[0] for line in val_captions.read_text().splitlines() if line}
    if train_videos & val_videos:
        raise ValueError("Train/val video intersection")
    by_query = {row["cap_id"]: row for row in manifest["train"]}
    counts = collections.Counter()
    frames = root / "frames/id_pilot_frames_8"
    frame_names = set()
    with train_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            cap_id, kind = row["id"].rsplit("#", 1)
            source = by_query[cap_id]
            expected_video = source[{"pos": "positive_video", "rand": "random_negative_video", "hard": "hard_negative_video"}[kind]]
            expected_label = "yes" if kind == "pos" else "no"
            assert row["state"] == {"query": source["query"]}
            assert row["targets"] == {"noul": {"label": expected_label}}
            assert row["images"] == [f"{expected_video}/frame_{i:02d}.png" for i in range(8)]
            assert expected_video in train_videos and expected_video not in val_videos
            if kind != "pos":
                assert expected_video not in {source["positive_video"], *source.get("extra_positive_videos", [])}
            clean = compile_row(row, include_targets=False)
            supervised = compile_row(row, include_targets=True)
            assert clean.messages == supervised.messages
            assert clean.targets is None
            flipped = dict(row, targets={"noul": {"label": "no" if expected_label == "yes" else "yes"}})
            assert compile_row(flipped).messages == supervised.messages
            assert "<decision>" in clean.messages[-1]["content"]
            counts[kind] += 1
            frame_names.update(row["images"])
    missing = sorted(name for name in frame_names if not (frames / name).is_file())
    if missing:
        raise ValueError(f"Missing training frames: {missing[:3]}")
    selected = {row["cap_id"]: row for row in manifest["val"]}
    assert len(selected) == len(ranks["rows"]) == 200
    assert ranks["library_videos"] == len(val_videos) == 1334
    for row in ranks["rows"]:
        assert row["cap_id"] in selected
        assert row["query"] == selected[row["cap_id"]]["query"]
        assert row["positive_video"] == selected[row["cap_id"]]["positive_video"]
        assert len(row["ranking"]) == len(set(row["ranking"])) == 1334
        assert set(row["ranking"]) == val_videos
        assert row["top20"] == row["ranking"][:20]
        assert row["positive_rank"] == row["ranking"].index(row["positive_video"]) + 1
    used_videos = sorted({p["video_id"] for p in frozen["pairs"]})
    split = freeze_groups(ranks["rows"], used_videos)
    steps = []
    truncated_tail = False
    lines = (run / "scaleup_steps.jsonl").read_text().splitlines()
    for i, line in enumerate(lines):
        try:
            steps.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            truncated_tail = True
    source_files = ["src/inputs/schema.py", "src/inputs/masked.py", "src/model/qwen.py",
                    "src/model/size_config.py", "configs/training/qwen35.py", "runtime/xtuner/sitecustomize.py"]
    audit = {
        "result": "pass", "actual_loader_path": str(train_path), "train_sha256": digest(train_path),
        "manifest_sha256": digest(manifest_path), "ranking_sha256": digest(ranks_path),
        "caption_hashes": {"train": digest(train_captions), "val": digest(val_captions)},
        "train_queries": len(by_query), "train_videos": len(train_videos), "val_videos": len(val_videos),
        "train_val_video_intersection": [], "pair_counts": dict(counts), "training_frame_files": len(frame_names),
        "target_flip_preserves_all_compiled_prompts": True,
        "valid_original_step_rows": len(steps), "original_truncated_tail_ignored_not_modified": truncated_tail,
        "source_revision": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "xtuner_revision": subprocess.check_output(["git", "-C", str(repo / "third_party/xtuner"), "rev-parse", "HEAD"], text=True).strip(),
        "active_source_hashes": {name: digest(repo / name) for name in source_files},
        "differences_from_logged_training_sources": [name for name in source_files if digest(repo / name) != config["source_hashes"][name]],
        "released_revision_metadata": (root / "models/Intern-Decision-2B/.cache/huggingface/download/config.json.metadata").read_text().splitlines()[0],
        "consumed_pair_counts": dict(collections.Counter(s["pair_type"] for r in steps for s in r["samples"])),
        "consumed_target_counts": dict(collections.Counter(s["label"] for r in steps for s in r["samples"])),
        "final_50_group_losses": {
            kind: {"n": len(values), "mean_ce": sum(values) / len(values)}
            for kind in ["positive", "random", "hard"]
            if (values := [r["loss"] for r in steps[-50:] if any(s["pair_type"] == kind for s in r["samples"])])
        },
    }
    protocol = {
        "version": "goal-20261003-v1", "seed": 42, "split_policy": "positive-video groups; all previously diagnosed validation videos forced to development, then shuffled groups until >=64 queries",
        "used_diagnostic_videos": used_videos, **split,
        "ranking_source": str(ranks_path), "ranking_sha256": digest(ranks_path),
        "manifest_sha256": digest(manifest_path), "frozen_diagnostic_sha256": digest(args.frozen_pairs),
        "queries": 200, "library_videos": 1334, "candidates_per_query": 20,
        "frame_policy": "existing eight ordered uniform frames; no GT timestamps; MAX_PIXELS=16777216",
        "selection": "development Rsum; tie R@1, then earlier checkpoint; starting model competes",
        "metrics": ["R@1", "R@5", "R@10", "R@100", "Rsum", "MRR@20", "candidate Recall@20"],
        "rsum": "sum of four recalls in percentage points",
        "ground_truth": "caption-owning positive_video, matching original PRVR first-stage evaluation; no relabeling from same-text negatives",
        "tie_rule": "score descending; exact ties retain frozen original candidate order; flag |margin difference|<=0.002 as numerically unresolved",
        "beyond_top20": "preserve original order and assert R@100 unchanged",
        "confirmation_policy": "do not inspect scores/metrics until final checkpoint/config locked; original retriever selected on validation, not an independent system test",
        "acceptance": "confirmation Rsum >= stronger(A,B)+2pp and R@1 >= same baseline; baseline tie R@1 then A",
        "logit_abs_tolerance": 0.001, "score_abs_tolerance": 0.002,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, payload in [("input_audit.json", audit), ("evaluation_protocol.json", protocol)]:
        path = args.output / name
        if path.exists():
            raise FileExistsError(f"Refuse overwrite: {path}")
        path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({**{k: v for k, v in audit.items() if k != "active_source_hashes"},
                      "development_queries": len(split["development"]), "confirmation_queries": len(split["confirmation"])}))


if __name__ == "__main__":
    main()
