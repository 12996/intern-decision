"""Read-only source-host inventory for the pre-existing confirmation split."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    data = root / "data/full-20260929"
    source = json.loads((data / "manifest.json").read_text())
    ranks = json.loads((data / "val_top20.json").read_text())
    protocol = json.loads((root / "outputs/goal-20261003-quick-v1/code/evaluation_protocol_v2_20261003.json").read_text())
    proxy = json.loads((root / "outputs/goal-20261003-quick-v1/code/quick_validation_pairs_20261003.json").read_text())
    selected = [r for r in ranks["rows"] if r["cap_id"] in set(protocol["confirmation"])]
    hits = [r for r in selected if r["positive_rank"] <= 20]
    required = {v for r in hits for v in r["top20"]}
    previous = {p["video_id"] for p in proxy["pairs"]}
    frames = root / "frames/id_pilot_frames_8"
    missing = [v for v in sorted(required) if not all((frames / v / f"frame_{i:02d}.png").is_file() for i in range(8))]
    captions = Path(source["sources"]["val_captions"])
    print(json.dumps({"cpu_only": True, "confirmation_queries": len(selected),
        "full_validation_queries": source["sources"]["val_rows"],
        "query_fraction": len(selected) / source["sources"]["val_rows"],
        "library_videos": ranks["library_videos"], "candidate_hit_queries": len(hits),
        "metric_contributing_pairs_per_model": len(hits) * 20,
        "unique_metric_contributing_videos": len(required), "previously_cached_videos": len(required & previous),
        "uncached_metric_contributing_videos": len(required - previous),
        "missing_frame_videos": missing, "all_confirmation_top20_videos": len({v for r in selected for v in r["top20"]}),
        "caption_sha256": hashlib.sha256(captions.read_bytes()).hexdigest(),
        "ranking_sha256": hashlib.sha256((data / "val_top20.json").read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
