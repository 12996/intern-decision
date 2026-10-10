"""Freeze four validation pairs before any E03 margin scoring.

The four queries are the first four rows of the frozen full-manifest
validation list. Rankings come from the frozen val_top20 file. Negatives
are validation videos only. This script does not score a model.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


MANIFEST = Path("/home/zhaopu/intern-decision/data/full-20260929/manifest.json")
TOP20 = Path("/home/zhaopu/intern-decision/data/full-20260929/val_top20.json")
CAPTIONS = Path(
    "/data/zhaopu/wang-2024-gmmformer-v2/data/prvr/charades/TextData/charadesval.caption.txt"
)
FRAMES = Path("/home/zhaopu/intern-decision/frames/id_pilot_frames_8")
OUTPUT = Path("/home/zhaopu/intern-decision/outputs/e03-check-20261002/frozen_pairs.json")
N_QUERIES = 4
SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_captions(path: Path) -> list[dict[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cap_id, text = line.split(" ", 1)
        rows.append({"cap_id": cap_id, "video_id": cap_id.split("#", 1)[0], "text": text})
    return rows


def frame_count(video_id: str) -> int:
    folder = FRAMES / video_id
    return sum(1 for index in range(8) if (folder / f"frame_{index:02d}.png").is_file())


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    top = json.loads(TOP20.read_text(encoding="utf-8"))
    if len(manifest["val"]) != 200 or len(top["rows"]) != 200:
        raise SystemExit(f"expected 200 val rows, manifest={len(manifest['val'])} top={len(top['rows'])}")
    by_id = {row["cap_id"]: row for row in top["rows"]}
    if set(by_id) != {row["cap_id"] for row in manifest["val"]}:
        raise SystemExit("val_top20 query ids differ from the frozen manifest")
    captions = parse_captions(CAPTIONS)
    text_to_videos: dict[str, set[str]] = {}
    caption_by_id = {}
    for row in captions:
        caption_by_id[row["cap_id"]] = row
        text_to_videos.setdefault(row["text"], set()).add(row["video_id"])
    selected = manifest["val"][:N_QUERIES]
    if len(selected) != N_QUERIES:
        raise SystemExit("manifest val prefix is short")
    rng = random.Random(SEED)
    pairs = []
    queries = []
    for example in selected:
        cap_id = example["cap_id"]
        ranked = by_id[cap_id]
        if ranked["query"] != example["query"] or ranked["positive_video"] != example["positive_video"]:
            raise SystemExit(f"frozen fields drifted for {cap_id}")
        if cap_id not in caption_by_id:
            raise SystemExit(f"caption missing for {cap_id}")
        if caption_by_id[cap_id]["text"] != example["query"]:
            raise SystemExit(f"caption text drifted for {cap_id}")
        if len(ranked["ranking"]) != 1334:
            raise SystemExit(f"ranking length {len(ranked['ranking'])} for {cap_id}")
        library = ranked["ranking"]
        if len(set(library)) != 1334:
            raise SystemExit(f"duplicate videos in ranking for {cap_id}")
        known = set(text_to_videos[example["query"]])
        known.add(example["positive_video"])
        if example["positive_video"] not in known:
            raise SystemExit(f"positive missing from known set for {cap_id}")
        hard = None
        hard_rank = None
        for rank, video_id in enumerate(library, start=1):
            if video_id in known:
                continue
            hard = video_id
            hard_rank = rank
            break
        if hard is None:
            raise SystemExit(f"no nonpositive candidate for {cap_id}")
        pool = sorted(video_id for video_id in set(library) if video_id not in known and video_id != hard)
        if not pool:
            raise SystemExit(f"empty random pool for {cap_id}")
        random_video = rng.choice(pool)
        roles = [
            ("positive", example["positive_video"], "yes", ranked["positive_rank"]),
            ("random", random_video, "no", library.index(random_video) + 1),
            ("hard", hard, "no", hard_rank),
        ]
        query_pairs = []
        for role, video_id, label, rank in roles:
            count = frame_count(video_id)
            if count != 8:
                raise SystemExit(f"{cap_id} {role} {video_id} has {count} frames")
            query_pairs.append(
                {
                    "id": f"{cap_id}#{role}",
                    "cap_id": cap_id,
                    "query": example["query"],
                    "video_id": video_id,
                    "pair_type": role,
                    "label": label,
                    "control": "matched",
                    "frame_count": count,
                    "library_rank": rank,
                }
            )
        videos = [item["video_id"] for item in query_pairs]
        if len(set(videos)) != 3:
            raise SystemExit(f"pair videos are not distinct for {cap_id}: {videos}")
        pairs.extend(query_pairs)
        queries.append(
            {
                "cap_id": cap_id,
                "query": example["query"],
                "positive_video": example["positive_video"],
                "known_positives": sorted(known),
                "random_video": random_video,
                "hard_video": hard,
                "hard_rank": hard_rank,
                "positive_rank": ranked["positive_rank"],
                "random_pool": len(pool),
            }
        )
    if len(pairs) != 12:
        raise SystemExit(f"expected 12 pairs, got {len(pairs)}")
    payload = {
        "protocol": {
            "queries": "first 4 rows of data/full-20260929/manifest.json val, the frozen 200",
            "random_negative": "Random(42) choice, one draw per query in that order, from sorted validation videos excluding known same-text positives and the hard video",
            "hard_negative": "highest-ranked validation video that is not a known same-text positive",
            "not_retrieval": True,
            "seed": SEED,
        },
        "sources": {
            "manifest_sha256": sha256_file(MANIFEST),
            "val_top20_sha256": sha256_file(TOP20),
            "captions_sha256": sha256_file(CAPTIONS),
        },
        "queries": queries,
        "pairs": pairs,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"pairs": len(pairs), "queries": [item["cap_id"] for item in queries], "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
