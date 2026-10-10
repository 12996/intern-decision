"""One-variable balanced-sampling fork, preserving every original training row."""

import argparse
import collections
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.source.read_bytes()
    if hashlib.sha256(data).hexdigest() != "a0e8ed904d294decda2495065dab98d02ad34bbf79b43a7ac310897cd33bdde7":
        raise ValueError("Original training rows changed")
    if args.output.exists():
        raise FileExistsError("Do not overwrite a training fork")
    entries = []
    counts = collections.Counter()
    for line in data.decode().splitlines():
        row = json.loads(line)
        label = row["targets"]["noul"]["label"]
        for replica in range(2 if label == "yes" else 1):
            h = hashlib.sha256(f"balanced42|{row['id']}|{replica}".encode()).hexdigest()
            entries.append((h, line))
            counts[label] += 1
    if counts != {"yes": 24816, "no": 24816}:
        raise ValueError("Unexpected original class counts")
    entries.sort()
    with args.output.open("w") as handle:
        for _, line in entries:
            handle.write(line + "\n")
    result = {"hypothesis": "Balance1:1 relevance supervision to test negative-dominant CE warning; no new targets/frames/loss",
              "seed": 42, "source_sha256": hashlib.sha256(data).hexdigest(),
              "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
              "rows": len(entries), "classes": dict(counts), "positive_duplication": 2,
              "algorithm": "Sort SHA256(balanced42|original_row_id|replica); preserve original row bytes",
              "fresh_optimizer_fork_not_DCP750_resume": True}
    args.output.with_suffix(".provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
