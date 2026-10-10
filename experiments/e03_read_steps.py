"""Read scaleup_steps.jsonl without rewriting it.

A truncated final line is ignored. Any earlier malformed line still fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_steps(path: Path) -> list[dict]:
    lines = path.read_bytes().splitlines()
    rows = []
    for index, raw in enumerate(lines):
        if not raw.strip():
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise
    return rows


def main() -> None:
    path = Path(sys.argv[1])
    raw_lines = path.read_bytes().splitlines()
    rows = load_steps(path)
    steps = [row.get("step") for row in rows]
    print(
        json.dumps(
            {
                "path": str(path),
                "raw_lines": len(raw_lines),
                "valid_rows": len(rows),
                "ignored_final_malformed": len(raw_lines) != len(rows),
                "first_step": steps[0] if steps else None,
                "last_step": steps[-1] if steps else None,
                "steps_contiguous_from_one": steps == list(range(1, len(steps) + 1)),
                "nonfinite_losses": sum(1 for row in rows if row.get("loss_finite") is not True),
            }
        )
    )


if __name__ == "__main__":
    main()
