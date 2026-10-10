"""Read-only resume metadata and checkpoint-size gate. Does not load model weights."""

from __future__ import annotations

import json
import os
from pathlib import Path

RUN = Path("/home/zhaopu/intern-decision/runs/charades-id2b-e03-20260929-224918")
EXP = RUN / "20260929225053"
CKPT = EXP / "checkpoints/ckpt-step-750"
TRAINER = Path("/home/zhaopu/intern-decision/Intern-Decision/third_party/xtuner/xtuner/v1/train/trainer.py")


def df_avail(path: str) -> int:
    line = os.popen(f"df -B1 -P {path}").read().splitlines()[1]
    return int(line.split()[3])


def du(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def matching_lines(path: Path, needles: tuple[str, ...], limit: int = 30) -> list[str]:
    found = []
    for index, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        if any(needle in line for needle in needles):
            found.append(f"{index}:{line.strip()[:200]}")
            if len(found) >= limit:
                break
    return found


def main() -> None:
    hf = du(EXP / "hf-800")
    dcp = du(CKPT)
    remaining = 2 * hf + dcp
    reserve = 50 * 1024**3
    avail = df_avail("/home")
    payload = {
        "home_avail_bytes": avail,
        "root_avail_bytes": df_avail("/"),
        "data_avail_bytes": df_avail("/data"),
        "hf800_bytes": hf,
        "dcp750_bytes": dcp,
        "remaining_checkpoint_bytes": remaining,
        "reserve_bytes": reserve,
        "need_bytes": remaining + reserve,
        "shortfall_bytes": remaining + reserve - avail,
        "trainer_hits": matching_lines(TRAINER, ("consumed_samples", "def _save_hf", "hf-", "exist_ok", "shutil.rmtree")),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
