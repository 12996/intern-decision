"""Print one terminal line for the E03 margin check, or nothing while it is healthy."""

from __future__ import annotations

import json
import time
from pathlib import Path

OUT = Path("/home/zhaopu/intern-decision/outputs/e03-check-20261002")
LAUNCHER = OUT / "launcher.log"
STATUS = OUT / "score-status.jsonl"
SCORES = OUT / "scores.jsonl"
SUMMARY = OUT / "summary-status.txt"


def alive() -> bool:
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            command = (path / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore")
        except OSError:
            continue
        if "e03_margin_score.py" in command or "launch_e03_margin_check.sh" in command:
            return True
    return False


def valid_count(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise
        count += 1
    return count


def main() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8", errors="ignore") if LAUNCHER.is_file() else ""
    status = STATUS.read_text(encoding="utf-8", errors="ignore") if STATUS.is_file() else ""
    if "CHECK_DONE" in launcher:
        summary = SUMMARY.read_text(encoding="utf-8", errors="ignore").strip() if SUMMARY.is_file() else ""
        print("DONE " + summary.replace("\n", " | "))
        return
    if "score_error" in status or "CUDA out of memory" in launcher:
        print("FAILED " + (status.strip().splitlines()[-1] if status.strip() else "cuda_or_score_error"))
        return
    running = alive()
    count = valid_count(SCORES)
    if not running:
        print(f"FAILED process_exited scores={count}")
        return
    started = None
    for line in launcher.splitlines():
        if line.startswith("DEADLINE_EPOCH "):
            started = int(line.split()[1]) - 3600
    if started is not None and count == 0 and time.time() - started > 1200:
        print("ACTION_REQUIRED stalled_before_first_score")
        return
    if SCORES.is_file() and count > 0 and time.time() - SCORES.stat().st_mtime > 900:
        print(f"ACTION_REQUIRED stalled scores={count}")


if __name__ == "__main__":
    main()
