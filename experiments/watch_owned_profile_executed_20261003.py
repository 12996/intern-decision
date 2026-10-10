"""Deadline safeguard for torchrun's separate-session worker; never target others."""

import argparse
import json
import os
from pathlib import Path
import signal
import time


def identity(pid, output):
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        return None
    command = (proc / "cmdline").read_bytes().split(b"\0")
    if os.fsencode(str(output / "run_goal_profile.py")) not in command or os.fsencode(str(output)) not in command:
        raise RuntimeError("Refuse unrelated or reused PID")
    return (proc / "stat").read_text().split()[21]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    original = identity(args.pid, args.output)
    deadline = json.loads((args.output / "profile_budget.json").read_text())["deadline_epoch"]
    time.sleep(max(0, deadline - time.time()))
    record = {"pid": args.pid, "deadline_epoch": deadline, "at": time.time(), "signals": []}
    if original is not None and identity(args.pid, args.output) == original:
        os.kill(args.pid, signal.SIGTERM)
        record["signals"].append("SIGTERM")
        time.sleep(1)
        if identity(args.pid, args.output) == original:
            os.kill(args.pid, signal.SIGKILL)
            record["signals"].append("SIGKILL")
    record["end_epoch"] = time.time()
    (args.output / "owned_worker_stop.json").write_text(json.dumps(record, indent=2) + "\n")
