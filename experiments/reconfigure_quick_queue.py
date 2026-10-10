"""Stop only the verified CPU waiter, under the existing single-writer gate."""

import fcntl
import json
import os
from pathlib import Path
import signal
import time

from project_cleanup_20261003 import process_identity


ROOT = Path("/home/zhaopu/intern-decision")
OUT = ROOT / "outputs/goal-20261003-quick-v1"


def main():
    with (ROOT / "runs/.charades-pilot.lock").open("a") as gate:
        fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / "training_budget.json").exists():
            raise ValueError("Attempt already started; do not stop or overwrite its code")
        expected = json.loads((ROOT / "outputs/cleanup-20261003/plan.json").read_text())["queue"]
        current = process_identity(expected["pid"])
        if current != expected:
            raise ValueError("Original CPU waiter identity changed")
        deadline = json.loads((OUT / "queue_status.json").read_text())["deadline_epoch"]
        os.kill(current["pid"], signal.SIGTERM)
        for _ in range(50):
            if process_identity(current["pid"]) is None:
                break
            time.sleep(0.2)
        else:
            raise TimeoutError("Original waiter did not stop; do not launch another")
        if (OUT / "training_budget.json").exists():
            raise ValueError("Unexpected attempt appeared")
        report = {"stopped_cpu_waiter": current["pid"], "start_ticks": current["start_ticks"],
                  "original_deadline_epoch": deadline, "gpu_work_started": False,
                  "at": time.time(), "reason": "User-requested lower-storage policy; no deadline/budget reset"}
        (OUT / "queue_reconfigured.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
