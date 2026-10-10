#!/usr/bin/env python3
"""Detached 5-minute health monitor for one owned E03 resume.

It records liveness, step, finite loss, group loss, checkpoint, and disk.
It does not restart training. A disk or 12-hour limit stops only processes
whose cwd, command, or environment contains this attempt's work directory.

Official trainer.py installs no SIGTERM handler. Stopping the attempt does
not create a checkpoint. Verified checkpoint directories are left in place.
"""
from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

POLL_SECONDS = 300
RESERVE = 50 * 1024 ** 3
UNCERTAINTY = 1024 ** 3
WALL_LIMIT_S = 12 * 3600
WORK_ROOT = "/home/zhaopu/intern-decision/runs/"


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def alive(pid):
    if not pid:
        return False
    return Path(f"/proc/{pid}").exists()


def cmdline(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def owned(pid, work_dir):
    if not alive(pid) or pid == os.getpid():
        return False
    command = cmdline(pid)
    if "e03_resume_monitor.py" in command:
        return False
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        cwd = ""
    try:
        env = Path(f"/proc/{pid}/environ").read_bytes().decode("utf-8", "replace")
    except OSError:
        env = ""
    return work_dir in command or cwd == work_dir or cwd.startswith(work_dir + "/") or work_dir in env


def descendants(pid):
    children = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text(encoding="utf-8", errors="replace").split()
            ppid = int(fields[3])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(entry.name))
    found = []
    stack = [pid]
    while stack:
        current = stack.pop()
        for child in children.get(current, []):
            found.append(child)
            stack.append(child)
    return found


def last_step(work_dir):
    path = Path(work_dir) / "scaleup_steps.jsonl"
    last = None
    finite_after = 0
    rows = 0
    if not path.is_file():
        return last, finite_after, rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows += 1
        last = item
        step = item.get("step")
        if isinstance(step, int) and step > 750 and item.get("loss_finite") is True:
            finite_after += 1
    return last, finite_after, rows


def newest_checkpoint(work_dir):
    root = Path(work_dir)
    best = None
    best_mtime = None
    if not root.is_dir():
        return None
    for path in root.glob("*/*"):
        if path.is_dir() and (path.name.startswith("ckpt-step-") or path.name.startswith("hf-")):
            mtime = path.stat().st_mtime
            if best_mtime is None or mtime > best_mtime:
                best, best_mtime = str(path), mtime
    for path in root.glob("*/*/*"):
        if path.is_dir() and (path.name.startswith("ckpt-step-") or path.name.startswith("hf-")):
            mtime = path.stat().st_mtime
            if best_mtime is None or mtime > best_mtime:
                best, best_mtime = str(path), mtime
    return best


def disk_free():
    st = os.statvfs("/home")
    return st.f_bavail * st.f_frsize


def next_write(step, hf_bytes, dcp_bytes):
    """Bytes of the only remaining save: final HF, final DCP, HF transient, 1GiB.

    HF_INTERVAL and DCP_INTERVAL both land on step 1000 and on no earlier
    new step. Steps 800 and 900 write nothing.
    """
    if step >= 1000:
        return 0
    return hf_bytes + dcp_bytes + hf_bytes + UNCERTAINTY


def safe_stop(work_dir, launcher_pid, reason, status_path):
    targets = []
    if owned(launcher_pid, work_dir):
        targets.append(launcher_pid)
        targets.extend(pid for pid in descendants(launcher_pid) if owned(pid, work_dir))
    trainer = read_text_pid(Path(work_dir) / "trainer.pid")
    if owned(trainer, work_dir):
        targets.append(trainer)
        targets.extend(pid for pid in descendants(trainer) if owned(pid, work_dir))
    unique = []
    for pid in targets:
        if pid not in unique:
            unique.append(pid)
    record = {
        "action": "safe_stop",
        "reason": reason,
        "pids": unique,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint_on_signal": False,
        "preserved_checkpoints": True,
        "note": "SIGTERM does not save a new checkpoint. Official trainer.py has no signal handler. Existing checkpoint directories were not deleted.",
    }
    for pid in unique:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as exc:
            record.setdefault("errors", []).append(f"{pid}:{exc}")
    deadline = time.time() + 180
    while time.time() < deadline and any(alive(pid) for pid in unique):
        time.sleep(5)
    record["still_alive"] = [pid for pid in unique if alive(pid)]
    if record["still_alive"] and disk_free() < 2 * 1024 ** 3:
        for pid in record["still_alive"]:
            if owned(pid, work_dir):
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError as exc:
                    record.setdefault("errors", []).append(f"kill {pid}:{exc}")
        time.sleep(2)
        record["still_alive"] = [pid for pid in unique if alive(pid)]
    write_status(status_path, record)
    print(json.dumps(record), flush=True)
    return record


def read_text_pid(path):
    try:
        return int(Path(path).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0


def write_status(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    attempt = json.loads(Path(os.environ["E03_ATTEMPT"]).read_text(encoding="utf-8"))
    work_dir = attempt["work_dir"]
    if not work_dir.startswith(WORK_ROOT + "charades-id2b-e03-resume750-"):
        raise SystemExit(f"REFUSE_MONITOR_WORK_DIR {work_dir}")
    launcher_pid = int(attempt["launcher_pid"])
    started = float(attempt["started_epoch"])
    hf_bytes = int(attempt.get("hf_bytes") or os.environ["E03_HF_BYTES"])
    dcp_bytes = int(attempt.get("dcp_bytes") or os.environ["E03_DCP_BYTES"])
    status_path = os.environ["E03_MONITOR_STATUS"]
    while True:
        last, finite_after, rows = last_step(work_dir)
        step = int(last.get("step") or 750) if isinstance(last, dict) else 750
        trainer_pid = read_text_pid(Path(work_dir) / "trainer.pid")
        launcher_alive = alive(launcher_pid)
        trainer_alive = alive(trainer_pid) and owned(trainer_pid, work_dir)
        free = disk_free()
        upcoming = next_write(step, hf_bytes, dcp_bytes)
        groups = None if not isinstance(last, dict) else last.get("groups_cumulative")
        payload = {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "work_dir": work_dir,
            "launcher_pid": launcher_pid,
            "launcher_alive": launcher_alive,
            "trainer_pid": trainer_pid,
            "trainer_alive": trainer_alive,
            "step": None if not isinstance(last, dict) else last.get("step"),
            "loss": None if not isinstance(last, dict) else last.get("loss"),
            "loss_finite": None if not isinstance(last, dict) else last.get("loss_finite"),
            "groups_cumulative": groups,
            "finite_updates_after_750": finite_after,
            "new_rows": rows,
            "checkpoint": newest_checkpoint(work_dir),
            "disk_available_bytes": free,
            "next_write_bytes": upcoming,
            "next_write_at_step": None if step >= 1000 else 1000,
            "checkpoint_on_signal": False,
            "retention_policy": attempt.get("retention_policy"),
            "best_model": False,
            "elapsed_s": time.time() - started,
        }
        log = Path(attempt["log"])
        if log.is_file():
            text = log.read_text(encoding="utf-8", errors="replace")
            for needle in ("Traceback", "CUDA out of memory", "No space left", "REFUSE_"):
                if needle in text:
                    payload["error_needle"] = needle
                    break
        write_status(status_path, payload)
        if (not launcher_alive) and (not trainer_alive):
            payload["state"] = "exited"
            write_status(status_path, payload)
            print(json.dumps({"state": "exited", "step": payload["step"]}), flush=True)
            return
        if payload["elapsed_s"] >= WALL_LIMIT_S and step < 1000:
            safe_stop(work_dir, launcher_pid, "wall_clock_12h", status_path)
            return
        if step > 1000:
            safe_stop(work_dir, launcher_pid, "horizon_exceeded", status_path)
            return
        if upcoming and free < RESERVE + upcoming:
            safe_stop(work_dir, launcher_pid, "disk_below_reserve_plus_next_write", status_path)
            return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
