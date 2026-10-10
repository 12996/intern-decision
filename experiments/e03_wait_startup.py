#!/usr/bin/env python3
"""Block until three resumed optimizer steps are visible, or the launch fails."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

DEADLINE_S = int(os.environ.get("E03_STARTUP_DEADLINE_S", "2700"))


def load_rows(path):
    rows = []
    malformed = 0
    if not path.is_file():
        return rows, malformed
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            malformed += 1
    return rows, malformed


def sample_ids(row):
    return [sample.get("id") for sample in row.get("samples") or [] if isinstance(sample, dict)]


def expected_ids(history):
    found = {}
    for line in history.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = item.get("step")
        if step in (751, 752, 753) and step not in found:
            found[step] = sample_ids(item)
    return [found.get(step) for step in (751, 752, 753)]


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def restored_state_errors(text, history_info, event):
    errors = []
    if not isinstance(history_info, dict):
        errors.append("history_restore_missing")
    else:
        if history_info.get("max_step") != 750:
            errors.append("history_max_step")
        if history_info.get("last_step") != 750:
            errors.append("history_last_step")
        if not isinstance(history_info.get("skipped_future"), int) or history_info["skipped_future"] < 1:
            errors.append("history_future_not_skipped")
    if not isinstance(event, dict):
        errors.append("resumed_event_missing")
        return errors
    checkpoint = str(event.get("checkpoint") or "")
    if not checkpoint.endswith("ckpt-step-750"):
        errors.append("load_path")
    if event.get("total_step") != 1000:
        errors.append("total_step")
    scheduler = event.get("scheduler") or {}
    if scheduler.get("last_epoch") != 750:
        errors.append("scheduler_last_epoch")
    if scheduler.get("milestones") != [30]:
        errors.append("warmup_milestone")
    children = scheduler.get("children") or []
    if not any(isinstance(child, dict) and child.get("T_max") == 970 for child in children):
        errors.append("cosine_horizon")
    optimizer = event.get("optimizer") or {}
    if optimizer.get("empty") is True or not _finite(optimizer.get("step")) or float(optimizer.get("step")) < 750:
        errors.append("optimizer_step")
    if "cpu" not in str(optimizer.get("device") or ""):
        errors.append("optimizer_device")
    if not isinstance(event.get("data_state"), dict):
        errors.append("data_state")
    if "SCALEUP_RNG_RESTORED " not in text:
        errors.append("rng_restore")
    if "SCALEUP_SWAP_OFFLOAD " not in text:
        errors.append("cpu_moment_rebind")
    freeze = None
    for line in text.splitlines():
        if line.startswith("SCALEUP_FREEZE "):
            try:
                freeze = json.loads(line.split(" ", 1)[1])
            except json.JSONDecodeError:
                errors.append("freeze_json")
    if not isinstance(freeze, dict):
        errors.append("freeze_missing")
    else:
        for name in ("vision_tower", "multi_modal_projector"):
            item = freeze.get(name) or {}
            if item.get("trainable") != 0 or not isinstance(item.get("frozen"), int) or item["frozen"] <= 0:
                errors.append(name)
        language = freeze.get("language_model") or {}
        if not isinstance(language.get("trainable"), int) or language["trainable"] <= 0:
            errors.append("language_model")
    return errors


def process_snapshot(work_dir):
    found = []
    proc = Path("/proc")
    if not proc.is_dir():
        return found
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if work_dir in command or "xtuner.v1.train.cli.sft" in command or "torch.distributed.run" in command:
            try:
                env = (entry / "environ").read_bytes().decode("utf-8", "replace")
            except OSError:
                env = ""
            if work_dir in command or work_dir in env:
                found.append({"pid": pid, "cmd": command[:240]})
    return found


def assess(text, rows, events, expected, trainer_alive, launcher_alive):
    refuse = [line for line in text.splitlines() if "REFUSE_" in line]
    resumed = [row for row in rows if isinstance(row.get("step"), int) and row["step"] > 750]
    history_line = ""
    for line in text.splitlines():
        if line.startswith("SCALEUP_HISTORY_RESTORED "):
            history_line = line
    history_info = None
    if history_line:
        try:
            history_info = json.loads(history_line.split(" ", 1)[1])
        except json.JSONDecodeError:
            history_info = None
    event = next((item for item in events if item.get("event") == "resumed"), None)
    if len(resumed) >= 3:
        steps = resumed[:3]
        got_steps = [row.get("step") for row in steps]
        got_ids = [sample_ids(row) for row in steps]
        problems = []
        if got_steps != [751, 752, 753]:
            problems.append("step_numbers")
        if not all(row.get("loss_finite") is True and _finite(row.get("loss")) for row in steps):
            problems.append("loss")
        if got_ids != expected or any(not ids for ids in got_ids):
            problems.append("data_ids")
        if any(row.get("total_step") != 1000 for row in steps):
            problems.append("step_horizon")
        problems.extend(restored_state_errors(text, history_info, event))
        report = {
            "status": "startup_ok" if not problems else "fail",
            "problems": problems,
            "history": history_info,
            "resumed_event": event,
            "steps": [
                {
                    "step": row.get("step"),
                    "loss": row.get("loss"),
                    "loss_finite": row.get("loss_finite"),
                    "lr": row.get("lr"),
                    "total_step": row.get("total_step"),
                    "ids": sample_ids(row),
                }
                for row in steps
            ],
            "expected_ids": expected,
            "actual_ids": got_ids,
            "trainer_alive": trainer_alive,
            "launcher_alive": launcher_alive,
        }
        if problems:
            return report, 3
        if not trainer_alive:
            report["status"] = "fail"
            report["problems"] = ["trainer_exited_after_steps"]
            return report, 3
        return report, 0
    if refuse and not trainer_alive and not launcher_alive:
        return {"status": "blocked", "refuse": refuse[-8:]}, 2
    if not trainer_alive and not launcher_alive and ("Traceback" in text or "Error" in text or "REFUSE_" in text):
        return {"status": "fail", "refuse": refuse[-8:], "log_tail": text[-2000:]}, 3
    return None, None


def main():
    work = Path(sys_argv_work())
    log = Path(sys.argv[2])
    history = Path(sys.argv[3])
    deadline = time.time() + DEADLINE_S
    expected = expected_ids(history)
    if expected != [ids for ids in expected if ids]:
        print(json.dumps({"status": "fail", "problems": ["history_ids_missing"], "expected_ids": expected}, indent=2))
        return 3
    while time.time() < deadline:
        text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
        rows, _malformed = load_rows(work / "scaleup_steps.jsonl")
        events, _ignored = load_rows(work / "scaleup_events.jsonl")
        trainer_alive = False
        trainer_pid = None
        pid_path = work / "trainer.pid"
        if pid_path.is_file():
            try:
                trainer_pid = int(pid_path.read_text(encoding="utf-8").strip())
                trainer_alive = Path(f"/proc/{trainer_pid}").exists()
            except ValueError:
                trainer_alive = False
        launcher_pid = int(os.environ.get("E03_LAUNCHER_PID", "0") or 0)
        launcher_alive = Path(f"/proc/{launcher_pid}").exists()
        report, code = assess(text, rows, events, expected, trainer_alive, launcher_alive)
        if report is not None:
            gpu = None
            for line in text.splitlines():
                if line.startswith("SELECTED_GPU ") or line.startswith("LAUNCH_GPU "):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        gpu = int(parts[1])
            report.update(
                {
                    "work_dir": str(work),
                    "log": str(log),
                    "gpu": gpu,
                    "trainer_pid": trainer_pid,
                    "launcher_pid": launcher_pid,
                    "processes": process_snapshot(str(work)),
                }
            )
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return code
        if not launcher_alive and not trainer_alive and log.is_file() and time.time() > deadline - DEADLINE_S + 60:
            print(json.dumps({"status": "fail", "log_tail": text[-2000:], "work_dir": str(work)}, indent=2))
            return 3
        time.sleep(20)
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    print(json.dumps({"status": "startup_timeout", "log_tail": text[-2000:], "work_dir": str(work)}, indent=2))
    return 4


def sys_argv_work():
    return sys.argv[1]


import sys


if __name__ == "__main__":
    sys.exit(main())
