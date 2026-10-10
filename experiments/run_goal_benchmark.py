"""Single-writer, detached CPU supervisor for the bounded real-GPU check."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def select_gpu(csv_text):
    candidates = []
    for line in csv_text.splitlines():
        index, total, free = [int(v.strip()) for v in line.split(",")]
        if index != 2 and free >= 18000:
            candidates.append((free, index, min(0.85, (free - 1024) / total)))
    if not candidates:
        raise RuntimeError("blocked: no permitted GPU has 18000MiB free")
    return max(candidates)


def write_status(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/zhaopu/intern-decision"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-budget", action="store_true")
    args = parser.parse_args()
    root, out = args.root, args.output
    out.mkdir(parents=True, exist_ok=True)
    gate = (root / "runs/.charades-pilot.lock").open("a")
    fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
    budget_path = out / "benchmark_budget.json"
    if budget_path.exists() and not args.resume_budget:
        raise FileExistsError("Existing first-check budget: do not reset/relaunch automatically")
    protocol = json.loads((out / "evaluation_protocol.json").read_text())
    assert len(protocol["development"]) == 64 and len(protocol["confirmation"]) == 136
    audit = json.loads((out / "input_audit.json").read_text())
    assert audit["result"] == "pass"
    if os.statvfs(root).f_bavail * os.statvfs(root).f_frsize < 50 * 1024**3 + 512 * 1024**2:
        raise RuntimeError("blocked: /home below 50GiB reserve plus small outputs")
    started = time.time()
    if args.resume_budget:
        budget = json.loads(budget_path.read_text())
        old_pid = budget["supervisor_pid"]
        try:
            os.kill(old_pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("Refuse resume while original supervisor still exists")
        deadline = budget["deadline_epoch"]
        if deadline <= started:
            raise TimeoutError("Original first-check deadline expired; do not reset")
        budget.setdefault("prior_attempt_results", []).append({k: budget[k] for k in ["supervisor_pid", "result", "elapsed_s", "error"] if k in budget})
        budget.update(supervisor_pid=os.getpid(), result="running")
        for key in ["error", "end_epoch", "elapsed_s", "health"]:
            budget.pop(key, None)
    else:
        deadline = started + 1800
        budget = {"start_epoch": started, "deadline_epoch": deadline, "max_forward_attempts": 12,
                  "supervisor_pid": os.getpid(), "result": "running", "models": []}
    write_status(budget_path, budget)
    result = "fail"
    try:
        for name, checkpoint in [
            ("released", root / "models/Intern-Decision-2B"),
            ("hf800", root / "runs/charades-id2b-e03-20260929-224918/20260929225053/hf-800"),
        ]:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("First-check 1800second deadline")
            if os.statvfs(root).f_bavail * os.statvfs(root).f_frsize < 50 * 1024**3:
                raise RuntimeError("blocked: disk reserve crossed")
            gpu_csv = subprocess.check_output([
                "nvidia-smi", "--query-gpu=index,memory.total,memory.free", "--format=csv,noheader,nounits"], text=True)
            free, gpu, fraction = select_gpu(gpu_csv)
            repo = root / "Intern-Decision"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MEM_FRACTION=str(fraction),
                       INTERN_DECISION_REPO=str(repo), DECISION_ROOT=str(root),
                       MEDIA_ROOT=str(root / "frames/id_pilot_frames_8"),
                       PYTHONPATH=f"{repo}/runtime/xtuner:{repo}:{repo}/third_party/xtuner",
                       XTUNER_HF_IMPL="1", XTUNER_USE_FA3="0", XTUNER_GC_ENABLE="1",
                       XTUNER_ACTIVATION_OFFLOAD="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       TOKENIZERS_PARALLELISM="false", PYTORCH_ALLOC_CONF="expandable_segments:True",
                       GOAL_OUT=str(out), GOAL_DEADLINE=str(deadline), MODEL_NAME=name, CHECKPOINT=str(checkpoint))
            cmd = [str(root / ".venv-xt310/bin/python"), "-m", "torch.distributed.run", "--standalone",
                   "--nproc-per-node=1", str(root / "decision_scorer.py")]
            with (out / f"score-{name}.log").open("a") as log:
                process = subprocess.Popen(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                record = {"model": name, "gpu": gpu, "initial_free_mib": free, "mem_fraction": fraction,
                          "pid": process.pid, "start_epoch": time.time(), "command": cmd, "log": str(out / f"score-{name}.log")}
                budget["models"].append(record)
                write_status(budget_path, budget)
                print(json.dumps(record), flush=True)
                try:
                    while True:
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            raise TimeoutError("First-check deadline while scoring")
                        try:
                            code = process.wait(timeout=min(300, remaining))
                            break
                        except subprocess.TimeoutExpired:
                            ledger = out / "forward_attempts.jsonl"
                            budget["health"] = {"at": time.time(), "owned_pid": process.pid,
                                                "forward_attempts": len(ledger.read_text().splitlines()) if ledger.exists() else 0,
                                                "disk_free_bytes": os.statvfs(root).f_bavail * os.statvfs(root).f_frsize}
                            write_status(budget_path, budget)
                            if budget["health"]["disk_free_bytes"] < 50 * 1024**3:
                                raise RuntimeError("Disk reserve crossed during owned inference")
                except BaseException:
                    # Inference has no checkpoint to save; kill only this owned group.
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                    raise
                record.update(exit_code=code, end_epoch=time.time())
                write_status(budget_path, budget)
                if code != 0:
                    raise RuntimeError(f"{name} scoring failed with exit {code}; inspect same log")
        result = "pass"
    except BaseException as exc:
        budget["error"] = str(exc)
        raise
    finally:
        end = time.time()
        budget.update(result=result, end_epoch=end, elapsed_s=end - budget["start_epoch"],
                      gpu_hours_charged=budget.get("gpu_hours_charged", 0) + (end - started) / 3600)
        write_status(budget_path, budget)
        fcntl.flock(gate, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
