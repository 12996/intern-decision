"""One-shot single-GPU supervisor with an inherited total budget and no retries."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def atomic_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def append(path, payload):
    with path.open("a") as handle:
        handle.write(json.dumps({"at": time.time(), **payload}) + "\n")
        handle.flush()


def select_gpu(csv):
    candidates = []
    for line in csv.strip().splitlines():
        index, total, free, utilization = [int(v.strip()) for v in line.split(",")]
        if index != 2 and free >= 22000 and utilization < 5:
            candidates.append((free, index, min(.85, (free - 2048) / total)))
    if not candidates:
        raise RuntimeError("blocked: no permitted idle GPU with22GiB free")
    return max(candidates)


def budget_seconds(prior):
    if not math.isfinite(prior) or prior < 0 or prior >= 24:
        raise ValueError("Cumulative budget exhausted or invalid")
    return min(10 * 3600, (24 - prior) * 3600)


def repair_envelope(original_prior, failed, now):
    charge = failed["gpu_hours_charged"]
    if (failed["result"] != "fail" or failed["error"] != "dict() got multiple values for keyword argument 'at'"
            or failed["prior_gpu_hours"] != original_prior or not math.isfinite(charge) or charge < 0
            or abs(failed["total_gpu_hours_charged"] - original_prior - charge) > 1e-9
            or failed["deadline_epoch"] <= now):
        raise ValueError("Cannot inherit failed confirmation envelope")
    return failed["total_gpu_hours_charged"], failed["deadline_epoch"]


def stop_owned(process):
    if process.poll() is None:
        # The process group was created here, and its leader is still our child.
        if os.getpgid(process.pid) != process.pid:
            raise RuntimeError("Owned process group identity changed")
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/zhaopu/intern-decision"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spawn", action="store_true")
    parser.add_argument("--repair-from", type=Path)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if not out.is_relative_to(root / "outputs"):
        raise ValueError("Output outside authorized project")
    if args.spawn:
        if (out / "confirmation_budget.json").exists():
            raise FileExistsError("No automatic retry or budget reset")
        with (out / "supervisor.log").open("x") as log:
            cmd = [sys.executable, str(Path(__file__).resolve()), "--root", str(root), "--out", str(out)]
            if args.repair_from:
                cmd += ["--repair-from", str(args.repair_from)]
            process = subprocess.Popen(cmd,
                cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"supervisor_pid": process.pid, "out": str(out)}))
        return
    import fcntl
    gate = (root / "runs/.charades-pilot.lock").open("a")
    fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
    budget_path = out / "confirmation_budget.json"
    if budget_path.exists():
        raise FileExistsError("Existing confirmation attempt")
    ledger = json.loads((root / "outputs/goal-20261008-resume25-v1/training_budget.json").read_text())
    if ledger["result"] != "pass" or ledger["owned_remaining"]:
        raise ValueError("Training not terminal")
    prior = ledger["total_gpu_hours_charged"]
    duration = budget_seconds(prior)
    if abs(prior - 6.138003225922584) > 1e-9:
        raise ValueError("Unexpected cumulative budget; re-audit before spending")
    inherited_deadline = None
    if args.repair_from:
        previous_path = root / "outputs/goal-20261008-confirm-v1/confirmation_budget.json"
        if args.repair_from.resolve() != previous_path or out != root / "outputs/goal-20261008-confirm-v2":
            raise ValueError("Only this specifically repaired terminal attempt may be inherited")
        if hashlib.sha256(previous_path.read_bytes()).hexdigest() != "39664ecbebb041aef5e7cf4c58f00f1bcea7cfa6e6de30076005b7465b993a58":
            raise ValueError("Prior failure receipt changed")
        previous = json.loads(previous_path.read_text())
        for pid in [previous["supervisor_pid"], *[r["launcher_pid"] for r in previous["models"]]]:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("Prior owned process still exists; no relaunch")
        prior, inherited_deadline = repair_envelope(prior, previous, time.time())
    manifest_path = out / "confirmation_manifest.json"
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if manifest_sha != "be6f62ea57ff09b282201667d6d83ae4464d301d784d2a8fd7757c7e701c46e9":
        raise ValueError("Prospective confirmation selection changed")
    if json.loads((out / "frame_transfer.json").read_text())["result"] != "pass":
        raise ValueError("Required frame staging incomplete")
    if shutil.disk_usage(root).free < 57 * 1024 ** 3:
        raise RuntimeError("blocked: disk below50GiB reserve plus6GiB cache/1GiB logs")
    repo = root / "Intern-Decision"
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
                   PYTHONPATH=f"{out}/code:{repo}/runtime/xtuner:{repo}:{root}:{repo}/third_party/xtuner")
    cpu = subprocess.run([str(root / ".venv-xt310/bin/python"), "-m", "unittest", "test_confirmation_protocol", "test_run_confirmation", "test_confirmation_restore", "test_confirmation_assets", "-v"],
        cwd=out / "code", env=cpu_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    (out / "cpu_tests.log").write_text(cpu.stdout)
    if cpu.returncode:
        raise RuntimeError("CPU protocol/supervisor tests failed")
    snapshot = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    free, gpu, fraction = select_gpu(snapshot)
    start = time.time()
    deadline = inherited_deadline or start + duration
    budget = {"result": "running", "supervisor_pid": os.getpid(), "gpu": gpu, "start_epoch": start,
        "deadline_epoch": deadline, "max_gpu_seconds": duration, "prior_gpu_hours": prior, "models": [],
        "manifest_sha256": manifest_sha, "queries": 136, "pairs_per_model": 620, "no_automatic_retry": True,
        "repair_from": str(args.repair_from) if args.repair_from else None,
        "deadline_inherited_not_reset": bool(inherited_deadline),
        "gpu2_forbidden": True, "reserve_gib": 50, "new_cache_max_gib": 6,
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / "code").glob("*.py")}}
    atomic_json(budget_path, budget)
    append(out / "events.jsonl", {"event": "launch", "gpu": gpu, "supervisor_pid": os.getpid()})
    result, gpu_started, gpu_ended = "fail", None, None
    try:
        for name in ("released", "step50", "step100"):
            if time.time() >= deadline or shutil.disk_usage(root).free < 50 * 1024 ** 3:
                raise TimeoutError("Time/disk gate before next fixed model")
            fresh = subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=index,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits"], text=True)
            select_gpu(fresh)
            repo = root / "Intern-Decision"
            code = out / "code"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MEM_FRACTION=str(fraction),
                INTERN_DECISION_REPO=str(repo), DECISION_ROOT=str(root), MEDIA_ROOT=str(root / "frames/id_pilot_frames_8"),
                PYTHONPATH=f"{code}:{repo}/runtime/xtuner:{repo}:{root}:{repo}/third_party/xtuner",
                XTUNER_HF_IMPL="1", XTUNER_USE_FA3="0", XTUNER_GC_ENABLE="1", XTUNER_ACTIVATION_OFFLOAD="0",
                HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                OMP_NUM_THREADS="2", PYTORCH_ALLOC_CONF="expandable_segments:True",
                CONFIRM_OUT=str(out), CONFIRM_MODEL=name, CONFIRM_DEADLINE=str(deadline), CONFIRM_MANIFEST_SHA256=manifest_sha)
            cmd = [str(root / ".venv-xt310/bin/python"), "-m", "torch.distributed.run", "--standalone",
                   "--nproc-per-node=1", str(code / "confirmation_worker.py")]
            with (out / f"worker-{name}.log").open("x") as log:
                if gpu_started is None:
                    gpu_started = time.time()
                child = subprocess.Popen(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                record = {"model": name, "launcher_pid": child.pid, "start_epoch": time.time(), "command": cmd}
                budget["models"].append(record)
                atomic_json(budget_path, budget)
                append(out / "events.jsonl", {"event": "model_start", "model": name, "launcher_pid": child.pid, "gpu": gpu})
                next_health = time.time()
                try:
                    while child.poll() is None:
                        now = time.time()
                        if now >= deadline or shutil.disk_usage(root).free < 50 * 1024 ** 3:
                            raise TimeoutError("Confirmation time/disk limit")
                        if now - record["start_epoch"] > 1800 and not (out / f"load-and-parity-{name}.json").exists():
                            raise TimeoutError("No model load/sentinel pass within30min")
                        if now >= next_health:
                            progress = json.loads((out / "progress.json").read_text()) if (out / "progress.json").exists() else None
                            if progress and progress["model"] != name:
                                progress = None
                            budget["health"] = {"at": now, "model": name, "launcher_pid": child.pid,
                                "disk_free_bytes": shutil.disk_usage(root).free, "progress": progress}
                            atomic_json(budget_path, budget)
                            append(out / "events.jsonl", dict(event="health", **budget["health"]))
                            next_health = now + 300
                        time.sleep(min(10, max(.01, deadline - now)))
                    record.update(exit_code=child.returncode, end_epoch=time.time())
                    if child.returncode != 0:
                        raise RuntimeError(f"{name} evaluation failed; no automatic retry")
                finally:
                    stop_owned(child)
            append(out / "events.jsonl", {"event": "model_complete", "model": name})
        gpu_ended = time.time()
        subprocess.run([str(root / ".venv-xt310/bin/python"), str(out / "code/confirmation_report.py"), "--out", str(out)],
                       cwd=root, check=True, timeout=300)
        result = "pass"
    except BaseException as exc:
        budget["error"] = str(exc)
        raise
    finally:
        end = time.time()
        charge = ((gpu_ended or end) - gpu_started) / 3600 if gpu_started is not None else 0
        budget.update(result=result, end_epoch=end, gpu_hours_charged=charge, total_gpu_hours_charged=prior + charge)
        atomic_json(budget_path, budget)
        append(out / "events.jsonl", {"event": "terminal", "result": result, "gpu_hours_charged": charge})
        fcntl.flock(gate, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
