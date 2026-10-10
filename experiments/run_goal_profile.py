"""Second authorized diagnostic: independent <=6-forward/900s budget, no training."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new_budget(first, first_hash, now, pid):
    remaining = 24 * 3600 - first["gpu_hours_charged"] * 3600
    if remaining <= 0:
        raise TimeoutError("Original24GPUh total exhausted")
    return {"start_epoch": now, "deadline_epoch": now + min(900, remaining),
            "max_forward_attempts": 6, "prior_gpu_hours": first["gpu_hours_charged"],
            "first_budget_sha256": first_hash, "gpu_hours_charged": 0,
            "supervisor_pid": pid, "result": "running", "processes": []}


def check_resume(budget, now, attempts):
    if now >= budget["deadline_epoch"] or attempts >= 6:
        raise TimeoutError("Original second-round deadline/forward limit exhausted; do not reset")
    if budget["prior_gpu_hours"] + budget["gpu_hours_charged"] >= 24:
        raise TimeoutError("Original24GPUh total exhausted")


def supervise(args):
    import fcntl
    from run_goal_benchmark import select_gpu, write_status

    root, out = args.root.resolve(), args.output.resolve()
    first_out = root / "outputs/goal-20261003-v2"
    first_path = first_out / "benchmark_budget.json"
    first_hash = digest(first_path)
    first = json.loads(first_path.read_text())
    if first["result"] not in ("fail", "pass"):
        raise RuntimeError("First check is not terminal")
    if first_hash != "49f78acd32d8a10b599b232c07afe0beb46ee151289590c035979460642bf2c6":
        raise RuntimeError("First-check ledger changed; reconcile charge before launching")
    if digest(first_out / "evaluation_protocol.json") != "f3ee6629e24ad932018ee2aea7e87a4a6a58191367521c210add5beccc74a095":
        raise RuntimeError("Frozen protocol changed")
    with (root / "runs/.charades-pilot.lock").open("a") as gate:
        fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out.mkdir(parents=True, exist_ok=True)
        budget_path = out / "profile_budget.json"
        ledger = out / "forward_attempts.jsonl"
        attempts = len(ledger.read_text().splitlines()) if ledger.exists() else 0
        now = time.time()
        if args.resume_budget:
            budget = json.loads(budget_path.read_text())
            check_resume(budget, now, attempts)
            try:
                os.kill(budget["supervisor_pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("Previous supervisor still exists")
            if budget["first_budget_sha256"] != first_hash:
                raise RuntimeError("Prior ledger identity differs")
            budget.update(supervisor_pid=os.getpid(), result="running")
            budget.pop("error", None)
        else:
            if budget_path.exists() or attempts:
                raise FileExistsError("Second-round budget exists; no automatic reset")
            budget = new_budget(first, first_hash, now, os.getpid())
        write_status(budget_path, budget)
        record, process = None, None
        result = "fail"
        try:
            disk_free = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
            if disk_free < 50 * 1024**3 + 512 * 1024**2:
                raise RuntimeError("blocked: /home below50GiB plus bounded diagnostic outputs")
            gpu_csv = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.total,memory.free",
                                               "--format=csv,noheader,nounits"], text=True)
            free, gpu, fraction = select_gpu(gpu_csv)
            repo = root / "Intern-Decision"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), MEM_FRACTION=str(fraction),
                       INTERN_DECISION_REPO=str(repo), DECISION_ROOT=str(root),
                       MEDIA_ROOT=str(root / "frames/id_pilot_frames_8"),
                       PYTHONPATH=f"{repo}/runtime/xtuner:{root}:{repo}:{repo}/third_party/xtuner",
                       XTUNER_HF_IMPL="1", XTUNER_USE_FA3="0", XTUNER_GC_ENABLE="1",
                       XTUNER_ACTIVATION_OFFLOAD="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       TOKENIZERS_PARALLELISM="false", PYTORCH_ALLOC_CONF="expandable_segments:True",
                       GOAL_PROFILE_WORKER="1", GOAL_DEADLINE=str(budget["deadline_epoch"]))
            cmd = [str(root / ".venv-xt310/bin/python"), "-m", "torch.distributed.run", "--standalone",
                   "--nproc-per-node=1", str(Path(__file__).resolve()), "--root", str(root),
                   "--output", str(out), "--mode", args.mode]
            log_path = out / f"{args.mode}.log"
            with log_path.open("a") as log:
                process = subprocess.Popen(cmd, cwd=repo, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                record = {"mode": args.mode, "gpu": gpu, "pid": process.pid, "start_epoch": time.time(),
                          "initial_free_mib": free, "disk_free_bytes": disk_free,
                          "command": cmd, "log": str(log_path)}
                budget["processes"].append(record)
                write_status(budget_path, budget)
                print(json.dumps(record), flush=True)
                while True:
                    remaining = budget["deadline_epoch"] - time.time()
                    if remaining <= 0:
                        raise TimeoutError("Second-round900second wall deadline reached")
                    try:
                        code = process.wait(timeout=min(300, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        disk_free = os.statvfs(root).f_bavail * os.statvfs(root).f_frsize
                        budget["health"] = {"at": time.time(), "owned_pid": process.pid,
                                            "disk_free_bytes": disk_free,
                                            "forward_attempts": len(ledger.read_text().splitlines()) if ledger.exists() else 0}
                        write_status(budget_path, budget)
                        if disk_free < 50 * 1024**3:
                            raise RuntimeError("Disk reserve crossed during diagnostic")
                record["exit_code"] = code
                if code != 0:
                    raise RuntimeError(f"Diagnostic exit{code}; inspect same log, no automatic retry")
                result = "pass"
        except BaseException as exc:
            budget["error"] = str(exc)
            raise
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if record is not None:
                record["end_epoch"] = time.time()
                record.setdefault("exit_code", process.returncode)
            budget.update(result=result, end_epoch=time.time(),
                          gpu_hours_charged=sum((r["end_epoch"] - r["start_epoch"]) / 3600
                                                for r in budget["processes"]),
                          first_ledger_unchanged=digest(first_path) == first_hash)
            budget["total_gpu_hours_charged"] = budget["prior_gpu_hours"] + budget["gpu_hours_charged"]
            write_status(budget_path, budget)


def worker(args):
    import torch
    import torch.distributed as dist
    from decision_scorer import DecisionScorer, append_record

    if args.mode != "profile":
        raise ValueError("No evidence-selected candidate implemented yet")
    root, out = args.root, args.output
    first_out = root / "outputs/goal-20261003-v2"
    deadline = float(os.environ["GOAL_DEADLINE"])
    if time.time() >= deadline:
        raise TimeoutError("Budget exhausted before load")
    torch.manual_seed(42)
    loaded = time.perf_counter()
    scorer = DecisionScorer(root / "models/Intern-Decision-2B")
    load_seconds = time.perf_counter() - loaded
    pair = json.loads((first_out / "frozen_pairs.json").read_text())["pairs"][0]
    reference = next(r for r in map(json.loads, (first_out / "benchmark_scores.jsonl").read_text().splitlines())
                     if r["model"] == "released" and r["mode"] == "reference_ce" and r["pair_id"] == pair["id"])
    protocol = json.loads((first_out / "evaluation_protocol.json").read_text())
    ledger = out / "forward_attempts.jsonl"
    attempts = len(ledger.read_text().splitlines()) if ledger.exists() else 0
    if attempts >= 6 or time.time() >= deadline:
        raise TimeoutError("Second-round forward/time cap before preparation")
    prepared = scorer.prepare(pair)
    if prepared[2] != reference["input"]:
        raise ValueError("Frozen reference input changed")
    if time.time() >= deadline:
        raise TimeoutError("Second-round deadline after preparation")
    append_record(ledger, {"mode": args.mode, "pair_id": pair["id"], "at": time.time()})
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False, profile_memory=False, with_stack=False) as prof:
        result = scorer.forward(prepared, detail_profile=True)
    raw_error = max(abs(result["raw_logits"][k] - reference["raw_logits"][k]) for k in ("no", "yes"))
    score_error = abs(result["score"] - reference["score"])
    passed = raw_error <= protocol["logit_abs_tolerance"] and score_error <= protocol["score_abs_tolerance"]
    averages = list(prof.key_averages())
    def top(key):
        return [{"op": e.key, "self_cpu_ms": e.self_cpu_time_total / 1000,
                 "cpu_ms": e.cpu_time_total / 1000, "self_device_ms": e.self_device_time_total / 1000,
                 "device_ms": e.device_time_total / 1000, "calls": e.count}
                for e in sorted(averages, key=lambda e: getattr(e, key), reverse=True)[:20]]
    trace_path = out / f"profile-trace-{attempts + 1}.json"
    prof.export_chrome_trace(str(trace_path))
    result.update(pair_id=pair["id"], model="released", diagnostic_mode=args.mode,
                  load_seconds=load_seconds, cpu_threads=torch.get_num_threads(),
                  interop_threads=torch.get_num_interop_threads(),
                  raw_logit_abs_error=raw_error, score_abs_error=score_error, parity_pass=passed,
                  trace_path=str(trace_path), scorer_sha256=digest(out / "decision_scorer.py"),
                  runner_sha256=digest(Path(__file__).resolve()),
                  top_cpu_ops=top("self_cpu_time_total"), top_device_ops=top("self_device_time_total"))
    append_record(out / "profile_scores.jsonl", result)
    print(json.dumps(result), flush=True)
    if not passed:
        raise ValueError("Unchanged prospective tolerance failed")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/zhaopu/intern-decision"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["profile"], default="profile")
    parser.add_argument("--resume-budget", action="store_true")
    args = parser.parse_args()
    if os.environ.get("GOAL_PROFILE_WORKER") == "1":
        worker(args)
    else:
        supervise(args)
