"""Detached single-writer supervisor; no retry/reset or unrelated process kill."""

import argparse
import fcntl
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


GIB = 1024 ** 3
REQUIRED_FREE = int(76.25 * GIB)


def required_free(args):
    return 51 * GIB if args.retry_source else REQUIRED_FREE


def smoke_passed(out, start, work, baseline_sha256, resume_step=0):
    path = Path(out) / "smoke_pass.json"
    if not path.exists():
        return False
    receipt = json.loads(path.read_text())
    proof = receipt.get("update_proof", {})
    at = receipt.get("at")
    after = proof.get("after_sha256")
    valid_hash = isinstance(after, str) and len(after) == 64 and all(c in "0123456789abcdef" for c in after)
    if not (receipt.get("pass") is True and receipt.get("step") == resume_step + 5
            and isinstance(at, (int, float)) and math.isfinite(at) and start <= at < start + 1800
            and receipt.get("work_dir") == str(work)
            and receipt.get("frozen_visual_unchanged") is True
            and receipt.get("finite_loss_and_gradient") is True
            and receipt.get("first_five_steps_are_part_of_the100_update_pilot") is True
            and proof.get("step") == resume_step + 5 and proof.get("full_hash_checked") is True
            and proof.get("update_detected") is True and valid_hash
            and proof.get("before_sha256") == baseline_sha256 and after != baseline_sha256):
        raise ValueError("Invalid five-step smoke receipt; continuation forbidden")
    if resume_step and not (receipt.get("resume_step") == resume_step and receipt.get("updates_since_restore") == 5
                            and receipt.get("total_step") == 100 and proof.get("updates_since_restore") == 5):
        raise ValueError("Invalid resumed five-step receipt; continuation forbidden")
    return True


def attempt_smoke(out, start, work, retry, resume=None):
    if retry is None:
        return True
    baseline = retry["restore"]["weights"]["language_model"]
    if resume is not None:
        restore_path = Path(out) / "initial_restore.json"
        if not restore_path.exists():
            return False
        restore = json.loads(restore_path.read_text())
        if not (restore.get("pass") is True and restore.get("step") == resume["step"]
                and restore.get("work_dir") == str(work) and restore.get("total_step") == 100
                and restore.get("all_model_optimizer_entries_exact") is True
                and restore.get("scheduler_data_rng_exact") is True):
            raise ValueError("Invalid post-update restore receipt")
        baseline = restore["weights"]["language_model"]
    return smoke_passed(out, start, work, baseline, resume_step=resume["step"] if resume else 0)


def atomic_json(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, path)


def identity(pid):
    try:
        path = Path("/proc") / str(pid)
        fields = (path / "stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "ppid": int(fields[1]), "start_ticks": int(fields[19]),
                "cmdline": (path / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def discover_owned(parent, token, previous):
    records = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            record = identity(int(path.name))
            if record:
                records[record["pid"]] = record
    roots = {parent["pid"]}
    live_parent = records.get(parent["pid"])
    if live_parent is None or live_parent["start_ticks"] != parent["start_ticks"]:
        roots = set()
    for old in previous:
        current = records.get(old["pid"])
        if current and current["start_ticks"] == old["start_ticks"]:
            roots.add(old["pid"])
    changed = True
    while changed:
        changed = False
        for pid, record in records.items():
            if pid not in roots and record["ppid"] in roots:
                roots.add(pid)
                changed = True
    # torchrun may put workers in independent process groups; retain their exact
    # PID/start identity, rather than assuming killpg(parent) covers them.
    return [records[pid] for pid in roots if pid in records and
            (pid == parent["pid"] or token in records[pid]["cmdline"] or records[pid]["ppid"] in roots)]


def stop_owned(records):
    for sig in [signal.SIGTERM, signal.SIGKILL]:
        for record in sorted(records, key=lambda r: r["pid"], reverse=True):
            current = identity(record["pid"])
            if current and current["start_ticks"] == record["start_ticks"] and current["cmdline"] == record["cmdline"]:
                try:
                    os.kill(record["pid"], sig)
                except ProcessLookupError:
                    pass
        time.sleep(3 if sig == signal.SIGTERM else 1)


def choose_gpu():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    candidates = []
    for line in output.splitlines():
        index, free, util = map(int, line.split(","))
        if index != 2 and free >= 26720 and util <= 5:
            candidates.append((free, index))
    if not candidates:
        raise RuntimeError("blocked: no permittedGPU with26720MiB free/util<=5")
    return max(candidates)


def supervise(args):
    root, out = args.root.resolve(), args.output.resolve()
    code, repo = out / "code", root / "Intern-Decision"
    budget_path = out / "training_budget.json"
    if budget_path.exists():
        raise FileExistsError("Pilot already attempted; do not reset its budget")
    first_path = root / "outputs/goal-20261003-v2/benchmark_budget.json"
    second_path = root / "outputs/goal-20261003-profile-v1/profile_budget.json"
    if hashlib.sha256(first_path.read_bytes()).hexdigest() != "49f78acd32d8a10b599b232c07afe0beb46ee151289590c035979460642bf2c6":
        raise ValueError("Historical first ledger changed")
    first, second = json.loads(first_path.read_text()), json.loads(second_path.read_text())
    if first["result"] not in ["fail", "pass"] or second["result"] not in ["fail", "pass"]:
        raise ValueError("Earlier compute is not terminal")
    prior = first["gpu_hours_charged"] + second["gpu_hours_charged"]
    retry = None
    if args.retry_source:
        from retry_inputs import validate_retry_source
        retry = validate_retry_source(root, args.retry_source)
        if abs(retry["ledger"]["prior_gpu_hours"] - prior) > 1e-9:
            raise ValueError("Historical GPU accounting mismatch")
        prior = retry["prior_gpu_hours"]
    previous_attempt = None
    if args.fresh_destination_baseline != (args.previous_attempt is not None):
        raise ValueError("Authorized rebaseline requires explicit previous-attempt accounting")
    if args.previous_attempt:
        if retry is None:
            raise ValueError("Rebaseline requires the verified old step0/input source")
        from retry_inputs import validate_previous_attempt
        previous_attempt = validate_previous_attempt(root, args.previous_attempt, prior)
        prior = previous_attempt["prior_gpu_hours"]
    resume = None
    if args.resume_source:
        from resume_state import resume_source
        resume = resume_source(root, args.resume_source)
        if (not retry or not args.fresh_destination_baseline or args.previous_attempt.resolve() != resume["source"]
                or abs(prior - resume["ledger"]["total_gpu_hours_charged"]) > 1e-9):
            raise ValueError("Continuation requires full failed-attempt accounting and verified old inputs")
        if not (out / "resume_reference.json").is_file():
            raise ValueError("Missing independently loaded CPU checkpoint reference")
    if prior >= 24:
        raise TimeoutError("24GPUh exhausted")
    with (root / "runs/.charades-pilot.lock").open("a") as gate:
        fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Start with one restore-test DCP+cache+misc+50GiB reserve. Subsequent
        # saves inspect actual free space; skip a new DCP rather than deleting
        # the previous durable last or lowering the reserve. Best still validates.
        free_disk = shutil.disk_usage(root).free
        if free_disk < required_free(args):
            raise RuntimeError(f"blocked: need{required_free(args) / GIB:.2f}GiB free for checkpoint/cache/reserve, have{free_disk / GIB:.3f}")
        ram_kib = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")))
        if ram_kib < 32 * 1024 ** 2:
            raise RuntimeError("blocked: insufficient availableRAM")
        free_mib, gpu = choose_gpu()
        work = root / "runs" / time.strftime("charades-id2b-e04-balanced-%Y%m%d-%H%M%S")
        if work.exists():
            raise FileExistsError(work)
        work.mkdir()
        start = time.time()
        deadline = start + (24 - prior) * 3600 - 10
        record = {"result": "running", "phase": "setup_then_baseline_then_training", "start_epoch": start,
                  "deadline_epoch": deadline, "prior_gpu_hours": prior, "gpu_hours_charged": 0,
                  "supervisor_pid": os.getpid(), "gpu": gpu, "free_mib_at_launch": free_mib,
                  "disk_free_bytes_at_launch": free_disk, "work_dir": str(work), "out": str(out),
                  "max_optimizer_updates": 100, "max_training_phase_seconds": 14400,
                  "no_automatic_retry": True, "health_interval_seconds": 300,
                  "storage_policy": {"start_required_gib": required_free(args) / GIB, "reserve_gib": 50,
                                     "best_export_max_gib": 9, "dcp_max_gib": 23,
                                     "dcp_save_requires_free_gib": 74,
                                     "skip_dcp_if_insufficient_space_keep_previous_last": True,
                                     "final_resume_step_may_precede_final_weights": True},
                  "first_ledger_sha256": hashlib.sha256(first_path.read_bytes()).hexdigest(),
                  "second_ledger_sha256": hashlib.sha256(second_path.read_bytes()).hexdigest(),
                  "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob("*.py")}}
        record["baseline_policy"] = "fresh_destination" if args.fresh_destination_baseline else "historical_replay"
        if previous_attempt:
            record["previous_attempt_accounting"] = previous_attempt
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), INTERN_DECISION_REPO=str(repo),
                   DECISION_ROOT=str(root), MEDIA_ROOT=str(root / "frames/id_pilot_frames_8"),
                   MODEL_PATH=str(root / "models/Intern-Decision-2B"), DATA_PATH=str(out / "balanced_train.jsonl"),
                   WORK_DIR=str(work), QUICK_OUT=str(out), QUICK_MANIFEST=str(code / "quick_validation_pairs_20261003.json"),
                   QUICK_MANIFEST_SHA256="c226b985145344d7415c8e46f4e601f31fbf6147f669d144d53caa05c61ef355",
                   QUICK_GLOBAL_DEADLINE=str(deadline), TOTAL_STEP="100", GLOBAL_BATCH_SIZE="1", SWAP_OPTIMIZER="1",
                   SAMPLE_MAX_LENGTH="2066", PACK_MAX_LENGTH="2066", MAX_PIXELS="16777216",
                   DATALOADER_NUM_WORKERS="0", PACK_WORKERS="1", MODEL_COMPILE="0",
                   LR="2e-6", WEIGHT_DECAY="0.01", WARMUP_RATIO="0.03", LR_MIN="2e-7",
                   LOSS_MODE="chunk", LOSS_CHUNK_SIZE="1024", LOSS_REDUCTION="token",
                   RECOMPUTE_RATIO="1.0", VISION_RECOMPUTE_RATIO="0.0", HF_INTERVAL="0", DCP_INTERVAL="25",
                   CHECKPOINT_INTERVAL="0", AUTO_RESUME="0", XTUNER_HF_IMPL="1", XTUNER_USE_FA3="0",
                   XTUNER_GC_ENABLE="1", XTUNER_ACTIVATION_OFFLOAD="0", HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", WANDB_MODE="offline", TOKENIZERS_PARALLELISM="false",
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                   PYTORCH_ALLOC_CONF="expandable_segments:True", WEIGHT_AUDIT=str(work / "weight-audit"),
                   PYTHONPATH=f"{code}:{repo}/runtime/xtuner:{repo}:{root}:{repo}/third_party/xtuner")
        from quick_validation import digest
        env["QUICK_BASELINE_POLICY"] = record["baseline_policy"]
        input_dir = retry["source"] if retry else out
        env["DATA_PATH"] = str(input_dir / "balanced_train.jsonl")
        if retry:
            env.update(QUICK_RETRY_SOURCE=str(retry["source"]), QUICK_SMOKE_DEADLINE=str(start + 1800))
            record.update(retry_source=str(retry["source"]), smoke_max_steps=5, smoke_max_seconds=1800,
                          previous_terminal_ledger_sha256=hashlib.sha256((retry["source"] / "training_budget.json").read_bytes()).hexdigest())
        if resume:
            env.update(QUICK_RESUME_SOURCE=str(resume["source"]),
                       QUICK_RESUME_REFERENCE_SHA256=digest(out / "resume_reference.json"))
            record.update(resume_source=str(resume["source"]), resume_step=resume["step"],
                          remaining_optimizer_updates=100 - resume["step"], smoke_absolute_step=resume["step"] + 5,
                          reference_sha256=env["QUICK_RESUME_REFERENCE_SHA256"],
                          baseline_reused_from=str(resume["source"] / "metrics-step-0.json"))
        if digest(input_dir / "balanced_train.jsonl") != json.loads((input_dir / "balanced_train.provenance.json").read_text())["output_sha256"]:
            raise ValueError("Balanced training input drift")
        record["training_source"] = json.loads((input_dir / "balanced_train.provenance.json").read_text())
        command = [str(root / ".venv-xt310/bin/python"), "-m", "torch.distributed.run", "--standalone",
                   "--nproc-per-node=1", "-m", "xtuner.v1.train.cli.sft", "--config", str(code / "quick_train_config.py")]
        record["command"] = command
        process, parent, owned = None, None, []
        requested_stop = []
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, lambda number, frame: requested_stop.append(number))
        atomic_json(budget_path, record)
        try:
            with (out / "training.log").open("a") as log:
                process = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                parent = identity(process.pid)
                if parent is None:
                    raise RuntimeError("Launcher identity unavailable")
                record["launcher"] = parent
                record["gpu_start_epoch"] = time.time()
                atomic_json(budget_path, record)
                next_health = time.time()
                while process.poll() is None:
                    now = time.time()
                    owned = discover_owned(parent, str(code / "quick_train_config.py"), owned)
                    started_path = out / "training_started.json"
                    training_phase = json.loads(started_path.read_text()) if started_path.exists() else None
                    verified_smoke = attempt_smoke(out, start, work, retry, resume)
                    if retry and not verified_smoke and now >= start + 1800 - 5:
                        raise TimeoutError("Authorized five-step/30minute smoke expired; no automatic retry")
                    if requested_stop or now >= deadline or (training_phase and now >= training_phase["deadline_epoch"] - 5):
                        raise TimeoutError("Requested stop or authorized wall budget reached; only already saved last is guaranteed")
                    if shutil.disk_usage(root).free < 50 * GIB:
                        raise RuntimeError("Disk fell below50GiB reserve")
                    if now >= next_health:
                        record["health"] = {"at": now, "owned": owned, "disk_free_bytes": shutil.disk_usage(root).free,
                                            "phase": "training_or_periodic_validation" if training_phase else "setup_or_initial_validation"}
                        steps = work / "scaleup_steps.jsonl"
                        if steps.exists() and steps.stat().st_size:
                            record["health"]["last_step"] = json.loads(steps.read_text().splitlines()[-1])
                        for name in ["last.json", "proxy_best.json"]:
                            if (out / name).exists():
                                record["health"][name] = json.loads((out / name).read_text())
                        record["gpu_hours_charged"] = (now - record["gpu_start_epoch"]) / 3600
                        atomic_json(budget_path, record)
                        print(json.dumps(record["health"]), flush=True)
                        next_health = now + 300
                    time.sleep(min(5, max(0.1, deadline - now)))
                record["exit_code"] = process.returncode
                verified_smoke = attempt_smoke(out, start, work, retry, resume)
                record["result"] = "pass" if process.returncode == 0 and verified_smoke else "fail"
        except BaseException as error:
            record.update(result="fail", error=str(error))
        finally:
            if parent is not None:
                owned = discover_owned(parent, str(code / "quick_train_config.py"), owned)
                stop_owned(owned)
            if process is not None:
                process.wait(timeout=10)
                record["exit_code"] = process.returncode
            record["end_epoch"] = time.time()
            record["gpu_hours_charged"] = (record["end_epoch"] - record.get("gpu_start_epoch", record["end_epoch"])) / 3600
            record["total_gpu_hours_charged"] = prior + record["gpu_hours_charged"]
            record["owned_remaining"] = discover_owned(parent, str(code / "quick_train_config.py"), owned) if parent else []
            atomic_json(budget_path, record)
        print(json.dumps({k: record[k] for k in ["result", "work_dir", "gpu_hours_charged", "total_gpu_hours_charged"]}), flush=True)


def wait_then_supervise(args):
    if not 0 < args.wait_seconds <= 14400:
        raise ValueError("Resource wait must be bounded to<=4hours")
    out, root = args.output.resolve(), args.root.resolve()
    if args.retry_source:
        from retry_inputs import validate_retry_source
        validate_retry_source(root, args.retry_source)
    queue_path = out / "queue_status.json"
    if (out / "training_budget.json").exists():
        raise FileExistsError("Existing attempt; do not silently relaunch")
    old = json.loads(queue_path.read_text()) if queue_path.exists() else None
    if old is not None:
        if not args.resume_existing_queue or identity(old["pid"]) is not None:
            raise FileExistsError("Existing live/unapproved queue; do not silently relaunch")
        if old["gpu_work_started"] or old["deadline_epoch"] <= time.time():
            raise ValueError("Queue cannot be resumed after compute or its original deadline")
    elif args.resume_existing_queue:
        raise FileNotFoundError("No existing queue to resume")
    stopped, wake = [], []
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, lambda number, frame: stopped.append(number))
    signal.signal(signal.SIGUSR1, lambda number, frame: wake.append(number))
    deadline = old["deadline_epoch"] if old is not None else time.time() + args.wait_seconds
    record = {"phase": "waiting_for_resources_not_training", "pid": os.getpid(),
              "deadline_epoch": deadline, "gpu_work_started": False, "poll_seconds": 300}
    if old is not None:
        record.update(previous_queue_pid=old["pid"], reconfigured_at=time.time(), original_deadline_preserved=True)
    while time.time() < deadline and not stopped:
        try:
            record.pop("gpu_candidate", None)
            record.pop("free_mib", None)
            record.pop("error", None)
            disk = shutil.disk_usage(root).free
            if disk < required_free(args):
                raise RuntimeError(f"blocked: disk below{required_free(args) / GIB:.2f}GiB start gate")
            free_mib, gpu = choose_gpu()
            record.update(disk_free_bytes=disk, gpu_candidate=gpu, free_mib=free_mib, at=time.time())
            atomic_json(queue_path, {**record, "phase": "resources_ready_rechecking_single_writer_gate"})
            supervise(args)
            atomic_json(queue_path, {**record, "phase": "supervisor_terminal", "ended_at": time.time()})
            return
        except RuntimeError as error:
            if (out / "training_budget.json").exists():
                raise
            record.update(error=str(error), disk_free_bytes=shutil.disk_usage(root).free, at=time.time())
            atomic_json(queue_path, record)
            print(json.dumps(record), flush=True)
        except BlockingIOError:
            record.update(error="blocked: shared single-writer lock busy", at=time.time())
            atomic_json(queue_path, record)
        next_poll = min(time.time() + 300, deadline)
        while time.time() < next_poll and not stopped and not wake:
            time.sleep(min(2, max(0.1, next_poll - time.time())))
        wake.clear()
    atomic_json(queue_path, {**record, "phase": "cancelled" if stopped else "blocked_wait_limit",
                            "ended_at": time.time(), "gpu_work_started": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=0)
    parser.add_argument("--resume-existing-queue", action="store_true")
    parser.add_argument("--retry-source", type=Path)
    parser.add_argument("--fresh-destination-baseline", action="store_true")
    parser.add_argument("--previous-attempt", type=Path)
    parser.add_argument("--resume-source", type=Path)
    arguments = parser.parse_args()
    if arguments.wait_seconds:
        wait_then_supervise(arguments)
    else:
        supervise(arguments)
