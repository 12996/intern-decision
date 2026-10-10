"""Read-only step50 testing on a separate GPU with a reserved concurrent budget."""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from confirmation_protocol import digest
from run_confirmation import select_gpu
from run_quick_training import atomic_json, identity, discover_owned, stop_owned
from selected_eval_protocol import MANIFEST_SHA, report_selected, reserved_seconds


def choose_independent(csv, training_gpu):
    filtered = "\n".join(line for line in csv.splitlines() if line.strip()
                         and int(line.split(",")[0]) != training_gpu)
    return select_gpu(filtered)


def validate_inputs(root, out):
    shared = root / "outputs/goal-20261008-confirm-v2"
    old = json.loads((shared / "confirmation_budget.json").read_text())
    if old["result"] == "running":
        raise ValueError("Shared confirmation inputs still have an active writer")
    for pid in [old["supervisor_pid"], *[m["launcher_pid"] for m in old["models"]]]:
        if identity(pid) is not None:
            raise ValueError("Old evaluator process remains live")
    if digest(out / "confirmation_manifest.json") != MANIFEST_SHA or digest(shared / "confirmation_manifest.json") != MANIFEST_SHA:
        raise ValueError("Frozen query/candidate protocol drift")
    for name in ("frame_sources.json", "frame_transfer.json"):
        if digest(out / name) != digest(shared / name):
            raise ValueError("Staged frame receipt differs")
    if json.loads((out / "frame_transfer.json").read_text())["result"] != "pass":
        raise ValueError("Frame transfer incomplete")
    history = root / "outputs/goal-20261008-resume25-v1"
    best = json.loads((history / "proxy_best.json").read_text())
    expected = root / "runs/charades-id2b-e04-balanced-20261008-213420/20261008213430/proxy-best-step-50"
    if best["step"] != 50 or Path(best["weights"]).resolve() != expected:
        raise ValueError("Immutable selected model changed")
    index = json.loads((expected / "model.safetensors.index.json").read_text())
    if len(index["weight_map"]) != 617 or any(not (expected / name).is_file()
            or (expected / name).stat().st_size == 0 for name in set(index["weight_map"].values())):
        raise ValueError("Selected-model export incomplete")
    scorer_sha = "8a2af7aa75fcc3a70fe130968747b3ab12725ebffc68d18ea3c96b724c357dd5"
    if digest(out / "code/decision_scorer.py") != scorer_sha:
        raise ValueError("Scoring implementation changed")
    training = json.loads((root / "outputs/goal-20261009-extend100-v1/training_budget.json").read_text())
    if training["prior_gpu_hours"] != old["total_gpu_hours_charged"] or training["max_gpu_seconds"] != 14400:
        raise ValueError("Concurrent budget chain changed")
    seconds = reserved_seconds(old["total_gpu_hours_charged"], training["max_gpu_seconds"])
    return old, training, seconds


def combined_charge(prior, training, evaluation_charge, now):
    if training["result"] == "running":
        training_charge = max(0, now - training["start_epoch"]) / 3600
    else:
        training_charge = training["gpu_hours_charged"]
    return {"pre_concurrent_gpu_hours": prior, "training_gpu_hours": training_charge,
        "evaluation_gpu_hours": evaluation_charge, "combined_gpu_hours": prior + training_charge + evaluation_charge,
        "training_terminal": training["result"] != "running",
        "not_a_serial_parent_of_running_training": True}


def supervise(root, out):
    import fcntl
    ledger_path = out / "evaluation_budget.json"
    if ledger_path.exists():
        raise FileExistsError("One-shot evaluation; no automatic retry")
    old, training, seconds = validate_inputs(root, out)
    code, repo = out / "code", root / "Intern-Decision"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2", PYTHONPATH=f"{code}:{repo}/runtime/xtuner:{repo}:{root}:{repo}/third_party/xtuner")
    tests = [str(root / ".venv-xt310/bin/python"), "-m", "unittest", "test_selected_eval_protocol",
        "test_confirmation_protocol", "test_confirmation_restore", "test_run_confirmation", "test_confirmation_assets", "-v"]
    with (out / "cpu_tests.log").open("x") as log:
        result = subprocess.run(tests, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    if result.returncode:
        raise RuntimeError("CPU preflight failed; no GPU work started")
    # This gate serializes read-only snapshot evaluations, not training writers.
    # No training lock is bypassed: step50 is a retained, immutable old artifact;
    # training, legacy/shared caches and all source logs are never written here.
    with (root / "runs/.charades-selected-eval.lock").open("a") as gate:
        fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old, training, seconds = validate_inputs(root, out)
        if shutil.disk_usage(root).free < 57 * 1024 ** 3:
            raise RuntimeError("blocked: require50GiBreserve+6GiBcache+1GiBlogs")
        snapshot = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True)
        free, gpu, fraction = choose_independent(snapshot, training["gpu"])
        started = time.time()
        deadline = started + seconds
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), MEM_FRACTION=str(fraction), INTERN_DECISION_REPO=str(repo),
            DECISION_ROOT=str(root), MEDIA_ROOT=str(root / "frames/id_pilot_frames_8"),
            XTUNER_HF_IMPL="1", XTUNER_USE_FA3="0", XTUNER_GC_ENABLE="1", XTUNER_ACTIVATION_OFFLOAD="0",
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
            PYTORCH_ALLOC_CONF="expandable_segments:True", CONFIRM_OUT=str(out), CONFIRM_MODEL="step50",
            CONFIRM_DEADLINE=str(deadline), CONFIRM_MANIFEST_SHA256=MANIFEST_SHA, CONFIRM_SELECTED_FIRST="1")
        command = [str(root / ".venv-xt310/bin/python"), "-m", "torch.distributed.run", "--standalone",
            "--nproc-per-node=1", str(code / "confirmation_worker.py")]
        ledger = {"result": "running", "supervisor_pid": os.getpid(), "gpu": gpu, "start_epoch": started,
            "deadline_epoch": deadline, "max_gpu_seconds": seconds, "prior_gpu_hours": old["total_gpu_hours_charged"],
            "reserved_training_gpu_hours": 4, "training_output": str(root / "outputs/goal-20261009-extend100-v1"),
            "training_gpu_excluded": training["gpu"], "gpu2_forbidden": True, "model": "step50",
            "manifest_sha256": MANIFEST_SHA, "queries": 136, "scored_pairs": 620, "out": str(out),
            "no_weights_saved": True, "legacy_and_shared_cache_read_only": True, "own_cache_max_gib": 6,
            "no_checkpoint_selection_from_confirmation": True, "no_automatic_retry": True, "command": command,
            "code_sha256": {p.name: digest(p) for p in code.glob("*.py")},
            "partial_released_scores_sha256": digest(root / "outputs/goal-20261008-confirm-v2/scores-released.jsonl"),
            "selected_weights": json.loads((root / "outputs/goal-20261008-resume25-v1/proxy_best.json").read_text())["weights"]}
        atomic_json(ledger_path, ledger)
        stopped = []
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda number, frame: stopped.append(number))
        child, parent, owned = None, None, []
        try:
            with (out / "worker-step50.log").open("x") as log:
                child = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                parent = identity(child.pid)
                if parent is None:
                    raise RuntimeError("Owned evaluation launcher disappeared")
                ledger["launcher"] = parent
                atomic_json(ledger_path, ledger)
                next_health = started
                while child.poll() is None:
                    now = time.time()
                    owned = discover_owned(parent, str(code / "confirmation_worker.py"), owned)
                    if stopped or now >= deadline or shutil.disk_usage(root).free < 50 * 1024 ** 3:
                        raise TimeoutError("Evaluation stop/time/disk limit")
                    if now >= started + 1800 and not (out / "load-and-parity-step50.json").exists():
                        raise TimeoutError("No selected-model load/sentinel pass within30minutes")
                    if now >= next_health:
                        progress = json.loads((out / "progress.json").read_text()) if (out / "progress.json").exists() else None
                        current_training = json.loads((root / "outputs/goal-20261009-extend100-v1/training_budget.json").read_text())
                        accounting = combined_charge(ledger["prior_gpu_hours"], current_training, (now - started) / 3600, now)
                        if accounting["combined_gpu_hours"] >= 24:
                            raise TimeoutError("Combined24GPUhbudget exhausted")
                        health = {"at": now, "owned": owned, "progress": progress,
                            "disk_free_bytes": shutil.disk_usage(root).free, "concurrent_accounting": accounting}
                        ledger["health"] = health
                        atomic_json(ledger_path, ledger)
                        with (out / "events.jsonl").open("a") as handle:
                            handle.write(json.dumps(dict(event="health", **health)) + "\n")
                        next_health = now + 300
                    time.sleep(5)
                if child.returncode != 0:
                    raise RuntimeError(f"Selected-model worker exit{child.returncode}")
            report = report_selected(out)
            with (out / "selected_model_report.json").open("x") as handle:
                json.dump(report, handle, indent=2)
            ledger["result"] = "pass"
        except BaseException as exc:
            ledger.update(result="fail", error=str(exc))
        finally:
            if parent:
                owned = discover_owned(parent, str(code / "confirmation_worker.py"), owned)
                stop_owned(owned)
            if child:
                child.wait(timeout=10)
                ledger["exit_code"] = child.returncode
            end = time.time()
            current_training = json.loads((root / "outputs/goal-20261009-extend100-v1/training_budget.json").read_text())
            charge = (end - started) / 3600
            ledger.update(end_epoch=end, gpu_hours_charged=charge,
                concurrent_accounting=combined_charge(ledger["prior_gpu_hours"], current_training, charge, end),
                owned_remaining=discover_owned(parent, str(code / "confirmation_worker.py"), owned) if parent else [])
            atomic_json(ledger_path, ledger)
            with (out / "events.jsonl").open("a") as handle:
                handle.write(json.dumps({"event": "terminal", "at": end, "result": ledger["result"],
                    "error": ledger.get("error"), "gpu_hours_charged": charge}) + "\n")
        print(json.dumps(ledger), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/zhaopu/intern-decision"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spawn", action="store_true")
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if out != root / "outputs/goal-20261009-test-step50-v1":
        raise ValueError("Only this bounded selected-model evaluation output is permitted")
    if args.spawn:
        with (out / "launch_once.json").open("x") as handle:
            json.dump({"at": time.time(), "no_automatic_retry": True}, handle)
        with (out / "supervisor.log").open("x") as log:
            command = [sys.executable, str(Path(__file__).resolve()), "--root", str(root), "--out", str(out)]
            child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"supervisor_pid": child.pid, "out": str(out)}), flush=True)
    else:
        supervise(root, out)


if __name__ == "__main__":
    main()
