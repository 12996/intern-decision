"""One bounded E06 continuation; preparation precedes owned evaluator cancellation."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from run_quick_training import atomic_json, identity, discover_owned, stop_owned, choose_gpu


def cancel_evaluation(root, out):
    evaluation = root / "outputs/goal-20261008-confirm-v2"
    ledger_path = evaluation / "confirmation_budget.json"
    ledger = json.loads(ledger_path.read_text())
    if ledger["result"] != "running":
        return ledger
    parent = identity(ledger["supervisor_pid"])
    if parent is None or str(evaluation / "code/run_confirmation.py") not in parent["cmdline"]:
        raise ValueError("Evaluator supervisor identity not verified")
    model = ledger["models"][-1]
    child = identity(model["launcher_pid"])
    token = str(evaluation / "code/confirmation_worker.py")
    if child is None or child["ppid"] != parent["pid"] or token not in child["cmdline"]:
        raise ValueError("Evaluator child identity not verified")
    owned = discover_owned(child, token, [])
    atomic_json(out / "evaluation_cancel_request.json", {"at": time.time(),
        "reason": "user_requested_training_priority", "supervisor": parent, "owned": owned,
        "progress": json.loads((evaluation / "progress.json").read_text()),
        "scores_and_cache_preserved": True})
    stop_owned(owned)
    deadline = time.time() + 45
    while time.time() < deadline:
        ledger = json.loads(ledger_path.read_text())
        if ledger["result"] != "running" and identity(parent["pid"]) is None:
            break
        time.sleep(1)
    else:
        raise RuntimeError("Evaluator did not finalize its budget/release gate")
    if discover_owned(child, token, owned):
        raise RuntimeError("Evaluator descendants remain")
    # Keep the evaluator's terminal ledger unchanged; cancellation intent is a
    # separate receipt because its unmodified supervisor labels SIGTERM as fail.
    atomic_json(out / "evaluation_terminal.json", ledger)
    return ledger


def prepare(root, out):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Preparation must explicitly hide GPUs")
    from extension_state import source_info, MANIFEST_SHA
    from resume_state import make_reference
    from quick_validation import digest
    source = source_info(root)
    if digest(out / "code/quick_validation_pairs_20261003.json") != MANIFEST_SHA:
        raise ValueError("Proxy manifest changed")
    reference = make_reference(source["checkpoint"], step=100, total_step=500)
    with (out / "resume_reference.json").open("x") as handle:
        json.dump(reference, handle, indent=2)
    atomic_json(out / "prepare_pass.json", {"pass": True, "at": time.time(),
        "reference_sha256": digest(out / "resume_reference.json"), "step": 100,
        "total_step": 500, "entries": len(reference["state"]),
        "code_sha256": {p.name: digest(p) for p in (out / "code").glob("*.py")}})


def supervise(root, out, takeover):
    from quick_validation import digest
    from extension_state import source_info, MANIFEST_SHA
    source_info(root)
    preparation = json.loads((out / "prepare_pass.json").read_text())
    if not preparation["pass"] or preparation["reference_sha256"] != digest(out / "resume_reference.json"):
        raise ValueError("Preparation/reference changed")
    if preparation["code_sha256"] != {p.name: digest(p) for p in (out / "code").glob("*.py")}:
        raise ValueError("Prepared source changed")
    if (out / "training_budget.json").exists():
        raise FileExistsError("No retry or budget reset")
    if shutil.disk_usage(root).free < 84 * 1024 ** 3:
        raise RuntimeError("blocked: require50GiB reserve plusDCP/best/write margin")
    if int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:"))) < 32 * 1024 ** 2:
        raise RuntimeError("blocked: insufficientRAM")
    # Confirm a permitted idle GPU before interrupting the user's evaluator.
    choose_gpu()
    evaluation = root / "outputs/goal-20261008-confirm-v2/confirmation_budget.json"
    if takeover:
        previous = cancel_evaluation(root, out)
    else:
        previous = json.loads(evaluation.read_text())
    if previous["result"] == "running":
        raise ValueError("Live evaluator; takeover was not authorized")
    prior = previous["total_gpu_hours_charged"]
    if not 0 <= prior < 24:
        raise ValueError("24GPUh exhausted or accounting invalid")
    with (root / "runs/.charades-pilot.lock").open("a") as gate:
        fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        free_mib, gpu = choose_gpu()
        start = time.time()
        deadline = start + min(4 * 3600, (24 - prior) * 3600 - 10)
        work = root / "runs" / time.strftime("charades-id2b-e06-extend100-%Y%m%d-%H%M%S")
        work.mkdir()
        repo, code = root / "Intern-Decision", out / "code"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), INTERN_DECISION_REPO=str(repo),
            DECISION_ROOT=str(root), MEDIA_ROOT=str(root / "frames/id_pilot_frames_8"),
            MODEL_PATH=str(root / "models/Intern-Decision-2B"),
            DATA_PATH=str(root / "outputs/goal-20261003-quick-v1/balanced_train.jsonl"),
            WORK_DIR=str(work), QUICK_OUT=str(out), QUICK_MANIFEST=str(code / "quick_validation_pairs_20261003.json"),
            QUICK_MANIFEST_SHA256=MANIFEST_SHA, QUICK_GLOBAL_DEADLINE=str(deadline),
            QUICK_SMOKE_DEADLINE=str(start + 1800), QUICK_BASELINE_POLICY="fresh_destination",
            QUICK_RESUME_REFERENCE_SHA256=preparation["reference_sha256"],
            TOTAL_STEP="500", GLOBAL_BATCH_SIZE="1", SWAP_OPTIMIZER="1", SAMPLE_MAX_LENGTH="2066",
            PACK_MAX_LENGTH="2066", MAX_PIXELS="16777216", DATALOADER_NUM_WORKERS="0", PACK_WORKERS="1",
            MODEL_COMPILE="0", LR="2e-6", WEIGHT_DECAY="0.01", WARMUP_RATIO="0.03", LR_MIN="2e-7",
            LOSS_MODE="chunk", LOSS_CHUNK_SIZE="1024", LOSS_REDUCTION="token", RECOMPUTE_RATIO="1.0",
            VISION_RECOMPUTE_RATIO="0.0", HF_INTERVAL="0", DCP_INTERVAL="50", CHECKPOINT_INTERVAL="0",
            AUTO_RESUME="0", XTUNER_HF_IMPL="1", XTUNER_USE_FA3="0", XTUNER_GC_ENABLE="1",
            XTUNER_ACTIVATION_OFFLOAD="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WANDB_MODE="offline",
            TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
            PYTORCH_ALLOC_CONF="expandable_segments:True", WEIGHT_AUDIT=str(work / "weight-audit"),
            PYTHONPATH=f"{code}:{repo}/runtime/xtuner:{repo}:{root}:{repo}/third_party/xtuner")
        command = [str(root / ".venv-xt310/bin/python"), "-m", "torch.distributed.run", "--standalone",
            "--nproc-per-node=1", "-m", "xtuner.v1.train.cli.sft", "--config", str(code / "extension_train_config.py")]
        budget = {"result": "running", "supervisor_pid": os.getpid(), "gpu": gpu,
            "start_epoch": start, "deadline_epoch": deadline, "prior_gpu_hours": prior,
            "max_additional_updates": 400, "absolute_total_step": 500, "max_gpu_seconds": deadline - start,
            "work_dir": str(work), "out": str(out), "command": command, "code_sha256": preparation["code_sha256"],
            "evaluation_terminal_sha256": digest(evaluation), "no_automatic_retry": True,
            "gpu2_forbidden": True, "validation_interval": 50, "selection": "fixed32query160pair_proxy_MRR5",
            "source_best50_preserved": True, "source_DCP100_preserved": True,
            "training_mode": "full_language_tuning_not_LoRA", "reserve_gib": 50}
        budget_path = out / "training_budget.json"
        atomic_json(budget_path, budget)
        stopped = []
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, lambda number, frame: stopped.append(number))
        child, parent, owned = None, None, []
        try:
            with (out / "training.log").open("x") as log:
                child = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                parent = identity(child.pid)
                if parent is None:
                    raise RuntimeError("Launcher disappeared")
                budget["launcher"] = parent
                atomic_json(budget_path, budget)
                next_health = start
                while child.poll() is None:
                    now = time.time()
                    owned = discover_owned(parent, str(code / "extension_train_config.py"), owned)
                    if stopped or now >= deadline or shutil.disk_usage(root).free < 50 * 1024 ** 3:
                        raise TimeoutError("Requested stop/time/disk limit")
                    if now >= start + 1800 and not (out / "smoke_pass.json").exists():
                        raise TimeoutError("No five-update smoke within30minutes")
                    if now >= next_health:
                        health = {"at": now, "owned": owned, "disk_free_bytes": shutil.disk_usage(root).free}
                        steps = work / "scaleup_steps.jsonl"
                        if steps.exists():
                            health["last_step"] = json.loads(steps.read_text().splitlines()[-1])
                        for name in ["last.json", "proxy_best.json"]:
                            if (out / name).exists():
                                health[name] = json.loads((out / name).read_text())
                        budget["health"] = health
                        atomic_json(budget_path, budget)
                        with (out / "health.jsonl").open("a") as handle:
                            handle.write(json.dumps(health) + "\n")
                        next_health = now + 300
                    time.sleep(5)
                if child.returncode != 0:
                    raise RuntimeError(f"Training worker exit{child.returncode}")
                smoke = json.loads((out / "smoke_pass.json").read_text())
                if not (smoke["pass"] and smoke["step"] == 105 and smoke["total_step"] == 500
                        and smoke["update_proof"]["full_hash_checked"] and smoke["update_proof"]["update_detected"]):
                    raise ValueError("Invalid smoke receipt")
                if not json.loads((out / "best_export_reload_parity.json").read_text())["pass"]:
                    raise ValueError("Invalid best export")
                budget["result"] = "pass"
        except BaseException as exc:
            budget.update(result="fail", error=str(exc))
        finally:
            if parent:
                owned = discover_owned(parent, str(code / "extension_train_config.py"), owned)
                stop_owned(owned)
            if child:
                child.wait(timeout=10)
                budget["exit_code"] = child.returncode
            end = time.time()
            budget.update(end_epoch=end, gpu_hours_charged=(end - start) / 3600,
                total_gpu_hours_charged=prior + (end - start) / 3600,
                owned_remaining=discover_owned(parent, str(code / "extension_train_config.py"), owned) if parent else [])
            atomic_json(budget_path, budget)
        print(json.dumps(budget), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/zhaopu/intern-decision"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--spawn", action="store_true")
    parser.add_argument("--takeover-evaluation", action="store_true")
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if out != root / "outputs/goal-20261009-extend100-v1":
        raise ValueError("Only this approved new output is permitted")
    if args.prepare:
        prepare(root, out)
    elif args.spawn:
        with (out / "launch_once.json").open("x") as handle:
            json.dump({"at": time.time(), "no_automatic_retry": True}, handle)
        with (out / "supervisor.log").open("x") as log:
            command = [sys.executable, str(Path(__file__).resolve()), "--root", str(root), "--out", str(out)]
            if args.takeover_evaluation:
                command.append("--takeover-evaluation")
            child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"supervisor_pid": child.pid, "out": str(out)}), flush=True)
    else:
        supervise(root, out, args.takeover_evaluation)


if __name__ == "__main__":
    main()
