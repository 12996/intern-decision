"""One cached baseline completion, with terminal concurrent accounting preserved."""

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
from released_completion_protocol import report_completed, verify_seed
from run_confirmation import select_gpu
from run_quick_training import atomic_json, identity, discover_owned, stop_owned
from selected_eval_protocol import MANIFEST_SHA


def inputs(root, out):
    training = json.loads((root / "outputs/goal-20261009-extend100-v1/training_budget.json").read_text())
    selected_dir = root / "outputs/goal-20261009-test-step50-v1"
    selected_budget = json.loads((selected_dir / "evaluation_budget.json").read_text())
    for ledger in (training, selected_budget):
        if ledger["result"] != "pass" or ledger["owned_remaining"]:
            raise ValueError("Source work must be terminal and valid")
        current = identity(ledger["supervisor_pid"])
        if current and str(Path(ledger["out"]) / "code") in current["cmdline"]:
            raise ValueError("Source supervisor still live")
    if training["prior_gpu_hours"] != selected_budget["prior_gpu_hours"]:
        raise ValueError("Concurrent accounting does not share the same parent")
    prior = training["prior_gpu_hours"] + training["gpu_hours_charged"] + selected_budget["gpu_hours_charged"]
    if abs(prior - 11.247872815728186) > 1e-9 or prior >= 23.5:
        raise ValueError("Unexpected prior accounting or insufficient30minute budget")
    if digest(out / "confirmation_manifest.json") != MANIFEST_SHA:
        raise ValueError("Protocol changed")
    old = root / "outputs/goal-20261008-confirm-v2"
    seed_sha = selected_budget["partial_released_scores_sha256"]
    if digest(out / "scores-released-seed.jsonl") != seed_sha or digest(old / "scores-released.jsonl") != seed_sha:
        raise ValueError("Original partial scoring evidence changed")
    selected_sha = "27234234addc88b7562fc94c2bec7da5610f273f4083a8c6a200ff59fb4d75b3"
    if digest(out / "scores-step50.jsonl") != selected_sha or digest(selected_dir / "scores-step50.jsonl") != selected_sha:
        raise ValueError("Selected model scoring evidence changed")
    for name in ("frame_sources.json", "frame_transfer.json", "metrics-step50.json"):
        if digest(out / name) != digest(selected_dir / name):
            raise ValueError("Staged comparison receipt changed")
    if digest(out / "code/decision_scorer.py") != "8a2af7aa75fcc3a70fe130968747b3ab12725ebffc68d18ea3c96b724c357dd5":
        raise ValueError("Scorer changed")
    manifest = json.loads((out / "confirmation_manifest.json").read_text())
    seed = [json.loads(line) for line in (out / "scores-released-seed.jsonl").read_text().splitlines()]
    selected = [json.loads(line) for line in (out / "scores-step50.jsonl").read_text().splitlines()]
    verify_seed(manifest, seed, selected)
    if len(seed) != 369:
        raise ValueError("Unexpected preserved prefix")
    return prior, seed_sha


def supervise(root, out):
    import fcntl
    ledger_path = out / "baseline_budget.json"
    if ledger_path.exists():
        raise FileExistsError("One-shot completion; no automatic retry")
    prior, seed_sha = inputs(root, out)
    code, repo = out / "code", root / "Intern-Decision"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2", PYTHONPATH=f"{code}:{repo}/runtime/xtuner:{repo}:{root}:{repo}/third_party/xtuner")
    with (out / "cpu_tests.log").open("x") as log:
        result = subprocess.run([str(root / ".venv-xt310/bin/python"), "-m", "unittest",
            "test_released_completion", "test_selected_eval_protocol", "test_confirmation_protocol",
            "test_confirmation_restore", "test_run_confirmation", "test_confirmation_assets", "-v"],
            cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    if result.returncode:
        raise RuntimeError("CPU tests failed; no GPU work started")
    with (root / "runs/.charades-pilot.lock").open("a") as train_gate, (root / "runs/.charades-selected-eval.lock").open("a") as eval_gate:
        fcntl.flock(train_gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(eval_gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior, seed_sha = inputs(root, out)
        if shutil.disk_usage(root).free < 51 * 1024 ** 3:
            raise RuntimeError("blocked:50GiBreserveplus1GiBlogsrequired")
        snapshot = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True)
        free, gpu, fraction = select_gpu(snapshot)
        start = time.time()
        deadline = start + 1790
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), MEM_FRACTION=str(fraction), INTERN_DECISION_REPO=str(repo),
            DECISION_ROOT=str(root), MEDIA_ROOT=str(root / "frames/id_pilot_frames_8"), XTUNER_HF_IMPL="1",
            XTUNER_USE_FA3="0", XTUNER_GC_ENABLE="1", XTUNER_ACTIVATION_OFFLOAD="0", HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTORCH_ALLOC_CONF="expandable_segments:True",
            CONFIRM_OUT=str(out), CONFIRM_MODEL="released", CONFIRM_MANIFEST_SHA256=MANIFEST_SHA,
            CONFIRM_DEADLINE=str(deadline), CONFIRM_SELECTED_FIRST="0", CONFIRM_RELEASED_COMPLETION="1",
            CONFIRM_SEED_SHA256=seed_sha)
        command = [str(root / ".venv-xt310/bin/python"), "-m", "torch.distributed.run", "--standalone",
            "--nproc-per-node=1", str(code / "confirmation_worker.py")]
        ledger = {"result": "running", "supervisor_pid": os.getpid(), "gpu": gpu, "start_epoch": start,
            "deadline_epoch": deadline, "max_gpu_seconds": 1790, "prior_gpu_hours": prior,
            "model": "released", "queries": 136, "score_pairs": 620, "reused_pairs": 369, "new_pairs": 251,
            "all_caches_read_only": True, "no_weights_saved": True, "no_automatic_retry": True,
            "out": str(out), "command": command, "seed_sha256": seed_sha,
            "code_sha256": {p.name: digest(p) for p in code.glob("*.py")}}
        atomic_json(ledger_path, ledger)
        stopped = []
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda number, frame: stopped.append(number))
        child, parent, owned = None, None, []
        try:
            with (out / "worker-released.log").open("x") as log:
                child = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                parent = identity(child.pid)
                if parent is None:
                    raise RuntimeError("Owned launcher disappeared")
                ledger["launcher"] = parent
                atomic_json(ledger_path, ledger)
                next_health = start
                while child.poll() is None:
                    now = time.time()
                    owned = discover_owned(parent, str(code / "confirmation_worker.py"), owned)
                    if stopped or now >= deadline or shutil.disk_usage(root).free < 50 * 1024 ** 3:
                        raise TimeoutError("Completion stop/time/disk limit")
                    if now >= next_health:
                        progress = json.loads((out / "progress.json").read_text()) if (out / "progress.json").exists() else None
                        health = {"at": now, "owned": owned, "progress": progress}
                        ledger["health"] = health
                        atomic_json(ledger_path, ledger)
                        with (out / "events.jsonl").open("a") as handle:
                            handle.write(json.dumps(dict(event="health", **health)) + "\n")
                        next_health = now + 30
                    time.sleep(2)
                if child.returncode != 0:
                    raise RuntimeError(f"Baseline worker exit{child.returncode}")
            report = report_completed(out)
            with (out / "baseline_comparison.json").open("x") as handle:
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
            ledger.update(end_epoch=end, gpu_hours_charged=(end-start)/3600,
                total_gpu_hours_charged=prior+(end-start)/3600,
                owned_remaining=discover_owned(parent, str(code / "confirmation_worker.py"), owned) if parent else [])
            atomic_json(ledger_path, ledger)
            with (out / "events.jsonl").open("a") as handle:
                handle.write(json.dumps({"event": "terminal", "at": end, "result": ledger["result"],
                    "error": ledger.get("error"), "gpu_hours_charged": ledger["gpu_hours_charged"]}) + "\n")
        print(json.dumps(ledger), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/zhaopu/intern-decision"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spawn", action="store_true")
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if out != root / "outputs/goal-20261009-baseline-complete-v1":
        raise ValueError("Only this authorized completion output is permitted")
    if args.spawn:
        with (out / "launch_once.json").open("x") as handle:
            json.dump({"at": time.time(), "no_automatic_retry": True}, handle)
        with (out / "supervisor.log").open("x") as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--root", str(root), "--out", str(out)],
                cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"supervisor_pid": child.pid, "out": str(out)}), flush=True)
    else:
        supervise(root, out)


if __name__ == "__main__":
    main()
