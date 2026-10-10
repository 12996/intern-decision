"""One CPU-preflight then user-authorized rebaseline training attempt."""

import json
import argparse
import os
from pathlib import Path
import subprocess
import time


ROOT = Path("/home/zhaopu/intern-decision")
OUT = ROOT / "outputs/goal-20261008-rebase-v1"
CODE = OUT / "code"
REPO = ROOT / "Intern-Decision"
PYTHON = ROOT / ".venv-xt310/bin/python"
COMMON = ["--root", str(ROOT), "--output", str(OUT), "--retry-source",
          str(ROOT / "outputs/goal-20261003-quick-v1"), "--fresh-destination-baseline",
          "--previous-attempt", str(ROOT / "outputs/goal-20261008-retry-v1")]


def main():
    global OUT, CODE, COMMON
    parser = argparse.ArgumentParser()
    parser.add_argument("--swapfix", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.swapfix and args.resume:
        raise ValueError("Choose only one explicitly authorized attempt")
    if args.swapfix:
        OUT = ROOT / "outputs/goal-20261008-swapfix-v1"
        CODE = OUT / "code"
        COMMON = ["--root", str(ROOT), "--output", str(OUT), "--retry-source",
                  str(ROOT / "outputs/goal-20261003-quick-v1"), "--fresh-destination-baseline",
                  "--previous-attempt", str(ROOT / "outputs/goal-20261008-rebase-v1")]
    if args.resume:
        OUT = ROOT / "outputs/goal-20261008-resume25-v1"
        CODE = OUT / "code"
        COMMON = ["--root", str(ROOT), "--output", str(OUT), "--retry-source",
                  str(ROOT / "outputs/goal-20261003-quick-v1"), "--fresh-destination-baseline",
                  "--previous-attempt", str(ROOT / "outputs/goal-20261008-swapfix-v1"),
                  "--resume-source", str(ROOT / "outputs/goal-20261008-swapfix-v1")]
    with (OUT / "launch_once.json").open("x") as marker:
        json.dump({"at": time.time(), "pid": os.getpid(), "no_automatic_retry": True}, marker)
    temp = OUT / "cpu-tests"
    temp.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TMPDIR=str(temp),
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONPATH=f"{CODE}:{REPO}/runtime/xtuner:{REPO}:{ROOT}:{REPO}/third_party/xtuner")
    tests = ["quick_train_tests", "test_train_update_probe", "test_baseline_policy"]
    if args.swapfix or args.resume:
        tests.append("test_swap_restore_fix")
    if args.resume:
        tests += ["test_fp32_hf_export", "test_resume_state"]
    checks = [([str(PYTHON), "-m", "unittest", *tests, "-v"], "cpu_tests_runtime.log", 120)]
    if args.resume:
        checks.append(([str(PYTHON), str(CODE / "resume_state.py"), "--root", str(ROOT),
                        "--source", str(ROOT / "outputs/goal-20261008-swapfix-v1"), "--output", str(OUT)],
                       "resume_reference.log", 180))
    checks.append(([str(PYTHON), str(CODE / "validate_quick_setup.py"), *COMMON], "preflight.log", 180))
    for command, filename, limit in checks:
        with (OUT / filename).open("x") as log:
            result = subprocess.run(command, env=env, cwd=REPO, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=limit)
        if result.returncode:
            raise RuntimeError(f"CPU preparation failed; do not retry: {filename}")
    report = json.loads((OUT / "setup_preflight.json").read_text())
    if report["disk_gate"] != "pass" or report["gpu_gate"]["result"] != "pass":
        raise RuntimeError("Resource blocked; no GPU attempt launched")
    if (OUT / "training_budget.json").exists():
        raise FileExistsError("Do not replace an existing attempt ledger")
    command = [str(PYTHON), str(CODE / "run_quick_training.py"), *COMMON]
    with (OUT / "supervisor.log").open("x") as log:
        child = subprocess.Popen(command, env=env, cwd=REPO, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(2)
    if child.poll() is not None:
        raise RuntimeError("Supervisor exited immediately; no automatic retry")
    print(json.dumps({"supervisor_pid": child.pid, "output": str(OUT),
                      "baseline_policy": "fresh_destination", "optimizer_proof_pending": True}), flush=True)


if __name__ == "__main__":
    main()
