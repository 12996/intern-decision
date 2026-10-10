"""CPU wait/preflight then one authorized GPU supervisor; no automatic retries."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path("/home/zhaopu/intern-decision")
BOOTSTRAP = Path("/home/zhaopu/.e04-migration-20261008")
OUT = ROOT / "outputs/goal-20261008-retry-v1"
SOURCE = ROOT / "outputs/goal-20261003-quick-v1"
DIRECT_APPROVED = False
DELTA_APPROVED = False


def status(phase, **details):
    record = {"phase": phase, "pid": os.getpid(), "at": time.time(), **details}
    path = BOOTSTRAP / ("post_delta_status.json" if DELTA_APPROVED else
                        "post_direct_status.json" if DIRECT_APPROVED else "post_status.json")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, path)
    print(json.dumps(record), flush=True)


def remaining(deadline):
    seconds = deadline - time.time()
    if seconds <= 0:
        raise TimeoutError("Original migration/preflight deadline expired; do not launch")
    return seconds


def cpu_command(command, env, log, deadline, limit):
    remaining(deadline)
    with log.open("x") as handle:
        result = subprocess.run(command, env=env, cwd=ROOT / "Intern-Decision",
                                stdout=handle, stderr=subprocess.STDOUT,
                                timeout=min(limit, remaining(deadline)))
    if result.returncode:
        raise RuntimeError(f"CPU check failed, exit={result.returncode}; see {log}")


def run(deadline):
    try:
        status("waiting_for_verified_migration_not_training", deadline_epoch=deadline)
        while True:
            remaining(deadline)
            migration = json.loads((ROOT / ("migration_delta_status.json" if DELTA_APPROVED else
                                           "migration_status.json")).read_text())
            if migration.get("deadline_epoch") != deadline:
                raise ValueError("Migration deadline drift")
            if DELTA_APPROVED and migration.get("delta_approved") is not True:
                raise ValueError("Not the approved delta receipt")
            if DIRECT_APPROVED and migration.get("direct_approved") is not True:
                time.sleep(min(5, remaining(deadline)))
                continue
            if migration.get("result") == "fail":
                raise RuntimeError("Migration recovery failed; no GPU retry allowed")
            if migration.get("result") == "pass":
                if migration["phase"] != "copied_sha256_verified_environment_not_yet_tested":
                    raise ValueError("Unrecognized migration pass")
                break
            time.sleep(min(30, remaining(deadline)))
        code, repo = OUT / "code", ROOT / "Intern-Decision"
        python = ROOT / ".venv-xt310/bin/python"
        temp = OUT / "cpu-tests"
        temp.mkdir()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TMPDIR=str(temp),
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PYTHONPATH=f"{code}:{repo}/runtime/xtuner:{repo}:{ROOT}:{repo}/third_party/xtuner")
        status("migration_pass_cpu_environment_checks_not_training", migration=migration)
        probe = ("import sys,torch,json; import scaleup_observe; "
                 "assert torch.__version__=='2.9.1+cu128'; "
                 "assert torch.version.cuda=='12.8'; "
                 "assert not torch.cuda.is_available(); "
                 "print(json.dumps({'python':sys.version,'torch':torch.__version__,"
                 "'torch_file':torch.__file__,'observe_file':scaleup_observe.__file__}))")
        cpu_command([str(python), "-c", probe], env, OUT / "relocated_environment.log", deadline, 90)
        cpu_command([str(python), "-m", "unittest", "quick_train_tests", "test_train_update_probe", "-v"],
                    env, OUT / "cpu_tests_migrated.log", deadline, 120)
        cpu_command([str(python), str(code / "validate_quick_setup.py"), "--root", str(ROOT),
                     "--output", str(OUT), "--retry-source", str(SOURCE)],
                    env, OUT / "setup_preflight_migrated.log", deadline, 180)
        preflight = json.loads((OUT / "setup_preflight.json").read_text())
        if preflight["config_check"] != "pass" or preflight["reuse_check"]["input_check"] != "pass":
            raise ValueError("Configuration/reuse preflight did not pass")
        if preflight["disk_gate"] != "pass" or preflight["gpu_gate"]["result"] != "pass":
            status("blocked_resources_no_gpu_work", preflight=preflight)
            return
        remaining(deadline)
        if (OUT / "training_budget.json").exists():
            raise FileExistsError("Never relaunch an existing GPU attempt")
        command = [str(python), str(code / "run_quick_training.py"), "--root", str(ROOT),
                   "--output", str(OUT), "--retry-source", str(SOURCE)]
        with (OUT / "supervisor.log").open("x") as log:
            process = subprocess.Popen(command, env=env, cwd=repo, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(2)
        if process.poll() is not None:
            raise RuntimeError("GPU supervisor exited at launch; inspect supervisor.log, do not retry")
        status("supervisor_started_initialization_not_optimizer_proof", supervisor_pid=process.pid,
               command=command, output=str(OUT), original_prior_gpu_hours=5.138736795054542,
               smoke_seconds=1800, smoke_steps=5, conditional_total_steps=100,
               supervisor_log=str(OUT / "supervisor.log"))
    except BaseException as error:
        status("fail_or_blocked_no_automatic_retry", error=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--spawn", action="store_true")
    parser.add_argument("--direct-approved", action="store_true")
    parser.add_argument("--delta-approved", action="store_true")
    args = parser.parse_args()
    DIRECT_APPROVED = args.direct_approved
    DELTA_APPROVED = args.delta_approved
    if DIRECT_APPROVED and DELTA_APPROVED:
        raise ValueError("Select exactly one authorized channel")
    if args.spawn:
        remaining(args.deadline_epoch)
        suffix = "_delta" if DELTA_APPROVED else "_direct" if DIRECT_APPROVED else ""
        with (BOOTSTRAP / f"post{suffix}_start.json").open("x") as marker:
            with (BOOTSTRAP / f"post{suffix}.log").open("x") as log:
                command = [sys.executable, str(Path(__file__).resolve()), "--deadline-epoch", str(args.deadline_epoch)]
                if DIRECT_APPROVED:
                    command.append("--direct-approved")
                if DELTA_APPROVED:
                    command.append("--delta-approved")
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record = {"pid": process.pid, "command": command, "deadline_epoch": args.deadline_epoch}
            marker.write(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    else:
        run(args.deadline_epoch)
