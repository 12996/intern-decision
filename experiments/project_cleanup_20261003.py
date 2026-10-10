"""One-shot, fixed allowlist cleanup. No datasets, environments or whole projects."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import time


HOME = Path("/home/zhaopu")
ROOT = HOME / "intern-decision"
RUN = ROOT / "runs/charades-id2b-e03-20260929-224918"
EXP = RUN / "20260929225053"
OUT = ROOT / "outputs/cleanup-20261003"
QUICK = ROOT / "outputs/goal-20261003-quick-v1"
GIB = 1024 ** 3
CACHE = [HOME / ".cache/pip/http-v2", ROOT / "pip-cache/http-v2",
         ROOT / "pip-cache-310/http-v2"]
TEMP_NAMES = """
pip-unpack-58qr27nr pip-metadata-wqoz2jfm pip-metadata-5sq2imx2
pip-unpack-7irjmnr0 pip-unpack-lks_bnhu pip-unpack-qlau5ypu
pip-unpack-9x42nz9o pip-unpack-ba6tnb34 pip-unpack-mnq5khd9
pip-unpack-9ts6wi5y pip-unpack-4k75j84b pip-build-tracker-9p0wo0ag
pip-unpack-7pw88e4h pip-ephem-wheel-cache-79zo7_id pip-metadata-77adj5ze
pip-unpack-_hys2jlw pip-unpack-d_qaa1b_ pip-build-tracker-4y83vfq1
pip-unpack-va_wm462 pip-unpack-jgho718c pip-unpack-qljh6y3w
pip-metadata-gqppm4vb pip-metadata-u0knhq9f pip-metadata-n1v1g1ao
pip-unpack-76w8hr69 pip-unpack-1i4qyvp3 pip-install-l93lrqr1
pip-ephem-wheel-cache-po159tjb pip-metadata-7kzyyj8t pip-unpack-uuc7lv9g
pip-unpack-_940aku7 tmpsmx4a77u pip-unpack-j_wsrb8u
pip-unpack-7uz5t8e8 pip-metadata-abjln2vj pip-unpack-icv5u5p7
pip-unpack-hhkdduc9 pip-install-6jjcf0un tmpwtswuc74 pip-unpack-88wbgs51
""".split()
TEMPS = [ROOT / "tmp" / name for name in TEMP_NAMES]
HF = [EXP / f"hf-{step}" for step in range(200, 800, 100)]
ALLOW = CACHE + TEMPS + HF
KEEP_TREES = [ROOT / "models/Intern-Decision-2B", EXP / "hf-100",
              EXP / "hf-800", EXP / "checkpoints/ckpt-step-750"]


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def literal(path):
    if not path.is_absolute() or not path.is_relative_to(HOME) or path == HOME:
        raise ValueError(f"Outside allowed scope: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"Symlink component: {current}")
    if path.resolve() != path:
        raise ValueError(f"Path substitution: {path}")
    if path.stat().st_dev != HOME.stat().st_dev:
        raise ValueError(f"Different filesystem: {path}")


def snapshot(path):
    literal(path)
    entries = [path]
    if path.is_dir():
        for parent, directories, files in os.walk(path, followlinks=False):
            entries.extend(Path(parent) / name for name in sorted(directories + files))
    records, logical, allocated, newest = [], 0, 0, 0
    device = HOME.stat().st_dev
    for entry in entries:
        info = entry.lstat()
        if info.st_dev != device or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise ValueError(f"Symlink, special file or mount: {entry}")
        records.append([str(entry.relative_to(path)), info.st_dev, info.st_ino,
                        info.st_mode, info.st_size, info.st_mtime_ns, info.st_nlink])
        newest = max(newest, info.st_mtime)
        if stat.S_ISREG(info.st_mode):
            logical += info.st_size
            allocated += info.st_blocks * 512
    return {"path": str(path), "tree_sha256": canonical_hash(records),
            "logical_bytes": logical, "allocated_file_bytes": allocated,
            "entry_count": len(records), "newest_mtime": newest}


def protected():
    result = {"trees": [snapshot(path) for path in KEEP_TREES], "records": {}, "identities": {}}
    for path in KEEP_TREES[:3]:
        index = json.loads((path / "model.safetensors.index.json").read_text())
        for name in set(index["weight_map"].values()):
            shard = path / name
            if not shard.resolve().is_relative_to(path) or shard.stat().st_size <= 0:
                raise ValueError(f"Incomplete protected HF export: {path}")
    for path in [RUN / "log.txt", RUN / "scaleup_steps.jsonl", RUN / ".xtuner",
                 ROOT / "data/full-20260929/train.jsonl",
                 QUICK / "balanced_train.provenance.json"]:
        literal(path)
        result["records"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in [ROOT / ".venv-xt310", ROOT / "torch-reuse-cp310",
                 ROOT / "Intern-Decision", ROOT / "data/full-20260929",
                 ROOT / "frames/id_pilot_frames_8", QUICK / "code",
                 HOME / "msc-prvr-data"]:
        literal(path)
        info = path.stat()
        result["identities"][str(path)] = [info.st_dev, info.st_ino]
    latest = EXP / "hf-latest"
    if not latest.is_symlink() or latest.resolve() != EXP / "hf-800":
        raise ValueError("hf-latest must reference protected hf-800")
    result["latest"] = os.readlink(latest)
    return result


def process_identity(pid):
    path = Path("/proc") / str(pid)
    try:
        fields = (path / "stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "start_ticks": int(fields[19]),
                "cmdline": (path / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")}
    except (FileNotFoundError, ProcessLookupError):
        return None


def queue_identity():
    record = process_identity(1898359)
    expected = [str(ROOT / ".venv-xt310/bin/python"),
                str(QUICK / "code/run_quick_training.py"), "--root", str(ROOT),
                "--output", str(QUICK), "--wait-seconds", "14400"]
    if record is None:
        return None
    if record["cmdline"].split() != expected:
        raise ValueError("Queue PID identity does not match authorized launcher")
    return record


def session_helper(command):
    return (command.strip() in ["/usr/lib/systemd/systemd --user", "/lib/systemd/systemd --user", "(sd-pam)"]
            or command.strip().startswith("sshd: zhaopu@"))


def active_references(targets):
    hits, denied, helpers, foreign = [], [], [], 0
    target_names = [str(path) for path in targets]

    def matches(value):
        return [name for name in target_names if name in value]

    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        same_user, command, found = False, "", set()
        try:
            same_user = proc.stat().st_uid == os.getuid()
            identity = process_identity(int(proc.name))
            if identity is None:
                continue
            command = identity["cmdline"]
            found = set(matches(command))
            if same_user:
                found.update(matches(os.readlink(proc / "cwd")))
                for fd in (proc / "fd").iterdir():
                    try:
                        found.update(matches(os.readlink(fd)))
                    except FileNotFoundError:
                        pass
                found.update(matches((proc / "maps").read_text()))
            else:
                foreign += 1
            if found:
                hits.append({"pid": int(proc.name), "targets": sorted(found)})
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            if found:
                hits.append({"pid": int(proc.name), "targets": sorted(found)})
            if same_user and session_helper(command) and not found:
                # Privilege-separated login helpers are non-dumpable. Do not
                # silently waive unreadable training or installer processes.
                helpers.append(int(proc.name))
            elif same_user:
                denied.append(int(proc.name))
            else:
                foreign += 1
    return {"hits": hits, "denied_same_uid_pids": denied,
            "non_dumpable_session_helper_pids": helpers,
            "foreign_processes_not_fully_inspectable": foreign,
            "coverage": "same-UID cmdline/cwd/fd/maps; visible foreign cmdlines only"}


def check_readers(paths):
    result = active_references(paths)
    if result["hits"] or result["denied_same_uid_pids"]:
        raise ValueError("Active readers or incomplete same-user inspection: " + json.dumps(result))
    return result


def main():
    import fcntl

    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-plan-sha256")
    args = parser.parse_args()
    if HOME.resolve() != HOME or HOME.is_symlink():
        raise ValueError("Authorized home directory is substituted")
    for path in [ROOT, ROOT / "outputs", ROOT / "runs", QUICK]:
        literal(path)
    if OUT.exists() or OUT.is_symlink():
        literal(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    literal(OUT)
    plan_path, result_path = OUT / "plan.json", OUT / "result.json"
    lock_path = ROOT / "runs/.charades-pilot.lock"
    if lock_path.exists() or lock_path.is_symlink():
        literal(lock_path)
    with lock_path.open("a") as gate:
        fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (QUICK / "training_budget.json").exists():
            raise ValueError("Training attempt already exists; do not clean during execution")
        if not args.execute_plan_sha256:
            if plan_path.exists() or result_path.exists():
                raise FileExistsError("One-shot cleanup plan already exists")
            targets = []
            for path in ALLOW:
                if not os.path.lexists(path):
                    continue
                record = snapshot(path)
                if path in TEMPS and time.time() - record["newest_mtime"] < 48 * 3600:
                    raise ValueError(f"Temporary artifact newer than 48 hours: {path}")
                targets.append(record)
            if sum(item["allocated_file_bytes"] for item in targets) > 35 * GIB:
                raise ValueError("Cleanup exceeds 35 GiB fixed scope")
            plan = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "disk_before": shutil.disk_usage(HOME).free,
                    "targets": targets, "protected": protected(), "queue": queue_identity(),
                    "reader_check": check_readers([Path(item["path"]) for item in targets])}
            write_json(plan_path, plan)
            print(json.dumps({"plan_sha256": canonical_hash(plan), "target_count": len(targets),
                              "allocated_file_gib": sum(t["allocated_file_bytes"] for t in targets) / GIB,
                              "disk_available_gib": plan["disk_before"] / GIB,
                              "reader_check": plan["reader_check"]}, indent=2))
            return
        if result_path.exists():
            raise FileExistsError("Cleanup attempt already recorded; no automatic rerun")
        plan = json.loads(plan_path.read_text())
        if canonical_hash(plan) != args.execute_plan_sha256:
            raise ValueError("Cleanup plan hash mismatch")
        paths = [Path(item["path"]) for item in plan["targets"]]
        if len(set(paths)) != len(paths) or any(path not in ALLOW for path in paths):
            raise ValueError("Non-allowlisted or duplicate cleanup path")
        if protected() != plan["protected"]:
            raise ValueError("Protected assets changed since planning")
        for item, path in zip(plan["targets"], paths):
            if snapshot(path) != item:
                raise ValueError(f"Target changed since planning: {path}")
        reader_check = check_readers(paths)
        report = {"status": "running", "plan_sha256": args.execute_plan_sha256,
                  "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "disk_before": shutil.disk_usage(HOME).free,
                  "reader_check": reader_check, "deleted": []}
        write_json(result_path, report)
        try:
            if not shutil.rmtree.avoids_symlink_attacks:
                raise ValueError("Platform lacks symlink-safe recursive removal")
            for item, path in zip(plan["targets"], paths):
                if snapshot(path) != item:
                    raise ValueError(f"Target changed immediately before removal: {path}")
                check_readers([path])
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                report["deleted"].append(item)
                write_json(result_path, report)
            report["protected_unchanged"] = protected() == plan["protected"]
            if not report["protected_unchanged"]:
                raise ValueError("Protected assets changed during cleanup")
            report["status"] = "pass"
        except BaseException as error:
            report.update(status="fail", error=str(error))
            raise
        finally:
            report["disk_after"] = shutil.disk_usage(HOME).free
            report["net_available_delta_bytes"] = report["disk_after"] - report["disk_before"]
            report["deleted_allocated_file_bytes"] = sum(t["allocated_file_bytes"] for t in report["deleted"])
            report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            write_json(result_path, report)
    # Wake the existing queue only after releasing its single-writer lock.
    expected_queue = plan["queue"]
    current_queue = queue_identity()
    if current_queue and current_queue == expected_queue and time.time() < json.loads((QUICK / "queue_status.json").read_text())["deadline_epoch"]:
        os.kill(current_queue["pid"], signal.SIGUSR1)
        report["queue_woken"] = {"pid": current_queue["pid"], "start_ticks": current_queue["start_ticks"]}
        write_json(result_path, report)
    print(json.dumps({k: v for k, v in report.items() if k != "deleted"}, indent=2))


if __name__ == "__main__":
    main()
