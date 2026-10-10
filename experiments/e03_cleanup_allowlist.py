#!/usr/bin/env python3
"""Delete only the four approved full-state checkpoints on 5090_1.

Missing targets are recorded and skipped. A symlink, path substitution,
or live reader aborts before any deletion. HF weights and DCP750 stay.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path("/home/zhaopu/intern-decision")
ALLOW = [
    ROOT / "runs/charades-id2b-e03-20260929-224918/20260929225053/checkpoints/ckpt-step-250",
    ROOT / "runs/charades-id2b-e03-20260929-224918/20260929225053/checkpoints/ckpt-step-500",
    ROOT / "runs/charades-id2b-dcp-smoke-20260929-205009/20260929205218/checkpoints/ckpt-step-2",
    ROOT / "runs/charades-id2b-dcp-smoke-20260929-205009/20260929205218/checkpoints/ckpt-step-3",
]
KEEP_DCP = ROOT / "runs/charades-id2b-e03-20260929-224918/20260929225053/checkpoints/ckpt-step-750"
EXP = ROOT / "runs/charades-id2b-e03-20260929-224918/20260929225053"
KEEP_HF = [EXP / f"hf-{step}" for step in range(100, 900, 100)]
OLD_LOG = ROOT / "runs/charades-id2b-e03-20260929-224918/log.txt"
OLD_JSONL = ROOT / "runs/charades-id2b-e03-20260929-224918/scaleup_steps.jsonl"
OLD_META = ROOT / "runs/charades-id2b-e03-20260929-224918/.xtuner"
OUT = ROOT / "outputs/e03-resume-20261002"


def du(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        base = Path(dirpath)
        if base.is_symlink():
            raise SystemExit(f"refuse symlink directory {base}")
        for name in list(dirnames):
            child = base / name
            if child.is_symlink():
                raise SystemExit(f"refuse symlink child {child}")
        for name in filenames:
            item = base / name
            if item.is_symlink():
                raise SystemExit(f"refuse symlink file {item}")
            total += item.stat().st_size
    return total


def df_home():
    st = os.statvfs("/home")
    return {
        "available_bytes": st.f_bavail * st.f_frsize,
        "free_bytes": st.f_bfree * st.f_frsize,
        "total_bytes": st.f_blocks * st.f_frsize,
    }


def literal_dir(path):
    """Scoped literal directory: absolute, no symlink component, inside ROOT.

    A missing path is reported and not created. This check has no parent-name
    rule, because preserved HF exports sit under the experiment timestamp
    while deletion targets sit under checkpoints/.
    """
    if not isinstance(path, Path):
        path = Path(path)
    if not path.is_absolute():
        raise SystemExit(f"refuse relative path {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if os.path.islink(current):
            raise SystemExit(f"refuse symlink component {current} -> {os.readlink(current)}")
    if not os.path.lexists(path):
        return "missing"
    resolved = path.resolve()
    if str(resolved) != str(path):
        raise SystemExit(f"refuse path substitution {path} -> {resolved}")
    if not str(resolved).startswith(str(ROOT) + os.sep):
        raise SystemExit(f"refuse path outside project root {resolved}")
    if not path.is_dir():
        raise SystemExit(f"refuse non-directory {path}")
    return "present"


def deletion_target_dir(path):
    """Four-target allowlist only: the directory parent must be checkpoints/."""
    if not isinstance(path, Path):
        path = Path(path)
    if path.parent.name != "checkpoints":
        raise SystemExit(f"refuse unexpected parent {path}")
    return literal_dir(path)


def readers(path):
    target = str(path)
    found = []
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            cwd = os.readlink(entry / "cwd")
        except OSError:
            continue
        owned = "intern-decision" in command or "intern-decision" in cwd
        hit = target in command or cwd == target or cwd.startswith(target + os.sep)
        if not hit:
            fd_dir = entry / "fd"
            try:
                for fd in fd_dir.iterdir():
                    try:
                        link = os.readlink(fd)
                    except OSError:
                        continue
                    if link == target or link.startswith(target + os.sep):
                        hit = True
                        break
            except OSError:
                pass
        if hit:
            found.append({"pid": int(entry.name), "project_process": owned})
    return found


def fingerprint(path):
    info = {"path": str(path), "exists": path.exists(), "symlink": os.path.islink(path)}
    if path.is_file() and not os.path.islink(path):
        stat = path.stat()
        info.update({"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return info


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    dry_run = False
    if args == ["--dry-run"]:
        dry_run = True
    elif args:
        raise SystemExit(f"refuse unknown arguments {args}")
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dry_run": dry_run,
        "disk_before": df_home(),
        "targets": [],
        "deleted": [],
        "skipped_missing": [],
        "kept": {},
    }
    busy = []
    present = []
    for path in ALLOW:
        state = deletion_target_dir(path)
        item = {"path": str(path), "state": state}
        if state == "present":
            users = readers(path)
            item["bytes"] = du(path)
            item["readers"] = users
            if users:
                busy.append(item)
            present.append(path)
        report["targets"].append(item)
    if literal_dir(KEEP_DCP) != "present":
        raise SystemExit("refuse cleanup because DCP750 is absent")
    keep_readers = readers(KEEP_DCP)
    if keep_readers:
        raise SystemExit("refuse cleanup because DCP750 has readers " + json.dumps(keep_readers))
    for hf in KEEP_HF:
        if literal_dir(hf) != "present":
            raise SystemExit(f"refuse cleanup because required HF is absent {hf}")
    if busy:
        report["status"] = "aborted_readers"
        report["busy"] = busy
        print(json.dumps(report, indent=2))
        raise SystemExit("refuse cleanup because a target has an active reader")
    report["kept_before"] = {
        "dcp750_bytes": du(KEEP_DCP),
        "log": fingerprint(OLD_LOG),
        "jsonl": fingerprint(OLD_JSONL),
        "meta": fingerprint(OLD_META),
        "hf_shards": [],
    }
    for hf in KEEP_HF:
        for shard in sorted(hf.glob("*.safetensors")):
            report["kept_before"]["hf_shards"].append(fingerprint(shard))
    if dry_run:
        report["would_delete"] = [str(path) for path in present]
        report["would_delete_bytes"] = sum(item.get("bytes", 0) for item in report["targets"] if item["state"] == "present")
        report["status"] = "dry_run"
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        destination = OUT / "cleanup-dry-run.json"
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": report["status"],
            "would_delete": report["would_delete"],
            "would_delete_bytes": report["would_delete_bytes"],
            "skipped_missing": [str(path) for path in ALLOW if str(path) not in report["would_delete"]],
            "disk_before": report["disk_before"],
            "dcp750_bytes": report["kept_before"]["dcp750_bytes"],
            "hf_count": len(KEEP_HF),
            "hf_shard_count": len(report["kept_before"]["hf_shards"]),
            "record": str(destination),
        }, indent=2))
        return
    if OLD_META.is_file() and not os.path.islink(OLD_META):
        backup = OUT / "xtuner-before-cleanup.json"
        backup.write_bytes(OLD_META.read_bytes())
    for path in present:
        print(f"DELETE {path}", flush=True)
        shutil.rmtree(path)
        if path.exists() or os.path.islink(path):
            raise SystemExit(f"delete failed {path}")
        report["deleted"].append(str(path))
    for path in ALLOW:
        if str(path) not in report["deleted"]:
            report["skipped_missing"].append(str(path))
    kept_after = {
        "dcp750_bytes": du(KEEP_DCP),
        "log": fingerprint(OLD_LOG),
        "jsonl": fingerprint(OLD_JSONL),
        "meta": fingerprint(OLD_META),
        "hf_shards": [fingerprint(shard) for hf in KEEP_HF for shard in sorted(hf.glob("*.safetensors"))],
    }
    if kept_after["dcp750_bytes"] != report["kept_before"]["dcp750_bytes"]:
        raise SystemExit("DCP750 bytes changed during cleanup")
    if kept_after["log"] != report["kept_before"]["log"] or kept_after["jsonl"] != report["kept_before"]["jsonl"]:
        raise SystemExit("original log or JSONL changed during cleanup")
    if kept_after["meta"] != report["kept_before"]["meta"]:
        raise SystemExit("original .xtuner changed during cleanup")
    if kept_after["hf_shards"] != report["kept_before"]["hf_shards"]:
        raise SystemExit("HF shard fingerprint changed during cleanup")
    report["kept_after"] = {"dcp750_bytes": kept_after["dcp750_bytes"], "hf_count": len(KEEP_HF)}
    report["disk_after"] = df_home()
    report["deleted_bytes"] = sum(item.get("bytes", 0) for item in report["targets"] if item["path"] in report["deleted"])
    report["status"] = "deleted"
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    destination = OUT / "cleanup-allowlist.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "deleted": report["deleted"],
        "skipped_missing": report["skipped_missing"],
        "deleted_bytes": report["deleted_bytes"],
        "disk_before": report["disk_before"],
        "disk_after": report["disk_after"],
        "dcp750_bytes": kept_after["dcp750_bytes"],
        "record": str(destination),
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"CLEANUP_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
