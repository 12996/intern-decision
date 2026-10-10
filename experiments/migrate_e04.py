"""One bounded binary relay for the approved E04 assets; no deletes/downloads."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import threading
import time


ROOT = Path("/home/zhaopu/intern-decision")
BASE = Path("/data/zhaopu/wang-2024-gmmformer-v2/conda")
NEW_BASE = ROOT / "runtime/conda-xt310"
BOOTSTRAP = "/home/zhaopu/.e04-migration-20261008/migrate_e04.py"
MAX_BYTES = 65 * 1024 ** 3
RESUME = False
DIRECT_APPROVED = False
FIRST_ATTEMPT_BOUND = 5 * 1024 ** 3
SECOND_ATTEMPT_BOUND = 10 * 1024 ** 3
CHUNK = 1024 ** 2
STEP0 = "runs/charades-id2b-e04-balanced-20261003-144345/20261003144427/checkpoints/ckpt-step-0"
SELECTED = [
    "models/Intern-Decision-2B", "frames/id_pilot_frames_8", "Intern-Decision",
    ".venv-xt310", "torch-reuse-cp310", "diagnose_loss_contrast.py",
    "outputs/goal-20261003-quick-v1", STEP0,
    "outputs/goal-20261003-v2/benchmark_budget.json",
    "outputs/goal-20261003-profile-v1/profile_budget.json",
    "outputs/goal-20261008-retry-v1/code",
]


def check_deadline(deadline):
    if time.time() >= deadline:
        raise TimeoutError("Authorized migration deadline expired")


def destination(relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(p in (".", "..") for p in path.parts):
        raise ValueError("Unsafe relative destination")
    result = ROOT / str(path)
    if not result.resolve().is_relative_to(ROOT):
        raise ValueError("Destination escapes approved new project")
    return result


def relocated_link(source, target):
    resolved = (source.parent / target).resolve() if not os.path.isabs(target) else Path(target).resolve()
    if resolved.is_relative_to(BASE):
        return str(NEW_BASE / resolved.relative_to(BASE))
    if resolved.is_relative_to(ROOT):
        return str(resolved)
    raise ValueError(f"Unapproved external symlink: {source}")


def entries():
    pairs = [(ROOT / item, item) for item in SELECTED] + [(BASE, "runtime/conda-xt310")]
    def walk(source, relative):
        info = source.lstat()
        header = {"path": relative, "mode": stat.S_IMODE(info.st_mode) & 0o777}
        if stat.S_ISLNK(info.st_mode):
            header.update(type="link", target=relocated_link(source, os.readlink(source)))
        elif stat.S_ISDIR(info.st_mode):
            header["type"] = "directory"
        elif stat.S_ISREG(info.st_mode):
            header.update(type="file", size=info.st_size)
        else:
            raise ValueError(f"Special files are not copied: {source}")
        yield source, header
        if header["type"] == "directory":
            for child in sorted(source.iterdir()):
                yield from walk(child, relative + "/" + child.name)
    for source, relative in pairs:
        yield from walk(source, relative)


def inventory(deadline):
    count, size, links = 0, 0, 0
    for source, header in entries():
        check_deadline(deadline)
        count += 1
        size += header.get("size", 0)
        links += header["type"] == "link"
        if size > MAX_BYTES:
            raise ValueError("Source logical bytes exceed approved65GiB")
    result = {"result": "pass", "phase": "inventory_not_training", "entries": count,
              "logical_file_bytes": size, "symlinks": links, "selected": SELECTED,
              "base_environment": str(BASE), "deadline_epoch": deadline, "limit_bytes": MAX_BYTES}
    print(json.dumps(result), flush=True)


def write_header(stream, header):
    stream.write(json.dumps(header, separators=(",", ":")).encode() + b"\n")


def read_header(stream):
    line = stream.readline(65537)
    if not line or len(line) > 65536 or not line.endswith(b"\n"):
        raise ValueError("Truncated/oversized migration header")
    return json.loads(line)


def append_manifest(digest, header, content_sha=""):
    digest.update(json.dumps([header, content_sha], sort_keys=True).encode() + b"\n")


def send(deadline):
    stream = gzip.GzipFile(fileobj=sys.stdout.buffer, mode="wb", compresslevel=1)
    manifest = hashlib.sha256()
    count, size = 0, 0
    write_header(stream, {"type": "start", "deadline_epoch": deadline, "limit_bytes": MAX_BYTES})
    for source, header in entries():
        check_deadline(deadline)
        count += 1
        size += header.get("size", 0)
        if size > MAX_BYTES:
            raise ValueError("65GiB source limit reached")
        write_header(stream, header)
        content_sha = ""
        if header["type"] == "file":
            before = source.stat()
            if before.st_size != header["size"]:
                raise ValueError("Source size changed before transfer")
            digest = hashlib.sha256()
            with source.open("rb") as handle:
                remaining = header["size"]
                while remaining:
                    check_deadline(deadline)
                    chunk = handle.read(min(CHUNK, remaining))
                    if not chunk:
                        raise ValueError("Source file shrank during transfer")
                    stream.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError("Source changed during transfer")
            content_sha = digest.hexdigest()
            write_header(stream, {"sha256": content_sha})
        append_manifest(manifest, header, content_sha)
    write_header(stream, {"type": "end", "entries": count, "logical_file_bytes": size,
                          "manifest_sha256": manifest.hexdigest()})
    stream.close()
    sys.stdout.buffer.flush()


def atomic_json(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, path)


def receive(deadline):
    previous = None
    if RESUME:
        previous = json.loads((ROOT / "migration_status.json").read_text())
        if (ROOT.is_symlink() or previous.get("result") != "fail"
                or previous.get("phase") != "incomplete_copy_no_training"
                or previous.get("deadline_epoch") != deadline
                or previous.get("source_alias") != "5090_1"
                or previous.get("destination_alias") != "5090"
                or (previous.get("previous_attempt") is not None and not DIRECT_APPROVED)
                or (DIRECT_APPROVED and (previous.get("previous_attempt") is None
                                        or previous.get("direct_approved") is True))):
            raise ValueError("Only one bounded recovery of this owned failed copy is allowed")
    elif ROOT.exists() or ROOT.is_symlink():
        raise FileExistsError("Never overwrite an existing destination project")
    if shutil.disk_usage(ROOT.parent).free < MAX_BYTES + 100 * 1024 ** 3:
        raise RuntimeError("Destination has insufficient transfer/save/reserve space")
    stream = gzip.GzipFile(fileobj=sys.stdin.buffer, mode="rb")
    manifest = hashlib.sha256()
    start = read_header(stream)
    if start != {"type": "start", "deadline_epoch": deadline, "limit_bytes": MAX_BYTES}:
        raise ValueError("Migration scope/deadline mismatch")
    ROOT.mkdir(exist_ok=RESUME)
    status = ROOT / "migration_status.json"
    count, size, next_report = 0, 0, 0
    record = {"phase": "copying_not_training", "pid": os.getpid(), "deadline_epoch": deadline,
              "start_epoch": time.time(), "source_alias": "5090_1", "destination_alias": "5090"}
    if previous:
        prior_bound = FIRST_ATTEMPT_BOUND + (SECOND_ATTEMPT_BOUND if DIRECT_APPROVED else 0)
        record.update(previous_attempt=previous, prior_attempt_bytes_charged_upper_bound=prior_bound)
    if DIRECT_APPROVED:
        record.update(direct_approved=True, transfer_channel="forced_command_server_to_server_ssh",
                      attempt_limit_bytes=MAX_BYTES)
    try:
        while True:
            check_deadline(deadline)
            header = read_header(stream)
            if header["type"] == "end":
                if header != {"type": "end", "entries": count, "logical_file_bytes": size,
                              "manifest_sha256": manifest.hexdigest()}:
                    raise ValueError("Whole transfer manifest mismatch")
                break
            path = destination(header["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            content_sha = ""
            if header["type"] == "directory":
                path.mkdir(exist_ok=True)
            elif header["type"] == "link":
                if not Path(header["target"]).resolve().is_relative_to(ROOT):
                    raise ValueError("Relocated symlink escapes destination")
                if RESUME and path.is_symlink() and os.readlink(path) == header["target"]:
                    pass
                else:
                    path.symlink_to(header["target"])
            elif header["type"] == "file":
                length = header["size"]
                if not isinstance(length, int) or length < 0 or size + length > MAX_BYTES:
                    raise ValueError("Approved65GiB file-byte limit exceeded")
                digest = hashlib.sha256()
                existing = RESUME and path.is_file() and not path.is_symlink()
                if existing and path.stat().st_size != length:
                    # Keep the owned incomplete artifact instead of deleting it.
                    suffix = ".migration-incomplete-a2" if DIRECT_APPROVED else ".migration-incomplete-a1"
                    partial = path.with_name(path.name + suffix)
                    if partial.exists():
                        raise FileExistsError("Do not overwrite an earlier partial artifact")
                    path.rename(partial)
                    existing = False
                with path.open("rb" if existing else "xb") as handle:
                    remaining = length
                    while remaining:
                        check_deadline(deadline)
                        chunk = stream.read(min(CHUNK, remaining))
                        if not chunk:
                            raise ValueError("Truncated file; no migration pass")
                        if existing:
                            if handle.read(len(chunk)) != chunk:
                                raise ValueError("Existing destination file differs; never overwrite it")
                        else:
                            handle.write(chunk)
                        digest.update(chunk)
                        remaining -= len(chunk)
                        if time.time() >= next_report:
                            atomic_json(status, {**record, "entries_complete": count, "file_bytes_complete": size,
                                                 "current_file_bytes": length - remaining,
                                                 "last_path": header["path"], "at": time.time()})
                            next_report = time.time() + 300
                content_sha = digest.hexdigest()
                if read_header(stream) != {"sha256": content_sha}:
                    raise ValueError("Copied file SHA256 mismatch")
                size += length
                if not existing:
                    os.chmod(path, header["mode"])
            else:
                raise ValueError("Unknown migration entry type")
            count += 1
            append_manifest(manifest, header, content_sha)
            if time.time() >= next_report:
                atomic_json(status, {**record, "entries_complete": count, "file_bytes_complete": size,
                                     "last_path": header["path"], "at": time.time()})
                next_report = time.time() + 300
        # Only this copied, generated environment file needs a prefix change.
        cfg = ROOT / ".venv-xt310/pyvenv.cfg"
        old = cfg.read_text()
        if f"home = {BASE}/bin" not in old:
            raise ValueError("Unexpected venv prefix; refuse blind relocation")
        cfg.write_text(old.replace(str(BASE), str(NEW_BASE)))
        record.update(result="pass", phase="copied_sha256_verified_environment_not_yet_tested",
                      entries_complete=count, file_bytes_complete=size,
                      manifest_sha256=manifest.hexdigest(), end_epoch=time.time(),
                      venv_prefix_relocated=True, original_source_untouched=True)
        atomic_json(status, record)
        print(json.dumps(record), flush=True)
    except BaseException as error:
        atomic_json(status, {**record, "result": "fail", "phase": "incomplete_copy_no_training",
                             "error": str(error), "entries_complete": count, "file_bytes_complete": size})
        raise


def relay(deadline):
    check_deadline(deadline)
    ssh_options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=10"]
    def command(alias, mode):
        recovery = " --resume-incomplete" if RESUME else ""
        return ["ssh", *ssh_options, alias, f"python3 {BOOTSTRAP} {mode} --deadline-epoch {deadline}{recovery}"]
    processes = []
    def stop():
        for process in processes:
            if process.poll() is None:
                process.kill()
    timer = threading.Timer(max(0, deadline - time.time()), stop)
    timer.daemon = True
    timer.start()
    try:
        receiver = subprocess.Popen(command("5090", "receive"), stdin=subprocess.PIPE)
        processes.append(receiver)
        sender = subprocess.Popen(command("5090_1", "send"), stdout=subprocess.PIPE)
        processes.append(sender)
        transferred, next_report = 0, time.time()
        while True:
            check_deadline(deadline)
            chunk = sender.stdout.read(CHUNK)
            if not chunk:
                break
            transferred += len(chunk)
            if transferred > MAX_BYTES:
                raise ValueError("Approved65GiB relay wire-byte limit reached")
            receiver.stdin.write(chunk)
            if time.time() >= next_report:
                print(json.dumps({"phase": "relaying_not_training", "wire_bytes": transferred,
                                  "at": time.time(), "deadline_epoch": deadline}), flush=True)
                next_report = time.time() + 300
        receiver.stdin.close()
        sender_code, receiver_code = sender.wait(), receiver.wait()
        if sender_code or receiver_code:
            raise RuntimeError(f"Migration failed: sender={sender_code}, receiver={receiver_code}")
        print(json.dumps({"relay_result": "pass", "wire_bytes": transferred}), flush=True)
    finally:
        timer.cancel()
        stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["inventory", "send", "receive", "relay"])
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--resume-incomplete", action="store_true")
    parser.add_argument("--direct-approved", action="store_true")
    args = parser.parse_args()
    if args.direct_approved:
        if args.mode not in ["send", "receive"] or args.resume_incomplete:
            raise ValueError("Approved direct copy is not another laptop relay")
        DIRECT_APPROVED = RESUME = True
        MAX_BYTES -= FIRST_ATTEMPT_BOUND + SECOND_ATTEMPT_BOUND
    elif args.resume_incomplete:
        RESUME = True
        MAX_BYTES -= FIRST_ATTEMPT_BOUND
    globals()[args.mode](args.deadline_epoch)
