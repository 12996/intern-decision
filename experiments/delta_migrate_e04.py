"""One approved missing-file transfer; rehash the prefix without retransmitting it."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import migrate_e04 as m


BOOT = Path("/home/zhaopu/.e04-migration-20261008")
DEADLINE = 1791453122.0
LIMIT = 5 * 1024 ** 3
PREFIX = 147061
EXPECTED_BYTES = 45899393017
EXPECTED_ENTRIES = 162489
OLD_DEADLINE = 1791451152.0
KEY = BOOT / "e04_delta_ed25519"
STATUS = m.ROOT / "migration_delta_status.json"


def check():
    m.check_deadline(DEADLINE)


def record(**data):
    m.atomic_json(STATUS, {"pid": os.getpid(), "at": time.time(),
                          "deadline_epoch": DEADLINE, "delta_approved": True, **data})


def digest_file(path):
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as handle:
        while True:
            check()
            chunk = handle.read(m.CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("Input changed while hashing")
    return digest.hexdigest()


def send():
    stream = sys.stdout.buffer
    manifest = hashlib.sha256()
    payload = size = count = 0
    m.write_header(stream, {"type": "delta-start", "deadline_epoch": DEADLINE,
                            "prefix": PREFIX, "payload_limit": LIMIT})
    for source, header in m.entries():
        check()
        if count == PREFIX and header["path"] != "runtime/conda-xt310/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so":
            raise ValueError("Previously verified prefix no longer matches source traversal")
        reuse = count < PREFIX
        m.write_header(stream, {"header": header, "reuse": reuse})
        sha = ""
        if header["type"] == "file":
            size += header["size"]
            if reuse:
                stream.flush()
                sha = digest_file(source)
            else:
                payload += header["size"]
                if payload > LIMIT:
                    raise ValueError("Missing-file payload exceeds approved5GiB")
                before = source.stat()
                digest = hashlib.sha256()
                with source.open("rb") as handle:
                    remaining = header["size"]
                    while remaining:
                        check()
                        chunk = handle.read(min(m.CHUNK, remaining))
                        if not chunk:
                            raise ValueError("Source truncated")
                        stream.write(chunk)
                        digest.update(chunk)
                        remaining -= len(chunk)
                after = source.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                    raise ValueError("Source changed")
                sha = digest.hexdigest()
            m.write_header(stream, {"sha256": sha})
        m.append_manifest(manifest, header, sha)
        count += 1
    if (count, size) != (EXPECTED_ENTRIES, EXPECTED_BYTES):
        raise ValueError("Source inventory changed since approved migration")
    m.write_header(stream, {"type": "delta-end", "entries": count, "logical_file_bytes": size,
                            "payload_bytes": payload, "manifest_sha256": manifest.hexdigest()})
    stream.flush()


def receive():
    previous = json.loads((m.ROOT / "migration_status.json").read_text())
    if (m.ROOT.is_symlink() or previous.get("result") != "fail"
            or previous.get("deadline_epoch") != OLD_DEADLINE
            or previous.get("entries_complete") != PREFIX
            or previous.get("file_bytes_complete") != 43917905972
            or previous.get("direct_approved") is not True):
        raise ValueError("Not the approved failed-owned prefix")
    with (BOOT / "delta_receive_once.json").open("x") as marker:
        json.dump({"pid": os.getpid(), "deadline_epoch": DEADLINE}, marker)
    if shutil.disk_usage(m.ROOT).free < 100 * 1024 ** 3 + LIMIT:
        raise RuntimeError("Insufficient transfer/checkpoint/reserve space")
    stream, manifest = sys.stdin.buffer, hashlib.sha256()
    start = m.read_header(stream)
    if start != {"type": "delta-start", "deadline_epoch": DEADLINE, "prefix": PREFIX, "payload_limit": LIMIT}:
        raise ValueError("Delta scope mismatch")
    count = size = payload = 0
    next_report = 0
    try:
        while True:
            check()
            message = m.read_header(stream)
            if message.get("type") == "delta-end":
                expected = {"type": "delta-end", "entries": count, "logical_file_bytes": size,
                            "payload_bytes": payload, "manifest_sha256": manifest.hexdigest()}
                if message != expected or (count, size) != (EXPECTED_ENTRIES, EXPECTED_BYTES):
                    raise ValueError("Whole delta manifest mismatch")
                break
            header, reuse = message["header"], message["reuse"]
            if reuse != (count < PREFIX):
                raise ValueError("Prefix reuse boundary drift")
            path = m.destination(header["path"])
            kind, sha = header["type"], ""
            if kind == "directory":
                if path.is_symlink():
                    raise ValueError("Directory replaced by symlink")
                path.mkdir(parents=True, exist_ok=True)
            elif kind == "link":
                if not Path(header["target"]).resolve().is_relative_to(m.ROOT):
                    raise ValueError("External relocated link")
                if path.is_symlink() and os.readlink(path) == header["target"]:
                    pass
                elif not path.exists() and not path.is_symlink() and not reuse:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.symlink_to(header["target"])
                else:
                    raise ValueError("Existing link differs")
            elif kind == "file":
                length = header["size"]
                if not isinstance(length, int) or length < 0 or size + length > EXPECTED_BYTES:
                    raise ValueError("Invalid file length")
                if path.is_symlink():
                    raise ValueError("File replaced by symlink")
                if reuse:
                    if not path.is_file() or path.stat().st_size != length:
                        raise ValueError("Verified prefix missing or changed")
                    sha = digest_file(path)
                else:
                    payload += length
                    if payload > LIMIT:
                        raise ValueError("Delta payload exceeds5GiB")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.exists():
                        if path.stat().st_size == length:
                            raise ValueError("Unexpected complete file beyond verified boundary")
                        partial = path.with_name(path.name + ".migration-incomplete-delta")
                        if partial.exists():
                            raise FileExistsError("Preserve previous partial artifact")
                        path.rename(partial)
                    digest = hashlib.sha256()
                    with path.open("xb") as handle:
                        remaining = length
                        while remaining:
                            check()
                            chunk = stream.read(min(m.CHUNK, remaining))
                            if not chunk:
                                raise ValueError("Truncated delta")
                            handle.write(chunk)
                            digest.update(chunk)
                            remaining -= len(chunk)
                    os.chmod(path, header["mode"])
                    sha = digest.hexdigest()
                if m.read_header(stream) != {"sha256": sha}:
                    raise ValueError("Delta file SHA mismatch")
                size += length
            else:
                raise ValueError("Invalid entry type")
            m.append_manifest(manifest, header, sha)
            count += 1
            if time.time() >= next_report:
                record(phase="delta_hashing_or_copying_not_training", entries_complete=count,
                       file_bytes_complete=size, new_payload_bytes=payload, last_path=header["path"])
                next_report = time.time() + 60
        cfg = m.ROOT / ".venv-xt310/pyvenv.cfg"
        old = cfg.read_text()
        if f"home = {m.BASE}/bin" not in old:
            raise ValueError("Unexpected copied Python prefix")
        cfg.write_text(old.replace(str(m.BASE), str(m.NEW_BASE)))
        record(result="pass", phase="copied_sha256_verified_environment_not_yet_tested",
               entries_complete=count, file_bytes_complete=size, new_payload_bytes=payload,
               manifest_sha256=manifest.hexdigest(), venv_prefix_relocated=True,
               original_source_untouched=True)
    except BaseException as error:
        record(result="fail", phase="delta_failed_no_automatic_retry", error=str(error),
               entries_complete=count, file_bytes_complete=size, new_payload_bytes=payload)
        raise


def ssh():
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ForwardAgent=no",
            "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=10", "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={BOOT / 'target_known_hosts'}", "-i", str(KEY),
            "zhaopu@202.207.1.22", "e04-delta-receive"]


def controller():
    children = []
    def stop():
        for child in children:
            if child.poll() is None:
                child.kill()
    timer = threading.Timer(max(0, DEADLINE - time.time()), stop)
    timer.daemon = True
    timer.start()
    wire = 0
    try:
        check()
        with (BOOT / "delta_receiver.log").open("x") as log:
            receiver = subprocess.Popen(ssh(), stdin=subprocess.PIPE, stdout=log)
            sender = subprocess.Popen([sys.executable, __file__, "send"], stdout=subprocess.PIPE)
            children.extend([receiver, sender])
            while True:
                check()
                chunk = sender.stdout.read1(m.CHUNK)
                if not chunk:
                    break
                wire += len(chunk)
                if wire > LIMIT:
                    raise ValueError("New wire allowance exceeded5GiB")
                receiver.stdin.write(chunk)
                receiver.stdin.flush()
            receiver.stdin.close()
            codes = [sender.wait(), receiver.wait()]
            if codes != [0, 0]:
                raise RuntimeError(f"Delta subprocess failure: {codes}")
        result = {"result": "pass", "wire_bytes": wire, "deadline_epoch": DEADLINE}
    except BaseException as error:
        result = {"result": "fail", "error": str(error), "wire_bytes": wire, "deadline_epoch": DEADLINE}
        raise
    finally:
        timer.cancel()
        stop()
        m.atomic_json(BOOT / "delta_sender_status.json", result)
        for path in [KEY, KEY.with_suffix(".pub")]:
            if path.is_symlink() or path.parent.resolve() != BOOT:
                raise ValueError("Unexpected owned key path")
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["send", "receive", "controller", "spawn", "generate", "prepare"])
    args = parser.parse_args()
    if args.mode == "prepare":
        check()
        if STATUS.exists():
            raise FileExistsError("Do not restart a delta attempt")
        charged = 15 * 1024 ** 3 + EXPECTED_BYTES + LIMIT
        if charged > 65 * 1024 ** 3:
            raise ValueError("Cumulative migration allowance exceeded")
        record(phase="waiting_for_delta_not_training", cumulative_charge_upper_bound_bytes=charged,
               new_wire_and_payload_limit_bytes=LIMIT, gpu_work=False)
    elif args.mode == "generate":
        check()
        if KEY.exists() or KEY.with_suffix(".pub").exists():
            raise FileExistsError("Do not replace an existing temporary key")
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C",
                        "e04-delta-only-20261008", "-f", str(KEY)], check=True)
        print(KEY.with_suffix(".pub").read_text().strip(), flush=True)
    elif args.mode == "spawn":
        check()
        with (BOOT / "delta_start.json").open("x") as marker:
            with (BOOT / "delta_transfer.log").open("x") as log:
                child = subprocess.Popen([sys.executable, __file__, "controller"], stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            json.dump({"pid": child.pid, "deadline_epoch": DEADLINE}, marker)
        print(json.dumps({"pid": child.pid, "deadline_epoch": DEADLINE}), flush=True)
    else:
        globals()[args.mode]()
