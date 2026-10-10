"""Temporary source-only, forced-command key; revoke only our exact key line."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time


BOOT = Path("/home/zhaopu/.e04-migration-20261008")
KEYS = Path("/home/zhaopu/.ssh/authorized_keys")
DEADLINE = 1791451152.0
RECORD = BOOT / "direct_access.json"
AUTH_SCRIPT = BOOT / "e04_direct_access.py"
REVOCATION_LOG = BOOT / "direct_revoke.log"


def save(record):
    temporary = RECORD.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, RECORD)


def revoke():
    record = json.loads(RECORD.read_text())
    line = record["authorized_line"].encode() + b"\n"
    if KEYS.is_symlink() or not KEYS.resolve().is_relative_to(Path("/home/zhaopu")):
        raise ValueError("Refuse auth edit outside the approved account")
    with KEYS.open("rb") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        before = os.fstat(handle.fileno())
        content = handle.read()
        rows = content.splitlines(keepends=True)
        matches = sum(row == line for row in rows)
        if matches > 1:
            raise ValueError("Unexpected duplicate temporary key; do not rewrite")
        if matches:
            updated = b"".join(row for row in rows if row != line)
            temporary = KEYS.with_name("authorized_keys.e04-revoke.tmp")
            with temporary.open("xb") as output:
                output.write(updated)
                output.flush()
                os.fsync(output.fileno())
            os.chmod(temporary, stat.S_IMODE(before.st_mode))
            os.chown(temporary, before.st_uid, before.st_gid)
            now = KEYS.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (now.st_ino, now.st_size, now.st_mtime_ns):
                raise ValueError("Concurrent authorization edit; do not overwrite it")
            os.replace(temporary, KEYS)
    record.update(revoked=True, revoked_at=time.time(), removed_lines=matches,
                  unrelated_authorized_lines_preserved=True)
    save(record)
    print(json.dumps({k: record[k] for k in ["revoked", "revoked_at", "removed_lines"]}), flush=True)


def install(public_key):
    if time.time() >= DEADLINE or RECORD.exists():
        raise ValueError("Expired/previous direct authorization; do not reinstall")
    parts = public_key.strip().split()
    if len(parts) < 2 or parts[0] != "ssh-ed25519" or any(c in parts[1] for c in '\"\\\n\r '):
        raise ValueError("Unexpected ephemeral public key")
    script = AUTH_SCRIPT
    command = f"/home/zhaopu/anaconda3/bin/python3 {script} forced"
    expiry = time.strftime("%Y%m%d%H%M%SZ", time.gmtime(DEADLINE))
    line = (f'from="202.207.1.21",restrict,expiry-time="{expiry}",command="{command}" '
            f"ssh-ed25519 {parts[1]} e04-migration-only-20261008")
    parent = KEYS.parent
    if parent.is_symlink() or KEYS.is_symlink() or not parent.resolve().is_relative_to(Path("/home/zhaopu")):
        raise ValueError("Unapproved authorization path")
    parent.mkdir(mode=0o700, exist_ok=True)
    descriptor = os.open(KEYS, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        info = os.fstat(handle.fileno())
        if info.st_uid != os.getuid():
            raise ValueError("Authorization file is not owned by this account")
        needs_newline = False
        if info.st_size:
            handle.seek(-1, os.SEEK_END)
            needs_newline = handle.read(1) != b"\n"
        handle.write((b"\n" if needs_newline else b"") + line.encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    record = {"installed_at": time.time(), "authorized_line": line, "deadline_epoch": DEADLINE,
              "source_ip_only": "202.207.1.21", "forced_command_only": True,
              "no_forwarding_or_pty": True, "revoked": False,
              "public_key_sha256": hashlib.sha256(parts[1].encode()).hexdigest()}
    save(record)
    with REVOCATION_LOG.open("x") as log:
        process = subprocess.Popen([sys.executable, str(script), "expiry"],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record["expiry_revoker_pid"] = process.pid
    save(record)
    print(json.dumps({k: record[k] for k in ["deadline_epoch", "source_ip_only", "public_key_sha256", "expiry_revoker_pid"]}), flush=True)


def forced():
    if time.time() >= DEADLINE:
        revoke()
        raise TimeoutError("Temporary migration authorization expired")
    request = os.environ.get("SSH_ORIGINAL_COMMAND", "")
    if request == "e04-probe":
        print(json.dumps({"restricted_direct_auth": "pass", "deadline_epoch": DEADLINE,
                          "allowed_request": "e04-receive", "gpu_work": False}), flush=True)
        return
    if request != "e04-receive":
        raise PermissionError("Temporary key permits only this migration receive")
    import migrate_e04
    migrate_e04.RESUME = migrate_e04.DIRECT_APPROVED = True
    migrate_e04.MAX_BYTES = 50 * 1024 ** 3
    def interrupted(number, frame):
        raise InterruptedError("Owned direct migration connection interrupted")
    for number in [signal.SIGTERM, signal.SIGHUP]:
        signal.signal(number, interrupted)
    try:
        migrate_e04.receive(DEADLINE)
    finally:
        revoke()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["install", "forced", "revoke", "expiry"])
    parser.add_argument("--public-key")
    args = parser.parse_args()
    if args.mode == "install":
        install(args.public_key)
    elif args.mode == "expiry":
        while time.time() < DEADLINE and not json.loads(RECORD.read_text()).get("revoked"):
            time.sleep(min(30, max(0, DEADLINE - time.time())))
        if not json.loads(RECORD.read_text()).get("revoked"):
            revoke()
    else:
        globals()[args.mode]()
