"""Source-host binary transfer through one temporary forced-command SSH key."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
import time


BOOT = Path("/home/zhaopu/.e04-migration-20261008")
KEY = BOOT / "e04_direct_ed25519"
DEADLINE = 1791451152.0
LIMIT = 50 * 1024 ** 3


def ssh(request):
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ForwardAgent=no",
            "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=10", "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={BOOT / 'target_known_hosts'}", "-i", str(KEY),
            "zhaopu@202.207.1.22", request]


def record(**details):
    path = BOOT / "direct_transfer_status.json"
    temporary = path.with_suffix(".tmp")
    data = {"pid": __import__("os").getpid(), "at": time.time(), "deadline_epoch": DEADLINE, **details}
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps(data), flush=True)


def run():
    processes = []
    def stop():
        for process in processes:
            if process.poll() is None:
                process.kill()
    timer = threading.Timer(max(0, DEADLINE - time.time()), stop)
    timer.daemon = True
    timer.start()
    transferred = 0
    try:
        if time.time() >= DEADLINE:
            raise TimeoutError("Original migration deadline expired")
        subprocess.run(ssh("e04-probe"), check=True, timeout=20)
        record(phase="direct_copy_not_training", source_alias="5090_1", destination_alias="5090",
               prior_attempts_charge_upper_bound_bytes=15 * 1024 ** 3, new_wire_logical_limit_bytes=LIMIT)
        with (BOOT / "direct_receiver.log").open("x") as receipt:
            receiver = subprocess.Popen(ssh("e04-receive"), stdin=subprocess.PIPE, stdout=receipt)
            processes.append(receiver)
            sender = subprocess.Popen([sys.executable, str(BOOT / "migrate_e04.py"), "send",
                                       "--deadline-epoch", str(DEADLINE), "--direct-approved"], stdout=subprocess.PIPE)
            processes.append(sender)
            next_report = 0
            while True:
                if time.time() >= DEADLINE:
                    raise TimeoutError("Original migration deadline expired")
                chunk = sender.stdout.read(1024 ** 2)
                if not chunk:
                    break
                transferred += len(chunk)
                if transferred > LIMIT:
                    raise ValueError("Cumulative migration allowance would exceed65GiB")
                receiver.stdin.write(chunk)
                if time.time() >= next_report:
                    record(phase="direct_copy_not_training", compressed_wire_bytes=transferred)
                    next_report = time.time() + 60
            receiver.stdin.close()
            sender_code, receiver_code = sender.wait(), receiver.wait()
            if sender_code or receiver_code:
                raise RuntimeError(f"Direct transfer failed: sender={sender_code}, receiver={receiver_code}")
        record(result="pass", phase="direct_transfer_sha256_verified_not_training", compressed_wire_bytes=transferred)
    except BaseException as error:
        record(result="fail", phase="direct_transfer_failed_no_automatic_retry", error=str(error),
               compressed_wire_bytes=transferred)
        raise
    finally:
        timer.cancel()
        stop()
        # These are only the temporary key artifacts generated for this transfer.
        for path in [KEY, KEY.with_suffix(".pub")]:
            if path.is_symlink() or path.parent.resolve() != BOOT:
                raise ValueError("Refuse cleanup outside the owned bootstrap key paths")
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spawn", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.generate:
        if time.time() >= DEADLINE or KEY.exists() or KEY.with_suffix(".pub").exists():
            raise ValueError("Do not replace existing/expired key artifacts")
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C",
                        "e04-migration-only-20261008", "-f", str(KEY)], check=True)
        print(json.dumps({"public_key": KEY.with_suffix(".pub").read_text().strip()}), flush=True)
    elif args.probe:
        subprocess.run(ssh("e04-probe"), check=True, timeout=20)
    elif args.spawn:
        if time.time() >= DEADLINE:
            raise TimeoutError("Expired migration scope")
        with (BOOT / "direct_start.json").open("x") as marker:
            with (BOOT / "direct_transfer.log").open("x") as log:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve())],
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            data = {"pid": process.pid, "deadline_epoch": DEADLINE}
            marker.write(json.dumps(data, indent=2) + "\n")
        print(json.dumps(data), flush=True)
    else:
        run()
