"""New bounded authorization for only the approved missing-file receive."""

import argparse
import json
import os
from pathlib import Path
import signal
import time

import e04_direct_access as access
import delta_migrate_e04 as delta


access.DEADLINE = delta.DEADLINE
access.RECORD = access.BOOT / "direct_access_delta.json"
access.AUTH_SCRIPT = access.BOOT / "e04_delta_access.py"
access.REVOCATION_LOG = access.BOOT / "delta_revoke.log"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["install", "forced", "expiry", "revoke"])
    parser.add_argument("--public-key")
    args = parser.parse_args()
    if args.mode == "install":
        access.install(args.public_key)
    elif args.mode == "revoke":
        access.revoke()
    elif args.mode == "expiry":
        while time.time() < delta.DEADLINE and not json.loads(access.RECORD.read_text()).get("revoked"):
            time.sleep(min(15, max(0, delta.DEADLINE - time.time())))
        if not json.loads(access.RECORD.read_text()).get("revoked"):
            access.revoke()
    else:
        delta.check()
        if os.environ.get("SSH_ORIGINAL_COMMAND") != "e04-delta-receive":
            raise PermissionError("This key permits only the approved delta receive")
        def interrupted(number, frame):
            raise InterruptedError("Owned delta connection interrupted")
        for number in [signal.SIGTERM, signal.SIGHUP]:
            signal.signal(number, interrupted)
        try:
            delta.receive()
        finally:
            access.revoke()
