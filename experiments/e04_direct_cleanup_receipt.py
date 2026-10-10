"""Emit only non-secret cleanup evidence for the expired owned SSH channel."""

import argparse
import json
from pathlib import Path
import time


BOOT = Path("/home/zhaopu/.e04-migration-20261008")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("host", choices=["source", "destination"])
    parser.add_argument("--delta", action="store_true")
    args = parser.parse_args()
    receipt = {"host_role": args.host, "checked_at": time.time(), "gpu_work": False}
    if args.host == "source":
        key = BOOT / ("e04_delta_ed25519" if args.delta else "e04_direct_ed25519")
        receipt["temporary_private_key_absent"] = not key.exists() and not key.is_symlink()
        public = key.with_suffix(".pub")
        receipt["temporary_public_key_absent"] = not public.exists() and not public.is_symlink()
    else:
        record = json.loads((BOOT / ("direct_access_delta.json" if args.delta else "direct_access.json")).read_text())
        receipt.update({name: record[name] for name in [
            "revoked", "revoked_at", "removed_lines", "unrelated_authorized_lines_preserved",
            "deadline_epoch", "source_ip_only", "forced_command_only", "no_forwarding_or_pty"]})
        keys = Path("/home/zhaopu/.ssh/authorized_keys")
        if keys.is_symlink() or not keys.resolve().is_relative_to(Path("/home/zhaopu")):
            raise ValueError("Unexpected authorization path")
        receipt["temporary_authorization_absent"] = record["authorized_line"] not in keys.read_text().splitlines()
    suffix = "_delta" if args.delta else ""
    path = BOOT / f"direct_cleanup_{args.host}{suffix}.json"
    with path.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
