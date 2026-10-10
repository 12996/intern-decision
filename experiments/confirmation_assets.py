"""Stage only required existing frames; never overwrite a mismatched asset."""

import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tarfile


class Base64Reader:
    """ASCII transport avoids native Windows SSH binary-stream translation."""

    def __init__(self, stream):
        self.stream, self.pending = stream, b""

    def read(self, size):
        while len(self.pending) < size:
            line = self.stream.readline(65539)
            if not line:
                break
            if len(line) > 65538:
                raise ValueError("Oversized base64 line")
            self.pending += base64.b64decode(line.strip(), validate=True)
        result, self.pending = self.pending[:size], self.pending[size:]
        return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required(manifest):
    return sorted({f"{v}/frame_{i:02d}.png" for r in manifest["rows"] if r["positive_rank"] <= 20
                   for v in r["top20"] for i in range(8)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("pack", "receive", "transfer"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base64", action="store_true")
    args = parser.parse_args()
    if args.action == "transfer":
        remote = PurePosixPath(args.out)
        if not remote.is_relative_to("/home/zhaopu/intern-decision/outputs") or ".." in remote.parts:
            raise ValueError("Remote staging path outside authorized output tree")
        remote_out = shlex.quote(str(remote))
        prefix = f"head -c 65536 {remote_out}/frames.tar"
        expected = subprocess.check_output(["ssh", "5090_1", f"{prefix} | sha256sum"], timeout=60).split()[0]
        probe = subprocess.Popen(["ssh", "5090_1", f"{prefix} | base64 -w 65536"], stdout=subprocess.PIPE)
        try:
            actual = subprocess.check_output(["ssh", "5090", "base64 -d | sha256sum"], stdin=probe.stdout, timeout=60).split()[0]
            probe.stdout.close()
            if probe.wait(timeout=15) or actual != expected:
                raise ValueError("Bounded base64 transport checksum probe failed")
        finally:
            if probe.poll() is None:
                probe.terminate()
                probe.wait(timeout=15)
        print(json.dumps({"transport_probe": "pass", "decoded_bytes": 65536, "sha256": actual.decode()}), flush=True)
        src = subprocess.Popen(["ssh", "5090_1", f"base64 -w 65536 {remote_out}/frames.tar"], stdout=subprocess.PIPE)
        dst = subprocess.Popen(["ssh", "5090", f"python3 {remote_out}/code/confirmation_assets.py receive --base64 --root /home/zhaopu/intern-decision --out {remote_out}"], stdin=src.stdout)
        src.stdout.close()
        try:
            dst_code = dst.wait(timeout=900)
            src_code = src.wait(timeout=20)
        finally:
            for process in (src, dst):
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
        if src_code or dst_code:
            raise SystemExit(f"Frame transfer failed: source={src_code}, destination={dst_code}")
        return
    args.out = Path(args.out)
    manifest = json.loads((args.out / "confirmation_manifest.json").read_text())
    names = required(manifest)
    frames = (args.root / "frames/id_pilot_frames_8").resolve()
    if args.action == "pack":
        files = []
        for name in names:
            path = frames / name
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(frames):
                raise ValueError("Missing/unsafe source frame")
            files.append(path)
        total = sum(p.stat().st_size for p in files)
        if total > 1024 ** 3:
            raise ValueError("Frame staging exceeds 1GiB bound")
        receipt = {"files": len(names), "bytes": total, "frame_sha256": {n: sha(p) for n, p in zip(names, files)}}
        with (args.out / "frame_sources.json").open("x") as handle:
            json.dump(receipt, handle, indent=2)
        with tarfile.open(args.out / "frames.tar", "x") as archive:
            for name, path in zip(names, files):
                archive.add(path, arcname=name, recursive=False)
        print(json.dumps({"result": "pass", "files": len(names), "bytes": total}))
    else:
        receipt = json.loads((args.out / "frame_sources.json").read_text())
        if set(receipt["frame_sha256"]) != set(names) or receipt["bytes"] > 1024 ** 3:
            raise ValueError("Frame receipt scope mismatch")
        seen, copied, reused, total = set(), 0, 0, 0
        stream = Base64Reader(sys.stdin.buffer) if args.base64 else sys.stdin.buffer
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for member in archive:
                if member.name not in receipt["frame_sha256"] or member.name in seen or not member.isfile():
                    raise ValueError("Archive contains extra/duplicate/nonregular member")
                path = frames / member.name
                if not path.resolve().is_relative_to(frames) or path.is_symlink():
                    raise ValueError("Frame target escapes authorized root")
                total += member.size
                if total > 1024 ** 3 or member.size > 16 * 1024 ** 2:
                    raise ValueError("Frame stream size bound exceeded")
                content = archive.extractfile(member).read()
                if len(content) != member.size or hashlib.sha256(content).hexdigest() != receipt["frame_sha256"][member.name]:
                    raise ValueError("Transferred frame checksum mismatch")
                if path.exists():
                    if sha(path) != receipt["frame_sha256"][member.name]:
                        raise ValueError("Refuse overwriting existing mismatched frame")
                    reused += 1
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as handle:
                        handle.write(content)
                    copied += 1
                seen.add(member.name)
        if seen != set(names) or total != receipt["bytes"]:
            raise ValueError("Incomplete frame transfer")
        result = {"result": "pass", "files": len(seen), "copied": copied, "reused": reused, "bytes": total}
        with (args.out / "frame_transfer.json").open("x") as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
