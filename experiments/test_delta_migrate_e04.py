"""CPU-only delta integrity and no-retransmit checks."""

import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import delta_migrate_e04 as d


class DeltaTests(unittest.TestCase):
    def fixture(self, root, corrupt=False):
        cfg = f"home = {Path('/fixture/base')}/bin\n".encode()
        data = [("old.bin", b"reused-payload-unique"),
                ("runtime/conda-xt310/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so", b"missing-payload"),
                (".venv-xt310/pyvenv.cfg", cfg)]
        source = root / "source"
        target = root / "target"
        boot = root / "boot"
        for path in [source, target, boot]:
            path.mkdir()
        headers = []
        for name, content in data:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            headers.append((path, {"path": name, "type": "file", "mode": 0o644, "size": len(content)}))
        (target / "old.bin").write_bytes(b"corrupt-payload-equal" if corrupt else data[0][1])
        (target / "migration_status.json").write_text(json.dumps({
            "result": "fail", "deadline_epoch": d.OLD_DEADLINE, "entries_complete": 1,
            "file_bytes_complete": 43917905972, "direct_approved": True}))
        patches = [(d, "PREFIX", 1), (d, "EXPECTED_ENTRIES", 3),
                   (d, "EXPECTED_BYTES", sum(len(content) for _, content in data)),
                   (d, "BOOT", boot), (d, "STATUS", target / "migration_delta_status.json"),
                   (d.m, "BASE", Path("/fixture/base")), (d.m, "ROOT", target),
                   (d.m, "NEW_BASE", target / "runtime/conda-xt310")]
        for module, name, value in patches:
            p = patch.object(module, name, value)
            p.start()
            self.addCleanup(p.stop)
        return headers, target, data

    def transfer(self, headers, limit=None):
        wire = io.BytesIO()
        with patch.object(d.m, "entries", return_value=iter(headers)), \
                patch.object(d.sys, "stdout", SimpleNamespace(buffer=wire)), \
                patch.object(d, "check"):
            d.send()
        raw = wire.getvalue()
        self.assertNotIn(b"reused-payload-unique", raw)
        wire.seek(0)
        with patch.object(d.sys, "stdin", SimpleNamespace(buffer=wire)), \
                patch.object(d, "check"), \
                patch.object(d.shutil, "disk_usage", return_value=SimpleNamespace(free=500 * 1024 ** 3)):
            d.receive()

    def test_reuse_is_rehashed_without_network_payload_and_delta_manifest_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            headers, target, data = self.fixture(Path(folder))
            self.transfer(headers)
            receipt = json.loads(d.STATUS.read_text())
            self.assertEqual(receipt["result"], "pass")
            self.assertEqual(receipt["new_payload_bytes"], sum(len(c) for _, c in data[1:]))
            self.assertEqual((target / data[1][0]).read_bytes(), data[1][1])

    def test_changed_reused_file_fails_hash_check(self):
        with tempfile.TemporaryDirectory() as folder:
            headers, target, data = self.fixture(Path(folder), corrupt=True)
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                self.transfer(headers)
            self.assertEqual(json.loads(d.STATUS.read_text())["result"], "fail")

    def test_new_payload_budget_is_enforced(self):
        with tempfile.TemporaryDirectory() as folder:
            headers, target, data = self.fixture(Path(folder))
            with patch.object(d, "LIMIT", 1), self.assertRaisesRegex(ValueError, "payload exceeds"):
                self.transfer(headers)


if __name__ == "__main__":
    unittest.main()
