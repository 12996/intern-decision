"""CPU-only checks of copy scope, framing and relocated-link confinement."""

import io
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import migrate_e04 as migration


class MigrationTests(unittest.TestCase):
    def setUp(self):
        root = (Path.cwd() / "migration-fixture-destination").resolve()
        base = (Path.cwd() / "migration-fixture-base").resolve()
        for name, value in [("ROOT", root), ("BASE", base), ("NEW_BASE", root / "runtime/conda-xt310")]:
            fixture = patch.object(migration, name, value)
            fixture.start()
            self.addCleanup(fixture.stop)

    def test_destination_is_confined(self):
        self.assertEqual(migration.destination("models/model.bin"), migration.ROOT / "models/model.bin")
        for value in ["/tmp/escape", "../escape", "models/../../escape", ""]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                migration.destination(value)

    def test_referenced_python_and_torch_links_are_relocated(self):
        source = migration.ROOT / ".venv-xt310/bin/python"
        self.assertEqual(migration.relocated_link(source, str(migration.BASE / "bin/python3.10")),
                         str(migration.NEW_BASE / "bin/python3.10"))
        self.assertEqual(migration.relocated_link(migration.ROOT / "torch-reuse-cp310/torch",
                                                 str(migration.BASE / "lib/python3.10/site-packages/torch")),
                         str(migration.NEW_BASE / "lib/python3.10/site-packages/torch"))

    def test_unknown_external_links_are_not_followed(self):
        with self.assertRaisesRegex(ValueError, "external symlink"):
            migration.relocated_link(migration.ROOT / "unknown", "/etc/unrelated")

    def test_headers_round_trip_and_truncation_is_rejected(self):
        stream = io.BytesIO()
        header = {"type": "file", "path": "models/x", "size": 42}
        migration.write_header(stream, header)
        stream.seek(0)
        self.assertEqual(migration.read_header(stream), header)
        for raw in [b"", b'{"type":"file"}', b"x" * 65537]:
            with self.subTest(raw=raw[:10]), self.assertRaises(ValueError):
                migration.read_header(io.BytesIO(raw))

    def test_expired_migration_does_not_start(self):
        with self.assertRaises(TimeoutError):
            migration.check_deadline(0)

    def wire(self, contents):
        raw, manifest = io.BytesIO(), hashlib.sha256()
        migration.write_header(raw, {"type": "start", "deadline_epoch": 1000,
                                     "limit_bytes": migration.MAX_BYTES})
        total = 0
        for name, data in contents:
            header = {"path": name, "mode": 0o644, "type": "file", "size": len(data)}
            migration.write_header(raw, header)
            raw.write(data)
            digest = hashlib.sha256(data).hexdigest()
            migration.write_header(raw, {"sha256": digest})
            migration.append_manifest(manifest, header, digest)
            total += len(data)
        migration.write_header(raw, {"type": "end", "entries": len(contents),
                                     "logical_file_bytes": total, "manifest_sha256": manifest.hexdigest()})
        return SimpleNamespace(buffer=io.BytesIO(gzip.compress(raw.getvalue())))

    def receive_fixture(self, root, data, resume=False):
        cfg = f"home = {migration.BASE}/bin\n".encode()
        wire = self.wire([("weights.bin", data), (".venv-xt310/pyvenv.cfg", cfg)])
        with patch.object(migration, "ROOT", root), patch.object(migration, "RESUME", resume), \
                patch.object(migration.sys, "stdin", wire), patch.object(migration.time, "time", return_value=101), \
                patch.object(migration.shutil, "disk_usage", return_value=SimpleNamespace(free=500 * 1024 ** 3)), \
                patch("builtins.print"):
            migration.receive(1000)

    def prepare_failed(self, root, contents):
        root.mkdir()
        (root / "weights.bin").write_bytes(contents)
        (root / "migration_status.json").write_text(json.dumps({
            "result": "fail", "phase": "incomplete_copy_no_training", "deadline_epoch": 1000,
            "source_alias": "5090_1", "destination_alias": "5090"}))

    def test_receive_verifies_whole_manifest_before_pass(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder) / "new-project"
            self.receive_fixture(root, b"complete weights")
            receipt = json.loads((root / "migration_status.json").read_text())
            self.assertEqual(receipt["result"], "pass")
            self.assertEqual((root / "weights.bin").read_bytes(), b"complete weights")

    def test_recovery_preserves_owned_partial_and_rechecks_existing_bytes(self):
        for existing in [b"part", b"complete weights"]:
            with self.subTest(existing=existing), tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
                root = Path(folder) / "new-project"
                self.prepare_failed(root, existing)
                self.receive_fixture(root, b"complete weights", resume=True)
                self.assertEqual((root / "weights.bin").read_bytes(), b"complete weights")
                partial = root / "weights.bin.migration-incomplete-a1"
                self.assertEqual(partial.exists(), existing == b"part")
                if partial.exists():
                    self.assertEqual(partial.read_bytes(), existing)

    def test_same_length_conflicting_existing_file_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder) / "new-project"
            self.prepare_failed(root, b"conflict weights")
            with self.assertRaisesRegex(ValueError, "never overwrite"):
                self.receive_fixture(root, b"complete weights", resume=True)
            self.assertEqual((root / "weights.bin").read_bytes(), b"conflict weights")

    def test_direct_recovery_preserves_partial_and_charges_both_prior_attempts(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder) / "new-project"
            self.prepare_failed(root, b"part")
            status = root / "migration_status.json"
            prior = json.loads(status.read_text())
            prior["previous_attempt"] = {"result": "fail"}
            status.write_text(json.dumps(prior))
            with patch.object(migration, "DIRECT_APPROVED", True), \
                    patch.object(migration, "MAX_BYTES", 50 * 1024 ** 3):
                self.receive_fixture(root, b"complete weights", resume=True)
            receipt = json.loads(status.read_text())
            self.assertTrue(receipt["direct_approved"])
            self.assertEqual(receipt["prior_attempt_bytes_charged_upper_bound"], 15 * 1024 ** 3)
            self.assertEqual(receipt["attempt_limit_bytes"], 50 * 1024 ** 3)
            self.assertEqual((root / "weights.bin.migration-incomplete-a2").read_bytes(), b"part")

    def test_failed_direct_attempt_cannot_be_retried_again(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            root = Path(folder) / "new-project"
            self.prepare_failed(root, b"part")
            status = root / "migration_status.json"
            prior = json.loads(status.read_text())
            prior.update(previous_attempt={"result": "fail"}, direct_approved=True)
            status.write_text(json.dumps(prior))
            with patch.object(migration, "DIRECT_APPROVED", True), \
                    self.assertRaisesRegex(ValueError, "Only one bounded recovery"):
                self.receive_fixture(root, b"complete weights", resume=True)
            self.assertEqual((root / "weights.bin").read_bytes(), b"part")


if __name__ == "__main__":
    unittest.main()
