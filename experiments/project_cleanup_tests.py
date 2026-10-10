"""CPU-only cleanup guard tests; no deletion or writes."""

from pathlib import Path
import unittest
from unittest.mock import patch

import project_cleanup_20261003 as cleanup


class CleanupTests(unittest.TestCase):
    def test_allowlist_has_only_exact_approved_paths(self):
        self.assertEqual(len(cleanup.ALLOW), len(set(cleanup.ALLOW)))
        self.assertEqual([p.name for p in cleanup.HF],
                         [f"hf-{i}" for i in range(200, 800, 100)])
        for path in cleanup.ALLOW:
            self.assertTrue(path.is_relative_to(cleanup.HOME))
            self.assertNotEqual(path, cleanup.HOME)
            self.assertNotIn(path, cleanup.KEEP_TREES)
        self.assertNotIn(cleanup.ROOT / "tmp", cleanup.ALLOW)

    def test_plan_hash_is_order_independent(self):
        self.assertEqual(cleanup.canonical_hash({"a": 1, "b": 2}),
                         cleanup.canonical_hash({"b": 2, "a": 1}))
        self.assertNotEqual(cleanup.canonical_hash({"a": 1}),
                            cleanup.canonical_hash({"a": 2}))

    def test_home_and_outside_paths_refused(self):
        home = Path.cwd()
        with patch.object(cleanup, "HOME", home):
            for path in [home, Path("outside"), home.parent]:
                with self.assertRaises(ValueError):
                    cleanup.literal(path)

    def test_symlink_component_refused(self):
        home = Path.cwd()
        target = home / "docs"
        with patch.object(cleanup, "HOME", home), patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaisesRegex(ValueError, "Symlink component"):
                cleanup.literal(target)

    def test_reader_and_permission_denial_both_stop_cleanup(self):
        for record in [{"hits": [{"pid": 42}], "denied_same_uid_pids": []},
                       {"hits": [], "denied_same_uid_pids": [42]}]:
            with patch.object(cleanup, "active_references", return_value=record):
                with self.assertRaises(ValueError):
                    cleanup.check_readers(cleanup.HF)

    def test_queue_pid_reuse_refused(self):
        with patch.object(cleanup, "process_identity", return_value={"pid": 1898359, "cmdline": "different job"}):
            with self.assertRaisesRegex(ValueError, "identity"):
                cleanup.queue_identity()

    def test_missing_queue_does_not_request_wake(self):
        with patch.object(cleanup, "process_identity", return_value=None):
            self.assertIsNone(cleanup.queue_identity())

    def test_session_helper_exception_is_narrow(self):
        for command in ["/usr/lib/systemd/systemd --user", "(sd-pam)", "sshd: zhaopu@notty"]:
            self.assertTrue(cleanup.session_helper(command))
        for command in ["python train.py", "pip install torch", "sshd: anotheruser@notty", "/usr/lib/systemd/systemd --system"]:
            self.assertFalse(cleanup.session_helper(command))


if __name__ == "__main__":
    unittest.main()
