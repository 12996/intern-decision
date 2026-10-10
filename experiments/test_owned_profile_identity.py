"""CPU fixtures for fail-closed worker identity; no processes are signalled."""

import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from watch_owned_profile import identity


class IdentityTests(unittest.TestCase):
    def probe(self, state="S", command=None, missing=False):
        output = Path("/scope")
        stat, cmdline, proc = MagicMock(), MagicMock(), MagicMock()
        stat.read_text.return_value = "123 (python) " + state + " " + "0 " * 18 + "456"
        if missing:
            stat.read_text.side_effect = FileNotFoundError()
        default_command = (b"python\0" + os.fsencode(str(output / "run_goal_profile.py")) +
                           b"\0--output\0" + os.fsencode(str(output)) + b"\0")
        cmdline.read_bytes.return_value = command if command is not None else default_command
        proc.__truediv__.side_effect = lambda name: stat if name == "stat" else cmdline
        with patch("watch_owned_profile.Path", return_value=proc):
            return identity(123, output)

    def test_exact_live_identity(self):
        self.assertEqual(self.probe(), "456")

    def test_zombie_and_missing_are_terminal_not_reused(self):
        self.assertIsNone(self.probe(state="Z", command=b""))
        self.assertIsNone(self.probe(missing=True))

    def test_unrelated_command_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "Refuse unrelated"):
            self.probe(command=b"python\0/other/job.py\0")


if __name__ == "__main__":
    unittest.main()
