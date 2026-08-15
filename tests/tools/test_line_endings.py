"""Test patch: tools/check_line_endings.py (the LF-policy scanner).

Two jobs here, and they fail for different reasons.

The unit tests pin the scanner's classification rules against synthetic content,
so a rule that quietly stops firing is caught. They matter because a checker that
passes everything looks exactly like a clean repository.

The repository scan is the failsafe itself, wired where a maintainer meets it:
every suite run. `.gitattributes` already states the LF policy and git already
enforces it at commit time, but that is one line acting at one moment, and it is
invisible while it works. This case reads the bytes on disk and decides
independently, so drift is reported whether or not that line is still applying.

Repo root resolution: LINDBLAD_REPO_ROOT env var (set by ctest), falling back
to the path of this file (tests/tools/ -> repo root two levels up).
"""

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(os.environ.get("LINDBLAD_REPO_ROOT",
                                Path(__file__).resolve().parents[2]))
SCRIPT = REPO_ROOT / "tools" / "check_line_endings.py"


def load_module():
    """Import the script by path: tools/ is not a package."""
    spec = importlib.util.spec_from_file_location("check_line_endings", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(SCRIPT.exists(), f"{SCRIPT} not present")
class CheckBytesTest(unittest.TestCase):
    """Classification rules, driven directly with bytes."""

    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    def test_lf_terminated_text_is_clean(self):
        self.assertEqual(self.mod.check_bytes(b"one\ntwo\n"), [])

    def test_crlf_is_reported_with_a_count(self):
        findings = self.mod.check_bytes(b"one\r\ntwo\r\n")
        self.assertEqual(len(findings), 1)
        self.assertIn("2 CRLF", findings[0])

    def test_lone_cr_is_reported_separately_from_crlf(self):
        # A classic-Mac ending is not a CRLF with the LF missing: tools that
        # split on "\n" leave it attached to the previous line, so it hides
        # rather than announcing itself. Counted apart for that reason.
        findings = self.mod.check_bytes(b"one\rtwo\r\nthree\n")
        self.assertEqual(len(findings), 2)
        self.assertIn("1 CRLF", findings[0])
        self.assertIn("1 lone CR", findings[1])

    def test_missing_final_newline_is_reported(self):
        self.assertEqual(self.mod.check_bytes(b"one\ntwo"), ["no final newline"])

    def test_truncated_file_reports_only_the_missing_newline(self):
        # The shape README.md was in for eight releases: content cut off
        # mid-sentence, every ending it does have correct.
        findings = self.mod.check_bytes(b"a line\nan unfinished sen")
        self.assertEqual(findings, ["no final newline"])

    def test_empty_file_is_clean(self):
        # Nothing to terminate. Requiring a newline here would demand that an
        # intentionally empty file stop being empty.
        self.assertEqual(self.mod.check_bytes(b""), [])

    def test_binary_content_is_exempt(self):
        # Exempt by content rather than by extension, so a new binary type
        # needs no change to the scanner.
        self.assertEqual(self.mod.check_bytes(b"\x89PNG\x00\r\n\x1a\n\xff"), [])

    def test_binary_detection_reads_only_the_leading_block(self):
        payload = b"text\r\n" * 4000 + b"\x00"
        self.assertGreater(len(payload), self.mod.BINARY_SNIFF_BYTES)
        self.assertFalse(self.mod.is_binary(payload))
        self.assertTrue(self.mod.is_binary(b"\x00" + payload))


@unittest.skipUnless(SCRIPT.exists(), f"{SCRIPT} not present")
class RepairTest(unittest.TestCase):
    """--fix output, which has to be idempotent to be safe to re-run."""

    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    def test_repair_normalises_and_terminates(self):
        self.assertEqual(self.mod.repair(b"one\r\ntwo\rthree"), b"one\ntwo\nthree\n")

    def test_repair_leaves_clean_content_byte_identical(self):
        clean = b"one\ntwo\n"
        self.assertEqual(self.mod.repair(clean), clean)

    def test_repair_is_idempotent(self):
        once = self.mod.repair(b"one\r\ntwo")
        self.assertEqual(self.mod.repair(once), once)

    def test_repair_leaves_an_empty_file_empty(self):
        self.assertEqual(self.mod.repair(b""), b"")

    def test_repaired_content_passes_the_check(self):
        for raw in (b"a\r\nb", b"a\rb\r\nc", b"no newline here"):
            with self.subTest(raw=raw):
                self.assertEqual(self.mod.check_bytes(self.mod.repair(raw)), [])


@unittest.skipUnless(SCRIPT.exists(), f"{SCRIPT} not present")
class RepositoryScanTest(unittest.TestCase):
    """The failsafe: the real tree, checked end to end through the real script."""

    def test_repository_is_clean(self):
        result = subprocess.run([sys.executable, str(SCRIPT)],
                                capture_output=True, text=True,
                                env={**os.environ, "LINDBLAD_REPO_ROOT": str(REPO_ROOT)})
        self.assertEqual(result.returncode, 0,
                         "tracked files violate the LF policy in .gitattributes:\n"
                         f"{result.stdout}\n{result.stderr}\n"
                         "Repair with: python tools/check_line_endings.py --fix")


if __name__ == "__main__":
    unittest.main()
