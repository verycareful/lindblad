"""1.1.28.1 test wave: the theta harvest's shape histogram.

A LINDBLAD_MPS_THETA_HARVEST=ON build writes theta_harvest/shapes.txt when
the process exits, from a static destructor, so no test inside that process
can read it. This runs the C++ harvest suite (V11281ThetaHarvest) in a fresh
process with a temporary working directory and checks the histogram it leaves
behind. The expected counts are exactly the offers that suite makes; changing
an offer there changes this file.

Skips unless LINDBLAD_HARVEST_TEST_BINARY names a lindblad_tests binary, so
the python_tools ctest entry and CI are unaffected. A binary that is named
but was built without the harvest is a failure, not a skip: the variable
claims otherwise.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

BINARY = os.environ.get("LINDBLAD_HARVEST_TEST_BINARY")

# (rows, cols) -> offers, derived from tests/test_v11281_theta_harvest.cpp:
# the direct offers use odd shapes the simulator cannot form, the hand-built
# chain forms 2x26, and the three-qubit GHZ forms 2x2 then 4x2.
EXPECTED = {
    (1, 3): 1,   # EntriesAreInRowMajorOrder
    (1, 5): 1,   # NonFiniteValuesRoundTrip
    (2, 2): 1,   # SimulatorRunOffersEverySplit, first cx
    (2, 26): 1,  # LibraryBlockIsTheOneTheSvdReceives
    (3, 5): 1,   # RoundTripIsBitExact
    (3, 7): 1,   # HeaderNamesTheShapeAndTheOrder
    (4, 2): 1,   # SimulatorRunOffersEverySplit, second cx
    (5, 7): 2,   # FirstSightWins, offered twice
}


@unittest.skipUnless(BINARY, "set LINDBLAD_HARVEST_TEST_BINARY to a harvest-enabled lindblad_tests")
class ThetaHarvestHistogram(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.binary = Path(BINARY).resolve()
        if not cls.binary.exists():
            raise AssertionError("LINDBLAD_HARVEST_TEST_BINARY does not exist: %s" % cls.binary)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cwd = Path(cls.tmp.name)
        cls.proc = subprocess.run(
            [str(cls.binary), "--gtest_filter=V11281ThetaHarvest.*"],
            cwd=str(cls.cwd), capture_output=True, text=True)
        cls.histogram = cls.cwd / "theta_harvest" / "shapes.txt"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_suite_ran_and_passed(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stdout + self.proc.stderr)
        self.assertNotIn("[  SKIPPED ]", self.proc.stdout,
                         "the named binary skipped the harvest suite, so it was "
                         "built without LINDBLAD_MPS_THETA_HARVEST:\n" + self.proc.stdout)

    def test_histogram_is_written_at_exit(self):
        self.assertTrue(self.histogram.exists(),
                        "no shapes.txt under the working directory the process "
                        "exited in; listing: %s" % sorted(os.listdir(self.cwd)))

    def parsed(self):
        lines = self.histogram.read_text().split("\n")
        self.assertEqual(lines[0], "# rows cols count")
        body = [l for l in lines[1:] if l and not l.startswith("#")]
        trailer = [l for l in lines[1:] if l.startswith("#")]
        counts = {}
        for line in body:
            r, c, n = line.split()
            counts[(int(r), int(c))] = int(n)
        self.assertEqual(len(trailer), 1, lines)
        return counts, trailer[0], body

    def test_every_offer_is_counted_once_per_offer(self):
        counts, _, _ = self.parsed()
        self.assertEqual(counts, EXPECTED)

    def test_shapes_are_listed_in_ascending_order(self):
        # An ordered map, so two runs of the same circuit are diffable.
        _, _, body = self.parsed()
        keys = [tuple(int(t) for t in l.split()[:2]) for l in body]
        self.assertEqual(keys, sorted(keys))

    def test_trailer_totals_match_the_body(self):
        counts, trailer, _ = self.parsed()
        self.assertEqual(trailer, "# %d distinct shapes, %d splits"
                         % (len(counts), sum(counts.values())))


if __name__ == "__main__":
    unittest.main()
