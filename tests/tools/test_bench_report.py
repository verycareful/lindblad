"""R.1.14.1 test patch -- tools/bench_report.py (the benchmark report generator).

Runs the real script end-to-end via subprocess against synthetic fixtures
(exit codes are part of its contract) and unit-tests the parity-gate math via
direct import. No third-party dependencies; registered in ctest only when a
Python3 interpreter is found (see tests/CMakeLists.txt).

Repo root resolution: LINDBLAD_REPO_ROOT env var (set by ctest), falling back
to the path of this file (tests/tools/ -> repo root two levels up).
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(os.environ.get("LINDBLAD_REPO_ROOT",
                                Path(__file__).resolve().parents[2]))
SCRIPT = REPO_ROOT / "tools" / "bench_report.py"


def gbench_fixture():
    return {
        "context": {"num_cpus": 16},
        "benchmarks": [
            # Aggregate path (the normal --benchmark_repetitions=5 shape).
            {"name": "BM_CmpSV/sv__scaling__n10_median",
             "run_name": "BM_CmpSV/sv__scaling__n10", "run_type": "aggregate",
             "aggregate_name": "median", "real_time": 0.42, "time_unit": "ms"},
            # Counters travel on the aggregate entry.
            {"name": "BM_CmpTrans/trans__qv27__linear27__o2_median",
             "run_name": "BM_CmpTrans/trans__qv27__linear27__o2",
             "run_type": "aggregate", "aggregate_name": "median",
             "real_time": 12.5, "time_unit": "ms",
             "twoq_out": 310, "depth_out": 190},
            # Iteration-only path (no aggregates): median of raw repetitions.
            {"name": "BM_CmpEst/est__heisenberg__n12__exact",
             "run_name": "BM_CmpEst/est__heisenberg__n12__exact",
             "run_type": "iteration", "real_time": 1.5e6, "time_unit": "ns"},
            {"name": "BM_CmpEst/est__heisenberg__n12__exact",
             "run_name": "BM_CmpEst/est__heisenberg__n12__exact",
             "run_type": "iteration", "real_time": 1.6e6, "time_unit": "ns"},
            {"name": "BM_CmpEst/est__heisenberg__n12__exact",
             "run_name": "BM_CmpEst/est__heisenberg__n12__exact",
             "run_type": "iteration", "real_time": 1.7e6, "time_unit": "ns"},
        ],
    }


def aer_fixture():
    return {
        "context": {"python": "3.12", "qiskit": "2.5.0", "qiskit_aer": "0.17.2",
                    "cpu_model": "fixture", "cpu_count": 16,
                    "omp_num_threads": "(unset)"},
        "benchmarks": [
            {"name": "sv__scaling__n10", "median_ms": 0.63, "mean_ms": 0.65,
             "stddev_ms": 0.02, "reps": 5},
            {"name": "trans__qv27__linear27__o2", "median_ms": 55.0,
             "mean_ms": 56.0, "stddev_ms": 1.0, "reps": 5,
             "counters": {"twoq_out": 402, "depth_out": 260}},
            {"name": "est__heisenberg__n12__exact", "median_ms": 3.1,
             "mean_ms": 3.2, "stddev_ms": 0.1, "reps": 5},
            # Deliberately unpaired on the Lindblad side.
            {"name": "clifford__ladder__n160", "median_ms": 9.0,
             "mean_ms": 9.1, "stddev_ms": 0.2, "reps": 5},
        ],
    }


def validation_fixtures():
    lb = {"engine": "lindblad", "version": "R.1.14.1", "shots": 8192, "seed": 42,
          "counts": {"val__sv__scaling__n8": {"00000000": 4096, "00000001": 4096}},
          "expectation": {"val__est__heisenberg__n8": -1.234567890123}}
    aer = {"engine": "qiskit-aer", "shots": 8192, "seed": 42,
           "counts": {"val__sv__scaling__n8": {"00000000": 4090, "00000001": 4102}},
           "expectation": {"val__est__heisenberg__n8": -1.234567890201}}
    return lb, aer


class BenchReportEndToEnd(unittest.TestCase):
    """Subprocess runs: exit codes and emitted report content."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def write(self, name, payload):
        path = self.dir / name
        path.write_text(json.dumps(payload))
        return str(path)

    def run_script(self, lb_val, aer_val, extra=()):
        out = self.dir / "report.md"
        cmd = [sys.executable, str(SCRIPT),
               "--lindblad", self.write("gb.json", gbench_fixture()),
               "--aer", self.write("aer.json", aer_fixture()),
               "--validate-lindblad", self.write("vlb.json", lb_val),
               "--validate-aer", self.write("vaer.json", aer_val),
               "--out", str(out), *extra]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        report = out.read_text() if out.exists() else ""
        return proc, report

    def test_pass_run_exits_zero_with_paired_speedups(self):
        lb, aer = validation_fixtures()
        proc, report = self.run_script(lb, aer)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("overall: PASS", report)
        self.assertIn("sv__scaling__n10", report)
        self.assertIn("1.50x", report)          # 0.63 / 0.42
        self.assertIn("310 / 190", report)      # transpiler quality counters
        self.assertNotIn("WARNING", report)

    def test_iteration_fallback_uses_median_of_repetitions(self):
        lb, aer = validation_fixtures()
        _, report = self.run_script(lb, aer)
        # 1.5/1.6/1.7 ms iterations -> median 1.600.
        line = next(l for l in report.splitlines()
                    if l.startswith("est__heisenberg__n12__exact"))
        self.assertIn("1.600", line)

    def test_unpaired_entry_renders_placeholders(self):
        lb, aer = validation_fixtures()
        _, report = self.run_script(lb, aer)
        line = next(l for l in report.splitlines()
                    if l.startswith("clifford__ladder__n160"))
        self.assertIn("--", line)

    def test_parity_failure_exits_two_and_stamps_banner(self):
        lb, aer = validation_fixtures()
        aer["expectation"]["val__est__heisenberg__n8"] = -0.9
        proc, report = self.run_script(lb, aer)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("overall: FAIL", report)
        self.assertIn("WARNING", report)

    def test_stale_version_exits_one_without_report(self):
        lb, aer = validation_fixtures()
        proc, report = self.run_script(lb, aer,
                                       extra=("--expect-version", "R.9.9.9"))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("stale binary", proc.stderr)
        self.assertEqual(report, "")

    def test_expected_version_match_passes(self):
        lb, aer = validation_fixtures()
        proc, _ = self.run_script(lb, aer,
                                  extra=("--expect-version", "R.1.14.1"))
        self.assertEqual(proc.returncode, 0, proc.stderr)


class ParityGateMath(unittest.TestCase):
    """Direct-import unit tests for tvd() and run_parity() thresholds."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("bench_report", SCRIPT)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def parity_status(self, delta):
        """run_parity() on a 2-outcome, 8192-shot pair differing by delta."""
        lb = {"shots": 8192,
              "counts": {"k": {"a": 4096, "b": 4096}}, "expectation": {}}
        aer = {"counts": {"k": {"a": 4096 + delta, "b": 4096 - delta}},
               "expectation": {}}
        _, worst = self.mod.run_parity(lb, aer)
        return worst

    def test_tvd_exact_value_and_support(self):
        dist, support = self.mod.tvd({"a": 75, "b": 25}, {"a": 50, "c": 50})
        # |0.75-0.50|/2 + |0.25-0|/2 + |0-0.50|/2 = 0.5
        self.assertAlmostEqual(dist, 0.5)
        self.assertEqual(support, 3)

    def test_threshold_bands_pass_warn_fail(self):
        # support 2, shots 8192: noise scale ~0.00882;
        # PASS <= 0.0232, WARN <= 0.0421, FAIL beyond.
        self.assertEqual(self.parity_status(0), "PASS")
        self.assertEqual(self.parity_status(123), "PASS")   # TVD 0.0150
        self.assertEqual(self.parity_status(246), "WARN")   # TVD 0.0300
        self.assertEqual(self.parity_status(800), "FAIL")   # TVD 0.0977

    def test_missing_counterpart_is_fail(self):
        lb = {"shots": 8192, "counts": {"k": {"a": 1}}, "expectation": {}}
        aer = {"counts": {}, "expectation": {}}
        _, worst = self.mod.run_parity(lb, aer)
        self.assertEqual(worst, "FAIL")


if __name__ == "__main__":
    unittest.main()
