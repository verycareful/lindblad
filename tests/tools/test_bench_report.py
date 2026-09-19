# Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
# SPDX-License-Identifier: LicenseRef-Lindblad-2.3
#
# This file is part of the Lindblad Quantum Computing Framework and is
# licensed under the Lindblad Software License Agreement, Version 2.3. The
# full text is in the LICENSE file at the root of the repository. Free for
# non-commercial and academic use; commercial use requires a separate
# Commercial License Agreement with the Author.

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


# =============================================================================
# 1.1.28.1: natural ordering, the svd column, and the committed page
# =============================================================================

def gbench_fixture_with_mps():
    """The base fixture plus an MPS sweep: rows whose keys embed numbers that
    lexicographic order misplaces, one with the svd counters, one without, and
    one with a zero counter, which is a legitimate reading and not a missing
    one."""
    data = gbench_fixture()
    mps = [
        ("mps__brickwork__n24__chi16", 101.0, {"svd_ms": 23.23, "svd_calls": 276}),
        ("mps__brickwork__n24__chi8", 19.0, {"svd_ms": 8.1, "svd_calls": 276}),
        ("mps__brickwork__n24__chi64", 6200.0, {}),
        ("mps__scaling__n160__chi32", 5.0, {"svd_ms": 0.0, "svd_calls": 0}),
        ("mps__scaling__n20__chi32", 1.0, {"svd_ms": 0.1, "svd_calls": 57}),
    ]
    for key, ms, counters in mps:
        data["benchmarks"].append({
            "name": "BM_CmpMPS/%s_median" % key, "run_name": "BM_CmpMPS/%s" % key,
            "run_type": "aggregate", "aggregate_name": "median",
            "real_time": ms, "time_unit": "ms", **counters})
    return data


def aer_fixture_with_mps():
    data = aer_fixture()
    for key, ms in [("mps__brickwork__n24__chi16", 40.0), ("mps__brickwork__n24__chi8", 28.0),
                    ("mps__brickwork__n24__chi64", 330.0), ("mps__scaling__n160__chi32", 50.0),
                    ("mps__scaling__n20__chi32", 10.0)]:
        data["benchmarks"].append({"name": key, "median_ms": ms, "mean_ms": ms,
                                   "stddev_ms": 0.1, "reps": 5})
    return data


class NaturalOrdering(unittest.TestCase):
    """natural_key: embedded numbers compare as numbers."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("bench_report", SCRIPT)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def sort(self, keys):
        return sorted(keys, key=self.mod.natural_key)

    def test_chi_sweep_reads_as_a_sweep(self):
        self.assertEqual(self.sort(["chi16", "chi8", "chi64", "chi32"]),
                         ["chi8", "chi16", "chi32", "chi64"])

    def test_register_sizes_read_as_a_sweep(self):
        self.assertEqual(self.sort(["n160", "n20", "n40", "n80"]),
                         ["n20", "n40", "n80", "n160"])
        self.assertEqual(self.sort(["s10", "s12", "s8"]), ["s8", "s10", "s12"])

    def test_full_keys_order_by_every_embedded_number(self):
        keys = ["mps__scaling__n24__chi16", "mps__scaling__n24__chi8",
                "mps__scaling__n16__chi32", "mps__scaling__n24__chi64",
                "mps__brickwork__n24__chi8"]
        self.assertEqual(self.sort(keys), [
            "mps__brickwork__n24__chi8",
            "mps__scaling__n16__chi32",
            "mps__scaling__n24__chi8",
            "mps__scaling__n24__chi16",
            "mps__scaling__n24__chi64"])

    def test_keys_without_digits_sort_by_string(self):
        self.assertEqual(self.sort(["beta", "alpha", "gamma"]), ["alpha", "beta", "gamma"])

    def test_domain_rows_come_out_in_natural_order(self):
        lb = {"mps__scaling__n24__chi16": {"median_ms": 1.0, "counters": {}},
              "mps__scaling__n24__chi8": {"median_ms": 1.0, "counters": {}},
              "mps__scaling__n24__chi64": {"median_ms": 1.0, "counters": {}},
              "val__mps__qv__n8": {"median_ms": 1.0, "counters": {}},
              "sv__scaling__n10": {"median_ms": 1.0, "counters": {}}}
        rows = self.mod.domain_rows("mps", lb, {})
        self.assertEqual([k for k, _, _ in rows],
                         ["mps__scaling__n24__chi8", "mps__scaling__n24__chi16",
                          "mps__scaling__n24__chi64"])


class SvdShare(unittest.TestCase):
    """svd_share: a zero counter is a reading, a missing one is not."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("bench_report", SCRIPT)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def share(self, rec):
        return self.mod.svd_share(rec)

    def test_missing_record_reports_nothing(self):
        self.assertIsNone(self.share(None))

    def test_missing_counter_reports_nothing(self):
        self.assertIsNone(self.share({"median_ms": 10.0, "counters": {}}))
        self.assertIsNone(self.share({"median_ms": 10.0}))

    def test_zero_counter_is_zero_percent(self):
        self.assertEqual(self.share({"median_ms": 10.0, "counters": {"svd_ms": 0.0}}), 0.0)

    def test_non_positive_median_reports_nothing(self):
        self.assertIsNone(self.share({"median_ms": 0.0, "counters": {"svd_ms": 1.0}}))

    def test_share_is_the_ratio_in_percent(self):
        self.assertAlmostEqual(self.share({"median_ms": 19.0, "counters": {"svd_ms": 8.1}}),
                               100.0 * 8.1 / 19.0)

    def test_column_distinguishes_zero_from_missing(self):
        rows = [
            ("mps__a", {"median_ms": 10.0, "counters": {"svd_ms": 0.0, "svd_calls": 0}}, None),
            ("mps__b", {"median_ms": 10.0, "counters": {}}, None),
            ("mps__c", {"median_ms": 10.0, "counters": {"svd_ms": 2.5, "svd_calls": 7}}, None),
            ("mps__d", None, {"median_ms": 10.0, "counters": {}}),
        ]
        lines = self.mod.emit_domain_block(rows, with_quality=False, with_svd=True)
        self.assertIn("lb svd share", lines[0])
        self.assertIn("lb splits", lines[0])
        by_key = {l.split()[0]: l for l in lines[2:]}
        self.assertRegex(by_key["mps__a"], r"0\.0%\s+0$")
        self.assertRegex(by_key["mps__b"], r"--\s+--$")
        self.assertRegex(by_key["mps__c"], r"25\.0%\s+7$")
        self.assertRegex(by_key["mps__d"], r"--\s+--$")

    def test_column_is_absent_unless_asked_for(self):
        rows = [("sv__a", {"median_ms": 10.0, "counters": {"svd_ms": 1.0}}, None)]
        plain = self.mod.emit_domain_block(rows, with_quality=False)
        explicit = self.mod.emit_domain_block(rows, with_quality=False, with_svd=False)
        self.assertEqual(plain, explicit)
        self.assertNotIn("svd", lines_text(plain))
        # Every line is as wide as the header, so the block still reads as a
        # table without the column.
        self.assertTrue(all(len(l) == len(plain[0]) for l in plain))


def lines_text(lines):
    return "\n".join(lines)


class SvdColumnEndToEnd(unittest.TestCase):
    """The MPS block of a generated report carries the column; no other does."""

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
               "--lindblad", self.write("gb.json", gbench_fixture_with_mps()),
               "--aer", self.write("aer.json", aer_fixture_with_mps()),
               "--validate-lindblad", self.write("vlb.json", lb_val),
               "--validate-aer", self.write("vaer.json", aer_val),
               "--out", str(out), *extra]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        report = out.read_text() if out.exists() else ""
        return proc, report

    def mps_block(self, report):
        lines = report.splitlines()
        start = lines.index("## Matrix Product State")
        end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("## "))
        return lines[start:end]

    def test_mps_block_has_the_column_and_the_sweep_order(self):
        lb, aer = validation_fixtures()
        proc, report = self.run_script(lb, aer)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        block = self.mps_block(report)
        header = next(l for l in block if l.startswith("workload"))
        self.assertIn("lb svd share", header)
        rows = [l for l in block if l.startswith("mps__")]
        self.assertEqual([r.split()[0] for r in rows], [
            "mps__brickwork__n24__chi8",
            "mps__brickwork__n24__chi16",
            "mps__brickwork__n24__chi64",
            "mps__scaling__n20__chi32",
            "mps__scaling__n160__chi32"])
        by_key = {r.split()[0]: r for r in rows}
        self.assertRegex(by_key["mps__brickwork__n24__chi8"], r"42\.6%\s+276$")
        self.assertRegex(by_key["mps__brickwork__n24__chi16"], r"23\.0%\s+276$")
        self.assertRegex(by_key["mps__brickwork__n24__chi64"], r"--\s+--$")
        self.assertRegex(by_key["mps__scaling__n160__chi32"], r"0\.0%\s+0$")

    def test_other_blocks_do_not_carry_the_column(self):
        lb, aer = validation_fixtures()
        _, report = self.run_script(lb, aer)
        lines = report.splitlines()
        for title in ("## Statevector", "## Transpiler", "## Estimator", "## Clifford / Stabilizer"):
            start = lines.index(title)
            header = next(l for l in lines[start:] if l.startswith("workload"))
            self.assertNotIn("svd", header, title)


class CommittedBenchmarksPage(unittest.TestCase):
    """docs/Benchmarks.md as committed: generated by the tool, so it has to
    show what the tool now guarantees. A regeneration with a regressed tool
    would land here."""

    PAGE = REPO_ROOT / "docs" / "Benchmarks.md"
    CORPUS = REPO_ROOT / "benchmarks" / "compare" / "circuits"

    @classmethod
    def setUpClass(cls):
        cls.lines = cls.PAGE.read_text().splitlines()
        start = cls.lines.index("## Matrix Product State")
        end = next(i for i in range(start + 1, len(cls.lines)) if cls.lines[i].startswith("## "))
        cls.block = cls.lines[start:end]
        cls.rows = [l for l in cls.block if l.startswith("mps__")]

    def test_brickwork_sweep_is_published_in_cap_order_with_every_cell(self):
        keys = [r.split()[0] for r in self.rows if r.startswith("mps__brickwork__")]
        self.assertEqual(keys, ["mps__brickwork__n24__chi%d" % c for c in (8, 16, 32, 64)])
        for r in self.rows:
            if r.startswith("mps__brickwork__"):
                self.assertNotIn("--", r, "a brickwork row is missing a cell: " + r)

    def test_published_splits_are_the_corpus_cx_count(self):
        # One split per adjacent cx, and every brickwork cx is adjacent, so the
        # `lb splits` cell is the number of cx lines in the corpus file.
        text = (self.CORPUS / "brickwork_n24.qasm").read_text()
        cx = sum(1 for l in text.splitlines() if l.startswith("cx "))
        for r in self.rows:
            if r.startswith("mps__brickwork__"):
                self.assertEqual(int(r.split()[-1]), cx, r)

    def test_mps_parity_member_is_published_and_passes(self):
        line = next(l for l in self.lines if l.startswith("val__mps__qv__n8"))
        self.assertTrue(line.rstrip().endswith("PASS"), line)

    def test_rows_are_in_natural_order(self):
        spec = importlib.util.spec_from_file_location("bench_report", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        keys = [r.split()[0] for r in self.rows]
        self.assertEqual(keys, sorted(keys, key=mod.natural_key))


if __name__ == "__main__":
    unittest.main()
