"""1.1.28.1 test wave: the two comparison-benchmark harnesses name the same rows.

bench_compare_mps.cpp and bench_validate.cpp on the Lindblad side, and
aer_bench.py on the Aer side, each register their workloads under string keys,
and bench_report.py pairs rows by key. A key present on one side only renders
as a placeholder in the timing tables and as a hard FAIL in the parity gate,
so a mismatch surfaces late and in a generated page. The 1.1.28.0 release added
four MPS timing rows and the first MPS parity member and checked the two sides
by hand; this makes that check permanent.

aer_bench.py imports Qiskit at module level. The tests stub those modules so
the file can be imported and its work lists walked on a machine without
Qiskit; nothing is simulated.

Repo root resolution: LINDBLAD_REPO_ROOT env var (set by ctest), falling back
to the path of this file (tests/tools/ -> repo root two levels up).
"""

import importlib.util
import os
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(os.environ.get("LINDBLAD_REPO_ROOT",
                                Path(__file__).resolve().parents[2]))
AER_BENCH = REPO_ROOT / "benchmarks" / "compare" / "aer_bench.py"
BENCH_MPS = REPO_ROOT / "benchmarks" / "bench_compare_mps.cpp"
BENCH_VALIDATE = REPO_ROOT / "benchmarks" / "bench_validate.cpp"
CORPUS = REPO_ROOT / "benchmarks" / "compare" / "circuits"

QISKIT_MODULES = ("qiskit", "qiskit.quantum_info", "qiskit.transpiler",
                  "qiskit_aer", "qiskit_aer.noise")

# BENCHMARK_CAPTURE(BM_CmpMPS, key, "file.qasm", chi)
MPS_ROW_RE = re.compile(
    r'BENCHMARK_CAPTURE\(BM_CmpMPS,\s*(mps__\w+),\s*"([\w.]+)",\s*(\d+)\)')
VAL_KEY_RE = re.compile(r"\b(val__\w+)\b")
CPP_INT_RE = r"constexpr int %s = ([^;]+);"


def load_aer_bench():
    """Import aer_bench.py with Qiskit stubbed out."""
    stubs = {name: mock.MagicMock() for name in QISKIT_MODULES}
    with mock.patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location("aer_bench", AER_BENCH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def aer_mps_timing_keys(mod):
    """The keys domain_mps registers, in order, with the simulator stubbed."""
    keys = []
    with mock.patch.object(mod, "timed", lambda fn: {}), \
         mock.patch.object(mod, "record",
                           lambda results, key, timing, **c: keys.append(key)), \
         mock.patch.object(mod, "AerSimulator", mock.MagicMock()), \
         mock.patch.object(mod, "transpile", mock.MagicMock()), \
         mock.patch.object(mod, "load_qasm", mock.MagicMock()):
        mod.domain_mps({})
    return keys


def aer_validation_keys(mod):
    """The keys run_validation writes, counts then expectation."""
    with mock.patch.object(mod, "AerSimulator", mock.MagicMock()), \
         mock.patch.object(mod, "transpile", mock.MagicMock()), \
         mock.patch.object(mod, "load_qasm", mock.MagicMock()), \
         mock.patch.object(mod, "load_observable", mock.MagicMock()), \
         mock.patch.object(mod, "make_noise_model", mock.MagicMock()), \
         mock.patch.object(mod, "Statevector", mock.MagicMock()):
        out = mod.run_validation()
    return sorted(out["counts"]) + sorted(out["expectation"])


class AerSide(unittest.TestCase):
    """Keys as aer_bench.py itself produces them, with nothing run."""

    @classmethod
    def setUpClass(cls):
        cls.mod = load_aer_bench()

    def test_brickwork_sweep_names_the_four_caps(self):
        keys = aer_mps_timing_keys(self.mod)
        for chi in self.mod.MPS_BRICKWORK_CHI_SWEEP:
            self.assertIn("mps__brickwork__n24__chi%d" % chi, keys)
        self.assertEqual(len(keys), len(set(keys)), "duplicate MPS key")

    def test_validation_bond_is_derived_from_the_width(self):
        # 2^(n/2): the widest cut of an n-qubit chain, so the cap cannot bind.
        n = self.mod.MPS_VALIDATION_QUBITS
        self.assertEqual(self.mod.MPS_VALIDATION_BOND, 2 ** (n // 2))


class LindbladSide(unittest.TestCase):
    """Keys as the C++ harnesses register them, read from the source."""

    @classmethod
    def setUpClass(cls):
        cls.mps_src = BENCH_MPS.read_text()
        cls.val_src = BENCH_VALIDATE.read_text()
        cls.rows = MPS_ROW_RE.findall(cls.mps_src)

    def test_every_mps_row_names_its_own_file_and_cap(self):
        # mps__<family>__n<N>__chi<C> must run <family>_n<N>.qasm at cap C.
        # The label is what the page shows; the arguments are what ran. A
        # row whose label and arguments disagree publishes one measurement
        # under another's name, which is the shape of the defect the sweep
        # itself had.
        self.assertTrue(self.rows, "no BM_CmpMPS rows found")
        for key, fname, chi in self.rows:
            m = re.match(r"mps__(\w+?)__n(\d+)__chi(\d+)$", key)
            self.assertIsNotNone(m, key)
            family, n, cap = m.group(1), m.group(2), m.group(3)
            self.assertEqual(fname, "%s_n%s.qasm" % (family, n), key)
            self.assertEqual(chi, cap, key)
            self.assertTrue((CORPUS / fname).exists(), "%s: %s missing" % (key, fname))

    def test_validation_bond_is_derived_from_the_width(self):
        n = re.search(CPP_INT_RE % "kMpsValidationQubits", self.val_src)
        bond = re.search(CPP_INT_RE % "kMpsValidationBond", self.val_src)
        self.assertIsNotNone(n)
        self.assertIsNotNone(bond)
        self.assertEqual(n.group(1).strip(), "8")
        self.assertEqual(bond.group(1).strip(), "1 << (kMpsValidationQubits / 2)")


class BothSides(unittest.TestCase):
    """The pairing bench_report.py performs, checked before any run."""

    @classmethod
    def setUpClass(cls):
        cls.mod = load_aer_bench()
        cls.mps_rows = MPS_ROW_RE.findall(BENCH_MPS.read_text())
        cls.val_src = BENCH_VALIDATE.read_text()

    def test_mps_timing_keys_agree(self):
        lindblad = sorted(key for key, _, _ in self.mps_rows)
        aer = sorted(aer_mps_timing_keys(self.mod))
        self.assertEqual(lindblad, aer)

    def test_validation_keys_agree(self):
        lindblad = sorted(set(VAL_KEY_RE.findall(self.val_src)))
        aer = sorted(set(aer_validation_keys(self.mod)))
        self.assertEqual(lindblad, aer)
        self.assertIn("val__mps__qv__n8", lindblad)

    def test_mps_validation_constants_agree(self):
        mod = self.mod
        n = re.search(CPP_INT_RE % "kMpsValidationQubits", self.val_src)
        self.assertEqual(int(n.group(1)), mod.MPS_VALIDATION_QUBITS)
        self.assertEqual(1 << (mod.MPS_VALIDATION_QUBITS // 2), mod.MPS_VALIDATION_BOND)


if __name__ == "__main__":
    unittest.main()
