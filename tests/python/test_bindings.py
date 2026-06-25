"""R.1.12.1 total-coverage suite, Batch 5: Python bindings smoke + convention tests.

Plan: docs (R.1.12.1 coverage plan), section "Batch 5".

Runs only when the pybind module is importable (built with -DLINDBLAD_BUILD_PYTHON=ON
and on sys.path); every test is skipped otherwise so the default C++-only build is
unaffected. Covers circuit construction, the statevector simulator, primitives, the
local backend, transpile, drawing, and two project-convention spot-checks
(LSB-first Pauli expectation, qubit-0-rightmost counts keys) plus exception
translation across the binding boundary.
"""

import unittest

lindblad = None
for _name in ("lindblad", "lindblad_python"):
    try:
        lindblad = __import__(_name)
        break
    except ImportError:
        continue

HAVE_MODULE = lindblad is not None


@unittest.skipUnless(HAVE_MODULE, "lindblad python module not built")
class TestBindings(unittest.TestCase):
    def test_circuit_construction_and_analysis(self):
        qc = lindblad.QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        self.assertEqual(qc.n_qubits, 2)
        self.assertEqual(qc.depth(), 2)
        ops = qc.count_ops()
        self.assertEqual(ops["h"], 1)
        self.assertEqual(ops["cx"], 1)

    def test_statevector_simulator_bell(self):
        qc = lindblad.QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        sim = lindblad.StatevectorSimulator()
        res = sim.run(qc, 2000, 1)
        self.assertTrue(res.success)
        total = 0
        for key, n in res.counts.items():
            self.assertIn(key, ("00", "11"))  # correlated outcomes only
            total += n
        self.assertEqual(total, 2000)

    def test_counts_key_is_qubit0_rightmost(self):
        qc = lindblad.QuantumCircuit(2, 2)
        qc.x(0)  # q0 = 1, q1 = 0
        qc.measure_all()
        res = lindblad.StatevectorSimulator().run(qc, 64, 1)
        keys = list(res.counts.keys())
        self.assertEqual(keys, ["01"])  # qubit 0 is the rightmost char

    def test_pauli_expectation_is_lsb_first(self):
        # |10> in amplitude terms is q0=0,q1=1; "ZI" = Z on qubit 0 -> +1.
        qc = lindblad.QuantumCircuit(2)
        qc.x(1)  # q1 = 1
        sv = lindblad.StatevectorSimulator().run(qc, 0, 0).final_state
        zi = lindblad.SparsePauliOp.from_list([("ZI", lindblad.Complex128(1.0, 0.0))])
        iz = lindblad.SparsePauliOp.from_list([("IZ", lindblad.Complex128(1.0, 0.0))])
        self.assertAlmostEqual(zi.expectation_value(sv), 1.0, places=6)   # Z on q0 (=0)
        self.assertAlmostEqual(iz.expectation_value(sv), -1.0, places=6)  # Z on q1 (=1)

    def test_estimator_and_sampler(self):
        qc = lindblad.QuantumCircuit(1)
        qc.h(0)
        x = lindblad.SparsePauliOp.from_list([("X", lindblad.Complex128(1.0, 0.0))])
        est = lindblad.Estimator()
        self.assertAlmostEqual(est.run_single(qc, x, []), 1.0, places=6)

        qm = lindblad.QuantumCircuit(1, 1)
        qm.h(0)
        qm.measure(0, 0)
        counts = lindblad.Sampler().run_single(qm, [])
        self.assertEqual(sum(counts.values()) > 0, True)

    def test_local_backend_and_transpile(self):
        qc = lindblad.QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        be = lindblad.LocalBackend()
        res = be.run(qc, 512, 1)
        self.assertTrue(res.success)
        self.assertEqual(sum(res.counts.values()), 512)

        unrouted = lindblad.QuantumCircuit(2)
        unrouted.h(0)
        unrouted.cx(0, 1)
        t = lindblad.transpile(unrouted)
        self.assertEqual(t.n_qubits, 2)

    def test_drawing_and_file_output(self):
        import os
        import tempfile
        qc = lindblad.QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        ascii_art = qc.draw()
        self.assertIsInstance(ascii_art, str)
        self.assertGreater(len(ascii_art), 0)

        path = os.path.join(tempfile.gettempdir(), "lindblad_py_draw.txt")
        qc.draw_to_file(path)
        self.assertTrue(os.path.exists(path))
        os.remove(path)

    def test_exception_translation(self):
        qc = lindblad.QuantumCircuit(2)
        # Identical operands on a two-qubit gate must raise across the boundary.
        with self.assertRaises(Exception):
            qc.cx(0, 0)


if __name__ == "__main__":
    unittest.main()
