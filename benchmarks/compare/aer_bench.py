#!/usr/bin/env python3
"""Qiskit / Qiskit Aer half of the Lindblad comparison suite (R.1.14).

Runs the SAME committed QASM2 corpus (circuits/) that the bench_compare_*
Google Benchmark binaries run, under a mirrored protocol:

  - circuit loading / backend transpilation happens in setup, outside timing
    (the C++ side likewise parses QASM outside the timed loop)
  - 1 warmup run, then REPS timed runs; median is the headline number
  - sampling workloads: shots = 256, seed = 42; each harness appends its own
    full-register terminal measurement to the gate-only corpus circuits
  - out-of-box engine configuration on BOTH sides (Aer keeps gate fusion and
    its own threading; Lindblad keeps its compiled flags); every relevant
    setting is recorded in the output context

Benchmark keys ("sv__scaling__n10", ...) match the BENCHMARK_CAPTURE labels
in benchmarks/bench_compare_*.cpp one-to-one; tools/bench_report.py pairs on
them. Twin definitions that CANNOT come from the corpus and must stay in sync
with the C++ side by hand:

  - the DM noise model (make_noise_model() here == make_dm_noise() in
    bench_compare_dm.cpp / bench_validate.cpp)
  - protocol constants (SHOTS/SEED/VALIDATION_SHOTS == compare_common.hpp)

Pauli-order convention: obs_*.txt strings are in LINDBLAD order (char q acts
on qubit q, LSB first). Qiskit labels are MSB first, so this harness REVERSES
every string. Getting this wrong flips the observable; the exact-expectation
validation entry exists to catch exactly that.

Usage:
  python3 aer_bench.py --out aer_results.json            # timed suite
  python3 aer_bench.py --validate --out aer_validation.json
  python3 aer_bench.py --domains sv,trans --out sv_trans.json
"""

import argparse
import json
import math
import os
import statistics
import sys
import time

from qiskit import qasm2, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error, depolarizing_error

# Protocol constants: mirrored in benchmarks/compare_common.hpp.
SHOTS = 256
SEED = 42
VALIDATION_SHOTS = 8192
WARMUP = 1
REPS = 5

CIRCUITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "circuits")

# Workload lists: mirrored against the BENCHMARK_CAPTURE registrations.
SV_SCALING = [10, 14, 18, 22, 26]
SV_QFT = [10, 14, 18, 22]
SV_QV = [10, 14, 18, 22]
SV_GROVER = [8, 10, 12]
DM_SCALING = [4, 6, 8, 10]
MPS_N_SWEEP = [16, 24, 32, 40]          # at chi = 32
MPS_CHI_SWEEP = [8, 16, 64]             # at n = 24 (chi = 32 comes from above)
CLIFFORD_SIZES = [20, 40, 80, 160]
# Transpiler workloads: (qasm file, circuit key, coupling map). Circuits are
# sized EXACTLY to their map, and both engines route WITHOUT basis translation
# (see bench_compare_transpiler.cpp header for the tracked Lindblad defects
# behind both constraints).
TRANS_WORK = [
    ("qv_n27.qasm",  "qv27",  "linear27"),
    ("qv_n25.qasm",  "qv25",  "grid25"),
    ("qv_n27.qasm",  "qv27",  "heavyhex27"),
    ("qft_n27.qasm", "qft27", "linear27"),
    ("qft_n25.qasm", "qft25", "grid25"),
    ("qft_n27.qasm", "qft27", "heavyhex27"),
]
TRANS_OPT_LEVELS = [2, 3]
EST_SIZES = [12, 16, 20]
EST_SHOTS = 4096


# =============================================================================
# Corpus loading
# =============================================================================

def load_qasm(name, add_measure):
    # LEGACY_CUSTOM_INSTRUCTIONS: Qiskit's strict qasm2 parser implements the
    # PAPER qelib1.inc, which lacks cp (and p, sx, rxx, ...). The documented
    # legacy set restores them as native gates, matching the gate vocabulary
    # of Lindblad's QASM2 parser, so the corpus stays byte-identical.
    qc = qasm2.load(os.path.join(CIRCUITS_DIR, name),
                    custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    if add_measure:
        # Adds a barrier + full-register measurement into a fresh creg. The
        # barrier is semantically inert here (no gates follow); bitstring keys
        # are qubit-0-rightmost, matching Lindblad's convention.
        qc.measure_all()
    return qc


def load_observable(name):
    terms = []
    with open(os.path.join(CIRCUITS_DIR, name)) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            coeff, pauli = line.split()
            # Lindblad order (LSB first) -> Qiskit label (MSB first).
            terms.append((pauli[::-1], float(coeff)))
    return SparsePauliOp.from_list(terms)


def load_coupling(name):
    edges = []
    n = None
    with open(os.path.join(CIRCUITS_DIR, name)) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if n is None:
                n = int(line)
                continue
            u, v = (int(x) for x in line.split())
            edges.append([u, v])
            edges.append([v, u])  # symmetrise, as the C++ loader does
    return CouplingMap(couplinglist=edges)


def make_noise_model():
    """Twin of make_dm_noise() in bench_compare_dm.cpp: keep in sync by hand."""
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ["cx"])
    nm.add_all_qubit_quantum_error(amplitude_damping_error(0.005), ["h"])
    return nm


# =============================================================================
# Timing protocol
# =============================================================================

def timed(fn):
    """1 warmup + REPS timed calls; median/mean/stddev in ms."""
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "stddev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "reps": REPS,
    }


def record(results, key, timing, **counters):
    entry = {"name": key}
    entry.update(timing)
    if counters:
        entry["counters"] = {k: float(v) for k, v in counters.items()}
    results.append(entry)
    print("  %-40s median %10.3f ms" % (key, timing["median_ms"]))


# =============================================================================
# Domains
# =============================================================================

def run_simulator_domain(results, sim, files_and_keys, shots=SHOTS):
    for fname, key in files_and_keys:
        qc = load_qasm(fname, add_measure=True)
        tqc = transpile(qc, sim)  # backend mapping only, outside timing
        run = lambda: sim.run(tqc, shots=shots, seed_simulator=SEED).result().get_counts()
        record(results, key, timed(run))


def domain_sv(results):
    sim = AerSimulator(method="statevector")
    work = [("scaling_n%d.qasm" % n, "sv__scaling__n%d" % n) for n in SV_SCALING]
    work += [("qft_n%d.qasm" % n, "sv__qft__n%d" % n) for n in SV_QFT]
    work += [("qv_n%d.qasm" % n, "sv__qv__n%d" % n) for n in SV_QV]
    work += [("grover_n%d.qasm" % s, "sv__grover__s%d" % s) for s in SV_GROVER]
    run_simulator_domain(results, sim, work)


def domain_dm(results):
    sim = AerSimulator(method="density_matrix", noise_model=make_noise_model())
    work = [("dmscaling_n%d.qasm" % n, "dm__scaling__n%d" % n) for n in DM_SCALING]
    run_simulator_domain(results, sim, work)


def domain_mps(results):
    def sim_for(chi):
        return AerSimulator(method="matrix_product_state",
                            matrix_product_state_max_bond_dimension=chi)
    work = [(sim_for(32), "scaling_n%d.qasm" % n, "mps__scaling__n%d__chi32" % n)
            for n in MPS_N_SWEEP]
    work += [(sim_for(chi), "scaling_n24.qasm", "mps__scaling__n24__chi%d" % chi)
             for chi in MPS_CHI_SWEEP]
    for sim, fname, key in work:
        qc = load_qasm(fname, add_measure=True)
        tqc = transpile(qc, sim)
        run = lambda: sim.run(tqc, shots=SHOTS, seed_simulator=SEED).result().get_counts()
        record(results, key, timed(run))


def domain_clifford(results):
    sim = AerSimulator(method="stabilizer")
    work = [("clifford_n%d.qasm" % n, "clifford__ladder__n%d" % n) for n in CLIFFORD_SIZES]
    run_simulator_domain(results, sim, work)


def domain_trans(results):
    for fname, ckey, map_name in TRANS_WORK:
        qc = load_qasm(fname, add_measure=False)
        cmap = load_coupling("coupling_%s.edges" % map_name)
        for opt in TRANS_OPT_LEVELS:
            out_holder = {}

            def run():
                # basis_gates=None: routing-only, mirroring the Lindblad side
                # (its transpile() currently performs no basis translation).
                out_holder["qc"] = transpile(
                    qc, coupling_map=cmap, basis_gates=None,
                    optimization_level=opt, seed_transpiler=SEED)

            timing = timed(run)
            out = out_holder["qc"]
            record(results, "trans__%s__%s__o%d" % (ckey, map_name, opt),
                   timing, twoq_out=out.num_nonlocal_gates(), depth_out=out.depth())


def domain_est(results):
    from qiskit_aer.primitives import EstimatorV2
    precision = 1.0 / math.sqrt(EST_SHOTS)
    est = EstimatorV2(options={"backend_options": {"seed_simulator": SEED}})
    for n in EST_SIZES:
        qc = load_qasm("ansatz_n%d.qasm" % n, add_measure=False)
        obs = load_observable("obs_heisenberg_n%d.txt" % n)

        # Exact reference path: dense statevector expectation, the standard
        # Qiskit answer to Lindblad's Estimator at shots = 0.
        run_exact = lambda: Statevector(qc).expectation_value(obs).real
        record(results, "est__heisenberg__n%d__exact" % n, timed(run_exact))

        # Sampled path at matched statistics (precision = 1/sqrt(shots)).
        run_sampled = lambda: float(
            est.run([(qc, obs)], precision=precision).result()[0].data.evs)
        record(results, "est__heisenberg__n%d__shots4096" % n, timed(run_sampled))


DOMAINS = {
    "sv": domain_sv,
    "dm": domain_dm,
    "mps": domain_mps,
    "clifford": domain_clifford,
    "trans": domain_trans,
    "est": domain_est,
}


# =============================================================================
# Validation mode (twin of bench_validate.cpp)
# =============================================================================

def run_validation():
    counts = {}

    def sample(sim, fname, key):
        qc = load_qasm(fname, add_measure=True)
        tqc = transpile(qc, sim)
        result = sim.run(tqc, shots=VALIDATION_SHOTS, seed_simulator=SEED).result()
        counts[key] = dict(result.get_counts())

    sv = AerSimulator(method="statevector")
    sample(sv, "scaling_n8.qasm", "val__sv__scaling__n8")
    sample(sv, "qft_n8.qasm", "val__sv__qft__n8")
    sample(sv, "qv_n8.qasm", "val__sv__qv__n8")
    sample(sv, "grover_n8.qasm", "val__sv__grover__s8")
    sample(AerSimulator(method="stabilizer"), "clifford_n8.qasm",
           "val__clifford__ladder__n8")
    sample(AerSimulator(method="density_matrix", noise_model=make_noise_model()),
           "dmscaling_n6.qasm", "val__dm__scaling__n6")

    expectation = float(Statevector(load_qasm("ansatz_n8.qasm", False))
                        .expectation_value(load_observable("obs_heisenberg_n8.txt")).real)
    return {
        "engine": "qiskit-aer",
        "shots": VALIDATION_SHOTS,
        "seed": SEED,
        "counts": counts,
        "expectation": {"val__est__heisenberg__n8": expectation},
    }


# =============================================================================
# Context capture + main
# =============================================================================

def capture_context():
    import qiskit
    import qiskit_aer
    cpu_model = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "python": sys.version.split()[0],
        "qiskit": qiskit.__version__,
        "qiskit_aer": qiskit_aer.__version__,
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "(unset)"),
        "shots": SHOTS,
        "seed": SEED,
        "warmup": WARMUP,
        "reps": REPS,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="aer_results.json")
    ap.add_argument("--domains", default="all",
                    help="comma-separated subset of %s (default: all)" % ",".join(DOMAINS))
    ap.add_argument("--validate", action="store_true",
                    help="emit the result-parity validation file instead of timings")
    args = ap.parse_args()

    if args.validate:
        payload = run_validation()
        payload["context"] = capture_context()
    else:
        selected = list(DOMAINS) if args.domains == "all" else args.domains.split(",")
        unknown = [d for d in selected if d not in DOMAINS]
        if unknown:
            ap.error("unknown domain(s): %s" % ",".join(unknown))
        results = []
        for name in selected:
            print("== domain: %s ==" % name)
            DOMAINS[name](results)
        payload = {"context": capture_context(), "benchmarks": results}

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
