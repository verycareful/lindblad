#!/usr/bin/env python3
"""Merge Lindblad and Qiskit/Aer benchmark runs into docs/Benchmarks.md.

Inputs
  --lindblad FILE...   Google Benchmark JSON files (--benchmark_format=json,
                       ideally --benchmark_repetitions=5; the per-benchmark
                       median aggregate is used, falling back to the median of
                       raw iterations)
  --aer FILE           aer_bench.py timing output
  --validate-lindblad FILE, --validate-aer FILE
                       result-parity files (bench_validate /
                       aer_bench.py --validate). REQUIRED: this script refuses
                       to write a report without the correctness gate, because
                       a speed number on a wrong answer is not a result.
  --expect-version R.X.X.X
                       optional stale-binary guard: hard-fails when the
                       version stamped in the Lindblad validation JSON differs
                       (see CLAUDE.md "Build and Test")
  --note TEXT          free-form environment note (compiler, flags, machine)
  --out FILE           default docs/Benchmarks.md

Pairing: benchmark keys ("sv__scaling__n10") are shared verbatim between the
BENCHMARK_CAPTURE labels (C++) and aer_bench.py. speedup = aer_ms / lindblad_ms,
so values above 1 mean Lindblad is faster.

Parity gate: sampled counts are compared by total-variation distance with a
sampling-noise-aware threshold. For two independent N-shot samples of the same
K-outcome distribution the expected TVD scales like sqrt(K / (pi * N)); checks
pass within 1.5x expected + 0.01, warn within 2.5x expected + 0.02, and fail
beyond that (a qubit-ordering or semantics bug produces TVD near 1.0). Exact
expectation values must agree to 1e-6. Any FAIL exits nonzero and stamps a
warning banner into the report.

Exit codes: 0 ok, 1 usage/input error, 2 parity FAIL.
"""

import argparse
import json
import math
import statistics
import sys
from datetime import date

DOMAIN_TITLES = [
    ("sv", "Statevector"),
    ("dm", "Density Matrix with Noise"),
    ("mps", "Matrix Product State"),
    ("clifford", "Clifford / Stabilizer"),
    ("trans", "Transpiler"),
    ("est", "Estimator"),
]

DOMAIN_NOTES = {
    "sv": ("Layered scaling circuit, QFT, QV-style random circuits, and a "
           "ccx-lowered Grover, 256 shots each, versus Aer method=statevector."),
    "dm": ("Twin noise model on both engines: 2-qubit depolarizing p=0.01 after "
           "every cx, amplitude damping gamma=0.005 after every h, versus Aer "
           "method=density_matrix."),
    "mps": ("Layered scaling circuit; chi is the bond-dimension cap on both "
            "engines (Aer matrix_product_state_max_bond_dimension). Lindblad "
            "uses Jacobi SVD (accuracy-first default since R.1.13)."),
    "clifford": ("H/CX/S ladder circuits versus Aer method=stabilizer; sizes "
                 "beyond statevector reach."),
    "trans": ("Full pipeline on both engines: layout, routing, optimization, "
              "and {cx, u3} basis translation on identical coupling graphs "
              "from the shared edge lists. Circuits are SMALLER than their "
              "maps (n=22 on 25/27-slot devices), exercising layout expansion "
              "/ ancilla allocation. twoq/depth are output-quality metrics: "
              "lower is better at equal legality."),
    "est": ("Heisenberg-chain observable on a measure-free ansatz. exact = "
            "Lindblad Estimator at shots=0 versus Qiskit dense-statevector "
            "expectation; shots4096 = Lindblad sampled Estimator versus Aer "
            "EstimatorV2 at precision 1/sqrt(4096)."),
}


# =============================================================================
# Input parsing
# =============================================================================

def die(msg):
    print("bench_report: error: " + msg, file=sys.stderr)
    sys.exit(1)


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        die("cannot read %s (%s)" % (path, e))


def to_ms(value, unit):
    factor = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}.get(unit)
    if factor is None:
        die("unknown Google Benchmark time_unit %r" % unit)
    return value * factor


def parse_gbench(paths):
    """{key: {"median_ms", "counters", "source"}} from Google Benchmark JSON."""
    out = {}
    contexts = []
    for path in paths:
        data = load_json(path)
        contexts.append(data.get("context", {}))
        by_run = {}
        for entry in data.get("benchmarks", []):
            run_name = entry.get("run_name", entry.get("name", ""))
            if "/" not in run_name:
                continue
            key = run_name.split("/", 1)[1]
            rec = by_run.setdefault(key, {"iters": [], "median": None, "counters": {}})
            if entry.get("run_type") == "aggregate":
                if entry.get("aggregate_name") == "median":
                    rec["median"] = to_ms(entry["real_time"], entry.get("time_unit", "ns"))
                    rec["counters"] = extract_counters(entry)
            else:
                rec["iters"].append(to_ms(entry["real_time"], entry.get("time_unit", "ns")))
                if not rec["counters"]:
                    rec["counters"] = extract_counters(entry)
        for key, rec in by_run.items():
            median = rec["median"]
            if median is None and rec["iters"]:
                median = statistics.median(rec["iters"])
            if median is None:
                continue
            if key in out:
                die("duplicate benchmark key %r across Lindblad inputs" % key)
            out[key] = {"median_ms": median, "counters": rec["counters"], "source": path}
    return out, contexts


GBENCH_NON_COUNTER_FIELDS = {
    "name", "run_name", "run_type", "family_index", "per_family_instance_index",
    "repetitions", "repetition_index", "threads", "iterations", "real_time",
    "cpu_time", "time_unit", "aggregate_name", "aggregate_unit",
}


def extract_counters(entry):
    return {k: v for k, v in entry.items()
            if k not in GBENCH_NON_COUNTER_FIELDS and isinstance(v, (int, float))}


def parse_aer(path):
    data = load_json(path)
    out = {}
    for entry in data.get("benchmarks", []):
        out[entry["name"]] = {
            "median_ms": entry["median_ms"],
            "counters": entry.get("counters", {}),
        }
    return out, data.get("context", {})


# =============================================================================
# Result-parity gate
# =============================================================================

def tvd(counts_a, counts_b):
    na, nb = sum(counts_a.values()), sum(counts_b.values())
    keys = set(counts_a) | set(counts_b)
    dist = 0.5 * sum(abs(counts_a.get(k, 0) / na - counts_b.get(k, 0) / nb)
                     for k in keys)
    return dist, len(keys)


def run_parity(val_lb, val_aer):
    """Returns (lines_for_doc, worst_status). Status order: PASS < WARN < FAIL."""
    lines = []
    worst = "PASS"

    def bump(status):
        nonlocal worst
        order = {"PASS": 0, "WARN": 1, "FAIL": 2}
        if order[status] > order[worst]:
            worst = status

    shots = val_lb.get("shots", 8192)
    keys = sorted(set(val_lb.get("counts", {})) | set(val_aer.get("counts", {})))
    for key in keys:
        a = val_lb.get("counts", {}).get(key)
        b = val_aer.get("counts", {}).get(key)
        if a is None or b is None:
            lines.append("%-32s MISSING on %s side" % (key, "Lindblad" if a is None else "Aer"))
            bump("FAIL")
            continue
        dist, support = tvd(a, b)
        expected = math.sqrt(support / (math.pi * shots))
        if dist <= 1.5 * expected + 0.01:
            status = "PASS"
        elif dist <= 2.5 * expected + 0.02:
            status = "WARN"
        else:
            status = "FAIL"
        bump(status)
        lines.append("%-32s TVD %.4f (support %4d, sampling-noise scale %.4f)  %s"
                     % (key, dist, support, expected, status))

    exp_keys = sorted(set(val_lb.get("expectation", {})) | set(val_aer.get("expectation", {})))
    for key in exp_keys:
        a = val_lb.get("expectation", {}).get(key)
        b = val_aer.get("expectation", {}).get(key)
        if a is None or b is None:
            lines.append("%-32s MISSING on %s side" % (key, "Lindblad" if a is None else "Aer"))
            bump("FAIL")
            continue
        diff = abs(a - b)
        status = "PASS" if diff < 1e-6 else "FAIL"
        bump(status)
        lines.append("%-32s |lindblad - aer| = %.3e (%.9f vs %.9f)  %s"
                     % (key, diff, a, b, status))
    return lines, worst


# =============================================================================
# Report emission
# =============================================================================

def fmt_ms(v):
    return "%10.3f" % v


def domain_rows(domain, lb, aer):
    keys = sorted(k for k in set(lb) | set(aer)
                  if k.split("__", 1)[0] == domain and not k.startswith("val__"))
    rows = []
    for key in keys:
        l, a = lb.get(key), aer.get(key)
        rows.append((key, l, a))
    return rows


def emit_domain_block(rows, with_quality):
    header = "%-34s %14s %14s %9s" % ("workload", "lindblad (ms)", "aer (ms)", "speedup")
    if with_quality:
        header += "   %19s %19s" % ("lb twoq/depth", "aer twoq/depth")
    lines = [header, "-" * len(header)]
    for key, l, a in rows:
        lb_s = fmt_ms(l["median_ms"]) if l else "         --"
        aer_s = fmt_ms(a["median_ms"]) if a else "         --"
        if l and a:
            speedup = "%8.2fx" % (a["median_ms"] / l["median_ms"])
        else:
            speedup = "      --"
        line = "%-34s %14s %14s %9s" % (key, lb_s, aer_s, speedup)
        if with_quality:
            def quality(rec):
                if not rec:
                    return "--"
                c = rec.get("counters", {})
                if "twoq_out" not in c:
                    return "--"
                return "%d / %d" % (int(c.get("twoq_out", 0)), int(c.get("depth_out", 0)))
            line += "   %19s %19s" % (quality(l), quality(a))
        lines.append(line)
    return lines


def build_report(lb, aer, parity_lines, parity_status, env_lines):
    doc = []
    doc.append("# Benchmarks")
    doc.append("")
    doc.append("Head-to-head comparison of Lindblad against Qiskit / Qiskit Aer on a shared,")
    doc.append("committed QASM2 corpus (`benchmarks/compare/circuits/`). Both engines run")
    doc.append("natively: Lindblad through the `bench_compare_*` Google Benchmark binaries,")
    doc.append("Qiskit/Aer through `benchmarks/compare/aer_bench.py` under a mirrored")
    doc.append("protocol. This page is GENERATED by `tools/bench_report.py`; do not edit the")
    doc.append("result blocks by hand.")
    doc.append("")
    doc.append("Benchmark runs are refreshed periodically, not for every release: the library")
    doc.append("version recorded under Environment identifies the run that produced this page")
    doc.append("and may lag the current release version.")
    doc.append("")
    if parity_status == "FAIL":
        doc.append("WARNING: the result-parity gate FAILED for at least one workload. The")
        doc.append("timing tables below are NOT comparable until the discrepancy is resolved.")
        doc.append("")
    doc.append("## Methodology")
    doc.append("")
    doc.append("- Identical gate content: both engines load the same gate-only QASM2 files and")
    doc.append("  append their own full-register terminal measurement where sampling is timed.")
    doc.append("- Protocol: circuit loading and backend mapping excluded from timing; warmup")
    doc.append("  then 5 repetitions; the median is reported. Sampling workloads use 256 shots,")
    doc.append("  seed 42.")
    doc.append("- Out-of-box configuration on both sides: Aer keeps gate fusion and its own")
    doc.append("  threading; Lindblad keeps its compiled optimization flags. Settings are")
    doc.append("  recorded under Environment.")
    doc.append("- speedup = aer_ms / lindblad_ms: values above 1.00x mean Lindblad is faster.")
    doc.append("- Every published table is gated on the result-parity checks below; a speed")
    doc.append("  number on a wrong answer is not a result.")
    doc.append("")
    doc.append("## Environment")
    doc.append("")
    doc.extend("- " + line for line in env_lines)
    doc.append("")
    doc.append("## Result Parity")
    doc.append("")
    doc.append("Cross-engine correctness gate (`bench_validate` vs `aer_bench.py --validate`,")
    doc.append("8192 shots): total-variation distance on sampled counts with a sampling-noise")
    doc.append("threshold, exact agreement (1e-6) on estimator expectation values. The Grover")
    doc.append("entry peaks on a non-symmetric marked state, so it doubles as an end-to-end")
    doc.append("qubit-ordering convention check between the engines.")
    doc.append("")
    doc.append("```text")
    doc.extend(parity_lines)
    doc.append("overall: " + parity_status)
    doc.append("```")
    doc.append("")
    for domain, title in DOMAIN_TITLES:
        rows = domain_rows(domain, lb, aer)
        if not rows:
            continue
        doc.append("## " + title)
        doc.append("")
        doc.append(DOMAIN_NOTES[domain])
        doc.append("")
        doc.append("```text")
        doc.extend(emit_domain_block(rows, with_quality=(domain == "trans")))
        doc.append("```")
        doc.append("")
    doc.append("## Reproducing")
    doc.append("")
    doc.append("See `docs/BuildAndTest.md` (section \"Comparison benchmarks\") for the full")
    doc.append("WSL command sequence: build the `bench_compare_*` targets, run each with")
    doc.append("`--benchmark_repetitions=5 --benchmark_format=json`, run `aer_bench.py` and")
    doc.append("both validation halves, then regenerate this page with `tools/bench_report.py`.")
    doc.append("")
    return "\n".join(doc)


def main():
    ap = argparse.ArgumentParser(description="Merge benchmark runs into docs/Benchmarks.md")
    ap.add_argument("--lindblad", nargs="+", required=True)
    ap.add_argument("--aer", required=True)
    ap.add_argument("--validate-lindblad", required=True)
    ap.add_argument("--validate-aer", required=True)
    ap.add_argument("--expect-version")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default="docs/Benchmarks.md")
    args = ap.parse_args()

    lb, gb_contexts = parse_gbench(args.lindblad)
    aer, aer_context = parse_aer(args.aer)
    val_lb = load_json(args.validate_lindblad)
    val_aer = load_json(args.validate_aer)

    lb_version = val_lb.get("version", "(unknown)")
    if args.expect_version and lb_version != args.expect_version:
        die("stale binary: validation file was produced by %s, expected %s "
            "(clean-rebuild the benchmark targets)" % (lb_version, args.expect_version))

    parity_lines, parity_status = run_parity(val_lb, val_aer)

    env_lines = ["Date: %s" % date.today().isoformat(),
                 "Lindblad version: %s" % lb_version]
    if gb_contexts:
        ctx = gb_contexts[0]
        env_lines.append("Host CPUs: %s (Google Benchmark context)" % ctx.get("num_cpus", "?"))
    if aer_context:
        env_lines.append("CPU model: %s" % (aer_context.get("cpu_model") or "(unknown)"))
        env_lines.append("Python %s, qiskit %s, qiskit-aer %s" % (
            aer_context.get("python", "?"), aer_context.get("qiskit", "?"),
            aer_context.get("qiskit_aer", "?")))
        env_lines.append("OMP_NUM_THREADS: %s" % aer_context.get("omp_num_threads", "(unset)"))
    if args.note:
        env_lines.append("Note: " + args.note)

    report = build_report(lb, aer, parity_lines, parity_status, env_lines)
    with open(args.out, "w", newline="\n") as f:
        f.write(report)

    paired = sum(1 for k in lb if k in aer)
    print("bench_report: %d Lindblad keys, %d Aer keys, %d paired; parity %s; wrote %s"
          % (len(lb), len(aer), paired, parity_status, args.out))
    if parity_status == "FAIL":
        print("bench_report: PARITY FAILURE: report stamped with a warning banner",
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
