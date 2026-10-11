#!/usr/bin/env python3
# Copyright (c) 2026 Sricharan Suresh (github.com/verycareful)
# SPDX-License-Identifier: LicenseRef-Lindblad-2.3
#
# This file is part of the Lindblad Quantum Computing Framework and is
# licensed under the Lindblad Software License Agreement, Version 2.3. The
# full text is in the LICENSE file at the root of the repository. Free for
# non-commercial and academic use; commercial use requires a separate
# Commercial License Agreement with the Author.

"""Compare two builds of a Google Benchmark binary, interleaved, on one machine.

A before/after comparison made from two separate sessions measures the
machine's mood as much as the change: clock boost, heat, and whatever else is
running all drift between sessions. This runs the BEFORE and AFTER binaries in
alternating order (round 1 before-then-after, round 2 after-then-before, ...)
inside one session, so slow drift lands on both sides equally, and reports
each benchmark's change with the spread across rounds next to it, so a change
smaller than the noise is reported as unresolved rather than as a result.

Usage
  bench_ab.py --before OLD_BINARY --after NEW_BINARY --out DIR
              [--rounds N] [--filter REGEX] [--repetitions R]
              [--min-time SECONDS] [--threshold FRACTION] [--pin]
              [--noise-floor A_A_SUMMARY] [-- extra args]

  --before, --after   the two benchmark binaries; they may be the same file,
                      which measures the noise floor (an A/A run)
  --out DIR           where the per-round JSON, the environment record and
                      the report go; created if missing
  --rounds N          alternating rounds, default 4 (two in each order)
  --filter REGEX      passed as --benchmark_filter to both sides
  --repetitions R     --benchmark_repetitions per side per round, default 3
  --min-time SECONDS  --benchmark_min_time per repetition, default 0.5
  --threshold F       smallest relative change reported as resolved, default
                      0.02; a measured A/A run says what this machine supports
  --pin               run both sides with OMP_PLACES=threads and
                      OMP_PROC_BIND=close. Without it the operating system
                      places threads afresh in every process: which core a
                      serial run lands on, and how a parallel run splits across
                      cache domains, then vary from run to run by more than
                      most changes being measured. Pinning is a measurement
                      condition for comparing two builds of the same code; it
                      says nothing about how either build performs unpinned.
  --noise-floor FILE  the summary.json of an A/A run (the same binary on both
                      sides) made under the same conditions: the same --pin,
                      --rounds, --repetitions and --min-time, else it is
                      refused. Each row then has to clear the larger of
                      --threshold and that row's own A/A ratio range to count
                      as a change. Rows the A/A did not cover fall back to
                      --threshold and are listed.
  -- ARGS             anything after a bare -- goes to both binaries verbatim

Reading the report
  speedup = median(before) / median(after), so above 1 means AFTER is faster.
  Each round gives one ratio, before over after, taken within that round; [lo,
  hi] is the range of those ratios. Drift that moves both sides together
  cancels in a ratio, so the range is what decides: FASTER or SLOWER only when
  every round clears 1 +- the row's bound, NOISY when the median moves by more
  than the bound but the rounds disagree, SAME otherwise (no change resolved).
  The bound is --threshold, raised to the row's A/A range under --noise-floor.
  Each side's own spread across rounds is reported beside it, for reading how
  steady the machine was.

The environment is recorded before the first and after the last round: power
profile, load average, the busiest other processes, CPU temperatures, the
OpenMP environment the two sides ran with, the repository commit and whether
the tree is dirty, and a SHA-256 of each binary. A benchmark that errors on either side, or a name that
exists on one side only, makes the exit status nonzero.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

UNIT_TO_MS = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}
PIN_ENV = {"OMP_PLACES": "threads", "OMP_PROC_BIND": "close"}
OMP_KEYS = ("OMP_NUM_THREADS", "OMP_PLACES", "OMP_PROC_BIND")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_quiet(cmd):
    """Output of a read-only probe command, or None if it is unavailable."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def cpu_temperatures():
    """hwmon temperatures in degrees C, keyed by sensor name and label."""
    temps = {}
    root = Path("/sys/class/hwmon")
    if not root.is_dir():
        return temps
    for hw in sorted(root.iterdir()):
        try:
            name = (hw / "name").read_text().strip()
        except OSError:
            continue
        for inp in sorted(hw.glob("temp*_input")):
            label_file = inp.with_name(inp.name.replace("_input", "_label"))
            try:
                label = label_file.read_text().strip() if label_file.exists() else inp.name
                temps[f"{name}/{label}"] = int(inp.read_text().strip()) / 1000.0
            except (OSError, ValueError):
                continue
    return temps


def environment(repo):
    env = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "power_profile": run_quiet(["powerprofilesctl", "get"]),
        "loadavg": None,
        "busiest_processes": run_quiet(["ps", "-eo", "pid,pcpu,comm", "--sort=-pcpu", "--no-headers"]),
        "temperatures_c": cpu_temperatures(),
        "commit": run_quiet(["git", "-C", str(repo), "rev-parse", "HEAD"]),
        "dirty": None,
    }
    try:
        env["loadavg"] = Path("/proc/loadavg").read_text().split()[:3]
    except OSError:
        pass
    if env["busiest_processes"]:
        # This script's own process is not load on the machine being measured.
        own = str(os.getpid())
        others = [line.split(None, 1)[1] for line in env["busiest_processes"].splitlines()
                  if line.split(None, 1)[0] != own]
        env["busiest_processes"] = others[:8]
    status = run_quiet(["git", "-C", str(repo), "status", "--porcelain"])
    if status is not None:
        env["dirty"] = bool(status)
    return env


def run_side(binary, out_file, args, extra, child_env):
    cmd = [str(binary),
           f"--benchmark_repetitions={args.repetitions}",
           "--benchmark_report_aggregates_only=true",
           f"--benchmark_min_time={args.min_time}s",
           "--benchmark_format=json",
           f"--benchmark_out={out_file}"]
    if args.filter:
        cmd.append(f"--benchmark_filter={args.filter}")
    cmd.extend(extra)
    print("  " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, env=child_env)
    if proc.returncode != 0:
        sys.exit(f"bench_ab: {binary} exited with status {proc.returncode}")


def medians(json_file):
    """name -> median real time in ms; names that errored are returned apart."""
    with open(json_file) as f:
        data = json.load(f)
    out, errors = {}, set()
    for b in data.get("benchmarks", []):
        name = b.get("run_name", b.get("name"))
        if b.get("error_occurred"):
            errors.add(name)
            continue
        if b.get("aggregate_name") == "median":
            out[name] = b["real_time"] * UNIT_TO_MS[b.get("time_unit", "ns")]
    return out, errors


def load_noise_floor(path, args):
    """name -> the A/A ratio half-range, from an A/A summary.json made under
    the conditions this run uses. Exits when the file is not such a run."""
    with open(path) as f:
        aa = json.load(f)
    if not aa.get("same_binary"):
        sys.exit(f"bench_ab: {path} is not an A/A run (its two sides differ)")
    mismatched = [k for k, mine in (("pinned", args.pin), ("rounds", args.rounds),
                                    ("repetitions", args.repetitions),
                                    ("min_time_s", args.min_time))
                  if aa.get(k, False if k == "pinned" else None) != mine]
    if mismatched:
        sys.exit(f"bench_ab: {path} was measured under different conditions "
                 f"({', '.join(mismatched)}), so its floor does not apply here")
    return {name: max(1.0 - row["ratio_lo"], row["ratio_hi"] - 1.0, 0.0)
            for name, row in aa.get("rows", {}).items()}


def classify(before, after, threshold, floor):
    """before, after: per-round medians (ms); floor: the row's A/A half-range,
    or None. Returns the report row."""
    speedup = statistics.median(before) / statistics.median(after)
    ratios = [b / a for b, a in zip(before, after)]
    lo, hi = min(ratios), max(ratios)
    spread_b = (max(before) - min(before)) / statistics.median(before)
    spread_a = (max(after) - min(after)) / statistics.median(after)
    bound = max(threshold, floor) if floor is not None else threshold
    if lo > 1.0 + bound:
        verdict = "FASTER"
    elif hi < 1.0 - bound:
        verdict = "SLOWER"
    elif abs(speedup - 1.0) > bound:
        verdict = "NOISY"
    else:
        verdict = "SAME"
    return {
        "floor": floor,
        "bound": bound,
        "before_ms": statistics.median(before),
        "after_ms": statistics.median(after),
        "speedup": speedup,
        "ratio_lo": lo,
        "ratio_hi": hi,
        "spread_before": spread_b,
        "spread_after": spread_a,
        "verdict": verdict,
    }


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1:]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--before", required=True, type=Path)
    ap.add_argument("--after", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--filter", default="")
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--min-time", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--noise-floor", type=Path)
    args = ap.parse_args(argv)

    if args.rounds < 2:
        ap.error("--rounds must be at least 2, so each order runs once")
    for b in (args.before, args.after):
        if not (b.is_file() and os.access(b, os.X_OK)):
            ap.error(f"{b} is not an executable file")
    floors = load_noise_floor(args.noise_floor, args) if args.noise_floor else None
    args.out.mkdir(parents=True, exist_ok=True)

    child_env = dict(os.environ)
    if args.pin:
        child_env.update(PIN_ENV)

    repo = Path(__file__).resolve().parent.parent
    record = {
        "before": {"path": str(args.before), "sha256": sha256(args.before)},
        "after": {"path": str(args.after), "sha256": sha256(args.after)},
        "same_binary": None,
        "rounds": args.rounds, "repetitions": args.repetitions,
        "min_time_s": args.min_time, "filter": args.filter, "extra_args": extra,
        "threshold": args.threshold,
        "pinned": args.pin,
        "omp_env": {k: child_env.get(k, "(unset)") for k in OMP_KEYS},
        "noise_floor": str(args.noise_floor) if args.noise_floor else None,
        "environment_start": environment(repo),
    }
    record["same_binary"] = record["before"]["sha256"] == record["after"]["sha256"]
    if record["same_binary"]:
        print("bench_ab: both sides are the same binary; this is an A/A run of the noise floor")

    per_round = {"before": [], "after": []}
    errors = {"before": set(), "after": set()}
    for r in range(1, args.rounds + 1):
        order = ("before", "after") if r % 2 == 1 else ("after", "before")
        print(f"round {r}/{args.rounds}: {order[0]} then {order[1]}", flush=True)
        for side in order:
            out_file = args.out / f"round{r}-{side}.json"
            run_side(getattr(args, side), out_file, args, extra, child_env)
            m, e = medians(out_file)
            per_round[side].append(m)
            errors[side] |= e
    record["environment_end"] = environment(repo)

    names_b = set.intersection(*(set(m) for m in per_round["before"]))
    names_a = set.intersection(*(set(m) for m in per_round["after"]))
    one_sided = sorted(names_b ^ names_a)
    rows = {}
    for name in sorted(names_b & names_a):
        floor = floors.get(name) if floors is not None else None
        rows[name] = classify([m[name] for m in per_round["before"]],
                              [m[name] for m in per_round["after"]], args.threshold, floor)
    record["rows"] = rows
    record["errors"] = {k: sorted(v) for k, v in errors.items()}
    record["one_sided"] = one_sided

    with open(args.out / "summary.json", "w") as f:
        json.dump(record, f, indent=1)

    width = max([len(n) for n in rows] + [9])
    lines = [f"{'benchmark':<{width}}  {'before ms':>11}  {'after ms':>11}  {'speedup':>8}  "
             f"{'round range':>15}  {'bound':>6}  {'spread b/a':>11}  verdict"]
    for name, row in rows.items():
        lines.append(
            f"{name:<{width}}  {row['before_ms']:>11.4f}  {row['after_ms']:>11.4f}  "
            f"{row['speedup']:>7.3f}x  [{row['ratio_lo']:.3f}, {row['ratio_hi']:.3f}]  "
            f"{100 * row['bound']:>5.1f}%  "
            f"{100 * row['spread_before']:>4.1f}%/{100 * row['spread_after']:<4.1f}%  {row['verdict']}")
    counts = {v: sum(1 for r in rows.values() if r["verdict"] == v)
              for v in ("FASTER", "SLOWER", "SAME", "NOISY")}
    lines.append("")
    lines.append("verdicts: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    for side in ("before", "after"):
        if errors[side]:
            lines.append(f"ERRORED on the {side} side: " + ", ".join(sorted(errors[side])))
    if one_sided:
        lines.append("on one side only: " + ", ".join(one_sided))
    if floors is not None:
        unfloored = [n for n, r in rows.items() if r["floor"] is None]
        if unfloored:
            lines.append("not covered by the noise floor (bound = --threshold): " + ", ".join(unfloored))
    report = "\n".join(lines) + "\n"
    (args.out / "report.txt").write_text(report)
    print(report)

    if errors["before"] or errors["after"] or one_sided:
        sys.exit(1)


if __name__ == "__main__":
    main()
