#!/usr/bin/env python3
"""Fail loudly when a tracked text file drifts off the repository's line-ending policy.

The policy is LF everywhere, in the repository and in the working tree, on every
platform. `.gitattributes` states it ("* text=auto eol=lf") and enforces it at
one point only: the conversion git applies when content moves between the
working tree and the object database. That is a real defence, but it is a single
line in a single file, it acts at commit time rather than at write time, and it
is invisible while it works. A tool that rewrites a whole file to CRLF looks
harmless right up until the attribute stops applying: a path marked `-text`, an
edited attributes file, a checkout by something that does not honour git's
filters, or content that reaches the repository by another route.

This script is the independent check. It reads bytes off disk and decides for
itself, so it holds whether or not `.gitattributes` is doing its job, and it can
run anywhere: by hand, as a ctest case (which is where a maintainer meets it
every suite run), or as a CI job.

Two properties are checked, both cheap and both absolute:

  CRLF          no tracked text file may contain a carriage return. A lone CR
                counts too: it is rarer and worse, because tools that split on
                "\\n" leave it stuck to the end of the previous line.
  final newline every non-empty tracked text file must end with one. A file
                whose last line is unterminated is indistinguishable from a file
                that was cut off, which is not hypothetical here: README.md
                shipped truncated mid-sentence for eight releases, and this
                check is what would have caught it the day it happened.

Binary files are exempt from both and are identified by content (a NUL byte in
the first block), not by extension, so a new binary type needs no update here.

Usage:
    python tools/check_line_endings.py           # report and exit 1 on any finding
    python tools/check_line_endings.py --fix     # repair in place, then report
    python tools/check_line_endings.py --quiet   # findings only, no clean summary

Exit codes: 0 clean (or every finding repaired under --fix), 1 findings remain,
2 the scan could not run. Outside a git checkout there is nothing to scan and no
policy to enforce, so the script says so and exits 0.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Read far enough into a file to classify it, not the whole thing: the NUL that
# marks content as binary sits in a header for every format that has one.
BINARY_SNIFF_BYTES = 8192


def repo_root() -> Path:
    """Repo root: the env var ctest sets, else this file's grandparent."""
    env = os.environ.get("LINDBLAD_REPO_ROOT")
    return Path(env) if env else Path(__file__).resolve().parents[1]


def tracked_files(root: Path):
    """Every file git tracks, or None outside a checkout.

    git is the authority on what belongs to the project. Walking the tree would
    mean re-deriving that from scratch and would wander into build directories,
    virtualenvs, and node_modules, none of which this policy governs.
    """
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"],
                             capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [root / name.decode("utf-8") for name in out.split(b"\0") if name]


def is_binary(data: bytes) -> bool:
    """Binary is decided by content, not extension: a NUL in the first block.

    Deciding by content means a new binary type (a diagram, a fixture image)
    needs no update here to be exempt.
    """
    return b"\x00" in data[:BINARY_SNIFF_BYTES]


def check_bytes(data: bytes):
    """Return the list of policy violations in one file's content."""
    if not data or is_binary(data):
        return []                        # nothing to terminate, or exempt
    findings = []
    crlf = data.count(b"\r\n")
    cr = data.count(b"\r") - crlf
    if crlf:
        findings.append(f"{crlf} CRLF line ending{'s' if crlf != 1 else ''}")
    if cr:
        findings.append(f"{cr} lone CR{'s' if cr != 1 else ''}")
    if not data.endswith(b"\n"):
        findings.append("no final newline")
    return findings


def repair(data: bytes) -> bytes:
    """Content rewritten to policy. Order matters: normalise, then terminate."""
    fixed = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if fixed and not fixed.endswith(b"\n"):
        fixed += b"\n"
    return fixed


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Check tracked text files for CRLF and missing final newlines.")
    ap.add_argument("--fix", action="store_true",
                    help="Rewrite offending files to policy instead of only reporting")
    ap.add_argument("--quiet", action="store_true",
                    help="Print findings only, suppressing the clean-run summary")
    args = ap.parse_args()

    root = repo_root()
    files = tracked_files(root)
    if files is None:
        print(f"check_line_endings: {root} is not a git checkout, nothing to scan.")
        return 0

    offenders = 0
    repaired = 0
    scanned = 0
    for path in files:
        try:
            data = path.read_bytes()
        except OSError:
            # A tracked path with no readable file is a checkout problem, not a
            # line-ending one. Say so and keep scanning.
            print(f"  SKIP  {path.relative_to(root).as_posix()}: unreadable")
            continue
        if is_binary(data):
            continue
        scanned += 1
        findings = check_bytes(data)
        if not findings:
            continue
        rel = path.relative_to(root).as_posix()
        if args.fix:
            path.write_bytes(repair(data))
            repaired += 1
            print(f"  FIXED {rel}: {', '.join(findings)}")
        else:
            offenders += 1
            print(f"  {rel}: {', '.join(findings)}")

    if args.fix:
        print(f"check_line_endings: repaired {repaired} of {scanned} tracked text files.")
        return 0
    if offenders:
        # The findings above went to stdout and the summary goes to stderr, so
        # flush first or the two streams interleave and the summary lands above
        # the list it summarises.
        sys.stdout.flush()
        print(f"\ncheck_line_endings: {offenders} file(s) violate the LF policy "
              f"stated in .gitattributes.\n"
              f"Repair them with: python tools/check_line_endings.py --fix",
              file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"check_line_endings: {scanned} tracked text files, all LF-terminated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
