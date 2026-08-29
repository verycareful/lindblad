#!/usr/bin/env python3
"""Print one version's section from CHANGELOG.md.

Used by the release workflow to fill a GitHub Release body with the notes that
were already written for that version, so the release page and the changelog
cannot disagree. Nobody retypes anything, and there is one place to fix a typo.

Usage:
    python tools/changelog_section.py 1.1.23.0
    python tools/changelog_section.py 1.1.23.0 --changelog path/to/CHANGELOG.md

Exits non-zero when the version has no section, which is deliberate: a release
whose notes were never written should fail loudly at tag time rather than
publish an empty page.
"""

import argparse
import re
import sys
from pathlib import Path

# A section runs from its own header to the next one, or to end of file.
# Historical headers carry several label forms, so the label is captured whole
# and split afterwards rather than matched by a single pattern:
#   ## [1.1.23.0] - 2026-08-29                 the current form
#   ## [R.1.22.2/1.1.22.2] - 2026-08-29        both schemes, while they coexisted
#   ## [R.1.22.0] - 2026-08-26                 the older scheme alone
HEADER = re.compile(r"^## \[([^\]]+)\]")


def labels_of(header_label):
    """Every version string a header claims to be.

    The dual form names one release twice, so a tag matches if it equals either
    half. Whitespace is stripped because a header is written by hand.
    """
    return [part.strip() for part in header_label.split("/")]


def matches(header_label, version):
    """Whether this header is the section for `version`.

    Also accepts the R-prefixed spelling of the same number, so a tag of
    1.22.0 finds a section headed R.1.22.0 and the workflow does not need to
    know which scheme a given release was cut under.
    """
    wanted = {version, "R." + version}
    if version.startswith("R."):
        wanted.add(version[2:])
    return any(label in wanted for label in labels_of(header_label))


def extract(text, version):
    """Return the body under `version`'s header, or None if there is no such header."""
    lines = text.splitlines()
    start = None

    for i, line in enumerate(lines):
        m = HEADER.match(line)
        if m is None:
            continue
        if start is None:
            if matches(m.group(1), version):
                start = i + 1  # body begins after the header line
            continue
        # A second header closes the section that was open.
        return "\n".join(lines[start:i]).strip("\n")

    if start is None:
        return None
    return "\n".join(lines[start:]).strip("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version", help="version label, e.g. 1.1.23.0")
    ap.add_argument("--changelog", default="CHANGELOG.md",
                    help="path to the changelog (default: CHANGELOG.md)")
    args = ap.parse_args()

    path = Path(args.changelog)
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 2

    body = extract(path.read_text(encoding="utf-8"), args.version)
    if body is None:
        print(f"error: no '## [{args.version}]' section in {path}", file=sys.stderr)
        print("       the release notes have not been written yet", file=sys.stderr)
        return 1
    if not body.strip():
        print(f"error: the section for {args.version} is empty", file=sys.stderr)
        return 1

    # No trailing newline juggling: the caller redirects this into a file that
    # becomes the release body verbatim.
    print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
