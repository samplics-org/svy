"""Extract a package's release notes from its CHANGELOG, for `gh release create`.

Run from a wheels workflow after a tag push. Writes the matching CHANGELOG
section to a file and exits non-zero if it cannot, which is deliberate: a release
with no notes, or one whose tag disagrees with the committed version, is a
mistake worth stopping on rather than papering over.

Usage:
    python .github/scripts/release_notes.py --package svy --tag svy-v0.26.0 \
        --out release-notes.md
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


def fail(message: str) -> None:
    # `::error::` surfaces this in the Actions log and the job summary.
    sys.exit(f"::error::{message}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", required=True, help="directory under packages/")
    ap.add_argument("--tag", required=True, help="e.g. svy-v0.26.0")
    ap.add_argument("--out", default="release-notes.md")
    args = ap.parse_args()

    root = pathlib.Path("packages") / args.package
    prefix = f"{args.package}-v"
    if not args.tag.startswith(prefix):
        fail(f"tag {args.tag!r} does not start with {prefix!r}")
    version = args.tag[len(prefix) :]

    # The tagged commit must already carry this version. Tagging before the
    # release PR lands would publish a wheel whose version disagrees with its
    # tag — catchable here, and not on PyPI, where it cannot be undone.
    pyproject = root / "pyproject.toml"
    declared = re.search(r'^version = "([^"]+)"', pyproject.read_text(), re.M)
    if not declared:
        fail(f"no version found in {pyproject}")
    if declared.group(1) != version:
        fail(
            f"tag {args.tag} does not match {pyproject} version {declared.group(1)}. "
            "The version bump must be committed before the tag is pushed."
        )

    # Headings look like: '## [0.26.0] — 2026-08-26' (em dash).
    changelog = root / "CHANGELOG.md"
    text = changelog.read_text()
    heading = re.search(rf"^## \[{re.escape(version)}\][^\n]*$", text, re.M)
    if not heading:
        fail(
            f"no '## [{version}]' section in {changelog}. A release without notes is "
            "almost certainly a mistake — stamp the CHANGELOG before tagging."
        )

    rest = text[heading.end() :]
    following = re.search(r"^## \[", rest, re.M)
    body = (rest[: following.start()] if following else rest).strip()
    if not body:
        fail(f"the '## [{version}]' section in {changelog} is empty")

    pypi = f"https://pypi.org/project/{args.package}/{version}/"
    link = (
        f"https://github.com/samplics-org/svy/blob/main/packages/"
        f"{args.package}/CHANGELOG.md"
    )
    body += (
        f"\n\n---\n\nPublished to PyPI as [`{args.package} {version}`]({pypi}).\n"
        f"Full changelog: [`packages/{args.package}/CHANGELOG.md`]({link})\n"
    )

    pathlib.Path(args.out).write_text(body)
    print(f"{len(body.splitlines())} lines of notes for {args.package} {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
