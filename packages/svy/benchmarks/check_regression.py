"""Compare a benchmark run against a recorded baseline and flag regressions.

Deliberately a LOCAL tool, not a CI gate. A committed baseline holds absolute
milliseconds, which only means anything on the machine that recorded it —
hosted CI runners are shared vCPUs whose run-to-run variance would swamp any
regression worth catching, so a CI comparison would mostly teach people to
ignore it. Run locally, on one machine, before a release or after touching a
hot path.

The payoff of absolute numbers is that the baseline doubles as a record across
releases: it answers "did we get faster since 0.22.1?", which is the question
the published performance notes exist to answer. Recording OVERWRITES the
default baseline and re-stamps the version, so archive the old one first --
``baselines/kernel-<version>.json`` -- and reach it later with ``--baseline``:

    uv run python benchmarks/check_regression.py \
        --baseline benchmarks/baselines/kernel-0.23.0.json

Usage:
    # record (do this deliberately, on a quiet machine, from a release build)
    make release-svy-rs && make bench-record

    # check the working tree against the recorded baseline
    make bench-check

    # compare two saved BENCH outputs instead of running anything
    uv run python benchmarks/check_regression.py --before a.txt --after b.txt

Exit codes: 0 clean, 1 regression, 2 the run looks like a debug build.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys

from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_BASELINE = HERE / "baselines" / "kernel.json"

#: A real regression hits one or two cases. A debug build hits EVERYTHING, by
#: roughly the same large factor -- `maturin develop` defaults to the debug
#: profile, which runs 13-20x slower than the release wheel. Diagnosing that
#: explicitly beats printing twenty regressions and letting someone conclude
#: they broke the library.
DEBUG_BUILD_MEDIAN_RATIO = 3.0

#: Regressions below this are noise. Local spread is 1-5% at 250k rows and up
#: to 25% at 32k, so a tight threshold produces false alarms at small sizes.
DEFAULT_THRESHOLD = 1.25


def run_bench(rows: list[int], reps: int | None) -> str:
    """Run the harness and return its stdout."""
    # bench_kernel's --rows is nargs="+", so every size goes after one flag.
    cmd = [sys.executable, str(HERE / "bench_kernel.py"), "--rows", *[str(n) for n in rows]]
    if reps:
        cmd += ["--reps", str(reps)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"benchmark harness failed with code {proc.returncode}")
    # SKIP lines go to stderr; surface them so a silently-missing case is visible.
    for line in proc.stderr.splitlines():
        if line.startswith("SKIP"):
            print(f"  skipped: {line.split(chr(9))[1]}")
    return proc.stdout


def parse_bench(text: str) -> dict[str, dict[str, float]]:
    """Parse ``BENCH<TAB>label<TAB>rows<TAB>ms`` lines into {label: {rows: ms}}."""
    out: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 4 and parts[0] == "BENCH":
            out.setdefault(parts[1], {})[parts[2]] = float(parts[3])
    return out


def machine_id() -> str:
    return f"{platform.system()} {platform.machine()} / py{platform.python_version()}"


def record(path: Path, results: dict, note: str) -> None:
    try:
        import svy

        version = getattr(svy, "__version__", "unknown")
    except Exception:  # pragma: no cover - recording should not hard-fail here
        version = "unknown"

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "svy_version": version,
        "machine": machine_id(),
        "note": note,
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    n = sum(len(v) for v in results.values())
    print(f"Recorded {n} measurements for svy {version} -> {path}")
    print(f"  machine: {payload['machine']}")


def compare(before: dict, after: dict, threshold: float) -> int:
    """Print a comparison table. Returns an exit code."""
    common = [
        (label, rows)
        for label in sorted(after)
        for rows in sorted(after[label], key=int)
        if label in before and rows in before[label]
    ]
    if not common:
        print("No overlapping benchmarks to compare.")
        return 1

    ratios = [after[lbl][r] / before[lbl][r] for lbl, r in common if before[lbl][r] > 0]
    median_ratio = statistics.median(ratios)

    if median_ratio > DEBUG_BUILD_MEDIAN_RATIO:
        print()
        print(f"!! Every benchmark is ~{median_ratio:.0f}x slower than the baseline.")
        print("   That is the signature of a DEBUG build, not a regression:")
        print("   `maturin develop` builds the debug profile by default.")
        print()
        print("   Rebuild optimized, then re-run:")
        print("     make release-svy-rs && make bench-check")
        print()
        return 2

    width = max(len(lbl) for lbl, _ in common) + 2
    print(f"\n{'benchmark':<{width}}{'rows':>10}{'baseline':>12}{'now':>10}{'change':>10}")
    print("-" * (width + 42))

    regressions: list[tuple[str, str, float]] = []
    improvements = 0
    for label, rows in common:
        b, a = before[label][rows], after[label][rows]
        ratio = a / b
        pct = (ratio - 1.0) * 100.0
        flag = ""
        if ratio > threshold:
            flag = "  REGRESSION"
            regressions.append((label, rows, ratio))
        elif ratio < 1 / threshold:
            flag = "  faster"
            improvements += 1
        print(f"{label:<{width}}{int(rows):>10,}{b:>10.2f}ms{a:>8.2f}ms{pct:>+9.1f}%{flag}")

    new = [(lbl, r) for lbl in after for r in after[lbl] if (lbl, r) not in set(common)]
    missing = [(lbl, r) for lbl in before for r in before[lbl] if (lbl, r) not in set(common)]
    if new:
        print(f"\nNew since baseline (not compared): {', '.join(sorted({n[0] for n in new}))}")
    if missing:
        print(f"Missing from this run: {', '.join(sorted({m[0] for m in missing}))}")

    print()
    if regressions:
        print(f"FAIL: {len(regressions)} regression(s) beyond {threshold:.2f}x")
        for label, rows, ratio in regressions:
            print(f"  {label} @ {int(rows):,} rows: {ratio:.2f}x")
        print("\nIf this is expected, re-record with --record.")
        return 1

    msg = f"OK: no regression beyond {threshold:.2f}x"
    if improvements:
        msg += f" ({improvements} case(s) faster — consider re-recording)"
    print(msg)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", action="store_true", help="write a new baseline")
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--before", type=Path, help="a saved BENCH output to use as baseline")
    ap.add_argument("--after", type=Path, help="a saved BENCH output to check")
    ap.add_argument("--rows", type=int, action="append", help="row counts (repeatable)")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--note", default="release build", help="recorded with the baseline")
    args = ap.parse_args()

    # 250k is the sweet spot: spread is 1-5% there versus up to 25% at 32k,
    # where fixed overheads dominate, while the whole suite still runs in a
    # few seconds. 1M is included because it is only ~7s and exercises the
    # scaling behaviour the small size cannot.
    rows = args.rows or [250_000, 1_000_000]

    if args.before and args.after:
        before = parse_bench(args.before.read_text())
        after = parse_bench(args.after.read_text())
        return compare(before, after, args.threshold)

    if args.record:
        print(f"Recording baseline at {', '.join(f'{n:,}' for n in rows)} rows...")
        results = parse_bench(run_bench(rows, args.reps))
        record(args.baseline, results, args.note)
        return 0

    if not args.baseline.exists():
        print(f"No baseline at {args.baseline}. Create one with --record.")
        return 1

    payload = json.loads(args.baseline.read_text())
    print(f"Baseline: svy {payload.get('svy_version')} on {payload.get('machine')}")
    if payload.get("machine") != machine_id():
        print(f"  WARNING: this machine is {machine_id()}.")
        print("  Absolute timings are not comparable across machines; treat")
        print("  any difference below as unreliable.")
    print(f"Running benchmarks at {', '.join(f'{n:,}' for n in rows)} rows...")
    after = parse_bench(run_bench(rows, args.reps))
    return compare(payload["results"], after, args.threshold)


if __name__ == "__main__":
    raise SystemExit(main())
