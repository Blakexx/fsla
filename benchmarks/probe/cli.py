"""`rola-probe`: the one way a number gets measured and the only way one gets recorded.

    python benchmarks/rola_probe.py duration consumer_forward --ab global:raw --tier landing
    python benchmarks/rola_probe.py counters consumer_forward --arm global --section RolaMemOcc
    python benchmarks/rola_probe.py sanitize consumer_forward --tool memcheck
    python benchmarks/rola_probe.py sass --json bodies.json [--diff-against before.json]
    python benchmarks/rola_probe.py report [--probe consumer_forward] [--format md]

The subcommands share one registry (:mod:`probe.registry`), one stopwatch
(:mod:`bench.pairing`), one preconditions block (:mod:`probe.discipline`) and one
record (:mod:`bench.ledger`). What the harness enforces, and why each step is
enforced rather than documented, is in `docs/measurement.md`.

`sass` and `report` take no lock and settle no clock: neither runs a kernel. That is
a property of those questions, not an exemption -- every subcommand that reaches the
device goes through :func:`probe.discipline.disciplined`, and there is no flag that
opts out of it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probe import counters, duration, report, sanitize, sass  # noqa: E402
from probe.discipline import TIERS, disciplined  # noqa: E402
from probe.registry import PROBES  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="rola-probe", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    d = sub.add_parser("duration", help="interleaved paired A/B, recorded")
    d.add_argument("probe", choices=sorted(PROBES))
    d.add_argument("--ab", required=True, metavar="A:B",
                   help="the two arm names to interleave, e.g. global:raw")
    d.add_argument("--rounds", type=int,
                   help="override the tier's round count (the tier still stamps)")
    d.add_argument("--tier", choices=TIERS, default="light")

    c = sub.add_parser("counters", help="ncu counter capture from a committed section")
    c.add_argument("probe", choices=sorted(PROBES))
    c.add_argument("--arm", required=True)
    c.add_argument("--section", default="RolaMemOcc")
    c.add_argument("--replay-mode", default="kernel", choices=counters.REPLAY_MODES,
                   help="RECORDED in every row; `application` is mandatory for a "
                        "kernel that carries state across launches")
    c.add_argument("--tier", choices=TIERS, default="light")

    s = sub.add_parser("sass", help="per-instantiation SASS body map, or the census")
    s.add_argument("--json", metavar="PATH", help="write the body-hash map")
    s.add_argument("--diff-against", metavar="PATH",
                   help="a map from THIS tool, built at the SAME path")
    s.add_argument("--census", action="store_true",
                   help="ratify's forcing census: registers and spills per key")

    n = sub.add_parser("sanitize",
                       help="compute-sanitizer over one probe; a NIGHTLY lane, "
                            "never a PR gate")
    n.add_argument("probe", choices=sorted(PROBES))
    n.add_argument("--tool", default="memcheck", choices=sanitize.TOOLS)
    n.add_argument("--tier", choices=TIERS, default="light")

    r = sub.add_parser("report", help="the ledger, as a table")
    r.add_argument("--fingerprint", help="a fingerprint prefix to filter on")
    r.add_argument("--probe", choices=sorted(PROBES))
    r.add_argument("--format", choices=("md", "csv"), default="md")

    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    cmd = "rola-probe " + " ".join(argv if argv is not None else sys.argv[1:])

    if a.command == "sass":
        return sass.run(out_json=a.json, diff_against=a.diff_against,
                        census=a.census)
    if a.command == "report":
        return report.run(fingerprint_prefix=a.fingerprint, probe=a.probe,
                          fmt=a.format)
    with disciplined(tier=a.tier, cmd=cmd) as run_ctx:
        if a.command == "duration":
            return duration.run(run_ctx=run_ctx, probe_name=a.probe, ab=a.ab,
                                rounds=a.rounds)
        if a.command == "sanitize":
            return sanitize.run(run_ctx=run_ctx, probe_name=a.probe, tool=a.tool)
        return counters.run(run_ctx=run_ctx, probe_name=a.probe, arm=a.arm,
                            section=a.section, replay_mode=a.replay_mode)


if __name__ == "__main__":
    raise SystemExit(main())
