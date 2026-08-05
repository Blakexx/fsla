"""WHAT can be probed: one entry per question, driving every subcommand.

The registry is the piece that makes this a harness rather than four scripts. A
probe names its fixture (a :mod:`bench.cells` cell), its arms, and -- the part the
timing side never needed -- its KERNEL NAME REGEX, because counter capture and SASS
comparison have to address the binary, not the fixture. One entry therefore serves
`duration`, `counters` and `sass` alike, and a probe cannot exist for one of them
and be undefined for another.

`version` is the probe's SEMANTIC version and it is inside the ledger fingerprint:
bumping it breaks the series on purpose, because a redefined question's numbers are
not the old question's numbers.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    name: str
    version: int
    cell: str
    #: `-k regex:` for ncu and the SASS census's filter. The kernel, not the launcher.
    kernel_regex: str
    #: `(norm, decay_temp)` pairs, keyed by the short name the CLI takes.
    arms: dict


#: The committed probes. Small and named, for the same reason
#: `bench_regression.ARMS` is: a registry that grows with every question asked
#: becomes a registry nobody can run, and the full grid is the nightly job.
PROBES = {
    "consumer_forward": Probe(
        name="consumer_forward",
        version=1,
        cell="topo-D2-w64x64-BC64",
        kernel_regex="consumer_forward_kernel",
        arms={"global": ("global", None),
              "raw": ("raw", None),
              "warm": ("global", "warm")},
    ),
    "consumer_forward_deep": Probe(
        name="consumer_forward_deep",
        version=1,
        cell="topo-D4-w8x8x8x16-BC16",
        kernel_regex="consumer_forward_kernel",
        arms={"global": ("global", None),
              "raw": ("raw", None)},
    ),
}


def get(name: str) -> Probe:
    if name not in PROBES:
        raise SystemExit(f"unknown probe {name!r}; the committed probes are "
                         f"{', '.join(sorted(PROBES))}")
    return PROBES[name]


def arm_id(probe: Probe, arm: str) -> str:
    """The arm's name in the ledger: the same `cell|norm|decay` string the latency
    baseline uses, so one arm has one name across both records."""
    norm, decay = probe.arms[arm]
    return f"{probe.cell}|{norm}|{decay or 'none'}"
