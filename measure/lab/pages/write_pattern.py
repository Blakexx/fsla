# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The predictions and the measured section of `docs/lab/write_pattern.md`: `RUNS` states each run's bands once
(`_banded.py`'s form), the newest sample read against its own run's (`lab page write_pattern --instance LABEL`,
docs/internals/tools/lab.md)."""

from __future__ import annotations

import pathlib

from _banded import PENDING, Claim, write_banded

CELL = "the producer's 323.0 MB at the cell, three CTAs an SM"
SLICE = CELL + ": a share's bytes one contiguous run"
WHOLE = CELL + ": its regions, every sector whole"
OWN = CELL + ": its regions as the producer lays them out"

#: each run's predictions, in the order they ran
RUNS = (
    (
        PENDING,
        "2026-11-28",
        "the floor prices the producer's 323.0 MB at HBM's write rate as stream_rate measured it, each CTA one "
        "contiguous slice: 3,170 GB/s, 101.9 us; the library's stores alone at the cell (produce_skeleton, run "
        "20261128T130545Z-524606-99b4f497) took 161.4 us, ~2.0 TB/s, at the same grid and occupancy as these rows, and "
        "ncu on its traced twin read DRAM at 58% of its peak whether one CTA an SM or two (warps active 11% and 22%), "
        "16.2 MB READ back during a launch that reads nothing, and L2 write sectors 1.68x DRAM's bytes. THE HYPOTHESIS, "
        "NOT A FINDING until these rows name the price: the scales' 4-byte words, token-major over the four heads, "
        "leave every 32-byte sector of the 4.2 MB of scales written in four pieces by four CTAs at different times, so "
        "the L2 fetches each partial sector from DRAM to merge it (131K sectors x 4 pieces x 32 bytes = 16.8 MB, the "
        "reads seen) and HBM turns its bus between those scattered reads and the write stream. So THE SLICE, the same "
        "bytes one run a share, writes at the floor's rate (102-115 us: 101.9 and the launch's last partial wave of "
        "the 5.2 at 396 CTAs); WHOLE SECTORS, every run 256 bytes or more and sector-whole, writes near it (104-122 us); "
        "THE PRODUCER'S OWN reproduces the stores alone (152-170 us). The rival: the ~8 regions a CTA writes cost HBM's "
        "page locality, which would leave WHOLE SECTORS with THE PRODUCER'S OWN (above 145 us) and the slice alone at "
        "the floor; a slice above 125 us would say the form itself (256 threads, 8 in flight, three CTAs an SM) does "
        "not reach HBM's write rate and name the store form",
        (
            Claim(SLICE, lambda R: R[SLICE], 102.0, 115.0, "us"),
            Claim(WHOLE, lambda R: R[WHOLE], 104.0, 122.0, "us"),
            Claim(OWN, lambda R: R[OWN], 152.0, 170.0, "us"),
            Claim(OWN + ": over the slice", lambda R: R[OWN] / R[SLICE], 1.35, 1.65, "times"),
        ),
    ),
)


def write(root: pathlib.Path, data: pathlib.Path, instance: str) -> list[pathlib.Path]:
    return write_banded(root, data, instance, "write_pattern", RUNS)
