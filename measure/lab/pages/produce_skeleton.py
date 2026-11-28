# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The predictions and the measured section of `docs/lab/produce_skeleton.md`: `RUNS` states each run's bands once
(`_banded.py`'s form), the newest sample read against its own run's (`lab page produce_skeleton --instance LABEL`,
docs/internals/tools/lab.md)."""

from __future__ import annotations

import pathlib

from _banded import Claim, write_banded

CELL = "the empty pipeline at the cell"
NO_STAGES = CELL + ", hidden 0"
TWO = CELL + ", two CTAs an SM"
LO = CELL + ", the lo plane"
STORES = CELL + ", the stores alone, three CTAs an SM"

#: THE FLOOR (measure/floors/produce.work.json at measure/floors/h100.rates.json, warpline's floor): the cell's
#: outputs at HBM's write rate, without and with the lo plane
FLOOR_US = 101.875
LO_FLOOR_US = 186.55

#: each run's predictions, in the order they ran
RUNS = (
    (
        "20261128T081918Z-081208-723dc457",
        "2026-11-28",
        "warpline's floor at the cell is 101.9 us, its 323.0 MB at HBM's write rate (measure/floors/produce.work.json). "
        "THE HYPOTHESIS, NOT A FINDING until these rows name the price: at one CTA an SM a CTA's non-store time (launch, "
        "setup, its 20 ring stages' edges, the tickets, its exit) runs in series with its own writes, ~10K cycles a CTA "
        "(the CTA's ~25K warp instructions of tools/budgets/produce.json at ~2.5 a cycle) over 2,048 CTAs on 132 SMs, "
        "15.5 waves at ~1.8 GHz: ~85 us past the floor, ~187 us; at hidden 0 the 20 stages' edges go (~100 cycles each, "
        "~17 us a launch); at two CTAs an SM one CTA's serial time runs under the other's writes, back near the floor. "
        "The rival (the stub stores' instruction issue, their runtime division 29 of a trip's ~46 instructions) would "
        "leave hidden 0 and two CTAs an SM where the plain row is, its time inside the traced rows' epilogue phases; the "
        "traced twins' timelines (lab trace: each role's phases, the path) say which. The lo plane adds its 268.4 MB at "
        "the write rate",
        (
            Claim(CELL, lambda R: R[CELL], 170.0, 205.0, "us"),
            Claim(CELL + ": over its floor", lambda R: R[CELL] / FLOOR_US, 1.67, 2.01, "times"),
            Claim(NO_STAGES, lambda R: R[NO_STAGES], 152.0, 190.0, "us"),
            Claim(TWO, lambda R: R[TWO], 104.0, 128.0, "us"),
            Claim(LO, lambda R: R[LO], 255.0, 292.0, "us"),
            Claim(LO + ": over its floor", lambda R: R[LO] / LO_FLOOR_US, 1.37, 1.57, "times"),
        ),
    ),
    (
        "20261128T095100Z-086748-f002e06d",
        "2026-11-28",
        "the first run held at one CTA an SM (202.8 us, 1.99x the floor), at hidden 0 (186.2) and with the lo plane (267.1, "
        "1.43x) and MISSED at two CTAs an SM (164.0 against 104-128): its traced twins name why. A lone CTA's 25-27K "
        "cycles are five job windows of ~4,550: the ring's empty edges ~1,350 (a K-step's round trip, the loader waiting "
        "on the release, the math warps on the arrival) and the EPILOGUE's stores ~2,900, 64% of the CTA, at ~11 bytes a "
        "cycle an SM; with no ring (hidden 0) a CTA still takes ~24K cycles and the launch writes at 1,736 GB/s, 55% of "
        "HBM's write rate; at two CTAs an SM each CTA takes ~40K, twice a lone one. So the price was not a serial "
        "non-store latency (the hypothesis, refuted as the main price) but the stub stores' instruction stream on the "
        "SM: ~46 instructions a 16-byte chunk, 29 of them the runtime division's dependent chain, with 8 math warps to "
        "hide it. THE STORE COMPONENT has no division (power-of-two-shaped runs: a row tile in a thread's 128-chunk "
        "stride, a share's 64 tokens two threads each): ~6 instructions a 16-byte store, ~14.2K warp instructions a CTA "
        "against ~25K (tools/budgets/produce.json). Its stores then run at the SM's write path, so a launch lies between "
        "the floor (101.9 us, every SM's writes overlapping the others' ring edges) and each CTA's serial sum (its five "
        "ring windows' ~6.75K cycles plus its 157.7 KB at HBM's write share, 13.2 bytes a cycle an SM: ~11.9K, and ~2K "
        "of setup and tickets: ~20.7K over 15.5 waves at 1.97 GHz, 163 us); at hidden 0 the sum is ~13.9K a CTA, 110 "
        "us; at two CTAs an SM one CTA's ring runs under the other's stores; the lo plane adds its 268.4 MB at the write "
        "rate",
        (
            Claim(CELL, lambda R: R[CELL], 110.0, 165.0, "us"),
            Claim(CELL + ": over its floor", lambda R: R[CELL] / FLOOR_US, 1.08, 1.62, "times"),
            Claim(NO_STAGES, lambda R: R[NO_STAGES], 100.0, 120.0, "us"),
            Claim(TWO, lambda R: R[TWO], 100.0, 125.0, "us"),
            Claim(LO, lambda R: R[LO], 190.0, 250.0, "us"),
        ),
    ),
    (
        "20261128T121503Z-274366-44f4c87b",
        "2026-11-28",
        "the store component's run held its coverage and MISSED every band (plain 204.4 us, hidden 0 195.7, two CTAs an "
        "SM 185.5, the lo plane 290.6): its stores' instructions fell ~25K to ~14.1K a CTA (ncu) and the launch did not "
        "move, so the instruction stream was not the price. ncu on the traced launch: DRAM 49% of its peak (321.5 MB "
        "written, the declared bytes), 28.8M warp instructions (~28 us of issue), no drain, store-queue or MIO stall, "
        "LONG SCOREBOARD 5 a issue, warps active 12% of peak: latency with ~2 warps a scheduler to hide it -- each "
        "16-byte store's address reused by the next trip of a rolled loop, each ring edge a polled wait. THIS FORM: four "
        "16-byte stores in flight a thread (each its own address), a token's run by consecutive threads (coalesced), "
        "the stub's ring one stage a job. A CTA's ~25K cycles were ~2,900 a job's epilogue (~10 latency-bound trips of "
        "~290) and ~1,350 its ring edges; now ~4 trips a job (~1,200) and one round trip (~340): ~9.7K a CTA, under its "
        "157.7 KB at HBM's write share (13.2 bytes a cycle an SM: 11.9K), so the launch meets the floor's write rate: "
        "101.9 us and the tail's wave; hidden 0 and two CTAs an SM the same; the lo plane at its 186.6 us floor",
        (
            Claim(CELL, lambda R: R[CELL], 104.0, 130.0, "us"),
            Claim(CELL + ": over its floor", lambda R: R[CELL] / FLOOR_US, 1.02, 1.28, "times"),
            Claim(NO_STAGES, lambda R: R[NO_STAGES], 102.0, 128.0, "us"),
            Claim(TWO, lambda R: R[TWO], 102.0, 128.0, "us"),
            Claim(LO, lambda R: R[LO], 188.0, 225.0, "us"),
        ),
    ),
    (
        "20261128T130545Z-524606-99b4f497",
        "2026-11-28",
        "four stores in flight and coalesced token runs moved every row ~17% (plain 170.1 us, hidden 0 163.8, two CTAs "
        "an SM 161.9, the lo plane 261.6) and missed every band: ncu on the traced launch -- DRAM 58% of its peak, lts "
        "55%, l1tex 25%, long scoreboard 4 a issue (9 at two CTAs an SM), warps active 11% -- still latency with ~9 warps "
        "an SM. The move from one store a trip to four (a row tile's 8 trips to 2) bought 17%, so a trip's latency is "
        "~a third of a CTA's time and the rest is the stall a instruction (~7 a issue) of the rest of its stream. THIS "
        "FORM: eight stores in flight a thread, a share's row tile in one trip (2 -> 1): the trips' third shrinks by "
        "about half, ~6% of the launch; the rest stays. THE OCCUPANCY ROW tests the claim that what remains is the CTA's ~9 "
        "warps an SM: the stores alone (the epilogues and tickets, no ring, no reserved shared memory) at three CTAs "
        "an SM, ~27 warps (four would spill at 56 registers a thread): if occupancy is the gap they reach the floor's write rate, 102-122 us; at two CTAs an SM the "
        "launch closed only 5% (170.1 -> 161.9), so a row near 155-165 would refute it and name the SM's store path",
        (
            Claim(CELL, lambda R: R[CELL], 150.0, 168.0, "us"),
            Claim(NO_STAGES, lambda R: R[NO_STAGES], 144.0, 162.0, "us"),
            Claim(TWO, lambda R: R[TWO], 143.0, 161.0, "us"),
            Claim(LO, lambda R: R[LO], 235.0, 259.0, "us"),
            Claim(STORES, lambda R: R[STORES], 102.0, 122.0, "us"),
        ),
    ),
)


def write(root: pathlib.Path, data: pathlib.Path, instance: str) -> list[pathlib.Path]:
    return write_banded(root, data, instance, "produce_skeleton", RUNS)
