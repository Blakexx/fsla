#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE HOPPER CARRY'S FLOOR, COUNTED: what one token costs an SM, term by term, for a declared routing shape, a box
and a cluster -- before any hopper carry exists, so a design is priced by one model and not by arithmetic redone a
conversation.

    bazel run //tools:carry_floor -- --widths 256 256 --read-sparse 0 1 --write-sparse 1 0 --k 4 \\
        --box 16 16 --cluster 1 1 [--pair fp32 | f16] [--window 1024] [--dv 64] [--tokens 65536] [--json floor.json]

THE SHAPE. `--widths` a level; `--read-sparse` / `--write-sparse` a flag a level (a sparse level holds `--k` live
digits a token, drawn uniformly; a dense one all of them). A BOX is `--box` digits of each level; a CLUSTER is
`--cluster` boxes along each level. A token's digits are independent draws, so every count below is an expectation.

THE TERMS, SM-cycles a token (each line of the report says which rate it used and whether that rate is MEASURED --
read from an H100 lab record in the pinned data (`data:lab/`, measure/data.py) at run time -- or ESTIMATED, a constant of this file with
its reason):
- tensor: the leaves a token reads, times the channels, over the pipe's products a cycle (`wgmma_rate`);
  a value width under 64 runs wgmma's 64 rows (the idle half), and the readout's leaves of a list in a box run
  whole steps of sixteen (its K). THE FOLD is priced by its steps' TIME, MEASURED (`wgmma_width`): a token in a
  list of a box is a sixteenth of a step as wide as the list's leaves in the box; a step of 8 columns is over half
  a step of 64, and one past 64 is whole 64-column steps;
- the readout's operand: its (token, leaf) coefficients expanded into shared memory, at the SM's shared access rate
  (`byte_expand`, its bank-spread form);
- the output's sum: by default (`--sum none`) every box drains its own partial to global memory (`drain_rate`:
  from rows no other CTA adds into, to rows spread over a window that every CTA shares), and a cluster with no
  sum is refused. Summed in a cluster first by one of two forms, it is then one fp32 drain a (token, cluster read). Both
  forms are MEASURED (`cluster_sum`, eight collectors a cluster, its rows at two round sizes split into a cost a
  round and a cost a token), and neither pays: `slots`, the fp32 sum the format states -- a box's partial staged and
  sent by a bulk copy into its own slot of a collector, which adds a token's slots; `s64` -- the partial as 64-bit
  integers by a bulk reduction into the collector's panel, outside the fp32 the format states, counted for
  comparison. A cluster's CTAs hand over once a ROUND of `--round` partials a CTA, and the round's cost is its own
  term (beyond 64 partials a round it is the line's, UNMEASURED);
- THE PAIR (`--pair f16 | fp32`, the width the plan states): a CTA holds two boxes neighbouring along a level the
  read side is dense on, one readout warpgroup runs each read digit's step on both boxes' images into one
  accumulator, and drain warps add ONE line a token. `f16`: the steps, the panel and the drain MEASURED whole
  (`readout_role`, by 4 to 1 drain warps), in place of the readout's tensor term and a drain a box. `fp32`: that
  measurement and one more line of the drain (`drain_rate`), an ESTIMATED sum. Both store the state's image once a
  window (`state_image`). Priced at 64 channels and for sides sparse on different levels alone: nothing else was
  measured;
- the fold's operands, one of two forms, the cheaper at its low end. LISTED: a (token, list, box) entry's value and coefficients
  fetched from scattered rows into the list's tiles by producer warps, MEASURED whole (`pool_fill`, at 16 leaves a
  list; a list's leaves past 16 are priced at the shared access rate, ESTIMATED). TILED: every box takes the window's whole value tile packed
  (`state_move`; every token's bytes), expands its coefficients into shared memory and gathers the values into
  list order (that copy ESTIMATED at the shared access rate; no lab has read it). A cluster shares no load: a
  multicast copy costs each CTA more than its own does (`value_cast`);
- the state in and out, a call (24-bit state, at a memory bandwidth).

NOT COUNTED: the hand-over between producer and consumer warps, the den's readout, the divergent carve's register
swap, paging's atom loads, the facts pass, the intra and the producer. FlashAttention's tensor floor at the same
tokens and width is printed beside the total.

Docs: docs/internals/tools/carry_floor.md.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from measure import data  # noqa: E402 -- the path insert above must precede it

#: wgmma's rows: the channels a warpgroup's MMA carries
MMA_ROWS = 64
#: the output widths `wgmma_width` reads a step's own time at; a step past 64 columns is whole 64-column steps
STEP_WIDTHS = (8, 16, 32, 64)
#: ESTIMATED, each with its reason: nothing on an H100 has measured these
ESTIMATES = {
    "clock": (1.8e9, "SM cycles a second: inferred from the card's published tensor rate, not measured"),
    "memory_low": (2.0e12, "bytes a second across memory, the low guess"),
    "memory_high": (3.0e12, "bytes a second across memory, the high guess"),
}
#: an SM's count on the card the records are from
SMS = 132


def median(lab: str, row: str) -> float:
    """A lab record's row on the H100: its median (refused if the record or the row is gone)."""
    path = data.lab(lab) / "sm_90a.json"
    if not path.exists():
        raise SystemExit(f"carry_floor: no record data:lab/{lab}/sm_90a.json")
    for result in json.loads(path.read_text())["results"]:
        if result["name"] == row:
            return float(result["median"])
    raise SystemExit(f"carry_floor: data:lab/{lab}/sm_90a.json has no row `{row}`")


def measured() -> dict[str, tuple[float, str]]:
    """THE MEASURED RATES, each with the record row it is read from."""
    pipe_row = "6 wgmmas a group, A registers, two warpgroups"
    spread_row = "the expansion, a warp's lanes on eight bank groups, 8 warps a CTA"
    #: twelve m64n64k16 wgmmas an SM a step; a lane's 6 stores and 1 load of 16 bytes, 32 lanes, 8 warps a tile time
    products = 12 * 64 * 64 * 16 / median("wgmma_rate", pipe_row)
    shared = 8 * 7 * 16 * 32 / median("byte_expand", spread_row)
    return {
        "products": (products, f"wgmma_rate / {pipe_row}"),
        "shared": (shared, f"byte_expand / {spread_row} (7 16-byte accesses a lane)"),
        "drain_own": (median("drain_rate", "each CTA its own rows, 4 warps") / 256.0,
                      "drain_rate / each CTA its own rows (a byte; the same at 4, 8 and 16 warps)"),
        "drain_spread": (median("drain_rate", "rows spread over 1,024, 4 warps") / 256.0,
                         "drain_rate / rows spread over 1,024 that every CTA shares (a byte)"),
        "load_packed": (median("state_move", "one bulk copy in from global memory, blocks of 8192 bytes") * 128 / 1024,
                        "state_move / one bulk copy in from global memory, blocks of 8192 bytes (128 bytes of it)"),
        **{f"step_{n}": ((median("wgmma_width", f"{n} columns, 8 steps a group")
                          - median("wgmma_width", f"{n} columns, 4 steps a group")) / 4,
                         f"wgmma_width / {n} columns: a step more, from 4 a group to 8")
           for n in STEP_WIDTHS},
        "line_low": (median("readout_role", "one readout warpgroup over two images, 4 drain warps"),
                     "readout_role / one readout warpgroup over two images, 4 drain warps (a token a pair)"),
        "line_high": (median("readout_role", "one readout warpgroup over two images, 1 drain warp"),
                      "readout_role / one readout warpgroup over two images, 1 drain warp"),
        "image": (median("state_image", "a value a 2-byte store, the image alone"),
                  "state_image / a value a 2-byte store, the image alone (a window a CTA of two boxes)"),
        "entry_low": (median("pool_fill", "the producers and the fold, 8 producers"),
                      "pool_fill / the producers and the fold, 8 producers (an entry's value and 16 coefficients)"),
        "entry_high": (median("pool_fill", "the producers and the fold, 2 producers"),
                       "pool_fill / the producers and the fold, 2 producers"),
        **cluster_rates(),
    }


def cluster_rates() -> dict[str, tuple[float, str]]:
    """A cluster form's two costs from `cluster_sum`'s rows at 16 and at 64 tokens a round, eight collectors: what a
    round costs a CTA whatever it holds, and what a partial costs a byte (256 a token in the lab)."""
    out = {}
    for name, form in (("slots", "fp32 slots by a bulk copy"), ("s64", "64-bit integers by a bulk reduction")):
        at16, at64 = (median("cluster_sum", f"{form}, eight collectors, rounds of {n} tokens") for n in (16, 64))
        a_round = (at16 - at64) / (1 / 16 - 1 / 64)
        where = f"cluster_sum / {form}, eight collectors, rounds of 16 and of 64 tokens"
        out[f"round_{name}"] = (a_round, f"{where}: the line's cost a round, a CTA")
        out[f"hop_{name}"] = ((at64 - a_round / 64) / 256.0, f"{where}: the line's cost a token (a byte of partial)")
    return out


@dataclass(frozen=True)
class Side:
    """One side's expected counts a token."""
    leaves: float
    boxes: float
    clusters: float
    lists: float


def distinct(ranges: float, draws: int) -> float:
    """The expected distinct ranges `draws` uniform digits fall in, of `ranges` equal ones."""
    return ranges * (1.0 - (1.0 - 1.0 / ranges) ** draws)


def side(widths, sparse, k: int, box, cluster) -> Side:
    """A side's counts: the leaves a token touches, the boxes and clusters those fall in, and the lists it sits in a
    box -- the box's leaves it touches over its digits on the level the MMA runs along (a dense level of the side
    inside the box where there is one, else its level of most digits)."""
    leaves = boxes = clusters = 1.0
    in_box = []
    for width, flag, b, c in zip(widths, sparse, box, cluster):
        digits = min(k, width) if flag else width
        ranges = width / b
        touched = distinct(ranges, digits) if flag else ranges
        leaves *= digits
        boxes *= touched
        clusters *= distinct(ranges / c, digits) if flag else ranges / c
        in_box.append(digits / touched)
    dense = [n for n, flag in zip(in_box, sparse) if not flag]
    along = max(dense) if dense else max(in_box)
    return Side(leaves, boxes, clusters, (leaves / boxes) / along)


def floor(widths, read_listed, write_listed, k: int, box, cluster, dv: int, tokens: int, form: str = "none",
          round_partials: int = 64, pair: str = "none", window: int = 1024) -> dict:
    """The report: each term's SM-cycles a token as `(low, high)`, with the rate it used."""
    rates = measured()
    est = {name: value for name, (value, _) in ESTIMATES.items()}
    read = side(widths, read_listed, k, box, cluster)
    write = side(widths, write_listed, k, box, cluster)
    rows = max(dv, MMA_ROWS)
    clustered = math.prod(cluster) > 1
    if clustered and form == "none":
        raise SystemExit("carry_floor: a cluster that sums nothing changes nothing (it shares no load): name its "
                         "`--sum`, or no cluster")
    if form != "none" and not clustered:
        raise SystemExit(f"carry_floor: `--sum {form}` is a cluster's sum, and no cluster is named")
    products, shared = rates["products"][0], rates["shared"][0]
    drain = (rates["drain_own"][0], rates["drain_spread"][0])
    value_requests = math.ceil(dv * 2 / 128)
    summed = clustered
    hop = rates[f"hop_{form}"][0] * dv * 4 if summed else 0.0
    barriers = read.boxes * rates[f"round_{form}"][0] / round_partials if summed else 0.0

    def both(x: float) -> tuple[float, float]:
        return (x, x)

    all_boxes = math.prod(w / b for w, b in zip(widths, box))

    drains = read.clusters if summed else read.boxes
    #: THE PAIR (the module's notes): its line and image are whole measurements at 64 channels of a pair that reads
    #: images, so only that width and only a divergent carve are priced
    paired = pair != "none"
    if paired:
        if clustered or not any(not flag and width // b >= 2 for width, flag, b in zip(widths, read_listed, box)):
            raise SystemExit("carry_floor: a pair is two boxes along a level the read side is dense on, with no "
                             "cluster")
        if tuple(read_listed) == tuple(write_listed):
            raise SystemExit("carry_floor: a pair's readout reads both boxes' images; where the two sides are sparse "
                             "on the same levels the image is laid along a sparse digit's dense digits, which "
                             "the lab `carry_window` measures and this tool has no term for")
        if dv != MMA_ROWS:
            raise SystemExit(f"carry_floor: a pair's line and image are measured at {MMA_ROWS} channels alone")
        #: an fp32 line is the f16 line's measurement and one more line of the drain (`drain_rate`), ESTIMATED so
        more = rates["drain_own"][0] * 128 if pair == "fp32" else 0.0
        line = (read.boxes / 2 * (rates["line_low"][0] + more), read.boxes / 2 * (rates["line_high"][0] + more))
        image = both(all_boxes / 2 * rates["image"][0] / window)
    else:
        line = image = both(0.0)
    #: THE FOLD'S OPERANDS, the cheaper at its low end of LISTED (an entry at `pool_fill`'s cost, a list's leaves past its 16 at the
    #: shared access rate) and TILED (the window's packed tile a box, expanded and gathered): see the module's notes
    entries = write.boxes * write.lists * value_requests
    wide = write.boxes * write.lists * max(0.0, write.leaves / (write.boxes * write.lists) - 16) * 2 / shared
    listed = (entries * rates["entry_low"][0] + wide, entries * rates["entry_high"][0] + wide)
    tiled = both(all_boxes * value_requests * rates["load_packed"][0] + write.leaves * 2 / shared
                 + write.boxes * write.lists * dv * 2 * 2 / shared)
    operands, operands_by = ((listed, "LISTED: entry_low .. entry_high, a list's leaves past 16 at shared (ESTIMATED)") if listed[0] <= tiled[0] else
                             (tiled, "TILED: load_packed, shared, and the gather ESTIMATED"))

    #: A FOLD STEP IS PRICED BY ITS TIME (`wgmma_width`): a token in a list of a box is a sixteenth of a step as
    #: wide as the list's leaves in the box -- the narrowest measured width that holds them, whole 64-column steps
    #: past 64
    def fold_step(a: Side) -> tuple[float, str]:
        along = a.leaves / (a.boxes * a.lists)
        width = next((n for n in STEP_WIDTHS if n >= along - 1e-9), 64)
        return rates[f"step_{width}"][0] * math.ceil(along / 64 - 1e-9), f"step_{width}"

    #: THE READOUT'S PADDING: a list's leaves in a box are its K, whole steps of sixteen

    def padded(a: Side) -> float:
        along = a.leaves / (a.boxes * a.lists)
        return a.boxes * a.lists * 16 * math.ceil(along / 16 - 1e-9)

    terms = {
        "tensor, readout": (both(0.0 if paired else padded(read) * rows / products),
                            "in the pair's line" if paired else "products"),
        "tensor, fold": (both(write.boxes * write.lists * fold_step(write)[0] / 16), fold_step(write)[1]),
        "readout operand, expanded": (both(read.leaves * 2 / shared), "shared"),
        "sum: partials into the cluster": (both(read.boxes * hop), f"hop_{form}" if summed else "no cluster sum"),
        "sum: the rounds' barriers": (both(barriers), f"round_{form}" if summed else "no cluster sum"),
        "sum: drains to global memory": (tuple(0.0 if paired else drains * (dv * 4 + 4) * d for d in drain),
                                         "in the pair's line" if paired else "drain_own .. drain_spread"),
        "readout and sum: a pair's line": (line, {"none": "no pair", "f16": "line_low .. line_high",
                                                  "fp32": "line_low .. line_high, and a line of drain_own "
                                                          "(ESTIMATED sum)"}[pair]),
        "state image, a window": (image, "image" if paired else "no pair"),
        "fold operands": (operands, operands_by),
    }
    total = tuple(sum(v[0][i] for v in terms.values()) for i in (0, 1))
    seconds = tuple(t * tokens / SMS / est["clock"] for t in total)
    state = math.prod(widths) * (dv + 1) * 3 * 2
    flash = tokens / 2 * dv * 2 / products
    return {
        "shape": {"widths": list(widths), "read_listed": list(read_listed), "write_listed": list(write_listed), "k": k,
                  "box": list(box), "cluster": list(cluster), "dv": dv, "tokens": tokens, "sum": form},
        "counts": {"read": vars(read), "write": vars(write)},
        "terms": {name: {"cycles": list(cycles), "rate": rate} for name, (cycles, rate) in terms.items()},
        "total_cycles_a_token": list(total),
        "call_seconds": list(seconds),
        "state_seconds_a_call": [state / est["memory_high"], state / est["memory_low"]],
        "flash_cycles_a_token": flash,
        "flash_call_seconds": flash * tokens / SMS / est["clock"],
        "measured": {name: {"value": value, "from": where} for name, (value, where) in rates.items()},
        "estimated": {name: {"value": value, "why": why} for name, (value, why) in ESTIMATES.items()},
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--widths", type=int, nargs="+", required=True)
    ap.add_argument("--read-sparse", dest="read_listed", type=int, nargs="+", required=True, help="a flag a level: 1 sparse (walked by its lists), 0 dense")
    ap.add_argument("--write-sparse", dest="write_listed", type=int, nargs="+", required=True)
    ap.add_argument("--k", type=int, required=True, help="a sparse level's live digits a token")
    ap.add_argument("--box", type=int, nargs="+", required=True, help="a box's digits of each level")
    ap.add_argument("--cluster", type=int, nargs="+", required=True, help="a cluster's boxes along each level")
    ap.add_argument("--sum", choices=("none", "slots", "s64"), default="none",
                    help="the cluster's sum: none (every box drains its own), fp32 slots by a bulk copy, or 64-bit "
                         "integers by a bulk reduction")
    ap.add_argument("--round", type=int, default=64,
                    help="the partials a CTA hands over between its barriers (64 is the round the lab read)")
    ap.add_argument("--pair", choices=("none", "fp32", "f16"), default="none",
                    help="a CTA's two boxes along a read-dense level drain one f16 line a token")
    ap.add_argument("--window", type=int, default=1024, help="the tokens a window: a pair's state image is stored once")
    ap.add_argument("--dv", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=65536)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)

    depth = len(a.widths)
    for name, flag in (("read_listed", "--read-sparse"), ("write_listed", "--write-sparse"), ("box", "--box"), ("cluster", "--cluster")):
        if len(getattr(a, name)) != depth:
            raise SystemExit(f"carry_floor: {flag} takes one value a level ({depth})")
    for width, b, c in zip(a.widths, a.box, a.cluster):
        if width % b or (width // b) % c:
            raise SystemExit(f"carry_floor: a level of {width} digits is not whole boxes of {b} in whole clusters of {c}")

    report = floor(a.widths, a.read_listed, a.write_listed, a.k, a.box, a.cluster, a.dv, a.tokens, a.sum, a.round,
                   a.pair, a.window)
    read, write = report["counts"]["read"], report["counts"]["write"]
    print(f"a token reads {read['leaves']:.0f} leaves in {read['boxes']:.1f} boxes of {read['clusters']:.2f} clusters; "
          f"writes {write['leaves']:.0f} in {write['boxes']:.1f} boxes of {write['clusters']:.2f} clusters, "
          f"{write['lists']:.2f} lists a box")
    for name, term in report["terms"].items():
        low, high = term["cycles"]
        span = f"{low:8.1f}" if low == high else f"{low:8.1f} .. {high:.1f}"
        print(f"  {name:32s} {span:22s} {term['rate']}")
    low, high = report["total_cycles_a_token"]
    print(f"  {'TOTAL, SM-cycles a token':32s} {low:8.1f} .. {high:.1f}")
    s_low, s_high = report["call_seconds"]
    print(f"a call of {a.tokens} tokens: {1e6 * s_low:.0f} .. {1e6 * s_high:.0f} us (ESTIMATED clock, {SMS} SMs); "
          f"its state in and out {1e6 * report['state_seconds_a_call'][0]:.1f} .. "
          f"{1e6 * report['state_seconds_a_call'][1]:.1f} us more (ESTIMATED memory)")
    print(f"FlashAttention's tensor floor: {report['flash_cycles_a_token']:.0f} SM-cycles a token, "
          f"{1e6 * report['flash_call_seconds']:.0f} us the call")
    if a.json:
        a.json.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
