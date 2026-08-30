"""WHAT is measured: the frozen chunk-arm cell registry and its committed manifest.

**A CELL IS ITS CONSTRUCTOR AND ITS SEED.** Every field of :class:`Cell` is a
construction parameter, `build()` is deterministic from them, and documents cite a cell
BY NAME (`chunk-p73-pinned`, `chunk-sparse-gain8`) and never by a description of its
density. Two numbers carrying the same cell name are commensurable; two carrying
different names are not.

The rule is inherited rather than invented: the retired tiled registry states the two
occasions this repository lost a fixture by describing it with statistics instead of
constructing it, and its at-scale cell is permanently non-reproducible as a result. A
STATISTIC IS NOT A FIXTURE.

:func:`check_manifest` is the standing gate. Every realized statistic in the manifest is
INTEGER-DERIVED -- support counts and live-atom counts, never a float reduction -- so a
drift in torch's RNG, in the entmax solver or in the producer announces itself as a diff
rather than as a tolerance question.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from math import prod
from pathlib import Path

import torch

from rola import RoLA, RouteProducer, dense_routing, union_routing
from rola.engine.dags.chunk_dag import ChunkContext, build_chunk_plan
from rola.engine.facts import planes

MANIFEST = Path(__file__).resolve().parents[1] / "cells" / "chunk_manifest.json"

_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


@dataclass(frozen=True)
class Cell:
    """One frozen chunk-arm fixture: the constructor, the seed, and nothing derived."""

    name: str
    widths: tuple[int, ...]
    B: int
    T: int
    H: int = 8
    d_v: int = 64
    hidden: int = 0
    #: `None` selects `dense_routing()`; a float selects `union_routing(alpha)`, the
    #: sparse write side the paper's cells are stated at.
    alpha: float | None = 1.5
    #: The producer's routing projection is scaled by this before any input is drawn.
    #: It is the gain knob the density sweep is stated on, so it is a CELL field.
    logit_gain: float = 1.0
    dtype: str = "bf16"
    seed: int = 0
    #: Decode steps taken after the prefill, for the continuation cells. `0` is a
    #: prefill-only cell and its decode subjects are not applicable.
    decode_steps: int = 0
    note: str = ""

    @property
    def hidden_size(self) -> int:
        return self.hidden or self.H * self.d_v

    @property
    def N(self) -> int:
        return prod(self.widths)

    @property
    def torch_dtype(self):
        return _DTYPES[self.dtype]

    def build(self) -> dict:
        """The fixture, built deterministically from this cell alone.

        The module seed and the input seed are separate draws: parameters come from
        `manual_seed(seed)` immediately before construction (the convention every layer
        bench and layer test uses), inputs from an explicit device generator at
        `seed + 500`, so changing `T` does not move the parameters.
        """
        torch.manual_seed(self.seed)
        template = (union_routing(1, self.alpha) if self.alpha is not None
                   else dense_routing(1))
        levels = [template.at(w) for w in self.widths]
        producer = RouteProducer(levels, hidden_size=self.hidden_size, num_heads=self.H,
                                 normalization="global")
        layer = RoLA(producer, d_v=self.d_v, decay=None, layer_idx=0)
        layer = layer.to("cuda", self.torch_dtype).eval()
        if self.logit_gain != 1.0:
            with torch.no_grad():
                producer.route_W.mul_(self.logit_gain)
        for param in layer.parameters():
            param.requires_grad_(False)

        g = torch.Generator(device="cuda").manual_seed(self.seed + 500)
        total = self.T + self.decode_steps
        x = torch.randn((self.B, total, self.hidden_size), device="cuda",
                        dtype=self.torch_dtype, generator=g)
        v = torch.randn((self.B, total, self.H, self.d_v), device="cuda",
                        dtype=self.torch_dtype, generator=g)

        #: `v_prefill` is the operand in the CELL's dtype, which is what the layer and
        #: the op facade take; `v_kernel` is the bf16 contiguous form the raw kernel
        #: entries require, narrowed once here rather than inside a timed callable
        #: where it would be priced as part of the launch.
        v_prefill = v[:, :self.T].contiguous()
        #: THE PLAN IS THE FIXTURE'S PACKING, and it is the production one: every
        #: launch argument a subject reads -- the packed sides, the write gain, the
        #: read mass, the arm, the union table -- is a field the chunk DAG produced
        #: for a real call, not a bench-local rebuild that could drift from it.
        #: Built here, once, so nothing it costs lands inside a timed callable.
        with torch.no_grad():
            routes = layer.routes(x[:, :self.T])
            plan = build_chunk_plan(ChunkContext(routes=routes, v=v_prefill))
        return {"cell": self, "layer": layer, "producer": layer.routes, "x": x, "v": v,
                "routes": routes, "plan": plan, "C": plan.arm.c, "BC": plan.arm.bc,
                "v_prefill": v_prefill,
                "v_kernel": v[:, :self.T].to(torch.bfloat16).contiguous()}


def _cells() -> dict[str, Cell]:
    cs = [
        #: **THE PINNED CELL.** Byte for byte the constructor P73's inventory and every
        #: stage measurement of it were taken on (`workflows/e2e-cuda-complete/
        #: inventory.md`: hidden 512, H=8, d_v=64, B=2, T=2048, widths (64,64),
        #: `union_routing(1.5)`, bf16, eval, no_grad, arm (C=32, BC=128)). Numbers
        #: quoted against this name ARE commensurable with the step's 2.520 -> 2.031 ms
        #: ladder.
        Cell(name="chunk-p73-pinned", widths=(64, 64), B=2, T=2048, H=8, hidden=512,
             alpha=1.5, seed=0,
             note="THE pinned cell: P73's inventory constructor, unchanged"),

        #: **THE SPARSE CELL.** The pinned constructor with the routing gain the density
        #: sweep is stated at, so the write support is genuinely sparse and the
        #: dead-block share the sparse grid exploits is large. `T` is a quarter of the
        #: pinned cell's because the axis under study is DENSITY, not length.
        Cell(name="chunk-sparse-gain8", widths=(64, 64), B=2, T=512, H=8, hidden=512,
             alpha=1.5, logit_gain=8.0, seed=1,
             note="the pinned constructor at routing gain 8: 4.6% per-token write "
                  "support, and a SEQUENCE union that is still fully live"),

        #: **THE DEAD-BLOCK CELL.** Per-token sparsity is not the same fact as liveness,
        #: and the registry says so with a cell rather than a footnote: MEASURED here,
        #: the atom union saturates with `T` at fixed gain -- 0.52 live at `T = 64`,
        #: 0.79 at 64 tokens and gain 8, 0.96 at 128, 0.998 at 256 -- so every cell above
        #: is fully live and the page plan and the sparse grid have nothing to skip on
        #: any of them. This is the arm where they do. Getting that wrong once made a
        #: whole gate vacuous on the tiled side (0.0 % read-only pairs), which is why it
        #: is a manifest column and not a comment.
        Cell(name="chunk-deadblocks-T64", widths=(64, 64), B=2, T=64, H=8, hidden=512,
             alpha=1.5, logit_gain=16.0, seed=5,
             note="~52% live atoms: the arm with dead blocks for the plan to skip"),

        #: **THE DENSE CELL.** `dense_routing()` on the same geometry: every owner is a
        #: candidate on every tile. It is the other end of the distribution the kernel
        #: is designed across, and it is the arm a sparse-only conclusion would be an
        #: artifact of.
        Cell(name="chunk-dense-D2", widths=(64, 64), B=2, T=512, H=8, hidden=512,
             alpha=None, seed=2,
             note="softmax routing: the dense end of the distribution space"),

        #: **THE DENSE N=L ANCHOR.** The dense constructor at `T = N = 4096`: the
        #: worst-case O(L*N) regime the parity claim is stated in, on the pinned cell's
        #: own geometry and duty pair (softmax read AND softmax write -- the d1-analog on
        #: the square topology). The rental measures N = L = 65536; this cell is the
        #: LOCAL anchor and the curve's second point, so the at-scale number is an
        #: extrapolation from two constructed fixtures rather than from one.
        Cell(name="chunk-dense-NL4096", widths=(64, 64), B=2, T=4096, H=8, hidden=512,
             alpha=None, seed=6,
             note="dense routing at T = N = 4096: the local N=L anchor for the "
                  "worst-case O(L*N) parity regime"),

        #: **THE REVERSE PASS' SECOND POINT.** Widths `(16, 16)` at `T = 2048`: the arm
        #: `(32, 64, 2, 16)`, `N = 256`, which is the grid-starved regime the backward
        #: profile's second cell was taken in and where the two scans' overlap is
        #: unambiguous. `chunk-decode-w16` shares the widths and is NOT it -- that cell is
        #: `T = 128, H = 2, fp32`, three constructor fields away.
        Cell(name="chunk-bwd-smalln", widths=(16, 16), B=2, T=2048, H=8, hidden=512,
             alpha=1.5, seed=7,
             note="the reverse pass' second point: N=256 at T=2048, the profile's "
                  "small-N cell"),

        #: **THE ONE-BLOCK BACKWARD CELL.** Widths `(8, 8)` at `T = 2048`: the arm
        #: `(32, 64, 2, 8)`, where `N = BC` and the block lattice is therefore ONE
        #: block -- the scans' walls carry no fan-in and no lattice term, so a
        #: regression here is a regression in a scan's own body rather than in the grid
        #: it is spread over. Dense on both duties: a sparse cell would fold the
        #: compaction's price into the same row.
        Cell(name="chunk-bwd-dense-N64", widths=(8, 8), B=2, T=2048, H=8, hidden=512,
             alpha=None, seed=8,
             note="N = BC = 64, dense: the one-block lattice, the scans' bodies "
                  "with no fan-in term"),

        #: **THE BACKWARD'S SPLIT-K CELL.** Widths `(256, 256)` -- the arm
        #: `(32, 128, 2, 256)`, `N = 65536`, the scale geometry, and a block lattice of
        #: 512. It is where the reverse pass' fan-in is stated at the depth the campaign
        #: actually runs: `dv` takes 512 partial contributions per (bh, token, channel)
        #: against the pinned cell's 32, and `nblk = N / BC` is the growth axis every
        #: fan-in design has to answer on. `B = 1, H = 2, T = 512`: the cell is sized so
        #: the LATTICE is at scale and the operand plane is not.
        Cell(name="chunk-bwd-splitk", widths=(256, 256), B=1, T=512, H=2, hidden=512,
             alpha=1.5, seed=9,
             note="N = 65536 at BC = 128: a 512-block lattice, the split-K fan-in's "
                  "own depth"),

        #: **THE DEEP-ARM CELL.** `D = 4` at width 8 -- the deepest topology the built
        #: manifest carries, and the one the paging bench uses for the same reason. It
        #: moves the level count while `N = 4096` is held identical to the pinned cell's,
        #: so a per-level cost is separable from a per-leaf one.
        Cell(name="chunk-deep-D4-w8", widths=(16, 16, 16, 16), B=2, T=512, H=8, hidden=512,
             alpha=1.5, logit_gain=8.0, seed=3,
             note="D=4 at N=4096: the level axis at fixed leaf count"),

        #: **THE DECODE CELL.** A short prefill and then single-token steps through the
        #: same carried state -- the shape the continuation subject and the decode-step
        #: subject both need. It is small on purpose: a decode step is a per-token
        #: latency, and a long prefill only lengthens the setup it is measured after.
        Cell(name="chunk-decode-w16", widths=(16, 16), B=2, T=128, H=2, hidden=128,
             alpha=1.5, dtype="fp32", seed=4, decode_steps=8,
             note="prefill + 8 carried single-token steps: the decode and "
                  "continuation cell"),

        #: **THE LA-CONTROL DECODE CELL.** `D = 1` at width 256 with dense routing on
        #: both duties: structurally a linear-attention step -- one flat state, no
        #: levels to expand, dense read/write -- on our own stack, at
        #: `chunk-decode-w16`'s exact B/T/H/d_v/dtype and the SAME `N = 256`. The pair
        #: isolates ROUTING'S OWN decode premium: the (16, 16) arm addresses the same
        #: 256 states through two 16-wide factors where a d_qk=16 LA head addresses 16,
        #: so any per-step gap is the price of that 16x addressability and nothing else.
        Cell(name="chunk-decode-la-w256", widths=(256,), B=2, T=128, H=2, hidden=128,
             alpha=None, dtype="fp32", seed=10, decode_steps=8,
             note="D=1 flat dense at N=256: the internal LA-control for the routed "
                  "decode cell"),
    ]
    return {c.name: c for c in cs}


CELLS = _cells()

#: Named GROUPS, so a run is cited by a name rather than by a slice of the registry.
GROUPS = {
    "all": list(CELLS),
    "prefill": [n for n, c in CELLS.items() if c.decode_steps == 0],
    "decode": [n for n, c in CELLS.items() if c.decode_steps > 0],
    #: The baseline session's roster. Every cell, because a baseline is a measurement of
    #: the registry and a partial one compares two sessions that were never comparable.
    "baseline": list(CELLS),
}


def realized(cell: Cell, fx: dict | None = None) -> dict:
    """This cell's INTEGER statistics, as the fixture actually realizes them.

    Support counts rather than amplitude sums: a count is bit-exact across runs, so the
    manifest diffs on a routing change and never on a reduction order.
    """
    fx = cell.build() if fx is None else fx
    plan = fx["plan"]
    #: The bitmap is taken from the PRIMITIVE, not off the plan: these cells bind no
    #: state, so the facts pass ran with the atom-grain emission skipped and `plan`
    #: carries no bitmap to read. This is the same reduction over the same plane.
    live = planes.written_atoms(planes.atom_bits(plan.write_plane, plan.widths))
    return {
        "N": cell.N,
        "BH": cell.B * cell.H,
        "C": fx["C"],
        "BC": fx["BC"],
        "read_nonzero": int((plan.read_plane != 0).sum()),
        "write_nonzero": int((plan.write_plane != 0).sum()),
        "amplitudes_total": int(plan.write_plane.numel()),
        "live_atoms": int(live.sum()),
        "atoms_total": int(live.numel()),
    }


def cell_manifest(cell: Cell, fx: dict | None = None) -> dict:
    out = {"name": cell.name, "widths": list(cell.widths), "B": cell.B, "T": cell.T,
           "H": cell.H, "d_v": cell.d_v, "hidden": cell.hidden_size, "alpha": cell.alpha,
           "logit_gain": cell.logit_gain, "dtype": cell.dtype, "seed": cell.seed,
           "decode_steps": cell.decode_steps}
    out.update(realized(cell, fx))
    out["write_support_frac"] = round(out["write_nonzero"] / out["amplitudes_total"], 6)
    out["live_atom_frac"] = round(out["live_atoms"] / out["atoms_total"], 6)
    return out


def build_manifest(names=None) -> dict:
    return {n: cell_manifest(CELLS[n]) for n in (names or CELLS)}


def check_manifest(names=None) -> bool:
    """A DRIFTED cell is a cell whose numbers are no longer comparable to the ones
    already published under its name, so this is a hard check and not a report."""
    if not MANIFEST.exists():
        raise SystemExit(f"no committed manifest at {MANIFEST}; record one with "
                         f"`python -c \"from bench.chunk_cells import write_manifest; write_manifest()\"` "
                         f"(the bench_chunk.py front door left with the K31 deletion batch)")
    want = json.loads(MANIFEST.read_text())
    got = build_manifest(names)
    bad = []
    for name in sorted(got):
        if want.get(name) != got[name]:
            bad.append(name)
            print(f"  DRIFT {name}\n    committed {want.get(name)}\n    realized  {got[name]}")
    print(f"{len(got) - len(bad)}/{len(got)} cells match the committed manifest")
    return not bad


def write_manifest() -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(build_manifest(), indent=1, sort_keys=True) + "\n")
