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

from rola import RoLA, RouteGateway, SimplexProducer, dense_routing, union_routing
from rola.ops import chunk as chunk_arm

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
        routing = union_routing(self.alpha) if self.alpha is not None else dense_routing()
        producer = SimplexProducer(tuple(self.widths), routing,
                                   hidden_size=self.hidden_size, num_heads=self.H)
        gateway = RouteGateway([producer], num_heads=self.H,
                               hidden_size=self.hidden_size, normalization="global")
        layer = RoLA(gateway, d_v=self.d_v, decay=None, layer_idx=0)
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

        with torch.no_grad():
            routes = layer.routes(x[:, :self.T])
        packed = chunk_arm.pack(routes)
        c, bc = chunk_arm.arm(routes)
        return {"cell": self, "layer": layer, "gateway": layer.routes, "x": x, "v": v,
                "routes": routes, "packed": packed, "C": c, "BC": bc,
                #: `v_prefill` is the operand in the CELL's dtype, which is what the layer
                #: and the op facade take; `v_kernel` is the bf16 contiguous form the raw
                #: kernel entries require, narrowed once here rather than inside a timed
                #: callable where it would be priced as part of the launch.
                "v_prefill": v[:, :self.T].contiguous(),
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

        #: **THE DEEP-ARM CELL.** `D = 4` at width 8 -- the deepest topology the built
        #: manifest carries, and the one the paging bench uses for the same reason. It
        #: moves the level count while `N = 4096` is held identical to the pinned cell's,
        #: so a per-level cost is separable from a per-leaf one.
        Cell(name="chunk-deep-D4-w8", widths=(8, 8, 8, 8), B=2, T=512, H=8, hidden=512,
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
    packed = fx["packed"]
    live = chunk_arm.stats(packed).bool()
    return {
        "N": cell.N,
        "BH": cell.B * cell.H,
        "C": fx["C"],
        "BC": fx["BC"],
        "read_nonzero": int((packed.pr != 0).sum()),
        "write_nonzero": int((packed.pw != 0).sum()),
        "amplitudes_total": int(packed.pw.numel()),
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
                         f"`bench_chunk.py manifest --write`")
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
