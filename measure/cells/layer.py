# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""FSLA'S LAYER CONSTRUCTIONS: what a FSLA layer is built as on a central layer cell.

A central layer cell (`fsla_devtools.cells.layer`) is only an input: hidden states and values drawn from a seed. How a
FSLA layer is built to read it -- the routing widths, the entmax alpha (none selects dense softmax routing) and the logit
gain the producer's projection is scaled by -- is FSLA's, so it lives here as a named CONSTRUCTION with the input cells
it was declared for. The names are the layer cells' names before the inputs moved to fsla-devtools (each central
layer cell's note names them), so a construction reads as the cell a published number cites.

**A CONSTRUCTION ON A CELL IS DETERMINISTIC.** `build` seeds the parameters from the cell's own seed immediately before
construction and takes the inputs from the cell's draw, which never moves with the construction.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Construction:
    name: str
    widths: tuple[int, ...]
    alpha: float | None
    logit_gain: float
    #: the central layer cells this construction is declared for
    cells: tuple[str, ...]
    note: str


def _c(name, widths, alpha, gain, cell, note):
    return Construction(name, tuple(widths), alpha, gain, (cell,), note)


CONSTRUCTIONS = {c.name: c for c in (
    _c("chunk-p73-pinned", (64, 64), 1.5, 1.0, "layer-B2-T2048-H8-h512-dv64-bf16-s0",
       "the inventory constructor every published step number was taken on"),
    _c("chunk-dense-D2", (64, 64), None, 1.0, "layer-B2-T512-H8-h512-dv64-bf16-s2", "softmax: the dense end"),
    _c("chunk-dense-NL4096", (64, 64), None, 1.0, "layer-B2-T4096-H8-h512-dv64-bf16-s6", "dense at T = N = 4096"),
    _c("chunk-deep-D4-w8", (8, 8, 8, 8), 1.5, 8.0, "layer-B2-T512-H8-h512-dv64-bf16-s3", "D = 4 at N = 4096"),
    _c("chunk-decode-w16", (16, 16), 1.5, 1.0, "layer-B2-T128-H2-h128-dv64-fp32-s4-dec8", "prefill + 8 carried steps"),
    _c("chunk-decode-la-w256", (256,), None, 1.0, "layer-B2-T128-H2-h128-dv64-fp32-s10-dec8",
       "D = 1 flat dense: the LA control for the FSLA decode"),
    #: THE CAPACITY-FAIR LONG LAYER: T = N = 65,536 a head, four heads, the alternating split, at rising logit gains --
    #: the weights realize the support (`tools/realized_support.py`), so a layer reading spans it
    *(_c("nl64k-alt-h4" + ("" if gain == 1.0 else f"-g{gain:g}"), (256, 256), 1.5, gain,
         "layer-B1-T65536-H4-h256-dv64-bf16-s11", f"the capacity-fair long layer at logit gain {gain:g}")
      for gain in (1.0, 4.0, 16.0)),
    #: THE PRODUCER SWEEP: one entmax constructor at rising logit gains; the realized density must fall across it
    *(_c(f"producer-sweep-g{gain:g}", (64, 64), 1.5, gain, "layer-B1-T128-H4-h64-dv64-fp32-s201",
         "the producer sweep: the logit gain moves the realized density") for gain in (0.25, 1.0, 4.0, 16.0)),
)}


def _routing(construction: Construction):
    """The construction's routing, one read head's tuple for every head. Every side has a dense level (docs/api.md
    section 2.1), so an entmax construction is the ALTERNATING split: entmax on the write side at the even levels
    (read-dense) and on the read side at the odd (write-dense). A softmax construction, and any depth-one one, is dense
    on both sides."""
    from fsla import Routing, entmax, softmax

    widths = construction.widths
    if construction.alpha is None or len(widths) == 1:
        return Routing(topology=tuple(widths), write=(softmax(),) * len(widths), read=((softmax(),) * len(widths),))
    sparse = [level % 2 == 0 for level in range(len(widths))]
    return Routing(topology=tuple(widths),
                   write=tuple(entmax(construction.alpha) if s else softmax() for s in sparse),
                   read=(tuple(softmax() if s else entmax(construction.alpha) for s in sparse),))


class Fixture(dict):
    """A fixture whose parts are each built on first read: an arm builds only what its launch reads, so the solve sets up
    on a line whose producer is not built, and only the arm that reads the produced routes asks for the producer."""

    def __init__(self, parts: dict, **given) -> None:
        super().__init__(**given)
        self._parts = parts

    def __missing__(self, key):
        if key not in self._parts:
            raise KeyError(key)
        value = self[key] = self._parts[key](self)
        return value


def _layer(fx):
    """The layer built by the construction in the cell's dtype, its parameters seeded from the cell's seed immediately
    before construction."""
    import torch

    from fsla import FSLA

    cell, construction = fx["cell"], fx["construction"]
    torch.manual_seed(cell.seed)
    layer = FSLA(_routing(construction), hidden_size=cell.hidden_size, num_heads=cell.H, d_v=cell.dv, decay=None,
                 layer_idx=0).to("cuda", cell.torch_dtype).eval()
    if construction.logit_gain != 1.0:
        with torch.no_grad():
            layer.route_W.mul_(construction.logit_gain)
    for param in layer.parameters():
        param.requires_grad_(False)
    return layer


def _drawn(fx):
    from fsla_devtools.cells.layer import realize

    return realize(fx["cell"])


def _routes(fx):
    """The routes the layer's producer makes for the prefill."""
    import torch

    with torch.no_grad():
        return fx["layer"]._produce(fx["x"][:, :fx["cell"].tokens])[0]


#: each part of a layer fixture, built from the parts it reads; the inputs come from the cell's own generator, so the order
#: the parts are built in never moves them
PARTS = {
    "layer": _layer,
    "drawn": _drawn,
    "x": lambda fx: fx["drawn"].x,
    "v": lambda fx: fx["drawn"].v,
    "routes": _routes,
}


def build(cell, construction: Construction) -> Fixture:
    """The fixture: the layer built by `construction` (`layer`), the cell's inputs (`x`, `v`) and the routes the layer
    produces for its prefill (`routes`), each built when an arm first reads it.

    IT BUILDS NO CALL PLAN. What a subject times is the launch it names, and the packing a particular op seam does for
    it is that subject's own business.
    """
    if cell.name not in construction.cells:
        raise ValueError(f"{construction.name} is declared for {list(construction.cells)}, not {cell.name}")
    return Fixture(PARTS, cell=cell, construction=construction)


__all__ = ["CONSTRUCTIONS", "PARTS", "Construction", "Fixture", "build"]
