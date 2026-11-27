# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""FSLA'S LAYER ON A LAYER CELL: the FSLA layer a central layer cell's routing and gain build, and the fixture its arms
read.

A central layer cell (`fsla_devtools.cells.layer`, kind `layer` of the one form) states its input -- hidden states and
values drawn from a seed -- and how the layer is built to read it, in FSLA's own terms: the routing (the topology, a
write activation a level, a read tuple a read head) and the logit gain the routing projection is scaled by. `routing`
turns the cell's entries into an `fsla.Routing`; `build` makes the fixture.

**A LAYER ON A CELL IS DETERMINISTIC.** The layer's parameters are seeded from the cell's own seed immediately before
construction and the inputs come from the cell's draw, which never moves with how the layer is built.
"""
from __future__ import annotations


def routing(cell):
    """The cell's routing as an `fsla.Routing`: each entry its activation (`softmax`, `entmax<alpha>`, `tied`)."""
    from fsla_devtools.cells.form import entry

    from fsla import Routing, entmax, softmax, tied

    def activation(word):
        op, alpha = entry(word)
        return {"softmax": softmax, "tied": tied}[op]() if alpha is None else entmax(alpha)

    r = cell.routing
    return Routing(topology=r.topology, write=tuple(map(activation, r.write)),
                   read=tuple(tuple(map(activation, head)) for head in r.read))


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
    """The layer the cell builds, in the cell's dtype, its parameters seeded from the cell's seed immediately before
    construction and its routing projection scaled by the cell's logit gain."""
    import torch

    from fsla import FSLA

    cell = fx["cell"]
    torch.manual_seed(cell.seed)
    layer = FSLA(routing(cell), hidden_size=cell.hidden_size, num_heads=cell.H, group=cell.group, d_v=cell.dv,
                 decay=None, layer_idx=0).to("cuda", cell.torch_dtype).eval()
    if cell.gain != 1.0:
        with torch.no_grad():
            layer.route_W.mul_(cell.gain)
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


def build(cell) -> Fixture:
    """The fixture: the layer the cell builds (`layer`), the cell's inputs (`x`, `v`) and the routes the layer produces
    for its prefill (`routes`), each built when an arm first reads it.

    IT BUILDS NO CALL PLAN. What a subject times is the launch it names, and the packing a particular op seam does for
    it is that subject's own business.
    """
    if cell.kind != "layer":
        raise TypeError(f"{cell.name} is a {cell.kind} cell; a layer is built on a layer cell")
    return Fixture(PARTS, cell=cell)


__all__ = ["PARTS", "Fixture", "build", "routing"]
