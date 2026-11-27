# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""WHAT is timed: one lean callable per kernel this line carries.

A subject takes a REGISTRY cell, builds everything the launch does not pay for, and
returns a :class:`Launch` whose call issues exactly the launch being priced.
The roster mirrors the oracle roster one for one -- a bench roster that does not match
the correctness roster is a roster with kernels nobody measures -- and it includes the
carry. A subject that is absent from the roster while its kernel exists is the defect the
mirroring exists to catch; a subject whose kernel is unbuilt refuses by name, which is an
honest report rather than a number for something that did not run.

**TWO CELL KINDS, ONE FORM** (fsla-devtools' central cells, `fsla_devtools.cells.form.Cell`). A subject declares which
kind it takes. ``kind="carry"`` is a drawn cell (`fsla_devtools.cells.carry`): amplitudes put directly on the simplex,
with the state the sequence enters with, which is what the carry family's oracle and probe run -- read through
`measure.cells`. ``kind="layer"`` is a layer cell (`fsla_devtools.cells.layer`): an input and the routing and logit gain
its layer is built with (`measure.cells.layer`), from which the amplitudes are PRODUCED -- the only way to price the
producer's own solve or a decode step through the layer.

**EVERYTHING OUTSIDE THE LAUNCH IS BUILT ONCE, HERE.** The bundle and its descriptor:
all of it happens before the callable is returned, so a timed span wraps the launch and nothing else.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Launch:
    """One thing to time: a name and the zero-argument callable that issues it. `reset`, for a launch that changes what
    its next call reads (a carried state, a decode step), restores exactly what the first call saw, and is run untimed
    before every call, so every call does the same work on the same data. `outside_allocator`, for a launch that holds
    device memory torch's caching allocator does not see, returns those bytes now."""

    name: str
    call: Callable[[], object]
    reset: Callable[[], None] | None = None
    outside_allocator: Callable[[], int] | None = None


#: THE LEVELS A SUBJECT MAY PRICE, and what each owes (`Subject.level`). The layer subject is one stateless segment: a
#: call the layer splits into segments carries a state between them, which the prefill refuses until the carry's segment
#: boundary is built (`docs/open-work.md` C-e). The op-level paths are the stateless prefill and decode's step.
LEVELS = ("kernel", "op", "layer")


@dataclass(frozen=True)
class Subject:
    """One registered bench: what it prices, at which LEVEL, which cell kind it takes, how it builds."""

    name: str
    kind: str
    build: Callable[[dict], Launch]
    what: str
    #: THE LEVEL IT PRICES (Blake, 2026-09-15), recorded with every sample so a reading never crosses two of them:
    #: `kernel` is one family's launches and nothing around them; `op` is everything the op does over bare operands and
    #: a state -- the facts, the paging, the packing and the kernels -- as `prefill` calls it; `layer` adds the
    #: projections that produce the routing, and is the ONLY level at which this library is compared with another.
    level: str
    #: THE BODY'S OWN NAME, for `ncu -k regex:`. A PREFIX would capture whichever kernel
    #: of the family launches first, and that is not the same kernel on two binaries -- a
    #: tip whose family launches a preliminary pass ahead of the body matches that pass
    #: while its parent matches the body, and every ratio taken from the pair is
    #: meaningless. Naming the body makes the two tips capture the same kernel or capture
    #: nothing.
    symbol: str
    #: The device-side stamp entry that proves THIS binary carries the family (a path or
    #: a hash cannot: only a fact the device produces can catch a stale binary).
    stamp: str
    #: THE CALL COUNTS it is priced at: the cell's sequence as that many carried calls. One is
    #: the whole sequence in one launch set; a count above one prices what each call's fixed
    #: cost adds, and the difference between the two rows IS the per-call price.
    calls: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"subject {self.name!r} prices level {self.level!r}; a subject prices one of {LEVELS}")


# ------------------------------------------------------------------ carry cells

def prefill_op(fx) -> Launch:
    """THE PREFILL AS THE OP RUNS IT: `fsla.prefill` over the cell's bundle, stateless -- the validation, the
    carry, the intra, the glue between them and the readout. A cell the envelope refuses
    raises its refusing clause at the first call."""
    from fsla import prefill
    from measure.cells import realize

    spec = fx["cell"]
    routes, values, masks, density = bundle_of(spec, realize(spec))

    def run():
        return prefill(routes, values, masks, density)

    run()
    return Launch(name=f"prefill|{spec.name}", call=run)


def bundle_of(spec, drawn, tokens: slice = slice(None)):
    """``(routes, values, masks, density)``: the cell's draw over ``tokens`` in the format the kernels read, the
    masks and the density the library's own from the bundle's rows (``fsla.masks_and_density``)."""
    from fsla.routing import quant
    from fsla.routing.factors import RouteFactors
    from fsla.routing.masks import masks_and_density
    from fsla.routing.types import Topology

    topology = Topology.opaque(spec.routing.topology)
    #: the cell's drawn write gain folded into write level 0, as a producer folds its own (docs/CONSTITUTION.md#r23)
    read = [x[:, tokens] for x in drawn.read]
    write = [x[:, tokens] for x in drawn.write]
    write[0] = write[0] * drawn.gain[:, tokens, :, None].float()
    routes = RouteFactors.from_levels(topology, read, write)
    masks, density = masks_and_density(routes)
    values = quant.ScaledBlock.quantize(drawn.v[:, tokens], quant.VALUES)
    return routes, values, masks, density


# ------------------------------------------------------------------ layer cells

def entmax_solve(fx) -> Launch:
    """The producer's batched multi-level solve, on this cell's own routing.

    The logits are drawn once from the cell's seed and held: what is priced is the
    SOLVE, not the projection that produced them.
    """
    from fsla.routing.entmax.production import production_routing_levels

    spec, layer = fx["cell"], fx["layer"]
    layout = layer.routing.layout(layer.group)
    gen = torch.Generator(device="cuda").manual_seed(spec.seed + 900)
    shape = (spec.B * spec.H, spec.tokens, layout.packed_router_width)
    logits = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=gen)

    def run():
        return production_routing_levels(logits, layout)

    return Launch(name=f"entmax_solve|{spec.name}", call=run)


def layer_prefill(fx) -> Launch:
    """THE LAYER'S STATELESS PREFILL, one segment: the hidden states to the op's output -- the producer (every
    projection in) and the op. The output projection is the same GEMM as attention's and is outside both entries."""
    spec, layer = fx["cell"], fx["layer"]
    x = fx["x"][:, :spec.tokens].contiguous()

    def run():
        with torch.no_grad():
            return layer._prefill(x, None, None)[0]

    return Launch(name=f"layer_prefill|{spec.name}", call=run)


def layer_op(fx) -> Launch:
    """THE OP ALONE on the layer's own bundle (the producer run once, untimed): `fsla.prefill`, stateless --
    the carry, the intra, the glue and the readout. Beside `layer_prefill` it splits the
    layer into the producer and the op; beside the stage subjects, the op into its launches and its glue."""
    from fsla import prefill

    spec, layer = fx["cell"], fx["layer"]
    with torch.no_grad():
        routes, values, masks, density = layer._produce(fx["x"][:, :spec.tokens].contiguous())

    def run():
        return prefill(routes, values, masks, density)[0]

    run()
    return Launch(name=f"layer_op|{spec.name}", call=run)


def decode_step(fx) -> Launch:
    """ONE carried single-token step, on a state the seeding call populated.

    The state is MUTATED by the call, which is what a decode step is, so the launch's reset
    seeds a fresh state before every call: each timed call is the same step, at the same
    context position, over the same plane. What the step is HANDED is built once, here: the
    token's operands produced by the layer's producer (the subject prices the op, `op`, and the
    producer is the layer's), and in every reset the fresh state's per-sequence scratch, which
    its first step would otherwise allocate and zero inside the timed call.
    """
    import fsla
    from fsla.engine.decode_step import decode_call, descriptor_of, scratch
    from fsla.engine.facts import planes
    from fsla.ops.produce import produce_decode
    from fsla.routing.quant import VALUES

    spec = fx["cell"]
    if spec.decode_steps == 0:
        raise ValueError(f"{spec.name} is a prefill-only cell; it has no decode subject")
    #: The seed binds through the state's own entry and commits a zeroed plane -- a decode step's
    #: cost is a function of the routing, not of the numbers stored, so nothing has to have computed them.
    layer = fx["layer"]
    routes = fx["routes"]
    token = fx["x"][:, spec.tokens:spec.tokens + 1]
    with torch.no_grad():
        step_routes, _values, step_masks = produce_decode(token, layer.routing, layer._weights(token), group=layer.group)
        values = fsla.ScaledBlock.quantize(fx["v"][:, spec.tokens:spec.tokens + 1].contiguous(), VALUES)
    BH, device = decode_call(step_routes)
    descriptor = descriptor_of(step_routes, values, None, BH, device)
    seeded: dict = {}

    def reset():
        seeded.clear()
        state = fsla.dense_state()
        state._kernel_entry(routes, d_v=spec.dv)
        state._kernel_commit(planes.state_plane(routes, d_v=spec.dv, zeros=True))
        scratch(state, descriptor, BH, device)
        seeded.update(state=state)

    reset()

    def run():
        with torch.no_grad():
            y, _state = fsla.decode_step_gated(step_routes, values, step_masks, seeded["state"])
        return y

    return Launch(name=f"decode_step|{spec.name}", call=run, reset=reset)


#: THE ROSTER: one entry per kernel this line carries. The op-level subjects' `symbol` is the carry's, the first launch of the op
#: once it is built: no kernel of the op is built on this line, so a capture of it finds nothing and a launch refuses by name.
SUBJECTS = {
    "prefill": Subject("prefill", "carry", prefill_op,
                       "the stateless prefill as `prefill` runs it: carry, intra, glue and readout",
                       level="op", symbol="carry_kernel", stamp="csrc_build_stamp"),
    "entmax_solve": Subject("entmax_solve", "layer", entmax_solve,
                            "the producer's batched multi-level solve",
                            level="kernel", symbol="(factor|softmax)_forward_kernel",
                            stamp="csrc_build_stamp"),
    "layer_prefill": Subject("layer_prefill", "layer", layer_prefill,
                             "the layer's stateless prefill, one segment: the producer and the op, the output "
                             "projection outside it as it is outside attention's",
                             level="layer", symbol="carry_kernel", stamp="csrc_build_stamp"),
    "layer_op": Subject("layer_op", "layer", layer_op,
                        "the op alone on the layer's own bundle: den, carry, intra, glue and readout",
                        level="op", symbol="carry_kernel", stamp="csrc_build_stamp"),
    "decode_step": Subject("decode_step", "layer", decode_step,
                           "one carried single-token step through the engine over the dense plane, "
                           "as the op runs it",
                           level="op", symbol="decode_step_kernel", stamp="fsla_decode_build_stamp"),
}


def _shared_window() -> int:
    from fsla.ops.window import window_of

    return window_of()


def _splits(tokens: int, calls: int) -> bool:
    """Whether a sequence runs as `calls` carried calls: one call takes any length, and a
    chain hands its state off on window boundaries, so each of its calls is whole windows."""
    return calls == 1 or tokens % (calls * _shared_window()) == 0


def _intra_reads(widths) -> bool:
    from fsla.ops import intra as intra_ops

    return intra_ops.reads(widths)


def applicable(spec, calls: int = 1) -> list[str]:
    """Which subjects a cell can carry at a call count, by the cell's own FACTS: its kind, the counts a subject is
    priced at, the chain's whole windows, its backing, and each subject's own condition (its decode steps, its
    widths). A limit of a kernel is the kernel's to refuse, by name, when it is called: none is restated here."""
    out = []
    for name, subject in SUBJECTS.items():
        if subject.kind != spec.kind or calls not in subject.calls or not _splits(spec.tokens, calls):
            continue
        if name == "decode_step" and spec.decode_steps == 0:
            continue
        #: the state is the dense state alone (docs/api.md section 1): a paged backing has no constructor
        if spec.backing != "dense":
            continue
        #: the intra runs at widths it reads (any length: a partial last window has its own arms)
        if name == "prefill" and not _intra_reads(spec.routing.topology):
            continue
        out.append(name)
    return out
