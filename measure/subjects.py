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

**TWO CELL KINDS, ONE REGISTRY** (rola-devtools' central cells). A subject declares which kind it takes.
``kind="carry"`` is a drawn cell (`rola_devtools.cells.carry`): amplitudes put directly on the simplex, with the state
the sequence enters with, which is what the carry family's oracle and probe run -- read through `measure.cells`.
``kind="layer"`` is a layer input (`rola_devtools.cells.layer`) under one of RoLA's constructions
(`measure.cells.layer`): a producer, a routing template and a gain, from which the amplitudes are PRODUCED -- the
only way to price the producer's own solve or a decode step through the layer.

**EVERYTHING OUTSIDE THE LAUNCH IS BUILT ONCE, HERE.** The bundle, the facts pass's lists, the carve, the den pass's
units: all of it happens before the callable is returned, so a timed span wraps the launch and nothing else.
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


#: THE LEVELS A SUBJECT MAY PRICE, and what each owes (`Subject.level`). rola has no `layer` subject on this line: the
#: prefill arm (`rola.interface.rola_op`) refuses since the rebuild deleted the shipped chunk consumer, so neither the
#: op's prefill path nor a layer around it has anything to time. Decode is the one op-level path that runs.
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
    #: a state -- the facts, the paging, the packing and the kernels -- as `rola_op` calls it; `layer` adds the
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

def cell_carve(spec):
    """The carve the planner picks for a cell's declared sparsity and value width, from the record alone (no draw)."""
    from rola.engine.carve import plan_carve

    declared = spec.declared_sparsity()
    return plan_carve(tuple(spec.widths), *(tuple(level[side] for level in declared) for side in (0, 1)), spec.dv)


def read_row_of(spec):
    """The read pool row a cell's lists address: the carry arm its carve selects, on the device's card
    (`rola.ops.carry.form`)."""
    from rola.ops import carry as carry_ops

    return carry_ops.form(carry_ops.arm_key(cell_carve(spec))).read_row


def carry_operands(spec):
    """``(routes, values, facts, carve, unit)``: what the carry launch takes for a cell, from the bundle's rows and
    values, the facts pass's lists, the carve the planner picks and the den pass's units (zeros: the launch reads them,
    the pass that writes them is not timed)."""
    from measure.cells import realize
    from rola.ops import facts as facts_ops

    routes, values, layout, statics = bundle_of(spec, realize(spec))
    carve = plan_carve_of(routes, values)
    facts = facts_ops.facts(routes, values, layout, *statics, read_row_of(spec))
    unit = torch.zeros((routes.batch * routes.heads, routes.tokens), dtype=torch.int8, device="cuda")
    return routes, values, facts, carve, unit


def carry_forward(fx) -> Launch:
    """The inter term: one carry launch over a whole cell (`carry_operands`)."""
    from rola.ops import carry as carry_ops

    spec = fx["cell"]
    operands = carry_operands(spec)

    def run():
        return carry_ops.carry(*operands)

    return Launch(name=f"carry_forward|{spec.name}", call=run)


def carry_intra(fx) -> Launch:
    """THE TWO KERNELS AS ONE OPERATOR: the carry and the intra over one cell's bundle, facts the caller built.

    A KERNEL-LEVEL SUBJECT, not the op (Blake, 2026-09-15): the facts pass runs once here, untimed, as the prefill's
    contract says it is the caller's; producing the bundle, the den pass, the epilogue and the pages are not timed. At
    ``calls`` above one the sequence runs as that many calls over whole windows, the state carried between them: what a
    chunked prefill pays. RED BY PLAN until each piece is ported: the intra raises by name, and a chain refuses until the
    carry's segment boundary carries a state between calls.
    """
    from measure.cells import realize
    from rola.ops import carry as carry_ops
    from rola.ops import facts as facts_ops
    from rola.ops import intra as intra_ops

    spec = fx["cell"]
    calls = fx.get("calls", 1)
    if calls > 1:
        raise NotImplementedError(f"carry_intra at calls={calls}: the carry's segment boundary is unbuilt, so a chain "
                                  "carries no state between its calls")
    routes, values, layout, statics = bundle_of(spec, realize(spec))
    carve = plan_carve_of(routes, values)
    facts = facts_ops.facts(routes, values, layout, *statics, read_row_of(spec))
    unit = torch.zeros((routes.batch * routes.heads, routes.tokens), dtype=torch.int8, device="cuda")
    bh = routes.batch * routes.heads
    num = torch.zeros(bh, spec.tokens, intra_ops.VALUE_WIDTH, device="cuda")
    den = torch.zeros(bh, spec.tokens, device="cuda")

    def run():
        carry_ops.carry(routes, values, facts, carve, unit)
        return intra_ops.intra_forward(routes, values, facts.unions, layout, *statics, num, den)

    def reset():
        num.zero_()
        den.zero_()

    return Launch(name=f"carry_intra|{spec.name}", call=run, reset=reset)


def plan_carve_of(routes, values):
    """The carve the planner picks for a bundle (`rola.engine.carve.plan_carve`)."""
    from rola.engine.carve import plan_carve

    return plan_carve(tuple(routes.topology.padded_widths), *routes.sparse, values.v.shape[-1])


def intra_forward(fx) -> Launch:
    """The within-window term on the cell's bundle, at the window the two kernels share, over the facts pass's unions.

    THE WINDOW IS THE CARRY'S CONSTANT: the two kernels run ONE grid, so a cell whose L is not a whole number of it has
    no grid for this subject and is not applicable to it. The bundle and its unions are built once, here; the planes
    the launch reduces into are zeroed untimed before every call.
    """
    from measure.cells import realize
    from rola.ops import facts as facts_ops
    from rola.ops import intra as intra_ops

    spec = fx["cell"]
    routes, values, layout, statics = bundle_of(spec, realize(spec))
    unions = facts_ops.facts(routes, values, layout, *statics, read_row_of(spec)).unions
    bh = routes.batch * routes.heads
    num = torch.zeros(bh, spec.tokens, intra_ops.VALUE_WIDTH, device="cuda")
    den = torch.zeros(bh, spec.tokens, device="cuda")

    def run():
        return intra_ops.intra_forward(routes, values, unions, layout, *statics, num, den)

    def reset():
        num.zero_()
        den.zero_()

    return Launch(name=f"intra_forward|{spec.name}", call=run, reset=reset)


def bundle_of(spec, drawn, tokens: slice = slice(None)):
    """``(routes, values, layout, (read_statics, write_statics))``: the cell's draw over ``tokens`` in the format the
    kernels read, a side dense where the cell declares it so."""
    from rola.engine.facts import liveness as lv
    from rola.ops import facts as facts_ops
    from rola.routing.factors import RouteFactors
    from rola.routing.types import OpaqueRouting, Topology
    from rola.values import Values

    declared = spec.declared_sparsity()
    topology = Topology(levels=tuple(OpaqueRouting(width=w) for w in spec.widths))
    routes = RouteFactors.from_levels(topology, [x[:, tokens] for x in drawn.read],
                                      [x[:, tokens] for x in drawn.write], drawn.gain[:, tokens].float(),
                                      sparse=tuple(tuple(sparse[side] for sparse in declared) for side in (0, 1)))
    values = Values.from_projection(drawn.v[:, tokens])
    layout = facts_ops.call_layout(routes)
    statics = tuple(lv.side_statics(layout, tuple(not sparse[side] for sparse in declared), topology.widths)
                    for side in (0, 1))
    return routes, values, layout, statics


def facts_pass(fx) -> Launch:
    """Both sides' class-1 words, the window's columns, the write lists' exponents and each side's lists in the cell's
    carry form, from the cell's draw in the bundle's format: the pass the prefill's facts node launches, a side dense
    where the cell declares it so (its rows unread)."""
    from measure.cells import realize
    from rola.ops import facts as facts_ops

    spec = fx["cell"]
    routes, values, layout, statics = bundle_of(spec, realize(spec))
    read_row = read_row_of(spec)

    def run():
        return facts_ops.facts(routes, values, layout, *statics, read_row)

    return Launch(name=f"facts_pass|{spec.name}", call=run)


# ------------------------------------------------------------------ layer cells

def entmax_solve(fx) -> Launch:
    """The producer's batched multi-level solve, on this cell's own routing.

    The logits are drawn once from the cell's seed and held: what is priced is the
    SOLVE, not the projection that produced them.
    """
    from rola.routing.entmax.production import production_routing_factor_levels

    spec = fx["cell"]
    routing = fx["layer"].routing.resolved
    gen = torch.Generator(device="cuda").manual_seed(spec.seed + 900)
    shape = (spec.B * spec.H, spec.tokens, routing.packed_logit_width)
    read_logits = torch.randn(shape, device="cuda", dtype=torch.float32, generator=gen)
    write_logits = torch.randn(shape, device="cuda", dtype=torch.float32, generator=gen)

    def run():
        return production_routing_factor_levels(read_logits, write_logits, routing)

    return Launch(name=f"entmax_solve|{spec.name}|{fx['construction'].name}", call=run)


def decode_step(fx) -> Launch:
    """ONE carried single-token step, on a state the seeding call populated.

    The state is MUTATED by the call, which is what a decode step is, so the launch's reset
    seeds a fresh state from the prefill before every call: each timed call is the same
    step, at the same context position, over the same residency.
    """
    import rola
    from rola.engine.constants import box_leaves
    from rola.engine.dags.decode_dag import admit_growth, decode_step_gated, growth_pending
    from rola.engine.facts import liveness as lv
    from rola.ops import facts as facts_ops

    spec = fx["cell"]
    if spec.decode_steps == 0:
        raise ValueError(f"{spec.name} is a prefill-only cell; it has no decode subject")
    #: The seed binds through the facade's own entry and commits exactly the pages the
    #: routing writes -- a decode step's cost is a function of residency and routing,
    #: not of the numbers stored, so nothing has to have computed them.
    layer = fx["layer"]
    routes = fx["routes"]
    layout = facts_ops.call_layout(routes)
    statics = facts_ops.call_statics(routes, layout)
    words = facts_ops.facts(routes, rola.Values.from_projection(fx["v_prefill"]), layout, *statics,
                            read_row_of(spec)).words
    bits = lv.page_bits(words, layout, *statics)
    seeded: dict = {}

    def reset():
        seeded.clear()
        state = rola.state()
        with torch.no_grad():
            _s, arena = state._kernel_entry(routes, bits, d_v=spec.dv, paging=True, BC=box_leaves(spec.dv, 8))
            arena.wait()
        seeded.update(state=state, arena=arena)

    reset()
    token = fx["x"][:, spec.tokens:spec.tokens + 1]
    v_token = fx["v"][:, spec.tokens:spec.tokens + 1].contiguous()

    def run():
        with torch.no_grad():
            routes, values = layer._produce(token)[0], rola.Values.from_projection(v_token)
            y, state = decode_step_gated(routes, values, seeded["state"])
            if growth_pending(state):
                admit_growth(state)
                y, state = decode_step_gated(routes, values, state)
        return y

    #: THE ARENA'S PHYSICAL PAGES: under the VMM backing the driver maps them outside the caching allocator, so a memory
    #: measurement adds its own receipt; the dense bridge's one plane is a torch tensor the allocator already counts.
    return Launch(name=f"decode_step|{spec.name}|{fx['construction'].name}", call=run, reset=reset,
                  outside_allocator=lambda: seeded["arena"].committed_bytes if seeded["arena"].backing == "vmm" else 0)


#: THE ROSTER: one entry per kernel this line carries.
SUBJECTS = {
    "carry_forward": Subject("carry_forward", "carry", carry_forward,
                             "the inter term: one carry launch over a whole cell",
                             level="kernel", symbol="carry_kernel", stamp="carry_build_stamp"),
    "carry_intra": Subject("carry_intra", "carry", carry_intra,
                           "the carry and the intra as one operator, over facts the caller built: the two kernels, "
                           "never the op's facts, den pass, epilogue or paging",
                           level="kernel", symbol="carry_kernel", stamp="carry_build_stamp", calls=(1, 4)),
    "intra_forward": Subject("intra_forward", "carry", intra_forward,
                             "the within-window term at the shared window",
                             level="kernel", symbol="intra_kernel", stamp="intra_build_stamp"),
    "facts_pass": Subject("facts_pass", "carry", facts_pass,
                          "both sides' class-1 liveness words and the fold's window scales, from the bundle",
                          level="kernel", symbol="facts_pass", stamp="csrc_build_stamp"),
    "entmax_solve": Subject("entmax_solve", "layer", entmax_solve,
                            "the producer's batched multi-level solve",
                            level="kernel", symbol="(union|factor|softmax)_forward_kernel",
                            stamp="csrc_build_stamp"),
    "decode_step": Subject("decode_step", "layer", decode_step,
                           "one carried single-token step through the engine: the facts pass, the pages it commits "
                           "and the kernels, as the op runs them",
                           level="op", symbol="decode_step_kernel", stamp="rola_decode_build_stamp"),
}


def _shared_window() -> int:
    from rola.ops import carry as carry_ops

    return carry_ops.WINDOW


def _splits(tokens: int, calls: int) -> bool:
    """Whether a sequence runs as `calls` carried calls: one call takes any length, and a
    chain hands its state off on window boundaries, so each of its calls is whole windows."""
    return calls == 1 or tokens % (calls * _shared_window()) == 0


def _intra_reads(widths) -> bool:
    from rola.ops import intra as intra_ops

    return intra_ops.reads(widths)


def applicable(spec, kind: str, calls: int = 1) -> list[str]:
    """Which subjects a cell can carry at a call count: its KIND, the counts a subject is
    priced at, the chain's whole windows, and each subject's own condition."""
    out = []
    for name, subject in SUBJECTS.items():
        if subject.kind != kind or calls not in subject.calls or not _splits(spec.tokens, calls):
            continue
        if name == "decode_step" and getattr(spec, "decode_steps", 0) == 0:
            continue
        #: the intra runs its tile grid (whole windows) at widths it reads
        if name in ("intra_forward", "carry_intra") and (spec.tokens % _shared_window() or not _intra_reads(spec.widths)):
            continue
        out.append(name)
    return out
