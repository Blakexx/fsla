# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""WHAT is timed: one lean callable per kernel this line carries.

A subject takes a REGISTRY cell, builds everything the launch does not pay for, and
returns a :class:`bench.pairing.Arm` whose call issues exactly the launch being priced.
The roster mirrors the oracle roster one for one -- a bench roster that does not match
the correctness roster is a roster with kernels nobody measures -- and it includes the
carry, which has no body on this line and whose arm therefore refuses by name. A subject
that is absent from the roster while its kernel exists is the defect the mirroring
exists to catch; a subject that refuses is an honest report.

**TWO CELL KINDS, ONE REGISTRY.** A subject declares which kind it takes.
``kind="carry"`` is a drawn cell (`benchmarks/cells/carry_cells.json`): amplitudes put
directly on the simplex, which is what the carry family's oracle and probe run.
``kind="layer"`` is a constructed one (`layer_cells.json`): a producer, a routing
template and a gain, from which the amplitudes are PRODUCED -- the only way to price the
producer's own solve or a decode step through the layer.

**EVERYTHING OUTSIDE THE LAUNCH IS BUILT ONCE, HERE.** The packed sides, the support
words, the state plane, the liveness words, the routing: all of it happens before the
callable is returned, so a timed span wraps the launch and nothing else.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from bench.pairing import Arm


@dataclass(frozen=True)
class Subject:
    """One registered bench: what it prices, which cell kind it takes, how it builds."""

    name: str
    kind: str
    build: Callable[[dict], Arm]
    what: str
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


# ------------------------------------------------------------------ carry cells

def _carry_operands(fx):
    from benchmarks.cells import carry_call

    return carry_call(fx["cell"])


def carry_forward(fx) -> Arm:
    """The inter term: one carry launch over a whole cell, state advanced in place.

    The operands are the registry's own `carry_call`, so this prices the same tensors
    the oracle tier checks. THERE IS NO BODY ON THIS LINE: the call reaches the launch
    surface's refusal, which is what a bench of an unbuilt kernel must do rather than
    report a number for something that did not run.
    """
    from rola.ops import carry as carry_ops

    spec = fx["cell"]
    _drawn, call = _carry_operands(fx)
    routes, v = call.pop("routes"), call.pop("v")
    plane = carry_ops.state_plane(call["descriptor"], 1)
    #: THE STATE ARM. `fresh` binds one exit plane over a zero entry state. `continuation`
    #: passes the SAME plane in and out, which is what a growing sequence runs and the one
    #: shape where the entry sweep's loads and the exit sweep's stores address the same
    #: pages -- a different memory schedule, not a different amount of the same one.
    state_in = plane if fx.get("state_arm") == "continuation" else None

    def run():
        return carry_ops.carry_forward(routes, v, state_in=state_in, state_out=plane,
                                       **call)

    return Arm(name=f"carry_forward|{spec.name}", call=run)


def intra_forward(fx) -> Arm:
    """The within-window term on the same cell, at the window the two kernels share.

    THE WINDOW IS THE CARRY'S CONSTANT, not the intra arm's default: the two kernels run
    ONE grid, so pricing the intra at a window the carry does not run would be pricing a
    different decomposition. A cell whose L is not a whole number of that window has no
    grid for this subject and is not applicable to it.
    """
    from benchmarks.cells import realize
    from rola.ops import carry as carry_ops
    from rola.ops import intra as intra_ops

    spec = fx["cell"]
    drawn = realize(spec)
    modes = _modes_of(spec)
    gwrite = drawn.gain.permute(0, 2, 1).reshape(-1, spec.tokens).contiguous()
    v = drawn.v.contiguous()
    read, write = drawn.read, drawn.write

    def run():
        return intra_ops.intra_forward_padded(read, write, gwrite, v, modes,
                                              window=carry_ops.WINDOW)

    return Arm(name=f"intra_forward|{spec.name}", call=run)


def liveness_pass(fx) -> Arm:
    """The class-1 liveness words for both sides -- the pass every carry call reads."""
    from benchmarks.cells import realize
    from rola.engine.facts import liveness as lv
    from rola.ops.carry import pack_side
    from rola.ops.liveness import liveness_words

    spec = fx["cell"]
    drawn = realize(spec)
    read, write = pack_side(drawn.read), pack_side(drawn.write)
    layout = lv.LivenessLayout(D=spec.D, B=spec.widths, L=spec.tokens)
    statics = lv.side_statics(layout, (False,) * spec.D, layout.B)

    def run():
        return liveness_words(read, write, layout, statics, statics)

    return Arm(name=f"liveness_pass|{spec.name}", call=run)


def _modes_of(spec):
    """The cell's DRAW as the per-level support declaration.

    DERIVED from the draw rather than carried in the record, because the declaration IS
    a statement about the draw and two places to state one fact is the drift this family
    has already paid for once. The oracle tier derives it the same way.
    """
    from rola.ops import intra as intra_ops

    if spec.draw == "dense":
        return (intra_ops.DENSE_BOTH,) * spec.D
    if spec.draw == "both":
        return tuple(intra_ops.BOTH_SPARSE if level == 0 else intra_ops.DENSE_BOTH
                     for level in range(spec.D))
    return tuple(intra_ops.READ_SPARSE if level % 2 else intra_ops.WRITE_SPARSE
                 for level in range(spec.D))


# ------------------------------------------------------------------ layer cells

def entmax_solve(fx) -> Arm:
    """The producer's batched multi-level solve, on this cell's own routing.

    The logits are drawn once from the cell's seed and held: what is priced is the
    SOLVE, not the projection that produced them.
    """
    from rola.routing.entmax.production import production_routing_factor_levels

    spec = fx["cell"]
    producer = fx["layer"].routes
    routing = producer.routing
    gen = torch.Generator(device="cuda").manual_seed(spec.seed + 900)
    shape = (spec.B * spec.H, spec.tokens, routing.packed_logit_width)
    read_logits = torch.randn(shape, device="cuda", dtype=torch.float32, generator=gen)
    write_logits = torch.randn(shape, device="cuda", dtype=torch.float32, generator=gen)

    def run():
        return production_routing_factor_levels(read_logits, write_logits, routing)

    return Arm(name=f"entmax_solve|{spec.name}", call=run)


def decode_step(fx) -> Arm:
    """ONE carried single-token step, on a state the seeding call populated.

    The state is MUTATED by the call, which is what a decode step is; repeated timing
    therefore walks the sequence forward. The arena's growth saturates within the first
    few steps at these widths, so the timed reps price a steady-state step and not the
    admission that precedes it -- which the warmup launches are what guarantee.
    """
    import rola
    from rola.engine.dags.decode_dag import decode_forward
    from rola.engine.facts import planes
    from rola.ops.carry import box_leaves

    spec = fx["cell"]
    if spec.decode_steps == 0:
        raise ValueError(f"{spec.name} is a prefill-only cell; it has no decode subject")
    #: The seed binds through the facade's own entry and commits exactly the pages the
    #: routing writes -- a decode step's cost is a function of residency and routing,
    #: not of the numbers stored, so nothing has to have computed them.
    producer = fx["layer"].routes
    state = rola.state()
    with torch.no_grad():
        routes = fx["routes"]
        bits = planes.atom_bits(planes.pack_side(routes.write), spec.widths)
        _s, arena = state._kernel_entry(routes, bits, d_v=spec.dv, paging=True,
                                        BC=box_leaves(spec.dv, 8))
        arena.wait()
    token = fx["x"][:, spec.tokens:spec.tokens + 1]
    v_token = fx["v"][:, spec.tokens:spec.tokens + 1].contiguous()

    def run():
        with torch.no_grad():
            y, _ = decode_forward(producer(token), v_token, state)
        return y

    return Arm(name=f"decode_step|{spec.name}", call=run)


#: THE ROSTER: one entry per kernel this line carries, plus the carry, whose arm refuses.
SUBJECTS = {
    "carry_forward": Subject("carry_forward", "carry", carry_forward,
                             "the inter term: one carry launch over a whole cell",
                             symbol="carry_kernel", stamp="carry_build_stamp"),
    "intra_forward": Subject("intra_forward", "carry", intra_forward,
                             "the within-window term at the shared window",
                             symbol="intra_kernel", stamp="intra_build_stamp"),
    "liveness_pass": Subject("liveness_pass", "carry", liveness_pass,
                             "both sides' class-1 liveness words",
                             symbol="liveness_pass", stamp="csrc_build_stamp"),
    "entmax_solve": Subject("entmax_solve", "layer", entmax_solve,
                            "the producer's batched multi-level solve",
                            symbol="(union|factor|softmax)_forward_kernel",
                            stamp="csrc_build_stamp"),
    "decode_step": Subject("decode_step", "layer", decode_step,
                           "one carried single-token step",
                           symbol="decode_step_kernel", stamp="rola_decode_build_stamp"),
}


def _shared_window() -> int:
    from rola.ops import carry as carry_ops

    return carry_ops.WINDOW


def applicable(spec, kind: str) -> list[str]:
    """Which subjects a cell can carry: its KIND, and the decode subject's own condition."""
    out = []
    for name, subject in SUBJECTS.items():
        if subject.kind != kind:
            continue
        if name == "decode_step" and getattr(spec, "decode_steps", 0) == 0:
            continue
        if name == "intra_forward" and spec.tokens % _shared_window():
            continue
        out.append(name)
    return out
