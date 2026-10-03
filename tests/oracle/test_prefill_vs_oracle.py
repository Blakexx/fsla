# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE COMBINED OPERATOR against the canonical fp64 oracle.

THE REFERENCE IS `rola.ops.naive.naive_rola` DIRECTLY, on the bundle's own dequantized
levels and values -- the op gate compares on the same quantized inputs the kernels read --
and that is the point of this file. Each family's own conformance would pass on two different window grids; here the
two kernels run as one call and the whole answer is compared against the recurrence
itself, so a window-grid disagreement, a double-counted diagonal or a dropped tile pair
has nowhere to hide.

RED BY DESIGN (card `development/queue/C_CLEAN_SLATE.md`, "TEST-DRIVEN"): a cell whose
carry arm this line does not build yet fails, by name. The window is no longer a cell's to name -- the axis law made ``W`` a kernel constant (`rola.ops.carry.WINDOW`) and the intra family's
arm must be built at the same number, because the two kernels run ONE grid. Neither is
``(k, m)``: the box is derived at launch from the descriptor and the launch shape.

Symbols, each at first use: ``D`` = routing depth, ``B_l`` = level ``l``'s padded digit
count, ``N = prod_l B_l`` = leaf capacity, ``DV`` = the padded value width, ``L`` =
tokens, ``BH`` = batch times heads, ``W`` = the kernel's fixed window, ``modes`` = the
per-level support-mode DECLARATION the intra's zero certificate reads (a statement about
the routing, never a switch: an undeclared side reads all-ones and cannot move a number).
"""
from __future__ import annotations

import pytest
import torch

from measure.cells import carry_cells, conservative_activity, descriptor, launch, realize
from rola.engine.facts import liveness as lv
from rola.ops import carry as carry_ops
from rola.ops import facts as facts_ops
from rola.ops import intra as intra_ops
from rola.ops.paging import bytes_equal
from rola.ops.prefill import _carry_levels, prefill, readout
from rola.routing.factors import RouteFactors
from rola.routing.types import IndependentRouting, OpaqueRouting, SoftmaxActivation, Topology
from rola.values import Values
from tests.oracle.fixtures import assert_planted_errors_fail, assert_slots_close, canonical_from_plane, oracle_run, relative
from tests.oracle.tolerances import BF16_RTOL, PREFILL_READOUT, PREFILL_STATE

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

#: The registry's cells the combined operator can run: the dense backing, a fresh state,
#: the shipped value width, and widths the intra reads. A cell whose carry arm no
#: build carries yet, or whose window is partial, stays selected and red (TEST-DRIVEN).
CELLS = tuple(c for c in carry_cells(tier="oracle")
              if c.backing == "dense" and c.state == "fresh" and c.dv == 64
              and intra_ops.reads(c.widths))
IDS = [c.name for c in CELLS]


def modes_of(spec):
    """The cell's DRAW as the intra's per-level declaration.

    ``alt``/``cohort`` alternate the sparse side by level (even write-sparse, odd
    read-sparse); ``both`` is sparse on both sides at level zero; ``dense`` declares
    nothing. It is DERIVED from the draw rather than carried in the record because the
    declaration IS a statement about the draw, and two places to state one fact is the
    drift shape this family has already paid for once.
    """
    if spec.draw == "dense":
        return (intra_ops.DENSE_BOTH,) * spec.D
    if spec.draw == "both":
        return tuple(intra_ops.BOTH_SPARSE if level == 0 else intra_ops.DENSE_BOTH
                     for level in range(spec.D))
    return tuple(intra_ops.READ_SPARSE if level % 2 else intra_ops.WRITE_SPARSE
                 for level in range(spec.D))


def fresh_plane(spec, bh: int = 1):
    """the plane a fresh sequence hands out: the op allocates none."""
    from rola.ops.prefill import _descriptor

    return carry_ops.state_plane(_descriptor(list(spec.widths), spec.dv, bh), bh)


def bundle(drawn, spec):
    """The cell's draw in the format the kernels read: u8 rows, f16 values."""
    topology = Topology(levels=tuple(OpaqueRouting(width=w) for w in spec.widths))
    return (RouteFactors.from_levels(topology, drawn.read, drawn.write, drawn.gain.float()),
            Values.from_projection(drawn.v))


def run(spec, routes, values, *, state_in=None, state_out=None):
    """The facts pass and the combined operator on one bundle, a side dense where the cell declares it so."""
    layout = facts_ops.call_layout(routes)
    declared = spec.declared_sparsity()
    read_statics, write_statics = (lv.side_statics(layout, tuple(not s[side] for s in declared), spec.widths)
                                   for side in (0, 1))
    out = facts_ops.facts(routes, values, layout, read_statics, write_statics)
    return prefill(routes, values, modes=modes_of(spec), liveness=carry_ops.LivenessWords(words=out.words),
                   unions=out.unions,
                   layout=layout, read_statics=read_statics, write_statics=write_statics, state_in=state_in,
                   state_out=state_out)


def call(spec, state_in=None):
    """One combined call on the FINAL surface: the cell's bundle, no window, no ``(k, m)``, no carve."""
    routes, values = bundle(realize(spec), spec)
    return (routes, values), run(spec, routes, values, state_in=state_in,
                                 state_out=fresh_plane(spec) if state_in is None else state_in)


def oracle(routes, values, state_in=None, *, carry=False):
    """The fp64 recurrence with each slot's envelope (`tests.oracle.fixtures.oracle_run`), on the bundle's own
    dequantized levels and values -- the intra's inputs -- or, with ``carry``, the state's writers as the carry leg's
    bf16 planes hold them, the inputs the state is written from until the carry reads the rows."""
    (read, write), v, gain = routes.dequantized(), values.dequantized(), routes.g_write
    if carry:
        _, write, v = _carry_levels(routes, values)
        write, v, gain = tuple(w.bfloat16() for w in write), v.bfloat16(), gain.bfloat16()
    routing = IndependentRouting(width=1, read=SoftmaxActivation(),
                                 write=SoftmaxActivation())
    topo = Topology(levels=tuple(routing.at(level.shape[-1]) for level in read))
    return oracle_run(v.double(), tuple(r.double() for r in read), tuple(w.double() for w in write), gain.double(),
                      topo, entry=state_in)


# ------------------------------------------- the grid, and the surface it runs on

def test_the_two_kernels_run_one_window_grid():
    """``W`` IS A CONSTANT AND IT IS SHARED. The carry's window left the surface at K50;
    the intra's is still an arm axis, so what must hold is that the intra family is BUILT
    at the carry's constant. If it were not, the combined operator would have no grid and
    the cells below would be comparing two different decompositions."""
    assert intra_ops.WINDOW == carry_ops.WINDOW, (
        f"the carry's fixed window {carry_ops.WINDOW} is not the intra's {intra_ops.WINDOW}; one operator needs one "
        "grid")


def test_the_combined_operator_reaches_the_carry_launch_surface():
    """THE ONE CONTRACT ITEM THIS FILE STILL OWES (recorded at C1c): the combined
    operator's carry leg must go through `rola.ops.carry.carry_forward`.

    `rola.ops.prefill` still names the pre-C0 bindings directly, so its carry calls are
    written against a signature the extension no longer carries and against axes K50
    deleted. The cells below therefore fail at prefill's own seam rather than at the
    kernel boundary -- a different fact reported as the same red. Re-pointing those three
    call sites is what makes this file's reds the ONE reason the carry oracle tier
    already reports."""
    import inspect

    from rola.ops import prefill as prefill_mod

    body = inspect.getsource(prefill_mod)
    stale = [name for name in ("carry_forward_inter", "carry_backward_inter")
             if name in body]
    assert not stale, (
        f"rola/ops/prefill.py still calls {stale} directly; the combined operator's "
        f"carry leg is rola.ops.carry.carry_forward/carry_backward, which is the only "
        f"place the descriptor, the geometry block, the liveness words and the launch "
        f"shape are checked")


# ------------------------------------------------- the arithmetic, cell by cell

@pytest.mark.parametrize("spec", CELLS, ids=IDS)
def test_prefill_matches_the_fp64_oracle(spec):
    (routes, values), (num, den, plane) = call(spec)
    ref = oracle(routes, values)
    assert_slots_close(readout(num, den, routes), ref.y, output=PREFILL_READOUT, envelope=ref.y_envelope,
                       what=f"{spec.name} readout")
    ref = oracle(routes, values, carry=True)
    shape = (1, spec.N, spec.dv + 1)
    assert_slots_close(canonical_from_plane(plane), ref.state.reshape(shape), output=PREFILL_STATE,
                       envelope=ref.state_envelope.reshape(shape), what=f"{spec.name} state")
    del num, den, plane, ref
    torch.cuda.empty_cache()


def test_the_per_slot_rule_fails_what_a_global_max_passes():
    """The rule the cells above are graded in has to be able to fail (docs/testing.md).

    A global max grades every slot against the largest one, so a slot below the band's share of it can be anything at
    all. On the kernel's own readout: every slot under one percent of the largest ZEROED, which the global form passes
    by construction and the per-slot rule must fail, and the shared planted errors (a wiped median slot, the smallest
    slots past their allowance).
    """
    spec = next(c for c in CELLS if c.name == "flagship-dense")
    (routes, values), (num, den, _plane) = call(spec)
    ref = oracle(routes, values)
    y_ref, y_env = ref.y, ref.y_envelope
    y = readout(num, den, routes).double()
    assert_planted_errors_fail(y, y_ref, output=PREFILL_READOUT, envelope=y_env, what=f"{spec.name} readout")
    small = y_ref.abs() < 0.99 * BF16_RTOL * float(y_ref.abs().max())
    zeroed = torch.where(small, torch.zeros_like(y), y)
    assert bool(small.any()) and relative(zeroed, y_ref) < BF16_RTOL, (
        f"zeroing the small slots moved the global form to {relative(zeroed, y_ref):.3e}; the premise is that it "
        "passes this mutant")
    with pytest.raises(AssertionError, match="slots are off the oracle"):
        assert_slots_close(zeroed, y_ref, output=PREFILL_READOUT, envelope=y_env,
                           what="the readout with its small slots zeroed")


def test_a_chained_call_equals_one_long_call():
    """THE STATE IS THE WHOLE CONTRACT of a chunked prefill: two calls over the halves,
    the first's ``state_out`` seeding the second, must be the one long call. The split
    sits on a WINDOW line, which is where a chunk boundary is legal at all -- the window
    grid is global, so a split inside a window would ask the second call to carry pairs
    neither term owns.

    THE STATE IS BIT-IDENTICAL and that is the claim that matters: an owner writes its
    own leaves and nothing reduces across CTAs there. The OUTPUT is not, and cannot be:
    ``num``/``den`` are accumulated by a RED at the top, so the order the owners'
    contributions land in is the scheduler's -- for the whole call as much as for the
    chain. Agreement to fp32 rounding is what a reordered fp32 sum can promise, and it is
    far tighter than the bf16 band."""
    spec = next(c for c in CELLS if c.name == "flagship-dense")
    routes, values = bundle(realize(spec), spec)
    whole = run(spec, routes, values, state_out=fresh_plane(spec))

    cut = carry_ops.WINDOW

    def part(sl):
        return (RouteFactors(topology=routes.topology, amp_read=tuple(r[:, sl] for r in routes.amp_read),
                             amp_write=tuple(r[:, sl] for r in routes.amp_write),
                             scale_read=tuple(s[:, sl] for s in routes.scale_read),
                             scale_write=tuple(s[:, sl] for s in routes.scale_write), g_write=routes.g_write[:, sl]),
                Values(v=values.v[:, sl], scale=values.scale[:, sl]))

    head_routes, head_values = part(slice(0, cut))
    tail_routes, tail_values = part(slice(cut, None))
    head = run(spec, head_routes, head_values, state_out=fresh_plane(spec))
    tail = run(spec, tail_routes, tail_values, state_in=head[2], state_out=head[2])

    assert bytes_equal(tail[2], whole[2])
    num = torch.cat((head[0], tail[0]), dim=1)
    den = torch.cat((head[1], tail[1]), dim=1)
    assert relative(num, whole[0]) < 1e-5
    assert relative(den, whole[1]) < 1e-5
    ref = oracle(routes, values)
    assert_slots_close(readout(num, den, routes), ref.y, output=PREFILL_READOUT, envelope=ref.y_envelope,
                       what="the chained readout")


def test_the_intra_term_is_actually_present():
    """NON-VACUITY. The carry alone must NOT pass the oracle gate: if it did, every cell
    above would be testing the carry and calling it the operator. This is also the one
    cell in this file that drives the carry LAUNCH SURFACE directly, so it is where the
    combined operator's carry leg and the kernel boundary are the same call."""
    spec = next(c for c in CELLS if c.name == "flagship-dense")
    routes, values = bundle(realize(spec), spec)
    read, write, v = _carry_levels(routes, values)
    desc = descriptor(spec)
    shape = launch()
    layout = facts_ops.call_layout(routes)
    words = facts_ops.facts(routes, values, layout, *facts_ops.call_statics(routes, layout)).words
    num, den = carry_ops.carry_forward(
        carry_ops.route_planes(read, write, routes.g_write),
        v.bfloat16().permute(0, 2, 1, 3).reshape(1, spec.tokens, spec.dv).contiguous(),
        descriptor=desc, geometry=carry_ops.geometry_block(desc, shape),
        liveness=carry_ops.LivenessWords(words=words), activity=conservative_activity(desc, 1, "cuda"),
        launch=shape, state_out=carry_ops.state_plane(desc, 1))
    ref = oracle(routes, values)
    with pytest.raises(AssertionError, match="slots are off the oracle"):
        assert_slots_close(readout(num, den, routes), ref.y, output=PREFILL_READOUT, envelope=ref.y_envelope,
                           what="the inter term alone")


def test_the_combined_operator_refuses_what_it_has_no_kernel_for():
    """The refusal is STRUCTURAL, and stating it is how a gap stays a gap rather than
    becoming a host-side second code path: widths the intra does not read (off its 16-digit
    granule) are an inter-only cell, and the combined entry says so instead of half-serving it."""
    spec = next(c for c in CELLS if c.name == "flagship-dense")
    routes, values = bundle(realize(spec), spec)
    layout = facts_ops.call_layout(routes)
    statics = facts_ops.call_statics(routes, layout)
    out = facts_ops.facts(routes, values, layout, *statics)
    words = carry_ops.LivenessWords(words=out.words)
    for width in (24, 8):
        narrow = RouteFactors(topology=Topology(levels=(OpaqueRouting(width=width),) * 2),
                              amp_read=tuple(r[..., :width] for r in routes.amp_read),
                              amp_write=tuple(r[..., :width] for r in routes.amp_write),
                              scale_read=routes.scale_read, scale_write=routes.scale_write, g_write=routes.g_write)
        with pytest.raises(ValueError, match="the intra does not read level widths"):
            prefill(narrow, values, modes=modes_of(spec), liveness=words, unions=out.unions, layout=layout, read_statics=statics[0],
                    write_statics=statics[1])
