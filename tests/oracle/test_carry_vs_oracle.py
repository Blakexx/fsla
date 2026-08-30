# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CARRY KERNEL AGAINST THE fp64 REFERENCE -- the executable spec of the C3 line.

THE GATE IS THE fp64 ORACLE, NOT A SHIPPED KERNEL. There is no bit-identity target
across a redesign; the only defensible reference is the arithmetic itself, and the band
is the repository's one bf16 family bound.

THESE CELLS ARE RED BY DESIGN AND STAY RED (card `development/queue/C_CLEAN_SLATE.md`,
"TEST-DRIVEN"). The carry kernel has no implementation on this line, so every cell below
fails at the extension's boundary for that ONE reason -- not a skip, not an xfail, not a
collection error, and not a designed refusal.
`test_every_carry_cell_fails_for_the_one_reason` asserts exactly that, and it is the
file's own honesty gate: the day a cell fails for a different reason, something other
than "the kernel is not built" is wrong.

THE CELLS ARE DATA (`benchmarks/cells/carry_cells.json`), read through
`benchmarks.cells`, so the oracle tier, the probe and the benches name one set of cells
and one draw. What a record declares is a SHAPE and a DRAW -- never an arm, never a
window, never a ``(k, m)`` box, because K50's axis law left the kernel nothing else to
be told.

Symbols, each at first use: ``D`` = routing depth, ``B_l`` = level ``l``'s padded digit
count, ``N = prod_l B_l`` = leaf capacity, ``DV`` = the padded value width, ``L`` =
tokens, ``BH`` = batch times heads, ``k_tok`` = a token's nonzero digits per sparse
level (a property of the DRAW, never told to the kernel), ``W`` = the kernel's fixed
window, which the reference mirrors because the inter/intra split is defined on it.
"""
from __future__ import annotations

import re

import pytest
import torch

from benchmarks.cells import carry_cells, conservative_activity, liveness_words, realize
from rola.ops import carry as carry_ops
from rola.ops.paging import bytes_equal
from tests.oracle import reference
from tests.oracle.fixtures import canonical_from_plane, relative
from tests.oracle.tolerances import BF16_RTOL

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

#: THE ONE REASON, as `csrc/rola/rola_api.cpp`'s `kCarryUnimplemented` spells it.
NO_IMPLEMENTATION = re.compile(r"no implementation on this line")

ORACLE_CELLS = carry_cells("oracle")
IDS = [cell.name for cell in ORACLE_CELLS]


def bind(spec, bh=1, device="cuda"):
    """One cell's whole call: the descriptor, the launch and every operand it takes."""
    drawn = realize(spec, device=device)
    desc = spec.descriptor(BH=bh)
    launch = spec.launch()
    return drawn, dict(
        routes=carry_ops.route_planes(drawn.read, drawn.write, drawn.gain),
        v=drawn.v.permute(0, 2, 1, 3).reshape(bh, spec.tokens, spec.dv).contiguous(),
        descriptor=desc, geometry=carry_ops.geometry_block(desc, launch),
        liveness=liveness_words(drawn, desc),
        activity=conservative_activity(desc, bh, device), launch=launch)


def run(spec, state_in=None, state_out=None, page_table=None, bh=1):
    drawn, call = bind(spec, bh=bh)
    routes, v = call.pop("routes"), call.pop("v")
    if state_out is None:
        state_out = carry_ops.state_plane(call["descriptor"], bh)
    num, den = carry_ops.carry_forward(routes, v, state_in=state_in, state_out=state_out,
                                       page_table=page_table, **call)
    return drawn, num, den, state_out


def ref(drawn, state_in=None):
    read, write, gain, v = drawn.doubles()
    return reference.inter_reference(read, write, gain, v, drawn.spec.widths,
                                     carry_ops.WINDOW, state_in=state_in)


def check(spec, num, den, plane, want, bh=1):
    n_ref, d_ref, s_ref = want
    assert relative(num.reshape(bh, 1, spec.tokens, spec.dv).permute(0, 2, 1, 3),
                    n_ref) < BF16_RTOL
    assert relative(den.reshape(bh, 1, spec.tokens).permute(0, 2, 1), d_ref) < BF16_RTOL
    assert relative(canonical_from_plane(plane),
                    s_ref.reshape(bh, spec.N, spec.dv + 1)) < BF16_RTOL


# ------------------------------------------------------------- the one reason

@pytest.mark.parametrize("spec", ORACLE_CELLS, ids=IDS)
def test_every_carry_cell_fails_for_the_one_reason(spec):
    """THE FILE'S HONESTY GATE. Every cell in this file drives a LAWFUL call, so the only
    thing that may refuse it is the absent implementation.

    A cell that failed for a SHAPE reason would mean the registry declares something the
    kernel boundary does not admit -- a contract error, not a missing body -- and a cell
    that PASSED would mean it never reached the kernel at all."""
    with pytest.raises(RuntimeError, match=NO_IMPLEMENTATION):
        run(spec)


def test_the_registry_declares_only_lawful_shapes():
    """NON-VACUITY for the gate above: every cell's descriptor, launch shape and
    geometry block are built here, so a cell the kernel boundary would refuse fails HERE,
    where the reason is the registry, instead of as a mystery inside a call."""
    for spec in carry_cells():
        desc = spec.descriptor()
        launch = spec.launch()
        carry_ops.geometry_block(desc, launch)
        assert desc.DV in carry_ops.SHIPPED_DV
        assert launch.warps_per_cta in carry_ops.WARPS_PER_CTA


# ------------------------------------------------- the arithmetic, cell by cell

@pytest.mark.parametrize("spec", ORACLE_CELLS, ids=IDS)
def test_the_inter_term_is_the_fp64_reference(spec):
    """``num``, ``den`` AND the state plane, against the recurrence in CANONICAL order.

    The state is asserted on every cell, not only the small ones: it is the accumulator
    in the write carve's order, so a wrong permutation, a wrong swizzle or a dropped mass
    column cannot pass, and it is the only output carrying a cross-window rejoin error
    that the readout of the SAME window would hide."""
    drawn, num, den, plane = run(spec)
    check(spec, num, den, plane, ref(drawn))


def test_a_carried_state_is_advanced_in_place():
    """THE CONTINUATION: the same plane in and out, with the reference chained the same
    way. A stateful chain allocates ONE plane for the sequence."""
    spec = next(c for c in ORACLE_CELLS if c.name == "flat-small-carried")
    plane = carry_ops.state_plane(spec.descriptor(), 1)
    drawn, num, den, plane = run(spec, state_in=plane, state_out=plane)
    check(spec, num, den, plane, ref(drawn))


def test_an_idle_resident_page_is_neither_loaded_nor_stored():
    """P80b's cell: a call reaching a STRICT SUBSET of the pages a previous one wrote
    must neither load nor store the rest.

    The census is ASSERTED, not inferred: an idle page's bytes must be BIT-identical
    across the second call, and perturbing one must not move the readout. The truncated
    draw (``support`` below one) is what produces an untouched page at all -- an
    unstructured ``k_tok`` still reaches every page at this length."""
    first = next(c for c in ORACLE_CELLS if c.name == "flat-small-dense")
    second = next(c for c in ORACLE_CELLS if c.name == "flat-small-idle-resident")
    plane = carry_ops.state_plane(first.descriptor(), 1)
    run(first, state_out=plane)

    before = plane.clone()
    _, num, den, plane = run(second, state_in=plane, state_out=plane)
    moved = (before != plane).any(-1).any(-1)
    assert not bool(moved.all()), "the truncated cell reached every page the first did"

    perturbed = before.clone()
    perturbed[~moved] = -1234.5
    _, num2, den2, _ = run(second, state_in=perturbed, state_out=perturbed)
    assert torch.equal(num, num2) and torch.equal(den, den2), (
        "perturbing an idle-resident page moved the readout: the entry sweep is reading "
        "a page this call does not read")


# ------------------------------------------------- the dense / paged byte gate

def test_the_two_backings_write_the_same_bytes():
    """ONE KERNEL, ONE ABI, ONE ACCUMULATION ORDER. The dense backing is the paged one
    under the identity slot table, so a paged run whose table is a PERMUTATION of the
    slots must write, page for page, the same bytes the dense run wrote.

    A permutation rather than the identity is the point: an identity table would also
    pass for a kernel that ignored the table entirely, which is the shape this gate
    exists to exclude."""
    spec = next(c for c in ORACLE_CELLS if c.backing == "paged")
    desc = spec.descriptor()
    pages = desc.N // carry_ops.PAGE_LEAVES

    dense = carry_ops.state_plane(desc, 1)
    run(spec, state_out=dense)

    gen = torch.Generator(device="cuda").manual_seed(spec.seed)
    slots = torch.argsort(torch.rand(pages, device="cuda", generator=gen))
    paged = carry_ops.state_plane(desc, 1)
    run(spec, state_out=paged, page_table=slots.to(torch.int32).reshape(1, pages))

    assert bytes_equal(paged[0, slots], dense[0]), (
        "the paged backing wrote different bytes than the dense one at the same pages; "
        "the two backings are one kernel and one accumulation order, so a difference is "
        "an addressing bug and never a rounding one")


# --------------------------------------------------------------- the lattice

def test_lattice_is_a_permutation():
    for widths, k, m in (((64, 64), 4, 8), ((64, 64), 4, 4), ((256, 256), 4, 8)):
        pi = carry_ops.lattice_of_canonical(widths, k, m)
        n = 1
        for width in widths:
            n *= width
        assert torch.equal(torch.sort(pi).values, torch.arange(n, device=pi.device))
        _, _, _, bc, owners = carry_ops.box_shape(widths, k, m)
        assert int((pi // bc).max()) == owners - 1


def test_the_decomposition_is_exact():
    """``inter(W) + intra(W)`` is the SAME full recurrence for every ``W``, and equals
    the canonical fp64 oracle. This is the study's no-double-count claim, and the reason
    an inter-only kernel can be gated at all. It runs NO kernel, which is why it is the
    one test in this file that is green today."""
    from rola.ops.constants import READOUT_EPS
    from rola.ops.naive import naive_rola
    from rola.routing.types import IndependentRouting, SoftmaxActivation, Topology

    spec = next(c for c in ORACLE_CELLS if c.name == "flat-small-dense")
    drawn = realize(spec)
    read, write, gain, v = drawn.doubles()
    tot = None
    for W in (1, 8, 64):
        ni, di, _ = reference.inter_reference(read, write, gain, v, spec.widths, W)
        na, da = reference.intra_reference(read, write, gain, v, spec.widths, W)
        cur = (ni + na, di + da)
        if tot is None:
            tot = cur
        else:
            assert relative(cur[0], tot[0]) < 1e-12
            assert relative(cur[1], tot[1]) < 1e-12
    routing = IndependentRouting(width=1, tied=False, read=SoftmaxActivation(),
                                 write=SoftmaxActivation())
    topo = Topology(levels=tuple(routing.at(w) for w in spec.widths))
    y, _ = naive_rola(v, read, write, gain, topo, None, output_final_state=False)
    mine = tot[0] / (tot[1][..., None] + READOUT_EPS)
    assert relative(mine, y) < 1e-10
