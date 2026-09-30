"""Gate for the FACTS PASS (csrc/rola/src/facts/): its words ARE the model's, bit for bit, and its zeta is the int8
reference's, per slot.

`tests/unit/test_liveness_contract.py` gates the words' contract and the pure-torch model without a device; this file
holds the one pass to that model on the bundle's own rows -- u8 and f16, separate tensors and views of one packed plane
-- which makes every property the model proves (sparse rows are the digits' support, a dense level's rows are the
static width mask and cost no read, pad digits are dead, a token past `L` votes dead) a property of the kernel. Zeta is
held to `int_reference.fold_zeta` per window and batch-head within the fp32 rounding of its products
(docs/internals/facts/facts.md#zeta). The class-2 folds the words feed are gated here too: the exact per-page byte
against the general box fold at the atom box set and against the leaf-level truth.

Symbols: ``D`` routing depth; ``B_l`` the PADDED per-level width; ``b_l`` the LOGICAL one (the rows' own width);
``L`` tokens; ``BH`` = batch x heads; atom = 16 canonical leaves = the page granule.
"""
from __future__ import annotations

from math import prod

import pytest
import torch

from rola.engine.facts import liveness as lv
from rola.engine.facts.planes import ATOM_READ, ATOM_WRITTEN
from rola.ops import facts as facts_ops
from rola.routing.factors import RouteFactors
from rola.routing.topology import padded_level_width
from rola.routing.types import OpaqueRouting, Topology
from rola.values import Values
from tests.oracle.int_reference import fold_zeta

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the pass is a CUDA kernel")

#: (b_l per level, L): one level, two and three levels, an `L` that is not a whole number of words, a level wider
#: than a 32-digit group, and a many-window call with a tail window and a tail word.
_SHAPES = (((32,), 32), ((16, 16), 64), ((32, 16), 40), ((16, 16, 16), 96), ((64, 16), 33), ((256, 256), 1100))

#: Which levels each side declares DENSE: the pass reads neither side's dense levels.
_MODES = ((), (0,), "all")

#: zeta's per-slot bound, relative: the kernel's fp32 product of at most six factors, a rounding each.
_ZETA_RTOL = 2.0 ** -20

_FLT_MIN = torch.finfo(torch.float32).tiny


def _dtypes(widths):
    """The row formats a depth admits: u8 through depth 2, f16 at any depth."""
    return (torch.uint8, torch.float16) if len(widths) <= 2 else (torch.float16,)


def _statics(layout, dense, logical):
    levels = range(layout.D) if dense == "all" else dense
    return lv.side_statics(layout, tuple(level in levels for level in range(layout.D)), logical)


def _bundle(widths, L, statics, dtype, *, B=1, H=2, seed=0, packed=False):
    """A bundle at logical widths ``widths``: a sparse level's rows a random support (at least one live digit a token),
    a dense level's mostly nonzero with some zero digits -- a value that rounded to zero, which the width mask reports
    live all the same. ``packed`` gives each side's levels as views of one ``[B, L, H, sum b_l]`` plane, the producer's
    layout; otherwise each level is its own tensor."""
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for side in statics:
        levels = []
        for level, width in enumerate(widths):
            digits = torch.randint(1, 256, (B, L, H, width), generator=gen)
            keep = torch.rand((B, L, H, width), generator=gen) < (0.9 if side.dense_levels[level] else 0.4)
            if not side.dense_levels[level]:
                keep[..., 0] |= ~keep.any(-1)
            digits = digits * keep
            levels.append(digits.to(torch.uint8) if dtype is torch.uint8 else (digits / 255.0).to(torch.float16))
        if packed:
            plane = torch.cat(levels, -1).cuda()
            offsets = [sum(widths[:level]) for level in range(len(widths))]
            rows.append(tuple(plane[..., o:o + w] for o, w in zip(offsets, widths)))
        else:
            rows.append(tuple(level.cuda() for level in levels))

    def column():
        return (torch.rand((B, L, H), generator=gen) + 0.1).cuda()

    D = len(widths)
    routes = RouteFactors(topology=Topology(levels=tuple(OpaqueRouting(width=w) for w in widths)),
                          amp_read=rows[0], amp_write=rows[1], scale_read=tuple(column() for _ in range(D)),
                          scale_write=tuple(column() for _ in range(D)), g_write=column())
    values = Values(v_hat=torch.zeros((B, L, H, 8), dtype=torch.int8, device="cuda"), v_lo=None, scale=column())
    return routes, values


def _layout(widths, L):
    return lv.LivenessLayout(D=len(widths), B=tuple(padded_level_width(w) for w in widths), L=L)


def _model(routes, layout, read_statics, write_statics):
    return lv.liveness_words(tuple(r.cpu() for r in routes.amp_read), tuple(r.cpu() for r in routes.amp_write),
                             layout, read_statics, write_statics)


# --------------------------------------------------------------- the words are the model


@pytest.mark.parametrize(("widths", "L"), _SHAPES)
@pytest.mark.parametrize("dense", _MODES)
@pytest.mark.parametrize("packed", (False, True))
def test_the_words_are_the_model_bit_for_bit(widths, L, dense, packed):
    layout = _layout(widths, L)
    statics = _statics(layout, dense, widths)
    for dtype in _dtypes(widths):
        routes, values = _bundle(widths, L, (statics, statics), dtype, packed=packed)
        got = facts_ops.facts(routes, values, layout, statics, statics)
        assert got.words.shape == (routes.batch * routes.heads, lv.SIDES, layout.rows, layout.words)
        assert torch.equal(got.words.cpu(), _model(routes, layout, statics, statics)), dtype


@pytest.mark.parametrize(("widths", "L"), _SHAPES)
def test_the_two_sides_are_voted_independently(widths, L):
    """One launch, two sides, each with its own statics: the write side's rows are not the read side's."""
    layout = _layout(widths, L)
    read_statics, write_statics = _statics(layout, (), widths), _statics(layout, (0,), widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics), _dtypes(widths)[0])
    got = facts_ops.facts(routes, values, layout, read_statics, write_statics).words
    assert torch.equal(got.cpu(), _model(routes, layout, read_statics, write_statics))
    assert not torch.equal(got[:, lv.READ], got[:, lv.WRITE])


@pytest.mark.parametrize(("widths", "L"), _SHAPES[:5])
@pytest.mark.parametrize("dense", _MODES)
def test_a_padded_topology_is_the_model_and_its_pad_digits_are_dead(widths, L, dense):
    """Logical widths short of the padded ones: the rows are the logical width, the words the padded layout."""
    logical = tuple(max(1, width - 3) for width in widths)
    layout = _layout(widths, L)
    statics = _statics(layout, dense, logical)
    routes, values = _bundle(logical, L, (statics, statics), _dtypes(widths)[0])
    got = facts_ops.facts(routes, values, layout, statics, statics).words.cpu()
    assert torch.equal(got, _model(routes, layout, statics, statics))
    bits = lv.unpack(got, layout)
    for level in range(layout.D):
        base = layout.row_base(level)
        assert not bits[:, :, base + logical[level]:base + layout.B[level], :].any(), level


def test_a_dense_level_is_never_read():
    """BLIND the dense level's rows: a pass that read them would report it dead."""
    widths, L = (32, 16), 64
    layout = _layout(widths, L)
    statics = _statics(layout, (0,), (20, 16))
    routes, values = _bundle((20, 16), L, (statics, statics), torch.uint8)
    seen = facts_ops.facts(routes, values, layout, statics, statics).words
    for side in (routes.amp_read, routes.amp_write):
        side[0].zero_()
    blinded = facts_ops.facts(routes, values, layout, statics, statics).words
    assert torch.equal(seen, blinded)
    mask = torch.tensor(statics.digit_mask[:32], dtype=torch.bool)
    dense_rows = lv.unpack(blinded.cpu(), layout)[:, lv.READ, :32]
    assert torch.equal(dense_rows, mask[None, :, None].expand(dense_rows.shape))


def test_a_token_past_the_end_votes_dead():
    widths, L = (16, 16), 33
    layout = _layout(widths, L)
    statics = _statics(layout, (), widths)
    routes, values = _bundle(widths, L, (statics, statics), torch.uint8)
    for side in (routes.amp_read, routes.amp_write):
        for rows in side:
            rows.fill_(1)
    got = facts_ops.facts(routes, values, layout, statics, statics).words.cpu()
    assert layout.words == 2
    assert torch.equal(got[:, :, :, 1], torch.ones_like(got[:, :, :, 1]))


def test_negative_zero_is_dead():
    widths, L = (16, 16), 32
    layout = _layout(widths, L)
    statics = _statics(layout, (), widths)
    routes, values = _bundle(widths, L, (statics, statics), torch.float16, H=1)
    for side in (routes.amp_read, routes.amp_write):
        for rows in side:
            rows.fill_(-0.0)
        side[0][0, 0, 0, 5] = 1.0
    bits = lv.unpack(facts_ops.facts(routes, values, layout, statics, statics).words.cpu(), layout)
    assert bits[:, lv.READ].sum() == 1 and bits[0, lv.READ, 5, 0]


# --------------------------------------------------------------- zeta is the reference's


@pytest.mark.parametrize(("widths", "L"), _SHAPES)
def test_zeta_is_the_reference_per_window(widths, L):
    layout = _layout(widths, L)
    statics = _statics(layout, (), widths)
    for dtype in _dtypes(widths):
        routes, values = _bundle(widths, L, (statics, statics), dtype, B=2)
        got = facts_ops.facts(routes, values, layout, statics, statics).zeta.double()
        m_w = prod(scale.double() for scale in routes.scale_write)
        want = fold_zeta(routes.g_write, values.scale, m_w, facts_ops.WINDOW)
        want = want.reshape(got.shape).clamp_min(_FLT_MIN)
        assert ((got - want).abs() <= _ZETA_RTOL * want).all(), (got - want).abs().div(want).max()


def test_a_dead_window_keeps_the_floor():
    """A window whose every token carries a zero gain stores the smallest normal, a divisor the carry's fold can take."""
    widths, L = (16, 16), 64
    layout = _layout(widths, L)
    statics = _statics(layout, (), widths)
    routes, values = _bundle(widths, L, (statics, statics), torch.uint8)
    routes.g_write.zero_()
    zeta = facts_ops.facts(routes, values, layout, statics, statics).zeta
    assert torch.equal(zeta, torch.full_like(zeta, _FLT_MIN))


def test_a_bound_ledger_books_every_phase_and_changes_nothing():
    """The phase ledger (docs/internals/facts/facts.md#phase-ledger): every warp books its units' vote and store and
    the window's zeta, and a bound ledger moves no output."""
    widths, L = (256, 256), 1100
    layout = _layout(widths, L)
    statics = _statics(layout, (), widths)
    routes, values = _bundle(widths, L, (statics, statics), torch.uint8)
    plain = facts_ops.facts(routes, values, layout, statics, statics)
    ledger = torch.zeros((facts_ops.LEDGER_WARPS, len(facts_ops.LEDGER_PHASES)), dtype=torch.int64, device="cuda")
    facts_ops.ledger_bind(ledger)
    try:
        booked = facts_ops.facts(routes, values, layout, statics, statics)
    finally:
        facts_ops.ledger_bind(None)
    assert torch.equal(plain.words, booked.words) and torch.equal(plain.zeta, booked.zeta)
    assert (ledger > 0).all(), ledger


# --------------------------------------------------------------- the seam's refusals


def test_the_seam_refuses_rows_that_are_not_the_format():
    widths, L = (16, 16), 32
    layout = _layout(widths, L)
    statics = _statics(layout, (), widths)
    routes, values = _bundle(widths, L, (statics, statics), torch.uint8)
    wide = [rows.float() for rows in routes.amp_read]
    with pytest.raises(RuntimeError, match="u8 or f16"):
        facts_ops.extension().rola_facts_forward(
            wide, wide, list(routes.scale_write), routes.g_write, values.scale,
            torch.empty((2, 2, 32, 1), dtype=torch.int32, device="cuda"),
            torch.empty((2, 1, 2), device="cuda"), list(layout.B), 0, [statics.mask_word(0)], 0,
            [statics.mask_word(0)], facts_ops.WINDOW)


def test_the_seam_refuses_a_mask_past_the_rows():
    """A mask naming a digit past a level's rows would read another level's digits: refused, not guessed."""
    widths, L = (16, 16), 32
    layout = _layout(widths, L)
    wide = _statics(layout, (), widths)
    routes, values = _bundle((12, 16), L, (wide, wide), torch.uint8)
    with pytest.raises(RuntimeError, match="past its 12-digit rows"):
        facts_ops.facts(routes, values, layout, wide, wide)


def test_the_seam_refuses_another_window():
    widths, L = (16, 16), 32
    layout = _layout(widths, L)
    statics = _statics(layout, (), widths)
    routes, values = _bundle(widths, L, (statics, statics), torch.uint8)
    with pytest.raises(RuntimeError, match="window is the carry's 512"):
        facts_ops.extension().rola_facts_forward(
            list(routes.amp_read), list(routes.amp_write), list(routes.scale_write), routes.g_write, values.scale,
            torch.empty((2, 2, 32, 1), dtype=torch.int32, device="cuda"),
            torch.empty((2, 1, 2), device="cuda"), list(layout.B), 0, [statics.mask_word(0)], 0,
            [statics.mask_word(0)], 256)


# --------------------------------------------------------------- the producer's bundle, end to end


def test_the_producers_bundle_through_its_topologys_statics():
    """A real producer's bundle: the layout and statics its topology declares, the kernel against the model."""
    import rola

    torch.manual_seed(0)
    routing = rola.Routing([rola.split_routing(32, sparse_duty="write"), rola.union_routing(32)])
    layer = rola.RoLA(routing, hidden_size=64, num_heads=2, d_v=32).cuda().eval()
    with torch.no_grad():
        routes, values = layer._produce(torch.randn(2, 600, 64, device="cuda"))
    layout = facts_ops.call_layout(routes)
    read_statics, write_statics = facts_ops.call_statics(routes, layout)
    assert read_statics.dense_levels == (True, False) and write_statics.dense_levels == (False, False)
    got = facts_ops.facts(routes, values, layout, read_statics, write_statics)
    assert torch.equal(got.words.cpu(), _model(routes, layout, read_statics, write_statics))


# --------------------------------------------------------------- the folds it feeds


@pytest.mark.parametrize(("widths", "L"), _SHAPES[:5])
def test_the_page_byte_is_the_general_fold_at_the_atom_box_set(widths, L):
    """`page_bits` is not a second derivation: it is `page_activity` over `atom_boxes`."""
    layout = _layout(widths, L)
    statics = _statics(layout, (), widths)
    routes, values = _bundle(widths, L, (statics, statics), _dtypes(widths)[0])
    words = facts_ops.facts(routes, values, layout, statics, statics).words.cpu()
    boxes = lv.atom_boxes(layout)
    assert len(boxes) == layout.atoms
    want = lv.page_activity(layout, boxes, lv.box_activity(words, layout, boxes, lv.READ),
                            boxes, lv.box_activity(words, layout, boxes, lv.WRITE))
    assert torch.equal(lv.page_bits(words, layout), want)


@pytest.mark.parametrize(("widths", "L"), _SHAPES[:5])
def test_the_page_byte_is_the_leaf_level_truth_exactly(widths, L):
    """EXACT, not merely sound: at the atom grain the box->page map is a bijection, so a page carries a bit if and only
    if it holds a live leaf."""
    layout = _layout(widths, L)
    statics = _statics(layout, (), widths)
    routes, values = _bundle(widths, L, (statics, statics), _dtypes(widths)[0])
    words = facts_ops.facts(routes, values, layout, statics, statics).words.cpu()
    got = lv.page_bits(words, layout)
    bits = lv.unpack(words, layout)
    strides = layout.radix_strides
    for side, flag in ((lv.READ, ATOM_READ), (lv.WRITE, ATOM_WRITTEN)):
        truth = torch.zeros_like(got, dtype=torch.bool)
        for bh in range(bits.shape[0]):
            for token in range(layout.L):
                leaves = torch.zeros(1, dtype=torch.int64)
                for level in range(layout.D):
                    base = layout.row_base(level)
                    digits = bits[bh, side, base:base + layout.B[level], token].nonzero().flatten()
                    leaves = (leaves[:, None] + (digits * strides[level])[None, :]).reshape(-1)
                truth[bh, torch.unique(leaves // lv.ATOM_LEAVES)] = True
        assert torch.equal((got & flag).bool(), truth)
