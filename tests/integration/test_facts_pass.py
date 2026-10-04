"""Gate for the FACTS PASS (csrc/rola/src/facts/): its words ARE the model's, bit for bit, and its list exponents are
the fp16 fold reference's, per slot.

`tests/unit/test_liveness_contract.py` gates the words' contract and the pure-torch model without a device; this file
holds the one pass to that model on the bundle's own rows -- u8 and f16, separate tensors and views of one packed plane
-- at every SPARSE level, which makes every property the model proves (sparse rows are the digits' support, pad digits
are dead, a token past `L` votes dead) a property of the kernel; a dense level's rows and groups are the static width
mask, which the folds take from the statics, so the pass neither reads nor writes them. The
list exponents are held to `fold_reference.list_exponents` per window, digit and batch-head, exactly except where a
maximum sits within fp32 rounding of a power of two (docs/internals/facts/facts.md#lists). The class-2 folds the words feed are gated here too: the exact per-page byte
against the general box fold at the atom box set and against the leaf-level truth.

Symbols: ``D`` routing depth; ``B_l`` the PADDED per-level width; ``b_l`` the LOGICAL one (the rows' own width);
``L`` tokens; ``BH`` = batch x heads; atom = 16 canonical leaves = the page granule.
"""
from __future__ import annotations

import pytest
import torch

from rola.engine.facts import liveness as lv
from rola.engine.facts.planes import ATOM_READ, ATOM_WRITTEN
from rola.ops import facts as facts_ops
from rola.routing import quant
from rola.routing.factors import RouteFactors
from rola.routing.topology import padded_level_width
from rola.routing.types import OpaqueRouting, Topology
from rola.values import Values
from tests.oracle.fold_reference import exponent_bounds, list_exponents

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the pass is a CUDA kernel")

#: (b_l per level, L): one level, two and three levels, an `L` that is not a whole number of words, a level wider
#: than a 32-digit group, and a many-window call with a tail window and a tail word.
_SHAPES = (((32,), 32), ((16, 16), 64), ((32, 16), 40), ((16, 16, 16), 96), ((64, 16), 33), ((256, 256), 1100))

#: Which levels each side declares DENSE: the pass neither reads nor writes either side's dense levels.
_MODES = ((), (0,), "all")

#: the list maxima's bound, relative: the kernel's fp32 product of at most seven factors, a rounding each.
_LIST_RTOL = 2.0 ** -20


def _statics(layout, dense, logical):
    levels = range(layout.D) if dense == "all" else dense
    return lv.side_statics(layout, tuple(level in levels for level in range(layout.D)), logical)


def _bundle(widths, L, statics, *, B=1, H=2, seed=0, packed=False):
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
            levels.append(digits.to(torch.uint8))
        if packed:
            slots = [quant.row_slot(w) for w in widths]
            plane = torch.zeros((B, L, H, sum(slots)), dtype=torch.uint8)
            offsets = [sum(slots[:level]) for level in range(len(widths))]
            for o, w, level in zip(offsets, widths, levels):
                plane[..., o:o + w] = level
            plane = plane.cuda()
            rows.append(tuple(plane[..., o:o + w] for o, w in zip(offsets, widths)))
        else:
            rows.append(tuple(quant.aligned_rows(B, L, H, w, "cuda").copy_(level) for w, level in zip(widths, levels)))

    def column():
        return (torch.rand((B, L, H), generator=gen) + 0.1).cuda()

    D = len(widths)
    routes = RouteFactors(topology=Topology(levels=tuple(OpaqueRouting(width=w) for w in widths)),
                          amp_read=rows[0], amp_write=rows[1], scale_read=tuple(column() for _ in range(D)),
                          scale_write=tuple(column() for _ in range(D)), g_write=column())
    values = Values(v=torch.zeros((B, L, H, 8), dtype=torch.float16, device="cuda"), scale=column())
    return routes, values


def _layout(widths, L):
    return lv.LivenessLayout(D=len(widths), B=tuple(padded_level_width(w) for w in widths), L=L)


def _model(routes, layout, read_statics, write_statics):
    return lv.liveness_words(tuple(r.cpu() for r in routes.amp_read), tuple(r.cpu() for r in routes.amp_write),
                             layout, read_statics, write_statics)


def _assert_written(got, want, layout, read_statics, write_statics):
    """The pass's words and unions are the model's at every sparse level of each side."""
    unions = lv.tile_unions(want, layout)
    for side, statics in enumerate((read_statics, write_statics)):
        for level, dense in enumerate(statics.dense_levels):
            if dense:
                continue
            rows = slice(layout.row_base(level), layout.row_base(level) + layout.B[level])
            groups = slice(layout.group_base(level), layout.group_base(level + 1))
            assert torch.equal(got.words[:, side, rows].cpu(), want[:, side, rows]), (side, level)
            assert torch.equal(got.unions[:, side, groups].cpu(), unions[:, side, groups]), (side, level)


# --------------------------------------------------------------- the words are the model


@pytest.mark.parametrize(("widths", "L"), _SHAPES)
@pytest.mark.parametrize("dense", _MODES)
@pytest.mark.parametrize("packed", (False, True))
def test_the_words_are_the_model_bit_for_bit(widths, L, dense, packed):
    layout = _layout(widths, L)
    statics = _statics(layout, dense, widths)
    routes, values = _bundle(widths, L, (statics, statics), packed=packed)
    got = facts_ops.facts(routes, values, layout, statics, statics)
    assert got.words.shape == (routes.batch * routes.heads, lv.SIDES, layout.rows, layout.words)
    _assert_written(got, _model(routes, layout, statics, statics), layout, statics, statics)


@pytest.mark.parametrize(("widths", "L"), _SHAPES)
def test_the_two_sides_are_voted_independently(widths, L):
    """One launch, two sides, each with its own statics and rows: each side's sparse levels are its own rows' votes."""
    layout = _layout(widths, L)
    read_statics, write_statics = _statics(layout, (), widths), _statics(layout, (0,), widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    got = facts_ops.facts(routes, values, layout, read_statics, write_statics)
    _assert_written(got, _model(routes, layout, read_statics, write_statics), layout, read_statics, write_statics)


@pytest.mark.parametrize(("widths", "L"), _SHAPES[:5])
@pytest.mark.parametrize("dense", _MODES)
def test_a_padded_topology_is_the_model_and_its_pad_digits_are_dead(widths, L, dense):
    """Logical widths short of the padded ones: the rows are the logical width, the words the padded layout."""
    logical = tuple(max(1, width - 3) for width in widths)
    layout = _layout(widths, L)
    statics = _statics(layout, dense, logical)
    routes, values = _bundle(logical, L, (statics, statics))
    got = facts_ops.facts(routes, values, layout, statics, statics)
    _assert_written(got, _model(routes, layout, statics, statics), layout, statics, statics)
    bits = lv.unpack(got.words.cpu(), layout)
    for level in range(layout.D):
        base = layout.row_base(level)
        if not statics.dense_levels[level]:
            assert not bits[:, :, base + logical[level]:base + layout.B[level], :].any(), level


def test_a_dense_level_is_neither_read_nor_written():
    """BLIND the dense level's rows and POISON the outputs: a pass that read the level would vote it dead and one that
    wrote it would clear the poison, while the sparse level is the model's all the same. The dense level is the write
    lists' level, so its list unit runs and writes only its exponents."""
    widths, L = (32, 16), 64
    layout = _layout(widths, L)
    statics = _statics(layout, (0,), (20, 16))
    routes, values = _bundle((20, 16), L, (statics, statics))
    want = _model(routes, layout, statics, statics)
    for side in (routes.amp_read, routes.amp_write):
        side[0].zero_()
    BH, poison, level = routes.batch * routes.heads, 0x5A5A5A5A, facts_ops.list_level(statics)
    assert level == 0
    words = torch.full((BH, lv.SIDES, layout.rows, layout.words), poison, dtype=torch.int32, device="cuda")
    unions = torch.full((BH, lv.SIDES, layout.groups, layout.tiles), poison, dtype=torch.int32, device="cuda")
    list_exp = torch.full((BH, 1, layout.B[level], 2), 0x5A5A, dtype=torch.int16, device="cuda")
    masks = [statics.mask_word(i) for i in range(lv.mask_words(layout.rows))]
    facts_ops.extension().rola_facts_forward(
        list(routes.amp_read), list(routes.amp_write), list(routes.scale_write), routes.g_write, values.scale,
        words, unions, list_exp, list(layout.B), 1, masks, 1, masks, level,
        torch.zeros(2, dtype=torch.int32, device="cuda"), facts_ops.WINDOW)
    _assert_written(facts_ops.Facts(words, unions, list_exp), want, layout, statics, statics)
    assert (words[:, :, :layout.B[0]] == poison).all() and (unions[:, :, :layout.group_base(1)] == poison).all()
    assert (list_exp != 0x5A5A).all()


def test_a_token_past_the_end_votes_dead():
    widths, L = (16, 16), 33
    layout = _layout(widths, L)
    statics = _statics(layout, (), widths)
    routes, values = _bundle(widths, L, (statics, statics))
    for side in (routes.amp_read, routes.amp_write):
        for rows in side:
            rows.fill_(1)
    got = facts_ops.facts(routes, values, layout, statics, statics).words.cpu()
    assert layout.words == 2
    assert torch.equal(got[:, :, :, 1], torch.ones_like(got[:, :, :, 1]))


# --------------------------------------------------------------- the list exponents are the reference's


def _list_exp(routes, values, layout, read_statics, write_statics):
    """``(got, lo, hi)`` as ``[BH, windows, B_list, 2]`` int64: the pass's exponents and the reference's bounds."""
    got = facts_ops.facts(routes, values, layout, read_statics, write_statics).list_exp.long()
    level = facts_ops.list_level(read_statics)
    lo, hi = exponent_bounds(routes, values, level, facts_ops.WINDOW, layout.B[level], _LIST_RTOL)
    return got, lo.reshape(got.shape), hi.reshape(got.shape)


@pytest.mark.parametrize(("widths", "L"), [shape for shape in _SHAPES if len(shape[0]) <= 2])
@pytest.mark.parametrize("dense", _MODES)
def test_the_list_exponents_are_the_reference_per_window(widths, L, dense):
    """Per window and digit of the list level, the smallest K with every token's ``b m`` (and ``b m s``) at most
    2^K -- a dense list level read for it, its zeros at the floor -- within the fp32 rounding of the products."""
    layout = _layout(widths, L)
    statics = _statics(layout, dense, widths)
    routes, values = _bundle(widths, L, (statics, statics), B=2)
    got, lo, hi = _list_exp(routes, values, layout, statics, statics)
    assert ((got >= lo) & (got <= hi)).all(), (got - lo).abs().max()
    level = facts_ops.list_level(statics)
    want = list_exponents(routes, values, level, facts_ops.WINDOW, layout.B[level]).reshape(got.shape)
    away = lo == hi
    assert torch.equal(got[away], want[away]), "an exponent off a power of two's rounding band differs"
    assert (got > -126).any(), "every list dead: the cell is vacuous"


def test_a_dead_window_keeps_the_floor():
    """A window whose every token carries a zero gain writes nothing: every list's exponent is the floor, -126."""
    widths, L = (16, 16), 64
    layout = _layout(widths, L)
    statics = _statics(layout, (), widths)
    routes, values = _bundle(widths, L, (statics, statics))
    routes.g_write.zero_()
    list_exp = facts_ops.facts(routes, values, layout, statics, statics).list_exp
    assert torch.equal(list_exp, torch.full_like(list_exp, -126))


def test_deeper_calls_carry_no_list_level():
    """Past depth two the carry is not built: the pass writes the words and no list exponents."""
    widths, L = (16, 16, 16), 96
    layout = _layout(widths, L)
    statics = _statics(layout, (), widths)
    routes, values = _bundle(widths, L, (statics, statics))
    assert facts_ops.facts(routes, values, layout, statics, statics).list_exp.shape[2] == 0


def test_a_bound_ledger_books_every_phase_and_changes_nothing():
    """The phase ledger (docs/internals/facts/facts.md#phase-ledger): every warp books the list items' column staging
    and its units' vote and store, and a bound ledger moves no output."""
    widths, L = (256, 256), 1100
    layout = _layout(widths, L)
    statics = _statics(layout, (), widths)
    routes, values = _bundle(widths, L, (statics, statics))
    plain = facts_ops.facts(routes, values, layout, statics, statics)
    ledger = torch.zeros((facts_ops.LEDGER_WARPS, len(facts_ops.LEDGER_PHASES)), dtype=torch.int64, device="cuda")
    facts_ops.ledger_bind(ledger)
    try:
        booked = facts_ops.facts(routes, values, layout, statics, statics)
    finally:
        facts_ops.ledger_bind(None)
    assert torch.equal(plain.words, booked.words) and torch.equal(plain.list_exp, booked.list_exp)
    assert (ledger > 0).all(), ledger


# --------------------------------------------------------------- the seam's refusals


def test_the_seam_refuses_rows_that_are_not_the_format():
    widths, L = (16, 16), 32
    layout = _layout(widths, L)
    statics = _statics(layout, (), widths)
    routes, values = _bundle(widths, L, (statics, statics))
    wide = [rows.float() for rows in routes.amp_read]
    with pytest.raises(RuntimeError, match="a CUDA u8 tensor"):
        facts_ops.extension().rola_facts_forward(
            wide, wide, list(routes.scale_write), routes.g_write, values.scale,
            torch.empty((2, 2, 32, 1), dtype=torch.int32, device="cuda"),
            torch.empty((2, 2, 1, 1), dtype=torch.int32, device="cuda"), torch.empty((2, 1, 16, 2), dtype=torch.int16, device="cuda"), list(layout.B), 0,
            [statics.mask_word(0)], 0, [statics.mask_word(0)], 1,
            torch.zeros(2, dtype=torch.int32, device="cuda"), facts_ops.WINDOW)


def test_the_seam_refuses_a_mask_past_the_rows():
    """A mask naming a digit past a level's rows would read another level's digits: refused, not guessed."""
    widths, L = (16, 16), 32
    layout = _layout(widths, L)
    wide = _statics(layout, (), widths)
    routes, values = _bundle((12, 16), L, (wide, wide))
    with pytest.raises(RuntimeError, match="past its 12-digit rows"):
        facts_ops.facts(routes, values, layout, wide, wide)


def test_the_seam_refuses_another_window():
    widths, L = (16, 16), 32
    layout = _layout(widths, L)
    statics = _statics(layout, (), widths)
    routes, values = _bundle(widths, L, (statics, statics))
    with pytest.raises(RuntimeError, match="window is the carry's 512"):
        facts_ops.extension().rola_facts_forward(
            list(routes.amp_read), list(routes.amp_write), list(routes.scale_write), routes.g_write, values.scale,
            torch.empty((2, 2, 32, 1), dtype=torch.int32, device="cuda"),
            torch.empty((2, 2, 1, 1), dtype=torch.int32, device="cuda"), torch.empty((2, 1, 16, 2), dtype=torch.int16, device="cuda"), list(layout.B), 0,
            [statics.mask_word(0)], 0, [statics.mask_word(0)], 1,
            torch.zeros(2, dtype=torch.int32, device="cuda"), 256)


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
    _assert_written(got, _model(routes, layout, read_statics, write_statics), layout, read_statics, write_statics)


# --------------------------------------------------------------- the folds it feeds


@pytest.mark.parametrize(("widths", "L"), _SHAPES[:5])
def test_the_page_byte_is_the_general_fold_at_the_atom_box_set(widths, L):
    """`page_bits` is not a second derivation: it is `page_activity` over `atom_boxes`."""
    layout = _layout(widths, L)
    statics = _statics(layout, (), widths)
    routes, values = _bundle(widths, L, (statics, statics))
    words = facts_ops.facts(routes, values, layout, statics, statics).words.cpu()
    boxes = lv.atom_boxes(layout)
    assert len(boxes) == layout.atoms
    want = lv.page_activity(layout, boxes, lv.box_activity(words, layout, boxes, lv.READ, statics),
                            boxes, lv.box_activity(words, layout, boxes, lv.WRITE, statics))
    assert torch.equal(lv.page_bits(words, layout, statics, statics), want)


@pytest.mark.parametrize(("widths", "L"), _SHAPES[:5])
def test_the_page_byte_is_the_leaf_level_truth_exactly(widths, L):
    """EXACT, not merely sound: at the atom grain the box->page map is a bijection, so a page carries a bit if and only
    if it holds a live leaf."""
    layout = _layout(widths, L)
    statics = _statics(layout, (), widths)
    routes, values = _bundle(widths, L, (statics, statics))
    words = facts_ops.facts(routes, values, layout, statics, statics).words.cpu()
    got = lv.page_bits(words, layout, statics, statics)
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
