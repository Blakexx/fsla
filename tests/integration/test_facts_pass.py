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

from rola.engine.facts import lists as list_model
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

#: Which levels each side declares DENSE (the pass neither reads nor writes either side's dense levels), every side
#: with at least one (docs/api.md section 2.1): "alternating" -- read-dense on the even levels, write-dense on the odd
#: (and, at depth one, on level 0), so each side has a sparse level from depth two; "tied" -- level 0 dense on both
#: sides, the rest sparse; "all" -- every level dense on both.
_MODES = ("alternating", "tied", "all")

#: the list maxima's bound, relative: the kernel's fp32 product of at most seven factors, a rounding each.
_LIST_RTOL = 2.0 ** -20


def _flags(D, mode):
    """``(read, write)`` dense flags a level for one of `_MODES`."""
    if mode == "all":
        return (True,) * D, (True,) * D
    if mode == "tied":
        return (True,) + (False,) * (D - 1), (True,) + (False,) * (D - 1)
    return (tuple(level % 2 == 0 for level in range(D)),
            tuple(level % 2 == 1 or D == 1 for level in range(D)))


def _sides(layout, mode, logical):
    """``(read_statics, write_statics)`` of a mode."""
    return tuple(lv.side_statics(layout, flags, logical) for flags in _flags(layout.D, mode))


def _statics(layout, dense, logical):
    """One side's statics at dense levels ``dense`` (a tuple of levels)."""
    return lv.side_statics(layout, tuple(level in dense for level in range(layout.D)), logical)


def _bundle(widths, L, statics, *, B=1, H=2, seed=0, packed=False):
    """A bundle at logical widths ``widths`` in the format: a sparse level's rows a random support (at least one live
    digit a token) as u8; a dense read level's mostly nonzero u8 with some zero digits -- a value that rounded to zero,
    which the width mask reports live all the same; a dense write level's the same values as f16 at the floor
    (`rola.routing.quant`). ``packed`` gives each side's levels as views of the producer's planes -- one u8 plane, and
    on the write side an f16 plane of its dense levels -- otherwise each level is its own tensor."""
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for index, side in enumerate(statics):
        levels = []
        for level, width in enumerate(widths):
            digits = torch.randint(1, 256, (B, L, H, width), generator=gen)
            keep = torch.rand((B, L, H, width), generator=gen) < (0.9 if side.dense_levels[level] else 0.4)
            if not side.dense_levels[level]:
                keep[..., 0] |= ~keep.any(-1)
            digits = digits * keep
            if index == 1 and side.dense_levels[level]:
                levels.append((digits / quant.AMP_LEVELS).clamp_min(quant.DENSE_FLOOR).to(quant.DENSE_DTYPE))
            else:
                levels.append(digits.to(torch.uint8))
        if packed:
            out = [None] * len(widths)
            for dtype in {level.dtype for level in levels}:
                size = torch.empty((), dtype=dtype).element_size()
                members = [i for i, level in enumerate(levels) if level.dtype is dtype]
                slots = {i: quant.row_slot(widths[i] * size) // size for i in members}
                plane = torch.zeros((B, L, H, sum(slots.values())), dtype=dtype)
                offset = 0
                for i in members:
                    plane[..., offset:offset + widths[i]] = levels[i]
                    out[i] = (offset, widths[i])
                    offset += slots[i]
                plane = plane.cuda()
                for i in members:
                    o, w = out[i]
                    out[i] = plane[..., o:o + w]
            rows.append(tuple(out))
        else:
            rows.append(tuple(quant.aligned_rows(B, L, H, w, "cuda", level.dtype).copy_(level)
                              for w, level in zip(widths, levels)))

    def column():
        return (torch.rand((B, L, H), generator=gen) + 0.1).cuda()

    D = len(widths)
    routes = RouteFactors(topology=Topology(levels=tuple(OpaqueRouting(width=w) for w in widths)),
                          amp_read=rows[0], amp_write=rows[1], scale_read=tuple(column() for _ in range(D)),
                          scale_write=tuple(column() for _ in range(D)), g_write=column(),
                          sparse=tuple(tuple(not dense for dense in side.dense_levels) for side in statics))
    values = Values(v=torch.zeros((B, L, H, 8), dtype=torch.float16, device="cuda"), scale=column())
    return routes, values


def _layout(widths, L):
    return lv.LivenessLayout(D=len(widths), B=tuple(padded_level_width(w) for w in widths), L=L)


def _outputs(BH, L, windows=1):
    """The list outputs of a call that lists nothing: empty lists, and the gains planes a list level writes."""
    empty = [torch.empty(0, dtype=torch.int16, device="cuda") for _ in range(8)]
    return empty + [torch.empty((BH, L, 2), dtype=torch.float16, device="cuda"),
                    torch.empty((BH, windows, 2), dtype=torch.int16, device="cuda")]


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
    read_statics, write_statics = _sides(layout, dense, widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics), packed=packed)
    got = facts_ops.facts(routes, values, layout, read_statics, write_statics)
    assert got.words.shape == (routes.batch * routes.heads, lv.SIDES, layout.rows, layout.words)
    _assert_written(got, _model(routes, layout, read_statics, write_statics), layout, read_statics, write_statics)


@pytest.mark.parametrize(("widths", "L"), _SHAPES)
def test_the_two_sides_are_voted_independently(widths, L):
    """One launch, two sides, each with its own statics and rows: each side's sparse levels are its own rows' votes."""
    layout = _layout(widths, L)
    read_statics = _sides(layout, "alternating", widths)[0]
    write_statics = _sides(layout, "tied", widths)[1]
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    got = facts_ops.facts(routes, values, layout, read_statics, write_statics)
    _assert_written(got, _model(routes, layout, read_statics, write_statics), layout, read_statics, write_statics)


@pytest.mark.parametrize(("widths", "L"), _SHAPES[:5])
@pytest.mark.parametrize("dense", _MODES)
def test_a_padded_topology_is_the_model_and_its_pad_digits_are_dead(widths, L, dense):
    """Logical widths short of the padded ones: the rows are the logical width, the words the padded layout."""
    logical = tuple(max(1, width - 3) for width in widths)
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, dense, logical)
    routes, values = _bundle(logical, L, (read_statics, write_statics))
    got = facts_ops.facts(routes, values, layout, read_statics, write_statics)
    _assert_written(got, _model(routes, layout, read_statics, write_statics), layout, read_statics, write_statics)
    bits = lv.unpack(got.words.cpu(), layout)
    for side, statics in enumerate((read_statics, write_statics)):
        for level in range(layout.D):
            base = layout.row_base(level)
            if not statics.dense_levels[level]:
                assert not bits[:, side, base + logical[level]:base + layout.B[level], :].any(), (side, level)


def test_a_dense_level_is_neither_read_nor_written():
    """BLIND the dense level's rows and POISON the outputs: a pass that read the level would vote it dead and one that
    wrote it would clear the poison, while the sparse level is the model's all the same. The dense level is the write
    lists' level, so its list unit runs and writes only its exponents."""
    widths, L = (32, 16), 64
    layout = _layout(widths, L)
    statics = _statics(layout, (0,), (20, 16))
    read_statics = write_statics = statics
    routes, values = _bundle((20, 16), L, (read_statics, write_statics))
    want = _model(routes, layout, read_statics, write_statics)
    for side in (routes.amp_read, routes.amp_write):
        side[0].zero_()
    BH, poison, level = routes.batch * routes.heads, 0x5A5A5A5A, facts_ops.list_level(read_statics)
    assert level == 0
    words = torch.full((BH, lv.SIDES, layout.rows, layout.words), poison, dtype=torch.int32, device="cuda")
    unions = torch.full((BH, lv.SIDES, layout.groups, layout.tiles), poison, dtype=torch.int32, device="cuda")
    list_exp = torch.full((BH, 1, layout.B[level], 2), 0x5A5A, dtype=torch.int16, device="cuda")
    fm = torch.zeros((BH, layout.L, 2), dtype=torch.float32, device="cuda")
    masks = [statics.mask_word(i) for i in range(lv.mask_words(layout.rows))]
    facts_ops.extension().rola_facts_forward(
        list(routes.amp_read), list(routes.amp_write), list(routes.scale_write), routes.g_write, values.scale,
        words, unions, list_exp, fm, _outputs(BH, L), list(layout.B), 1, masks, 1, masks, level, -1, facts_ops.WINDOW)
    _assert_written(facts_ops.Facts(words, unions, list_exp, fm, *([None] * 4)), want, layout, read_statics, write_statics)
    assert (words[:, :, :layout.B[0]] == poison).all() and (unions[:, :, :layout.group_base(1)] == poison).all()
    assert (list_exp != 0x5A5A).all()


def test_a_token_past_the_end_votes_dead():
    widths, L = (16, 16), 33
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, "alternating", widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    for side in (routes.amp_read, routes.amp_write):
        for rows in side:
            rows.fill_(1)
    got = facts_ops.facts(routes, values, layout, read_statics, write_statics).words.cpu()
    assert layout.words == 2
    for side, statics in enumerate((read_statics, write_statics)):
        for level, dense in enumerate(statics.dense_levels):
            if not dense:
                rows = got[:, side, layout.row_base(level):layout.row_base(level) + layout.B[level], 1]
                assert torch.equal(rows, torch.ones_like(rows)), (side, level)


# --------------------------------------------------------------- the factor products are the declared ones


@pytest.mark.parametrize(("widths", "L"), [((32, 64), 600), ((16,), 1100)])
def test_the_factor_products_are_the_declared_fp32_ones(widths, L):
    """``fm`` holds each token's ``(F, m)`` = ``(((g sw_0) sw_1) vs, (g sw_0) sw_1)``, multiplied in fp32 in that order,
    bit for bit, at every token of every window (docs/internals/facts/facts.md#factor-products)."""
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, "alternating", widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    got = facts_ops.facts(routes, values, layout, read_statics, write_statics).fm
    m = routes.g_write.float()
    for scale in routes.scale_write:
        m = m * scale.float()
    F = m * values.scale.float()
    want = torch.stack((F, m), -1).permute(0, 2, 1, 3).reshape(routes.batch * routes.heads, L, 2)
    assert torch.equal(got, want)


# --------------------------------------------------------------- the lists and gains are the model


def _lists_want(routes, values, layout, read_statics, write_statics, got):
    """``(write, read)``: each side's lists from the model at its listed level, ``None`` where the level is dense or
    absent; the write side's coefficient pairs from the pass's own exponents (gated against the reference above)."""
    bits = lv.unpack(_model(routes, layout, read_statics, write_statics), layout)
    wl, rl = facts_ops.list_level(read_statics), facts_ops.read_list_level(read_statics)
    write = read = None
    if wl >= 0 and not write_statics.dense_levels[wl]:
        write = list_model.side_lists(bits[:, lv.WRITE], layout, wl, rows=routes.amp_write[wl],
                                      cols=list_model.columns(routes, values), list_exp=got.list_exp.cpu())
    if rl >= 0 and not read_statics.dense_levels[rl]:
        read = list_model.side_lists(bits[:, lv.READ], layout, rl, rows=None, cols=None, list_exp=None)
    return write, read


def _assert_lists(got: list_model.ListSet, want: list_model.ListSet):
    """Every defined slot of the pass's lists is the model's: the unions to their counts, the starts, each list to its
    end, and on the write side the coefficient pairs' bits."""
    count = got.count.cpu().long()
    assert torch.equal(count, want.count)
    upto = torch.arange(list_model.WINDOW) < count[..., None]
    assert torch.equal((got.union.cpu().long() & 0xFFFF)[upto], want.union[upto])
    at = got.at.cpu().long() & 0xFFFF
    assert torch.equal(at, want.at)
    end = torch.arange(list_model.LIST_CAP) < at[..., -1:]
    assert torch.equal((got.slot.cpu().long() & 0xFFFF)[end], want.slot[end])
    if want.cv is not None:
        assert torch.equal(got.cv.cpu().view(torch.int16)[end], want.cv.view(torch.int16)[end])
        assert torch.equal(got.cm.cpu().view(torch.int16)[end], want.cm.view(torch.int16)[end])


@pytest.mark.parametrize(("widths", "L"), [shape for shape in _SHAPES if len(shape[0]) <= 2])
@pytest.mark.parametrize("dense", _MODES)
def test_the_lists_are_the_model_bit_for_bit(widths, L, dense):
    """Each side's lists at its listed level (docs/internals/facts/facts.md#lists-out): the unions ranked into slots,
    each digit's entries round-robin by slot residue within its passes, the starts, and the write entries' coefficient
    pairs -- bit for bit, at every window."""
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, dense, widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    got = facts_ops.facts(routes, values, layout, read_statics, write_statics)
    write, read = _lists_want(routes, values, layout, read_statics, write_statics, got)
    for side, want in ((got.write, write), (got.read, read)):
        if want is None:
            assert side.slot.shape[2] == 0
        else:
            _assert_lists(side, want)


def test_the_lists_cover_many_passes():
    """A window whose groups' unions span every pass: the passes split, each its own round robin."""
    widths, L = (64, 64), 1024
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, "alternating", widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    got = facts_ops.facts(routes, values, layout, read_statics, write_statics)
    write, read = _lists_want(routes, values, layout, read_statics, write_statics, got)
    assert (write.count > 3 * list_model.PASS_SLOTS).any(), "no union spans four passes: the cell is vacuous"
    _assert_lists(got.write, write)
    _assert_lists(got.read, read)


@pytest.mark.parametrize(("widths", "L"), [((32, 64), 600), ((16,), 1100)])
def test_the_gains_are_the_model_bit_for_bit(widths, L):
    """Each token's ``(m, m s)`` as f16 at its window's powers of two, and those powers (facts.md#gains)."""
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, "alternating", widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    got = facts_ops.facts(routes, values, layout, read_statics, write_statics)
    want16, want_exp = list_model.gains(routes, values, L)
    assert torch.equal(got.gains_exp.cpu().long(), want_exp.long())
    assert torch.equal(got.gains16.cpu().view(torch.int16), want16.view(torch.int16))


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
    read_statics, write_statics = _sides(layout, dense, widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics), B=2)
    got, lo, hi = _list_exp(routes, values, layout, read_statics, write_statics)
    assert ((got >= lo) & (got <= hi)).all(), (got - lo).abs().max()
    level = facts_ops.list_level(read_statics)
    want = list_exponents(routes, values, level, facts_ops.WINDOW, layout.B[level]).reshape(got.shape)
    away = lo == hi
    assert torch.equal(got[away], want[away]), "an exponent off a power of two's rounding band differs"
    assert (got > -126).any(), "every list dead: the cell is vacuous"


def test_a_dead_window_keeps_the_floor():
    """A window whose every token carries a zero gain writes nothing: every list's exponent is the floor, -126."""
    widths, L = (16, 16), 64
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, "alternating", widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    routes.g_write.zero_()
    list_exp = facts_ops.facts(routes, values, layout, read_statics, write_statics).list_exp
    assert torch.equal(list_exp, torch.full_like(list_exp, -126))


def test_deeper_calls_carry_no_list_level():
    """Past depth two the carry is not built: the pass writes the words and no list exponents."""
    widths, L = (16, 16, 16), 96
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, "alternating", widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    assert facts_ops.facts(routes, values, layout, read_statics, write_statics).list_exp.shape[2] == 0


def test_a_bound_ledger_books_every_phase_and_changes_nothing():
    """The phase ledger (docs/internals/facts/facts.md#phase-ledger): every warp books the list items' column staging
    and its units' vote and store, and a bound ledger moves no output."""
    widths, L = (256, 256), 1100
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, "alternating", widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    plain = facts_ops.facts(routes, values, layout, read_statics, write_statics)
    ledger = torch.zeros((facts_ops.LEDGER_WARPS, len(facts_ops.LEDGER_PHASES)), dtype=torch.int64, device="cuda")
    facts_ops.ledger_bind(ledger)
    try:
        booked = facts_ops.facts(routes, values, layout, read_statics, write_statics)
    finally:
        facts_ops.ledger_bind(None)
    want = _model(routes, layout, read_statics, write_statics)
    _assert_written(plain, want, layout, read_statics, write_statics)
    _assert_written(booked, want, layout, read_statics, write_statics)
    assert torch.equal(plain.list_exp, booked.list_exp)
    assert (ledger > 0).all(), ledger


# --------------------------------------------------------------- the seam's refusals


def test_the_seam_refuses_rows_that_are_not_the_format():
    widths, L = (16, 16), 32
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, "alternating", widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    wide = [rows.float() for rows in routes.amp_read]
    with pytest.raises(RuntimeError, match="a CUDA u8 tensor"):
        facts_ops.extension().rola_facts_forward(
            wide, wide, list(routes.scale_write), routes.g_write, values.scale,
            torch.empty((2, 2, 32, 1), dtype=torch.int32, device="cuda"),
            torch.empty((2, 2, 1, 1), dtype=torch.int32, device="cuda"), torch.empty((2, 1, 16, 2), dtype=torch.int16, device="cuda"),
            torch.empty((2, L, 2), dtype=torch.float32, device="cuda"), _outputs(2, L), list(layout.B), 0,
            [read_statics.mask_word(0)], 0, [write_statics.mask_word(0)], 1, -1, facts_ops.WINDOW)


def test_the_seam_refuses_a_mask_past_the_rows():
    """A mask naming a digit past a level's rows would read another level's digits: refused, not guessed."""
    widths, L = (16, 16), 32
    layout = _layout(widths, L)
    wide = _sides(layout, "alternating", widths)
    routes, values = _bundle((12, 16), L, wide)
    with pytest.raises(RuntimeError, match="past its 12-digit rows"):
        facts_ops.facts(routes, values, layout, *wide)


def test_the_seam_refuses_another_window():
    widths, L = (16, 16), 32
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, "alternating", widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    with pytest.raises(RuntimeError, match="window is the carry's 512"):
        facts_ops.extension().rola_facts_forward(
            list(routes.amp_read), list(routes.amp_write), list(routes.scale_write), routes.g_write, values.scale,
            torch.empty((2, 2, 32, 1), dtype=torch.int32, device="cuda"),
            torch.empty((2, 2, 1, 1), dtype=torch.int32, device="cuda"), torch.empty((2, 1, 16, 2), dtype=torch.int16, device="cuda"),
            torch.empty((2, L, 2), dtype=torch.float32, device="cuda"), _outputs(2, L), list(layout.B), 0,
            [read_statics.mask_word(0)], 0, [write_statics.mask_word(0)], 1, -1, 256)


# --------------------------------------------------------------- the producer's bundle, end to end


def test_the_producers_bundle_through_its_topologys_statics():
    """A real producer's bundle: the layout and statics its topology declares, the kernel against the model."""
    import rola

    torch.manual_seed(0)
    routing = rola.Routing([rola.split_routing(32, sparse_duty="write"), rola.split_routing(32, sparse_duty="read")])
    layer = rola.RoLA(routing, hidden_size=64, num_heads=2, d_v=32).cuda().eval()
    with torch.no_grad():
        routes, values = layer._produce(torch.randn(2, 600, 64, device="cuda"))
    layout = facts_ops.call_layout(routes)
    read_statics, write_statics = facts_ops.call_statics(routes, layout)
    assert read_statics.dense_levels == (True, False) and write_statics.dense_levels == (False, True)
    got = facts_ops.facts(routes, values, layout, read_statics, write_statics)
    _assert_written(got, _model(routes, layout, read_statics, write_statics), layout, read_statics, write_statics)


# --------------------------------------------------------------- the folds it feeds


@pytest.mark.parametrize(("widths", "L"), _SHAPES[:5])
def test_the_page_byte_is_the_general_fold_at_the_atom_box_set(widths, L):
    """`page_bits` is not a second derivation: it is `page_activity` over `atom_boxes`."""
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, "alternating", widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    words = facts_ops.facts(routes, values, layout, read_statics, write_statics).words.cpu()
    boxes = lv.atom_boxes(layout)
    assert len(boxes) == layout.atoms
    want = lv.page_activity(layout, boxes, lv.box_activity(words, layout, boxes, lv.READ, read_statics),
                            boxes, lv.box_activity(words, layout, boxes, lv.WRITE, write_statics))
    assert torch.equal(lv.page_bits(words, layout, read_statics, write_statics), want)


@pytest.mark.parametrize(("widths", "L"), _SHAPES[:5])
def test_the_page_byte_is_the_leaf_level_truth_exactly(widths, L):
    """EXACT, not merely sound: at the atom grain the box->page map is a bijection, so a page carries a bit if and only
    if it holds a live leaf."""
    layout = _layout(widths, L)
    read_statics, write_statics = _sides(layout, "alternating", widths)
    routes, values = _bundle(widths, L, (read_statics, write_statics))
    words = facts_ops.facts(routes, values, layout, read_statics, write_statics).words.cpu()
    got = lv.page_bits(words, layout, read_statics, write_statics)
    bits = lv.unpack(words, layout)
    strides = layout.radix_strides
    for side, flag in ((lv.READ, ATOM_READ), (lv.WRITE, ATOM_WRITTEN)):
        truth = torch.zeros_like(got, dtype=torch.bool)
        for bh in range(bits.shape[0]):
            for token in range(layout.L):
                leaves = torch.zeros(1, dtype=torch.int64)
                for level in range(layout.D):
                    base = layout.row_base(level)
                    if (read_statics, write_statics)[side].dense_levels[level]:
                        digits = torch.arange(layout.B[level])
                    else:
                        digits = bits[bh, side, base:base + layout.B[level], token].nonzero().flatten()
                    leaves = (leaves[:, None] + (digits * strides[level])[None, :]).reshape(-1)
                truth[bh, torch.unique(leaves // lv.ATOM_LEAVES)] = True
        assert torch.equal((got & flag).bool(), truth)
