"""The sparse grid computes what the full grid computes.

The persistent launch skips blocks. Two claims have to hold for that to be an
optimization rather than a semantic change, and they are gated differently
because the operator's own arithmetic is not reproducible against itself:

**`s_out` IS BIT-IDENTITY, and it is the real gate.** Each atom of the state
plane is stored by exactly one CTA through a plain store -- no atomics, no
ordering freedom -- so the two launches must agree byte for byte on every
backing and every call class. Any difference is a bug.

**`y` and `den` ARE NOT.** The per-block fan-in is a global `atomicAdd`, so the
summation order is the hardware scheduler's and the FULL grid is not bit-identical
against ITSELF run to run. A bit-identity gate on `y` would therefore be a gate on
the scheduler. Two gates replace it: a DETERMINISTIC cell where `y` has exactly one
contributor per token and bit-identity IS available, and a distribution gate on the
general cell whose bound is the full grid's own measured run-to-run spread.

Vocabulary, defined here and used unqualified below: `D` = level count; `B` = one
uniform level width; `N = B ** D` = leaves; `BC` = the launch's block width in
leaves; `blocks = N // BC`; `C` = the chunk (32 tokens); `L`/`T` = tokens;
`BH` = batch * heads; `bh` = the (batch, head) index; **atom** = the 16-leaf page
granule; `rho` = the live-block fraction; `y` = the operator's output;
`den` = the global denominator plane.

**The structural claim carries the exactness, not the numeric gate.** A skipped
block's contribution is the exact additive identity: its accumulators are
initialised to `+0.0`, are only ever written inside the liveness gates, and
`x + (+0.0)` is the identity in round-to-nearest for every `x` a zero-initialised
accumulator can hold. `test_a_single_live_block_is_bit_identical` gates that claim
directly, where the scheduler cannot interfere with it.
"""
from __future__ import annotations

import pytest
import torch

from rola.ops._ext import extension

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the launch is a CUDA kernel"),
]

_C, _DV = 32, 64
_WIDTHS = [64, 64]          # D = 2, B = 64 -> N = 4096
_N = 64 * 64
_ATOM = 16
_COLS = _DV + 1

#: PRE-REGISTERED, before any run: the relative bound the distribution gate falls
#: back to when the full grid's own spread reads exactly zero on the device under
#: test. A zero spread is a SCHEDULING ARTIFACT, not a contract -- the persistent
#: grid changes the schedule by construction -- so it must never be promoted into
#: a bit-identity gate. The value is the relative spread the full grid measured on
#: sm_86 (workflows/p68-sparse-grid/s0_measurement.md 3.4: 4.1e-07 at L=64,
#: 4.6e-07 at L=256), rounded up.
_SIGMA_FLOOR_REL = 1e-06

#: The runs the distribution gate establishes its instrument from.
_SIGMA_RUNS = 20


def _routing(bh, L, live_blocks, bc, seed, *, split_sides=False):
    """`(pr, pw, gw, v)` whose level-0 support makes exactly `live_blocks` blocks live.

    At `D = 2, B = 64` leaf `s = d0 * 64 + d1` and `block = d0 // (BC // 64)`, so
    restricting the level-0 digits restricts the blocks. `split_sides` gives the
    read and write sides DISJOINT level-0 supports, which is the cell where the
    stateless predicate's `R && W` term is the one doing the skipping.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    per_block = max(1, bc // _WIDTHS[1])
    d0 = min(_WIDTHS[0], live_blocks * per_block)
    pr = torch.zeros(1, bh, L, sum(_WIDTHS), device="cuda")
    pw = torch.zeros_like(pr)
    for side, lo in ((pr, 0), (pw, d0 if split_sides else 0)):
        hi = min(_WIDTHS[0], lo + d0)
        a0 = torch.rand(bh, L, hi - lo, device="cuda", generator=g) + 0.5
        a1 = torch.rand(bh, L, _WIDTHS[1], device="cuda", generator=g) + 0.5
        side[0, :, :, lo:hi] = a0 / a0.sum(-1, keepdim=True)
        side[0, :, :, _WIDTHS[0]:] = a1 / a1.sum(-1, keepdim=True)
    gw = torch.rand(1, bh, L, device="cuda", generator=g) + 0.5
    v = torch.randn(1, bh, L, _DV, device="cuda", generator=g)
    return (pr.to(torch.bfloat16).contiguous(), pw.to(torch.bfloat16).contiguous(),
            gw.to(torch.bfloat16).contiguous(), v.to(torch.bfloat16).contiguous())


def _bits(pr, pw, v, bc, L, page_tbl=None):
    m = extension()
    tbl = m.chunk_union_table(pr, pw, v, _C, bc, _WIDTHS)
    return tbl, m.chunk_block_bits(tbl, bc, _WIDTHS, L, page_tbl)


def _launch(pr, pw, gw, v, tbl, bc, *, bits=None, s_in=None, s_out=None, page_tbl=None):
    return extension().chunk_forward_raw(pr, pw, gw, v, tbl, _C, bc, _WIDTHS, s_in, s_out,
                                         page_tbl, bits)


def _assert_it_skips(bits):
    """NON-VACUITY. An identity that holds because nothing was skipped is not
    evidence about skipping; every cell that claims to exercise the predicate says
    how much of the grid it actually removed."""
    dead = int((bits == 0).sum())
    assert 2 * dead >= bits.numel(), (
        f"only {dead} of {bits.numel()} blocks are dead; this cell does not "
        f"exercise the skip and its identity claim is vacuous")


def _dense_plane(bh, poison):
    """A dense `[BH, N/16, 16, d_v+1]` plane, POISONED.

    `torch.empty` is what production allocates, and an uninitialised plane cannot
    tell "written with the right value" from "not written at all". A poison value
    can: any survivor is a block the launch failed to store.
    """
    return torch.full((bh, _N // _ATOM, _ATOM, _COLS), poison, device="cuda",
                      dtype=torch.float32)


def _paged(pw, bh):
    """`(page_table, plane)` for a PAGED backing over exactly this call's write set.

    The plan's bitmap is the write set exactly (that is P67's guarantee, and what
    makes skipping an unmapped atom exact rather than defensive), so the arena is
    modelled here by admitting precisely the atoms `chunk_stats` names and zeroing
    them, which is what admission does.
    """
    atoms = extension().chunk_stats(pw, _WIDTHS).bool()
    slot = torch.full((bh, _N // _ATOM), -1, device="cuda", dtype=torch.int32)
    n = int(atoms.sum())
    slot[atoms] = torch.arange(n, device="cuda", dtype=torch.int32)
    return slot.contiguous(), torch.zeros((max(n, 1), _ATOM, _COLS), device="cuda",
                                          dtype=torch.float32)


# ---------------------------------------------------------------------------
# s_out: strict bit-identity, every call class and both backings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bc", [64, 128])
@pytest.mark.parametrize("live", [1, 4, 16])
def test_the_final_state_is_bit_identical_dense(bc, live):
    """Dense `s_out`, poisoned: the plane must be written WHOLE and identically.

    The dense backing's predicate skips nothing, precisely because this plane is
    written whole by construction. Poisoning it makes that a fact the test can
    see rather than an argument: a survivor is a block whose atoms nobody stored.
    """
    bh, L = 4, 64
    pr, pw, gw, v = _routing(bh, L, live, bc, seed=live * 31 + bc)
    tbl, bits = _bits(pr, pw, v, bc, L)
    full, pers = _dense_plane(bh, float("nan")), _dense_plane(bh, float("nan"))
    _launch(pr, pw, gw, v, tbl, bc, s_out=full)
    _launch(pr, pw, gw, v, tbl, bc, bits=bits, s_out=pers)
    assert not full.isnan().any(), "the FULL grid left the dense plane partly unwritten"
    assert not pers.isnan().any(), (
        "the persistent grid left poison in the dense state plane: a block was "
        "skipped whose atoms it owed a store")
    assert torch.equal(full, pers)


@pytest.mark.parametrize("bc", [64, 128])
@pytest.mark.parametrize("live", [1, 4, 16])
def test_the_final_state_is_bit_identical_paged(bc, live):
    """Paged `s_out`: an unwritten block must leave its slot UNCHANGED, which is
    what skipping does -- so the two launches agree byte for byte."""
    bh, L = 4, 64
    pr, pw, gw, v = _routing(bh, L, live, bc, seed=live * 17 + bc + 1)
    page_tbl, full = _paged(pw, bh)
    pers = full.clone()
    tbl, bits = _bits(pr, pw, v, bc, L, page_tbl)
    _assert_it_skips(bits)
    _launch(pr, pw, gw, v, tbl, bc, s_out=full, page_tbl=page_tbl)
    _launch(pr, pw, gw, v, tbl, bc, bits=bits, s_out=pers, page_tbl=page_tbl)
    assert torch.equal(full, pers)


@pytest.mark.parametrize("bc", [64, 128])
def test_the_final_state_is_bit_identical_on_a_continuation(bc):
    """The state-in call class: a carried state makes a read-live block that
    writes nothing NO LONGER INERT, so the predicate needs its third bit. This is
    the cell that catches a predicate which forgot it."""
    bh, L, live = 4, 64, 4
    pr, pw, gw, v = _routing(bh, L, live, bc, seed=bc + 7)
    page_tbl, s_in = _paged(pw, bh)
    #: a carried state that is NOT zero, so reading it is observable.
    s_in.normal_(generator=torch.Generator(device="cuda").manual_seed(3))
    tbl, bits = _bits(pr, pw, v, bc, L, page_tbl)
    full, pers = s_in.clone(), s_in.clone()
    n_full, d_full = _launch(pr, pw, gw, v, tbl, bc, s_in=s_in, s_out=full, page_tbl=page_tbl)
    n_pers, d_pers = _launch(pr, pw, gw, v, tbl, bc, bits=bits, s_in=s_in, s_out=pers,
                             page_tbl=page_tbl)
    assert torch.equal(full, pers)
    _assert_within_the_full_grids_spread(
        n_full, d_full, n_pers, d_pers,
        lambda: _launch(pr, pw, gw, v, tbl, bc, s_in=s_in, s_out=s_in.clone(),
                        page_tbl=page_tbl))


# ---------------------------------------------------------------------------
# y: the deterministic cell, then the distribution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bc", [64, 128])
def test_a_single_live_block_is_bit_identical(bc):
    """THE DETERMINISTIC CELL, and it gates the exactness argument directly.

    With one live block every token has exactly ONE contributor to `o` and to
    `den`, so the fan-in's `atomicAdd` has nothing to reorder and `y` IS
    reproducible. Every other block is dead and contributes the additive identity;
    if that claim were merely approximate rather than exact, this is where it
    would show as a differing bit.
    """
    bh, L = 2, 64
    pr, pw, gw, v = _routing(bh, L, 1, bc, seed=bc)
    tbl, bits = _bits(pr, pw, v, bc, L)
    assert int((bits & 3).eq(3).sum(-1).max()) == 1, "the cell is not single-live-block"
    n_full, d_full = _launch(pr, pw, gw, v, tbl, bc)
    n_pers, d_pers = _launch(pr, pw, gw, v, tbl, bc, bits=bits)
    assert torch.equal(n_full, n_pers), "the numerator differs on a one-contributor cell"
    assert torch.equal(d_full, d_pers), "the denominator differs on a one-contributor cell"


def _assert_within_the_full_grids_spread(n_full, d_full, n_pers, d_pers, again):
    """The distribution gate: the instrument is established, then applied.

    The bound is the FULL grid's own run-to-run spread on identical inputs -- the
    only honest one, since the persistent grid changes a schedule that the full
    grid does not reproduce against itself either. `_SIGMA_FLOOR_REL` is the
    pre-registered fallback for a device on which that spread reads zero, and is
    NOT a licence to be looser than the instrument: the bound is the larger of the
    two, and a zero spread is recorded rather than promoted to bit-identity.
    """
    spread_n = spread_d = 0.0
    for _ in range(_SIGMA_RUNS):
        n, d = again()
        spread_n = max(spread_n, float((n - n_full).abs().max()))
        spread_d = max(spread_d, float((d - d_full).abs().max()))
    scale_n = float(n_full.abs().max())
    scale_d = float(d_full.abs().max())
    bound_n = max(spread_n, _SIGMA_FLOOR_REL * scale_n)
    bound_d = max(spread_d, _SIGMA_FLOOR_REL * scale_d)
    got_n = float((n_pers - n_full).abs().max())
    got_d = float((d_pers - d_full).abs().max())
    assert got_n <= bound_n, (
        f"the numerator moved {got_n:.3e}, outside the full grid's own spread "
        f"{spread_n:.3e} (bound {bound_n:.3e})")
    assert got_d <= bound_d, (
        f"the denominator moved {got_d:.3e}, outside the full grid's own spread "
        f"{spread_d:.3e} (bound {bound_d:.3e})")


@pytest.mark.parametrize("bc", [64, 128])
@pytest.mark.parametrize("live,split", [(4, False), (16, False), (8, True), (64, False)])
def test_y_stays_inside_the_full_grids_own_spread(bc, live, split):
    """The general cell, across densities and with the sides split.

    `split` gives the two sides disjoint level-0 supports, so blocks are read-live
    or write-live but rarely both -- the cell where the stateless predicate's
    `R && W` term skips blocks a write-side-only bitmap could not.
    """
    bh, L = 4, 128
    live = min(live, _N // bc)
    pr, pw, gw, v = _routing(bh, L, live, bc, seed=live * 5 + bc + split, split_sides=split)
    tbl, bits = _bits(pr, pw, v, bc, L)
    if live * 2 < _N // bc:
        _assert_it_skips(bits)
    n_full, d_full = _launch(pr, pw, gw, v, tbl, bc)
    n_pers, d_pers = _launch(pr, pw, gw, v, tbl, bc, bits=bits)
    _assert_within_the_full_grids_spread(n_full, d_full, n_pers, d_pers,
                                         lambda: _launch(pr, pw, gw, v, tbl, bc))


# ---------------------------------------------------------------------------
# the gates' teeth
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bc", [64, 128])
def test_a_bitmap_that_skips_everything_is_caught_by_every_gate(bc):
    """THE PLANTED VIOLATION. A gate that cannot be shown failing is decoration.

    An all-dead bitmap is the maximal skip: the work loop then runs no block at
    all. `y`, `den` and the dense state plane must each move far outside what the
    gates above allow -- if any of them did not, that gate was passing on
    something other than the property it names.
    """
    bh, L = 4, 64
    pr, pw, gw, v = _routing(bh, L, 8, bc, seed=bc + 11)
    tbl, bits = _bits(pr, pw, v, bc, L)
    dead = torch.zeros_like(bits)
    n_full, d_full = _launch(pr, pw, gw, v, tbl, bc)
    n_none, d_none = _launch(pr, pw, gw, v, tbl, bc, bits=dead)
    assert float(n_full.abs().max()) > 0 and float(d_full.abs().max()) > 0
    assert not torch.equal(n_full, n_none) and not torch.equal(d_full, d_none)
    assert float((n_none - n_full).abs().max()) > _SIGMA_FLOOR_REL * float(n_full.abs().max())
    #: the state plane's own teeth, and they belong on the PAGED backing: the
    #: dense predicate is `always run`, so a bitmap cannot defeat that gate at all
    #: -- which is the property, not a hole in the test.
    page_tbl, full_plane = _paged(pw, bh)
    none_plane = full_plane.clone()
    _launch(pr, pw, gw, v, tbl, bc, s_out=full_plane, page_tbl=page_tbl)
    _launch(pr, pw, gw, v, tbl, bc, bits=dead, s_out=none_plane, page_tbl=page_tbl)
    assert not torch.equal(full_plane, none_plane), (
        "a launch that ran no block wrote the same paged state as one that ran "
        "every block: the bit-identity gate would pass on an unwritten plane")
    dense = _dense_plane(bh, float("nan"))
    _launch(pr, pw, gw, v, tbl, bc, bits=dead, s_out=dense)
    assert not dense.isnan().any(), (
        "an all-dead bitmap left the DENSE plane partly unwritten; the dense "
        "state-out predicate must run every block whatever the bitmap says")


# ---------------------------------------------------------------------------
# the refusals
# ---------------------------------------------------------------------------

def test_an_arm_without_the_work_loop_refuses_by_name():
    """`(C=64, BC=128, D=2, B=64)` carries no persistent variant: its register
    census does not hold the work loop. Serving it the full grid instead would
    attribute a measurement to a launch that never ran, so it raises."""
    bh, L, bc = 2, 64, 128
    pr, pw, gw, v = _routing(bh, L, 4, bc, seed=99)
    m = extension()
    tbl = m.chunk_union_table(pr, pw, v, 64, bc, _WIDTHS)
    bits = m.chunk_block_bits(tbl, bc, _WIDTHS, L, None)
    with pytest.raises(RuntimeError, match="sparse grid is not built"):
        m.chunk_forward_raw(pr, pw, gw, v, tbl, 64, bc, _WIDTHS, None, None, None, bits)


def test_a_bitmap_built_for_another_arm_is_refused():
    """The bitmap's row count IS its block grain; a mismatch is a different arm."""
    bh, L = 2, 64
    pr, pw, gw, v = _routing(bh, L, 4, 64, seed=5)
    m = extension()
    tbl64 = m.chunk_union_table(pr, pw, v, _C, 64, _WIDTHS)
    tbl128 = m.chunk_union_table(pr, pw, v, _C, 128, _WIDTHS)
    bits128 = m.chunk_block_bits(tbl128, 128, _WIDTHS, L, None)
    with pytest.raises(RuntimeError, match="built for a different arm"):
        m.chunk_forward_raw(pr, pw, gw, v, tbl64, _C, 64, _WIDTHS, None, None, None, bits128)


def test_the_seam_reaches_the_sparse_grid():
    """`rola.ops.chunk.run(sparse_grid=True)` -- the packing the facade calls.

    The plan's choice travels through the seam, and the seam's own permute-and-cast
    is exactly the step a hand-built fixture skips.
    """
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from conftest import build_routes

    from rola.ops import chunk
    from rola.routing.producer import union_routing

    torch.manual_seed(0)
    gateway = build_routes(hidden_size=64, num_heads=2, widths=(64, 64),
                           routing=union_routing()).cuda()
    x = torch.randn(2, 64, 64, device="cuda")
    routes = gateway(x)
    v = torch.randn(2, 64, 2, _DV, device="cuda")
    y_full, _ = chunk.run(chunk.pack(routes), v)
    y_sparse, _ = chunk.run(chunk.pack(routes), v, sparse_grid=True)
    scale = float(y_full.abs().max())
    assert float((y_sparse - y_full).abs().max()) <= _SIGMA_FLOOR_REL * max(scale, 1.0)


def _sparse_bundle(B, T, H, live_d0, seed=11):
    """A `RouteFactors` whose level-0 support is `live_d0` digits wide.

    Direct construction, which the bundle's contract permits: the point is a cell
    with a KNOWN dead-block count, and a producer would only reach one by accident.
    """
    from rola.routing.factors import RouteFactors
    from rola.routing.producer import union_routing
    from rola.routing.types import LevelConfig, Topology

    g = torch.Generator(device="cuda").manual_seed(seed)
    topology = Topology(levels=tuple(
        LevelConfig(width, union_routing(1.5)) for width in _WIDTHS))

    def side():
        level0 = torch.zeros(B, T, H, _WIDTHS[0], device="cuda")
        a = torch.rand(B, T, H, live_d0, device="cuda", generator=g) + 0.5
        level0[..., :live_d0] = a / a.sum(-1, keepdim=True)
        b = torch.rand(B, T, H, _WIDTHS[1], device="cuda", generator=g) + 0.5
        return (level0, b / b.sum(-1, keepdim=True))

    return RouteFactors(
        topology=topology, normalization="global", read=side(), write=side(),
        g_read=None, g_write=torch.rand(B, T, H, device="cuda", generator=g) + 0.5)


@pytest.mark.parametrize("paging", [False, True])
def test_the_facade_reaches_the_sparse_grid_and_the_state_is_bit_identical(paging):
    """R5: `PlanOverrides(sparse_grid=True)` is the persistent grid, THROUGH `rola_op`.

    P68 measured the win and left it unreachable -- `sparse_grid` existed only on
    `rola.ops.chunk.run`, which the facade never set. This row is the reachability
    claim and the correctness claim in one: the same call, expert-pinned both ways,
    must agree on `s_out` BYTE FOR BYTE (the module docstring's real gate -- each
    atom has exactly one storing CTA, so there is no ordering freedom to hide in),
    on the dense plane and on the paged one alike.
    """
    import rola
    from rola.expert import PlanOverrides

    B, T, H, live = 1, 64, 2, 4
    routes = _sparse_bundle(B, T, H, live)
    v = torch.randn(B, T, H, _DV, device="cuda")
    #: NON-VACUITY: the cell must actually have dead blocks, or the gate is a tautology.
    touched = int((routes.write[0].abs().amax(dim=(0, 1, 2)) > 0).sum())
    assert touched == live, f"the fixture's level-0 support is {touched}, not {live}"
    assert touched * _WIDTHS[1] < _N, "every block is live; there is nothing to skip"

    planes = []
    for sparse_grid in (False, True):
        state = rola.state()
        with torch.no_grad():
            rola.rola_op(routes, v, state=state,
                         expert=PlanOverrides(paging=None if paging else False,
                                              sparse_grid=sparse_grid))
        planes.append(state.materialize().clone())
    assert torch.equal(*planes), (
        "the facade's sparse grid and its full grid disagree on the carried state")


def test_the_expert_pin_is_what_the_launch_actually_receives():
    """THE TOOTH under the row above: a bit-identity gate on `s_out` passes just as
    well if the flag never left the facade. This one watches the value arrive.
    """
    import rola
    from rola.expert import PlanOverrides
    from rola.ops import chunk

    routes = _sparse_bundle(1, 64, 2, 4)
    v = torch.randn(1, 64, 2, _DV, device="cuda")
    seen, run = [], chunk.run

    def spy(packed, value, **kwargs):
        seen.append(kwargs.get("sparse_grid"))
        return run(packed, value, **kwargs)

    chunk.run = spy
    try:
        with torch.no_grad():
            for pin in (None, False, True):
                rola.rola_op(routes, v, expert=PlanOverrides(sparse_grid=pin))
            rola.rola_op(routes, v)
    finally:
        chunk.run = run
    assert seen == [False, False, True, False], (
        f"the expert's sparse_grid pin did not arrive at the launch: {seen}")
