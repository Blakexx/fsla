# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""Dense and paged are ONE kernel and ONE base translation -- bit for bit.

The carry consumer's state I/O addresses a page per atom. Which page is the only thing
a backing changes: without a table the slot IS the atom and the plane is the dense
`[BH, N/16, 16, cols]` one; with a table the slot is what the plan committed and the
plane holds only those pages. Same instruction stream, same accumulation order, same
ABI -- so the claim here is `torch.equal` and not a tolerance. A tolerance would be the
wrong instrument entirely: any difference at all is an ADDRESSING difference, and
addressing is exact or wrong.

This file is `675d6b6^`'s of the same name, RE-ANCHORED onto the K31 carry family after
the deleted consumer took the original with it (`docs/internals/DELETIONS.md`). Its
claims and their derivations are unchanged; what moved is the seam they are made at.

Vocabulary, defined here and used unqualified below: `D` = level count; `B` = one
uniform level width; `N = B ** D` = leaves; `T` = tokens; `BH = batch * heads`;
`d_v` = the value width; `cols = d_v + 1`; **atom** = the 16-leaf page granule
(`rola.ops.paging.MMA_K_QUANTUM`); **slot** = the physical page an atom is mapped to,
`-1` for an atom the plan committed nothing for; `BC` = leaves one owner block owns;
`owners = N / BC`.

**THE PLAN IS THE SHIPPED ONE, UNMODIFIED.** The bitmap planned here is
`rola.engine.facts.planes.atom_bits` exactly as published -- the kernel's own exact
write set, canonically keyed -- and the table is `PageArena.plan_exact`'s. That is only
possible because the consumer re-keys its own atom IDS to the canonical numbering at its
prologue (`box.cuh`'s `canon_atom`, P80): the two leaf orders partition the leaves into
the SAME atoms (K2/K3) and agree on the row inside one, so the re-key is arithmetic on
ids and never a permutation of data. One representation, shared by the arena,
`AtomKeying`, `materialize` and the decode step, is what makes this file's `torch.equal`
a statement about addressing rather than about a convention.
"""
from __future__ import annotations

from math import prod

import pytest
import torch

from rola.engine.facts import planes
from rola.engine.types import stored_cols
from rola.ops import carry as carry_ops
from rola.ops.paging import MMA_K_QUANTUM, PageArena, bytes_equal, from_split_planes
from tests.oracle.fixtures import assert_fresh_binary, require_arm

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the carry is a CUDA kernel"),
]

_DV = carry_ops.D_V
_COLS = _DV + 1
#: The page's STORED width: `_COLS` plus the pad that puts every slot's hi rows on a
#: 128 B line (docs/internals/state.md#format). It is what a state PLANE is shaped by;
#: `_COLS` is what a materialized, rejoined state is shaped by.
_STORED = stored_cols(_COLS)
_POISON = -1234.5

#: `_launch`'s "the caller said nothing", which is not the same as an explicit `None`.
_UNSET = object()

#: `(widths, k, m, window, level_modes)` -- one arm per depth the carry's list carries,
#: covering BOTH bodies: `level_modes = 0` is DENSE_BOTH and takes the channel-split
#: body, a nonzero declaration takes the carved one. Row 9's `(8,)*4` is the MULTI-LEVEL
#: carve, and row 1 is the shipped flagship.
_ARMS = (
    ((64, 64), 4, 8, 64, 0),
    ((64, 64), 4, 8, 64, 6),
    ((16, 16, 16), 2, 16, 64, 0),
    ((8, 8, 8, 8), 2, 8, 64, 102),
    ((256, 256), 4, 8, 384, 6),
)


def _arm_id(a):
    return f"{'x'.join(map(str, a[0]))}-W{a[3]}-mo{a[4]}"


#: the fraction of each level's digits the WRITE side is allowed to reach. Residency is
#: a property of the write support's STRUCTURE, not of its density: unstructured
#: sparsity at long `T` still reaches every atom, so a cell that wants absent atoms must
#: restrict digits.
_SUPPORTS = {"sparse": 0.25, "mixed": 0.5, "dense": 1.0}


def _need(arm):
    widths, k, m, window, modes = arm
    require_arm(widths, k, m, window, 32, carry_ops.LEAF_SPLIT, modes)


def _simplex(shape, generator, live=None):
    """Rows on the simplex, optionally supported on the first `live` digits."""
    x = torch.rand(shape, device="cuda", dtype=torch.float64, generator=generator)
    if live is not None:
        x[..., live:] = 0.0
    return (x / x.sum(-1, keepdim=True)).to(torch.bfloat16)


def _fixture(widths, *, T, support=1.0, read_support=1.0, seed=0, B=1, H=1):
    """`read_support`-structured read support, `support`-structured write support.

    Both truncations keep a level's FIRST digits, so a smaller support is a SUBSET of a
    larger one at every level and therefore over the atoms too -- which is what lets a
    continuation's second call touch a strict subset of its first call's atoms.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    read = tuple(_simplex((B, T, H, w), g, live=max(1, int(w * read_support)))
                 for w in widths)
    write = tuple(_simplex((B, T, H, w), g, live=max(1, int(w * support))) for w in widths)
    g_write = torch.rand((B, T, H), device="cuda", generator=g).to(torch.bfloat16)
    v = torch.randn((B, T, H, _DV), device="cuda", generator=g).to(torch.bfloat16)
    return read, write, g_write, v


def _BH(fixture):
    return fixture[3].shape[0] * fixture[3].shape[2]


def _launch(fixture, arm, *, s_in=None, s_out=None, page_tbl=None, bits=_UNSET):
    widths, k, m, window, modes = arm
    read, write, g_write, v = fixture
    if bits is _UNSET:
        bits = None if (s_in is None and s_out is None) else _bitmap(fixture, arm)
    return carry_ops.carry_inter(read, write, g_write, v, widths, k=k, m=m, window=window,
                                 chunk=32, carve=carry_ops.LEAF_SPLIT, level_modes=modes,
                                 state_in=s_in, state_out=s_out, page_table=page_tbl,
                                 atom_bits=bits)


def _dense(fixture, arm, *, s_in=None, plane=None):
    widths = arm[0]
    if plane is None:
        plane = carry_ops.state_plane(widths, _BH(fixture), _DV)
    y, den, out = _launch(fixture, arm, s_in=s_in, s_out=plane)
    return y, den, from_split_planes(out).reshape(_BH(fixture), prod(widths), _COLS)


def _side(fixture, which):
    """One side as the `[B, H, L, sum_l w_l]` packed plane the facts pass reads -- the
    producer's own layout, not the kernel seam's `[BH, L, sum]` one."""
    return torch.cat(fixture[which], -1).permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()


def _bitmap(fixture, arm):
    """The kernel's own EXACT atom-grain ACTIVITY byte, as the shipped pass publishes it.

    Bit 0 WRITTEN, bit 1 READ, over BOTH sides' packed planes (P80b). No re-keying:
    since P80 the consumer addresses its state by the CANONICAL atom id this bitmap is
    indexed by, so the plan the engine would build is the plan this file hands the arena
    and the byte the engine would pass is the byte this file hands the kernel.
    """
    widths = arm[0]
    bits = planes.atom_bits(_side(fixture, 1), widths, read_plane=_side(fixture, 0))
    return bits.reshape(_BH(fixture), -1)


def _plan(fixture, arm):
    """The WRITE half alone: residency is keyed on the atoms a call writes."""
    return planes.written_atoms(_bitmap(fixture, arm))


def _arena(arm, BH, pool_slack=0.0):
    widths, k, m, _, _ = arm
    BC = carry_ops.box_shape(widths, k, m)[3]
    return PageArena(widths=widths, BC=BC, cols=_COLS, BH=BH, device=torch.device("cuda"),
                     pool_slack=pool_slack)


def _paged(fixture, arm, arena, *, s_in=None):
    arena.wait()
    y, den, _ = _launch(fixture, arm, s_in=s_in, s_out=arena.state,
                        page_tbl=arena.page_table)
    return y, den, arena.materialize().reshape(_BH(fixture), prod(arm[0]), _COLS)


def _stored(arena) -> torch.Tensor:
    """The arena's pages gathered into CANONICAL atom order, still in the STORED form.

    `materialize` rejoins hi||lo; this one does not, so the comparison it feeds is the
    BYTE claim over the split layout itself -- a backing that agreed on values while
    disagreeing on which plane a half lives in would pass the first and fail this.
    """
    table = arena.page_table
    gathered = arena.state.index_select(0, table.long().clamp_min(0).reshape(-1))
    stored = torch.zeros_like(gathered)
    present = (table >= 0).reshape(-1)
    stored[present] = gathered[present]
    return stored


def _y_is_exact(arm) -> bool:
    """`y` is fanned in with `atomicAdd` across owner blocks, so at `owners > 1` the
    path is not bit-reproducible against ITSELF and the comparison is against the
    reordering envelope with a null control asserted to fit inside it. The STATE is
    exact on every arm regardless: each atom is stored by exactly one CTA."""
    widths, k, m = arm[0], arm[1], arm[2]
    return prod(widths) // carry_ops.box_shape(widths, k, m)[3] == 1


def _envelope(y, arm):
    n = prod(arm[0]) // carry_ops.box_shape(arm[0], arm[1], arm[2])[3]
    return n * n * torch.finfo(torch.float32).eps * y.abs().max()


def _assert_y(y_dense, y_paged, y_control, arm, what):
    if _y_is_exact(arm):
        assert torch.equal(y_dense, y_control), "y is not reproducible against itself"
        assert torch.equal(y_dense, y_paged), what
    else:
        bound = _envelope(y_dense, arm)
        assert (y_dense - y_control).abs().max() <= bound, "the null control blew its envelope"
        assert (y_dense - y_paged).abs().max() <= bound, what


# ---------------------------------------------------------------------------
# 1. dense and paged are the same call
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("support", list(_SUPPORTS), ids=list(_SUPPORTS))
@pytest.mark.parametrize("arm", _ARMS, ids=_arm_id)
def test_the_paged_backing_is_bit_identical_to_the_dense_one(arm, support):
    assert_fresh_binary()
    _need(arm)
    fixture = _fixture(arm[0], T=2 * arm[3], support=_SUPPORTS[support],
                       seed=len(arm[0]) * 97 + arm[3])
    BH = _BH(fixture)

    dense_plane = carry_ops.state_plane(arm[0], BH, _DV)
    y_dense, _, dense = _dense(fixture, arm, plane=dense_plane)
    y_control, _, _ = _dense(fixture, arm)
    arena = _arena(arm, BH)
    result = arena.plan_exact(_plan(fixture, arm))
    y_paged, _, paged = _paged(fixture, arm, arena)

    assert result.allocated_pages > 0 and (paged != 0).any(), (
        "the plan committed nothing, or the pages came back empty")
    identity = torch.arange(arena.logical_pages, device="cuda",
                            dtype=torch.int32).view_as(arena.page_table)
    assert support == "dense" or not torch.equal(arena.page_table, identity), (
        "the plan mapped every atom to its own id, so this cell never exercised "
        "a translation at all")
    assert bytes_equal(_stored(arena),
                       dense_plane.reshape(-1, MMA_K_QUANTUM, _STORED)), (
        "the two backings disagree on the STORED bytes: the split layout is part of "
        "the backing-invisibility claim, not only the values it rejoins to")
    assert torch.equal(dense, paged), (
        f"the paged state differs from the dense one by "
        f"{(dense - paged).abs().max().item():.3e} over "
        f"{int((dense != paged).sum())} elements")
    _assert_y(y_dense, y_paged, y_control, arm, "the paged y differs from the dense y")


# ---------------------------------------------------------------------------
# 2. a zero page is inert (the design's E.4)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", _ARMS, ids=_arm_id)
def test_a_zero_page_is_inert(arm):
    """Force atoms the routing never writes to be resident, and nothing moves.

    This is what licenses every SUPERSET residency derivation as safe-direction, and it
    is the mechanical form of the self-normalizing readout's dead-leaf argument.
    """
    assert_fresh_binary()
    _need(arm)
    fixture = _fixture(arm[0], T=2 * arm[3], support=0.25, seed=len(arm[0]) * 11 + arm[3])
    BH = _BH(fixture)
    exact = _plan(fixture, arm)
    if bool(exact.all()):
        pytest.skip("this arm's routing already reaches every atom; there is no zero page")

    exactly = _arena(arm, BH)
    exactly.plan_exact(exact)
    y_exact, _, s_exact = _paged(fixture, arm, exactly)

    superset = _arena(arm, BH)
    superset.plan_exact(torch.ones_like(exact))
    y_super, _, s_super = _paged(fixture, arm, superset)

    #: THE NULL CONTROL is the exact plan run a second time into its own arena: at
    #: `owners > 1` the fan-in's atomics make `y` non-reproducible against ITSELF, so the
    #: control is what bounds the comparison. The STATE is exact on every arm.
    control = _arena(arm, BH)
    control.plan_exact(exact)
    y_control, _, _ = _paged(fixture, arm, control)

    assert superset.resident_pages > exactly.resident_pages, "the superset admitted nothing extra"
    assert torch.equal(s_exact, s_super), "a resident never-written page moved the state"
    _assert_y(y_exact, y_super, y_control, arm, "a resident never-written page moved y")


# ---------------------------------------------------------------------------
# 3. an absent atom is skipped, not written
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", _ARMS, ids=_arm_id)
def test_an_absent_atom_is_skipped_and_never_written(arm):
    """A structured-support run leaves every unadmitted page exactly as it found it.

    Checked against a poison fill of the arena's whole backing: the exit epilogue stores
    only PLANNED atoms, so a page outside the plan keeps its poison, and `y` still
    equals the dense run's.
    """
    assert_fresh_binary()
    _need(arm)
    fixture = _fixture(arm[0], T=2 * arm[3], support=0.25, seed=len(arm[0]) * 13 + arm[3])
    BH = _BH(fixture)
    bitmap = _plan(fixture, arm)
    if bool(bitmap.all()):
        pytest.skip("this arm's routing reaches every atom; no atom is absent to skip")

    y_dense, _, dense = _dense(fixture, arm)
    y_control, _, _ = _dense(fixture, arm)

    arena = _arena(arm, BH)
    arena.plan_exact(bitmap)
    arena.wait()
    #: THE CLAIM LIVES ON THE RAW POOL, not on `materialize()`, which gathers through the
    #: table and sends an absent atom to zeros BY CONSTRUCTION. What must not move is a
    #: physical slot the table names for nothing: an atom the plan skipped has no slot,
    #: so a kernel that addressed it anyway would scribble at a NEGATIVE displacement.
    #: The whole pool is poisoned -- planned slots included, which a fresh call rewrites
    #: whole -- and the unmapped slots must come back untouched.
    used = arena.page_table[arena.page_table >= 0].long()
    spare = torch.ones(arena.state.shape[0], dtype=torch.bool, device="cuda")
    spare[used] = False
    if not bool(spare.any()):
        pytest.skip("this arena's pool is exactly full; no unmapped slot to poison")
    arena.state.fill_(_POISON)
    y_paged, _, paged = _paged(fixture, arm, arena)

    live = bitmap.repeat_interleave(MMA_K_QUANTUM, dim=1)
    assert torch.equal(dense[live], paged[live]), "a planned atom differs across the backings"
    assert bool((arena.state[spare] == _POISON).all()), (
        "an unmapped slot was written: the exit epilogue is addressing atoms the plan "
        "committed nothing for")
    _assert_y(y_dense, y_paged, y_control, arm, "the skipping run's y differs from the dense y")


# ---------------------------------------------------------------------------
# 4. a reserved slack pool is bitwise invisible
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", _ARMS, ids=_arm_id)
def test_a_reserved_slack_pool_is_bitwise_invisible(arm):
    """A pool moves every residency slot and must move NO byte of the state.

    Its slots come off the SAME allocator, ahead of the routing's own, so a pooled arena
    assigns different slot ids to the same atoms -- and "which slot" is exactly what a
    backing is not allowed to be able to say.
    """
    assert_fresh_binary()
    _need(arm)
    fixture = _fixture(arm[0], T=2 * arm[3], support=0.5, seed=len(arm[0]) * 31 + arm[3])
    BH = _BH(fixture)
    bitmap = _plan(fixture, arm)

    y_dense, _, dense = _dense(fixture, arm)
    y_control, _, _ = _dense(fixture, arm)

    plain = _arena(arm, BH)
    plain.plan_exact(bitmap)
    pooled = _arena(arm, BH, pool_slack=0.25)
    pooled.plan_exact(bitmap)
    y_pooled, _, state_pooled = _paged(fixture, arm, pooled)

    assert pooled.pool_capacity > 0, "this cell needs a pool"
    assert not torch.equal(plain.page_table, pooled.page_table), (
        "the pool did not move a single slot id, so this cell exercised no translation")
    assert torch.equal(dense, state_pooled), (
        "the reserved pool moved the state: some address is keyed on atom RANK rather "
        "than on the page table")
    _assert_y(y_dense, y_pooled, y_control, arm, "the pooled arena's y differs from the dense y")


# ---------------------------------------------------------------------------
# 5. a continuation is bit-identical across the two backings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", _ARMS, ids=_arm_id)
def test_a_continuation_is_bit_identical_across_the_two_backings(arm):
    """Two halves chained through the state, once dense and once in the arena.

    BOTH backings continue IN PLACE: the entry state and the final state are the SAME
    plane, resolved through the same table, which is the aliasing the design permits
    (`state.md#the-dense-continuation`) and the case a growing sequence actually runs.
    The second half's plan is `fresh=False` -- the atoms the first half committed stay
    committed, and only the newly written ones are admitted and zeroed.
    """
    assert_fresh_binary()
    _need(arm)
    widths, k, m, window, _ = arm
    T = 4 * window
    fixture = _fixture(widths, T=T, support=0.5, seed=31)
    read, write, g_write, v = fixture
    BH = _BH(fixture)
    half = T // 2
    first = (tuple(x[:, :half] for x in read), tuple(x[:, :half] for x in write),
             g_write[:, :half], v[:, :half])
    second = (tuple(x[:, half:] for x in read), tuple(x[:, half:] for x in write),
              g_write[:, half:], v[:, half:])

    plane = carry_ops.state_plane(widths, BH, _DV)
    y1_dense, _, _ = _dense(first, arm, plane=plane)
    s1 = from_split_planes(plane.clone()).reshape(BH, prod(widths), _COLS)
    #: THE SAME PLANE IN AND OUT: a stateful chain allocates ONE plane for the SEQUENCE.
    y2_dense, _, s2 = _dense(second, arm, s_in=plane, plane=plane)
    s2 = s2.clone()
    y1_control, _, _ = _dense(first, arm)

    arena = _arena(arm, BH)
    arena.plan_exact(_plan(first, arm))
    y1_paged, _, p1 = _paged(first, arm, arena)
    assert torch.equal(s1, p1), "the first half's state differs across the backings"

    arena.plan_exact(_plan(second, arm), fresh=False)
    y2_paged, _, p2 = _paged(second, arm, arena, s_in=arena.state)
    assert torch.equal(s2, p2), (
        f"the continued state differs across the backings by "
        f"{(s2 - p2).abs().max().item():.3e}")
    _assert_y(y1_dense, y1_paged, y1_control, arm, "the first half's y differs")
    _assert_y(y2_dense, y2_paged, y2_dense, arm, "the second half's y differs")


# ---------------------------------------------------------------------------
# 6. a stateless call runs a variant with no state I/O
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", _ARMS, ids=_arm_id)
def test_a_stateless_call_carries_no_state(arm):
    """The predicate is `state is not None`, at this seam as at the facade's.

    A call handing NEITHER plane reaches an arm variant with no state I/O compiled into
    it, allocates nothing, and computes the same `y` a fresh stateful call does -- a
    fresh entry state is zero, and the store is the only other thing the variant drops.
    """
    assert_fresh_binary()
    _need(arm)
    fixture = _fixture(arm[0], T=2 * arm[3], support=1.0, seed=len(arm[0]) * 17 + arm[3])
    y_state, _, _ = _dense(fixture, arm)
    y_none, _, none = _launch(fixture, arm)
    assert none is None, "a stateless call handed back a state plane"
    _assert_y(y_state, y_none, y_state, arm, "the stateless variant's y differs from the "
              "stateful one's over a zero entry state")


# ---------------------------------------------------------------------------
# 7. a RESIDENT BUT IDLE atom costs a continuation nothing (P80b)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", _ARMS, ids=_arm_id)
def test_a_resident_but_idle_atom_is_neither_loaded_nor_stored(arm):
    """The second call's routing touches a STRICT SUBSET of the first call's atoms.

    Residency is the arena's fact and activity is the facts pass' (P80b): the union
    residency a continuation plans keeps the first call's atoms mapped, and the second
    call must touch none of the ones it neither reads nor writes. The claim is made
    twice, because the two halves of the predicate fail differently:

    * THE STORE SKIP is observable, and is checked against a poison fill of exactly the
      idle-resident slots -- they must come back holding it.
    * THE LOAD SKIP is not observable by construction: an idle atom is not read, so its
      contents cannot reach `y` whether the sweep loads them or not. That is the whole
      point (skipping is exact, not an approximation), and what it saves is DRAM traffic
      -- measured by the probe's ncu pass, never by an assertion here.

    Correctness is the same `torch.equal` the rest of this file makes: the continued
    state and `y` are identical to the dense continuation's, which runs the identical
    predicate over a null table.
    """
    assert_fresh_binary()
    _need(arm)
    widths, k, m, window, _ = arm
    BH = 1
    first = _fixture(widths, T=2 * window, support=0.5, read_support=0.5, seed=61)
    second = _fixture(widths, T=2 * window, support=0.25, read_support=0.25, seed=62)

    resident = _plan(first, arm)
    active = _bitmap(second, arm) != 0
    idle = resident & ~active
    if not bool(idle.any()):
        pytest.skip("this arm's second call reaches every atom the first one committed")
    assert bool((_plan(second, arm) & ~resident).sum() == 0), (
        "the second call writes an atom the first did not: this cell is about the union "
        "residency the first call already committed")

    plane = carry_ops.state_plane(widths, BH, _DV)
    _dense(first, arm, plane=plane)
    y2_dense, _, s2 = _dense(second, arm, s_in=plane, plane=plane)
    s2 = s2.clone()

    arena = _arena(arm, BH)
    arena.plan_exact(resident)
    _paged(first, arm, arena)
    arena.plan_exact(_plan(second, arm), fresh=False)
    y2_paged, _, p2 = _paged(second, arm, arena, s_in=arena.state)
    assert torch.equal(s2, p2), (
        f"the continued state differs across the backings by "
        f"{(s2 - p2).abs().max().item():.3e}; the two backings ran different predicates")

    #: THE STORE SKIP, on a second arena carrying the same first half: poison exactly
    #: the idle-resident slots and they must survive the second call untouched.
    poisoned = _arena(arm, BH)
    poisoned.plan_exact(resident)
    _paged(first, arm, poisoned)
    poisoned.plan_exact(_plan(second, arm), fresh=False)
    poisoned.wait()
    slots = poisoned.page_table[idle].long()
    assert int(slots.numel()) > 0, "no idle-resident slot to poison"
    poisoned.state[slots] = _POISON
    y2_poison, _, _ = _paged(second, arm, poisoned, s_in=poisoned.state)
    assert bool((poisoned.state[slots] == _POISON).all()), (
        "an idle-resident atom was written: the exit sweep is storing atoms this call "
        "does not write")
    _assert_y(y2_paged, y2_poison, y2_dense, arm,
              "poisoning an idle-resident atom moved y: the entry sweep is reading an "
              "atom this call does not read")
