"""Spec §7.1 page commitment: the shipping gates.

The one that ships this increment is **bit-identity of the STATE, paged versus
unpaged, with `y` held to the kernel's own null control**. Paging is memory
management; it decides where a value panel LIVES, never what it is.

That split is forced by a property of the consumer, not by paging, and it was
MEASURED rather than assumed (findings `paging.md` §2):

- The **state** is stored once per owner, by exactly one CTA, with no atomic. It
  is a deterministic function of the inputs, so `torch.equal` is the right gate and
  paging must not move a bit of it.
- **`y`** is accumulated across owners with `atomicAdd` (§5's split-K readout).
  Float atomics do not commute in floating point and their order is not
  reproducible, so `y` is not bit-reproducible against ITSELF: running the unpaged
  consumer twice on identical inputs already differs (measured: `1.9e-6` max
  absolute at the fixture below). Asserting `torch.equal(y_paged, y_unpaged)` would
  therefore be asserting something false about the KERNEL and would fail for a
  reason that has nothing to do with paging.

So `y` is gated two ways, neither of which is a tuned tolerance:

- against a **derived reordering bound**. One output element is a sum of at most
  `owners_total` fp32 partials, and reassociating an `n`-term fp32 sum perturbs it by
  at most `(n-1) * eps * sum|terms|`; taking `sum|terms| <= n * max|y|` as the
  reachable scale gives `bound = (n-1) * n * eps_fp32 * max|y|`, shipped as
  `owners_total^2 * eps_fp32 * max|y|` (`_reordering_bound` below -- fix pass R2,
  `workflows/20260730-wiring/paging-review.md`: an earlier revision of this
  docstring under-stated the derivation by one factor of `n`, 16x looser than the
  code at `n = 16`, though the code itself was always the two-factor bound). That
  is arithmetic, not a threshold. **The STATE `torch.equal` gate is what actually
  catches a paging defect** (fix pass R3): the `y` bound above is a sanity
  envelope on the kernel's own atomic reassociation, loose enough that even a
  single missed page commitment clears it by only ~2x, so a bound violation on `y`
  alone is not the signal to trust -- the state comparison is.
- against a **null control** that also has to fit inside that bound. The unpaged
  path run twice must itself land under `bound`, which is what keeps the bound
  honest: if the model of the error were wrong, the control would blow it and the
  test would fail before ever looking at the paged path.

A single control draw is deliberately NOT used as the bound directly. The control
is a random variable (which atomics land in which order), so one draw
under-estimates its own range and a strict `paged <= control_draw` comparison flakes
at the ULP scale -- measured, at `7.45e-9` against a `5.59e-9` draw, both pure fp32
noise. Comparing one draw of one random variable against one draw of another is not
a test of anything.

The oracle is untouched here on purpose. `tests/ops/test_rola_v3_consumer.py`
already gates the unpaged consumer against the canonical fp64
`naive_rola`; this file gates the paged consumer against THAT consumer, so
correctness composes transitively through a reference that is already canonical.
Writing a fresh reference for the paged path is exactly the
[[matching-the-naive-antipattern]] the shared-router bug walked through, and spec
§2.1 forbids it.

Vocabulary, defined at first use: `N` = leaf count = `prod_l width_l`;
`BC` = leaves per owner (spec §3's owner rectangle); `owners_total` = `N / BC`;
`BH` = `batch * heads`; `BT` = the plan's token tile, a compile-time constant in
`{16, 32}`; `cols` = `d_v` plus one mass column under `normalization='global'`.
"""

from __future__ import annotations

import pytest
import torch

from rola.ops.consumer import _consumer_forward
from rola.ops.paging import MMA_K_QUANTUM, PageArena
from rola.planner.schedule import Plan, build_schedule
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    LeafMassDecay,
    LevelConfig,
    SoftmaxActivation,
    Topology,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="V3 paging is a CUDA feature"),
]

#: A level's WRITE activation is what decides whether its factor can be exactly
#: zero, and §7.1's mark phase reads that structurally (`can_zero`) rather than
#: measuring it -- a softmax level cannot emit an exact zero, so the planner takes
#: the dense short-circuit and commits the whole page domain.
#:
#: The fixtures must therefore DECLARE what they emit. A fixture that injects exact
#: zeros under a declared softmax write router is describing an unreachable model
#: (sparsity comes from entmax, spec §1.2), and it silently tested the dense path
#: while looking like it tested the sparse one. Untied sparse/sparse is rejected by
#: `IndependentRouting`, so the sparse fixture is dense-read / sparse-write.
_DENSE_ROUTING = IndependentRouting(
    tied=False, read_activation=SoftmaxActivation(), write_activation=SoftmaxActivation())
_SPARSE_WRITE_ROUTING = IndependentRouting(
    tied=False, read_activation=SoftmaxActivation(), write_activation=EntmaxActivation(1.5))


def _topology(widths, d_v, sparse_write=True):
    routing = _SPARSE_WRITE_ROUTING if sparse_write else _DENSE_ROUTING
    return Topology(levels=tuple(LevelConfig(w, routing) for w in widths), d_v=d_v)


def _simplex(shape, p_nonzero, generator, device):
    """A routing allocation: rows on the simplex, with EXACT support zeros.

    `p_nonzero < 1` is what makes the mark phase's predicate reachable -- entmax's
    exact zeros are the scheduling authority (spec §1.2), and a fixture that never
    emits one would exercise only the dense short-circuit.
    """
    x = torch.rand(shape, device=device, dtype=torch.float64, generator=generator)
    if p_nonzero < 1.0:
        keep = torch.rand(shape, device=device, dtype=torch.float64, generator=generator) < p_nonzero
        x = x * keep
    x = torch.where(x.sum(-1, keepdim=True) == 0, torch.ones_like(x), x)
    return (x / x.sum(-1, keepdim=True)).to(torch.float32)


def _fixture(widths, d_v, *, B=2, T=48, H=2, p_nonzero=0.4, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    read = tuple(_simplex((B, T, H, w), p_nonzero, g, device) for w in widths)
    write = tuple(_simplex((B, T, H, w), p_nonzero, g, device) for w in widths)
    g_read = torch.rand((B, T, H), device=device, generator=g)
    g_write = torch.rand((B, T, H), device=device, generator=g)
    v = torch.randn((B, T, H, d_v), device=device, generator=g)
    return read, write, g_read, g_write, v


def _plan(widths, d_v, *, BT, BC, normalization, decay, sparse_write=True, device="cuda"):
    topology = _topology(widths, d_v, sparse_write=sparse_write)
    return Plan(topology=topology, normalization=normalization, BT=BT, BC=BC,
                  warp_tile=(1, max(1, d_v // 32)), decay=decay,
                  mma_precision="tf32x3")


def _decay(topology, device, H, value=0.9):
    return LeafMassDecay(dials=tuple(
        torch.full((H, w), value, device=device, dtype=torch.float64)
        for w in topology.widths))


def _run(plan, fixture, *, arena=None, initial_state=None, decay=None, continue_paging=False):
    read, write, g_read, g_write, v = fixture
    global_norm = plan.normalization == "global"
    schedule = build_schedule(read, write, None if global_norm else g_read, g_write, plan)
    return _consumer_forward(
        v, read, write, None if global_norm else g_read, g_write, plan,
        initial_state=initial_state, decay=decay, schedule=schedule, paging=arena,
        continue_paging=continue_paging)


def _reordering_bound(y: torch.Tensor, plan) -> torch.Tensor:
    """The most `y` can move under a reassociation of the §5 readout's atomics.

    One output element is a sum of at most `owners_total` fp32 partials, and
    reassociating an `n`-term fp32 sum perturbs it by at most
    `(n-1) * eps * sum|terms|`. Bounding `sum|terms|` by `n * max|y|` (the reachable
    scale when the partials do not catastrophically cancel) gives the expression
    below. Derived arithmetic, not a tuned tolerance -- and the null control is
    asserted to fit inside it on every run, so a wrong model fails loudly.
    """
    n = plan.geometry.owners_total
    return n * n * torch.finfo(torch.float32).eps * y.abs().max()


def _arena(plan, *, BH, max_physical_pages=None):
    d_v = plan.topology.d_v
    cols = d_v + (1 if plan.normalization == "global" else 0)
    return PageArena(widths=plan.topology.widths, BC=plan.BC, cols=cols, BH=BH,
                       device=torch.device("cuda"),
                       max_physical_pages=max_physical_pages)


# ---------------------------------------------------------------------------
# The shipping gate: bit-identity across the full fixture grid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decay_on", [False, True], ids=["decay-none", "leaf-mass"])
@pytest.mark.parametrize("normalization", ["raw", "global"])
@pytest.mark.parametrize("p_nonzero", [0.4, 1.0], ids=["sparse", "dense"])
@pytest.mark.parametrize("BT", [16, 32])
@pytest.mark.parametrize("BC", [16, 64], ids=["BC16-QPO1", "BC64-QPO4"])
def test_paging_is_bitwise_invisible(decay_on, normalization, p_nonzero, BT, BC):
    """State exactly equal; `y` within the kernel's own reordering envelope.

    Covers both decay arms, both normalizations, sparse and dense routing, BOTH
    values of `BT` -- because §7.1's `token_tile` coupling is settled PARAMETRIC
    (findings `paging.md` D3), so `BT = 16` is a gated configuration rather than an
    untested one -- and BOTH instantiated `BC`. `p_nonzero = 1.0` is the dense limit,
    where every owner is live and the mark phase takes its short-circuit.

    **`BC = 64` IS THE CELL THAT WAS MISSING, AND IT IS THE ONE THAT MATTERS.**
    Every cell of this file used to pin `BC = 16`, where `QPO = BC / MMA_K_QUANTUM`
    is 1 and an ATOM IS AN OWNER -- so the owner keying and the atom keying are
    indistinguishable and a half-landed re-key passes. At `BC = 64`, `QPO = 4`: an
    owner is four independently paged atoms whose slots need not be contiguous, the
    paged state is `[slots, 16, cols]` where an owner-keyed kernel expects
    `[pages, 64, cols]`, and the page table's stride is `N/16` where an owner-keyed
    kernel reads `N/BC`. `widths = (4, 8, 8)` admits it (suffix product `8*8 = 64`,
    so `j = 0`, `slice_digits = 1`, 4 owners). This is also THE FLAGSHIP DEFAULT --
    `largest_feasible_bc` resolves to 64 on the flagship topology -- so before this
    cell existed, no gated configuration was the one production takes.
    """
    widths, d_v = (4, 8, 8), 64
    plan = _plan(widths, d_v, BT=BT, BC=BC, normalization=normalization, decay=decay_on,
                 sparse_write=p_nonzero < 1.0)
    fixture = _fixture(widths, d_v, p_nonzero=p_nonzero, seed=11)
    B, T, H, _ = fixture[4].shape
    decay = _decay(plan.topology, "cuda", H) if decay_on else None

    y_dense, state_dense, _ = _run(plan, fixture, decay=decay)
    # The null control: the SAME code path, run twice.
    y_control, state_control, _ = _run(plan, fixture, decay=decay)
    arena = _arena(plan, BH=B * H)
    y_paged, returned, _ = _run(plan, fixture, arena=arena, decay=decay)

    assert returned is arena, "under paging the arena IS the carried state"
    cols = state_dense.shape[-1]
    # THE GATE. The state has no atomic, so it is exactly reproducible -- and paging
    # must not move a bit of it. Every paging defect (a wrong page id, a missed
    # commitment, a mis-ordered admission) lands here.
    assert torch.equal(state_dense, state_control), (
        "the state must be deterministic; if this fails the readout comparison below "
        "is meaningless and the kernel, not paging, is what changed")
    assert torch.equal(state_dense.reshape(B * H, -1, cols), arena.materialize())

    bound = _reordering_bound(y_dense, plan)
    assert (y_dense - y_control).abs().max() <= bound, (
        "the unpaged readout's own run-to-run spread exceeded the derived fp32 "
        "reassociation bound; the bound's model is wrong, not the paged path")
    assert (y_dense - y_paged).abs().max() <= bound


@pytest.mark.parametrize("normalization", ["raw", "global"])
def test_paging_is_bitwise_invisible_under_the_panel_lowering(normalization):
    """The same shipping gate, on the decay arm's DENSE-LIMIT intra lowering.

    `panel` changes what the kernel does with the owner's resident state inside a
    tile: the entry state is pre-scaled into a per-leaf reference frame and
    post-scaled out (§12 rule 1's split), where the walk multiplies it per token.
    Paging changes only WHERE those `BC` rows live, so the two must be orthogonal
    -- and "must be" is the reason to check it: the pre-scale is the first thing in
    this kernel that writes the resident state before any deposit, which is
    exactly the window a non-resident (unpaged, zero) owner would be visible in if
    the residency guard were wrong."""
    widths, d_v, BC = (4, 8, 8), 64, 16
    plan = _plan(widths, d_v, BT=32, BC=BC, normalization=normalization, decay=True)
    fixture = _fixture(widths, d_v, p_nonzero=0.4, seed=11)
    B, T, H, _ = fixture[4].shape
    decay = _decay(plan.topology, "cuda", H)

    y_dense, state_dense, _ = _run(plan, fixture, decay=decay)
    y_control, state_control, _ = _run(plan, fixture, decay=decay)
    arena = _arena(plan, BH=B * H)
    y_paged, returned, _ = _run(plan, fixture, arena=arena, decay=decay)

    assert returned is arena
    cols = state_dense.shape[-1]
    assert torch.equal(state_dense, state_control)
    assert torch.equal(state_dense.reshape(B * H, -1, cols), arena.materialize())
    bound = _reordering_bound(y_dense, plan)
    assert (y_dense - y_control).abs().max() <= bound
    assert (y_dense - y_paged).abs().max() <= bound


@pytest.mark.parametrize("decay_on", [False, True], ids=["decay-none", "leaf-mass"])
@pytest.mark.parametrize("normalization", ["raw", "global"])
def test_paging_is_bitwise_invisible_on_a_carried_state(decay_on, normalization):
    """Continuation, both decay arms: a carried state adopted into the arena.

    `adopt_dense` is the EXPLICIT bridge across the call boundary. What SHOULD
    cross it under paging is spec §7.3's question -- §7.1 says in terms that it does
    not settle continuation -- so this gate proves the bridge is lossless rather
    than asserting a continuation contract that has not been decided.
    """
    widths, d_v, BC = (4, 8, 8), 64, 16
    plan = _plan(widths, d_v, BT=32, BC=BC, normalization=normalization, decay=decay_on)
    first = _fixture(widths, d_v, p_nonzero=0.4, seed=3)
    second = _fixture(widths, d_v, p_nonzero=0.4, seed=4)
    B, T, H, _ = first[4].shape
    decay = _decay(plan.topology, "cuda", H) if decay_on else None

    _, carried, _ = _run(plan, first, decay=decay)
    y_dense, state_dense, _ = _run(plan, second, initial_state=carried, decay=decay)

    arena = _arena(plan, BH=B * H)
    cols = carried.shape[-1]
    arena.adopt_dense(carried.reshape(B * H, -1, cols).contiguous())
    y_control, _, _ = _run(plan, second, initial_state=carried, decay=decay)
    # `continue_paging=True`: `adopt_dense` just committed the carried state onto
    # this arena, and the call-scoped-by-default reset (fix pass BLOCKER B1) would
    # otherwise wipe it out from under this call.
    y_paged, _, _ = _run(plan, second, arena=arena, decay=decay, continue_paging=True)

    assert torch.equal(state_dense.reshape(B * H, -1, cols), arena.materialize())
    bound = _reordering_bound(y_dense, plan)
    assert (y_dense - y_control).abs().max() <= bound
    assert (y_dense - y_paged).abs().max() <= bound


@pytest.mark.parametrize("decay_on", [False, True], ids=["decay-none", "leaf-mass"])
@pytest.mark.parametrize("normalization", ["raw", "global"])
@pytest.mark.parametrize("BT", [16, 32])
def test_paging_is_bitwise_invisible_at_partial_residency(decay_on, normalization, BT):
    """The one cell `test_paging_is_bitwise_invisible`'s grid never reaches.

    Fix pass R1 (`workflows/20260730-wiring/paging-review.md`): every cell of the
    grid above -- and of `test_paging_is_bitwise_invisible_on_a_carried_state` --
    lands at 100% (or 127/128) resident, because the per-token live-owner union
    saturates. No gated cell therefore ever has a NON-RESIDENT owner, which is the
    one path "a non-resident atom reads zeros and skips its store" exists to
    exercise. This fixture -- measured by the
    reviewer at 90.6% resident with a nearby `(16,8,8) d_v=32 BC=16 p_nonzero=0.1
    T=16` cell (findings §04); this file's default fixture shape (`B=2, H=2`)
    needs `T=8` at the same widths/`p_nonzero` to land away from saturation
    (measured **87.1%** resident here) -- small and sparse enough that the union
    does not saturate, so this is the same gate as `test_paging_is_bitwise_invisible`
    run where it actually matters, over the same norm x decay x BT grid. `BT`
    cannot move the owner set (`test_both_BT_values_commit_the_same_owner_set`), so
    the residency fraction is identical across all 8 cells of this grid.
    """
    widths, d_v, BC = (16, 8, 8), 32, 16
    plan = _plan(widths, d_v, BT=BT, BC=BC, normalization=normalization, decay=decay_on)
    fixture = _fixture(widths, d_v, p_nonzero=0.1, seed=23, T=8)
    B, T, H, _ = fixture[4].shape
    decay = _decay(plan.topology, "cuda", H) if decay_on else None

    y_dense, state_dense, _ = _run(plan, fixture, decay=decay)
    y_control, state_control, _ = _run(plan, fixture, decay=decay)
    arena = _arena(plan, BH=B * H)
    y_paged, returned, _ = _run(plan, fixture, arena=arena, decay=decay)

    result = arena.last_result
    assert result.resident_pages < arena.logical_pages, (
        "this fixture must exercise partial residency, or it is not testing what "
        "R1 requires; if this fails the fixture saturated and needs to shrink "
        "further (paging-review.md §04)")
    assert result.resident_fraction < 0.95

    assert returned is arena, "under paging the arena IS the carried state"
    cols = state_dense.shape[-1]
    assert torch.equal(state_dense, state_control), (
        "the state must be deterministic; if this fails the readout comparison "
        "below is meaningless and the kernel, not paging, is what changed")
    assert torch.equal(state_dense.reshape(B * H, -1, cols), arena.materialize())

    bound = _reordering_bound(y_dense, plan)
    assert (y_dense - y_control).abs().max() <= bound
    assert (y_dense - y_paged).abs().max() <= bound


@pytest.mark.parametrize("decay_on", [False, True], ids=["decay-none", "leaf-mass"])
@pytest.mark.parametrize("BT", [16, 32])
@pytest.mark.parametrize("normalization", ["raw", "global"])
def test_at_BC64_an_owners_atoms_are_paged_INDEPENDENTLY(normalization, BT, decay_on):
    """The structural claim of the §4.5 re-key, asserted rather than inferred.

    `BC = 64` gives `QPO = 4`. The re-key's whole content is that those four atoms
    are SEPARATE residency decisions: a partially-written owner commits only the
    atoms its routing touches, and an owner-keyed kernel -- which resolves ONE base
    per owner and reads `BC` contiguous rows from it -- cannot express that. So this
    cell drives a fixture sparse enough that some owner is SPLIT (at least one atom
    resident and at least one not) and then asserts bit-identity through it.

    Without the split assertion this would silently degrade into
    `test_paging_is_bitwise_invisible`'s `BC64` cell whenever the fixture saturated,
    which is exactly the way the `BC = 16` grid managed to gate nothing.

    H5 (coverage-closure): promoted from a single `(BT=32, no-decay)` cell to the
    GRID ARM -- norm x BT x decay -- because the main grid's `BC = 64` cells
    saturate (every owner fully resident), so split owners were exercised on
    exactly one code-path combination while the kernel branches on all three axes.
    """
    widths, d_v, BC = (16, 8, 8), 32, 64
    plan = _plan(widths, d_v, BT=BT, BC=BC, normalization=normalization,
                 decay=decay_on)
    fixture = _fixture(widths, d_v, p_nonzero=0.1, seed=23, T=8)
    B, T, H, _ = fixture[4].shape
    decay = _decay(plan.topology, "cuda", H) if decay_on else None

    y_dense, state_dense, _ = _run(plan, fixture, decay=decay)
    y_control, _, _ = _run(plan, fixture, decay=decay)
    arena = _arena(plan, BH=B * H)
    y_paged, returned, _ = _run(plan, fixture, arena=arena, decay=decay)

    QPO = BC // MMA_K_QUANTUM
    assert QPO == 4
    present = (arena.page_table >= 0).reshape(B * H, -1, QPO)
    split = present.any(dim=-1) & ~present.all(dim=-1)
    assert bool(split.any()), (
        "no owner was SPLIT across its atoms, so this cell is not exercising what "
        "the re-key is for; the fixture saturated and needs to be sparser")

    assert returned is arena
    cols = state_dense.shape[-1]
    assert torch.equal(state_dense.reshape(B * H, -1, cols), arena.materialize())
    bound = _reordering_bound(y_dense, plan)
    assert (y_dense - y_control).abs().max() <= bound
    assert (y_dense - y_paged).abs().max() <= bound


# ---------------------------------------------------------------------------
# Commitment scales with realized usage, not with N
# ---------------------------------------------------------------------------

def test_committed_pages_track_realized_owners_not_N():
    """The claim the feature exists to make, asserted rather than assumed."""
    widths, d_v, BC = (8, 8, 8), 32, 16
    plan = _plan(widths, d_v, BT=32, BC=BC, normalization="raw", decay=False)
    fixture = _fixture(widths, d_v, p_nonzero=0.15, seed=7, T=32)
    B, T, H, _ = fixture[4].shape
    arena = _arena(plan, BH=B * H)
    _run(plan, fixture, arena=arena)
    result = arena.last_result

    assert 0 < result.resident_pages < arena.logical_pages
    assert result.resident_bytes < result.dense_bytes
    # The page set is exactly the WRITE-live owner set: reads never commit.
    assert result.resident_pages == int((arena.page_table >= 0).sum().item())


def test_a_dense_write_route_commits_every_owner():
    """The dense limit is the honest ceiling: paging must not under-commit there."""
    widths, d_v, BC = (4, 8, 8), 32, 16
    plan = _plan(widths, d_v, BT=32, BC=BC, normalization="raw", decay=False,
                 sparse_write=False)
    fixture = _fixture(widths, d_v, p_nonzero=1.0, seed=9)
    B, T, H, _ = fixture[4].shape
    arena = _arena(plan, BH=B * H)
    _run(plan, fixture, arena=arena)
    assert arena.last_result.resident_pages == arena.logical_pages


# ---------------------------------------------------------------------------
# The BT coupling (V14), settled parametric
# ---------------------------------------------------------------------------

def test_both_BT_values_commit_the_same_owner_set():
    """`BT` changes the tile GRANULARITY of `first_write_tile`, never WHICH owners
    are written. If it changed the page set, the coupling would not be a free
    parameter and D3's settlement would be wrong."""
    widths, d_v, BC = (4, 8, 8), 32, 16
    fixture = _fixture(widths, d_v, p_nonzero=0.35, seed=17, T=64)
    B, T, H, _ = fixture[4].shape
    tables = []
    for BT in (16, 32):
        plan = _plan(widths, d_v, BT=BT, BC=BC, normalization="raw", decay=False)
        arena = _arena(plan, BH=B * H)
        _run(plan, fixture, arena=arena)
        tables.append((arena.page_table >= 0).clone())
    assert torch.equal(tables[0], tables[1])


# ---------------------------------------------------------------------------
# Contract surfaces that must stay loud
# ---------------------------------------------------------------------------

def test_paging_refuses_an_implicit_dense_initial_state():
    widths, d_v, BC = (4, 8, 8), 32, 16
    plan = _plan(widths, d_v, BT=32, BC=BC, normalization="raw", decay=False)
    fixture = _fixture(widths, d_v, seed=2)
    B, T, H, _ = fixture[4].shape
    arena = _arena(plan, BH=B * H)
    read, write, g_read, g_write, v = fixture
    state = torch.zeros((B, H, plan.topology.N, d_v), device="cuda")
    with pytest.raises(ValueError, match="adopt_dense"):
        _consumer_forward(v, read, write, g_read, g_write, plan,
                            initial_state=state, paging=arena)


# ---------------------------------------------------------------------------
# BLOCKER B1 fix pass: the ops driver is call-scoped by default, and an explicit
# `continue_paging=True` is the only way to get a continuation
# (`workflows/20260730-wiring/paging-review.md`)
# ---------------------------------------------------------------------------

def test_a_reused_arena_does_not_silently_continue_by_default():
    """Reproduces the reviewer's adversarial sequence at the ops level: forward A
    on one fixture, then forward B on a DIFFERENT fixture reusing A's arena with
    no flag. The reused-arena result must match a cold arena's (and the unpaged
    path's) result -- not a continuation of A."""
    widths, d_v, BC = (4, 8, 8), 32, 16
    plan = _plan(widths, d_v, BT=32, BC=BC, normalization="raw", decay=False)
    fixture_a = _fixture(widths, d_v, p_nonzero=0.4, seed=31)
    fixture_b = _fixture(widths, d_v, p_nonzero=0.4, seed=32)
    B, T, H, _ = fixture_b[4].shape

    reused = _arena(plan, BH=B * H)
    _run(plan, fixture_a, arena=reused)                             # forward A
    y_reused, arena_out, _ = _run(plan, fixture_b, arena=reused)     # forward B, no flag

    y_cold, state_cold, _ = _run(plan, fixture_b)                    # unpaged: definitionally cold
    cold_arena = _arena(plan, BH=B * H)
    _run(plan, fixture_b, arena=cold_arena)

    assert arena_out is reused
    cols = state_cold.shape[-1]
    assert torch.equal(state_cold.reshape(B * H, -1, cols), reused.materialize()), (
        "a reused arena must compute the SAME function as a cold one by default -- "
        "if this fails the arena silently continued forward A's pages into forward "
        "B (BLOCKER B1)")
    assert torch.equal(reused.materialize(), cold_arena.materialize())
    bound = _reordering_bound(y_cold, plan)
    assert (y_cold - y_reused).abs().max() <= bound


def test_continue_paging_true_legitimately_continues_the_arena():
    """The OTHER direction of BLOCKER B1: freshness-by-default must not have made
    continuation unreachable. Forward A commits pages; forward B on the SAME arena
    with `continue_paging=True` must carry A's pages forward as a UNION (every page
    A committed that B alone would not have written must still be resident), and
    the resulting page set must differ from a cold B's -- proving the flag actually
    continues rather than silently resetting anyway.
    """
    # A small owner space (`widths=(4,8,8)`, `owners_total=16`) saturates too fast
    # for A and B to disagree on any owner at all -- this needs the same wider,
    # sparser shape `test_paging_is_bitwise_invisible_at_partial_residency` (R1)
    # uses to stay away from saturation (measured: 52/256 owners A-exclusive here).
    widths, d_v, BC = (16, 8, 8), 32, 16
    plan = _plan(widths, d_v, BT=32, BC=BC, normalization="raw", decay=False)
    fixture_a = _fixture(widths, d_v, p_nonzero=0.1, seed=41, T=8)
    fixture_b = _fixture(widths, d_v, p_nonzero=0.1, seed=42, T=8)
    B, T, H, _ = fixture_b[4].shape

    arena = _arena(plan, BH=B * H)
    _run(plan, fixture_a, arena=arena)
    pages_after_a = (arena.page_table >= 0).clone()
    resident_after_a = arena.last_result.resident_pages

    _, arena_out, _ = _run(plan, fixture_b, arena=arena, continue_paging=True)
    pages_after_continuation = (arena.page_table >= 0).clone()

    cold_arena = _arena(plan, BH=B * H)
    _run(plan, fixture_b, arena=cold_arena)
    pages_cold_b_only = (cold_arena.page_table >= 0)

    assert arena_out is arena
    assert arena.last_result.resident_pages >= resident_after_a
    a_exclusive = pages_after_a & ~pages_cold_b_only
    assert a_exclusive.any(), (
        "fixture produced no A-exclusive page; reseed so this gate means something")
    assert torch.equal(pages_after_continuation & a_exclusive, a_exclusive), (
        "continue_paging=True must keep every page forward A committed that "
        "forward B alone would not have written")
    assert not torch.equal(pages_after_continuation, pages_cold_b_only), (
        "continue_paging=True produced the same page set as a cold arena -- the "
        "flag is not actually continuing anything")


def test_continue_paging_requires_an_arena():
    widths, d_v, BC = (4, 8, 8), 32, 16
    plan = _plan(widths, d_v, BT=32, BC=BC, normalization="raw", decay=False)
    fixture = _fixture(widths, d_v, seed=51)
    read, write, g_read, g_write, v = fixture
    with pytest.raises(ValueError, match="continue_paging=True is meaningless"):
        _consumer_forward(v, read, write, g_read, g_write, plan,
                            continue_paging=True)


def test_a_fragmented_non_monotone_slot_arena_is_bitwise_invisible():
    """H5 (coverage-closure): bit-identity through an arena whose SLOT ORDER is
    scrambled relative to logical atom order.

    Every other cell plans FRESH, so slots are assigned in ascending logical atom
    id and `slot == rank(atom)` -- a page table any owner-keyed indexing bug can
    survive, because a wrong-but-monotone address maps to a plausibly-right slot.
    Here the schedule's EVEN atoms are pre-admitted alone (slots `0..k`), then the
    consumer's own `plan(fresh=False)` admits the odd remainder into LATER slots:
    the table interleaves low and high slot ids atom by atom, which is exactly the
    fragmentation a long-lived arena accumulates. The kernel must address through
    the table -- asserted by bit-identity of the materialized state against the
    unpaged dense run -- and the non-monotonicity is asserted so the cell cannot
    silently degrade into the fresh layout.
    """
    from rola.ops.paging import atom_bitmap_from_schedule

    widths, d_v, BC = (16, 8, 8), 32, 64
    plan = _plan(widths, d_v, BT=32, BC=BC, normalization="raw", decay=False)
    fixture = _fixture(widths, d_v, p_nonzero=0.35, seed=67, T=16)
    read, write, g_read, g_write, v = fixture
    B, T, H, _ = v.shape

    schedule = build_schedule(read, write, g_read, g_write, plan)
    arena = _arena(plan, BH=B * H)
    bitmap = atom_bitmap_from_schedule(schedule, arena.keying)
    assert int(bitmap.sum()) >= 8, "fixture too sparse to fragment anything"

    # Pre-admit only the EVEN-indexed touched atoms (flat order), fresh.
    flat = bitmap.reshape(-1)
    touched = flat.nonzero(as_tuple=True)[0]
    even = touched[::2]
    partial = torch.zeros_like(flat)
    partial[even] = True
    arena.plan_exact(partial.view_as(bitmap), fresh=True)

    # The consumer's own planning admits the odd remainder into LATER slots.
    y_dense, state_dense, _ = _run(plan, fixture)
    y_paged, returned, _ = _run(plan, fixture, arena=arena, continue_paging=True)
    assert returned is arena

    table = arena.page_table.reshape(-1)
    slots = table[touched]
    assert bool((slots >= 0).all()), "an admitted atom lost its slot"
    diffs = slots[1:] - slots[:-1]
    assert bool((diffs < 0).any()), (
        "the slot order came out monotone in atom id; the cell is not exercising "
        "a fragmented table and needs a different admission split")

    cols = state_dense.shape[-1]
    assert torch.equal(state_dense.reshape(B * H, -1, cols), arena.materialize()), (
        "a non-monotone slot layout moved the state: the kernel (or materialize) "
        "is addressing by atom RANK instead of through the page table")
    bound = _reordering_bound(y_dense, plan)
    assert (y_dense - y_paged).abs().max() <= bound


@pytest.mark.parametrize("P", [1, 2, 3])
def test_a_near_full_but_fitting_arena_survives_a_real_launch_at_P(P):
    """H6 (coverage-closure): the arena at EXACTLY its demand, through a real
    launch, crossed with §6.4's sequence parallelism.

    `max_physical_pages` is set to the schedule's exact atom demand -- one page of
    headroom less would raise -- so the allocator's ceiling arithmetic, the
    planned-exact popcount and the launch's addressing are all exercised at the
    boundary instead of in the roomy middle every other cell occupies. `P > 1`
    adds the three-pass path over the same paged state (paging and `P` had never
    been composed in any gate). The `P = 1` cell is bit-identity as usual; the
    `P > 1` cells hold paged-vs-unpaged AT THE SAME `P` to bit-identity of the
    state (no atomic touches it) and `y` to the derived envelope.
    """
    from rola.ops.paging import atom_bitmap_from_schedule
    from rola.routing.sequence_parallel import (
        SequenceParallelPlan,
        SequenceParallelPolicy,
        owner_slot_counts,
        select_sequence_parallelism,
    )

    widths, d_v, BC = (16, 8, 8), 32, 16
    plan = _plan(widths, d_v, BT=32, BC=BC, normalization="global", decay=False)
    fixture = _fixture(widths, d_v, p_nonzero=0.15, seed=91, T=16)
    read, write, g_read, g_write, v = fixture
    B, T, H, _ = v.shape
    schedule = build_schedule(read, write, None, g_write, plan)

    sp = None
    if P > 1:
        base = select_sequence_parallelism(
            plan, schedule, policy=SequenceParallelPolicy(workspace_cap=1 << 30))
        n_owners = schedule.owners.shape[0]
        Ps = tuple([P] * n_owners)
        sp = SequenceParallelPlan(
            P=Ps, limiting_gate=(None,) * n_owners, binding_gate=None,
            sms=base.sms, blocks_per_sm=base.blocks_per_sm,
            waves_before=base.waves_before, waves_after=0.0,
            base_traffic_bytes=0, roundtrip_bytes=0, roundtrip_seconds=0.0,
            base_traffic_seconds=0.0, ledger=base.ledger,
            n_slots=sum(owner_slot_counts(Ps)), BC=base.BC, cols=base.cols)

    probe = _arena(plan, BH=B * H)
    demand = int(atom_bitmap_from_schedule(schedule, probe.keying).sum())
    assert 0 < demand < probe.logical_pages, (
        "the fixture saturated (or wrote nothing); a near-full cell needs a real "
        "partial demand")

    y_dense, state_dense, _ = _run_sp(plan, fixture, sp=sp)
    arena = _arena(plan, BH=B * H, max_physical_pages=demand)
    y_paged, returned, _ = _run_sp(plan, fixture, arena=arena, sp=sp)

    assert returned is arena
    assert arena.resident_pages == demand == arena.max_physical_pages, (
        "the cell must run the arena AT its ceiling, or it is not the near-full "
        "gate it claims to be")
    cols = state_dense.shape[-1]
    assert torch.equal(state_dense.reshape(B * H, -1, cols), arena.materialize())
    bound = _reordering_bound(y_dense, plan)
    assert (y_dense - y_paged).abs().max() <= bound


def _run_sp(plan, fixture, *, arena=None, sp=None):
    """`_run`, with an optional pre-built sequence-parallel plan."""
    read, write, g_read, g_write, v = fixture
    global_norm = plan.normalization == "global"
    schedule = build_schedule(read, write, None if global_norm else g_read, g_write, plan)
    return _consumer_forward(
        v, read, write, None if global_norm else g_read, g_write, plan,
        schedule=schedule, paging=arena, sequence_parallel=sp)
