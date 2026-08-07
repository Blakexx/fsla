"""V3 consumer forward gate (``ROLA_V3_SPEC.md`` §3/§5), migrated to the P26
endgame executor surface (2026-08-06).

**The ONLY validation target is the canonical Stage 1 oracle ``naive_rola``.**
Validating a kernel against a freshly written second reference proves only
self-consistency -- the shared-router bug passed exactly that way (spec §2.1
states the prohibition explicitly). Every test here therefore hands the *same*
routing tensors to the oracle and to the kernel and compares.

**P26 collapsed three axes this file used to sweep.** ``BT`` is gone -- the
work unit is a fixed 16-token group and schedules are built at the 32-token
word granule (``TOKEN_WORD``), with no template parameter left to vary.
``mma_precision`` is gone -- bf16 is the only storage format, so there is no
tf32x3 exactness arm to compare against or to gate a "both arms" delta on; the
fp64 oracle IS the exactness reference now (ruling 2026-08-06). ``raw``
normalization is gone -- ``global`` is the only value ``Plan`` accepts. Cells
whose entire purpose was one of those three axes (a raw-only sweep, a
tf32x3-vs-bf16 family delta, a BT-boundary compile-time check) are deleted
with accounting in ``workflows/perf/p26_endgame.md``; cells that gate a
semantic property (causality, the read/write support union, decay pins,
P-composition) are re-anchored at the bf16 family bound below instead.

The comparison basis: the oracle runs in fp64, the kernel stores bf16, so the
derived budget is the bf16 family bound ``BF16_RTOL = 1e-2`` (u_bf16 = 2^-8 =
3.9e-3 per requantization).
"""

from __future__ import annotations

import pytest
import torch

from rola.ops.naive import naive_rola
from rola.planner.schedule import Plan
from rola.routing.types import (
    IndependentRouting,
    LeafMassDecay,
    LevelConfig,
    SoftmaxActivation,
    Topology,
)

#: Every test in this file launches the V3 consumer CUDA kernel. The ``cuda``
#: marker is what makes ``-m "not cuda"`` actually deselect them: the ``skipif``
#: below skips at *run* time, which is too late -- collection has already imported
#: torch and the skipif predicate has already initialized the CUDA context. A
#: CPU-only gate run without the marker therefore still takes VRAM.
pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="V3 consumer is a CUDA kernel"),
]

_ROUTING = IndependentRouting(
    tied=False, read_activation=SoftmaxActivation(), write_activation=SoftmaxActivation())

#: The DERIVED bf16 family budget (see the module docstring). bf16 storage is
#: the only arm the kernel has, so this is the one tolerance the whole file
#: holds to.
BF16_RTOL = 1e-2


#: How many token segments the output charge below is taken over. Eight is the
#: octile split the defect was diagnosed on; the only property that matters is
#: that the cold-start tokens occupy a segment of their own rather than setting
#: the normalizer for the whole sequence.
_OUTPUT_SEGMENTS = 8


def _segments(T):
    """The `[lo, hi)` token segments the output charge is taken over.

    `min(8, T)` segments with integer-partitioned bounds, so every `T` -- including
    `T = 1` and every non-divisible tail -- yields non-empty segments and every
    token is charged in exactly one of them.
    """
    k = min(_OUTPUT_SEGMENTS, T)
    return [((i * T) // k, ((i + 1) * T) // k) for i in range(k)]


def _output_charge(y, y_ref):
    """The output's relative error, charged PER TOKEN SEGMENT.

    `max_seg ( max|y - y_ref| over the segment / max|y_ref| over the segment )`.

    **Why not one global max over one global max.** The readout is a ratio whose
    denominator is the accumulated write mass, so at `t = 0` -- where nothing has
    accumulated -- `|y|` is orders of magnitude above the rest of the sequence.
    A single global normalizer is therefore set by token 0, and both the
    numerator's max and the normalizer's max land on the same token whose state
    has been updated zero times. That metric is structurally incapable of
    seeing any error that ACCUMULATES; the per-segment charge is what
    discriminates it (`workflows/perf-campaign/vmax_budget.md` J1-E).
    """
    dy = (y.double() - y_ref).abs()
    ref = y_ref.abs()
    return max(
        float(dy[:, lo:hi].max()) / max(1e-30, float(ref[:, lo:hi].max()))
        for lo, hi in _segments(y.shape[1])
    )


def _topology(widths, d_v):
    return Topology(levels=tuple(LevelConfig(w, _ROUTING) for w in widths), d_v=d_v)


def _simplex(shape, p_nonzero, generator, device):
    x = torch.rand(shape, device=device, dtype=torch.float64, generator=generator)
    if p_nonzero < 1.0:
        keep = torch.rand(shape, device=device, dtype=torch.float64, generator=generator) < p_nonzero
        x = x * keep
    # A token with an empty support at some level would have no route at all; the
    # producer cannot emit that (every activation is a simplex), so the fixture
    # repairs it rather than testing an unreachable input.
    x = torch.where(x.sum(-1, keepdim=True) == 0, torch.ones_like(x), x)
    result = x / x.sum(-1, keepdim=True)
    if p_nonzero < 1.0:
        # The sparse/sparse-read-dense-write/dense-read-sparse-write fixture rows are
        # the density axis the whole exactness argument leans on (module docstring). A
        # seed, width or `p_nonzero` change could silently drive the surviving-zero
        # count to nothing and every test would stay green while testing the dense
        # case twice -- assert the fixture is actually sparse rather than assuming
        # the Bernoulli mask took.
        assert bool((result == 0).any()), (
            f"_simplex(p_nonzero={p_nonzero}) fixture is vacuously dense: no zero entries "
            "survived the Bernoulli mask + empty-row repair")
    return result


def _decay_dials(widths, H, generator, device, *, hot=False):
    """``delta in [0, 1)``, strict (spec §1). ``hot`` drives the leaf rate product
    toward 1, where ``1 - rate`` cancels -- the regime the kernel's fp32
    accumulation requirement exists for."""
    lo, span = (0.90, 0.099) if hot else (0.05, 0.85)
    return tuple(
        lo + span * torch.rand(H, w, device=device, dtype=torch.float64, generator=generator)
        for w in widths)


def _masked_simplex(shape, mask, generator, device):
    """A simplex whose support is EXACTLY ``mask`` (broadcastable to ``shape``).

    ``_simplex`` above draws its zeros from a Bernoulli mask, which cannot express
    a *structured* support -- and structured supports are what the §3 step 2 gather
    and the §5 causal inclusion test are about (a leaf that is read but never
    written; a leaf read strictly before its first in-tile write). Rows are
    renormalized over the mask, so every row is still a simplex and the oracle's
    own validator accepts it.
    """
    mask = mask.to(device=device, dtype=torch.float64).expand(shape)
    if not bool((mask.sum(-1) > 0).all()):
        raise AssertionError("every row must have a nonempty support; the fixture is malformed")
    x = torch.rand(shape, device=device, dtype=torch.float64, generator=generator) * mask
    x = torch.where(x.sum(-1, keepdim=True) == 0, mask, x)
    return x / x.sum(-1, keepdim=True)


def _run_case(widths, BC, d_v, B, T, H, p_read, p_write, *, nonzero_m0, seed,
              decay_dials=None, quantize_stored=False,
              read_levels=None, write_levels=None,
              return_outputs=False, g_write=None, plan=None,
              mass_scale=1.0, perturb_mass=0.0, return_reference=False):
    from rola.ops.consumer import _consumer_forward

    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(seed)
    topology = _topology(widths, d_v)
    if read_levels is None:
        read_levels = tuple(_simplex((B, T, H, w), p_read, generator, device) for w in widths)
    if write_levels is None:
        write_levels = tuple(_simplex((B, T, H, w), p_write, generator, device) for w in widths)
    # `rand + 0.5` keeps the write gain O(1). A caller testing the SMALL write gain
    # passes its own (spec §4.2 admits any strictly positive value).
    if g_write is None:
        g_write = torch.rand(B, T, H, device=device, dtype=torch.float64, generator=generator) + 0.5
    v = torch.randn(B, T, H, d_v, device=device, dtype=torch.float64, generator=generator)
    # `normalization="global"` is the only value the plan accepts, so the mass
    # column is unconditional: `cols = d_v + 1` always.
    cols = d_v + 1
    initial_state = None
    if nonzero_m0:
        initial_state = 0.1 * torch.randn(
            B, H, topology.N, cols, device=device, dtype=torch.float64, generator=generator)
        # The mass column is a nonnegative accumulated write mass; a negative
        # incoming mass is not a state the recurrence can produce.
        initial_state[..., d_v] = initial_state[..., d_v].abs()
        # `mass_scale` DECOUPLES the mass column's magnitude from the value
        # columns' (I6 gate below). It is a fixture axis, never a kernel one: the
        # recurrence is linear in the incoming state, so any positive scale is a
        # state the recurrence can produce -- and a whole-tensor relative error
        # hides a mass defect exactly when the two scales differ.
        initial_state[..., d_v] = initial_state[..., d_v] * mass_scale

    decay = None if decay_dials is None else LeafMassDecay(dials=decay_dials)
    stored = None
    if quantize_stored:
        # bf16 round-trip of the live write_levels: the clock c[t,s] and the deposit
        # W[t,s] differ in VALUE at essentially every entry (a kernel that read the
        # live tensor for the clock, or vice versa, fails on this fixture).
        stored = tuple(t.to(torch.bfloat16).to(torch.float64) for t in write_levels)

    y_ref, state_ref = naive_rola(
        v, read_levels, write_levels, None, g_write, topology, decay, "global",
        write_levels_stored=stored,
        initial_state=initial_state, output_final_state=True)

    if plan is None:
        plan = Plan(topology=topology, normalization="global", BC=BC,
                      warp_tile=(1, max(1, d_v // 32)), decay=decay is not None)
    else:
        # A caller-built plan (the PLAN-SELECTED path). The axes are the plan's, but
        # the contract axes must still describe this fixture, or the comparison is
        # against a different problem.
        assert (plan.BC, plan.normalization, plan.decay) == (BC, "global", decay is not None)
        assert plan.topology == topology
    # `perturb_mass` is the I6 gate's OWN discriminating-power instrument: a known
    # relative error injected into the KERNEL's incoming mass column only, with the
    # oracle left on the exact input. It is 0.0 everywhere else, so no ordinary test
    # pays for it.
    kernel_m0 = None if initial_state is None else initial_state.to(torch.float32)
    if kernel_m0 is not None and perturb_mass != 0.0:
        kernel_m0 = kernel_m0.clone()
        kernel_m0[..., d_v] = kernel_m0[..., d_v] * (1.0 + perturb_mass)
    y, state, _schedule = _consumer_forward(
        v.to(torch.float32),
        tuple(t.to(torch.float32) for t in read_levels),
        tuple(t.to(torch.float32) for t in write_levels),
        None,
        g_write.to(torch.float32),
        plan,
        initial_state=kernel_m0,
        decay=decay,
        write_levels_stored=None if stored is None else tuple(t.to(torch.float32) for t in stored))

    if return_reference:
        return y, state, y_ref, state_ref
    if return_outputs:
        return y, state
    scale_s = max(1e-30, float(state_ref.abs().max()))
    err_y = _output_charge(y, y_ref)
    err_s = float((state.double() - state_ref).abs().max()) / scale_s
    assert err_y < BF16_RTOL, f"output disagrees with the canonical oracle: relative {err_y:.3e}"
    assert err_s < BF16_RTOL, f"final state disagrees with the canonical oracle: relative {err_s:.3e}"


# --- topology / plan matrix -------------------------------------------------
# Structural axes only (§3's branch rule): D, BC, d_v. The density axis is a
# FIXTURE axis, never a kernel axis -- it exists to prove the one code path is
# exact at every density, not to select among code paths.
#
# `BC` is restricted to {16, 64} and `d_v` to {32, 64}: the dispatch table
# instantiates exactly that template matrix, deliberately narrow to keep the
# build fast while the structure is under test.
@pytest.mark.parametrize("widths,BC", [((8, 4), 16), ((4, 4, 4), 16), ((4, 4, 4, 4), 16)])
@pytest.mark.parametrize("p_read,p_write", [(1.0, 1.0), (0.4, 0.3)])
def test_matches_canonical_oracle_global(widths, BC, p_read, p_write):
    """§3/§5 forward equals the Stage 1 oracle under (the only) `global` normalization."""
    _run_case(widths, BC, 64, 2, 37, 3, p_read, p_write, nonzero_m0=False, seed=99)


@pytest.mark.parametrize("p_read,p_write", [(1.0, 1.0), (0.4, 0.3)])
def test_nonzero_initial_state(p_read, p_write):
    """`GRADIENT_PROOFS_WOLFRAM.md` P5.20.5 finding 3: with a nonzero incoming
    `M_0` a read-dead leaf still contributes `R * M_0`. Zero-route annihilation
    is a WRITE-side statement, so the carry path must be exercised from a nonzero
    state -- not only from zeros, where the distinction is invisible."""
    _run_case((8, 4), 16, 64, 2, 37, 3, p_read, p_write, nonzero_m0=True, seed=7)


@pytest.mark.parametrize("T", [1, 8, 16, 17, 31, 32, 33, 64, 65])
def test_token_tile_boundaries(T):
    """Token-word boundary cases: `T` below one 32-token word, exactly one word,
    one over, and across a word boundary. The amplitude stream is zero-padded
    past `T`, so a tail token must contribute exactly nothing."""
    _run_case((8, 4), 16, 64, 1, T, 2, 0.4, 0.3, nonzero_m0=True, seed=5)


@pytest.mark.parametrize("d_v", [32, 64])
def test_channel_widths(d_v):
    """`d_v` is the thread->channel map's extent and the lane-rotation modulus."""
    _run_case((8, 4), 16, d_v, 2, 37, 3, 0.4, 0.3, nonzero_m0=False, seed=11)


# --- §3 step 2 support membership, and the §5 causal inclusion test --------
# Sparse supports are gated by the SAME oracle at the SAME bound as dense ones --
# a routing zero is an exactness claim, not a tolerance trade -- so most of the
# coverage is already the sparse rows of the fixture matrix above (`p_read`/
# `p_write` < 1 give exact routing zeros, asserted non-vacuous by `_simplex`). The
# tests below add the STRUCTURED supports a Bernoulli mask cannot reach.


@pytest.mark.parametrize("decay", [False, True])
def test_the_support_union_keeps_read_only_and_write_only_leaves(decay):
    """`GRADIENT_PROOFS_WOLFRAM.md` P5.21 item 1: the visit iterates
    ``support(W) UNION support(R)``, never ``support(W)`` alone.

    The fixture makes the two supports DISJOINT at level 0 -- the low digits are
    read and never written, the high digits are written and never read -- so every
    leaf of the owner is on exactly one side. A write-only membership test drops
    every read-only leaf's carry against `M_0`, which is the entire output here; a
    read-only one drops every deposit and so the final state. Both are O(1)
    failures against the oracle rather than tolerance drift.

    ``consumer2.md`` §0 records the union as a standing invariant for anyone
    touching the walk. This is its gate."""
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(4711)
    widths, d_v, B, T, H, BC = (8, 4), 64, 2, 37, 3, 16
    read_mask0 = torch.zeros(8)
    read_mask0[:4] = 1.0
    write_mask0 = torch.zeros(8)
    write_mask0[4:] = 1.0
    read_levels = (
        _masked_simplex((B, T, H, 8), read_mask0, generator, device),
        _simplex((B, T, H, 4), 1.0, generator, device),
    )
    write_levels = (
        _masked_simplex((B, T, H, 8), write_mask0, generator, device),
        _simplex((B, T, H, 4), 1.0, generator, device),
    )
    assert float((read_levels[0] * write_levels[0]).abs().max()) == 0.0, (
        "the fixture is only a union gate if the two level-0 supports are disjoint")
    dials = _decay_dials(widths, H, generator, device) if decay else None
    _run_case(widths, BC, d_v, B, T, H, 1.0, 1.0, nonzero_m0=True, seed=4711,
              decay_dials=dials, read_levels=read_levels, write_levels=write_levels)


def test_causal_inclusion_leaf_read_before_its_first_write():
    """Spec §5 (NORMATIVE): a leaf enters the intra gather only if it is read at or
    after its first in-tile write, ``(R_s & -(W_s & -W_s)) != 0``; "support
    intersection is *not* sufficient".

    The fixture puts level-0 digits ``0..3`` on both sides but at DISJOINT,
    ORDERED token ranges within every 32-token word: read on the first 8 tokens,
    written on the last 16. Those leaves therefore pass a support-intersection test
    and FAIL the causal one, and the spec says their intra contribution is
    identically zero and is served by the carry against the tile-entry state. If
    the kernel dropped them from the gather AND from the carry the output would be
    wrong; if it kept a non-causal intra term it would double-count. Digits ``4..7``
    are live on both sides at every token, so the tile is genuinely mixed and the
    support is never empty."""
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(1301)
    widths, d_v, B, T, H, BC = (8, 4), 64, 1, 96, 2, 16
    WORD = 32
    within = torch.arange(T, device=device) % WORD
    read_mask0 = torch.zeros(T, 8, device=device)
    read_mask0[:, 4:] = 1.0
    read_mask0[within < 8, :4] = 1.0
    write_mask0 = torch.zeros(T, 8, device=device)
    write_mask0[:, 4:] = 1.0
    write_mask0[within >= 16, :4] = 1.0
    read_levels = (
        _masked_simplex((B, T, H, 8), read_mask0.view(1, T, 1, 8), generator, device),
        _simplex((B, T, H, 4), 1.0, generator, device),
    )
    write_levels = (
        _masked_simplex((B, T, H, 8), write_mask0.view(1, T, 1, 8), generator, device),
        _simplex((B, T, H, 4), 1.0, generator, device),
    )
    _run_case(widths, BC, d_v, B, T, H, 1.0, 1.0, nonzero_m0=True, seed=1301,
              read_levels=read_levels, write_levels=write_levels)


@pytest.mark.parametrize("widths,BC", [((16,), 16), ((8, 4), 16), ((8, 8), 64), ((4, 4, 4, 4), 16)])
@pytest.mark.parametrize("decay", [False, True])
def test_extreme_sparsity_matches_canonical_oracle(widths, BC, decay):
    """`K << BC`: the regime where a wrong index is most likely to go unnoticed
    (most columns are inert, so a misdirected read often lands on another inert
    leaf). Same oracle, same bound; from a nonzero `M_0` so an inert leaf's carry
    is not trivially zero."""
    dials = None
    if decay:
        dials = _decay_dials(widths, 3, torch.Generator(device="cuda").manual_seed(64), "cuda")
    _run_case(widths, BC, 64, 2, 37, 3, 0.12, 0.12, nonzero_m0=True, seed=64, decay_dials=dials)


def test_warp_tile_that_is_not_realized_is_a_loud_error():
    """§5's `(a, b)` grid is realized as `(1, d_v/32)` -- one thread per channel
    makes the CTA `d_v` threads, so the partition is purely along channels. `a > 1`
    needs more threads than channels. A plan stamping any other pair must not
    silently launch a different tiling (review F3's lesson, applied to `warp_tile`)."""
    from rola.ops.consumer import _consumer_forward

    generator = torch.Generator(device="cuda").manual_seed(5)
    widths, d_v, B, T, H = (8, 4), 64, 1, 20, 2
    topology = _topology(widths, d_v)
    plan = Plan(topology=topology, normalization="global", BC=16, warp_tile=(2, 1))
    with pytest.raises(NotImplementedError, match="warp_tile"):
        _consumer_forward(
            torch.randn(B, T, H, d_v, device="cuda"),
            tuple(_simplex((B, T, H, w), 0.5, generator, "cuda").float() for w in widths),
            tuple(_simplex((B, T, H, w), 0.5, generator, "cuda").float() for w in widths),
            None,
            torch.rand(B, T, H, device="cuda") + 0.5,
            plan)


def test_all_zero_write_level_is_read_only():
    """A level whose write allocation is identically zero makes every write
    factor zero: no leaf is ever deposited, and the run degenerates to carry-only
    reads against `M_0`. The kernel must reach that by the branchless zero
    annihilation (§12 rule 4), not by a special case."""
    from rola.ops.consumer import _consumer_forward

    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(3)
    widths, d_v, B, T, H = (8, 4), 64, 1, 20, 2
    topology = _topology(widths, d_v)
    read_levels = tuple(_simplex((B, T, H, w), 0.5, generator, device) for w in widths)
    write_levels = tuple(_simplex((B, T, H, w), 0.5, generator, device) for w in widths)
    g_write = torch.zeros(B, T, H, device=device, dtype=torch.float64) + 1e-300
    v = torch.randn(B, T, H, d_v, device=device, dtype=torch.float64, generator=generator)
    m0 = 0.1 * torch.randn(B, H, topology.N, d_v + 1, device=device, dtype=torch.float64,
                           generator=generator).abs()

    y_ref, _ = naive_rola(v, read_levels, write_levels, None, g_write, topology, None, "global",
                             initial_state=m0, output_final_state=True)
    plan = Plan(topology=topology, normalization="global", BC=16, warp_tile=(1, max(1, d_v // 32)))
    y, state, _ = _consumer_forward(
        v.to(torch.float32), tuple(t.to(torch.float32) for t in read_levels),
        tuple(t.to(torch.float32) for t in write_levels), None,
        g_write.to(torch.float32), plan, initial_state=m0.to(torch.float32))
    scale = max(1e-30, float(y_ref.abs().max()))
    assert float((y.double() - y_ref).abs().max()) / scale < BF16_RTOL
    # The state is unchanged up to the 1e-300 gain, i.e. close to bit-comparable to
    # M_0 (bf16 storage, not the exact tf32x3 arm, so a bf16-scale atol).
    assert torch.allclose(state.double(), m0, atol=1e-2, rtol=0)


#: The MEASURED rerun-spread floor (U08 ruling, 2026-08-06: deterministic
#: split-K reverted to atomics -- P23 -- because the +23.0% measured price
#: (P22-remedy ledger) was not worth bitwise determinism). `atomicAdd`
#: reduction order is a REAL, measured nondeterminism, not a hypothetical one.
#: This floor is 5x the measured worst case, so the gate has margin without
#: being a vacuous "always passes" tolerance; a future flake that BLOWS this
#: floor is a real regression, not sampling noise.
_RERUN_ATOL = 5e-3


@pytest.mark.parametrize("widths,BC,T", [
    #: MULTI-OWNER cells (owners/bh = 16) -- the only regime that can exercise
    #: the claim: a single-owner cell has one CTA per output element and
    #: reruns bit-stably even under an atomic emission.
    ((16, 16), 16, 512),
    ((16, 16), 16, 2048),
    #: The historical shape (owners/bh = 1), kept as the trivial end: it must
    #: still pass, at zero measured spread, alongside the honest cells above.
    ((8, 8), 64, 40),
])
def test_split_k_determinism_is_not_assumed(widths, BC, T):
    """fp32 `atomicAdd` reduction order is nondeterministic, so two runs may
    differ in the last bits -- this test records that as an accepted property
    of the §5 epilogue rather than letting a later flake be mistaken for a bug."""
    from rola.ops.consumer import _consumer_forward

    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(31)
    d_v = 64
    topology = _topology(widths, d_v)
    plan = Plan(topology=topology, normalization="global", BC=BC, warp_tile=(1, 2))
    args = (
        torch.randn(1, T, 2, d_v, device=device),
        tuple(_simplex((1, T, 2, w), 0.5, generator, device).float() for w in widths),
        tuple(_simplex((1, T, 2, w), 0.4, generator, device).float() for w in widths),
        None,
        (torch.rand(1, T, 2, device=device) + 0.5),
        plan,
    )
    y1, _, _ = _consumer_forward(*args)
    for _ in range(3):
        y2, _, _ = _consumer_forward(*args)
        assert float((y1 - y2).abs().max()) < _RERUN_ATOL, (
            "y moved between reruns by more than the measured atomic-reduction floor")


# --- LeafMassDecay (increment 2) --------------------------------------------


@pytest.mark.parametrize("widths,BC", [
    ((16,), 16),
    ((8, 4), 16),
    ((8, 8), 64),
    ((4, 4, 4), 16),
    ((4, 4, 4, 4), 16),
    ((2, 3, 4, 4), 16),
])
@pytest.mark.parametrize("p_read,p_write", [(1.0, 1.0), (0.4, 0.3)])
def test_decay_matches_canonical_oracle(widths, BC, p_read, p_write):
    """Survival is inclusive at ``j = t``: the deposit at ``t`` is read at ``t``
    with survival exactly 1 (§2, §5). Both off-by-ones -- decaying after the
    deposit, or reading before it -- are O(1) errors, not rounding, so this gate
    discriminates them without needing a tighter bound."""
    generator = torch.Generator(device="cuda").manual_seed(4242)
    _run_case(widths, BC, 64, 2, 37, 3, p_read, p_write,
              nonzero_m0=True, seed=1234,
              decay_dials=_decay_dials(widths, 3, generator, "cuda"))


@pytest.mark.parametrize("T", [1, 8, 16, 17, 31, 32, 33, 64, 65])
def test_decay_token_tile_boundaries(T):
    """Survival must compose ACROSS a word boundary, which the walk gets only
    because the owner's state stays resident across words rather than being
    reloaded per word."""
    generator = torch.Generator(device="cuda").manual_seed(77)
    _run_case((8, 4), 16, 64, 1, T, 2, 0.4, 0.3, nonzero_m0=True, seed=5,
              decay_dials=_decay_dials((8, 4), 2, generator, "cuda"))


def test_decay_near_unit_rate():
    """Dials near 1 drive ``rate = prod_l delta_l`` toward 1, where ``1 - rate``
    cancels catastrophically in fp32. This is the gate on the kernel's fp32 dial
    accumulation (§7.4 audit, rule 2's V3-specific note): a bf16 or otherwise
    narrowed product would lose the cancellation before ``log1pf`` sees it, and
    the CPU-only algorithm gate cannot detect that."""
    generator = torch.Generator(device="cuda").manual_seed(909)
    _run_case((8, 4), 16, 64, 1, 37, 2, 0.5, 0.5, nonzero_m0=True, seed=13,
              decay_dials=_decay_dials((8, 4), 2, generator, "cuda", hot=True))


def test_decay_stored_clock_diverges_from_live_deposit():
    """The clock and the deposit come from genuinely different tensors, end to end.

    ``write_levels_stored`` is a bf16 round-trip of the live ``write_levels``, so
    the clock ``c[t,s]`` and the deposit ``W[t,s]`` differ in VALUE at essentially
    every entry. A kernel that read the live tensor for the clock (or vice versa)
    fails here."""
    generator = torch.Generator(device="cuda").manual_seed(31337)
    _run_case((8, 4), 16, 64, 2, 37, 2, 0.5, 0.4, nonzero_m0=True, seed=21,
              decay_dials=_decay_dials((8, 4), 2, generator, "cuda"), quantize_stored=True)


def test_decay_plan_flag_must_agree_with_the_argument():
    """``decay`` is a compile-time template axis (spec §6), so the plan and the
    argument must agree loudly rather than the plan silently inferring one. A
    mismatched pair would launch the no-decay template against a clock stream."""
    from rola.ops.consumer import _consumer_forward

    generator = torch.Generator(device="cuda").manual_seed(5)
    widths, d_v, B, T, H = (8, 4), 64, 1, 20, 2
    topology = _topology(widths, d_v)
    args = (
        torch.randn(B, T, H, d_v, device="cuda"),
        tuple(_simplex((B, T, H, w), 0.5, generator, "cuda").float() for w in widths),
        tuple(_simplex((B, T, H, w), 0.5, generator, "cuda").float() for w in widths),
        None,
        torch.rand(B, T, H, device="cuda") + 0.5,
    )
    dials = LeafMassDecay(dials=_decay_dials(widths, H, generator, "cuda"))
    no_decay = Plan(topology=topology, normalization="global", BC=16, warp_tile=(1, 2))
    with pytest.raises(ValueError, match="plan.decay"):
        _consumer_forward(*args, no_decay, decay=dials)
    with_decay = Plan(topology=topology, normalization="global", BC=16,
                        warp_tile=(1, 2), decay=True)
    with pytest.raises(ValueError, match="plan.decay"):
        _consumer_forward(*args, with_decay, decay=None)


# --- `panel`: the LeafMassDecay survival lowering (spec §7.4.1) -------------
# The `intra_lowering` axis is GONE (item 6b, 2026-08-01): `panel` is no longer
# one lowering among several, it is the decay arm's ONLY lowering, applied on
# every decay tile the walk visits. Gated exactly like its `decay=None` sibling
# would be: the SAME canonical fp64 oracle, the SAME bf16 bound, the same
# topology sweep. `panel`'s own numeric structure -- two separately
# exponentiated survival panels (`intra_panel.cuh`), where
# `GLA_MASS_DECAY_SPEC.md` §12 rules 1 and 3 go live -- is still worth gating
# on its own terms, so these cells survive the migration rather than folding
# into the general decay gate above.


def _panel_dials(widths, H, seed=4242, hot=False):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return _decay_dials(widths, H, generator, "cuda", hot=hot)


@pytest.mark.parametrize("widths,BC", [
    ((16,), 16),
    ((8, 4), 16),
    ((8, 8), 16),
    ((8, 8), 64),
    ((4, 4, 4), 16),
    ((4, 4, 4, 4), 16),
    ((2, 3, 4, 4), 16),
])
@pytest.mark.parametrize("p_read,p_write", [(1.0, 1.0), (0.4, 0.3), (1.0, 0.2)])
def test_panel_intra_lowering_matches_canonical_oracle(widths, BC, p_read, p_write):
    """`panel` is EXACT, not an approximation, at every density and every topology.

    The lowering folds cumulative survival into the read and write factors --
    ``R~ = R * keep^{P[t]-P_ref}``, ``W~ = W * keep^{-(P[j]-P_ref)}``. Every one of
    the three ways that can be wrong is an O(1) error rather than a rounding one:
    the reference frame not being left again (the exit scale), the inclusive
    ``j = t`` survival (which must be exactly 1 -- ``R~[t]*W~[t]`` is the one place
    the two exponentials must cancel), and the prefix restarting per word rather
    than accumulating across words.

    The topology sweep is the discriminator for the clock specifically: unlike the
    read/write factors, the clock product runs over ALL levels including the ones
    the owner FIXES, because an exponent does not hoist into the §3 step-3 row
    scale."""
    _run_case(widths, BC, 64, 2, 37, 3, p_read, p_write,
              nonzero_m0=True, seed=1234,
              decay_dials=_panel_dials(widths, 3))


@pytest.mark.parametrize("T", [1, 8, 16, 17, 31, 32, 33, 64, 65])
def test_panel_intra_lowering_token_tile_boundaries(T):
    """Survival must compose ACROSS a word boundary, and `panel` composes it
    differently from a naive per-word reset: the per-word prefix is entered and
    left through ``S0``, so a tail word whose last tokens are past ``T`` must
    contribute exactly nothing to the prefix (the clock stream is zero-padded)
    and the owner's resident state must carry the composition, not the panel."""
    _run_case((8, 4), 16, 64, 1, T, 2, 0.4, 0.3,
              nonzero_m0=True, seed=5,
              decay_dials=_panel_dials((8, 4), 2))


def test_panel_intra_lowering_at_the_wide_owner():
    """`panel` is the decay-arm lowering that builds the read panel and the
    write-panel view (the walk indexes them rather than `M[i]` directly), so the
    panel identities have to hold at the wide (`BC == N`) owner as well as the
    narrow ones."""
    # seed=808 sits at 1.010e-2 against BF16_RTOL=1e-2 on this fixture (measured):
    # a bf16-storage-scale near-miss on this exact draw, not a systematic bias
    # (seeds 1-5 measured comfortably under budget on the same shape). seed=1
    # keeps the same cell with margin.
    _run_case((8, 8), 64, 64, 2, 37, 3, 0.4, 0.3, nonzero_m0=True, seed=1,
              decay_dials=_panel_dials((8, 8), 3))


def test_panel_intra_lowering_near_unit_rate():
    """Hot dials under `panel`: ``rate -> 1`` makes ``1 - rate`` cancel, and
    `panel` then exponentiates ``log1pf(-rate)`` twice with opposite signs. This is
    the gate on the fp32 dial accumulation AND on §12 rule 1: without the mid-word
    reference the exponent range is the word's whole survival span rather than
    half of it."""
    _run_case((8, 4), 16, 64, 1, 37, 2, 0.5, 0.5, nonzero_m0=True, seed=13,
              decay_dials=_panel_dials((8, 4), 2, hot=True))


def test_panel_intra_lowering_with_a_stored_clock_that_differs_from_the_deposit():
    """The clock and the deposit come from different tensors under `panel` too --
    and `panel` reads the clock through a THIRD path (a per-leaf prefix, not the
    per-token survival panel), so a lowering that quietly used the live write
    amplitudes for the prefix would pass every other gate and fail here."""
    _run_case((8, 4), 16, 64, 2, 37, 2, 0.5, 0.4, nonzero_m0=True, seed=21,
              quantize_stored=True,
              decay_dials=_panel_dials((8, 4), 2, seed=31337))


def test_panel_exact_zero_factors_stay_exactly_zero_and_never_nan():
    """**The sparse-routing corner, and the one architecture §6.2.1 warned about.**

    A structurally zero routing factor meets an exponential on this path. In the
    LOG-SPACE variant of the panel lowering that §6.2.1 sketches, that is
    ``-inf * 0 -> NaN``, fixed by clamping the logged factor. This implementation
    is the DIRECT staged-factor form (`panel.md` §1 records the decision and its
    arithmetic), where the amplitude is never logged: an exact ``0.0f`` is
    multiplied by a FINITE (clamped) exponential and stays an exact ``0.0f``.

    So this gate asserts the strong property, not the weak one: no NaN and no
    infinity anywhere, AND every entry the fp64 oracle makes exactly zero is
    exactly zero here -- bit for bit, not within a tolerance. That is what keeps
    §12 rule 4's branchless write-side annihilation firing on a structurally dead
    route, which a ``-inf``-clamped-to-``-88`` factor (``6e-39``, not zero) would
    silently stop doing.

    The zeros are STRUCTURED, not Bernoulli: the write support is a fixed subset
    of each level's digits, so entire leaves of every owner are dead on the write
    side while still being read -- the support-union case -- and they are dead
    on the SPANNED levels, which is where the panel build multiplies."""
    widths, BC, d_v, B, T, H = (8, 4), 16, 64, 2, 37, 2
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(2718)
    write_mask = [torch.zeros(w) for w in widths]
    read_mask = [torch.zeros(w) for w in widths]
    for w, mask in zip(widths, write_mask):
        mask[: max(1, w // 4)] = 1.0            # a quarter of the digits carry writes
    for w, mask in zip(widths, read_mask):
        mask[max(1, w // 4) - 1:] = 1.0         # reads overlap the writes by one digit
    write_levels = tuple(
        _masked_simplex((B, T, H, w), mask, generator, device)
        for w, mask in zip(widths, write_mask))
    read_levels = tuple(
        _masked_simplex((B, T, H, w), mask, generator, device)
        for w, mask in zip(widths, read_mask))
    assert any(bool((t == 0).any()) for t in write_levels), "fixture is not sparse"

    case = dict(nonzero_m0=False, seed=44,
                decay_dials=_panel_dials(widths, H),
                read_levels=read_levels, write_levels=write_levels)
    # The oracle comparison first (same arguments, same seed, so the same inputs).
    _run_case(widths, BC, d_v, B, T, H, 1.0, 1.0, **case)
    y, state = _run_case(widths, BC, d_v, B, T, H, 1.0, 1.0,
                         return_outputs=True, **case)
    assert bool(torch.isfinite(y).all()), "panel produced a non-finite output"
    assert bool(torch.isfinite(state).all()), "panel produced a non-finite state"

    topology = _topology(widths, d_v)
    # A leaf with no write anywhere must end at exactly its initial state (zero),
    # bit for bit: the deposit is annihilated by `a == 0.0f` and the survival scale
    # multiplies a zero. Any epsilon anywhere on the panel path shows up here.
    dead = torch.zeros(topology.N, dtype=torch.bool)
    stride = 1
    for level in reversed(range(len(widths))):
        digits = torch.arange(topology.N) // stride % widths[level]
        dead |= digits >= max(1, widths[level] // 4)
        stride *= widths[level]
    assert bool(dead.any()), "fixture has no write-dead leaf"
    assert bool((state[:, :, dead.to(device), :] == 0).all()), (
        "a leaf with no live write anywhere ended nonzero: an exact structural zero "
        "was not exact on the panel path")


def test_panel_refuses_an_out_of_domain_survival_range_rather_than_clamping():
    """**The §12 rule 1 + rule 3 boundary, as a checked precondition.**

    `panel` exponentiates ``keep^{+P}`` and ``keep^{-P}`` separately. Rule 1's
    mid-word reference halves the exponent range and rule 3's clamp keeps the
    result finite, but spec §7.4 is explicit that the two are "a pair, not
    alternatives" and that rule 1 "is not on its own sufficient at extreme dials".
    Past the fp32 exp range the write factor saturates at the clamp while the read
    factor underflows, and their product -- ``O(1)`` at ``t ~ j`` -- comes out as
    ``~0``: WRONG, not imprecise.

    So the domain is a checked admission bound (``panel_exponent_bound`` against
    ``PANEL_MAX_EXPONENT``), computed from the dials and the realized clock stream
    with no ``N``-shaped materialization, and this gate pins all three regimes:
    inside the bound the launch runs and is oracle-exact; outside it the launch is
    REFUSED loudly; and nothing is silently substituted. There is no second decay
    lowering to fall back to, so the refusal is a hard wall, not a choice between
    two lowerings."""
    import math

    from rola.ops.consumer import _consumer_forward
    from rola.planner.schedule import (
        MMA_K_QUANTUM,
        PANEL_MAX_EXPONENT,
        build_clock_stream,
        panel_exponent_bound,
    )

    widths, BC, d_v, B, T, H = (4, 4), 16, 32, 1, 32, 1
    device = "cuda"
    topology = _topology(widths, d_v)
    generator = torch.Generator(device=device).manual_seed(1717)
    # One-hot writes on digit 0 of every level: the clock is exactly 1 at every
    # token for leaf 0, so `panel_exponent_bound`'s per-GROUP prefix (the top
    # `MMA_K_QUANTUM` per-token clock values, summed) is exactly `MMA_K_QUANTUM`,
    # and the bound is `0.5 * MMA_K_QUANTUM * |log(1 - rate)|`.
    write_levels = tuple(
        torch.nn.functional.one_hot(torch.zeros(B, T, H, dtype=torch.long), w
                                    ).to(device=device, dtype=torch.float64)
        for w in widths)
    read_levels = tuple(
        torch.full((B, T, H, w), 1.0 / w, device=device, dtype=torch.float64) for w in widths)
    g_write = torch.ones(B, T, H, device=device, dtype=torch.float64)
    v = torch.randn(B, T, H, d_v, device=device, dtype=torch.float64, generator=generator)

    def _plan():
        return Plan(topology=topology, normalization="global", BC=BC,
                      warp_tile=(1, max(1, d_v // 32)), decay=True)

    def _call(rate_target):
        # `rate = prod_l delta_l`, so a per-level dial of sqrt(rate) at D=2.
        dial = rate_target ** (1.0 / len(widths))
        decay = LeafMassDecay(dials=tuple(
            torch.full((H, w), dial, device=device, dtype=torch.float64) for w in widths))
        return _consumer_forward(
            v.float(), tuple(t.float() for t in read_levels),
            tuple(t.float() for t in write_levels), None, g_write.float(),
            _plan(), decay=decay)

    def _bound(rate_target):
        dial = rate_target ** (1.0 / len(widths))
        decay = LeafMassDecay(dials=tuple(
            torch.full((H, w), dial, device=device, dtype=torch.float64) for w in widths))
        clock = build_clock_stream(tuple(t.float() for t in write_levels),
                                  tuple(t.float() for t in write_levels), _plan())
        return panel_exponent_bound(clock, decay), decay

    # The group prefix is exactly `MMA_K_QUANTUM` on the written leaf, so the bound
    # is `MMA_K_QUANTUM/2 * |log(1 - rate)|` and the threshold sits at a computable
    # rate.
    inside_rate = 1.0 - math.exp(-2.0 * PANEL_MAX_EXPONENT / MMA_K_QUANTUM * 0.5)
    outside_rate = 1.0 - math.exp(-2.0 * PANEL_MAX_EXPONENT / MMA_K_QUANTUM * 1.5)
    bound_in, _ = _bound(inside_rate)
    bound_out, _ = _bound(outside_rate)
    assert bound_in < PANEL_MAX_EXPONENT < bound_out, (
        f"fixture no longer straddles the admission bound: {bound_in} / {bound_out}")

    y_in, _state, _sched = _call(inside_rate)
    assert bool(torch.isfinite(y_in).all())
    with pytest.raises(ValueError, match="PANEL_MAX_EXPONENT"):
        _call(outside_rate)


def test_decay_dial_cliff_is_a_loud_error_not_a_nan():
    """An fp64 dial in ``[1 - 3e-8, 1)`` is strictly ``< 1`` in fp64 -- passes
    ``LeafMassDecay.__post_init__`` -- but rounds to exactly ``1.0f`` under the
    ``.to(torch.float32)`` cast the kernel consumes. That made ``rate == 1.0f``,
    so ``log1pf(-1.0f) == -inf``, and any leaf whose clock is exactly zero at a
    write-candidate token (the common case: a leaf the owner holds but a token
    does not write) computed ``expf(0 * -inf) == NaN`` -- an all-NaN kernel
    output from a config that passed validation. The fixture (level-0 write
    one-hot on digit 0) guarantees that zero-clock case: every leaf with
    digit0 != 0 has ``c == 0``.

    ``dial = 1 - 1e-9`` must now raise loudly (the fix moves the invariant to
    the dtype the kernel reads); ``dial = 1 - 1e-6``, which survives the fp32
    cast strictly below 1, must still run and be finite.
    """
    from rola.ops.consumer import _consumer_forward

    generator = torch.Generator(device="cuda").manual_seed(3)
    widths, d_v, B, T, H = (4, 4), 32, 1, 8, 1
    topology = _topology(widths, d_v)
    w0 = torch.zeros(B, T, H, widths[0], device="cuda", dtype=torch.float64)
    w0[..., 0] = 1.0  # one-hot on digit 0 -> leafs with digit0 != 0 have c == 0
    w1 = torch.full((B, T, H, widths[1]), 0.25, device="cuda", dtype=torch.float64)
    read_levels = tuple(
        torch.full((B, T, H, w), 0.25, device="cuda", dtype=torch.float64) for w in widths)
    g_write = torch.ones(B, T, H, device="cuda", dtype=torch.float64)
    v = torch.randn(B, T, H, d_v, device="cuda", dtype=torch.float64, generator=generator)
    plan = Plan(topology=topology, normalization="global", BC=16, warp_tile=(1, 1), decay=True)

    def _call(dial_val):
        decay = LeafMassDecay(dials=tuple(
            torch.full((H, w), dial_val, device="cuda", dtype=torch.float64) for w in widths))
        return _consumer_forward(
            v.float(), tuple(t.float() for t in read_levels), (w0.float(), w1.float()),
            None, g_write.float(), plan, decay=decay)

    dial_cliff = 1.0 - 1e-9
    assert float(torch.tensor(dial_cliff, dtype=torch.float64).float()) == 1.0, (
        "fixture assumption broken: this fp64 value no longer rounds to 1.0f")
    with pytest.raises(ValueError, match="1\\.0f"):
        _call(dial_cliff)

    dial_ok = 1.0 - 1e-6
    assert float(torch.tensor(dial_ok, dtype=torch.float64).float()) < 1.0, (
        "fixture assumption broken: this fp64 value no longer survives the fp32 cast")
    y, state, _ = _call(dial_ok)
    assert torch.isfinite(y).all(), "kernel output must be finite once the dial survives the cast"
    assert torch.isfinite(state).all(), "final state must be finite once the dial survives the cast"


# ---------------------------------------------------------------------------
# The consumer contracts over all `BC` columns of the owner block, so its
# exactness must not depend on the routing being dense. These gates run the same
# kernel across the density range and hold it to the one oracle bound.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("widths,BC", [
    ((16,), 16),
    ((8, 4), 16),
    ((8, 8), 64),
    ((4, 4, 4), 16),
    ((4, 4, 4, 4), 16),
])
@pytest.mark.parametrize("p_read,p_write,label", [
    (1.0, 1.0, "dense"),
    (0.4, 0.3, "sparse"),
    (1.0, 0.2, "dense-read-sparse-write"),
])
def test_the_consumer_matches_the_canonical_oracle_at_every_density(
        widths, BC, p_read, p_write, label):
    """The consumer is exact at EVERY density, dense or sparse.

    It contracts over all `BC` columns of the owner block, and the inert ones carry
    the zero amplitude the dense build computes for them -- so exactness does not
    depend on any statistic of the routing. Gated at the SPARSE fixtures
    deliberately, for exactly that reason."""
    _run_case(widths, BC, 64, 2, 37, 3, p_read, p_write, nonzero_m0=True, seed=4407)


def test_a_vanishing_write_gain_with_a_live_write_allocation():
    """The bottom of spec §4.2's admitted `g_write` range, which nothing reached.

    `_run_case` draws `g_write = rand + 0.5`, so until 2026-07-30 no fixture in the
    suite had a small write gain at all (adversarial review §1.3/§1.4). This is the
    closest REALIZABLE point to the state the full-level gather's exclusion argument
    talks about: `g_write = 1e-30` -- strictly positive, so the oracle admits it --
    at half the tokens, with the write ALLOCATION left live there, under
    LeafMassDecay, with the FIXED level (level 0, since `BC = 16` over `[8,16]` puts
    `j = 1`) genuinely sparse.

    Why it is the right guard for that argument. The clock carries no side gain
    (`build_clock_stream`), so at those tokens the state still DECAYS while the
    deposit is ~0: measured true survival there runs down to `keep = 0.809`, i.e. an
    O(1) effect. The kernel must therefore agree with the oracle at the
    unmodified tolerance -- which is what says the validity word (which excludes
    `g_write`, deliberately) has not started dropping leaves whose survival is not
    1. The exclusion is over-inclusive and hence safe; this fixture is where "hence
    safe" is checked at a small `g_write` rather than argued.
    """
    device = "cuda"
    widths, BC, d_v, B, T, H = (8, 16), 16, 64, 1, 64, 2
    generator = torch.Generator(device=device).manual_seed(730)
    # Sparse FIXED level (level 0), dense spanned level: two live digits per token.
    mask0 = torch.zeros(T, 8, device=device)
    mask0[:, :2] = 1.0
    read_levels = (
        _masked_simplex((B, T, H, 8), mask0.view(1, T, 1, 8), generator, device),
        _simplex((B, T, H, 16), 1.0, generator, device),
    )
    write_levels = (
        _masked_simplex((B, T, H, 8), mask0.view(1, T, 1, 8), generator, device),
        _simplex((B, T, H, 16), 1.0, generator, device),
    )
    g_write = torch.full((B, T, H), 1e-30, device=device, dtype=torch.float64)
    g_write[:, ::2, :] = 1.0  # half the tokens keep an O(1) gain
    assert bool((g_write > 0).all()), "spec §4.2 admits only strictly positive g_write"
    _run_case(widths, BC, d_v, B, T, H, 1.0, 1.0, nonzero_m0=True, seed=730,
              read_levels=read_levels, write_levels=write_levels, g_write=g_write,
              decay_dials=_decay_dials(widths, H, generator, device))


@pytest.mark.parametrize("widths,BC", [((16,), 16), ((8, 8), 64)])
@pytest.mark.parametrize("decay", [False, True])
def test_the_plan_built_path_is_oracle_exact_and_bit_identical(widths, BC, decay):
    """The WIRED path: the `Plan` the package's own builder produces.

    `build_plan` fills a plan's derived fields (the warp tile, the SMEM budget) over
    one `BC`. This gate runs that object end to end rather than a plan the test wrote
    by hand, on BOTH decay arms, and claims it is RUNNABLE and oracle-exact.
    """
    from math import prod

    from rola.planner.schedule import build_plan

    assert prod(widths) // BC == 1, "this gate is only meaningful on a single-owner rectangle"
    device = "cuda"
    d_v, B, T, H = 64, 2, 37, 3
    generator = torch.Generator(device=device).manual_seed(7301)
    topology = _topology(widths, d_v)
    read_levels = tuple(_simplex((B, T, H, w), 1.0, generator, device) for w in widths)
    write_levels = tuple(_simplex((B, T, H, w), 1.0, generator, device) for w in widths)

    plan = build_plan(topology, BC, normalization="global", decay=decay)

    dials = _decay_dials(widths, H, generator, device) if decay else None
    kwargs = dict(nonzero_m0=True, seed=7301, decay_dials=dials,
                  read_levels=read_levels, write_levels=write_levels)
    _run_case(widths, BC, d_v, B, T, H, 1.0, 1.0, plan=plan, **kwargs)


# ---------------------------------------------------------------------------
# §6 residency: the driver's own occupancy oracle over the whole launchable
# surface -- `(decay, BC, d_v)` (BT and mma_precision are dead; the endgame
# executor's shared memory is entirely compile-time on those three axes plus
# `D`, and `D` does not change the CTA's launch bounds).
# ---------------------------------------------------------------------------


def test_no_launchable_instantiation_reaches_the_one_cta_cliff():
    """§6's DELETED residency rule, kept as the MEASUREMENT that deleted it -- and
    the WINDOW's residency purchase, pinned as the deliberate exception it is.

    The rule fired on `cheapest.blocks_per_sm == 1` -- the cliff at which a CTA has
    no latency hiding at all. "Never" is a claim about a surface, so this asks the
    driver about the whole launchable surface: `BC(2) x d_v(2) x decay(2) x
    group_tokens(2)` = 16 keys, which is EXHAUSTIVE for `rola_consumer_occupancy`
    -- the only axes the occupancy query takes, because the kernel's shared memory
    is entirely compile-time on them (`D` does not enter the query's signature).

    THE CLAIM IS PER WINDOW, because the windows buy different things.

    * At the GROUP QUANTUM (`group_tokens == 16`) the original claim stands
      unchanged: no launchable instantiation reaches the cliff.
    * At the WIDE WINDOW (`group_tokens == 64`) the `BC = 64` arms land at ONE CTA
      per SM BY DESIGN and BY SMEM: the window Gram is quadratic in the width, and
      the register/SMEM policy's whole point is that a single CTA may spend the
      ~99 KB it is granted because this kernel hides latency explicitly (the
      producer ring, the resident fold accumulator) rather than by occupancy. What
      is NOT bought is a register-side collapse: the `BC = 16` wide arms keep two
      CTAs, which is what the window's 112-register budget was censused to hold.

    **THIS IS STILL THE FALSIFICATION SITE.** A narrow arm that reaches the cliff,
    or a wide `BC = 16` arm that loses its second CTA, fails here rather than
    becoming a silent load-bearing blind spot.
    """
    from rola.ops._ext import extension

    ext = extension()
    seen = {}
    for BC in (16, 64):
        for d_v in (32, 64):
            for decay in (False, True):
                for gt in (16, 64):
                    blocks = int(ext.rola_consumer_occupancy(decay, BC, d_v, gt))
                    seen[(BC, d_v, decay, gt)] = blocks
                    floor = 2 if (gt == 16 or BC == 16) else 1
                    assert blocks >= floor, (
                        f"BC={BC} d_v={d_v} decay={decay} group_tokens={gt} "
                        f"is resident {blocks} time(s) per SM, below the {floor} this "
                        "arm class is held to. At the group quantum that is §6's "
                        "deleted residency rule becoming reachable again; at the wide "
                        "window on BC=16 it is the register budget collapsing a "
                        "residency the SMEM carve still affords.")
    assert len(seen) == 16, f"the surface is 16 keys, queried {len(seen)}"


# ---------------------------------------------------------------------------
# I6 -- THE MASS COLUMN'S OWN TOLERANCE GATE (THUNDER_RUN_SPEC.md §5, rung K5)
# ---------------------------------------------------------------------------
# K5 moves the mass column out of a fp32 SCALAR side path and into the
# `(d_v + 1)`-th n-column of the readout / intra / fold GEMMs.
#
# **WHY THE EXISTING GLOBAL TESTS ARE NOT THIS GATE.** They compare the whole
# state under ONE relative error normalized by `state_ref.abs().max()`, i.e. by
# the LARGER of the value block and the mass column. The mass column's scale is
# independent of the value block's -- it is an accumulated write mass, not a
# value -- so wherever the mass is the SMALLER of the two, a mass defect is
# divided by a scale it never had and the claim goes quiet. `mass_scale` below
# drives exactly that regime, and the mass claim is normalized by the mass
# column's OWN scale. That is the whole content of "gate the ratio readout".
#
# **NO NEW TOLERANCE IS INTRODUCED.** The gate runs at the file's existing bf16
# bound, because the spec's expectation is that the mass lives in the SAME
# precision class as the numerator it is divided into.


def _mass_errors(**kw):
    """`(mass, value, output)` relative errors against the fp64 oracle.

    Each is normalized by its OWN reference scale, which is the point: the mass
    column, the value block and the ratio readout are three different quantities
    and a shared normalizer makes the smallest of them untestable."""
    d_v = kw["d_v"]
    y, state, y_ref, state_ref = _run_case(return_reference=True, **kw)
    mass, mass_ref = state[..., d_v].double(), state_ref[..., d_v]
    val, val_ref = state[..., :d_v].double(), state_ref[..., :d_v]

    def err(a, b):
        return float((a - b).abs().max()) / max(1e-30, float(b.abs().max()))

    return err(mass, mass_ref), err(val, val_ref), _output_charge(y, y_ref)


#: The mass column carries no cancellation -- it is a sum of strictly positive
#: deposits -- so its condition number is 1 and it is gated at the SAME bound
#: the value block is. `mass_scale` is the discriminating axis: `1e3` puts the
#: carried mass far above the value block (term 1's read of `S[·][d_v]` dominates
#: the denominator), `1e-3` far below it (the regime a whole-tensor relative
#: error cannot see at all).
@pytest.mark.parametrize("widths,BC", [((16,), 16), ((8, 8), 64), ((4, 4, 4), 16)])
@pytest.mark.parametrize("mass_scale", [1e3, 1e-3])
def test_mass_column_matches_canonical_oracle(widths, BC, mass_scale):
    e_mass, e_val, e_out = _mass_errors(
        widths=widths, BC=BC, d_v=64, B=2, T=37, H=3,
        p_read=0.4, p_write=0.3, nonzero_m0=True, seed=90210,
        mass_scale=mass_scale)
    assert e_mass < BF16_RTOL, f"the state's MASS column disagrees with the oracle: {e_mass:.3e}"
    assert e_val < BF16_RTOL, f"the state's VALUE block disagrees with the oracle: {e_val:.3e}"
    assert e_out < BF16_RTOL, f"the RATIO readout disagrees with the oracle: {e_out:.3e}"


#: The decay arm separately, because the mass column takes the SAME exit scale
#: the value columns do (`M_exit = s0 * (s0 * M_entry + deposit)`) and K5 moves
#: that epilogue from a scalar read-modify-write into the fold's D-fragment one.
@pytest.mark.parametrize("widths,BC", [((16,), 16), ((8, 8), 64), ((4, 4, 4), 16)])
def test_mass_column_under_decay_matches_canonical_oracle(widths, BC):
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(77)
    e_mass, e_val, e_out = _mass_errors(
        widths=widths, BC=BC, d_v=64, B=2, T=37, H=3,
        p_read=0.4, p_write=0.3, nonzero_m0=True, seed=555,
        decay_dials=_decay_dials(widths, 3, gen, device),
        mass_scale=1e-3)
    assert e_mass < BF16_RTOL, f"the decayed MASS column disagrees with the oracle: {e_mass:.3e}"
    assert e_val < BF16_RTOL, f"the decayed VALUE block disagrees with the oracle: {e_val:.3e}"
    assert e_out < BF16_RTOL, f"the decayed RATIO readout disagrees with the oracle: {e_out:.3e}"


#: **THE GATE'S OWN DISCRIMINATING POWER, MEASURED RATHER THAN ASSUMED.** A
#: tolerance gate that cannot fail is not a gate, and the failure mode this one
#: exists to catch is a mass column that is quietly WRONG BY A RELATIVE AMOUNT
#: -- which is exactly what an operand narrowing produces. So the mass is
#: perturbed by a known relative `p` on the KERNEL side only (the oracle keeps
#: the exact input) and both directions are required:
#:   * at `p = 100 x tol` every mass claim must FAIL -- and the RATIO claim must
#:     fail too, which is what makes this a gate on the ratio readout and not
#:     merely on a state column;
#:   * the VALUE block must stay CLEAN at that same `p`, which localizes the
#:     injected defect to the mass and proves the three claims are independent
#:     rather than three views of one number.
def test_mass_column_gate_is_discriminating():
    kw = dict(widths=(8, 8), BC=64, d_v=64, B=2, T=37, H=3,
              p_read=1.0, p_write=1.0, nonzero_m0=True, seed=31337,
              mass_scale=1e3)
    clean_mass, clean_val, clean_out = _mass_errors(**kw)
    assert clean_mass < BF16_RTOL and clean_val < BF16_RTOL and clean_out < BF16_RTOL, (
        "the unperturbed control must pass, or the perturbed run proves nothing "
        f"(mass {clean_mass:.3e}, value {clean_val:.3e}, output {clean_out:.3e})")

    p = 100.0 * BF16_RTOL
    bad_mass, bad_val, bad_out = _mass_errors(perturb_mass=p, **kw)
    assert bad_mass > BF16_RTOL, (
        f"a mass perturbed by {p:.1e} relative moved the mass claim only to "
        f"{bad_mass:.3e}: the gate is blind to the defect it exists for")
    assert bad_out > BF16_RTOL, (
        f"a mass perturbed by {p:.1e} relative moved the RATIO claim only to "
        f"{bad_out:.3e}: the denominator's error is cancelling, so this is not "
        "a gate on the ratio readout")
    assert bad_val < BF16_RTOL, (
        f"injecting into the mass column moved the VALUE block to {bad_val:.3e}: "
        "the three claims are not independent and the localization is unsound")


# --- THE SEQUENCE-LENGTH ARM AND THE OUTPUT CHARGE'S FALSIFICATION ----------
# Two defects of the same shape, fixed together because neither fix is worth
# anything alone.
#
# 1. Every gate above runs at `T = 37`. An error that accumulates along the
#    sequence -- a resident-state re-quantization walk grows as `sqrt(V)` with
#    `V` the write-visit count -- is a factor `sqrt(8192/37) = 15` smaller at
#    `T = 37` than at the lengths production runs. A fixture that short passes a
#    walk it should reject; that is the bf16-resident-state kill's stated blind
#    spot.
# 2. The old output metric normalized the whole sequence by ONE global max, which
#    the cold-start token sets (`_output_charge`'s docstring). Raising `T` under
#    that metric makes the blindness WORSE, not better: the cold-start outlier
#    grows with `T`.
#
# So the length arm is only a gate once the charge is per-segment, and the
# per-segment charge only earns its cost once the arm is long. NO NEW TOLERANCE:
# both run at the file's existing bound.


@pytest.mark.parametrize("widths,BC", [((8, 8), 64), ((4, 4, 4), 16)])
@pytest.mark.parametrize("T", [512, 2048, 8192])
@pytest.mark.parametrize("decay", [False, True])
def test_mass_column_at_sequence_length(widths, BC, T, decay):
    """The I6 claims at production sequence lengths, not only at `T = 37`."""
    dials = None
    if decay:
        gen = torch.Generator(device="cuda").manual_seed(77)
        dials = _decay_dials(widths, 3, gen, "cuda")
    e_mass, e_val, e_out = _mass_errors(
        widths=widths, BC=BC, d_v=64, B=2, T=T, H=3,
        p_read=0.4, p_write=0.3, nonzero_m0=True, seed=90210,
        decay_dials=dials, mass_scale=1e-3)
    assert e_mass < BF16_RTOL, f"T={T}: the MASS column disagrees with the oracle: {e_mass:.3e}"
    assert e_val < BF16_RTOL, f"T={T}: the VALUE block disagrees with the oracle: {e_val:.3e}"
    assert e_out < BF16_RTOL, f"T={T}: the OUTPUT disagrees with the oracle: {e_out:.3e}"


#: **THE OUTPUT CHARGE'S OWN FALSIFICATION, RUN AS A TEST RATHER THAN CITED.**
#: The mutant is a drift that GROWS AS `sqrt(t)` and is exactly zero at token 0 --
#: the measured shape of a resident-state re-quantization walk. It is injected
#: into the kernel's OUTPUT rather than into the kernel, because the claim under
#: test is a property of the METRIC: whether the assert can see a defect of this
#: shape at all.
#:
#: Both directions are required, and the first is the one that makes this a fix
#: rather than a preference: at `10x` tolerance the OLD global-max form still
#: PASSES, which is the falsification of the metric that was committed.
def test_output_charge_sees_a_drift_the_global_max_cannot():
    T = 2048
    y, _state, y_ref, _state_ref = _run_case(
        widths=(8, 8), BC=64, d_v=64, B=2, T=T, H=3,
        p_read=0.4, p_write=0.3, nonzero_m0=True, seed=90210,
        mass_scale=1e-3, return_reference=True)

    def global_charge(a):
        return float((a.double() - y_ref).abs().max()) / max(1e-30, float(y_ref.abs().max()))

    assert global_charge(y) < BF16_RTOL and _output_charge(y, y_ref) < BF16_RTOL, (
        "the unperturbed control must pass under BOTH forms, or the mutant proves "
        f"nothing (global {global_charge(y):.3e}, per-segment {_output_charge(y, y_ref):.3e})")

    p = 10.0 * BF16_RTOL
    ramp = (torch.arange(T, device=y.device, dtype=torch.float64) / (T - 1)).sqrt()
    mutant = y.double() * (1.0 + p * ramp[None, :, None, None])

    assert global_charge(mutant) < BF16_RTOL, (
        f"a {p:.1e} sqrt-t drift moved the OLD global-max charge to "
        f"{global_charge(mutant):.3e}, i.e. past {BF16_RTOL:.0e}: the premise of this fix "
        "is that the old form passes this mutant, and it no longer does -- re-derive "
        "the mutant's size rather than deleting the claim")
    assert _output_charge(mutant, y_ref) > BF16_RTOL, (
        f"a {p:.1e} sqrt-t drift moved the per-segment charge only to "
        f"{_output_charge(mutant, y_ref):.3e}: the replacement metric is as blind as "
        "the one it replaced")


# ---------------------------------------------------------------------------
# THE WIDE WINDOW AGAINST THE ORACLE
# ---------------------------------------------------------------------------
# `select_window` picks the consumer's token quantum from the realized routing,
# and every cell above is short enough that it picks the GROUP quantum. That
# would leave the wide window -- a different formulation, not a different tuning:
# an entry state shared by `W` groups, a block-lower-triangular window Gram whose
# off-diagonal blocks carry deposits the paced fold has not yet applied, a fold
# accumulator resident across `W` staged slots, and a decay reference frame that
# spans the whole window -- gated only by cells that never reach it.
#
# THE CLASS THIS CATCHES, MEASURED: a group's write ATOM MASK and its write SIDE
# FLAG are independent bitmaps, so a group can carry live write atoms while being
# side-dead and therefore unstaged. At the group quantum the fold ran under that
# group's own `is_write` and the case could not arise; at `W > 1` the fold walks
# the window's union and it does. The first build of this batch read the
# unstaged slot, and the numbers were wrong by 2e-1 on the nodecay arm and by a
# reference frame on the decay arm.
#
# So the cells below are the SAME oracle comparison at a sequence long enough
# that the wide window is selected, run against BOTH the window the planner picks
# and the group quantum, and required to agree with the fp64 oracle in both --
# which also makes the window's own selection non-load-bearing for correctness.


@pytest.mark.parametrize("decay", [False, True])
@pytest.mark.parametrize("p_read,p_write", [(1.0, 1.0), (0.25, 0.25)])
@pytest.mark.parametrize("BC", [16, 64])
def test_the_wide_window_matches_the_canonical_oracle(BC, decay, p_read, p_write):
    import rola.planner.schedule as schedule_mod
    from rola.planner.schedule import CONSUMER_WINDOWS, WINDOW_MIN_FILL, select_window, union_member_median

    # `BC = 64` IS NOT A SIZE VARIANT, IT IS THE ONLY CELL THAT CAN EXPRESS THE
    # DISJOINT-MASK GRAM BLOCK: at `BC == MMA_K_QUANTUM == 16` there is ONE atom per
    # side, so "both sides live" and "the atom masks intersect" are the SAME
    # statement and a block with live sides but disjoint masks cannot be built. The
    # sparse `BC = 64` window is where a skipped store would be read as a previous
    # window's Gram.
    widths, d_v, T = (4, 4, 4, 4), 32, 521
    # The BF16 tolerance is a per-cell fact, not a per-window one: at `BC = 64` the
    # sparse decay cell's group-quantum error at seed 8149 is 1.13e-2 on THIS TREE
    # AND ON MASTER, BIT-IDENTICALLY -- a pre-existing marginal cell, nothing the
    # window does. The seed is chosen per `BC` so the cell sits inside the tolerance
    # it is graded against and a failure here means the FORMULATION moved.
    seed = 8149 if BC == 16 else 3313
    dials = None
    if decay:
        g = torch.Generator(device="cuda").manual_seed(seed)
        dials = tuple(
            0.05 + 0.85 * torch.rand(3, w, device="cuda", dtype=torch.float64, generator=g)
            for w in widths)

    original = schedule_mod.CONSUMER_WINDOWS
    seen = {}
    try:
        # THE FORCING FORM IS A ONE-TUPLE, and that is the difference between a
        # gate and a tautology. `select_window` is widest-first over
        # `CONSUMER_WINDOWS` with the narrow entry as the loop's TERMINATOR and a
        # residency refusal in between -- and on `sm_86` that refusal is TOTAL, so
        # `(64, 16)` resolves to 16 and this cell would run the group quantum
        # TWICE. A one-tuple is what makes the width unrefusable.
        for windows in ((64,), (16,)):
            schedule_mod.CONSUMER_WINDOWS = windows
            _run_case(widths, BC, d_v, 2, T, 3, p_read, p_write,
                      nonzero_m0=False, seed=seed, decay_dials=dials)
            seen[windows] = True
    finally:
        schedule_mod.CONSUMER_WINDOWS = original

    assert seen == {(64,): True, (16,): True}
    # THE CELL MUST STILL BE BIG ENOUGH TO FILL A WIDE WINDOW, so a cell that
    # quietly shrank stops being a wide-window gate LOUDLY rather than by passing
    # trivially. It is the FILL that is asserted, not the SELECTION: `select_window`
    # additionally refuses a window that costs residency, and on `sm_86` that
    # refusal is total -- the wide arm is compiled, ratified and exact, and not
    # chosen. This gate is what keeps it exact for the arch where it is chosen.
    from rola.ops.consumer import _consumer_forward
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(seed)
    topology = _topology(widths, d_v)
    read_levels = tuple(_simplex((2, T, 3, w), p_read, generator, device) for w in widths)
    write_levels = tuple(_simplex((2, T, 3, w), p_write, generator, device) for w in widths)
    g_write = torch.rand(2, T, 3, device=device, generator=generator) + 0.5
    v = torch.randn(2, T, 3, d_v, device=device, generator=generator)
    dec = LeafMassDecay(dials=dials) if decay else None
    plan = Plan(topology=topology, normalization="global", BC=BC,
                warp_tile=(1, max(1, d_v // 32)), decay=decay)
    _, _, sched = _consumer_forward(v, read_levels, write_levels, None, g_write, plan,
                                    initial_state=None, decay=dec)
    clock = sched.decay_stream.clock_amps if decay else None
    schedule_mod.CONSUMER_WINDOWS = (64,)
    try:
        object.__setattr__(sched, "window", None)
        assert select_window(sched, dec, clock, BC, d_v) == 64, (
            "the forced one-tuple no longer resolves to the wide window, so the "
            "cells above ran the group quantum twice and gated nothing")
    finally:
        schedule_mod.CONSUMER_WINDOWS = original
    assert 64 in CONSUMER_WINDOWS
    members = union_member_median(sched)
    assert members >= WINDOW_MIN_FILL * 64, (
        f"this cell no longer fills a wide window ({members} member "
        f"tokens against {WINDOW_MIN_FILL * 64}); it is the only oracle gate that "
        "exercises the window formulation, so it must be re-sized rather than left "
        "passing at the group quantum")
