"""V3 consumer forward gate (``ROLA_V3_SPEC.md`` §3/§5).

**The ONLY validation target is the canonical Stage 1 oracle ``naive_rola``.**
Validating a kernel against a freshly written second reference proves only
self-consistency -- the shared-router bug passed exactly that way (spec §2.1
states the prohibition explicitly). Every test here therefore hands the *same*
routing tensors to the oracle and to the kernel and compares.

The comparison basis is deliberate: the oracle runs in fp64 and the kernel in
fp32 with an fp32 split-K accumulation, so ``RTOL`` below is a genuine
fp32-accumulation tolerance, not a slack knob.

The MMA readout landed with ``mma_precision`` as a compile-time NUMERIC FORMAT
axis (findings D4.3), and the two arms are gated separately rather than under one
widened number:

* ``tf32x3`` -- the exactness arm, and the default everywhere in this file. It
  runs against the SAME ``RTOL`` the scalar readout did; it was not loosened by a
  single digit when the MMA landed, which is the whole point of gating a novel
  contraction at near-fp32 precision.
* ``bf16``   -- the performance arm, gated at its own ``BF16_RTOL`` on the same
  fixtures. That number is an honest format tolerance (bf16 carries 8 mantissa
  bits, so ~4e-3 relative per product), not a relaxation of the exactness gate.
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

#: fp32 accumulation against an fp64 oracle. Unchanged across the MMA increment:
#: the tf32x3 arm meets it. Tightened rather than loosened would be the correct
#: direction for any future change.
RTOL = 2e-5

#: The bf16 MMA arm's own tolerance. bf16 A/B operands carry 8 mantissa bits
#: (~4e-3 relative per product) and the readout contracts over `BC` of them, so
#: this is a FORMAT tolerance, not slack: it is what the chosen numeric format
#: costs, measured, and it applies only to `mma_precision='bf16'`.
BF16_RTOL = 1e-2


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
        # V12: the sparse/sparse-read-dense-write/dense-read-sparse-write fixture rows are
        # the density axis the whole exactness argument leans on (module docstring). A seed,
        # width or `p_nonzero` change could silently drive the surviving-zero count to
        # nothing and every test would stay green while testing the dense case twice --
        # assert the fixture is actually sparse rather than assuming the Bernoulli mask took.
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


def _run_case(widths, BC, BT, d_v, B, T, H, norm, p_read, p_write, *, nonzero_m0, seed,
              decay_dials=None, quantize_stored=False, mma_precision="tf32x3",
              read_levels=None, write_levels=None,
              compact=True, return_outputs=False, g_write=None, plan=None,
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
    # passes its own (spec §4.2 admits any strictly positive value, and nothing in
    # the suite exercised the bottom of that range until 2026-07-30).
    if g_write is None:
        g_write = torch.rand(B, T, H, device=device, dtype=torch.float64, generator=generator) + 0.5
    g_read = (None if norm == "global"
              else torch.rand(B, T, H, device=device, dtype=torch.float64, generator=generator) + 0.5)
    v = torch.randn(B, T, H, d_v, device=device, dtype=torch.float64, generator=generator)
    cols = d_v + (1 if norm == "global" else 0)
    initial_state = None
    if nonzero_m0:
        initial_state = 0.1 * torch.randn(
            B, H, topology.N, cols, device=device, dtype=torch.float64, generator=generator)
        if norm == "global":
            # The mass column is a nonnegative accumulated write mass; a negative
            # incoming mass is not a state the recurrence can produce.
            initial_state[..., d_v] = initial_state[..., d_v].abs()
            # `mass_scale` DECOUPLES the mass column's magnitude from the value
            # columns' (I6 gate below). It is a fixture axis, never a kernel one:
            # the recurrence is linear in the incoming state, so any positive scale
            # is a state the recurrence can produce -- and a whole-tensor relative
            # error hides a mass defect exactly when the two scales differ.
            initial_state[..., d_v] = initial_state[..., d_v] * mass_scale

    decay = None if decay_dials is None else LeafMassDecay(dials=decay_dials)
    stored = None
    if quantize_stored:
        # bf16 round-trip of the live write_levels: the clock c[t,s] and the deposit
        # W[t,s] differ in VALUE at essentially every entry (a kernel that read the
        # live tensor for the clock, or vice versa, fails on this fixture). It does
        # NOT reach P5.20.5 finding 2's actual condition (stored == 0 with live !=
        # 0) -- review finding F1 measured bf16's flush-to-zero threshold at
        # ~9.2e-41 against O(1/width) simplex entries, so that condition is
        # unreachable here (0 of 824 live-nonzero entries flush on this exact
        # fixture); it is reached instead by direct injection at the builder
        # boundary in
        # test_rola_v3_schedule.py::test_build_clock_stream_accepts_stored_zero_with_live_nonzero.
        stored = tuple(t.to(torch.bfloat16).to(torch.float64) for t in write_levels)

    y_ref, state_ref = naive_rola(
        v, read_levels, write_levels, g_read, g_write, topology, decay, norm,
        write_levels_stored=stored,
        initial_state=initial_state, output_final_state=True)

    if plan is None:
        plan = Plan(topology=topology, normalization=norm, BT=BT, BC=BC,
                      warp_tile=(1, max(1, d_v // 32)), decay=decay is not None,
                      mma_precision=mma_precision, compact=compact)
    else:
        # A caller-built plan (the PLAN-SELECTED path). The axes are the plan's, but
        # the contract axes must still describe this fixture, or the comparison is
        # against a different problem.
        assert (plan.BC, plan.normalization, plan.decay) == (BC, norm, decay is not None)
        assert plan.topology == topology
    # `perturb_mass` is the I6 gate's OWN discriminating-power instrument: a known
    # relative error injected into the KERNEL's incoming mass column only, with the
    # oracle left on the exact input. It is 0.0 everywhere else, so no ordinary test
    # pays for it.
    kernel_m0 = None if initial_state is None else initial_state.to(torch.float32)
    if kernel_m0 is not None and perturb_mass != 0.0 and norm == "global":
        kernel_m0 = kernel_m0.clone()
        kernel_m0[..., d_v] = kernel_m0[..., d_v] * (1.0 + perturb_mass)
    y, state, _schedule = _consumer_forward(
        v.to(torch.float32),
        tuple(t.to(torch.float32) for t in read_levels),
        tuple(t.to(torch.float32) for t in write_levels),
        None if g_read is None else g_read.to(torch.float32),
        g_write.to(torch.float32),
        plan,
        initial_state=kernel_m0,
        decay=decay,
        write_levels_stored=None if stored is None else tuple(t.to(torch.float32) for t in stored))

    if return_reference:
        return y, state, y_ref, state_ref
    if return_outputs:
        return y, state
    tol = RTOL if mma_precision == "tf32x3" else BF16_RTOL
    scale_y = max(1e-30, float(y_ref.abs().max()))
    scale_s = max(1e-30, float(state_ref.abs().max()))
    err_y = float((y.double() - y_ref).abs().max()) / scale_y
    err_s = float((state.double() - state_ref).abs().max()) / scale_s
    assert err_y < tol, f"output disagrees with the canonical oracle: relative {err_y:.3e}"
    assert err_s < tol, f"final state disagrees with the canonical oracle: relative {err_s:.3e}"


# --- topology / plan matrix -------------------------------------------------
# Structural axes only (§3's branch rule): D, BT, BC, d_v, normalization. The
# density axis is a FIXTURE axis, never a kernel axis -- it exists to prove the
# one code path is exact at every density, not to select among code paths.
#
# `BC` is restricted to {16, 64} and `d_v` to {32, 64}: increment 1 instantiates
# exactly that template matrix (see `v3_consumer.cu`'s dispatch), deliberately
# narrow to keep the build fast while the structure is under test. Widening the
# matrix is a build-time cost, not a design change.
#
# Each entry names a DIFFERENT owner rectangle, which is the point -- `j` and `c`
# are derived from `(widths, BC)` (§3), so this is the real structural axis:
#   (16,)      BC=16 -> j=0, c=16, 1 owner, no fixed level
#   (8,4)      BC=16 -> j=0, c=4,  2 owners, no fixed level
#   (8,8)      BC=16 -> j=0, c=2,  4 owners, no fixed level
#   (8,8)      BC=64 -> j=0, c=8,  1 owner (BC == N: span everything)
#   (4,4,4)    BC=16 -> j=1, c=4,  4 owners, ONE fixed level
#   (4,4,4,4)  BC=16 -> j=2, c=4, 16 owners, TWO fixed levels (the j+1 AND chain)
#   (2,3,4,4)  BC=16 -> j=2, c=4,  6 owners, two fixed, non-power-of-two widths
#   (8,4,2)    BC=16 -> j=0, c=2,  4 owners
#   (2,2,2,2)  BC=16 -> j=0, c=2,  1 owner, D at its cap
@pytest.mark.parametrize("widths,BC", [
    ((16,), 16),
    ((8, 4), 16),
    ((8, 8), 16),
    ((8, 8), 64),
    ((4, 4, 4), 16),
    ((4, 4, 4, 4), 16),
    ((2, 3, 4, 4), 16),
    ((8, 4, 2), 16),
    ((2, 2, 2, 2), 16),
])
@pytest.mark.parametrize("BT", [16, 32])
@pytest.mark.parametrize("p_read,p_write,label", [
    (1.0, 1.0, "dense"),
    (0.4, 0.3, "sparse"),
    (0.4, 1.0, "sparse-read-dense-write"),
    (1.0, 0.2, "dense-read-sparse-write"),
])
def test_matches_canonical_oracle_raw(widths, BC, BT, p_read, p_write, label):
    """§3/§5 forward equals the Stage 1 oracle under `raw` normalization."""
    _run_case(widths, BC, BT, 64, 2, 37, 3, "raw", p_read, p_write,
              nonzero_m0=False, seed=1234)


@pytest.mark.parametrize("widths,BC", [((8, 4), 16), ((4, 4, 4), 16), ((4, 4, 4, 4), 16)])
@pytest.mark.parametrize("p_read,p_write", [(1.0, 1.0), (0.4, 0.3)])
def test_matches_canonical_oracle_global(widths, BC, p_read, p_write):
    """`global` normalization: `num` and `den` are split-K accumulated separately
    and divided in an epilogue (decision D4.6), because `den` is only complete
    after every owner has landed."""
    _run_case(widths, BC, 32, 64, 2, 37, 3, "global", p_read, p_write,
              nonzero_m0=False, seed=99)


@pytest.mark.parametrize("norm", ["raw", "global"])
@pytest.mark.parametrize("p_read,p_write", [(1.0, 1.0), (0.4, 0.3)])
def test_nonzero_initial_state(norm, p_read, p_write):
    """`GRADIENT_PROOFS_WOLFRAM.md` P5.20.5 finding 3: with a nonzero incoming
    `M_0` a read-dead leaf still contributes `R * M_0`. Zero-route annihilation
    is a WRITE-side statement, so the carry path must be exercised from a nonzero
    state -- not only from zeros, where the distinction is invisible."""
    _run_case((8, 4), 16, 32, 64, 2, 37, 3, norm, p_read, p_write,
              nonzero_m0=True, seed=7)


@pytest.mark.parametrize("T", [1, 8, 16, 17, 31, 32, 33, 64, 65])
def test_token_tile_boundaries(T):
    """`BT` boundary cases: `T` below one tile, exactly one tile, one over, and
    across a word boundary. The amplitude stream is zero-padded past `T`, so a
    tail token must contribute exactly nothing."""
    _run_case((8, 4), 16, 32, 64, 1, T, 2, "raw", 0.4, 0.3, nonzero_m0=True, seed=5)


@pytest.mark.parametrize("d_v", [32, 64])
def test_channel_widths(d_v):
    """`d_v` is the thread->channel map's extent and the lane-rotation modulus."""
    _run_case((8, 4), 16, 32, d_v, 2, 37, 3, "raw", 0.4, 0.3, nonzero_m0=False, seed=11)


# --- MMA precision axis (findings D4.3) -------------------------------------
# The exactness arm (tf32x3) is the default on every test above, so it is already
# gated across the whole structural matrix. The bf16 arm gets its own sweep at its
# own FORMAT tolerance rather than a widened shared one.


@pytest.mark.parametrize("widths,BC", [((16,), 16), ((8, 4), 16), ((8, 8), 64), ((4, 4, 4), 16)])
@pytest.mark.parametrize("BT", [16, 32])
@pytest.mark.parametrize("norm", ["raw", "global"])
def test_bf16_mma_arm_matches_canonical_oracle(widths, BC, BT, norm):
    """The bf16 readout arm, against the SAME canonical oracle at `BF16_RTOL`.

    This is the arm on which `GLA_MASS_DECAY_SPEC.md` §12 rule 3 is live: the A/B
    operands narrow to bf16, so they are clamped to bf16's max finite magnitude
    (never to fp32 `FLT_MAX`, whose gap rounds to bf16 `inf`). Structural coverage
    is the k-step count (`BC/16`: one step at `BC=16`, four at `BC=64`), the
    m-tile count (`BT/16`), and both `d_v` warp counts via `norm`-independent
    fixtures; the fragment index map is what these discriminate.

    Run from a nonzero `M_0` so the CARRY term -- the only term this arm changes --
    is nonzero even on a read-only tile."""
    _run_case(widths, BC, BT, 64, 2, 37, 3, norm, 0.4, 0.3,
              nonzero_m0=True, seed=4242, mma_precision="bf16")


@pytest.mark.parametrize("d_v", [32, 64])
def test_bf16_mma_arm_channel_widths(d_v):
    """`d_v` sets the CTA's warp count (`W = d_v/32`) and hence which 8-channel
    column blocks each warp's fragments cover -- the axis a wrong `ntg` map breaks."""
    _run_case((8, 4), 16, 32, d_v, 2, 37, 3, "raw", 0.4, 0.3,
              nonzero_m0=True, seed=11, mma_precision="bf16")


# --- the `gram` intra lowering (findings §1a; the decay=None dense limit) ----


# --- §3 step 2 support compaction, and the §5 causal inclusion test ---------
# The compacted paths are gated by the SAME oracle at the SAME `RTOL` as the dense
# ones -- compaction is an exactness claim, not a tolerance trade -- so most of its
# coverage is already the sparse rows of the fixture matrix above (`p_read`/
# `p_write` < 1 give exact routing zeros, asserted non-vacuous by `_simplex`). The
# tests below add the STRUCTURED supports a Bernoulli mask cannot reach.


@pytest.mark.parametrize("widths,BC", [((16,), 16), ((8, 4), 16), ((8, 8), 64), ((4, 4, 4), 16)])
@pytest.mark.parametrize("BT", [16, 32])
@pytest.mark.parametrize("norm", ["raw", "global"])
def test_bf16_readout_under_decay_matches_canonical_oracle(widths, BC, BT, norm):
    """`mma_precision` is a live numeric-format axis on BOTH decay arms.

    A READ-ONLY tile under `LeafMassDecay` has `c = 0` on every leaf the owner
    holds, hence `keep = 1` EXACTLY (spec §2's write-conditioned quiescence), no
    deposit and no intra term -- so it takes the §5 tensor-core readout, and the
    format that readout narrows to is a real choice there. This test replaces the
    former `test_bf16_precision_is_rejected_under_decay_rather_than_ignored`,
    whose premise ("LeafMassDecay has no MMA readout") stopped being true with the
    compaction increment; a rejection kept past its own reason would be the
    review-F3 failure mode with the sign flipped.

    Run from a nonzero `M_0` so the carry -- the only term the format changes --
    is nonzero on the read-only tiles this arm exercises."""
    _run_case(widths, BC, BT, 64, 2, 37, 3, norm, 0.4, 0.3, nonzero_m0=True, seed=808,
              decay_dials=_decay_dials(
                  widths, 3, torch.Generator(device="cuda").manual_seed(808), "cuda"),
              mma_precision="bf16")


@pytest.mark.parametrize("norm", ["raw", "global"])
@pytest.mark.parametrize("decay", [False, True])
def test_union_compaction_keeps_read_only_and_write_only_leaves(norm, decay):
    """`GRADIENT_PROOFS_WOLFRAM.md` P5.21 item 1: the gather iterates
    ``support(W) UNION support(R)``, never ``support(W)`` alone.

    The fixture makes the two supports DISJOINT at level 0 -- the low digits are
    read and never written, the high digits are written and never read -- so every
    leaf of the owner is on exactly one side. A write-only compaction (the
    optimization P5.21 warns is invisible to a forward gate on a zero `M_0`) drops
    every read-only leaf's carry against `M_0`, which is the entire output here; a
    read-only compaction drops every deposit and so the final state. Both are
    O(1) failures against the oracle rather than tolerance drift.

    ``consumer2.md`` §0 records the union as a standing invariant for anyone
    touching the walk. This is its gate."""
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(4711)
    widths, d_v, B, T, H, BC, BT = (8, 4), 64, 2, 37, 3, 16, 32
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
    _run_case(widths, BC, BT, d_v, B, T, H, norm, 1.0, 1.0, nonzero_m0=True, seed=4711,
              decay_dials=dials, read_levels=read_levels, write_levels=write_levels)


@pytest.mark.parametrize("norm", ["raw", "global"])
def test_causal_inclusion_leaf_read_before_its_first_write(norm):
    """Spec §5 (NORMATIVE): a leaf enters the intra gather only if it is read at or
    after its first in-tile write, ``(R_s & -(W_s & -W_s)) != 0``; "support
    intersection is *not* sufficient".

    The fixture puts level-0 digits ``0..3`` on both sides but at DISJOINT,
    ORDERED token ranges within every ``BT = 32`` tile: read on the first 8 tokens,
    written on the last 16. Those leaves therefore pass a support-intersection test
    and FAIL the causal one, and the spec says their intra contribution is
    identically zero and is served by the carry against the tile-entry state. If
    the kernel dropped them from the gather AND from the carry the output would be
    wrong; if it kept a non-causal intra term it would double-count. Digits ``4..7``
    are live on both sides at every token, so the tile is genuinely mixed and the
    support is never empty."""
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(1301)
    widths, d_v, B, T, H, BC, BT = (8, 4), 64, 1, 96, 2, 16, 32
    within = torch.arange(T, device=device) % BT
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
    _run_case(widths, BC, BT, d_v, B, T, H, norm, 1.0, 1.0, nonzero_m0=True, seed=1301,
              read_levels=read_levels, write_levels=write_levels)


@pytest.mark.parametrize("widths,BC", [((16,), 16), ((8, 4), 16), ((8, 8), 64), ((4, 4, 4, 4), 16)])
@pytest.mark.parametrize("norm", ["raw", "global"])
@pytest.mark.parametrize("decay", [False, True])
def test_extreme_sparsity_matches_canonical_oracle(widths, BC, norm, decay):
    """`K << BC`: the regime compaction exists for, and the one where a wrong
    compacted index is most likely to go unnoticed (most columns are inert, so a
    misdirected gather often lands on another inert leaf). Same oracle, same
    `RTOL`; from a nonzero `M_0` so an inert leaf's carry is not trivially zero."""
    dials = None
    if decay:
        dials = _decay_dials(widths, 3, torch.Generator(device="cuda").manual_seed(64), "cuda")
    _run_case(widths, BC, 32, 64, 2, 37, 3, norm, 0.12, 0.12, nonzero_m0=True, seed=64,
              decay_dials=dials)


def test_warp_tile_that_is_not_realized_is_a_loud_error():
    """§5's `(a, b)` grid is realized as `(1, d_v/32)` -- one thread per channel
    makes the CTA `d_v` threads, so the partition is purely along channels. `a > 1`
    needs more threads than channels. A plan stamping any other pair must not
    silently launch a different tiling (review F3's lesson, applied to `warp_tile`)."""
    from rola.ops.consumer import _consumer_forward

    generator = torch.Generator(device="cuda").manual_seed(5)
    widths, d_v, B, T, H = (8, 4), 64, 1, 20, 2
    topology = _topology(widths, d_v)
    plan = Plan(topology=topology, normalization="raw", BT=32, BC=16, warp_tile=(2, 1))
    with pytest.raises(NotImplementedError, match="warp_tile"):
        _consumer_forward(
            torch.randn(B, T, H, d_v, device="cuda"),
            tuple(_simplex((B, T, H, w), 0.5, generator, "cuda").float() for w in widths),
            tuple(_simplex((B, T, H, w), 0.5, generator, "cuda").float() for w in widths),
            torch.rand(B, T, H, device="cuda") + 0.5,
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
    write_levels = [_simplex((B, T, H, w), 0.5, generator, device) for w in widths]
    # A structurally dead level: every digit zero. This is not a simplex, so the
    # oracle's validator would reject it -- instead drive the gain to zero, which
    # is the reachable form of "this token deposits nothing".
    write_levels = tuple(write_levels)
    g_write = torch.zeros(B, T, H, device=device, dtype=torch.float64) + 1e-300
    g_read = torch.rand(B, T, H, device=device, dtype=torch.float64, generator=generator) + 0.5
    v = torch.randn(B, T, H, d_v, device=device, dtype=torch.float64, generator=generator)
    m0 = 0.1 * torch.randn(B, H, topology.N, d_v, device=device, dtype=torch.float64,
                           generator=generator)

    y_ref, _ = naive_rola(v, read_levels, write_levels, g_read, g_write, topology, None, "raw",
                             initial_state=m0, output_final_state=True)
    # `mma_precision` is pinned to the EXACT arm, not defaulted: the plan default is
    # `bf16` (production), whose ~1e-2 per-product error would swamp the fp64-oracle
    # comparison below -- the split `MMA_PRECISIONS` documents.
    plan = Plan(topology=topology, normalization="raw", BT=32, BC=16, warp_tile=(1, 2),
                  mma_precision="tf32x3")
    y, state, _ = _consumer_forward(
        v.to(torch.float32), tuple(t.to(torch.float32) for t in read_levels),
        tuple(t.to(torch.float32) for t in write_levels), g_read.to(torch.float32),
        g_write.to(torch.float32), plan, initial_state=m0.to(torch.float32))
    scale = max(1e-30, float(y_ref.abs().max()))
    assert float((y.double() - y_ref).abs().max()) / scale < RTOL
    # The state is unchanged up to the 1e-300 gain, i.e. bit-comparable to M_0.
    assert torch.allclose(state.double(), m0, atol=1e-6, rtol=0)


def test_split_k_determinism_is_not_assumed():
    """fp32 `atomicAdd` reduction order is nondeterministic, so two runs may
    differ in the last bits. This test records that as an accepted property of
    the §5 epilogue rather than letting a later flake be mistaken for a bug: the
    two runs must agree to the fp32 accumulation tolerance, NOT bitwise."""
    from rola.ops.consumer import _consumer_forward

    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(31)
    widths, d_v = (8, 8), 64
    topology = _topology(widths, d_v)
    plan = Plan(topology=topology, normalization="raw", BT=32, BC=64, warp_tile=(1, 2))
    args = (
        torch.randn(1, 40, 2, d_v, device=device),
        tuple(_simplex((1, 40, 2, w), 0.5, generator, device).float() for w in widths),
        tuple(_simplex((1, 40, 2, w), 0.4, generator, device).float() for w in widths),
        (torch.rand(1, 40, 2, device=device) + 0.5),
        (torch.rand(1, 40, 2, device=device) + 0.5),
        plan,
    )
    y1, _, _ = _consumer_forward(*args)
    y2, _, _ = _consumer_forward(*args)
    scale = max(1e-30, float(y1.abs().max()))
    assert float((y1 - y2).abs().max()) / scale < RTOL


# --- LeafMassDecay (increment 2) --------------------------------------------


@pytest.mark.parametrize("widths,BC", [
    ((16,), 16),
    ((8, 4), 16),
    ((8, 8), 64),
    ((4, 4, 4), 16),
    ((4, 4, 4, 4), 16),
    ((2, 3, 4, 4), 16),
])
@pytest.mark.parametrize("BT", [16, 32])
@pytest.mark.parametrize("norm", ["raw", "global"])
@pytest.mark.parametrize("p_read,p_write", [(1.0, 1.0), (0.4, 0.3)])
def test_decay_matches_canonical_oracle(widths, BC, BT, norm, p_read, p_write):
    """Survival is inclusive at ``j = t``: the deposit at ``t`` is read at ``t``
    with survival exactly 1 (§2, §5). Both off-by-ones -- decaying after the
    deposit, or reading before it -- are O(1) errors, not rounding, so this gate
    discriminates them at the fp32 tolerance without needing a tighter one."""
    generator = torch.Generator(device="cuda").manual_seed(4242)
    _run_case(widths, BC, BT, 64, 2, 37, 3, norm, p_read, p_write,
              nonzero_m0=True, seed=1234,
              decay_dials=_decay_dials(widths, 3, generator, "cuda"))


@pytest.mark.parametrize("T", [1, 8, 16, 17, 31, 32, 33, 64, 65])
def test_decay_token_tile_boundaries(T):
    """Survival must compose ACROSS a tile boundary, which the token-sequential
    walk gets only because the owner's state stays resident across tiles rather
    than being reloaded per tile."""
    generator = torch.Generator(device="cuda").manual_seed(77)
    _run_case((8, 4), 16, 32, 64, 1, T, 2, "raw", 0.4, 0.3, nonzero_m0=True, seed=5,
              decay_dials=_decay_dials((8, 4), 2, generator, "cuda"))


@pytest.mark.parametrize("norm", ["raw", "global"])
def test_decay_near_unit_rate(norm):
    """Dials near 1 drive ``rate = prod_l delta_l`` toward 1, where ``1 - rate``
    cancels catastrophically in fp32. This is the gate on the kernel's fp32 dial
    accumulation (§7.4 audit, rule 2's V3-specific note): a bf16 or otherwise
    narrowed product would lose the cancellation before ``log1pf`` sees it, and
    the CPU-only algorithm gate cannot detect that."""
    generator = torch.Generator(device="cuda").manual_seed(909)
    _run_case((8, 4), 16, 32, 64, 1, 37, 2, norm, 0.5, 0.5, nonzero_m0=True, seed=13,
              decay_dials=_decay_dials((8, 4), 2, generator, "cuda", hot=True))


@pytest.mark.parametrize("norm", ["raw", "global"])
def test_decay_stored_clock_diverges_from_live_deposit(norm):
    """The clock and the deposit come from genuinely different tensors, end to end.

    ``write_levels_stored`` is a bf16 round-trip of the live ``write_levels``, so
    the clock ``c[t,s]`` and the deposit ``W[t,s]`` differ in VALUE at essentially
    every entry. That is what this fixture exercises, and it is worth having: a
    kernel that read the live tensor for the clock (or vice versa) fails here.

    What it does **not** reach -- measured, review finding F1 -- is P5.20.5
    finding 2's actual condition, ``stored == 0`` with ``live != 0``. bf16's
    flush-to-zero threshold is ~9.2e-41 and simplex entries are ``O(1/width)``, so
    that condition is *unreachable* by a bf16 round-trip of a simplex: 0 of 824
    live-nonzero entries flush on this exact fixture. It is reached by direct
    injection at the builder boundary instead, in
    ``test_rola_v3_schedule.py::test_build_clock_stream_accepts_stored_zero_with_live_nonzero``
    (review B2). P5.20.5 finding 2 itself rests on the structural argument
    (``build_schedule`` has no ``write_levels_stored`` parameter at all), never
    on this fixture."""
    generator = torch.Generator(device="cuda").manual_seed(31337)
    _run_case((8, 4), 16, 32, 64, 2, 37, 2, norm, 0.5, 0.4, nonzero_m0=True, seed=21,
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
        torch.rand(B, T, H, device="cuda") + 0.5,
        torch.rand(B, T, H, device="cuda") + 0.5,
    )
    dials = LeafMassDecay(dials=_decay_dials(widths, H, generator, "cuda"))
    no_decay = Plan(topology=topology, normalization="raw", BT=32, BC=16, warp_tile=(1, 2))
    with pytest.raises(ValueError, match="plan.decay"):
        _consumer_forward(*args, no_decay, decay=dials)
    with_decay = Plan(topology=topology, normalization="raw", BT=32, BC=16,
                        warp_tile=(1, 2), decay=True)
    with pytest.raises(ValueError, match="plan.decay"):
        _consumer_forward(*args, with_decay, decay=None)


# --- `panel`: the LeafMassDecay dense-limit intra lowering (spec §7.4.1) -----
# Gated exactly like its `decay=None` sibling `gram`: the SAME canonical fp64
# oracle, the SAME `RTOL`, the same topology sweep. `panel` is an EXACTNESS claim
# with a numeric structure of its own -- two separately exponentiated survival
# panels (`intra_panel.cuh`), where `GLA_MASS_DECAY_SPEC.md` §12 rules 1 and 3 go
# live for the first time in this design -- so it is not enough that it is fast.


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
@pytest.mark.parametrize("BT", [16, 32])
@pytest.mark.parametrize("p_read,p_write", [(1.0, 1.0), (0.4, 0.3), (1.0, 0.2)])
def test_panel_intra_lowering_matches_canonical_oracle(widths, BC, BT, p_read, p_write):
    """`panel` is EXACT, not an approximation, at every density and every topology.

    The lowering folds cumulative survival into the read and write factors --
    ``R~ = R * keep^{P[t]-P_ref}``, ``W~ = W * keep^{-(P[j]-P_ref)}`` -- so a mixed
    LeafMassDecay tile takes the ``decay = None`` tile path verbatim (MMA readout,
    causal Gram, deposit) instead of the token-sequential walk. Every one of the
    three ways that can be wrong is an O(1) error rather than a rounding one and
    is discriminated at the unchanged ``RTOL``: the reference frame not being
    left again (the exit scale), the inclusive ``j = t`` survival (which must be
    exactly 1 -- ``R~[t]*W~[t]`` is the one place the two exponentials must
    cancel), and the prefix restarting per tile rather than accumulating across
    tiles.

    The topology sweep is the discriminator for the clock specifically: unlike the
    read/write factors, the clock product runs over ALL levels including the ones
    the owner FIXES, because an exponent does not hoist into the §3 step-3 row
    scale."""
    _run_case(widths, BC, BT, 64, 2, 37, 3, "raw", p_read, p_write,
              nonzero_m0=True, seed=1234,
              decay_dials=_panel_dials(widths, 3))


@pytest.mark.parametrize("widths,BC", [((8, 4), 16), ((4, 4, 4), 16), ((8, 8), 64)])
@pytest.mark.parametrize("mma_precision", ["tf32x3", "bf16"])
def test_panel_intra_lowering_under_global_and_both_mma_arms(widths, BC, mma_precision):
    """`global` and the numeric-format axis, crossed with `panel`.

    Both are live here in a way they are NOT under the walk: `panel`'s mixed tile
    runs the tensor-core readout (so `mma_precision` selects a real format on a
    decay MIXED tile for the first time) and shares one Gram between ``num`` and
    ``den`` (so the mass recurrence takes the same entry/exit scale the state
    does). §12 rule 3's clamp is a bf16-finite clamp, which makes the bf16 arm the
    one where an unclamped panel entry would show up as an ``inf``."""
    _run_case(widths, BC, 32, 64, 2, 37, 3, "global", 0.4, 0.3,
              nonzero_m0=True, seed=99,
              mma_precision=mma_precision, decay_dials=_panel_dials(widths, 3))


@pytest.mark.parametrize("T", [1, 8, 16, 17, 31, 32, 33, 64, 65])
def test_panel_intra_lowering_token_tile_boundaries(T):
    """Survival must compose ACROSS a tile boundary, and `panel` composes it
    differently from the walk: the per-tile prefix is entered and left through
    ``S0``, so a tail tile whose last tokens are past ``T`` must contribute
    exactly nothing to the prefix (the clock stream is zero-padded) and the
    owner's resident state must carry the composition, not the panel."""
    _run_case((8, 4), 16, 32, 64, 1, T, 2, "raw", 0.4, 0.3,
              nonzero_m0=True, seed=5,
              decay_dials=_panel_dials((8, 4), 2))


@pytest.mark.parametrize("compact", [True, False])
@pytest.mark.parametrize("norm", ["raw", "global"])
def test_panel_intra_lowering_on_both_compact_arms(compact, norm):
    """The §6 `compact` axis, crossed with `panel`.

    `panel` is the first decay-arm lowering that uses the COMPACTED read panel and
    the compacted write-panel view (the walk uses neither -- it indexes ``M[i]``
    directly), so the gather's scatter-fill identities have to hold for the two
    survival-folded panels as well as for the raw ones."""
    _run_case((8, 8), 64, 32, 64, 2, 37, 3, norm, 0.4, 0.3, nonzero_m0=True, seed=808,
              compact=compact,
              decay_dials=_panel_dials((8, 8), 3))


@pytest.mark.parametrize("widths,BC", [((8, 4), 16), ((8, 8), 64)])
@pytest.mark.parametrize("d_v", [32, 64])
@pytest.mark.parametrize("decay_on", [False, True])
@pytest.mark.parametrize("seed", [101, 202, 303])
def test_compacted_write_panel_view_is_race_free_under_non_identity_compaction(
        widths, BC, d_v, decay_on, seed):
    """Regression probe for the `kWritePanelC` latent barrier race
    (`panel-review.md` R3, fixed by the `__syncthreads()` added at the compacted
    write-panel copy in ``v3_consumer.cu``).

    The bug: the compacted copy reads ``panel_write[t*BC + c_idx[q]]`` from
    thread ``(t*BC + q) % DV``, while thread ``(t*BC + c_idx[q]) % DV`` is the one
    that WROTE that slot in the build immediately above. Those threads differ
    whenever the compaction is not the identity (``c_idx[q] != q (mod DV)``),
    which needs sparse, non-aligned routing to force -- a dense or structured
    fixture keeps the compaction close enough to the identity that most pairs
    land same-warp and the race stays invisible even without the barrier
    (`panel-review.md` §6 measured this: the fix passes 141/141 targeted tests
    AND a 72-cell non-identity probe even with the barrier deleted, on this box).
    This fixture is deliberately UNSTRUCTURED and asymmetric (``p_read=0.35``,
    ``p_write=0.25``, no shared support) specifically to make ``c_idx`` a
    non-trivial permutation rather than a near-identity one, and sweeps ``d_v``
    (warp count), ``BC`` (owner width) and the two lowerings that reach
    ``kWritePanelC`` (``realized_leaf`` at ``decay=None``, ``panel`` at
    ``decay=LeafMassDecay``) against the canonical oracle at the standing
    ``RTOL``.

    This test runs against the FIXED kernel and is therefore a coverage
    expansion for this code path, not a proof the race cannot recur: because the
    race is scheduling-dependent, a regression that removed the barrier again
    could still pass this test on this box/driver/compiler combination. It is
    committed anyway because the un-compaction shape it exercises (asymmetric
    sparse read/write support driving `c_idx` away from the identity) was not
    covered by the existing compact-arm gates, which use dense or lightly-sparse
    shared-support fixtures."""
    H = 2
    decay_dials = _panel_dials(widths, H, seed=seed) if decay_on else None
    _run_case(widths, BC, 32, d_v, 2, 257, H, "raw", 0.35, 0.25,
              nonzero_m0=True, seed=seed, compact=True,
              decay_dials=decay_dials)


@pytest.mark.parametrize("norm", ["raw", "global"])
def test_panel_intra_lowering_near_unit_rate(norm):
    """Hot dials under `panel`: ``rate -> 1`` makes ``1 - rate`` cancel, and
    `panel` then exponentiates ``log1pf(-rate)`` twice with opposite signs. This is
    the gate on the fp32 dial accumulation AND on §12 rule 1: without the mid-tile
    reference the exponent range is the tile's whole survival span rather than
    half of it."""
    _run_case((8, 4), 16, 32, 64, 1, 37, 2, norm, 0.5, 0.5, nonzero_m0=True, seed=13,
              decay_dials=_panel_dials((8, 4), 2, hot=True))


def test_panel_intra_lowering_with_a_stored_clock_that_differs_from_the_deposit():
    """The clock and the deposit come from different tensors under `panel` too --
    and `panel` reads the clock through a THIRD path (a per-leaf prefix, not the
    per-token survival panel), so a lowering that quietly used the live write
    amplitudes for the prefix would pass every other gate and fail here."""
    _run_case((8, 4), 16, 32, 64, 2, 37, 2, "raw", 0.5, 0.4, nonzero_m0=True, seed=21,
              quantize_stored=True,
              decay_dials=_panel_dials((8, 4), 2, seed=31337))


@pytest.mark.parametrize("norm", ["raw", "global"])
def test_panel_exact_zero_factors_stay_exactly_zero_and_never_nan(norm):
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
    side while still being read -- the union-compaction case -- and they are dead
    on the SPANNED levels, which is where the panel build multiplies."""
    widths, BC, BT, d_v, B, T, H = (8, 4), 16, 32, 64, 2, 37, 2
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
    _run_case(widths, BC, BT, d_v, B, T, H, norm, 1.0, 1.0, **case)
    y, state = _run_case(widths, BC, BT, d_v, B, T, H, norm, 1.0, 1.0,
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
    mid-tile reference halves the exponent range and rule 3's clamp keeps the
    result finite, but spec §7.4 is explicit that the two are "a pair, not
    alternatives" and that rule 1 "is not on its own sufficient at extreme dials".
    Past the fp32 exp range the write factor saturates at the clamp while the read
    factor underflows, and their product -- ``O(1)`` at ``t ~ j`` -- comes out as
    ``~0``: WRONG, not imprecise.

    So the domain is a checked admission bound (``panel_exponent_bound`` against
    ``PANEL_MAX_EXPONENT``), computed from the dials and the realized clock stream
    with no ``N``-shaped materialization, and this gate pins all three regimes:
    inside the bound the launch runs and is oracle-exact; outside it the launch is
    REFUSED loudly; and nothing is silently substituted (``realized_leaf`` on the
    identical routing still runs, which is what makes the refusal a real choice
    rather than a dead end).

    The fixture is the only regime that reaches the boundary: a near-one-hot write
    clock (so the per-tile clock prefix is ~``BT``) meeting a near-unit leaf rate
    (so ``|log keep|`` is large). Either alone is far inside the domain, which is
    why every other gate in this file is."""
    import math

    from rola.ops.consumer import _consumer_forward
    from rola.planner.schedule import PANEL_MAX_EXPONENT, build_clock_stream, panel_exponent_bound

    widths, BC, BT, d_v, B, T, H = (4, 4), 16, 16, 32, 1, 32, 1
    device = "cuda"
    topology = _topology(widths, d_v)
    generator = torch.Generator(device=device).manual_seed(1717)
    # One-hot writes on digit 0 of every level: the clock is exactly 1 at every
    # token for leaf 0, so its per-tile prefix is BT and `P_ref = BT/2 = 8`.
    write_levels = tuple(
        torch.nn.functional.one_hot(torch.zeros(B, T, H, dtype=torch.long), w
                                    ).to(device=device, dtype=torch.float64)
        for w in widths)
    read_levels = tuple(
        torch.full((B, T, H, w), 1.0 / w, device=device, dtype=torch.float64) for w in widths)
    g_write = torch.ones(B, T, H, device=device, dtype=torch.float64)
    g_read = torch.ones(B, T, H, device=device, dtype=torch.float64)
    v = torch.randn(B, T, H, d_v, device=device, dtype=torch.float64, generator=generator)

    def _plan():
        return Plan(topology=topology, normalization="raw", BT=BT, BC=BC,
                      warp_tile=(1, max(1, d_v // 32)), decay=True)

    def _call(rate_target):
        # `rate = prod_l delta_l`, so a per-level dial of sqrt(rate) at D=2.
        dial = rate_target ** (1.0 / len(widths))
        decay = LeafMassDecay(dials=tuple(
            torch.full((H, w), dial, device=device, dtype=torch.float64) for w in widths))
        return _consumer_forward(
            v.float(), tuple(t.float() for t in read_levels),
            tuple(t.float() for t in write_levels), g_read.float(), g_write.float(),
            _plan(), decay=decay)

    def _bound(rate_target):
        dial = rate_target ** (1.0 / len(widths))
        decay = LeafMassDecay(dials=tuple(
            torch.full((H, w), dial, device=device, dtype=torch.float64) for w in widths))
        clock = build_clock_stream(tuple(t.float() for t in write_levels),
                                  tuple(t.float() for t in write_levels), _plan())
        return panel_exponent_bound(clock, decay, BT), decay

    # The clock prefix is exactly `BT` on the written leaf, so the bound is
    # `BT/2 * |log(1 - rate)|` and the threshold sits at a computable rate.
    inside_rate = 1.0 - math.exp(-2.0 * PANEL_MAX_EXPONENT / BT * 0.5)
    outside_rate = 1.0 - math.exp(-2.0 * PANEL_MAX_EXPONENT / BT * 1.5)
    bound_in, _ = _bound(inside_rate)
    bound_out, _ = _bound(outside_rate)
    assert bound_in < PANEL_MAX_EXPONENT < bound_out, (
        f"fixture no longer straddles the admission bound: {bound_in} / {bound_out}")

    y_in, _state, _sched = _call(inside_rate)
    assert bool(torch.isfinite(y_in).all())
    with pytest.raises(ValueError, match="PANEL_MAX_EXPONENT"):
        _call(outside_rate)
    # **THE "NOT A DEAD END" HALF IS DELETED** (item 6b, 2026-08-01). It ran the
    # identical routing through `intra_lowering='realized_leaf'` -- the old
    # consumer's token-sequential walk, which had no exponent range -- to show the
    # refusal was a choice between two lowerings rather than a wall. There is one
    # lowering now, so it IS a wall, and the refusal says so: an out-of-domain
    # routing is refused outright rather than clamped into a wrong answer. That is
    # the property this gate now pins, and it is strictly the stronger claim.


# --- B1: fp64-valid dial that rounds to 1.0f under the kernel's cast --------


@pytest.mark.parametrize("norm", ["raw", "global"])
def test_decay_dial_cliff_is_a_loud_error_not_a_nan(norm):
    """B1 (blocker, review fix pass). An fp64 dial in ``[1 - 3e-8, 1)`` is
    strictly ``< 1`` in fp64 -- passes ``LeafMassDecay.__post_init__`` -- but
    rounds to exactly ``1.0f`` under the ``.to(torch.float32)`` cast the kernel
    consumes. That made ``rate == 1.0f``, so ``log1pf(-1.0f) == -inf``
    (``v3_consumer.cu``), and any leaf whose clock is exactly zero at a
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
    g_read = None if norm == "global" else torch.ones(B, T, H, device="cuda", dtype=torch.float64)
    v = torch.randn(B, T, H, d_v, device="cuda", dtype=torch.float64, generator=generator)
    plan = Plan(topology=topology, normalization=norm, BT=32, BC=16, warp_tile=(1, 1), decay=True)

    def _call(dial_val):
        decay = LeafMassDecay(dials=tuple(
            torch.full((H, w), dial_val, device="cuda", dtype=torch.float64) for w in widths))
        return _consumer_forward(
            v.float(), tuple(t.float() for t in read_levels), (w0.float(), w1.float()),
            None if g_read is None else g_read.float(), g_write.float(), plan, decay=decay)

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
# The `compact` compile axis (spec §6, ratified rev `2026-07-29.5` Q1)
#
# `Plan.compact` selects between two INSTANTIATIONS of the one kernel: the
# gather compiled in, and the gather compiled out at its `K == BC` degeneracy.
# It is a plan-time structure choice from measured supports (§7.4.1's sanctioned
# pattern), so both instantiations must be exact at every density -- the planner's
# job is to pick the faster one, never to pick the correct one. These gates are
# what make that true rather than assumed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("widths,BC", [
    ((16,), 16),
    ((8, 4), 16),
    ((8, 8), 64),
    ((4, 4, 4), 16),
    ((4, 4, 4, 4), 16),
])
@pytest.mark.parametrize("BT", [16, 32])
@pytest.mark.parametrize("norm", ["raw", "global"])
@pytest.mark.parametrize("p_read,p_write,label", [
    (1.0, 1.0, "dense"),
    (0.4, 0.3, "sparse"),
    (1.0, 0.2, "dense-read-sparse-write"),
])
def test_the_compiled_out_gather_matches_the_canonical_oracle(
        widths, BC, BT, norm, p_read, p_write, label):
    """`compact=False` is exact at EVERY density, not only where it is selected.

    The planner picks this instantiation from measured tile unions, and a measured
    statistic is a mean: individual tiles below it are still sparse. Compiled out,
    the kernel simply contracts over all `BC` columns, and the inert ones carry the
    zero amplitude the dense build computes for them -- so exactness does not depend
    on the planner having been right. Gated at the SPARSE fixtures deliberately, for
    exactly that reason."""
    _run_case(widths, BC, BT, 64, 2, 37, 3, norm, p_read, p_write,
              nonzero_m0=True, seed=4407, compact=False)


@pytest.mark.parametrize("widths,BC", [((8, 4), 16), ((4, 4, 4), 16)])
@pytest.mark.parametrize("norm", ["raw", "global"])
@pytest.mark.parametrize("mma_precision", ["tf32x3", "bf16"])
def test_the_compiled_out_gather_is_exact_under_decay(widths, BC, norm, mma_precision):
    """Both decay arms and both MMA formats reach the compiled-out instantiation.

    Under LeafMassDecay the compiled-out path also drops the `c_slot` predicate on
    the SURVIVAL panel, so every leaf's `keep` is evaluated rather than identity-
    filled. That is the same value by construction (an inert leaf has `c == 0`, so
    `expf(0 * log_keep) == 1`), and this is where that "by construction" is
    checked against the fp64 oracle instead of argued."""
    _run_case(widths, BC, 32, 64, 2, 37, 3, norm, 0.4, 0.3,
              nonzero_m0=True, seed=911, compact=False,
              decay_dials=_decay_dials(widths, 3, torch.Generator(device="cuda").manual_seed(911),
                                       "cuda"),
              mma_precision=mma_precision)


@pytest.mark.parametrize("compact", [True, False])
def test_a_vanishing_write_gain_with_a_live_write_allocation(compact):
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
    O(1) effect. Both compact arms must therefore agree with the oracle at the
    unmodified tolerances -- which is what says the validity word (which excludes
    `g_write`, deliberately) has not started dropping leaves whose survival is not
    1. The exclusion is over-inclusive and hence safe; this fixture is where "hence
    safe" is checked at a small `g_write` rather than argued.

    NOTE, recorded because the commit message and `compaction2.md` §A1.2 originally
    rested the exclusion on the O(1) error being REACHABLE: it is not. `g_write == 0`
    exactly is rejected by the oracle (`naive.py:116`, spec §4.2) and cannot be
    produced (`g_write = project_side_gains(...) * write_mass`, both positive). The
    exclusion is right because it is over-inclusive and keeps the word meaning one
    thing on both sides, NOT because the alternative errs on a reachable input. The
    one route that would reach it is an fp64-positive `g_write` underflowing the
    fp32 the kernel reads -- which is what `1e-30` here approaches without crossing.
    """
    device = "cuda"
    widths, BC, BT, d_v, B, T, H = (8, 16), 16, 16, 64, 1, 64, 2
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
    _run_case(widths, BC, BT, d_v, B, T, H, "raw", 1.0, 1.0, nonzero_m0=True, seed=730,
              read_levels=read_levels, write_levels=write_levels, g_write=g_write,
              compact=compact,
              decay_dials=_decay_dials(widths, H, generator, device))


@pytest.mark.parametrize("widths,BC", [((16,), 16), ((8, 8), 64)])
@pytest.mark.parametrize("norm", ["raw", "global"])
@pytest.mark.parametrize("decay", [False, True])
def test_the_plan_selected_path_is_oracle_exact_and_bit_identical(widths, BC, norm, decay):
    """The WIRED path: a plan whose §6 axes are SELECTED, not hand-written.

    `select_plan` is the plan build -- it measures the realized routing tensors,
    prices §6's three axes jointly, and returns a `Plan`. Until 2026-07-30 the
    selector was called from no path at all (adversarial review §4.3), so every
    "plan-selected" number in a findings doc described a function no build invoked.
    This gate runs the selector's own output end to end.

    Two properties, and they are the two the increment claims:

    1. **The selected plan is RUNNABLE and oracle-exact** -- on BOTH decay arms.
       That is not free: the decay arm's dense limit (`panel`) is unimplemented, and
       selection ranges over the arm's implemented lowerings so the plan the driver
       receives can actually launch.
    2. **The selected compact axis is a performance decision with no numeric
       consequence** at saturation: this fixture is fully dense (`K == BC`), so the
       selected instantiation and its opposite must agree BIT for BIT. Single-owner
       rectangles only, for the split-K reason `test_the_two_compact_arms_are_bit_
       identical_at_saturation` states.
    """
    from math import prod

    from planner_calibration.cost_model import select_plan  # G16

    assert prod(widths) // BC == 1, "this gate is only meaningful on a single-owner rectangle"
    device = "cuda"
    d_v, B, T, H = 64, 2, 37, 3
    generator = torch.Generator(device=device).manual_seed(7301)
    topology = _topology(widths, d_v)
    read_levels = tuple(_simplex((B, T, H, w), 1.0, generator, device) for w in widths)
    write_levels = tuple(_simplex((B, T, H, w), 1.0, generator, device) for w in widths)

    # The SELECTED axes are what this gate is about; `mma_precision` is not one of
    # them (it is a numeric-format input, not a cost axis) and is pinned to the exact
    # arm so the fp64-oracle comparison below measures structure, not bf16 rounding.
    plan, axes = select_plan(
        tuple(t.float() for t in read_levels), tuple(t.float() for t in write_levels),
        topology, BC, normalization=norm, decay=decay, mma_precision="tf32x3")
    # The fixture is dense, so the selector must see saturation and price the
    # gather out: at K == BC it is the identity and can only cost.
    assert axes.stats.mean_K_candidate == BC
    assert plan.compact is False

    dials = _decay_dials(widths, H, generator, device) if decay else None
    kwargs = dict(nonzero_m0=True, seed=7301, decay_dials=dials,
                  read_levels=read_levels, write_levels=write_levels)
    # 1. oracle-exact on the selected plan.
    _run_case(widths, BC, plan.BT, d_v, B, T, H, norm, 1.0, 1.0, plan=plan, **kwargs)
    # 2. bit-identical against the opposite compact instantiation.
    from dataclasses import replace as _replace
    y_sel, state_sel = _run_case(widths, BC, plan.BT, d_v, B, T, H, norm, 1.0, 1.0,
                                 plan=plan, return_outputs=True, **kwargs)
    other = _replace(plan, compact=not plan.compact)
    y_oth, state_oth = _run_case(widths, BC, plan.BT, d_v, B, T, H, norm, 1.0, 1.0,
                                 plan=other, return_outputs=True, **kwargs)
    for name, a, b in (("y", y_sel, y_oth), ("final state", state_sel, state_oth)):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32)), (
            f"{name} differs between the SELECTED plan (compact={plan.compact}) and its "
            f"opposite at a saturated fixture: max |delta| = {float((a - b).abs().max()):.3e}")


# ---------------------------------------------------------------------------
# §6 residency (increment item 2(a)): the planner's SMEM model against the
# launcher's own, and the 1-vs-many cliff against the driver
# ---------------------------------------------------------------------------


def test_no_launchable_instantiation_reaches_the_one_cta_cliff():
    """§6's DELETED residency rule, kept as the MEASUREMENT that deleted it.

    The rule fired on `cheapest.blocks_per_sm == 1` -- the cliff at which a CTA has
    no latency hiding at all. That cliff was a fact about the DELETED consumer
    (72448 B at `BT = 32, BC = 64, d_v = 64`, one CTA per SM). This consumer's
    footprint never reaches it, so the guard could never be true and the rule was
    dead code with a permanently-`None` `residency_override`.

    "Never" is a claim about the whole launchable surface, so this asks the driver
    about the whole launchable surface: `BT(2) x BC(2) x d_v(2) x decay(2) x
    compact(2)` = 32 keys, which is EXHAUSTIVE -- those five are the only axes the
    occupancy query has, because the kernel's shared memory is entirely
    compile-time and `gram_cols`/`ring_slots`/`mma_precision` do not enter it.

    **THIS IS THE FALSIFICATION SITE.** If a future axis, budget or arch makes some
    instantiation land at 1 CTA/SM, this fails and the rule's deletion is what has
    to be revisited -- loudly, instead of the blind spot silently becoming
    load-bearing again. It replaces `test_the_planner_smem_model_matches_the_launcher`
    and `test_the_residency_predictor_agrees_with_the_driver_at_the_cliff`, whose
    subject -- the analytic shared-memory mirror `v3_smem_bytes` /
    `predicted_blocks_per_sm` -- modelled the deleted consumer and was measured
    INVERTING the bound it claimed against this one (mirror 2, driver 3 at the
    at-scale `BC = 64` cell). There is one accounting now, so there is nothing to
    pin a second one against.
    """
    from rola.ops._ext import extension

    ext = extension()
    seen = {}
    for BT in (16, 32):
        for BC in (16, 64):
            for d_v in (32, 64):
                for decay in (False, True):
                    for compact in (False, True):
                        blocks = int(ext.rola_consumer_occupancy(decay, compact, BC, BT, d_v))
                        seen[(BT, BC, d_v, decay, compact)] = blocks
                        assert blocks >= 2, (
                            f"BT={BT} BC={BC} d_v={d_v} decay={decay} compact={compact} "
                            f"is resident {blocks} time(s) per SM. The 1-CTA cliff is "
                            "reachable again, and §6's residency rule was deleted on the "
                            "measurement that it is not -- see `select_plan_axes`.")
    assert len(seen) == 32, f"the surface is 32 keys, queried {len(seen)}"
    # Recorded rather than asserted: the exact integers are an architecture fact.
    # What IS asserted is the ORDERING a constant-returning stub would fail --
    # `BT = 32` doubles the y-fragment count and the token panels, so it can never
    # be MORE resident than `BT = 16` at the same other axes.
    for BC in (16, 64):
        for d_v in (32, 64):
            for decay in (False, True):
                for compact in (False, True):
                    assert seen[(32, BC, d_v, decay, compact)] <= seen[(16, BC, d_v, decay,
                                                                        compact)], (
                        f"BT=32 reported MORE CTAs/SM than BT=16 at BC={BC} d_v={d_v} "
                        f"decay={decay} compact={compact}; the query is not reading the "
                        "instantiation it names")


# ---------------------------------------------------------------------------
# I6 -- THE MASS COLUMN'S OWN TOLERANCE GATE (THUNDER_RUN_SPEC.md §5, rung K5)
# ---------------------------------------------------------------------------
# K5 moves the mass column out of a fp32 SCALAR side path and into the
# `(d_v + 1)`-th n-column of the readout / intra / fold GEMMs. Its three
# multiplicands move fp32 -> bf16 OPERAND on the production arm:
#
#   term 1 (readout)  `sum_k R[t,k] * S[k][d_v]`   `S[·][d_v]` fp32 -> bf16 B
#   term 2 (intra)    `sum_u G[t,u] * sw[u]`       `sw[u]`     fp32 -> bf16 B
#   the fold          `sum_t W[t,i] * sw[t]`       `sw[t]`     fp32 -> bf16 B
#
# §5 requires this gate to be WRITTEN FIRST, before the kernel change, so that
# the tolerance is a prediction rather than a description of whatever landed.
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
# **NO NEW TOLERANCE IS INTRODUCED.** The gate runs at the file's existing two
# numbers -- `RTOL` on the exactness arm, `BF16_RTOL` on the production arm --
# because the spec's expectation is that the mass moves into the SAME precision
# class as the numerator it is divided into. A mass column that needed a wider
# number than the value columns would be the spec's stated contingency ("the
# finding is: mass needs a tf32 n-tile"), and it would show up here as a
# failure rather than as an edited constant.


def _mass_errors(**kw):
    """`(mass, value, ratio)` relative errors against the fp64 oracle.

    Each is normalized by its OWN reference scale, which is the point: the mass
    column, the value block and the ratio are three different quantities and a
    shared normalizer makes the smallest of them untestable.
    """
    d_v = kw["d_v"]
    y, state, y_ref, state_ref = _run_case(return_reference=True, **kw)
    mass, mass_ref = state[..., d_v].double(), state_ref[..., d_v]
    val, val_ref = state[..., :d_v].double(), state_ref[..., :d_v]

    def err(a, b):
        return float((a - b).abs().max()) / max(1e-30, float(b.abs().max()))

    return err(mass, mass_ref), err(val, val_ref), err(y.double(), y_ref)


#: The mass column carries no cancellation -- it is a sum of strictly positive
#: deposits -- so its condition number is 1 and it is gated at the SAME numbers
#: the value block is. `mass_scale` is the discriminating axis: `1e3` puts the
#: carried mass far above the value block (term 1's read of `S[·][d_v]` dominates
#: the denominator), `1e-3` far below it (the regime a whole-tensor relative
#: error cannot see at all).
@pytest.mark.parametrize("widths,BC", [((16,), 16), ((8, 8), 64), ((4, 4, 4), 16)])
@pytest.mark.parametrize("BT", [16, 32])
@pytest.mark.parametrize("mma_precision", ["tf32x3", "bf16"])
@pytest.mark.parametrize("mass_scale", [1e3, 1e-3])
def test_mass_column_matches_canonical_oracle(widths, BC, BT, mma_precision, mass_scale):
    tol = RTOL if mma_precision == "tf32x3" else BF16_RTOL
    e_mass, e_val, e_ratio = _mass_errors(
        widths=widths, BC=BC, BT=BT, d_v=64, B=2, T=37, H=3, norm="global",
        p_read=0.4, p_write=0.3, nonzero_m0=True, seed=90210,
        mma_precision=mma_precision, mass_scale=mass_scale)
    assert e_mass < tol, f"the state's MASS column disagrees with the oracle: {e_mass:.3e}"
    assert e_val < tol, f"the state's VALUE block disagrees with the oracle: {e_val:.3e}"
    assert e_ratio < tol, f"the RATIO readout disagrees with the oracle: {e_ratio:.3e}"


#: The decay arm separately, because the mass column takes the SAME exit scale
#: the value columns do (`M_exit = s0 * (s0 * M_entry + deposit)`) and K5 moves
#: that epilogue from a scalar read-modify-write into the fold's D-fragment one.
@pytest.mark.parametrize("widths,BC", [((16,), 16), ((8, 8), 64), ((4, 4, 4), 16)])
@pytest.mark.parametrize("mma_precision", ["tf32x3", "bf16"])
def test_mass_column_under_decay_matches_canonical_oracle(widths, BC, mma_precision):
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(77)
    tol = RTOL if mma_precision == "tf32x3" else BF16_RTOL
    e_mass, e_val, e_ratio = _mass_errors(
        widths=widths, BC=BC, BT=32, d_v=64, B=2, T=37, H=3, norm="global",
        p_read=0.4, p_write=0.3, nonzero_m0=True, seed=555,
        decay_dials=_decay_dials(widths, 3, gen, device), mma_precision=mma_precision,
        mass_scale=1e-3)
    assert e_mass < tol, f"the decayed MASS column disagrees with the oracle: {e_mass:.3e}"
    assert e_val < tol, f"the decayed VALUE block disagrees with the oracle: {e_val:.3e}"
    assert e_ratio < tol, f"the decayed RATIO readout disagrees with the oracle: {e_ratio:.3e}"


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
@pytest.mark.parametrize("mma_precision", ["tf32x3", "bf16"])
def test_mass_column_gate_is_discriminating(mma_precision):
    tol = RTOL if mma_precision == "tf32x3" else BF16_RTOL
    kw = dict(widths=(8, 8), BC=64, BT=16, d_v=64, B=2, T=37, H=3, norm="global",
              p_read=1.0, p_write=1.0, nonzero_m0=True, seed=31337,
              mma_precision=mma_precision, mass_scale=1e3)
    clean_mass, clean_val, clean_ratio = _mass_errors(**kw)
    assert clean_mass < tol and clean_val < tol and clean_ratio < tol, (
        "the unperturbed control must pass, or the perturbed run proves nothing "
        f"(mass {clean_mass:.3e}, value {clean_val:.3e}, ratio {clean_ratio:.3e})")

    p = 100.0 * tol
    bad_mass, bad_val, bad_ratio = _mass_errors(perturb_mass=p, **kw)
    assert bad_mass > tol, (
        f"a mass perturbed by {p:.1e} relative moved the mass claim only to "
        f"{bad_mass:.3e}: the gate is blind to the defect it exists for")
    assert bad_ratio > tol, (
        f"a mass perturbed by {p:.1e} relative moved the RATIO claim only to "
        f"{bad_ratio:.3e}: the denominator's error is cancelling, so this is not "
        "a gate on the ratio readout")
    assert bad_val < tol, (
        f"injecting into the mass column moved the VALUE block to {bad_val:.3e}: "
        "the three claims are not independent and the localization is unsound")
