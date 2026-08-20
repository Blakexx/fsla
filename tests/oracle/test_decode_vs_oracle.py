# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""TIER 1 — the T=1 decode path against the canonical fp64 oracle.

Gates G1-G7 of the decode design (docs/internals/decode/decode.md).

**WHY "BIT-IDENTITY VS THE PREFILL CONSUMER" IS NOT ONE OF THEM, and this has to be
said before the tolerances make sense.** Three independent facts make byte-equality
between decode and the prefill kernel unattainable, so a gate asking for it would be
asking for something no implementation can deliver:

1. The prefill consumer's ``y`` is not reproducible against ITSELF wherever a ``bh``
   holds more than one owner block — it fans partials in by ``atomicAdd`` and CUDA
   fixes no inter-CTA order (Track K measured 384 of 976 fixture cases differing
   between two runs of one binary). Decode is deterministic by design.
2. Summation order differs by construction: the prefill kernel contracts inside an
   MMA fragment tree per owner then sums owners by atomic; decode contracts in leaf
   order in fp32 FMA chains and folds 8 warps then ``n_split`` CTAs in fixed order.
3. The precision path differs: the prefill readout narrows its operands to bf16 for
   the MMA; decode has no MMA and stays fp32. Decode is the *more* accurate of the
   two, which is the safe direction but is still a difference.

So the gates are: byte-equality where decode is *supposed* to be exact (G1, G2, the
BC-blindness row in tier 2), and the ORACLE band everywhere else. Decode's byte gates
are STRONGER than the prefill kernel's, not weaker — G1 is a gate no MMA path passes.

**P67 D2-b / K31.** The prefill leg here is the fp64 ORACLE now: the tiled consumer
left at D2, and the chunk arm left with the K31 deletion batch (baseline = tag
``baseline/pre-k31``).
That took one gate with it and improved another:

* **G5** — decode against the TILED consumer at ``T = 1`` — RETIRES. Its subject was
  the second prefill kernel as a cross-check, and the chunk arm cannot run ``T = 1``
  at all (``T`` must be a multiple of ``rola.engine.rules.arm.CHUNK_TOKENS``). What G5
  actually established beyond the oracle rows was consumer-to-consumer agreement, and
  after D2 there is one prefill consumer.
* **G7** — the prefill/decode seam — is now graded against the fp64 ORACLE over the
  whole ``L + 1`` sequence rather than against a second prefill of ``L + 1``. It has
  to be: ``L`` and ``L + 1`` cannot both be chunk multiples. The replacement is an
  INDEPENDENT reference for the seam instead of the same family of kernel twice,
  which is the direction the standing rule points.

Run: ``pytest tests/oracle/test_decode_vs_oracle.py``
"""

from __future__ import annotations

import math

import pytest
import torch

from rola.ops.decode import _decode_step, derive_decode_geometry
from rola.ops.naive import naive_rola
from rola.routing.types import LeafMassDecay
from tests.oracle.oracle_fixtures import _decay_dials, _simplex, _topology
from tests.oracle.tolerances import BF16_RTOL

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="decode is a CUDA kernel"),
]

#: DECODE'S TOLERANCE IS THE bf16 FAMILY BOUND, ADOPTED RATHER THAN INVENTED.
DECODE_RTOL = BF16_RTOL


def _chunk_tokens() -> int:
    """The prefill arm's chunk length. Imported through the function so this
    module stays importable without the extension for a collection-only run."""
    from rola.engine.rules.arm import CHUNK_TOKENS

    return CHUNK_TOKENS

#: THE TOPOLOGY AXIS, and the last three rows are the reason it is written out rather
#: than reduced. The bitmap expansion copies an ``S_{l+1}``-bit block into each live
#: digit's range; when ``S_{l+1}`` is a multiple of 32 that is a word copy, and
#: otherwise it is a funnel shift across a word boundary. An off-by-one in that shift is
#: exactly the defect these gates exist to catch, so the axis must contain a topology
#: with ``S_{D-1} < 32`` and one with ``S_{D-1} % 32 != 0``:
#:
#:   (8, 16)     S_1 = 16   < 32
#:   (4, 4, 4)   S_2 = 4    < 32,  S_1 = 16 < 32
#:   (3, 5, 7)   S_2 = 7    < 32,  S_1 = 35 % 32 = 3   <- the funnel-shift case
#:   (2, 3, 4, 4) S_3 = 4,  S_2 = 16, S_1 = 48 % 32 = 16
#:   (7, 9)      S_1 = 9    < 32,  N = 63 (a ragged last word)
#:   (33,)       D = 1,     N = 33 spans two words with a ragged tail
#:   (64, 64)    the aligned case, N = 4096
#: `kCoarseGroupWords * 32` -- the leaves one coarse group covers.
kCOARSE_GROUP_LEAVES = 32 * 32

_TOPOLOGIES = [(16,), (33,), (8, 16), (7, 9), (4, 4, 4), (3, 5, 7), (2, 3, 4, 4), (64, 64)]


def _leaf_support(levels, widths, B, H, device):
    """``[BH, N]`` bool: leaf ``s`` is live iff EVERY level's digit is exact-nonzero.

    This is `schedule.py`'s own definition of the support bitmap
    (``read_mask = tuple(level != 0 ...)``) taken to leaf granularity, written here in
    torch so the kernel's expansion is checked against the definition rather than
    against a second copy of the kernel.
    """
    D = len(widths)
    N = math.prod(widths)
    strides = [math.prod(widths[l + 1:]) for l in range(D)]
    idx = torch.arange(N, device=device)
    mask = torch.ones(B * H, N, dtype=torch.bool, device=device)
    for level, width in enumerate(widths):
        digit = (idx // strides[level]) % width
        folded = levels[level].permute(0, 2, 1, 3).reshape(B * H, width)
        mask &= folded[:, digit] != 0
    return mask


def _kernel_expansion(levels, widths, B, H, device, d_v=64):
    """The step kernel's OWN expansion of one side's amplitudes, as ``[BH, N]`` bool.

    The expansion is ONE device function and it does not know which side it is serving,
    so the amplitudes go in through the WRITE port, where the leaf set they expand to is
    exactly the set of state rows the step deposits into. Membership is then read off the
    STATE DIFF -- observable at any topology, and bit for bit, which is what the fused
    step's shared-memory bitmaps are not.

    The read side is left empty so the walk's union is the expansion under test, the
    state starts at zero and ``v`` is all ones, so a deposited row is a nonzero row and
    the correspondence is exact rather than statistical.
    """
    topology = _topology(widths)
    N = math.prod(widths)
    BH = B * H
    config = derive_decode_geometry(topology, d_v=d_v, decay=False, BH=BH, device=device)
    f32 = dict(device=device, dtype=torch.float32)
    empty_read = tuple(torch.zeros(B, 1, H, w, **f32) for w in widths)
    write = tuple(t.float() for t in levels)
    g_write = torch.ones(B, 1, H, **f32)
    v = torch.ones(B, 1, H, d_v, **f32)
    state = torch.zeros(B, H, N, config.cols, **f32)
    _y, state, _ws = _decode_step(v, empty_read, write, None, g_write, config, state)
    return (state.reshape(BH, N, config.cols) != 0).any(-1)


def _case(widths, d_v, B, H, norm, p_read, p_write, *, seed, decay_dials=None,
          n_split=None, nonzero_m0=True, steps=1, stored_bf16=False):
    """Build one fixture, run the oracle and decode, return everything both produced."""
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(seed)
    topology = _topology(widths)
    N = math.prod(widths)
    cols = d_v + (1 if norm == "global" else 0)

    read_levels, write_levels, g_reads, g_writes, vs = [], [], [], [], []
    for _ in range(steps):
        read_levels.append(tuple(_simplex((B, 1, H, w), p_read, generator, device) for w in widths))
        write = tuple(_simplex((B, 1, H, w), p_write, generator, device) for w in widths)
        #: ONE representation, and it is the STORED one: the producer emits bf16, so
        #: the clock and the deposit read the same tensor and a flushed amplitude is
        #: absent from both. Rounding here, before either side is called, is what makes
        #: that identity the fixture rather than an argument.
        if stored_bf16:
            write = tuple(t.to(torch.bfloat16).to(t.dtype) for t in write)
        write_levels.append(write)
        g_writes.append(torch.rand(B, 1, H, device=device, dtype=torch.float64, generator=generator) + 0.5)
        g_reads.append(None if norm == "global"
                       else torch.rand(B, 1, H, device=device, dtype=torch.float64,
                                       generator=generator) + 0.5)
        vs.append(torch.randn(B, 1, H, d_v, device=device, dtype=torch.float64, generator=generator))

    initial_state = None
    if nonzero_m0:
        initial_state = 0.1 * torch.randn(B, H, N, cols, device=device, dtype=torch.float64,
                                          generator=generator)
        if norm == "global":
            initial_state[..., d_v] = initial_state[..., d_v].abs()

    decay = None if decay_dials is None else LeafMassDecay(dials=decay_dials)

    # ---- the oracle, chained one token at a time (which is what decode does) ----
    ref_state = initial_state
    y_refs = []
    for s in range(steps):
        y_ref, ref_state = naive_rola(
            vs[s], read_levels[s], write_levels[s], g_reads[s], g_writes[s], topology,
            decay, norm, initial_state=ref_state, output_final_state=True)
        y_refs.append(y_ref)

    # ---- decode ---------------------------------------------------------------
    config = derive_decode_geometry(topology, d_v=d_v, decay=decay is not None, BH=B * H,
                                  device=device, n_split=n_split)
    state = (torch.zeros(B, H, N, cols, device=device, dtype=torch.float32)
             if initial_state is None else initial_state.to(torch.float32).clone())
    workspace = None
    y_outs = []
    for s in range(steps):
        y, state, workspace = _decode_step(
            vs[s].float(), tuple(t.float() for t in read_levels[s]),
            tuple(t.float() for t in write_levels[s]),
            None if g_reads[s] is None else g_reads[s].float(), g_writes[s].float(),
            config, state, decay=decay, workspace=workspace)
        y_outs.append(y)

    return dict(y=y_outs, state=state, y_ref=y_refs, state_ref=ref_state, config=config,
                workspace=workspace, read_levels=read_levels, write_levels=write_levels,
                widths=widths, B=B, H=H, N=N, device=device)


def _assert_oracle(out, tol=DECODE_RTOL):
    y, y_ref = out["y"][-1], out["y_ref"][-1]
    state, state_ref = out["state"], out["state_ref"]
    scale_y = max(1e-30, float(y_ref.abs().max()))
    scale_s = max(1e-30, float(state_ref.abs().max()))
    err_y = float((y.double() - y_ref).abs().max()) / scale_y
    err_s = float((state.double() - state_ref).abs().max()) / scale_s
    assert err_y < tol, f"decode output disagrees with the canonical oracle: relative {err_y:.3e}"
    assert err_s < tol, f"decode state disagrees with the canonical oracle: relative {err_s:.3e}"


# ---------------------------------------------------------------------------
# G3 -- the support authority, bit for bit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("widths", _TOPOLOGIES)
@pytest.mark.parametrize("p_read,p_write", [(1.0, 1.0), (0.4, 0.3)])
def test_g3_expansion_is_the_exact_leaf_support(widths, p_read, p_write):
    """The expansion IS the per-level exact-nonzero support, leaf for leaf.

    This is the one place the decode path's "membership is ``amp != 0``" claim is
    checked rather than argued. It is a bit-for-bit gate on a bit-for-bit object: a
    funnel shift that is off by one at an ``S_{l+1} % 32 != 0`` boundary moves exactly
    one bit, and nothing downstream would necessarily notice — the leaf it wrongly
    includes may carry a zero amplitude and contribute nothing to ``y``.

    Both sides go through the SAME device function, so both are stated here.
    """
    out = _case(widths, 64, 2, 3, "global", p_read, p_write, seed=17)
    B, H, device = out["B"], out["H"], out["device"]
    want_r = _leaf_support(out["read_levels"][0], widths, B, H, device)
    want_w = _leaf_support(out["write_levels"][0], widths, B, H, device)
    assert torch.equal(_kernel_expansion(out["read_levels"][0], widths, B, H, device),
                       want_r), "the read expansion is not the exact read support"
    assert torch.equal(_kernel_expansion(out["write_levels"][0], widths, B, H, device),
                       want_w), "the write expansion is not the exact write support"


def test_g3_the_coarse_index_carries_a_sparse_multi_group_topology():
    """``pc[g]`` is the exclusive popcount prefix, gated where it is load-bearing.

    The coarse index lives and dies inside the CTA that builds it, so what states it is
    the WALK it drives: a prefix that is wrong by one skips a group that has rows in it,
    and those rows vanish from ``y`` and from the state rather than failing loudly. The
    cell therefore asserts the fixture actually has empty groups to skip and a row range
    to partition, and then holds the whole step to the fp64 oracle.
    """
    widths = (64, 64)
    out = _case(widths, 64, 2, 3, "global", 0.05, 0.05, seed=99)
    config = out["config"]
    assert config.n_groups > 1, "the fixture is one coarse group: nothing is being skipped"
    assert config.n_split > 1, "the fixture is one CTA: no row range is being resolved"
    union = (_leaf_support(out["read_levels"][0], widths, out["B"], out["H"], out["device"])
             | _leaf_support(out["write_levels"][0], widths, out["B"], out["H"], out["device"]))
    per_group = union.view(-1, config.n_groups, kCOARSE_GROUP_LEAVES).any(-1)
    assert bool(union.any()), "the fixture touches no leaf at all"
    assert bool((~per_group).any()), "the fixture has no empty group for the walk to skip"
    _assert_oracle(out)


# ---------------------------------------------------------------------------
# G4 -- the fp64 oracle, over the instantiation matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("widths", _TOPOLOGIES)
@pytest.mark.parametrize("p_read,p_write", [(1.0, 1.0), (0.4, 0.3)])
def test_g4_matches_the_canonical_oracle(widths, p_read, p_write):
    _assert_oracle(_case(widths, 64, 2, 3, "global", p_read, p_write, seed=1234))


@pytest.mark.parametrize("d_v", [32, 64])
def test_g4_both_value_widths(d_v):
    _assert_oracle(_case((8, 16), d_v, 2, 2, "global", 0.5, 0.5, seed=77))


def test_the_empty_support_corner_is_inert_rather_than_undefined():
    """``M = 0``: no leaf is touched, ``y`` is zero, and the state is byte-unchanged.

    **This is a ROBUSTNESS row, not a semantics row, and the distinction is itself a
    finding.** ``M = 0`` requires some level to have no live digit on EITHER side, and
    every routing level is a SIMPLEX — it sums to 1, so it always has one. The canonical
    oracle enforces exactly that (``_oracle_contract.validate_simplex`` refuses an all-zero level),
    which means the corner is UNREACHABLE from a valid producer and cannot be gated
    against the oracle at all: there is no reference answer because there is no legal
    input. Asserting against the oracle here would have meant weakening the oracle.

    It is still worth pinning, because the empty ROW RANGE it exercises is very much
    reachable — the frozen ``n_split`` sizes the grid from a CEILING, so ordinary steps
    launch CTAs with empty ranges — and the degenerate forms of the coarse-index binary
    search, the boundary bit cuts and the split-K combine all live on that path. So the
    assertion is against the arithmetic identity (a sum over no terms is zero, and
    ``0 / (0 + eps)`` is zero), not against a reference.
    """
    device = "cuda"
    widths, d_v, B, H = (8, 16), 64, 1, 2
    topology = _topology(widths)
    zeros = tuple(torch.zeros(B, 1, H, w, device=device, dtype=torch.float32) for w in widths)
    g_write = torch.ones(B, 1, H, device=device, dtype=torch.float32)
    v = torch.randn(B, 1, H, d_v, device=device, dtype=torch.float32)
    config = derive_decode_geometry(topology, d_v=d_v, decay=False, BH=B * H, device=device)
    m0 = 0.1 * torch.randn(B, H, config.N, config.cols, device=device, dtype=torch.float32)
    state = m0.clone()
    y, state, _workspace = _decode_step(v, zeros, zeros, None, g_write, config, state)
    assert not _leaf_support(zeros, widths, B, H, device).any(), (
        "the fixture did not produce M = 0")
    assert torch.equal(y, torch.zeros_like(y))
    assert torch.equal(state, m0), "an untouched state must be byte-unchanged"


@pytest.mark.parametrize("n_split", [7, 64])
def test_more_ctas_than_rows_is_inert(n_split):
    """``n_split > M``: the surplus CTAs contribute nothing and the answer does not move.

    The reachable half of the corner above, and what makes a FROZEN ``n_split`` safe: the
    carrier sizes the grid from a ceiling, so a step whose realized support is far below
    it launches empty CTAs on every call. They must contribute an exact zero to the
    fixed-order combine, not an uninitialized workspace slot.
    """
    ref = _case((8, 16), 64, 1, 2, "global", 0.3, 0.3, seed=4, n_split=1)
    got = _case((8, 16), 64, 1, 2, "global", 0.3, 0.3, seed=4, n_split=n_split)
    assert torch.equal(ref["state"], got["state"])
    scale = max(1e-30, float(ref["y"][-1].abs().max()))
    assert float((got["y"][-1] - ref["y"][-1]).abs().max()) / scale < DECODE_RTOL


def test_the_global_residence_expansion_arm():
    """The expansion's SECOND residence arm, which no other row reaches.

    The arm is chosen from a topology fact (``n_words``) against a hardware fact (the
    per-block SMEM budget): the ping-pong pair is ``2 * ceil(N/32)`` words, so below
    roughly ``N = 196,608`` on an sm_86 device the intermediate levels live in SMEM and
    above it they live in global memory. Every other topology in this file fits SMEM, so
    without this row the global arm would ship having never executed — which is the
    failure mode the two-arm design is most exposed to, since the arms differ only in
    WHERE the intermediates live and a pointer mistake in the one that is never run
    looks exactly like a correct program.

    ``widths = (256, 256, 4)`` gives ``N = 262,144`` (``n_words = 8192``, a 64 KiB
    ping-pong pair, over the budget) and ``S_2 = 4 < 32``, so it exercises the
    shift-and-merge copy on the global arm rather than only the aligned one.
    """
    widths = (256, 256, 4)
    n_words = (math.prod(widths) + 31) // 32
    assert 2 * n_words * 4 > 48 * 1024, "this fixture no longer selects the global arm"
    #: `p = 0.5` rather than something thinner, and `B = 2, H = 3` rather than `1, 2`:
    #: `_simplex` refuses a vacuously dense fixture, and the width-4 level is narrow
    #: enough that a thin Bernoulli mask zeroes every entry and the empty-row repair
    #: hands back a uniform (dense) row. The sparsity that matters for this row is the
    #: TOPOLOGY's size, not the support's.
    _assert_oracle(_case(widths, 32, 2, 3, "global", 0.5, 0.5, seed=262144))


# ---------------------------------------------------------------------------
# G4 -- decay
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("widths", [(16,), (8, 16), (3, 5, 7), (4, 4, 4)])
@pytest.mark.parametrize("p_read,p_write", [(1.0, 1.0), (0.4, 0.3)])
def test_g4_decay_matches_the_canonical_oracle(widths, p_read, p_write):
    """Survival is inclusive at ``j = t``: the deposit at ``t`` is read at ``t`` with
    survival exactly 1.

    Decode realizes that by UPDATING the row and then reading it back, which is where
    the tiled kernel's causal intra Gram collapses to at ``T = 1``. Both off-by-ones —
    decaying after the deposit, or reading the row before the deposit — are O(1) errors
    rather than rounding, so this discriminates them at the fp32 tolerance without
    needing a tighter one. (An earlier revision of the kernel had exactly the second
    one; this row is what would have caught it.)
    """
    generator = torch.Generator(device="cuda").manual_seed(4242)
    dials = _decay_dials(widths, 3, generator, "cuda")
    _assert_oracle(_case(widths, 64, 2, 3, "global", p_read, p_write, seed=1234, decay_dials=dials))


@pytest.mark.parametrize("widths", [(8, 16), (3, 5, 7)])
def test_g4_decay_hot_dials(widths):
    """Near-unit retention, where ``1 - rate`` cancels catastrophically and only the
    ``log1pf`` composition recovers the bits."""
    generator = torch.Generator(device="cuda").manual_seed(5150)
    dials = _decay_dials(widths, 3, generator, "cuda", hot=True)
    _assert_oracle(_case(widths, 64, 2, 3, "global", 0.6, 0.5, seed=606, decay_dials=dials))


def test_g4_decay_on_the_stored_write_representation():
    """The clock and the deposit read ONE representation, and it is the bf16 one.

    An amplitude that flushed to zero in storage is absent from the deposit AND from
    the clock, so ``keep == 1`` there with nothing deposited — the regime the shipped
    producer actually hands decode, run against the oracle on the same tensors.
    """
    generator = torch.Generator(device="cuda").manual_seed(31337)
    dials = _decay_dials((8, 16), 3, generator, "cuda")
    _assert_oracle(_case((8, 16), 64, 2, 3, "global", 0.5, 0.4, seed=808,
                         decay_dials=dials, stored_bf16=True))


# ---------------------------------------------------------------------------
# G1 / G2 -- determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_split", [1, 2, 7, 64])
@pytest.mark.parametrize("widths", [(8, 16), (64, 64)])
def test_g1_is_byte_reproducible_against_itself(n_split, widths):
    """TWO RUNS OF THIS BINARY GIVE BYTE-IDENTICAL ``y`` AND ``state``.

    This is the gate the tiled consumer cannot pass and decode must. It is what the
    fixed-order slot combine buys, and a failure means either that combine is not
    fixed-order or the row partition is not a partition.
    """
    kwargs = dict(seed=2024, n_split=n_split)
    a = _case(widths, 64, 2, 3, "global", 0.4, 0.3, **kwargs)
    b = _case(widths, 64, 2, 3, "global", 0.4, 0.3, **kwargs)
    assert torch.equal(a["y"][-1], b["y"][-1]), "decode's y is not run-reproducible"
    assert torch.equal(a["state"], b["state"]), "decode's state is not run-reproducible"


@pytest.mark.parametrize("widths", [(8, 16), (64, 64)])
def test_g2_state_is_invariant_across_n_split(widths):
    """``state`` is byte-identical across ``n_split``, UNCONDITIONALLY.

    Unlike ``y``, the state is not a reduction: the row partition hands every touched
    row to exactly one CTA, warp and lane, so changing how the rows are split cannot
    change a single stored bit. ``y`` is a reduction and fp32 addition is not
    associative, so ``y`` is only byte-stable at FIXED ``n_split`` (which is what ships,
    since ``n_split`` is a pure function of the frozen config) and is checked at the
    oracle band across ``n_split``.
    """
    ref = _case(widths, 64, 2, 3, "global", 0.4, 0.3, seed=555, n_split=1)
    for n_split in (2, 7, 64):
        got = _case(widths, 64, 2, 3, "global", 0.4, 0.3, seed=555, n_split=n_split)
        assert torch.equal(ref["state"], got["state"]), (
            f"state moved with n_split={n_split}; the row partition is not a partition")
        scale = max(1e-30, float(ref["y"][-1].abs().max()))
        err = float((got["y"][-1] - ref["y"][-1]).abs().max()) / scale
        assert err < DECODE_RTOL, f"y across n_split={n_split} exceeds the band: {err:.3e}"


# ---------------------------------------------------------------------------
# G6 -- the multi-step recurrence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("widths", [(8, 16), (7, 9)])
@pytest.mark.parametrize("decay", [False, True])
def test_g6_sixty_four_chained_steps(widths, decay):
    """64 decode steps against 64 chained oracle steps.

    A single step hides a decay sign error whose effect is second order in one
    application and first order in sixty-four, and it hides a mass-column drift
    entirely. This is also the only row that exercises workspace REUSE across steps and
    the ``ctr`` reset the last-arriving CTA performs — a counter that failed to reset
    would deadlock or double-count on step 2, not step 1.
    """
    dials = None
    if decay:
        generator = torch.Generator(device="cuda").manual_seed(11)
        dials = _decay_dials(widths, 3, generator, "cuda")
    #: `B = 2, H = 3` rather than `1, 2`: `_simplex` refuses a vacuously dense fixture,
    #: and a narrow level drawn only twice reliably survives the Bernoulli mask intact.
    #: The sparsity is the point of the row, so the draw count has to support it.
    out = _case(widths, 64, 2, 3, "global", 0.5, 0.4, seed=64, steps=64, decay_dials=dials)
    _assert_oracle(out)


# ---------------------------------------------------------------------------
# G7 -- the prefill/decode seam, graded against the oracle
# ---------------------------------------------------------------------------


def _oracle_prefill(widths, d_v, B, T, H, generator, device):
    """The prefill leg on the fp64 ORACLE; return what decode needs.

    The PRODUCTION prefill arm ran this leg until the K31 deletion batch removed
    it (baseline = tag `baseline/pre-k31`); the seam this gate owns -- decode
    continuing from a canonical leaf-major entry state under the frozen carrier
    config -- survives it, because the oracle's `output_final_state` IS that
    state's definition. A hand-built bundle and not the producer: the fixture
    plants a specific density on each side, which a producer cannot be asked for.
    """
    topology = _topology(widths)
    read_levels = tuple(_simplex((B, T, H, w), 0.5, generator, device) for w in widths)
    write_levels = tuple(_simplex((B, T, H, w), 0.4, generator, device) for w in widths)
    g_write = torch.rand(B, T, H, device=device, dtype=torch.float64, generator=generator) + 0.5
    v = torch.randn(B, T, H, d_v, device=device, dtype=torch.float64, generator=generator)
    y, state = naive_rola(v, read_levels, write_levels, None, g_write,
                          topology, None, "global", output_final_state=True)
    return dict(topology=topology, read_levels=read_levels, write_levels=write_levels,
                g_read=None, g_write=g_write, v=v, y=y,
                state=state.float())


@pytest.mark.parametrize("widths", [(16, 16), (16, 16, 16)],
                         ids=lambda w: "x".join(map(str, w)))
def test_g7_prefill_decode_seam(widths):
    """Prefill ``L`` on the fp64 oracle, decode token ``L + 1`` on the KERNEL,
    and grade the pair against the fp64 oracle over the whole ``L + 1`` sequence.

    This is the gate on the frozen carrier being the RIGHT config: an ``eps``
    that disagreed with the readout's, a head fold that disagreed with
    ``h = bh % H``, or a state layout that was not leaf-major would each show up
    here and nowhere else. The prefill leg ran on the production chunk arm until
    the K31 deletion (its charge was measured at 5.2e-3 / 6.4e-3 against the
    1e-2 band, with an oracle CONTROL to keep the blame assignable); the seam's
    entry state is now the oracle's own, narrowed to the kernel's fp32, so the
    number is decode's alone.
    """
    device = "cuda"
    d_v, B, H = 64, 2, 2
    L = 4 * _chunk_tokens()
    seed = 909
    pre = _oracle_prefill(widths, d_v, B, L, H,
                          torch.Generator(device=device).manual_seed(seed), device)

    # the step's own token: drawn from the same generator so the fixture is one
    # sequence of L + 1 tokens rather than a prefill and an unrelated step.
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    step_read = tuple(_simplex((B, 1, H, w), 0.5, gen, device) for w in widths)
    step_write = tuple(_simplex((B, 1, H, w), 0.4, gen, device) for w in widths)
    step_gw = torch.rand(B, 1, H, device=device, dtype=torch.float64, generator=gen) + 0.5
    step_v = torch.randn(B, 1, H, d_v, device=device, dtype=torch.float64, generator=gen)

    config = derive_decode_geometry(pre["topology"], d_v=d_v, decay=False, BH=B * H,
                                  device=device)
    y_dec, state_dec, _ = _decode_step(
        step_v.float(), tuple(t.float() for t in step_read),
        tuple(t.float() for t in step_write), None, step_gw.float(),
        config, pre["state"].clone())

    #: THE WHOLE SEQUENCE, on the oracle: prefill and step concatenated, so the
    #: reference sees exactly what the two kernel legs together saw.
    cat = lambda a, b: tuple(torch.cat([x, s], dim=1) for x, s in zip(a, b))  # noqa: E731
    y_ref, state_ref = naive_rola(
        torch.cat([pre["v"], step_v], dim=1),
        cat(pre["read_levels"], step_read), cat(pre["write_levels"], step_write),
        None, torch.cat([pre["g_write"], step_gw], dim=1),
        pre["topology"], None, "global", output_final_state=True)

    scale_y = max(1e-30, float(y_ref[:, L:L + 1].abs().max()))
    scale_s = max(1e-30, float(state_ref.abs().max()))
    err_y = float((y_dec.double() - y_ref[:, L:L + 1]).abs().max()) / scale_y
    err_s = float((state_dec.double() - state_ref).abs().max()) / scale_s
    assert err_y < DECODE_RTOL, f"the prefill/decode seam moved y: {err_y:.3e}"
    assert err_s < DECODE_RTOL, f"the prefill/decode seam moved state: {err_s:.3e}"
    assert pre["y"].shape[1] == L
