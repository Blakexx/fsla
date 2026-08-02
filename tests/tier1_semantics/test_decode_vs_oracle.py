# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""TIER 1 — the T=1 decode path against the canonical fp64 oracle and the tiled kernel.

Gates G1-G7 of the Track D design (`rola-scratch/workflows/20260802-thunder/trackD.md`).

**WHY "BIT-IDENTITY VS THE TILED CONSUMER" IS NOT ONE OF THEM, and this has to be said
before the tolerances make sense.** Three independent facts make byte-equality between
decode and the tiled consumer unattainable, so a gate asking for it would be asking for
something no implementation can deliver:

1. The tiled consumer's ``y`` is not reproducible against ITSELF wherever a ``bh`` holds
   more than one owner — it emits split-K partials through ``atomicAdd`` and CUDA fixes
   no inter-CTA order (Track K measured 384 of 976 fixture cases differing between two
   runs of one binary). Decode is deterministic by design.
2. Summation order differs by construction: the tiled kernel contracts inside an
   ``m16n8k16`` fragment tree per owner then sums owners by atomic; decode contracts in
   leaf order in fp32 FMA chains and folds 8 warps then ``n_split`` CTAs in fixed order.
3. The precision path differs: the tiled readout narrows its operands to bf16 (or
   tf32x3) for the MMA; decode has no MMA and stays fp32. Decode is the *more* accurate
   of the two, which is the safe direction but is still a difference.

So the gates are: byte-equality where decode is *supposed* to be exact (G1, G2, the
BC-blindness row in tier 2), and the oracle band everywhere the two consumers are
compared (G4, G5, G6, G7). Decode's byte gates are STRONGER than the tiled kernel's,
not weaker — G1 is a gate the tiled consumer cannot pass.

Run: ``pytest tests/tier1_semantics/test_decode_vs_oracle.py``
"""

from __future__ import annotations

import math

import pytest
import torch

from rola.ops.decode import decode_forward, freeze_decode_config
from rola.ops.naive import naive_rola
from rola.routing.types import LeafMassDecay
from tests.tier1_semantics.test_consumer_vs_oracle import (
    RTOL,
    _decay_dials,
    _simplex,
    _topology,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="decode is a CUDA kernel"),
]

#: DECODE'S TOLERANCE IS THE TILED KERNEL'S ``tf32x3`` NUMBER, ADOPTED RATHER THAN
#: INVENTED. Both consumers accumulate the readout in fp32 over at most ``N`` leaf
#: terms; the tiled ``tf32x3`` arm reaches that accuracy by splitting an fp32 operand
#: into three tf32 pieces, decode by never leaving fp32 (it has no MMA). Decode's error
#: is therefore bounded by the same fp32-accumulation analysis ``RTOL`` was derived
#: from, and is expected to land strictly INSIDE it. A decode-specific constant would
#: be a number with no derivation of its own, which `docs/testing.md` calls a defect.
DECODE_RTOL = RTOL

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


def _pack(mask):
    """``[BH, N]`` bool -> ``[BH, ceil(N/32)]`` int64 holding the packed words."""
    BH, N = mask.shape
    n_words = (N + 31) // 32
    padded = torch.zeros(BH, n_words * 32, dtype=torch.int64, device=mask.device)
    padded[:, :N] = mask.to(torch.int64)
    weights = (1 << torch.arange(32, device=mask.device, dtype=torch.int64))
    return (padded.view(BH, n_words, 32) * weights).sum(-1)


def _u32(t):
    """The kernel's int32 words as unsigned values, for comparison against ``_pack``."""
    return t.to(torch.int64) & 0xFFFFFFFF


def _case(widths, d_v, B, H, norm, p_read, p_write, *, seed, decay_dials=None,
          n_split=None, nonzero_m0=True, steps=1, quantize_stored=False):
    """Build one fixture, run the oracle and decode, return everything both produced."""
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(seed)
    topology = _topology(widths, d_v)
    N = math.prod(widths)
    cols = d_v + (1 if norm == "global" else 0)

    read_levels, write_levels, g_reads, g_writes, vs = [], [], [], [], []
    for _ in range(steps):
        read_levels.append(tuple(_simplex((B, 1, H, w), p_read, generator, device) for w in widths))
        write_levels.append(tuple(_simplex((B, 1, H, w), p_write, generator, device) for w in widths))
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
        stored = None
        if quantize_stored:
            stored = tuple(t.to(torch.bfloat16).to(torch.float64) for t in write_levels[s])
        y_ref, ref_state = naive_rola(
            vs[s], read_levels[s], write_levels[s], g_reads[s], g_writes[s], topology,
            decay, norm, write_levels_stored=stored,
            initial_state=ref_state, output_final_state=True)
        y_refs.append(y_ref)

    # ---- decode ---------------------------------------------------------------
    config = freeze_decode_config(topology, norm, decay=decay is not None, BH=B * H,
                                  device=device, n_split=n_split)
    state = (torch.zeros(B, H, N, cols, device=device, dtype=torch.float32)
             if initial_state is None else initial_state.to(torch.float32).clone())
    workspace = None
    y_outs = []
    for s in range(steps):
        stored = None
        if quantize_stored:
            stored = tuple(t.to(torch.bfloat16).to(torch.float32) for t in write_levels[s])
        y, state, workspace = decode_forward(
            vs[s].float(), tuple(t.float() for t in read_levels[s]),
            tuple(t.float() for t in write_levels[s]),
            None if g_reads[s] is None else g_reads[s].float(), g_writes[s].float(),
            config, state, decay=decay, write_levels_stored=stored, workspace=workspace)
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
    """The expanded bitmaps ARE the per-level exact-nonzero support, leaf for leaf.

    This is the one place the decode path's "membership is ``amp != 0``" claim is
    checked rather than argued. It is a bit-for-bit gate on a bit-for-bit object: a
    funnel shift that is off by one at an ``S_{l+1} % 32 != 0`` boundary moves exactly
    one bit, and nothing downstream would necessarily notice — the leaf it wrongly
    includes may carry a zero amplitude and contribute nothing to ``y``.
    """
    out = _case(widths, 64, 2, 3, "global", p_read, p_write, seed=17)
    BH, n_words = out["B"] * out["H"], out["config"].n_words
    bits = out["workspace"].bits
    got_r = _u32(bits[:BH * n_words].view(BH, n_words))
    got_w = _u32(bits[BH * n_words:2 * BH * n_words].view(BH, n_words))

    want_r_mask = _leaf_support(out["read_levels"][0], widths, out["B"], out["H"], out["device"])
    want_w_mask = _leaf_support(out["write_levels"][0], widths, out["B"], out["H"], out["device"])
    assert torch.equal(got_r, _pack(want_r_mask)), "the read bitmap is not the exact read support"
    assert torch.equal(got_w, _pack(want_w_mask)), "the write bitmap is not the exact write support"

    # and the coarse index's total is the touched-row count the walk will visit
    union = want_r_mask | want_w_mask
    assert torch.equal(out["workspace"].pc_coarse[:, -1].to(torch.int64),
                       union.sum(-1)), "pc_coarse's total is not |supp(R) union supp(W)|"


@pytest.mark.parametrize("widths", [(8, 16), (3, 5, 7)])
def test_g3_coarse_index_is_the_group_prefix(widths):
    """``pc_coarse[g]`` is the exclusive popcount prefix over 32-word groups.

    Gated separately from the total because the WALK consults it to skip empty groups: a
    prefix that is right only in aggregate would skip a group that has rows in it, and
    those rows would silently vanish from ``y`` rather than fail loudly.
    """
    out = _case(widths, 64, 2, 3, "global", 0.4, 0.3, seed=99)
    BH, n_words = out["B"] * out["H"], out["config"].n_words
    bits = out["workspace"].bits
    got_r = _u32(bits[:BH * n_words].view(BH, n_words))
    got_w = _u32(bits[BH * n_words:2 * BH * n_words].view(BH, n_words))
    per_word = torch.tensor(
        [[bin(int(x)).count("1") for x in row] for row in (got_r | got_w).cpu()],
        dtype=torch.int64)
    n_groups = out["config"].n_groups
    padded = torch.zeros(BH, n_groups * 32, dtype=torch.int64)
    padded[:, :n_words] = per_word
    group = padded.view(BH, n_groups, 32).sum(-1)
    want = torch.cat([torch.zeros(BH, 1, dtype=torch.int64), group.cumsum(-1)], dim=-1)
    assert torch.equal(out["workspace"].pc_coarse.cpu().to(torch.int64), want)


# ---------------------------------------------------------------------------
# G4 -- the fp64 oracle, over the instantiation matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("widths", _TOPOLOGIES)
@pytest.mark.parametrize("norm", ["raw", "global"])
@pytest.mark.parametrize("p_read,p_write", [(1.0, 1.0), (0.4, 0.3)])
def test_g4_matches_the_canonical_oracle(widths, norm, p_read, p_write):
    _assert_oracle(_case(widths, 64, 2, 3, norm, p_read, p_write, seed=1234))


@pytest.mark.parametrize("d_v", [32, 64])
@pytest.mark.parametrize("norm", ["raw", "global"])
def test_g4_both_value_widths(d_v, norm):
    _assert_oracle(_case((8, 16), d_v, 2, 2, norm, 0.5, 0.5, seed=77))


def test_the_empty_support_corner_is_inert_rather_than_undefined():
    """``M = 0``: no leaf is touched, ``y`` is zero, and the state is byte-unchanged.

    **This is a ROBUSTNESS row, not a semantics row, and the distinction is itself a
    finding.** ``M = 0`` requires some level to have no live digit on EITHER side, and
    every routing level is a SIMPLEX — it sums to 1, so it always has one. The canonical
    oracle enforces exactly that (``naive._validate_simplex`` refuses an all-zero level),
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
    topology = _topology(widths, d_v)
    zeros = tuple(torch.zeros(B, 1, H, w, device=device, dtype=torch.float32) for w in widths)
    g_write = torch.ones(B, 1, H, device=device, dtype=torch.float32)
    v = torch.randn(B, 1, H, d_v, device=device, dtype=torch.float32)
    config = freeze_decode_config(topology, "global", decay=False, BH=B * H, device=device)
    m0 = 0.1 * torch.randn(B, H, config.N, config.cols, device=device, dtype=torch.float32)
    state = m0.clone()
    y, state, workspace = decode_forward(v, zeros, zeros, None, g_write, config, state)
    assert int(workspace.pc_coarse[:, -1].max()) == 0, "the fixture did not produce M = 0"
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
@pytest.mark.parametrize("norm", ["raw", "global"])
@pytest.mark.parametrize("p_read,p_write", [(1.0, 1.0), (0.4, 0.3)])
def test_g4_decay_matches_the_canonical_oracle(widths, norm, p_read, p_write):
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
    _assert_oracle(_case(widths, 64, 2, 3, norm, p_read, p_write, seed=1234, decay_dials=dials))


@pytest.mark.parametrize("widths", [(8, 16), (3, 5, 7)])
def test_g4_decay_hot_dials(widths):
    """Near-unit retention, where ``1 - rate`` cancels catastrophically and only the
    ``log1pf`` composition recovers the bits."""
    generator = torch.Generator(device="cuda").manual_seed(5150)
    dials = _decay_dials(widths, 3, generator, "cuda", hot=True)
    _assert_oracle(_case(widths, 64, 2, 3, "global", 0.6, 0.5, seed=606, decay_dials=dials))


def test_g4_decay_with_a_quantized_clock():
    """The clock is the STORED write allocation, the deposit is the LIVE one.

    A leaf whose stored amplitude flushed to zero has ``keep == 1`` with a live deposit,
    and decode must still deposit it — the clock may not gate membership.
    """
    generator = torch.Generator(device="cuda").manual_seed(31337)
    dials = _decay_dials((8, 16), 3, generator, "cuda")
    _assert_oracle(_case((8, 16), 64, 2, 3, "global", 0.5, 0.4, seed=808,
                         decay_dials=dials, quantize_stored=True))


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
# G5 / G7 -- against the tiled consumer, and across the prefill/decode seam
# ---------------------------------------------------------------------------

def _prefill(widths, d_v, B, T, H, norm, generator, device, BC, *, decay=None):
    """Run the TILED consumer over ``T`` tokens and return everything decode needs."""
    from rola.ops.consumer import _consumer_forward
    from rola.planner.schedule import Plan

    topology = _topology(widths, d_v)
    read_levels = tuple(_simplex((B, T, H, w), 0.5, generator, device) for w in widths)
    write_levels = tuple(_simplex((B, T, H, w), 0.4, generator, device) for w in widths)
    g_write = torch.rand(B, T, H, device=device, dtype=torch.float64, generator=generator) + 0.5
    g_read = (None if norm == "global"
              else torch.rand(B, T, H, device=device, dtype=torch.float64, generator=generator) + 0.5)
    v = torch.randn(B, T, H, d_v, device=device, dtype=torch.float64, generator=generator)
    plan = Plan(topology=topology, normalization=norm, BT=16, BC=BC,
                warp_tile=(1, max(1, d_v // 32)), decay=decay is not None,
                mma_precision="tf32x3", compact=True)
    y, state, _ = _consumer_forward(
        v.float(), tuple(t.float() for t in read_levels), tuple(t.float() for t in write_levels),
        None if g_read is None else g_read.float(), g_write.float(), plan, decay=decay)
    return dict(topology=topology, read_levels=read_levels, write_levels=write_levels,
                g_read=g_read, g_write=g_write, v=v, y=y, state=state, plan=plan)


@pytest.mark.parametrize("widths,BC", [((8, 16), 16), ((64, 64), 64), ((4, 4, 4), 16)])
@pytest.mark.parametrize("norm", ["raw", "global"])
def test_g5_agrees_with_the_tiled_consumer_at_T_equals_one(widths, BC, norm):
    """Both consumers, the same one-token input, the same entry state.

    At the oracle band and NOT at byte level — see the module docstring for the three
    independent reasons byte-equality is unavailable here.
    """
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(4321)
    pre = _prefill(widths, 64, 2, 1, 2, norm, generator, device, BC)

    config = freeze_decode_config(pre["topology"], norm, decay=False, BH=2 * 2, device=device)
    state = torch.zeros(2, 2, config.N, config.cols, device=device, dtype=torch.float32)
    y, state, _ = decode_forward(
        pre["v"].float(), tuple(t.float() for t in pre["read_levels"]),
        tuple(t.float() for t in pre["write_levels"]),
        None if pre["g_read"] is None else pre["g_read"].float(), pre["g_write"].float(),
        config, state)

    scale_y = max(1e-30, float(pre["y"].abs().max()))
    scale_s = max(1e-30, float(pre["state"].abs().max()))
    err_y = float((y - pre["y"]).abs().max()) / scale_y
    err_s = float((state - pre["state"]).abs().max()) / scale_s
    assert err_y < DECODE_RTOL, f"decode disagrees with the tiled consumer on y: {err_y:.3e}"
    assert err_s < DECODE_RTOL, f"decode disagrees with the tiled consumer on state: {err_s:.3e}"


@pytest.mark.parametrize("widths,BC", [((8, 16), 16), ((4, 4, 4), 16)])
@pytest.mark.parametrize("norm", ["raw", "global"])
def test_g7_prefill_decode_seam(widths, BC, norm):
    """Prefill ``L`` with the tiled kernel then decode token ``L+1``, versus prefilling
    ``L+1``.

    This is the gate on the frozen carrier being the RIGHT config: an ``eps`` that
    disagreed with ``divide_kernel``'s, a head fold that disagreed with the consumer's
    ``h = bh % H``, or a state layout that was not leaf-major would each show up here
    and nowhere else.
    """
    device = "cuda"
    d_v, B, H, L = 64, 2, 2, 33
    seed = 909
    full = _prefill(widths, d_v, B, L + 1, H, norm,
                    torch.Generator(device=device).manual_seed(seed), device, BC)
    # the same routing, truncated: same generator, same draw order
    gen = torch.Generator(device=device).manual_seed(seed)
    _ = _prefill(widths, d_v, B, L + 1, H, norm, gen, device, BC)  # replay to fix the draws

    from rola.ops.consumer import _consumer_forward
    head = {k: (tuple(t[:, :L] for t in v) if isinstance(v, tuple)
                else (v[:, :L] if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == L + 1 else v))
            for k, v in full.items() if k in ("read_levels", "write_levels", "g_read", "g_write", "v")}
    y_pre, state_pre, _ = _consumer_forward(
        head["v"].float(), tuple(t.float() for t in head["read_levels"]),
        tuple(t.float() for t in head["write_levels"]),
        None if head["g_read"] is None else head["g_read"].float(), head["g_write"].float(),
        full["plan"])

    config = freeze_decode_config(full["topology"], norm, decay=False, BH=B * H, device=device)
    tail = {k: (tuple(t[:, L:L + 1] for t in v) if isinstance(v, tuple)
                else (v[:, L:L + 1] if torch.is_tensor(v) and v.dim() >= 2 else v))
            for k, v in full.items() if k in ("read_levels", "write_levels", "g_read", "g_write", "v")}
    y_dec, state_dec, _ = decode_forward(
        tail["v"].float(), tuple(t.float() for t in tail["read_levels"]),
        tuple(t.float() for t in tail["write_levels"]),
        None if tail["g_read"] is None else tail["g_read"].float(), tail["g_write"].float(),
        config, state_pre.clone())

    scale_y = max(1e-30, float(full["y"][:, L:L + 1].abs().max()))
    scale_s = max(1e-30, float(full["state"].abs().max()))
    err_y = float((y_dec - full["y"][:, L:L + 1]).abs().max()) / scale_y
    err_s = float((state_dec - full["state"]).abs().max()) / scale_s
    assert err_y < DECODE_RTOL, f"the prefill/decode seam moved y: {err_y:.3e}"
    assert err_s < DECODE_RTOL, f"the prefill/decode seam moved state: {err_s:.3e}"
    assert y_pre.shape[1] == L
