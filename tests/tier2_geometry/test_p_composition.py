"""§6.4 sequence parallelism (`P` per owner): the kernel-side gate.

``ROLA_V3_SPEC.md`` §6.4, contract and policy NORMATIVE, amended 2026-07-30.

**The state discipline this file enforces is threefold, and the amendment that put
it there is worth restating because it is easy to misread as a relaxation.** The
§6.4 monoid ``(A1,B1) o (A2,B2) = (A2*A1, A2*B1 + B2)`` is associative and exact in
EXACT arithmetic. The kernel accumulates state in fp32, where addition is not
associative, so the fold is *re-association-equivalent* to the serial ``P = 1``
order rather than bitwise identical to it -- ``P = 1`` interleaves the entry state
into every FMA, the fold multiplies it by a separately accumulated ``A`` and adds
once. Measured: 0 of 8 fixtures bitwise, at 1-6 fp32 ulp
(``workflows/20260730-wiring/pincrement.md`` §1-§2). So:

1. ``P > 1`` is graded against the canonical fp64 oracle ``naive_rola`` at
   :data:`BF16_RTOL` -- **the same number ``P = 1`` is held to**, imported from the
   ``P = 1`` gate file rather than restated, so it cannot drift apart from it.
2. **The unwritten-leaf carve-out is BITWISE and is gated as such.** A leaf with no
   write in a segment has ``A = 1`` and ``B = 0`` exactly, and the scan never visits
   it, so it must be ``torch.equal`` to the serial result. This is the part that
   detects genuine fold bugs, and every test that asserts it FIRST asserts that its
   own fixture contains written leaves that are *not* bit-identical -- otherwise the
   test would pass on a kernel that had no fold at all.
3. A derived re-association envelope bounds ``P > 1`` against ``P = 1``, with a null
   control, and is documented as an ENVELOPE, NOT A DETECTOR.

**The serial reference is `P = 1` itself, and that is not a re-derived reference.**
Where a test needs "what the serial order would have produced", it uses a
``P = 1`` launch of the same fixture -- the canonical path, already gated against
the fp64 oracle by ``test_rola_v3_consumer.py``. The
``matching-the-naive-antipattern`` note is why: a fold checked against a
re-derived fold proves self-consistency and nothing else.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rola.ops.consumer import _consumer_forward
from rola.ops.naive import naive_rola
from rola.ops.paging import PageArena
from rola.planner.schedule import Plan, build_schedule
from rola.routing.sequence_parallel import (
    SequenceParallelPlan,
    SequenceParallelPolicy,
    build_sequence_parallel_tensors,
    owner_slot_counts,
    select_sequence_parallelism,
)
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    LeafMassDecay,
    LevelConfig,
    SoftmaxActivation,
    Topology,
)

# The `P = 1` gate's own tolerance, IMPORTED rather than restated. §6.4's amended
# discipline is "the same BF16_RTOL as P = 1, no tolerance widened anywhere"; importing
# it is what makes that structural instead of a promise.
from tests.tier1_semantics.test_consumer_vs_oracle import BF16_RTOL

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the V3 consumer is a CUDA kernel"),
]

_ROUTING = IndependentRouting(
    tied=False, read_activation=SoftmaxActivation(), write_activation=SoftmaxActivation())

#: §7.1's mark phase reads the WRITE activation structurally (`can_zero`): a softmax
#: level cannot emit an exact zero, so the arena takes its dense short-circuit and
#: commits every page. A partially-resident arena -- the composition R4 is about --
#: therefore requires an entmax-declared write router, and the fixture must declare
#: what it emits ([[sparsity-comes-from-entmax]]).
_SPARSE_WRITE_ROUTING = IndependentRouting(
    tied=False, read_activation=SoftmaxActivation(), write_activation=EntmaxActivation(1.5))


def _topology(widths, d_v, routing=_ROUTING):
    return Topology(levels=tuple(LevelConfig(w, routing) for w in widths), d_v=d_v)


def _simplex(shape, p_nonzero, generator, device):
    x = torch.rand(shape, device=device, dtype=torch.float64, generator=generator)
    if p_nonzero < 1.0:
        keep = torch.rand(shape, device=device, dtype=torch.float64, generator=generator) < p_nonzero
        x = x * keep
    x = torch.where(x.sum(-1, keepdim=True) == 0, torch.ones_like(x), x)
    result = x / x.sum(-1, keepdim=True)
    if p_nonzero < 1.0:
        assert bool((result == 0).any()), "the sparse fixture is vacuously dense"
    return result


def _masked_simplex(shape, mask, generator, device):
    """A simplex whose support is EXACTLY ``mask``. Structured supports are what the
    carve-out is about; a Bernoulli mask cannot express "this digit is dead over
    exactly this token range"."""
    mask = mask.to(device=device, dtype=torch.float64).expand(shape)
    assert bool((mask.sum(-1) > 0).all()), "every row must have a nonempty support"
    x = torch.rand(shape, device=device, dtype=torch.float64, generator=generator) * mask
    x = torch.where(x.sum(-1, keepdim=True) == 0, mask, x)
    return x / x.sum(-1, keepdim=True)


class _Fixture:
    """One routing realization plus the plan and schedule built from it."""

    def __init__(self, widths, BC, d_v, B, T, H, p_read, p_write, seed, *,
                 decay_on=False, nonzero_m0=False,
                 read_levels=None, write_levels=None, routing=_ROUTING):
        device = "cuda"
        gen = torch.Generator(device=device).manual_seed(seed)
        self.topology = _topology(widths, d_v, routing)
        self.d_v = d_v
        # `normalization="global"` is the only value `Plan` accepts, so the mass
        # column is unconditional: `cols = d_v + 1` always.
        self.cols = d_v + 1
        self.read_levels = read_levels or tuple(
            _simplex((B, T, H, w), p_read, gen, device) for w in widths)
        self.write_levels = write_levels or tuple(
            _simplex((B, T, H, w), p_write, gen, device) for w in widths)
        self.g_write = torch.rand(B, T, H, device=device, dtype=torch.float64, generator=gen) + 0.5
        self.g_read = None
        self.v = torch.randn(B, T, H, d_v, device=device, dtype=torch.float64, generator=gen)
        self.decay = None
        if decay_on:
            self.decay = LeafMassDecay(dials=tuple(
                0.05 + 0.85 * torch.rand(H, w, device=device, dtype=torch.float64, generator=gen)
                for w in widths))
        self.initial_state = None
        if nonzero_m0:
            m0 = 0.1 * torch.randn(B, H, self.topology.N, self.cols, device=device,
                                   dtype=torch.float64, generator=gen)
            # The mass column is a nonnegative accumulated write mass; a negative
            # incoming mass is not a state the recurrence can produce.
            m0[..., d_v] = m0[..., d_v].abs()
            self.initial_state = m0
        self.plan = Plan(topology=self.topology, normalization="global", BC=BC,
                           warp_tile=(1, max(1, d_v // 32)), decay=decay_on)
        self._f32 = dict(
            v=self.v.to(torch.float32),
            read_levels=tuple(t.to(torch.float32) for t in self.read_levels),
            write_levels=tuple(t.to(torch.float32) for t in self.write_levels),
            g_read=None if self.g_read is None else self.g_read.to(torch.float32),
            g_write=self.g_write.to(torch.float32))
        self.schedule = build_schedule(
            self._f32["read_levels"], self._f32["write_levels"],
            self._f32["g_read"], self._f32["g_write"], self.plan)
        self.n_owners = int(self.schedule.owners.shape[0])

    def oracle(self):
        return naive_rola(
            self.v, self.read_levels, self.write_levels, self.g_read, self.g_write,
            self.topology, self.decay, "global",
            initial_state=self.initial_state, output_final_state=True)

    def run(self, P=1, *, sequence_parallel=None):
        """Launch the consumer. ``P=1`` passes NO sequence-parallel plan at all."""
        sp = sequence_parallel
        if sp is None and P != 1:
            sp = self.forced(P)
        return _consumer_forward(
            self._f32["v"], self._f32["read_levels"], self._f32["write_levels"],
            self._f32["g_read"], self._f32["g_write"], self.plan,
            schedule=self.schedule, sequence_parallel=sp, decay=self.decay,
            initial_state=(None if self.initial_state is None
                           else self.initial_state.to(torch.float32)))

    def policy_plan(self, **kwargs):
        return select_sequence_parallelism(self.plan, self.schedule, **kwargs)

    def forced(self, P):
        """A plan that splits EVERY owner ``P`` ways, bypassing the admission gates.

        The gates are a performance policy; the kernel's exactness must not depend on
        which `P` the policy happened to admit, so the numeric gates drive `P`
        directly. The policy's own behaviour is gated in
        ``test_rola_v3_sequence_parallel_policy.py``.
        """
        base = self.policy_plan(policy=SequenceParallelPolicy(workspace_cap=1 << 30))
        n_tiles = self.schedule.n_twords
        assert n_tiles >= P, (
            f"a fixture cannot be split {P} ways over {n_tiles} tiles; raise T or lower P, "
            "rather than letting the split silently degenerate to empty segments")
        Ps = tuple([P] * self.n_owners)
        return SequenceParallelPlan(
            P=Ps, limiting_gate=(None,) * self.n_owners, binding_gate=None,
            sms=base.sms, blocks_per_sm=base.blocks_per_sm,
            waves_before=base.waves_before, waves_after=0.0,
            base_traffic_bytes=0, roundtrip_bytes=0, roundtrip_seconds=0.0,
            base_traffic_seconds=0.0, ledger=base.ledger,
            n_slots=sum(owner_slot_counts(Ps)), BC=base.BC, cols=base.cols)


def _relative(actual: torch.Tensor, reference: torch.Tensor) -> float:
    scale = max(1e-30, float(reference.abs().max()))
    return float((actual.double() - reference).abs().max()) / scale


# ---------------------------------------------------------------------------
# 1. The oracle grid, at every P, at the P = 1 tolerance
# ---------------------------------------------------------------------------

#: Structural axes only. `P` composes with the EXISTING plan axes, which is the
#: claim the grid exists to test rather than assume: each row names a different
#: owner rectangle, and every row is run at both decay arms, sparse and dense,
#: and with and without a carried state. `BT` is gone (no template axis left to
#: vary) and `normalization` is pinned "global" (the only value `Plan` accepts).
_GRID = [
    ((8, 8), 16, 64),      # widths, BC, d_v -- j=0, c=2, 4 owners/bh
    ((4, 4, 4), 16, 64),   # j=1, one FIXED level
    ((2, 3, 4, 4), 16, 64),  # j=2, two fixed levels, non-power-of-two widths
    ((8, 8), 64, 64),      # BC == N: span everything
    ((16,), 16, 32),       # D = 1, d_v = 32
]


@pytest.mark.parametrize("widths,BC,d_v", _GRID)
@pytest.mark.parametrize("P", [1, 2, 4])
@pytest.mark.parametrize("decay_on", [False, True])
def test_every_P_agrees_with_the_canonical_oracle_at_the_P_equals_one_tolerance(
        widths, BC, d_v, P, decay_on):
    """§6.4 discipline 1: no tolerance is widened for `P > 1`, anywhere.

    ``BF16_RTOL`` is imported from the ``P = 1`` gate, so this cannot drift from it.
    """
    fx = _Fixture(widths, BC, d_v, B=1, T=128, H=2,
                  p_read=1.0, p_write=1.0, seed=1000 + P, decay_on=decay_on)
    y_ref, state_ref = fx.oracle()
    y, state, _ = fx.run(P)
    assert _relative(y, y_ref) < BF16_RTOL, f"y disagrees with the oracle at P={P}"
    assert _relative(state, state_ref) < BF16_RTOL, f"state disagrees with the oracle at P={P}"


@pytest.mark.parametrize("P", [1, 2, 4])
@pytest.mark.parametrize("p_read,p_write", [(0.4, 0.4), (1.0, 0.3), (0.3, 1.0)])
@pytest.mark.parametrize("nonzero_m0", [False, True])
def test_the_sparse_and_carried_state_arms_hold_at_every_P(P, p_read, p_write, nonzero_m0):
    """The density axis is a FIXTURE axis, never a kernel axis -- and a carried
    ``M_0`` is the case where the fold's re-association actually bites at every
    segment boundary rather than only past the first.

    P26: ``T=128``, not the old ``96``. The scan word is now the fixed 32-token
    granule (``TOKEN_WORD``, no more ``BT=16``), so ``n_twords = ceil(T/32)``
    must be at least ``P`` -- ``96`` gives only 3 words, one short of the
    ``P=4`` cell.
    """
    fx = _Fixture((8, 8), 16, 64, B=2, T=128, H=2,
                  p_read=p_read, p_write=p_write, seed=2000 + P, nonzero_m0=nonzero_m0)
    y_ref, state_ref = fx.oracle()
    y, state, _ = fx.run(P)
    assert _relative(y, y_ref) < BF16_RTOL
    assert _relative(state, state_ref) < BF16_RTOL


@pytest.mark.parametrize("P", [1, 2, 3, 4])
def test_a_ragged_token_tail_holds_at_every_P(P):
    """``T`` not a multiple of the 32-token scan word puts a partial word at the
    end, and an uneven ``n_twords / P`` puts differently-sized segments in one
    owner. Both are the off-by-one surface the tile-range mask owns.

    P26: ``T=131``, not the old ``83`` -- ``TOKEN_WORD=32`` (no more ``BT=16``)
    means ``n_twords = ceil(T/32)``, and the `P=4` cell needs at least 4 words
    (``83`` gave only 3); ``131`` is still ragged (not a multiple of 32) and
    gives 5.
    """
    fx = _Fixture((8, 8), 16, 64, B=1, T=131, H=2,
                  p_read=0.6, p_write=0.6, seed=3000 + P)
    y_ref, state_ref = fx.oracle()
    y, state, _ = fx.run(P)
    assert _relative(y, y_ref) < BF16_RTOL
    assert _relative(state, state_ref) < BF16_RTOL


@pytest.mark.parametrize("P", [1, 2, 3, 4])
@pytest.mark.parametrize("decay_on", [False, True])
def test_P_composes_with_a_carried_entry_state_on_both_decay_arms(P, decay_on):
    """`P` is an axis of the WORK LIST, and pass 1 is where it meets the kernel's own
    structures: pass 1 ALIASES its summary state onto the read-coefficient buffer and
    forces ``is_read = false``, which is the predicate every staged buffer's lifetime
    is argued from. That the split composes with a CARRIED entry state -- sparse
    routing, ``M_0 != 0`` -- is a claim to verify rather than assume, and it is the
    only place ``P = 3`` meets it.

    The decay arm is the harder composition and is why it is a row here: its summary
    ``A`` is not accumulated per TOKEN. It is multiplied once per tile by ``S0*S0``,
    deliberately in the same factors and the same order the state itself takes, so
    pass 2's fold sees a summary that cannot have drifted from its own ``B``. A fold
    that agreed with a token-serial walk but not with the panel would show up here and
    nowhere else."""
    fx = _Fixture((8, 8), 16, 64, B=1, T=128, H=2, p_read=0.5, p_write=0.5,
                  seed=9000 + P, decay_on=decay_on, nonzero_m0=True)
    y_ref, state_ref = fx.oracle()
    y, state, _ = fx.run(P)
    assert _relative(y, y_ref) < BF16_RTOL
    assert _relative(state, state_ref) < BF16_RTOL


# ---------------------------------------------------------------------------
# 2. P = 1 is the primary configuration and is BIT-IDENTICAL
# ---------------------------------------------------------------------------


def test_an_unsplit_plan_is_bit_identical_to_passing_no_plan_at_all():
    """`P = 1` is the primary benchmarked configuration everywhere (§6.4), so it
    must reach the kernel through exactly the arguments it reached it through before
    `P` existed -- a perturbed baseline is not a baseline.

    An unsplit plan carries no work list and no workspace, so the consumer takes the
    single-launch path with every sequence-parallel pointer null.

    **The comparison is on STATE, and ``y`` is deliberately excluded.** ``y`` is not
    bitwise reproducible at ``P = 1`` either: the §5 readout lands through
    ``atomicAdd`` into ``num``/``den`` from concurrently scheduled owners, so its
    summation order is a race by construction and two identical launches already
    disagree in the last bits. State is the output ``P`` actually moves from
    deterministic to re-associated -- it is per-owner and published exactly once --
    so it is the output the bitwise discipline can and does apply to. Asserting
    ``torch.equal`` on ``y`` here would be asserting a property the kernel has never
    had, and it would fail on the unmodified kernel.
    """
    fx = _Fixture((8, 8), 16, 64, B=1, T=128, H=2,
                  p_read=1.0, p_write=1.0, seed=4001)
    unsplit = fx.policy_plan(policy=SequenceParallelPolicy(workspace_cap=0))
    assert not unsplit.split, "the capacity gate at cap=0 must admit no split"
    _, s_none, _ = fx.run(1)
    _, s_plan, _ = fx.run(1, sequence_parallel=unsplit)
    assert torch.equal(s_none, s_plan)


def test_P_equals_one_launched_twice_is_bit_identical_on_state():
    """The NULL CONTROL for the envelope below, and a real invariant in its own
    right: the state is per-owner and published once, so it does not inherit the
    split-K epilogue's atomic non-determinism the way ``y`` does."""
    fx = _Fixture((8, 8), 16, 64, B=1, T=128, H=2,
                  p_read=1.0, p_write=1.0, seed=4002)
    _, s_a, _ = fx.run(1)
    _, s_b, _ = fx.run(1)
    assert torch.equal(s_a, s_b), (
        "P = 1 is not reproducible on STATE, which would invalidate the envelope "
        "below: a P > 1 difference could then be launch noise rather than the fold")


def test_P_equals_two_from_a_zero_initial_state_is_bit_identical_on_state():
    """A property of THIS implementation shape, stronger than §6.4 requires, so it
    is pinned rather than left to be rediscovered.

    Segment 0's summary is accumulated from zero, which at ``M_0 = 0`` is exactly the
    serial prefix; the scan's ``fmaf(A, 0, B)`` returns it unchanged; and the LAST
    segment runs serially from that boundary and publishes. At ``P = 2`` there is
    therefore no re-associated boundary at all. The property fails as soon as either
    premise does -- ``P >= 3`` (an interior boundary) or ``M_0 != 0`` -- and both of
    those are asserted here too, so this test cannot pass on a kernel whose fold is
    a no-op.
    """
    zero_m0 = _Fixture((8, 8), 16, 64, B=1, T=128, H=2,
                       p_read=1.0, p_write=1.0, seed=4003)
    _, s1, _ = zero_m0.run(1)
    _, s2, _ = zero_m0.run(2)
    assert torch.equal(s2, s1), "P = 2 from a zero initial state must be bit-identical"

    _, s4, _ = zero_m0.run(4)
    assert not torch.equal(s4, s1), (
        "P = 4 introduced no re-association, so the fold is not running and the "
        "bit-identity above is vacuous")

    carried = _Fixture((8, 8), 16, 64, B=1, T=128, H=2,
                       p_read=1.0, p_write=1.0, seed=4003, nonzero_m0=True)
    _, c1, _ = carried.run(1)
    _, c2, _ = carried.run(2)
    assert not torch.equal(c2, c1), (
        "P = 2 with a NONZERO initial state must re-associate (`A*M_0 + B` against "
        "the serial interleaving); bit-identity here would mean M_0 is being dropped")


# ---------------------------------------------------------------------------
# 3. The unwritten-leaf carve-out: BITWISE, and a real detector
# ---------------------------------------------------------------------------


def _partial_write_support_fixture(seed, *, live_first, live_second, boundary, T=128,
                                   widths=(8, 8)):
    """A fixture whose level-0 WRITE support is ``live_first`` digits before token
    ``boundary`` and ``live_second`` after it.

    ``boundary`` must be a SEGMENT boundary, not an arbitrary token: the carve-out is
    "unwritten in every segment after the first", so a support that changes in the
    middle of a segment leaves the digits it drops still written *within* that
    segment, and they are then correctly re-associated rather than carved out. That
    was a real defect in the first version of this fixture, and it is exactly the
    kind of thing the carve-out gate is supposed to be sharp enough to catch.
    """
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)
    B, H = 1, 1
    mask0 = torch.zeros(1, T, 1, widths[0], dtype=torch.float64)
    mask0[:, :boundary, :, :live_first] = 1.0
    mask0[:, boundary:, :, :live_second] = 1.0
    write_levels = (
        _masked_simplex((B, T, H, widths[0]), mask0, gen, device),
        _masked_simplex((B, T, H, widths[1]), torch.ones(1, 1, 1, widths[1]), gen, device))
    read_levels = tuple(
        _masked_simplex((B, T, H, w), torch.ones(1, 1, 1, w), gen, device) for w in widths)
    return read_levels, write_levels


def test_the_unwritten_leaf_carve_out_is_bitwise_and_the_written_leaves_are_not():
    """§6.4 discipline 2, NORMATIVE. This is the 48/48-against-0/16 exhibit,
    promoted from evidence to a committed invariant.

    A leaf with no write in a segment has ``A = 1`` and ``B = 0`` EXACTLY, so it is
    absent from the summary's written-leaf list and the scan never visits it: its
    carried value passes through untouched. That makes the carve-out bitwise by
    STRUCTURE (the leaf is never read, written or arithmetically touched) rather
    than by an identity like ``fmaf(1, x, 0) == x``, which is why it detects genuine
    fold bugs -- a fold that scattered the wrong leaf, mis-indexed the list, or
    dropped the count would corrupt exactly these rows.

    **The test asserts its own discriminating power first**: the fixture must
    contain written leaves that are NOT bit-identical. Without that, a kernel with
    no fold at all would pass.

    **Two review NOTEs, recorded here because they bound what a pass means.**

    * *The 48 "unwritten" leaves are two populations, not one* (review NOTE-3). 32
      of them (level-0 digits 4-7) are never written in ANY segment and are never
      visited by the scan -- the structural carve-out, separately gated by
      ``test_an_owner_with_no_write_anywhere_is_bit_identical_at_every_P``. The other
      16 (digits 2-3) ARE written in segment 0, so the scan does visit them at
      ``s = 0``, and they return bitwise because ``M_0 = 0`` makes ``fmaf(A, 0, B)``
      exactly ``B``. That mixture is not a defect -- it is what gives this fixture
      its detection power (the reviewer's mutation of segment 0's summary by a
      relative ``1e-7`` is caught THROUGH those 16 leaves) -- but the claim "bitwise
      by structure, not by an arithmetic identity" is exact only for the first 32.
    * *This gate observes the OUTCOME, not the structure* (review NOTE-4). "A leaf
      absent from the list is never read, written or arithmetically touched" is an
      argument about the source. A mutation that touches every unwritten leaf with
      an inert ``fmaf(1, x, 0)`` was measured to be caught by NOTHING in this suite.
      A refactor that keeps the arithmetic inert while losing the structure would
      therefore pass here; it must be caught in review of ``v3_scan_kernel``.
    """
    from rola.planner.schedule import TOKEN_WORD

    widths, BC, d_v, T, P = (8, 8), 16, 64, 128, 3
    # The write support must change at a SEGMENT boundary. Segment 0 of a `P`-way
    # split over `n_twords` covers words `[0, n_twords/P)`, so its last token is
    # `(n_twords // P) * TOKEN_WORD - 1`. Derived rather than hard-coded, so the
    # fixture cannot silently stop being a carve-out fixture if `T` or `P` move.
    n_tiles = -(-T // TOKEN_WORD)
    boundary = (n_tiles // P) * TOKEN_WORD
    assert 0 < boundary < T
    read_levels, write_levels = _partial_write_support_fixture(
        5001, live_first=4, live_second=2, boundary=boundary, T=T, widths=widths)
    fx = _Fixture(widths, BC, d_v, B=1, T=T, H=1, p_read=1.0, p_write=1.0,
                  seed=5001, read_levels=read_levels, write_levels=write_levels)
    assert fx.schedule.n_twords == n_tiles

    # `P = 3` puts an INTERIOR segment boundary in, which is what actually
    # re-associates (see the `P = 2` test above for why `P = 2` alone would not).
    _, serial, _ = fx.run(1)
    _, folded, _ = fx.run(P)

    N = fx.topology.N
    # A leaf is written after the first segment iff its level-0 digit is live there.
    leaf = torch.arange(N, device=serial.device)
    written_later = (leaf // widths[1]) < 2
    per_leaf_equal = (folded == serial).reshape(-1, N, fx.cols).all(dim=-1).reshape(-1)
    unwritten = ~written_later
    assert int(unwritten.sum()) > 0 and int(written_later.sum()) > 0, "malformed fixture"

    n_written_identical = int((per_leaf_equal & written_later).sum())
    assert n_written_identical == 0, (
        f"{n_written_identical} of {int(written_later.sum())} WRITTEN leaves came back "
        "bit-identical, so the fold is not re-associating anything and the carve-out "
        "assertion below would be vacuous")
    n_unwritten_identical = int((per_leaf_equal & unwritten).sum())
    assert n_unwritten_identical == int(unwritten.sum()), (
        f"only {n_unwritten_identical} of {int(unwritten.sum())} UNWRITTEN leaves are "
        "bit-identical; §6.4's carve-out is NORMATIVE and bitwise -- an unwritten leaf "
        "must not be visited by the fold at all")


def test_an_owner_with_no_write_anywhere_is_bit_identical_at_every_P():
    """The degenerate end of the same carve-out: a read-only owner has `(1, 0)` on
    every segment, so `P` cannot move its state by a single bit. Its state is also
    exactly its initial state, which is what makes this checkable at all."""
    widths, BC, d_v, T = (8, 8), 16, 64, 128
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(5002)
    # Level 0's write support is confined to digits {0,1}; owners over digits >= 2
    # are never written on any segment.
    mask0 = torch.zeros(1, 1, 1, widths[0], dtype=torch.float64)
    mask0[..., :2] = 1.0
    write_levels = (
        _masked_simplex((1, T, 1, widths[0]), mask0, gen, device),
        _masked_simplex((1, T, 1, widths[1]), torch.ones(1, 1, 1, widths[1]), gen, device))
    read_levels = tuple(
        _masked_simplex((1, T, 1, w), torch.ones(1, 1, 1, w), gen, device) for w in widths)
    fx = _Fixture(widths, BC, d_v, B=1, T=T, H=1, p_read=1.0, p_write=1.0,
                  seed=5002, read_levels=read_levels, write_levels=write_levels,
                  nonzero_m0=True)
    _, serial, _ = fx.run(1)
    leaf = torch.arange(fx.topology.N, device=serial.device)
    never_written = (leaf // widths[1]) >= 2
    assert int(never_written.sum()) > 0
    for P in (2, 3, 4):
        _, folded, _ = fx.run(P)
        rows = (folded == serial).reshape(-1, fx.topology.N, fx.cols).all(dim=-1).reshape(-1)
        assert bool(rows[never_written].all()), (
            f"a never-written leaf moved at P={P}; §6.4's `(1, 0)` is exact, not approximate")


# ---------------------------------------------------------------------------
# 4. The re-association envelope -- AN ENVELOPE, NOT A DETECTOR
# ---------------------------------------------------------------------------

#: **Read this before treating a failure here as a numerics bug, and before
#: treating a pass as evidence of correctness.**
#:
#: `P > 1` re-associates the state accumulation, so it differs from `P = 1` by
#: rounding. This bound is DERIVED, not fitted: the fold re-associates a sum of at
#: most `T` fp32 deposits per leaf, so the relative difference is bounded
#: by the usual `n * eps` sequential-summation bound with `eps = 2^-24`, times a
#: factor for the `A` product. At the fixtures below (`T <= 128`) that is
#: `128 * 2^-24 ~ 7.6e-6`; the constant is set an order above it.
#:
#: It is an ENVELOPE. It answers "did the difference stay the size rounding can
#: explain", and it CANNOT detect a fold bug whose error is small -- the carve-out
#: test above is the detector, because it is bitwise and structural. The null
#: control below is what keeps the envelope honest: a bound that a `P = 1`-vs-`P = 1`
#: comparison could also fail would be measuring launch noise, not the fold.
#:
#: **The slack is 215x-546x over the real signal, and that is stated rather than
#: implied** (review §4). Measured deltas across the twelve cells below are
#: 1.8e-7..4.7e-7 against this 1e-4 bound. For scale, §7.1's paging `y` gate -- the
#: discipline this one is said to mirror -- clears its derived bound by only ~2x on a
#: real defect, so the two are NOT comparable instruments. Confirmed by mutation: a
#: genuine fold bug injected at 1e-7 relative moves the delta from 2.0e-7 to
#: 2.5e-7/3.0e-7 and this bound does not notice, while the bitwise carve-out gate
#: catches it. A tighter derivation is available (sqrt(n) rather than worst-case n
#: growth, or the measured `A`-product factor, ~30x tighter and still derived); it is
#: not taken here because tightening an envelope does not make it a detector, and the
#: detector already exists.
REASSOCIATION_ENVELOPE = 1e-4


@pytest.mark.parametrize("P", [2, 3, 4])
@pytest.mark.parametrize("decay_on,nonzero_m0", [
    (False, False),
    (True, True),
    (False, True),
    (True, False),
])
def test_the_reassociation_envelope_holds_with_a_null_control(P, decay_on, nonzero_m0):
    fx = _Fixture((8, 8), 16, 64, B=1, T=128, H=2, p_read=1.0, p_write=1.0,
                  seed=6000 + P, decay_on=decay_on, nonzero_m0=nonzero_m0)
    _, serial, _ = fx.run(1)

    # NULL CONTROL. The same comparison, between two `P = 1` launches. It must be
    # EXACTLY zero -- not merely inside the envelope -- or the envelope below is
    # bounding launch noise rather than the fold's re-association.
    _, serial_again, _ = fx.run(1)
    null = _relative(serial_again, serial.double())
    assert null == 0.0, (
        f"the null control is {null:.3e}, not 0; `P = 1` is not reproducible on state, "
        "so nothing measured against it attributes to the fold")

    _, folded, _ = fx.run(P)
    delta = _relative(folded, serial.double())
    # The envelope must not be vacuous -- but "must differ" is only true where the
    # fold actually has a re-associated boundary. `P = 2` from a ZERO initial state
    # has none (segment 0's summary from zero IS the serial prefix, and the last
    # segment runs serially from it), and that is a pinned property of this
    # implementation, not an accident -- see
    # `test_P_equals_two_from_a_zero_initial_state_is_bit_identical_on_state`. So the
    # non-vacuity assertion applies exactly where the theory says it should.
    reassociates = P >= 3 or nonzero_m0
    if reassociates:
        assert delta > 0.0, (
            f"P={P} reproduced `P = 1` on state exactly, but this fixture HAS a "
            "re-associated boundary (P >= 3, or a nonzero initial state at P = 2), so "
            "the split is not running and the envelope is vacuous")
    else:
        assert delta == 0.0, (
            f"P={P} from a zero initial state must be bit-identical (no re-associated "
            f"boundary exists), but differs by {delta:.3e}")
    assert delta < REASSOCIATION_ENVELOPE, (
        f"P={P} state differs from `P = 1` by {delta:.3e}, outside the derived "
        f"re-association envelope {REASSOCIATION_ENVELOPE:.0e}. This is an ENVELOPE, not "
        "a detector: exceeding it means the difference is larger than rounding can "
        "explain, which is a real defect, but staying inside it is not proof of "
        "correctness -- that is what the bitwise carve-out gate is for")


# ---------------------------------------------------------------------------
# 5. Composition with §7.1 paging
# ---------------------------------------------------------------------------
#
# Fix pass R4. The scan RE-IMPLEMENTS §7.1's residency addressing (`v3_scan_kernel`
# resolves the owner's page exactly as the consumer does), and pass 3's entry-state
# stash now travels the same path -- so `P` over a paged arena is code on the
# increment's critical path that no committed test reached: neither sequence-parallel
# file mentioned `paging`, `page_table` or `admit` before this section.
#
# The discipline is the ruled one, unchanged by paging: fp64 ORACLE at the `P = 1`
# `BF16_RTOL`, the bitwise CARVE-OUT, and the ENVELOPE. Paging adds its own §7.1 gate on
# top (state bit-identical paged vs unpaged), which composes here as `torch.equal`
# at each fixed `P`: paging decides where a panel lives, never what it is.


def _paged_fixture(seed=7301, *, T=64, live_digits=4):
    """A sparse-write fixture whose arena lands PARTIALLY resident, by STRUCTURE.

    Partial residency is the cell the composition is actually about (a non-resident
    owner reads zeros and skips its store -- in the consumer AND, since `P`, in the
    scan's own page resolution). A Bernoulli write density does not reach it: the
    per-owner union over `T` tokens saturates, which is the same saturation
    ``paging-review.md`` R1 had to design around (measured here: 256 of 256 pages
    resident at ``p_write = 0.05``). So the write support is confined to the first
    ``live_digits`` digits of level 0, which leaves whole owners unwritten however
    long the sequence is.

    Reads are dense, so the SCHEDULE still carries every owner -- a split owner with
    no page at all is exactly the composition's interesting item, and it is in the
    grid rather than filtered out of it.
    """
    widths, BC, d_v, H = (16, 8, 8), 16, 32, 1
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)
    mask0 = torch.zeros(1, 1, 1, widths[0], dtype=torch.float64)
    mask0[..., :live_digits] = 1.0
    write_levels = (_masked_simplex((1, T, H, widths[0]), mask0, gen, device),) + tuple(
        _masked_simplex((1, T, H, w), torch.ones(1, 1, 1, w), gen, device)
        for w in widths[1:])
    read_levels = tuple(
        _masked_simplex((1, T, H, w), torch.ones(1, 1, 1, w), gen, device) for w in widths)
    return _Fixture(widths, BC, d_v, B=1, T=T, H=H, p_read=1.0, p_write=1.0,
                    seed=seed, routing=_SPARSE_WRITE_ROUTING,
                    read_levels=read_levels, write_levels=write_levels)


def _arena_for(fx, *, BH, max_physical_pages=None):
    return PageArena(widths=fx.topology.widths, BC=fx.plan.BC, cols=fx.cols, BH=BH,
                       device=torch.device("cuda"),
                       max_physical_pages=max_physical_pages)


def _run_paged(fx, arena, P=1):
    sp = None if P == 1 else fx.forced(P)
    return _consumer_forward(
        fx._f32["v"], fx._f32["read_levels"], fx._f32["write_levels"],
        fx._f32["g_read"], fx._f32["g_write"], fx.plan, schedule=fx.schedule,
        sequence_parallel=sp, decay=fx.decay, paging=arena)


@pytest.mark.parametrize("P", [2, 3])
def test_P_composes_with_paging_over_a_partially_resident_arena(P):
    """R4: the three-pass split, run against §7.1's page table rather than a dense
    state, at the same discipline and with paging's own bit-identity gate on top.

    **THE FIXTURE IS `T = 192`, NOT A SMALLER `T`, AND THAT IS THE `P >= 3`
    NON-VACUITY CONTROL'S OWN REQUIREMENT** (`rewrite.md` §2.31). A re-associated
    boundary exists only where some SUMMARIZED segment carries two or more
    words: pass 1 summarizes segments `0 .. P - 2` from ZERO, the boundaries
    are those summaries accumulated in order, and a one-word summary is a
    single deposit -- which sums into the boundary in exactly the serial
    order. P26: the scan word is now the fixed 32-token `TOKEN_WORD` (no more
    `BT=16`), so `n_twords = ceil(T/32)`; the `P = 3` cell needs at least 6
    words for its summarized segment to span two or more (`T=192` gives
    exactly 6). A smaller `T` -- the old `T=128` gave only 4 words under the
    32-token granule -- splits `1 / 1 / 2`, whose only two-word segment is the
    LAST one, which pass 1 does not summarize: the split would be
    bit-identical to `P = 1` BY CONSTRUCTION and the control below would fire
    on a fixture that never had teeth.
    """
    fx = _paged_fixture(T=192)
    #: The structural precondition, asserted rather than assumed -- a later `T` edit
    #: must fail here rather than silently re-vacate the control below.
    assert fx.schedule.n_twords >= 6, (
        f"{fx.schedule.n_twords} tiles cannot give `P = 3` a summarized segment of two "
        "or more tiles, so the non-vacuity assertion below would have no teeth")
    B, H = 1, 1
    arena = _arena_for(fx, BH=B * H)
    y_p1, _, _ = _run_paged(fx, arena, 1)
    state_p1 = arena.materialize().clone()
    result = arena.last_result
    assert 0 < result.resident_pages < arena.logical_pages, (
        "this fixture must be PARTIALLY resident or it is not testing the composition "
        "R4 asks for; a saturated arena exercises only the dense addressing")
    page_table = arena.page_table.clone()

    # The null control for the envelope, on the PAGED path: the state is per-owner and
    # published once, so a second identical paged launch must reproduce it exactly.
    control = _arena_for(fx, BH=B * H)
    _run_paged(fx, control, 1)
    assert torch.equal(control.materialize(), state_p1)

    split_arena = _arena_for(fx, BH=B * H)
    y_pk, _, _ = _run_paged(fx, split_arena, P)
    state_pk = split_arena.materialize()
    assert torch.equal(split_arena.page_table, page_table), (
        "the split changed WHICH owners got pages; `P` is a work-list axis and must "
        "not touch §7.1's commitment, which is keyed on the write-live owner set")

    # 1. The fp64 oracle, at the `P = 1` tolerance -- unchanged by paging.
    y_ref, state_ref = fx.oracle()
    N, cols = fx.topology.N, fx.cols
    assert _relative(y_pk, y_ref) < BF16_RTOL
    assert _relative(state_pk.reshape(B, H, N, cols), state_ref) < BF16_RTOL

    # 2. The bitwise carve-out. A never-written owner has no page at all, so its rows
    # read as zeros at every `P` -- the structural end of §6.4's carve-out, composed
    # with §7.1's "a non-resident owner reads zeros and skips its store".
    absent = (page_table.reshape(-1) < 0)
    assert int(absent.sum()) > 0
    rows_pk = state_pk.reshape(-1, N // fx.plan.BC, fx.plan.BC, cols)
    assert bool((rows_pk.reshape(-1, fx.plan.BC * cols)[absent] == 0).all())
    # 3. The envelope (an ENVELOPE, not a detector -- see its definition above), with
    # the non-vacuity assertion conditioned exactly as the unpaged envelope test's is.
    # §7.1 forbids an implicit dense `initial_state` (the arena is the carrier of
    # prior state), so `M_0 = 0` here ALWAYS -- and at `M_0 = 0` the `P = 2` split has
    # no re-associated boundary at all and is bit-identical, which is this
    # implementation's pinned property, not an accident
    # (`test_P_equals_two_from_a_zero_initial_state_is_bit_identical_on_state`).
    delta = _relative(state_pk, state_p1.double())
    per_leaf_equal = (state_pk == state_p1).reshape(-1, N, cols).all(dim=-1)
    if P >= 3:
        assert not bool(per_leaf_equal.all()), (
            "the paged split reproduced `P = 1` on every leaf, so the fold is not "
            "running and the carve-out assertion above is vacuous")
        assert 0.0 < delta < REASSOCIATION_ENVELOPE
    else:
        assert delta == 0.0, (
            f"P = 2 from a zero initial state must be bit-identical, but differs by "
            f"{delta:.3e}; paging cannot be what re-associates it")

    # §7.1's own gate, which the composition must not weaken: the same routing at the
    # same `P` is BIT-IDENTICAL paged and unpaged. Paging decides where a value panel
    # lives, never what it is.
    _, state_dense, _ = fx.run(P)
    assert torch.equal(state_dense.reshape(B * H, N, cols), state_pk)


def _run_paged_plan(fx, arena, sp):
    return _consumer_forward(
        fx._f32["v"], fx._f32["read_levels"], fx._f32["write_levels"],
        fx._f32["g_read"], fx._f32["g_write"], fx.plan, schedule=fx.schedule,
        sequence_parallel=sp, decay=fx.decay, paging=arena)


# ---------------------------------------------------------------------------
# 6. The workspace
# ---------------------------------------------------------------------------


def test_a_dirty_workspace_changes_nothing():
    """The workspace is allocated with ``torch.empty``, so "every byte is written
    before it is read" is a claim the kernel makes and this test checks.

    Poisoning it with NaN is the right poison: a NaN that reaches the fold
    propagates into the state and cannot be cancelled by a later multiply, so any
    read of an unwritten slot byte shows up as a NaN in the output rather than as a
    small perturbation.
    """
    fx = _Fixture((8, 8), 16, 64, B=1, T=128, H=2,
                  p_read=0.5, p_write=0.5, seed=7001, nonzero_m0=True)
    sp = fx.forced(3)

    def _launch(tensors):
        return _consumer_forward(
            fx._f32["v"], fx._f32["read_levels"], fx._f32["write_levels"],
            fx._f32["g_read"], fx._f32["g_write"], fx.plan, schedule=fx.schedule,
            sequence_parallel=sp, sequence_parallel_tensors=tensors, decay=fx.decay,
            initial_state=fx.initial_state.to(torch.float32))

    clean = build_sequence_parallel_tensors(sp, fx.schedule)
    dirty = build_sequence_parallel_tensors(sp, fx.schedule)
    dirty.ws_state.fill_(float("nan"))
    dirty.ws_scale.fill_(float("nan"))
    dirty.ws_leaf.fill_(0)      # a VALID leaf index, so a stale read is silent, not a fault
    dirty.ws_count.fill_(sp.BC)  # the maximum, so a stale count over-reads rather than under-

    _, state_clean, _ = _launch(clean)
    _, state_dirty, _ = _launch(dirty)
    assert not bool(torch.isnan(state_dirty).any()), (
        "a NaN from the poisoned workspace reached the state: some byte of it is read "
        "before it is written, and `torch.empty` therefore makes the kernel's output "
        "depend on whatever the caching allocator last left there")
    assert torch.equal(state_dirty, state_clean), (
        "the poisoned workspace changed the answer, so the workspace is not fully "
        "written before it is read")


def test_the_sparse_summary_is_exercised_between_its_two_degeneracies():
    """Fix pass R3: a fixture that reaches ``0 < K < BC``, which nothing else does.

    ``K = ws_count[slot]`` is the length of §6.4's NORMATIVE sparse written-leaf
    summary, and it is the trip count of both the ballot compaction that emits it
    (`v3_consumer.cu`) and the scan's ``K * cols`` fold. Over every other committed
    fixture ``K`` takes only its two DEGENERATE values: the reviewer measured
    ``K == BC == 16`` on 24 of 24 slots even at ``p_write = 0.15``, because the union
    of writes over a whole SEGMENT saturates the owner rectangle however sparse any
    single token is, and ``K = 0`` on a segment with no write candidate at all. A
    Bernoulli density cannot reach the middle; a STRUCTURED support can.

    Here level 1's write support is confined to 3 of its 8 digits for every token.
    An owner spans ``c = 2`` digits of level 0 and all 8 of level 1, so its written
    leaves are exactly ``2 * 3 = 6`` of its ``BC = 16`` -- a partial compaction whose
    ascending leaf list is not the identity, which is the only regime where a
    mis-indexed scatter or a dropped count is observable at all.
    """
    widths, BC, d_v, T, P = (8, 8), 16, 64, 128, 3
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(5101)
    live = 3
    mask1 = torch.zeros(1, 1, 1, widths[1], dtype=torch.float64)
    mask1[..., :live] = 1.0
    write_levels = (
        _masked_simplex((1, T, 1, widths[0]), torch.ones(1, 1, 1, widths[0]), gen, device),
        _masked_simplex((1, T, 1, widths[1]), mask1, gen, device))
    read_levels = tuple(
        _masked_simplex((1, T, 1, w), torch.ones(1, 1, 1, w), gen, device) for w in widths)
    fx = _Fixture(widths, BC, d_v, B=1, T=T, H=1, p_read=1.0, p_write=1.0,
                  seed=5101, read_levels=read_levels, write_levels=write_levels,
                  nonzero_m0=True)
    sp = fx.forced(P)
    t = build_sequence_parallel_tensors(sp, fx.schedule)
    # `-1` is not a reachable count, so it distinguishes "pass 1 wrote this" from
    # "nothing did" -- which is also how the stash slots are identified below.
    t.ws_count.fill_(-1)

    y, state, _ = _consumer_forward(
        fx._f32["v"], fx._f32["read_levels"], fx._f32["write_levels"],
        fx._f32["g_read"], fx._f32["g_write"], fx.plan, schedule=fx.schedule,
        sequence_parallel=sp, sequence_parallel_tensors=t, decay=fx.decay,
        initial_state=fx.initial_state.to(torch.float32))

    counts = t.ws_count.tolist()
    offsets, n_owner_slots = sp.slot_offsets(), owner_slot_counts(sp.P)
    stash = {o for o, n in zip(offsets, n_owner_slots) if n}
    summary_counts = [k for s, k in enumerate(counts) if s not in stash]
    stash_counts = [k for s, k in enumerate(counts) if s in stash]
    # The entry-state stash carries no sparse summary, so pass 1 must not have
    # touched its metadata -- and the scan must not have read it.
    assert set(stash_counts) == {-1}, (
        "a stash slot's `ws_count` was written; the summary slots are `slot0 + 1 + s` "
        "and the stash is `slot0`, so this is an off-by-one in the slot layout")
    assert summary_counts, "the fixture produced no summary slots at all"
    assert all(k == 2 * live for k in summary_counts), (
        f"expected K = {2 * live} on every summary slot (c=2 level-0 digits x {live} "
        f"live level-1 digits), got {sorted(set(summary_counts))}")
    # The point of the fixture, asserted as a property rather than as a number:
    # strictly between the two degeneracies the rest of the suite lives at.
    assert all(0 < k < sp.BC for k in summary_counts)

    y_ref, state_ref = fx.oracle()
    assert _relative(y, y_ref) < BF16_RTOL
    assert _relative(state, state_ref) < BF16_RTOL


def test_the_pass_three_work_list_order_is_not_load_bearing():
    """Fix pass R1: pass 3's reads must be disjoint from pass 3's writes BY
    CONSTRUCTION, not by the order the work list happens to be in.

    A split owner's LAST segment writes the owner's ``p.state`` rows; before the fix
    its segment 0 READ those same rows, from a different CTA of the SAME launch.
    CUDA orders nothing between CTAs of one launch, so correctness rested on the
    builder emitting an owner's segments adjacently and ascending -- a property
    nothing in the kernel, the parameter block or the builder stated as load-bearing,
    in a codebase that already cost-sorts owner lists elsewhere. The reviewer
    demonstrated it: permuting the pass-3 work list so that every owner's LAST
    segment precedes its segment 0 produced ``y`` with relative error **1.0**,
    silently, with the state untouched (so no state-based gate could see it).

    Since the fix, segment 0 reads the entry-state stash the SCAN published in pass
    2 -- an earlier launch -- so this permutation is arithmetically inert. This test
    is the permutation attack, committed: it must pass now and it FAILS on the
    pre-fix kernel, which is what makes it a mutation gate rather than a decoration.

    The permutation is applied to the pass-3 lists only; passes 1 and 2 are
    untouched, so what is under test is exactly pass 3's read/write disjointness.
    """
    # **The grid must exceed the machine's resident CTA capacity, and that is the
    # fixture's whole design.** The hazard is a race between a CTA that writes
    # `p.state` and a sibling that reads it, so it can only MANIFEST when some CTAs
    # start after others have finished -- i.e. in a second wave. Measured on the
    # pre-fix kernel: 64 owners x `P = 8` = 512 CTAs, below this machine's 80 x 8 =
    # 640 resident slots, and the permutation is harmless because every CTA loads its
    # entry state before any CTA stores; 128 owners x `P = 8` = 1024 CTAs (the
    # reviewer's probe shape, ~1.6x capacity) reproduces the corruption. A carried
    # `M_0` is equally required: at `M_0 = 0` a clobbered entry-state read is
    # indistinguishable from a correct one, so the attack would be inert for the
    # wrong reason.
    #
    # P26: `T=256`, not the old `128`. `TOKEN_WORD=32` (no more `BT=16`) means
    # `n_twords = ceil(T/32)`, and `forced(8)` needs at least 8 words to split
    # 8 ways (`128` gave only 4).
    fx = _Fixture((8, 16), 16, 64, B=2, T=256, H=8, p_read=1.0, p_write=1.0,
                  seed=8101, nonzero_m0=True)
    assert fx.n_owners == 128, "the attack needs more CTAs than the GPU can hold at once"
    sp = fx.forced(8)
    canonical = build_sequence_parallel_tensors(sp, fx.schedule)

    # "Last segments first": stable-sort the pass-3 work list by DESCENDING segment
    # index. Every owner's last segment then precedes its segment 0, which is the
    # ordering the pre-fix kernel got wrong.
    order = torch.argsort(-canonical.work_seg.to(torch.int64), stable=True)
    permuted = replace(
        canonical,
        owners=canonical.owners[order].contiguous(),
        work_seg=canonical.work_seg[order].contiguous(),
        work_nseg=canonical.work_nseg[order].contiguous(),
        work_slot=canonical.work_slot[order].contiguous())

    # Discriminating power, asserted on the fixture rather than assumed: the
    # permutation must actually place a last segment before its own segment 0.
    first_seg_rows = (permuted.work_seg == 0).nonzero().flatten()
    last_seg_rows = (permuted.work_seg == permuted.work_nseg - 1).nonzero().flatten()
    assert int(last_seg_rows.max()) < int(first_seg_rows.min()), (
        "the permutation did not put every last segment ahead of every segment 0, so "
        "it is not the ordering that broke the pre-fix kernel")

    def _launch(tensors):
        return _consumer_forward(
            fx._f32["v"], fx._f32["read_levels"], fx._f32["write_levels"],
            fx._f32["g_read"], fx._f32["g_write"], fx.plan, schedule=fx.schedule,
            sequence_parallel=sp, sequence_parallel_tensors=tensors, decay=fx.decay,
            initial_state=fx.initial_state.to(torch.float32))

    y_ref, state_ref = fx.oracle()
    y_c, state_c, _ = _launch(canonical)
    y_p, state_p, _ = _launch(permuted)

    # `y` is the output the hazard corrupted (to relative error 1.0), and it is the
    # gate that matters here. It is graded against the fp64 oracle rather than
    # bitwise against the canonical order, because the §5 readout's `atomicAdd`
    # ordering is a race at every `P` including 1 -- a permuted work list legitimately
    # changes the last bits of `y` and nothing else.
    assert _relative(y_p, y_ref) < BF16_RTOL, (
        "the permuted pass-3 work list changed `y`; some pass-3 CTA is reading state "
        "another CTA of the same launch writes")
    assert _relative(y_c, y_ref) < BF16_RTOL
    # State is per-owner and published once, so the permutation must not move a bit
    # of it: each owner's segments perform the identical arithmetic in the identical
    # order whatever position they occupy in the list.
    assert torch.equal(state_p, state_c), (
        "the permuted work list moved the STATE, so an owner's arithmetic depends on "
        "where its segments landed in the list")


def test_the_three_pass_work_lists_have_the_shape_the_contract_names():
    """§6.4's three passes, as grids. The relationship between them is load-bearing:
    pass 1 omits every owner's LAST segment because that segment's outcome is the
    final state, which pass 3 publishes directly -- so an owner emits ``P - 1``
    summaries while pass 3 runs ``P`` segments.

    The workspace is ``P`` slots per split owner, not ``P - 1``: the ``P - 1``
    segment boundaries plus the ENTRY-STATE STASH pass 2 publishes for segment 0
    (fix pass R1; spec §6.4's ``(P-1) * BC * d_v`` bound was amended to
    ``P * BC * d_v`` in the same pass). An unsplit owner still costs zero slots.
    """
    fx = _Fixture((8, 8), 16, 64, B=1, T=128, H=2,
                  p_read=1.0, p_write=1.0, seed=8001)
    sp = fx.forced(3)
    t = build_sequence_parallel_tensors(sp, fx.schedule)
    n = fx.n_owners
    assert t.owners.shape[0] == sum(sp.P) == 3 * n
    assert t.summary_owners.shape[0] == sum(p - 1 for p in sp.P) == 2 * n
    assert t.scan_owners.shape[0] == sum(1 for p in sp.P if p > 1) == n
    assert sp.n_slots == t.ws_state.shape[0] == 3 * n
    assert tuple(t.ws_state.shape) == (3 * n, sp.BC, sp.cols)
    # Segment `p` of owner `o` reads slot `slot_offsets()[o] + p`, so the owner's
    # slot runs must tile `[0, n_slots)` without overlapping: two owners sharing a
    # slot would be two entry states in one place.
    offsets = sp.slot_offsets()
    counts = owner_slot_counts(sp.P)
    assert offsets == tuple(sum(counts[:i]) for i in range(n))
    assert offsets[-1] + counts[-1] == sp.n_slots
    # Segments are ascending within an owner and owners keep `schedule.owners`'
    # order (head-major: `bh` ascending, cost descending within a `bh`), so the
    # pass-3 grid's first `n` rows are that owner list with segment 0.
    assert torch.equal(t.owners[t.work_seg == 0], fx.schedule.owners.to(t.owners.dtype))
    assert torch.equal(t.work_seg.reshape(n, 3),
                       torch.arange(3, device=t.work_seg.device, dtype=t.work_seg.dtype)
                       .expand(n, 3))
