"""§6.4's `P` policy: targeted selection, the three admission gates, the ledger.

``ROLA_V3_SPEC.md`` §6.4's policy block (NORMATIVE, ratified 2026-07-29).

These tests are about the PLANNER, so they are CPU tests wherever they can be:
``sms`` and ``blocks_per_sm`` are passed explicitly rather than read off whatever
GPU the suite happens to run on, which is what makes a constructed
occupancy-starved / bandwidth-heavy / capacity-tight scenario the same scenario
everywhere. The two that genuinely need a device -- the driver-answered occupancy
query and the ``None`` cap's resolution against total device memory -- say so and
carry the ``cuda`` marker.

**Each of the three gates has a test that forces it to be the limiting one**, per
§6.4's "record the limiting gate in the plan whenever `P_o` is clamped". A gate that
is never observed to bind is a gate whose code has never run.
"""

from __future__ import annotations

import pytest
import torch

from rola.planner.schedule import Plan, build_schedule
from rola.routing.sequence_parallel import (
    GATES,
    MEASURED_HBM_BYTES_PER_S,
    MEASURED_STATE_ROUNDTRIP_BYTES_PER_S,
    SequenceParallelPolicy,
    build_sequence_parallel_tensors,
    consumer_blocks_per_sm,
    select_sequence_parallelism,
    slot_bytes,
    state_roundtrip_bytes_per_s,
)
from rola.routing.types import (
    IndependentRouting,
    LevelConfig,
    SoftmaxActivation,
    Topology,
)

_ROUTING = IndependentRouting(
    tied=False, read_activation=SoftmaxActivation(), write_activation=SoftmaxActivation())

#: A cap far above anything these fixtures need, so the capacity gate is out of the
#: way except in the test that is about it. Passed explicitly rather than left to
#: resolve, so no CPU test touches the device.
_WIDE_CAP = 1 << 34


def _fixture(widths=(8, 8), BC=16, d_v=64, B=1, T=128, H=2, BT=16, norm="raw",
             device="cpu", seed=11):
    gen = torch.Generator(device=device).manual_seed(seed)
    topology = Topology(levels=tuple(LevelConfig(w, _ROUTING) for w in widths), d_v=d_v)
    read_levels = tuple(
        torch.rand(B, T, H, w, generator=gen, device=device).softmax(-1) for w in widths)
    write_levels = tuple(
        torch.rand(B, T, H, w, generator=gen, device=device).softmax(-1) for w in widths)
    g_write = torch.rand(B, T, H, generator=gen, device=device) + 0.5
    g_read = (None if norm == "global"
              else torch.rand(B, T, H, generator=gen, device=device) + 0.5)
    plan = Plan(topology=topology, normalization=norm, BT=BT, BC=BC,
                  warp_tile=(1, max(1, d_v // 32)), decay=False,
                  mma_precision="tf32x3", compact=False)
    schedule = build_schedule(read_levels, write_levels, g_read, g_write, plan)
    return plan, schedule


# ---------------------------------------------------------------------------
# Targeted selection
# ---------------------------------------------------------------------------


def test_the_default_is_P_equals_one_and_carries_no_workspace_at_all():
    """§6.4: "`P = 1` is the default and the PRIMARY benchmarked configuration
    everywhere". An unsplit plan must therefore cost nothing -- not a small
    workspace, not an empty work list: nothing."""
    plan, schedule = _fixture()
    sp = select_sequence_parallelism(
        plan, schedule, policy=SequenceParallelPolicy(workspace_cap=0),
        sms=80, blocks_per_sm=8)
    assert not sp.split
    assert sp.n_slots == 0
    assert sp.ledger.workspace_bytes == 0
    assert sp.roundtrip_bytes == 0
    assert set(sp.P) == {1}
    with pytest.raises(ValueError, match="unsplit plan"):
        build_sequence_parallel_tensors(sp, schedule)


def test_P_is_apportioned_in_proportion_to_the_owner_live_tile_count():
    """§6.4: "assigns `P_o` per owner PROPORTIONAL to that owner's live-tile count".

    `schedule.owner_costs` IS that count -- §4's builder forms it as read-candidate
    tiles plus `write_cost_weight` times write-candidate tiles over the realized
    schedule -- so the apportionment must be monotone non-increasing IN COST.
    Monotonicity is the checkable form of "proportional" that does not depend on
    the exact rounding rule.

    **The claim is stated against cost, not against row index, and that is the
    point.** `schedule.owners` is head-major (`bh` ascending, cost descending
    within a `bh`), so cost is NOT globally descending down the owner list any
    more; a test that read monotonicity off the row order would have been reading
    the launch's L2-locality key, which the apportionment has nothing to do with.
    Re-sorting by cost here restores the property the spec actually asserts. The
    head-major structure is a separate claim with its own gate,
    `test_planner.py::test_the_owner_order_is_head_major` (which lives there
    because it needs a fixture whose owners have DIFFERENT costs -- this file's
    dense-softmax fixture gives every owner the same cost, so it could not tell
    the two keys apart).
    """
    plan, schedule = _fixture(widths=(8, 8), T=512)
    sp = select_sequence_parallelism(
        plan, schedule, policy=SequenceParallelPolicy(workspace_cap=_WIDE_CAP),
        sms=80, blocks_per_sm=8)
    costs = schedule.owner_costs.tolist()
    by_cost = sorted(range(len(costs)), key=lambda o: -costs[o])
    P_by_cost = [sp.P[o] for o in by_cost]
    assert P_by_cost == sorted(P_by_cost, reverse=True), (
        f"P is not monotone in the live-tile count: costs={costs} P={sp.P}")


def test_over_splitting_is_refused_beyond_the_wave_target():
    """§6.4: "raises total segments only to the SMALLEST count clearing the waves/SM
    target (~1-2). Over-splitting pays fold overhead for nothing."

    So the admitted total must not exceed the target, and lowering the target must
    not raise the split.
    """
    plan, schedule = _fixture(widths=(8, 8), T=512)
    wide = SequenceParallelPolicy(workspace_cap=_WIDE_CAP, waves_target=1.0)
    narrow = SequenceParallelPolicy(workspace_cap=_WIDE_CAP, waves_target=0.25)
    big = select_sequence_parallelism(plan, schedule, policy=wide, sms=8, blocks_per_sm=1)
    small = select_sequence_parallelism(plan, schedule, policy=narrow, sms=8, blocks_per_sm=1)
    assert big.total_segments <= int(1.0 * 8 * 1) or big.binding_gate is not None
    assert small.total_segments <= big.total_segments


def test_an_owner_is_never_split_past_its_own_tile_count_or_past_max_p():
    """Both ceilings are STRUCTURAL bounds -- a tile count and a policy constant --
    not thresholds on a measured density, so they are admissible under §3's branch
    rule and belong in the planner rather than anywhere near the kernel."""
    plan, schedule = _fixture(widths=(8, 8), T=64)      # n_tiles = 4 at BT = 16
    assert schedule.n_tiles == 4
    sp = select_sequence_parallelism(
        plan, schedule, policy=SequenceParallelPolicy(workspace_cap=_WIDE_CAP, max_p=64),
        sms=1024, blocks_per_sm=8)
    assert max(sp.P) <= schedule.n_tiles
    capped = select_sequence_parallelism(
        plan, schedule, policy=SequenceParallelPolicy(workspace_cap=_WIDE_CAP, max_p=2),
        sms=1024, blocks_per_sm=8)
    assert max(capped.P) <= 2


def test_selection_is_deterministic():
    """The apportionment is largest-remainder with ties broken by descending weight
    and then ascending index, so it is a pure function of the inputs and does not
    depend on sort stability or on float comparison order."""
    plan, schedule = _fixture(T=256)
    policy = SequenceParallelPolicy(workspace_cap=_WIDE_CAP)
    runs = [select_sequence_parallelism(plan, schedule, policy=policy, sms=64, blocks_per_sm=4)
            for _ in range(4)]
    assert all(r.P == runs[0].P for r in runs)
    assert all(r.limiting_gate == runs[0].limiting_gate for r in runs)


# ---------------------------------------------------------------------------
# The three admission gates: each forced to be the limiting one
# ---------------------------------------------------------------------------


def test_the_occupancy_gate_is_the_limiting_one_when_the_grid_already_saturates():
    """Gate 1. §6.4: `P` is an occupancy dial for the LOW-OWNER corner and "at design
    scale the liveness-filtered grid saturates naturally". A saturated grid must
    therefore close the gate outright -- `P = 1` everywhere, and the plan says why.

    Forced by shrinking the machine until the existing owner count alone clears the
    wave target, which is the same condition a large-`N` topology reaches by having
    more owners.
    """
    plan, schedule = _fixture()
    n_owners = int(schedule.owners.shape[0])
    assert n_owners > 0
    sp = select_sequence_parallelism(
        plan, schedule, policy=SequenceParallelPolicy(workspace_cap=_WIDE_CAP),
        sms=1, blocks_per_sm=1)
    assert sp.waves_before >= 1.0, "the fixture did not actually saturate the grid"
    assert sp.binding_gate == "occupancy"
    assert set(sp.limiting_gate) == {"occupancy"}
    assert set(sp.P) == {1}
    assert not sp.split

    # The control: the SAME fixture on a machine it does not saturate must split, or
    # the assertion above is passing for the wrong reason.
    starved = select_sequence_parallelism(
        plan, schedule, policy=SequenceParallelPolicy(workspace_cap=_WIDE_CAP),
        sms=80, blocks_per_sm=8)
    assert starved.waves_before < 1.0
    assert starved.split, "the occupancy test's control does not split, so it proves nothing"


def test_the_bandwidth_gate_is_the_limiting_one_on_a_traffic_heavy_launch():
    """Gate 2. §6.4: the split's roundtrip bytes must stay under the achievable
    bandwidth with margin, because "on a saturated bus, extra roundtrips worsen
    wall-clock while IMPROVING occupancy" -- the one failure mode where the occupancy
    gate on its own would make things worse.

    Forced by driving the margin to zero: no added roundtrip traffic is affordable,
    so the gate admits nothing however starved the grid is. The control is the same
    plan at a generous margin, which must split.
    """
    plan, schedule = _fixture(T=256)
    tight = select_sequence_parallelism(
        plan, schedule,
        policy=SequenceParallelPolicy(workspace_cap=_WIDE_CAP, bandwidth_margin=0.0),
        sms=80, blocks_per_sm=8)
    assert tight.waves_before < 1.0, "the fixture must be occupancy-starved for this to bind"
    assert tight.binding_gate == "bandwidth"
    assert set(tight.P) == {1}
    assert "bandwidth" in tight.limiting_gate

    loose = select_sequence_parallelism(
        plan, schedule,
        policy=SequenceParallelPolicy(workspace_cap=_WIDE_CAP, bandwidth_margin=1e6),
        sms=80, blocks_per_sm=8)
    assert loose.split, "the bandwidth control does not split, so the tight case proves nothing"
    assert loose.binding_gate != "bandwidth"


def test_the_bandwidth_gate_clamps_PARTIALLY_through_its_measured_relative_form():
    """Fix pass R2: the gate forced through the arithmetic, not around it.

    The test above forces the gate with `bandwidth_margin = 0.0`, which zeroes the
    admitted slot count BEFORE `base_traffic_seconds`, `MEASURED_HBM_BYTES_PER_S` or
    `state_roundtrip_bytes_per_s()` is consulted -- an implementation that simply
    returned `0` would pass it identically, and no test observed a PARTIAL clamp.
    This one is the L = 256 cell the flagship benchmark actually binds at, lifted
    into the suite: `B=1 H=8 L=256 d_v=64 branches=(8,16) BC=16`, 64 owners, at the
    DEFAULT margin. Measured (reviewer's reproduction and this test's own
    assertions): `binding_gate = 'bandwidth'` with `P_max = 3` -- the relative form
    admitting a nonzero, sub-requested number of slots.

    Three things are asserted, and each kills a different degenerate implementation:
    a nonzero admission (a `return 0` stub), equality with the measured-constants
    formula (any other pricing), and strict monotonicity in the margin (a constant).
    """
    from math import floor

    plan, schedule = _fixture(widths=(8, 16), BC=16, d_v=64, B=1, T=256, H=8, BT=16)
    policy = SequenceParallelPolicy(workspace_cap=_WIDE_CAP)   # the DEFAULT margin
    sp = select_sequence_parallelism(plan, schedule, policy=policy, sms=80, blocks_per_sm=8)

    assert sp.waves_before < 1.0, "the cell must be occupancy-starved for gate 2 to bind"
    assert sp.binding_gate == "bandwidth"
    assert sp.split, "a bandwidth clamp that admits nothing is the degenerate case R2 is about"
    assert max(sp.P) == 3, f"expected the measured partial clamp to P_max = 3, got {max(sp.P)}"

    # The admitted extra-segment count IS the relative form's arithmetic, evaluated
    # from the module's measured constants. `base_traffic_seconds` prices the traffic
    # the launch already moves against the measured HBM figure; the split's slots are
    # priced at the `bench_state_roundtrip` point matching their per-CTA footprint.
    per_slot = sp.BC * sp.cols * 4 * 2
    admitted = floor(policy.bandwidth_margin * sp.base_traffic_seconds
                     * state_roundtrip_bytes_per_s(sp.BC * sp.cols * 4) / per_slot)
    assert admitted > 0
    assert sum(p - 1 for p in sp.P) == admitted, (
        "the clamp does not equal the relative form evaluated at the measured "
        "constants, so the gate is not the gate its docstring describes")
    assert sp.base_traffic_seconds == pytest.approx(
        sp.base_traffic_bytes / MEASURED_HBM_BYTES_PER_S)

    # Monotone in the margin, and still the binding gate at twice it: a clamp that did
    # not move with the budget would be a constant wearing the gate's name.
    wider = select_sequence_parallelism(
        plan, schedule,
        policy=SequenceParallelPolicy(workspace_cap=_WIDE_CAP, bandwidth_margin=0.20),
        sms=80, blocks_per_sm=8)
    assert wider.binding_gate == "bandwidth"
    assert sum(p - 1 for p in wider.P) > sum(p - 1 for p in sp.P)


def test_the_capacity_gate_is_the_limiting_one_under_a_tight_cap():
    """Gate 3. §6.4: "Workspace fits the resolved cap. Degrade `P_o` to fit."

    Forced by resolving the cap to exactly the non-degradable terms plus room for
    two slots: the plan must come back with exactly two slots and record `capacity`.
    A cap expressed in slots rather than a magic byte count is what makes the
    assertion exact rather than approximate.
    """
    plan, schedule = _fixture(T=256)
    cols = plan.topology.d_v
    per_slot = slot_bytes(plan.BC, cols)
    materializations = schedule.schedule_bytes() + schedule.stream_bytes()
    cap = materializations + 2 * per_slot
    sp = select_sequence_parallelism(
        plan, schedule,
        policy=SequenceParallelPolicy(workspace_cap=cap, bandwidth_margin=1e6),
        sms=80, blocks_per_sm=8)
    assert sp.n_slots == 2, f"expected exactly two slots to fit the cap, got {sp.n_slots}"
    assert sp.binding_gate == "capacity"
    assert "capacity" in sp.limiting_gate
    assert sp.ledger.fits
    assert sp.ledger.projected_peak_bytes <= cap


def test_state_pages_are_never_the_degradable_item():
    """§6.4 gate 3, verbatim: "Degrade `P_o` to fit; state pages are NEVER the
    degradable item."

    A cap smaller than the arena's committed pages alone must collapse `P` to 1 and
    report `capacity` -- it must NOT reduce the arena term, and the ledger must
    report the overflow honestly rather than hiding it by shrinking the wrong line.
    """
    plan, schedule = _fixture(T=256)
    arena = 64 << 20
    sp = select_sequence_parallelism(
        plan, schedule,
        policy=SequenceParallelPolicy(workspace_cap=1 << 20, bandwidth_margin=1e6),
        arena_page_bytes=arena, sms=80, blocks_per_sm=8)
    assert set(sp.P) == {1}
    assert sp.binding_gate == "capacity"
    assert sp.ledger.arena_page_bytes == arena, "the arena term was degraded"
    assert sp.ledger.workspace_bytes == 0
    assert sp.ledger.workspace_headroom_bytes == 0
    assert not sp.ledger.fits, (
        "the ledger reports a fit while the arena alone exceeds the cap; the overflow "
        "must be visible, not absorbed")


def test_the_gates_are_conjunctive_and_the_tightest_one_is_what_is_recorded():
    """All three are applied, in §6.4's order, and each may only lower the split.
    `limiting_gate` names the gate that actually DETERMINED each `P_o` -- the last
    one to reduce it -- rather than the first one with an opinion about it."""
    plan, schedule = _fixture(T=256)
    base = SequenceParallelPolicy(workspace_cap=_WIDE_CAP, bandwidth_margin=1e6)
    unconstrained = select_sequence_parallelism(
        plan, schedule, policy=base, sms=80, blocks_per_sm=8)
    assert unconstrained.split
    for policy, gate in (
        (SequenceParallelPolicy(workspace_cap=_WIDE_CAP, bandwidth_margin=0.0), "bandwidth"),
        (SequenceParallelPolicy(workspace_cap=0, bandwidth_margin=1e6), "capacity"),
    ):
        tighter = select_sequence_parallelism(plan, schedule, policy=policy,
                                              sms=80, blocks_per_sm=8)
        assert tighter.total_segments <= unconstrained.total_segments
        assert tighter.binding_gate == gate
    assert set(GATES) == {"occupancy", "bandwidth", "capacity"}


# ---------------------------------------------------------------------------
# The ledger and the cap
# ---------------------------------------------------------------------------


def test_the_ledger_accounts_arena_workspace_and_materializations_against_one_cap():
    """§6.4: "What is unified is the LEDGER, not the allocator." The workspace is a
    launch-scoped torch tensor and the arena is the §7.1 page allocator -- different
    owners, different lifetimes, deliberately -- but one account."""
    plan, schedule = _fixture(T=256)
    arena = 1 << 20
    sp = select_sequence_parallelism(
        plan, schedule,
        policy=SequenceParallelPolicy(workspace_cap=_WIDE_CAP, bandwidth_margin=1e6),
        arena_page_bytes=arena, sms=80, blocks_per_sm=8)
    ledger = sp.ledger
    assert ledger.arena_page_bytes == arena
    assert ledger.materialization_bytes == schedule.schedule_bytes() + schedule.stream_bytes()
    assert ledger.workspace_bytes == sp.n_slots * slot_bytes(sp.BC, sp.cols)
    assert (ledger.projected_peak_bytes
            == ledger.arena_page_bytes + ledger.workspace_bytes + ledger.materialization_bytes)
    assert ledger.fits


def test_a_user_supplied_cap_wins_over_the_fraction_of_total_memory():
    plan, schedule = _fixture()
    sp = select_sequence_parallelism(
        plan, schedule, policy=SequenceParallelPolicy(workspace_cap=1 << 22),
        sms=80, blocks_per_sm=8)
    assert sp.ledger.cap_bytes == 1 << 22
    assert sp.ledger.cap_source == "user"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="resolves against device memory")
def test_an_unset_cap_resolves_to_a_deterministic_fraction_of_total_memory():
    """§6.4: "`None` resolves to a small fixed fraction of TOTAL device memory (a few
    percent, the cuBLAS/FA workspace convention) -- DETERMINISTIC across runs, unlike
    free-memory queries, and safe against activations allocated later in the step."

    Determinism is the property under test: the same call must resolve to the same
    cap while other allocations come and go.
    """
    plan, schedule = _fixture(device="cuda")
    policy = SequenceParallelPolicy()
    first = select_sequence_parallelism(plan, schedule, policy=policy, sms=80, blocks_per_sm=8)
    ballast = torch.empty(1 << 26, dtype=torch.uint8, device="cuda")
    second = select_sequence_parallelism(plan, schedule, policy=policy, sms=80, blocks_per_sm=8)
    del ballast
    total = torch.cuda.get_device_properties("cuda").total_memory
    assert first.ledger.cap_source == "fraction_of_total"
    assert first.ledger.cap_bytes == second.ledger.cap_bytes, (
        "the resolved cap moved when unrelated memory was allocated, so it is reading "
        "free memory rather than total")
    assert first.ledger.cap_bytes == int(policy.cap_fraction * total)


# ---------------------------------------------------------------------------
# Measured constants, and the occupancy the gate actually uses
# ---------------------------------------------------------------------------


def test_the_roundtrip_price_is_a_step_over_the_measured_points_not_an_extrapolation():
    """The two `bench_state_roundtrip` points differ by WHICH MEMORY serves them (L2
    versus an L2/HBM blend), so a line drawn between them would be a line through a
    regime change. Below the smaller footprint the smaller point is used
    unextrapolated -- it is the closest measurement that exists."""
    small, large = sorted(MEASURED_STATE_ROUNDTRIP_BYTES_PER_S)
    assert state_roundtrip_bytes_per_s(1 << 10) == MEASURED_STATE_ROUNDTRIP_BYTES_PER_S[small]
    assert state_roundtrip_bytes_per_s(small) == MEASURED_STATE_ROUNDTRIP_BYTES_PER_S[small]
    assert state_roundtrip_bytes_per_s(small + 1) == MEASURED_STATE_ROUNDTRIP_BYTES_PER_S[large]
    assert state_roundtrip_bytes_per_s(1 << 30) == MEASURED_STATE_ROUNDTRIP_BYTES_PER_S[large]
    # The roundtrip points are L2-blended and are ABOVE the clean-HBM figure; that is
    # explained in the source measurement, not an error, and the planner must not
    # quietly "fix" it by clamping.
    assert MEASURED_STATE_ROUNDTRIP_BYTES_PER_S[small] > MEASURED_HBM_BYTES_PER_S


def test_the_roundtrip_bytes_are_the_formula_the_spec_writes():
    """§6.4 gate 2, verbatim: `(P_o - 1) * BC * cols * 4 B * 2`."""
    plan, schedule = _fixture(T=256)
    sp = select_sequence_parallelism(
        plan, schedule,
        policy=SequenceParallelPolicy(workspace_cap=_WIDE_CAP, bandwidth_margin=1e6),
        sms=80, blocks_per_sm=8)
    assert sp.split
    expected = sum((p - 1) * sp.BC * sp.cols * 4 * 2 for p in sp.P)
    assert sp.roundtrip_bytes == expected
    assert sp.roundtrip_seconds == pytest.approx(
        expected / state_roundtrip_bytes_per_s(sp.BC * sp.cols * 4))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="asks the CUDA driver")
def test_the_occupancy_gate_reads_the_driver_for_the_instantiation_it_will_launch():
    """The occupancy gate's `blocks_per_sm` is
    `cudaOccupancyMaxActiveBlocksPerMultiprocessor` on the REAL templated kernel at
    its REAL dynamic SMEM size -- not a register-count table for a differently
    shaped kernel. It must therefore respond to the plan axes that change the
    kernel's footprint.

    Fix pass R5: this asserted only `>= 1` for three plan variants, which a stub
    returning a constant passes. What moves the answer is the geometry the plan's
    own axes imply -- `BC` sizes the state panel and the gather buffers, `BT` sizes
    the token panels -- so that is what is asserted, as strict inequalities.

    **THE ANALYTIC-MIRROR HALF OF THIS TEST IS DELETED** (item 6b, 2026-08-01). It
    asserted `plan.smem_bytes` and `plan.predicted_blocks_per_sm` -- a
    shared-memory-only model of the DELETED consumer's footprint, measured
    INVERTING the bound it claimed against this consumer's driver answer (mirror 2
    against driver 3 at the at-scale `BC = 64` cell). There is one accounting now,
    the driver's, so there is nothing to pin a second one against.

    `global` is likewise no longer asserted as a footprint axis. It is a template
    axis of the kernel, but the occupancy query PINS it to `global`, which costs
    strictly more shared memory (one `[BT]` float `den` row) and therefore can only
    under-report a raw plan -- see `schedule_v3.consumer_occupancy`.
    """
    from dataclasses import replace

    plan, _ = _fixture(device="cuda")
    base = consumer_blocks_per_sm(plan)
    assert base >= 1
    bigger_BC = replace(plan, BC=64, warp_tile=(1, max(1, plan.topology.d_v // 32)))
    coarser_BT = replace(plan, BT=32)
    assert consumer_blocks_per_sm(bigger_BC) < base, (
        "a 4x larger owner rectangle did not lower CTAs/SM, so the query is not "
        "reading the SMEM size of the instantiation the plan will launch")
    assert consumer_blocks_per_sm(coarser_BT) < base, (
        "a 2x larger token tile did not lower CTAs/SM, so the query is insensitive to "
        "the other axis that sizes the kernel's SMEM")
    assert consumer_blocks_per_sm(bigger_BC) >= 1
    # The DECAY and COMPACT arms are the query's other two axes, and what a
    # constant-returning stub fails is that neither can report MORE CTAs than the
    # arm carrying strictly less shared memory: the decay arm carries extra `[BC]`
    # vectors and the compact arm carries `c_idx`/`c_slot`/`c_count`.
    for arm in (replace(plan, decay=True), replace(plan, compact=True)):
        assert consumer_blocks_per_sm(arm) <= base, (
            "an arm that carries strictly more shared memory reported MORE CTAs/SM; "
            "the query is not reading the instantiation the plan will launch")
