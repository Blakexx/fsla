"""THE runtime planner (``THUNDER_RUN_SPEC.md`` §4.3): a pure function of recorded inputs.

``select_config`` chooses ``(BC, BT, feeder_flags)`` from three arguments and
nothing else -- a static configuration record, a hardware-facts record, and one
:class:`~rola.routing.summaries.RoutingSummaries` from the producer epilogue. It reads
no tensor, issues no reduction, allocates nothing, and does not touch the routing. That
is what makes it unit-testable against recorded rows, which is the whole point of the
end-state ruling: what audits selection is an offline agreement gate, not a runtime
model that audits itself.

Symbols: ``L``/``T`` tokens; ``N`` leaves; ``D`` levels; ``b_l`` widths; ``BC`` leaves
per owner block; ``BT`` token tile; ``BH`` = batch x heads; ``d_v`` value dim;
``MMA_K_QUANTUM = 16``; ``QPO = BC / MMA_K_QUANTUM``; ``ds = ceil(BC/MMA_K_QUANTUM)``
the DENSE quantum count; ``k_hat`` the mean of ``ceil(K_hat/MMA_K_QUANTUM)`` over
candidate pairs.

---- WHAT MOVED, AND THE ONE INTERPRETATION THIS FILE MAKES -------------------

§6 P4 retires "census + ``plan_cost_per_token`` + constants" to
``rola/planner/calibration/``. What that leaves the runtime is read here as §9
step 2 spells it -- "re-calibration on the isolating cells ...; **rule-table
recompile**" -- i.e. the FITTING INSTRUMENT and the exact census leave, and their
compiled output ships as a small RULE TABLE. So:

* DELETED from the hot path: the exact per-``(owner, tile)`` census, ``TileUnionStats``,
  ``_tile_union_stats_multi``, and the argmin over measured statistics. With them goes
  the ``torch.compile``/inductor RUNTIME dependency.
* KEPT, as :data:`RULE_TABLE_PROVENANCE`'s four coefficients: the per-token polynomial's
  constants. §4.3 asks for an "arithmetic argmin", and `PLANNER_ENDSTATE_DESIGN.md`
  §1.3's own instruction is **"ship the inequality, not the constant"** -- its margin at
  ``BC = 16`` / no-decay is 5 %, thin enough that a baked-in answer would be a guess.
  Shipping the inequality requires its coefficients.

The distinction that matters: nothing here MEASURES. Every coefficient is a recorded
number with a named isolating cell, refitted offline and recompiled into this table.

---- THE DECISION ORDER, AND WHY IT IS ACYCLIC -------------------------------

1. ``feeder_flags`` = static density OR detected saturation (§4.4's two triggers).
   Depends on nothing else.
2. ``(BC, BT)`` argmin: for each feasible pair, query the occupancy oracle for the
   footprint that pair launches (§4.3), form ``cost_per_token * waves + atomic``,
   take the minimum.

There is no back-edge: the consumer instantiates one executor, so the footprint the
oracle is asked about is the footprint that launches, and nothing downstream of the
argmin feeds back into it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import prod

from rola.planner.feeder import FeederFlags

#: §6's MMA k-extent. A SHAPE of the instruction the kernel issues, never a threshold.
MMA_K_QUANTUM = 16

#: The ``BT`` the consumer instantiates, finest first. The finest is also the
#: granularity §4.2 specifies the layer's summary call at.
SUPPORTED_BT = (16, 32)

#: The ``BC`` the consumer instantiates (``v3_consumer.cu``'s ``V3_SMEM_BC`` ladder).
SUPPORTED_BC = (16, 64)


# ---------------------------------------------------------------------------
# §8 CALIBRATION PLACEHOLDERS
# ---------------------------------------------------------------------------
#
# I5: calibration-owned constants ship as NAMED PLACEHOLDERS with the formula that
# will fill them, and the code runs on the stated conservative default. §8's rule for
# choosing those defaults is that a WRONG placeholder degrades to today's behaviour and
# never to a new one.

#: **§8 PLACEHOLDER.** The occupancy fraction at or above which a ``(side, level)`` cell
#: is treated as dense (§4.4's detection trigger). FILLED BY: an occupancy sweep on the
#: recorded cells. DEFAULT ``1.0`` -- EXACT-dense only.
#:
#: The default is not merely conservative, it is the only defensible value until the
#: sweep exists: any structural zero at all breaks the closed form the flag licenses,
#: and a soft epsilon would re-introduce exactly the distributional branch I2 forbids.
C_SATURATION: float = 1.0

#: **§8 PLACEHOLDER.** The cross-level collision correction ``K_est = gamma * K_hat``.
#: FILLED BY: a census fit against the exact per-pair ``K``, per ``D``, on untrained and
#: trained routing (at ``D = 1`` the factorization is the identity and ``gamma == 1``
#: exactly). DEFAULT ``1.0`` -- the bound uncorrected.
#:
#: **IT IS APPLIED IN THE SUMMARY PASS, NOT HERE, and that placement is load-bearing.**
#: §1.4 of `PLANNER_ENDSTATE_DESIGN.md` requires it per ``(owner, tile)`` BEFORE
#: quantization, "never to the mean -- because ``ceil`` is nonlinear and the ``K = 0``
#: mass is a distribution fact the mean cannot carry". A histogram of already-quantized
#: values cannot express that, so scaling ``k_hat`` here would be a different and
#: LESS conservative correction. The layer therefore hands this constant to
#: :func:`rola.routing.summaries.summarize_routing`, which applies it before its
#: ``ceil``. At the default it is bit-identically inert either way.
C_COLLISION_GAMMA: float = 1.0

#: **§8 PLACEHOLDER.** The split-K atomic coefficient: the per-lane-op cost of the
#: ``N/BC``-way reduction, in the same work-units as the rule table. FILLED BY: r1
#: M0's ``N/BC`` sweep. DEFAULT ``0.0`` -- the term is INERT.
#:
#: It is the biggest at-scale term of the ``BC`` choice and the reason ``BC`` is a
#: planner axis at all (`CONSUMER_ACT3_PRICING.md` §9 derives 3.44e10 lane-ops at the
#: parity cell). At ``0.0`` the ``BC`` argmin reduces to per-leaf amortization plus
#: grid fill, which is today's behaviour -- ``BC`` was not selected at all before.
C_ATOMIC_COEFF: float = 0.0


# ---------------------------------------------------------------------------
# The RULE TABLE -- the compiled output of the offline calibration (§9 step 2)
# ---------------------------------------------------------------------------
#
# Every coefficient below was measured by `ncu --clock-control base`,
# `smsp__inst_executed.sum`, ONE profiled launch per arm, on an ISOLATING cell or cell
# PAIR rather than fitted to the rows it must reproduce. Their full derivations,
# falsifiers and known residuals live with the fitting instrument in
# `rola/planner/calibration/cost_model.py`; what is repeated here is the number,
# its cell, and the one sentence that says what it prices. A reader who wants to
# CHALLENGE a value goes to the instrument; a reader who wants to know what the planner
# believes reads this.

#: Per-candidate-tile fixed work that does not scale with the gathered extent -- epoch
#: bookkeeping, row scales, amplitude staging, block barriers, the write-conditioned
#: publish. The term that makes a LARGER `BT` cheaper, and the only one that does.
#: Cell: two independent isolating solves over `iso-tile-*`, intercepts 3507.0 / 3513.7.
RULE_TILE_FIXED_WORK = (3507.0 + 3513.7) / 2.0

#: Per-k-step cost of the readout, the fold and the panel builds -- the terms that ride
#: the k-quantum count. Cell: `atscale-L2048`.
RULE_READOUT_FOLD_KSTEP_WORK = 1755.0

#: Per-k-step cost of the intra Gram's own contraction. Cell: `wk1-D2-w64x64-BC64`
#: against `topo-D2-w64x64-BC64`, which differ only in `KI`.
RULE_INTRA_GRAM_KSTEP_WORK = 144.0

#: Per-k-step cost of the LeafMassDecay arm on top of the same tile. Two isolating
#: pairs, slopes 1106.3 / 959.7, both intercepts within 10 % of zero -- decay rides the
#: `ceil(K/16)` quantum exactly like the readout, and the `BT*BC` alternative is
#: refuted by 15.8x.
RULE_DECAY_KSTEP_WORK = (1106.3 + 959.7) / 2.0

RULE_TABLE_PROVENANCE = (
    "RTX 3080 Ti, `ncu --clock-control base`, `smsp__inst_executed.sum`, ONE profiled "
    "launch per arm (`--launch-skip 1 --launch-count 1`), every run flock-serialized; "
    "re-based 2026-08-01 onto the surviving consumer (the previous basis measured a "
    "kernel that no longer exists). This table is the COMPILED "
    "OUTPUT of `rola/planner/calibration/cost_model.py`; it is recompiled by §9 "
    "step 2 whenever a kernel or a constant moves, and WHAT BINDS IT is the auditor's "
    "agreement gate against the eight measured argmins of the §2.32 matrix -- measured "
    "argmins, not model outputs, which is exactly the right anchor for a selector that "
    "no longer runs a model. "
    "KNOWN AND MEASURED, NOT MODELLED, so the next reader does not rediscover it: "
    "(1) a per-COHORT remainder of 4036-9506 instructions (CTA prologue/epilogue, "
    "state load/store) -- BT- and arm-invariant per token, hence unable to move any "
    "argmin, and below 1 % of the at-scale cells; (2) 74.6 per candidate tile per FIXED "
    "level, the only geometry dependence that survives measurement (the per-staged-Gram-"
    "column price is EXACTLY 0.000, measured byte-identical across a 2.1x range of "
    "gram_cols); (3) the dense k-quantum is OVER-priced at 1899 (readout+intra) against "
    "1033/1155/1316 measured at D=1/2/3, the leading absolute-total residual, growing "
    "~141 per level per quantum. "
    "KNOWN LIMITATION, PROVED RATHER THAN SUSPECTED: on a SATURATED-DENSE cell with "
    "decay the per-slice BT preference INVERTS against any work model, because the "
    "machine is stall-dominated there (BT=32 retires 14 % FEWER instructions in "
    "5.7-27.2 % MORE time) and a work model does not model stalls.")


def quanta_per_owner(BC: int) -> int:
    """``QPO = ceil(BC / MMA_K_QUANTUM)`` -- the dense k-quantum count ``ds``.

    I1: every quanta-shaped quantity in this module goes through this function, so a
    ``BC``-derived literal cannot exist. The ``ceil`` form is kept although §1b
    guarantees ``MMA_K_QUANTUM | BC``, because the guarantee belongs to the geometry
    validator and restating it as a division would make this expression depend on it.
    """
    return -(-int(BC) // MMA_K_QUANTUM)


# ---------------------------------------------------------------------------
# The three inputs (§4.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StaticPlanInputs:
    """Tier 0: everything derivable at layer construction. Never re-derived per step.

    ``feasible_BC`` is the alignment-admissible set intersected with the instantiated
    ladder (§6 P6). ``pinned_BC`` is the expert override: when it is set the feasible
    set is that single value, which is what "``state_block`` remains only as an expert
    pin that shrinks the feasible set to itself" (§4.3) means operationally.
    """

    widths: tuple[int, ...]
    d_v: int
    normalization: str
    decay: bool
    mma_precision: str
    density: object                      # rola.planner.schedule.StaticDensity
    feasible_BC: tuple[int, ...]
    BH: int
    pinned_BC: int | None = None

    @property
    def D(self) -> int:
        return len(self.widths)

    @property
    def N(self) -> int:
        return prod(self.widths)

    def owners_total(self, BC: int) -> int:
        return self.N // BC

    def n_cohorts(self, BC: int) -> int:
        return self.owners_total(BC) * self.BH


@dataclass(frozen=True, slots=True)
class HardwareFacts:
    """Tier H: measured once per (device, install), carried with its provenance.

    ``occupancy`` is the DRIVER's
    ``cudaOccupancyMaxActiveBlocksPerMultiprocessor`` on the real templated kernel at
    its real dynamic shared-memory size -- the SOLE AUTHORITY for physical facts
    (standing ruling). It enters selection as an INPUT and is recorded next to the
    decision it drove, never queried ambiently, which is what keeps ``select_config``
    a pure function of its recorded inputs.
    """

    sm_count: int
    occupancy: object                    # rola.planner.schedule.OccupancyOracle
    hbm_bytes_per_s: float | None = None
    #: ``C_ATOMIC_COEFF``'s per-install override; ``None`` uses the §8 default.
    atomic_coeff: float | None = None

    def resolved_atomic_coeff(self) -> float:
        return C_ATOMIC_COEFF if self.atomic_coeff is None else float(self.atomic_coeff)


@dataclass(frozen=True, slots=True)
class ConfigChoice:
    """§4.3's frozen tuple. Nothing else lives here.

    Evidence -- the residency the decision was reasoned at, the wave count, the key it
    minimized -- is deliberately NOT a field: §4.3 defines
    ``ConfigChoice = (BC, BT, feeder_flags)`` and neither track may edit it.
    :func:`select_config_with_evidence` returns the evidence beside the choice for the
    auditor and the shadow ledger.
    """

    BC: int
    BT: int
    feeder_flags: FeederFlags


@dataclass(frozen=True, slots=True)
class ConfigEvidence:
    """Why a :class:`ConfigChoice` is what it is. Recorded, never re-read by a rule."""

    blocks_per_sm: int
    waves: float
    cost_per_token: float
    effective_key: float
    k_hat: float
    candidate_frac: float
    #: ``{(BC, BT): key}`` over the whole feasible grid -- the margin the decision won
    #: by, which is what the approximation contract (§2) says a decision must carry.
    grid: dict[tuple[int, int], float] = field(default_factory=dict)
    provenance: str = ""

    @property
    def margin(self) -> float:
        """Relative gap to the runner-up, or ``inf`` when there is only one candidate."""
        keys = sorted(self.grid.values())
        if len(keys) < 2 or keys[0] <= 0.0:
            return float("inf")
        return (keys[1] - keys[0]) / keys[0]


# ---------------------------------------------------------------------------
# The feasible set (§6 P6)
# ---------------------------------------------------------------------------


def feasible_bc(widths: tuple[int, ...], *,
                instantiated: tuple[int, ...] = SUPPORTED_BC,
                pinned: int | None = None) -> tuple[int, ...]:
    """``{BC in instantiated : owner_geometry(widths, BC) admits}``, ascending.

    ALIGNMENT is the filter, and it is not a formality: §3 requires a ``BC``-aligned
    rectangle, and a topology whose narrow trailing levels cannot compose one at a
    given ``BC`` has no owner geometry there at all. ``widths = (2, 4, 16)`` admits
    ``BC = 16`` and refuses ``BC = 64`` (``64`` would need 8 digits of a width-4 level).

    ``pinned`` is ``state_block``, the EXPERT OVERRIDE: it shrinks the set to itself
    and is validated against alignment exactly like any other member, so an
    inadmissible pin is a construction-time error rather than a plan-time surprise
    (§6 P6). It is NOT filtered against ``instantiated``: a caller pinning a geometry
    off the ladder is taking responsibility for supplying an oracle, which is the
    documented contract of the injection interface.

    **HONEST SCOPE.** ``instantiated`` has TWO elements today, so the "search" is a
    binary choice. Its at-scale value is contingent on a ``BC = 128`` instantiation,
    which `CONSUMER_ACT3_PRICING.md` §9 prices as SMEM-blocked. This function is
    written generically because I1 requires it and because the ladder is the thing
    expected to grow -- not because a two-element set needs an optimizer.
    """
    from rola.planner.schedule import owner_geometry

    candidates = (int(pinned),) if pinned is not None else tuple(sorted(instantiated))
    out = []
    for BC in candidates:
        try:
            owner_geometry(tuple(widths), BC)
        except ValueError:
            continue
        out.append(BC)
    if not out:
        raise ValueError(
            f"no feasible BC for widths={tuple(widths)}: none of {candidates} names a "
            "BC-aligned owner rectangle (spec §3 requires the block to be exactly one "
            "rectangle -- fixed digits x an aligned sub-range of the boundary level x a "
            "full span below it). This is a TOPOLOGY defect, not a runtime condition: "
            "permute a wider level innermost, or instantiate a BC the topology admits.")
    return tuple(out)


def largest_feasible_bc(widths: tuple[int, ...], **kwargs) -> int:
    """The §6 P6 DEFAULT, replacing ``widths[-1]``.

    ``widths[-1]`` was never a planner decision -- it is the innermost level's width,
    which happens to name an aligned rectangle and happens to be instantiated on the
    freeze topology. It is neither on a topology whose last level is wide (it exceeds
    the ladder) or narrow (it wastes the state tile).

    The largest feasible ``BC`` is the right default for a reason the cost model states
    and `CONSUMER_ACT3_PRICING.md` §9 sharpens: per-token cost per LEAF is
    ``c(BT)/BC``, so a larger block amortizes the per-tile fixed work over more leaves
    (DERIVED from the rule table: 1.95x cheaper per leaf at ``BC = 64`` than at 16
    without decay, 1.69x with), and the split-K reduction is ``N/BC``-way, so a larger
    block is ``BC``-fold less atomic traffic -- the term that dominates at scale and the
    reason ``BC`` is a planner axis at all. Both point the same way; the argmin in
    :func:`select_config` is what arbitrates when grid fill points the other way.
    """
    return feasible_bc(widths, **kwargs)[-1]


# ---------------------------------------------------------------------------
# The closed forms
# ---------------------------------------------------------------------------


def mean_quantum(quanta_hist) -> float:
    """``k_hat = sum_q q * hist[q]`` -- the mean gathered quantum over candidate pairs.

    The HISTOGRAM is what §4.2 emits and this is only one of its moments; the full
    distribution is retained on the summaries object because a breakeven keyed on the
    ``q = 0`` mass (39 % of candidate pairs at ``atscale-L2048``, MEASURED -- the pairs
    that gather nothing and skip every GEMM) needs no new statistic, only a different
    reduction. Taking the mean here is a property of TODAY's rule, not of the data.
    """
    return float(sum(q * float(quanta_hist[q]) for q in range(len(quanta_hist))))


def cost_per_token(BC: int, BT: int, *, decay: bool,
                   candidate_frac: float) -> float:
    """The rule table's per-token polynomial, in relative work-units.

    Per CANDIDATE tile, then divided by ``BT`` so the result is comparable across
    ``BT``, then scaled by the candidate fraction::

        W = TILE_FIXED
          + (READOUT + [decay] DECAY) * ds * (BT / 16)
          + INTRA                     * ds * (BT (BT+1) / 272)

        cost_per_token = candidate_frac * W / BT

    The k-extent is the DENSE quantum count ``ds`` and never a realized one: the
    consumer walks the full ``BC`` whatever the routing realizes, so the extents are
    the template's and not the routing's. ``owners_total`` is ``BT``-invariant and
    cancels from every comparison at fixed ``BC``; the ``BC`` comparison restores it
    through the wave term and the atomic term, which is where it genuinely differs.

    **``candidate_frac`` is a DEFINITION, not a correction.** A ``(owner, tile)`` pair
    outside the §4 bitmap is never enumerated by any CTA and costs nothing at all, and
    there are ``candidate_frac * owners_total / BT`` visited tiles per token. Omitting
    it asserts ``candidate_frac == 1``, which is true on a saturated cell and MEASURED
    wrong by 1.8x on a routed one.
    """
    k = float(quanta_per_owner(BC))
    w_tile = RULE_TILE_FIXED_WORK
    w_tile += ((RULE_READOUT_FOLD_KSTEP_WORK + (RULE_DECAY_KSTEP_WORK if decay else 0.0))
               * k * (BT / MMA_K_QUANTUM))
    w_tile += (RULE_INTRA_GRAM_KSTEP_WORK * k
               * (BT * (BT + 1) / float(MMA_K_QUANTUM * (MMA_K_QUANTUM + 1))))
    return float(candidate_frac) * w_tile / BT


def waves(n_cohorts: int, blocks_per_sm: int, sm_count: int) -> float:
    """``max(1, n_cohorts / (sm_count * blocks_per_sm))``.

    Below one wave the whole grid is resident, so elapsed time is one CTA's latency and
    residency cannot help a machine that is already holding everything. Above it the
    CTAs serialize in batches of ``sm_count * blocks_per_sm`` and a candidate that fits
    fewer CTAs per SM pays proportionally. The floor at 1 is structural -- a wave count
    cannot be less than one wave of latency -- not a threshold.

    This is the ONE physical term in the argmin and it is what the eight measured
    argmins partition on: every many-wave cell wants the smaller tile, every one-wave
    cell the larger, and NEITHER ``BC`` NOR ``decay`` appears in that partition.
    """
    if sm_count < 1:
        raise ValueError(f"sm_count must be >= 1, got {sm_count}")
    if blocks_per_sm < 1:
        raise ValueError(f"blocks_per_sm must be >= 1, got {blocks_per_sm}")
    return max(1.0, n_cohorts / float(sm_count * blocks_per_sm))


def atomic_work_per_token(static: StaticPlanInputs, BC: int, coeff: float) -> float:
    """``C_ATOMIC_COEFF * owners_total * d_v`` -- the split-K reduction, per token per head.

    `CONSUMER_ACT3_PRICING.md` §9 counts the launch's atomic lane-ops as
    ``L * H * (N/BC) * DV``; dividing by ``L * H`` puts it in the same per-token
    currency as :func:`cost_per_token`. It is the term that makes ``BC`` worth choosing
    at scale -- 16x larger at the freeze cell than at any cell the instruction census
    was taken on, and 16x again at ``N = 10^6`` -- and it is INERT at the §8 default of
    ``0.0``, which is stated rather than hidden: until r1's M0(d) sweep exists, the
    ``BC`` argmin is per-leaf amortization and grid fill only.
    """
    return float(coeff) * static.owners_total(BC) * static.d_v


# ---------------------------------------------------------------------------
# §4.3's two entry points
# ---------------------------------------------------------------------------


def select_config(static: StaticPlanInputs, hw: HardwareFacts,
                  summaries) -> ConfigChoice:
    """§4.3's frozen signature. See :func:`select_config_with_evidence`."""
    return select_config_with_evidence(static, hw, summaries)[0]


def select_config_with_evidence(static: StaticPlanInputs, hw: HardwareFacts,
                                summaries, *,
                                summaries_by_bt=None,
                                ) -> tuple[ConfigChoice, ConfigEvidence]:
    """The joint ``(BC, BT)`` argmin, with the evidence the auditor and shadow need.

    ``summaries_by_bt`` is an OPTIONAL ``{BT: RoutingSummaries}`` mapping. §4.2 freezes
    ``candidate_frac`` and ``quanta_hist`` as single-granularity fields measured "at
    finest BT", but both are statistics OF A TILING and neither is ``BT``-invariant:
    ``cf(16) = 0.1936`` against ``cf(32) = 0.3574`` is MEASURED at
    ``atscale-L2048-BC16``, a ratio of 1.846. Reusing the finest-``BT`` value at
    ``BT = 32`` under-prices the coarser tile (``cf`` too small), in the direction
    §4.2's own safety argument runs against.

    This argument is how a caller supplies the per-``BT`` statistic WITHOUT any change
    to §4.2's struct or to :func:`select_config`'s frozen signature: it calls the
    summary pass once per candidate granularity (the pass is
    ``O(BH * n_tiles * owners_total)``, so a second call is affordable) and hands the
    mapping here. With it absent the single object is used at every ``BT`` and the bias
    above is live -- recorded in ``ConfigEvidence.provenance`` so a plan states which
    arm it took rather than hiding it.
    """
    flags = FeederFlags.derive(static.density, summaries.occupancy, saturation=C_SATURATION)
    feasible = (feasible_bc(static.widths, pinned=static.pinned_BC)
                if static.pinned_BC is not None else static.feasible_BC)
    if not feasible:
        raise ValueError(
            "StaticPlanInputs carries an empty feasible_BC set; the topology admits no "
            "instantiated BC-aligned owner rectangle and no plan exists for it")
    coeff = hw.resolved_atomic_coeff()

    best = None
    grid: dict[tuple[int, int], float] = {}
    for BC in feasible:
        for BT in SUPPORTED_BT:
            per_bt = summaries if summaries_by_bt is None else summaries_by_bt[BT]
            k_hat = mean_quantum(per_bt.quanta_hist)
            cf = per_bt.candidate_frac
            blocks = hw.occupancy(BT, BC, static.d_v, static.normalization == "global",
                                  static.decay, static.mma_precision,
                                  _gram_cols(static.widths, BC), 0)
            if blocks == 0:
                raise ValueError(
                    f"the candidate BC={BC}, BT={BT} does not fit an "
                    "SM: the occupancy oracle reports 0 CTAs/SM, so its footprint "
                    "exceeds the shared memory and the launch would fail")
            cost = cost_per_token(BC, BT, decay=static.decay, candidate_frac=cf)
            w = waves(static.n_cohorts(BC), blocks, hw.sm_count)
            key = cost * w + atomic_work_per_token(static, BC, coeff)
            grid[(BC, BT)] = key
            if best is None or key < best[0]:
                best = (key, BC, BT, blocks, w, cost, k_hat, cf)

    _, BC, BT, blocks, w, cost, k_hat, cf = best
    granularity = ("per-BT summaries supplied"
                   if summaries_by_bt is not None
                   else "SINGLE-granularity summaries reused at every BT (§4.2 frozen "
                        "shape; cf and k_hat are not BT-invariant -- see the findings)")
    return ConfigChoice(BC=BC, BT=BT, feeder_flags=flags), ConfigEvidence(
        blocks_per_sm=blocks, waves=w, cost_per_token=cost, effective_key=grid[(BC, BT)],
        k_hat=k_hat, candidate_frac=cf, grid=grid,
        provenance=f"waves={w:.4f} at {blocks} CTAs/SM; {granularity}")


def _gram_cols(widths: tuple[int, ...], BC: int) -> int:
    """``sum_{l >= j} m_l`` over the owner rectangle's spanned moduli.

    Passed to the occupancy oracle to satisfy its signature. The DRIVER oracle ignores
    it -- the consumer's shared memory is entirely compile-time -- but an injected
    oracle is a caller taking responsibility for the machine and may key on whatever it
    likes, so the argument is computed rather than stubbed. Derived from
    ``owner_geometry`` and never re-derived here: the rectangle has one owner.
    """
    from rola.planner.schedule import owner_geometry

    geometry = owner_geometry(tuple(widths), BC)
    j = geometry.first_spanned_level
    return geometry.slice_digits + sum(widths[j + 1:])


def select_p(plan, schedule, hw: HardwareFacts, *, policy=None,
             arena_page_bytes: int = 0):
    """§4.3's second entry point: §6.4's sequence-parallel policy, UNCHANGED.

    It is a SEPARATE function from :func:`select_config` for a structural reason and
    not an implementation one: every input §6.4's three gates read --
    ``schedule.owner_costs``, ``stream_bytes()``, ``schedule_bytes()``, ``n_tiles``,
    ``BH``, ``T`` -- is a ``Schedule`` field, and the ``Schedule`` is ``BT``-keyed, so
    it cannot exist before ``BT`` is chosen. ``P`` is downstream of the schedule by
    construction. `PLANNER_STATS_SPEC.md` §4's single-function shape is amended here
    rather than worked around.

    The gates themselves are untouched; this only routes the machine facts through
    :class:`HardwareFacts` so the whole planner reads ONE hardware record.
    """
    from rola.routing.sequence_parallel import select_sequence_parallelism

    blocks = hw.occupancy(plan.BT, plan.BC, plan.topology.d_v,
                          plan.normalization == "global", plan.decay,
                          plan.mma_precision, 0, plan.ring_slots)
    return select_sequence_parallelism(
        plan, schedule, policy=policy, arena_page_bytes=arena_page_bytes,
        blocks_per_sm=blocks, sms=hw.sm_count)


# ---------------------------------------------------------------------------
# §6b: DECODE BYPASSES SELECTION ARCHITECTURALLY (user ruling, 2026-08-02)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrozenConfig:
    """The config the PREFILL -> DECODE boundary freezes, and the reason it can.

    At ``T = 1`` there is exactly ONE token tile whatever ``BT`` is, one candidate
    window, and one owner set -- so every statistic §4.2 measures is a one-sample
    estimate of a distribution the step does not have. Selecting from it is not
    cheap-and-approximate, it is meaningless: ``candidate_frac`` is the fraction of
    owners live in the single tile, the quantum histogram has one nonzero bin, and the
    wave term is computed for a grid that does not change from step to step anyway.

    So the decode path does not select. It carries this.

    **WHY EACH FIELD IS WHAT IT IS.**

    ``BC`` -- FROZEN BY THE STATE, not by a policy. ``BC`` fixes the state layout and
    the paging atom keying; the state is live during decode by definition, so it cannot
    move. This is the same latch `PLANNER_ENDSTATE_DESIGN.md` §3.2 puts at state init.

    ``BT`` -- the MINIMAL tile. At ``T = 1`` the tile is one token whatever the template
    says, so the only thing ``BT`` still buys is the size of the per-tile fixed cost,
    and the smallest instantiated tile is the cheapest. Structural: it is the no-tiling
    case, not a comparison against a dimension.

    ``feeder_flags`` -- from the STATIC declarations only. The detection trigger reads a
    measured occupancy, and the decode path does not measure one; a flag set from a
    single token's support would be a claim about the sequence made from one sample,
    which is exactly the unsound direction (a false dense flag is a correctness defect
    under §5.4's factored feeder, not a mis-pricing).

    ``P = 1`` -- §6.4 gate 1 closes at decode anyway (one token, no owner has segments
    to split), and running the policy on a degenerate schedule would apportion over a
    ``owner_costs`` vector of ones.
    """

    choice: ConfigChoice
    #: The prefill config this was frozen from, or ``None`` when decode began cold.
    frozen_from: ConfigChoice | None = None

    @property
    def sequence_parallel(self) -> None:
        """``P = 1``, expressed as "there is no plan" rather than as a plan of ones."""
        return None


def is_decode_step(T: int) -> bool:
    """The STRUCTURAL predicate: is this the no-tiling case?

    ``T == 1`` is a statement about the SHAPE of the schedule -- one tile, one token,
    no distribution -- in the same class as `if ND_V == 1` (a tile count). It is not
    `if T < 64` and it is not a density: there is no value of ``T`` above 1 that takes
    this branch, and no configuration that makes ``T == 1`` take the other one.
    """
    return int(T) == 1


def freeze_for_decode(static: StaticPlanInputs, *,
                      prefill: ConfigChoice | None = None) -> FrozenConfig:
    """Build §6b's carrier at the prefill -> decode boundary.

    ``prefill`` is the config the prefill launch ran, when there was one. Its ``BC`` is
    ADOPTED (the state it wrote is laid out for it and cannot be re-laid), and
    everything else is re-derived from the rules above rather than inherited -- a
    prefill that chose ``BT = 32`` chose it for a statistic decode does not have.
    """
    if prefill is not None:
        BC = prefill.BC
    elif static.pinned_BC is not None:
        BC = static.pinned_BC
    else:
        BC = largest_feasible_bc(static.widths)
    flags = FeederFlags.derive(static.density, None, saturation=C_SATURATION)
    return FrozenConfig(
        choice=ConfigChoice(BC=BC, BT=SUPPORTED_BT[0], feeder_flags=flags),
        frozen_from=prefill)


def config_for_step(static: StaticPlanInputs, hw: HardwareFacts, T: int, *,
                    summaries=None, frozen: FrozenConfig | None = None,
                    **kwargs) -> tuple[ConfigChoice, ConfigEvidence | None]:
    """THE DISPATCH. One structural branch, and the decode arm touches no statistic.

    On the decode arm this function does not call :func:`select_config`, does not call
    the summary pass, and does not call :func:`select_p`. It reads a carrier and
    returns. That is the whole of §6b: the bypass is architectural, so there is no path
    on which a degenerate statistic can reach a rule.

    ``summaries`` is therefore OPTIONAL, and passing ``None`` at ``T > 1`` is an error
    rather than a signal -- a caller that skipped the summary pass on a prefill step has
    a bug, and inferring "decode" from a missing argument would turn a bug into a
    silently different plan.
    """
    if is_decode_step(T):
        return (frozen or freeze_for_decode(static)).choice, None
    if summaries is None:
        raise ValueError(
            f"select_config needs the producer epilogue's RoutingSummaries at T={T}; "
            "only the T == 1 decode arm bypasses selection (§6b), and it is taken on the "
            "shape of the schedule, never on a missing argument")
    return select_config_with_evidence(static, hw, summaries, **kwargs)


# ---------------------------------------------------------------------------
# SHADOW MODE (§6 P3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShadowRecord:
    """One sampled step's (decision, exact decision, margin), for the auditor ledger.

    ``margin`` is the CLOSED FORM's own relative gap to its runner-up -- how much room
    the decision had, in the currency the decision was made in. It is the quantity the
    approximation contract (§2) says every decision must carry: "decisions with margins
    large vs residuals" is the validity domain, and a disagreement at a 30 % margin is a
    different finding from one at 1 %.

    ``regret`` is the disagreement's cost in the RETIRED currency, and it is ``None``
    when the two agree. It is denominated in the object being retired on purpose: that
    is the only currency in which "the closed form picked worse" is even a statement,
    and it is exactly why the shadow lives in the bench and not in the runtime.
    """

    decision: ConfigChoice
    exact: ConfigChoice | None
    margin: float
    regret: float | None
    step: int


def shadow_step(static: StaticPlanInputs, hw: HardwareFacts, summaries, *,
                exact_argmin, step: int = 0, **kwargs) -> ShadowRecord:
    """Run the closed form and the exact argmin side by side on one sampled step.

    ``exact_argmin`` is INJECTED -- ``rola.planner.calibration.auditor``'s
    exact-census argmin in the bench, a recorded table in a unit gate. It is not
    imported here, and that is the design: the exact argmin is the object the end-state
    ruling RETIRES, so a runtime that imported it would keep alive exactly what the
    ruling deletes. A shipped install without the bench therefore cannot enable the
    shadow, which is intended rather than incidental.

    THE CLOSED FORM IS WHAT DECIDES. This function returns the closed form's choice
    whatever the exact argmin says; it records, it does not arbitrate.
    """
    choice, evidence = select_config_with_evidence(static, hw, summaries, **kwargs)
    exact = exact_argmin(static, hw, summaries)
    if exact is None:
        return ShadowRecord(decision=choice, exact=None, margin=evidence.margin,
                            regret=None, step=step)
    exact_choice, exact_key = exact
    agreed = (exact_choice.BC, exact_choice.BT) == (choice.BC, choice.BT)
    regret = None if agreed else (evidence.grid[(choice.BC, choice.BT)] - exact_key) / exact_key
    return ShadowRecord(decision=choice, exact=exact_choice, margin=evidence.margin,
                        regret=regret, step=step)


__all__ = [
    "C_ATOMIC_COEFF",
    "C_COLLISION_GAMMA",
    "C_SATURATION",
    "MMA_K_QUANTUM",
    "RULE_TABLE_PROVENANCE",
    "SUPPORTED_BC",
    "SUPPORTED_BT",
    "ConfigChoice",
    "ConfigEvidence",
    "FrozenConfig",
    "HardwareFacts",
    "ShadowRecord",
    "StaticPlanInputs",
    "atomic_work_per_token",
    "config_for_step",
    "cost_per_token",
    "feasible_bc",
    "freeze_for_decode",
    "is_decode_step",
    "largest_feasible_bc",
    "mean_quantum",
    "quanta_per_owner",
    "select_config",
    "select_config_with_evidence",
    "select_p",
    "shadow_step",
    "waves",
]
