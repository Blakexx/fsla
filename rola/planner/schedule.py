"""V3 schedule builder (``ROLA_V3_SPEC.md`` §4) and plan object (§6).

The V3 schedule is deliberately tiny and is **all** of what §4 authorizes:

```text
per (level, digit): a TILE-liveness bitmap, one bit per token tile
per boundary sub-range: a slice-union bitmap (b_j / c entries)
```

Everything else the consumer needs -- token-level liveness and the routing
amplitudes -- comes from the compact route stream it must load anyway, because a
nonzero amplitude *is* a live token (§4). Leaf-level liveness never reaches the
schedule; it is operand assembly inside the CTA.

This module does **not** reuse ``routing/schedule.py``. That file is the V2
prefix-run planner (``PREFIX_SCHEDULE_ABI_VERSION = 9``, ``RouteDescriptor``,
run-length tile records): a different schedule representation keyed to the Q/K-era
descriptor ABI, in which the §4 representation is not expressible. The only thing
shared is the token-tile concept, and V3's tile size ``BT`` is a plan parameter in
``{16, 32}`` (§6) rather than V2's hard-pinned 32.

Two orientations are emitted, matching §4's table:

| level role                     | orientation                        |
|--------------------------------|------------------------------------|
| levels the owner **fixes**     | transposed ``(level,digit) -> tile bits`` |
| innermost level it **spans**   | raw ``token -> digit amplitudes`` (the route stream) |

The transpose is the schedule's entire value; the raw side is not stored twice.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product as _iter_product
from math import gcd, log1p, prod

import torch

from rola.routing.types import Topology, validate_normalization

# Bumped whenever the emitted layout changes in a way the CUDA consumer must
# observe. The consumer asserts on it at load time.
ROLA_SCHEDULE_ABI_VERSION = 2  # v2: §4's support word (`read/write_token_bits`)

#: A ``cp.async`` warp-wide transaction is 32 lanes x 16 B = 512 B (§5 "Staging":
#: "innermost rows (512 B contiguous, cp.async-shaped)").
CP_ASYNC_ROW_BYTES = 512

_SUPPORTED_BT = (16, 32)

#: The `BC` and `d_v` the consumer extension INSTANTIATES (`v3_consumer.cu`'s
#: `V3_SMEM_BC` / `V3_SMEM_DV` ladders, which `TORCH_CHECK` outside these). Selection
#: validates against them up front so the occupancy oracle is only ever asked about
#: launchable instantiations.
_SUPPORTED_BC = (16, 64)
_SUPPORTED_DV = (32, 64)

#: Ring DEPTHS the consumer implements, deepest first. Depth `n` means `n` staging
#: slots: one being consumed and `n-1` being filled by `cp.async` for the next
#: candidate tiles. **Depth 1 is the ring DECLINED** -- the kernel stages
#: synchronously inside the visit, which is the pre-ring dataflow exactly.
#:
#: This is a PLAN axis carried at RUNTIME, not a template parameter, and that is
#: what makes it free: the launch's dynamic shared-memory size is already a runtime
#: argument, so the slot count sizes the buffers with no new instantiation and the
#: 256-tuple matrix is untouched. The kernel branches on it as a block-uniform,
#: plan-selected STRUCTURAL integer -- the same class as `compact`, never a density.
#:
#: The accounting below is generic in the depth; the KERNEL currently implements
#: 1 and 2, which is what this tuple advertises. A depth-3 ring needs the lookahead
#: iterator to name two tiles ahead and is not built.
_RING_DEPTHS = (2, 1)


#: An OCCUPANCY ORACLE: `(BT, BC, d_v, global_norm, decay, mma_precision, compact,
#: gram_cols, ring_slots) -> CTAs/SM`.
#:
#: **`mma_precision` IS PART OF THE INSTANTIATION IDENTITY and is passed per query,
#: never closed over.** `MmaPrec` is a real template axis with its own register
#: pressure, so pricing a bf16 plan against the tf32x3 kernel's occupancy is exactly
#: the register-blindness this interface exists to remove.
#:
#: **A MACHINE FACT ENTERING THE PLAN THROUGH THE INTERFACE THE PLAN ALREADY HAS FOR
#: MACHINE FACTS** (coordinator ruling 2026-07-31): supplied as an input, recorded
#: next to the decision it drove, never queried ambiently from inside selection.
#: Selection is therefore a PURE FUNCTION of its recorded inputs -- same inputs, same
#: plan -- which is the machine-independence §10 actually needs, as distinct from
#: "never look at the machine"; and tests inject a table.
#:
#: **THERE IS NO ANALYTIC MIRROR AND NO FALLBACK ORACLE** (item 6b, 2026-08-01). The
#: shared-memory-only `mirror_occupancy` modelled the DELETED consumer's footprint,
#: and the exhaustive driver probe measured it INVERTING the bound it claimed (mirror
#: 2 against the driver's 3 at the at-scale `BC = 64` cell), so it was not merely
#: loose but wrong in direction. A planning call with no injected oracle and no bound
#: extension is now an ImportError, not a silently worse plan.
OccupancyOracle = Callable[..., int]


def consumer_occupancy() -> OccupancyOracle:
    """THE occupancy oracle: `cudaOccupancyMaxActiveBlocksPerMultiprocessor` on the
    real templated kernel at its real dynamic SMEM size.

    Register-aware, which is the entire point: a shared-memory-only model cannot see
    registers, and at `BC = 16` the REGISTER FILE is the binding limit.

    **THE MIS-KEY THIS REPLACED IS A MEASURED DEFECT, NOT A HYPOTHETICAL ONE**
    (`rewrite.md` §2.34). The previous default queried the REGISTER-RESIDENT
    consumer (deleted at item 6b; its header was also named `v3_consumer_kernel.cuh`
    -- readable at `c3c2f845`, and a different kernel from today's file of that
    name), which has a different footprint -- 31680 B against 44672 at
    `BT = 16, BC = 64, d_v = 64`, 44416 against 72448 at `BT = 32` -- so it reported
    `2 / 1` CTAs/SM where the launched kernel runs at `3 / 2`, and the joint argmin
    reproduced 4 of its 8 measured rows instead of 8. With the right key `ncu`'s
    `launch__occupancy_limit_registers` and the driver agree on all 16 profiled
    instantiations, byte for byte on `launch__shared_mem_per_block_dynamic`.

    **`global_norm` IS A REAL TEMPLATE AXIS THAT THIS ORACLE DOES NOT RESOLVE, and
    the pinning STANDS** (coordinator ruling, 2026-08-01, `rewrite.md` §2.37).
    `launch_consumer` instantiates `GLOBAL` both ways; `rola_consumer_occupancy`
    pins it to `true`. That is CONSERVATIVE rather than sloppy and the reason is at
    the `del` in the query body: `global` costs strictly more shared memory, so the
    pinned answer can only UNDER-report a raw plan's residency, and an under-report
    makes the wave term charge MORE. A raw plan can be mis-ranked only toward
    caution. Making it a real query axis is a C++ dispatch change, not a docstring.

    The kernel's shared memory is entirely COMPILE-TIME
    (`consumer::consumer_smem_bytes<BT, BC, DV, D=2, prec, decay, global, compact>`), so
    `gram_cols`, `ring_slots` and `mma_precision` are not query axes for it: they are
    accepted to satisfy the :data:`OccupancyOracle` contract -- one signature, so a
    caller cannot silently hand the wrong oracle the wrong arity -- and ignored with
    that reason stated rather than dropped from the signature.
    """
    from rola.ops._ext import extension
    ext = extension()

    def query(BT: int, BC: int, d_v: int, global_norm: bool, decay: bool,
              mma_precision: str, compact: bool,
              gram_cols: int, ring_slots: int) -> int:
        # `global_norm` IS a template axis of the kernel but is PINNED TO `global`
        # inside `rola_consumer_occupancy`, and that is CONSERVATIVE rather than
        # sloppy: `global` costs strictly more shared memory than `raw` (one extra
        # `[BT]` float row, the `den` accumulator -- `consumer_smem_bytes`'s
        # `(GLOBAL ? 3 : 2) * BT * sizeof(float)` term), so the pinned answer can
        # only ever UNDER-report a raw plan's residency, never over-report it. An
        # under-report makes the wave term charge MORE, so a raw plan can be
        # mis-ranked only in the direction of caution. Recorded in `rewrite.md`
        # §2.36 as the one axis the oracle does not resolve.
        del global_norm, mma_precision, gram_cols, ring_slots
        return int(ext.rola_consumer_occupancy(decay, compact, BC, BT, d_v))

    return query


def resolve_occupancy(occupancy: OccupancyOracle | None, *,
                      BC: int, d_v: int) -> OccupancyOracle:
    """The CALL-SITE default: the injected oracle if there is one, the driver else.

    This is the only place that decides; `select_plan_axes` consults the resolved
    oracle and nothing else, which is what keeps selection a pure function of its
    inputs. An INJECTED oracle is the caller taking responsibility for the machine --
    a synthetic one in the planner gates, a real table in production -- so no
    instantiation guard applies to it: the caller answers for whatever it is asked.

    **THE INSTANTIATION GUARD IS NOW A REFUSAL, NOT A FALLBACK** (item 6b,
    2026-08-01). It used to divert an un-instantiated `(BC, d_v)` to the analytic
    mirror, so a geometry the kernel cannot launch still got a plan -- a plausible
    number for an impossible launch. The mirror is deleted (it modelled the deleted
    consumer and was measured inverting the bound it claimed), and the honest
    replacement is not a different fallback but none: a geometry outside
    `_SUPPORTED_BC` x `_SUPPORTED_DV` has no kernel, so it has no residency, so it
    has no plan. Callers that legitimately plan such a geometry -- CPU-only planner
    gates on toy topologies -- INJECT an oracle, which is what the injection
    interface is for and is stated above.
    """
    if occupancy is not None:
        return occupancy
    if BC not in _SUPPORTED_BC or d_v not in _SUPPORTED_DV:
        raise NotImplementedError(
            f"the consumer instantiates BC in {_SUPPORTED_BC} and d_v in "
            f"{_SUPPORTED_DV}; this plan asks for BC={BC}, d_v={d_v}, which no "
            "kernel implements, so the driver has no residency to report for it. "
            "Pass `occupancy=` a table if you are deliberately planning a geometry "
            "off the instantiation matrix (the CPU-only planner gates do); there is "
            "no analytic fallback any more, because the one that existed modelled a "
            "different kernel and was measured inverting the bound it claimed.")
    return consumer_occupancy()

_BITS_PER_WORD = 32

#: Placeholder relative cost of a write-live tile against a read-live tile in the
#: owner sort. §8 states reads are ~99.9% of arithmetic and that a write-hot,
#: read-cold "junk" owner contributes zero split-K traffic -- but its deposits and
#: epoch bookkeeping are not free, so the weight is small and nonzero rather than
#: zero. The value itself is NOT measured (§8's measurement basis does not price
#: schedule construction or deposits); it is a plan parameter so that a measured
#: value replaces it without a code change.
DEFAULT_WRITE_COST_WEIGHT = 0.125

#: MMA operand precision for the §5 readout -- a compile-time NUMERIC FORMAT axis
#: (findings D4.3), never a tolerance knob and never a density branch.
#:
#: - ``tf32x3``: three ``m16n8k8`` tf32 passes, near-fp32. What the exactness gate
#:   against the fp64 oracle runs: bf16 A/B carry ~1e-2 relative error per product
#:   and would swamp any structural disagreement.
#: - ``bf16``: one ``m16n8k16`` pass with fp32 accumulate. **THE PRODUCTION
#:   DEFAULT** since 2026-07-31, reported against its own honestly looser
#:   tolerance, and the arm on which ``GLA_MASS_DECAY_SPEC.md`` §12 rule 3's finite
#:   clamp is live.
#:
#: **The default is bf16 while the exactness gates pin tf32x3 EXPLICITLY, and that
#: split is deliberate.** `harvest.md` §30 measured bf16 at -9.5 % at scale, -7.3 %
#: mid and -6.6 % at the gram crossover, with 43 bf16 correctness gates green at
#: their own declared `BF16_RTOL = 1e-2` against the fp64 oracle (tolerance
#: untouched); ncu attributed the modest SIZE of that win to a tensor pipe that is
#: 97 % idle either way, not to the format. bf16 is therefore free at its own
#: tolerance and worth ~7-10 %, which is what a production default should be --
#: while a structural gate must keep the arm whose own error cannot swamp the
#: disagreement it exists to detect.
MMA_PRECISIONS = ("tf32x3", "bf16")

#: Per-SM shared-memory budget, in bytes, when no device can be asked -- the CPU-only
#: fallback for §6 selection. 100 KiB is `cudaDevAttrMaxSharedMemoryPerMultiprocessor`
#: on Ampere (SM 8.0 and 8.6 alike) and is DEVICE-QUERIED whenever a device exists
#: (`rola_sm_shared_bytes`). It is a **plan field**, not a hidden default, so a
#: plan built on a CPU carries the number it reasoned with and a reader can tell a
#: fallback-selected plan from a device-selected one (§10's ruling).
DEFAULT_SM_SHARED_BYTES = 100 * 1024

#: SM count, when no device can be asked -- the CPU-only fallback for §6's WAVE
#: term (`plan_waves`). 80 is the RTX 3080 Ti this repo's measurement basis was
#: taken on. Unlike `DEFAULT_SM_SHARED_BYTES` there is NO architectural value to
#: fall back to -- SM count is a per-SKU fact, not a per-architecture one -- so this
#: is a declared target rather than a safe default, and the number is carried on
#: `PlanAxes` for exactly that reason: a plan states the machine it reasoned for.
DEFAULT_SM_COUNT = 80

#: The candidate-tile word budget the kernel holds in SMEM per CTA
#: (`v3_consumer.cuh::kMaxWords`). Mirrored here because it is a term of the
#: footprint; the equality gate below pins it.
_K_MAX_WORDS = 128


# ---------------------------------------------------------------------------
# Owner geometry (§3 "Owner geometry")
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OwnerGeometry:
    """The unique ``BC``-aligned rectangle decomposition of the leaf space (§3).

    An aligned block is always exactly one rectangle::

        fixed digits (levels 0..j-1) x aligned sub-range of c digits (level j)
                                     x full span (levels j+1..D-1)

    ``first_spanned_level = j`` and ``slice_digits = c`` are *derived* from
    ``(widths, BC)``; they are not free parameters. Unaligned ranges would
    decompose into up to ``2D-1`` rectangles each with its own AND chain,
    row-scale and MMA, which is why §3 requires alignment.
    """

    widths: tuple[int, ...]
    BC: int
    first_spanned_level: int
    slice_digits: int

    @property
    def inner_extent(self) -> int:
        """``prod(widths[j+1:])`` -- the fully spanned levels below the boundary."""
        return prod(self.widths[self.first_spanned_level + 1:])

    @property
    def n_slices(self) -> int:
        """Number of aligned sub-ranges of the boundary level: ``b_j / c`` (§4)."""
        return self.widths[self.first_spanned_level] // self.slice_digits

    @property
    def fixed_levels(self) -> tuple[int, ...]:
        return tuple(range(self.first_spanned_level))

    @property
    def owners_total(self) -> int:
        return prod(self.widths) // self.BC

    def owner_digits(self, owner: int) -> tuple[tuple[int, ...], int]:
        """Decompose an owner id into ``(fixed digits, slice index)``.

        Inverse of the leaf-base identity ``owner * BC = sum_{l<j} d_l * suffix[l+1]
        + slice * c * suffix[j+1]``, which reduces exactly to
        ``owner = n_slices * mixed_radix(d_0..d_{j-1}) + slice``.
        """
        j = self.first_spanned_level
        slice_index = owner % self.n_slices
        rest = owner // self.n_slices
        digits = []
        for level in range(j):
            stride = prod(self.widths[level + 1:j])
            digits.append((rest // stride) % self.widths[level])
        return tuple(digits), slice_index

    def owner_of(self, fixed_digits: tuple[int, ...], slice_index: int) -> int:
        j = self.first_spanned_level
        rest = 0
        for level in range(j):
            rest += fixed_digits[level] * prod(self.widths[level + 1:j])
        return rest * self.n_slices + slice_index

    def leaf_base(self, owner: int) -> int:
        return owner * self.BC


def owner_geometry(widths: tuple[int, ...], BC: int) -> OwnerGeometry:
    """Derive the §3 owner rectangle for ``BC`` over an MSB-first mixed radix.

    Raises loudly (no fallback, no silent re-alignment) when ``BC`` does not
    produce an aligned rectangle. §6 makes ``BC`` a planner choice, so an
    inadmissible ``BC`` is a planner defect, not a runtime condition.
    """
    if BC < 1:
        raise ValueError(f"BC must be >= 1, got {BC}")
    N = prod(widths)
    if N % BC != 0:
        raise ValueError(f"BC={BC} must divide N={N} (owners partition the leaf space)")
    if BC > N:
        raise ValueError(f"BC={BC} exceeds the leaf capacity N={N}")
    D = len(widths)
    for j in range(D - 1, -1, -1):
        # The boundary level is the outermost-innermost level whose span already
        # reaches BC: prod(widths[j+1:]) < BC <= prod(widths[j:]). BC <= N makes
        # j = 0 the worst case, so the loop always returns.
        if prod(widths[j:]) < BC:
            continue
        inner = prod(widths[j + 1:])
        if BC % inner != 0:
            raise ValueError(
                f"BC={BC} is not a multiple of the spanned inner extent {inner} "
                f"(widths={widths}); it does not name a BC-aligned rectangle (spec §3)")
        c = BC // inner
        if c > widths[j] or widths[j] % c != 0:
            raise ValueError(
                f"BC={BC} implies {c} digits of level {j} (width {widths[j]}), which is not "
                f"an aligned sub-range (spec §3 requires c | width_j)")
        return OwnerGeometry(widths=tuple(widths), BC=BC, first_spanned_level=j, slice_digits=c)
    raise AssertionError(
        f"owner_geometry loop invariant violated (BC={BC}, widths={widths}); BC <= N was checked")


# ---------------------------------------------------------------------------
# Plan (§6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Plan:
    """Compile-time-selected kernel parameters (``ROLA_V3_SPEC.md`` §6).

    None of these is a kernel branch on a *distributional* property; every one is
    a validated structural invariant that the kernel may specialize on (§3 branch
    rule). ``BT``, ``BC``, ``(a, b)`` and ``D`` are the template parameters of the
    single topology-agnostic kernel family.
    """

    topology: Topology
    normalization: str
    BT: int
    BC: int
    warp_tile: tuple[int, int]
    decay: bool = False
    #: §3 step 2's support gather, as a COMPILE-TIME axis (spec §6, ratified rev
    #: `2026-07-29.5` Q1). `True` compiles the gather in; `False` compiles it out
    #: and the kernel takes the `K == BC` degeneracy directly -- the same code with
    #: `c_idx[k] == k` folded at compile time, so the two are BIT-IDENTICAL wherever
    #: the tile unions really do saturate, and the second is strictly cheaper (no
    #: probe, no ballot, two fewer barriers, and the gather's SMEM unallocated).
    #: `select_compact_and_BT` chooses it from the realized `TileUnionStats`; the
    #: default is `True` because it is the general path, and a hand-built plan that
    #: does not price its supports must not silently assume saturation.
    compact: bool = True
    mma_precision: str = "bf16"
    amp_dtype: torch.dtype = torch.float32
    write_cost_weight: float = DEFAULT_WRITE_COST_WEIGHT
    #: The per-SM shared-memory budget §6 reasoned about when this plan was selected,
    #: RECORDED rather than re-read at launch (§10's ruling). Device-queried when a
    #: device exists, `DEFAULT_SM_SHARED_BYTES` otherwise -- and carried either way,
    #: so a plan states the machine it was reasoned for instead of silently depending
    #: on the machine it is replayed on.
    #:
    #: **PROVENANCE ONLY: NO SELECTION RULE READS IT** (coordinator ruling,
    #: 2026-08-01, `rewrite.md` §2.37). It fed `mirror_occupancy` and nothing else,
    #: and the mirror died with the consumer it modelled (item 6b). It is RETAINED
    #: on the same contract `sm_count` has -- a plan states the machine it was
    #: reasoned for, so a replayed plan is auditable -- with the difference stated
    #: rather than hidden: `sm_count` IS read (by `plan_waves`), this is not.
    #: Deleting it cascades into `layers/rola.py`, the `rola_sm_shared_bytes`
    #: binding and `DEFAULT_SM_SHARED_BYTES`; it dies at the standalone-package
    #: extraction, not before.
    sm_shared_bytes: int = DEFAULT_SM_SHARED_BYTES
    #: Staging RING depth (§6 residency): slots of the `[BT, gram_cols]` staged rows.
    #: `>= 2` pipelines the next candidate tile's gathered columns with `cp.async`;
    #: `1` DECLINES the ring and stages synchronously inside the visit.
    #:
    #: STATED, not selected (M-1 RULING A, 2026-08-01): every currency the planner
    #: has -- the work model, the occupancy oracle, and the kernel's own measured
    #: instruction count -- is INVARIANT in this field, so `select_plan_axes` sets
    #: the deepest built depth and says why instead of running a `min` over equal
    #: keys. The axis survives because §2.3's item buffers may build a ring that is
    #: real, at which point its price is a NEW measurement on THAT kernel. Carried at
    #: RUNTIME -- the launch's dynamic SMEM size is already a runtime argument -- so
    #: it costs no template instantiation.
    ring_slots: int = _RING_DEPTHS[0]
    #: The CTAs/SM §6 reasoned about when this plan was selected, RECORDED next to
    #: the decision it drove -- the same contract `sm_shared_bytes` has, and for the
    #: same reason (coordinator ruling 2026-07-31): a machine fact enters the plan as
    #: an INPUT and is readable off the plan afterwards, so a plan states the
    #: residency it was reasoned for. `0` = hand-stamped, no occupancy consulted.
    occupancy_blocks_per_sm: int = 0

    def __post_init__(self) -> None:
        validate_normalization(self.normalization)
        if self.mma_precision not in MMA_PRECISIONS:
            raise ValueError(
                f"mma_precision must be one of {MMA_PRECISIONS}, got {self.mma_precision!r}")
        if self.ring_slots not in _RING_DEPTHS:
            raise ValueError(
                f"ring_slots must be one of {_RING_DEPTHS} (deepest first; 1 declines the "
                f"ring), got {self.ring_slots}")
        if self.BT not in _SUPPORTED_BT:
            raise ValueError(
                f"BT must be a compile-time constant in {_SUPPORTED_BT} (spec §6): below 16 "
                f"wastes MMA rows, intra cost is quadratic in BT; got {self.BT}")
        owner_geometry(self.topology.widths, self.BC)  # validates the BC alignment, loudly
        a, b = self.warp_tile
        if a < 1 or b < 1:
            raise ValueError(f"warp tile (a, b) must be positive, got {self.warp_tile}")
        warps = a * b
        if warps not in (1, 2, 4, 8):
            raise ValueError(f"a*b must be a supported CTA warp count in (1,2,4,8), got {warps}")
        if self.BC % a != 0:
            raise ValueError(f"a={a} must divide BC={self.BC} (leaf-group partition)")
        if self.topology.d_v % b != 0:
            raise ValueError(f"b={b} must divide d_v={self.topology.d_v} (channel-group partition)")

    @property
    def geometry(self) -> OwnerGeometry:
        return owner_geometry(self.topology.widths, self.BC)

    @property
    def warps(self) -> int:
        return self.warp_tile[0] * self.warp_tile[1]

    @property
    def state_registers_per_warp(self) -> int:
        """``BC * d_v / W`` -- invariant in ``(a, b)`` by construction (§5)."""
        return self.BC * self.topology.d_v // self.warps

    def amplitude_pitch(self, width: int) -> int:
        """Pad a level's width so one tile's rows are a whole number of 512 B lines.

        The stream is stored token-major and contiguous per tile (§5 "Staging"), so
        a tile block is ``BT * pitch`` elements; requiring that to be a multiple of
        ``CP_ASYNC_ROW_BYTES`` is what makes the innermost rows ``cp.async``-shaped.
        Padding columns hold structural zeros, which are inert under §12 rule 4's
        zero-route annihilation and never appear in a bitmap (built over ``width``
        only), so padding cannot manufacture a live leaf.
        """
        itemsize = torch.empty((), dtype=self.amp_dtype).element_size()
        elems_per_row = CP_ASYNC_ROW_BYTES // itemsize
        step = elems_per_row // gcd(self.BT, elems_per_row)
        return ((width + step - 1) // step) * step


# ---------------------------------------------------------------------------
# Span ranking (§6.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LevelSupportStats:
    """Per-level measured routing statistics consumed by the §6.1 ranking.

    ``s_r``/``s_w``: mean active read/write leaf counts *given* the level is live
    (§1's ``A_r = prod_l s_r[l]``). ``rho_r``/``rho_w``: read-live and write-live
    **token rates** -- the fraction of tokens for which the level is read (resp.
    written) at all. §6.1 is explicit that these are different quantities and that
    conflating them is the read-density-only degeneracy.
    """

    s_r: float
    s_w: float
    rho_r: float
    rho_w: float


def span_score(stats: LevelSupportStats, width: int) -> float:
    """§6.1's volume-weighted combined density: ``(s_r*rho_r + s_w*rho_w) / b_l``."""
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    return (stats.s_r * stats.rho_r + stats.s_w * stats.rho_w) / width


def spanned_moduli(topology: Topology, BC: int) -> tuple[int, ...]:
    """The owner's per-level moduli ``m_l`` for ``l >= j`` (§3's rectangle).

    ``m_j = slice_digits``, ``m_l = b_l`` for ``l > j``, and ``prod_{l>=j} m_l ==
    BC`` by construction. The FIXED levels are absent because the owner holds one
    digit of each and that factor hoists into the §3 step 3 row scale.
    """
    geometry = owner_geometry(topology.widths, BC)
    j = geometry.first_spanned_level
    return (geometry.slice_digits,) + tuple(topology.widths[j + 1:])


# ---------------------------------------------------------------------------
# §6 cost model -- ONE formula prices the intra lowering, the compact axis and BT
# ---------------------------------------------------------------------------

# `MMA_ISSUE_WORK` IS DELETED (2026-07-30, adversarial review §4.2), and the
# readout term it scaled with it. It priced the readout's MMA issues per active
# leaf column -- `d_v * w / (128 * k_unit)`, i.e. 0.375 work-units at `w = 6`,
# `d_v = 64`, `k_unit = 8` -- against an in-tile term of 136-264 work-units per
# token and a gather term of 13000 per tile. NO DECISION ANYWHERE IN THIS MODEL
# DEPENDS ON IT: it is absent from the intra crossover entirely, and it enters the
# compact comparison only as `(K - BC) * d_v * w / (128 * k_unit)`, which at the
# deepest measured deficit (`K = 7.2/16`) would need `w > 2.4e4` -- 4000x its own
# derivation -- to outweigh `gather_tile_work`. A constant that cannot influence a
# decision at any value within three orders of magnitude of its derivation is not
# a plan parameter; carrying it as one asserted a tunability the calibration gate
# could not gate (the review mutated it 60x and nothing moved). The work it named
# is real and is now un-modelled EXPLICITLY, which is checkable, rather than
# modelled by a number no gate can bind. `k_unit`, whose only use was that term,
# goes with it: `mma_precision` is a NUMERIC FORMAT axis of the kernel and is no
# longer an input to the cost model.
#
# The two constants that remain each have a value at which the calibration table
# fails -- `test_the_cost_model_reproduces_its_calibration` asserts exactly that,
# per constant, so "binding" is a gated property rather than a claim.

#: Per-CANDIDATE-TILE fixed work that does not scale with ``K``: the epoch
#: bookkeeping, the row scales, the amplitude staging, the block barriers, the
#: write-conditioned publish. Same units. This is the term that makes a LARGER
#: ``BT`` cheaper, and it is the only one that does.
#:
#: **RE-MEASURED 2026-08-01 (M-1), 120.0 -> 3510.35, AND IT ABSORBS THE DELETED
#: `STAGE_COLUMN_WORK`.** The old value was never isolated: it was the residue of an
#: old-consumer calibration, and the work it names was really being carried by the
#: staging term this repository has now retired (see the deletion note below).
#:
#: Instrument: `ncu --clock-control base`, `smsp__inst_executed.sum`, ONE profiled
#: launch per arm (`--launch-skip 1 --launch-count 1`), RTX 3080 Ti, 2026-08-01, on
#: the M-1 isolating cells in `benchmarks/bench/cells.py` (all dense/dense, so the
#: unions saturate and NO routing statistic moves between them).
#:
#: **TWO INDEPENDENT ISOLATING SOLVES, EACH OF THEM TWO-STAGE.** A per-candidate-
#: tile constant cannot be read off a single cell, because a launch's instructions
#: are `n_cohorts * (F + n_tiles * P)` -- a per-COHORT part `F` (CTA prologue and
#: epilogue, the owner's state load and store) and a per-TILE part `P`. So each
#: geometry is measured at three (or two) values of `n_tiles = T / BT` at fixed
#: everything else, `F` and `P` are separated, and only then is `P` split across
#: `BC` into a constant and a per-dense-k-quantum slope:
#:
#:   `iso-tile-w16x4x64-BC64-T{128,256,512}` -> P = 8771.0, F = 9041.0
#:   `iso-tile-w16x4x64-BC16-T{128,256,512}` -> P = 4823.0, F = 4535.0
#:       => slope 1316.0 / dense k-quantum, INTERCEPT **3507.0**
#:   `topo-D2-w64x64-BC64` + `iso-tile-w64x64-BC64-T256` -> P = 8135.0, F = 8434.0
#:   `topo-D2-w64x16-BC16` + `iso-tile-w64x16-BC16-T256` -> P = 4669.0, F = 4036.0
#:       => slope 1155.3 / dense k-quantum, INTERCEPT **3513.7**
#:
#: The two intercepts are 0.19 % apart across a different `D`, a different width
#: multiset and a 4x different owner count, and the three-point `T` sweep solves `F`
#: to the same value from both of its consecutive pairs (9041.0 / 9041.0), so the
#: `F + n_tiles * P` form is CHECKED rather than assumed. The landed value is the
#: mean of the two intercepts and
#: `test_the_cost_model_reproduces_its_calibration` re-runs every argmin across the
#: whole measured range `[3507.0, 3513.7]`.
#:
#: **ARM-INVARIANT, MEASURED.** On `iso-tile-w16x4x64-BC64-T{128,256}` the per-tile
#: solve is 8771.0 dense / 11125.9 compact / 13345.0 decay; the two deltas are
#: 2355.0 (against `GATHER_CANDIDATE_*`'s prediction of 2278.25, +3.4 %) and 4574.0
#: (against `DECAY_KSTEP_WORK * 4 = 4132.6`, +10.7 %), so this constant is the part
#: those two arms ADD TO, and both of them re-derive on a geometry they were not
#: measured on.
#:
#: **THE PER-COHORT REMAINDER `F` IS MEASURED AND DELIBERATELY NOT MODELLED**, which
#: is the `MMA_ISSUE_WORK` rule applied to a term that is real: `F` = 7999 (D=1),
#: 8434 (D=2), 9041 (D=3) at `BC = 64` and 4036 / 4535 at `BC = 16`, essentially
#: arm-invariant (dense 9041, compact 8990, decay 9506). It is a per-COHORT cost, so
#: per token it is `F / T` -- exactly `BT`-INVARIANT and arm-invariant, hence unable
#: to move the joint `(BT, compact)` argmin at any value -- and it is 12 % of the
#: T=128 gate cells but 0.9 % at `atscale-L2048` and below 0.1 % at the parity cell,
#: i.e. it vanishes in exactly the regime the planner is for. Carrying it would add
#: a constant no gate can bind and an `n_pairs / n_cohorts` input the currency does
#: not otherwise need. It is recorded here instead, which is checkable.
#:
#: FALSIFIABLE: an isolating `T` sweep whose intercept falls outside
#: `[3507.0, 3513.7]`, or a geometry on which `F + n_tiles * P` does not solve
#: consistently from two disjoint pairs.
TILE_FIXED_WORK = (3507.0 + 3513.7) / 2.0

#: Per-candidate-tile cost of §3 step 2's gather when it is COMPILED IN: the
#: per-(level,digit) probe over the staged rows, the per-leaf AND chain, the
#: ballot/popc compaction, two extra block barriers, and the index loads the
#: gathered operands pay. Same units. This is what the compact axis buys or pays.
#:
#: **MEASURED, and it REPLACES `GATHER_TILE_WORK = 13000` rather than rescaling it**
#: (§2.20, 2026-08-01). That constant priced the OLD consumer's gather, whose
#: dominant components -- the staging-ring magnitude probe, the dense-panel-to-
#: compacted COPY, the third block barrier -- do not exist in this kernel, and whose
#: own calibration could only bound it from BELOW because the 48-cell A/B never
#: observed break-even. Here break-even is not merely observable, it is crossed by
#: 1.75x in instructions at the flagship.
#:
#: **AND IT IS NOT ONE NUMBER: it scales with the DENSE extent, measured by an
#: isolating cell PAIR rather than assumed.** The ballot compaction probes every one
#: of the owner's `BC` leaf slots whatever the union realizes, so its cost tracks
#: `ceil(BC / 16)` -- one ballot round at `BC = 16`, two at `BC = 64` -- and pricing
#: it as a constant was wrong in the direction that matters: it made compaction
#: unreachable at `BC = 16`, where the dense extent it has to beat is a single
#: k-step.
#:
#: Instrument: `ncu --clock-control base`, `smsp__inst_executed.sum`, ONE profiled
#: launch per arm (2026-08-01), on `topo-D2-w64x64-BC64` and `topo-D2-w64x16-BC16`
#: -- the same topology, the same support, and unions that SATURATE on both
#: (`K = KI = BC` on every candidate pair, `--mode gatherstats`), so compaction
#: changes no extent at either and the whole delta IS the gather. Over 1024
#: candidate pairs each:
#:
#:   `BC = 64`: 11.743 M against  9.410 M  ->  2278.25 per candidate tile (2 quanta)
#:   `BC = 16`:  6.017 M against  5.298 M  ->   702.88 per candidate tile (1 quantum)
#:
#: A two-point solve over `ceil(BC/16)` gives a per-quantum slope of **525.1** and a
#: per-tile intercept of **177.8** -- both positive, unlike every affine fit the
#: decay constant admitted, and the intercept is a real fixed cost (the two extra
#: block barriers and the epoch bookkeeping the gather adds regardless of extent).
GATHER_CANDIDATE_FIXED_WORK = 177.75

#: The per-`ceil(BC/16)`-quantum half of the gather, from the same isolating pair.
GATHER_CANDIDATE_QUANTUM_WORK = 525.125

#: Per-K-STEP cost of the readout, the fold and the panel builds -- the three
#: extents compaction actually shortens. Same units, per candidate tile, at the
#: `BT = 16` the constants were measured at.
#:
#: Instrument: the same counters at `atscale-L2048`, where compaction DOES shorten
#: the extents: 1035.0 M against 1812.1 M over 208836 candidate pairs, a saving of
#: 3721 per tile across the 3.39 k-steps compaction removes there
#: (`4 - mean_ksteps_candidate`) = **1755** per k-step.
#:
#: **IT IS PRICED PER QUANTUM, NEVER PER LEAF.** `ceil(K/16)` is what the kernel
#: issues: a tile with `K = 1` pays a whole k-step and a tile with `K = 0` pays
#: none. At `atscale-L2048` 39 % of candidate pairs gather nothing, which is why
#: `mean_K_candidate/BC = 0.9724` and `mean_ksteps_candidate/(BC/16) = 0.6087` are
#: different numbers and only the second one prices the kernel.
READOUT_FOLD_KSTEP_WORK = 1755.0

#: Per-K-STEP cost of the intra Gram's own contraction, same units and same `BT`.
#:
#: Instrument: `wk1-D2-w64x64-BC64` against `topo-D2-w64x64-BC64` -- the same
#: widths, the same `BC`, the same saturated READ union, and a write of `k = 1` per
#: token that cuts `KI` from 64.00 to 0.33 while leaving `K = BC`. The delta is
#: therefore the intra extent and nothing else: **144** per k-step, a 12x demotion
#: of the term that used to dominate the `BT` decision.
INTRA_GRAM_KSTEP_WORK = 144.0

#: Per-K-STEP cost of the LeafMassDecay arm on top of the same tile: the clock
#: prefix, the two `expf` survival panels, the two coefficient folds and the `s0`
#: entry/exit sweeps, plus the `gram`/`realized_leaf` -> `panel` lowering the arm
#: forces. Same units, same `BT = 16`.
#:
#: **PINNED BY TWO ISOLATING CELL PAIRS, AND IT HAS NO PER-TILE FIXED COMPONENT.**
#: `topo-D2-w64x64-BC64` and `topo-D2-w64x16-BC16` are the same topology, the same
#: support, the same `BT`, and saturate identically -- so they differ ONLY in the
#: k-step count (4 against 1) and a two-point solve separates a per-tile constant
#: from a per-k-step one. Counters (`ncu --clock-control base`,
#: `smsp__inst_executed.sum`, one profiled launch per arm, 2026-08-01, over 1024
#: candidate pairs each):
#:
#:   dense arm:    4372.9 and 1054.1 per tile ->  slope **1106.3**, intercept  -52.3
#:   compact arm:  3948.8 and 1069.8 per tile ->  slope   **959.7**, intercept +110.1
#:
#: Both intercepts are within 10 % of the smaller delta of ZERO, so decay carries no
#: per-candidate-tile fixed cost: it rides the `ceil(K/16)` quantum exactly like the
#: readout. The landed value is the mean of the two slopes, and
#: `test_the_cost_model_reproduces_its_calibration` re-runs every ordering across
#: the WHOLE measured range `[959.7, 1106.3]`, so the residual spread is a gated
#: property rather than a rounding choice.
#:
#: **THE `BT * BC` ALTERNATIVE STRUCTURE IS REFUTED BY 15.8x**, and by a comparison
#: with no free parameter: `atscale-L2048`'s two compact arms have IDENTICAL `BT`
#: and `BC`, so a `BT * BC`-scaled decay term predicts the same delta on both.
#: Measured 4345.5 per tile dense against 275.8 compact. Only an extent-scaled term
#: can move there at all.
#:
#: **THE ONE REGIME IT OVERPRICES, RECORDED RATHER THAN FITTED.** At
#: `atscale-L2048` COMPACT the measured decay is 275.8 per tile over 0.6087
#: k-steps = 453 per k-step, 2.3x BELOW this constant, and no non-negative affine
#: function of `ceil(K/16)` fits that point and the four saturated-extent points
#: together (it would need a negative per-tile term). Fitting a second, compact-only
#: decay constant is FORBIDDEN by the standing rule -- its value could only be set
#: by the one cell it would then be validated on, which is exactly the reasoning
#: §2.32 rejected for a spill-byte term. The error's DIRECTION is stated instead and
#: is conservative for the decision it feeds: the model charges the compact decay
#: arm ~350 work-units per tile MORE than it costs, so wherever the joint argmin
#: still selects compaction under decay it does so against a handicap.
#:
#: FALSIFIABLE: a saturated-extent cell whose per-k-step decay delta falls outside
#: `[959.7, 1106.3]`, or any cell whose delta needs a per-tile constant term.
DECAY_KSTEP_WORK = (1106.3 + 959.7) / 2.0

# ---------------------------------------------------------------------------
# `STAGE_COLUMN_WORK` IS DELETED (M-1, RULING A, 2026-08-01) -- and it is deleted
# ON A MEASUREMENT, not on the source read that suspected it.
# ---------------------------------------------------------------------------
# It priced `2 * gram_cols * W / ring_slots` per candidate tile at `W = 85`, named
# for a staging ring, bracketed `(40.5, 143.3)` by three cells, and it was known to
# be load-bearing: setting it to 0 flipped FOUR of the eight measured argmins. The
# source read had already established that the surviving consumer has no staging
# ring (`p.ring_slots`, `p.gram_cols` and `p.level_gram_offset` are written by the
# host and read by ZERO sites in `consumer_kernel.cuh`), but a name being wrong is
# not permission to delete a term that carries a decision, so the disposition
# waited for the isolating cells.
#
# **WHY NO CALIBRATION CELL COULD SEE THE VARIABLE.** `gram_cols = sum(m_l)` over
# the spanned levels, and `sum(m_l) <= prod(m_l) = BC` with equality exactly when
# ONE level is spanned. All four calibration cells span one level, so `gram_cols`
# was IDENTICALLY `BC` on all eight argmin rows and all twelve absolute-total rows:
# the term was algebraically indistinguishable from `16 * W` per dense k-quantum,
# and its `gram_cols` shape -- the only thing making it a STAGING term rather than
# an extent term -- had never been observed at all.
#
# **THE MEASUREMENT, and it is an EXACT EQUALITY rather than a small residual.**
# Four cells at `BC = 64`, `D = 3`, `N = 4096`, the same width MULTISET permuted so
# that only the owner rectangle's level decomposition moves -- same grid, same 1024
# candidate tiles, same saturated unions (`ncu --clock-control base`,
# `smsp__inst_executed.sum`, one profiled launch each, RTX 3080 Ti, 2026-08-01):
#
#     `iso-gc-w32x16x8-BC64`         gram_cols = 16   10,062,336
#     `iso-gc-w64x16x4-BC64`         gram_cols = 20   10,062,336
#     `iso-gc-w64x32x2-BC64`         gram_cols = 34   10,062,336
#     `iso-tile-w16x4x64-BC64-T128`  gram_cols = 64   10,138,752
#     `iso-gc-w4x16x64-BC64`         gram_cols = 64   10,138,752
#
# The first three are BYTE-IDENTICAL across a 2.1x range of `gram_cols`. The
# per-staged-column price is therefore **exactly 0.000**, not 85: at
# `gram_cols = 34` the term was charging 2890 work-units per tile of a mechanism
# that retires no instruction. What the counts DO separate is the number of FIXED
# levels `j` (the two `gram_cols = 64` cells have `j = 2`, the other three
# `j = 1`): 76,416 instructions per launch = **74.6 per candidate tile per fixed
# level**, reproduced at the other `BC` by `iso-tile-w16x4x64-BC16-T128` (j=2)
# against `iso-gc-w64x16x4-BC16` (j=1) as 73.6 per tile -- the SAME delta at
# `d(gram_cols) = 8` that the `BC = 64` pair shows at `d(gram_cols) = 44`, which is
# the second, independent refutation of the column scaling.
#
# **WHERE THE WORK WENT.** The magnitude the term carried was real; its variable was
# not. It is now inside `TILE_FIXED_WORK`, re-measured by isolating cells from 120
# to 3510.35 -- a per-candidate-tile constant, `gram_cols`-invariant, which is the
# shape the eight argmins actually need (see that constant's own docstring). The
# 74.6-per-fixed-level geometry term is MEASURED AND NOT MODELLED: it is 2 % of the
# constant it would ride on and decision-inert, and this repository does not carry
# constants no gate can bind (`MMA_ISSUE_WORK`'s deletion, 2026-07-30).
#
# **THE RING AXIS IS NOW UNPRICED, AND THAT IS STATED RATHER THAN HIDDEN.**
# `ring_slots` was priced by this term and by nothing else; the occupancy oracle is
# ring-INVARIANT (the consumer's shared memory is entirely compile-time) and the
# kernel reads the field at zero sites, so after this deletion the depth is inert in
# every currency the planner has. `select_plan_axes` therefore STATES the depth
# instead of arbitrating it. If §2.3's item buffers ever build a real ring, its
# price is a new measurement on the kernel that has one -- not this constant
# revived.

#: Where the constants above come from, carried next to them so a number cannot be
#: quoted without its provenance (spec §8's standing warning).
#:
#: **RE-BASED 2026-08-01 ONTO THE NEW CONSUMER, and the old basis is retired rather
#: than amended.** The 48-cell `compaction2.md` §B2 grid measured the OLD consumer,
#: whose gather is a different algorithm (a scalar walk's trip count, a dense-panel
#: copy, a staging-ring probe); compaction LOST at every one of its cells, which is
#: why `GATHER_TILE_WORK` could only ever be a lower bound. Re-using it to bind
#: constants measured on this kernel would have been the two-kernel conflict §2.28
#: stopped on.
COST_MODEL_PROVENANCE = (
    "Re-based 2026-08-01 on the NEW consumer. RTX 3080 Ti, `ncu --clock-control "
    "base`, `smsp__inst_executed.sum`, ONE profiled launch per arm "
    "(`--launch-skip 1 --launch-count 1`), every run `flock`-serialized. Cells are "
    "the frozen registry in `benchmarks/ops/benchmark_rola_v3_new_consumer.py`, and "
    "each constant has an ISOLATING cell or cell pair rather than a fit: "
    "the GATHER_CANDIDATE_* pair from `topo-D2-w64x64-BC64` against "
    "`topo-D2-w64x16-BC16`, whose unions saturate so compaction changes no extent "
    "at either and only the dense quantum differs; "
    "READOUT_FOLD_KSTEP_WORK from `atscale-L2048`, "
    "where they do; INTRA_GRAM_KSTEP_WORK from `wk1-D2-w64x64-BC64` against "
    "`topo-D2-w64x64-BC64`, which differ only in KI; DECAY_KSTEP_WORK from the "
    "`BC=64`/`BC=16` pair of one topology at both compact arms, which differ only "
    "in the k-step count; and TILE_FIXED_WORK (M-1, 2026-08-01) from TWO "
    "independent isolating solves over the `iso-tile-*` cells, each separating a "
    "per-COHORT cost from a per-TILE one with a three-point `T` sweep before "
    "splitting the per-tile part across `BC`. The `iso-gc-*` cells belong to the "
    "same batch and are the one that removed a term: they hold `BC`, `D`, `N` and "
    "the grid fixed while permuting the owner rectangle, and measure the price of a "
    "staged Gram column as EXACTLY zero. "
    "KNOWN AND MEASURED, NOT MODELLED, so the next reader does not rediscover it: "
    "(1) a per-COHORT remainder of 4036-9506 instructions (CTA prologue/epilogue, "
    "state load/store) -- `BT`- and arm-invariant per token, hence unable to move "
    "the argmin, and below 1 % of the at-scale cells; (2) 74.6 per candidate tile "
    "per FIXED level, the only geometry dependence that survives measurement; "
    "(3) the dense k-quantum is OVER-priced by this model at 1899 (readout+intra) "
    "against 1033 (D=1) / 1155 (D=2) / 1316 (D=3) measured, which is now the "
    "LEADING absolute-total residual and grows ~141 per level per quantum -- an "
    "isolating re-measurement of READOUT_FOLD_KSTEP_WORK, queued, never a re-fit "
    "against the rows it must reproduce. ONE machine, ONE toolchain: the constants are plan "
    "parameters, not machine truths, and spec §8's measurement basis replaces them "
    "without a code change. WHAT BINDS THEM is "
    "`test_the_cost_model_reproduces_its_calibration`, which asserts the JOINT "
    "`(BT, compact)` ARGMIN -- the decision `select_plan_axes` actually makes -- "
    "against the 8 measured argmins of the §2.32 matrix, plus the 4 measured "
    "compact dispositions and the dense-toll signs. "
    "KNOWN LIMITATION, PROVED RATHER THAN SUSPECTED: on a SATURATED-DENSE cell "
    "(`K == BC` on every candidate pair) the per-slice `BT` preference at "
    "`decay + compact` INVERTS against this model, and no value of any constant can "
    "reach it. On such a cell compaction changes no extent, so adding it is exactly "
    "one positive constant G on both `BT` arms; the modelled ratio "
    "r(G) = (W32 + G) / (2 (W16 + G)) has dr/dG = (W16 - W32) / (2 (W16 + G)^2) < 0 "
    "whenever W32 > W16, i.e. whenever the model prefers BT=32 at all -- so "
    "compaction must DEEPEN a BT=32 preference, never invert it. The measurement "
    "inverts it, with BT=32 retiring 14 % FEWER instructions in 5.7-27.2 % MORE "
    "time: that is a STALL, and a work model does not model stalls. The row is "
    "never a decision (the joint argmin never selects a compacted arm on a "
    "saturated-dense cell) and "
    "`test_a_saturated_dense_cell_never_argmins_onto_a_compacted_arm` GUARDS that "
    "unreachability, so if a future axis ever makes the row reachable the guard "
    "fails loudly instead of the blind spot silently becoming load-bearing. A "
    "spill-byte term is FORBIDDEN as the fix: its constant could only be set by the "
    "two cells it would then have to reproduce."
)


def plan_cost_per_token(topology: Topology, BC: int, stats: TileUnionStats, *,
                        compact: bool, decay: bool = False,
                        tile_fixed_work: float = TILE_FIXED_WORK,
                        gather_candidate_fixed_work: float = GATHER_CANDIDATE_FIXED_WORK,
                        gather_candidate_quantum_work: float = GATHER_CANDIDATE_QUANTUM_WORK,
                        readout_fold_kstep_work: float = READOUT_FOLD_KSTEP_WORK,
                        intra_gram_kstep_work: float = INTRA_GRAM_KSTEP_WORK,
                        decay_kstep_work: float = DECAY_KSTEP_WORK) -> float:
    """§6's crossover pricing, in work-units per token, over the QUANTA the kernel
    issues rather than the leafs it names.

    Per CANDIDATE tile, and then divided by ``BT`` so the result is comparable
    across ``BT``::

        W_tile = tile_fixed_work
               + gather_candidate_fixed_work                      (if compact)
               + gather_candidate_quantum_work * ceil(BC / 16)     (if compact)
               + readout_fold_kstep_work * ksteps  * (BT / 16)
               + intra_gram_kstep_work   * kisteps * (BT * (BT + 1) / 272)
               + decay_kstep_work        * ksteps  * (BT / 16)    (if decay)

        cost_per_token = candidate_frac * W_tile / BT

    where ``candidate_frac`` is the realized share of ``(owner, tile)`` pairs the §4
    bitmap admits -- see the derivation at the return statement, which shows it is
    the DEFINITION of "per token" rather than an added term, carries no constant,
    and is identically ``1.0`` on every saturated-dense cell.

    with, on the COMPACT arm, ``ksteps = stats.mean_ksteps_candidate`` and
    ``kisteps`` the intra extent's own quantum; on the DENSE arm both degenerate to
    the compiled-out extent ``ceil(BC / 16)``, because the compiled-out kernel walks
    the full ``BC`` whatever the union realizes.

    **WHY QUANTA AND NOT ``mean_K``.** The kernel issues ``ceil(K / 16)`` MMA
    k-steps, so a tile with ``K = 1`` costs a whole one and a tile with ``K = 0``
    costs none. At ``atscale-L2048`` the two readings are ``0.9724`` of ``BC``
    against ``0.6087`` of ``BC/16``: 39 % of candidate pairs gather nothing and skip
    every GEMM, a saving ``mean_K`` structurally cannot express and the reason
    ``TileUnionStats`` accumulates the ceilings per pair.

    **WHAT REPLACED WHAT, AND WHY IT IS A REPLACEMENT.** The pre-2026-08-01 form
    priced a ``K_eff * (D - j)`` panel assembly and a ``BT * I_eff`` in-tile term
    against ``TILE_FIXED_WORK`` and ``GATHER_TILE_WORK = 13000``. Those two
    constants were an OLD-CONSUMER calibration in which compaction lost at every
    measured cell, so ``13000`` was documented as a lower bound and made compaction
    unreachable at any density the routing can produce -- a measured ~25 % of
    wall-clock left on the floor at the flagship, worsening with ``N``. The three
    replacement constants are each measured on THIS kernel at an isolating cell
    (see their own docstrings), and the panel assembly is inside
    ``readout_fold_kstep_work`` because that is what the isolating delta contains.

    **THE TWO ``BT`` SCALINGS ARE STRUCTURAL, and they are now GATED rather than
    posited.** The readout, the fold and the panel builds are ``[BT x K] x [K x DV]``
    and linear in ``BT``; the intra Gram is causal, ``BT(BT+1)/2`` entries. Both are
    normalized to the ``BT = 16`` cell the constants were measured at
    (``16 * 17 = 272``). §2.28 could only posit them; the §2.32 matrix measured the
    ``BT`` axis at both values on both ``BC`` and the calibration gate now binds the
    joint ``(BT, compact)`` argmin against it.

    **THE DECAY TERM RIDES THE SAME QUANTUM**, which is measured rather than assumed:
    two isolating cell pairs differing only in k-step count give slopes 1106.3 and
    959.7 with intercepts within 10 % of zero, and the ``BT * BC`` alternative is
    refuted by 15.8x at ``atscale-L2048`` (identical ``BT`` and ``BC`` on both
    compact arms, 15.8x different deltas). Its one overpricing regime is recorded on
    ``DECAY_KSTEP_WORK`` itself.

    **WHAT IS STILL NOT MODELLED, EXPLICITLY.** Residency and the WAVE REGIME are
    not here at all -- they are physical facts and ``select_plan_axes`` reads them
    from the occupancy oracle, per §6.5.

    **THE `intra_lowering` AXIS IS GONE** (item 6b, 2026-08-01), and with it the
    branch that priced ``gram``/``panel`` on the spanned moduli instead of the leaf
    list. The surviving consumer includes ``intra_panel.cuh`` unconditionally and
    does ONE intra lowering, so there is no second extent to price: ``kisteps`` is
    the realized intra quantum on the compact arm and the compiled-out ``ceil(BC/16)``
    on the dense one, exactly as ``ksteps`` is.
    """
    if stats.BC != BC:
        raise ValueError(f"stats were computed at BC={stats.BC}, priced at BC={BC}")
    # `topology` enters this function as a PRECONDITION and no longer as a price
    # (M-1, 2026-08-01): the owner rectangle must be admissible at this `BC` --
    # `owner_geometry` raises otherwise, and pricing an unaligned rectangle is
    # pricing a plan that cannot be built -- but its SHAPE costs nothing, which the
    # `iso-gc-*` cells measure as an EXACT equality across a 2.1x range of
    # `gram_cols`. Keeping the check is what makes that a stated finding rather than
    # a silently dropped argument.
    owner_geometry(topology.widths, BC)
    BT = stats.BT
    quantize = lambda n: float(-(-int(n) // MMA_K_QUANTUM))  # noqa: E731

    # The COMPILED-OUT kernel walks the full `BC` whatever the union realizes, so
    # the dense arm's extents are the template's, not the routing's. That is the
    # same "same formula at two values of the extent" that keeps the compact
    # decision consistent with the `BT` one instead of needing a second model.
    dense_steps = quantize(BC)
    ksteps = stats.mean_ksteps_candidate if compact else dense_steps
    kisteps = stats.mean_kisteps_candidate if compact else dense_steps

    w_tile = tile_fixed_work
    if compact:
        # The ballot probes all `BC` leaf slots whatever the union realizes, so the
        # gather is priced on the DENSE quantum -- never on `ksteps`, which is the
        # extent it PRODUCES rather than the one it scans.
        w_tile += gather_candidate_fixed_work
        w_tile += gather_candidate_quantum_work * dense_steps
    w_tile += readout_fold_kstep_work * ksteps * (BT / MMA_K_QUANTUM)
    w_tile += intra_gram_kstep_work * kisteps * (
        BT * (BT + 1) / float(MMA_K_QUANTUM * (MMA_K_QUANTUM + 1)))
    if decay:
        w_tile += decay_kstep_work * ksteps * (BT / MMA_K_QUANTUM)
    # ---- THE CANDIDATE FRACTION, and it is a DEFINITION, not a term ----------
    # `w_tile` is the cost of ONE CANDIDATE tile. The tiles an owner actually
    # visits are the CANDIDATE ones -- a `(owner, tile)` pair outside the §4 tile
    # bitmap is never enumerated by the CTA, so it costs nothing at all -- and there
    # are `candidate_frac * owners_total / BT` of them per token:
    #
    #   work / token = n_candidate_pairs * w_tile / (BH * T)
    #                = candidate_frac * owners_total * BH * (T / BT) * w_tile / (BH * T)
    #                = candidate_frac * owners_total * w_tile / BT
    #
    # `owners_total` is `BT`-INVARIANT and cell-constant, so it cancels out of every
    # comparison this function feeds and is not carried; `candidate_frac` is NOT
    # `BT`-invariant, and dropping it was a definitional error rather than a missing
    # effect. Omitting it asserted `candidate_frac == 1` -- that every owner is
    # candidate on every tile -- which is true on a dense cell and wrong by 1.8x on
    # a routed one.
    #
    # ZERO CONSTANTS: `candidate_frac` is an exact statistic of the realized
    # schedule, from the same reduction that produces `mean_ksteps_candidate` and in
    # the same class as every other input here. And it is IDENTICALLY 1.0 on all
    # four SATURATED-DENSE calibration cells, so it cannot have been fitted to the
    # rows it must reproduce: it moves only the two routed cells.
    #
    # MEASURED, 2026-08-01 (`ncu --clock-control base`, `smsp__inst_executed.sum`,
    # ONE profiled launch per arm), which is what identified it. At
    # `atscale-L2048-BC16` dense, the union coarsens from `candidate_frac = 0.1936`
    # at `BT = 16` to `0.3574` at `BT = 32` -- a 1.846x ratio, because a 32-token
    # tile union catches owners a 16-token one misses. The per-candidate-tile model
    # predicts `BT = 32` retires 0.821x the instructions; the machine retires
    # **1.481x**. `0.821 * 1.846 = 1.516` against 1.481, a 2.4 % residual. That 1.8x
    # WAS §2.33's "unexplained 2.3x", and it is a WORK MISCOUNT rather than the
    # stall it was filed as; the stall is the separate 1.42x issue-rate factor
    # recorded on `plan_waves`.
    #
    # FALSIFIABLE: a cell whose measured `BT = 32 / BT = 16` instruction ratio
    # departs from `candidate_frac`-corrected `w_tile / BT` by more than the ~2-3 %
    # the two at-scale cells show.
    return stats.candidate_frac * w_tile / BT


def plan_waves(n_cohorts: int, blocks_per_sm: int, sm_count: int) -> float:
    """The launch's WAVE COUNT, and the one regime distinction §6 is allowed to make.

    ``n_cohorts / (sm_count * blocks_per_sm)``, floored at 1. Every input is a
    RECORDED fact rather than a modelled one: the cohort count comes from the
    statistics pass, the CTAs/SM from the driver's occupancy oracle, the SM count
    from the device. Nothing here is calibrated and there is no constant to tune.

    **WHY IT IS A MULTIPLIER ON THE PER-CTA COST, AND WHY IT IS FLOORED AT 1.**
    Below one wave the whole grid is resident at once, so elapsed time is one CTA's
    latency and having fewer CTAs buys nothing -- residency cannot help a machine
    that is already holding everything. Above one wave the CTAs are serialized in
    batches of ``sm_count * blocks_per_sm``, so elapsed time is total work divided
    by that capacity and a candidate that fits FEWER CTAs per SM pays for it
    proportionally. ``max(1, waves)`` is the single expression that degenerates to
    both, and the ``1`` is structural (a wave count cannot be less than one wave of
    latency), not a threshold.

    **THIS IS THE STRUCTURE THE MEASUREMENT FOUND, and it is why
    ``_CALIBRATION_BT``'s KEY was wrong.** The §2.32 matrix's eight joint argmins
    partition perfectly on the wave regime -- every many-wave cell wants
    ``(BT = 16, compact)``, every one-wave cell wants ``(BT = 32, dense)`` -- and
    NEITHER ``BC`` NOR ``decay``, the two axes that table was keyed on, appears in
    the partition at all: both ``BC`` agree on every row and both decay arms agree
    on every row. ``waves_per_multiprocessor`` was 34-51 on the many-wave cells and
    0.40-0.80 on the one-wave cells.

    **AND IT IS NOT `harvest.md` §19.2's REJECTED GLOBAL DIVISOR.** That rejection
    was a measured counterexample: dividing by ``blocks_per_sm`` flipped the
    flagship decay cell to ``BT = 16`` where `compaction2.md` §B2 had measured
    ``BT = 32`` faster at 23 of 24 cell groups. §B2 measured the OLD consumer. On
    THIS kernel §2.32 measures ``BT = 16`` faster at the flagship on every arm, by
    89-153 % against paired IQRs of 0.005-0.025 -- i.e. the counterexample has been
    re-measured and now points the other way, which is the only thing that could
    have licensed the divisor and is recorded here as the licence.
    """
    if sm_count < 1:
        raise ValueError(f"sm_count must be >= 1, got {sm_count}")
    if blocks_per_sm < 1:
        raise ValueError(f"blocks_per_sm must be >= 1, got {blocks_per_sm}")
    return max(1.0, n_cohorts / float(sm_count * blocks_per_sm))


@dataclass(frozen=True, slots=True)
class PlanAxes:
    """The three axes §6 selects jointly from one cost model, plus the evidence."""

    BT: int
    compact: bool
    stats: TileUnionStats
    cost_per_token: float
    #: The residency the OCCUPANCY ORACLE reports for the selected instantiation
    #: (§6 residency, item 2(a)) -- the driver's number in production and the
    #: injected one in tests. There is no analytic mirror to confuse it with any
    #: more: `mirror_blocks_per_sm` and the SMEM footprint it was computed from were
    #: deleted with the consumer they modelled (item 6b), after the exhaustive driver
    #: probe measured the mirror INVERTING the bound it claimed.
    blocks_per_sm: int = 0
    #: The staging ring depth selected for this candidate, arbitrated by
    #: `cost_per_token / occupancy` within the candidate -- an effective cost, which
    #: selection MINIMIZES. `1` means the ring was DECLINED because the CTA it costs
    #: outweighed its overlap.
    ring_slots: int = _RING_DEPTHS[0]
    #: The launch's WAVE COUNT for the selected candidate (`plan_waves`), and the
    #: SM count it was computed against. EVIDENCE, and the evidence that matters
    #: most: the wave regime is what the §2.32 argmins partition on, so a plan that
    #: did not record it could not be checked against them.
    waves: float = 1.0
    sm_count: int = DEFAULT_SM_COUNT
    #: `cost_per_token * waves` -- what selection actually MINIMIZED. Carried
    #: separately from `cost_per_token` so the two halves of the decision (the work
    #: model and the physical regime) stay individually readable.
    effective_cost_per_token: float = 0.0


def select_plan_axes(read_levels: tuple[torch.Tensor, ...],
                     write_levels: tuple[torch.Tensor, ...],
                     topology: Topology, BC: int, *, decay: bool,
                     bt_candidates: tuple[int, ...] = _SUPPORTED_BT,
                     views: RoutingViews | None = None,
                     static_density: StaticDensity | None = None,
                     normalization: str = "global",
                     sm_count: int | None = None,
                     mma_precision: str = "bf16",
                     occupancy: OccupancyOracle | None = None,
                     **cost_kwargs) -> PlanAxes:
    """Choose ``BT``, ``compact`` and the ring depth JOINTLY (spec §6, rev .5).

    The realized routing tensors are measured ONCE at the finest candidate ``BT``
    and folded exactly to the coarser candidates
    (:func:`_tile_union_stats_multi` -- exact, over the realized schedule, not a
    distributional guess), every ``(BT, compact)`` combination is priced by
    :func:`plan_cost_per_token`, weighted by :func:`plan_waves`, and the cheapest is
    returned.

    **``intra_lowering`` IS NO LONGER AN AXIS** (item 6b, 2026-08-01). The consumer
    includes ``intra_panel.cuh`` unconditionally and does one intra lowering, so
    there is nothing to select; the whole crossover -- ``INTRA_LOWERINGS``,
    ``select_intra_lowering``, ``crossover_intra_lowering``,
    ``implemented_intra_lowerings`` and ``Plan.intra_lowering`` -- was
    old-consumer machinery and died with it.

    **This is the sanctioned §7.4.1 pattern and NOT a runtime density branch.** The
    result is compile-time template parameters stamped into ``Plan`` before the
    launch. Device code never sees ``K``, a saturation, or a fill; the kernel it
    runs is chosen, not steered.

    The ``compact=False`` instantiation is BIT-IDENTICAL to the compacted one
    wherever the unions genuinely saturate (`c_idx[k] == k` at ``K == BC``, so the
    compiled-out path issues the same loads in the same order), which is what makes
    selecting it a performance decision with no numeric consequence.
    """
    for BT in bt_candidates:
        if BT not in _SUPPORTED_BT:
            raise ValueError(f"BT candidates must lie in {_SUPPORTED_BT} (spec §6), got {BT}")
    if views is None:
        views = routing_views(read_levels, write_levels, topology)
    else:
        views.require_source(read_levels, write_levels)
    # ONE stats pass at the finest candidate granularity, folded up exactly. See
    # `_tile_union_stats_multi` for why the coarser candidates fold exactly rather
    # than approximately.
    # `static_density` (§2.7's ruling, fix brief #1) substitutes the closed form for
    # any side the CONFIGURATION makes full-support, so only genuinely-routed sides
    # pay the live-leaf expansion. It is a structural declaration, not a threshold:
    # nothing here compares a width, a density or a size.
    stats_by_bt = _tile_union_stats_multi(views, topology, BC, tuple(bt_candidates),
                                          static_density)

    if sm_count is None:
        sm_count = DEFAULT_SM_COUNT
    gram_cols = sum(spanned_moduli(topology, BC))

    # The occupancy oracle is an INPUT (coordinator ruling 2026-07-31), resolved at
    # the call site and consulted here as data. Selection therefore stays a pure
    # function of its recorded inputs -- the machine-independence §10 needs -- while
    # the ring's residency decision still uses the physical number rather than a
    # proxy the Gates proved lies.
    occupancy = resolve_occupancy(occupancy, BC=BC, d_v=topology.d_v)

    candidates: list[PlanAxes] = []
    for BT in bt_candidates:
        stats = stats_by_bt[BT]
        for compact in (True, False):
            # ---- THE RING DEPTH IS STATED, NOT ARBITRATED (M-1, 2026-08-01) --
            # It used to be arbitrated in the cost currency, because the ring's
            # BENEFIT was a term in `plan_cost_per_token` (`STAGE_COLUMN_WORK`,
            # divided by the depth) and its COST was the residency the deeper ring
            # might give up, read from the driver's occupancy oracle. RULING A
            # deleted the term on a measurement -- the per-staged-column price is
            # EXACTLY zero, see the deletion note beside `TILE_FIXED_WORK` -- and the
            # oracle is ring-INVARIANT because the consumer's shared memory is
            # entirely compile-time. Both sides of that arbitration are therefore
            # identically equal at every depth, and a `min` over equal keys is not a
            # decision: it is a tie-break pretending to be one.
            #
            # So the depth is STATED as `_RING_DEPTHS[0]`, the deepest built, with
            # the reason attached: nothing in the kernel, the occupancy or the work
            # model distinguishes the depths, and the field is read at zero sites in
            # `consumer_kernel.cuh`. The FEASIBILITY check survives unchanged,
            # because it never was about the ring: a candidate the oracle prices at
            # 0 CTAs/SM does not fit the machine and must be refused rather than
            # divided by.
            #
            # REOPENS ON A MEASUREMENT: if §2.3's item buffers build a real ring,
            # its price is measured on the kernel that has one and this becomes an
            # arbitration again -- with a constant isolated on that kernel, never
            # this one's revived.
            #
            # THE ORACLE IS THE SOLE AUTHORITY FOR PHYSICAL FACTS (user ruling
            # extension, 2026-07-31): the cost model owns WORK -- the routing
            # statistics and the measured per-tile quanta -- and the driver owns
            # every PHYSICAL constant. There is no analytic mirror any more: it
            # modelled the deleted consumer, and the exhaustive probe measured it
            # inverting the bound it claimed.
            ring_slots = _RING_DEPTHS[0]
            blocks = occupancy(BT, BC, topology.d_v, normalization == "global",
                               decay, mma_precision, compact, gram_cols, ring_slots)
            if blocks == 0:
                raise ValueError(
                    f"the candidate BT={BT}, BC={BC}, compact={compact} does not "
                    f"fit an SM: the occupancy oracle reports 0 CTAs/SM, so its "
                    "footprint exceeds the shared memory and the launch would fail")
            cost = plan_cost_per_token(topology, BC, stats, compact=compact,
                                       decay=decay, **cost_kwargs)
            # ---- THE WAVE REGIME, THE ONLY PHYSICAL TERM IN THE ARGMIN --------
            # `plan_cost_per_token` is per-CTA work. What a launch takes is that
            # work times the number of WAVES the grid is serialized into, and the
            # wave count is `n_cohorts / (sm_count * blocks_per_sm)` floored at 1 --
            # every input a recorded fact, no constant. Below one wave the term is
            # identically 1 and the argmin is the pure work model (which prefers the
            # LARGER tile, because the per-tile fixed cost amortizes); above it, a
            # candidate that fits fewer CTAs per SM pays proportionally (which
            # prefers the SMALLER tile, because `BT = 32` costs a CTA at both `BC`).
            # That is the partition the §2.32 argmins exhibit, and it is why
            # `_CALIBRATION_BT`'s `(BC, decay)` key could not express the answer.
            #
            waves = plan_waves(stats.n_cohorts, blocks, sm_count)
            candidates.append(PlanAxes(
                BT=BT, compact=compact, stats=stats,
                cost_per_token=cost, blocks_per_sm=blocks, ring_slots=ring_slots,
                waves=waves, sm_count=sm_count,
                effective_cost_per_token=cost * waves))

    assert candidates  # bt_candidates is non-empty by construction
    # ---- §6's RESIDENCY RULE IS DELETED (item 6b, 2026-08-01) ---------------
    # It fired on `cheapest.blocks_per_sm == 1` -- the 1-CTA/SM cliff, where there is
    # no latency hiding at all. That cliff was a fact about the DELETED
    # register-resident consumer, not this one (72448 B at `BT = 32, BC = 64, d_v = 64`). An EXHAUSTIVE query of this
    # consumer's occupancy oracle over the entire launchable surface --
    # `BT(2) x BC(2) x d_v(2) x decay(2) x compact(2)` = 32 keys -- measures
    # `BT = 16: 3 or 4` and `BT = 32: 2`, MINIMUM OVER ALL 32 = 2. The guard can
    # never be true, so the rule was dead code and `PlanAxes.residency_override` was
    # permanently `None`. Re-pointing its fixtures could not fix that: no cell
    # exists, in the frozen set or outside it, because no INSTANTIATION exists.
    #
    # The wave term is what carries residency into the argmin now, and it does so on
    # every candidate rather than at one threshold: `cost * max(1, n_cohorts /
    # (SMs * blocks_per_sm))`. Below one wave residency correctly drops out (the
    # whole grid is resident); above it, a candidate that fits fewer CTAs per SM pays
    # proportionally.
    #
    # FALSIFIABLE IN ONE LINE, and the falsification is a MEASUREMENT rather than an
    # argument: an instantiation this consumer can launch at 1 CTA/SM.
    # `test_no_launchable_instantiation_reaches_the_one_cta_cliff` is that probe.
    return min(candidates, key=lambda c: c.effective_cost_per_token)


def select_plan(read_levels: tuple[torch.Tensor, ...],
                   write_levels: tuple[torch.Tensor, ...],
                   topology: Topology, BC: int, *, normalization: str, decay: bool,
                   mma_precision: str = "bf16",
                   amp_dtype: torch.dtype = torch.float32,
                   bt_candidates: tuple[int, ...] = _SUPPORTED_BT,
                   views: RoutingViews | None = None,
                   static_density: StaticDensity | None = None,
                   sm_shared_bytes: int | None = None,
                   sm_count: int | None = None,
                   occupancy: OccupancyOracle | None = None,
                   **cost_kwargs) -> tuple[Plan, PlanAxes]:
    """THE plan build: a ``Plan`` whose §6 axes are SELECTED, not hand-written.

    Returns the plan and the :class:`PlanAxes` evidence it was built from (the
    realized :class:`TileUnionStats` and the winning cost), so a caller can record
    WHY a plan is what it is next to the number it produced. Everything §6 does not
    select is passed in: ``BC`` and ``normalization`` are contract, ``mma_precision``
    is a numeric-format axis (not a cost axis, though it IS part of the
    instantiation identity and is therefore forwarded to the occupancy oracle --
    `MmaPrec` has its own register pressure), and ``warp_tile`` is the realized
    ``(1, d_v/32)`` the consumer implements -- stated here rather than defaulted so
    the driver's check has something to check.

    **Selection is per ROUTING REALIZATION, and this function does not cache.** The
    statistics are exact over the realized schedule, so a plan is only as valid as
    the routing it was measured on; reusing one across steps is a policy decision
    with a correctness-free but performance-visible consequence, and it belongs to
    the caller that knows the step structure. Nothing here hides it.
    """
    if sm_shared_bytes is None:
        sm_shared_bytes = DEFAULT_SM_SHARED_BYTES
    axes = select_plan_axes(read_levels, write_levels, topology, BC, decay=decay,
                            bt_candidates=bt_candidates, views=views,
                            static_density=static_density,
                            normalization=normalization, sm_count=sm_count,
                            mma_precision=mma_precision, occupancy=occupancy,
                            **cost_kwargs)
    plan = Plan(topology=topology, normalization=normalization, BT=axes.BT, BC=BC,
                  warp_tile=(1, max(1, topology.d_v // 32)), decay=decay,
                  compact=axes.compact,
                  mma_precision=mma_precision, amp_dtype=amp_dtype,
                  sm_shared_bytes=sm_shared_bytes, ring_slots=axes.ring_slots,
                  occupancy_blocks_per_sm=axes.blocks_per_sm)
    return plan, axes


def plan_and_schedule(read_levels: tuple[torch.Tensor, ...],
                         write_levels: tuple[torch.Tensor, ...],
                         g_read: torch.Tensor | None,
                         g_write: torch.Tensor,
                         topology: Topology, BC: int, *, normalization: str, decay: bool,
                         mma_precision: str = "bf16",
                         amp_dtype: torch.dtype = torch.float32,
                         bt_candidates: tuple[int, ...] = _SUPPORTED_BT,
                         static_density: StaticDensity | None = None,
                         sm_shared_bytes: int | None = None,
                         sm_count: int | None = None,
                         occupancy: OccupancyOracle | None = None,
                         **cost_kwargs) -> tuple[Plan, PlanAxes, Schedule]:
    """THE planner plane: select the §6 axes and build the §4 schedule in ONE pass.

    :func:`select_plan` and :func:`build_schedule` walk the same routing
    tensors and independently recompute the same prefix -- the ``[B,T,H,w] ->
    [BH,T,w]`` fold (a real copy), the exact-nonzero mask, and the per-``(level,
    digit)`` §4 tile bitmap. This entry point builds one :class:`RoutingViews` and
    hands it to both, so the prefix is computed once and the builder's bitmaps at
    the selected ``BT`` are the ones the selection already formed (folded exactly
    from the finest candidate, never re-reduced).

    **Nothing about the SEMANTICS moves.** The returned plan is the plan
    :func:`select_plan` returns on the same inputs, bit for bit, and the
    schedule is the schedule :func:`build_schedule` builds from it; both remain
    callable standalone and construct their own view when none is supplied. Per-step
    selection stands -- a ``RoutingViews`` lives inside one forward and is not a
    cross-step cache.
    """
    views = routing_views(read_levels, write_levels, topology)
    plan, axes = select_plan(read_levels, write_levels, topology, BC,
                                normalization=normalization, decay=decay,
                                mma_precision=mma_precision, amp_dtype=amp_dtype,
                                bt_candidates=bt_candidates, views=views,
                                static_density=static_density,
                                sm_shared_bytes=sm_shared_bytes,
                                sm_count=sm_count, occupancy=occupancy,
                                **cost_kwargs)
    schedule = build_schedule(read_levels, write_levels, g_read, g_write, plan, views=views)
    return plan, axes, schedule


# ---------------------------------------------------------------------------
# Shared routing views -- the ONE pass the selector and the builder both walk
# ---------------------------------------------------------------------------


class RoutingViews:
    """The routing prefix the planner and the schedule builder BOTH need, computed once.

    Three things were being recomputed. (1) The ``[B,T,H,w] -> [BH,T,w]`` fold,
    which is a permute-then-reshape and therefore a real copy, done once by
    :func:`tile_union_stats` per ``BT`` candidate and again by
    :func:`build_schedule`. (2) The exact-nonzero liveness mask, same. (3) The
    per-``(level, digit)`` TILE bitmap ``[BH, n_tiles, w]``, which the selector
    forms at every candidate ``BT`` and the builder forms again at the selected
    one.

    Nothing here is a cache across STEPS: a ``RoutingViews`` is built from one
    realization of the routing tensors and is as short-lived as they are. It is
    memoization WITHIN one forward, which is why it cannot go stale -- the
    coordinator's per-step selection ruling is untouched, and the thing that dies
    is the redundant tensor work, not the per-step decision.

    ``read_mask``/``write_mask`` are exact nonzero-ness, never a threshold: entmax's
    exact support zeros are the scheduling authority (spec §1.2).
    """

    __slots__ = ("topology", "BH", "T", "read_folded", "write_folded",
                 "read_mask", "write_mask", "source_key", "_tile_cache")

    def __init__(self, topology: Topology, read_levels: tuple[torch.Tensor, ...],
                 write_levels: tuple[torch.Tensor, ...]) -> None:
        self.topology = topology
        self.source_key = _source_key(read_levels, write_levels)
        self.read_folded = tuple(_fold_bth(level) for level in read_levels)
        self.write_folded = tuple(_fold_bth(level) for level in write_levels)
        self.BH, self.T, _ = self.read_folded[0].shape
        self.read_mask = tuple(level != 0 for level in self.read_folded)
        self.write_mask = tuple(level != 0 for level in self.write_folded)
        self._tile_cache: dict[int, tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]] = {}

    def require_source(self, read_levels: tuple[torch.Tensor, ...],
                       write_levels: tuple[torch.Tensor, ...]) -> None:
        """Reject a ``views`` built from tensors OTHER than the ones being planned/built.

        A shared view is only sound if it is a view of *these* tensors. Passing a
        stale one would silently plan or schedule against a previous step's routing
        -- the exact staleness class the per-step selection ruling exists to
        prevent -- and every downstream shape check would still pass, so it would
        be silent. Checked on storage identity (``data_ptr``/shape/dtype), which is
        host-side and free.
        """
        key = _source_key(read_levels, write_levels)
        if key != self.source_key:
            raise ValueError(
                "RoutingViews was built from different routing tensors than the ones passed "
                "here (storage identity mismatch). A shared view must be a view of THESE "
                "tensors; reusing one across steps would silently plan against a previous "
                "step's routing.")

    def tile_live(self, BT: int) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """``(read, write)`` per-level ``[BH, n_tiles, width]`` §4 tile bitmaps at ``BT``.

        Memoized per ``BT``. Where a coarser ``BT`` is an exact multiple of one
        already computed, it is folded by OR-pairing adjacent tiles rather than
        re-reduced over the tokens -- the same identity
        :func:`_tile_union_stats_multi_enumerated` derives, applied one level down. An OR over
        ``2*BT`` tokens is the OR of the two halves' ORs, so the fold is exact, and
        a padded (all-zero) token tail contributes no bit either way.
        """
        if BT in self._tile_cache:
            return self._tile_cache[BT]
        source = max((base for base in self._tile_cache
                      if BT % base == 0 and not ((BT // base) & (BT // base - 1))),
                     default=None)
        n_tiles = (self.T + BT - 1) // BT
        if source is None:
            result = tuple(
                tuple(_tile_live(level, BT, n_tiles) for level in masks)
                for masks in (self.read_mask, self.write_mask))
        else:
            result = self._tile_cache[source]
            for _ in range((BT // source).bit_length() - 1):
                result = tuple(tuple(_fold_tile_bits(bits) for bits in side) for side in result)
        self._tile_cache[BT] = result
        return result


def _fold_tile_bits(bits: torch.Tensor) -> torch.Tensor:
    """OR-pair adjacent tiles of a ``[BH, n_tiles, width]`` bitmap (exact, see above)."""
    if bits.shape[1] % 2:
        bits = torch.nn.functional.pad(bits, (0, 0, 0, 1))
    return bits[:, 0::2] | bits[:, 1::2]


def _source_key(read_levels: tuple[torch.Tensor, ...],
                write_levels: tuple[torch.Tensor, ...]) -> tuple:
    """Storage-identity key: ``(data_ptr, shape, stride, dtype)`` per tensor.

    Soundness precondition (not enforced, only relied on): this key is sound
    for detecting "a different tensor" only because nothing in this codebase
    holds a ``RoutingViews`` -- or a view built from one -- across more than one
    forward call. Two gaps follow from that, neither reachable in-tree today:
    (a) an in-place mutation of ``read_levels``/``write_levels`` AFTER a
    ``RoutingViews`` is constructed leaves the key unchanged while the live
    data (and therefore the masks it should invalidate) has changed; (b) the
    caching allocator can reissue a freed address, so a view held past its
    producing tensors' lifetime could collide with an unrelated later tensor's
    ``data_ptr``. Both are excluded only by the call discipline in
    :func:`plan_and_schedule` (fresh ``RoutingViews`` per call, never stored
    past it) -- if that discipline is ever relaxed, this key must be
    strengthened (e.g. a version counter or content hash) rather than trusted
    as-is.
    """
    return tuple((level.data_ptr(), tuple(level.shape), tuple(level.stride()), level.dtype)
                 for level in (*read_levels, *write_levels))


def routing_views(read_levels: tuple[torch.Tensor, ...],
                  write_levels: tuple[torch.Tensor, ...],
                  topology: Topology) -> RoutingViews:
    """Fold the producer's per-level ``[B,T,H,w]`` allocations once (see :class:`RoutingViews`)."""
    if len(read_levels) != topology.D or len(write_levels) != topology.D:
        raise ValueError(f"read/write levels must contain one tensor per level (D={topology.D})")
    return RoutingViews(topology, tuple(read_levels), tuple(write_levels))


# ---------------------------------------------------------------------------
# Tile-union statistics (§3 step 2's gather, priced by §6)
# ---------------------------------------------------------------------------

#: The MMA fragment's k-extent, and therefore the QUANTUM every gathered extent is
#: rounded up to. This is a SHAPE of the instruction the kernel issues
#: (``m16n8k16`` at bf16, and the tf32x3 lowering keeps the same k-extent by
#: construction -- it splits the MANTISSA, not the contraction), not a tuning
#: threshold: ``K = 1`` and ``K = 16`` issue the same one k-step, and ``K = 0``
#: issues none. `--mode gatherstats` in
#: ``benchmarks/ops/benchmark_rola_v3_new_consumer.py`` measures the same quantity
#: from the schedule's token bitmaps and is the independent cross-check on it.
MMA_K_QUANTUM = 16


@dataclass(frozen=True, slots=True)
class StaticDensity:
    """Which ``(level, side)`` cells are FULL-SUPPORT BY CONSTRUCTION (§2.7 ruling).

    One boolean per level per side: ``True`` means the configured ACTIVATION for
    that cell cannot produce an exact structural zero, so every digit of that level
    is live at every token in every realization. That is a fact about the
    configuration, available at model build, and it is read from the ONE place
    activation properties live -- ``ACTIVATION_REGISTRY``'s ``exact_zeros`` column,
    via :meth:`Topology.activation_properties` -- never from a name list and never
    from a measurement of the data (`rola/routing/activations.py`: *"No case list
    over activations may exist anywhere outside this module"*).

    **Why this is an INPUT rather than something the planner derives from the
    ``topology`` it already holds.** The topology states what the model's producer
    will emit; the planner is handed tensors. Those agree in production (the
    producer applies exactly the configured activation), and they deliberately do
    NOT agree in the synthetic fixtures, where a support constructor imposes a
    density the declared activation does not describe (``benchmarks/bench/cells.py``:
    *"a synthetic cell's own sparsity comes from the support constructors"*, and
    ``tests/unit/test_planner.py``'s scenarios build sparse tensors under a softmax
    topology). Deriving the closed form from the tag alone would therefore price a
    density those callers do not have. Passing it makes the claim the CALLER's --
    :class:`~rola.layers.RoLA` builds one from its own topology once, at
    construction -- and leaves every caller that does not pass one on the fully
    data-dependent pass, bit for bit.

    **What a wrong declaration costs, stated exactly.** Only the COST MODEL: the
    §4 schedule the kernel executes is built by :func:`build_schedule` from the
    realized masks and never reads this object, so a side declared dense whose
    realization has zeros can only mis-price a plan, never mis-execute one. The
    one realistic way that happens is float underflow -- a softmax logit far
    enough below the row max that ``exp`` rounds to exactly ``0`` -- which shrinks
    the realized support without changing the activation's mathematics; the closed
    form then over-counts ``K`` by the underflowed fraction.
    """

    read: tuple[bool, ...]
    write: tuple[bool, ...]

    @classmethod
    def from_topology(cls, topology: Topology) -> StaticDensity:
        """Read the per-``(level, duty)`` ``exact_zeros`` column of the registry."""
        return cls(*(tuple(not topology.activation_properties(level, duty).exact_zeros
                           for level in range(topology.D))
                     for duty in ("read", "write")))

    def side_dense(self, duty: str) -> bool:
        """Is the WHOLE side full-support -- the condition the closed form needs?

        A leaf is live at a token iff EVERY level's digit is live there
        (:func:`tile_union_stats`), so one dense level among sparse ones does not
        make the side's support the full rectangle and the closed form below does
        not apply. The per-LEVEL resolution is kept because §2.7's other planes
        (the producer's support machinery, the consumer's feeder specialization)
        are per level; the planner's tile statistics are the one consumer of this
        object for which only the conjunction is actionable.
        """
        if duty == "read":
            return all(self.read)
        if duty == "write":
            return all(self.write)
        raise ValueError(f"duty must be 'read' or 'write', got {duty!r}")


@dataclass(frozen=True, slots=True)
class TileUnionStats:
    """Exact per-``(owner, tile)`` statistics of §3 step 2's gather.

    Every field is computed from the REALIZED routing tensors over the realized
    schedule -- not from a distributional model, and not from the per-level
    marginals :class:`LevelSupportStats` carries. That distinction is the whole
    point: ``K`` is a per-TILE UNION, and a union over ``BT`` tokens saturates far
    faster than any per-token marginal predicts.

    ``mean_K``/``saturated_frac`` are over ALL ``(owner, tile)`` pairs;
    ``mean_K_candidate``/``mean_KI_candidate`` are over the pairs the kernel
    actually walks (the §4 candidate bitmap), which is the population that pays
    the gather. They differ: a tile can be a candidate on the AND of per-level
    TILE bitmaps and still have ``K == 0``, because tile liveness is an OR over
    the tile's tokens while the leaf test is an AND at a common token.

    **``mean_ksteps_candidate``/``mean_kisteps_candidate`` are the QUANTA, and they
    are what §6 prices.** The kernel does not move ``K`` leafs, it moves
    ``ceil(K / MMA_K_QUANTUM)`` MMA k-steps: a candidate tile with ``K = 1`` costs
    the same k-step as one with ``K = 16``, and a candidate tile with ``K = 0``
    costs NONE. ``mean_K_candidate`` cannot express either end. Measured at
    ``atscale-L2048``: ``mean_K_candidate / BC = 0.9724`` against
    ``mean_ksteps_candidate / (BC/16) = 0.606`` -- the whole compact saving lives in
    that gap, and 39 % of that cell's candidate pairs gather ``K = 0`` and skip
    every GEMM. Both are means over the CANDIDATE population, like the two means
    above them, so a term may mix them freely.

    ``n_cohorts`` is ``owners_total * BH`` -- the CTA grid the launch requests, one
    CTA per ``(bh, owner)``. It is an UPPER BOUND on the live grid rather than the
    grid: an owner with no candidate tile at all still gets a CTA that exits early.
    Measured tightness (§2.32): ``atscale-L2048`` has 8180 live owners of 8192
    (99.9 %) and every ``topo`` cell is exact, so no separate liveness pass is
    warranted -- but it is a bound and §6's wave term states it as one.
    """

    BC: int
    BT: int
    mean_K: float
    saturated_frac: float
    mean_K_candidate: float
    mean_KI_candidate: float
    #: Mean ``ceil(K / MMA_K_QUANTUM)`` over CANDIDATE pairs -- the readout's and the
    #: fold's k-step count, which is what the kernel pays.
    mean_ksteps_candidate: float
    #: Mean ``ceil(KI / MMA_K_QUANTUM)`` over CANDIDATE pairs -- the intra Gram's.
    mean_kisteps_candidate: float
    candidate_frac: float
    n_pairs: int
    #: ``owners_total * BH``: the requested CTA grid, an upper bound on the live one.
    n_cohorts: int


def tile_union_stats(read_levels: tuple[torch.Tensor, ...],
                     write_levels: tuple[torch.Tensor, ...],
                     topology: Topology, BC: int, BT: int, *,
                     views: RoutingViews | None = None,
                     static_density: StaticDensity | None = None) -> TileUnionStats:
    """Compute :class:`TileUnionStats` exactly, for one ``(BC, BT)`` candidate.

    ``K(owner, tile) = |{ i in [0,BC) : exists t in the tile, R[t, leaf(i)] != 0
    or W[t, leaf(i)] != 0 }|``, over the UNION of the two supports
    (``GRADIENT_PROOFS_WOLFRAM.md`` P5.21 item 1 -- the backward iterates
    ``support(W) UNION support(R)``, so a write-only count would price a gather
    the kernel is forbidden to build).

    **All ``D`` levels enter, in the two different roles §3 gives them.** The
    SPANNED levels (``l >= j``) distinguish leafs of one owner, so they are ANDed
    per leaf. The FIXED levels (``l < j``) cannot distinguish two leafs -- but a
    zero there kills the whole token's row scale (§3 step 3), so they AND into
    TOKEN VALIDITY, which shrinks every leaf's support word identically. Omitting
    them is not a conservative approximation; it is a different and strictly
    weaker gather, and it is what the kernel and every instrument did before
    2026-07-30: at ``branches=[8,16], BC=16`` (``j = 1``) it made the whole
    width-8 level invisible and reported the flagship entmax-gain-8 cell as
    ``K = 15.91/16, sat 0.908`` against the exact ``11.13/16, sat 0.017``.

    The side gains are excluded, exactly as the kernel excludes them: ``g_write``
    scales the deposit but NOT the clock (:func:`build_clock_stream`), so folding
    it in would drop leafs whose survival still evolves.

    ``KI`` is §5's causal intra sub-list length -- a leaf is included iff it is
    read at or after its first in-tile write -- which is what prices the
    ``realized_leaf`` intra term (§6).

    This is a PLANNER function. Its results are reduced to compile-time template
    parameters before launch; no kernel path reads them (§3's branch rule).

    ``views`` is an optional :class:`RoutingViews` carrying the folded tensors and
    the memoized per-``BT`` tile bitmaps, so the selector and the schedule builder
    can share one pass over the routing (see :func:`plan_and_schedule`). Passing
    it changes NOTHING about what is computed -- it is the same reduction over the
    same tensors -- only how many times the shared prefix is recomputed.

    ``static_density`` declares which ``(level, side)`` cells are full-support BY
    CONFIGURATION (:class:`StaticDensity`). It changes NOTHING about what is
    computed either -- a dense side's statistics are substituted by the closed form
    they provably equal -- only how much of the data-dependent pass has to run to
    reach them.
    """
    if views is None:
        views = routing_views(read_levels, write_levels, topology)
    else:
        views.require_source(read_levels, write_levels)
    return _tile_union_stats_multi(views, topology, BC, (BT,), static_density)[BT]


def _bt_candidate_chain(bt_candidates: tuple[int, ...]) -> tuple[int, ...]:
    """Sorted unique candidates, validated as a power-of-two chain above the finest.

    Both statistics implementations require it, for the same structural reason and
    with the same message: the sparse counter groups tokens by ``t // BT`` inside
    ONE chunk of coarsest-candidate tiles (so every candidate's tile nests in the
    chunk), and the enumerated reference OR-pairs adjacent tiles.

    **Recorded, not enforced here (delta review R1):** the support-counted pass
    stores a token offset within the coarsest candidate tile as ``int16``, so it
    represents ``BT`` up to 32767. Nothing in this function bounds ``BT``; what
    bounds it is ``_SUPPORTED_BT = (16, 32)``, checked one layer up in
    :func:`select_plan_axes`, and spec §6's pinning of ``BT`` to that set. The
    guarantee is therefore real but lives elsewhere, and a future ``BT`` outside that
    set is a spec change that must revisit the offset width rather than a runtime
    condition to branch on.
    """
    cands = tuple(sorted({int(BT) for BT in bt_candidates}))
    if not cands:
        raise ValueError("bt_candidates must be non-empty")
    BT0 = cands[0]
    for BT in cands:
        ratio, rem = divmod(BT, BT0)
        if rem or (ratio & (ratio - 1)):
            raise ValueError(
                f"BT candidates must be a power-of-two chain above the finest ({BT0}); "
                f"got {cands}. The exact fold OR-pairs adjacent tiles, so a candidate that "
                "is not 2^k times the finest has no exact fold and would have to be "
                "re-measured -- which is a real option, but it is not silently taken here.")
    return cands


def _exclusive_cumsum(counts: torch.Tensor) -> torch.Tensor:
    """``[0, c0, c0+c1, ...]`` -- the start offset of each segment in a ragged flat array."""
    return torch.cumsum(counts, dim=0) - counts


def _live_leaf_scatter(counts: list[torch.Tensor], digits: list[torch.Tensor],
                       offsets: list[torch.Tensor], live: torch.Tensor,
                       strides: tuple[int, ...], lo: int, hi: int,
                       pair_cell: torch.Tensor, pair_token: torch.Tensor,
                       ) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(destination cell, token offset)`` of every live leaf, over pairs ``[lo, hi)``.

    **The product structure is what makes this ``O(support)``.** A leaf is live at a
    token iff EVERY level's digit is live there (:func:`tile_union_stats`), so one
    token's live leaf set is the CARTESIAN PRODUCT of the per-level supports and its
    size is ``prod_l n_l``. Expanding it is therefore a mixed-radix decomposition of a
    flat offset whose RADICES ARE THE SUPPORT COUNTS, not the widths, and it touches
    exactly one index per live leaf. Nothing here is proportional to ``N``.

    Everything that varies per ``(bh, token)`` rather than per leaf -- the
    destination row ``pair_cell``, the token offset ``pair_token`` that is the
    scattered value, the segment start, and each level's support offset, count and
    radix -- is stacked into ONE ``int32`` row per pair and expanded once, so the
    per-entry work is a fixed handful of ``int32`` lanes plus one support gather per
    level. **The layout is measured, not assumed** (dense-read fixture, 8.4 M
    entries, `harvest.md` §31): recomputing the destination from a per-entry pair id
    costs 7.8 ms, an ``int64`` row-major lane block 6.4 ms, and this column-contiguous
    ``int32`` form 3.5 ms -- at this size the pass is bound by the bytes it moves per
    entry and by nothing else, and a row-major lane block makes every lane read pull
    a full cache line.
    """
    m = live[lo:hi]
    total = int(m.sum())
    if total == 0:
        return None
    if total > _INT32_MAX:
        raise ValueError(
            f"{total} live entries in one chunk exceeds the int32 addressing this pass "
            f"uses; the chunk budget (_STATS_ENTRY_CHUNK={_STATS_ENTRY_CHUNK}) is meant "
            "to keep it far below that, so this means one BT_max tile alone is that "
            "dense -- which needs a finer chunk unit, not a wider index.")
    device = live.device
    D = len(strides)
    lanes = ([pair_cell[lo:hi], pair_token[lo:hi], _exclusive_cumsum(m)]
             + [offsets[level][lo:hi] for level in range(D)]
             + [counts[level][lo:hi] for level in range(1, D)])
    rows = torch.repeat_interleave(torch.stack(lanes, dim=0).to(torch.int32), m, dim=1)
    # Innermost digit first, so the mixed-radix decode is a running remainder: the
    # innermost level needs no division and the outermost needs no remainder.
    rest = torch.arange(total, dtype=torch.int32, device=device) - rows[2]
    index = rows[0]
    for level in range(D - 1, -1, -1):
        if level:
            rank = rest % rows[2 + D + level]
            rest = rest // rows[2 + D + level]
        else:
            rank = rest
        index = index + digits[level].index_select(0, rows[3 + level] + rank) * strides[level]
    return index.long(), rows[1].to(torch.int16)


#: Budgets for one chunk of the production statistics pass. Chunking is over whole
#: ``max(bt_candidates)`` tiles -- the coarsest granularity at which every candidate
#: ``BT``'s tiles nest inside one chunk, so a chunk boundary can never split a tile.
#: Two transients are bounded, and they are the pass's only two terms: the LIVE
#: ENTRIES it expands (``O(support)``) and the ``(tile, leaf)`` CELLS it reduces over
#: (``O(BH * n_tiles * N)`` -- the size of the answer, and ``BT`` times smaller than
#: the ``(token, leaf)`` rectangle the enumerated reference walks).
_STATS_ENTRY_CHUNK = 1 << 22
_STATS_TILE_CELL_CHUNK = 1 << 24

#: The addressing :func:`_live_leaf_scatter` expands in; both budgets sit far below it.
_INT32_MAX = (1 << 31) - 1


def _tile_union_stats_multi(views: RoutingViews, topology: Topology, BC: int,
                            bt_candidates: tuple[int, ...],
                            static_density: StaticDensity | None = None,
                            ) -> dict[int, TileUnionStats]:
    """:class:`TileUnionStats` for EVERY candidate ``BT``, counted from the SUPPORT.

    Bitwise-identical output to :func:`_tile_union_stats_multi_enumerated` (gated on
    the full planner grid), at ``O(support) + O(BH * n_tiles * N)`` instead of
    ``Theta(BH * T * N)``: the reference asks of every ``(token, owner, leaf)`` cell
    whether it is live; this walks the live entries and records WHERE they land, in
    the only object the answer needs -- a per-``(tile, leaf)`` pair of token extremes,
    which is ``BT`` times smaller than the reference's ``(token, leaf)`` rectangle.
    Four derivations make that possible, and each is what one accumulator comes from.

    **1. Owners are contiguous leaf blocks, and ``owner == leaf // BC``.** By
    :meth:`OwnerGeometry.leaf_base` the owner's leaf base is exactly ``owner * BC``
    and its ``BC`` leafs are the next ``BC`` ids, so the owner axis is a RESHAPE of
    the leaf axis -- no owner loop, no per-owner digit gather, and ``K`` for every
    owner at once is one ``[..., owners, BC] -> sum`` over the union bitmap.

    **2. Two token extremes carry both counts.** Per ``(tile, leaf)`` keep
    ``first_write`` (scatter-amin over the live WRITE entries) and ``last_read``
    (scatter-amax over the live READ entries), as offsets within the chunk's tile.
    Then ``leaf is live`` (the union, giving ``K``) is ``last_read >= 0 or
    first_write < BT_max``, and §5's causal intra bit (giving ``KI``) is
    ``first_write <= last_read``: a leaf is in the intra sub-list iff some in-tile
    read is at or after some in-tile write, which over a whole tile is exactly the
    comparison of the two extremes. The reference reaches the same bit by a token
    cummax plus an ``Rany_B AND Wany_A`` carry across each fold boundary; on extremes
    the carry is not needed, because ``min``/``max`` fold with no cross term.

    **3. The coarser candidates fold by ``min``/``max``.** Offsets are stored
    relative to the COARSEST candidate tile, so folding two adjacent finest tiles is
    ``min(first_write)`` and ``max(last_read)`` -- the same exact fold identity the
    reference derives, one representation down. Padded tiles beyond ``ceil(T/BT)``
    hold the untouched sentinels, i.e. they are dead, exactly as the reference's
    zero-padded token tail is.

    **4. Candidacy is a PRODUCT SET, so its count is closed-form.** A candidate owner
    at a tile is one whose boundary slice is tile-live AND whose every fixed digit is
    tile-live (§4), i.e. the candidate set on one side is
    ``{live slices} x prod_{l<j} {live digits of level l}``, whose size is the
    product of the §4 TILE BITMAPS' popcounts. ``cand = cand_r OR cand_w`` is a union
    of two product sets and the intersection of two product sets is the product of
    the per-axis intersections, so inclusion-exclusion gives ``|A| + |B| - |A and B|``
    exactly, from popcounts alone -- ``O(BH * n_tiles * sum(widths))``, independent of
    ``N`` and of the support.

    **``sum_K_candidate == sum_K`` and ``sum_KI_candidate == sum_KI``, proved not
    assumed.** If ``K(owner, tile) > 0`` then some leaf of that owner is live on some
    side at some token of the tile, so EVERY level's digit for that leaf is live at
    that token, so every level's tile bitmap bit is set -- the boundary level's
    through its slice OR -- and the owner is a candidate on that side. Every pair with
    ``K > 0`` is therefore a candidate, and restricting the sum to candidates removes
    only zero terms; ``KI > 0`` implies ``K > 0`` a fortiori. The equal-to-the-
    reference gate exercises this on the sparse and zero-level scenarios, where the
    two populations genuinely differ.

    **Where the live pass is NOT a win, and what replaces it.** At a dense-READ
    routing configuration the support IS the full rectangle (``prod_l w_l`` live
    leafs per token), so expanding it costs what the reference costs and the win
    collapses to the ``BT``-fold of the reduction -- no algorithm counts a dense
    union in less than its own size (measured both ways in `harvest.md` §31). What
    that argument leaves open is whether the union has to be COUNTED at all, and
    when the side is dense BY CONFIGURATION (``static_density``, §2.7's ruling) it
    does not: every one of the side's statistics is then a closed form in
    ``(BC, BT, T, BH, owners_total)`` alone, derived below, and the live pass runs
    only for the genuinely-routed side. Passing no ``static_density`` leaves this
    function exactly as it was, on every path.

    **THE CLOSED FORM, derived (fix brief #1, 2026-08-01).** Let one side be dense,
    i.e. every level's digit live at every token. Then for every ``(owner, tile)``
    pair whose tile holds at least one token -- and there are exactly
    ``ceil(T / BT)`` such tiles per ``bh``, which is ``n_tiles`` -- every one of the
    owner's ``BC`` leafs is live on that side at every token of the tile. Hence:

    * ``K = BC`` on every pair, so ``sum_K = n_pairs * BC``, ``mean_K = BC`` and
      ``saturated_frac = 1``; the tiles the buffers pad beyond ``ceil(T/BT)``
      contribute ``K = 0`` in the live pass and are excluded from ``n_pairs`` in
      both, so they cancel rather than needing a term.
    * ``ksteps = ceil(BC / MMA_K_QUANTUM)`` on every pair, by the same argument
      applied to the quantum (``sum_ks`` is summed BEFORE the mean, as always).
    * ``candidate_frac = 1``: derivation 4's ``p`` for the dense side is
      ``n_slices * prod_{l<j} w_l``, which is ``owners_total`` identically, and
      ``cand = p_read + p_write - p_both`` with ``p_both = p_sparse <= p_dense``
      collapses to ``owners_total``. The tile bitmaps are therefore not needed at
      all, which is why the candidacy pass below is skipped whole rather than
      simplified.
    * ``KI`` STAYS DATA-DEPENDENT, and it is the only thing that does. The intra
      bit is ``first_write <= last_read`` (derivation 2). With the READ side dense,
      ``last_read`` is the tile's own last token, so the comparison degenerates to
      ``first_write < BT_max`` -- "the leaf is written somewhere in the tile" -- and
      the WRITE scatter still has to establish it. With the WRITE side dense,
      ``first_write == 0`` and it degenerates to ``last_read >= 0`` symmetrically.
      With BOTH dense the bit is constant ``True`` and ``KI = BC``, so nothing is
      scattered at all and the whole pass is arithmetic.
    """
    cands = _bt_candidate_chain(bt_candidates)
    BT0, BT_max = cands[0], cands[-1]
    geometry = owner_geometry(topology.widths, BC)
    widths = topology.widths
    D = topology.D
    j = geometry.first_spanned_level
    c = geometry.slice_digits
    n_slices = geometry.n_slices
    owners_total = geometry.owners_total
    N = prod(widths)
    BH, T = views.BH, views.T
    device = views.read_mask[0].device
    n_units = (T + BT_max - 1) // BT_max
    tiles_per_unit = BT_max // BT0
    strides = tuple(prod(widths[level + 1:]) for level in range(D))

    # Per side, the support in its sparse currency: the live digits in (pair, digit)
    # order, the per-pair count, and that count's exclusive prefix (the addressing).
    #
    # HOST SYNCS, counted (delta review R2). This pass adds `2 * D` -- one `nonzero()`
    # per level per side, here -- plus one `per_unit.cpu()` for the chunk split and one
    # `int(m.sum())` per chunk per side inside `_live_leaf_scatter`. At the at-scale
    # cell that is 5 syncs (one chunk), against a plane §7.4.2 tracks as launch- and
    # sync-bound: measured 236 launches for 2.4 ms of device time against 5.9 ms of
    # wall (`harvest.md` §31.2). They are the price of the sparse currency -- a
    # ragged expansion has to know its own length -- and they are named here so the
    # parked device-side-planner item inherits the number rather than rediscovering it.
    # THE STATIC/DYNAMIC SPLIT (fix brief #1). `dense_side[side]` is a statement
    # about the CONFIGURATION -- see the closed form in the docstring -- and every
    # skip below is keyed on it alone, never on a width, a density measurement or a
    # size threshold. With no declaration both sides are live and this whole
    # function is the pass it was.
    dense_side = (False, False) if static_density is None else (
        static_density.side_dense("read"), static_density.side_dense("write"))
    live_sides = tuple(side for side in (0, 1) if not dense_side[side])

    counts: list[list[torch.Tensor]] = [[], []]
    digits: list[list[torch.Tensor]] = [[], []]
    offsets: list[list[torch.Tensor]] = [[], []]
    live_per_pair: list[torch.Tensor | None] = [None, None]
    for side in live_sides:
        masks = (views.read_mask, views.write_mask)[side]
        flat = [mask.reshape(BH * T, width) for mask, width in zip(masks, widths)]
        side_counts = [mask.sum(dim=-1) for mask in flat]
        counts[side] = side_counts
        digits[side] = [mask.nonzero()[:, 1].to(torch.int32) for mask in flat]
        offsets[side] = [_exclusive_cumsum(cnt) for cnt in side_counts]
        product = side_counts[0]
        for cnt in side_counts[1:]:
            product = product * cnt
        live_per_pair[side] = product

    zero = lambda: torch.zeros((), dtype=torch.int64, device=device)  # noqa: E731
    acc = {BT: {name: zero() for name in
                ("sum_K", "sum_sat", "sum_cand", "sum_KI", "sum_ks", "sum_kis")}
           for BT in cands}
    #: Totals that are arithmetic in the geometry, filled only where a side is
    #: dense by construction; they REPLACE the corresponding accumulator rather
    #: than being checked against it, so no reduction is issued for them at all.
    static_totals: dict[tuple[int, str], int] = {}

    n_tiles_at = {BT: (T + BT - 1) // BT for BT in cands}
    if any(dense_side):
        # Closed form, per the docstring's derivation. `ceil(BC / MMA_K_QUANTUM)`
        # is the per-pair k-step count at `K = BC`, summed (never averaged) first.
        bc_ksteps = -(-BC // MMA_K_QUANTUM)
        for BT in cands:
            n_pairs = owners_total * BH * n_tiles_at[BT]
            static_totals[(BT, "sum_K")] = n_pairs * BC
            static_totals[(BT, "sum_sat")] = n_pairs
            static_totals[(BT, "sum_cand")] = n_pairs
            static_totals[(BT, "sum_ks")] = n_pairs * bc_ksteps
            if all(dense_side):
                # Both sides dense: the causal intra bit is constant `True`.
                static_totals[(BT, "sum_KI")] = n_pairs * BC
                static_totals[(BT, "sum_kis")] = n_pairs * bc_ksteps
    else:
        # Derivation 4: candidacy, closed-form from the tile bitmaps' popcounts.
        for BT in cands:
            r_tile, w_tile = views.tile_live(BT)
            n_tiles = r_tile[0].shape[1]
            slice_r, slice_w = (side[j].reshape(BH, n_tiles, n_slices, c).any(dim=-1)
                                for side in (r_tile, w_tile))
            p_read = slice_r.sum(dim=-1)
            p_write = slice_w.sum(dim=-1)
            p_both = (slice_r & slice_w).sum(dim=-1)
            for level in range(j):
                p_read = p_read * r_tile[level].sum(dim=-1)
                p_write = p_write * w_tile[level].sum(dim=-1)
                p_both = p_both * (r_tile[level] & w_tile[level]).sum(dim=-1)
            acc[BT]["sum_cand"] = (p_read + p_write - p_both).sum()

    # A chunk is a run of whole BT_max units, which is a contiguous run of (bh, token)
    # pairs; both budgets above bound it. Everything that varies per pair rather than
    # per leaf is resolved here, once: the destination ROW of the pair's finest tile
    # (chunk-relative at use) and the token offset that is the scattered value.
    if live_sides:
        pair_ids = torch.arange(BH * T, device=device)
        token_of_pair = pair_ids % T
        unit_of_pair = (pair_ids // T) * n_units + token_of_pair // BT_max
        token_in_unit = token_of_pair % BT_max
        cell_of_pair = (unit_of_pair * tiles_per_unit + token_in_unit // BT0) * N
        per_unit = torch.zeros(BH * n_units, dtype=torch.int64, device=device)
        load_per_pair = live_per_pair[live_sides[0]]
        for side in live_sides[1:]:
            load_per_pair = load_per_pair + live_per_pair[side]
        per_unit.index_add_(0, unit_of_pair, load_per_pair)
        unit_load = per_unit.cpu().tolist()
        units_per_chunk = max(1, _STATS_TILE_CELL_CHUNK // max(1, tiles_per_unit * N))

        chunks: list[tuple[int, int]] = []
        start, load = 0, 0
        for unit, unit_entries in enumerate(unit_load):
            if unit > start and (load + unit_entries > _STATS_ENTRY_CHUNK
                                 or unit - start >= units_per_chunk):
                chunks.append((start, unit))
                start, load = unit, 0
            load += unit_entries
        chunks.append((start, BH * n_units))
    else:
        # Both sides static: nothing is expanded, so there is nothing to chunk and
        # not one of this pass's syncs is issued.
        chunks = []

    for unit_lo, unit_hi in chunks:
        n_chunk = unit_hi - unit_lo
        lo = (unit_lo // n_units) * T + (unit_lo % n_units) * BT_max
        last = unit_hi - 1
        hi = (last // n_units) * T + min(T, (last % n_units + 1) * BT_max)
        cells = n_chunk * tiles_per_unit * N
        # Derivation 2: the two token extremes, per (finest tile, leaf). The
        # sentinels are self-excluding -- a tile-leaf no entry ever reached keeps
        # `BT_max`/`-1`, which reads as dead on the union test and fails the causal
        # comparison in both orders. A side that is dense BY CONFIGURATION has no
        # extreme to establish (the docstring's closed form gives its side of every
        # predicate in constant form), so neither its buffer nor its scatter exists.
        extreme: list[torch.Tensor | None] = [None, None]
        chunk_cell = cell_of_pair - unit_lo * tiles_per_unit * N
        for side, sentinel, reduce_op in ((0, -1, "amax"), (1, BT_max, "amin")):
            if dense_side[side]:
                continue
            buffer = torch.full((cells,), sentinel, dtype=torch.int16, device=device)
            entries = _live_leaf_scatter(counts[side], digits[side], offsets[side],
                                         live_per_pair[side], strides, lo, hi,
                                         chunk_cell, token_in_unit)
            if entries is not None:
                index, value = entries
                buffer.scatter_reduce_(0, index, value, reduce=reduce_op)
            extreme[side] = buffer.view(n_chunk, tiles_per_unit, N)
        last_read, first_write = extreme

        for step, BT in enumerate(cands):
            if step:                                    # derivation 3: fold by min/max
                for _ in range((BT // cands[step - 1]).bit_length() - 1):
                    if first_write is not None:
                        first_write = first_write.view(
                            n_chunk, first_write.shape[1] // 2, 2, N).amin(dim=2)
                    if last_read is not None:
                        last_read = last_read.view(
                            n_chunk, last_read.shape[1] // 2, 2, N).amax(dim=2)
            bucket = acc[BT]
            if last_read is None:
                # READ dense: `K` and its quantum are static (they are in
                # `static_totals`), and the causal bit is "written in this tile".
                ki_bit = first_write < BT_max
            elif first_write is None:
                # WRITE dense: symmetric -- the causal bit is "read in this tile".
                ki_bit = last_read >= 0
            else:
                union = (last_read >= 0) | (first_write < BT_max)
                k_count = union.view(n_chunk, -1, owners_total, BC).sum(dim=-1)
                bucket["sum_K"] += k_count.sum()
                bucket["sum_sat"] += (k_count == BC).sum()
                # The QUANTA, in the same chunk loop and off the same per-pair
                # counts: the kernel's k-step is `ceil(K / MMA_K_QUANTUM)`, which is
                # 0 at K = 0 and 1 for every K in [1, 16]. Summing the ceilings is
                # the only order that survives -- `ceil(mean)` is not `mean(ceil)`
                # and the difference IS the compact saving (`TileUnionStats`' note).
                bucket["sum_ks"] += ((k_count + (MMA_K_QUANTUM - 1)) // MMA_K_QUANTUM).sum()
                ki_bit = first_write <= last_read
            ki_count = ki_bit.view(n_chunk, -1, owners_total, BC).sum(dim=-1)
            bucket["sum_KI"] += ki_count.sum()
            bucket["sum_kis"] += ((ki_count + (MMA_K_QUANTUM - 1)) // MMA_K_QUANTUM).sum()

    order_keys = [(BT, name) for BT in cands
                  for name in ("sum_K", "sum_sat", "sum_cand", "sum_KI",
                               "sum_ks", "sum_kis")]
    dynamic_keys = [key for key in order_keys if key not in static_totals]
    totals: dict[tuple[int, str], int] = dict(static_totals)
    if dynamic_keys:
        flat = torch.stack([acc[BT][name] for BT, name in dynamic_keys]).cpu().tolist()
        totals.update(zip(dynamic_keys, flat))

    out: dict[int, TileUnionStats] = {}
    for BT in cands:
        n_pairs = owners_total * BH * n_tiles_at[BT]
        n_cand = totals[(BT, "sum_cand")]
        out[BT] = TileUnionStats(
            BC=BC, BT=BT,
            mean_K=totals[(BT, "sum_K")] / n_pairs if n_pairs else 0.0,
            saturated_frac=totals[(BT, "sum_sat")] / n_pairs if n_pairs else 0.0,
            mean_K_candidate=totals[(BT, "sum_K")] / n_cand if n_cand else 0.0,
            mean_KI_candidate=totals[(BT, "sum_KI")] / n_cand if n_cand else 0.0,
            # `sum_ks`/`sum_kis` are over ALL pairs, divided by the CANDIDATE count
            # -- valid for the same reason `sum_K_candidate == sum_K` is (docstring
            # above): `K > 0` implies candidacy, and `ceil(0/16) == 0`, so the
            # non-candidate pairs contribute exactly zero to both numerators.
            mean_ksteps_candidate=totals[(BT, "sum_ks")] / n_cand if n_cand else 0.0,
            mean_kisteps_candidate=totals[(BT, "sum_kis")] / n_cand if n_cand else 0.0,
            candidate_frac=n_cand / n_pairs if n_pairs else 0.0,
            n_pairs=n_pairs,
            n_cohorts=owners_total * BH)
    return out


#: Element budget for one owner CHUNK of the statistics pass, in booleans per side.
#: The pass is vectorized OVER OWNERS, so the transient leaf tensor is
#: ``[BH, n_tiles, BT, chunk, BC]`` -- i.e. ``BH * T * chunk * BC`` booleans. The
#: chunk is sized from this budget so the transient never grows with ``N``: at
#: ``chunk == owners_total`` the tensor would be ``BH * T * N``, which is exactly
#: the ``O(L*N)`` object §4 forbids materializing. Chunking is therefore a
#: STRUCTURAL bound, not a tuning knob -- the standing "design for nc ~ 65K" rule
#: is what it exists for, and the flagship cell (``BH*T*BC = 262144``) fits a
#: 64-owner chunk inside it, so the flagship pays exactly one chunk.
_STATS_OWNER_CHUNK_ELEMS = 1 << 24


def _tile_union_stats_multi_enumerated(views: RoutingViews, topology: Topology, BC: int,
                                      bt_candidates: tuple[int, ...]) -> dict[int, TileUnionStats]:
    """:class:`TileUnionStats` for EVERY candidate ``BT``, from ONE pass at the finest.

    **A TEST-ONLY REFERENCE IMPLEMENTATION. It must never feed a production
    decision**. It enumerates
    the FULL ``[BH, n_tiles, BT, owners, BC]`` leaf rectangle in owner chunks, so
    it costs ``Theta(BH * T * N)`` boolean work no matter how sparse the routing
    is -- which made ``select_plan`` 59-74 % of the layer at scale. The
    production path is :func:`_tile_union_stats_multi`, which counts the same
    statistics from the live entries alone. This one survives because it is
    written from the fold identity below and is therefore an independent second
    derivation to gate the sparse counter against.

    **The fold is an identity, not an approximation, and this is the derivation.**
    Fix an owner and a coarse tile of ``2*BT`` tokens, split into halves ``A``
    (first ``BT``) and ``B`` (second). Write ``live_r[t, i]`` / ``live_w[t, i]``
    for the per-token, per-leaf read/write liveness (the AND over the spanned
    levels' digits AND the fixed levels' token validity -- see
    :func:`tile_union_stats`), and per half define the four leaf BITMAPS

        U    [i] = OR_t (live_r | live_w)        popcount -> K
        Rany [i] = OR_t live_r
        Wany [i] = OR_t live_w
        I    [i] = OR_t (live_r[t,i] AND OR_{t' <= t} live_w[t',i])   popcount -> KI

    Then, over the coarse tile,

        U    = U_A | U_B                                   (an OR over tokens is associative)
        Rany = Rany_A | Rany_B
        Wany = Wany_A | Wany_B
        I    = I_A | I_B | (Rany_B AND Wany_A)

    ``U``/``Rany``/``Wany`` are plain token-ORs and fold trivially. ``I`` does not,
    because §5's causal prefix ``OR_{t' <= t} live_w`` CROSSES the half boundary:
    a leaf read in ``B`` counts if it was written anywhere in ``A``, which neither
    half's own ``I`` records. That carry is exactly one term, ``Rany_B AND Wany_A``,
    and adding it makes the fold exact -- which is why the fold has to be over
    bitmaps and cannot be over the counts ``K``/``KI`` (a popcount cannot be
    un-ORed).

    Tile CANDIDACY does not fold from the per-owner candidate flags either: it is
    an AND (over fixed levels) of ORs (over the boundary slice's digits and over
    the tile's tokens), and AND-of-OR does not commute with a further OR over
    tokens. It IS exact when folded one level down, on the per-``(level, digit)``
    TILE bitmaps -- ``r_tile_2BT[l, d] = r_tile_A[l, d] | r_tile_B[l, d]`` -- and
    that is what :meth:`RoutingViews.tile_live` does, forming the candidacy AND
    fresh at each candidate ``BT`` from the folded bitmaps.

    The padded token tail is inert on both counts: a padded token is exactly zero
    in every level, so it is dead in ``live_r``/``live_w`` and contributes no bit
    anywhere. ``ceil(ceil(T/BT)/2) == ceil(T/(2*BT))`` makes the tile count agree
    with a direct computation at the coarse ``BT``, so an odd fine tile count folds
    against an all-dead partner and lands on the right number of coarse tiles.

    Everything is accumulated as EXACT int64 counts and reduced to floats once, on
    the host, after a SINGLE device transfer -- the per-owner boolean-mask indexing
    the previous form used (``k.flatten()[cand]``) forced a synchronization per
    owner per side.
    """
    cands = _bt_candidate_chain(bt_candidates)
    BT0 = cands[0]

    geometry = owner_geometry(topology.widths, BC)
    j = geometry.first_spanned_level
    widths = topology.widths
    D = topology.D
    c = geometry.slice_digits
    n_slices = geometry.n_slices
    owners_total = geometry.owners_total
    BH, T = views.BH, views.T
    device = views.read_mask[0].device
    n_tiles0 = (T + BT0 - 1) // BT0

    def _tiled(mask: torch.Tensor) -> torch.Tensor:
        pad = n_tiles0 * BT0 - T
        if pad:
            mask = torch.nn.functional.pad(mask, (0, 0, 0, pad))
        return mask.view(BH, n_tiles0, BT0, mask.shape[-1])

    r = [_tiled(mask) for mask in views.read_mask]      # [BH, n_tiles0, BT0, width]
    w = [_tiled(mask) for mask in views.write_mask]

    # Owner decomposition, vectorized: the inverse of `OwnerGeometry.owner_digits`
    # evaluated for every owner at once, on device, so no Python loop is O(N/BC).
    owner_ids = torch.arange(owners_total, device=device)
    slice_ids = owner_ids % n_slices                                   # [owners]
    rest = owner_ids // n_slices
    fixed_digit = [(rest // max(1, prod(widths[level + 1:j]))) % widths[level]
                   for level in range(j)]                              # each [owners]

    # Leaf-local digit of level `l >= j`: `base_l + (i / stride_l) % m_l`, the same
    # (base, stride, modulus) triple `level_descriptors` hands the kernel -- now
    # broadcast over owners rather than recomputed per owner.
    local = torch.arange(BC, device=device)
    span_index = []
    for level in range(j, D):
        stride = max(1, prod(widths[level + 1:]))
        modulus = c if level == j else widths[level]
        digits = (local // stride) % modulus                           # [BC]
        if level == j:
            digits = slice_ids.unsqueeze(1) * c + digits.unsqueeze(0)  # [owners, BC]
        else:
            digits = digits.unsqueeze(0).expand(owners_total, BC)
        span_index.append(digits)

    # §4 candidacy is formed per candidate `BT` from that candidate's folded
    # per-(level, digit) tile bitmaps -- see the docstring on why it cannot be
    # folded from the per-owner flags.
    slice_any: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    tile_bitmaps: dict[int, tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]] = {}
    n_tiles_at: dict[int, int] = {}
    for BT in cands:
        r_tile, w_tile = views.tile_live(BT)
        tile_bitmaps[BT] = (r_tile, w_tile)
        n_tiles_at[BT] = r_tile[0].shape[1]
        slice_any[BT] = tuple(
            side[j].view(BH, n_tiles_at[BT], n_slices, c).any(dim=-1)  # [BH, n_tiles, n_slices]
            for side in (r_tile, w_tile))

    # Six exact int64 accumulators per candidate; ONE host transfer at the end.
    zero = lambda: torch.zeros((), dtype=torch.int64, device=device)  # noqa: E731
    acc = {BT: {name: zero() for name in
                ("sum_K", "sum_sat", "sum_cand", "sum_K_cand", "sum_KI_cand",
                 "sum_ks_cand", "sum_kis_cand")}
           for BT in cands}

    per_owner = max(1, BH * n_tiles0 * BT0 * BC)
    chunk = max(1, min(owners_total, _STATS_OWNER_CHUNK_ELEMS // per_owner))

    for lo in range(0, owners_total, chunk):
        hi = min(lo + chunk, owners_total)
        C = hi - lo

        live_r = live_w = None
        for k_level, level in enumerate(range(j, D)):
            index = span_index[k_level][lo:hi].reshape(-1)             # [C*BC]
            rl = r[level][..., index].view(BH, n_tiles0, BT0, C, BC)
            wl = w[level][..., index].view(BH, n_tiles0, BT0, C, BC)
            live_r = rl if live_r is None else (live_r & rl)
            live_w = wl if live_w is None else (live_w & wl)
        # FIXED levels -> per-token validity, broadcast across the BC leaf axis.
        for level in range(j):
            index = fixed_digit[level][lo:hi]                          # [C]
            live_r = live_r & r[level][..., index].unsqueeze(-1)
            live_w = live_w & w[level][..., index].unsqueeze(-1)

        # The four leaf bitmaps at the FINEST granularity: [BH, n_tiles0, C, BC].
        u = (live_r | live_w).any(dim=2)
        r_any = live_r.any(dim=2)
        w_any = live_w.any(dim=2)
        i_bits = (live_r & live_w.cummax(dim=2).values).any(dim=2)
        del live_r, live_w

        for step, BT in enumerate(cands):
            if step:
                factor = BT // cands[step - 1]
                for _ in range(factor.bit_length() - 1):
                    if u.shape[1] % 2:                                 # pad with a dead tile
                        u, r_any, w_any, i_bits = (
                            torch.nn.functional.pad(t, (0, 0, 0, 0, 0, 1))
                            for t in (u, r_any, w_any, i_bits))
                    a, b = slice(0, None, 2), slice(1, None, 2)
                    i_bits = (i_bits[:, a] | i_bits[:, b]) | (r_any[:, b] & w_any[:, a])
                    u = u[:, a] | u[:, b]
                    r_any = r_any[:, a] | r_any[:, b]
                    w_any = w_any[:, a] | w_any[:, b]
            assert u.shape[1] == n_tiles_at[BT], (
                f"fold landed on {u.shape[1]} tiles at BT={BT}, expected {n_tiles_at[BT]}")

            k_count = u.sum(dim=-1)                                    # [BH, n_tiles, C] int64
            ki_count = i_bits.sum(dim=-1)

            r_tile, w_tile = tile_bitmaps[BT]
            slice_r, slice_w = slice_any[BT]
            cand_r = slice_r[..., slice_ids[lo:hi]]                    # [BH, n_tiles, C]
            cand_w = slice_w[..., slice_ids[lo:hi]]
            for level in range(j):
                index = fixed_digit[level][lo:hi]
                cand_r = cand_r & r_tile[level][..., index]
                cand_w = cand_w & w_tile[level][..., index]
            cand = cand_r | cand_w

            bucket = acc[BT]
            bucket["sum_K"] += k_count.sum()
            bucket["sum_sat"] += (k_count == BC).sum()
            bucket["sum_cand"] += cand.sum()
            bucket["sum_K_cand"] += (k_count * cand).sum()
            bucket["sum_KI_cand"] += (ki_count * cand).sum()
            q = MMA_K_QUANTUM
            bucket["sum_ks_cand"] += (((k_count + q - 1) // q) * cand).sum()
            bucket["sum_kis_cand"] += (((ki_count + q - 1) // q) * cand).sum()

    order = [(BT, name) for BT in cands for name in
             ("sum_K", "sum_sat", "sum_cand", "sum_K_cand", "sum_KI_cand",
              "sum_ks_cand", "sum_kis_cand")]
    flat = torch.stack([acc[BT][name] for BT, name in order]).cpu().tolist()
    totals = {key: value for key, value in zip(order, flat)}

    out: dict[int, TileUnionStats] = {}
    for BT in cands:
        n_pairs = owners_total * BH * n_tiles_at[BT]
        n_cand = totals[(BT, "sum_cand")]
        out[BT] = TileUnionStats(
            BC=BC, BT=BT,
            mean_K=totals[(BT, "sum_K")] / n_pairs if n_pairs else 0.0,
            saturated_frac=totals[(BT, "sum_sat")] / n_pairs if n_pairs else 0.0,
            mean_K_candidate=totals[(BT, "sum_K_cand")] / n_cand if n_cand else 0.0,
            mean_KI_candidate=totals[(BT, "sum_KI_cand")] / n_cand if n_cand else 0.0,
            mean_ksteps_candidate=totals[(BT, "sum_ks_cand")] / n_cand if n_cand else 0.0,
            mean_kisteps_candidate=totals[(BT, "sum_kis_cand")] / n_cand if n_cand else 0.0,
            candidate_frac=totals[(BT, "sum_cand")] / n_pairs if n_pairs else 0.0,
            n_pairs=n_pairs,
            n_cohorts=owners_total * BH)
    return out


def rank_spans(topology: Topology, stats: tuple[LevelSupportStats, ...]) -> tuple[int, ...]:
    """Rank levels by §6.1's score, best (highest) first.

    **Structural decision recorded here, not applied implicitly.** Making a
    particular level set the owner's resident dimension requires *permuting* it
    innermost (§3: "level order is semantically free ... to span a particular set
    of levels, permute them innermost and take aligned blocks"). A permutation
    changes the mixed-radix leaf indexing, so it is a **config transform** applied
    to the ``Topology`` and to the routing tensors together --
    ``permuted_topology`` below -- never a hidden kernel behaviour. The ranking is
    advisory output of the planner; the kernel only ever sees the already-permuted
    topology and derives ``j``/``c`` from ``BC``.
    """
    if len(stats) != topology.D:
        raise ValueError(f"stats must contain one entry per level (D={topology.D}), got {len(stats)}")
    scored = [(span_score(stat, width), level) for level, (stat, width) in enumerate(zip(stats, topology.widths))]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(level for _, level in scored)


def permuted_topology(topology: Topology, order: tuple[int, ...]) -> Topology:
    """Reorder levels; ``order[k]`` is the source level placed at position ``k``.

    Position ``D-1`` is innermost (fastest-varying digit under the MSB-first
    mixed radix), so a span ranking is applied by placing the highest-scoring
    levels **last**. Callers must apply the same permutation to every per-level
    tensor (``read_levels``, ``write_levels``, decay dials) -- the helper does not
    see them, and silently permuting one side would be a correctness bug.
    """
    from rola.routing.types import LevelConfig  # local: avoid cycle at import time

    if sorted(order) != list(range(topology.D)):
        raise ValueError(f"order must be a permutation of range({topology.D}), got {order}")
    return Topology(levels=tuple(LevelConfig(width=topology.levels[src].width,
                                             routing=topology.levels[src].routing)
                                 for src in order),
                    d_v=topology.d_v)


# ---------------------------------------------------------------------------
# Bit packing helpers
# ---------------------------------------------------------------------------


def _pack_bits(flags: torch.Tensor) -> torch.Tensor:
    """Pack a trailing boolean axis into little-endian ``uint32`` bit patterns.

    Returns ``torch.int32`` holding the raw bit pattern (torch's uint32 support in
    2.6 does not cover the arithmetic used here); the CUDA side reinterprets it as
    ``uint32_t``. Bit ``k`` of word ``w`` is element ``32*w + k``.
    """
    *lead, n = flags.shape
    n_words = (n + _BITS_PER_WORD - 1) // _BITS_PER_WORD
    padded = torch.zeros((*lead, n_words * _BITS_PER_WORD), dtype=torch.bool, device=flags.device)
    padded[..., :n] = flags
    grouped = padded.reshape(*lead, n_words, _BITS_PER_WORD).to(torch.int64)
    weights = torch.ones((), dtype=torch.int64, device=flags.device) << torch.arange(
        _BITS_PER_WORD, device=flags.device, dtype=torch.int64)
    words = (grouped * weights).sum(dim=-1)
    # Reinterpret as signed int32 without changing the bit pattern.
    words = torch.where(words >= (1 << 31), words - (1 << 32), words)
    return words.to(torch.int32)


def _pack_token_bits(views: RoutingViews, BH: int, T: int, BT: int, n_tiles: int,
                     total_rows: int, level_offsets: list[int],
                     widths: tuple[int, ...]) -> torch.Tensor:
    """§4's support word: ``[2, BH, n_tiles, total_rows]`` int32, bit ``t`` = token ``tile*BT+t``.

    **Tile-major**, unlike the tile bitmap: one consumer visit reads one tile's
    slab, so a level's ``modulus`` digit words are adjacent and coalesce, and the
    per-level assignment below needs no transpose either.

    Side 0 is read, side 1 is write. Liveness is EXACT nonzero-ness of the
    amplitude (:class:`RoutingViews`' ``read_mask``/``write_mask``, spec §1.2) --
    never a threshold, and never a magnitude test on anything downstream of it.
    The padded token tail (``n_tiles*BT - T`` tokens) contributes no bit, exactly
    as it contributes none to the tile bitmap.

    Accumulated per (side, level) into a preallocated buffer rather than through
    :func:`_pack_bits` on one stacked boolean, for two reasons that both matter at
    the scale the standing rule designs for. (1) :func:`_pack_bits`' ``int64``
    intermediate would be the whole ``[2, BH, total_rows, n_tiles*BT]`` at once;
    this one's is a single level's. (2) It accumulates in **int32**, so the largest
    transient is exactly the size of one folded per-level tensor -- the weights
    carry bit 31 as ``-2**31`` and the summed powers of two are DISJOINT, so the
    partial sums lie in ``[-2**31, 2**31 - 1]`` and the int32 arithmetic is exact
    with no wraparound to argue about. Launch count is comparable to
    :func:`_pack_bits`', which is the axis the planner plane is bound on.
    """
    if not (1 <= BT <= _BITS_PER_WORD):
        raise ValueError(f"BT must be in [1, 32] (spec §6) to carry a token bit each, got {BT}")
    device = views.read_mask[0].device
    pad = n_tiles * BT - T
    weights = torch.ones((), dtype=torch.int64, device=device) << torch.arange(
        BT, device=device, dtype=torch.int64)
    weights = torch.where(weights >= (1 << 31), weights - (1 << 32), weights)
    weights = weights.to(torch.int32).view(1, 1, BT, 1)
    words = torch.zeros((2, BH, n_tiles, total_rows), dtype=torch.int32, device=device)
    for side, masks in enumerate((views.read_mask, views.write_mask)):
        for level, mask in enumerate(masks):
            width = widths[level]
            if pad:
                mask = torch.nn.functional.pad(mask, (0, 0, 0, pad))
            start = level_offsets[level]
            words[side, :, :, start:start + width] = (
                mask.reshape(BH, n_tiles, BT, width) * weights).sum(dim=2, dtype=torch.int32)
    return words


def unpack_bits(words: torch.Tensor, n: int) -> torch.Tensor:
    """Inverse of :func:`_pack_bits`, for tests and the CPU reference walk."""
    as_u64 = words.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(_BITS_PER_WORD, device=words.device, dtype=torch.int64)
    bits = ((as_u64.unsqueeze(-1) >> shifts) & 1).to(torch.bool)
    flat = bits.reshape(*words.shape[:-1], -1)
    return flat[..., :n]


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecayStream:
    """The LeafMassDecay artifacts a consumer launch needs, built ONCE per routing.

    `build_clock_stream` + `leaf_rate_dials` + `panel_exponent_bound` are a
    per-CALL cost of the decay arm and are pure functions of the write-side routing
    tensors, the `LeafMassDecay` and `BT`. `_consumer_forward` used to rebuild all
    three on EVERY call, including calls handed an already-built `schedule=` --
    measured at **0.78-1.40 ms per call** on the cells in
    `benchmarks/ops/benchmark_rola_v3_new_consumer.py`, which is a pure waste term
    on every decay launch that reuses a schedule.

    It is a SEPARATE object carried BY the schedule rather than a field built with
    it, because the clock is deliberately a separate builder from the schedule: the
    STORED write tensor must never key a tile bitmap (P5.20.5 finding 2), so the
    schedule cannot be allowed to depend on it.

    ---- WHY THE VALIDITY KEY IS IDENTITY-PLUS-VERSION AND HOLDS STRONG REFERENCES --

    `_source_key`'s `(data_ptr, shape, stride, dtype)` key is documented as sound
    ONLY under the discipline that nothing holds a view across more than one forward
    call, and it names the two gaps that discipline excludes: (a) an in-place
    mutation after construction leaves the key unchanged, and (b) the caching
    allocator can reissue a freed address, so a stale key can collide with an
    unrelated later tensor.

    **Caching across calls is exactly the discipline that key relies on being
    absent**, so this record does not use it. Instead it holds STRONG REFERENCES to
    the tensor objects it was derived from and validates with `is` identity plus
    `Tensor._version`:

    * holding the reference makes gap (b) UNREACHABLE -- the allocator cannot
      reissue an address whose tensor is still alive;
    * `_version` is torch's own in-place-mutation counter and closes gap (a).

    The cost of holding the references is that a cached schedule pins the write-side
    routing tensors alive. That is the correct trade: they are the tensors the
    schedule already describes, and a schedule outliving its routing is not a
    supported use.
    """

    clock_amps: tuple[torch.Tensor, ...]
    dials: torch.Tensor
    panel_bound: float | None
    src_stored: tuple[torch.Tensor, ...]
    src_write: tuple[torch.Tensor, ...]
    src_versions: tuple[int, ...]
    decay: object
    BT: int

    @staticmethod
    def versions(stored, write) -> tuple[int, ...]:
        return tuple(t._version for t in (*stored, *write))

    def matches(self, stored, write, decay, BT) -> bool:
        """Every component is necessary; none is a heuristic."""
        return (decay is self.decay and BT == self.BT
                and len(stored) == len(self.src_stored)
                and len(write) == len(self.src_write)
                and all(a is b for a, b in zip(stored, self.src_stored))
                and all(a is b for a, b in zip(write, self.src_write))
                and self.versions(stored, write) == self.src_versions)


@dataclass(frozen=True, slots=True)
class Schedule:
    """The §4 schedule plus the token-major route stream the consumer loads.

    ``read_tile_bits``/``write_tile_bits``: ``[BH, sum_l width_l, n_words]``
    ``int32`` bit patterns, level ``l``'s digits at rows
    ``[level_offsets[l], level_offsets[l] + width_l)``. Bit ``k`` of word ``w`` is
    token tile ``32*w + k``.

    ``read_token_bits``/``write_token_bits``: ``[BH, n_tiles, sum_l width_l]``
    ``int32`` -- **§4's SUPPORT WORD, materialized** (conformance audit item 3,
    S1). Same level-digit row indexing as the tile bitmaps but TILE-MAJOR, because
    one consumer visit reads exactly one tile's slab; bit ``t`` of entry
    ``[bh, tile, row]`` is token ``tile*BT + t`` (``BT <= 32``, spec §6). It is
    exact nonzero-ness of the routing amplitude, from :class:`RoutingViews`'
    ``read_mask``/``write_mask`` -- the SAME bits the tile bitmap is a reduction
    of, so ``read_tile_bits`` is now literally derived from it here rather than
    computed on a second path.

    **Orientation, and the §4 sentence it does not match literally.** §4's table
    gives the spanned level the RAW orientation, ``token -> digit support word``.
    What is emitted is its per-tile TRANSPOSE, ``(level, digit) -> token word``,
    for two reasons that are §4's own stated criterion ("each matching its access
    pattern"): (1) the consumer's gather ANDs one word per (level, digit) per leaf
    (``O(D)`` per leaf), whereas a token-major word would force a per-(leaf, token)
    bit test (``O(BT)`` per leaf); (2) an owner spans an ALIGNED SUB-RANGE of the
    boundary level's digits, so a digit-major row lets it read exactly its own
    ``gram_cols`` and no more, while a token-major word is packed over the level's
    FULL width and would be read and masked. Recorded as a spec-text reconciliation
    (same class as S2), not as a silent divergence.

    ``read_slice_union``/``write_slice_union``: ``[BH, n_slices, n_words]`` -- the
    OR over the ``c`` boundary-level digits of each aligned sub-range (§4's
    "``b_j / c`` entries"). This is the only union the owner cannot recover with a
    single row read.

    ``read_amps``/``write_amps``: one ``[BH, n_tiles, BT, pitch_l]`` tensor per
    level, token-major and contiguous per tile (§5 "Staging"). A nonzero entry *is*
    a live token (§4), so no separate token mask is stored.

    ``owners``/``owner_costs``: active owners sorted by estimated cost, descending.
    An owner with no candidate tile on either side is absent entirely.

    **Per-owner candidate bitmaps are deliberately NOT a field.** §4: the
    intersection "is not materialized per owner in memory (Pi-shaped, one reader
    each, strictly more traffic than recomputing)". The builder forms them
    transiently for the cost sort and drops them.
    """

    plan: Plan
    BH: int
    T: int
    n_tiles: int
    n_words: int
    level_offsets: tuple[int, ...]
    read_tile_bits: torch.Tensor
    write_tile_bits: torch.Tensor
    read_token_bits: torch.Tensor
    write_token_bits: torch.Tensor
    read_slice_union: torch.Tensor
    write_slice_union: torch.Tensor
    read_amps: tuple[torch.Tensor, ...]
    write_amps: tuple[torch.Tensor, ...]
    g_read: torch.Tensor | None
    g_write: torch.Tensor
    owners: torch.Tensor
    owner_costs: torch.Tensor
    abi_version: int = ROLA_SCHEDULE_ABI_VERSION
    #: Attached lazily by `_consumer_forward` on the decay arm; see
    #: :class:`DecayStream`. `None` on the `decay=None` arm, always.
    decay_stream: DecayStream | None = None

    def decay_artifacts(self, stored, write, decay, BT, *, build):
        """Return this schedule's decay artifacts, building them at most once.

        `build()` is called only on a miss. The result is attached to this frozen
        instance with `object.__setattr__` -- deliberate and confined to this one
        method: the record is a MEMO of a pure function of inputs the schedule
        already describes, so attaching it changes no observable value, and
        `DecayStream.matches` is what makes that claim checkable rather than
        assumed.
        """
        cached = self.decay_stream
        if cached is not None and cached.matches(stored, write, decay, BT):
            return cached
        fresh = build()
        object.__setattr__(self, "decay_stream", fresh)
        return fresh

    def schedule_bytes(self) -> int:
        """Bitmap-only footprint -- the quantity §4's O(L*N) assertion is about."""
        return sum(t.numel() * t.element_size() for t in (
            self.read_tile_bits, self.write_tile_bits,
            self.read_token_bits, self.write_token_bits,
            self.read_slice_union, self.write_slice_union))

    def stream_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (*self.read_amps, *self.write_amps))


def _fold_bth(tensor: torch.Tensor) -> torch.Tensor:
    """``[B, T, H, *] -> [B*H, T, *]`` (the convention every routing module uses)."""
    return tensor.permute(0, 2, 1, 3).reshape(
        tensor.shape[0] * tensor.shape[2], tensor.shape[1], *tensor.shape[3:])


def _tile_live(mask: torch.Tensor, BT: int, n_tiles: int) -> torch.Tensor:
    """``[BH,T,width]`` bool mask ``-> [BH,n_tiles,width]`` bool: any live token in the tile.

    Liveness is exact nonzero-ness of the amplitude, never a threshold: entmax's
    exact support zeros are the scheduling authority (spec §1.2), and a
    tolerance here would either invent or destroy support. The ``!= 0`` that makes
    the mask lives in :class:`RoutingViews`, which is where both callers now get
    it from; this function is the tile reduction only.
    """
    BH, T, width = mask.shape
    pad = n_tiles * BT - T
    if pad:
        mask = torch.nn.functional.pad(mask, (0, 0, 0, pad))
    return mask.view(BH, n_tiles, BT, width).any(dim=2)


def _tile_major_amps(level: torch.Tensor, BT: int, n_tiles: int, pitch: int,
                     dtype: torch.dtype) -> torch.Tensor:
    """``[BH,T,width] -> [BH,n_tiles,BT,pitch]``, token-major, zero padded."""
    BH, T, width = level.shape
    out = level.new_zeros((BH, n_tiles * BT, pitch), dtype=dtype)
    out[:, :T, :width] = level.to(dtype)
    return out.view(BH, n_tiles, BT, pitch).contiguous()


def build_schedule(
    read_levels: tuple[torch.Tensor, ...],
    write_levels: tuple[torch.Tensor, ...],
    g_read: torch.Tensor | None,
    g_write: torch.Tensor,
    plan: Plan,
    views: RoutingViews | None = None,
) -> Schedule:
    """Build the §4 schedule from the producer's per-level routing allocations.

    Inputs are exactly ``naive_rola``'s routing inputs (``[B,T,H,width_l]`` per
    level, ``[B,T,H]`` gains) so the builder and the canonical oracle consume one
    contract and cannot drift.

    **Write liveness is derived from ``write_levels`` (the live deposit), never
    from ``write_levels_stored`` (the clock).** `GRADIENT_PROOFS_WOLFRAM.md`
    P5.20.5 finding 2 is binding: a storage path may flush a small live
    ``p_write`` to a stored zero, giving ``c == 0`` (``keep == 1``) with a live
    deposit ``W != 0``. Such a leaf is touched and must appear in the write
    support; a clock-keyed summary would silently drop the deposit. This function
    therefore does not accept ``write_levels_stored`` at all -- the wrong tensor
    is not merely discouraged, it is unreachable.

    ``views`` is an optional :class:`RoutingViews`, normally the SAME one the
    plan was selected from (:func:`plan_and_schedule`), carrying the folded
    tensors and the §4 tile bitmaps at ``plan.BT`` -- both of which the selector
    has already computed over these exact tensors. It changes what is RECOMPUTED,
    never what is built: the fold, the ``!= 0`` and the tile reduction are the same
    operations on the same inputs either way, and the shape validation below runs
    identically in both cases.
    """
    topology = plan.topology
    widths = topology.widths
    if len(read_levels) != topology.D or len(write_levels) != topology.D:
        raise ValueError(f"read/write levels must contain one tensor per level (D={topology.D})")
    B, T, H, _ = read_levels[0].shape
    BH = B * H
    BT = plan.BT
    n_tiles = (T + BT - 1) // BT
    n_words = (n_tiles + _BITS_PER_WORD - 1) // _BITS_PER_WORD
    geometry = plan.geometry

    if views is None:
        views = routing_views(read_levels, write_levels, topology)
    else:
        views.require_source(read_levels, write_levels)
    read_f, write_f = views.read_folded, views.write_folded
    for name, levels in (("read_levels", read_f), ("write_levels", write_f)):
        for level, (tensor, width) in enumerate(zip(levels, widths)):
            if tensor.shape != (BH, T, width):
                raise ValueError(
                    f"{name}[{level}] must fold to [{BH},{T},{width}], got {tuple(tensor.shape)}")

    read_live, write_live = views.tile_live(BT)                              # [BH,n_tiles,w]

    level_offsets = []
    offset = 0
    for width in widths:
        level_offsets.append(offset)
        offset += width
    total_rows = offset

    j = geometry.first_spanned_level
    c = geometry.slice_digits
    n_slices = geometry.n_slices

    # §4's SUPPORT WORD, materialized (conformance item 3 / S1). One `int32` per
    # (side, bh, level-digit row, token tile); bit `t` is token `tile*BT + t`.
    # `BT <= 32` is spec §6 and is asserted rather than assumed, because the whole
    # object is a bit per token of a tile.
    #
    # THE TILE BITMAP IS NOW A REDUCTION OF THIS, not a second path. `harvest.md` §23.2's
    # verification is the reason it can be: "the schedule computes [token-granular
    # exact support] and then discards the token axis", so keeping the axis is an
    # ABI addition and not new information -- and deriving `tile_bits` from
    # `token_bits` makes "a tile is live iff some token in it is live" true BY
    # CONSTRUCTION rather than by two agreeing reductions.
    token_bits = _pack_token_bits(views, BH, T, BT, n_tiles, total_rows,
                                  level_offsets, widths)
    read_token_bits, write_token_bits = token_bits[0].contiguous(), token_bits[1].contiguous()
    # `[2,BH,n_tiles,total_rows] -> [2,BH,total_rows,n_words]`: a tile is live iff
    # some token in it is, so this is the tile bitmap BY CONSTRUCTION rather than by
    # a second reduction that has to be argued to agree.
    tile_bits = _pack_bits((token_bits != 0).permute(0, 1, 3, 2))
    read_tile_bits, write_tile_bits = tile_bits[0].contiguous(), tile_bits[1].contiguous()

    # [BH,n_tiles,width_j] -> [BH,n_slices,n_tiles] -> packed, both sides at once.
    grouped = torch.stack(
        [live[j].permute(0, 2, 1).reshape(BH, n_slices, c, n_tiles).any(dim=2)
         for live in (read_live, write_live)])                      # [2,BH,n_slices,n_tiles]
    slice_union = _pack_bits(grouped)
    read_slice_union = slice_union[0].contiguous()
    write_slice_union = slice_union[1].contiguous()

    read_amps = tuple(
        _tile_major_amps(level, BT, n_tiles, plan.amplitude_pitch(width), plan.amp_dtype)
        for level, width in zip(read_f, widths))
    write_amps = tuple(
        _tile_major_amps(level, BT, n_tiles, plan.amplitude_pitch(width), plan.amp_dtype)
        for level, width in zip(write_f, widths))

    # `grouped` is exactly the per-(slice, tile) boundary liveness the owner sort
    # needs; it is handed over rather than recomputed there.
    owners, owner_costs = _sort_owners(
        read_live, write_live, geometry, plan, n_tiles=n_tiles,
        slice_live=(grouped[0], grouped[1]))

    g_read_f = None if g_read is None else _fold_bth(g_read.unsqueeze(-1)).squeeze(-1).contiguous()
    g_write_f = _fold_bth(g_write.unsqueeze(-1)).squeeze(-1).contiguous()

    return Schedule(
        plan=plan, BH=BH, T=T, n_tiles=n_tiles, n_words=n_words,
        level_offsets=tuple(level_offsets),
        read_tile_bits=read_tile_bits, write_tile_bits=write_tile_bits,
        read_token_bits=read_token_bits, write_token_bits=write_token_bits,
        read_slice_union=read_slice_union, write_slice_union=write_slice_union,
        read_amps=read_amps, write_amps=write_amps,
        g_read=g_read_f, g_write=g_write_f,
        owners=owners, owner_costs=owner_costs)


def build_clock_stream(
    write_levels_stored: tuple[torch.Tensor, ...],
    write_levels: tuple[torch.Tensor, ...],
    plan: Plan,
) -> tuple[torch.Tensor, ...]:
    """Stage the ``LeafMassDecay`` **clock** amplitudes, and nothing else.

    Returns one ``[BH, n_tiles, BT, pitch_l]`` tensor per level, laid out exactly
    like ``Schedule.write_amps`` so the kernel walks it with the same
    ``(base, stride, modulus)`` triple. The clock is ``c[t,s] = prod_l
    p_write_stored[t, d_l(s)]`` (spec §2) -- a pure product of the *stored* write
    allocations with **no side gain**: ``g_write`` scales the deposit ``W``, not
    the clock.

    **This is a separate function from :func:`build_schedule` on purpose, and
    the separation is the enforcement mechanism for `GRADIENT_PROOFS_WOLFRAM.md`
    P5.20.5 finding 2.** That finding binds tile liveness to the *live* deposit
    ``write_levels``, never to the stored clock: a storage path may flush a small
    live ``p_write`` to a stored zero, giving ``c == 0`` (``keep == 1``) with a
    live deposit ``W != 0``, and a clock-keyed bitmap would silently drop that
    deposit. Because the stored tensor never enters ``build_schedule``'s scope
    at all, keying a bitmap on it is not merely forbidden, it is unreachable --
    which is a stronger guarantee than a comment, and it survives refactors.
    ``write_levels`` (the live allocation) is accepted HERE, by the clock
    builder, purely so it can be checked against; that is the allowed direction
    (the clock builder may see ``live``) and is not the forbidden one (the
    bitmap builder seeing ``stored``).

    The converse direction is what makes tile skipping still exact under decay:
    quantization can only flush a value toward zero, never create a nonzero from
    a zero, so ``stored != 0 => live != 0``, hence the clock-live tile set is a
    SUBSET of the write-live tile set. Outside a write-live tile every leaf of the
    owner has ``c = 0`` and therefore ``keep = 1`` exactly -- §2's
    write-conditioned quiescence, which is why skipping is bit-identical rather
    than approximately right.

    **This premise is now a checked runtime precondition, not an assumption
    (review B2).** ``stored != 0 => live != 0`` is enforced below by exact-zero
    comparison over every level, elementwise, before staging. A caller whose
    storage path can manufacture a nonzero from a zero (stochastic rounding,
    additive noise, a learned quantizer with a nonzero zero-point, ...) is
    rejected loudly here rather than silently producing an O(1) survival error
    on a path with no other equivalence check (spec §2.1).
    """
    topology = plan.topology
    widths = topology.widths
    stored = tuple(write_levels_stored)
    live = tuple(write_levels)
    if len(stored) != topology.D:
        raise ValueError(
            f"write_levels_stored must contain one tensor per level (D={topology.D}), "
            f"got {len(stored)}")
    if len(live) != topology.D:
        raise ValueError(
            f"write_levels must contain one tensor per level (D={topology.D}), "
            f"got {len(live)}")
    for level, (stored_level, live_level) in enumerate(zip(stored, live)):
        if stored_level.shape != live_level.shape:
            raise ValueError(
                f"write_levels_stored[{level}].shape={tuple(stored_level.shape)} must match "
                f"write_levels[{level}].shape={tuple(live_level.shape)}")
        violation = (stored_level != 0) & (live_level == 0)
        if torch.any(violation):
            n_bad = int(violation.sum())
            raise ValueError(
                f"write_levels_stored[{level}] has {n_bad} entry(ies) that are nonzero while "
                "the corresponding write_levels (live) entry is exactly zero. The bit-identical "
                "tile-skip argument in this docstring requires 'stored != 0 => live != 0' -- "
                "quantization may flush a live value TOWARD zero, never manufacture a nonzero "
                "FROM zero. This storage path violates that (stochastic rounding, additive "
                "noise, and a learned quantizer with a nonzero zero-point are the usual "
                "causes); it is rejected rather than silently dropping the leaf's survival to "
                "an O(1)-wrong value on the one path with no equivalence-based validation.")
    B, T, H, _ = stored[0].shape
    BT = plan.BT
    n_tiles = (T + BT - 1) // BT
    folded = tuple(_fold_bth(level) for level in stored)
    for level, (tensor, width) in enumerate(zip(folded, widths)):
        if tensor.shape != (B * H, T, width):
            raise ValueError(
                f"write_levels_stored[{level}] must fold to [{B * H},{T},{width}], "
                f"got {tuple(tensor.shape)}")
    return tuple(
        _tile_major_amps(level, BT, n_tiles, plan.amplitude_pitch(width), plan.amp_dtype)
        for level, width in zip(folded, widths))


#: The largest per-leaf reference exponent ``panel`` will accept, in units of
#: ``|log keep|`` (i.e. of a natural log). See :func:`panel_exponent_bound` for
#: what is bounded and why the bound has to exist at all.
#:
#: **Where the number comes from.** Inside the domain every quantity the lowering
#: forms is bounded by ``exp(PANEL_MAX_EXPONENT)``: a read-panel entry, a
#: write-panel entry, a ``BT``-term deposit sum (``<= 32 * e^40``), and the
#: pairwise products the MMA takes of them (``e^80 ~ 5.5e34``, against fp32's
#: ``3.4e38``). It is deliberately far below the kernel's own §12 rule 3 clamp at
#: ``88``: the clamp is the LAST-RESORT finiteness guard and this is the value at
#: which the clamp is provably unreachable, which is what makes the clamp a
#: structural guarantee rather than a silent approximation. Every realistic cell
#: sits one to two orders of magnitude below it (measured bounds on the gate
#: fixtures are 1e-3 to ~15), so the threshold is not a tuning knob for
#: performance -- it is where "the two separately exponentiated panels stop being
#: an exact rewrite" begins.
PANEL_MAX_EXPONENT = 40.0


def panel_exponent_bound(clock_amps: tuple[torch.Tensor, ...], decay,
                         BT: int) -> float:
    """An EXACT UPPER BOUND on ``panel``'s largest per-leaf reference exponent.

    ``intra_panel.cuh`` forms ``keep_i^{+(P[t,i]-P_ref[i])}`` and
    ``keep_i^{-(P[j,i]-P_ref[i])}`` as two SEPARATE exponentials, with §12 rule
    1's minimax reference ``P_ref[i] = P[BT-1,i]/2``. The largest exponent either
    of them takes, over one tile, is therefore

        E_max = max_{i, tile} P_ref[i] * |log_keep[i]|
              = max_{i, tile} (1/2) * (sum_{t in tile} c[t,i]) * (-log1p(-rate_i))

    and this function bounds it from above **without materializing anything of
    size N**::

        prod_l c_l[t, d_l(i)]  <=  prod_l max_d c_l[t, d]        (per token)
        rate_i = prod_l delta_l[d_l(i)]  <=  prod_l max_d delta_l[d]

    -- two independent maxima, each over one level's own axis, so the cost is
    ``O(T * sum_l width_l)`` and the answer is conservative (it can refuse a
    launch that would have been fine; it can never admit one that is not).

    **Why a bound is required rather than a caveat.** Rules 1 and 3 are "a pair,
    not alternatives" (spec §7.4) and the pair is still not sufficient without
    one: past ``E_max ~ 88`` the write factor saturates at the clamp while the
    read factor underflows toward zero, and their product -- which is ``O(1)``
    when ``t`` is close to ``j`` -- comes out as ``~0``. That is a WRONG answer,
    not a rounding one, and it is exactly the class this design refuses to leave
    to a clamp. The driver compares this bound against
    :data:`PANEL_MAX_EXPONENT` and refuses the launch above it, so the domain is
    a checked precondition of the lowering rather than a documented hazard.

    This is a PLAN/DRIVER-time quantity computed from the realized routing, in
    the sanctioned §7.4.1 pattern: no kernel path reads it, and nothing branches
    on it at runtime.
    """
    if not clock_amps:
        raise ValueError("panel_exponent_bound needs the staged clock stream")
    rate_max = 1.0
    for dial in decay.dials:
        rate_max *= float(dial.max())
    if not rate_max < 1.0:
        # `leaf_rate_dials` rejects a dial that rounds to 1.0f, so this is the
        # product of several sub-unit maxima reaching 1 -- unreachable in
        # practice, and infinity is the honest bound rather than a log domain
        # error.
        return float("inf")
    g_max = -log1p(-rate_max)
    # Per-token upper bound on the leaf clock, then the per-TILE sum (the prefix
    # is over one tile, never across tiles: `P` restarts every tile because the
    # reference frame is entered and left inside it).
    per_token = None
    for amps in clock_amps:            # [BH, n_tiles, BT, pitch], zero padded
        level_max = amps.amax(dim=-1)  # [BH, n_tiles, BT]
        per_token = level_max if per_token is None else per_token * level_max
    if per_token.shape[-1] != BT:
        raise ValueError(
            f"the staged clock stream has {per_token.shape[-1]} tokens per tile but the "
            f"plan says BT={BT}; the bound is over one tile's prefix, so the two must agree")
    p_hat = float(per_token.sum(dim=-1).max()) if per_token.numel() else 0.0
    return 0.5 * p_hat * g_max


def leaf_rate_dials(decay, topology: Topology) -> torch.Tensor:
    """Flatten ``LeafMassDecay.dials`` to ``[H, sum_l width_l]``, rows at ``level_offsets``.

    The leaf rate is ``rate[s] = prod_l delta_l[d_l(s)]`` (spec §2) -- a plain
    product of **dials**, not an AND over per-branch retentions as in V2. It is
    accumulated by the kernel in fp32 and never in bf16: ``rate`` near 1 makes
    ``1 - rate`` catastrophically cancelling, and ``log1p(-rate)`` cannot repair a
    cancellation that already happened in the product (§7.4, this workflow's
    numerics audit).

    **Cast-boundary re-validation (fix pass, review B1).**
    ``LeafMassDecay.__post_init__`` enforces ``0 <= delta < 1`` in the dial
    tensor's OWN dtype (``types.py``). That is not the dtype the kernel reads: an
    fp64 dial in ``[1 - 3e-8, 1)`` is strictly ``< 1`` in fp64 and passes, but
    round-to-nearest at the ``.to(torch.float32)`` cast below lands it on exactly
    ``1.0f``. A leaf rate of exactly 1 makes ``log1pf(-1.0f) = -inf`` in the
    kernel, and any leaf whose clock is exactly zero at a token then computes
    ``expf(0 * -inf) = NaN`` -- the strict-upper-bound invariant the kernel's
    survival math depends on is checked in the wrong dtype. The contract is
    "the dtype the kernel consumes", so it is re-checked HERE, after the cast,
    rather than loosened or floored: this is not a fallback, it is the same
    invariant enforced at the boundary where it actually has to hold.
    """
    dials = tuple(decay.dials)
    if len(dials) != topology.D:
        raise ValueError(
            f"decay.dials must contain one tensor per level (D={topology.D}), got {len(dials)}")
    cast = [dial.to(torch.float32) for dial in dials]
    for level, (original, rounded) in enumerate(zip(dials, cast)):
        if torch.any(rounded >= 1.0):
            bad = int(torch.sum(rounded >= 1.0))
            worst = float(original[rounded >= 1.0].max())
            raise ValueError(
                f"LeafMassDecay.dials[{level}] has {bad} entry(ies) that round to exactly "
                f"1.0f under the fp32 cast the kernel consumes (worst fp64 value: {worst!r}), "
                "even though every entry satisfies the strict '0 <= delta < 1' invariant in "
                "its own (fp64) dtype. This is fp64->fp32 round-to-nearest, not a bug in the "
                "input: any dial within roughly 3e-8 of 1 rounds up. rate == 1.0f makes "
                "log1pf(-rate) == -inf in the kernel and NaNs any leaf with a zero clock at a "
                "write-candidate token (v3_consumer.cu). Move the dial strictly below "
                "1 - 3e-8 in fp64 (or otherwise keep it < 1.0f after casting to fp32); this is "
                "not silently clamped or nudged because that would be a hidden fallback.")
    return torch.cat(cast, dim=1).contiguous()


def _sort_owners(
    read_live: tuple[torch.Tensor, ...],
    write_live: tuple[torch.Tensor, ...],
    geometry: OwnerGeometry,
    plan: Plan,
    *,
    n_tiles: int,
    slice_live: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enumerate active owners and sort them by estimated cost, descending.

    **No ``O(L*N)`` or ``O(N)`` object is ever formed.** Owners are enumerated over
    the *Cartesian product of per-level live digit sets* on the fixed levels, so
    the enumeration is bounded by the routing support actually present, not by the
    leaf capacity ``N``. A level with no live digit at all makes the whole product
    empty and the schedule launches nothing.

    Candidate tile bitmaps are formed transiently, per owner, and dropped: §4
    forbids materializing them.

    The enumeration is Python (it must be -- it is a Cartesian product over live
    digit sets, whose *size* is data-dependent), but the per-owner tile counting is
    a single batched reduction. The distinction matters: the earlier form counted
    one owner at a time with a ``int(tensor.sum())``, which is a device
    synchronization per owner, and measured **28.5 ms of a 28.8 ms schedule
    build** on the Stage 0 cell (64 owners) -- 13x the consumer kernel it was
    scheduling. Nothing about the algorithm changed here; only the number of
    synchronizations, from ``O(owners)`` to ``O(1)``.

    **Host-transfer discipline (planner increment, 2026-07-30).** The enumeration
    is host work by construction, so what it must not do is pay a per-element ATen
    call or a per-level ``H2D`` on top of it. The liveness array is read as ONE
    ``.tolist()`` rather than ``BH * width_l`` scalar ``bool(tensor[bh, d])``
    indexings, and the whole per-owner index block (``bh``, the fixed digits, the
    mixed-radix rest) is assembled as one CPU tensor and transferred once instead
    of once per level per side. The enumeration ORDER -- ``(bh, fixed-digit
    lexicographic, slice)`` -- is byte-for-byte what it was, which matters because
    the sort is ``stable`` and a reorder would silently change the launch on cost
    ties.
    """
    device = read_live[0].device
    BH = read_live[0].shape[0]
    j = geometry.first_spanned_level
    widths = geometry.widths

    # Per (bh, level, digit) tile-liveness on either side; a fixed level's digit is
    # a candidate if it is live on either side (a write-only owner still runs).
    # One transfer for the whole array rather than one per (bh, level, digit).
    any_live = [
        (read_live[level] | write_live[level]).any(dim=1).tolist()   # [BH][width]
        for level in range(j)
    ]

    # Enumeration order is (bh, fixed-digit lexicographic, slice) and is preserved
    # exactly: the owner sort is `stable`, so a different enumeration order would
    # silently reorder cost ties and change the launch.
    bh_ids: list[int] = []
    fixed_digits: list[tuple[int, ...]] = []
    for bh in range(BH):
        live_digits = [
            [d for d in range(widths[level]) if any_live[level][bh][d]]
            for level in range(j)
        ]
        if any(len(digits) == 0 for digits in live_digits):
            continue
        for fixed in _cartesian(live_digits):
            bh_ids.append(bh)
            fixed_digits.append(tuple(fixed))

    n_slices = geometry.n_slices
    if not bh_ids:
        empty_i = torch.zeros((0, 2), dtype=torch.int32, device=device)
        return empty_i, torch.zeros(0, dtype=torch.float32, device=device)

    # ONE host->device transfer for the whole index block: column 0 is `bh`,
    # columns 1..j are the fixed digits, column j+1 is the mixed-radix rest.
    radix = [prod(widths[level + 1:j]) for level in range(j)]
    block = torch.tensor(
        [[bh, *fixed, sum(fixed[level] * radix[level] for level in range(j))]
         for bh, fixed in zip(bh_ids, fixed_digits)], dtype=torch.int64).to(device)
    bh_idx = block[:, 0]                                                 # [K]
    fixed_idx = [block[:, 1 + level] for level in range(j)]
    rests = block[:, j + 1]

    def _counts(live: tuple[torch.Tensor, ...], boundary: torch.Tensor) -> torch.Tensor:
        # `boundary` is the caller's [BH, n_slices, n_tiles] slice-union liveness,
        # which the schedule builder has already formed for `*_slice_union`.
        mask = torch.ones((len(bh_ids), n_tiles), dtype=torch.bool, device=device)
        for level in range(j):
            mask &= live[level][bh_idx, :, fixed_idx[level]]             # [K, n_tiles]
        # [K, 1, n_tiles] & [K, n_slices, n_tiles] -> [K, n_slices]
        return (mask.unsqueeze(1) & boundary[bh_idx]).sum(dim=2)

    read_counts = _counts(read_live, slice_live[0])
    write_counts = _counts(write_live, slice_live[1])
    costs_all = read_counts.float() + plan.write_cost_weight * write_counts.float()
    active = (read_counts > 0) | (write_counts > 0)                      # [K, n_slices]

    # `owner = mixed_radix(fixed digits) * n_slices + slice` (OwnerGeometry's
    # leaf-base identity), formed for every (K, slice) pair and then masked.
    slice_ids = torch.arange(n_slices, dtype=torch.int64, device=device)
    owner_all = rests.unsqueeze(1) * n_slices + slice_ids.unsqueeze(0)    # [K, n_slices]
    bh_all = bh_idx.unsqueeze(1).expand(-1, n_slices)

    # ONE synchronization for the whole selection. `nonzero` returns the live flat
    # positions in ascending order, so gathering by them is IDENTICAL to the
    # boolean-mask selects this replaces -- but the mask form ran `nonzero`
    # internally once per selected tensor (three here) and `bool(flat.any())` was a
    # fourth. The empty case is read off the index tensor, which is already resident
    # on the host side of that one sync.
    active_index = active.reshape(-1).nonzero(as_tuple=True)[0]
    if active_index.numel() == 0:
        empty_i = torch.zeros((0, 2), dtype=torch.int32, device=device)
        return empty_i, torch.zeros(0, dtype=torch.float32, device=device)
    costs = costs_all.reshape(-1)[active_index]
    order = torch.argsort(costs, descending=True, stable=True)
    pairs = torch.stack((bh_all.reshape(-1)[active_index].to(torch.int32),
                         owner_all.reshape(-1)[active_index].to(torch.int32)), dim=1)
    return pairs[order].contiguous(), costs[order].contiguous()


def _cartesian(sets: list[list[int]]):
    if not sets:
        yield ()
        return
    yield from _iter_product(*sets)


__all__ = [
    "COST_MODEL_PROVENANCE",
    "CP_ASYNC_ROW_BYTES",
    "DECAY_KSTEP_WORK",
    "DEFAULT_SM_COUNT",
    "DEFAULT_SM_SHARED_BYTES",
    "DEFAULT_WRITE_COST_WEIGHT",
    "GATHER_CANDIDATE_FIXED_WORK",
    "GATHER_CANDIDATE_QUANTUM_WORK",
    "INTRA_GRAM_KSTEP_WORK",
    "MMA_K_QUANTUM",
    "MMA_PRECISIONS",
    "PANEL_MAX_EXPONENT",
    "READOUT_FOLD_KSTEP_WORK",
    "ROLA_SCHEDULE_ABI_VERSION",
    "TILE_FIXED_WORK",
    "LevelSupportStats",
    "OwnerGeometry",
    "Plan",
    "PlanAxes",
    "RoutingViews",
    "Schedule",
    "TileUnionStats",
    "build_clock_stream",
    "build_schedule",
    "leaf_rate_dials",
    "owner_geometry",
    "panel_exponent_bound",
    "permuted_topology",
    "plan_and_schedule",
    "plan_cost_per_token",
    "plan_waves",
    "rank_spans",
    "routing_views",
    "select_plan",
    "select_plan_axes",
    "span_score",
    "spanned_moduli",
    "tile_union_stats",
    "unpack_bits",
]
