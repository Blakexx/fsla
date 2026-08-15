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
shared is the token-word concept (the 32-token metadata granule) in
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
from functools import cache
from itertools import product as _iter_product
from math import gcd, log1p, prod

import torch

from rola.routing.geometry import (  # noqa: F401  -- re-exported: this module's
    OwnerGeometry,  # own consumers still import it from here,
    owner_geometry,  # and they all retire with it (P67 D2-c).
)
from rola.routing.types import Topology, validate_normalization

# Bumped whenever the emitted layout changes in a way the CUDA consumer must
# observe. The consumer asserts on it at load time.
ROLA_SCHEDULE_ABI_VERSION = 4  # v4: 32-token word granule + per-token atom-union bytes

#: A ``cp.async`` warp-wide transaction is 32 lanes x 16 B = 512 B (§5 "Staging":
#: "innermost rows (512 B contiguous, cp.async-shaped)").
CP_ASYNC_ROW_BYTES = 512

#: The scan stream's METADATA GRANULE: the 32-token machine word. The BT axis is
#: DEAD (endgame executor: groups of MMA_K_QUANTUM gathered tokens are the work
#: unit; the word is bit-packing, not a boundary).
TOKEN_WORD = 32

def consumer_occupancy() -> Callable[..., int]:
    """THE occupancy oracle: `cudaOccupancyMaxActiveBlocksPerMultiprocessor` on the
    real templated kernel at its real dynamic SMEM size.

    Register-aware, which is the entire point: a shared-memory-only model cannot see
    registers, and at `BC = 16` the REGISTER FILE is the binding limit.

    **THE MIS-KEY THIS REPLACED IS A MEASURED DEFECT, NOT A HYPOTHETICAL ONE**
    (`rewrite.md` §2.34). The previous default queried the REGISTER-RESIDENT
    consumer (deleted at item 6b; its header was also named `v3_consumer_kernel.cuh`
    -- readable at `c3c2f845` in the source repository `docs/provenance.md` names,
    and a different kernel from today's file of that name), which has a different footprint -- 31680 B against 44672 at
    `BT = 16, BC = 64, d_v = 64`, 44416 against 72448 at `BT = 32` -- so it reported
    `2 / 1` CTAs/SM where the launched kernel runs at `3 / 2`, and the joint argmin
    reproduced 4 of its 8 measured rows instead of 8. With the right key `ncu`'s
    `launch__occupancy_limit_registers` and the driver agree on all 16 profiled
    instantiations, byte for byte on `launch__shared_mem_per_block_dynamic`.

    The kernel's shared memory is entirely COMPILE-TIME
    (`consumer::consumer_smem_bytes<BC, DV, D, decay, GT>`), so its template axes are
    the whole query -- `D` among them, because the amplitude prefetch slab carries a
    16-byte chunk per token per side for every level beyond the first.
    """
    from rola.ops._ext import extension
    ext = extension()

    def query(BC: int, d_v: int, decay: bool, D: int = 2,
              group_tokens: int = MMA_K_QUANTUM) -> int:
        return int(ext.rola_consumer_occupancy(decay, BC, d_v, D, group_tokens))

    return query


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
#: The precision axis is DEAD: bf16 storage, fp32 accumulation, fp64 reference
#: as oracle (revival = git).

#: Per-SM shared-memory budget, in bytes, when no device can be asked -- the CPU-only
#: fallback for §6 selection. 100 KiB is `cudaDevAttrMaxSharedMemoryPerMultiprocessor`
#: on Ampere (SM 8.0 and 8.6 alike) and is DEVICE-QUERIED whenever a device exists
#: (`rola_sm_shared_bytes`). It is a **plan field**, not a hidden default, so a
#: plan built on a CPU carries the number it reasoned with and a reader can tell a
#: fallback-selected plan from a device-selected one (§10's ruling).
DEFAULT_SM_SHARED_BYTES = 100 * 1024

#: SM count, when no device can be asked -- the CPU-only fallback for the fill
#: ceiling. 80 is the RTX 3080 Ti this repo's measurement basis was taken on. Unlike
#: `DEFAULT_SM_SHARED_BYTES` there is NO architectural value to fall back to -- SM
#: count is a per-SKU fact, not a per-architecture one -- so this is a declared target
#: rather than a safe default.
DEFAULT_SM_COUNT = 80

#: The candidate-tile word budget the kernel holds in SMEM per CTA
#: (`v3_consumer.cuh::kMaxWords`). Mirrored here because it is a term of the
#: footprint; the equality gate below pins it.
_K_MAX_WORDS = 128


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
    BC: int
    warp_tile: tuple[int, int]
    decay: bool = False
    write_cost_weight: float = DEFAULT_WRITE_COST_WEIGHT
    #: The per-SM shared-memory budget §6 reasoned about when this plan was selected,
    #: RECORDED rather than re-read at launch (§10's ruling). Device-queried when a
    #: device exists, `DEFAULT_SM_SHARED_BYTES` otherwise -- and carried either way,
    #: so a plan states the machine it was reasoned for instead of silently depending
    #: on the machine it is replayed on.
    #:
    #: **PROVENANCE ONLY: NO DERIVATION RULE READS IT.** The box term prices the
    #: PER-CTA budget (`cudaFuncAttributeMaxDynamicSharedMemorySize`), not the per-SM
    #: one, because the endgame executor hides latency inside one CTA. This field is
    #: RETAINED so a replayed plan states the machine it was built for; it dies at the
    #: standalone-package extraction, together with the `rola_sm_shared_bytes` binding
    #: and `DEFAULT_SM_SHARED_BYTES`.
    sm_shared_bytes: int = DEFAULT_SM_SHARED_BYTES

    def __post_init__(self) -> None:
        validate_normalization(self.normalization)
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

    def level_amp_dtype(self, level: int) -> torch.dtype:
        """Storage dtype of ``level``'s amplitude stream, split by LEVEL ROLE.

        SPANNED levels (``level >= j``) store bf16 -- they feed the per-leaf
        coefficients, whose independent roundings partially cancel over the
        ``BC``-term contraction; the DECAY arm applies its survival panels in
        fp16 against the same bf16 storage (U07-d -- the P09 decay carve is
        dead). FIXED levels (``level < j``) feed the per-token ROW SCALES, whose
        error is uniform on the whole output row and does not cancel, so they
        never narrow (Option B): fp32 always. The decay clock is fp32 always
        (:func:`build_clock_stream`).
        """
        if level < self.geometry.first_spanned_level:
            return torch.float32
        return torch.bfloat16

    def amplitude_pitch(self, width: int, dtype: torch.dtype | None = None) -> int:
        """Pad a level's width so one tile's rows are a whole number of 512 B lines.

        The stream is stored token-major and contiguous per tile (§5 "Staging"), so
        a tile block is ``BT * pitch`` elements; requiring that to be a multiple of
        ``CP_ASYNC_ROW_BYTES`` is what makes the innermost rows ``cp.async``-shaped.
        Padding columns hold structural zeros, which are inert under §12 rule 4's
        zero-route annihilation and never appear in a bitmap (built over ``width``
        only), so padding cannot manufacture a live leaf.

        ``dtype`` is the stream's STORAGE dtype (the pitch is an element count, so
        it depends on the itemsize); default is the spanned bf16. A caller
        staging a differently-typed stream (a fixed level, the clock) passes
        that stream's dtype. The block quantum is the 32-token machine word
        (``TOKEN_WORD``) -- the BT axis is dead.
        """
        itemsize = torch.empty((), dtype=dtype or torch.bfloat16).element_size()
        elems_per_row = CP_ASYNC_ROW_BYTES // itemsize
        step = elems_per_row // gcd(TOKEN_WORD, elems_per_row)
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
# Shared routing views -- the ONE pass the statistics and the builder both walk
# ---------------------------------------------------------------------------


class RoutingViews:
    """The routing prefix the planner and the schedule builder BOTH need, computed once.

    Three things would otherwise be recomputed by any caller that wants both the
    statistics and the schedule. (1) The ``[B,T,H,w] -> [BH,T,w]`` fold, which is a
    permute-then-reshape and therefore a real copy. (2) The exact-nonzero liveness
    mask. (3) The per-``(level, digit)`` TILE bitmap ``[BH, n_tiles, w]``.

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

    @staticmethod
    def versions(stored, write) -> tuple[int, ...]:
        return tuple(t._version for t in (*stored, *write))

    def matches(self, stored, write, decay) -> bool:
        """Every component is necessary; none is a heuristic."""
        return (decay is self.decay
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

    ``slice_union_tok``: ``[BH, n_slices, n_tiles]`` ``int32`` (ABI v3) -- the
    SAME union at TOKEN granularity, sides OR'd: bit ``t`` of entry
    ``[bh, s, tile]`` is "some digit of slice ``s`` at the boundary level is
    read- or write-live at token ``tile*BT + t``". One word per candidate visit;
    the token-gather ballot reads it and ANDs it under the owner's fixed-level
    ``token_bits`` rows to form the per-tile membership word. Derived from
    ``token_bits``, so it can only agree with the fine authority.

    ``read_amps``/``write_amps``: one ``[BH, n_tiles, BT, pitch_l]`` tensor per
    level, token-major and contiguous per tile (§5 "Staging"). A nonzero entry *is*
    a live token (§4), so no separate token mask is stored.

    ``owners``/``owner_costs``: active owners in HEAD-MAJOR order -- ``bh``
    ascending, estimated cost descending within a ``bh`` (:func:`_sort_owners`).
    An owner with no candidate tile on either side is absent entirely.

    **Per-owner candidate bitmaps are deliberately NOT a field.** §4: the
    intersection "is not materialized per owner in memory (Pi-shaped, one reader
    each, strictly more traffic than recomputing)". The builder forms them
    transiently for the cost sort and drops them.
    """

    plan: Plan
    BH: int
    T: int
    n_twords: int  # ceil(T / 32): the scan stream's word count
    level_offsets: tuple[int, ...]
    read_tile_bits: torch.Tensor
    write_tile_bits: torch.Tensor
    read_token_bits: torch.Tensor
    write_token_bits: torch.Tensor
    read_slice_union: torch.Tensor
    write_slice_union: torch.Tensor
    slice_union_tok: torch.Tensor
    #: TWO UNION FORMS, form (b): the ATOM-MAJOR CONTIGUOUS union bitlists,
    #: ``[BH, n_slices, QPO, n_twords]`` int32 per side -- bit ``t`` of word
    #: ``[bh, s, q, w]`` = "some leaf of atom ``q`` of slice ``s`` has its WHOLE
    #: spanned product live at token ``w*32+t`` on this side". The walk's
    #: per-side atom masks, consumed word-wise at the ballot; atom-major so the
    #: per-atom fetch era streams one atom's row contiguously (no migration).
    #: DERIVED from the same masks the token bits are (the per-(owner,word)
    #: pre-OR'd form is FORBIDDEN -- the per-visit OR is the kernel's job).
    read_atom_bits: torch.Tensor
    write_atom_bits: torch.Tensor
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
    #: Attached lazily by :func:`select_window`: the MEDIAN OWNER'S MEMBER COUNT,
    #: the one routing statistic the window selection consults. It is a pure
    #: function of `slice_union_tok`, which is frozen with the rest of the
    #: schedule, so caching it changes no observable value -- and computing it is
    #: a device-side popcount over the whole union bitmap, which is a real cost
    #: beside a small kernel and must not be paid once per LAUNCH.
    union_members: int | None = None
    #: Attached lazily by :func:`select_window`: the chosen token quantum. The
    #: choice is a pure function of this schedule, the decay stream and the plan's
    #: `(BC, d_v)`, all of which are fixed for the schedule's lifetime -- and the
    #: driver queries it rests on are per-arm constants. A LAUNCH must not ask the
    #: driver a question with a fixed answer.
    window: int | None = None

    def decay_artifacts(self, stored, write, decay, *, build):
        """Return this schedule's decay artifacts, building them at most once.

        `build()` is called only on a miss. The result is attached to this frozen
        instance with `object.__setattr__` -- deliberate and confined to this one
        method: the record is a MEMO of a pure function of inputs the schedule
        already describes, so attaching it changes no observable value, and
        `DecayStream.matches` is what makes that claim checkable rather than
        assumed.
        """
        cached = self.decay_stream
        if cached is not None and cached.matches(stored, write, decay):
            return cached
        fresh = build()
        object.__setattr__(self, "decay_stream", fresh)
        return fresh

    def schedule_bytes(self) -> int:
        """Bitmap-only footprint -- the quantity §4's O(L*N) assertion is about."""
        return sum(t.numel() * t.element_size() for t in (
            self.read_tile_bits, self.write_tile_bits,
            self.read_token_bits, self.write_token_bits,
            self.read_slice_union, self.write_slice_union,
            self.slice_union_tok, self.read_atom_bits, self.write_atom_bits))

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


def _atom_union_bits(views: RoutingViews, geometry: OwnerGeometry, BH: int, T: int,
                     level_offsets: list[int], widths: tuple[int, ...]
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """ATOM-MAJOR CONTIGUOUS union bitlists, ``[BH, n_slices, QPO, n_twords]``
    int32 per side (Blake's confirmed layout, 2026-08-06).

    Bit ``t`` of word ``[bh, s, q, w]`` = "some leaf of atom ``q`` of slice
    ``s`` has EVERY spanned level's digit live at token ``w*32 + t``" (an atom
    = ``MMA_K_QUANTUM`` consecutive leaves in canonical order;
    ``QPO = BC/16``). ATOM-MAJOR so one atom's whole token-liveness row is
    CONTIGUOUS -- the layout the per-atom fetch era streams without a
    migration; the ballot consumes it word-wise (atom row word AND the taken
    word). Fixed levels do not enter: they are the membership ballot's own AND
    terms, and the row is per-SLICE precisely so it covers the structure every
    owner of the slice shares.

    Chunked over leaves so no ``[BH, T, N]`` tensor is materialized at once
    (design-for-1M: the production form at scale is a producer-epilogue
    OR-reduce over the same authority; this is its host reference). The flat
    global-leaf mixed radix below is exact because ``BC = c * inner`` makes the
    slice offset ``s*BC`` decompose into ``s*c`` boundary digits and a multiple
    of every lower level's width.
    """
    device = views.read_mask[0].device
    j = geometry.first_spanned_level
    D = len(widths)
    BC = geometry.BC
    n_slices = geometry.n_slices
    qpo = BC // MMA_K_QUANTUM
    n_span = n_slices * BC
    leaf = torch.arange(n_span, device=device)
    digit_idx = []
    for level in range(j, D):
        stride = 1
        for width in widths[level + 1:]:
            stride *= width
        digit_idx.append((leaf // stride) % widths[level])
    sides = []
    chunk = 64 * MMA_K_QUANTUM
    for masks in (views.read_mask, views.write_mask):
        atoms_live = torch.zeros((BH, T, n_span // MMA_K_QUANTUM), dtype=torch.bool,
                                 device=device)
        for lo in range(0, n_span, chunk):
            hi = min(lo + chunk, n_span)
            live = None
            for k, level in enumerate(range(j, D)):
                m = masks[level][:, :, digit_idx[k][lo:hi]]  # [BH, T, hi-lo]
                live = m if live is None else live & m
            atoms_live[:, :, lo // MMA_K_QUANTUM:hi // MMA_K_QUANTUM] = live.view(
                BH, T, (hi - lo) // MMA_K_QUANTUM, MMA_K_QUANTUM).any(dim=-1)
        # [BH, T, n_slices, qpo] -> [BH, n_slices, qpo, T] -> T packed to words.
        rows = atoms_live.view(BH, T, n_slices, qpo).permute(0, 2, 3, 1)
        sides.append(_pack_bits(rows).contiguous())
    return sides[0], sides[1]


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

    ``views`` is an optional :class:`RoutingViews`, normally the SAME one the caller
    already built (:func:`plan_and_schedule`), carrying the folded tensors and the §4
    tile bitmaps over these exact tensors. It changes what is RECOMPUTED,
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
    BT = TOKEN_WORD  # the 32-token machine word: the metadata granule (no BT axis)
    n_tiles = (T + BT - 1) // BT  # = n_twords
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

    # The TOKEN-GRANULAR slice union (ABI v3): one 32-bit token word per
    # (bh, slice, tile), bit ``t`` = "some digit of this slice at level ``j`` is
    # live -- on EITHER side -- at token ``tile*BT + t``". It is the same
    # reduction as ``slice_union`` one granularity finer (token bits instead of
    # tile bits, sides OR'd because the walk visits the read|write union), and
    # it is derived from ``token_bits`` so the two authorities cannot disagree.
    # The kernel's token-gather ballot reads exactly this word; the per-side
    # fine rows it ANDs against are the fixed levels' ``token_bits`` rows.
    lj = token_bits[:, :, :, level_offsets[j]:level_offsets[j] + widths[j]]
    union_j = (lj[0] | lj[1]).reshape(BH, n_tiles, n_slices, c)
    while union_j.shape[-1] > 1:
        half = union_j.shape[-1] // 2
        head = union_j[..., :half] | union_j[..., half:2 * half]
        union_j = (torch.cat([head, union_j[..., 2 * half:]], dim=-1)
                   if union_j.shape[-1] % 2 else head)
    slice_union_tok = union_j.squeeze(-1).permute(0, 2, 1).contiguous()

    # TWO UNION FORMS, form (b): the ATOM-MAJOR CONTIGUOUS union bitlists. An
    # atom is MMA_K_QUANTUM consecutive leaves of a slice's spanned rectangle;
    # its bit is set at a token iff SOME leaf of it has EVERY spanned level's
    # digit live there (the fixed levels enter membership separately, at the
    # ballot's AND). Derived from the same masks as the token bits -- one
    # authority -- and CHUNKED over atoms so no [BH, T, N] leaf tensor is ever
    # materialized (the production path at scale is a producer-epilogue
    # OR-reduce; this host builder is its reference).
    if geometry.BC % MMA_K_QUANTUM != 0:
        raise ValueError(
            f"BC={geometry.BC} is not a multiple of MMA_K_QUANTUM={MMA_K_QUANTUM}: the "
            "executor's atom union has no concept of a sub-quantum owner rectangle (every "
            "atom is MMA_K_QUANTUM consecutive leaves, QPO=BC/MMA_K_QUANTUM); a sub-aligned "
            "BC has no atom-bits construction, not merely an inefficient one. Declared "
            "restriction, not a fallback -- pick a BC that is a multiple of "
            f"{MMA_K_QUANTUM} (owner_geometry's own alignment rules already require BC to "
            "name an aligned rectangle; this adds the executor's coarser quantum on top).")
    read_atom_bits, write_atom_bits = _atom_union_bits(
        views, geometry, BH, T, level_offsets, widths)

    # The amplitude CAST, strictly BELOW support selection (the bitmaps above are
    # built from the exact fp32 folds -- gate I6-B §2.B4 pins the ordering). The
    # dtype is per LEVEL ROLE: spanned levels take `plan.amp_dtype`, fixed levels
    # never narrow (`Plan.level_amp_dtype`).
    read_amps = tuple(
        _tile_major_amps(level, BT, n_tiles,
                         plan.amplitude_pitch(width, plan.level_amp_dtype(idx)),
                         plan.level_amp_dtype(idx))
        for idx, (level, width) in enumerate(zip(read_f, widths)))
    write_amps = tuple(
        _tile_major_amps(level, BT, n_tiles,
                         plan.amplitude_pitch(width, plan.level_amp_dtype(idx)),
                         plan.level_amp_dtype(idx))
        for idx, (level, width) in enumerate(zip(write_f, widths)))

    # `grouped` is exactly the per-(slice, tile) boundary liveness the owner sort
    # needs; it is handed over rather than recomputed there.
    owners, owner_costs = _sort_owners(
        read_live, write_live, geometry, plan, n_tiles=n_tiles,
        slice_live=(grouped[0], grouped[1]))

    g_read_f = None if g_read is None else _fold_bth(g_read.unsqueeze(-1)).squeeze(-1).contiguous()
    g_write_f = _fold_bth(g_write.unsqueeze(-1)).squeeze(-1).contiguous()

    return Schedule(
        plan=plan, BH=BH, T=T, n_twords=n_tiles,
        level_offsets=tuple(level_offsets),
        read_tile_bits=read_tile_bits, write_tile_bits=write_tile_bits,
        read_token_bits=read_token_bits, write_token_bits=write_token_bits,
        read_slice_union=read_slice_union, write_slice_union=write_slice_union,
        slice_union_tok=slice_union_tok,
        read_atom_bits=read_atom_bits, write_atom_bits=write_atom_bits,
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
    BT = TOKEN_WORD
    n_tiles = (T + BT - 1) // BT
    folded = tuple(_fold_bth(level) for level in stored)
    for level, (tensor, width) in enumerate(zip(folded, widths)):
        if tensor.shape != (B * H, T, width):
            raise ValueError(
                f"write_levels_stored[{level}] must fold to [{B * H},{T},{width}], "
                f"got {tuple(tensor.shape)}")
    # THE CLOCK IS PINNED: it never follows a narrow `amp_dtype`. The clock is an
    # exponent -- a relative error `eps` on the prefix propagates as
    # `exp(eps * E) - 1` on the survival factors with `|E|` up to the panel clamp
    # (88), i.e. a bf16 clock is a ~19% error where a bf16 amplitude is a ~0.2%
    # one, worsening with sequence length and decay strength (design §4.7; gate
    # I6-B mutant M2 enforces this pin by measurement). A WIDER amp_dtype (the
    # fp64 reference path) still applies.
    clock_dtype = torch.float32
    return tuple(
        _tile_major_amps(level, BT, n_tiles,
                         plan.amplitude_pitch(width, clock_dtype), clock_dtype)
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
                         group_tokens: int = MMA_K_QUANTUM) -> float:
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
    for amps in clock_amps:            # [BH, n_twords, 32, pitch], zero padded
        level_max = amps.amax(dim=-1)  # [BH, n_twords, 32]
        per_token = level_max if per_token is None else per_token * level_max
    if per_token.shape[-1] != TOKEN_WORD:
        raise ValueError(
            f"the staged clock stream has {per_token.shape[-1]} tokens per word; the "
            f"granule is the {TOKEN_WORD}-token machine word")
    # A WINDOW is any `group_tokens` MEMBER tokens of the whole sequence (dynamic
    # boundaries -- alignment gives no bound), so the per-window prefix is bounded
    # by the sum of the `group_tokens` LARGEST per-token clock bounds. Conservative
    # (refusal-side only), exact when those tokens happen to share a window.
    #
    # **THE BOUND SCALES WITH THE WINDOW, AND THAT IS THE WINDOW'S ONE REAL PRICE
    # ON THE DECAY ARM.** The reference frame is entered and left inside a window,
    # so `P` restarts every window and a window four times as wide sums four times
    # as many clocks. Widening the consumer's token quantum therefore CONTRACTS the
    # admitted routing domain in proportion, which is why `select_window` consults
    # this function per candidate rather than once.
    flat = per_token.reshape(per_token.shape[0], -1)  # [BH, Tpad]
    if flat.numel() == 0:
        p_hat = 0.0
    else:
        k = min(group_tokens, flat.shape[-1])
        p_hat = float(flat.topk(k, dim=-1).values.sum(dim=-1).max())
    return 0.5 * p_hat * g_max


#: THE WINDOWS THIS BINARY CARRIES, widest first. A closed set, matching the `GT`
#: axis of `tools/gen_shards.py`: selection picks from it or refuses, never
#: interpolates. `128` is absent because the window Gram's `[GT][GT+8]` storage is
#: QUADRATIC in `GT` and a `GT = 128` CTA does not fit `sm_86`'s 99 KB opt-in at
#: the flagship shape.
CONSUMER_WINDOWS = (64, 16)

#: How many full windows a typical owner must have before a wide one is selected.
#: The window's only waste is its RAGGED TAIL -- groups are always full by
#: construction, so sparsity does not thin a window, it only shortens the owner's
#: group stream -- and a tail of at most `W - 1` dead groups out of
#: `WINDOW_MIN_FILL` windows is bounded by `1 / (2 * WINDOW_MIN_FILL)` of the
#: owner's work.
WINDOW_MIN_FILL = 4


@cache
def _max_dynamic_smem_bytes() -> int:
    from rola.ops._ext import extension
    return int(extension().rola_max_dynamic_smem_bytes())


@cache
def _arm_occupancy(decay: bool, BC: int, d_v: int, D: int, gt: int) -> int:
    from rola.ops._ext import extension
    return int(extension().rola_consumer_occupancy(decay, BC, d_v, D, gt))


@cache
def _arm_smem_bytes(decay: bool, BC: int, d_v: int, D: int, gt: int) -> int:
    from rola.ops._ext import extension
    return int(extension().rola_consumer_smem_bytes(decay, BC, d_v, D, gt))


def union_member_median(schedule: Schedule) -> int:
    """The MEDIAN OWNER'S MEMBER COUNT -- the one routing statistic the window
    selection consults, and an upper bound on how long an owner's group stream is.

    Exact at dense routing, where the fixed-level rows are all ones and an owner's
    membership IS its coarse slice union; an over-estimate where the fixed levels
    thin the union further. The over-estimate is on the PERFORMANCE side and never
    the correctness side: an over-wide window computes the same answer with dead
    tile.

    Memoized on the schedule, because it is a pure function of `slice_union_tok`
    -- which is frozen with the rest of it -- and computing it is a device-side
    sweep of the whole union bitmap, a real cost beside a small kernel.
    """
    members = schedule.union_members
    if members is not None:
        return members
    union = schedule.slice_union_tok  # [BH, n_slices, n_twords] int32
    if union.numel() == 0:
        members = 0
    else:
        # THE SWAR POPCOUNT, six elementwise passes over the bitmap rather than
        # the thirty-two a shift-and-test loop takes.
        x = union.to(torch.int64) & 0xFFFFFFFF
        x = x - ((x >> 1) & 0x5555555555555555)
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
        per_owner = ((x * 0x0101010101010101) >> 56).sum(dim=-1).reshape(-1)
        members = int(per_owner.median())
    object.__setattr__(schedule, "union_members", members)
    return members


def select_window(schedule: Schedule, decay, clock_amps, BC: int, d_v: int) -> int:
    """The consumer's token quantum for THIS routing -- a plan-time choice.

    The window is a DISTRIBUTION-DEPENDENT structural axis, never a raised
    constant: a wide window amortises every per-visit term (the three barriers,
    the fold's read-modify-write of the state, the descriptor decode) ``W``-fold
    and pays for it with a Gram quadratic in ``W``, so which width wins is a
    property of the routing rather than of the kernel. Widest-first over
    :data:`CONSUMER_WINDOWS`, with four refusals.

    **The tail.** Groups are always full (the executor packs ``MMA_K_QUANTUM``
    members per group), so a wide window is not thinned by sparsity -- it is only
    wasted when an owner's whole group stream is shorter than a few windows. That
    stream length is bounded above by the popcount of the owner's COARSE SLICE
    UNION, exact at dense routing (where the fixed-level rows are all ones) and an
    over-estimate where the fixed levels thin the union further. The over-estimate
    is on the PERFORMANCE side, never the correctness side: an over-wide window
    computes the same answer with dead tile.

    **The device.** The window Gram's storage is quadratic in the width, so a wide
    arm is the first one a device cannot grant a block; the same inequality the
    launch applies is applied here, before selecting.

    **RESIDENCY, AND THIS IS THE BINDING ONE ON `sm_86`.** A wider window is taken
    only if it costs NO CTA per SM against the group quantum. The window's whole
    trade is per-visit overhead amortised `W`-fold against a Gram quadratic in
    `W`; measured on this box, the amortisation is worth far less than the CTAs
    the wider carve gives up, and the loss tracks the residency given up rather
    than the extra arithmetic (`BC = 16`, 4 CTAs -> 2: +43% at
    `topo-D4-w8x8x8x16-BC16`; `BC = 64`, 3 -> 1: +84% whole-layer at the dense
    `N = L = 4096` cell). That is `sm_86`'s 100 KB/SM speaking, not the
    formulation: the kernel's panel builds gather amplitudes straight from global
    memory and are NOT prefetched, so warps are still what hides that latency --
    which is also what P31 measured directly (66% of cycles with no eligible
    warp). On an arch whose SMEM budget carries the wider carve at the same
    occupancy the rule admits it with no code change, and the answer is re-derived
    from the driver rather than from this paragraph.
    Evidence: `workflows/perf/p36_quantum.md`.

    **The panel domain**, decay only. :func:`panel_exponent_bound` grows with the
    window, so a width whose bound clears :data:`PANEL_MAX_EXPONENT` at 16 may not
    clear it at 64. The narrower window is then selected rather than the launch
    refused -- the domain contraction is the window's price, and declining to pay
    it is the right answer, not an error.
    """
    if schedule.window is not None:
        return schedule.window

    # THE REFUSALS RUN CHEAPEST-FIRST, AND THAT ORDER IS LOAD-BEARING RATHER THAN
    # TIDY. This runs on the LAUNCH path, beside kernels that take a fifth of a
    # millisecond, so a refusal that costs a device-side sweep of the union bitmap
    # must never be reached when a refusal costing one cached driver fact already
    # settles it. On a machine whose SMEM budget cannot carry the wider carve at
    # the same occupancy -- `sm_86`, every arm -- the first two refusals answer and
    # nothing else is computed at all.
    narrow = CONSUMER_WINDOWS[-1]
    decayed = decay is not None
    #: THE TOPOLOGY DEPTH IS A RESIDENCY AXIS. The prefetch slab carries one extra
    #: 16-byte chunk per token per side for every level beyond the first, so an
    #: occupancy answer taken at the wrong `D` is simply the wrong arm's.
    levels = len(schedule.read_amps)
    granted = _max_dynamic_smem_bytes()
    resident = _arm_occupancy(decayed, BC, d_v, levels, narrow)
    for gt in CONSUMER_WINDOWS:
        if gt == narrow:
            break
        # THE SAME ADMISSION THE LAUNCH APPLIES, applied before selecting rather
        # than discovered as a hard refusal: the window Gram's storage is quadratic
        # in `gt`, so a wide window is the first arm a small-SMEM device cannot
        # grant a block. `consumer.cuh:row_admitted` is the inequality mirrored.
        if _arm_smem_bytes(decayed, BC, d_v, levels, gt) > granted:
            continue
        # THE RESIDENCY RULE, AND IT IS THE ONE THE MEASUREMENT WROTE. See the
        # docstring: a window bought with occupancy is a window that loses.
        if _arm_occupancy(decayed, BC, d_v, levels, gt) < resident:
            continue
        if decayed and clock_amps is not None:
            if not panel_exponent_bound(clock_amps, decay, gt) <= PANEL_MAX_EXPONENT:
                continue
        if union_member_median(schedule) < WINDOW_MIN_FILL * gt:
            continue
        object.__setattr__(schedule, "window", gt)
        return gt
    object.__setattr__(schedule, "window", narrow)
    return narrow


def _sort_owners(
    read_live: tuple[torch.Tensor, ...],
    write_live: tuple[torch.Tensor, ...],
    geometry: OwnerGeometry,
    plan: Plan,
    *,
    n_tiles: int,
    slice_live: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enumerate active owners and sort them HEAD-MAJOR: ``bh`` ascending, and
    estimated cost descending within each ``bh``.

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
    bh_sel = bh_all.reshape(-1)[active_index]
    # HEAD-MAJOR (`bh` primary ascending, cost descending within a `bh`). The reason
    # is L2 RESIDENCY, not load balance: the amplitude tensors a CTA gathers are
    # indexed by `bh`, so the working set the device is actually holding is the
    # union over the `bh` of the resident `blockIdx.x` window. Cost-descending alone
    # is not head-monotone and interleaved every head across that window -- the
    # window spanned 8.00 of 8 heads, and the amplitude stream was re-fetched
    # 24.8-26.9x its 0.336 GB footprint. Keying on `bh` first collapses the span to
    # 1.88 heads, which is the "at most 2 batch-heads" AMP_STAGING_DESIGN §4.3
    # assumes and had never had. MEASURED at the parity flagship (N=L=16384): L2
    # read hit rate 27.09 % -> 70.67 %, DRAM 20.27 -> 6.38 GB, step 51.41 -> 48.21 ms
    # (-6.23 %), with the request side untouched (L2 read requests -0.06 %, global
    # RED sectors bit-identical, occupancy identical) -- so the win is entirely
    # hit/miss, not less work. `state` is byte-equal across the re-key.
    #
    # Cost-descending is retained WITHIN a `bh` (it is what the sequence-parallel
    # apportionment and the tail-effect argument read), and BOTH sorts are `stable`,
    # so a cost tie still resolves to the enumeration order above -- the
    # byte-for-byte reproducibility the enumeration comment relies on is preserved:
    # a stable sort of a stable order is that order restricted to each `bh` group.
    by_cost = torch.argsort(costs, descending=True, stable=True)
    order = by_cost[torch.argsort(bh_sel[by_cost], stable=True)]
    pairs = torch.stack((bh_sel.to(torch.int32),
                         owner_all.reshape(-1)[active_index].to(torch.int32)), dim=1)
    return pairs[order].contiguous(), costs[order].contiguous()


def _cartesian(sets: list[list[int]]):
    if not sets:
        yield ()
        return
    yield from _iter_product(*sets)


# ---------------------------------------------------------------------------
# Plan assembly
# ---------------------------------------------------------------------------


def build_plan(topology: Topology, BC: int, *, normalization: str, decay: bool,
               sm_shared_bytes: int | None = None) -> Plan:
    """A ``Plan`` over one topology at one ``BC``, with the derived fields filled.

    ``BC`` is an ARGUMENT: :func:`rola.planner.derive.derive_bc` is what supplies it on
    the dispatch path, and a bench or a conformance gate supplies its own so it can
    exercise a geometry the derivation would not pick. Everything else here is either a
    model fact (``topology``, ``normalization``, ``decay``) or derived from one -- the
    warp tile is the ``(1, d_v/32)`` the consumer implements, stated rather than
    defaulted so the driver's check has something to check.
    """
    return Plan(
        topology=topology, normalization=normalization, BC=int(BC), decay=bool(decay),
        warp_tile=(1, max(1, topology.d_v // 32)),
        sm_shared_bytes=(DEFAULT_SM_SHARED_BYTES if sm_shared_bytes is None
                         else int(sm_shared_bytes)))


def plan_and_schedule(read_levels: tuple[torch.Tensor, ...],
                      write_levels: tuple[torch.Tensor, ...],
                      g_read: torch.Tensor | None,
                      g_write: torch.Tensor,
                      topology: Topology, BC: int, *, normalization: str, decay: bool,
                      sm_shared_bytes: int | None = None) -> tuple[Plan, Schedule]:
    """The plan and its §4 schedule from ONE pass over the routing tensors.

    :func:`build_schedule` builds its own :class:`RoutingViews` when none is supplied;
    building it here and handing it over is what keeps the ``[B,T,H,w] -> [BH,T,w]``
    fold, the exact-nonzero mask and the per-``(level, digit)`` tile bitmap computed
    once for a caller that wants both objects. Nothing about the result moves: the plan
    is :func:`build_plan`'s and the schedule is :func:`build_schedule`'s.
    """
    views = routing_views(read_levels, write_levels, topology)
    plan = build_plan(topology, BC, normalization=normalization, decay=decay,
                      sm_shared_bytes=sm_shared_bytes)
    schedule = build_schedule(read_levels, write_levels, g_read, g_write, plan, views=views)
    return plan, schedule


__all__ = [
    "CONSUMER_WINDOWS",
    "CP_ASYNC_ROW_BYTES",
    "DEFAULT_SM_COUNT",
    "DEFAULT_SM_SHARED_BYTES",
    "DEFAULT_WRITE_COST_WEIGHT",
    "MMA_K_QUANTUM",
    "PANEL_MAX_EXPONENT",
    "ROLA_SCHEDULE_ABI_VERSION",
    "WINDOW_MIN_FILL",
    "LevelSupportStats",
    "OwnerGeometry",
    "Plan",
    "RoutingViews",
    "Schedule",
    "build_clock_stream",
    "build_plan",
    "build_schedule",
    "owner_geometry",
    "panel_exponent_bound",
    "permuted_topology",
    "plan_and_schedule",
    "rank_spans",
    "routing_views",
    "select_window",
    "span_score",
    "spanned_moduli",
    "union_member_median",
    "unpack_bits",
]

