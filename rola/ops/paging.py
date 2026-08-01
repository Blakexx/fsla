"""Spec §7.1 page commitment: the Python surface over the ``p3_*`` paging arena.

What this module is for, in one sentence: make the state a layer actually keeps
resident scale with the owners its routing REALIZES, instead of with the ``N`` its
topology PROVISIONS.

The machinery underneath is V2's, re-keyed rather than rewritten (§7.1 "the
machinery is reused in structure; what changes is one constant, one degenerate
axis, and one status code"):

``page_states := BC``
    One page is one owner. ``logical page id = bh * pages_per_bh + owner`` with
    ``pages_per_bh = owners_total = N / BC``. This is V2's arithmetic at a chosen
    constant, not a new keying.

``dqk := 1``
    The mapped ``d_qk`` axis is an ordinary routing level folded into ``N`` in V3,
    so the ABI axis degenerates and ``paged_value_offset`` collapses to
    ``(physical * BC + row) * cols + v``. The axis is set to 1, not deleted: no
    accessor path is added or removed.

``CAPPED``
    A third terminal status. V2 treated "more pages required than the allocator
    can hold" as an error; architecture §4.5 promises the cap degrades gracefully,
    so pages are admitted in increasing ``first_write_tile`` order until the cap is
    reached and the rest are DENIED. A denied owner's write factor is annihilated
    by the admit mask -- the same rule as a structurally zero write. Capping
    CHANGES THE FUNCTION COMPUTED, so :attr:`PagingResult.capped_pages` is
    reported and :meth:`V3PageArena.plan` raises unless the caller opted in.

Vocabulary, defined here and used unqualified below: ``N`` = leaf count; ``BC`` =
leaves per owner (spec §3's owner rectangle); ``cols`` = ``d_v`` plus one mass
column under ``normalization='global'``; ``BH`` = ``batch * heads``; ``BT`` = the
plan's token tile; ``owners_total`` = ``N / BC``.

**What this module does NOT settle.** Continuation across a call boundary is
§7.3's, and §7.1 says so explicitly ("What this section does not settle:
continuation (§7.3) still owns what crosses a call boundary, including whether the
page table and the touched set persist across calls or are rebuilt"). The arena
is therefore *call-scoped by default, enforced*: :meth:`plan` resets the arena
before every call unless the caller passes ``fresh=False``. That is a fix-pass
correction (`workflows/20260730-wiring/paging-review.md` BLOCKER B1) -- an
earlier revision of this module *claimed* call-scoping here without enforcing
it, so a caller that reused one arena across two independent forwards (as
`v3_consumer_forward`'s ``paging=`` argument invites) silently computed a
continuation. ``fresh=False`` is the explicit, named opt-in for legitimate
continuation (e.g. successive chunks of one sequence kept resident without a
round trip through :meth:`adopt_dense`); :meth:`adopt_dense` /
:meth:`materialize` remain the explicit, lossless bridges to the existing dense
cache contract for state that must cross backends or re-enter after a
materialized gap.

**Arena exclusion (§6.4) is respected literally.** The arena holds persistent
cross-chunk state pages ONLY. The plan workspace below is a launch-scoped set of
ordinary torch tensors owned by the caching allocator, never mapped under the
compactor.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch

_ABSENT_PAGE = -1
#: Guard on the per-token owner Cartesian walk in the mark kernel. A DEFECT bound
#: (planner work), never the page cap -- exceeding it raises ``OVERFLOW``.
_DEFAULT_MAX_SUFFIX_COMBINATIONS = 1 << 20


def _write_can_zero(level) -> bool:
    """Can this level's write factor be EXACTLY zero?

    A structural property of the routing operator, not a measured density: entmax
    has exact support zeros (spec §1.2), softmax does not, and a `UnionRouting`
    level always can. It selects §7.1's dense-write short-circuit -- when no level
    can zero, every owner is written by the first token and enumerating owners per
    token would be `O(L * N / BC)` planner work for a constant answer. Mirrors
    `RouteDescriptor.write_level_can_zero`, which is the same rule on the V2-era
    descriptor rather than on a `Topology`.
    """
    from rola.routing.types import UnionRouting

    if isinstance(level.routing, UnionRouting):
        return True
    return level.routing.write_activation.can_zero


@dataclass(frozen=True, slots=True)
class PagingResult:
    """One call's committed-state accounting (spec §7.1's reported quantities)."""

    #: ``'READY'`` or ``'CAPPED'``. Both are successful terminal states; ``CAPPED``
    #: additionally means the function computed was degraded.
    status: str
    #: Owners with a committed page after this call's publication.
    resident_pages: int
    #: High-water mark of :attr:`resident_pages` over the arena's lifetime.
    peak_pages: int
    #: New pages DENIED by the allocator cap this call. Nonzero iff ``CAPPED``.
    capped_pages: int
    #: Logical pages the mark phase touched -- the owners this call's routing
    #: actually realized on the write side.
    realized_owners: int
    #: ``owners_total`` -- what a dense ``[N, cols]`` state would provision.
    owners_total: int
    #: Bytes of fp32 state actually committed (``resident_pages * BC * cols * 4``).
    resident_bytes: int
    #: Bytes a dense ``[BH, N, cols]`` fp32 state would occupy. The counterfactual
    #: the feature exists to beat.
    dense_bytes: int
    #: Driver-committed bytes when the arena is VMM-backed; equals
    #: :attr:`resident_bytes` rounded up to the mapped chunk, else ``None``.
    committed_bytes: int | None

    @property
    def resident_fraction(self) -> float:
        return self.resident_bytes / self.dense_bytes if self.dense_bytes else 0.0


class V3PageArena:
    """A pooled, per-layer page arena over V3 value panels.

    **Pooled, not per-(batch, head)** -- this is V2's shape and it is the reason
    the savings math works (§7.1). One arena backs every ``(batch, head)`` pair in
    a layer, so it tracks the live fraction of the WHOLE layer. A per-pair arena
    would round each pair's tiny live fraction up to at least one growth chunk and
    cap savings at the chunk-to-footprint ratio.

    ``max_physical_pages`` is the allocator hard cap. ``None`` provisions the
    dense limit (``BH * owners_total``), i.e. paging can never deny a write and
    ``CAPPED`` is unreachable -- the configuration the dense-limit overhead
    measurement runs at. A finite value is what makes §4.5's graceful degradation
    reachable.
    """

    def __init__(self, *, widths: tuple[int, ...], BC: int, cols: int, BH: int,
                 device: torch.device, max_physical_pages: int | None = None,
                 touched_capacity: int | None = None,
                 max_new_pages: int | None = None,
                 allow_capping: bool = False) -> None:
        from rola.ops._ext import extension

        self._ext = extension()
        N = prod(widths)
        if N % BC:
            raise ValueError(f"BC={BC} must divide N={N} (spec §7.1: one page is one owner)")
        self.widths = tuple(widths)
        self.BC = int(BC)
        self.cols = int(cols)
        self.BH = int(BH)
        self.N = int(N)
        self.owners_total = N // BC
        self.device = torch.device(device)
        self.logical_pages = self.BH * self.owners_total
        dense_limit = self.logical_pages
        self.max_physical_pages = dense_limit if max_physical_pages is None else int(max_physical_pages)
        if not 1 <= self.max_physical_pages <= dense_limit:
            raise ValueError(
                f"max_physical_pages must be in [1, {dense_limit}] (the dense limit); "
                f"got {self.max_physical_pages}")
        self.allow_capping = bool(allow_capping)
        if self.max_physical_pages < dense_limit and not allow_capping:
            # Loud by construction: a cap that can bind changes the function
            # computed, and §7.1 requires that be a reported decision rather than a
            # discovered one.
            raise ValueError(
                "max_physical_pages below the dense limit makes spec §7.1's CAPPED status "
                "reachable, which CHANGES THE FUNCTION COMPUTED (denied writes are "
                "annihilated). Pass allow_capping=True to opt in.")

        self._touched_capacity = int(touched_capacity or self.logical_pages)
        self._max_new_pages = int(max_new_pages or self._touched_capacity)
        if not 0 < self._max_new_pages <= self._touched_capacity:
            raise ValueError("max_new_pages must be in (0, touched_capacity]")

        self._generation = 1
        self._epoch = 0
        self.peak_pages = 0
        self._planned_BT: int | None = None
        #: The most recent :class:`PagingResult`, or ``None``.
        self.last_result: PagingResult | None = None
        self._allocate()

    # -- storage ----------------------------------------------------------

    def _allocate(self) -> None:
        dev = self.device
        z64 = lambda n: torch.zeros(n, dtype=torch.uint64, device=dev)          # noqa: E731
        z32 = lambda n: torch.zeros(n, dtype=torch.uint32, device=dev)          # noqa: E731
        scalar64 = lambda: torch.zeros((), dtype=torch.uint64, device=dev)      # noqa: E731
        scalar32 = lambda: torch.zeros((), dtype=torch.uint32, device=dev)      # noqa: E731
        tc, mn = self._touched_capacity, self._max_new_pages
        workspace = self._ext.p3_page_plan_workspace_bytes(tc)

        # Plan workspace. §6.4: launch-scoped-in-lifetime torch tensors, owned by
        # the caching allocator. They are persistent across calls only because
        # re-allocating them every step would be pure launch overhead -- they are
        # NOT arena pages and never go under the compactor.
        self._plan = [
            z64(self.logical_pages),                 # touched_tags
            z64(self.logical_pages),                 # earliest_epoch_tile (scratch)
            torch.full((tc,), 2**64 - 1, dtype=torch.uint64, device=dev),  # touched_ids
            z64(tc),                                 # sorted_ids
            scalar32(),                              # touched_count
            z32(tc), z32(tc),                        # new_flags, new_ranks
            z64(mn), z64(mn),                        # new_logical_ids, new_earliest_tiles
            scalar32(),                              # new_count
            scalar64(),                              # required_slots
            scalar32(),                              # status
            scalar64(), scalar64(), scalar64(),      # generation, epoch, last_epoch
            torch.zeros(max(workspace, 1), dtype=torch.uint8, device=dev),
            z64(mn), z64(mn),                        # admit_keys, admit_sorted
            scalar32(),                              # capped_pages
        ]

        # Cache metadata + the page arena itself.
        self._page_table = torch.full((self.BH, self.owners_total), _ABSENT_PAGE,
                                      dtype=torch.int32, device=dev)
        #: `[physical_pages, BC, 1, cols]` -- the `dqk := 1` degeneracy kept as a
        #: real axis so `paged_value_offset` is the SAME expression V2 used.
        self._value_base = torch.zeros(
            (self.max_physical_pages, self.BC, 1, self.cols), dtype=torch.float32, device=dev)
        self._cache = [
            self._page_table,
            z64(self.logical_pages),                 # earliest_write_epoch_tile
            torch.full((self.max_physical_pages,), _ABSENT_PAGE, dtype=torch.int32, device=dev),
            scalar32(),                              # allocated_count
            torch.full((), self.max_physical_pages, dtype=torch.uint64, device=dev),
            scalar64(),                              # next_compact_slot
            torch.full((), self._generation, dtype=torch.uint64, device=dev),
            self._value_base,
            scalar64(), scalar64(),                  # hit_count, logical_allocation_count
        ]
        self._admit = torch.zeros((self.BH, (self.owners_total + 31) // 32),
                                  dtype=torch.uint32, device=dev)

    # -- properties -------------------------------------------------------

    @property
    def planned_BT(self) -> int | None:
        """The ``BT`` this arena's live pages recorded ``first_write_tile`` at."""
        return self._planned_BT

    @property
    def page_table(self) -> torch.Tensor:
        """``[BH, owners_total]`` int32; ``-1`` means the owner has no page."""
        return self._page_table

    @property
    def state(self) -> torch.Tensor:
        """``[physical_pages, BC, cols]`` -- what the consumer indexes."""
        return self._value_base.view(self.max_physical_pages, self.BC, self.cols)

    @property
    def admit(self) -> torch.Tensor:
        """§7.1's admit mask, ``[BH, ceil(owners_total/32)]`` int32 for the consumer."""
        return self._admit.view(torch.int32)

    @property
    def resident_pages(self) -> int:
        return int(self._cache[3].item())

    def first_write_tiles(self) -> torch.Tensor:
        """``[BH, owners_total]`` int64: each committed page's ``first_write_tile``.

        The EXACT per-page minimum the mark phase recorded, denominated in units of
        the plan's ``BT`` (findings ``paging.md`` D3). ``-1`` where the page has no
        recorded write. This is the quantity §7.1's admission order is defined on,
        so it is exposed rather than inferred.
        """
        packed = self._cache[1].view(self.BH, self.owners_total).to(torch.int64)
        tiles = (packed & 0xFFFFFFFF).to(torch.int64)
        return torch.where(self._page_table >= 0, tiles, torch.full_like(tiles, -1))

    def dense_bytes(self) -> int:
        return self.BH * self.N * self.cols * 4

    def resident_bytes(self) -> int:
        return self.resident_pages * self.BC * self.cols * 4

    # -- lifecycle --------------------------------------------------------

    def reset(self) -> None:
        """Drop every committed page. The arena is reusable afterwards.

        The page STORAGE is deliberately not zeroed here. Dropping the page table
        makes every owner unmapped, so the next plan sees each of them as a NEW page
        and `publish` zeroes it (`zero_new_pages_kernel`) before any read can reach
        it. Blanket-zeroing `max_physical_pages` worth of storage on every reset
        would touch the DENSE footprint -- exactly the allocation §7.1 exists to
        avoid -- to re-do work publication already does on the realized set only.
        """
        self._ext.rola_p3_page_cache_reset(cache_tensors=self._cache)
        self._admit.zero_()
        self._generation += 1
        self._cache[6].fill_(self._generation)
        self._epoch = 0
        self.peak_pages = 0
        self._planned_BT = None

    def plan(self, schedule, *, fresh: bool = True) -> PagingResult:
        """Run §7.1's ``prepare -> (grow) -> publish`` against one V3 schedule.

        ``schedule`` is a :class:`~rola.planner.schedule.V3Schedule`.
        Only its ``write_amps`` and its plan's owner geometry are read: page
        commitment keys on the LIVE DEPOSIT, exactly as the §4 bitmap builder does,
        never on the clock (``build_clock_stream``'s stored tensor is not an input
        here and cannot be passed).

        ``fresh`` (default ``True``): call :meth:`reset` before planning, so this
        call is call-scoped -- it computes as if nothing preceded it, matching the
        unpaged path's "allocate zeros every call" semantics. Pass ``fresh=False``
        for an explicit, legitimate continuation: the pages this arena already
        holds resident stay resident and this call's routing is committed on top
        of them. There is no implicit third option; a caller that wants
        call-scoped-by-default behaviour gets it without asking, and a caller that
        wants to continue must say so at the call (fix-pass BLOCKER B1,
        `workflows/20260730-wiring/paging-review.md`).
        """
        if fresh:
            self.reset()
        plan = schedule.plan
        geometry = plan.geometry
        if self._planned_BT is not None and self._planned_BT != plan.BT:
            # `first_write_tile` is denominated in units of the plan's `BT`
            # (findings `paging.md` D3). A `BT` change while pages carry recorded
            # tiles would silently reinterpret them, so it is a hard error, not a
            # re-derivation.
            raise ValueError(
                f"this arena's live pages recorded first_write_tile at BT={self._planned_BT}; "
                f"the schedule plans BT={plan.BT}. Reset the arena before changing BT.")
        if tuple(plan.topology.widths) != self.widths or plan.BC != self.BC:
            raise ValueError("schedule topology/BC does not match this arena")

        self._epoch += 1
        widths = list(plan.topology.widths)
        pitches = [plan.amplitude_pitch(w) for w in widths]
        j = geometry.first_spanned_level
        # `owner_stride[l]` for a FIXED level is `prod(widths[l+1:j])`, the same
        # mixed radix `OwnerGeometry.owner_of` uses; spanned levels never index the
        # owner id, so their stride is an unused 1.
        owner_stride = [max(1, prod(widths[l + 1:j])) if l < j else 1 for l in range(len(widths))]
        # A level whose write factor can be exactly zero. This is a STRUCTURAL
        # property of the routing operator (entmax has exact support zeros, softmax
        # does not) and the topology already carries it as
        # `Topology.write_level_can_zero`. Reading it off the realized amplitudes
        # instead would be branching on a measured density -- and would cost one
        # device sync per level per step on a plane that is already launch-bound.
        can_zero = [int(_write_can_zero(level)) for level in plan.topology.levels]

        status, required, new_count, capped = self._ext.rola_p3_page_plan_prepare(
            write_amps=[t.contiguous() for t in schedule.write_amps],
            level_widths=widths, level_pitches=pitches,
            level_write_can_zero=can_zero, owner_stride=owner_stride,
            token_count=schedule.T, token_tile=plan.BT,
            first_spanned_level=j, slice_digits=geometry.slice_digits,
            n_slices=geometry.n_slices,
            plan_tensors=self._plan, cache_tensors=self._cache, mass_base=None,
            page_states=self.BC, pages_per_bh=self.owners_total,
            max_suffix_combinations=_DEFAULT_MAX_SUFFIX_COMBINATIONS,
            generation=self._generation, epoch=self._epoch,
            identity_physical_ids=False)

        code = int(status.item())
        codes = self._ext.p3_page_plan_status_codes()
        if code == codes["NEED_GROWTH"]:
            # This arena preallocates its physical extent, so growth is a driver
            # concern only for a VMM-backed plane; here it can only mean the cap
            # itself was exceeded, which `finalize` has already turned into CAPPED.
            raise RuntimeError(
                "page plan reported NEED_GROWTH against a fully provisioned arena; "
                "this is a planner defect, not a capacity condition")
        if code == codes["OVERFLOW"]:
            raise RuntimeError(
                "page plan reported OVERFLOW -- a plan-workspace capacity or "
                "generation/epoch invariant was violated (spec §7.1: OVERFLOW is a "
                "DEFECT, distinct from the CAPPED allocator limit)")
        if code not in (codes["READY"], codes["CAPPED"]):
            raise RuntimeError(f"page plan returned an unknown status {code}")

        self._ext.rola_p3_page_plan_publish(
            plan_tensors=self._plan, cache_tensors=self._cache, mass_base=None,
            generation=self._generation, epoch=self._epoch,
            page_states=self.BC, dqk=1, dv=self.cols, identity_physical_ids=False)
        self._ext.rola_p3_page_admit_mask(page_table=self._page_table, admit=self._admit)

        resident = self.resident_pages
        self.peak_pages = max(self.peak_pages, resident)
        capped_pages = int(capped.item())
        realized = int(self._plan[4].item())
        if capped_pages and not self.allow_capping:
            raise RuntimeError(
                f"page plan CAPPED {capped_pages} owners without allow_capping=True")
        self._planned_BT = plan.BT
        # `resident` above already paid the one DtoH `.item()` this quantity costs;
        # `self.resident_bytes()` would read `self.resident_pages` (and therefore
        # `.item()`) a second time for the same value. Computed inline instead --
        # a free saving on the host-round-trip-bound plane (fix pass N3,
        # `workflows/20260730-wiring/paging-review.md`).
        self.last_result = PagingResult(
            status="CAPPED" if capped_pages else "READY",
            resident_pages=resident, peak_pages=self.peak_pages,
            capped_pages=capped_pages, realized_owners=realized,
            owners_total=self.owners_total,
            resident_bytes=resident * self.BC * self.cols * 4, dense_bytes=self.dense_bytes(),
            committed_bytes=None)
        return self.last_result

    # -- dense bridges (the §7.3 boundary, crossed explicitly) -------------

    def adopt_dense(self, state: torch.Tensor) -> None:
        """Commit a page for every owner of a dense ``[BH, N, cols]`` state.

        The explicit bridge INTO the arena. It provisions the dense limit, so it
        is the wrong thing to do at scale and is here for exactly one reason: a
        carried continuation state produced by the unpaged path (or by the oracle)
        must be loadable, and §7.1 defers to §7.3 on what should cross a call
        boundary instead. Named a bridge rather than a mode so no measurement
        quotes it as the feature.
        """
        expected = (self.BH, self.N, self.cols)
        if tuple(state.shape) != expected:
            raise ValueError(f"dense state must be {expected}, got {tuple(state.shape)}")
        if self.max_physical_pages < self.logical_pages:
            raise ValueError(
                "adopt_dense commits every owner and needs the dense limit; this arena "
                f"caps at {self.max_physical_pages} of {self.logical_pages} pages")
        table = torch.arange(self.logical_pages, dtype=torch.int32, device=self.device)
        self._page_table.copy_(table.view(self.BH, self.owners_total))
        self._value_base.view(self.logical_pages, self.BC, self.cols).copy_(
            state.reshape(self.BH * self.owners_total, self.BC, self.cols))
        self._cache[2].copy_(table)
        self._cache[3].fill_(self.logical_pages)
        self._cache[5].fill_(self.logical_pages)
        self._ext.rola_p3_page_admit_mask(page_table=self._page_table, admit=self._admit)
        self.peak_pages = max(self.peak_pages, self.logical_pages)

    def materialize(self) -> torch.Tensor:
        """Gather the committed pages back into a dense ``[BH, N, cols]`` state.

        The explicit bridge OUT. Allocating ``[BH, N, cols]`` is precisely what
        paging exists to avoid, so this is for oracle comparison and for the
        existing dense cache contract -- never on a measured path.
        """
        dense = torch.zeros((self.BH, self.owners_total, self.BC, self.cols),
                            dtype=torch.float32, device=self.device)
        present = self._page_table >= 0
        idx = self._page_table.long().clamp_min(0)
        gathered = self.state.index_select(0, idx.reshape(-1)).view(
            self.BH, self.owners_total, self.BC, self.cols)
        dense[present] = gathered[present]
        return dense.reshape(self.BH, self.N, self.cols)


__all__ = ["V3PageArena", "PagingResult"]
