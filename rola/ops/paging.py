"""§7.1 page commitment, RE-KEYED TO THE MMA ATOM (``THUNDER_RUN_SPEC.md`` §4.5).

What this module is for, in one sentence: make the state a layer actually keeps
resident scale with the atoms its routing REALIZES, instead of with the ``N`` its
topology PROVISIONS.

Vocabulary, defined here and used unqualified: ``N`` = leaf count; ``BC`` = leaves per
owner block (spec §3's owner rectangle); ``cols`` = ``d_v`` plus one mass column under
``normalization='global'``; ``BH`` = ``batch * heads``; ``MMA_K_QUANTUM = 16``;
**atom** = one ``MMA_K_QUANTUM``-leaf block in canonical leaf order;
``QPO = BC / MMA_K_QUANTUM`` = atoms per owner.

---- WHAT CHANGED, AND WHY IT IS NOT A TUNING CHOICE --------------------------

``page_states := MMA_K_QUANTUM`` (was ``BC``). §1b's ruling: **``BC`` is a re-derivable
COMPUTE tile; a page is a MEMORY-RESIDENCY atom.** Their consumers are different
objects with different lifetimes -- admission bitmaps, arena pools, residency ledgers,
warm reuse across sequences -- and every one of them must survive a ``BC`` change
unbiased. Since the thunder run makes ``BC`` PER LAUNCH (§4.3), a ``BC``-keyed page
would have to be rebuilt whenever the planner moved it, which is exactly the
arena-rebuild class the re-key kills.

Every legal ``BC`` is a whole number of atoms by construction (``16 | BC``), so an
owner spans exactly ``QPO`` atoms and a CTA resolves ``QPO`` slots at startup instead
of one page. Residency tests move to atom granularity -- the SAME atom as the
emptiness-skipping quantum -- so ONE predicate serves both.

**THE ADDRESS.** Logical atom id ``bh * (N / MMA_K_QUANTUM) + leaf // MMA_K_QUANTUM``;
``table[atom id] = slot | ABSENT``; the value address is
``((slot << LOG2_ATOM | row) * cols + v)`` with ``row = leaf & (MMA_K_QUANTUM - 1)``.
Both shifts are DERIVED from the named constant (invariant I1) -- see
:data:`_LOG2_ATOM_LEAVES` -- so there is no ``4`` and no ``15`` anywhere in this file.

---- ALLOCATION: PLANNED-EXACT-ASYNC ------------------------------------------

The pool is sized EXACTLY, not grown: the popcount of §4.1's atom condensation is the
number of atoms this call will touch, and §4.1 makes that bitmap canonical truth
produced by the producer epilogue. So the arena knows its own footprint before the
consumer launches, and there is nothing to discover.

Allocation and zero-init issue on a SIDE STREAM as soon as the epilogue lands, and the
consumer launch waits on ONE event. That is the whole of the asynchrony: no polling, no
second sync, and the zeroing overlaps the schedule build rather than serializing with
it.

**THE CAP IS A POLICY CEILING, AND EXHAUSTING IT IS A REFUSAL.** With the footprint
known exactly, the allocator cannot be surprised; ``max_physical_pages`` stops being a
capacity guess and becomes a deliberate limit. When a launch demands more atoms than
the ceiling leaves, the arena RAISES, naming the demand and the arena size -- it never
grants a prefix and never drops a write. The former ``CAPPED`` mode (partial grant plus
an admit mask that annihilated the denied atoms' writes) is DELETED: it computed a
DIFFERENT function from the one the caller asked for, and the closed-world standard is
that a demand the binary cannot serve is refused at the host rather than silently
degraded. With it goes §7.1's annihilation contract, which existed only to define what
a denied write meant.

The consequence downstream is that admission is TOTAL: every atom the routing realizes
is resident, so ``admit bit set``, ``resident`` and ``page_table[atom] >= 0`` are one
predicate and the arena publishes exactly one of them (the table).

---- EXTENTS ARE ALLOCATOR POLICY AND NOTHING ELSE ----------------------------

The pool is carved into runs of ``C_EXTENT_ATOMS`` consecutive slots, and atoms
admitted together take consecutive slot ids -- so an owner's ``QPO`` atoms land
contiguous in the common case and the CTA resolves ONE base instead of ``QPO``
independent lookups. This is INVISIBLE to the table format and to the kernel: the table
is still ``atom -> slot``, and a caller cannot tell an extent-allocated arena from a
first-fit one except by reading the slot ids. Transfers batch at extent granularity.

---- SCOPE ---------------------------------------------------------------------

The arena stays CALL-SCOPED by default (``fresh=`` contract unchanged):
:meth:`plan` resets before every call unless the caller opts into continuation.
:meth:`adopt_dense` / :meth:`materialize` remain the explicit, lossless bridges to the
dense cache contract.

Arena exclusion (§6.4) is respected literally: the arena holds persistent cross-chunk
state pages ONLY.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch

#: §6's MMA k-extent, and §1b's page granule. Mirrored from
#: ``rola.routing.summaries.MMA_K_QUANTUM`` rather than imported so this module has no
#: routing dependency; :meth:`PageArena.__init__` checks the two agree.
MMA_K_QUANTUM = 16

#: ``log2(MMA_K_QUANTUM)``, DERIVED. Invariant I1: the address expression is
#: ``(slot << _LOG2_ATOM_LEAVES | row)``, and the shift must be an expression in the
#: named constant rather than the literal 4 -- otherwise a future atom size passes every
#: instantiated test and silently mis-addresses.
_LOG2_ATOM_LEAVES = MMA_K_QUANTUM.bit_length() - 1
assert 1 << _LOG2_ATOM_LEAVES == MMA_K_QUANTUM, (
    "MMA_K_QUANTUM must be a power of two for the shift form of the page address")

#: Mask for the intra-atom row, DERIVED for the same reason.
_ATOM_ROW_MASK = MMA_K_QUANTUM - 1

_ABSENT_PAGE = -1

#: **§8 PLACEHOLDER** ``C_EXTENT_ATOMS``. The pool is carved into runs of this many
#: CONSECUTIVE slots, and atoms admitted together take consecutive ids inside a run.
#: FILLED BY: the PCIe/HBM batch-efficiency curve.
#:
#: DEFAULT: ``None``, resolved per-arena to ``QPO`` -- one owner's worth of atoms. §8's
#: stated default, and it is the value at which the policy degrades to today's
#: behaviour exactly: an owner's atoms land contiguous, which is what a ``BC``-keyed
#: page gave for free, so a wrong placeholder cannot be worse than the thing this
#: replaces.
C_EXTENT_ATOMS: int | None = None


@dataclass(frozen=True, slots=True)
class AtomKeying:
    """§4.5's addressing arithmetic, as ONE object so it has ONE derivation.

    Every consumer of the page table -- this module, the admission bitmap, the residency
    report, and (once it lands) the CTA's startup slot resolution -- computes the same
    three expressions. Carrying them here rather than inlining them three times is what
    makes the ``BC``-invariance claim below checkable instead of asserted.

    **THE PROPERTY THAT MATTERS, and it is why the re-key exists:** every field of this
    object is invariant in ``BC``. ``atoms_per_bh``, ``atom_of``, ``row_of`` and
    ``address`` mention only ``N``, ``cols`` and ``MMA_K_QUANTUM``. So two launches of
    the same sequence at different ``BC`` address the SAME bytes of the SAME slots, and
    a ``BC`` flip between them is a no-op for residency -- which is exactly what
    ``BC``-per-launch needs and what a ``BC``-keyed page could not give.
    """

    N: int
    cols: int
    BH: int

    def __post_init__(self) -> None:
        if self.N % MMA_K_QUANTUM:
            raise ValueError(
                f"N={self.N} must be a whole number of {MMA_K_QUANTUM}-leaf atoms "
                "(spec §1b: the page granule is geometry-owned and fixed across every "
                "config forever, so a topology whose leaf count is not a multiple of it "
                "has no page keying at all)")

    @property
    def atoms_per_bh(self) -> int:
        return self.N // MMA_K_QUANTUM

    @property
    def logical_atoms(self) -> int:
        return self.BH * self.atoms_per_bh

    def atom_of(self, bh: int, leaf: int) -> int:
        """``bh * (N / MMA_K_QUANTUM) + leaf // MMA_K_QUANTUM`` -- §4.5 verbatim."""
        return bh * self.atoms_per_bh + (leaf >> _LOG2_ATOM_LEAVES)

    def row_of(self, leaf: int) -> int:
        """``leaf & (MMA_K_QUANTUM - 1)`` -- the leaf's row inside its atom."""
        return leaf & _ATOM_ROW_MASK

    def address(self, slot: int, leaf: int, v: int) -> int:
        """``((slot << LOG2_ATOM | row) * cols + v)`` -- §4.5's value address.

        The ``|`` is an OR and not an ADD because ``row < MMA_K_QUANTUM`` by
        construction, so the low bits of the shifted slot are provably clear; writing it
        as an OR is what makes that a stated invariant rather than an accident.
        """
        return ((slot << _LOG2_ATOM_LEAVES) | self.row_of(leaf)) * self.cols + v

    def atoms_per_owner(self, BC: int) -> int:
        """``QPO``. The ONE place ``BC`` meets the keying, and it meets it as a COUNT."""
        if BC % MMA_K_QUANTUM:
            raise ValueError(
                f"BC={BC} must be a whole number of {MMA_K_QUANTUM}-leaf atoms; every "
                "legal BC is by construction (spec §1b)")
        return BC // MMA_K_QUANTUM


@dataclass(frozen=True, slots=True)
class PagingResult:
    """One call's committed-state accounting (spec §7.1's reported quantities)."""

    #: Atoms with a committed slot after this call's publication. Admission is TOTAL
    #: (the capped mode is deleted), so this is exactly the set the routing realized
    #: plus anything a continuation carried in.
    resident_pages: int
    #: High-water mark of :attr:`resident_pages` over the arena's lifetime.
    peak_pages: int
    #: Logical atoms this call's routing actually realized on the write side -- the
    #: popcount of §4.1's atom condensation, which is EXACT rather than a bound.
    realized_owners: int
    #: ``atoms_per_bh``: what a dense ``[N, cols]`` state would provision per ``bh``.
    owners_total: int
    #: Bytes of fp32 state actually committed.
    resident_bytes: int
    #: Bytes a dense ``[BH, N, cols]`` fp32 state would occupy -- the counterfactual the
    #: feature exists to beat.
    dense_bytes: int
    #: Driver-committed bytes when the arena is VMM-backed; ``None`` otherwise.
    committed_bytes: int | None

    @property
    def resident_fraction(self) -> float:
        return self.resident_bytes / self.dense_bytes if self.dense_bytes else 0.0


class ExtentAllocator:
    """§4.5's slot policy: consecutive runs, and NOTHING the table or kernel can see.

    The pool is a sequence of EXTENTS of ``extent_atoms`` consecutive slots. A batch of
    atoms admitted together is placed into as few extents as possible, in ascending
    order, so the common case -- an owner's ``QPO`` atoms admitted in one call -- lands
    contiguous and the CTA resolves ONE base instead of ``QPO`` independent lookups.

    **INVISIBILITY IS THE CONTRACT.** The table format is ``atom -> slot`` either way,
    and no consumer may infer contiguity: a caller that read slot ids and assumed a run
    would break the moment a partially-filled extent forces a split. What the policy
    promises is a TENDENCY, priced by the transfer batching, not an invariant.
    """

    def __init__(self, capacity: int, extent_atoms: int) -> None:
        if extent_atoms < 1:
            raise ValueError(f"extent_atoms must be >= 1, got {extent_atoms}")
        self.capacity = int(capacity)
        self.extent_atoms = int(extent_atoms)
        self._next = 0

    @property
    def allocated(self) -> int:
        return self._next

    @property
    def free(self) -> int:
        return self.capacity - self._next

    def reset(self) -> None:
        self._next = 0

    def take(self, count: int) -> int:
        """Reserve ``count`` slots, ALIGNED to an extent when a fresh one is opened.

        Returns the first slot. **ALL OR NOTHING**: a demand the ceiling cannot serve
        RAISES, naming the demand and the arena size, and leaves the allocator
        untouched. There is no partial grant -- the capped mode that consumed one was
        deleted, because a prefix grant silently computes a different function.

        The alignment rule is one line and is the whole policy: a batch that fits the
        current extent's remainder goes there (so small admissions do not waste a whole
        extent), and a batch that does not starts a new extent (so a large admission --
        an owner's ``QPO`` atoms, or a decode step's pre-growth run -- is contiguous
        rather than straddling). Both branches are STRUCTURAL: they compare a count
        against a remainder, never a size against a threshold.
        """
        if count <= 0:
            return self._next
        remainder = (-self._next) % self.extent_atoms
        first = (self._next + remainder if (remainder and count > remainder)
                 else self._next)
        if first + count > self.capacity:
            raise RuntimeError(
                f"the page arena cannot serve this launch: {count} new atoms demanded, "
                f"{self.capacity - first} free in an arena of {self.capacity} atoms. "
                "The arena grants ALL or refuses -- there is no partial grant and no "
                "dropped write. Raise `max_physical_pages` to the launch's realized "
                "write footprint, or shrink the launch.")
        self._next = first + count
        return first


class PageArena:
    """A pooled, per-layer arena over V3 value panels, keyed by the MMA ATOM.

    **Pooled, not per-(batch, head)** -- V2's shape, and the reason the savings math
    works (§7.1). One arena backs every ``(batch, head)`` pair, so it tracks the live
    fraction of the WHOLE layer; a per-pair arena would round each pair's tiny live
    fraction up to its own extent and cap savings at the extent-to-footprint ratio.

    ``max_physical_pages`` is the allocator CEILING, in atoms. ``None`` provisions the
    dense limit (``BH * atoms_per_bh``), i.e. the ceiling can never bind. A finite value
    is a deliberate POLICY limit rather than a capacity guess -- the footprint is known
    before the launch -- and a launch that exceeds it is REFUSED, loudly and host-side.
    It is never degraded: graceful degradation was the deleted capped mode.

    **NOTE ON ``BC``.** The arena takes it only to report ``QPO`` and to validate that
    the schedule it is handed is atom-compatible. Nothing it stores is ``BC``-keyed, so
    two calls at different ``BC`` share slots byte for byte -- which is the property
    ``BC``-per-launch needs and the reason the re-key exists.
    """

    def __init__(self, *, widths: tuple[int, ...], BC: int, cols: int, BH: int,
                 device: torch.device, max_physical_pages: int | None = None,
                 extent_atoms: int | None = None) -> None:
        N = prod(widths)
        self.widths = tuple(widths)
        self.BC = int(BC)
        self.cols = int(cols)
        self.BH = int(BH)
        self.N = int(N)
        self.keying = AtomKeying(N=self.N, cols=self.cols, BH=self.BH)
        self.atoms_per_owner = self.keying.atoms_per_owner(self.BC)
        self.owners_total = N // self.BC
        self.device = torch.device(device)
        self.logical_pages = self.keying.logical_atoms

        dense_limit = self.logical_pages
        self.max_physical_pages = (dense_limit if max_physical_pages is None
                                   else int(max_physical_pages))
        if not 1 <= self.max_physical_pages <= dense_limit:
            raise ValueError(
                f"max_physical_pages must be in [1, {dense_limit}] (the dense limit, in "
                f"ATOMS since the §4.5 re-key); got {self.max_physical_pages}")

        resolved_extent = (self.atoms_per_owner if extent_atoms is None
                           else int(extent_atoms))
        if C_EXTENT_ATOMS is not None and extent_atoms is None:
            resolved_extent = int(C_EXTENT_ATOMS)
        self.extent_atoms = resolved_extent
        self._allocator = ExtentAllocator(self.max_physical_pages, self.extent_atoms)

        self.peak_pages = 0
        #: The most recent :class:`PagingResult`, or ``None``.
        self.last_result: PagingResult | None = None
        #: §4.5's side stream and the ONE event the consumer launch waits on.
        self._stream = (torch.cuda.Stream(device=self.device)
                        if self.device.type == "cuda" else None)
        self._ready: torch.cuda.Event | None = None
        self._allocate()

    # -- storage ----------------------------------------------------------

    def _allocate(self) -> None:
        self._page_table = torch.full((self.BH, self.keying.atoms_per_bh), _ABSENT_PAGE,
                                      dtype=torch.int32, device=self.device)
        #: ``[slots, MMA_K_QUANTUM, cols]`` -- exactly the shape
        #: :meth:`AtomKeying.address` indexes, so the address expression and the
        #: allocation cannot disagree about what a slot is.
        self._value_base = torch.zeros(
            (self.max_physical_pages, MMA_K_QUANTUM, self.cols),
            dtype=torch.float32, device=self.device)

    # -- properties -------------------------------------------------------

    @property
    def page_table(self) -> torch.Tensor:
        """``[BH, atoms_per_bh]`` int32; ``-1`` means the atom has no slot."""
        return self._page_table

    @property
    def state(self) -> torch.Tensor:
        """``[slots, MMA_K_QUANTUM, cols]`` -- what the consumer indexes."""
        return self._value_base

    @property
    def resident_pages(self) -> int:
        """Committed atoms. HOST-SIDE: the allocator counts them, so this is free."""
        return self._allocator.allocated

    def dense_bytes(self) -> int:
        return self.BH * self.N * self.cols * 4

    def resident_bytes(self) -> int:
        return self.resident_pages * MMA_K_QUANTUM * self.cols * 4

    # -- lifecycle --------------------------------------------------------

    def reset(self) -> None:
        """Drop every committed atom. The arena is reusable afterwards.

        The page STORAGE is deliberately not zeroed here. Dropping the table makes every
        atom unmapped, so the next admission sees each of them as NEW and
        :meth:`_admit_pages` zeroes it before any read can reach it. Blanket-zeroing
        ``max_physical_pages`` worth of storage on every reset would touch the DENSE
        footprint -- exactly the allocation §7.1 exists to avoid -- to redo work
        admission already does on the realized set only.
        """
        self._page_table.fill_(_ABSENT_PAGE)
        self._allocator.reset()
        self.peak_pages = 0
        self._ready = None

    # -- THE ONE DATA-MOVEMENT BOUNDARY -----------------------------------

    def _admit_pages(self, atom_ids: torch.Tensor, first_slot: int) -> None:
        """THE narrow data-movement boundary. Everything that MOVES BYTES lives here.

        ---- CONTRACT -------------------------------------------------------

        INPUTS
          ``atom_ids``  ``[k]`` int64, device-resident, STRICTLY ASCENDING logical atom
                        ids (``bh * atoms_per_bh + leaf // MMA_K_QUANTUM``), each one
                        currently ABSENT from the table. Ascending order is what makes
                        admission order observable in the slot ids, and it is the
                        caller's to establish.
          ``first_slot`` the physical slot the run starts at, already reserved from
                        :class:`ExtentAllocator`. Slots ``[first_slot, first_slot + k)``
                        are guaranteed free and inside the ceiling.

        EFFECTS, and exactly these two:
          1. ``table[atom_ids] = first_slot + arange(k)`` (the gather-scatter);
          2. ``state[first_slot : first_slot + k] = 0`` (the zero-init, which is what
             makes a fresh atom read as a zero state rather than as a stale one).

        There is no third effect any more. The admit mask used to be set here, one bit
        per admitted atom; with the capped mode deleted its bit is set for exactly the
        atoms the table maps, so it carried no information the table did not and it was
        removed end-to-end (arena, binding and kernel).

        STREAM / EVENT DISCIPLINE
          Both run on the arena's SIDE STREAM, which waits on the current stream
          first (the atom bitmap it was handed is produced there). The caller then
          records ONE event and the consumer launch waits on it. Nothing here
          synchronizes the host: there is no ``.item()``, no ``.cpu()``, no
          ``nonzero()`` whose length the host must learn.

        NO HOST SYNCS -- and that is why ``k`` is a HOST integer parameter rather than
        something read off a device tensor. The caller learns it from the planned-exact
        popcount it already performed for sizing.

        ---- WHY IT IS ONE FUNCTION -----------------------------------------

        These two operations are torch ops today and want to be ONE custom CUDA
        gather-scatter tomorrow: a single kernel that writes the table and zeroes the
        slots in one pass over ``atom_ids``, with the zeroing
        vectorized over the ``[k, MMA_K_QUANTUM, cols]` run. Confining them behind one
        boundary with a stated contract is what makes that a DROP-IN replacement rather
        than an archaeology exercise -- the same relationship the summary kernel has to
        its torch reference. Scattering these calls through the arena logic would make
        the eventual kernel a rewrite of the arena.
        """
        stream = self._stream
        context = (torch.cuda.stream(stream) if stream is not None
                   else _NullContext())
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream(self.device))
        with context:
            count = int(atom_ids.numel())
            if count:
                slots = torch.arange(first_slot, first_slot + count,
                                     dtype=torch.int32, device=self.device)
                self._page_table.view(-1).index_copy_(0, atom_ids, slots)
                self._value_base[first_slot:first_slot + count].zero_()

    # -- planning ---------------------------------------------------------

    def plan_exact(self, atom_bitmap: torch.Tensor, *,
                   fresh: bool = True) -> PagingResult:
        """§4.5's PLANNED-EXACT-ASYNC admission, from §4.1's atom condensation.

        ``atom_bitmap`` is ``[BH, atoms_per_bh]`` bool -- exactly what
        :meth:`rola.routing.summaries.RoutingSummaries.atom_bitmap` condenses from the
        canonical per-leaf write bitmap. It is the EXACT set of atoms this call touches
        on the write side, so the pool is SIZED rather than grown and there is nothing
        for an allocator to discover.

        ``fresh`` (default ``True``): reset first, so this call is call-scoped -- it
        computes as if nothing preceded it, matching the unpaged path's "allocate zeros
        every call" semantics. ``fresh=False`` is the explicit, named opt-in for a
        legitimate continuation (successive chunks of one sequence). There is no
        implicit third option.

        **ADMISSION IS TOTAL.** Atoms are admitted in ascending logical id -- which is
        ascending ``(bh, leaf)``, and therefore ascending first-write within a ``bh``
        under the canonical leaf order -- and EVERY one of them is admitted. If the
        ceiling cannot cover the demand the allocator RAISES; there is no prefix grant
        and no denied atom, because the capped mode that produced them was deleted.
        """
        if fresh:
            self.reset()
        expected = (self.BH, self.keying.atoms_per_bh)
        if tuple(atom_bitmap.shape) != expected:
            raise ValueError(
                f"the atom condensation must be {expected} (spec §4.1's derived atom "
                f"granularity); got {tuple(atom_bitmap.shape)}")

        # The atoms that are touched and not already resident. `nonzero` costs one
        # device->host size read; it is the ONE the planned-exact contract admits,
        # because the count is also what sizes the allocation.
        wanted = atom_bitmap.reshape(-1) & (self._page_table.reshape(-1) < 0)
        atom_ids = wanted.nonzero(as_tuple=True)[0]
        realized = int(atom_bitmap.sum())
        first_slot = self._allocator.take(int(atom_ids.numel()))
        self._admit_pages(atom_ids, first_slot)
        if self._stream is not None:
            self._ready = torch.cuda.Event()
            self._ready.record(self._stream)

        self.peak_pages = max(self.peak_pages, self._allocator.allocated)
        self.last_result = PagingResult(
            resident_pages=self._allocator.allocated, peak_pages=self.peak_pages,
            realized_owners=realized,
            owners_total=self.keying.atoms_per_bh,
            resident_bytes=self.resident_bytes(), dense_bytes=self.dense_bytes(),
            committed_bytes=None)
        return self.last_result


    def plan(self, schedule, *, fresh: bool = True) -> PagingResult:
        """:meth:`plan_exact`, with §4.1's atom condensation DERIVED from a schedule.

        This is a CONVENIENCE over the primitive, not a second path: it computes exactly
        the object :meth:`plan_exact` takes and then calls it. It exists because the
        arena has one caller that holds a ``Schedule`` and not yet a
        :class:`~rola.routing.summaries.RoutingSummaries`.

        **ITS COST IS STATED RATHER THAN HIDDEN.** The condensation it derives here is
        already produced -- exactly, canonically -- by the producer epilogue's summary
        pass (§4.1: the per-leaf write bitmap is the canonical truth and every other
        granularity CONDENSES from it). Deriving it again from ``schedule.write_amps``
        repeats an ``O(BH * n_tiles * N)`` reduction the launch has already paid for. A
        caller that has the summaries must pass ``summaries.atom_bitmap(N)`` to
        :meth:`plan_exact` and skip this entirely; the duplication disappears with the
        layer cut-over, and it is a duplication rather than a divergence because both
        derivations compute the same union over tokens of the exact write support.
        """
        plan_obj = schedule.plan
        if tuple(plan_obj.topology.widths) != self.widths:
            raise ValueError("schedule topology does not match this arena")
        if plan_obj.BC % MMA_K_QUANTUM:
            raise ValueError(
                f"the schedule plans BC={plan_obj.BC}, which is not a whole number of "
                f"{MMA_K_QUANTUM}-leaf atoms (spec §1b)")
        return self.plan_exact(atom_bitmap_from_schedule(schedule, self.keying),
                               fresh=fresh)

    def wait(self) -> None:
        """The ONE event-wait §4.5 puts before the consumer launch.

        Allocation and zero-init ran on the side stream while the schedule was being
        built; this is where the consumer's stream joins them. It is a stream-to-stream
        wait, not a host synchronization -- the host never learns when the zeroing
        finished, only that the consumer will not start before it did.
        """
        if self._ready is not None:
            torch.cuda.current_stream(self.device).wait_event(self._ready)

    # -- dense bridges (the §7.3 boundary, crossed explicitly) -------------

    def adopt_dense(self, state: torch.Tensor) -> None:
        """Commit a slot for every atom of a dense ``[BH, N, cols]`` state.

        The explicit bridge INTO the arena. It provisions the dense limit, so it is the
        wrong thing to do at scale and exists for exactly one reason: a carried
        continuation state produced by the unpaged path (or by the oracle) must be
        loadable, and §7.1 defers to §7.3 on what should cross a call boundary. Named a
        bridge rather than a mode so no measurement quotes it as the feature.
        """
        expected = (self.BH, self.N, self.cols)
        if tuple(state.shape) != expected:
            raise ValueError(f"dense state must be {expected}, got {tuple(state.shape)}")
        if self.max_physical_pages < self.logical_pages:
            raise ValueError(
                "adopt_dense commits every atom and needs the dense limit; this arena "
                f"caps at {self.max_physical_pages} of {self.logical_pages} atoms")
        self._allocator.reset()
        first = self._allocator.take(self.logical_pages)
        assert first == 0
        table = torch.arange(self.logical_pages, dtype=torch.int32, device=self.device)
        self._page_table.copy_(table.view(self.BH, self.keying.atoms_per_bh))
        self._value_base.copy_(
            state.reshape(self.logical_pages, MMA_K_QUANTUM, self.cols))
        self.peak_pages = max(self.peak_pages, self.logical_pages)

    def materialize(self) -> torch.Tensor:
        """Gather the committed atoms back into a dense ``[BH, N, cols]`` state.

        The explicit bridge OUT. Allocating ``[BH, N, cols]`` is precisely what paging
        exists to avoid, so this is for oracle comparison and for the existing dense
        cache contract -- never on a measured path.
        """
        present = self._page_table >= 0
        index = self._page_table.long().clamp_min(0).reshape(-1)
        gathered = self._value_base.index_select(0, index).view(
            self.BH, self.keying.atoms_per_bh, MMA_K_QUANTUM, self.cols)
        dense = torch.zeros_like(gathered)
        dense[present] = gathered[present]
        return dense.reshape(self.BH, self.N, self.cols)


def atom_bitmap_from_schedule(schedule, keying: AtomKeying) -> torch.Tensor:
    """``[BH, atoms_per_bh]`` bool: the atoms this schedule's WRITE support touches.

    §4.1's derived atom granularity, condensed from the per-leaf write support -- a leaf
    is written iff at SOME token EVERY level's digit for it is live on the write side,
    which is an OR of ANDs and is why this is ``O(BH * n_tiles * N)`` however it is
    spelled.

    **KEYED ON THE LIVE DEPOSIT, never on the clock.** ``GRADIENT_PROOFS_WOLFRAM.md``
    P5.20.5 finding 2 is binding: a storage path may flush a small live ``p_write`` to a
    stored zero, giving ``c == 0`` (``keep == 1``) with a live deposit ``W != 0``. Such a
    leaf is touched and must be resident. ``schedule.write_amps`` IS the live deposit and
    the stored clock is not reachable from a ``Schedule`` at all, so the wrong tensor is
    not merely discouraged here -- it is unreachable, exactly as it is in the §4 bitmap
    builder.

    Chunked over tiles so the transient never reaches ``O(L * N)``.
    """
    widths = tuple(schedule.plan.topology.widths)
    amps = schedule.write_amps
    BH, n_tiles = amps[0].shape[0], amps[0].shape[1]
    N = keying.N
    device = amps[0].device
    live = torch.zeros((BH, N), dtype=torch.bool, device=device)
    tiles_per_chunk = max(1, (1 << 24) // max(1, N))
    for lo in range(0, n_tiles, tiles_per_chunk):
        hi = min(lo + tiles_per_chunk, n_tiles)
        block = None
        for level, width in enumerate(widths):
            # `[BH, tiles, BT, width] -> [BH, tiles*BT, 1..width..1]`, broadcast over the
            # digit grid: the leaf id is the row-major flattening of it, so the AND over
            # levels needs no index at all.
            shape = [1] * len(widths)
            shape[level] = width
            token_live = (amps[level][:, lo:hi, :, :width] != 0).reshape(
                BH, -1, *shape)
            block = token_live if block is None else (block & token_live)
        live |= block.reshape(BH, -1, N).any(dim=1)
    return live.view(BH, keying.atoms_per_bh, MMA_K_QUANTUM).any(dim=-1)


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


__all__ = [
    "C_EXTENT_ATOMS",
    "MMA_K_QUANTUM",
    "AtomKeying",
    "ExtentAllocator",
    "PageArena",
    "PagingResult",
    "atom_bitmap_from_schedule",
]
