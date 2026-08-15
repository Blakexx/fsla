# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The carried recurrent state: A PAGED TENSOR, and nothing more.

Storage (an arena or a dense plane), a page table, the logical shape
``[B, H, N, cols]``, a device and a dtype, and the storage-mechanics stamp
(``strategy``). Everything else that used to ride here failed one question --
*would a paged tensor know this?* -- and lives where it belongs now
(docs/internals/state.md).

    s = rola.state()                     # unbound; the first call binds the shape
    y1, s = rola.rola_op(routes1, v1, state=s)
    y2, s = rola.rola_op(routes2, v2, state=s)

``state=None`` is the stateless call and returns ``(y, None)``. That predicate --
``state is not None`` -- is the whole engagement signal at the op and at the layer.

**A fixed-size recurrent state is not a KV cache.** A KV cache grows with the
sequence; this shape does not mention ``T`` at all, which is why nothing here borrows
``transformers``' cache vocabulary (concatenation, eviction, window rolling).

Using one across a branch, or against a call it was not bound by, is the CALLER's
business exactly as it is for a ``torch`` tensor: a disagreement fails as a shape
error where the bytes are addressed, and nothing here polices it in advance.
"""

from __future__ import annotations

import torch

from rola.ops.paging import AtomKeying, PageArena

__all__ = ["RoLAState", "state"]

#: The paging strategies this version BUILDS. `"cow"` is named in the design as the
#: next one and is refused by name rather than by a generic message, so a caller who
#: asks for it learns it is scheduled rather than misspelled.
_BUILT_STRATEGIES = ("mutable",)
_NAMED_UNBUILT_STRATEGIES = ("cow",)


def state(strategy: str = "mutable") -> RoLAState:
    """A fresh UNBOUND state. Everything but the strategy binds at the first call."""
    return RoLAState(strategy=strategy)


class RoLAState:
    """One sequence's carried state, mutated in place and handed back.

    Construct through :func:`rola.state`. The first stateful call binds the shape, the
    device and the backing from the call itself -- binding is SHAPE ACQUISITION, not a
    fingerprint, and there is no later compatibility interview.

    ``strategy`` is a storage-mechanics fact, the same class as a tensor's layout: it
    governs what a write does to this object's own storage. ``"mutable"`` is the built
    one -- the call writes in place and returns THIS object. Changing it is
    :meth:`clone`, never a mode flip on a live object.
    """

    __slots__ = ("strategy", "_shape", "_device", "_paging", "_arena", "_plane",
                 "__weakref__")

    def __init__(self, *, strategy: str = "mutable") -> None:
        if strategy in _NAMED_UNBUILT_STRATEGIES:
            raise NotImplementedError(
                f"strategy={strategy!r} is designed and NOT BUILT: copy-on-write state "
                "branching is named future work (s0_design.md), and its slot refcounts "
                "and per-branch tables do not exist yet. The built strategy is 'mutable'.")
        if strategy not in _BUILT_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {_BUILT_STRATEGIES}, got {strategy!r}")
        self.strategy = strategy
        self._shape: tuple[int, int, int, int] | None = None
        self._device: torch.device | None = None
        self._paging = True
        self._arena: PageArena | None = None
        self._plane: torch.Tensor | None = None

    # -- what a caller may ask -------------------------------------------------

    def __repr__(self) -> str:
        if not self.bound:
            return f"RoLAState(strategy={self.strategy!r}, unbound)"
        return (f"RoLAState(strategy={self.strategy!r}, shape={self._shape}, "
                f"backing={'paged' if self.paged else 'dense'}, device={self._device})")

    @property
    def bound(self) -> bool:
        """Whether a call has bound this state's shape, device and backing yet."""
        return self._shape is not None

    @property
    def shape(self) -> tuple[int, int, int, int] | None:
        """``[B, H, N, cols]``, or ``None`` while unbound."""
        return self._shape

    @property
    def device(self) -> torch.device | None:
        return self._device

    @property
    def dtype(self) -> torch.dtype | None:
        """The STORAGE's dtype, or ``None`` while nothing is carried."""
        if self._arena is not None:
            return self._arena.state.dtype
        return None if self._plane is None else self._plane.dtype

    @property
    def paged(self) -> bool:
        """Whether the CURRENT backing is an arena. Not a mode: a footprint."""
        return self._arena is not None

    @property
    def carries(self) -> bool:
        """Whether anything is STORED yet. A bound state can still be empty.

        Binding fixes the shape and the backing kind; contents arrive with the first
        call that finishes. The two are separate because a call may bind a state and
        then have nothing to carry (a fresh sequence's entry state is no state at all).
        """
        return self._arena is not None or self._plane is not None

    @property
    def page_table(self) -> torch.Tensor | None:
        """The arena's ``[BH, N/16]`` table, or ``None`` for a dense backing."""
        return None if self._arena is None else self._arena.page_table

    def materialize(self) -> torch.Tensor | None:
        """`[B, H, N, cols]`, or `None` while nothing is carried. ORACLE AND BRIDGE ONLY.

        Allocating `[B, H, N, cols]` is precisely what paging exists to avoid, so
        this is for the fp64 oracle, for a cross-arm hand-off and for a test -- never
        on a measured path.
        """
        if not self.carries:
            return None
        if self._arena is not None:
            return self._arena.materialize().view(self._shape)
        return self._plane.view(self._shape)

    def clone(self, strategy: str | None = None) -> RoLAState:
        """An EAGER deep copy, and the only sanctioned strategy change.

        The copy shares nothing: a paged state's copy gets its own arena with the
        same residency, a dense one's its own plane. Changing strategy crosses a
        clone by rule (`s0_design.md` amendment 10 revised) -- a live ancestor is a
        standing alias onto shared slots, so a cheaper transition would hand a
        descendant a guarantee its ancestor can break.
        """
        out = RoLAState(strategy=self.strategy if strategy is None else strategy)
        if not self.bound:
            return out
        out._shape, out._device, out._paging = self._shape, self._device, self._paging
        if self._arena is None:
            if self._plane is not None:
                out._plane = self._plane.clone()
            return out
        arena = out._new_arena(self._arena.widths, self._arena.BC)
        arena.plan_exact(self._arena.page_table >= 0, fresh=True)
        arena.wait()
        source, destination = self._arena.page_table, arena.page_table
        present = source >= 0
        arena.state.index_copy_(
            0, destination[present].long(),
            self._arena.state.index_select(0, source[present].long()))
        out._arena = arena
        return out

    # -- the facade's seam (rola/interface.py and rola/ops/decode.py) -----------

    @property
    def _keying(self) -> AtomKeying:
        B, H, N, cols = self._shape
        return AtomKeying(N=N, cols=cols, BH=B * H)

    def _wants_pages(self, paging: bool) -> bool:
        """Whether this call needs an atom bitmap -- i.e. whether it will PAGE.

        Asked by the facade before the stats pass, so a dense-backed state does not
        pay for a bitmap nothing reads. Once bound, the state's own backing preference
        answers it; the argument is only consulted on the call that binds.
        """
        return self._paging if self.bound else bool(paging)

    def _kernel_entry(self, routes, atom_bits: torch.Tensor | None, *, d_v: int,
                      paging: bool, BC: int):
        """`(s_in, arena)` for the kernel arm, with this call's pages committed.

        The host order (docs/internals/state.md#host-order): the caller has already run the
        stats pass, so the only thing that happens here before the launch is
        `plan_exact` on the arena's side stream -- one device-to-host size read, no
        other synchronization. The launch's `arena.wait()` (in `rola.ops.chunk.run`)
        is what puts the admission before the kernel.

        `BC` arrives FROM THE CALL and is never stored: it is allocator policy (the
        extent default is one owner's atoms), and every ADDRESS is `BC`-invariant by
        the atom re-key, so a `BC` flip between calls neither rebuilds the arena nor
        moves a byte.
        """
        if not self.bound:
            self._shape = (routes.batch, routes.heads, routes.topology.N, d_v + 1)
            self._device = routes.device
            self._paging = bool(paging)
        bitmap = (None if atom_bits is None else
                  atom_bits.reshape(self._keying.BH, self._keying.atoms_per_bh).bool())

        if self._arena is not None:
            #: a continuation: residency becomes the UNION, which is what continuing
            #: a sequence means, and the atoms already committed keep their contents.
            self._arena.plan_exact(bitmap, fresh=False)
            return self._arena.state, self._arena
        if self._plane is not None:
            return self._plane, None
        if not self._paging:
            return None, None
        arena = self._new_arena(routes.topology.widths, BC)
        arena.plan_exact(bitmap, fresh=True)
        self._arena = arena
        return None, arena

    def _kernel_commit(self, plane: torch.Tensor) -> None:
        """Take the kernel's final-state plane. Paged states already own theirs."""
        if self._arena is None:
            self._plane = plane

    def _decode_entry(self, routes, bitmap: torch.Tensor | None):
        """`(plane, page_table, arena)` for one decode step, with its atoms committed.

        A step CONTINUES a sequence, so unlike :meth:`_kernel_entry` there is no fresh
        case to serve: the state carries, and the only question is whether this step
        writes an atom that is not resident yet. `plan_exact(..., fresh=False)` answers
        it -- residency becomes the UNION, admission is total, and the one size read it
        costs is the only host synchronization in the step.

        Dense states hand back a VIEW of their own plane, which is what makes the
        kernel's in-place update the state's update.
        """
        if not self.carries:
            raise RuntimeError(
                "a decode step continues a carried state, and this one carries nothing: "
                "at T = 1 there is no prefill to run and no entry state to read. Run the "
                "sequence's first tokens through rola_op and step from the state it "
                "returns.")
        if self._arena is not None:
            #: `bitmap` is `None` when the growth probe says nothing is new. Admission
            #: runs on the ARENA's side stream, so `decode_forward`'s `arena.wait()` is
            #: what orders it before the launch (docs/internals/state.md#host-order).
            if bitmap is not None:
                self._arena.plan_exact(bitmap, fresh=False)
            return self._arena.state, self._arena.page_table, self._arena
        return self._plane.view(self._shape), None, None

    # -- backing internals -----------------------------------------------------

    def _new_arena(self, widths: tuple[int, ...], BC: int) -> PageArena:
        keying = self._keying
        return PageArena(widths=widths, BC=int(BC), cols=keying.cols, BH=keying.BH,
                         device=self._device)
