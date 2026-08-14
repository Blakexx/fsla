# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The carried recurrent state: ONE object, one predicate (docs/internals/state.md).

**A fixed-size recurrent state is not a KV cache.** A KV cache grows with the
sequence; RoLA's carried state is `[B, H, N, cols]` -- a shape that does not mention
`T` at all -- which is why nothing here borrows `transformers`' cache vocabulary
(concatenation, eviction, window rolling) and why the package's only hard
dependencies stay `torch` + `einops`.

    s = rola.state()                     # unbound; keying binds at the first call
    y1, s = rola.rola_op(routes1, v1, state=s)
    y2, s = rola.rola_op(routes2, v2, state=s)

`state=None` is the stateless call and returns `(y, None)`. That predicate --
`state is not None` -- is the whole engagement signal at the op and at the layer.

What the object holds and what it does NOT hold is the design's closed surface
(`s0_design.md` amendment 11): a strategy stamp, the keying it bound to, and the
backing. No step stamp, no lineage, no staleness machinery, no wrapper tier. A
state is a buffer with a contract, exactly like a `torch` tensor is, and using one
across a branch is the caller's business rather than something this class polices.

The backing is DENSE or PAGED and that is invisible to a caller: both spell the
same math and the same bytes (`tests/tier2_geometry/test_chunk_paging_equivalence.py`
asserts it with `torch.equal`), and only the footprint differs.
"""

from __future__ import annotations

import torch

from rola.ops.paging import MMA_K_QUANTUM, AtomKeying, PageArena

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

    Construct through :func:`rola.state`. The first stateful call binds the keying
    (`N`, `cols`, `BH`), the topology fingerprint, the device and the backing from
    the call itself; every later call is checked against those for
    MATH-COMPATIBILITY only -- a disagreement is the wrong-shaped-tensor class of
    error, not a policy refusal.

    ``strategy`` is the construction-time stamp. ``"mutable"`` is the built one:
    the call writes in place and returns THIS object. Changing a state's strategy is
    :meth:`clone`, never a mode flip on a live object.
    """

    __slots__ = ("strategy", "conv_state", "_keying", "_widths", "_normalization",
                 "_batch", "_heads", "_device", "_paging", "_BC", "_arena", "_plane",
                 "_decode_carrier_cache", "_decode_workspace")

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
        #: The layer's short-convolution rings, carried ALONGSIDE the routed state.
        #: They are the layer's lifecycle sugar and the op never reads them.
        self.conv_state: tuple[torch.Tensor | None, ...] | None = None
        #: The `T = 1` path's LAUNCH MACHINERY -- the frozen carrier and its scratch --
        #: cached where the sequence is, for the same reason `conv_state` is: it is
        #: per-sequence and it is not a caller's burden. Neither is read by the chunk
        #: arm, and a clone does not copy them (they rebuild from their own key).
        self._decode_carrier_cache = None
        self._decode_workspace = None
        self._keying: AtomKeying | None = None
        self._widths: tuple[int, ...] | None = None
        self._normalization: str | None = None
        self._batch = 0
        self._heads = 0
        self._device: torch.device | None = None
        self._paging = True
        self._BC = 0
        self._arena: PageArena | None = None
        self._plane: torch.Tensor | None = None

    # -- what a caller may ask -------------------------------------------------

    def __repr__(self) -> str:
        if not self.bound:
            return f"RoLAState(strategy={self.strategy!r}, unbound)"
        return (f"RoLAState(strategy={self.strategy!r}, widths={self._widths}, "
                f"BH={self._keying.BH}, cols={self._keying.cols}, "
                f"backing={'paged' if self.paged else 'dense'}, device={self._device})")

    @property
    def bound(self) -> bool:
        """Whether a call has bound this state's keying, device and backing yet."""
        return self._keying is not None

    @property
    def paged(self) -> bool:
        """Whether the CURRENT backing is an arena. Not a mode: a footprint."""
        return self._arena is not None

    def _wants_pages(self, paging: bool) -> bool:
        """Whether this call needs an atom bitmap -- i.e. whether it will PAGE.

        Asked by the facade before the stats pass, so a dense-backed state does not
        pay for a bitmap nothing reads. Once bound, the state's own backing preference
        answers it; the argument is only consulted on the call that binds.
        """
        return self._paging if self.bound else bool(paging)

    @property
    def carries(self) -> bool:
        """Whether anything is STORED yet. A bound state can still be empty.

        Binding fixes the keying and the backing kind; contents arrive with the first
        call that finishes. The two are separate because a call may bind a state and
        then have nothing to carry (a fresh sequence's entry state is no state at all).
        """
        return self._arena is not None or self._plane is not None

    @property
    def keying(self) -> AtomKeying | None:
        return self._keying

    @property
    def device(self) -> torch.device | None:
        return self._device

    def materialize(self) -> torch.Tensor | None:
        """`[B, H, N, cols]`, or `None` while nothing is carried. ORACLE AND BRIDGE ONLY.

        Allocating `[B, H, N, cols]` is precisely what paging exists to avoid, so
        this is for the fp64 oracle, for a cross-arm hand-off and for a test -- never
        on a measured path.
        """
        if not self.carries:
            return None
        shape = (self._batch, self._heads, self._keying.N, self._keying.cols)
        if self._arena is not None:
            return self._arena.materialize().view(shape)
        return self._plane.view(shape)

    def clone(self, strategy: str | None = None) -> RoLAState:
        """An EAGER deep copy, and the only sanctioned strategy change.

        The copy shares nothing: a paged state's copy gets its own arena with the
        same residency, a dense one's its own plane. Changing strategy crosses a
        clone by rule (`s0_design.md` amendment 10 revised) -- a live ancestor is a
        standing alias onto shared slots, so a cheaper transition would hand a
        descendant a guarantee its ancestor can break.
        """
        out = RoLAState(strategy=self.strategy if strategy is None else strategy)
        out.conv_state = None if self.conv_state is None else tuple(
            None if t is None else t.clone() for t in self.conv_state)
        if not self.bound:
            return out
        out._keying, out._widths = self._keying, self._widths
        out._normalization, out._batch, out._heads = self._normalization, self._batch, self._heads
        out._device, out._paging, out._BC = self._device, self._paging, self._BC
        if self._arena is None:
            out._plane = self._plane.clone()
            return out
        arena = out._new_arena()
        arena.plan_exact(self._arena.page_table >= 0, fresh=True)
        arena.wait()
        source, destination = self._arena.page_table, arena.page_table
        present = source >= 0
        arena.state.index_copy_(
            0, destination[present].long(),
            self._arena.state.index_select(0, source[present].long()))
        out._arena = arena
        return out

    def adopt_dense_contents(self, dense: torch.Tensor) -> None:
        """Store a dense `[B, H, N, cols]` state, replacing whatever backed this one.

        The bridge back FROM the reference arm, which keeps raw tensors by spec
        (`rola/ops/naive.py`'s signature is the oracle's contract). The backing
        becomes the dense plane holding exactly these contents, in the dtype they
        arrive in -- the fp64 oracle's result stays fp64, so a reference-arm chain
        never rounds through fp32. A kernel-arm call on such a state re-admits from
        the contents' own atom occupancy (:meth:`_kernel_entry`).

        Deliberately NOT `PageArena.adopt_dense`: that commits the dense limit, which
        is the footprint the paged backing exists to avoid.
        """
        if not self.bound:
            raise RuntimeError(
                "adopt_dense_contents needs a bound state: the keying it would store "
                "against comes from a call, and this state has not seen one")
        expected = (self._batch, self._heads, self._keying.N, self._keying.cols)
        if tuple(dense.shape) != expected:
            raise ValueError(f"a dense state must be {expected}, got {tuple(dense.shape)}")
        self._arena = None
        self._plane = dense.reshape(
            self._keying.BH, self._keying.atoms_per_bh, MMA_K_QUANTUM,
            self._keying.cols).contiguous()

    # -- the facade's seam (rola/interface.py is the only caller) --------------

    def _bind(self, routes, *, paging: bool) -> None:
        """Take the keying, the topology fingerprint, the device and the backing kind.

        Binding happens ONCE, at the first call that receives this state. The paging
        preference is read here and nowhere else: a backing is chosen, and changing
        it afterwards is a :meth:`clone`, not a per-call flag.
        """
        topology = routes.topology
        self._widths = tuple(topology.widths)
        self._normalization = routes.normalization
        self._batch, self._heads = routes.batch, routes.heads
        self._device = routes.device
        self._paging = bool(paging)
        self._keying = AtomKeying(
            N=topology.N, cols=topology.d_v + (1 if routes.normalization == "global" else 0),
            BH=routes.batch * routes.heads)

    def _require_compatible(self, routes) -> None:
        """Math-compatibility, and only that: keying, topology, device.

        Each of these makes the call literally uncomputable against this state -- a
        different leaf count addresses different rows, a different `cols` a different
        stride, a different device different memory. Everything else (who else holds
        this object, what order the calls came in) is the caller's business.
        """
        topology = routes.topology
        if tuple(topology.widths) != self._widths:
            raise ValueError(
                f"this state is bound to widths={self._widths}; the call routes "
                f"{tuple(topology.widths)}")
        if routes.normalization != self._normalization:
            raise ValueError(
                f"this state is bound to normalization={self._normalization!r}; the call "
                f"is {routes.normalization!r}, which is a different column count")
        if (routes.batch, routes.heads) != (self._batch, self._heads):
            raise ValueError(
                f"this state is bound to (batch, heads)=({self._batch}, {self._heads}); "
                f"the call is ({routes.batch}, {routes.heads})")
        if routes.device != self._device:
            raise ValueError(
                f"this state lives on {self._device}; the call is on {routes.device}")

    def _kernel_entry(self, routes, atom_bits: torch.Tensor | None, *, paging: bool,
                      BC: int):
        """`(s_in, arena)` for the kernel arm, with this call's pages committed.

        The host order the design fixes (§A.4/§C.3): the caller has already run the
        stats pass, so the only thing that happens here before the launch is
        `plan_exact` on the arena's side stream -- one device-to-host size read, no
        other synchronization. The launch's `arena.wait()` (in `rola.ops.chunk.run`)
        is what puts the admission before the kernel.
        """
        if not self.bound:
            self._bind(routes, paging=paging)
        else:
            self._require_compatible(routes)
        #: `BC` reaches the arena as allocator POLICY only (the extent default is one
        #: owner's atoms); every ADDRESS is `BC`-invariant by the atom re-key, so a
        #: `BC` flip between calls neither rebuilds the arena nor moves a byte.
        self._BC = int(BC)
        bitmap = (None if atom_bits is None else
                  atom_bits.reshape(self._keying.BH, self._keying.atoms_per_bh).bool())

        if self._arena is not None:
            #: a continuation: residency becomes the UNION, which is what continuing
            #: a sequence means, and the atoms already committed keep their contents.
            self._arena.plan_exact(bitmap, fresh=False)
            return self._arena.state, self._arena
        if self._plane is None:
            #: nothing carried yet -- a fresh sequence, whichever call bound the state.
            if not self._paging:
                return None, None
            arena = self._new_arena()
            arena.plan_exact(bitmap, fresh=True)
            self._arena = arena
            return None, arena
        if self._paging:
            #: a reference-arm result returning to the kernel arm: the dense contents'
            #: OWN atom occupancy joins this call's write set, so the admission covers
            #: what is carried as well as what will be written.
            s_in = self._readmit(bitmap)
            return s_in, self._arena
        if self._plane.dtype is not torch.float32:
            self._plane = self._plane.to(torch.float32)
        return self._plane, None

    def _kernel_commit(self, plane: torch.Tensor) -> None:
        """Take the kernel's final-state plane. Paged states already own theirs."""
        if self._arena is None:
            self._plane = plane

    def _decode_carrier(self, routes, *, decay, mma_precision: str):
        """The `T = 1` frozen carrier for this sequence, built once and cached HERE.

        It freezes at the prefill/decode boundary and is immutable for the sequence
        (`docs/internals/decode.md`), so it is rebuilt only when its own key moves --
        `BH`, the device, and the decay arm, the three facts it is a function of that
        the state does not already pin. Everything else in it is a topology fact, and
        the topology is what this state is bound to.
        """
        from rola.ops.decode import freeze_decode_config

        key = (routes.batch * routes.heads, routes.device, decay is not None,
               mma_precision)
        cached = self._decode_carrier_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        config = freeze_decode_config(
            routes.topology, routes.normalization, decay=decay is not None,
            BH=key[0], device=routes.device, mma_precision=mma_precision)
        self._decode_carrier_cache = (key, config)
        return config

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
        self._require_compatible(routes)
        if self._arena is not None:
            #: GROWTH. The atoms this step deposits into join the resident set on the
            #: arena's side stream; `decode_forward`'s `arena.wait()` is what puts that
            #: admission before the launch. An atom neither resident nor written stays
            #: ABSENT and reads inert.
            self._arena.plan_exact(bitmap, fresh=False)
            return self._arena.state, self._arena.page_table, self._arena
        if self._plane.dtype is not torch.float32:
            self._plane = self._plane.to(torch.float32)
        return (self._plane.view(self._batch, self._heads, self._keying.N,
                                 self._keying.cols),
                None, None)

    def _dense_entry(self, routes, *, paging: bool) -> torch.Tensor | None:
        """The `[B, H, N, cols]` entry state the fp64 oracle takes, or `None`.

        The reference arm keeps RAW TENSORS by spec, so a paged state materializes
        here. It is the expensive direction of the bridge and it is the one the
        design accepts: the oracle is not a measured path.
        """
        if not self.bound:
            self._bind(routes, paging=paging)
            return None
        self._require_compatible(routes)
        return self.materialize()

    # -- backing internals -----------------------------------------------------

    def _new_arena(self) -> PageArena:
        return PageArena(widths=self._widths, BC=self._BC, cols=self._keying.cols,
                         BH=self._keying.BH, device=self._device)

    def _readmit(self, bitmap: torch.Tensor) -> torch.Tensor:
        """Dense contents -> pages, admitting occupancy UNION this call's write set.

        Occupancy is a nonzero reduction at atom grain, which is exact in the safe
        direction: an atom holding nothing but zeros reads as zeros whether it is
        resident or absent (the zero-page inertness gate), so admitting on nonzero
        contents can only ever admit a superset of what is carried.
        """
        plane = self._plane
        occupied = (plane != 0).any(dim=-1).any(dim=-1)
        arena = self._new_arena()
        arena.plan_exact(occupied | bitmap, fresh=True)
        arena.wait()
        table = arena.page_table
        present = table >= 0
        arena.state.index_copy_(
            0, table[present].long(), plane[present].to(torch.float32))
        self._arena, self._plane = arena, None
        return arena.state
