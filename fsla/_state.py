# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The carried recurrent state: an interface, :class:`FSLAState`, and its one implementation, :class:`DenseState`.

The interface is what a caller may ask of a state (its format, bound shape, device, dtype, whether it carries
anything, a ``materialize`` and a ``clone``) and the seam the ops take the plane in and hand it back through. The
dense state is the one implementation: one ``[BH, N/16, page_bytes]`` plane of the stored page format, the whole
leaf space resident. Another implementation takes whatever parameters its own storage needs and is built by its
own constructor (docs/internals/state.md#interface).

    s = fsla.dense_state()               # unbound; the first call binds the shape
    y1, s = fsla.prefill(routes1, values1, masks1, state=s)
    y2, s = fsla.prefill(routes2, values2, masks2, state=s)

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

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from fsla.engine.types import MMA_K_QUANTUM

__all__ = ["DenseState", "FSLAState", "StateFormat", "dense_state"]

#: The page rectangle, as a bit count: an atom is the trailing
#: ``log2(MMA_K_QUANTUM)`` canonical leaf bits, and the page IS that rectangle.
PAGE_RECTANGLE_BITS = MMA_K_QUANTUM.bit_length() - 1

#: The one canonical leaf order a descriptor may name today. It is a FIELD rather
#: than an assumption so a second order is a refusal at the seam and not a silent
#: reinterpretation of somebody's bytes.
CANONICAL_ORDER = "mixed-radix-msb-first"

#: THE STATE FORMAT (ruled 2026-09-28): a leaf-channel is an s24 at its (leaf, 32-channel group)'s s8
#: power-of-two exponent, stored as an s16 hi plane and a u8 lo plane; the mass is fp32
#: (`fsla.ops.paging.page_layout`, docs/internals/common/state_page.md).
STATE_DTYPE = "s24 hi/lo planes at an s8 group exponent, fp32 mass"

#: THE UNPORTED CARRY'S OWN PAGE, fp32 split into two bf16 planes: it describes the carry's
#: descriptor alone, and the carry port deletes it.
SPLIT_PLANE_DTYPE = "fp32 as split bf16 planes"

#: The per-level width floor (``B_l`` a power of two at or above 16 = ``K_max``), which
#: is what makes every span fit inside one level's run and the atom stay inside it.
MIN_LEVEL_WIDTH = 16


@dataclass(frozen=True, slots=True)
class StateFormat:
    """THE STATE OWNS ITS FORMAT: how a kernel reads these bytes.

    Bound from the FIRST call that receives the state and CHECKED against every later
    one, which is what replaces the old "later calls are NOT checked" clause: a state
    handed to a call it does not fit is refused at the seam, by name, instead of
    mis-addressing bytes somewhere inside a kernel.

    ``dtype`` is the LOGICAL element type -- fp32 -- and the split-plane form is how
    that word is laid out inside a page, not a narrowing of it
    (docs/internals/state.md#format).
    """

    D: int
    B: tuple[int, ...]
    order: str
    page_bits: int
    DV: int
    dtype: str
    ids: int

    def __post_init__(self) -> None:
        if len(self.B) != self.D:
            raise ValueError(f"a descriptor names D={self.D} levels and {len(self.B)} widths")
        for level, width in enumerate(self.B):
            if width & (width - 1) or width < MIN_LEVEL_WIDTH:
                raise ValueError(
                    f"B_l is a power of two at or above {MIN_LEVEL_WIDTH} (the axis law: the "
                    f"floor is K_max, which is what keeps every span inside one level's run and "
                    f"the atom inside one level); level {level} is {width}")
        if self.order != CANONICAL_ORDER:
            raise ValueError(f"the only leaf order this build addresses is {CANONICAL_ORDER!r}")
        if self.page_bits != PAGE_RECTANGLE_BITS:
            raise ValueError(
                f"the page rectangle is the trailing {PAGE_RECTANGLE_BITS} canonical bits")
        if self.dtype == STATE_DTYPE and self.DV % 32:
            raise ValueError(f"a channel group is 32 channels, so DV is a multiple of 32; got {self.DV}")
        if self.dtype == SPLIT_PLANE_DTYPE and self.DV % 2:
            raise ValueError(
                f"the split planes address a PAIR of value columns per 32-bit access, so DV is "
                f"even; got {self.DV}")
        if self.N % (1 << self.page_bits):
            raise ValueError(
                f"THERE IS NO RAGGED STATE: N = {self.N} is not a whole number of "
                f"{1 << self.page_bits}-leaf pages. Under the law above N = prod_l B_l with "
                f"every B_l a power of two at or above 16, so a lawful state is page aligned "
                f"by construction; a ragged one is refused here rather than padded or split "
                f"across a tail page (docs/internals/state.md#format)")

    @property
    def N(self) -> int:
        """Leaf capacity -- the mixed-radix product, derived and never configured."""
        n = 1
        for width in self.B:
            n *= width
        return n

    @property
    def cols(self) -> int:
        """``DV`` plus the mass column: the page's LOGICAL width, not its layout."""
        return self.DV + 1

    @property
    def atoms_per_bh(self) -> int:
        return self.N >> self.page_bits

    @property
    def page_bytes(self) -> int:
        """One page's bytes: the int8 state page (`fsla.ops.paging.page_layout`), or the carry's split fp32 page."""
        if self.dtype == STATE_DTYPE:
            from fsla.ops.paging import page_layout

            return page_layout(self.DV).bytes
        return (1 << self.page_bits) * self.cols * 4

    def check(self, other: StateFormat) -> None:
        """Refuse ``other`` against the bound format, naming the field that differs."""
        for field in ("D", "B", "order", "page_bits", "DV", "dtype", "ids"):
            mine, theirs = getattr(self, field), getattr(other, field)
            if mine != theirs:
                raise ValueError(
                    f"this state was bound with {field}={mine!r} and the call presents "
                    f"{field}={theirs!r}. A state carries its FORMAT: its leaf order, its "
                    f"page rectangle and its value width are what every kernel derives its "
                    f"addressing from, so a call that disagrees is refused here rather than "
                    f"reading somebody else's bytes. Bind a fresh fsla.dense_state() for this shape.")


def _format_of(routes, d_v: int, BH: int) -> StateFormat:
    """The descriptor a CALL presents: the routes' widths and the value stream's DV."""
    widths = tuple(routes.topology.widths)
    n = 1
    for width in widths:
        n *= width
    return StateFormat(D=len(widths), B=widths, order=CANONICAL_ORDER,
                       page_bits=PAGE_RECTANGLE_BITS, DV=int(d_v), dtype=STATE_DTYPE,
                       ids=BH * (n >> PAGE_RECTANGLE_BITS))


def dense_state() -> DenseState:
    """A fresh UNBOUND dense state. Everything binds at the first call that receives it."""
    return DenseState()


class FSLAState(ABC):
    """THE STATE INTERFACE: one sequence's carried state, mutated in place and handed back.

    What a caller may ask is the public half; the underscored half is the seam the ops take the plane in and hand it
    back through (`fsla/interface.py`, `fsla/engine/decode_step.py`). The first stateful call binds the shape and the
    device from the call itself -- binding is SHAPE ACQUISITION, not a fingerprint, and there is no later
    compatibility interview. An implementation owns its storage and the parameters that storage takes.
    """

    __slots__ = ()

    # -- what a caller may ask -------------------------------------------------

    @property
    @abstractmethod
    def format(self) -> StateFormat | None:
        """This state's FORMAT DESCRIPTOR, or ``None`` while unbound (docs/internals/state.md#format)."""

    @property
    @abstractmethod
    def bound(self) -> bool:
        """Whether a call has bound this state's shape and device yet."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int, int, int] | None:
        """``[B, H, N, cols]``, or ``None`` while unbound."""

    @property
    @abstractmethod
    def device(self) -> torch.device | None:
        """The bound device, or ``None`` while unbound."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype | None:
        """The STORAGE's dtype, or ``None`` while nothing is carried."""

    @property
    @abstractmethod
    def carries(self) -> bool:
        """Whether anything is STORED yet. A bound state can still be empty: binding fixes the shape, contents
        arrive with the first call that finishes."""

    @abstractmethod
    def materialize(self) -> torch.Tensor | None:
        """``[B, H, N, cols]`` fp32, or ``None`` while nothing is carried. ORACLE AND BRIDGE ONLY."""

    @abstractmethod
    def clone(self) -> FSLAState:
        """An EAGER deep copy: an independent sequence, either handle passable into any op with no effect on the other."""

    # -- the seam (fsla/interface.py, fsla/engine/decode_step.py) --------------

    @abstractmethod
    def _kernel_entry(self, routes, *, d_v: int) -> torch.Tensor | None:
        """Bind this state to the call's shape, device and format (or check the call against what it is bound to),
        and return the plane the kernel reads in, or ``None`` for a state that carries nothing yet."""

    @abstractmethod
    def _kernel_commit(self, plane: torch.Tensor) -> None:
        """Take the plane a call that bound a FRESH state stored into; a continuation has nothing to take."""

    @abstractmethod
    def _require_carrying(self) -> None:
        """Refuse a state with nothing to continue from: a decode step CONTINUES a sequence."""

    @abstractmethod
    def _decode_entry(self, routes) -> torch.Tensor:
        """The plane one decode step reads and updates in place, after checking the step's routing against the bound format."""


class DenseState(FSLAState):
    """The dense state: one ``[BH, N/16, page_bytes]`` plane of the stored page format, mutated in place.

    The plane is the whole leaf space, every page resident. A continuation advances it IN PLACE and returns THIS
    object; a fresh plane is ZEROED at bind (docs/internals/state.md#the-dense-continuation). It takes no parameter:
    its storage has nothing to configure.
    """

    __slots__ = ("_shape", "_device", "_plane", "_format", "__weakref__")

    def __init__(self) -> None:
        self._shape: tuple[int, int, int, int] | None = None
        self._device: torch.device | None = None
        self._plane: torch.Tensor | None = None
        self._format: StateFormat | None = None

    def __repr__(self) -> str:
        if not self.bound:
            return "DenseState(unbound)"
        return f"DenseState(shape={self._shape}, device={self._device})"

    @property
    def format(self) -> StateFormat | None:
        return self._format

    @property
    def bound(self) -> bool:
        return self._shape is not None

    @property
    def shape(self) -> tuple[int, int, int, int] | None:
        return self._shape

    @property
    def device(self) -> torch.device | None:
        return self._device

    @property
    def dtype(self) -> torch.dtype | None:
        return None if self._plane is None else self._plane.dtype

    @property
    def carries(self) -> bool:
        return self._plane is not None

    def materialize(self) -> torch.Tensor | None:
        """`[B, H, N, cols]`, or `None` while nothing is carried. ORACLE AND BRIDGE ONLY: it rejoins every page on the
        host, so it is for the fp64 oracle, for a cross-arm hand-off and for a test -- never on a measured path."""
        if self._plane is None:
            return None
        from fsla.ops.paging import decode_pages

        return decode_pages(self._plane, self._format.DV).view(self._shape)

    def clone(self) -> DenseState:
        out = DenseState()
        if not self.bound:
            return out
        out._shape, out._device, out._format = self._shape, self._device, self._format
        if self._plane is not None:
            out._plane = self._plane.clone()
        return out

    def _kernel_entry(self, routes, *, d_v: int) -> torch.Tensor | None:
        routes.refuse_grouped("the carried state")
        presented = _format_of(routes, d_v, routes.batch * routes.heads)
        if not self.bound:
            self._shape = (routes.batch, routes.heads, routes.topology.N, d_v + 1)
            self._device = routes.device
            self._format = presented
        else:
            self._format.check(presented)
        return self._plane

    def _kernel_commit(self, plane: torch.Tensor) -> None:
        """A continuation launched with `s_out` aliased onto its own plane, so its update landed in place; the one
        call that moves a reference here is the one that BOUND the state (docs/internals/state.md#the-dense-continuation)."""
        if self._plane is None:
            self._plane = plane

    def _require_carrying(self) -> None:
        if not self.carries:
            raise RuntimeError(
                "a decode step continues a carried state, and this one carries nothing: "
                "at T = 1 there is no prefill to run and no entry state to read. Run the "
                "sequence's first tokens through fsla.prefill and step from the state it "
                "returns.")

    def _decode_entry(self, routes) -> torch.Tensor:
        """The plane itself, not a copy: which is what makes the kernel's in-place update the state's update. The step's
        routing is CHECKED against the bound format first: a decode step derives its lattice and its unit from the
        descriptor, so a routing that disagrees is refused here rather than walked (docs/internals/state.md#format)."""
        self._require_carrying()
        self._format.check(_format_of(routes, self._format.DV,
                                      self._shape[0] * self._shape[1]))
        return self._plane
