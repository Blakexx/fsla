"""The routing's contract types: the activations, the tied marker, the topology and the decay dials.

A routing is declared ONCE as a :class:`Topology`: the level widths, one flat activation a write level, and the read
heads' activations -- one tuple a read head of the group, or one tuple for every head -- each entry an activation or the
:func:`tied` marker. The layer's declaration (:class:`fsla.routing.producer.Routing`) builds one, and a bundle
(:class:`fsla.routing.factors.RouteFactors`) carries the one its rows are shaped by.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from math import prod
from typing import Any, TypeAlias

import torch

from fsla.routing.activations import (
    ActivationProperties,
    ActivationTag,
    properties_for,
    validate_alpha,
)
from fsla.routing.topology import (
    padded_level_width,
    validate_level_width,
    validate_routing_shape,
)


def activation_properties(activation: Any, *, owner: str = "activation") -> ActivationProperties:
    """The registry row for any activation object, by its `tag`.

    The ONE way any rule in this module reads an activation's properties. Nothing
    reads a property off the activation object itself, because then an object could
    disagree with the table and the table would stop being the source of truth.
    """
    tag = getattr(activation, "tag", None)
    if tag is None:
        raise TypeError(
            f"{owner} must be a routing activation carrying a registry `tag`, got {activation!r}")
    return properties_for(tag, owner=owner)


@dataclass(frozen=True, slots=True)
class SoftmaxActivation:
    """The dense routing activation. A plain tagged constructor.

    Carries NO property metadata -- properties come from
    `fsla.routing.activations.ACTIVATION_REGISTRY`, keyed on :attr:`tag`. Metadata a
    caller could supply is metadata a caller could supply wrong, and a validator
    reading it would be checking a claim instead of a fact.
    """

    @property
    def tag(self) -> ActivationTag:
        return ("softmax", None)

    @property
    def properties(self) -> ActivationProperties:
        return properties_for(self.tag, owner="SoftmaxActivation")

    @property
    def can_zero(self) -> bool:
        """The `exact_zeros` column, under the name the rest of the tree uses."""
        return self.properties.exact_zeros


@dataclass(frozen=True, slots=True)
class EntmaxActivation:
    """An exact entmax routing activation, at a registry-ratified `alpha`.

    `alpha` is validated by REGISTRY LOOKUP (`("entmax", alpha)`), so the set of
    admissible alphas is the set of rows and there is no second list to keep in step.
    """

    alpha: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", validate_alpha(self.alpha, owner="EntmaxActivation"))

    @property
    def tag(self) -> ActivationTag:
        return ("entmax", self.alpha)

    @property
    def properties(self) -> ActivationProperties:
        return properties_for(self.tag, owner="EntmaxActivation")

    @property
    def can_zero(self) -> bool:
        return self.properties.exact_zeros


def entmax(alpha: float = 1.5) -> EntmaxActivation:
    """User-facing constructor. `entmax(2)` is sparsemax."""
    return EntmaxActivation(alpha)


def softmax() -> SoftmaxActivation:
    """User-facing constructor for the dense activation."""
    return SoftmaxActivation()


#: The activations the shipped producer solves (`fsla.routing.entmax.production`).
RoutingActivation: TypeAlias = SoftmaxActivation | EntmaxActivation

# ``Activation`` is the spec-facing alias for this union. Same type.
Activation: TypeAlias = RoutingActivation


@dataclass(frozen=True, slots=True)
class OpaqueActivation:
    """The activation of a cell this package did not solve.

    A caller of the op with its own levels builds the bundle with
    :meth:`~fsla.routing.factors.RouteFactors.from_levels`, so a level may be filled by a map with no registry row of
    its own. Its cell still has to answer every question the rules ask -- density and ratification -- and it answers
    them from ONE row, ``("opaque", None)``, whose columns are argued in ``fsla/routing/activations.py``.
    """

    @property
    def tag(self) -> ActivationTag:
        return ("opaque", None)

    @property
    def properties(self) -> ActivationProperties:
        return properties_for(self.tag, owner="OpaqueActivation")

    @property
    def can_zero(self) -> bool:
        return self.properties.exact_zeros


@dataclass(frozen=True, slots=True)
class Tied:
    """THE TIED MARKER, a read entry's position-held statement that this read head's level IS the write side's on that
    level: the same rows, one solve, no read logits of its own. It is a marker and not an activation -- it has no
    registry row; a rule asking about the cell reads the write side's activation on that level."""


def tied() -> Tied:
    """User-facing constructor of the tied marker (:class:`Tied`)."""
    return Tied()


#: One entry of a read head's tuple.
ReadEntry: TypeAlias = SoftmaxActivation | EntmaxActivation | OpaqueActivation | Tied


def _checked_activation(activation, *, owner: str) -> None:
    """A registry row, and the row nonnegative: v1's routing is nonnegative and its readout normalized."""
    if not activation_properties(activation, owner=owner).nonnegative:
        raise ValueError(f"{owner}: {activation!r}'s registry row is not nonnegative, and every routing activation is "
                         "(the readout is normalized; signed weights arrive with the unnormalized readout, together)")


@dataclass(frozen=True, slots=True)
class Topology:
    """THE ROUTING, DECLARED ONCE: the level widths, a flat activation a write level, and the read heads' tuples.

    ``widths``: one width a level, outermost first, ``D = len(widths)`` in ``[1, 4]`` and each in ``[1, 256]``
    (``validate_routing_shape``). ``write``: one activation a level. ``read``: one tuple a read head of the group, or
    ONE tuple that every head reads by -- each tuple one entry a level, an activation or :func:`tied`. Every read head
    has the write side's widths: the read side states none. Every activation is a registry row, and the row
    nonnegative.

    **The value width is not here, and that is the point.** ``d_v`` is the width of the state each leaf holds, a fact
    about the VALUE stream: the layer sizes ``v_proj``/``o_proj`` with it and the op reads it off the values at call
    time.
    """

    widths: tuple[int, ...]
    write: tuple[Any, ...]
    read: tuple[tuple[ReadEntry, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.widths, (tuple, list)):
            raise TypeError(f"Topology.widths is one width a level, a tuple of ints, got {self.widths!r}")
        widths = tuple(self.widths)
        for level, width in enumerate(widths):
            validate_level_width(width, owner=f"Topology.widths[{level}]")
        validate_routing_shape(widths, owner="Topology")
        D = len(widths)
        if not isinstance(self.write, (tuple, list)) or len(self.write) != D:
            raise ValueError(f"Topology.write is one activation a level (D={D}), got {self.write!r}")
        write = tuple(self.write)
        for level, activation in enumerate(write):
            _checked_activation(activation, owner=f"Topology.write[{level}]")
        if not isinstance(self.read, (tuple, list)) or not self.read:
            raise ValueError(f"Topology.read is one tuple a read head (or one tuple for every head), got {self.read!r}")
        read = []
        for head, entries in enumerate(self.read):
            if not isinstance(entries, (tuple, list)) or len(entries) != D:
                raise ValueError(f"Topology.read[{head}] is one entry a level, the topology's depth D={D}; got "
                                 f"{entries!r}")
            for level, entry in enumerate(entries):
                if type(entry) is not Tied:
                    _checked_activation(entry, owner=f"Topology.read[{head}][{level}]")
            read.append(tuple(entries))
        object.__setattr__(self, "widths", widths)
        object.__setattr__(self, "write", write)
        object.__setattr__(self, "read", tuple(read))

    @classmethod
    def opaque(cls, widths) -> Topology:
        """The topology of levels this package did not solve: every cell the ``("opaque", None)`` row."""
        widths = tuple(widths)
        return cls(widths=widths, write=(OpaqueActivation(),) * len(widths), read=((OpaqueActivation(),) * len(widths),))

    @property
    def D(self) -> int:
        return len(self.widths)

    @property
    def padded_widths(self) -> tuple[int, ...]:
        """Per-level ``B_l``: the next power of two at or above 16 that admits ``b_l``.

        The KERNEL-FACING widths (``StateFormat.B``'s law): every ``B_l`` this
        property names is already a legal ``fsla._state.StateFormat`` level width.
        Derived from :attr:`widths` alone -- never a second input a caller supplies,
        so a padded width can never disagree with the logical one it pads.
        """
        return tuple(padded_level_width(width) for width in self.widths)

    @property
    def is_padded(self) -> bool:
        """Whether every level is ALREADY at its padded width -- the dense fast path.

        ``widths == padded_widths`` for an ideal (power-of-two >= 16, every level)
        model: :meth:`padded` still returns an equivalent ``Topology`` in that case,
        so a caller may always call it uniformly (CONSTITUTION §R9: a structure
        branch is a derived fact, never a data-dependent shortcut).
        """
        return self.widths == self.padded_widths

    def padded(self) -> Topology:
        """The SAME routing at its kernel-facing widths: every activation and marker kept, each width padded. A
        convenience for a caller building the padded-shape SIDE of an equivalence directly
        (``tests/oracle/test_padding_seam.py``'s fp64 prefill proof); the op call surfaces that pad a live call derive
        their own padded widths per call (``fsla.ops.padding.PaddedShape``)."""
        return Topology(widths=self.padded_widths, write=self.write, read=self.read)

    @property
    def N(self) -> int:
        """Leaf capacity: ``N = prod_l width_l`` -- derived, never configured."""
        return prod(self.widths)

    @property
    def radix_strides(self) -> tuple[int, ...]:
        """MSB-first mixed-radix strides: ``stride_l = prod(width_{l+1..D-1})``."""
        widths = self.widths
        return tuple(prod(widths[index + 1:]) for index in range(len(widths)))

    def read_heads(self, group: int) -> tuple[tuple[ReadEntry, ...], ...]:
        """The ``group`` read heads' tuples: the declared ones, or the one tuple every head reads by. Refuses a count
        that is neither ``group`` nor one, naming both."""
        if type(group) is not int or group < 1:
            raise ValueError(f"a group is a positive exact int of read heads a write head, got {group!r}")
        if len(self.read) == group:
            return self.read
        if len(self.read) == 1:
            return self.read * group
        raise ValueError(f"the routing declares {len(self.read)} read heads' tuples and the layer reads each write head "
                         f"by G={group} query heads: declare G tuples, one a read head of the group, or one tuple for "
                         "every head")

    def read_activation(self, head: int, level: int):
        """Read head ``head``'s activation on ``level``, a tied entry the write side's (``head`` indexes :attr:`read`)."""
        entry = self.read[head][level]
        return self.write[level] if type(entry) is Tied else entry

    def can_zero(self, side: str, level: int) -> bool:
        """Whether ``side``'s ``level`` may hold exact zeros, from the registry: the write activation's ``exact_zeros``,
        and on the read side any read head's."""
        if side == "write":
            return activation_properties(self.write[level]).exact_zeros
        if side == "read":
            return any(activation_properties(self.read_activation(head, level)).exact_zeros
                       for head in range(len(self.read)))
        raise ValueError(f"side must be 'read' or 'write', got {side!r}")


@dataclass(frozen=True, slots=True)
class LeafMassDecay:
    """Complete-leaf write-conditioned decay dials.

    ``dials[l]`` has shape ``[H, width_l]`` with entries ``delta_l[d] in [0, 1)``.
    The upper bound is strict and load-bearing: it keeps
    ``rate_leaf[s] = prod_l delta_l[d_l(s)] < 1`` for every leaf ``s``, so
    ``keep = (1 - rate_leaf) ** c = 0`` is unreachable by construction.

    **This check is necessary but NOT sufficient.** It runs in
    ``dial``'s own dtype, which may be fp64; the kernel only ever reads an fp32
    cast of these dials, and an fp64 value within ~3e-8 of 1 is strictly ``< 1``
    here yet rounds to exactly ``1.0f`` on cast, which NaNs the kernel
    (``log1pf(-1.0f) = -inf``, then ``expf(0 * -inf)``). The invariant is
    re-checked in fp32 at the cast boundary in
    ``fsla/internal/decay.py::leaf_rate_dials`` -- that second check, not this
    one, is what the kernel's correctness actually depends on.
    """

    dials: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        dials = tuple(self.dials)
        if not dials:
            raise ValueError("LeafMassDecay.dials must be nonempty")
        heads = None
        for level, dial in enumerate(dials):
            if not isinstance(dial, torch.Tensor):
                raise TypeError(
                    f"LeafMassDecay.dials[{level}] must be a torch.Tensor, got {type(dial).__name__}")
            if dial.ndim != 2:
                raise ValueError(
                    f"LeafMassDecay.dials[{level}] must have shape [H, width_l], "
                    f"got {tuple(dial.shape)}")
            if heads is None:
                heads = dial.shape[0]
            elif dial.shape[0] != heads:
                raise ValueError("LeafMassDecay.dials must share one head count H across levels")
            if torch.any(dial < 0) or torch.any(dial >= 1):
                raise ValueError(
                    f"LeafMassDecay.dials[{level}] entries must satisfy the strict invariant "
                    "0 <= delta < 1 (upper bound strict; keeps rate_leaf < 1 so keep=0 is "
                    "unreachable by construction)")
        object.__setattr__(self, "dials", dials)

    def validate_against(self, topology: Topology) -> None:
        """Check level count and per-level width. The clock needs no property of the write side: it is each leaf's
        SHARE of the token's write, computed from the write rows themselves (docs/CONSTITUTION.md#r23), so it prices
        survival per unit of deposited mass whatever magnitude a level carries."""
        widths = topology.widths
        if len(self.dials) != len(widths):
            raise ValueError(
                f"LeafMassDecay.dials must contain one tensor per level (D={len(widths)}), "
                f"got {len(self.dials)}")
        for level, (dial, width) in enumerate(zip(self.dials, widths)):
            if dial.shape[-1] != width:
                raise ValueError(
                    f"LeafMassDecay.dials[{level}] must have width {width} matching "
                    f"Topology.widths[{level}], got {dial.shape[-1]}")


# ``kind: None | LeafMassDecay``. ``FactorizedMassDecay`` was removed and must not be
# reintroduced here.
DecayConfig: TypeAlias = None | LeafMassDecay


@cache
def activation_walks(topology) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    """WHICH LEVEL-SIDES CAN MAKE EXACT ZEROS, ``(read, write)`` a bool a level, from the declaration's activation metadata (an opaque level, which
    may have zeros, included; on the read side, any read head's) and never from the data; asked once a topology. It has two readers. THE PRODUCER'S
    WALK RULE: a side walks such a level BY ITS LISTS (the rest digit by digit), which :func:`fsla.ops.produce.level_facts` states as decode's
    ``write_listed_k`` facts. THE PREFILL'S MASKS: the producer and the masks' helpers (:mod:`fsla.routing.masks`) make a mask exactly for these
    level-sides, and leave the rest absent, dense."""
    return tuple(tuple(topology.can_zero(side, level) for level in range(topology.D)) for side in ("read", "write"))
