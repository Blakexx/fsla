"""Canonical, ops-internal routing plan types.

These types deliberately describe routing execution rather than the layer-facing
configuration vocabulary.  In particular, union routing is a fixed operation,
not a parameterized activation-family choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, TypeAlias

import torch

from rola.routing.activations import (
    ActivationProperties,
    ActivationTag,
    properties_for,
    validate_alpha,
)
from rola.routing.topology import (
    validate_routing_shape,
)

ROUTE_DESCRIPTOR_VERSION = 2
ROUTE_TILE_SIZE = 32
READ_SUPPORT = 1
WRITE_SUPPORT = 2


def level_logit_slice(branches: tuple[int, ...], index: int) -> slice:
    """The column span level ``index`` occupies in a packed per-side logit tensor.

    THE definition of the packed router-logit layout: one side's levels laid end to
    end, level ``l`` owning ``branches[l]`` columns, with no padding to
    ``max(branches)``. Free-standing (rather than only a ``ResolvedRouting`` method)
    because ``diagnostics/telemetry.py`` is handed the branch widths alone and must
    read the same layout -- a second offset expression there would be a second
    definition of the layout, and two definitions drift.
    """
    start = sum(branches[:index])
    return slice(start, start + branches[index])


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


def _validated_alpha(alpha: float, owner: str) -> float:
    """Kept as the module's spelling of the check; the check itself is the registry."""
    return validate_alpha(alpha, owner=owner)


@dataclass(frozen=True, slots=True)
class SoftmaxActivation:
    """The dense routing activation. A plain tagged constructor, per §4.6.

    Carries NO property metadata -- properties come from
    `rola.routing.activations.ACTIVATION_REGISTRY`, keyed on :attr:`tag`. Metadata a
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


RoutingActivation: TypeAlias = SoftmaxActivation | EntmaxActivation

# ROLA_V3_SPEC.md §1 names this union ``Activation``. Same type, spec-facing alias.
Activation: TypeAlias = RoutingActivation


@dataclass(frozen=True, slots=True)
class RouteDescriptor:
    """Versioned compact routing ABI shared by every routing consumer.

    Values are physically tile-major: ``[BH, ceil(T/32), sum_width, 32]``.
    Support is packed once per zero-capable level as signed ``int32`` storage
    carrying uint32 bit patterns in
    ``[BH, ceil(T/32), sum_sparse_width]``.  There is deliberately no token-major
    view, padded branch width, digits table, or per-level support tensor in this
    contract.  A cleared support bit guarantees that the corresponding stored
    side factor is zero; a set bit may conservatively accompany a stored zero.
    """

    read_values: torch.Tensor
    write_values: torch.Tensor
    support_words: torch.Tensor
    level_offsets: tuple[int, ...]
    support_offsets: tuple[int, ...]
    level_widths: tuple[int, ...]
    radix_strides: tuple[int, ...]
    support_side_flags: tuple[int, ...]
    token_origin: int
    token_count: int
    topology_stamp: tuple[int, ...]

    def __post_init__(self) -> None:
        for name, tensor in (
            ("read_values", self.read_values),
            ("write_values", self.write_values),
            ("support_words", self.support_words),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        if self.read_values.ndim != 4 or self.write_values.ndim != 4:
            raise ValueError("route values must have rank 4")
        if self.support_words.ndim != 3:
            raise ValueError("support_words must have rank 3")
        offsets = tuple(self.level_offsets)
        support_offsets = tuple(self.support_offsets)
        widths = tuple(self.level_widths)
        strides = tuple(self.radix_strides)
        flags = tuple(self.support_side_flags)
        stamp = tuple(self.topology_stamp)
        d = len(widths)
        if d == 0 or len(offsets) != d + 1:
            raise ValueError("level_offsets must contain D + 1 entries for a nonempty routing plan")
        validate_routing_shape(widths, owner="RouteDescriptor")
        if any(type(value) is not int for value in (*offsets, *support_offsets, *widths, *strides, *flags, *stamp)):
            raise TypeError("RouteDescriptor metadata must contain exact ints")
        if len(support_offsets) != d or len(strides) != d or len(flags) != d:
            raise ValueError("RouteDescriptor metadata must contain one entry per routing level")
        if offsets[0] != 0 or any(end - start != width or width < 2 for start, end, width in zip(offsets, offsets[1:], widths)):
            raise ValueError("level_offsets must be compact slices matching level_widths >= 2")
        expected_support_offset = 0
        for offset, width, flag in zip(support_offsets, widths, flags):
            if flag not in (0, READ_SUPPORT, WRITE_SUPPORT, READ_SUPPORT | WRITE_SUPPORT):
                raise ValueError("support_side_flags entries must be READ=1 and/or WRITE=2")
            if flag == 0:
                if offset != -1:
                    raise ValueError("dense routing levels must use support offset -1")
            else:
                if offset != expected_support_offset:
                    raise ValueError("support offsets must compactly represent each sparse level once")
                expected_support_offset += width
        expected_strides = tuple(prod(widths[index + 1:]) for index in range(d))
        if strides != expected_strides:
            raise ValueError("radix_strides must be canonical MSB-first mixed-radix strides")
        if type(self.token_origin) is not int or self.token_origin < 0:
            raise ValueError("token_origin must be a nonnegative exact int")
        if type(self.token_count) is not int or self.token_count < 0:
            raise ValueError("token_count must be a nonnegative exact int")
        route_tiles = (self.token_count + ROUTE_TILE_SIZE - 1) // ROUTE_TILE_SIZE
        expected_value_shape = (self.read_values.shape[0], route_tiles, offsets[-1], ROUTE_TILE_SIZE)
        if (
            self.read_values.dtype != torch.bfloat16
            or self.write_values.dtype != torch.bfloat16
            or self.read_values.shape != expected_value_shape
            or self.write_values.shape != expected_value_shape
            or not self.read_values.is_contiguous()
            or not self.write_values.is_contiguous()
        ):
            raise ValueError("route values must be contiguous bf16 [BH, route_tiles, sum_width, 32] arenas")
        if self.read_values.device != self.write_values.device:
            raise ValueError("read and write route arenas must share a device")
        expected_support_shape = (self.read_values.shape[0], route_tiles, expected_support_offset)
        if (
            self.support_words.dtype != torch.int32
            or self.support_words.shape != expected_support_shape
            or not self.support_words.is_contiguous()
            or self.support_words.device != self.read_values.device
        ):
            raise ValueError(
                "support_words must be contiguous int32 uint32-bit patterns "
                "[BH, token_words, sum_sparse_width]")
        expected_stamp = (ROUTE_DESCRIPTOR_VERSION, d, *widths, *strides, *flags)
        if stamp != expected_stamp:
            raise ValueError("topology_stamp does not match the RouteDescriptor ABI metadata")
        object.__setattr__(self, "level_offsets", offsets)
        object.__setattr__(self, "support_offsets", support_offsets)
        object.__setattr__(self, "level_widths", widths)
        object.__setattr__(self, "radix_strides", strides)
        object.__setattr__(self, "support_side_flags", flags)
        object.__setattr__(self, "topology_stamp", stamp)


@dataclass(frozen=True, slots=True)
class IndependentRouting:
    """Per-side routing with independently specified value activations."""

    tied: bool
    read_activation: RoutingActivation
    write_activation: RoutingActivation

    def __post_init__(self) -> None:
        if type(self.tied) is not bool:
            raise TypeError(f"IndependentRouting.tied must be bool, got {self.tied!r}")
        for name, activation in (
            ("read_activation", self.read_activation),
            ("write_activation", self.write_activation),
        ):
            # Admission is a REGISTRY lookup, not an isinstance list. An isinstance
            # list would be a second closed world, and a variant with a ratified row
            # would still be unconfigurable until someone remembered to widen it --
            # which is precisely the "one row and it participates in everything"
            # property §4.6 is built to give.
            activation_properties(activation, owner=f"IndependentRouting.{name}")
        if (
            activation_properties(self.read_activation).exact_zeros
            and activation_properties(self.write_activation).exact_zeros
            and not self.shares_solve
        ):
            raise ValueError(
                "IndependentRouting only permits two exact-zero (`exact_zeros`) sides when "
                "they share one tied solve; use UnionRouting for untied sparse/sparse "
                "routing. Stated against the registry column rather than against a list "
                "of activation names, so a new zero-capable variant is covered the "
                "moment its row exists")

    def activation(self, duty: str):
        """This level's activation for one DUTY, `'read'` or `'write'` (§4.6).

        Per-level AND per-duty configuration is what makes level-split
        (`docs/open-work.md` M-3) pure configuration rather than a code path, so the
        duty axis gets a named accessor instead of two attribute reads that a caller
        has to remember the spelling of.
        """
        if duty == "read":
            return self.read_activation
        if duty == "write":
            return self.write_activation
        raise ValueError(f"duty must be 'read' or 'write', got {duty!r}")

    @property
    def shares_solve(self) -> bool:
        """Whether tied logits and equal activations make one solve exact."""
        return self.tied and self.read_activation == self.write_activation


@dataclass(frozen=True, slots=True)
class UnionRouting:
    """Exact entmax shared support followed by read/write role splitting.

    The INTERSECTION GUARANTEE -- that one solve determines a support both duties
    then split across -- requires the activation to produce EXACT structural zeros.
    That precondition is checked against the registry's `exact_zeros` column, not
    against a list of activation names, so a future zero-capable family is admitted
    by its row and a future non-zero-capable one is refused by its row.
    """

    alpha: float = 1.5

    def __post_init__(self) -> None:
        alpha = validate_alpha(self.alpha, owner="UnionRouting")
        object.__setattr__(self, "alpha", alpha)
        if not properties_for(("entmax", alpha), owner="UnionRouting").exact_zeros:
            raise ValueError(
                f"UnionRouting(alpha={alpha}) names an activation whose registry row says "
                "`exact_zeros=False`. Union routing IS the shared-support construction: "
                "without exact structural zeros there is no support to intersect, only "
                "small numbers, and the read/write split would be silently meaningless")

    @property
    def properties(self) -> ActivationProperties:
        return properties_for(("entmax", self.alpha), owner="UnionRouting")

    def activation(self, duty: str) -> EntmaxActivation:
        """Union routing runs ONE solve, so both duties name the same activation."""
        if duty not in ("read", "write"):
            raise ValueError(f"duty must be 'read' or 'write', got {duty!r}")
        return EntmaxActivation(self.alpha)

LevelRouting: TypeAlias = IndependentRouting | UnionRouting


@dataclass(frozen=True, slots=True)
class ResolvedRouting:
    """Validated routing topology and packed projection layout for one RoLA plan."""

    levels: tuple[LevelRouting, ...]
    branches: tuple[int, ...]

    def __post_init__(self) -> None:
        levels = tuple(self.levels)
        branches = tuple(self.branches)
        if not levels:
            raise ValueError("ResolvedRouting.levels must be nonempty")
        if len(branches) != len(levels):
            raise ValueError(
                f"ResolvedRouting.branches must have D={len(levels)} entries, got {len(branches)}")
        validate_routing_shape(branches, owner="ResolvedRouting")
        for level in levels:
            if type(level) not in (IndependentRouting, UnionRouting):
                raise TypeError(f"ResolvedRouting.levels contains an invalid level: {level!r}")
        if any(type(branch) is not int for branch in branches):
            raise TypeError(f"ResolvedRouting.branches must contain exact ints, got {branches!r}")
        if any(branch < 2 for branch in branches):
            raise ValueError(f"ResolvedRouting.branches entries must be >= 2, got {branches!r}")
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "branches", branches)

    @property
    def D(self) -> int:
        return len(self.levels)

    @property
    def real_width(self) -> int:
        return max(self.branches)

    @property
    def state_count(self) -> int:
        return prod(self.branches)

    @property
    def untied_levels(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, level in enumerate(self.levels)
            if isinstance(level, UnionRouting) or not level.tied
        )

    @property
    def packed_slot_count(self) -> int:
        return self.D + len(self.untied_levels)

    @property
    def packed_read_slot_lookup(self) -> tuple[int, ...]:
        """Packed source slot for each level's read logits."""
        read_slots = {level: self.D + slot for slot, level in enumerate(self.untied_levels)}
        return tuple(read_slots.get(level, level) for level in range(self.D))

    @property
    def shared_solve_levels(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, level in enumerate(self.levels)
            if isinstance(level, IndependentRouting) and level.shares_solve
        )

    @property
    def read_level_can_zero(self) -> tuple[bool, ...]:
        """Exact-zero capability for every level on the read side."""
        return tuple(
            True if isinstance(level, UnionRouting)
            else activation_properties(level.read_activation).exact_zeros
            for level in self.levels
        )

    @property
    def write_level_can_zero(self) -> tuple[bool, ...]:
        """Exact-zero capability for every level on the write side."""
        return tuple(
            True if isinstance(level, UnionRouting)
            else activation_properties(level.write_activation).exact_zeros
            for level in self.levels
        )

    @property
    def level_offsets(self) -> tuple[int, ...]:
        offsets = [0]
        for width in self.branches:
            offsets.append(offsets[-1] + width)
        return tuple(offsets)

    @property
    def packed_logit_width(self) -> int:
        """Columns in ONE side's router-logit tensor -- tightly packed, no per-level padding.

        The router logits are packed on the same offsets the route arena already uses
        (``level_offsets``), NOT padded to ``real_width`` per level. Padding every level
        to the widest one made ``packed_slot_count * real_width`` columns where only
        ``2 * sum(branches)`` are ever read: at the parity cell (branches (256, 64),
        untied) that was 1024 computed for 640 consumed -- 37.5% of the projection GEMM
        and of the logit tensor it writes, thrown away. The waste GROWS with branch-width
        imbalance, so it is a scaling defect, not a constant.

        ``route_W`` keeps its ``[H, packed_slots, hidden, real_width]`` parameter layout:
        the packing is done on the WEIGHT, per side, before the GEMM (see
        ``routing/projection.py``), so no checkpoint and no parameter shape moves.
        """
        return self.level_offsets[-1]

    def level_logit_slice(self, index: int) -> slice:
        """The column span level ``index`` occupies in a packed per-side logit tensor.

        Delegates to the module-level :func:`level_logit_slice` so the production
        solve, the fp64 reference, telemetry and the budget-conditioning actuators
        all read ONE definition of the packed layout.
        """
        return level_logit_slice(self.branches, index)

    @property
    def radix_strides(self) -> tuple[int, ...]:
        return tuple(prod(self.branches[index + 1:]) for index in range(self.D))

    @property
    def support_side_flags(self) -> tuple[int, ...]:
        return tuple(
            (READ_SUPPORT if read_can_zero else 0) | (WRITE_SUPPORT if write_can_zero else 0)
            for read_can_zero, write_can_zero in zip(
                self.read_level_can_zero, self.write_level_can_zero)
        )

    @property
    def support_offsets(self) -> tuple[int, ...]:
        offsets: list[int] = []
        next_offset = 0
        for width, flag in zip(self.branches, self.support_side_flags):
            if flag == 0:
                offsets.append(-1)
            else:
                offsets.append(next_offset)
                next_offset += width
        return tuple(offsets)

    @property
    def topology_stamp(self) -> tuple[int, ...]:
        return (
            ROUTE_DESCRIPTOR_VERSION,
            self.D,
            *self.branches,
            *self.radix_strides,
            *self.support_side_flags,
        )

    @property
    def read_can_zero(self) -> bool:
        return any(self.read_level_can_zero)

    @property
    def write_can_zero(self) -> bool:
        return any(self.write_level_can_zero)

    def validate_packed_router(self, route_W: torch.Tensor, route_bias: torch.Tensor | None = None) -> int:
        """Validate packed router tensors and return their derived head count."""
        if not isinstance(route_W, torch.Tensor):
            raise TypeError(f"route_W must be a torch.Tensor, got {type(route_W).__name__}")
        if route_W.ndim != 4:
            raise ValueError(
                "route_W must have shape [H, packed_slots, hidden_size, real_width], "
                f"got {tuple(route_W.shape)}")
        heads, slots, _, width = route_W.shape
        if heads < 1 or slots != self.packed_slot_count or width != self.real_width:
            raise ValueError(
                "route_W must have shape "
                f"[H>=1, {self.packed_slot_count}, hidden_size, {self.real_width}], "
                f"got {tuple(route_W.shape)}")
        if route_bias is not None:
            if not isinstance(route_bias, torch.Tensor):
                raise TypeError(f"route_bias must be a torch.Tensor or None, got {type(route_bias).__name__}")
            expected_bias_shape = (heads, self.packed_slot_count, self.real_width)
            if route_bias.shape != expected_bias_shape:
                raise ValueError(
                    f"route_bias must have shape {expected_bias_shape}, got {tuple(route_bias.shape)}")
        return heads


# ---------------------------------------------------------------------------
# ROLA_V3_SPEC.md §1 config contract additions.
#
# ``Activation`` (above), ``LevelRouting`` (below, already ``Independent |
# Union``), ``IndependentRouting``, and ``UnionRouting`` are reused verbatim --
# they already satisfy the V3 §1 shapes. Only the topology/decay/normalization
# wrappers are new. These are producer-agnostic: they describe the routing
# contract the V3 naive oracle (``fla_rola/ops/rola/naive.py``) consumes, not a
# projection ABI. They deliberately do not reuse ``ResolvedRouting`` /
# ``RouteDescriptor``, which are packed-router / kernel ABI types tied to the
# Q/K-era producer and out of scope for V3 stage 1.
# ---------------------------------------------------------------------------

LevelRouting: TypeAlias = IndependentRouting | UnionRouting


@dataclass(frozen=True, slots=True)
class LevelConfig:
    """One routing level: its branch width and routing tag (``ROLA_V3_SPEC.md`` §1)."""

    width: int
    routing: LevelRouting

    def __post_init__(self) -> None:
        if type(self.width) is not int:
            raise TypeError(f"LevelConfig.width must be an exact int, got {self.width!r}")
        if self.width < 1:
            # Spec §1 states only the upper bound (``width <= 256``); a width < 1
            # is not a modeling choice, it makes N/radix_strides ill-defined
            # (division by zero), so it is rejected as a structural invariant.
            raise ValueError(f"LevelConfig.width must be >= 1, got {self.width}")
        if type(self.routing) not in (IndependentRouting, UnionRouting):
            raise TypeError(
                f"LevelConfig.routing must be IndependentRouting or UnionRouting, got {self.routing!r}")


@dataclass(frozen=True, slots=True)
class Topology:
    """Complete routing topology plus value width (``ROLA_V3_SPEC.md`` §1).

    ``D = len(levels)`` is validated in ``[1, 4]`` and every level's width in
    ``[1, 256]`` by ``validate_routing_shape`` -- the same validator ``V2``'s
    ``ResolvedRouting`` uses, so the two structural invariants are not
    reimplemented here.
    """

    levels: tuple[LevelConfig, ...]
    d_v: int

    def __post_init__(self) -> None:
        levels = tuple(self.levels)
        for index, level in enumerate(levels):
            if type(level) is not LevelConfig:
                raise TypeError(f"Topology.levels[{index}] must be a LevelConfig, got {level!r}")
        validate_routing_shape(tuple(level.width for level in levels), owner="Topology")
        if type(self.d_v) is not int or self.d_v < 1:
            raise ValueError(f"Topology.d_v must be a positive exact int, got {self.d_v!r}")
        object.__setattr__(self, "levels", levels)

    @property
    def D(self) -> int:
        return len(self.levels)

    @property
    def widths(self) -> tuple[int, ...]:
        return tuple(level.width for level in self.levels)

    @property
    def N(self) -> int:
        """Leaf capacity: ``N = prod_l width_l`` (spec §1, derived, never configured)."""
        return prod(self.widths)

    @property
    def radix_strides(self) -> tuple[int, ...]:
        """MSB-first mixed-radix strides: ``stride_l = prod(width_{l+1..D-1})``."""
        widths = self.widths
        return tuple(prod(widths[index + 1:]) for index in range(len(widths)))

    # --- §4.6: the per-level per-duty activation surface, read generically ----

    def activation(self, level: int, duty: str):
        """The activation configured for one `(level, duty)` cell."""
        return self.levels[level].routing.activation(duty)

    def activation_properties(self, level: int, duty: str) -> ActivationProperties:
        """That cell's registry row -- the ONLY source of activation properties."""
        return activation_properties(
            self.activation(level, duty), owner=f"Topology.levels[{level}].{duty}")

    def duty_cells(self) -> tuple[tuple[int, str], ...]:
        """Every `(level, duty)` cell, so a rule can quantify over the whole grid
        instead of over a hand-written pair of loops per rule."""
        return tuple((level, duty) for level in range(self.D) for duty in ("read", "write"))

    @property
    def unratified_activations(self) -> tuple[tuple[int, str, ActivationTag], ...]:
        """The cells whose registry row says `kerneled=False`.

        §4.6: a config naming one of these is a LABELED RESEARCH SURFACE -- it runs
        on the reference path carrying an explicit `not_kerneled` flag. It is not a
        kernel fallback, because the kernel is never asked and never declines;
        `launch_consumer` remains the single authority on supported shapes.
        """
        return tuple(
            (level, duty, self.activation(level, duty).tag)
            for level, duty in self.duty_cells()
            if not self.activation_properties(level, duty).kerneled
        )

    @property
    def kerneled(self) -> bool:
        """Whether every configured activation has a ratified kernel row."""
        return not self.unratified_activations


@dataclass(frozen=True, slots=True)
class LeafMassDecay:
    """Complete-leaf write-conditioned decay dials (``ROLA_V3_SPEC.md`` §1, §2).

    ``dials[l]`` has shape ``[H, width_l]`` with entries ``delta_l[d] in [0, 1)``.
    The upper bound is strict and load-bearing: it keeps
    ``rate_leaf[s] = prod_l delta_l[d_l(s)] < 1`` for every leaf ``s``, so
    ``keep = (1 - rate_leaf) ** c = 0`` is unreachable by construction (spec §1).

    **This check is necessary but NOT sufficient (review B1).** It runs in
    ``dial``'s own dtype, which may be fp64; the kernel only ever reads an fp32
    cast of these dials, and an fp64 value within ~3e-8 of 1 is strictly ``< 1``
    here yet rounds to exactly ``1.0f`` on cast, which NaNs the kernel
    (``log1pf(-1.0f) = -inf``, then ``expf(0 * -inf)``). The invariant is
    re-checked in fp32 at the cast boundary in
    ``routing/schedule_v3.py::leaf_rate_dials`` -- that second check, not this
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
                    "unreachable -- see ROLA_V3_SPEC.md §1)")
        object.__setattr__(self, "dials", dials)

    def validate_against(self, topology: Topology) -> None:
        """Check level count, per-level width, and the WRITE-side mass precondition.

        §4.6: **mass decay requires a `normalized` write side.** The clock is
        `c[t,s] = prod_l p_write_stored[t, d_l(s)]` and the survival factor is
        `(1 - rate_leaf)**c` -- an expression that prices a decay per unit of MASS
        DEPOSITED. If the write side does not land on the simplex, `c` is no longer
        a unit-mass share and the same routing decays by an amount that depends on
        the un-normalized scale of the logits, which is not a modeling choice anyone
        made. Stated against the registry column, so it covers every present and
        future activation without naming one.
        """
        widths = topology.widths
        unnormalized = [
            (level, topology.activation(level, "write").tag)
            for level in range(topology.D)
            if not topology.activation_properties(level, "write").normalized
        ]
        if unnormalized:
            raise ValueError(
                f"LeafMassDecay requires a `normalized` write side on every level; "
                f"levels {unnormalized} name activations whose registry row says "
                "`normalized=False`. The decay clock prices survival per unit of "
                "DEPOSITED MASS, so an un-normalized write side makes the amount "
                "decayed depend on the scale of the logits rather than on the "
                "routing. Configure a normalized write activation, or do not "
                "configure mass decay")
        if len(self.dials) != len(widths):
            raise ValueError(
                f"LeafMassDecay.dials must contain one tensor per level (D={len(widths)}), "
                f"got {len(self.dials)}")
        for level, (dial, width) in enumerate(zip(self.dials, widths)):
            if dial.shape[-1] != width:
                raise ValueError(
                    f"LeafMassDecay.dials[{level}] must have width {width} matching "
                    f"Topology.levels[{level}].width, got {dial.shape[-1]}")


# ``kind: None | LeafMassDecay`` (spec §1). ``FactorizedMassDecay`` is removed in
# V3 (spec §1 rationale) and must not be reintroduced here.
DecayConfig: TypeAlias = None | LeafMassDecay

NORMALIZATION_KINDS = ("raw", "global")
Normalization: TypeAlias = str


def validate_normalization(value: str) -> str:
    """Validate the V3 ``Normalization`` contract: ``raw | global`` (spec §1)."""
    if value not in NORMALIZATION_KINDS:
        raise ValueError(f"Normalization must be one of {NORMALIZATION_KINDS}, got {value!r}")
    return value
