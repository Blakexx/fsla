# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The activation registry: one table, and every rule reads it.

A routing activation used to be a class whose
properties were prose in a docstring and a hand-written `can_zero`. Now it is a TAG
with a ROW, and the row is the only place the properties live.

**The shape, and why it is this shape.**

* *Users* write plain tagged constructors -- `softmax()`, `entmax(1.5)` -- and never
  see or supply property metadata. Property metadata supplied by a caller is
  metadata that can be supplied WRONG, and a validator reading it would then be
  validating a claim rather than a fact.
* *The package* keeps :data:`ACTIVATION_REGISTRY`, `tag -> ActivationProperties`,
  as the single source of truth. It is a plain serializable table, centrally
  auditable, and it is deliberately the same closed-world pattern as the arch table
  and the spill manifest: a thing either has a ratified row or it does not run.
* *Every validator reads COLUMNS, never tags.* A rule says "independent routing refuses two
  `exact_zeros` sides", not "independent routing refuses two entmax sides". **No case list over activations
  may exist anywhere outside this module** -- that is the invariant that makes adding a
  variant a one-row change instead of a hunt.
* *An unknown tag is a hard panic at config construction*, not a default and not a
  warning. A closed-world lookup is what makes "every rule applies to every variant"
  true by construction rather than by review.

**The columns.**

| column | question it answers | who reads it |
|---|---|---|
| `exact_zeros` | does it produce exact structural zeros, not merely small values? | `fsla.routing.types.activation_walks` (which levels a side walks by lists, the producer's walk rule carried by the call's descriptor; and which prefill level-sides get a mask), the solve's support words (`fsla.routing.projection.RouterLayout`), the declaration's refusal (`fsla.routing.producer.Routing`) of an untied read and its write both making zeros on one level |
| `nonnegative` | is every output entry `>= 0`? | `fsla.routing.types.Topology`, which refuses a level-side whose activation is not: v1's routing is nonnegative and its readout normalized (docs/api.md section 2.3) |

A column is kept only while something reads it. Whether a map's output sums to 1 is not
a column: no level is required to (docs/CONSTITUTION.md#r23), so nothing asks.

The op is agnostic to activations: it eats tensors and masks and cannot tell what produced them, so no
activation reaches it and none is refused by it. What a producer supports is the producer's own registry
(`fsla.ops.produce`, `fsla.routing.producer`), and a layer's configuration selects one. **Extending the registry** is
one row plus one constructor, and it is a MINOR version bump.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

__all__ = [
    "ACTIVATION_REGISTRY",
    "ActivationProperties",
    "ActivationTag",
    "UnknownActivation",
    "properties_for",
    "ratified_entmax_alphas",
]


class UnknownActivation(ValueError):
    """Raised at config construction for a tag with no registry row."""


@dataclass(frozen=True, slots=True)
class ActivationProperties:
    """One registry row. Booleans; no behaviour, so it stays serializable."""

    exact_zeros: bool
    nonnegative: bool


#: `(family, parameter)`. The parameter is `None` for a family with no parameter,
#: so that every tag has the same arity and the table can be keyed uniformly.
ActivationTag = tuple[str, float | None]


#: THE TABLE. v0.1 has exactly two families and three tags, plus the opaque row. Every row is
#: nonnegative: a signed activation arrives with the unnormalized readout, together.
#:
#: * `softmax` -- dense: never exactly zero.
#: * `entmax` at `alpha = 1.5` and `alpha = 2` (the latter is sparsemax). Both produce
#:   EXACT structural zeros -- that exactness is the whole reason the production solve is
#:   an exact entmax and not an epsilon-floored approximation (sparsity comes from entmax,
#:   not from a floor).
ACTIVATION_REGISTRY: dict[ActivationTag, ActivationProperties] = {
    ("softmax", None): ActivationProperties(exact_zeros=False, nonnegative=True),
    ("entmax", 1.5): ActivationProperties(exact_zeros=True, nonnegative=True),
    ("entmax", 2.0): ActivationProperties(exact_zeros=True, nonnegative=True),
    ("opaque", None): ActivationProperties(exact_zeros=True, nonnegative=True),
}

#: THE ROW FOR A CELL WE DID NOT SOLVE. A caller with its own levels builds the bundle
#: with :meth:`~fsla.routing.factors.RouteFactors.from_levels`, so a level may be filled
#: by a map this table has never heard of. Such a cell is `opaque`, and `exact_zeros` is TRUE,
#: the CONSERVATIVE reading: it means "may produce zeros", so nothing may assume this cell's
#: support is the full rectangle. `nonnegative` is TRUE because the rows' form holds nothing else:
#: the factory refuses a negative entry (`fsla.routing.quant.ROWS`).

#: The column names, derived from the dataclass.
ACTIVATION_COLUMNS: tuple[str, ...] = tuple(ActivationProperties.__annotations__)


def properties_for(tag: ActivationTag, *, owner: str) -> ActivationProperties:
    """The row for `tag`, or a hard panic naming what would have to be added.

    `owner` is the config object being constructed, so that the panic says which
    configuration is unbuildable rather than merely that a lookup missed.
    """
    try:
        return ACTIVATION_REGISTRY[tag]
    except (KeyError, TypeError):
        raise UnknownActivation(
            f"{owner}: no ratified activation {tag!r}. The registry is CLOSED-WORLD "
            f"(fsla/routing/activations.py): known tags are "
            f"{sorted(ACTIVATION_REGISTRY, key=repr)}. A new variant needs a registry "
            "row -- at which point it participates in every validation rule "
            "immediately -- and adding one is a MINOR version bump. It does not get "
            "to run without one.") from None


def ratified_entmax_alphas() -> tuple[float, ...]:
    """The entmax `alpha` values with a row, READ OFF the registry.

    Exists so that `EntmaxActivation`'s validation is a registry lookup rather than a
    second list of alphas that has to be kept in step with the first one.
    """
    return tuple(sorted(
        parameter for family, parameter in ACTIVATION_REGISTRY
        if family == "entmax" and parameter is not None))


def validate_alpha(alpha: Any, *, owner: str) -> float:
    """A real scalar with an `entmax` row, or a panic. The ONLY alpha check."""
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError(f"{owner}.alpha must be a real scalar, got {alpha!r}")
    alpha = float(alpha)
    properties_for(("entmax", alpha), owner=owner)
    return alpha
