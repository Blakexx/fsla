"""Canonical mixed-radix routing topology."""


MAX_ROUTING_DEPTH = 4
MAX_BRANCH_WIDTH = 256


def validate_routing_shape(branches, *, owner="routing") -> tuple[int, ...]:
    """Validate the production routing-tree depth and per-level width limits."""
    branches = tuple(branches)
    if not branches:
        raise ValueError(f"{owner} must contain at least one routing level")
    if len(branches) > MAX_ROUTING_DEPTH:
        raise ValueError(
            f"{owner} routing depth D must be <= {MAX_ROUTING_DEPTH}; got D={len(branches)}")
    for level, width in enumerate(branches):
        if type(width) is not int:
            raise TypeError(f"{owner} branch widths must be exact ints, got {branches!r}")
        if width > MAX_BRANCH_WIDTH:
            raise ValueError(
                f"{owner} branch width must be <= {MAX_BRANCH_WIDTH}; "
                f"got level {level} width {width}")
    return branches


def resolve_b_spec(b, levels) -> tuple[int, ...]:
    """Normalize an integer or mixed-radix branch specification."""
    if type(levels) is not int:
        raise TypeError(f"levels must be an exact int, got {levels!r}")
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")
    if levels > MAX_ROUTING_DEPTH:
        raise ValueError(
            f"routing depth D must be <= {MAX_ROUTING_DEPTH}; got D={levels}")
    if type(b) is int:
        branches = [b] * levels
    else:
        try:
            branches = list(b)
        except TypeError as error:
            raise TypeError(
                "states_per_level must be an exact int or an iterable of exact ints"
            ) from error
        if any(type(width) is not int for width in branches):
            raise TypeError(f"states_per_level entries must be exact ints, got {branches!r}")
    if len(branches) != levels:
        raise ValueError(f"states_per_level must have exactly D={levels} entries, got {len(branches)}")
    if any(width < 2 for width in branches):
        raise ValueError(f"states_per_level entries must be >= 2, got {branches}")
    validate_routing_shape(branches, owner="states_per_level")
    return tuple(branches)


def canonical_radix_strides(branches) -> tuple[int, ...]:
    """Return MSB-first state-id to digit divisors without materializing digits."""
    branches = tuple(branches)
    if not branches or any(type(width) is not int or width < 2 for width in branches):
        raise ValueError("branches must be a nonempty sequence of exact ints >= 2")
    validate_routing_shape(branches)
    strides = []
    suffix = 1
    for width in reversed(branches):
        strides.append(suffix)
        suffix *= width
    return tuple(reversed(strides))
