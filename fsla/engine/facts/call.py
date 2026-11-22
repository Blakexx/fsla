# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""One call's facts, read off the routing bundle and the value stream, and the
envelope those facts fold into: whether the built arm has an arm for this call.

The rules fold VALUES -- that is what makes them testable with no device present --
so something has to read those values off the objects a caller actually holds. This
is that adapter and nothing else: it finds facts and applies the rules over them, and
it decides nothing of its own (docs/internals/engine/engine.md).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

from fsla.engine import BackwardNotImplemented, rules
from fsla.engine.facts.operands import operands_of, requires_backward
from fsla.ops._ext import KernelNotBuilt

if TYPE_CHECKING:
    from fsla.routing.factors import RouteFactors
    from fsla.values import Values


def widths_of(routes) -> tuple[int, ...]:
    """The per-level widths of this bundle's topology, in level order."""
    return tuple(level.width for level in routes.topology.levels)


def prefill_clauses(routes, decay: Any = None, *, d_v: int, stateful: bool = False) -> tuple:
    """The facts :func:`fsla.engine.rules.envelope.prefill_envelope` folds, for this segment: the call's shape,
    whether the intra reads its widths and whether a state rides the call."""
    from fsla.ops import intra as intra_ops

    widths = widths_of(routes)
    return (widths, d_v, decay, intra_ops.admits(widths, intra_ops.VALUE_WIDTH),
            intra_ops.VALUE_WIDTH, stateful)


def arm_refusal(routes, decay: Any = None, *, d_v: int, stateful: bool = False) -> str:
    """Why the prefill kernels refuse this segment (they always do while no carry is built)."""
    return rules.prefill_envelope(*prefill_clauses(routes, decay, d_v=d_v, stateful=stateful))


def envelope_refusal(routes: RouteFactors, values: Values, decay: Any = None, *, stateful: bool = False) -> str:
    """Why the built kernel arm has no arm for this call. IT ALWAYS REFUSES: no carry arm is built on this line, so there
    is no call the arm runs and this never returns ``None``; the sentence names the first clause the call fails.

    THE ONE AUTHORITY on the envelope's CONFIGURATION clauses, so a caller can ask
    before calling and get the sentence the refusal will carry. It is a CALLER of the
    rule layer: the sentences are :mod:`fsla.engine.rules.envelope`'s and this function reads
    the facts they fold. The clauses that exist only on the grad path are
    :func:`require_envelope`'s.
    """
    reason = rules.envelope(routes.device)
    if reason is not None:
        return reason
    return arm_refusal(routes, decay, d_v=values.d_v, stateful=stateful)


def require_envelope(routes: RouteFactors, values: Values, decay: Any = None,
                     state: Any = None) -> NoReturn:
    """Refuse anything the built kernels cannot run. THE FAST-FAIL, in one place. IT ALWAYS RAISES: no carry arm is
    built on this line (:func:`envelope_refusal` always has a sentence), so every call that reaches it is refused.

    There is no second arm to route to: the fp64 oracle is the executable spec, not a
    fallback, and a caller who wants it calls it (:func:`fsla.ops.naive.naive_fsla`).

    A GRADIENT IS REFUSED, loudly: the prefill has no reverse pass until the native
    backward is rebuilt, and a forward reached from a grad context must raise rather
    than sever the graph silently (the layer-dispatch autograd gap).
    """
    if requires_backward(*operands_of(routes, values, decay)):
        raise BackwardNotImplemented(
            "a FSLA PREFILL carrying a gradient: the prefill has no reverse pass until the native backward is "
            "rebuilt, and running it would sever the graph silently. Run under torch.no_grad(), freeze the "
            "operands, or train through the fp64 oracle (fsla.ops.naive.naive_fsla).")
    raise KernelNotBuilt(
        f"the prefill kernel is not built on this line: fsla_op runs the BUILT envelope only and there is no kernel "
        f"for this call: "
        f"{envelope_refusal(routes, values, decay, stateful=state is not None)}. Nothing else is reachable through "
        "this surface (docs/api.md §1).")
