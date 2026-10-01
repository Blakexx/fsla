# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""EnvelopeRule -- why the built kernel has no arm for a call, in one sentence.

THE BUILT ENVELOPE IS THE WHOLE SURFACE (docs/api.md §1), so these folds hold every
refusal TEXT the op can carry. They are three because the call sites need them at three
different moments -- the arm matrix, the operand facts around it, and the one
clause that exists only on the grad path -- and folding
them into one would evaluate clauses a caller has not reached yet
(docs/internals/engine/rules.md).
"""
from __future__ import annotations

from typing import Any

from rola.engine import rule


@rule(gives=("prefill_envelope",),
      needs=("widths", "d_v", "decay", "tokens", "window", "intra_reads", "intra_dv", "carry_dvs"))
def prefill_envelope(widths: tuple[int, ...], d_v: int, decay: Any, tokens: int, window: int,
                     intra_reads: bool, intra_dv: int, carry_dvs: tuple[int, ...]) -> str | None:
    """Why the prefill has no kernel for this segment, or None if it runs.

    ``intra_reads`` is whether the intra kernel reads these level widths (`rola.ops.intra.admits`) and ``intra_dv``
    the value width it reads; ``carry_dvs`` the value widths the carry ships at this depth -- the binary's own
    census, never the source tree's.
    """
    depth = len(widths)
    if decay is not None:
        return ("decay is not implemented in the operator (the decay retirement; docs/internals/chunk/chunk.md); "
                "use the reference arm")
    if not intra_reads:
        return (f"widths={widths}; the intra reads one or two levels, each a whole number of 16-digit granules up "
                f"to 256, within the device's shared memory")
    if d_v != intra_dv:
        return f"d_v={d_v}; the intra reads value width {intra_dv}"
    if not any(dv >= d_v for dv in carry_dvs):
        return f"d_v={d_v}; this build's carry at depth {depth} takes value widths {sorted(carry_dvs)}"
    if tokens % window:
        return f"T={tokens} is not a whole number of {window}-token windows"
    return None


@rule(gives=("envelope",), needs=("unratified", "device"))
def envelope(unratified: tuple[str, ...], device) -> str | None:
    """The operand-side clauses, which precede the prefill's own.

    They are separate from :func:`prefill_envelope` because they are facts about the
    operands rather than about the configuration, and because an unratified
    activation or a CPU tensor must be named BEFORE a question about the built
    matrix is asked of a device that may not have one.
    """
    if unratified:
        return (f"unratified activations {unratified}; no kernel was measured for them "
                "(docs/ratification.md)")
    if device.type != "cuda":
        return f"the operands are on {device}; the arm is a CUDA kernel"
    return None


