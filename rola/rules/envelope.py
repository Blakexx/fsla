# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""EnvelopeRule -- why the built kernel has no arm for a call, in one sentence.

THE BUILT ENVELOPE IS THE WHOLE SURFACE (docs/api.md §1), so these folds hold every
refusal TEXT the op can carry. They are four because the call sites need them at four
different moments -- the arm matrix, the operand facts around it, the reverse pass's
narrower matrix, and the one clause that exists only on the grad path -- and folding
them into one would evaluate clauses a caller has not reached yet
(docs/internals/facts/rules.md).
"""
from __future__ import annotations

from typing import Any

from rola.facts.runner import rule
from rola.facts.types import Arm


@rule(gives=("arm_envelope",),
      needs=("widths", "d_v", "normalization", "decay", "tokens", "arm"))
def arm_envelope(widths: tuple[int, ...], d_v: int, normalization: str, decay: Any,
                 tokens: int, arm: Arm | None) -> str | None:
    """Why the chunk arm refuses this call, or None if it runs.

    `d_v` is the value stream's per-head width, which is a fact about `v` and not
    about the routing -- the caller reads it off the operand and states it here.
    """
    if d_v != 64:
        return f"d_v={d_v}; the chunk matrix is built at d_v=64"
    if normalization != "global":
        return f"normalization={normalization!r}; the chunk consumer is global-only"
    if decay is not None:
        return "decay is not implemented in the operator (the decay retirement; docs/internals/chunk/chunk.md); use the reference arm"
    if len(set(widths)) != 1:
        return f"widths={widths}; the built manifest is uniform-width"
    if arm is None:
        return f"topology (D={len(widths)}, B={widths[0]}) has no built arm in the manifest"
    if tokens % arm.c:
        return f"T={tokens} is not a multiple of the chunk C={arm.c}"
    return None


@rule(gives=("envelope",), needs=("unratified", "device"))
def envelope(unratified: tuple[str, ...], device) -> str | None:
    """The operand-side clauses, which precede the arm matrix's own.

    They are separate from :func:`arm_envelope` because they are facts about the
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


@rule(gives=("backward_envelope",),
      needs=("arm_envelope", "widths", "arm", "scan_manifest"))
def backward_envelope(arm_envelope: str | None, widths: tuple[int, ...], arm: Arm,
                      scan_manifest: set[tuple[int, int, int, int]]) -> str | None:
    """Why a GRAD-BEARING call has no arm, or None if it trains.

    The forward's arm matrix is wider than the scans': a forward that runs and a
    backward that does not is exactly the known-broken path this refuses at the
    boundary instead of discovering three launches later.
    """
    if arm_envelope is not None:
        return arm_envelope
    want = (arm.c, arm.bc, len(widths), widths[0])
    if want not in scan_manifest:
        return (f"the reverse pass is not built for arm C={want[0]} BC={want[1]} "
                f"depth={want[2]} width={want[3]}; it carries "
                f"{sorted(scan_manifest)} (docs/internals/chunk/backward.md)")
    return None


@rule(gives=("training_envelope",), needs=("grad", "state"))
def training_envelope(grad: bool, state: Any) -> str | None:
    """The one clause that exists only on the grad path: a carried STATE.

    Training in v1 is stateless (P65 R-D) -- the state plane is caller-owned and
    mutated in place, and `d initial_state` is not in the scans -- so `state=` and a
    gradient are mutually exclusive, by name, at the boundary rather than silently.
    """
    if grad and state is not None:
        return (
            "rola_op trains STATELESS ONLY and this call carries both a gradient and "
            "a state plane. The state is caller-owned and mutated in place, and the "
            "reverse pass has no `d initial_state`: it rebuilds from zero. Detach the "
            "operands for a stateful step, or drop `state=` to train.")
    return None
