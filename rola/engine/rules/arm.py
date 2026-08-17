# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""ArmRule -- the `(C, BC)` arm a topology runs on, derived from the manifest."""
from __future__ import annotations

from rola.engine import Arm, rule

#: THE CHUNK LENGTH -- the operator's token quantum. `T` must be a multiple of it,
#: and shaping a sequence to it is the CALLER's job, which is why this is public
#: rather than only reachable through a refusal message.
CHUNK_TOKENS = 32

#: The pinned plan: a fixed preference order over MEASURED arms, never a tuned
#: heuristic. It generalizes only by falling back to the largest built BC when the
#: pinned one is not in this topology's manifest.
_PIN_C, _PIN_BC = CHUNK_TOKENS, 128


@rule(gives=("arm",), needs=("widths", "manifest"))
def arm(widths: tuple[int, ...], manifest: tuple[tuple[int, int, int, int], ...]) -> Arm | None:
    """`(C, BC)` -- the built arm this topology runs on, or None if it has none.

    DERIVED FROM THE MANIFEST, never from a threshold on `d_v` or `N`: what exists
    is what was built and measured, and the preference order over what exists is the
    whole of the policy.
    """
    depth, width = len(widths), widths[0]
    bcs = [bc for (c, bc, d, b) in manifest if d == depth and b == width and c == _PIN_C]
    if not bcs:
        return None
    return Arm(_PIN_C, _PIN_BC if _PIN_BC in bcs else max(bcs))
