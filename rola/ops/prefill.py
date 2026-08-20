# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""K31 — the box-native PREFILL: the carry and the intra as one call.

THE TWO KERNELS ARE ONE OPERATOR, and this module is the seam that says so. The
window grid is GLOBAL and FIXED, identical for both: the carry computes every
pair whose write fell in a STRICTLY EARLIER window than its read, and the intra
computes every pair INSIDE one window. Together they are the whole recurrence,
with no pair counted twice and none dropped — the study's exactness clause, which
`tests/k31/test_carry_vs_oracle.py::test_m2_decomposition_is_exact` states
independently in fp64.

Symbols, each at first use: ``widths`` the per-level digit counts, ``D`` levels,
``k`` the uniform sub-box span, ``m`` the capacity multiplier, ``W`` the window
length in tokens, ``C`` the carry's private batch tile, ``BC`` the leaves one
owner holds, ``d_v = 64`` value channels, ``modes`` the per-level support-mode
declaration the intra's zero certificate reads.

WHY THE INTRA GOES SECOND AND REDUCES.  The intra kernel's write-back is a RED
(``fan_in_add``), not a store, so it ADDS into whatever ``o``/``den`` already
hold. The carry allocates and fills them; the intra then folds its term into the
same buffers on the same stream. Nothing is summed on the host, and there is no
second output pair to reconcile — which is the whole reason the intra was built
as a reducer rather than as a returner.

WHAT THIS OPERATOR REFUSES, and why each refusal is structural rather than a
missing feature:

* ``L % W != 0`` — the intra's tile grid is a COMPILE-TIME count per arm, so a
  partial trailing window has no kernel. (The carry alone tolerates one; the
  combined operator cannot, and a host-side patch for the tail would be a second
  code path for the case the window grid exists to remove.)
* any topology but ``(256, 256)`` — the intra's level width is 256 by
  instantiation. The carry's ``(64, 64)`` arm therefore has no intra partner and
  is an INTER-ONLY conformance cell; see `docs/internals/intra/intra_kernel.md`.
"""
from __future__ import annotations

import torch

from rola.ops import carry as carry_ops
from rola.ops import intra as intra_ops
from rola.ops._ext import extension

#: the built windows, restated from the two families so a caller has one import.
WINDOW = carry_ops.WINDOW
WINDOW_WIDE = carry_ops.WINDOW_WIDE

#: the topology the intra kernel is instantiated at, and therefore the one the
#: combined operator has.
LEVEL_WIDTH = intra_ops.LEVEL_WIDTH


def prefill(read_levels, write_levels, g_write, v, widths, *, k: int, m: int,
            window: int, modes, chunk: int = 32,
            state_in: torch.Tensor | None = None):
    """The whole forward term, undivided: ``(num, den, state_out)``.

    ``num`` is ``[BH, T, d_v]`` fp32, ``den`` is ``[BH, T]`` fp32 and
    ``state_out`` is ``[BH, N/16, 16, d_v+1]`` fp32 in LATTICE leaf order.
    The readout a caller wants is ``num / (den + READOUT_EPS)``; this operator
    does not divide, because the state and the mass are what a chained call
    needs and a ratio is not.

    ``modes`` is one :mod:`rola.ops.intra` support mode per level. It is a
    DECLARATION about the routing, not a switch: the intra's slab gate reads
    all-ones for an undeclared side and therefore cannot move a number, so
    ``DENSE_BOTH`` everywhere is always correct and only ever slower.
    """
    D = len(widths)
    if len(modes) != D:
        raise ValueError(f"{D} levels want {D} support modes; got {len(modes)}")
    if any(w != LEVEL_WIDTH for w in widths):
        raise ValueError(
            f"the combined prefill is built at level width {LEVEL_WIDTH} only "
            f"(the intra kernel's instantiation); got widths {tuple(widths)}. "
            f"Narrower topologies are inter-only cells -- call "
            f"rola.ops.carry.carry_inter directly.")
    length = read_levels[0].shape[1]
    if length % window:
        raise ValueError(
            f"L = {length} is not a whole number of {window}-token windows; the "
            f"intra kernel's tile grid is a compile-time count per arm.")

    #: THE OPERANDS ARE PACKED ONCE.  Both kernels read the SAME planes, so
    #: routing them through `carry_inter` (which packs its own) would pack them
    #: twice and hand the intra a second, equal-valued copy -- two images of one
    #: operand is how a rounding difference becomes a conformance mystery.
    pread = carry_ops.pack_side(read_levels)
    pwrite = carry_ops.pack_side(write_levels)
    gwrite = g_write.permute(0, 2, 1).contiguous().to(torch.bfloat16)
    gwrite = gwrite.reshape(gwrite.shape[0] * gwrite.shape[1], -1)
    value = v.to(torch.bfloat16).contiguous()

    ext = extension()
    num, den, state_out = ext.carry_forward_inter(
        pread, pwrite, gwrite, value, list(widths), int(k), int(m), int(window),
        int(chunk), carry_ops.CHANNEL_SPLIT, state_in)
    #: SAME STREAM, SAME BUFFERS.  The intra REDs into `num`/`den`; the stream
    #: order is the only synchronization either kernel needs, and there is no
    #: host round trip between them.
    ext.intra_forward(pread, pwrite, gwrite, value, num, den, D,
                      intra_ops.pack_modes(tuple(modes)), int(window))
    return num, den, state_out


__all__ = ["LEVEL_WIDTH", "WINDOW", "WINDOW_WIDE", "prefill"]
