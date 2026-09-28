# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PRE-PASS, as the caller reaches it: den and the packed panel's per-token bound, from the bundle.

The pre-pass is the int8 carry's skeleton at two columns a leaf (mass and bound). It consumes the
internal format directly -- the u8 unit-max rows with their per-level maxima, the write gain, the
token's value scale -- and REDUCES two floats a token. Its declared arithmetic is
``tests/oracle/int_reference.prepass_reference``; the liveness words it walks follow the u8 digit
(a digit is live iff its byte is nonzero), which is the format's own support rule.
See docs/internals/prepass/prepass.md
"""
from __future__ import annotations

import torch

from rola.ops._ext import extension
from rola.routing.factors import RouteFactors
from rola.values import Values

__all__ = ["level_modes_of", "liveness_words_u8", "pack_rows", "prepass_forward", "prepass_plan"]


def level_modes_of(sparsity) -> int:
    """The mode word from ``[(read_sparse, write_sparse)] per level``: two bits a level, read then write."""
    word = 0
    for level, (read_sparse, write_sparse) in enumerate(sparsity):
        word |= (int(bool(read_sparse)) | (int(bool(write_sparse)) << 1)) << (2 * level)
    return word


def pack_rows(levels) -> torch.Tensor:
    """The per-level ``[B, T, H, B_l]`` u8 rows as one ``[BH, L, sum_l B_l]`` u8 plane, levels in canonical order."""
    Bn, T, H, _ = levels[0].shape
    out = torch.cat([level.permute(0, 2, 1, 3) for level in levels], dim=-1)
    return out.reshape(Bn * H, T, -1).contiguous()


def _pack_scalars(x: torch.Tensor) -> torch.Tensor:
    """``[B, T, H]`` -> ``[BH, L]`` fp32."""
    Bn, T, H = x.shape
    return x.permute(0, 2, 1).reshape(Bn * H, T).to(torch.float32).contiguous()


def _pack_scales(scales) -> torch.Tensor:
    """The per-level ``[B, T, H]`` maxima as ``[BH, L, D]`` fp32."""
    Bn, T, H = scales[0].shape
    return torch.stack([s.permute(0, 2, 1) for s in scales], dim=-1).reshape(Bn * H, T, -1).to(torch.float32).contiguous()


def liveness_words_u8(rows_read: torch.Tensor, rows_write: torch.Tensor) -> torch.Tensor:
    """``[BH, 2, sum_l B_l, ceil(L / 32)]`` int32: bit ``t % 32`` of word ``(row, t // 32)`` set when token ``t``'s
    u8 digit ``row`` is nonzero on that side -- THE FORMAT'S SUPPORT RULE, a torch fold of the rows."""
    BH, L, wtot = rows_read.shape
    words = (L + 31) // 32
    pad = words * 32 - L
    out = []
    for rows in (rows_read, rows_write):
        live = rows != 0                                              # [BH, L, wtot]
        if pad:
            live = torch.nn.functional.pad(live, (0, 0, 0, pad))
        bits = live.permute(0, 2, 1).reshape(BH, wtot, words, 32)     # [BH, rows, words, 32]
        weights = (1 << torch.arange(32, device=rows.device, dtype=torch.int64))
        packed = (bits.to(torch.int64) * weights).sum(-1)             # [BH, rows, words]
        out.append(((packed + 2**31) % 2**32 - 2**31).to(torch.int32))   # the u32 word as int32 bits
    return torch.stack(out, dim=1).contiguous()


def prepass_forward(factors: RouteFactors, values: Values, sparsity):
    """``(den, scale)`` as ``[BH, L]`` fp32, REDUCED into zeroed planes by the pre-pass kernel.

    ``sparsity`` is the per-level ``(read_sparse, write_sparse)`` declaration. The class-1 words are folded
    from the rows here, under the format's support rule.
    """
    widths = [int(w) for w in factors.topology.widths]
    rows_read, rows_write = pack_rows(factors.amp_read), pack_rows(factors.amp_write)
    scale_read, scale_write = _pack_scales(factors.scale_read), _pack_scales(factors.scale_write)
    gain, vscale = _pack_scalars(factors.g_write), _pack_scalars(values.scale)
    liveness = liveness_words_u8(rows_read, rows_write)
    BH, L = gain.shape
    den = torch.zeros(BH, L, dtype=torch.float32, device=gain.device)
    scale = torch.zeros(BH, L, dtype=torch.float32, device=gain.device)
    extension().prepass_forward(rows_read, rows_write, scale_read, scale_write, gain, vscale, den, scale,
                                widths, level_modes_of(sparsity), liveness)
    return den, scale


def prepass_plan(depth: int = 2) -> dict:
    """The compile-time plan as the extension reports it."""
    threads, smem, leaves, leaves_per_warp, pool_tok, chunks, rng, phases, regs, local = extension().prepass_plan(depth)
    return dict(threads=threads, smem_bytes=smem, leaves=leaves, leaves_per_warp=leaves_per_warp,
                pool_tok=pool_tok, chunks=chunks, range=rng, phases=phases, registers=regs, local_bytes=local)


def ledger_bind(ledger: torch.Tensor | None) -> None:
    """Bind (or unbind) the phase ledger the next launches add into: ``[ctas][8][phases]`` int64 on the device."""
    extension().prepass_ledger_bind(ledger)
