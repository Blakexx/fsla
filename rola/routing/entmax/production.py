"""Production routing factors: the exact sorted alpha=1.5/2.0 solve and its analytic VJP.

This module is deliberately separate from ``routing.entmax.reference``. The reference is an fp64
bisection oracle; this module owns the value path every runtime caller takes, and its analytic
fixed-alpha VJP. It offers the same solve through two plan-level surfaces:

* ``production_routing_factor_levels`` -- dense per-level ``[BH, T, width_l]`` factors, the contract
  V3's consumer and its fp64 oracle both consume. This is what the V3 producer calls.
* ``production_routing_factors`` -- the bf16 tile-major ``RouteDescriptor`` arena ABI.

and through two lowerings of one semantics: fused Triton on CUDA fp32, and the identical sorted
closed form in torch for the domains Triton does not cover (CPU, fp64). Neither surface ever calls
the reference; it is the test oracle, not a fallback.
"""

from __future__ import annotations

from math import prod

import torch
import triton
import triton.language as tl

from rola.routing.topology import MAX_BRANCH_WIDTH
from rola.routing.types import (
    ROUTE_TILE_SIZE as _ROUTE_TILE_SIZE_VALUE,
)
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    ResolvedRouting,
    RouteDescriptor,
    SoftmaxActivation,
    UnionRouting,
)

_SPARSE_ACTIVATIONS = {"entmax", "sparsemax"}
_SUPPORTED_PRODUCTION_ALPHAS = (1.5, 2.0)
_TRITON_FP32_MAX = tl.constexpr(3.4028234663852886e38)
_TRITON_FP32_HALF_MAX = tl.constexpr(1.7014117331926443e38)
_TRITON_ROUTE_TILE_SIZE = tl.constexpr(_ROUTE_TILE_SIZE_VALUE)
# Elements (rows x padded width) of one union-solve sub-tile; see `_union_row_tile`.
# 512 = the measured sm_86 knee (2 rows x 256 cols at the parity topology), from the
# p3 sweep of 1/2/4/8/16/32 rows at L=2048, median of 30 launches:
#   rows  32 -> 1120 us, 255 reg, 468 spilled, 32.8 KB smem   (the pre-p3 shape)
#   rows   8 ->  394 us, 255 reg,   0 spilled,  8.2 KB smem
#   rows   2 ->  348 us, 128 reg,   0 spilled,  4.1 KB smem   <- knee
#   rows   1 ->  700 us,  80 reg,   0 spilled,  1.0 KB smem   (sort too thin to amortize)
# Capping registers instead (`maxnreg` 64/96/128) reintroduces spills and is 1.4-3.4x
# slower; num_warps 2 or 8 are both worse. Shrinking the sub-tile is the only lever
# that moves register demand and sort scratch together.
_UNION_SOLVE_TILE = 512


def _production_alpha(alpha: float) -> float:
    """Validate the two exact production entmax specializations."""
    alpha = float(alpha)
    if alpha not in _SUPPORTED_PRODUCTION_ALPHAS:
        raise ValueError(
            "production routing supports exactly alpha=1.5 (entmax) and alpha=2.0 "
            f"(sparsemax), got {alpha!r}")
    return alpha


@triton.jit
def _production_fwd_kernel(
    z_ptr,
    p_ptr,
    p_second_ptr,
    support_words_ptr,
    mask_ptr,
    n_streams,
    tokens,
    n_words,
    n_cols,
    z_stream,
    z_token,
    z_col,
    p_stream,
    p_token,
    p_col,
    p_tile,
    p_lane,
    p_second_stream,
    p_second_token,
    p_second_col,
    p_second_tile,
    p_second_lane,
    support_stream_stride,
    support_word_stride,
    support_col_stride,
    m_stream,
    m_token,
    m_col,
    ALPHA: tl.constexpr,
    HAS_MASK: tl.constexpr,
    WRITE_SECOND: tl.constexpr,
    TILE_MAJOR: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    program = tl.program_id(0)
    stream = program // n_words
    word = program - stream * n_words
    token = word * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_COLS)
    rmask = (stream < n_streams) & (token < tokens)
    cmask = cols < n_cols
    valid = rmask[:, None] & cmask[None, :]
    if HAS_MASK:
        valid &= tl.load(
            mask_ptr + stream * m_stream + token[:, None] * m_token + cols[None, :] * m_col,
            mask=rmask[:, None] & cmask[None, :], other=0)
    z = tl.load(z_ptr + stream * z_stream + token[:, None] * z_token + cols[None, :] * z_col,
                mask=valid, other=0.0).to(tl.float32)
    valid_count = tl.sum(valid.to(tl.int32), axis=1)[:, None]
    has_valid = valid_count > 0
    row_max = tl.max(tl.where(valid, z, float("-inf")), axis=1)[:, None]
    row_max = tl.where(has_valid, row_max, 0.0)
    z = tl.where(valid, z - row_max, 0.0)

    # Both supported activations have an exact threshold after sorting.  For
    # sparsemax, tau_k = (sum(z[:k]) - 1) / k.  For alpha=1.5, solve
    # sum((z[:k] - tau)^2) = 4 for its smaller root.  The greatest valid k
    # defines the support, so no iterative tolerance enters the contract.
    # This Triton toolchain corrupts bitonic-sort lanes containing infinities.
    # A finite row-relative sentinel is strictly below every valid value and
    # preserves the same ordering without poisoning the prefix scan.
    row_min = tl.min(tl.where(valid, z, float("inf")), axis=1)[:, None]
    sentinel = tl.where(has_valid, row_min - 1.0, -1.0)
    sorted_z = tl.sort(tl.where(valid, z, sentinel), dim=1, descending=True)
    sorted_valid = cols[None, :] < valid_count
    prefix = tl.cumsum(sorted_z, axis=1)
    prefix_sq = tl.cumsum(sorted_z * sorted_z, axis=1)
    k = (cols[None, :] + 1).to(tl.float32)
    if ALPHA == 2.0:
        tau_candidates = (prefix - 1.0) / k
        candidate = sorted_valid & (sorted_z > tau_candidates)
    else:
        discriminant = prefix * prefix - k * (prefix_sq - 4.0)
        tau_candidates = (prefix - tl.sqrt(tl.maximum(discriminant, 0.0))) / k
        candidate = sorted_valid & (discriminant >= 0.0) & (sorted_z > tau_candidates)
    chosen_col = tl.max(tl.where(candidate, cols[None, :], -1), axis=1)[:, None]
    tau = tl.sum(tl.where(cols[None, :] == chosen_col, tau_candidates, 0.0), axis=1)[:, None]
    tau = tl.where(chosen_col >= 0, tau, 0.0)
    semantic_support = valid & (z > tau)
    base = (ALPHA - 1.0) * (z - tau)
    base = tl.where(semantic_support, tl.maximum(base, 0.0), 0.0)
    if ALPHA == 2.0:
        p = base
    else:
        p = base * base

    output_mask = rmask[:, None] & cmask[None, :]
    p_offsets = tl.where(
        TILE_MAJOR,
        stream * p_stream + (token[:, None] // _TRITON_ROUTE_TILE_SIZE) * p_tile
        + cols[None, :] * p_col + (token[:, None] % _TRITON_ROUTE_TILE_SIZE) * p_lane,
        stream * p_stream + token[:, None] * p_token + cols[None, :] * p_col,
    )
    tl.store(p_ptr + p_offsets, p,
             mask=output_mask)
    if WRITE_SECOND:
        p_second_offsets = tl.where(
            TILE_MAJOR,
            stream * p_second_stream + (token[:, None] // _TRITON_ROUTE_TILE_SIZE) * p_second_tile
            + cols[None, :] * p_second_col + (token[:, None] % _TRITON_ROUTE_TILE_SIZE) * p_second_lane,
            stream * p_second_stream + token[:, None] * p_second_token + cols[None, :] * p_second_col,
        )
        tl.store(
            p_second_ptr + p_second_offsets,
            p,
            mask=output_mask,
        )
    # Scheduling support is defined at the native bf16 storage boundary, not
    # by the pre-quantization semantic solve. This makes support and the route
    # stream consumed by CUDA one authority. The packed word is a bitwise OR
    # because every row contributes a unique token bit. Sum in int64: Triton's
    # uint32 reduction may lower through a lossy accumulator at bit 31.
    stored_support = semantic_support & (p.to(tl.bfloat16).to(tl.float32) != 0.0)
    bit_values = tl.where(
        stored_support,
        tl.full((BLOCK_ROWS, BLOCK_COLS), 1, tl.int64) << (token[:, None] % 32),
        tl.zeros((BLOCK_ROWS, BLOCK_COLS), tl.int64),
    )
    packed = tl.sum(bit_values, axis=0)
    tl.store(
        support_words_ptr + stream * support_stream_stride
        + word * support_word_stride + cols * support_col_stride,
        packed.to(tl.uint32).to(tl.int32, bitcast=True),
        mask=cmask,
    )


@triton.jit
def _production_bwd_kernel(
    p_ptr,
    support_words_ptr,
    mask_ptr,
    dp_ptr,
    dp_second_ptr,
    dz_ptr,
    n_streams,
    tokens,
    n_words,
    n_cols,
    p_stream,
    p_token,
    p_col,
    p_tile,
    p_lane,
    support_stream_stride,
    support_word_stride,
    support_col_stride,
    m_stream,
    m_token,
    m_col,
    dp_stream,
    dp_token,
    dp_col,
    dp_tile,
    dp_lane,
    dp_second_stream,
    dp_second_token,
    dp_second_col,
    dp_second_tile,
    dp_second_lane,
    dz_stream,
    dz_token,
    dz_col,
    ALPHA: tl.constexpr,
    HAS_MASK: tl.constexpr,
    HAS_SECOND_DP: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    TILE_MAJOR: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    program = tl.program_id(0)
    stream = program // n_words
    word = program - stream * n_words
    token = word * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_COLS)
    rmask = (stream < n_streams) & (token < tokens)
    cmask = cols < n_cols
    valid = rmask[:, None] & cmask[None, :]
    if HAS_MASK:
        valid &= tl.load(
            mask_ptr + stream * m_stream + token[:, None] * m_token + cols[None, :] * m_col,
            mask=rmask[:, None] & cmask[None, :], other=0)
    p_offsets = tl.where(
        TILE_MAJOR,
        stream * p_stream + (token[:, None] // _TRITON_ROUTE_TILE_SIZE) * p_tile
        + cols[None, :] * p_col + (token[:, None] % _TRITON_ROUTE_TILE_SIZE) * p_lane,
        stream * p_stream + token[:, None] * p_token + cols[None, :] * p_col,
    )
    p = tl.load(
        p_ptr + p_offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    support_words = tl.load(
        support_words_ptr + stream * support_stream_stride
        + word * support_word_stride + cols * support_col_stride,
        mask=cmask,
        other=0,
    ).to(tl.uint32, bitcast=True)
    support = (
        (support_words >> (token[:, None] % 32))
        & tl.full((BLOCK_ROWS, BLOCK_COLS), 1, tl.uint32)
    ) != 0
    if ALPHA == 2.0:
        s = tl.where(valid & support, 1.0, 0.0)
    else:
        s = tl.where(valid & support, tl.sqrt(p), 0.0)
    dp_offsets = tl.where(
        TILE_MAJOR,
        stream * dp_stream + (token[:, None] // _TRITON_ROUTE_TILE_SIZE) * dp_tile
        + cols[None, :] * dp_col + (token[:, None] % _TRITON_ROUTE_TILE_SIZE) * dp_lane,
        stream * dp_stream + token[:, None] * dp_token + cols[None, :] * dp_col,
    )
    dp = tl.load(dp_ptr + dp_offsets,
                 mask=valid, other=0.0).to(tl.float32)
    if HAS_SECOND_DP:
        dp_second_offsets = tl.where(
            TILE_MAJOR,
            stream * dp_second_stream + (token[:, None] // _TRITON_ROUTE_TILE_SIZE) * dp_second_tile
            + cols[None, :] * dp_second_col + (token[:, None] % _TRITON_ROUTE_TILE_SIZE) * dp_second_lane,
            stream * dp_second_stream + token[:, None] * dp_second_token + cols[None, :] * dp_second_col,
        )
        dp += tl.load(
            dp_second_ptr + dp_second_offsets,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
    num = tl.sum(s * dp, axis=1)[:, None]
    den = tl.sum(s, axis=1)[:, None]
    den = tl.where(den > 0.0, den, 1.0)
    dz = tl.where(valid, s * (dp - num / den), 0.0)
    dz_ptrs = dz_ptr + stream * dz_stream + token[:, None] * dz_token + cols[None, :] * dz_col
    if ACCUMULATE:
        tl.atomic_add(dz_ptrs, dz, mask=rmask[:, None] & cmask[None, :])
    else:
        tl.store(dz_ptrs, dz, mask=rmask[:, None] & cmask[None, :])


@triton.jit
def _production_softmax_fwd_kernel(
    z_ptr,
    p_ptr,
    p_second_ptr,
    n_streams,
    tokens,
    n_words,
    n_cols,
    z_stream,
    z_token,
    z_col,
    p_stream,
    p_token,
    p_col,
    p_tile,
    p_lane,
    p_second_stream,
    p_second_token,
    p_second_col,
    p_second_tile,
    p_second_lane,
    WRITE_SECOND: tl.constexpr,
    TILE_MAJOR: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    program = tl.program_id(0)
    stream = program // n_words
    word = program - stream * n_words
    token = word * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_COLS)
    rmask = (stream < n_streams) & (token < tokens)
    cmask = cols < n_cols
    mask = rmask[:, None] & cmask[None, :]
    z = tl.load(
        z_ptr + stream * z_stream + token[:, None] * z_token + cols[None, :] * z_col,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    z = z - tl.max(z, axis=1)[:, None]
    p = tl.exp(z)
    p /= tl.sum(p, axis=1)[:, None]
    p_offsets = tl.where(
        TILE_MAJOR,
        stream * p_stream + (token[:, None] // _TRITON_ROUTE_TILE_SIZE) * p_tile
        + cols[None, :] * p_col + (token[:, None] % _TRITON_ROUTE_TILE_SIZE) * p_lane,
        stream * p_stream + token[:, None] * p_token + cols[None, :] * p_col,
    )
    tl.store(p_ptr + p_offsets, p, mask=mask)
    if WRITE_SECOND:
        p_second_offsets = tl.where(
            TILE_MAJOR,
            stream * p_second_stream + (token[:, None] // _TRITON_ROUTE_TILE_SIZE) * p_second_tile
            + cols[None, :] * p_second_col + (token[:, None] % _TRITON_ROUTE_TILE_SIZE) * p_second_lane,
            stream * p_second_stream + token[:, None] * p_second_token + cols[None, :] * p_second_col,
        )
        tl.store(
            p_second_ptr + p_second_offsets,
            p,
            mask=mask,
        )


@triton.jit
def _production_softmax_bwd_kernel(
    p_ptr,
    dp_ptr,
    dp_second_ptr,
    dz_ptr,
    n_streams,
    tokens,
    n_words,
    n_cols,
    p_stream,
    p_token,
    p_col,
    p_tile,
    p_lane,
    dp_stream,
    dp_token,
    dp_col,
    dp_tile,
    dp_lane,
    dp_second_stream,
    dp_second_token,
    dp_second_col,
    dp_second_tile,
    dp_second_lane,
    dz_stream,
    dz_token,
    dz_col,
    HAS_SECOND_DP: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    TILE_MAJOR: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    program = tl.program_id(0)
    stream = program // n_words
    word = program - stream * n_words
    token = word * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_COLS)
    rmask = (stream < n_streams) & (token < tokens)
    cmask = cols < n_cols
    mask = rmask[:, None] & cmask[None, :]
    p_offsets = tl.where(
        TILE_MAJOR,
        stream * p_stream + (token[:, None] // _TRITON_ROUTE_TILE_SIZE) * p_tile
        + cols[None, :] * p_col + (token[:, None] % _TRITON_ROUTE_TILE_SIZE) * p_lane,
        stream * p_stream + token[:, None] * p_token + cols[None, :] * p_col,
    )
    p = tl.load(
        p_ptr + p_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    dp_offsets = tl.where(
        TILE_MAJOR,
        stream * dp_stream + (token[:, None] // _TRITON_ROUTE_TILE_SIZE) * dp_tile
        + cols[None, :] * dp_col + (token[:, None] % _TRITON_ROUTE_TILE_SIZE) * dp_lane,
        stream * dp_stream + token[:, None] * dp_token + cols[None, :] * dp_col,
    )
    dp = tl.load(dp_ptr + dp_offsets, mask=mask, other=0.0).to(tl.float32)
    if HAS_SECOND_DP:
        dp_second_offsets = tl.where(
            TILE_MAJOR,
            stream * dp_second_stream + (token[:, None] // _TRITON_ROUTE_TILE_SIZE) * dp_second_tile
            + cols[None, :] * dp_second_col + (token[:, None] % _TRITON_ROUTE_TILE_SIZE) * dp_second_lane,
            stream * dp_second_stream + token[:, None] * dp_second_token + cols[None, :] * dp_second_col,
        )
        dp += tl.load(
            dp_second_ptr + dp_second_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    dz = p * (dp - tl.sum(p * dp, axis=1)[:, None])
    dz_ptrs = dz_ptr + stream * dz_stream + token[:, None] * dz_token + cols[None, :] * dz_col
    if ACCUMULATE:
        tl.atomic_add(dz_ptrs, dz, mask=mask)
    else:
        tl.store(dz_ptrs, dz, mask=mask)


@triton.jit
def _production_union_fwd_kernel(
    read_logits_ptr,
    write_logits_ptr,
    midpoint_values_ptr,
    support_words_ptr,
    read_ptr,
    write_ptr,
    mask_ptr,
    n_streams,
    tokens,
    n_words,
    n_cols,
    read_logits_stream,
    read_logits_token,
    read_logits_col,
    write_logits_stream,
    write_logits_token,
    write_logits_col,
    midpoint_values_stream,
    midpoint_values_token,
    midpoint_values_col,
    read_stream,
    read_token,
    read_col,
    read_tile,
    read_lane,
    write_stream,
    write_token,
    write_col,
    write_tile,
    write_lane,
    support_stream_stride,
    support_word_stride,
    support_col_stride,
    mask_stream,
    mask_token,
    mask_col,
    ALPHA: tl.constexpr,
    HAS_MASK: tl.constexpr,
    TILE_MAJOR: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    ROW_TILE: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    program = tl.program_id(0)
    stream = program // n_words
    word = program - stream * n_words
    cols = tl.arange(0, BLOCK_COLS)
    cmask = cols < n_cols
    # One CTA still owns exactly the BLOCK_ROWS=32 tokens of one support word (the
    # `1 << (token % 32)` packing below depends on it), but the entmax solve runs on
    # ROW_TILE rows at a time. The SM-resident cost of this kernel is
    # ROW_TILE * BLOCK_COLS * (live fp32 tiles), so the sub-tile -- not the warp
    # count -- is what sets registers/CTA and the `tl.sort` shared scratch together.
    packed = tl.zeros((BLOCK_COLS,), tl.int64)
    for sub in range(BLOCK_ROWS // ROW_TILE):
        token = word * BLOCK_ROWS + sub * ROW_TILE + tl.arange(0, ROW_TILE)
        rmask = (stream < n_streams) & (token < tokens)
        valid = rmask[:, None] & cmask[None, :]
        if HAS_MASK:
            valid &= tl.load(
                mask_ptr + stream * mask_stream + token[:, None] * mask_token
                + cols[None, :] * mask_col,
                mask=rmask[:, None] & cmask[None, :], other=0)
        read_logits = tl.load(
            read_logits_ptr + stream * read_logits_stream + token[:, None] * read_logits_token
            + cols[None, :] * read_logits_col,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        write_logits = tl.load(
            write_logits_ptr + stream * write_logits_stream + token[:, None] * write_logits_token
            + cols[None, :] * write_logits_col,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        midpoint_logits = 0.5 * read_logits + 0.5 * write_logits
        valid_count = tl.sum(valid.to(tl.int32), axis=1)[:, None]
        has_valid = valid_count > 0
        row_max = tl.max(
            tl.where(valid, midpoint_logits, float("-inf")), axis=1)[:, None]
        row_max = tl.where(has_valid, row_max, 0.0)
        midpoint_logits = tl.where(valid, midpoint_logits - row_max, 0.0)

        # M - hi is exact where saturation is possible; clamp lo before adding magnitudes.
        # Hoisted ahead of the sort: `delta` is a pure elementwise fold of the two logit
        # tiles, so computing it here retires BOTH of them before the sort's live range
        # instead of carrying them across it (one live tile instead of two).
        read_negative = read_logits < 0.0
        write_negative = write_logits < 0.0
        opposite_sign = read_negative != write_negative
        read_magnitude = tl.abs(read_logits)
        write_magnitude = tl.abs(write_logits)
        hi_magnitude = tl.maximum(read_magnitude, write_magnitude)
        lo_magnitude = tl.minimum(read_magnitude, write_magnitude)
        magnitude_room = _TRITON_FP32_MAX - hi_magnitude
        opposite_magnitude = hi_magnitude + tl.minimum(lo_magnitude, magnitude_room)
        opposite_delta = tl.where(read_negative, -opposite_magnitude, opposite_magnitude)
        same_read = tl.where(opposite_sign, 0.0, read_logits)
        same_write = tl.where(opposite_sign, 0.0, write_logits)
        delta = tl.where(opposite_sign, opposite_delta, same_read - same_write)

        row_min = tl.min(
            tl.where(valid, midpoint_logits, float("inf")), axis=1)[:, None]
        sentinel = tl.where(has_valid, row_min - 1.0, -1.0)
        sorted_logits = tl.sort(
            tl.where(valid, midpoint_logits, sentinel), dim=1, descending=True)
        sorted_valid = cols[None, :] < valid_count
        prefix = tl.cumsum(sorted_logits, axis=1)
        prefix_sq = tl.cumsum(sorted_logits * sorted_logits, axis=1)
        k = (cols[None, :] + 1).to(tl.float32)
        if ALPHA == 2.0:
            tau_candidates = (prefix - 1.0) / k
            candidate = sorted_valid & (sorted_logits > tau_candidates)
        else:
            discriminant = prefix * prefix - k * (prefix_sq - 4.0)
            tau_candidates = (prefix - tl.sqrt(tl.maximum(discriminant, 0.0))) / k
            candidate = sorted_valid & (discriminant >= 0.0) & (sorted_logits > tau_candidates)
        chosen_col = tl.max(tl.where(candidate, cols[None, :], -1), axis=1)[:, None]
        tau = tl.sum(
            tl.where(cols[None, :] == chosen_col, tau_candidates, 0.0), axis=1)[:, None]
        tau = tl.where(chosen_col >= 0, tau, 0.0)
        semantic_support = valid & (midpoint_logits > tau)
        base = (ALPHA - 1.0) * (midpoint_logits - tau)
        base = tl.where(semantic_support, tl.maximum(base, 0.0), 0.0)
        if ALPHA == 2.0:
            midpoint_values = base
        else:
            midpoint_values = base * base

        read_gate = tl.sigmoid(delta)
        # log(sigmoid(-delta)) stays finite for large finite logits, unlike log(sigmoid()).
        log_write_gate = tl.where(delta > 0.0, -delta, 0.0) - tl.log(
            1.0 + tl.exp(-tl.abs(delta)))
        safe_values = tl.where(semantic_support, midpoint_values, 1.0)
        log_weight = tl.where(
            semantic_support,
            tl.log(safe_values) + log_write_gate,
            float("-inf"),
        )
        max_log_weight = tl.max(log_weight, axis=1)[:, None]
        has_mass = max_log_weight > float("-inf")
        safe_max_log_weight = tl.where(has_mass, max_log_weight, 0.0)
        scaled_weight = tl.where(
            semantic_support & has_mass,
            tl.exp(log_weight - safe_max_log_weight),
            0.0,
        )
        denominator = tl.sum(scaled_weight, axis=1)[:, None]
        has_weight = denominator > 0.0
        safe_denominator = tl.where(has_weight, denominator, 1.0)
        write = tl.where(
            semantic_support & has_weight,
            scaled_weight / safe_denominator,
            0.0,
        )
        read = midpoint_values * read_gate

        output_mask = rmask[:, None] & cmask[None, :]
        tl.store(
            midpoint_values_ptr + stream * midpoint_values_stream
            + token[:, None] * midpoint_values_token + cols[None, :] * midpoint_values_col,
            midpoint_values,
            mask=output_mask,
        )
        read_offsets = tl.where(
            TILE_MAJOR,
            stream * read_stream + (token[:, None] // _TRITON_ROUTE_TILE_SIZE) * read_tile
            + cols[None, :] * read_col + (token[:, None] % _TRITON_ROUTE_TILE_SIZE) * read_lane,
            stream * read_stream + token[:, None] * read_token + cols[None, :] * read_col,
        )
        write_offsets = tl.where(
            TILE_MAJOR,
            stream * write_stream + (token[:, None] // _TRITON_ROUTE_TILE_SIZE) * write_tile
            + cols[None, :] * write_col + (token[:, None] % _TRITON_ROUTE_TILE_SIZE) * write_lane,
            stream * write_stream + token[:, None] * write_token + cols[None, :] * write_col,
        )
        tl.store(read_ptr + read_offsets, read, mask=output_mask)
        tl.store(write_ptr + write_offsets, write, mask=output_mask)
        bit_values = tl.where(
            # Union has one scheduling stream for both roles. Pack the union of
            # post-cast nonzeros so either side may create harmless false-live work
            # but a numerically live stored factor is never skipped.
            semantic_support & (
                (read.to(tl.bfloat16).to(tl.float32) != 0.0)
                | (write.to(tl.bfloat16).to(tl.float32) != 0.0)
            ),
            tl.full((ROW_TILE, BLOCK_COLS), 1, tl.int64) << (token[:, None] % 32),
            tl.zeros((ROW_TILE, BLOCK_COLS), tl.int64),
        )
        # Sub-tiles own disjoint bit positions of the same word, so accumulating
        # their partial packs is exact integer OR, bit-for-bit the whole-word sum.
        packed += tl.sum(bit_values, axis=0)
    tl.store(
        support_words_ptr + stream * support_stream_stride
        + word * support_word_stride + cols * support_col_stride,
        packed.to(tl.uint32).to(tl.int32, bitcast=True),
        mask=cmask,
    )


@triton.jit
def _production_union_bwd_kernel(
    read_logits_ptr,
    write_logits_ptr,
    midpoint_values_ptr,
    support_words_ptr,
    mask_ptr,
    read_values_ptr,
    write_values_ptr,
    d_read_ptr,
    d_write_ptr,
    d_read_logits_ptr,
    d_write_logits_ptr,
    n_streams,
    tokens,
    n_words,
    n_cols,
    read_logits_stream,
    read_logits_token,
    read_logits_col,
    write_logits_stream,
    write_logits_token,
    write_logits_col,
    midpoint_values_stream,
    midpoint_values_token,
    midpoint_values_col,
    support_stream_stride,
    support_word_stride,
    support_col_stride,
    mask_stream,
    mask_token,
    mask_col,
    read_values_stream,
    read_values_token,
    read_values_col,
    read_values_tile,
    read_values_lane,
    write_values_stream,
    write_values_token,
    write_values_col,
    write_values_tile,
    write_values_lane,
    d_read_stream,
    d_read_token,
    d_read_col,
    d_read_tile,
    d_read_lane,
    d_write_stream,
    d_write_token,
    d_write_col,
    d_write_tile,
    d_write_lane,
    d_read_logits_stream,
    d_read_logits_token,
    d_read_logits_col,
    d_write_logits_stream,
    d_write_logits_token,
    d_write_logits_col,
    ALPHA: tl.constexpr,
    HAS_MASK: tl.constexpr,
    TILE_MAJOR: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    program = tl.program_id(0)
    stream = program // n_words
    word = program - stream * n_words
    token = word * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_COLS)
    rmask = (stream < n_streams) & (token < tokens)
    cmask = cols < n_cols
    valid = rmask[:, None] & cmask[None, :]
    if HAS_MASK:
        valid &= tl.load(
            mask_ptr + stream * mask_stream + token[:, None] * mask_token + cols[None, :] * mask_col,
            mask=rmask[:, None] & cmask[None, :], other=0)
    midpoint_values = tl.load(
        midpoint_values_ptr + stream * midpoint_values_stream
        + token[:, None] * midpoint_values_token + cols[None, :] * midpoint_values_col,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    read_logits = tl.load(
        read_logits_ptr + stream * read_logits_stream + token[:, None] * read_logits_token
        + cols[None, :] * read_logits_col,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    write_logits = tl.load(
        write_logits_ptr + stream * write_logits_stream + token[:, None] * write_logits_token
        + cols[None, :] * write_logits_col,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    read_values_offsets = tl.where(
        TILE_MAJOR,
        stream * read_values_stream + (token[:, None] // _TRITON_ROUTE_TILE_SIZE) * read_values_tile
        + cols[None, :] * read_values_col + (token[:, None] % _TRITON_ROUTE_TILE_SIZE) * read_values_lane,
        stream * read_values_stream + token[:, None] * read_values_token + cols[None, :] * read_values_col,
    )
    write_values_offsets = tl.where(
        TILE_MAJOR,
        stream * write_values_stream + (token[:, None] // _TRITON_ROUTE_TILE_SIZE) * write_values_tile
        + cols[None, :] * write_values_col + (token[:, None] % _TRITON_ROUTE_TILE_SIZE) * write_values_lane,
        stream * write_values_stream + token[:, None] * write_values_token + cols[None, :] * write_values_col,
    )
    # Returned bf16 amplitudes are the route-probability authority for the VJP.
    read_values = tl.load(
        read_values_ptr + read_values_offsets, mask=valid, other=0.0).to(tl.float32)
    write_values = tl.load(
        write_values_ptr + write_values_offsets, mask=valid, other=0.0).to(tl.float32)
    d_read_offsets = tl.where(
        TILE_MAJOR,
        stream * d_read_stream + (token[:, None] // _TRITON_ROUTE_TILE_SIZE) * d_read_tile
        + cols[None, :] * d_read_col + (token[:, None] % _TRITON_ROUTE_TILE_SIZE) * d_read_lane,
        stream * d_read_stream + token[:, None] * d_read_token + cols[None, :] * d_read_col,
    )
    d_write_offsets = tl.where(
        TILE_MAJOR,
        stream * d_write_stream + (token[:, None] // _TRITON_ROUTE_TILE_SIZE) * d_write_tile
        + cols[None, :] * d_write_col + (token[:, None] % _TRITON_ROUTE_TILE_SIZE) * d_write_lane,
        stream * d_write_stream + token[:, None] * d_write_token + cols[None, :] * d_write_col,
    )
    d_read = tl.load(d_read_ptr + d_read_offsets, mask=valid, other=0.0).to(tl.float32)
    d_write = tl.load(d_write_ptr + d_write_offsets, mask=valid, other=0.0).to(tl.float32)

    support_word = tl.load(
        support_words_ptr + stream * support_stream_stride
        + word * support_word_stride + cols * support_col_stride,
        mask=cmask,
        other=0,
    ).to(tl.uint32, bitcast=True)
    support = (
        (support_word >> (token[:, None] % 32))
        & tl.full((BLOCK_ROWS, BLOCK_COLS), 1, tl.uint32)
    ) != 0
    support = support & valid

    # Mirror forward's overflow-free delta and retain the clamp's direct-adjoint mask.
    read_negative = read_logits < 0.0
    write_negative = write_logits < 0.0
    opposite_sign = read_negative != write_negative
    read_magnitude = tl.abs(read_logits)
    write_magnitude = tl.abs(write_logits)
    hi_magnitude = tl.maximum(read_magnitude, write_magnitude)
    lo_magnitude = tl.minimum(read_magnitude, write_magnitude)
    magnitude_room = _TRITON_FP32_MAX - hi_magnitude
    opposite_magnitude = hi_magnitude + tl.minimum(lo_magnitude, magnitude_room)
    opposite_delta = tl.where(read_negative, -opposite_magnitude, opposite_magnitude)
    same_read = tl.where(opposite_sign, 0.0, read_logits)
    same_write = tl.where(opposite_sign, 0.0, write_logits)
    delta = tl.where(opposite_sign, opposite_delta, same_read - same_write)
    saturated = (
        opposite_sign
        & (hi_magnitude > _TRITON_FP32_HALF_MAX)
        & (lo_magnitude > magnitude_room)
    )
    read_gate = tl.sigmoid(delta)
    write_gate = tl.sigmoid(-delta)

    d_log_weight = write_values * (
        d_write - tl.sum(d_write * write_values, axis=1)[:, None])
    safe_gradient_values = tl.where(
        support & (midpoint_values != 0.0),
        midpoint_values,
        1.0,
    )
    d_midpoint_values = d_read * read_gate + tl.where(
        support,
        d_log_weight / safe_gradient_values,
        0.0,
    )
    if ALPHA == 2.0:
        s = tl.where(support, 1.0, 0.0)
    else:
        s = tl.where(support, tl.sqrt(midpoint_values), 0.0)
    numerator = tl.sum(s * d_midpoint_values, axis=1)[:, None]
    denominator = tl.sum(s, axis=1)[:, None]
    denominator = tl.where(denominator > 0.0, denominator, 1.0)
    d_midpoint_logits = tl.where(
        valid,
        s * (d_midpoint_values - numerator / denominator),
        0.0,
    )
    d_delta = (
        d_read * read_values * write_gate
        - d_log_weight * read_gate
    ) * (~saturated).to(tl.float32)
    d_read_logits = 0.5 * d_midpoint_logits + d_delta
    d_write_logits = 0.5 * d_midpoint_logits - d_delta
    tl.store(
        d_read_logits_ptr + stream * d_read_logits_stream
        + token[:, None] * d_read_logits_token + cols[None, :] * d_read_logits_col,
        d_read_logits,
        mask=rmask[:, None] & cmask[None, :],
    )
    tl.store(
        d_write_logits_ptr + stream * d_write_logits_stream
        + token[:, None] * d_write_logits_token + cols[None, :] * d_write_logits_col,
        d_write_logits,
        mask=rmask[:, None] & cmask[None, :],
    )


def _launch_fwd(z, mask, alpha, *, tokens, n_streams):
    rows, cols = z.shape
    block_rows = 32
    block_cols = triton.next_power_of_2(cols)
    p = torch.empty_like(z)
    n_words = triton.cdiv(tokens, block_rows)
    support_words = torch.empty(
        (n_streams, n_words, cols), dtype=torch.int32, device=z.device)
    if rows == 0:
        return p, support_words
    _production_fwd_kernel[(n_streams * n_words,)](
        z, p, p, support_words, mask,
        n_streams, tokens, n_words, cols,
        tokens * z.stride(0), z.stride(0), z.stride(1),
        tokens * p.stride(0), p.stride(0), p.stride(1),
        0, 0,
        tokens * p.stride(0), p.stride(0), p.stride(1),
        0, 0,
        support_words.stride(0), support_words.stride(1), support_words.stride(2),
        tokens * mask.stride(0), mask.stride(0), mask.stride(1),
        ALPHA=float(alpha), HAS_MASK=True, WRITE_SECOND=False, TILE_MAJOR=False,
        BLOCK_ROWS=block_rows, BLOCK_COLS=block_cols,
    )
    return p, support_words


class _ProductionFactorFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, mask, alpha, tokens, n_streams):
        p, support_words = _launch_fwd(
            z, mask, alpha, tokens=tokens, n_streams=n_streams)
        ctx.save_for_backward(p, support_words, mask)
        ctx.alpha = float(alpha)
        ctx.tokens = int(tokens)
        ctx.n_streams = int(n_streams)
        ctx.mark_non_differentiable(support_words)
        return p, support_words

    @staticmethod
    def backward(ctx, dp, _dsupport_words):
        if dp is None:
            return None, None, None, None, None
        p, support_words, mask = ctx.saved_tensors
        dp = dp.contiguous()
        rows, cols = p.shape
        block_rows = 32
        block_cols = triton.next_power_of_2(cols)
        dz = torch.empty_like(p)
        if rows == 0:
            return dz, None, None, None, None
        _production_bwd_kernel[(ctx.n_streams * triton.cdiv(ctx.tokens, block_rows),)](
            p, support_words, mask, dp, dp, dz,
            ctx.n_streams, ctx.tokens, triton.cdiv(ctx.tokens, block_rows), cols,
            ctx.tokens * p.stride(0), p.stride(0), p.stride(1),
            0, 0,
            support_words.stride(0), support_words.stride(1), support_words.stride(2),
            ctx.tokens * mask.stride(0), mask.stride(0), mask.stride(1),
            ctx.tokens * dp.stride(0), dp.stride(0), dp.stride(1),
            0, 0,
            ctx.tokens * dp.stride(0), dp.stride(0), dp.stride(1),
            0, 0,
            ctx.tokens * dz.stride(0), dz.stride(0), dz.stride(1),
            ALPHA=ctx.alpha, HAS_MASK=True, HAS_SECOND_DP=False, ACCUMULATE=False, TILE_MAJOR=False,
            BLOCK_ROWS=block_rows, BLOCK_COLS=block_cols,
        )
        return dz, None, None, None, None


def production_factor(z, activation="entmax", alpha=1.5, mask=None):
    """Return ``(values, packed_support_words)`` for an alpha=1.5 or alpha=2 CUDA routing solve.

    Logits and output factors are fp32. ``mask`` is a detached boolean candidate mask and is broadcast
    over leading dimensions. Support is packed over the penultimate (token) axis into signed ``int32``
    words whose bit pattern is uint32. Packed support describes factors that remain nonzero after the
    native bf16 route-storage cast; the returned fp32 factors retain the analytic solve values and are
    reused directly by the VJP.
    """
    if not z.is_cuda:
        raise RuntimeError("production_factor is CUDA-only; use the fp64 naive oracle on CPU")
    if z.dtype != torch.float32:
        raise TypeError(f"production_factor requires fp32 logits, got {z.dtype}")
    if z.dim() < 1 or z.shape[-1] < 1:
        raise ValueError(f"production_factor expects a non-empty branch axis, got {tuple(z.shape)}")
    if z.shape[-1] > MAX_BRANCH_WIDTH:
        raise ValueError(
            f"production_factor branch width must be <= {MAX_BRANCH_WIDTH}; "
            f"got {z.shape[-1]}")
    if activation not in _SPARSE_ACTIVATIONS:
        raise ValueError(f"production_factor requires entmax/sparsemax, got {activation!r}")
    alpha = _production_alpha(alpha)

    cols = z.shape[-1]
    if z.dim() == 1:
        tokens, n_streams, support_shape = 1, 1, (1, cols)
    else:
        tokens = z.shape[-2]
        n_streams = prod(z.shape[:-2])
        support_shape = (*z.shape[:-2], triton.cdiv(tokens, 32), cols)
    z_rows = z.reshape(n_streams * tokens, cols).contiguous()
    if mask is None:
        mask_rows = torch.ones_like(z_rows, dtype=torch.bool)
    else:
        mask_rows = mask.to(device=z.device, dtype=torch.bool).expand_as(z).reshape(-1, cols).contiguous()
    p, support_words = _ProductionFactorFn.apply(
        z_rows, mask_rows, float(alpha), tokens, n_streams)
    return p.reshape_as(z), support_words.reshape(support_shape)


def _arena_shape(logits: torch.Tensor) -> tuple[int, int]:
    if logits.ndim not in (3, 4):
        raise ValueError(
            "production routing tensors must be [BH, L, W] or [BH, L, D, W], got "
            f"{tuple(logits.shape)}")
    return logits.shape[0], logits.shape[1]


def _launch_factor_into(
    logits: torch.Tensor,
    output: torch.Tensor,
    output_second: torch.Tensor | None,
    support_words: torch.Tensor,
    alpha: float,
) -> None:
    n_streams, tokens = _arena_shape(logits)
    cols = logits.shape[-1]
    n_words = triton.cdiv(tokens, 32)
    _production_fwd_kernel[(n_streams * n_words,)](
        logits, output, output if output_second is None else output_second, support_words, logits,
        n_streams, tokens, n_words, cols,
        logits.stride(0), logits.stride(1), logits.stride(2),
        *_value_strides(output),
        *_value_strides(output if output_second is None else output_second),
        support_words.stride(0), support_words.stride(1), support_words.stride(2),
        logits.stride(0), logits.stride(1), logits.stride(2),
        ALPHA=alpha, HAS_MASK=False, WRITE_SECOND=output_second is not None,
        TILE_MAJOR=output.ndim == 4,
        BLOCK_ROWS=32, BLOCK_COLS=triton.next_power_of_2(cols),
    )


def _launch_factor_backward_into_logits(
    output: torch.Tensor,
    d_output: torch.Tensor,
    d_output_second: torch.Tensor | None,
    support_words: torch.Tensor,
    d_logits: torch.Tensor,
    alpha: float,
    *,
    accumulate: bool,
) -> None:
    n_streams, tokens = _arena_shape(d_logits)
    cols = d_logits.shape[-1]
    n_words = triton.cdiv(tokens, 32)
    p_stream, p_token, p_col, p_tile, p_lane = _value_strides(output)
    _production_bwd_kernel[(n_streams * n_words,)](
        output, support_words, output, d_output,
        d_output if d_output_second is None else d_output_second, d_logits,
        n_streams, tokens, n_words, cols,
        p_stream, p_token, p_col, p_tile, p_lane,
        support_words.stride(0), support_words.stride(1), support_words.stride(2),
        p_stream, p_token, p_col,
        *_value_strides(d_output),
        *_value_strides(d_output if d_output_second is None else d_output_second),
        d_logits.stride(0), d_logits.stride(1), d_logits.stride(2),
        ALPHA=alpha, HAS_MASK=False, HAS_SECOND_DP=d_output_second is not None,
        ACCUMULATE=accumulate, TILE_MAJOR=output.ndim == 4, BLOCK_ROWS=32,
        BLOCK_COLS=triton.next_power_of_2(cols),
    )


def _value_strides(values: torch.Tensor) -> tuple[int, int, int, int, int]:
    """``(stream, token, col, tile, lane)`` strides for either route-value layout.

    A tile-major arena slice is ``[BH, tiles, width, 32]`` and has no token stride of
    its own (the kernel reconstructs the token from ``tile``/``lane``); a dense level
    is ``[BH, T, width]`` and has no tile/lane strides. Deriving both from the tensor's
    own rank keeps one launch site per kernel instead of one per layout.
    """
    if values.ndim == 4:
        return (values.stride(0), 0, values.stride(2), values.stride(1), values.stride(3))
    if values.ndim == 3:
        return (values.stride(0), values.stride(1), values.stride(2), 0, 0)
    raise ValueError(f"route values must be rank 3 (dense) or rank 4 (tile-major), got {values.ndim}")


def _union_row_tile(block_cols: int) -> int:
    """Rows of one 32-token support word the union solve keeps in flight at once.

    The union forward kernel's SM cost is ``ROW_TILE * block_cols`` elements times the
    number of live fp32 tiles in the closed-form solve, in BOTH registers and the
    ``tl.sort`` shared scratch. Holding that product at ``_UNION_SOLVE_TILE`` keeps the
    solve's live set inside the register file (128 registers/thread, zero spilled bytes)
    and the sort scratch at 4 KB/CTA for every routing width, instead of letting both
    scale with the level's branch count -- which is what pinned the pre-p3 shape at
    ptxas's 255-register cap with 468 spilled bytes and 32.8 KB of scratch. This is a
    structural rule: the working set stays fixed as ``block_cols`` grows toward the
    wide-``nc`` regime, so it is not a per-width threshold.
    """
    return min(32, max(1, _UNION_SOLVE_TILE // block_cols))


def _launch_union_fwd(
    read_source: torch.Tensor,
    write_source: torch.Tensor,
    midpoint: torch.Tensor,
    support_words: torch.Tensor,
    read_output: torch.Tensor,
    write_output: torch.Tensor,
    alpha: float,
) -> None:
    n_streams, tokens = read_source.shape[0], read_source.shape[1]
    width = read_source.shape[-1]
    _production_union_fwd_kernel[(n_streams * triton.cdiv(tokens, 32),)](
        read_source, write_source, midpoint, support_words, read_output, write_output, read_source,
        n_streams, tokens, triton.cdiv(tokens, 32), width,
        read_source.stride(0), read_source.stride(1), read_source.stride(2),
        write_source.stride(0), write_source.stride(1), write_source.stride(2),
        midpoint.stride(0), midpoint.stride(1), midpoint.stride(2),
        *_value_strides(read_output),
        *_value_strides(write_output),
        support_words.stride(0), support_words.stride(1), support_words.stride(2),
        read_source.stride(0), read_source.stride(1), read_source.stride(2),
        ALPHA=alpha, HAS_MASK=False, TILE_MAJOR=read_output.ndim == 4, BLOCK_ROWS=32,
        ROW_TILE=_union_row_tile(triton.next_power_of_2(width)),
        BLOCK_COLS=triton.next_power_of_2(width),
    )


def _launch_union_bwd(
    read_logits_level: torch.Tensor,
    write_logits_level: torch.Tensor,
    midpoint: torch.Tensor,
    support_words: torch.Tensor,
    read_output: torch.Tensor,
    write_output: torch.Tensor,
    d_read_output: torch.Tensor,
    d_write_output: torch.Tensor,
    d_read_level: torch.Tensor,
    d_write_level: torch.Tensor,
    alpha: float,
) -> None:
    n_streams, tokens = read_logits_level.shape[0], read_logits_level.shape[1]
    width = read_logits_level.shape[-1]
    _production_union_bwd_kernel[(n_streams * triton.cdiv(tokens, 32),)](
        read_logits_level, write_logits_level, midpoint, support_words,
        read_logits_level, read_output, write_output,
        d_read_output, d_write_output, d_read_level, d_write_level,
        n_streams, tokens, triton.cdiv(tokens, 32), width,
        read_logits_level.stride(0), read_logits_level.stride(1), read_logits_level.stride(2),
        write_logits_level.stride(0), write_logits_level.stride(1), write_logits_level.stride(2),
        midpoint.stride(0), midpoint.stride(1), midpoint.stride(2),
        support_words.stride(0), support_words.stride(1), support_words.stride(2),
        read_logits_level.stride(0), read_logits_level.stride(1), read_logits_level.stride(2),
        *_value_strides(read_output),
        *_value_strides(write_output),
        *_value_strides(d_read_output),
        *_value_strides(d_write_output),
        d_read_level.stride(0), d_read_level.stride(1), d_read_level.stride(2),
        d_write_level.stride(0), d_write_level.stride(1), d_write_level.stride(2),
        ALPHA=alpha, HAS_MASK=False, TILE_MAJOR=read_output.ndim == 4, BLOCK_ROWS=32,
        BLOCK_COLS=triton.next_power_of_2(width),
    )


def _launch_softmax_into(
    logits: torch.Tensor,
    output: torch.Tensor,
    output_second: torch.Tensor | None,
) -> None:
    n_streams, tokens = _arena_shape(logits)
    cols = logits.shape[-1]
    n_words = triton.cdiv(tokens, 32)
    _production_softmax_fwd_kernel[(n_streams * n_words,)](
        logits, output, output if output_second is None else output_second,
        n_streams, tokens, n_words, cols,
        logits.stride(0), logits.stride(1), logits.stride(2),
        *_value_strides(output),
        *_value_strides(output if output_second is None else output_second),
        WRITE_SECOND=output_second is not None, TILE_MAJOR=output.ndim == 4,
        BLOCK_ROWS=32, BLOCK_COLS=triton.next_power_of_2(cols),
    )


def _launch_softmax_backward_into_logits(
    output: torch.Tensor,
    d_output: torch.Tensor,
    d_output_second: torch.Tensor | None,
    d_logits: torch.Tensor,
    *,
    accumulate: bool,
) -> None:
    n_streams, tokens = _arena_shape(d_logits)
    cols = d_logits.shape[-1]
    n_words = triton.cdiv(tokens, 32)
    _production_softmax_bwd_kernel[(n_streams * n_words,)](
        output, d_output, d_output if d_output_second is None else d_output_second, d_logits,
        n_streams, tokens, n_words, cols,
        *_value_strides(output),
        *_value_strides(d_output),
        *_value_strides(d_output if d_output_second is None else d_output_second),
        d_logits.stride(0), d_logits.stride(1), d_logits.stride(2),
        HAS_SECOND_DP=d_output_second is not None, ACCUMULATE=accumulate,
        TILE_MAJOR=output.ndim == 4,
        BLOCK_ROWS=32, BLOCK_COLS=triton.next_power_of_2(cols),
    )


class _ProductionRoutingArenaFn(torch.autograd.Function):
    """Own final bf16 arenas and retain their output references for routing VJPs.

    Union alone retains compact fp32 midpoint values and its original logits.
    """

    @staticmethod
    def forward(ctx, read_logits, write_logits, routing):
        n_streams, tokens = _arena_shape(read_logits)
        if read_logits.shape != write_logits.shape:
            raise ValueError("read/write routing logits must have identical shapes")
        if read_logits.dtype != torch.float32 or write_logits.dtype != torch.float32:
            raise TypeError("production routing arenas require fp32 logits")
        if not read_logits.is_cuda or not write_logits.is_cuda:
            raise RuntimeError("production routing arenas are CUDA-only")
        if read_logits.device != write_logits.device:
            raise ValueError("read/write routing logits must share one CUDA device")
        if read_logits.shape[-2:] != (routing.D, routing.real_width):
            raise ValueError("routing logits do not match the resolved routing layout")

        fully_shared = all(
            isinstance(level, IndependentRouting) and level.shares_solve for level in routing.levels)
        arena_shape = (
            n_streams,
            triton.cdiv(tokens, _ROUTE_TILE_SIZE_VALUE),
            routing.level_offsets[-1],
            _ROUTE_TILE_SIZE_VALUE,
        )
        read_arena = torch.zeros(arena_shape, dtype=torch.bfloat16, device=read_logits.device)
        write_arena = read_arena if fully_shared else torch.zeros_like(read_arena)
        support_words = torch.empty(
            (n_streams, triton.cdiv(tokens, _ROUTE_TILE_SIZE_VALUE),
             sum(width for width, offset in zip(routing.branches, routing.support_offsets) if offset >= 0)),
            dtype=torch.int32,
            device=read_logits.device,
        )
        union_midpoints: list[torch.Tensor] = []
        empty_launch = n_streams == 0 or tokens == 0

        for index, level in enumerate(routing.levels):
            width = routing.branches[index]
            read_source = read_logits[:, :, index, :width]
            write_source = write_logits[:, :, index, :width]
            offset = routing.level_offsets[index]
            read_output = read_arena[:, :, offset:offset + width, :]
            write_output = write_arena[:, :, offset:offset + width, :]
            support_offset = routing.support_offsets[index]

            if isinstance(level, UnionRouting):
                midpoint = torch.empty((n_streams, tokens, width), dtype=torch.float32, device=read_logits.device)
                union_midpoints.append(midpoint)
                if empty_launch:
                    continue
                words = support_words[:, :, support_offset:support_offset + width]
                _launch_union_fwd(
                    read_source, write_source, midpoint, words, read_output, write_output,
                    level.alpha)
                continue

            assert isinstance(level, IndependentRouting)
            if empty_launch:
                continue
            if level.shares_solve:
                if isinstance(level.write_activation, EntmaxActivation):
                    words = support_words[:, :, support_offset:support_offset + width]
                    _launch_factor_into(
                        write_source, read_output,
                        None if fully_shared else write_output, words, level.write_activation.alpha)
                else:
                    _launch_softmax_into(
                        write_source, read_output, None if fully_shared else write_output)
                continue

            for activation, source, output in (
                (level.read_activation, write_source if level.tied else read_source, read_output),
                (level.write_activation, write_source, write_output),
            ):
                if isinstance(activation, EntmaxActivation):
                    words = support_words[:, :, support_offset:support_offset + width]
                    _launch_factor_into(source, output, None, words, activation.alpha)
                else:
                    _launch_softmax_into(source, output, None)

        union_state = (read_logits, write_logits, *union_midpoints) if union_midpoints else ()
        ctx.save_for_backward(read_arena, write_arena, support_words, *union_state)
        ctx.routing = routing
        ctx.logits_shape = read_logits.shape
        ctx.union_count = len(union_midpoints)
        ctx.mark_non_differentiable(support_words)
        return read_arena, write_arena, support_words

    @staticmethod
    def backward(ctx, d_read, d_write, _d_support_words):
        saved = ctx.saved_tensors
        read_arena, write_arena, support_words = saved[:3]
        if ctx.union_count:
            read_logits, write_logits = saved[3:5]
            union_midpoints = saved[5:5 + ctx.union_count]
        else:
            read_logits = write_logits = None
            union_midpoints = ()
        routing = ctx.routing
        if d_read is None:
            d_read = torch.zeros_like(read_arena)
        if d_write is None:
            d_write = torch.zeros_like(write_arena)
        d_read_logits = torch.zeros(
            ctx.logits_shape, dtype=torch.float32, device=read_arena.device)
        d_write_logits = torch.zeros_like(d_read_logits)
        union_index = 0
        n_streams, tokens = ctx.logits_shape[:2]
        if n_streams == 0 or tokens == 0:
            return d_read_logits, d_write_logits, None

        for index, level in enumerate(routing.levels):
            width = routing.branches[index]
            offset = routing.level_offsets[index]
            read_output = read_arena[:, :, offset:offset + width, :]
            write_output = write_arena[:, :, offset:offset + width, :]
            d_read_output = d_read[:, :, offset:offset + width, :]
            d_write_output = d_write[:, :, offset:offset + width, :]
            d_read_level = d_read_logits[:, :, index, :width]
            d_write_level = d_write_logits[:, :, index, :width]
            support_offset = routing.support_offsets[index]

            if isinstance(level, UnionRouting):
                assert read_logits is not None and write_logits is not None
                midpoint = union_midpoints[union_index]
                union_index += 1
                words = support_words[:, :, support_offset:support_offset + width]
                _launch_union_bwd(
                    read_logits[:, :, index, :width], write_logits[:, :, index, :width],
                    midpoint, words, read_output, write_output,
                    d_read_output, d_write_output, d_read_level, d_write_level, level.alpha)
                continue

            assert isinstance(level, IndependentRouting)
            if level.shares_solve:
                if isinstance(level.write_activation, EntmaxActivation):
                    _launch_factor_backward_into_logits(
                        read_output, d_read_output, d_write_output,
                        support_words[:, :, support_offset:support_offset + width], d_write_level,
                        level.write_activation.alpha, accumulate=False)
                else:
                    _launch_softmax_backward_into_logits(
                        read_output, d_read_output, d_write_output, d_write_level, accumulate=False)
                continue

            first = True
            for activation, output, d_output, target in (
                (level.read_activation, read_output, d_read_output,
                 d_write_level if level.tied else d_read_level),
                (level.write_activation, write_output, d_write_output, d_write_level),
            ):
                if isinstance(activation, EntmaxActivation):
                    _launch_factor_backward_into_logits(
                        output, d_output, None,
                        support_words[:, :, support_offset:support_offset + width], target,
                        activation.alpha,
                        accumulate=level.tied and not first)
                else:
                    _launch_softmax_backward_into_logits(
                        output, d_output, None, target, accumulate=level.tied and not first)
                first = False
        return d_read_logits, d_write_logits, None


class _ProductionLevelsFn(torch.autograd.Function):
    """Solve a routing plan into per-level dense fp32 factors (V3's consumer contract).

    Structurally the same plan walk as ``_ProductionRoutingArenaFn`` -- the same
    kernels, the same shared/tied/union dispatch, the same packed support -- with the
    bf16 tile-major arena replaced by one dense ``[BH, T, width_l]`` fp32 tensor per
    level-side. V3's consumer (`v3_consumer.py`) and its fp64 oracle
    (`naive.naive_rola`) both take dense levels; the arena is the V2 chunk kernel's
    ABI, and routing through it would force a bf16 round trip that no V3 consumer asks
    for.

    Read and write are always distinct allocations, even where one solve serves both
    (``shares_solve``): the kernels already write a second output and accumulate a
    second incoming gradient, and returning one tensor twice from a single
    ``autograd.Function`` would alias two graph outputs.
    """

    @staticmethod
    def forward(ctx, read_logits, write_logits, routing):
        n_streams, tokens = _arena_shape(read_logits)
        device = read_logits.device
        support_words = torch.empty(
            (n_streams, triton.cdiv(tokens, _ROUTE_TILE_SIZE_VALUE),
             sum(width for width, offset in zip(routing.branches, routing.support_offsets) if offset >= 0)),
            dtype=torch.int32,
            device=device,
        )
        empty_launch = n_streams == 0 or tokens == 0
        read_levels: list[torch.Tensor] = []
        write_levels: list[torch.Tensor] = []
        union_midpoints: list[torch.Tensor] = []

        for index, level in enumerate(routing.levels):
            width = routing.branches[index]
            read_source = read_logits[:, :, index, :width]
            write_source = write_logits[:, :, index, :width]
            read_output = torch.zeros((n_streams, tokens, width), dtype=torch.float32, device=device)
            write_output = torch.zeros_like(read_output)
            read_levels.append(read_output)
            write_levels.append(write_output)
            support_offset = routing.support_offsets[index]
            words = None if support_offset < 0 else support_words[
                :, :, support_offset:support_offset + width]

            if isinstance(level, UnionRouting):
                midpoint = torch.empty((n_streams, tokens, width), dtype=torch.float32, device=device)
                union_midpoints.append(midpoint)
                if empty_launch:
                    continue
                _launch_union_fwd(
                    read_source, write_source, midpoint, words, read_output, write_output,
                    level.alpha)
                continue

            assert isinstance(level, IndependentRouting)
            if empty_launch:
                continue
            if level.shares_solve:
                if isinstance(level.write_activation, EntmaxActivation):
                    _launch_factor_into(
                        write_source, read_output, write_output, words,
                        level.write_activation.alpha)
                else:
                    _launch_softmax_into(write_source, read_output, write_output)
                continue

            for activation, source, output in (
                (level.read_activation, write_source if level.tied else read_source, read_output),
                (level.write_activation, write_source, write_output),
            ):
                if isinstance(activation, EntmaxActivation):
                    _launch_factor_into(source, output, None, words, activation.alpha)
                else:
                    _launch_softmax_into(source, output, None)

        union_state = (read_logits, write_logits, *union_midpoints) if union_midpoints else ()
        ctx.save_for_backward(*read_levels, *write_levels, support_words, *union_state)
        ctx.routing = routing
        ctx.logits_shape = read_logits.shape
        ctx.union_count = len(union_midpoints)
        return (*read_levels, *write_levels)

    @staticmethod
    def backward(ctx, *grads):
        routing = ctx.routing
        D = routing.D
        saved = ctx.saved_tensors
        read_levels = saved[:D]
        write_levels = saved[D:2 * D]
        support_words = saved[2 * D]
        if ctx.union_count:
            read_logits, write_logits = saved[2 * D + 1:2 * D + 3]
            union_midpoints = saved[2 * D + 3:]
        else:
            read_logits = write_logits = None
            union_midpoints = ()

        d_read_logits = torch.zeros(
            ctx.logits_shape, dtype=torch.float32, device=support_words.device)
        d_write_logits = torch.zeros_like(d_read_logits)
        n_streams, tokens = ctx.logits_shape[:2]
        if n_streams == 0 or tokens == 0:
            return d_read_logits, d_write_logits, None

        union_index = 0
        for index, level in enumerate(routing.levels):
            width = routing.branches[index]
            read_output = read_levels[index]
            write_output = write_levels[index]
            d_read_output = grads[index]
            d_write_output = grads[D + index]
            if d_read_output is None:
                d_read_output = torch.zeros_like(read_output)
            if d_write_output is None:
                d_write_output = torch.zeros_like(write_output)
            d_read_output = d_read_output.contiguous()
            d_write_output = d_write_output.contiguous()
            d_read_level = d_read_logits[:, :, index, :width]
            d_write_level = d_write_logits[:, :, index, :width]
            support_offset = routing.support_offsets[index]
            words = None if support_offset < 0 else support_words[
                :, :, support_offset:support_offset + width]

            if isinstance(level, UnionRouting):
                assert read_logits is not None and write_logits is not None
                midpoint = union_midpoints[union_index]
                union_index += 1
                _launch_union_bwd(
                    read_logits[:, :, index, :width], write_logits[:, :, index, :width],
                    midpoint, words, read_output, write_output,
                    d_read_output, d_write_output, d_read_level, d_write_level, level.alpha)
                continue

            assert isinstance(level, IndependentRouting)
            if level.shares_solve:
                if isinstance(level.write_activation, EntmaxActivation):
                    _launch_factor_backward_into_logits(
                        read_output, d_read_output, d_write_output, words, d_write_level,
                        level.write_activation.alpha, accumulate=False)
                else:
                    _launch_softmax_backward_into_logits(
                        read_output, d_read_output, d_write_output, d_write_level, accumulate=False)
                continue

            first = True
            for activation, output, d_output, target in (
                (level.read_activation, read_output, d_read_output,
                 d_write_level if level.tied else d_read_level),
                (level.write_activation, write_output, d_write_output, d_write_level),
            ):
                if isinstance(activation, EntmaxActivation):
                    _launch_factor_backward_into_logits(
                        output, d_output, None, words, target, activation.alpha,
                        accumulate=level.tied and not first)
                else:
                    _launch_softmax_backward_into_logits(
                        output, d_output, None, target, accumulate=level.tied and not first)
                first = False
        return d_read_logits, d_write_logits, None


# ---------------------------------------------------------------------------
# Torch lowering of the SAME exact solve, for the domains Triton does not cover
# (CPU, and fp64 -- the oracle-equivalence gate projects fp64 logits end to end).
#
# This is a second *lowering* of the production semantics, not a fallback to the
# reference: it computes the identical sorted closed form the Triton kernels do
# (row-max shift, descending sort, prefix/prefix-square scans, greatest valid k)
# with the identical analytic fixed-alpha VJP. The fp64 bisection in
# ``entmax/reference.py`` is reachable from no runtime path at all -- it is the
# test oracle, and the point of an oracle is that nothing in production shares
# its code.
# ---------------------------------------------------------------------------


def _sorted_solve(z: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact ``(values, support)`` for alpha=1.5/2.0 by the sorted closed form."""
    width = z.shape[-1]
    z = z - z.amax(dim=-1, keepdim=True)
    sorted_z, _ = torch.sort(z, dim=-1, descending=True)
    prefix = sorted_z.cumsum(dim=-1)
    k = torch.arange(1, width + 1, device=z.device, dtype=z.dtype)
    if alpha == 2.0:
        tau_candidates = (prefix - 1.0) / k
        candidate = sorted_z > tau_candidates
    else:
        prefix_sq = (sorted_z * sorted_z).cumsum(dim=-1)
        discriminant = prefix * prefix - k * (prefix_sq - 4.0)
        tau_candidates = (prefix - discriminant.clamp(min=0.0).sqrt()) / k
        candidate = (discriminant >= 0.0) & (sorted_z > tau_candidates)
    columns = torch.arange(width, device=z.device)
    chosen = torch.where(candidate, columns, torch.full_like(columns, -1)).amax(dim=-1, keepdim=True)
    tau = torch.gather(tau_candidates, -1, chosen.clamp(min=0))
    tau = torch.where(chosen >= 0, tau, torch.zeros_like(tau))
    support = z > tau
    base = ((alpha - 1.0) * (z - tau)).clamp(min=0.0)
    values = torch.where(support, base if alpha == 2.0 else base * base, torch.zeros_like(base))
    return values, support


class _SortedEntmaxFn(torch.autograd.Function):
    """Sorted exact solve with the analytic sparse fixed-alpha Jacobian-vector product."""

    @staticmethod
    def forward(ctx, z, alpha):
        values, support = _sorted_solve(z, float(alpha))
        s = torch.where(
            support, values.clamp(min=0.0).pow(2.0 - float(alpha)), torch.zeros_like(values))
        ctx.save_for_backward(s)
        return values

    @staticmethod
    def backward(ctx, dp):
        (s,) = ctx.saved_tensors
        num = (s * dp).sum(dim=-1, keepdim=True)
        den = s.sum(dim=-1, keepdim=True)
        den = torch.where(den > 0, den, torch.ones_like(den))
        return s * (dp - num / den), None


def _overflow_free_delta(read_logits: torch.Tensor, write_logits: torch.Tensor) -> torch.Tensor:
    """``read - write`` with the union kernel's saturating magnitude arithmetic.

    Opposite-sign finite extrema overflow a raw subtraction while their midpoint stays
    representable, so the kernel clamps the magnitude instead. Mirrored here so the two
    lowerings agree on the same near-max fixtures the CUDA gate already covers.
    """
    finfo = torch.finfo(read_logits.dtype)
    read_negative = read_logits < 0.0
    opposite_sign = read_negative != (write_logits < 0.0)
    read_magnitude, write_magnitude = read_logits.abs(), write_logits.abs()
    hi = torch.maximum(read_magnitude, write_magnitude)
    lo = torch.minimum(read_magnitude, write_magnitude)
    opposite_magnitude = hi + torch.minimum(lo, finfo.max - hi)
    opposite_delta = torch.where(read_negative, -opposite_magnitude, opposite_magnitude)
    zero = torch.zeros_like(read_logits)
    same_delta = torch.where(opposite_sign, zero, read_logits) - torch.where(
        opposite_sign, zero, write_logits)
    return torch.where(opposite_sign, opposite_delta, same_delta)


def _torch_union_split(
    read_logits: torch.Tensor, write_logits: torch.Tensor, alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = 0.5 * read_logits + 0.5 * write_logits
    values = _SortedEntmaxFn.apply(z, alpha)
    # Support comes from the same `z > tau` bits `_sorted_solve` (and the Triton union
    # kernel's `semantic_support = valid & (z > tau)`) compute, not from `values > 0`: the
    # two conventions agree everywhere except a genuinely-supported member whose amplitude
    # underflows fp32 to exactly 0 (alpha=1.5 squares `(z-tau)/2`, so `z-tau ~ 1e-25`
    # suffices) -- adversarial review Claim 2. Recomputed directly (no grad needed; support
    # is a discrete, non-differentiable mask) rather than threaded out of the autograd
    # Function, so the two lowerings share one support definition instead of two that
    # happen to usually agree.
    with torch.no_grad():
        _, support = _sorted_solve(z.detach(), alpha)
    delta = _overflow_free_delta(read_logits, write_logits)
    read = values * torch.sigmoid(delta)
    safe = torch.where(support, values, torch.ones_like(values))
    log_write = (safe.log() + torch.nn.functional.logsigmoid(-delta)).masked_fill(
        ~support, float("-inf"))
    has_support = support.any(dim=-1, keepdim=True)
    write = torch.softmax(
        torch.where(has_support, log_write, torch.zeros_like(log_write)), dim=-1)
    return read, torch.where(has_support & support, write, torch.zeros_like(write))


def _torch_activation(logits: torch.Tensor, activation) -> torch.Tensor:
    if isinstance(activation, SoftmaxActivation):
        return torch.softmax(logits, dim=-1)
    return _SortedEntmaxFn.apply(logits, activation.alpha)


def _torch_routing_factor_levels(
    read_logits: torch.Tensor, write_logits: torch.Tensor, routing: ResolvedRouting,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    reads, writes = [], []
    for index, level in enumerate(routing.levels):
        width = routing.branches[index]
        read_level = read_logits[..., index, :width]
        write_level = write_logits[..., index, :width]
        if isinstance(level, UnionRouting):
            read_factor, write_factor = _torch_union_split(read_level, write_level, level.alpha)
        elif level.shares_solve:
            write_factor = _torch_activation(write_level, level.write_activation)
            read_factor = write_factor
        elif level.tied:
            read_factor = _torch_activation(write_level, level.read_activation)
            write_factor = _torch_activation(write_level, level.write_activation)
        else:
            read_factor = _torch_activation(read_level, level.read_activation)
            write_factor = _torch_activation(write_level, level.write_activation)
        reads.append(read_factor)
        writes.append(write_factor)
    return tuple(reads), tuple(writes)


def production_routing_factor_levels(
    read_logits: torch.Tensor,
    write_logits: torch.Tensor,
    routing: ResolvedRouting,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Exact per-level ``(read, write)`` routing factors on the production solve.

    Returns one ``[BH, T, width_l]`` tensor per level-side -- the shape V3's producer
    hands ``naive_rola`` and ``build_schedule``. Lowered to the fused Triton
    kernels on CUDA fp32 and to the identical sorted closed form in torch elsewhere;
    the two agree by construction rather than by tolerance, since the threshold is a
    closed form and not an iterate.
    """
    if not isinstance(routing, ResolvedRouting):
        raise TypeError(f"routing must be ResolvedRouting, got {type(routing).__name__}")
    if read_logits.shape != write_logits.shape:
        raise ValueError(
            "read/write logits must have identical shapes, got "
            f"{read_logits.shape} and {write_logits.shape}")
    if read_logits.ndim != 4 or read_logits.shape[-2:] != (routing.D, routing.real_width):
        raise ValueError(
            "routing logits must end in "
            f"[D, real_width]=[{routing.D}, {routing.real_width}], got {tuple(read_logits.shape)}")
    if read_logits.device != write_logits.device or read_logits.dtype != write_logits.dtype:
        raise ValueError("read/write logits must share one device and dtype")

    if read_logits.is_cuda and read_logits.dtype == torch.float32:
        outputs = _ProductionLevelsFn.apply(
            read_logits.contiguous(), write_logits.contiguous(), routing)
        return tuple(outputs[:routing.D]), tuple(outputs[routing.D:])
    return _torch_routing_factor_levels(read_logits, write_logits, routing)


def production_routing_factors(
    read_logits: torch.Tensor,
    write_logits: torch.Tensor,
    routing: ResolvedRouting,
    *,
    token_origin: int = 0,
) -> RouteDescriptor:
    """Solve a canonical routing plan directly into the RouteDescriptor ABI."""
    if not isinstance(routing, ResolvedRouting):
        raise TypeError(f"routing must be ResolvedRouting, got {type(routing).__name__}")
    if read_logits.shape != write_logits.shape:
        raise ValueError(
            "read/write logits must have identical shapes, got "
            f"{read_logits.shape} and {write_logits.shape}")
    if type(token_origin) is not int or token_origin < 0:
        raise ValueError("token_origin must be a nonnegative exact int")
    if read_logits.ndim != 4 or read_logits.shape[-2:] != (routing.D, routing.real_width):
        raise ValueError(
            "routing logits must end in "
            f"[D, real_width]=[{routing.D}, {routing.real_width}], got {tuple(read_logits.shape)}")

    outputs = _ProductionRoutingArenaFn.apply(read_logits, write_logits, routing)
    read_values, write_values, support_words = outputs
    return RouteDescriptor(
        read_values, write_values, support_words,
        routing.level_offsets, routing.support_offsets, routing.branches,
        routing.radix_strides, routing.support_side_flags,
        token_origin, read_logits.shape[1], routing.topology_stamp,
    )


__all__ = [
    "production_factor",
    "production_routing_factor_levels",
    "production_routing_factors",
]
