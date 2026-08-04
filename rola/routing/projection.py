"""Packed router projection for a :class:`ResolvedRouting` plan."""

from __future__ import annotations

import torch

from rola.routing.types import ResolvedRouting


def _packed_router_logits(
    h: torch.Tensor,
    route_W: torch.Tensor,
    routing: ResolvedRouting,
    *,
    route_bias: torch.Tensor | None = None,
    h_w: torch.Tensor | None = None,
    projection_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project packed router weights in one explicitly selected arithmetic domain.

    Returns ``(read_logits, write_logits)``, each ``[B*H, T, routing.packed_logit_width]``:
    one side's levels laid out end to end on ``routing.level_offsets``, so level ``l`` is
    ``logits[:, :, routing.level_logit_slice(l)]``. There is no per-level padding to
    ``real_width`` -- see ``ResolvedRouting.packed_logit_width``.
    """
    if not isinstance(routing, ResolvedRouting):
        raise TypeError(f"routing must be ResolvedRouting, got {type(routing).__name__}")
    heads = routing.validate_packed_router(route_W, route_bias)

    def validate_hidden(hidden: torch.Tensor, name: str) -> None:
        if not isinstance(hidden, torch.Tensor) or hidden.ndim not in (3, 4):
            shape = getattr(hidden, "shape", None)
            raise ValueError(f"{name} must have rank 3 or 4, got {shape}")
        hidden_width = hidden.shape[-1]
        if hidden_width != route_W.shape[2]:
            raise ValueError(
                f"{name} hidden width must match route_W hidden width {route_W.shape[2]}, "
                f"got {hidden_width}")
        if hidden.ndim == 4 and hidden.shape[2] != heads:
            raise ValueError(
                f"{name} rank-4 head count must match route_W heads {heads}, "
                f"got {hidden.shape[2]}")

    validate_hidden(h, "h")
    if h_w is not None:
        validate_hidden(h_w, "h_w")
        if h_w.shape != h.shape or h_w.device != h.device:
            raise ValueError("h_w must match h shape and device")

    # PACK THE WEIGHT, NOT THE OUTPUT. `route_W` stores every slot padded to
    # `real_width`, but level `l` only ever has `branches[l]` real columns -- the rest
    # are computed by the GEMM, written to the logit tensor, and never read (their
    # gradients are identically zero, so they are dead parameters too). Gathering each
    # side's real column blocks into one contiguous `[H, hidden, packed_logit_width]`
    # weight makes the GEMM compute exactly the consumed columns. The gather is
    # L-INVARIANT (measured 0.036 ms/step at the parity cell: 16.0 MiB read -> 10.0 MiB
    # written) while the work it removes scales with L, so it is free at scale and the
    # margin only widens.
    #
    # BIT-IDENTICAL, and that is the whole safety argument: a column of the output is a
    # dot product over the full hidden dim and is unaffected by which OTHER columns share
    # the GEMM. Verified `torch.equal` on the consumed columns against the padded einsum
    # at L=2048/16384 x seeds 0/5150/6161 (0 of 83,886,080 words differing). So
    # `test_projection_tf32_support_gate.py` -- the gate that exists because effd23c
    # moved this GEMM's arithmetic and flipped 14-95 entmax support entries -- sees
    # byte-for-byte the logits it sees today, as do the fp64 oracle gates.
    def pack_weight(slots: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """One side's per-level column blocks, tightly packed on `level_offsets`."""
        weight = torch.cat(
            [route_W[:, slot, :, :width] for slot, width in zip(slots, routing.branches)], dim=2)
        if route_bias is None:
            return weight, None
        bias = torch.cat(
            [route_bias[:, slot, :width] for slot, width in zip(slots, routing.branches)], dim=1)
        return weight, bias

    def project(hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        equation = "bld,hdc->bhlc" if hidden.ndim == 3 else "blhd,hdc->bhlc"
        logits = torch.einsum(equation, hidden.to(projection_dtype), weight.to(projection_dtype))
        if bias is not None:
            logits = logits + bias.to(projection_dtype)[None, :, None, :]
        return logits.reshape(hidden.shape[0] * heads, hidden.shape[1], weight.shape[2])

    write_slots = tuple(range(routing.D))
    # `packed_read_slot_lookup` already resolves "which slot feeds this level's READ
    # logits" for tied, untied and mixed plans, so the three-branch slot arithmetic this
    # function used to carry collapses into one lookup. Fully tied is exactly
    # `read_slots == write_slots`, stated structurally rather than as `not untied_levels`.
    read_slots = routing.packed_read_slot_lookup
    fully_tied = read_slots == write_slots

    write_weight, write_bias = pack_weight(write_slots)
    if fully_tied:
        # ONE solve serves both duties, and callers (the budget-conditioning path, the
        # arena's `fully_shared`) rely on getting the SAME TENSOR OBJECT back twice.
        tied_logits = project(h if h_w is None else h_w, write_weight, write_bias)
        return tied_logits, tied_logits

    if h_w is None:
        # One stream, so both sides are columns of ONE GEMM -- the same single launch the
        # padded path used, now 640 columns wide instead of 1024 at the parity cell.
        read_weight, read_bias = pack_weight(read_slots)
        both = project(
            h,
            torch.cat([write_weight, read_weight], dim=2),
            None if route_bias is None else torch.cat([write_bias, read_bias], dim=1),
        )
        width = routing.packed_logit_width
        return both[:, :, width:], both[:, :, :width]

    # Split read/write streams: the write side is projected from `h_w`, and a TIED level
    # takes its read logits from that write projection (the write stream is its source),
    # so only the untied levels are projected from `h`.
    write_logits = project(h_w, write_weight, write_bias)
    untied = routing.untied_levels
    if len(untied) == routing.D:
        read_weight, read_bias = pack_weight(read_slots)
        return project(h, read_weight, read_bias), write_logits
    untied_weight = torch.cat(
        [route_W[:, read_slots[level], :, :routing.branches[level]] for level in untied], dim=2)
    untied_bias = None if route_bias is None else torch.cat(
        [route_bias[:, read_slots[level], :routing.branches[level]] for level in untied], dim=1)
    untied_logits = project(h, untied_weight, untied_bias)
    untied_offsets = {}
    cursor = 0
    for level in untied:
        untied_offsets[level] = cursor
        cursor += routing.branches[level]
    read_logits = torch.cat(
        [
            untied_logits[:, :, untied_offsets[level]:untied_offsets[level] + routing.branches[level]]
            if level in untied_offsets
            else write_logits[:, :, routing.level_logit_slice(level)]
            for level in range(routing.D)
        ],
        dim=2,
    )
    return read_logits, write_logits


def packed_router_logits(
    h: torch.Tensor,
    route_W: torch.Tensor,
    routing: ResolvedRouting,
    *,
    route_bias: torch.Tensor | None = None,
    h_w: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project production router logits in fp32."""
    return _packed_router_logits(
        h, route_W, routing, route_bias=route_bias, h_w=h_w,
        projection_dtype=torch.float32)


def packed_router_logits_reference(
    h: torch.Tensor,
    route_W: torch.Tensor,
    routing: ResolvedRouting,
    *,
    route_bias: torch.Tensor | None = None,
    h_w: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project canonical reference router logits in fp64."""
    return _packed_router_logits(
        h, route_W, routing, route_bias=route_bias, h_w=h_w,
        projection_dtype=torch.float64)
