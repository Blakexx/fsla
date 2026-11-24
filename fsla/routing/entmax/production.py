"""Production routing factors: the exact sorted alpha=1.5/2.0 solve and its analytic VJP.

This module is deliberately separate from ``routing.entmax.reference``. The reference is an fp64
bisection oracle; this module owns the value path every runtime caller takes, and its analytic
fixed-alpha VJP. Its plan-level surface is ``production_routing_levels``: dense per-level
``[BH, T, width_l]`` factors, the contract the consumer and its fp64 oracle both consume. This is what
the producer calls.

It runs through two lowerings of one semantics: hand-CUDA on bf16 CUDA tensors, and the identical
sorted closed form in torch for the domains the kernels do not cover (CPU, fp64). Neither surface
ever calls the reference; it is the test oracle, not a fallback.

THE KERNELS ARE `csrc/fsla/src/entmax/` -- `factor.cu` for the
independent per-side solves and the softmax pair. There is no Triton in this package and no torch
fallback on CUDA: `fsla.ops._ext` raises loudly if the wheel is missing
(docs/internals/entmax/factor.md).
"""

from __future__ import annotations

from functools import cache

import torch

from fsla.routing.types import (
    ROUTE_TILE_SIZE as _ROUTE_TILE_SIZE_VALUE,
)
from fsla.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    ResolvedRouting,
    SoftmaxActivation,
    TiedRouting,
)


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _extension():
    # NO FALLBACK, by policy: `fsla.ops._ext` raises loudly if the wheel is missing.
    from fsla.ops._ext import extension

    return extension()


def _one(width: int) -> list[int]:
    """The single-level table a one-level launch takes: offsets are already in the
    tensors' own `data_ptr`, because every caller here hands a level SLICE."""
    return [width]


def _arena_shape(logits: torch.Tensor) -> tuple[int, int]:
    """``(streams, tokens)`` for either logit layout.

    Rank 3 is the flat ``[BH, T, W]`` plane; rank 4 is the TOKEN-MAJOR ``[B, T, H, W]``
    the packed router GEMM writes directly, whose stream ``b*H + h`` is two axes. Which
    one arrived is `_heads`' question, and the solve carries the answer as a stride pair
    rather than as a copy.
    """
    if logits.ndim == 4:
        return logits.shape[0] * logits.shape[2], logits.shape[1]
    if logits.ndim != 3:
        raise ValueError(
            f"production routing logits must be [BH, T, W] or [B, T, H, W], got "
            f"{tuple(logits.shape)}")
    return logits.shape[0], logits.shape[1]


def _heads(logits: torch.Tensor) -> int:
    """The STREAM SPLIT the launch runs under: ``H`` for token-major logits, 1 for flat.

    NEVER 0, and that is the kernel's constraint rather than a convention: the split is
    unconditional there because guarding it cost the family its register budget, so
    ``heads = 1`` -- one head, ``sb = stream``, ``sh = 0`` -- is how a flat plane says it
    has no head axis (docs/internals/entmax/entmax.md#stream-split).
    """
    return logits.shape[2] if logits.ndim == 4 else 1


# ---------------------------------------------------------------------------
# THE LAUNCH PLAN -- one launch per (operator, alpha, width class, operand set),
# which is what makes `producer.py`'s "one batched solve" true rather than aspirational.
# ---------------------------------------------------------------------------

@cache
def _max_batched_levels() -> int:
    """The levels one launch's table holds: the matcher's bound (`common/plan/entmax_launch.h`), read from the extension. A topology deeper than this
    issues a second launch of the same arm, not a wider table."""
    return _extension().entmax_max_levels()


@cache
def _width_class(width: int) -> int:
    """The PADDED width, which is what selects the template arm: the matcher's answer (`common/plan/entmax_launch.h`), read from the extension.

    Two levels may share a launch only if they share this. `LW`/`IPT` set how the sort
    permutes and how the prefix scan associates, so running a narrow level in a wider
    arm would round `tau` differently and could move a support boundary -- the batch has
    to be the loop, not an approximation of it.
    """
    return _extension().entmax_width_class(int(width))


class _Launch:
    """The levels of ONE kernel launch: one operator, one alpha, one width class, one
    set of base tensors. Offsets are COLUMN indices into those bases, which is the unit
    the kernel's address expression adds them in."""

    __slots__ = ("kind", "alpha", "operands", "heads",
                 "logit", "value", "support", "widths")

    def __init__(self, kind, alpha, operands, heads):
        self.kind, self.alpha, self.operands, self.heads = kind, alpha, operands, heads
        self.logit, self.value, self.support, self.widths = [], [], [], []

    def full(self) -> bool:
        return len(self.widths) == _max_batched_levels()

    def add(self, width, logit_off, value_off, support_off) -> None:
        self.widths.append(int(width))
        self.logit.append(int(logit_off))
        self.value.append(int(value_off))
        self.support.append(int(support_off))

    def issue(self) -> None:
        extension = _extension()
        if self.kind == "entmax":
            logits, values, values_second, support = self.operands
            extension.entmax_factor_forward(
                logits, values, values_second, support,
                self.logit, self.value, self.support, self.widths, self.heads, self.alpha)
        else:
            logits, values, values_second = self.operands
            extension.routing_softmax_forward(
                logits, values, values_second, self.logit, self.value, self.widths, self.heads)


class _Plan:
    """The forward walk's launches, grouped by arm and issued in FIRST-APPEARANCE order.

    That order is what makes the grouping safe rather than merely faster. In the forward
    every level owns disjoint output columns, so there is no write hazard to preserve at
    all; the order is kept because a reader should be able to read the launch sequence
    off the plan walk.
    """

    def __init__(self):
        self._open: dict = {}
        self._issued: list[_Launch] = []

    def add(self, kind, alpha, operands, width, *, logit_off,
            value_off=0, support_off=0, heads=1):
        key = (kind, alpha, _width_class(width), heads, *(id(t) for t in operands))
        launch = self._open.get(key)
        if launch is None or launch.full():
            launch = _Launch(kind, alpha, operands, heads)
            self._open[key] = launch
            self._issued.append(launch)
        launch.add(width, logit_off, value_off, support_off)

    def issue(self) -> int:
        for launch in self._issued:
            launch.issue()
        return len(self._issued)


# ---------------------------------------------------------------------------
# THE BACKWARD stays one launch per level-side, and that is a MEASURED shape, not an
# oversight: autograd delivers D separate cotangent tensors with no common base, so
# batching them would mean copying D of them into one plane to save D-1 launches.
# ---------------------------------------------------------------------------


def _launch_factor_backward_into_logits(
    output: torch.Tensor,
    d_output: torch.Tensor,
    d_output_second: torch.Tensor | None,
    support_words: torch.Tensor,
    d_logits: torch.Tensor,
    alpha: float,
) -> None:
    _extension().entmax_factor_backward(
        output, support_words, d_output, d_output_second, d_logits,
        [0], [0], [0], _one(d_logits.shape[-1]), _heads(d_logits),
        float(alpha))


def _launch_softmax_backward_into_logits(
    output: torch.Tensor,
    d_output: torch.Tensor,
    d_output_second: torch.Tensor | None,
    d_logits: torch.Tensor,
) -> None:
    _extension().routing_softmax_backward(
        output, d_output, d_output_second, d_logits,
        [0], [0], _one(d_logits.shape[-1]), _heads(d_logits))


class _ProductionLevelsFn(torch.autograd.Function):
    """Solve a routing plan into per-level dense ``[BH, T, width_l]`` factors.

    The plan walk -- the same
    kernels, the same tied/independent dispatch, the same packed support -- against one
    dense plane per side rather than the tile-major arena the earlier chunk ABI took.

    The CUDA solve writes the STORED form (docs/internals/entmax/factor.md#stored-form):
    ``bfloat16`` is what the consumer reads and therefore what the producer emits, and
    the solve writes it directly rather than through a narrowing pass.

    Read and write are always distinct allocations, even where one solve serves both
    (a ``TiedRouting`` level): the kernels already write a second output and accumulate
    a second incoming gradient, and returning one tensor twice from a single
    ``autograd.Function`` would alias two graph outputs.
    """

    @staticmethod
    def forward(ctx, read_logits, write_logits, routing):
        n_streams, tokens = _arena_shape(read_logits)
        device = read_logits.device
        support_words = torch.empty(
            (n_streams, _cdiv(tokens, _ROUTE_TILE_SIZE_VALUE),
             sum(width for width, offset in zip(routing.branches, routing.support_offsets) if offset >= 0)),
            dtype=torch.int32,
            device=device,
        )
        empty_launch = n_streams == 0 or tokens == 0
        heads = _heads(read_logits)
        #: ONE PLANE PER SIDE, sliced into levels. Each level is still a distinct graph
        #: output, at its own columns of one allocation -- which is what lets the levels
        #: of a width class share a launch
        #: (docs/internals/entmax/factor.md#batched-levels), and what makes the whole
        #: side ONE contiguous tensor in the layout the consumer reads.
        #:
        #: UNINITIALIZED: a dense level is exactly the tokens, and every solve writes
        #: every element of one (unlike the tile-major arena, whose tail lanes past
        #: `tokens` no kernel touches).
        packed_width = routing.level_offsets[-1]
        read_plane = torch.empty(
            (n_streams, tokens, packed_width), dtype=torch.bfloat16, device=device)
        write_plane = torch.empty_like(read_plane)

        read_levels: list[torch.Tensor] = []
        write_levels: list[torch.Tensor] = []
        plan = _Plan()

        for index, level in enumerate(routing.levels):
            width = routing.branches[index]
            logit_offset = routing.level_logit_slice(index).start
            offset = routing.level_offsets[index]
            read_levels.append(read_plane[:, :, offset:offset + width])
            write_levels.append(write_plane[:, :, offset:offset + width])
            support_offset = routing.support_offsets[index]

            assert isinstance(level, (IndependentRouting, TiedRouting))
            if empty_launch:
                continue
            if isinstance(level, TiedRouting):
                shared_activation = level.activation("write")
                if isinstance(shared_activation, EntmaxActivation):
                    plan.add("entmax", shared_activation.alpha,
                             (write_logits, read_plane, write_plane, support_words), width,
                             logit_off=logit_offset, value_off=offset,
                             support_off=support_offset, heads=heads)
                else:
                    plan.add("softmax", 0.0, (write_logits, read_plane, write_plane), width,
                             logit_off=logit_offset, value_off=offset, heads=heads)
                continue

            for activation, source, output in (
                (level.read, read_logits, read_plane),
                (level.write, write_logits, write_plane),
            ):
                if isinstance(activation, EntmaxActivation):
                    plan.add("entmax", activation.alpha, (source, output, None, support_words),
                             width, logit_off=logit_offset, value_off=offset,
                             support_off=support_offset, heads=heads)
                else:
                    plan.add("softmax", 0.0, (source, output, None), width,
                             logit_off=logit_offset, value_off=offset, heads=heads)

        plan.issue()

        ctx.save_for_backward(*read_levels, *write_levels, support_words)
        ctx.routing = routing
        ctx.logits_shape = read_logits.shape
        ctx.logits_dtype = read_logits.dtype
        ctx.mark_non_differentiable(support_words)
        return (*read_levels, *write_levels, support_words)

    @staticmethod
    def backward(ctx, *grads):
        routing = ctx.routing
        D = routing.D
        saved = ctx.saved_tensors
        read_levels = saved[:D]
        write_levels = saved[D:2 * D]
        support_words = saved[2 * D]

        d_read_logits = torch.zeros(
            ctx.logits_shape, dtype=ctx.logits_dtype, device=support_words.device)
        d_write_logits = torch.zeros_like(d_read_logits)
        n_streams, tokens = ctx.logits_shape[:2]
        if n_streams == 0 or tokens == 0:
            return d_read_logits, d_write_logits, None

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
            span = routing.level_logit_slice(index)
            d_read_level = d_read_logits[..., span]
            d_write_level = d_write_logits[..., span]
            support_offset = routing.support_offsets[index]
            words = None if support_offset < 0 else support_words[
                :, :, support_offset:support_offset + width]

            assert isinstance(level, (IndependentRouting, TiedRouting))
            if isinstance(level, TiedRouting):
                shared_activation = level.activation("write")
                if isinstance(shared_activation, EntmaxActivation):
                    _launch_factor_backward_into_logits(
                        read_output, d_read_output, d_write_output, words, d_write_level,
                        shared_activation.alpha)
                else:
                    _launch_softmax_backward_into_logits(
                        read_output, d_read_output, d_write_output, d_write_level)
                continue

            for activation, output, d_output, target in (
                (level.read, read_output, d_read_output, d_read_level),
                (level.write, write_output, d_write_output, d_write_level),
            ):
                if isinstance(activation, EntmaxActivation):
                    _launch_factor_backward_into_logits(
                        output, d_output, None, words, target, activation.alpha)
                else:
                    _launch_softmax_backward_into_logits(
                        output, d_output, None, target)
        return d_read_logits, d_write_logits, None


# ---------------------------------------------------------------------------
# Torch lowering of the SAME exact solve, for the domains the kernels do not cover
# (CPU, and fp64 -- the oracle-equivalence gate projects fp64 logits end to end).
#
# This is a second *lowering* of the production semantics, not a fallback to the
# reference: it computes the identical sorted closed form the CUDA kernels do
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


def _torch_activation(logits: torch.Tensor, activation) -> torch.Tensor:
    if isinstance(activation, SoftmaxActivation):
        return torch.softmax(logits, dim=-1)
    return _SortedEntmaxFn.apply(logits, activation.alpha)


def _torch_routing_factor_levels(
    read_logits: torch.Tensor, write_logits: torch.Tensor, routing: ResolvedRouting,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    reads, writes = [], []
    for index, level in enumerate(routing.levels):
        span = routing.level_logit_slice(index)
        read_level = read_logits[..., span]
        write_level = write_logits[..., span]
        if isinstance(level, TiedRouting):
            # TiedRouting names ONE activation for its one solve through the same
            # `.activation('write')` accessor it answers `'read'` with; it carries no
            # separate `.write` attribute.
            write_factor = _torch_activation(write_level, level.activation("write"))
            read_factor = write_factor
        else:
            read_factor = _torch_activation(read_level, level.read)
            write_factor = _torch_activation(write_level, level.write)
        reads.append(read_factor)
        writes.append(write_factor)
    return tuple(reads), tuple(writes)


def production_routing_levels(
    read_logits: torch.Tensor,
    write_logits: torch.Tensor,
    routing: ResolvedRouting,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor | None]:
    """Exact per-level ``(read, write)`` routing factors on the production solve, and the solve's ``support_words``.

    THE SUPPORT WORDS ARE ROUTED OUT as the kernels write them: int32 ``[B * H, ceil(T / 32), sum of the supported
    levels' widths]``, stream ``b * H + h`` (so ``[B, H, ...]`` is a free view), bit ``t % 32`` of word ``t // 32`` at a
    digit's column = token ``t`` live at that digit, the levels' columns at ``routing.support_offsets``. ``None`` on
    the torch lowering, which writes no words.


    Returns one ``[BH, T, width_l]`` tensor per level-side -- the shape the producer
    hands ``naive_fsla`` and the consumer. Lowered to the hand-CUDA solve
    on CUDA bf16 and to the identical sorted closed form in torch elsewhere;
    the two agree by construction rather than by tolerance, since the threshold is a
    closed form and not an iterate.

    CUDA LOGITS ARRIVE IN ``bfloat16``, which is what the shipped projection emits -- bf16
    operands on the tensor cores, fp32 accumulate, bf16 out -- or in ``float64``, which takes
    the torch lowering (the oracle-equivalence gate runs there); any other CUDA dtype is
    refused by name. The kernels widen each logit IN REGISTER, so no pass over the plane
    exists in either direction (docs/internals/entmax/factor.md#logit-dtype).

    LAYOUT: flat ``[BH, T, packed_logit_width]``, or the
    TOKEN-MAJOR ``[B, T, H, packed_logit_width]`` the packed router GEMM writes without
    a transpose. The OUTPUT layout is the same either way. Only the CUDA solve reads the
    second one as it lies (it addresses ``b`` and ``h`` as two strides); the torch
    lowering flattens it first, which is the transpose this seam exists to avoid paying
    on the shipped path and does not mind paying on the oracle's.

    The CUDA solve's output is the STORED form: the consumer reads bf16 and nothing else
    reads these at all, so the kernel writes bf16 and no narrowing pass exists. The torch
    lowering returns its own dtype, the domains (CPU, fp64) where the stored form is not
    the question.
    """
    if not isinstance(routing, ResolvedRouting):
        raise TypeError(f"routing must be ResolvedRouting, got {type(routing).__name__}")
    if read_logits.shape != write_logits.shape:
        raise ValueError(
            "read/write logits must have identical shapes, got "
            f"{read_logits.shape} and {write_logits.shape}")
    if read_logits.ndim not in (3, 4) or read_logits.shape[-1] != routing.packed_logit_width:
        raise ValueError(
            "routing logits must be [BH, T, packed_logit_width] or "
            f"[B, T, H, packed_logit_width] with packed_logit_width="
            f"{routing.packed_logit_width}, got {tuple(read_logits.shape)}")
    if read_logits.device != write_logits.device or read_logits.dtype != write_logits.dtype:
        raise ValueError("read/write logits must share one device and dtype")

    if read_logits.is_cuda and read_logits.dtype not in (torch.bfloat16, torch.float64):
        raise TypeError(
            "the CUDA routing solve takes bfloat16 logits (the kernels) or float64 logits (the torch "
            f"lowering the oracle-equivalence gate runs), got {read_logits.dtype}")
    if read_logits.is_cuda and read_logits.dtype is torch.bfloat16:
        #: NOT `.contiguous()`: every solve entry below reads the logits through an
        #: explicit stride tuple, so a packed projection's half-slice -- and its head
        #: axis -- are addressable as they stand, and materializing either is a whole
        #: extra pass over the side.
        outputs = _ProductionLevelsFn.apply(read_logits, write_logits, routing)
        return tuple(outputs[:routing.D]), tuple(outputs[routing.D:2 * routing.D]), outputs[2 * routing.D]
    reads, writes = _torch_routing_factor_levels(_flat(read_logits), _flat(write_logits), routing)
    return reads, writes, None


def _flat(logits: torch.Tensor) -> torch.Tensor:
    """Token-major ``[B, T, H, W]`` as the flat ``[BH, T, W]`` plane, or itself.

    This IS the transpose the token-major layout exists to avoid; it is here because the
    torch lowering has no stride tuple to read the head axis through, and it runs on the
    fp64 oracle path where a pass over the plane is not the cost that matters.
    """
    if logits.ndim != 4:
        return logits
    return logits.permute(0, 2, 1, 3).reshape(
        logits.shape[0] * logits.shape[2], logits.shape[1], logits.shape[3])


__all__ = [
    "production_routing_levels",
]
