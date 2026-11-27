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

from fsla.routing.masks import WORD_BITS
from fsla.routing.projection import RouterLayout
from fsla.routing.types import EntmaxActivation, SoftmaxActivation


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
    (a tied read entry): the kernels already write a second output and accumulate
    a second incoming gradient, and returning one tensor twice from a single
    ``autograd.Function`` would alias two graph outputs.

    THE READ SIDE IS ``G`` HEADS A WRITE HEAD. The read plane is ``[B * H_w * G, T, ·]``, query head ``w * G + g``'s
    rows at stream ``(b * H_w + w) * G + g`` -- the public ``[B, T, H, ·]`` as a permute -- and each solve of head ``g``
    writes the strided view of its rows (:func:`_head`), whose streams are the logits' ``b * H_w + w``. A level's write
    solve writes the first tied head's rows as its second output; a further tied head's rows are copied from the write
    side's, and the backward hands the write solve the write side's and every tied head's incoming gradients, as the
    two cotangents it adds (their sum's two terms where there are more than two, :func:`_two_terms`).
    """

    @staticmethod
    def forward(ctx, logits, layout):
        n_streams, tokens = _arena_shape(logits)
        device, G = logits.device, layout.group
        words = _cdiv(tokens, WORD_BITS)
        write_support = torch.empty((n_streams, words, layout.write_support_width), dtype=torch.int32, device=device)
        read_support = torch.empty((n_streams * G, words, layout.read_support_width), dtype=torch.int32, device=device)
        empty_launch = n_streams == 0 or tokens == 0
        heads = _heads(logits)
        #: ONE PLANE PER SIDE, sliced into levels. Each level is still a distinct graph
        #: output, at its own columns of one allocation -- which is what lets the levels
        #: of a width class share a launch
        #: (docs/internals/entmax/factor.md#batched-levels), and what makes the whole
        #: side ONE contiguous tensor in the layout the consumer reads.
        #:
        #: UNINITIALIZED: a dense level is exactly the tokens, and every solve writes
        #: every element of one (unlike the tile-major arena, whose tail lanes past
        #: `tokens` no kernel touches).
        packed_width = layout.side_width
        write_plane = torch.empty((n_streams, tokens, packed_width), dtype=torch.bfloat16, device=device)
        read_plane = torch.empty((n_streams * G, tokens, packed_width), dtype=torch.bfloat16, device=device)
        read_heads = tuple(_head(read_plane, G, g) for g in range(G))
        support_heads = tuple(_head(read_support, G, g) for g in range(G))

        read_levels: list[torch.Tensor] = []
        write_levels: list[torch.Tensor] = []
        copies: list[tuple[int, int, int]] = []
        plan = _Plan()

        def solve(activation, output, second, support, width, *, logit_off, value_off, support_off):
            if isinstance(activation, EntmaxActivation):
                plan.add("entmax", activation.alpha, (logits, output, second, support), width,
                         logit_off=logit_off, value_off=value_off, support_off=support_off, heads=heads)
            else:
                plan.add("softmax", 0.0, (logits, output, second), width,
                         logit_off=logit_off, value_off=value_off, heads=heads)

        for index, width in enumerate(layout.widths):
            offset = layout.level_offsets[index]
            read_levels.append(read_plane[:, :, offset:offset + width])
            write_levels.append(write_plane[:, :, offset:offset + width])
            tied = layout.tied_heads(index)
            copies.extend((g, offset, width) for g in tied[1:])
            if empty_launch:
                continue
            solve(layout.topology.write[index], write_plane, read_heads[tied[0]] if tied else None, write_support,
                  width, logit_off=layout.write_columns(index).start, value_off=offset,
                  support_off=layout.write_support_offsets[index])
            for g, entries in enumerate(layout.heads):
                columns = layout.read_columns(g, index)
                if columns is not None:
                    solve(entries[index], read_heads[g], None, support_heads[g], width, logit_off=columns.start,
                          value_off=offset, support_off=layout.read_support_offsets[index])

        plan.issue()
        for g, offset, width in copies:
            read_heads[g][:, :, offset:offset + width].copy_(write_plane[:, :, offset:offset + width])

        ctx.save_for_backward(*read_levels, *write_levels, write_support, read_support)
        ctx.layout = layout
        ctx.logits_shape = logits.shape
        ctx.logits_dtype = logits.dtype
        ctx.mark_non_differentiable(write_support, read_support)
        return (*read_levels, *write_levels, write_support, read_support)

    @staticmethod
    def backward(ctx, *grads):
        layout = ctx.layout
        D, G = layout.D, layout.group
        saved = ctx.saved_tensors
        read_levels, write_levels = saved[:D], saved[D:2 * D]
        write_support, read_support = saved[2 * D], saved[2 * D + 1]

        d_logits = torch.zeros(ctx.logits_shape, dtype=ctx.logits_dtype, device=write_support.device)
        n_streams, tokens = _arena_shape(d_logits)
        if n_streams == 0 or tokens == 0:
            return d_logits, None

        def words_of(support, offset, width):
            return None if offset < 0 else support[:, :, offset:offset + width]

        def backward_into(activation, output, d_output, d_second, words, target):
            if isinstance(activation, EntmaxActivation):
                _launch_factor_backward_into_logits(output, d_output, d_second, words, target, activation.alpha)
            else:
                _launch_softmax_backward_into_logits(output, d_output, d_second, target)

        for index, width in enumerate(layout.widths):
            read_output, write_output = read_levels[index], write_levels[index]
            d_read = grads[index]
            d_write = grads[D + index]
            d_read = (torch.zeros_like(read_output) if d_read is None else d_read).contiguous()
            d_write = (torch.zeros_like(write_output) if d_write is None else d_write).contiguous()
            write_words = words_of(write_support, layout.write_support_offsets[index], width)
            write_target = d_logits[..., layout.write_columns(index)]
            tied = layout.tied_heads(index)
            if tied:
                #: the tied solve's form: the first tied head's rows; its gradient first and the write side's second,
                #: or with several tied heads the two terms of every cotangent's sum (`_two_terms`)
                first = tied[0]
                d_first, d_second = _head(d_read, G, first), d_write
                if len(tied) > 1:
                    d_first, d_second = _two_terms([d_write, *(_head(d_read, G, g) for g in tied)])
                backward_into(layout.topology.write[index], _head(read_output, G, first), d_first, d_second,
                              write_words, write_target)
            else:
                backward_into(layout.topology.write[index], write_output, d_write, None, write_words, write_target)
            read_words = words_of(read_support, layout.read_support_offsets[index], width)
            for g, entries in enumerate(layout.heads):
                columns = layout.read_columns(g, index)
                if columns is not None:
                    backward_into(entries[index], _head(read_output, G, g), _head(d_read, G, g), None,
                                  None if read_words is None else _head(read_words, G, g), d_logits[..., columns])
        return d_logits, None


def _two_terms(cotangents: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """The sum of three or more cotangents as the TWO terms the backward kernels add in fp32: the fp32 sum's rounding
    into the cotangents' type, and the remainder's. A sum formed in that type alone rounds at its 8 bits, and the
    solve's Jacobian carries the rounding into the logits' gradient (docs/internals/entmax/factor.md#read-heads)."""
    total = cotangents[0].float()
    for cotangent in cotangents[1:]:
        total = total + cotangent.float()
    high = total.to(cotangents[0].dtype)
    return high, (total - high.float()).to(cotangents[0].dtype)


def _head(plane: torch.Tensor, group: int, head: int) -> torch.Tensor:
    """Read head ``head``'s rows of a ``[S * group, ...]`` read-side plane as ``[S, ...]``, a strided view: stream ``s``
    of the logits is rows ``s * group + head``. At ``group`` 1 it is the plane itself, as a view."""
    return plane.view(plane.shape[0] // group, group, *plane.shape[1:])[:, head]


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
    logits: torch.Tensor, layout: RouterLayout,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    reads, writes = [], []
    for index in range(layout.D):
        write_factor = _torch_activation(logits[..., layout.write_columns(index)], layout.topology.write[index])
        heads = []
        for g, entries in enumerate(layout.heads):
            columns = layout.read_columns(g, index)
            heads.append(write_factor if columns is None else _torch_activation(logits[..., columns], entries[index]))
        reads.append(_interleaved(heads))
        writes.append(write_factor)
    return tuple(reads), tuple(writes)


def _interleaved(heads: list[torch.Tensor]) -> torch.Tensor:
    """``G`` read heads' ``[S, T, w]`` as the read side's ``[S * G, T, w]``, head ``g`` at rows ``s * G + g``; one head
    is itself."""
    return heads[0] if len(heads) == 1 else torch.stack(heads, dim=1).flatten(0, 1)


def production_routing_levels(
    logits: torch.Tensor,
    layout: RouterLayout,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, torch.Tensor] | None]:
    """Exact per-level ``(read, write)`` routing factors on the production solve, and the solve's support words.

    THE SUPPORT WORDS ARE ROUTED OUT as the kernels write them, ``(write, read)``: int32 ``[B * H_w, ceil(T / 32), ·]``
    on the write side and ``[B * H, ceil(T / 32), ·]`` on the read side (``H = H_w * G`` the query heads), stream
    ``b * H + h`` (so ``[B, H, ...]`` is a free view), bit ``t % 32`` of word ``t // 32`` at a digit's column = token
    ``t`` live at that digit, a level's columns at ``layout.write_support_offsets`` and ``layout.read_support_offsets``.
    ``None`` on the torch lowering, which writes no words.

    Returns one tensor per level-side: ``[B * H_w, T, width_l]`` write and ``[B * H, T, width_l]`` read, query head
    ``w * G + g`` read head ``g`` of write head ``w`` -- the shape the producer hands ``naive_fsla`` and the consumer.
    A tied read entry's rows are its write level's. Lowered to the hand-CUDA solve
    on CUDA bf16 and to the identical sorted closed form in torch elsewhere;
    the two agree by construction rather than by tolerance, since the threshold is a
    closed form and not an iterate.

    CUDA LOGITS ARRIVE IN ``bfloat16``, which is what the shipped projection emits -- bf16
    operands on the tensor cores, fp32 accumulate, bf16 out -- or in ``float64``, which takes
    the torch lowering (the oracle-equivalence gate runs there); any other CUDA dtype is
    refused by name. The kernels widen each logit IN REGISTER, so no pass over the plane
    exists in either direction (docs/internals/entmax/factor.md#logit-dtype).

    LAYOUT: the projection's logits (:func:`fsla.routing.projection.router_logits`), flat
    ``[B * H_w, T, packed_router_width]`` or the TOKEN-MAJOR ``[B, T, H_w, packed_router_width]`` the packed router
    GEMM writes without a transpose; every solve reads its level's column span of the one plane. The OUTPUT layout is
    the same either way. Only the CUDA solve reads the second one as it lies (it addresses ``b`` and ``h`` as two
    strides); the torch lowering flattens it first, which is the transpose this seam exists to avoid paying on the
    shipped path and does not mind paying on the oracle's.

    The CUDA solve's output is the STORED form: the consumer reads bf16 and nothing else
    reads these at all, so the kernel writes bf16 and no narrowing pass exists. The torch
    lowering returns its own dtype, the domains (CPU, fp64) where the stored form is not
    the question.
    """
    if type(layout) is not RouterLayout:
        raise TypeError(f"layout must be a RouterLayout, got {type(layout).__name__}")
    if not isinstance(logits, torch.Tensor) or logits.ndim not in (3, 4) \
            or logits.shape[-1] != layout.packed_router_width:
        raise ValueError(
            "routing logits must be [B * H_w, T, packed_router_width] or [B, T, H_w, packed_router_width] with "
            f"packed_router_width={layout.packed_router_width}, got {tuple(getattr(logits, 'shape', ()))}")
    if logits.is_cuda and logits.dtype not in (torch.bfloat16, torch.float64):
        raise TypeError(
            "the CUDA routing solve takes bfloat16 logits (the kernels) or float64 logits (the torch "
            f"lowering the oracle-equivalence gate runs), got {logits.dtype}")
    if logits.is_cuda and logits.dtype is torch.bfloat16:
        #: NOT `.contiguous()`: every solve entry below reads the logits through an
        #: explicit stride tuple, so the projection's routing slice -- and its head
        #: axis -- are addressable as they stand, and materializing either is a whole
        #: extra pass over the plane.
        outputs = _ProductionLevelsFn.apply(logits, layout)
        D = layout.D
        return tuple(outputs[:D]), tuple(outputs[D:2 * D]), (outputs[2 * D], outputs[2 * D + 1])
    reads, writes = _torch_routing_factor_levels(_flat(logits), layout)
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
