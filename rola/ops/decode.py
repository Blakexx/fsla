# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""The T=1 decode op — the frozen-config carrier, the workspace, and the guard.

Design ruling (docs/internals/decode.md): **T = 1 has no stage 2.** Every stage-2 decision is
degenerate at one token (``BT``: nothing to tile; ``P``: nothing to split;
statistics: a population of one), so the decode path bypasses selection
*architecturally* — the config freezes at
the prefill→decode boundary and per-step work is data plane only. Nothing here builds
a ``Plan``, a ``Schedule`` or a set of tile bitmaps, and that is the point rather than
an omission.

**WHY THERE IS NO SCHEDULE.** At ``T = 1`` a leaf is in the read support iff every
level's digit has a nonzero read amplitude, so the support is a Cartesian product and
its bitmap is a Kronecker product of the per-level indicators. The kernel expands that
exactly (``rola_decode_expand``) and walks it (``rola_decode_gemv``). The membership
predicate it uses — ``amp != 0`` — is not a threshold standing in for the schedule's
bitmap; it is that bitmap's *defining expression* (``schedule.py``'s
``read_mask = tuple(level != 0 ...)``), evaluated one step earlier so the object never
has to be materialized at tile granularity. ``tests/tier1_semantics/
test_decode_expand.py`` gates the two bit for bit.

**DECODE'S DECAY DOMAIN IS STRICTLY LARGER THAN THE TILED KERNEL'S, and that is a
consequence rather than a design goal.** The tiled consumer refuses a routing whose
per-leaf reference exponent exceeds ``PANEL_MAX_EXPONENT``, because its §12 rule-1
lowering exponentiates ``keep^{+P}`` and ``keep^{-P}`` separately and their product is
wrong (not merely rounded) once one saturates while the other underflows. At ``T = 1``
that rule-1 half-mass shift is *exactly the identity* — the readout's ``fr * s0``
collapses to ``keep^c`` and the fold's ``s0 * (s0 M + W fw Vw)`` to
``keep^c M + W Vw`` — so decode forms **one** exponential, never a quotient of two,
and has no admissibility precondition to check. It therefore does not import one.

**FORWARD ONLY, AND THE GUARD RAISES.** The decode kernels have no backward. The
layer's ``select_execution_backend`` already excludes training mode, gradient
reachability and non-CUDA devices before the decode fork is reached, so this module's
own check is defense in depth for a direct caller — and it *raises* rather than
silently redirecting to the oracle, because a silent redirect is the same class of
failure (two implementations behind one call) that the guard exists to prevent.

**UNPAGED ONLY, DELIBERATELY.** §4.5's page granule is the ATOM
(``page_states := MMA_K_QUANTUM``), which makes the paged address BC-free, but this
path does not implement it: a page table is REFUSED loudly rather than approximated,
so that decode's BC-blindness is a property of what it accepts and not of what a
caller happens to pass.

**THE KERNEL-LAYOUT CARRIER.** :class:`FrozenDecodeConfig` answers "what shape is the
launch?" -- widths, mixed-radix strides, spans, row offsets, word and group counts,
``d_v``, the normalization and decay arms, ``eps``, ``n_split``. None of that is a
derived axis. ``BC`` is deliberately NOT a field: owner blocking is a property of how
the tiled consumer amortizes a ``[BC, cols]`` block across many tokens, and with one
token there is nothing to amortize, so the decode kernels are BC-blind by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from rola.ops._ext import extension
from rola.ops.constants import READOUT_EPS
from rola.planner.schedule import leaf_rate_dials
from rola.routing.types import Topology

__all__ = [
    "C_DECODE_ROWS_PER_CTA",
    "C_DECODE_TARGET_CTAS_PER_SM",
    "DecodeWorkspace",
    "FrozenDecodeConfig",
    "decode_forward",
    "freeze_decode_config",
]


#: §8-class calibration placeholders. Both defaults are chosen so that a WRONG value
#: degrades to today's behaviour and never to a new behaviour: they only move how many
#: CTAs share a step's rows, and a CTA with no rows exits after one L2 read while a CTA
#: with too many merely serializes. The terminal cycle fills them from a rows-per-CTA
#: sweep at the at-scale cell and an occupancy-oracle query on the shipped footprint.
C_DECODE_ROWS_PER_CTA = 256
C_DECODE_TARGET_CTAS_PER_SM = 12  # 2 waves x the design's 6 CTAs/SM residency estimate


@dataclass(frozen=True, slots=True)
class FrozenDecodeConfig:
    """Frozen at the prefill→decode boundary, immutable for the sequence.

    ``BC`` IS NOT A FIELD and must not become one. Owner blocking is a property of how
    the tiled consumer amortizes a ``[BC, cols]`` block across many tokens; with one
    token there is nothing to amortize, and the decode kernels are BC-blind by
    construction (``tests/tier2_geometry/test_decode_bc_blindness.py`` asserts
    byte-identical decode output under prefill ``BC`` in ``{16, 32, 64}``).
    """

    D: int
    widths: tuple[int, ...]
    leaf_stride: tuple[int, ...]     # prod(widths[l+1:]) — the leaf mixed radix
    level_span: tuple[int, ...]      # prod(widths[l:])   — the expansion's copy geometry
    level_row_offset: tuple[int, ...]
    n_words: int
    n_groups: int
    d_v: int
    has_decay: bool
    eps: float
    n_split: int
    #: The layer's numeric-format pin, carried across the boundary so the carrier can
    #: never silently disagree with the layer that froze it (coverage-closure H9: the
    #: field was missing and the pin was dropped at ``L == 1``). At ``T = 1`` there is
    #: no MMA — the decode GEMV is scalar fp32 for EVERY pin, which is exact for both
    #: formats' contracts (fp32 accumulate is what the tiled consumer's tf32x3 and
    #: bf16 arms share; only their OPERAND format differs, and a GEMV has no operand
    #: tile to format). So the field selects nothing today; it exists so the pin is
    #: carried, validated, and checkable, and so the future M-4 decode kernel (§6b)
    #: inherits it instead of re-plumbing the boundary.
    mma_precision: str = "bf16"
    #: §4.5; ``None`` on the unpaged path, which is the only path decode implements.
    page_states: int | None = None
    atoms_per_bh: int | None = None

    @property
    def N(self) -> int:
        return self.level_span[0]

    @property
    def cols(self) -> int:
        return self.d_v + 1  # the mass column is unconditional (raw is dead)

    @property
    def total_rows(self) -> int:
        return sum(self.widths)


def freeze_decode_config(
    topology: Topology,
    normalization: str,
    *,
    decay: bool,
    BH: int,
    device: torch.device | str,
    eps: float = READOUT_EPS,
    n_split: int | None = None,
    mma_precision: str,
) -> FrozenDecodeConfig:
    """Build the carrier once, at the boundary.

    ``n_split`` is FROZEN here and never selected per step: re-deriving it would
    reintroduce exactly the per-step planner §6b dissolves. CTAs whose row range starts
    beyond the realized touched-row count read one L2 word and exit; that
    over-launching is the price of never reading a device value back to the host, which
    is the one thing a decode step must not do.
    """
    if mma_precision != "bf16":
        raise ValueError(
            "mma_precision is a dead axis: 'bf16' is the only value, "
            f"got {mma_precision!r}")
    widths = tuple(topology.widths)
    D = len(widths)
    N = math.prod(widths)
    leaf_stride = tuple(math.prod(widths[l + 1:]) for l in range(D))
    level_span = tuple(math.prod(widths[l:]) for l in range(D))
    row_offset = []
    acc = 0
    for w in widths:
        row_offset.append(acc)
        acc += w
    n_words = (N + 31) // 32
    n_groups = (n_words + 31) // 32

    if n_split is None:
        # M_ceiling: every level contributes its full width. The realized touched-row
        # count is a device fact and reading it back is the one thing a decode step must
        # not do, so the split is sized for the whole row space -- more CTAs, never
        # fewer.
        m_ceiling = N
        dev = torch.device(device)
        if dev.type == "cuda":
            sms = torch.cuda.get_device_properties(dev).multi_processor_count
        else:
            sms = 1
        cap = max(1, math.ceil(sms * C_DECODE_TARGET_CTAS_PER_SM / max(1, BH)))
        n_split = min(max(1, math.ceil(m_ceiling / C_DECODE_ROWS_PER_CTA)), cap)

    return FrozenDecodeConfig(
        D=D, widths=widths, leaf_stride=leaf_stride, level_span=level_span,
        level_row_offset=tuple(row_offset), n_words=n_words, n_groups=n_groups,
        d_v=topology.d_v, has_decay=decay,
        eps=eps, n_split=int(n_split),
        mma_precision=mma_precision)


@dataclass(slots=True)
class DecodeWorkspace:
    """Per-sequence scratch, allocated once and reused across steps.

    Rebuilt (not resized) whenever the shape key moves. ``ctr`` is zeroed exactly once
    here: the last-arriving CTA of every step resets it, so no ``memset`` launch ever
    stands between two decode steps and the whole step stays capturable in a CUDA
    graph.
    """

    bits: torch.Tensor
    pc_coarse: torch.Tensor
    ws: torch.Tensor
    ctr: torch.Tensor
    key: tuple = field(default=())

    @staticmethod
    def build(config: FrozenDecodeConfig, BH: int, device) -> DecodeWorkspace:
        i32 = dict(dtype=torch.int32, device=device)
        f32 = dict(dtype=torch.float32, device=device)
        return DecodeWorkspace(
            bits=torch.empty(4 * BH * config.n_words, **i32),
            pc_coarse=torch.empty(BH, config.n_groups + 1, **i32),
            ws=torch.empty(BH, config.n_split, config.cols, **f32),
            ctr=torch.zeros(BH, **i32),
            key=(BH, config.n_words, config.n_groups, config.n_split, config.cols,
                 torch.device(device)))

    def matches(self, config: FrozenDecodeConfig, BH: int, device) -> bool:
        return self.key == (BH, config.n_words, config.n_groups, config.n_split,
                            config.cols, torch.device(device))


def _fold(t: torch.Tensor) -> torch.Tensor:
    """``[B, 1, H, W] -> [B*H, W]`` with ``bh = b * H + h``.

    The fold order is the consumer's own (``h = bh % H`` in the kernel), restated here
    rather than imported because decode shares no schedule object with it — and a fold
    that disagreed would misroute every head's dials.
    """
    B, T, H, W = t.shape
    if T != 1:
        raise ValueError(f"the decode op consumes exactly one token, got T={T}")
    return t.permute(0, 2, 1, 3).reshape(B * H, W).contiguous()


def _fold_scalar(t: torch.Tensor) -> torch.Tensor:
    """``[B, 1, H] -> [B*H]``, the same order."""
    B, T, H = t.shape
    if T != 1:
        raise ValueError(f"the decode op consumes exactly one token, got T={T}")
    return t.permute(0, 2, 1).reshape(B * H).contiguous()


def _guard_no_grad(operands) -> None:
    if not torch.is_grad_enabled():
        return
    if any(t is not None and torch.is_tensor(t) and t.requires_grad for t in operands):
        raise RuntimeError(
            "rola decode is forward-only and has no native backward; reaching it from a "
            "grad-requiring context would return a tensor with no grad_fn and silently "
            "sever the graph (the x.grad-is-None failure class). Use eval mode / "
            "torch.no_grad, or the differentiable oracle path.")


def decode_forward(
    v: torch.Tensor,
    read_levels: tuple[torch.Tensor, ...],
    write_levels: tuple[torch.Tensor, ...],
    g_read: torch.Tensor | None,
    g_write: torch.Tensor,
    config: FrozenDecodeConfig,
    state: torch.Tensor,
    *,
    decay=None,
    write_levels_stored: tuple[torch.Tensor, ...] | None = None,
    workspace: DecodeWorkspace | None = None,
    paging=None,
) -> tuple[torch.Tensor, torch.Tensor, DecodeWorkspace]:
    """One decode step, in place on ``state``.

    ``v`` is ``[B, 1, H, d_v]``; the level tensors are ``[B, 1, H, width_l]``;
    ``g_write`` is ``[B, 1, H]`` and ``g_read`` is the same or ``None`` under the
    global arm. ``state`` is ``[B, H, N, d_v + 1]`` and is UPDATED IN PLACE —
    every touched row has exactly one owning CTA, warp and lane, so the update needs no
    atomic and is bitwise reproducible.

    Returns ``(y, state, workspace)`` with ``y`` shaped ``[B, 1, H, d_v]``.
    """
    if paging is not None:
        raise NotImplementedError(
            "the decode path is unpaged only. §4.5's page granule is the ATOM "
            "(page_states := MMA_K_QUANTUM), which is what keeps a paged address "
            "BC-free -- but this path does not implement paged addressing, so a page "
            "table is refused rather than approximated.")

    _guard_no_grad((v, g_read, g_write, *read_levels, *write_levels,
                    *(write_levels_stored or ()),
                    *(decay.dials if decay is not None else ())))

    if v.dim() != 4:
        raise ValueError(f"v must be [B, 1, H, d_v], got {tuple(v.shape)}")
    B, _T, H, d_v = v.shape
    BH = B * H
    if d_v != config.d_v:
        raise ValueError(f"v's d_v={d_v} disagrees with the frozen config's {config.d_v}")
    if g_read is not None:
        raise ValueError(
            "normalization='global' fixes g_read := 1 (raw is dead); pass g_read=None")
    if (decay is not None) != config.has_decay:
        raise ValueError(
            f"the frozen config says has_decay={config.has_decay} but decay is "
            f"{'absent' if decay is None else 'present'}; the config is frozen for the "
            "sequence and the decay arm is part of it")
    if state.shape != (B, H, config.N, config.cols):
        raise ValueError(
            f"state must be [B, H, N, d_v + 1] = "
            f"{(B, H, config.N, config.cols)}, got {tuple(state.shape)}")

    if workspace is None or not workspace.matches(config, BH, v.device):
        workspace = DecodeWorkspace.build(config, BH, v.device)

    amp_read = [_fold(t).float() for t in read_levels]
    amp_write = [_fold(t).float() for t in write_levels]

    amp_clock = None
    dials = None
    if decay is not None:
        stored = write_levels_stored if write_levels_stored is not None else write_levels
        for level, (s, w) in enumerate(zip(stored, write_levels)):
            # `stored != 0 => live != 0`, checked rather than assumed — the same
            # precondition `build_clock_stream` enforces, for the same reason: a
            # storage path that manufactures a nonzero from a zero would give a leaf a
            # clock (hence decay) with no deposit, an O(1) survival error on a path
            # with no other equivalence check.
            if bool(((s != 0) & (w == 0)).any()):
                raise ValueError(
                    f"write_levels_stored[{level}] is nonzero where write_levels[{level}] is "
                    "zero; quantization may only flush toward zero, never manufacture a "
                    "nonzero, and the clock's write-conditioned quiescence depends on it")
        # The clock is `c[t, s] = prod_l p_write_stored[t, d_l(s)]` (spec §2) — a pure
        # product of the STORED allocations with no side gain: `g_write` scales the
        # deposit, not the clock.
        amp_clock = [_fold(t).float() for t in stored]
        dials = leaf_rate_dials(decay, _TopologyShim(config)).to(v.device).float()

    y = extension().rola_decode_forward(
        amp_read=amp_read,
        amp_write=amp_write,
        amp_clock=amp_clock,
        dials=dials,
        g_write=_fold_scalar(g_write).float(),
        v=_fold(v).float(),
        state=state.reshape(BH, config.N, config.cols),
        bits=workspace.bits,
        pc_coarse=workspace.pc_coarse,
        ws=workspace.ws,
        ctr=workspace.ctr,
        level_widths=list(config.widths),
        level_strides=list(config.leaf_stride),
        level_spans=list(config.level_span),
        level_row_offsets=list(config.level_row_offset),
        heads=H,
        n_split=config.n_split,
        eps=config.eps,
    )
    return y.view(B, H, 1, d_v).permute(0, 2, 1, 3).contiguous(), state, workspace


class _TopologyShim:
    """The two attributes ``leaf_rate_dials`` reads, off the frozen config.

    ``leaf_rate_dials`` is REUSED rather than reimplemented — it is the one place the
    fp64→fp32 dial cast is re-checked for the ``delta < 1`` strictness the kernel's
    ``log1pf(-rate)`` depends on, and a second copy is a second thing that can drift.
    It wants a topology; the frozen config carries what it reads, and a shim is
    cheaper than making the decode path carry a ``Topology`` it uses for nothing else.
    """

    __slots__ = ("widths", "d_v", "D")

    def __init__(self, config: FrozenDecodeConfig):
        self.widths = config.widths
        self.d_v = config.d_v
        self.D = config.D
