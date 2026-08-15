# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""The T=1 decode op -- the frozen-config carrier, the workspace, and the guard.

    decode_forward(routes, v, state, decay=..., expert=...) -> (y, state)

The `T = 1` sibling of `rola_op`: one token, no schedule, no selection. The bypass is
architectural rather than a cheap default (docs/internals/decode/decode.md#no-stage-2),
the support is a Kronecker product the kernel expands exactly
(docs/internals/decode/decode.md#kronecker-support), decode's decay domain is strictly
larger than the chunk arm's as a CONSEQUENCE
(docs/internals/decode/decode.md#decay-domain), and the carrier and the workspace are
Part C of the same page.

FORWARD ONLY. The decode kernels have no backward; the guard below is defence in depth
behind the layer's own envelope check (docs/internals/decode/decode.md#forward-only).

PAGED AND DENSE ARE ONE WALK: the page granule is the ATOM, which is what makes the
paged address BC-free, so the whole difference is one indirection resolved once per
touched atom (docs/internals/decode/decode.md#the-atom).

GROWTH IS THE STEP'S OWN NEW WORK, and most steps have none -- the step asks on the
DEVICE first, because the admission costs several times the launch it precedes
(docs/internals/decode/decode.md#growth).
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, field

import torch

from rola.ops._ext import extension
from rola.ops.constants import READOUT_EPS
from rola.ops.decay import leaf_rate_dials
from rola.ops.paging import MMA_K_QUANTUM
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
    *,
    d_v: int,
    decay: bool,
    BH: int,
    device: torch.device | str,
    eps: float = READOUT_EPS,
    n_split: int | None = None,
    mma_precision: str,
) -> FrozenDecodeConfig:
    """Build the carrier once, at the boundary.

    ``n_split`` is FROZEN here and never selected per step: re-deriving it would
    reintroduce exactly the per-step selection §6b dissolves. CTAs whose row range starts
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
        d_v=int(d_v), has_decay=decay,
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
    #: The growth probe's device answer and its PINNED landing buffer. Pinned is what
    #: makes the read one DMA rather than a staging copy, and holding both here is what
    #: keeps a steady step free of allocation: docs/internals/decode/decode.md#growth-probe
    growth: torch.Tensor
    growth_host: torch.Tensor
    #: The probe's OTHER output: `[BH, N/16]` bool, the step's own write-atom set,
    #: condensed on the device from the expansion the verdict was read off. A growth
    #: step admits from THIS -- the host derives nothing: docs/internals/decode/decode.md#growth
    atom_bits: torch.Tensor
    key: tuple = field(default=())

    @staticmethod
    def build(config: FrozenDecodeConfig, BH: int, device) -> DecodeWorkspace:
        i32 = dict(dtype=torch.int32, device=device)
        f32 = dict(dtype=torch.float32, device=device)
        cuda = torch.device(device).type == "cuda"
        return DecodeWorkspace(
            bits=torch.empty(4 * BH * config.n_words, **i32),
            pc_coarse=torch.empty(BH, config.n_groups + 1, **i32),
            ws=torch.empty(BH, config.n_split, config.cols, **f32),
            ctr=torch.zeros(BH, **i32),
            growth=torch.empty(BH, **i32),
            growth_host=torch.empty(BH, dtype=torch.int32, pin_memory=cuda),
            atom_bits=torch.empty(BH, config.N // MMA_K_QUANTUM, dtype=torch.bool,
                                  device=device),
            key=(BH, config.N, config.n_words, config.n_groups, config.n_split, config.cols,
                 torch.device(device)))

    def matches(self, config: FrozenDecodeConfig, BH: int, device) -> bool:
        return self.key == (BH, config.N, config.n_words, config.n_groups, config.n_split,
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
        from rola.interface import BACKWARD_ARRIVES_WITH, BackwardNotImplemented

        raise BackwardNotImplemented(
            f"rola decode is forward-only: {BACKWARD_ARRIVES_WITH}. Reaching it from a "
            "grad-requiring context would return a tensor with no grad_fn and silently "
            "sever the graph (the x.grad-is-None failure class). Use eval mode / "
            "torch.no_grad.")


#: The workspace map holds its states WEAKLY: a dropped sequence must drop its scratch,
#: and two concurrent sequences must never share a buffer
#: (docs/internals/decode/decode.md#host-carrier).
_CARRIERS: dict[tuple, FrozenDecodeConfig] = {}
_WORKSPACES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _carrier(topology, *, d_v: int, decay: bool, BH: int, device,
             mma_precision: str) -> FrozenDecodeConfig:
    key = (tuple(topology.widths), d_v, decay, BH, torch.device(device), mma_precision)
    config = _CARRIERS.get(key)
    if config is None:
        config = freeze_decode_config(
            topology, d_v=d_v, decay=decay, BH=BH, device=device,
            mma_precision=mma_precision)
        _CARRIERS[key] = config
    return config


def decode_forward(routes, v: torch.Tensor, state, *, decay=None, expert=None):
    """ONE decode step over a routing bundle and a carried state -- the whole surface.

    ``routes`` is a one-token :class:`~rola.routing.factors.RouteFactors`; ``v`` is
    ``[B, 1, H, d_v]``; ``state`` is a :class:`~rola.RoLAState` that CARRIES
    (a step continues a sequence, so an empty state has nothing to decode from).
    Returns ``(y, state)`` -- ``y`` shaped ``[B, 1, H, d_v]`` and the SAME state
    object, updated in place, exactly as :func:`rola.rola_op` returns it.

    The launch machinery -- the frozen carrier and the scratch -- is memoized in this
    MODULE and never appears here, for the same reason the chunk arm's tables do not:
    it is how the step runs, not what it computes. The backing is likewise invisible,
    and its growth is handled below (docs/internals/decode/decode.md#growth).
    """
    from rola.expert import PlanOverrides

    overrides = PlanOverrides() if expert is None else PlanOverrides.of(expert)
    BH = routes.batch * routes.heads
    config = _carrier(routes.topology, d_v=v.shape[-1], decay=decay is not None, BH=BH,
                      device=routes.device, mma_precision=overrides.mma_precision)
    workspace = _WORKSPACES.get(state)
    if workspace is None or not workspace.matches(config, BH, routes.device):
        workspace = DecodeWorkspace.build(config, BH, routes.device)
    #: THE HOST ORDER, `rola_op`'s at `L = 1`: a paged step asks the DEVICE whether it
    #: grows, and only a step that does pays the admission
    #: (docs/internals/decode/decode.md#growth).
    bitmap = None
    if state.paged and _growth_probe(routes.write, config, state.page_table, workspace):
        bitmap = workspace.atom_bits
    plane, page_table, arena = state._decode_entry(routes, bitmap)
    if arena is not None and bitmap is not None:
        arena.wait()
    #: Decode is an fp32 boundary by contract (out of the P09 bf16-memory family's
    #: scope -- a `[BH, width]` object with no measured pressure). A bf16-projected `v`
    #: re-widens here and the cast is exact, the values being bf16-rounded already.
    y, _, _WORKSPACES[state] = _decode_step(
        v.to(torch.float32), routes.read, routes.write, routes.g_read, routes.g_write,
        config, plane, decay=decay, workspace=workspace,
        page_table=page_table)
    return y, state


def _growth_probe(write_levels, config: FrozenDecodeConfig, page_table: torch.Tensor,
                  workspace: DecodeWorkspace) -> bool:
    """Does this step deposit into an atom the page table does not map?

    THE CONDENSATION IS THE OUTPUT, not an internal: it lands in ``workspace.atom_bits``
    and IS what a growth step admits, so the write-atom set is derived exactly once per
    step, on the device. The read is BLOCKING by measurement
    (docs/internals/decode/decode.md#probe-blocking-read).
    """
    amp_write = [_fold(t).float() for t in write_levels]
    extension().rola_decode_growth_probe(
        amp_write=amp_write,
        page_table=page_table,
        bits=workspace.bits,
        growth=workspace.growth,
        atom_bits=workspace.atom_bits,
        level_widths=list(config.widths),
        level_strides=list(config.leaf_stride),
        level_spans=list(config.level_span))
    workspace.growth_host.copy_(workspace.growth)
    return bool(workspace.growth_host.any())


def _write_atom_bitmap(write_levels, config: FrozenDecodeConfig) -> torch.Tensor:
    """``[BH, N / 16]`` bool: the atoms this ONE-TOKEN step deposits into, EXACTLY.

    AN ENUMERATED REFERENCE -- ITS ONLY CALLERS ARE GATES. Deleting it would delete the
    gate's reference; calling it on a shipped path would delete the gate's independence
    (docs/internals/decode/decode.md#second-derivation).
    """
    device = write_levels[0].device
    index = torch.arange(config.N, device=device)
    live = None
    for level, width in enumerate(config.widths):
        digit = (index // config.leaf_stride[level]) % width
        column = _fold(write_levels[level])[:, digit] != 0
        live = column if live is None else (live & column)
    return live.view(-1, config.N // MMA_K_QUANTUM, MMA_K_QUANTUM).any(-1)


def _decode_step(
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
    page_table: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, DecodeWorkspace]:
    """The RAW-TENSOR launch, in place on ``state``. The prefill arm's sibling.

    ``v`` is ``[B, 1, H, d_v]``; the level tensors are ``[B, 1, H, width_l]``;
    ``g_write`` is ``[B, 1, H]`` and ``g_read`` is the same or ``None`` under the
    global arm. ``state`` is the dense ``[B, H, N, d_v + 1]`` plane, or the arena's
    ``[slots, 16, d_v + 1]`` one under ``page_table``; either is UPDATED IN PLACE --
    every touched row has exactly one owning CTA, warp and lane, so the update needs no
    atomic and is bitwise reproducible.

    Returns ``(y, state, workspace)`` with ``y`` shaped ``[B, 1, H, d_v]``.
    """
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
    #: The two backings, and the ONE shape rule each. Under a table the plane holds only
    #: the slots a plan committed, so its leading extent is the arena's and not `BH`;
    #: without one the plane IS the leaf space. Both are handed to the ABI in the shape
    #: the kernel indexes, so the reshape below is the whole difference.
    if page_table is None:
        if state.shape != (B, H, config.N, config.cols):
            raise ValueError(
                f"state must be [B, H, N, d_v + 1] = "
                f"{(B, H, config.N, config.cols)}, got {tuple(state.shape)}")
        plane = state.reshape(BH, config.N, config.cols)
    else:
        if state.dim() != 3 or state.shape[1:] != (MMA_K_QUANTUM, config.cols):
            raise ValueError(
                f"under a page table the state is the arena's plane, "
                f"[slots, {MMA_K_QUANTUM}, d_v + 1], got {tuple(state.shape)}")
        if tuple(page_table.shape) != (BH, config.N // MMA_K_QUANTUM):
            raise ValueError(
                f"page_table must be [BH, N / {MMA_K_QUANTUM}] = "
                f"{(BH, config.N // MMA_K_QUANTUM)}, got {tuple(page_table.shape)}")
        plane = state

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
        state=plane,
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
        page_table=page_table,
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
