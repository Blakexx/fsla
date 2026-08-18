# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""The T=1 decode family's BODIES: the geometry, the scratch, the launches.

The decode DAG (:mod:`rola.engine.dags.decode_dag`) declares these as nodes and composes the
step; the facade lives there with it. What is here is one function per thing that is
done, and the raw-tensor entry :func:`_decode_step` is those same functions in the
step's order, for a caller holding tensors rather than a bundle.

THE STEP IS ONE LAUNCH. Every per-token operand is folded inside the step kernel, from
the producer's own views, so there is no operand plane between them
(docs/internals/decode/decode_fold.md).

The bypass is architectural rather than a cheap default
(docs/internals/decode/decode.md#no-stage-2), the support is a Kronecker product the
kernel expands exactly (docs/internals/decode/decode.md#kronecker-support), and decode's
decay domain is strictly larger than the chunk arm's as a CONSEQUENCE
(docs/internals/decode/decode.md#decay-domain).

FORWARD ONLY. The decode kernels have no backward; the guard below is defence in depth
behind the layer's own envelope check (docs/internals/decode/decode.md#forward-only).

PAGED AND DENSE ARE ONE WALK: the page granule is the ATOM, which is what makes the
paged address BC-free, so the whole difference is one indirection resolved once per
touched atom (docs/internals/decode/decode.md#the-atom).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from rola.engine.plan import DecodeGeometry, DecodePlan
from rola.ops._ext import extension
from rola.ops.constants import READOUT_EPS
from rola.ops.decay import leaf_rate_dials
from rola.ops.paging import MMA_K_QUANTUM
from rola.routing.types import Topology

__all__ = [
    "C_DECODE_ROWS_PER_CTA",
    "C_DECODE_TARGET_CTAS_PER_SM",
    "DecodeScratch",
    "derive_decode_geometry",
    "growth_probe",
    "leaf_rate_dials_for",
    "step",
    "step_plane",
]


#: CALIBRATION PLACEHOLDERS. Both defaults are chosen so that a WRONG value
#: degrades to today's behaviour and never to a new behaviour: they only move how many
#: CTAs share a step's rows, and a CTA with no rows exits after one L2 read while a CTA
#: with too many merely serializes. The terminal cycle fills them from a rows-per-CTA
#: sweep at the at-scale cell and an occupancy-oracle query on the shipped footprint.
C_DECODE_ROWS_PER_CTA = 256
C_DECODE_TARGET_CTAS_PER_SM = 12  # 2 waves x the design's 6 CTAs/SM residency estimate


def derive_decode_geometry(
    topology: Topology,
    *,
    d_v: int,
    decay: bool,
    BH: int,
    device: torch.device | str,
    eps: float = READOUT_EPS,
    n_split: int | None = None,
) -> DecodeGeometry:
    """Build the carrier once, at the boundary.

    ``n_split`` is FROZEN here and never selected per step: re-deriving it would
    reintroduce exactly the per-step selection the design dissolves. CTAs whose row range starts
    beyond the realized touched-row count read one L2 word and exit; that
    over-launching is the price of never reading a device value back to the host, which
    is the one thing a decode step must not do.
    """
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

    return DecodeGeometry(
        D=D, widths=widths, leaf_stride=leaf_stride, level_span=level_span,
        level_row_offset=tuple(row_offset), n_words=n_words, n_groups=n_groups,
        d_v=int(d_v), has_decay=decay,
        eps=eps, n_split=int(n_split))


@dataclass(slots=True)
class DecodeScratch:
    """Per-sequence scratch, allocated once and reused across steps.

    ``ctr`` is zeroed exactly once here: the last-arriving CTA of every step resets it,
    so no ``memset`` launch ever stands between two decode steps and the whole step
    stays capturable in a CUDA graph.
    """

    #: The expansion's GLOBAL scratch, and empty at every shipped topology: the step and
    #: the probe each build their side maps in shared memory, and this is what they fall
    #: back to when the leaf space is too wide for it
    #: (docs/internals/decode/decode_expand.md#smem-arm).
    work: torch.Tensor
    #: `[BH, ceil(N/32)]` int32, THE PROBE'S PUBLISHED EXPANSION: the exact write-leaf map
    #: it condenses to atoms and ANDs against residency. The step publishes no such
    #: object -- its two maps live and die inside the CTA that built them
    #: (docs/internals/decode/decode_expand.md#growth-probe).
    probe_bits: torch.Tensor
    ws: torch.Tensor
    ctr: torch.Tensor
    #: The growth probe's device answer and its PINNED landing buffer. Pinned is what
    #: makes the read one DMA rather than a staging copy, and holding both here is what
    #: keeps a steady step free of allocation: docs/internals/decode/decode.md#growth-probe
    growth: torch.Tensor
    #: The probe's REDUCTION over batch-heads and the step's GATE are the same `[1]`
    #: int32 buffer: the step reads it on the device, the host reads its pinned mirror.
    #: `probe_ctr` is the reducing CTA's self-resetting arrival counter, zeroed exactly
    #: once here: docs/internals/decode/decode.md#the-gate
    growth_any: torch.Tensor
    probe_ctr: torch.Tensor
    growth_host: torch.Tensor
    #: Recorded after the verdict's copy and BEFORE the step is enqueued, so the host's
    #: wait for a 4-byte flag overlaps the launches it used to sit in front of.
    verdict: torch.cuda.Event | None
    #: The probe's OTHER output: `[BH, N/16]` bool, the step's own write-atom set,
    #: condensed on the device from the expansion the verdict was read off. A growth
    #: step admits from THIS -- the host derives nothing: docs/internals/decode/decode.md#growth
    atom_bits: torch.Tensor

    @staticmethod
    def build(config: DecodeGeometry, BH: int, device) -> DecodeScratch:
        i32 = dict(dtype=torch.int32, device=device)
        f32 = dict(dtype=torch.float32, device=device)
        cuda = torch.device(device).type == "cuda"
        #: The kernel's own sizing function answers whether the expansion fits in shared
        #: memory; a host-side mirror of that arithmetic is what would drift.
        per_cta = extension().rola_decode_work_words(
            level_widths=list(config.widths), d_v=config.d_v, decay=config.has_decay,
            device_index=torch.device(device).index or 0) if cuda else 0
        return DecodeScratch(
            work=torch.empty(per_cta * BH * config.n_split, **i32),
            probe_bits=torch.empty(BH, config.n_words, **i32),
            ws=torch.empty(BH, config.n_split, config.cols, **f32),
            ctr=torch.zeros(BH, **i32),
            growth=torch.empty(BH, **i32),
            growth_any=torch.zeros(1, **i32),
            probe_ctr=torch.zeros(1, **i32),
            growth_host=torch.empty(1, dtype=torch.int32, pin_memory=cuda),
            verdict=torch.cuda.Event() if cuda else None,
            atom_bits=torch.empty(BH, config.N // MMA_K_QUANTUM, dtype=torch.bool,
                                  device=device))


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


def fold_levels(levels) -> list[torch.Tensor]:
    """The per-level amplitude stream, in torch: ``[BH, width_l]`` fp32.

    AN ENUMERATED REFERENCE -- ITS ONLY CALLERS ARE GATES. The shipped path folds inside
    the step kernel, from the producer's own views; a gate that fed the kernel its own
    fold's output would prove nothing about the fold
    (docs/internals/decode/decode_fold.md#the-gate).
    """
    return [_fold(t).float() for t in levels]


def guard_no_grad(operands) -> None:
    if not torch.is_grad_enabled():
        return
    if any(t is not None and torch.is_tensor(t) and t.requires_grad for t in operands):
        from rola.interface import BackwardNotImplemented

        raise BackwardNotImplemented(
            "rola decode is forward-only: the single-token arm has no reverse pass -- "
            "the two-pass backward is the chunk arm's. Reaching it from a "
            "grad-requiring context would return a tensor with no grad_fn and silently "
            "sever the graph (the x.grad-is-None failure class). Use eval mode / "
            "torch.no_grad.")


def step_plane(v: torch.Tensor, config: DecodeGeometry, g_read, decay,
               state: torch.Tensor, page_table: torch.Tensor | None) -> torch.Tensor:
    """The plane the kernel indexes, and the per-step agreement checks that find it.

    The two backings have ONE shape rule each. Under a table the plane holds only the
    slots a plan committed, so its leading extent is the arena's and not ``BH``; without
    one the plane IS the leaf space. Both are handed to the ABI in the shape the kernel
    indexes, so the reshape below is the whole difference.
    """
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
    if page_table is None:
        if state.shape != (B, H, config.N, config.cols):
            raise ValueError(
                f"state must be [B, H, N, d_v + 1] = "
                f"{(B, H, config.N, config.cols)}, got {tuple(state.shape)}")
        return state.reshape(BH, config.N, config.cols)
    if state.dim() != 3 or state.shape[1:] != (MMA_K_QUANTUM, config.cols):
        raise ValueError(
            f"under a page table the state is the arena's plane, "
            f"[slots, {MMA_K_QUANTUM}, d_v + 1], got {tuple(state.shape)}")
    if tuple(page_table.shape) != (BH, config.N // MMA_K_QUANTUM):
        raise ValueError(
            f"page_table must be [BH, N / {MMA_K_QUANTUM}] = "
            f"{(BH, config.N // MMA_K_QUANTUM)}, got {tuple(page_table.shape)}")
    return state


def leaf_rate_dials_for(decay, config: DecodeGeometry, device) -> torch.Tensor | None:
    """``[H, sum_l width_l]`` fp32 -- the decay arm's per-HEAD constant, or ``None``.

    It is the ONE operand the step kernel does not fold: a per-HEAD constant rather than a
    per-token one, and its fp32 cast carries a refusal the host has to raise
    (:func:`rola.ops.decay.leaf_rate_dials`).
    """
    if decay is None:
        return None
    return leaf_rate_dials(decay, config).to(device).float()


def growth_probe(write_levels, heads: int, config: DecodeGeometry,
                 page_table: torch.Tensor, scratch: DecodeScratch) -> None:
    """Does this step deposit into an atom the page table does not map? ASKED, not read.

    It folds its own write side from the producer's views -- the same device function the
    step folds with -- so the probe and the step's admission agree about the write support
    by the call graph rather than by an audit.

    THE CONDENSATION IS THE OUTPUT, not an internal: it lands in ``scratch.atom_bits``
    and IS what a growth step admits, so the write-atom set is derived exactly once per
    step, on the device. The verdict lands in ``growth_any``, which the step reads on the
    device (docs/internals/decode/decode.md#the-gate) and the host reads after enqueueing
    it.
    """
    extension().rola_decode_growth_probe(
        write=list(write_levels),
        page_table=page_table,
        work=scratch.work,
        probe_bits=scratch.probe_bits,
        growth=scratch.growth,
        growth_any=scratch.growth_any,
        probe_ctr=scratch.probe_ctr,
        atom_bits=scratch.atom_bits,
        heads=heads,
        n_split=config.n_split,
        level_widths=list(config.widths),
        level_strides=list(config.leaf_stride),
        level_spans=list(config.level_span))


def step(plan: DecodePlan) -> torch.Tensor:
    """THE STEP, in ONE launch, in place on ``plan.state_plane``. ``y`` as ``[B, 1, H, d_v]``.

    Fold, expand and walk are one kernel: the per-token operands are gathered from the
    producer's own views straight into the CTA's shared memory, the support bitmap is
    built there, and the walk consumes both without either ever reaching DRAM
    (docs/internals/decode/decode.md#per-cta-prologue).

    Every touched row has exactly one owning CTA, warp and lane, so the update needs no
    atomic and is bitwise reproducible.
    """
    config = plan.config
    y = extension().rola_decode_forward(
        read=list(plan.read),
        write=list(plan.write),
        normalize=[int(bool(flag)) for flag in plan.normalize],
        dials=plan.dials,
        g_write=plan.g_write,
        v=plan.v,
        state=plan.state_plane,
        work=plan.scratch.work,
        ws=plan.scratch.ws,
        ctr=plan.scratch.ctr,
        level_widths=list(config.widths),
        level_strides=list(config.leaf_stride),
        level_spans=list(config.level_span),
        level_row_offsets=list(config.level_row_offset),
        n_split=config.n_split,
        eps=config.eps,
        page_table=plan.page_table,
        gate=plan.gate,
    )
    B, _T, H, _d_v = plan.v.shape
    return y.view(B, H, 1, config.d_v).permute(0, 2, 1, 3).contiguous()


def _write_atom_bitmap(write_levels, config: DecodeGeometry) -> torch.Tensor:
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
    config: DecodeGeometry,
    state: torch.Tensor,
    *,
    decay=None,
    normalize: tuple[bool, ...] | None = None,
    workspace: DecodeScratch | None = None,
    page_table: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, DecodeScratch]:
    """The RAW-TENSOR entry: the step's own bodies, in the step's order.

    ``v`` is ``[B, 1, H, d_v]``; the level tensors are ``[B, 1, H, width_l]``;
    ``g_write`` is ``[B, 1, H]`` and ``g_read`` is the same or ``None`` under the
    global arm. ``state`` is the dense ``[B, H, N, d_v + 1]`` plane, or the arena's
    ``[slots, 16, d_v + 1]`` one under ``page_table``.

    ``normalize`` is the per-read-level answer to "does this level still carry its own
    mass"; ``None`` is the ordinary case here, a caller already on the simplex. The
    bundle entry reads it off :meth:`~rola.routing.factors.RouteFactors.read_needs_normalization`
    instead.

    Returns ``(y, state, workspace)`` with ``y`` shaped ``[B, 1, H, d_v]``.
    """
    guard_no_grad((v, g_read, g_write, *read_levels, *write_levels,
                   *(decay.dials if decay is not None else ())))
    plane = step_plane(v, config, g_read, decay, state, page_table)
    BH = v.shape[0] * v.shape[2]
    if workspace is None:
        workspace = DecodeScratch.build(config, BH, v.device)
    y = step(DecodePlan(
        config=config, heads=v.shape[2], read=read_levels, write=write_levels,
        normalize=(False,) * config.D if normalize is None else tuple(normalize),
        dials=leaf_rate_dials_for(decay, config, v.device),
        g_write=g_write, v=v, state_plane=plane, page_table=page_table,
        scratch=workspace, gate=gate))
    return y, state, workspace
