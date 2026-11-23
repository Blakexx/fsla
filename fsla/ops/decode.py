# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""The T=1 decode family's BODIES: the geometry, the scratch, the launches.

The decode step (:mod:`fsla.engine.decode_step`) calls these in order; the facade lives there with it. What is here is one function per thing that is
done, and the raw-tensor entry :func:`_decode_step` is those same functions in the
step's order, for a caller holding tensors rather than a bundle.

THE STEP IS ONE LAUNCH. Every per-token operand is folded inside the step kernel, from
the producer's own views, so there is no operand plane between them
(docs/internals/decode/decode_fold.md).

The bypass is architectural rather than a cheap default
(docs/internals/decode/decode.md#no-stage-2), the support is a Kronecker product the
kernel WALKS from its factors rather than materializing
(docs/internals/decode/decode.md#kronecker-support), and decode's decay domain is
strictly larger than the chunk arm's as a CONSEQUENCE
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
from functools import cache

import torch

from fsla.engine.plan import DecodeLaunch, DecodeOperands, DecodeWorkspace
from fsla.ops._ext import extension
from fsla.ops.constants import READOUT_EPS
from fsla.ops.decay import leaf_rate_dials
from fsla.ops.descriptor import Descriptor
from fsla.ops.lattice import derive_lattice, permutation
from fsla.ops.padding import pad_v, slice_y
from fsla.ops.paging import MMA_K_QUANTUM, page_layout
from fsla.routing.masks import DECODE

__all__ = [
    "DecodeScratch",
    "arms",
    "lattice_of",
    "leaf_rate_dials_for",
    "step",
    "step_plane",
]


def arms() -> tuple[tuple[int, int, bool], ...]:
    """The decode arms THIS BINARY carries, as ``(d_v, D, decay)`` rows.

    A full build carries the whole declared matrix (``d_v x D x decay`` = 16 rows). An
    ITERATION build names a narrowed contract with ``--//:contract=<name>``, carries the
    arms that contract reaches and is NOT SHIPPABLE; this is how such a binary identifies
    itself, and the launch refuses a call outside the narrowed contract, naming it
    (``docs/build.md#narrowed``).
    """
    return tuple((int(dv), int(d), bool(decay)) for dv, d, decay in extension().fsla_decode_arms())


#: THE PHASE LEDGER'S SHAPE: `[ctas][warps][phases]`, the kernel's own `DecodePhase` order.
LEDGER_PHASES = ("fold", "factor", "admit", "walk", "combine")


def ledger_warps() -> int:
    """The phase ledger's warps: the step's plan's at the current card's build, as the C++ entry derives them."""
    return int(extension().fsla_decode_ledger_warps())


def ledger_bind(ledger: torch.Tensor | None) -> None:  #: reserved: the measurement instrument (docs/internals/decode/decode.md#phase-ledger)
    """Bind the phase ledger the next steps book into, ``[ctas, ledger_warps(), 5]`` int64 cycles, or ``None``."""
    extension().fsla_decode_ledger_bind(ledger)


@dataclass(frozen=True, slots=True)
class DecodeLaunchFacts:
    """What the planner's launch function makes of a descriptor that Python reads (``common/plan/decode_launch.h``): the arm (``d_v``,
    ``levels``, ``decay``), the zero columns the call's value width is padded by, the workspace the call needs (bytes, alignment) and
    each level's width (``widths``, one a level)."""

    d_v: int
    levels: int
    decay: int
    pad: int
    workspace_bytes: int
    workspace_alignment: int
    widths: tuple[int, ...]

    @property
    def N(self) -> int:
        return math.prod(self.widths)

    @property
    def cols(self) -> int:
        return self.d_v + 1  # the mass column is unconditional (raw is dead)


@cache
def decode_launch(descriptor: Descriptor) -> DecodeLaunchFacts:
    """The planner's decode launch for ``descriptor``, on its own device; the padded value width is the arm's ``d_v``."""
    words = [int(x) for x in extension().fsla_decode_launch(list(descriptor.words), descriptor.device)]
    return DecodeLaunchFacts(*words[:6], widths=tuple(words[6:6 + words[1]]))


@cache
def lattice_of(widths: tuple[int, ...]) -> tuple[int, int]:
    """The ``(k, m)`` box the state plane's leaves are ordered by, derived from the state's format
    (:func:`fsla.ops.lattice.derive_lattice`, the one authority); the leaf order travels with the plane."""
    return derive_lattice(widths)


@dataclass(slots=True)
class DecodeScratch:
    """Per-sequence scratch, allocated once and reused across steps.

    ``yacc``, ``queue``, ``growth_ctr`` and ``done`` are set exactly once here: every one of them
    is reset by the kernel that last reads it (a head's finisher, the last CTA out), so no
    ``memset`` launch ever stands between two decode steps and the whole step stays capturable
    in a CUDA graph.
    """

    #: `[BH, cols]` fp32, a head's readout sums, added into once by every CTA visit that walks it.
    yacc: torch.Tensor
    #: `[4 BH + 2]` int32, THE WORK QUEUE: per head the positions claimed, the positions finished,
    #: the position count (-1 until published) and the admission state, then the unstarted-head
    #: cursor and the exit count (docs/internals/decode/decode.md#queue).
    queue: torch.Tensor
    #: `[BH]` int32, THE PER-BATCH-HEAD VERDICT, and the `[1]` int32 OR of it that the
    #: host reads through a PINNED mirror -- pinned so the read is one DMA rather than a
    #: staging copy, and held here so a steady step allocates nothing
    #: (docs/internals/decode/decode.md#per-bh-verdict).
    growth: torch.Tensor
    growth_any: torch.Tensor
    #: The reducing CTA's self-resetting arrival counter.
    growth_ctr: torch.Tensor
    growth_host: torch.Tensor
    #: `[BH]` int32, THE DONE FLAGS: which batch-heads have already deposited this step.
    #: A replay re-walks a done batch-head for its `y` and deposits nothing, which is what
    #: makes the caller's replay advance every sequence EXACTLY ONCE
    #: (docs/internals/decode/decode.md#done-flags).
    done: torch.Tensor
    #: `[BH, N/16]` bool, the step's own write-atom set, condensed on the device from the
    #: expansion the verdict was read off. A host admission plans from THIS -- the host
    #: derives nothing: docs/internals/decode/decode.md#atom-bits
    atom_bits: torch.Tensor

    @staticmethod
    def build(descriptor: Descriptor, BH: int, device) -> DecodeScratch:
        launch = decode_launch(descriptor)
        i32 = dict(dtype=torch.int32, device=device)
        f32 = dict(dtype=torch.float32, device=device)
        cuda = torch.device(device).type == "cuda"
        return DecodeScratch(
            yacc=torch.zeros(BH, launch.cols, **f32),
            queue=_fresh_queue(BH, device),
            growth=torch.zeros(BH, **i32),
            growth_any=torch.zeros(1, **i32),
            growth_ctr=torch.zeros(1, **i32),
            growth_host=torch.empty(1, dtype=torch.int32, pin_memory=cuda),
            done=torch.zeros(BH, **i32),
            atom_bits=torch.empty(BH, launch.N // MMA_K_QUANTUM, dtype=torch.bool,
                                  device=device))


def _fresh_queue(BH: int, device) -> torch.Tensor:
    """The queue as a step finds it: counters zero, every position count unpublished (-1)."""
    queue = torch.zeros(4 * BH + 2, dtype=torch.int32, device=device)
    queue[2 * BH:3 * BH] = -1
    return queue


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
    """The per-level RAW coefficient values, in torch: ``[BH, width_l]`` fp32, each stored u8 digit as its integer, the
    values the step kernel's fold sums and scales.

    AN ENUMERATED REFERENCE -- ITS ONLY CALLERS ARE GATES. The shipped path folds inside
    the step kernel, from the producer's own views; a gate that fed the kernel its own
    fold's output would prove nothing about the fold
    (docs/internals/decode/decode_fold.md#the-gate).
    """
    return [_fold(rows).float() for rows in levels]


def check_descriptor(descriptor: Descriptor, routes, device: torch.device) -> None:
    """The descriptor against the call it is handed: a decode step's (its ``prefill`` fact 0), the level widths it states are the bundle's, its
    ``batch_heads`` is the bundle's B times H, and its device is the values'."""
    widths = tuple(routes.topology.widths)
    if descriptor["prefill"]:
        raise ValueError("the descriptor is a prefill's (its `prefill` fact is 1); fsla.ops.produce.decode_descriptor builds a decode step's")
    if widths != decode_launch(descriptor).widths:
        raise ValueError(f"the descriptor is for widths {decode_launch(descriptor).widths}, the bundle's are {widths}")
    if descriptor["batch_heads"] != routes.batch * routes.heads:
        raise ValueError(f"the descriptor states batch_heads={descriptor['batch_heads']}; the bundle's B times H is "
                         f"{routes.batch} * {routes.heads} = {routes.batch * routes.heads}: build the descriptor from this call "
                         "(fsla.ops.produce.decode_descriptor)")
    if descriptor.device != device.index:
        raise ValueError(f"the descriptor was made for device {descriptor.device}; the values are on {device}")


def check_decode_masks(masks, routes) -> None:
    """The decode step's masks are the DECODE layout, digit-packed int32 ``[B, H, ceil(width_l / 32)]`` a side a level,
    validated for shape and dtype only (:meth:`~fsla.routing.masks.Masks.check`)."""
    if masks.consumer != DECODE:
        raise ValueError(f"the decode step takes the {DECODE!r}-layout masks, got {masks.consumer!r}")
    masks.check(routes)


def guard_no_grad(operands) -> None:
    if not torch.is_grad_enabled():
        return
    if any(t is not None and torch.is_tensor(t) and t.requires_grad for t in operands):
        from fsla.interface import BackwardNotImplemented

        raise BackwardNotImplemented(
            "fsla decode is forward-only: the single-token arm has no reverse pass -- "
            "the two-pass backward is the chunk arm's. Reaching it from a "
            "grad-requiring context would return a tensor with no grad_fn and silently "
            "sever the graph (the x.grad-is-None failure class). Use eval mode / "
            "torch.no_grad.")


def step_plane(v: torch.Tensor, descriptor: Descriptor, decay,
               state: torch.Tensor, page_table: torch.Tensor | None) -> torch.Tensor:
    """The plane the kernel indexes, and the per-step agreement checks that find it.

    The two backings have ONE shape rule each. Under a table the plane holds only the
    slots a plan committed, so its leading extent is the arena's and not ``BH``; without
    one the plane IS the leaf space. Both are handed to the ABI in the shape the kernel
    indexes, so the reshape below is the whole difference.
    """
    if v.dim() != 4:
        raise ValueError(f"v must be [B, 1, H, d_v], got {tuple(v.shape)}")
    launch = decode_launch(descriptor)
    B, _T, H, d_v = v.shape
    BH = B * H
    if d_v != launch.d_v:
        raise ValueError(f"v's d_v={d_v} disagrees with the descriptor's arm width {launch.d_v}")
    if (decay is not None) != bool(launch.decay):
        raise ValueError(
            f"the descriptor says decay={launch.decay} but decay is "
            f"{'absent' if decay is None else 'present'}; the descriptor is for the "
            "sequence and the decay arm is part of it")
    page_bytes = page_layout(d_v).bytes
    if page_table is None:
        #: PREFILL AND DECODE SHARE ONE ADMISSION LAW (Blake, 2026-08-29): the dense
        #: plane is the same page sequence with every page resident, and `N` is page
        #: aligned by construction (docs/internals/state.md#format).
        expected = (BH, launch.N // MMA_K_QUANTUM, page_bytes)
        if tuple(state.shape) != expected:
            raise ValueError(f"a dense state is [BH, N / {MMA_K_QUANTUM}, page_bytes] = {expected}, "
                             f"got {tuple(state.shape)}")
        return state
    if state.dim() != 2 or state.shape[1] != page_bytes:
        raise ValueError(
            f"under a page table the state is the arena's plane, [slots, {page_bytes}], got {tuple(state.shape)}")
    if tuple(page_table.shape) != (BH, launch.N // MMA_K_QUANTUM):
        raise ValueError(
            f"page_table must be [BH, N / {MMA_K_QUANTUM}] = "
            f"{(BH, launch.N // MMA_K_QUANTUM)}, got {tuple(page_table.shape)}")
    return state


def leaf_rate_dials_for(decay, descriptor: Descriptor, device) -> torch.Tensor | None:
    """``[H, sum_l width_l]`` fp32 -- the decay arm's per-HEAD constant, or ``None``.

    It is the ONE operand the step kernel does not fold: a per-HEAD constant rather than a
    per-token one, and its fp32 cast carries a refusal the host has to raise
    (:func:`fsla.ops.decay.leaf_rate_dials`).
    """
    if decay is None:
        return None
    return leaf_rate_dials(decay, decode_launch(descriptor).levels).to(device).float()


def step(launch: DecodeLaunch, *, ctas: int | None = None, eps: float = READOUT_EPS) -> torch.Tensor:  #: reserved: ``ctas`` is a gate's forced CTA count (None is the launch function's own grid, docs/internals/decode/decode.md#the-split-policy), ``eps`` the readout's constant
    """THE STEP, in ONE launch, in place on ``launch.operands.state_plane``. ``y`` as ``[B, 1, H, d_v]``.

    Fold, expand and walk are one kernel: the per-token operands are gathered from the
    producer's own views straight into the CTA's shared memory, the digit masks and owner
    lists are built there, and the walk consumes both without either ever reaching DRAM
    (docs/internals/decode/decode.md#per-cta-prologue).

    Every written leaf has exactly one depositing lane group, so the state update needs no
    atomic and is bitwise reproducible; ``y`` is reproducible to the order the visits add
    its sums in (docs/internals/decode/decode.md#determinism).

    THE ADMISSION QUESTION IS PART OF IT: the write map the walk needs is the map the
    residency test reads, so the test is a pass over shared memory rather than a launch,
    and the verdict it publishes is per batch-head
    (docs/internals/decode/decode.md#absorbed-verdict).
    """
    op, pool = launch.operands, launch.workspace.pool
    scratch = launch.workspace.scratch
    y = extension().fsla_decode_forward(
        read=list(op.read),
        write=list(op.write),
        read_scale=list(op.read_scale),
        write_scale=list(op.write_scale),
        read_mask=list(op.read_mask),
        write_mask=list(op.write_mask),
        dials=op.dials,
        v=op.v,
        v_scale=op.v_scale,
        state=op.state_plane,
        yacc=scratch.yacc,
        queue=scratch.queue,
        growth=scratch.growth,
        growth_any=scratch.growth_any,
        growth_ctr=scratch.growth_ctr,
        done=scratch.done,
        atom_bits=scratch.atom_bits,
        lattice_k=op.state_lattice[0],
        lattice_m=op.state_lattice[1],
        ctas=-1 if ctas is None else ctas,
        eps=eps,
        descriptor=list(launch.descriptor.words),
        page_table=op.page_table,
        pool_slots=None if pool is None else pool.pool_slots,
        pool_cursor=None if pool is None else pool.pool_cursor,
    )
    B, _T, H, d_v = op.v.shape
    return y.view(B, H, 1, d_v).permute(0, 2, 1, 3).contiguous()


def _write_atom_bitmap(write_live, widths: tuple[int, ...], lattice: tuple[int, int]) -> torch.Tensor:
    """``[BH, N / 16]`` bool: the atoms this ONE-TOKEN step deposits into, EXACTLY, from each write level's live
    digits: the write masks' bits (:meth:`~fsla.routing.masks.Masks.bits`), ``[B, 1, H, width_l]`` bool a level.

    AN ENUMERATED REFERENCE -- ITS ONLY CALLERS ARE GATES. Deleting it would delete the
    gate's reference; calling it on a shipped path would delete the gate's independence
    (docs/internals/decode/decode.md#second-derivation). It derives the CANONICAL leaf
    support digit by digit and crosses into lattice order through ``pi``, so it shares no
    step of the kernel's own enumeration.
    """
    device = write_live[0].device
    index = torch.arange(math.prod(widths), device=device)
    live = None
    for level, width in enumerate(widths):
        digit = (index // math.prod(widths[level + 1:])) % width
        column = _fold(write_live[level])[:, digit]
        live = column if live is None else (live & column)
    pi = permutation(widths, *lattice, device=device)
    lattice_live = torch.empty_like(live)
    lattice_live.index_copy_(1, pi, live)
    return lattice_live.view(-1, math.prod(widths) // MMA_K_QUANTUM, MMA_K_QUANTUM).any(-1)


def _decode_step(
    routes,
    values,
    masks,
    descriptor: Descriptor,
    state: torch.Tensor,
    *,
    decay=None,
    workspace: DecodeScratch | None = None,
    page_table: torch.Tensor | None = None,
    pool=None,
    ctas: int | None = None,  #: reserved: a gate's forced CTA count, the step's own keyword
) -> tuple[torch.Tensor, torch.Tensor, DecodeScratch]:
    """THE BUNDLE ENTRY for one token: the step's own bodies, in the step's order.

    ``routes`` is a one-token :class:`~fsla.routing.factors.RouteFactors`, ``values`` a one-token
    :class:`~fsla.values.Values` and ``masks`` their DECODE-layout :class:`~fsla.routing.masks.Masks`; the kernel
    reads their rows, scales, values and mask words as they are. ``state`` is the dense
    ``[BH, N / 16, page_bytes]`` plane, or the arena's ``[slots, page_bytes]`` one under ``page_table``.

    Returns ``(y, state, workspace)`` with ``y`` shaped ``[B, 1, H, d_v]`` fp32.

    ``descriptor`` ALREADY HOLDS THE PADDED ``DV`` (:func:`decode_launch` maps the call's ``d_v`` to its arm's), which is what
    makes this the SEAM for decode's value width: the values are padded to the arm's width with zero columns and the returned
    ``y`` is sliced back to the caller's ``d_v`` (``fsla.ops.padding``). Per-level widths are NOT padded -- the descriptor
    refuses a width outside the contract when it is made, so the rows reach the kernel exactly as the bundle holds them.
    """
    guard_no_grad((*routes.tensors(), *values.tensors(), *(decay.dials if decay is not None else ())))
    check_decode_masks(masks, routes)
    routes.refuse_grouped("the decode step")
    if routes.tokens != 1:
        raise ValueError(f"the decode step takes one token, and a one-token bundle is all rows: the routes are "
                         f"[B, T={routes.tokens}, H]")
    check_descriptor(descriptor, routes, values.v.device)
    widths = tuple(routes.topology.widths)
    logical_d_v = values.d_v
    launch = decode_launch(descriptor)
    v = pad_v(values.v, launch.d_v)
    plane = step_plane(v, descriptor, decay, state, page_table)
    BH = values.batch * values.heads
    if workspace is None:
        workspace = DecodeScratch.build(descriptor, BH, v.device)
    y = step(DecodeLaunch(
        descriptor=descriptor,
        operands=DecodeOperands(
            read=routes.amp_read, write=routes.amp_write, read_scale=routes.scale_read, write_scale=routes.scale_write,
            read_mask=masks.read, write_mask=masks.write, dials=leaf_rate_dials_for(decay, descriptor, v.device), v=v,
            v_scale=values.scale, state_plane=plane, state_lattice=lattice_of(widths), page_table=page_table),
        workspace=DecodeWorkspace(scratch=workspace, pool=pool)), ctas=ctas)
    return slice_y(y, logical_d_v), state, workspace
