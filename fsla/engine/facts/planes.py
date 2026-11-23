# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The state plane and the atom bitmap: the packing facts, and their bodies."""
from __future__ import annotations

import torch

from fsla.engine.types import MMA_K_QUANTUM
from fsla.routing.masks import PREFILL


def state_plane(routes, *, d_v: int, zeros: bool = False) -> torch.Tensor:
    """A `[BH, N/16, page_bytes]` uint8 state plane for this routing: one int8-format page an atom, in canonical
    atom order (`fsla.ops.paging.page_layout`). An all-zero page is the zero state.

    Allocation is the caller's, not the kernel entry's -- which is what lets a
    carried state be handed straight back in, and a CONTINUATION does exactly that
    rather than reaching here at all (docs/internals/state.md#the-dense-continuation).
    """
    from fsla.ops.paging import page_layout

    widths = tuple(level.width for level in routes.topology.levels)
    n = 1
    for w in widths:
        n *= w
    shape = (routes.batch * routes.heads, n // MMA_K_QUANTUM, page_layout(d_v).bytes)
    build = torch.zeros if zeros else torch.empty
    return build(shape, device=routes.device, dtype=torch.uint8)


#: THE TWO ORTHOGONAL BITS the atom fold publishes per atom
#: (:func:`atom_bits`); the mirror of
#: `csrc/fsla/src/common/ops.cuh`'s `kAtomWritten`/`kAtomRead`, which is what the
#: kernel derives `load iff resident and (read or written)` / `store iff written` from.
ATOM_WRITTEN = 1
ATOM_READ = 2


def written_atoms(bits: torch.Tensor) -> torch.Tensor:
    """The WRITE set as a bool tensor: the page plan's input, and only ever that.

    Residency is keyed on the atoms a call WRITES -- reads of never-written state are
    inert by the self-normalizing readout -- so a planner that took the whole activity
    byte would admit read-only atoms and turn exact commitment into a superset.
    """
    return (bits & ATOM_WRITTEN).bool()


#: The page granule, in leaves: an atom is 16 consecutive leaves in canonical order.
ATOM_LEAVES = 16


def _atom_spans(widths: tuple[int, ...]) -> tuple[int, ...]:
    """THE ATOM AS A BOX: the per-level spans of the trailing 16 canonical leaves. Canonical order is the radix over the
    levels with the last one minor, so an atom IS an aligned rectangle: the last levels whole until their product
    reaches 16, and an aligned run in the level that completes it."""
    spans, remaining = [1] * len(widths), ATOM_LEAVES
    for level in reversed(range(len(widths))):
        span = min(widths[level], remaining)
        if remaining % span or widths[level] % span:
            raise ValueError(f"level {level}'s width {widths[level]} does not tile the "
                             f"{remaining} leaves the page atom still needs")
        spans[level], remaining = span, remaining // span
        if remaining == 1:
            break
    if remaining != 1:
        raise ValueError(f"the topology holds fewer than one page atom of leaves (widths {widths})")
    return tuple(spans)


def _side_atoms(level_masks, widths: tuple[int, ...], spans: tuple[int, ...]) -> torch.Tensor:
    """``[B, H, N/16]`` bool: atom ``a`` is live for head ``h`` when some token has, at every level, a live digit inside
    the atom's run (the OR over tokens of the AND over levels of the OR over the run), the digits at or past a level's
    logical width being the pad, dead."""
    clause = None
    for bits, width, span in zip(level_masks, widths, spans, strict=True):
        B, T, H, w = bits.shape
        bits = torch.nn.functional.pad(bits, (0, width - w))
        run = bits.reshape(B, T, H, width // span, span).any(dim=-1)
        clause = run if clause is None else (clause[..., :, None] & run[..., None, :]).reshape(B, T, H, -1)
    return clause.any(dim=1)


def atom_bits(masks, padded_widths: tuple[int, ...], tokens: int) -> torch.Tensor:
    """`[B H_w, N/16]` uint8: the EXACT activity byte, `ATOM_READ | ATOM_WRITTEN`, a write head's (its state's), folded
    from the PREFILL-layout masks and nothing else: READ by any of its ``g = H / H_w`` query heads (the read side's heads
    run ``g`` to a write head), WRITTEN by its write side. It is what `plan_exact` allocates on and what a residency
    check reads."""
    if masks.consumer != PREFILL:
        raise ValueError(f"the atom fold reads the {PREFILL!r}-layout masks, got {masks.consumer!r}")
    widths = tuple(padded_widths)
    spans = _atom_spans(widths)
    read, write = (
        _side_atoms(tuple(masks.bits(side, level, tokens, width) for level, width in enumerate(widths)), widths, spans)
        for side in ("read", "write"))
    B, H, atoms = read.shape
    heads = write.shape[1]
    read = read.reshape(B, heads, H // heads, atoms).any(dim=2)
    return (read.to(torch.uint8) * ATOM_READ | write.to(torch.uint8) * ATOM_WRITTEN).reshape(B * heads, atoms)
