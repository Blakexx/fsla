# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The operand planes and the write-atom bitmap: the packing facts, and their bodies.

Every chunk entry reads one `[B, H, T, sumW]` bf16 tensor per side and one atom-major
state plane, and the packing happens ONCE per call so every entry reads the same
bytes. On the shipped path the pack is a VIEW: the producer's solve writes the plane
directly and :func:`_side_plane` reconstructs it from the levels' strides
(docs/internals/chunk/chunk.md#the-side-plane). The copy survives for the sides that
are not already that layout -- a hand-built bundle, a foreign producer.
"""
from __future__ import annotations

import torch

from rola.engine import MMA_K_QUANTUM


def _side_plane(levels) -> torch.Tensor | None:
    """The `[B, H, T, sumW]` bf16 plane these levels are already views OF, or None.

    A STRIDE fact, not a provenance one: the levels are the plane iff each is the
    window of one bf16 allocation whose layout is exactly this side's, which is
    checkable and is what a hand-built bundle either satisfies or does not
    (docs/internals/chunk/chunk.md#the-side-plane).
    """
    batch, tokens, heads, _ = levels[0].shape
    widths = [level.shape[-1] for level in levels]
    total = sum(widths)
    want = (heads * tokens * total, total, tokens * total, 1)
    base = levels[0]
    offset = 0
    for level, width in zip(levels, widths):
        if (level.dtype is not torch.bfloat16
                or level.untyped_storage().data_ptr() != base.untyped_storage().data_ptr()
                or level.stride() != want
                or level.storage_offset() != base.storage_offset() + offset):
            return None
        offset += width
    span = base.storage_offset() + batch * heads * tokens * total
    if span * base.element_size() > base.untyped_storage().nbytes():
        return None
    return base.as_strided((batch, heads, tokens, total),
                           (heads * tokens * total, tokens * total, total, 1),
                           base.storage_offset())


def pack_side(levels) -> torch.Tensor:
    """The per-level `[B, T, H, width_l]` tuple as one `[B, H, T, sumW]` bf16 tensor.

    ONE destination and one pass per level: concatenating in the public layout
    materializes an intermediate no kernel reads, and the transpose and the
    narrowing are then a second pass over it. A side that is already the plane skips
    all of it -- the producer emits this layout.
    """
    plane = _side_plane(levels)
    if plane is not None:
        return plane
    batch, tokens, heads, _ = levels[0].shape
    widths = [level.shape[-1] for level in levels]
    packed = torch.empty((batch, heads, tokens, sum(widths)),
                         dtype=torch.bfloat16, device=levels[0].device)
    offset = 0
    for level, width in zip(levels, widths):
        packed[:, :, :, offset:offset + width].copy_(level.permute(0, 2, 1, 3))
        offset += width
    return packed


def state_plane(routes, *, d_v: int, zeros: bool = False) -> torch.Tensor:
    """A `[BH, N/16, page_bytes]` uint8 state plane for this routing: one int8-format page an atom, in canonical
    atom order (`rola.ops.paging.page_layout`). An all-zero page is the zero state.

    Allocation is the caller's, not the kernel entry's -- which is what lets a
    carried state be handed straight back in, and a CONTINUATION does exactly that
    rather than reaching here at all (docs/internals/state.md#the-dense-continuation).
    """
    from rola.ops.paging import page_layout

    widths = tuple(level.width for level in routes.topology.levels)
    n = 1
    for w in widths:
        n *= w
    shape = (routes.batch * routes.heads, n // MMA_K_QUANTUM, page_layout(d_v).bytes)
    build = torch.zeros if zeros else torch.empty
    return build(shape, device=routes.device, dtype=torch.uint8)


#: THE TWO ORTHOGONAL BITS the words' page fold publishes per atom
#: (`rola.engine.facts.liveness.page_bits`); the mirror of
#: `csrc/rola/src/common/ops.cuh`'s `kAtomWritten`/`kAtomRead`, which is what the
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
