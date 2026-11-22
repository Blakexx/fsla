"""Gate for the ATOM FOLD (`fsla.engine.facts.planes.atom_bits`): the page byte a call's masks imply.

The subject is a pure fold over torch tensors, so the file gates without the built extension and without a GPU. The
truth it is held to is brute force: the leaves each token's live digits name, the atoms they fall in.

Symbols: ``D`` routing depth; ``B_l`` the PADDED per-level width; ``L`` tokens; ``N = prod_l B_l`` leaves; atom = 16
canonical leaves = the page granule; ``H`` query heads, ``H_w`` write heads, ``g = H / H_w``.
"""
from __future__ import annotations

import pytest
import torch

from fsla.engine.facts.planes import ATOM_LEAVES, ATOM_READ, ATOM_WRITTEN, atom_bits, written_atoms
from fsla.routing.masks import DECODE, PREFILL, Masks, pack_tokens

#: (B_l per level, L): a padded one-level, a two-level and a three-level topology, an L that is not a whole number of
#: words, and a level whose run splits an atom.
_SHAPES = (((32,), 32), ((16, 16), 64), ((32, 16), 40), ((16, 16, 16), 96), ((64, 16), 33))


def _draw(widths, L, heads, *, logical=None, seed=0):
    """Each level's drawn bool bits ``[1, L, heads, b_l]`` at its logical width, a random support with at least one live
    digit per token."""
    logical = tuple(widths) if logical is None else tuple(logical)
    generator = torch.Generator().manual_seed(seed)
    out = []
    for width in logical:
        live = torch.rand((1, L, heads, width), generator=generator) < 0.3
        live[..., 0] |= ~live.any(dim=-1)
        out.append(live)
    return tuple(out)


def _brute(bits, widths):
    """``[1, heads, N/16]`` bool: an atom is live for a head when some token's live leaves reach it."""
    L, heads = bits[0].shape[1:3]
    strides, suffix = [], 1
    for width in reversed(widths):
        strides.append(suffix)
        suffix *= width
    strides.reverse()
    out = torch.zeros((1, heads, suffix // ATOM_LEAVES), dtype=torch.bool)
    for head in range(heads):
        for token in range(L):
            leaves = torch.zeros(1, dtype=torch.int64)
            for level, block in enumerate(bits):
                digits = block[0, token, head].nonzero().flatten()
                leaves = (leaves[:, None] + (digits * strides[level])[None, :]).reshape(-1)
            out[0, head, torch.unique(leaves // ATOM_LEAVES)] = True
    return out


@pytest.mark.parametrize(("widths", "L"), _SHAPES)
@pytest.mark.parametrize("groups", (1, 2))
def test_the_atom_byte_is_the_brute_force_set_of_atoms_the_live_leaves_reach(widths, L, groups):
    """EXACTNESS: WRITTEN is the write head's own atoms, READ the union over its ``g`` query heads'."""
    heads = 2
    read_bits = _draw(widths, L, heads * groups, seed=1)
    write_bits = _draw(widths, L, heads, seed=2)
    masks = Masks(PREFILL, tuple(pack_tokens(b) for b in read_bits), tuple(pack_tokens(b) for b in write_bits))
    got = atom_bits(masks, widths, L)
    read = _brute(read_bits, widths).reshape(1, heads, groups, -1).any(dim=2)
    write = _brute(write_bits, widths)
    want = (read.to(torch.uint8) * ATOM_READ | write.to(torch.uint8) * ATOM_WRITTEN).reshape(heads, -1)
    assert got.dtype is torch.uint8
    assert torch.equal(got, want)
    assert torch.equal(written_atoms(got), write.reshape(heads, -1))


def test_pad_digits_are_dead_by_construction():
    """A level's digits at or past its logical width are the pad: a level 6 wide in a 16 slot reaches only its own."""
    widths, L = (16, 16), 32
    bits = _draw(widths, L, 1, logical=(16, 6), seed=3)
    masks = Masks(PREFILL, tuple(pack_tokens(b) for b in bits), tuple(pack_tokens(b) for b in bits))
    got = atom_bits(masks, widths, L)
    want = _brute(tuple(torch.nn.functional.pad(b, (0, 16 - b.shape[-1])) for b in bits), widths)
    assert torch.equal(written_atoms(got), want.reshape(1, -1))


def test_the_fold_reads_the_prefill_layout_and_refuses_the_decode_one_by_name():
    widths = (16, 16)
    packed = tuple(torch.zeros((1, 1, 1), dtype=torch.int32) for _ in widths)
    with pytest.raises(ValueError, match="layout masks"):
        atom_bits(Masks(DECODE, packed, packed), widths, 1)


def test_a_topology_with_fewer_leaves_than_an_atom_is_refused():
    bits = _draw((8,), 32, 1)
    masks = Masks(PREFILL, (pack_tokens(bits[0]),), (pack_tokens(bits[0]),))
    with pytest.raises(ValueError, match="page atom"):
        atom_bits(masks, (8,), 32)
