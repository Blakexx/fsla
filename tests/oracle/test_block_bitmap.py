"""The liveness epilogue emits the EXACT block-grain read/write/state facts.

This is the sparse grid's predicate INPUT. If a bit is a SUPERSET the grid merely
does work it did not have to; if a bit is a SUBSET the grid drops a block that
contributes, and the operator returns a wrong answer with no other symptom. So
exactness is asserted as `torch.equal` against an independent reference in BOTH
directions, never as a containment.

Vocabulary, defined here and used unqualified below: `D` = level count;
`B` = one uniform level width; `N = B ** D` = leaves; `BC` = the launch's block
width in leaves; `blocks = N // BC`; `L`/`T` = tokens; `BH` = batch * heads;
**atom** = the 16-leaf page granule (`rola.ops.paging.MMA_K_QUANTUM`). The bitmap
is `[BH, blocks]` uint8 with bits `READ = 1`, `WRITE = 2`, `STATE = 4`.

**THE REFERENCE, and why it is independent.** READ and WRITE are checked against
an fp64 recomputation over the ORACLE's own leaf materialization
(`rola.ops.naive._leaf_product` over `rola.ops.naive._radix_strides`), which
derives leaf addresses separately from production -- the same reference
`test_atom_bitmap.py` uses, and for the same reason: a fresh reference written
beside the kernel proves self-consistency and nothing else.

**The exactness argument being tested.** The kernel evaluates, per token word,
the AND over the block's covering entry rows and ORs that over tokens. That is
exact at the block's own grain rather than a bound because a block is an aligned
digit-range rectangle: any combination of digits live at one token, one per
level, inside the rectangle IS a leaf of the rectangle.
`test_two_token_cross_product_is_excluded_at_block_grain` is the cell where the
loose form (OR over tokens first) disagrees, with the wrongly admitted blocks
named.

**The cross-grain gate.** `test_write_bit_agrees_with_the_atom_bitmap` joins this
reduction to P67's atom-grain one: two evaluations of the same conjunction at two
grains, which must agree by construction. It is the cheapest check that a
predicate bug in either has to survive.
"""
from __future__ import annotations

from math import prod

import pytest
import torch

from rola.engine.facts import planes
from rola.engine.types import Mask
from rola.ops._ext import extension
from rola.ops.naive import _leaf_product, _radix_strides
from rola.ops.paging import MMA_K_QUANTUM
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    SoftmaxActivation,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the epilogue is a CUDA kernel"),
]

#: The built `(D, B, BC)` grains, projected off the manifest itself so a new arm
#: is covered without editing this list. The chunk length is not an axis of the
#: reduction: it reduces the `(D, B, BC)` union table.
_GRAINS = sorted({(a[2], a[3], a[1]) for a in extension().chunk_arms()})

_SUPPORTS = {"sparse": 0.15, "mixed": 0.7, "dense": 1.0}

_ROUTING = IndependentRouting(
    width=1, tied=False, read=SoftmaxActivation(), write=EntmaxActivation(1.5))

_B, _T, _H, _DV, _C = 2, 64, 2, 64, 32

READ, WRITE, STATE = 1, 2, 4


def _simplex(shape, p_nonzero, generator):
    """Rows on the simplex with EXACT support zeros -- entmax's own output shape."""
    x = torch.rand(shape, device="cuda", dtype=torch.float64, generator=generator)
    if p_nonzero < 1.0:
        keep = torch.rand(shape, device="cuda", dtype=torch.float64,
                          generator=generator) < p_nonzero
        x = x * keep
    x = torch.where(x.sum(-1, keepdim=True) == 0, torch.ones_like(x), x)
    return (x / x.sum(-1, keepdim=True)).to(torch.float32)


def _pack(levels):
    return torch.cat(levels, -1).permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()


def _bits(read, write, widths, bc, page_tbl=None):
    pr, pw = _pack(read), _pack(write)
    for side in (read, write):
        flat = torch.cat(side, -1)
        assert torch.equal(flat != 0, flat.to(torch.bfloat16) != 0), (
            "fixture amplitudes flushed to zero in bf16; the comparison would be "
            "about the cast, not about the bitmap")
    T = read[0].shape[1]
    tbl = extension().chunk_facts(pr, pw, _C, bc, list(widths), Mask.M1, False)[0]
    return extension().chunk_block_bits(tbl, bc, list(widths), T, page_tbl)


def _oracle_blocks(levels, widths, bc):
    """THE REFERENCE: the block is live iff at SOME token one of its leaves has a
    nonzero fp64 leaf product. Built from the oracle's address map."""
    B, T, H = levels[0].shape[:3]
    N = prod(widths)
    levels64 = tuple(level.to(torch.float64) for level in levels)
    leaves = _leaf_product(levels64, _radix_strides(tuple(widths)), N)  # [B,T,H,N]
    live = (leaves != 0).permute(0, 2, 1, 3).reshape(B * H, T, N).any(dim=1)
    return live.view(B * H, N // bc, bc).any(dim=-1)


def _fixture(widths, p_nonzero, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    read = tuple(_simplex((_B, _T, _H, w), p_nonzero, g) for w in widths)
    write = tuple(_simplex((_B, _T, _H, w), p_nonzero, g) for w in widths)
    return read, write


@pytest.mark.parametrize("support", list(_SUPPORTS), ids=list(_SUPPORTS))
@pytest.mark.parametrize("grain", _GRAINS, ids=lambda g: f"d{g[0]}b{g[1]}bc{g[2]}")
def test_read_and_write_bits_are_exact(grain, support):
    """Exact tensor equality against the independent reference, per side, per grain."""
    D, B, bc = grain
    widths = (B,) * D
    read, write = _fixture(widths, _SUPPORTS[support], seed=D * 10000 + B * 100 + bc)
    bits = _bits(read, write, widths, bc)
    assert bits.shape == (_B * _H, prod(widths) // bc)
    for name, mask, levels in (("read", READ, read), ("write", WRITE, write)):
        got = (bits & mask).bool()
        want = _oracle_blocks(levels, widths, bc)
        assert torch.equal(got, want), (
            f"the {name} bit disagrees with the fp64 leaf-product reference on "
            f"{(got ^ want).sum().item()} of {got.numel()} blocks")


@pytest.mark.parametrize("grain", _GRAINS, ids=lambda g: f"d{g[0]}b{g[1]}bc{g[2]}")
def test_write_bit_agrees_with_the_atom_bitmap(grain):
    """THE CROSS-GRAIN GATE: one conjunction, evaluated at two grains.

    P67's stats pass answers the same question at the invariant 16-leaf quantum.
    OR-condensed over each block's `BC / 16` atoms it must equal this reduction's
    WRITE bit exactly -- neither artifact derives from the other.
    """
    D, B, bc = grain
    widths = (B,) * D
    read, write = _fixture(widths, 0.3, seed=D * 77 + B + bc)
    bits = _bits(read, write, widths, bc)
    atoms = planes.atom_bits(_pack(write), widths).bool()
    condensed = atoms.view(_B * _H, prod(widths) // bc, bc // MMA_K_QUANTUM).any(dim=-1)
    assert torch.equal((bits & WRITE).bool(), condensed), (
        "the block-grain WRITE bit and the atom-grain bitmap disagree; two "
        "evaluations of one conjunction have diverged")


def test_two_token_cross_product_is_excluded_at_block_grain():
    """THE ADVERSARIAL CELL, at the launch's grain rather than the atom's.

    At `D = 2, B = 64, BC = 64` leaf `s = d0 * 64 + d1`, so block `== d0`. The
    sequence is one chunk long and every token past the second is identically
    zero, which the table votes dead. Token 0 writes only `(0, 0)`; token 1 writes
    only `(1, 32)`. The exact answer is blocks `{0, 1}` on the write side. The
    trap this states is the READ side: token 0
    reads `(3, 0)` and token 1 reads `(5, 8)`, so blocks `{3, 5}` are read-live and
    nothing else is -- a derivation that ORed the level-0 and level-1 supports over
    tokens before ANDing them would still admit exactly those two blocks here,
    which is why the level-0 supports are made DISJOINT from the write side's: any
    leakage between the sides shows up as an extra block.
    """
    widths = (64, 64)
    read = tuple(torch.zeros((1, _C, 1, w), device="cuda", dtype=torch.float32) for w in widths)
    write = tuple(torch.zeros((1, _C, 1, w), device="cuda", dtype=torch.float32) for w in widths)
    write[0][0, 0, 0, 0] = 1.0
    write[1][0, 0, 0, 0] = 1.0
    write[0][0, 1, 0, 1] = 1.0
    write[1][0, 1, 0, 32] = 1.0
    read[0][0, 0, 0, 3] = 1.0
    read[1][0, 0, 0, 0] = 1.0
    read[0][0, 1, 0, 5] = 1.0
    read[1][0, 1, 0, 8] = 1.0

    bits = _bits(read, write, widths, 64)
    assert set(torch.nonzero(bits[0] & WRITE).flatten().tolist()) == {0, 1}
    assert set(torch.nonzero(bits[0] & READ).flatten().tolist()) == {3, 5}
    assert torch.equal((bits & WRITE).bool(), _oracle_blocks(write, widths, 64))
    assert torch.equal((bits & READ).bool(), _oracle_blocks(read, widths, 64))


@pytest.mark.parametrize("grain", _GRAINS, ids=lambda g: f"d{g[0]}b{g[1]}bc{g[2]}")
def test_the_state_bit_is_the_page_tables_residency(grain):
    """`STATE` is the OR of the block's atoms' residency, and zero without a table.

    An unmapped atom's entry-state fragments provably stay zero, so residency is
    the exact witness for "this block may carry state in". The dense backing has
    no such map, and the bit is then absent rather than guessed -- which is what
    makes the host's degraded dense predicate a recorded weakening and not a bug.
    """
    D, B, bc = grain
    widths = (B,) * D
    N = prod(widths)
    read, write = _fixture(widths, 0.5, seed=D + B + bc)
    assert not (_bits(read, write, widths, bc) & STATE).any(), (
        "the STATE bit is set with no page table; the dense backing has no "
        "residency map to read it from")

    atoms = N // MMA_K_QUANTUM
    g = torch.Generator(device="cuda").manual_seed(5)
    resident = torch.rand((_B * _H, atoms), device="cuda", generator=g) < 0.25
    page_tbl = torch.where(resident, torch.arange(atoms, device="cuda").expand(_B * _H, atoms),
                           torch.full((_B * _H, atoms), -1, device="cuda")).to(torch.int32)
    bits = _bits(read, write, widths, bc, page_tbl=page_tbl.contiguous())
    want = resident.view(_B * _H, N // bc, bc // MMA_K_QUANTUM).any(dim=-1)
    assert torch.equal((bits & STATE).bool(), want)
