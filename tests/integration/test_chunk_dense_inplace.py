# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""A dense CONTINUATION advances its plane in place, and that is bit-inert.

`s_out` aliases `s_in` for a continued dense sequence, so a chained call allocates
one plane for the sequence instead of one per call. The comparand is the DOUBLE-BUFFER
form -- a fresh whole-written plane per call, swapped in -- and the claim against it is
`torch.equal`, because the two differ in ADDRESSING only and addressing is exact or
wrong (docs/internals/state.md#the-dense-continuation).

Two properties carry the claim and this file separates them, which is what makes a
failure locatable:

* the ALIASING -- one plane at both ends of the launch. Safe by the CTA structure:
  each atom has one owner group in one CTA, the entry fold reads its atoms once in
  the prologue and the store epilogue writes the same ones once at the end, so no
  CTA can read an atom another may already have stored
  (`chunk_kernel.md#entry-state`);
* the SKIP ROW -- with the plane updated in place a dead block leaves its own atoms
  carrying, which is what makes the dense continuation's predicate `R | W` rather than
  always-run (`chunk/chunk.md#run-table`).

The middle arm below aliases under the OLD always-run table, so an aliasing fault and
a skip-row fault land on different cells.

Vocabulary: `D` = level count; `B` = one uniform level width; `N = B ** D`; `T` =
tokens; `BH = batch * heads`; `d_v = 64`; `cols = d_v + 1`; **atom** = the 16-leaf
page granule; `BC` = leaves per owner block.
"""
from __future__ import annotations

from math import prod

import pytest
import torch

from rola.engine.rules.run_table import run_table
from rola.engine.types import Mask
from rola.ops._ext import extension
from rola.ops.paging import MMA_K_QUANTUM

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the chunk consumer is a CUDA kernel"),
]

_DV = 64
_COLS = _DV + 1

#: `(widths, C, BC)`. The ONE-OWNER arm, where `y` is bit-reproducible against itself
#: and the claim on it can be `torch.equal`; the shallow multi-owner arm; the DEEP one
#: (four levels, where a block's leaf base is a sum over four digit places); and the
#: two-group arm, whose epilogue iterates its atoms per GROUP -- the one place where
#: two schedules inside one CTA touch one plane.
_ARMS = (
    ((8, 8), 32, 64),
    ((16, 16), 32, 64),
    ((8, 8, 8, 8), 32, 128),
    ((64, 64), 32, 256),
)

#: `T` per call in a chain. `36` and `19` are OFF the 32-token chunk on both ends of
#: the walk, which is the ragged tail the state fold must be indifferent to; `64` is
#: the aligned control.
_LENGTHS = ((64, 64, 64), (36, 19, 45), (19, 64, 36))

#: the double-buffer arm's table: a plane written whole may skip nothing. It is a
#: LITERAL and not a derivation, because the comparand must stay fixed under any later
#: edit to the rule -- a comparand that moves with its subject compares nothing.
_ALWAYS_RUN = 0xFF


def _simplex(shape, generator, *, live=None, off=0):
    """Rows on the simplex, optionally supported on a `live`-wide window at `off`."""
    x = torch.rand(shape, device="cuda", dtype=torch.float64, generator=generator)
    if live is not None:
        w = shape[-1]
        keep = torch.zeros(w, device="cuda", dtype=torch.bool)
        keep[(torch.arange(live, device="cuda") + off) % w] = True
        x = x * keep
    return (x / x.sum(-1, keepdim=True)).to(torch.float32)


def _fixture(widths, *, T, support, seed, off=0, split=False, B=2, H=2):
    """Dense read support; the write side on a `support`-wide window starting at `off`.

    The read side reaches every digit and the write side does not, which is what puts
    blocks in the READ-LIVE, WRITE-DEAD state -- and MOVING the window between the
    calls of a chain is what leaves those blocks carrying something nonzero. Without
    that, a dead block's state is zero and skipping it is unobservable whatever the
    predicate says.

    `split` gives the read side the window BEFORE the write side's instead of all of
    them, which is the only way to produce a WRITE-LIVE, READ-DEAD block -- the byte
    the `R | W` rule's second term is about.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    live = tuple(max(1, int(w * support)) for w in widths)
    read = tuple(_simplex((B, T, H, w), g,
                          live=(n if split else None), off=(off - 1) * n)
                 for w, n in zip(widths, live))
    write = tuple(_simplex((B, T, H, w), g, live=n, off=off * n)
                  for w, n in zip(widths, live))
    g_write = torch.rand((B, T, H), device="cuda", generator=g)
    v = torch.randn((B, T, H, _DV), device="cuda", generator=g)
    return read, write, g_write, v


def _packed(fixture):
    read, write, g_write, v = fixture
    return (torch.cat(read, -1).permute(0, 2, 1, 3).to(torch.bfloat16).contiguous(),
            torch.cat(write, -1).permute(0, 2, 1, 3).to(torch.bfloat16).contiguous(),
            g_write.permute(0, 2, 1).to(torch.bfloat16).contiguous(),
            v.to(torch.bfloat16).contiguous())


def _plane(widths, BH, *, zeros=False):
    shape = (BH, prod(widths) // MMA_K_QUANTUM, MMA_K_QUANTUM, _COLS)
    build = torch.zeros if zeros else torch.empty
    return build(shape, device="cuda", dtype=torch.float32)


def _launch(fixture, widths, C, BC, *, s_in, s_out, table):
    """One call, WITH the liveness bitmap -- which is what makes `table` mean anything.

    A launch handed no bitmap runs every block whatever the table says, so a cell that
    means to exercise a skip row has to build the bitmap the shipped seam builds.
    """
    m = extension()
    pr, pw, gw, vk = _packed(fixture)
    union = m.chunk_facts(pr, pw, C, BC, list(widths), int(Mask.M2), False)[0]
    bits = m.chunk_block_bits(union, BC, list(widths), vk.shape[1], None)
    y = m.chunk_forward(pr, pw, gw, vk, C, BC, list(widths), 1e-6, table,
                        None, torch.float32, s_in, s_out, None, bits)
    return y.permute(0, 2, 1, 3)


def _in_place(chain, widths, C, BC, *, table=None):
    """The shipped path: one plane for the sequence, `s_out` IS `s_in`."""
    BH = chain[0][3].shape[0] * chain[0][3].shape[2]
    plane, ys = None, []
    for fixture in chain:
        if plane is None:
            plane = _plane(widths, BH)
            ys.append(_launch(fixture, widths, C, BC, s_in=None, s_out=plane,
                              table=run_table(None, plane, None)))
            continue
        derived = run_table(plane, plane, None) if table is None else table
        ys.append(_launch(fixture, widths, C, BC, s_in=plane, s_out=plane, table=derived))
    return ys, plane


def _double_buffered(chain, widths, C, BC):
    """The comparand: a fresh whole-written plane per call, then a swap."""
    BH = chain[0][3].shape[0] * chain[0][3].shape[2]
    plane, ys = None, []
    for fixture in chain:
        out = _plane(widths, BH)
        ys.append(_launch(fixture, widths, C, BC, s_in=plane, s_out=out, table=_ALWAYS_RUN))
        plane = out
    return ys, plane


def _chain(widths, lengths, *, support, seed, split=False):
    return [_fixture(widths, T=T, support=support, seed=seed + 7 * i, off=i, split=split)
            for i, T in enumerate(lengths)]


def _assert_chain(a, b, control, widths, BC, what):
    """The STATE is `torch.equal` on every arm; `y` is claimed where `y` is exact.

    Each atom is stored by exactly one CTA, so the state is bit-reproducible whatever
    the arm -- that is the claim this file exists for, and it is never softened. `y` is
    fanned in with `atomicAdd` across owner blocks, so above one owner it is not
    reproducible against ITSELF; there it is held to the reordering envelope with the
    null control asserted to fit inside it, the same instrument
    `test_chunk_paging_equivalence.py` derives.
    """
    (ys_a, s_a), (ys_b, s_b), (ys_c, _) = a, b, control
    assert torch.equal(s_a, s_b), (
        f"{what}: the final state differs by {(s_a - s_b).abs().max().item():.3e} over "
        f"{int((s_a != s_b).sum())} elements")
    exact = prod(widths) // BC == 1
    for i, (y_a, y_b, y_c) in enumerate(zip(ys_a, ys_b, ys_c)):
        if exact:
            assert torch.equal(y_a, y_c), f"call {i}'s y is not reproducible against itself"
            assert torch.equal(y_a, y_b), f"{what}: call {i}'s y differs"
            continue
        n = prod(widths) // BC
        bound = n * n * torch.finfo(torch.float32).eps * y_a.abs().max()
        assert (y_a - y_c).abs().max() <= bound, f"call {i}'s null control blew its envelope"
        assert (y_a - y_b).abs().max() <= bound, (
            f"{what}: call {i}'s y differs by {(y_a - y_b).abs().max().item():.3e}, "
            f"outside the {bound.item():.3e} reordering envelope")


# ---------------------------------------------------------------------------
# 1. the aliasing alone -- same table on both sides
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lengths", _LENGTHS, ids=lambda ls: "-".join(map(str, ls)))
@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: f"{'x'.join(map(str, a[0]))}-C{a[1]}-BC{a[2]}")
def test_aliasing_the_two_state_planes_is_inert(arm, lengths):
    """`s_out` aliased onto `s_in` against the double-buffer form, ONE table apart.

    Both arms run the always-run table, so the only difference is whether the entry
    plane and the exit plane are the same bytes. That is the CTA-structure claim,
    isolated: read once at entry, store once at exit, same owner, nothing between.
    """
    widths, C, BC = arm
    chain = _chain(widths, lengths, support=0.5, seed=len(widths) * 131 + BC)
    _assert_chain(_in_place(chain, widths, C, BC, table=_ALWAYS_RUN),
                  _double_buffered(chain, widths, C, BC),
                  _double_buffered(chain, widths, C, BC),
                  widths, BC, "aliasing moved a bit")


# ---------------------------------------------------------------------------
# 2. the shipped path -- aliasing AND the `R | W` skip row
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("support", [0.25, 1.0], ids=["sparse", "dense"])
@pytest.mark.parametrize("lengths", _LENGTHS, ids=lambda ls: "-".join(map(str, ls)))
@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: f"{'x'.join(map(str, a[0]))}-C{a[1]}-BC{a[2]}")
def test_the_in_place_continuation_is_bit_identical_to_the_double_buffer(arm, lengths, support):
    """The production path end to end, against the double-buffer comparand.

    The in-place arm derives its own table, so a dense continuation SKIPS the blocks
    the new row lets it; the double-buffered arm runs every block into a fresh plane.
    Equal `y` at every call and an equal final state is the claim, and the sparse
    support is the cell where the skip is doing something.
    """
    widths, C, BC = arm
    chain = _chain(widths, lengths, support=support, seed=len(widths) * 977 + BC)
    _assert_chain(_in_place(chain, widths, C, BC),
                  _double_buffered(chain, widths, C, BC),
                  _double_buffered(chain, widths, C, BC),
                  widths, BC,
                  "the in-place continuation differs from the double-buffered one")


def test_the_skip_row_is_what_the_in_place_arm_actually_runs():
    """NON-VACUITY. The two arms above are only a claim about skipping if the tables
    they launch under differ -- and if the derived one leaves blocks out."""
    plane = _plane((16, 16), 4)
    derived = run_table(plane, plane, None)
    assert derived != _ALWAYS_RUN, (
        "the dense continuation derives the always-run table; the cells above compare "
        "one path with itself")
    #: `R | W`: the byte with neither entry is the one that may skip, at every value
    #: of the residency bit a dense call cannot witness.
    assert not derived >> 0 & 1 and not derived >> 4 & 1
    assert all(derived >> v & 1 for v in (1, 2, 3, 5, 6, 7))


@pytest.mark.parametrize("arm", _ARMS[1:], ids=lambda a: f"{'x'.join(map(str, a[0]))}-C{a[1]}-BC{a[2]}")
def test_a_write_live_read_dead_block_still_stores(arm):
    """The `R | W` rule's SECOND term, on the byte only split sides produce.

    With the read window disjoint from the write one a block can write without being
    read, and in place that store is the only thing that advances its atoms -- so the
    term is not decoration. The `R`-side term is exercised by every cell above; this
    is the one that needs the sides pulled apart to exist at all.
    """
    widths, C, BC = arm
    chain = _chain(widths, (36, 19, 45), support=0.25, seed=len(widths) * 613 + BC, split=True)
    _assert_chain(_in_place(chain, widths, C, BC),
                  _double_buffered(chain, widths, C, BC),
                  _double_buffered(chain, widths, C, BC),
                  widths, BC, "the split-sides continuation differs")
