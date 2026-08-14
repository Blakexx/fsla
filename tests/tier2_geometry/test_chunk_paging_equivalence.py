"""Dense and paged are ONE kernel and ONE base translation -- bit for bit.

The chunk consumer's state I/O addresses a page per atom. Which page is the only
thing a backing changes: without a table the slot IS the atom and the plane is
the dense `[BH, N/16, 16, cols]` one; with a table the slot is what the plan
committed and the plane holds only those pages. Same instruction stream, same
accumulation order, same ABI -- so the claim here is `torch.equal` and not a
tolerance. A tolerance would be the wrong instrument entirely: any difference at
all is an ADDRESSING difference, and addressing is exact or wrong.

Vocabulary, defined here and used unqualified below: `D` = level count;
`B` = one uniform level width; `N = B ** D` = leaves; `T` = tokens;
`BH = batch * heads`; `d_v = 64`; `cols = d_v + 1`; **atom** = the 16-leaf page
granule (`rola.ops.paging.MMA_K_QUANTUM`); **slot** = the physical page an atom
is mapped to, `-1` for an atom the plan committed nothing for; `BC` = leaves per
owner block; `owners = N / BC`.

**Three claims.**

1. *Bit identity.* One call dense and one paged over the same routing and the
   same `v`, with the arena planned from the kernel's own exact write bitmap
   (`rola.ops.chunk.stats` -> `PageArena.plan_exact`). `y` is equal and the
   states are equal once the paged pages are gathered back through the table.
   The gather sends an absent atom to zeros, which is the same thing the dense
   plane holds there, because the bitmap is the write set EXACTLY -- an atom
   outside it was never deposited into.

2. *A zero page is inert* (the design's E.4). Force an atom the routing never
   writes to be resident, and nothing moves: not `y`, not one bit of the state.
   This is what licenses every SUPERSET residency derivation as safe-direction,
   and it is the mechanical form of the self-normalizing readout's dead-leaf
   argument.

3. *An absent atom is skipped, not written.* A structured-support run whose
   bitmap genuinely excludes atoms leaves every unadmitted page of the arena's
   backing exactly as it found it -- checked against a poison fill -- while `y`
   still equals the dense run's.

`y`'s exactness is claimed only where `y` is reproducible at all: it is fanned
in with `atomicAdd` across owner blocks, so at `owners > 1` the path is not
bit-reproducible against ITSELF and the comparison is against the reordering
envelope with the null control asserted to fit inside it -- the derivation
`test_chunk_final_state.py` and `test_paging_equivalence.py` both ship. The
STATE is exact on every arm regardless: each atom is stored by exactly one CTA.
"""
from __future__ import annotations

from math import prod

import pytest
import torch

from rola.ops import chunk
from rola.ops._ext import extension
from rola.ops.paging import MMA_K_QUANTUM, PageArena

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the chunk consumer is a CUDA kernel"),
]

_DV = 64
_COLS = _DV + 1
_POISON = -1234.5

#: `(widths, C, BC)` -- one arm per depth the design's sweep names, plus the
#: TWO-GROUP arm, where the epilogue's atom count and its barriers are per group
#: rather than per CTA and a per-group residency divergence is therefore possible.
_ARMS = (
    ((16, 16), 32, 64),
    ((16, 16, 16), 32, 128),
    ((8, 8, 8, 8), 32, 128),
    ((64, 64), 32, 256),
)

#: the fraction of each level's digits the WRITE side is allowed to reach.
#: Residency is a property of the write support's STRUCTURE, not of its density:
#: unstructured sparsity at long `T` still reaches every atom (the stats pass's
#: own measurement), so a cell that wants absent atoms must restrict digits.
_SUPPORTS = {"sparse": 0.25, "mixed": 0.5, "dense": 1.0}


def _simplex(shape, generator, live=None):
    """Rows on the simplex, optionally supported on the first `live` digits."""
    x = torch.rand(shape, device="cuda", dtype=torch.float64, generator=generator)
    if live is not None:
        x[..., live:] = 0.0
    return (x / x.sum(-1, keepdim=True)).to(torch.float32)


def _fixture(widths, *, T, support=1.0, seed=0, B=2, H=2):
    """Dense read support, `support`-structured write support."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    read = tuple(_simplex((B, T, H, w), g) for w in widths)
    write = tuple(_simplex((B, T, H, w), g, live=max(1, int(w * support))) for w in widths)
    g_write = torch.rand((B, T, H), device="cuda", generator=g)
    v = torch.randn((B, T, H, _DV), device="cuda", generator=g)
    return read, write, g_write, v


def _packed(fixture):
    read, write, g_write, v = fixture
    return (torch.cat(read, -1).permute(0, 2, 1, 3).to(torch.bfloat16).contiguous(),
            torch.cat(write, -1).permute(0, 2, 1, 3).to(torch.bfloat16).contiguous(),
            g_write.permute(0, 2, 1).to(torch.bfloat16).contiguous(),
            v.permute(0, 2, 1, 3).to(torch.bfloat16).contiguous())


def _dense_plane(widths, BH):
    return torch.empty((BH, prod(widths) // MMA_K_QUANTUM, MMA_K_QUANTUM, _COLS),
                       device="cuda", dtype=torch.float32)


def _launch(fixture, widths, C, BC, *, s_in=None, s_out=None, page_tbl=None, eps=1e-6):
    pr, pw, gw, vk = _packed(fixture)
    y = extension().chunk_forward(pr, pw, gw, vk, C, BC, list(widths), eps, s_in, s_out, page_tbl)
    return y


def _dense(fixture, widths, C, BC, *, s_in=None):
    BH = fixture[3].shape[0] * fixture[3].shape[2]
    out = _dense_plane(widths, BH)
    y = _launch(fixture, widths, C, BC, s_in=s_in, s_out=out)
    return y, out.view(BH, prod(widths), _COLS)


def _bitmap(fixture, widths):
    """The kernel's own EXACT atom-grain write bitmap for this routing."""
    return extension().chunk_stats(_packed(fixture)[1], list(widths)).bool()


def _arena(widths, BC, BH):
    return PageArena(widths=widths, BC=BC, cols=_COLS, BH=BH, device=torch.device("cuda"))


def _paged(fixture, widths, C, BC, arena, *, s_in=None):
    arena.wait()
    y = _launch(fixture, widths, C, BC, s_in=s_in, s_out=arena.state,
                page_tbl=arena.page_table)
    return y, arena.materialize()


def _y_is_exact(widths, BC) -> bool:
    return prod(widths) // BC == 1


def _envelope(y, widths, BC):
    """The most `y` can move under a reassociation of the fan-in's atomics."""
    n = prod(widths) // BC
    return n * n * torch.finfo(torch.float32).eps * y.abs().max()


def _assert_y(y_dense, y_paged, y_control, widths, BC, what):
    if _y_is_exact(widths, BC):
        assert torch.equal(y_dense, y_control), "y is not reproducible against itself"
        assert torch.equal(y_dense, y_paged), what
    else:
        bound = _envelope(y_dense, widths, BC)
        assert (y_dense - y_control).abs().max() <= bound, "the null control blew its envelope"
        assert (y_dense - y_paged).abs().max() <= bound, what


# ---------------------------------------------------------------------------
# 1. dense and paged are the same call
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("support", list(_SUPPORTS), ids=list(_SUPPORTS))
@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: f"{'x'.join(map(str, a[0]))}-C{a[1]}-BC{a[2]}")
def test_the_paged_backing_is_bit_identical_to_the_dense_one(arm, support):
    widths, C, BC = arm
    fixture = _fixture(widths, T=2 * C, support=_SUPPORTS[support], seed=len(widths) * 97 + BC)
    BH = fixture[3].shape[0] * fixture[3].shape[2]

    y_dense, dense = _dense(fixture, widths, C, BC)
    y_control, _ = _dense(fixture, widths, C, BC)
    arena = _arena(widths, BC, BH)
    result = arena.plan_exact(_bitmap(fixture, widths))
    y_paged, paged = _paged(fixture, widths, C, BC, arena)

    assert result.allocated_pages > 0 and (paged != 0).any(), (
        "the plan committed nothing, or the pages came back empty")
    identity = torch.arange(arena.logical_pages, device="cuda",
                            dtype=torch.int32).view_as(arena.page_table)
    assert support == "dense" or not torch.equal(arena.page_table, identity), (
        "the plan mapped every atom to its own id, so this cell never exercised "
        "a translation at all")
    assert torch.equal(dense, paged), (
        f"the paged state differs from the dense one by "
        f"{(dense - paged).abs().max().item():.3e} over "
        f"{int((dense != paged).sum())} elements")
    _assert_y(y_dense, y_paged, y_control, widths, BC, "the paged y differs from the dense y")


@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: f"{'x'.join(map(str, a[0]))}-C{a[1]}-BC{a[2]}")
def test_a_continuation_is_bit_identical_across_the_two_backings(arm):
    """Two halves chained through the state, once dense and once in the arena.

    The paged half continues IN PLACE: the entry state and the final state are
    the same plane, resolved through the same table, which is the aliasing the
    design permits and the case a growing sequence actually runs. The second
    half's plan is `fresh=False` -- the atoms the first half committed stay
    committed, and only the newly written ones are admitted and zeroed.
    """
    widths, C, BC = arm
    T = 4 * C
    fixture = _fixture(widths, T=T, support=0.5, seed=31)
    read, write, g_write, v = fixture
    BH = v.shape[0] * v.shape[2]
    half = T // 2
    first = (tuple(x[:, :half] for x in read), tuple(x[:, :half] for x in write),
             g_write[:, :half], v[:, :half])
    second = (tuple(x[:, half:] for x in read), tuple(x[:, half:] for x in write),
              g_write[:, half:], v[:, half:])

    y1_dense, s1 = _dense(first, widths, C, BC)
    y2_dense, s2 = _dense(second, widths, C, BC,
                          s_in=s1.view(BH, prod(widths) // MMA_K_QUANTUM, MMA_K_QUANTUM, _COLS))
    y1_control, _ = _dense(first, widths, C, BC)

    arena = _arena(widths, BC, BH)
    arena.plan_exact(_bitmap(first, widths))
    y1_paged, p1 = _paged(first, widths, C, BC, arena)
    assert torch.equal(s1, p1), "the first half's state differs across the backings"

    arena.plan_exact(_bitmap(second, widths), fresh=False)
    y2_paged, p2 = _paged(second, widths, C, BC, arena, s_in=arena.state)
    assert torch.equal(s2, p2), (
        f"the continued state differs across the backings by "
        f"{(s2 - p2).abs().max().item():.3e}")
    _assert_y(y1_dense, y1_paged, y1_control, widths, BC, "the first half's y differs")
    _assert_y(y2_dense, y2_paged, y2_dense, widths, BC, "the second half's y differs")


# ---------------------------------------------------------------------------
# 2. a zero page is inert
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: f"{'x'.join(map(str, a[0]))}-C{a[1]}-BC{a[2]}")
def test_a_never_written_resident_page_moves_nothing(arm):
    """The design's E.4: a SUPERSET plan computes the exact plan's answer.

    An atom the routing never writes is forced resident. It is loaded as zero,
    it is stored as zero, and it is exactly what the dense plane holds there --
    so a residency derivation that overshoots is safe by construction and only
    the FOOTPRINT pays for the overshoot.
    """
    widths, C, BC = arm
    fixture = _fixture(widths, T=2 * C, support=0.25, seed=41)
    BH = fixture[3].shape[0] * fixture[3].shape[2]
    exact = _bitmap(fixture, widths)
    absent = (~exact).nonzero()
    assert absent.numel(), "no atom is unwritten; the superset injection has nothing to add"
    extra = exact.clone()
    extra[tuple(absent[0].tolist())] = True

    arena = _arena(widths, BC, BH)
    arena.plan_exact(exact)
    y_exact, s_exact = _paged(fixture, widths, C, BC, arena)

    over = _arena(widths, BC, BH)
    over.plan_exact(extra)
    y_over, s_over = _paged(fixture, widths, C, BC, over)

    assert over.allocated_pages == arena.allocated_pages + 1, "the injection changed nothing"
    assert torch.equal(s_exact, s_over), "a never-written resident page moved the state"
    assert torch.equal(y_exact, y_over) or not _y_is_exact(widths, BC), (
        "a never-written resident page moved y")


# ---------------------------------------------------------------------------
# 3. an absent atom is skipped
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: f"{'x'.join(map(str, a[0]))}-C{a[1]}-BC{a[2]}")
def test_an_absent_atom_leaves_its_backing_untouched(arm):
    """A page the plan did not commit is not written -- checked, not argued.

    The backing BEYOND the plan's cursor is poisoned, so anything the kernel touched
    there shows up as a lost poison value. Poisoning AFTER the plan rather than before
    is what keeps the cell honest under the VMM backing: those pages exist physically
    only once commitment has reached them, and admission has already zeroed everything
    below the cursor. The page TABLE is checked too: a slot resolve that let an absent
    atom through would address `-1 * 4160` bytes, which is off the front of the backing
    and into whatever the allocator put there.
    """
    widths, C, BC = arm
    fixture = _fixture(widths, T=2 * C, support=0.25, seed=53)
    BH = fixture[3].shape[0] * fixture[3].shape[2]
    y_dense, dense = _dense(fixture, widths, C, BC)
    y_control, _ = _dense(fixture, widths, C, BC)

    arena = _arena(widths, BC, BH)
    result = arena.plan_exact(_bitmap(fixture, widths))
    arena.wait()
    assert result.allocated_pages < arena.logical_pages, (
        "every atom is resident; this cell needs a routing that leaves some absent")
    arena.state[result.allocated_pages:].fill_(_POISON)
    table = arena.page_table.clone()
    y_paged, paged = _paged(fixture, widths, C, BC, arena)
    torch.cuda.synchronize()

    untouched = arena.state[result.allocated_pages:]
    assert torch.equal(untouched, torch.full_like(untouched, _POISON)), (
        f"{int((untouched != _POISON).sum())} floats of an uncommitted page were written")
    assert torch.equal(table, arena.page_table), "the launch wrote through the page table"
    assert torch.equal(dense, paged), "the state differs where atoms are absent"
    _assert_y(y_dense, y_paged, y_control, widths, BC, "y differs where atoms are absent")


# ---------------------------------------------------------------------------
# the seam and its refusals
# ---------------------------------------------------------------------------

def test_the_arena_travels_through_the_host_seam():
    """`rola.ops.chunk.run(..., arena=...)` on a producer's own bundle.

    The seam is what a facade will call: it takes the arena, waits on its
    admission stream and hands the kernel the table and the paged plane. The
    dense call beside it is the same routing through the same entry.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from conftest import build_routes

    from rola.routing.producer import sparse_routing

    widths, T = (16, 16), 64
    torch.manual_seed(0)
    gateway = build_routes(hidden_size=64, num_heads=2, d_v=_DV, widths=widths,
                           routing=sparse_routing()).cuda()
    x = torch.randn(2, T, 64, device="cuda")
    v = torch.randn(2, T, 2, _DV, device="cuda")

    routes = gateway(x)
    BC = 128  # the arm `rola.ops.chunk` pins for this topology
    y_dense, dense = chunk.run(routes, v, want_state=True)
    y_control, _ = chunk.run(routes, v, want_state=True)
    arena = _arena(widths, BC, routes.batch * routes.heads)
    arena.plan_exact(chunk.stats(routes).bool())
    y_paged, plane = chunk.run(routes, v, want_state=True, arena=arena)

    assert plane is arena.state
    assert torch.equal(dense.view(-1, prod(widths), _COLS), arena.materialize())
    _assert_y(y_dense, y_paged, y_control, widths, BC, "the seam's paged y differs")


def test_a_misshapen_page_table_is_refused():
    """One entry per atom of the launch's own topology, or a loud refusal."""
    widths = (16, 16)
    fixture = _fixture(widths, T=32, seed=3)
    BH = fixture[3].shape[0] * fixture[3].shape[2]
    arena = _arena(widths, 64, BH)
    arena.plan_exact(_bitmap(fixture, widths))
    with pytest.raises(RuntimeError, match="one per atom"):
        _launch(fixture, widths, 32, 64, s_out=arena.state,
                page_tbl=arena.page_table[:, :-1].contiguous())


def test_a_dense_plane_under_a_page_table_is_refused():
    """The table names slots; a plane shaped `[BH, N/16, 16, cols]` has none."""
    widths = (16, 16)
    fixture = _fixture(widths, T=32, seed=4)
    BH = fixture[3].shape[0] * fixture[3].shape[2]
    arena = _arena(widths, 64, BH)
    arena.plan_exact(_bitmap(fixture, widths))
    with pytest.raises(RuntimeError, match="paged plane"):
        _launch(fixture, widths, 32, 64, s_out=_dense_plane(widths, BH),
                page_tbl=arena.page_table)


def test_a_page_table_without_a_state_is_refused():
    """A call that carries no state has nothing for a table to resolve."""
    widths = (16, 16)
    fixture = _fixture(widths, T=32, seed=5)
    BH = fixture[3].shape[0] * fixture[3].shape[2]
    arena = _arena(widths, 64, BH)
    with pytest.raises(RuntimeError, match="nothing for it to resolve"):
        _launch(fixture, widths, 32, 64, page_tbl=arena.page_table)
