"""The stats pass emits the EXACT atom-grain write bitmap.

This is the gate the paging work's first stage exists to pass, and it is the one
that decides the shape of everything after it. Residency is planned from this
bitmap, so if the bitmap is a SUPERSET the arena's `resident == realized`
accounting stops being true and the "planned-exact footprint" contract becomes a
bound instead of a fact. Exactness is therefore asserted as `torch.equal`
against two independent references, never as a containment.

Vocabulary, defined here and used unqualified below: `D` = level count;
`B` = one uniform level width; `N = B ** D` = leaves; `L`/`T` = tokens;
`BH` = batch * heads; **atom** = the 16-leaf page granule
(`rola.ops.paging.MMA_K_QUANTUM`); `atoms = N // 16`; the **bitmap** is
`[BH, atoms]`, true where this routing's WRITE side touches the atom.

**The two references, and why there are two.**

1. `atom_bitmap_from_schedule` (`rola/ops/paging.py`) -- the production
   condensation the arena already admits pages from. Equality here is what makes
   the kernel a drop-in for it.
2. An independent fp64 recomputation built from the ORACLE's own leaf
   materialization (`rola.ops.naive._leaf_product` over
   `rola.ops.naive._radix_strides`), which derives leaf addresses separately from
   production on purpose. Reference 1 and the kernel could in principle share a
   wrong idea of what a leaf is; reference 2 is the second opinion that would
   disagree. Gating against a fresh reference ALONE is the
   [[matching-the-naive-antipattern]] this repository has already been bitten by.

**The adversarial cell is the point of the file.** Every looser derivation of
residency -- OR over tokens first, AND over levels second, or the same
conjunction evaluated on the `BC` rectangle -- agrees with the exact one on
ordinary routing. It disagrees on two tokens whose live digits differ per level,
where the loose form admits CROSS-PRODUCT leaves that were never live at any
single token. `test_two_token_cross_product_is_excluded` is that cell, with the
excluded atoms named explicitly.
"""
from __future__ import annotations

from math import prod

import pytest
import torch

from rola.ops._ext import extension
from rola.ops.naive import _leaf_product, _radix_strides
from rola.ops.paging import MMA_K_QUANTUM, AtomKeying, atom_bitmap_from_schedule
from rola.planner.schedule import Plan, build_schedule
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    LevelConfig,
    SoftmaxActivation,
    Topology,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the stats pass is a CUDA kernel"),
]

#: The built stats arms: `(D, B)`, the projection of the chunk arm list that
#: `tools/gen_shards.py` emits as `CHUNK_ATOM_ARMS_X`. Depths 2-4 are the sweep
#: the design asks for; the two `D = 1` arms are the flat-routing baselines,
#: carried because they are built and because `D = 1` is the shape with no upper
#: prefix at all -- the reduction's degenerate carve.
_ARMS = ((1, 64), (1, 256), (2, 8), (2, 16), (2, 64), (3, 16), (4, 8))

#: The support sweep. `1.0` is the dense limit, where every atom is live and the
#: reduction's early exit is the path taken.
_SUPPORTS = {"sparse": 0.35, "mixed": 0.7, "dense": 1.0}

_ROUTING = IndependentRouting(
    tied=False, read_activation=SoftmaxActivation(), write_activation=EntmaxActivation(1.5))

_B, _T, _H, _DV = 2, 48, 2, 64


def _simplex(shape, p_nonzero, generator):
    """Rows on the simplex with EXACT support zeros -- entmax's own output shape.

    A fixture with no exact zero tests only the dense limit
    ([[sparsity-comes-from-entmax]]).
    """
    x = torch.rand(shape, device="cuda", dtype=torch.float64, generator=generator)
    if p_nonzero < 1.0:
        keep = torch.rand(shape, device="cuda", dtype=torch.float64,
                          generator=generator) < p_nonzero
        x = x * keep
    x = torch.where(x.sum(-1, keepdim=True) == 0, torch.ones_like(x), x)
    return (x / x.sum(-1, keepdim=True)).to(torch.float32)


def _pack(write):
    """`[B,T,H,w] per level -> [B,H,T,sum(widths)]` bf16 -- the seam's packing."""
    return torch.cat(write, -1).permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()


def _kernel_bitmap(write, widths):
    packed = _pack(write)
    #: THE FIXTURE MUST SURVIVE ITS OWN CAST. The kernel reads bf16, the
    #: references read fp32/fp64; an amplitude that flushed to a bf16 zero would
    #: make the two disagree about the fixture rather than about the derivation.
    flat = torch.cat(write, -1)
    assert torch.equal(flat != 0, flat.to(torch.bfloat16) != 0), (
        "fixture amplitudes flushed to zero in bf16; the comparison would be "
        "about the cast, not about the bitmap")
    return extension().chunk_stats(packed, list(widths)).bool()


def _oracle_bitmap(write, widths):
    """Reference 2: the leaf is written iff at SOME token its fp64 leaf product
    is nonzero, condensed to atoms. Built from the oracle's address map."""
    B, T, H = write[0].shape[:3]
    N = prod(widths)
    levels64 = tuple(level.to(torch.float64) for level in write)
    leaves = _leaf_product(levels64, _radix_strides(tuple(widths)), N)  # [B,T,H,N]
    live = (leaves != 0).permute(0, 2, 1, 3).reshape(B * H, T, N).any(dim=1)
    return live.view(B * H, N // MMA_K_QUANTUM, MMA_K_QUANTUM).any(dim=-1)


def _schedule_bitmap(read, write, g_write, widths):
    """Reference 1: the production condensation, through a real `Schedule`.

    `BC` is a `Plan` field the schedule needs and the bitmap does not: the
    condensation is atom-keyed, which `test_the_bitmap_is_bc_free` states as a
    claim rather than leaving as an assumption.
    """
    topology = Topology(levels=tuple(LevelConfig(w, _ROUTING) for w in widths), d_v=_DV)
    plan = Plan(topology=topology, normalization="global", BC=MMA_K_QUANTUM,
                warp_tile=(1, _DV // 32), decay=None)
    schedule = build_schedule(read, write, None, g_write, plan)
    keying = AtomKeying(N=prod(widths), cols=_DV + 1, BH=read[0].shape[0] * read[0].shape[2])
    return atom_bitmap_from_schedule(schedule, keying)


def _fixture(widths, p_nonzero, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    read = tuple(_simplex((_B, _T, _H, w), p_nonzero, g) for w in widths)
    write = tuple(_simplex((_B, _T, _H, w), p_nonzero, g) for w in widths)
    g_write = torch.rand((_B, _T, _H), device="cuda", generator=g)
    return read, write, g_write


@pytest.mark.parametrize("support", list(_SUPPORTS), ids=list(_SUPPORTS))
@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: f"d{a[0]}b{a[1]}")
def test_atom_bitmap_is_exact(arm, support):
    """Exact tensor equality against BOTH references, on every built arm."""
    D, B = arm
    widths = (B,) * D
    read, write, g_write = _fixture(widths, _SUPPORTS[support], seed=D * 1000 + B)
    bits = _kernel_bitmap(write, widths)
    oracle = _oracle_bitmap(write, widths)
    schedule = _schedule_bitmap(read, write, g_write, widths)
    assert bits.shape == (_B * _H, prod(widths) // MMA_K_QUANTUM)
    assert torch.equal(bits, oracle), (
        f"kernel and the fp64 leaf-product reference disagree on "
        f"{(bits ^ oracle).sum().item()} atoms")
    assert torch.equal(bits, schedule), (
        f"kernel and atom_bitmap_from_schedule disagree on "
        f"{(bits ^ schedule).sum().item()} atoms")


def test_two_token_cross_product_is_excluded():
    """THE ADVERSARIAL CELL: two tokens, disjoint live digits per level.

    At `D = 2, B = 64` leaf `s = d0 * 64 + d1`. Token 0 writes only `(0, 0)`
    (leaf 0, atom 0); token 1 writes only `(1, 32)` (leaf 96, atom 6). The exact
    answer is `{0, 6}`. A derivation that ORs over tokens before ANDing over
    levels also admits `(0, 32)` -> leaf 32 -> atom 2 and `(1, 0)` -> leaf 64 ->
    atom 4: leaves that were never live at any single token. Those two atoms are
    named here so the assertion states the trap rather than merely avoiding it.
    """
    widths = (64, 64)
    write = tuple(torch.zeros((1, 2, 1, w), device="cuda", dtype=torch.float32) for w in widths)
    write[0][0, 0, 0, 0] = 1.0
    write[1][0, 0, 0, 0] = 1.0
    write[0][0, 1, 0, 1] = 1.0
    write[1][0, 1, 0, 32] = 1.0

    bits = _kernel_bitmap(write, widths)
    live = set(torch.nonzero(bits[0]).flatten().tolist())
    assert live == {0, 6}, f"expected atoms {{0, 6}}, got {sorted(live)}"
    assert 2 not in live and 4 not in live, (
        "the cross-product atoms are present: the reduction ORed over tokens "
        "before ANDing over levels")
    assert torch.equal(bits, _oracle_bitmap(write, widths))


def test_the_bitmap_is_bc_free():
    """One routing, two `Plan` `BC`s: the SAME bitmap, bit for bit.

    Residency keyed to the invariant quantum is the property the whole atom
    re-key exists for, and a block-grain derivation would fail this by
    construction. The kernel takes no `BC` at all, so what this checks is that
    the reference it is gated against does not either -- which is what makes the
    equality above a `BC`-free statement.
    """
    widths = (16, 16)
    read, write, g_write = _fixture(widths, 0.35, seed=99)
    bits = _kernel_bitmap(write, widths)
    topology = Topology(levels=tuple(LevelConfig(w, _ROUTING) for w in widths), d_v=_DV)
    keying = AtomKeying(N=prod(widths), cols=_DV + 1, BH=_B * _H)
    for BC in (16, 64):
        plan = Plan(topology=topology, normalization="global", BC=BC,
                    warp_tile=(1, _DV // 32), decay=None)
        schedule = build_schedule(read, write, None, g_write, plan)
        assert torch.equal(bits, atom_bitmap_from_schedule(schedule, keying)), (
            f"the bitmap moved with BC={BC}")


def test_stats_through_the_host_seam():
    """`rola.ops.chunk.stats(routes)` -- the packing the facade will call.

    A real producer's `RouteFactors` rather than hand-built tensors, because the
    permute-and-cast between the bundle's layout and the kernel's is exactly the
    step a hand-built fixture would skip.
    """
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from conftest import build_routes

    from rola.ops import chunk
    from rola.routing.producer import sparse_routing

    torch.manual_seed(0)
    gateway = build_routes(hidden_size=64, num_heads=_H, d_v=_DV, widths=(16, 16),
                           routing=sparse_routing()).cuda()
    x = torch.randn(_B, _T, 64, device="cuda")
    routes = gateway(x)
    bits = chunk.stats(routes)
    assert bits.dtype == torch.uint8 and bits.device.type == "cuda"
    assert torch.equal(bits.bool(), _oracle_bitmap(routes.write, (16, 16)))
