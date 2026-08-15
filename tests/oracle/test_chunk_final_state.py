"""The chunk consumer's carried state: the final-state store and the entry load.

The kernel keeps its state in registers for the whole launch, so until now a call
began at zero and ended by throwing the state away. This file gates the two ends
that change that -- the epilogue that stores the register fragments into the
dense state plane, and the prologue that seeds them from one.

Vocabulary, defined here and used unqualified below: `D` = level count;
`B` = one uniform level width; `N = B ** D` = leaves; `T`/`L` = tokens;
`BH` = batch * heads; `d_v = 64` = value channels; `cols = d_v + 1`, the mass
column that global normalization divides by; **atom** = the 16-leaf page granule
(`rola.ops.paging.MMA_K_QUANTUM`); the **state plane** is `[BH, N/16, 16, cols]`
fp32, the atom-major view of `[BH, N, cols]` byte for byte; `BC` = leaves per
owner block; `owners = N / BC` = the CTAs that share one output row.

**Three claims, three kinds of comparison, and none of the three is a tuned
tolerance.**

1. *The stored state is the recurrence's state.* Against the fp64 oracle
   (`naive_rola(..., output_final_state=True)`), under a bound DERIVED from what
   separates the two runs: the kernel's operands are bf16 and its sum is fp32,
   the oracle's are fp64 throughout. A state element is `sum_t W[t,s] * dep[t,c]`;
   each term carries `D + 2` bf16 roundings (the `D` level amplitudes chained,
   the write gain, the value) at relative `u = eps_bf16 / 2` each, and the fp32
   accumulation of `T` terms reassociates by at most `T * eps_fp32` of the same
   scale. So the bound is `((D + 2) * u + T * eps_fp32) * sum_t |terms|`, with
   the term sum computed in fp64 from the oracle's own leaf materialization --
   arithmetic about the two runs, not a threshold read off a failure.

   This is also the gate on the LEAF-ORDER LAW: the plane is addressed by
   `block_base(blk) / 16`, so a block whose leaves were not the contiguous
   aligned run that law claims would show up here as a PERMUTATION -- elements
   that are exactly some other element -- and not as noise.

2. *Seeding the state is continuing the sequence.* One call over `T` tokens
   against two calls over `T/2` each, the second handed the first's final state.
   With dense support and `T/2` a multiple of the chunk `C`, the super-chunk
   partition is the same tokens in the same order on both sides, and the store
   and load are exact fp32 moves -- so the claim is `torch.equal`, on the state
   always and on `y` wherever `y` is reproducible at all. It is reproducible when
   `owners == 1`: `y` is fanned in with `atomicAdd` across owner blocks, so at
   `owners > 1` the unpaged path is not bit-reproducible against ITSELF and the
   comparison is against the reordering envelope with the null control asserted
   to fit inside it (the same derivation
   `test_chunk_paging_equivalence.py` ships).

3. *A zero entry state is no entry state.* Bit-identical, both outputs. The
   recurrence is affine in the state with no decay in the operator, so a zero
   seed is the identity -- and this is the cell that would catch an entry load
   that ACCUMULATED into the fragments instead of setting them, or that ran
   after something had already written them.
"""
from __future__ import annotations

from math import prod

import pytest
import torch

from rola.ops import chunk
from rola.ops._ext import extension
from rola.ops.naive import _leaf_product, _radix_strides, naive_rola
from rola.ops.paging import MMA_K_QUANTUM
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    LevelConfig,
    SoftmaxActivation,
    Topology,
)

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the chunk consumer is a CUDA kernel"),
]

_DV = 64
_COLS = _DV + 1
_ROUTING = IndependentRouting(
    tied=False, read_activation=SoftmaxActivation(), write_activation=EntmaxActivation(1.5))

#: `(widths, C, BC)`. The first five are what `rola.ops.chunk._arm_for` picks for
#: those topologies; `BC = 256` is the TWO-GROUP arm, which no pinned preference
#: reaches and which is the one shape where the epilogue's staging buffer, its
#: atom count and its barriers are per-group rather than per-CTA.
_ARMS = (
    ((8, 8), 32, 64),
    ((16, 16), 32, 64),
    ((64, 64), 32, 128),
    ((16, 16, 16), 32, 128),
    ((8, 8, 8, 8), 32, 128),
    ((256,), 32, 64),
    ((64, 64), 32, 256),
)

_SUPPORTS = {"sparse": 0.35, "dense": 1.0}


def _simplex(shape, p_nonzero, generator):
    """Rows on the simplex with EXACT support zeros -- entmax's own output shape."""
    x = torch.rand(shape, device="cuda", dtype=torch.float64, generator=generator)
    if p_nonzero < 1.0:
        keep = torch.rand(shape, device="cuda", dtype=torch.float64,
                          generator=generator) < p_nonzero
        x = x * keep
    x = torch.where(x.sum(-1, keepdim=True) == 0, torch.ones_like(x), x)
    return (x / x.sum(-1, keepdim=True)).to(torch.float32)


def _fixture(widths, *, T, p_nonzero, seed, B=2, H=2):
    g = torch.Generator(device="cuda").manual_seed(seed)
    read = tuple(_simplex((B, T, H, w), p_nonzero, g) for w in widths)
    write = tuple(_simplex((B, T, H, w), p_nonzero, g) for w in widths)
    g_write = torch.rand((B, T, H), device="cuda", generator=g)
    v = torch.randn((B, T, H, _DV), device="cuda", generator=g)
    return read, write, g_write, v


def _plane(widths, B, H):
    return torch.empty((B * H, prod(widths) // MMA_K_QUANTUM, MMA_K_QUANTUM, _COLS),
                       device="cuda", dtype=torch.float32)


def _kernel(fixture, widths, C, BC, *, state_in=None, want_state=True, eps=1e-6):
    """One launch through the arm entry: `(y [B,H,T,d_v], state plane or None)`."""
    read, write, g_write, v = fixture
    B, H = v.shape[0], v.shape[2]
    pr = torch.cat(read, -1).permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()
    pw = torch.cat(write, -1).permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()
    gw = g_write.permute(0, 2, 1).to(torch.bfloat16).contiguous()
    vk = v.permute(0, 2, 1, 3).to(torch.bfloat16).contiguous()
    out = _plane(widths, B, H) if want_state else None
    y = extension().chunk_forward(pr, pw, gw, vk, C, BC, list(widths), eps, state_in, out, None)
    return y, out


def _oracle_state(fixture, widths):
    read, write, g_write, v = fixture
    topology = Topology(levels=tuple(LevelConfig(w, _ROUTING) for w in widths))
    _, state = naive_rola(v, read, write, None, g_write, topology, None, "global",
                          output_final_state=True)
    return state


def _state_bound(fixture, widths):
    """`[B,H,N,cols]`: the most the kernel's state may differ from the oracle's.

    `(D + 2)` bf16 roundings per term plus the fp32 reassociation of `T` of them,
    against the fp64 term sum. See this file's docstring for the derivation; the
    term sum is materialized through the ORACLE's address map (`_leaf_product`
    over `_radix_strides`) so the bound never borrows the kernel's idea of which
    leaf is which.
    """
    read, write, g_write, v = fixture
    B, T, H = v.shape[0], v.shape[1], v.shape[2]
    N = prod(widths)
    w64 = g_write.double()[..., None] * _leaf_product(
        tuple(level.double() for level in write), _radix_strides(tuple(widths)), N)
    deposit = torch.cat((v.double(), torch.ones((B, T, H, 1), device="cuda",
                                                dtype=torch.float64)), dim=-1)
    scale = torch.einsum("bthn,bthv->bhnv", w64.abs(), deposit.abs())
    u = torch.finfo(torch.bfloat16).eps / 2
    return ((len(widths) + 2) * u + T * torch.finfo(torch.float32).eps) * scale


def _reordering_bound(y: torch.Tensor, widths, BC) -> torch.Tensor:
    """The most `y` can move under a reassociation of the fan-in's atomics.

    One output element is a sum of at most `owners` fp32 partials, and
    reassociating an `n`-term fp32 sum perturbs it by at most
    `(n-1) * eps * sum|terms|`; bounding `sum|terms|` by `n * max|y|` gives
    `n * n * eps_fp32 * max|y|`. `eps` is fp32's here and not bf16's because this
    kernel entry hands back the fp32 accumulator undowncast.
    """
    n = prod(widths) // BC
    return n * n * torch.finfo(torch.float32).eps * y.abs().max()


# ---------------------------------------------------------------------------
# 1. the stored state IS the recurrence's state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("support", list(_SUPPORTS), ids=list(_SUPPORTS))
@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: f"{'x'.join(map(str, a[0]))}-C{a[1]}-BC{a[2]}")
def test_final_state_matches_the_fp64_oracle(arm, support):
    widths, C, BC = arm
    fixture = _fixture(widths, T=64, p_nonzero=_SUPPORTS[support], seed=len(widths) * 1000 + widths[0] + BC)
    _, plane = _kernel(fixture, widths, C, BC)
    B, H = fixture[3].shape[0], fixture[3].shape[2]
    got = plane.view(B, H, prod(widths), _COLS).double()
    want = _oracle_state(fixture, widths)
    bound = _state_bound(fixture, widths)
    bad = (got - want).abs() > bound
    assert not bad.any(), (
        f"{int(bad.sum())} of {got.numel()} state elements outside the derived "
        f"bound; worst ratio {((got - want).abs() / (bound + 1e-300)).max().item():.3f}")


def test_an_unwritten_leaf_stays_zero():
    """A leaf no token routes to is exactly zero in the plane, not merely small.

    The epilogue stores every atom of the dense plane unconditionally, so this is
    what says the fragments behind the never-written ones were never disturbed --
    the property the paged backing later turns into "absent reads inert".
    """
    widths = (16, 16)
    B, T, H = 1, 32, 1
    g = torch.Generator(device="cuda").manual_seed(7)
    write = tuple(torch.zeros((B, T, H, w), device="cuda") for w in widths)
    write[0][..., 0] = 1.0
    write[1][..., 0] = 1.0  # every token writes leaf 0 and nothing else
    read = tuple(_simplex((B, T, H, w), 1.0, g) for w in widths)
    g_write = torch.rand((B, T, H), device="cuda", generator=g)
    v = torch.randn((B, T, H, _DV), device="cuda", generator=g)
    _, plane = _kernel((read, write, g_write, v), widths, 32, 64)
    state = plane.view(B * H, prod(widths), _COLS)
    assert (state[:, 0] != 0).any(), "the one written leaf is empty"
    assert torch.equal(state[:, 1:], torch.zeros_like(state[:, 1:])), (
        "an unwritten leaf carries a nonzero state")


# ---------------------------------------------------------------------------
# 2. seeding the state IS continuing the sequence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: f"{'x'.join(map(str, a[0]))}-C{a[1]}-BC{a[2]}")
def test_a_continuation_reproduces_the_whole_sequence(arm):
    """Two halves chained through the state against one call over both.

    Dense support and a half that is a whole number of chunks: the walk then cuts
    the token axis at the same places on both sides, so this is the same
    arithmetic in the same order and the state claim is exact equality.
    """
    widths, C, BC = arm
    T = 4 * C
    fixture = _fixture(widths, T=T, p_nonzero=1.0, seed=11)
    read, write, g_write, v = fixture

    y_full, s_full = _kernel(fixture, widths, C, BC)
    y_null, s_null = _kernel(fixture, widths, C, BC)
    assert torch.equal(s_full, s_null), "the final state is not reproducible against itself"

    half = T // 2
    first = (tuple(x[:, :half] for x in read), tuple(x[:, :half] for x in write),
             g_write[:, :half], v[:, :half])
    second = (tuple(x[:, half:] for x in read), tuple(x[:, half:] for x in write),
              g_write[:, half:], v[:, half:])
    y1, s1 = _kernel(first, widths, C, BC)
    y2, s2 = _kernel(second, widths, C, BC, state_in=s1)

    assert torch.equal(s_full, s2), (
        f"the continued final state differs from the whole sequence's by "
        f"{(s_full - s2).abs().max().item():.3e}")
    y_chained = torch.cat([y1, y2], dim=2)
    if prod(widths) // BC == 1:
        assert torch.equal(y_full, y_null), "y is not reproducible at a single owner"
        assert torch.equal(y_full, y_chained), "the continued y differs at a single owner"
    else:
        bound = _reordering_bound(y_full, widths, BC)
        assert (y_full - y_null).abs().max() <= bound, "the null control blew its own envelope"
        assert (y_full - y_chained).abs().max() <= bound


@pytest.mark.parametrize("arm", _ARMS[:3], ids=lambda a: f"{'x'.join(map(str, a[0]))}-BC{a[2]}")
def test_a_sparse_continuation_matches_the_oracle(arm):
    """The seeding claim where the partition is NOT shared.

    Under sparse support the walk closes super-chunks on liveness, so splitting
    the sequence genuinely re-partitions it and nothing is owed bit for bit. What
    IS owed is the recurrence: the chained state must still be the oracle's final
    state over the whole sequence, under the same derived bound as cell 1.
    """
    widths, C, BC = arm
    T = 4 * C
    fixture = _fixture(widths, T=T, p_nonzero=0.35, seed=23)
    read, write, g_write, v = fixture
    half = T // 2
    first = (tuple(x[:, :half] for x in read), tuple(x[:, :half] for x in write),
             g_write[:, :half], v[:, :half])
    second = (tuple(x[:, half:] for x in read), tuple(x[:, half:] for x in write),
              g_write[:, half:], v[:, half:])
    _, s1 = _kernel(first, widths, C, BC)
    _, s2 = _kernel(second, widths, C, BC, state_in=s1)

    B, H = v.shape[0], v.shape[2]
    got = s2.view(B, H, prod(widths), _COLS).double()
    want = _oracle_state(fixture, widths)
    bound = _state_bound(fixture, widths)
    bad = (got - want).abs() > bound
    assert not bad.any(), f"{int(bad.sum())} chained state elements outside the derived bound"


# ---------------------------------------------------------------------------
# 3. a zero entry state is no entry state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: f"{'x'.join(map(str, a[0]))}-C{a[1]}-BC{a[2]}")
def test_a_zero_entry_state_is_the_absent_one(arm):
    widths, C, BC = arm
    fixture = _fixture(widths, T=64, p_nonzero=1.0, seed=5)
    B, H = fixture[3].shape[0], fixture[3].shape[2]
    y_none, s_none = _kernel(fixture, widths, C, BC)
    zeros = torch.zeros_like(_plane(widths, B, H))
    y_zero, s_zero = _kernel(fixture, widths, C, BC, state_in=zeros)
    assert torch.equal(s_none, s_zero), "a zero entry state moved the final state"
    if prod(widths) // BC == 1:
        assert torch.equal(y_none, y_zero), "a zero entry state moved y at a single owner"
    else:
        assert (y_none - y_zero).abs().max() <= _reordering_bound(y_none, widths, BC)


# ---------------------------------------------------------------------------
# the seam and its refusals
# ---------------------------------------------------------------------------

def test_the_state_travels_through_the_host_seam():
    """`rola.ops.chunk.run(routes, v, state_in, want_state)` on a real bundle.

    A producer's `RouteFactors` rather than hand-built tensors, because the
    packing between the bundle's layout and the kernel's -- and the plane the
    seam allocates -- is exactly what a hand-built fixture skips. The halves come
    from the producer run on the halves of `x`, which is the same routing: the
    producer is per-token.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from conftest import build_routes

    from rola.routing.producer import union_routing

    widths, T = (16, 16), 64
    torch.manual_seed(0)
    gateway = build_routes(hidden_size=64, num_heads=2, widths=widths,
                           routing=union_routing()).cuda()
    x = torch.randn(2, T, 64, device="cuda")

    y_full, s_full = chunk.run(chunk.pack(gateway(x)), torch.randn(2, T, 2, _DV, device="cuda"),
                               want_state=True)
    assert s_full.shape == (4, prod(widths) // MMA_K_QUANTUM, MMA_K_QUANTUM, _COLS)
    assert s_full.dtype == torch.float32

    y_stateless, none = chunk.run(chunk.pack(gateway(x)), torch.randn(2, T, 2, _DV, device="cuda"))
    assert none is None and y_stateless.shape == y_full.shape

    v = torch.randn(2, T, 2, _DV, device="cuda")
    half = T // 2
    _, s1 = chunk.run(chunk.pack(gateway(x[:, :half])), v[:, :half], want_state=True)
    _, s2 = chunk.run(chunk.pack(gateway(x[:, half:])), v[:, half:], state_in=s1, want_state=True)
    _, s_whole = chunk.run(chunk.pack(gateway(x)), v, want_state=True)
    assert torch.equal(s_whole, s2), (
        f"the seam's continuation differs by {(s_whole - s2).abs().max().item():.3e}")


#: THE PAGE TABLE'S REFUSAL IS GONE, and with it this file's cell for it: the
#: table is resolved rather than refused, and the paged backing's own gates --
#: bit identity, zero-page inertness, the absent-atom skip, and the refusals
#: that survive -- live in `test_chunk_paging_equivalence.py`.


def test_a_misshapen_state_plane_is_refused():
    """The plane's size is the topology's; a plane for another one is refused."""
    widths = (16, 16)
    fixture = _fixture(widths, T=32, p_nonzero=1.0, seed=4)
    wrong = _plane((16, 16, 16), 2, 2)
    with pytest.raises(RuntimeError, match="the entry state"):
        _kernel(fixture, widths, 32, 64, state_in=wrong, want_state=False)
