# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The facade's stateful half: one object, one predicate (docs/internals/state.md).

What this file gates, and nothing else: the SURFACE. That the predicate is
`state is not None`; that an unbound state binds from its first call and a bound one
continues; that a chained pair of halves computes what one whole call computes; that
`clone` is independent; that the two backings are the same answer; that a bound state
holds its SHAPE and no fingerprint. The kernel-level claims underneath it -- the
exact atom bitmap, dense-vs-paged bit identity at the ABI, zero-page inertness -- are
`tests/oracle/test_atom_bitmap.py` and
`tests/integration/test_chunk_paging_equivalence.py`, and are not restated here.

Vocabulary, defined here and used unqualified below: `N` = leaf count;
`cols` = `d_v + 1` under `normalization='global'`; `owners` = `N / BC` = the owner
blocks `y`'s fan-in reassociates over; `atom` = the 16-leaf page granule.

**Where `y` is compared exactly and where it is not.** `y` is fanned in with
`atomicAdd` across owner blocks, so at `owners > 1` it is not bit-reproducible against
ITSELF and every comparison here goes through the same reordering envelope the paging
suite derives, with a null control asserted to fit inside it. The STATE is exact on
every arm: each atom is stored by exactly one CTA.
"""
from __future__ import annotations

import sys
from math import prod
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import build_layer, build_routes  # noqa: E402

import rola  # noqa: E402
from rola.expert import PlanOverrides  # noqa: E402
from rola.ops import chunk as _chunk  # noqa: E402
from rola.routing.producer import union_routing  # noqa: E402

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel arm is a CUDA kernel"),
]

_HIDDEN, _H, _DV, _WIDTHS, _T = 64, 2, 64, (16, 16), 128
_DENSE = PlanOverrides(paging=False)


def _gateway(seed=0, widths=_WIDTHS):
    torch.manual_seed(seed)
    return build_routes(hidden_size=_HIDDEN, num_heads=_H, widths=widths,
                        routing=union_routing(1)).cuda()


def _inputs(seed=0, T=_T, B=2):
    torch.manual_seed(seed + 1000)
    return (torch.randn(B, T, _HIDDEN, device="cuda"),
            torch.randn(B, T, _H, _DV, device="cuda"))


def _envelope(y, routes):
    """The most `y` can move under a reassociation of the fan-in's atomics."""
    n = prod(routes.topology.widths) // _chunk.arm(routes)[1]
    return n * n * torch.finfo(torch.float32).eps * float(y.abs().max())


def _state_envelope(state, routes):
    """The most the STATE can move when the same tokens are summed in a different
    ASSOCIATION -- which is what splitting one sequence into two launches does.

    A leaf's state is a sum over the chunks that wrote it. One launch reassociates
    that sum over its own `T / C` chunks; two launches cut it in the middle and carry
    an fp32 partial across. Nothing is dropped and no chunk moves to another leaf, so
    the difference is bounded by the reassociation of that many fp32 additions --
    DERIVED here from the arm, exactly as the paging suite derives `y`'s.
    """
    chunks = routes.tokens // _chunk.arm(routes)[0]
    return chunks * chunks * torch.finfo(torch.float32).eps * float(state.abs().max())


# ---------------------------------------------------------------------------
# 1. the predicate
# ---------------------------------------------------------------------------

def test_a_stateless_call_returns_no_state():
    """`state=None` is the whole signal, and the second return is `None`.

    Not a formality: a stateless call runs the kernel variant with NO state I/O
    compiled into it, so a facade that quietly manufactured a state would price every
    stateless forward at the full state's bandwidth.
    """
    gateway, (x, v) = _gateway(), _inputs()
    with torch.no_grad():
        y, state = rola.rola_op(gateway(x), v)
    assert state is None
    assert y.shape == v.shape


def test_a_raw_tensor_is_not_a_state():
    gateway, (x, v) = _gateway(), _inputs()
    with torch.no_grad(), pytest.raises(TypeError, match="rola.RoLAState"):
        rola.rola_op(gateway(x), v, state=torch.zeros(1))


def test_an_unbuilt_strategy_is_refused_by_name():
    with pytest.raises(NotImplementedError, match="cow"):
        rola.state("cow")
    with pytest.raises(ValueError, match="strategy must be"):
        rola.state("mutable-ish")


# ---------------------------------------------------------------------------
# 2. bind, then continue
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expert", [None, _DENSE], ids=["paged", "dense"])
def test_two_chained_halves_are_one_whole_sequence(expert):
    """The entry state's gate at the facade: bind on the first call, continue on the
    second, and the pair must compute the one call over the whole sequence.

    Both comparisons go through a DERIVED envelope and neither is exact, for the same
    reason: cutting a sequence in two reassociates every per-leaf sum around the cut
    (an fp32 partial is carried across it), and `y` additionally fans in with
    `atomicAdd` over owner blocks. Nothing is dropped either way, which is what the
    envelopes bound.
    """
    gateway, (x, v) = _gateway(seed=3), _inputs(seed=3)
    half = _T // 2
    with torch.no_grad():
        chained = rola.state()
        y1, chained = rola.rola_op(gateway(x[:, :half]), v[:, :half], state=chained,
                                   expert=expert)
        y2, chained = rola.rola_op(gateway(x[:, half:]), v[:, half:], state=chained,
                                   expert=expert)
        routes = gateway(x)
        whole = rola.state()
        y, whole = rola.rola_op(routes, v, state=whole, expert=expert)

    assert chained.paged == (expert is None), "the expert's backing pin was not honoured"
    assert float(whole.materialize().abs().max()) > 0, "the state came back empty"
    got, want = chained.materialize(), whole.materialize()
    assert (got - want).abs().max() <= _state_envelope(want, routes), (
        "the chained state left the reassociation envelope, by "
        f"{(got - want).abs().max().item():.3e}")
    assert (y1 - y[:, :half]).abs().max() <= _envelope(y, routes), (
        "the first half's y moved: it consumes NO entry state, so it is the same "
        "arithmetic as the whole call's first half and owes the fan-in envelope")
    #: the SECOND half is a different claim and gets a different instrument. It reads
    #: through an entry state that the cut moved by the reassociation above, and the
    #: readout DIVIDES by an accumulated mass -- so a 1e-7 state difference reaches y
    #: amplified by 1/mass at the thinly-written leaves, which no reassociation bound
    #: covers. What is owed there is agreement at the family band, and that is what is
    #: asserted; the exact statement lives on the state.
    rel = float((y2 - y[:, half:]).norm() / y[:, half:].norm())
    assert rel < 1e-2, f"the continuation's y left the bf16 family band: rel={rel:.3e}"


def test_the_two_backings_compute_the_same_call():
    """One routing, two states, one of them pinned dense. The state is `torch.equal`.

    The ABI-level version of this is the paging-equivalence file; this is the FACADE's
    version, and what it adds is that the two backings are reachable through the public
    surface and are invisible in the answer.
    """
    gateway, (x, v) = _gateway(seed=5), _inputs(seed=5)
    with torch.no_grad():
        routes = gateway(x)
        paged, dense = rola.state(), rola.state()
        y_p, paged = rola.rola_op(routes, v, state=paged)
        y_d, dense = rola.rola_op(routes, v, state=dense, expert=_DENSE)
        y_control, _ = rola.rola_op(routes, v, state=rola.state(), expert=_DENSE)

    assert paged.paged and not dense.paged, "both states took the same backing"
    assert torch.equal(paged.materialize(), dense.materialize()), (
        "the paged and dense states differ; only the base translation should")
    bound = _envelope(y_d, routes)
    assert (y_d - y_control).abs().max() <= bound, "the null control blew its envelope"
    assert (y_d - y_p).abs().max() <= bound, "the paged y left the fan-in envelope"


@pytest.mark.parametrize("expert", [None, _DENSE], ids=["paged", "dense"])
def test_a_clone_is_independent(expert):
    """`clone()` is EAGER: the copy shares nothing, in either backing."""
    gateway, (x, v) = _gateway(seed=7), _inputs(seed=7)
    with torch.no_grad():
        state = rola.state()
        _y, state = rola.rola_op(gateway(x), v, state=state, expert=expert)
        copy = state.clone()
        snapshot = copy.materialize().clone()
        assert torch.equal(snapshot, state.materialize()), "the clone is not a copy"
        rola.rola_op(gateway(x), v, state=state, expert=expert)

    assert torch.equal(snapshot, copy.materialize()), (
        "mutating the original moved the clone: the copy shares its backing")
    assert not torch.equal(snapshot, state.materialize()), (
        "the second call moved nothing; the cell cannot see sharing")
    assert copy.paged == state.paged and copy.strategy == state.strategy


# ---------------------------------------------------------------------------
# 3. the shape a state is bound to
# ---------------------------------------------------------------------------

def test_a_bound_state_carries_the_shape_it_bound_and_nothing_else():
    """Binding is SHAPE ACQUISITION. The state took `[B, H, N, cols]`, a device and a
    backing from its first call, and it holds no fingerprint of the routing that gave
    them -- a widths/normalization interview would be a second contract to keep in
    step with the first, and a wrong-shaped call fails where the bytes are addressed
    exactly as it does for a tensor."""
    gateway, (x, v) = _gateway(seed=11), _inputs(seed=11)
    state = rola.state()
    assert not state.bound and state.shape is None
    with torch.no_grad():
        rola.rola_op(gateway(x), v, state=state)
    assert state.shape == (2, _H, prod(_WIDTHS), _DV + 1)
    assert state.device == torch.device("cuda", torch.cuda.current_device())
    assert state.dtype is torch.float32
    assert state.page_table is not None and state.paged


def test_a_call_the_state_was_not_bound_by_fails_where_the_bytes_are():
    """The named accepted hazard, gated at its real edge: a DIFFERENT-SHAPED call
    against a bound state raises (the addressing cannot be satisfied), and it raises
    from the shape machinery rather than from a policy interview."""
    gateway, (x, v) = _gateway(seed=11), _inputs(seed=11)
    other = _gateway(seed=12, widths=(64, 64))
    x2 = torch.randn(2, _T, _HIDDEN, device="cuda")
    v2 = torch.randn(2, _T, _H, 64, device="cuda")
    with torch.no_grad():
        state = rola.state()
        rola.rola_op(gateway(x), v, state=state)
        with pytest.raises(RuntimeError):
            rola.rola_op(other(x2), v2, state=state)


# ---------------------------------------------------------------------------
# 5. the layer is a pass-through
# ---------------------------------------------------------------------------

def test_the_layer_returns_the_layered_pair_always():
    """`forward(x, state=...) -> (y, (state, conv_rings))`, ALWAYS the pair -- and the
    rings are `None` here because no convolution is configured, which is the layer
    holding exactly the recurrences its own level introduced."""
    layer = build_layer(hidden_size=_HIDDEN, num_heads=_H, d_v=_DV, widths=_WIDTHS,
                        routing=union_routing(1), normalization="global", decay=None,
                        layer_idx=0).cuda().float().eval()
    x = torch.randn(2, _T, _HIDDEN, device="cuda")
    with torch.no_grad():
        y, (none, no_rings) = layer(x)
        assert none is None and no_rings is None
        state = rola.state()
        y1, (returned, rings) = layer(x, state=(state, None))
        assert returned is state, "the layer did not hand back the state it was given"
        assert state.paged and state.carries
        y2, (returned, rings) = layer(x, state=(returned, rings))
        assert rings is None, "a layer with no convolution has no ring to carry"
        with pytest.raises(TypeError, match=r"\(state, conv_rings\)"):
            layer(x, state=state)
    assert y.shape == y1.shape == y2.shape
    assert layer.last_execution_backend == "cuda"


# ---------------------------------------------------------------------------
# 6. a BC flip over a LIVE state
# ---------------------------------------------------------------------------

def test_a_BC_flip_over_a_live_state_neither_rebuilds_the_arena_nor_moves_a_byte(monkeypatch):
    """`BC` is chosen PER LAUNCH; residency is not. This is that claim at the facade.

    The retired `test_layer_paging.py` made it of a layer-held arena, forcing
    the decision through the function that made it. The arena is the STATE's now, and the
    chunk arm's `(C, BC)` comes from `rola.ops.chunk._arm_for` -- the one function both
    the facade (which needs `BC` for the arena's extent policy) and the launch read -- so
    scripting THAT is the same move against the same seam, not a reach around it.

    Two calls on one live state at two built `BC`s must: keep the SAME arena object (a
    rebuild would mean residency was keyed to the tile), let the state's extent policy
    follow the flip (it is policy, and policy may move), and compute the same bytes as a
    DENSE-backed state driven through the identical flip.
    `test_chunk_paging_equivalence.py` makes the ABI-level version of the byte claim; this
    one is about the object's lifecycle across the flip.
    """
    from rola.ops import chunk as chunk_module

    widths = (64, 64)  # the topology the built matrix gives more than one BC
    #: PINNED PER CALL, not per invocation of the seam: `_arm_for` is read several times
    #: inside one `rola_op` (the refusal check, the facade's extent policy, the launch),
    #: and all of those must agree or the call would plan one arm and launch another.
    forced = {"BC": 64}

    def scripted_arm_for(D, B):
        return (32, forced["BC"])

    gateway = _gateway(seed=11, widths=widths)
    x, v = _inputs(seed=11, T=64)
    with torch.no_grad():
        first, second = gateway(x[:, :32]), gateway(x[:, 32:])

    def chain(expert):
        state = rola.state()
        with torch.no_grad():
            forced["BC"] = 64
            rola.rola_op(first, v[:, :32], state=state, expert=expert)
            arena = state._arena
            bc = None if arena is None else arena.BC
            forced["BC"] = 128
            rola.rola_op(second, v[:, 32:], state=state, expert=expert)
        return state, arena, bc

    monkeypatch.setattr(chunk_module, "_arm_for", scripted_arm_for)
    paged, arena_after_first, bc_after_first = chain(None)
    dense, _, _ = chain(_DENSE)

    assert bc_after_first == 64 and forced["BC"] == 128, (
        "the script did not reach the state's extent policy, so the flip never happened")
    assert paged._arena is arena_after_first and paged._arena.BC == 64, (
        "the BC flip rebuilt the arena or re-keyed its extent policy; residency is not "
        "the compute tile's to move")
    assert paged.paged and not dense.paged, "both chains took the same backing"
    assert torch.equal(paged.materialize(), dense.materialize()), (
        "the paged chain across a BC flip differs from the dense one")
