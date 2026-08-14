# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The facade's stateful half: one object, one predicate (docs/internals/state.md).

What this file gates, and nothing else: the SURFACE. That the predicate is
`state is not None`; that an unbound state binds from its first call and a bound one
continues; that a chained pair of halves computes what one whole call computes; that
`clone` is independent; that the two backings are the same answer; that a state
crosses to the reference arm and back. The kernel-level claims underneath it -- the
exact atom bitmap, dense-vs-paged bit identity at the ABI, zero-page inertness -- are
`tests/tier2_geometry/test_atom_bitmap.py` and
`tests/tier2_geometry/test_chunk_paging_equivalence.py`, and are not restated here.

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
from rola.routing.producer import sparse_routing  # noqa: E402

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel arm is a CUDA kernel"),
]

_HIDDEN, _H, _DV, _WIDTHS, _T = 64, 2, 64, (16, 16), 128
_DENSE = PlanOverrides(paging=False)


def _gateway(seed=0, widths=_WIDTHS):
    torch.manual_seed(seed)
    return build_routes(hidden_size=_HIDDEN, num_heads=_H, d_v=_DV, widths=widths,
                        routing=sparse_routing()).cuda()


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
# 3. the refusals, which are computational impossibilities only
# ---------------------------------------------------------------------------

def test_a_bound_state_refuses_an_incompatible_call_by_naming_the_field():
    gateway, (x, v) = _gateway(seed=11), _inputs(seed=11)
    other = _gateway(seed=12, widths=(64, 64))
    x2 = torch.randn(2, _T, _HIDDEN, device="cuda")
    v2 = torch.randn(2, _T, _H, 64, device="cuda")
    with torch.no_grad():
        state = rola.state()
        rola.rola_op(gateway(x), v, state=state)
        with pytest.raises(ValueError, match="widths"):
            rola.rola_op(other(x2), v2, state=state)
        with pytest.raises(ValueError, match=r"\(batch, heads\)"):
            rola.rola_op(gateway(x[:1]), v[:1], state=state)


# ---------------------------------------------------------------------------
# 4. crossing the arms
# ---------------------------------------------------------------------------

def test_a_state_crosses_to_the_reference_arm_and_back():
    """The oracle keeps raw tensors, so the FACADE crosses the boundary both ways.

    A decay call dispatches to the reference arm; its fp64 result is stored back into
    the state as dense contents, and the next kernel-arm call RE-ADMITS them from the
    contents' own atom occupancy. The claim is that the round trip is lossless at the
    dtype it was stored in and that the continuation runs.
    """
    from rola.routing.decay import LearnedDecay

    gateway, (x, v) = _gateway(seed=13), _inputs(seed=13)
    decay = LearnedDecay(2.0 ** -8, widths=_WIDTHS, num_heads=_H).cuda()
    with torch.no_grad():
        state = rola.state()
        y_ref, state = rola.rola_op(gateway(x), v, decay(), state=state)
        assert y_ref.dtype == torch.float64, "the reference arm did not run"
        assert not state.paged, "the reference arm's result must be dense-backed"
        carried = state.materialize()
        assert carried.dtype == torch.float64, "the fp64 oracle's state was rounded"

        y, state = rola.rola_op(gateway(x), v, state=state)
    assert state.paged, "the kernel arm did not re-admit the dense contents into pages"
    assert torch.isfinite(y).all()
    occupied = (carried != 0).any(-1).any(-1).sum()
    resident = int((state._arena.page_table >= 0).sum())
    assert resident * 16 >= int(occupied), (
        "the re-admission committed fewer leaves than the dense contents occupy")


def test_a_readmitted_state_carries_the_same_contents():
    """The bridge is a MOVE, not a reset: what was carried is still there afterwards.

    Checked by comparing a reference-arm-then-kernel-arm continuation against the same
    contents continued on a state that never left the kernel arm. Both entry states are
    the same numbers; only the route they took into the launch differs.
    """
    gateway, (x, v) = _gateway(seed=17), _inputs(seed=17)
    half = _T // 2
    with torch.no_grad():
        first, second = gateway(x[:, :half]), gateway(x[:, half:])
        direct = rola.state()
        rola.rola_op(first, v[:, :half], state=direct)
        y_direct, direct = rola.rola_op(second, v[:, half:], state=direct)

        crossed = rola.state()
        rola.rola_op(first, v[:, :half], state=crossed)
        #: force the dense-contents route without changing the numbers: store the
        #: state's own contents back through the reference arm's bridge.
        crossed.adopt_dense_contents(crossed.materialize().clone())
        assert not crossed.paged
        y_crossed, crossed = rola.rola_op(second, v[:, half:], state=crossed)

    assert crossed.paged, "the re-admission did not happen"
    assert torch.equal(direct.materialize(), crossed.materialize()), (
        "the re-admitted continuation computed a different state")
    #: `y` is the fan-in's, so it is held to its envelope rather than to equality --
    #: the two runs differ in RESIDENCY (the re-admission unions the carried
    #: occupancy in), and a zero page is inert but an atomic order is not repeatable.
    assert (y_direct - y_crossed).abs().max() <= _envelope(y_direct, second)


# ---------------------------------------------------------------------------
# 5. the layer is a pass-through
# ---------------------------------------------------------------------------

def test_the_layer_mirrors_the_op():
    """`forward(x, state=...) -> (y, state)`, and a stateless layer call is `(y, None)`."""
    layer = build_layer(hidden_size=_HIDDEN, num_heads=_H, d_v=_DV, widths=_WIDTHS,
                        routing=sparse_routing(), normalization="global", decay=None,
                        layer_idx=0).cuda().float().eval()
    x = torch.randn(2, _T, _HIDDEN, device="cuda")
    with torch.no_grad():
        y, none = layer(x)
        assert none is None
        state = rola.state()
        y1, returned = layer(x, state=state)
        assert returned is state, "the layer did not hand back the state it was given"
        assert state.paged and state.carries
        y2, returned = layer(x, state=state)
    assert y.shape == y1.shape == y2.shape
    assert layer.last_execution_backend == "cuda"
