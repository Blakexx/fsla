# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""ONE sequence, TWO consumers, ONE state object — the prefill/decode seam at the facade.

`test_decode_vs_oracle.py` gates the decode kernel and `test_decode_paging.py` gates its
addressing. This file gates the SEAM: that a sequence prefilled on the chunk arm through
`rola_op` continues, on the same `RoLAState`, through `decode_forward` steps — and that
the pair computes what the fp64 oracle computes over the whole sequence in one go. The
state's backing (paged by default) is invisible across that seam, and the decode steps'
GROWTH happens inside it, which is what makes this the growth gate at the real path
rather than at the arena.

Vocabulary, defined here and used unqualified below: `T` = prefill tokens; `S` = decode
steps; `C` = the chunk arm's token chunk; `N` = leaf count; `cols = d_v + 1`;
`owners = N / BC`; **atom** = the 16-leaf page granule.

**WHERE THE COMPARISON IS EXACT AND WHERE IT IS NOT.** Nowhere here is it exact against
the oracle, and that is a property of the arithmetic rather than of the code:

* the chunk arm sums a leaf's tokens in an `m16n8k16` fragment tree per owner and fans
  owners in with `atomicAdd`; the oracle runs an fp64 serial recurrence. The STATE's
  difference is therefore a REASSOCIATION, and it gets a bound DERIVED from the arm
  (`chunks^2 * eps * max`, the instrument `test_state_facade.py` already derives) —
  never a tuned tolerance.
* `y` DIVIDES by an accumulated mass, so a state difference reaches it amplified by
  `1 / mass` at thinly-written leaves. No reassociation bound covers that, and the
  decode suite's family band (`DECODE_RTOL`, which is the tiled kernel's own bf16
  number) is what `y` is held to. This is exactly the D1/D2 split P67 S3b recorded for
  the chunk arm's own chained-halves cell.

Run: ``pytest tests/integration/test_decode_continuation.py``
"""
from __future__ import annotations

import sys
from math import prod
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import build_routes  # noqa: E402

import rola  # noqa: E402
from rola.expert import PlanOverrides  # noqa: E402
from rola.ops import chunk as _chunk  # noqa: E402
from rola.ops.decode import decode_forward  # noqa: E402
from rola.ops.naive import _leaf_product, _radix_strides, naive_rola  # noqa: E402
from rola.routing.producer import union_routing  # noqa: E402
from tests.oracle.test_decode_vs_oracle import DECODE_RTOL  # noqa: E402

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="both arms are CUDA kernels"),
]

#: `(16, 16)` at `d_v = 64` is the uniform-width topology the chunk arm has a built arm
#: for; `T` is a multiple of the arm's `C`. `union_routing` (exact entmax) is what makes
#: the write support genuinely sparse, so atoms are genuinely absent and the decode
#: steps' growth is reachable rather than vacuous.
_HIDDEN, _H, _DV, _WIDTHS, _T, _S, _B = 64, 2, 64, (16, 16), 128, 6, 2
_DENSE = PlanOverrides(paging=False)


def _producer(seed=0):
    torch.manual_seed(seed)
    return build_routes(hidden_size=_HIDDEN, num_heads=_H, widths=_WIDTHS,
                        routing=union_routing(1)).cuda()


def _inputs(seed=0):
    torch.manual_seed(seed + 500)
    return (torch.randn(_B, _T + _S, _HIDDEN, device="cuda"),
            torch.randn(_B, _T + _S, _H, _DV, device="cuda"))


def _state_bound(routes, v, prefill):
    """`[B, H, N, cols]`: the most the CHAINED state may differ from the oracle's.

    The instrument `test_chunk_final_state.py` derives, extended across the seam by the
    one fact that distinguishes the two arms: the chunk consumer narrows its operands to
    bf16 (`chunk.run`'s four `.to(torch.bfloat16)` casts), and decode has no MMA and
    never leaves fp32. So a leaf's state is a sum of `T + S` fp64-exact terms of which
    only the first `prefill` were rounded to bf16 on the way in:

        (D + 2) * u_bf16 * sum_{t < prefill} |w_t| |deposit_t|          the operand roundings
      + (T + S) * eps_fp32 * sum_{t} |w_t| |deposit_t|                  the accumulation

    `(D + 2)` is the `D` routing levels plus the side gain plus `v`. The term sums are
    materialized through the ORACLE's address map, so the bound never borrows either
    kernel's idea of which leaf is which.
    """
    B, T, H = v.shape[0], v.shape[1], v.shape[2]
    N = prod(routes.topology.widths)
    w64 = routes.g_write.double()[..., None] * _leaf_product(
        tuple(level.double() for level in routes.write),
        _radix_strides(tuple(routes.topology.widths)), N)
    deposit = torch.cat((v.double(), torch.ones((B, T, H, 1), device="cuda",
                                                dtype=torch.float64)), dim=-1)
    def _scale(upto):
        return torch.einsum("bthn,bthv->bhnv", w64[:, :upto].abs(),
                            deposit[:, :upto].abs())

    scale_all, scale_pre = _scale(T), _scale(prefill)
    u = torch.finfo(torch.bfloat16).eps / 2
    return ((len(routes.topology.widths) + 2) * u * scale_pre
            + T * torch.finfo(torch.float32).eps * scale_all)


def _state_envelope(state, routes, steps):
    """The most the STATE can move between TWO KERNEL runs of the same tokens.

    Both runs narrow the same operands the same way, so what is left between them is
    pure REASSOCIATION: the prefill sums over its `T / C` chunks and each decode step
    carries an fp32 partial forward. DERIVED from the arm, exactly as the facade suite
    derives it, never tuned -- and it is a much tighter instrument than
    :func:`_state_bound`, which is why the cross-backing cell uses this one and the
    oracle cells use that one.
    """
    n = routes.tokens // _chunk.arm(routes)[0] + steps
    return n * n * torch.finfo(torch.float32).eps * float(state.abs().max())


@torch.no_grad()
def _run(producer, x, v, *, expert=None):
    """Prefill on the chunk arm, then `S` decode steps, all on ONE state object.

    Under `no_grad` because carried-state execution is inference-only end to end -- the
    layer refuses it outright and the decode op's own guard refuses it again
    (the layer-dispatch autograd gap). The producer's parameters require grad, so this
    is the same context a served model runs in, not a convenience.
    """
    state = rola.state()
    y_pre, state = rola.rola_op(producer(x[:, :_T]), v[:, :_T], state=state, expert=expert)
    ys = [y_pre]
    for step in range(_S):
        token = slice(_T + step, _T + step + 1)
        y, state = decode_forward(producer(x[:, token]), v[:, token], state, expert=expert)
        ys.append(y)
    return torch.cat([t.double() for t in ys], dim=1), state


@torch.no_grad()
def _oracle(producer, x, v):
    routes = producer(x)
    return naive_rola(v.double(), routes.read_simplex(), routes.write, routes.g_read, routes.g_write,
                      routes.topology, None, routes.normalization, output_final_state=True)


# ---------------------------------------------------------------------------
# the fixture's own precondition
# ---------------------------------------------------------------------------

def test_the_producer_is_token_wise_so_the_whole_sequence_is_its_own_prefix():
    """Slicing the STREAM and slicing the ROUTING are the same object.

    Stated as a gate rather than assumed, because every cell below compares a chained
    run against an oracle run over the CONCATENATED sequence, and that comparison is
    only meaningful if the producer sees one token at a time. A producer that mixed
    tokens (a convolution, an attention) would make this file's whole construction
    invalid rather than merely inaccurate, and it would do so silently.
    """
    producer = _producer()
    x, _v = _inputs()
    full, head = producer(x), producer(x[:, :_T])
    for level, (a, b) in enumerate(zip(full.read, head.read)):
        assert torch.equal(a[:, :_T], b), f"read level {level} is not token-wise"
    for level, (a, b) in enumerate(zip(full.write, head.write)):
        assert torch.equal(a[:, :_T], b), f"write level {level} is not token-wise"
    assert torch.equal(full.g_write[:, :_T], head.g_write)


# ---------------------------------------------------------------------------
# 1. the seam, against the fp64 oracle
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backing", ["paged", "dense"], ids=["paged", "dense"])
def test_a_prefilled_sequence_continues_through_decode_steps(backing):
    """Prefill `T` on the chunk arm, step `S` tokens on decode, versus ONE oracle pass.

    The decode steps run through the state facade, so on the paged arm each of them
    plans its own write atoms into the live arena -- this is the growth path at the real
    seam. An entry state the decode kernel read at the wrong offset, a head fold that
    disagreed with the chunk arm's, or an admission that dropped a deposit would each
    show up here and nowhere else.
    """
    producer = _producer()
    x, v = _inputs()
    expert = None if backing == "paged" else _DENSE
    y, state = _run(producer, x, v, expert=expert)
    assert state.paged is (backing == "paged"), "the backing under test is not the one bound"

    y_ref, state_ref = _oracle(producer, x, v)
    got = state.materialize().double()
    with torch.no_grad():
        bound = _state_bound(producer(x), v, _T)
    excess = float(((got - state_ref).abs() - bound).max())
    assert excess <= 0.0, f"the chained state left its derived bound by {excess:.3e}"
    scale = max(1e-30, float(y_ref.abs().max()))
    err = float((y - y_ref).abs().max()) / scale
    assert err < DECODE_RTOL, f"the chained y left the family band: {err:.3e}"


def test_the_two_backings_compute_the_same_continuation():
    """Paged and dense are the same sequence of calls, so they are the same answer.

    Not `torch.equal` and that is honest rather than weak: the two runs differ in the
    CHUNK arm's `y` fan-in (`atomicAdd` over owners is not reproducible against itself),
    which the envelope covers; the decode halves of the two runs are bit-identical by
    `test_decode_paging.py`'s claim 2, and the STATE is compared exactly where the
    arm allows it.
    """
    producer = _producer(seed=3)
    x, v = _inputs(seed=3)
    y_paged, paged = _run(producer, x, v)
    y_dense, dense = _run(producer, x, v, expert=_DENSE)
    assert paged.paged and not dense.paged

    routes = producer(x[:, :_T])
    state_p, state_d = paged.materialize().double(), dense.materialize().double()
    bound = _state_envelope(state_d, routes, _S)
    assert float((state_p - state_d).abs().max()) <= bound
    owners = prod(_WIDTHS) // _chunk.arm(routes)[1]
    y_bound = owners * owners * torch.finfo(torch.float32).eps * float(y_dense.abs().max())
    assert float((y_paged - y_dense).abs().max()) <= y_bound


# ---------------------------------------------------------------------------
# 2. the seam, the other way
# ---------------------------------------------------------------------------

def test_a_decoded_sequence_continues_back_onto_the_chunk_arm():
    """decode -> chunk, which is the direction a speculative or chunked-prefill loop takes.

    The state is the only thing that crosses, so the arm that reads it next is the
    caller's business: the arena a decode step grew is the arena the chunk launch
    resolves through, with no re-admission and no copy.
    """
    producer = _producer(seed=5)
    x, v = _inputs(seed=5)
    y, state = _run(producer, x, v)
    resident = int(state.bound and (state.page_table >= 0).sum())

    tail = slice(_T + _S, None)
    x2 = torch.randn(_B, _chunk.arm(producer(x[:, :_T]))[0], _HIDDEN, device="cuda")
    v2 = torch.randn(_B, x2.shape[1], _H, _DV, device="cuda")
    with torch.no_grad():
        y2, state = rola.rola_op(producer(x2), v2, state=state)
    assert state.paged, "the chunk continuation abandoned the paged backing"
    assert int((state._arena.page_table >= 0).sum()) >= resident, (
        "the chunk continuation dropped atoms the decode steps had grown")

    #: the oracle over the WHOLE thing: prefill, the decode steps, then the chunk tail.
    x_all = torch.cat([x, x2], dim=1)
    v_all = torch.cat([v, v2], dim=1)
    _y_ref, state_ref = _oracle(producer, x_all, v_all)
    got = state.materialize().double()
    with torch.no_grad():
        #: the bf16 half of the bound covers the PREFILL and the chunk TAIL alike; the
        #: `S` decode tokens sit between them and are fp32 throughout, so charging them
        #: the operand rounding too is the safe direction and is stated rather than
        #: quietly taken.
        bound = _state_bound(producer(x_all), v_all, x_all.shape[1])
    excess = float(((got - state_ref).abs() - bound).max())
    assert excess <= 0.0, (
        f"the decode->chunk continuation left its derived bound by {excess:.3e}")
    assert tail.start == _T + _S


# ---------------------------------------------------------------------------
# 3. the refusal that is a computational impossibility
# ---------------------------------------------------------------------------

def test_a_decode_step_on_a_state_that_carries_nothing_is_refused_by_name():
    """A step CONTINUES; there is no prefill at one token and no entry state to read.

    Refused rather than silently prefilled, because a one-token call with no carried
    state is a degenerate PREFILL and belongs on the other arm -- which is exactly the
    predicate `RoLA.forward` uses (`tests/integration/test_decode_dispatch.py`).
    """
    producer = _producer(seed=7)
    x, v = _inputs(seed=7)
    with pytest.raises(RuntimeError, match="carries nothing"):
        decode_forward(producer(x[:, :1]), v[:, :1], rola.state())
