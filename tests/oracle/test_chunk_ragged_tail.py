"""A sequence length off the chunk quantum, against the fp64 oracle.

Vocabulary: `T` = tokens in the call; `C` = the arm's chunk, the MAXIMUM number of
live entries one super-chunk gathers; `widths` = the per-level routing widths.

The consumer's schedule is a COMPACTION over live tokens, not a tiling of the token
axis: the walk ballots one 32-token word at a time and a super-chunk closes when a
side reaches `C`, when the window line is crossed, or when the token axis runs out.
A super-chunk that closes short is therefore the ordinary outcome -- it is what every
sparsely-routed chunk already produced -- and the last one of a ragged call is the
same object. What made `T` off the quantum unreachable was host-side guards, not a
structure in the walk, which is why this file's claims are the same claims the
quantum-aligned cells make and not weaker ones.

A flat and a depth-four topology both run because the produce's outer chain differs
between them, and the reverse cells run every topology the scans carry, which is now
that same span; the stateful cells run because the state fold at a short final
super-chunk is the one place a wrong tail would show as a WRONG NUMBER rather than
as noise. The
continuation chain is the sharpest of them: a first call whose length is off the
quantum hands its final state to a second, and the pair must be the whole sequence.
"""
import pytest
import torch

import rola
from rola import rola_op
from rola.engine import rules
from rola.engine.facts import manifest
from rola.ops.naive import naive_rola
from tests.conftest import build_routes

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the chunk consumer is a CUDA kernel"),
]

_HIDDEN = 128
_HEADS = 2
_DV = 64

#: Around the 32-token ballot word and around the chunk `C = 32` both: one token, a
#: word that is only part-full, the last token before a word closes, the first token
#: after one, and two lengths whose tails are neither.
_RAGGED = (1, 5, 31, 33, 63, 100)

#: The band every comparison below is held to. It is the bf16 family band the rest of
#: the oracle suite carries: the kernel's operands are bf16 and its sums fp32, the
#: oracle is fp64 throughout, and nothing about a short final super-chunk changes
#: which roundings separate them.
_BAND = 1e-2


def _bundle(widths, T, seed):
    torch.manual_seed(seed)
    producer = build_routes(hidden_size=_HIDDEN, num_heads=_HEADS, widths=widths).cuda()
    x = torch.randn(1, T, _HIDDEN, device="cuda")
    v = torch.randn(1, T, _HEADS, _DV, device="cuda")
    return producer, x, v


def _oracle(routes, v, *, state=None, want_state=False):
    return naive_rola(v, routes.read_simplex(), routes.write, routes.g_read, routes.g_write,
                      routes.topology, None, routes.normalization,
                      initial_state=state, output_final_state=want_state)


def _relative(got, want):
    return float((got.double() - want.double()).norm() / want.double().norm())


@pytest.mark.parametrize("widths", [(64, 64), (8, 8, 8, 8)])
@pytest.mark.parametrize("T", _RAGGED)
def test_a_ragged_stateless_call_sits_at_the_oracle(widths, T):
    producer, x, v = _bundle(widths, T, seed=T)
    with torch.no_grad():
        routes = producer(x)
        y_ref, _ = _oracle(routes, v)
        y_ker, state = rola_op(routes, v)
    assert state is None
    rel = _relative(y_ker, y_ref)
    assert rel < _BAND, f"ragged T={T} off the oracle: rel={rel:.3e}"


@pytest.mark.parametrize("widths", [(64, 64), (8, 8, 8, 8)])
@pytest.mark.parametrize("T", _RAGGED)
def test_a_ragged_call_folds_the_state_the_recurrence_folds(widths, T):
    """The tail's STATE, which is the claim a short final super-chunk could break.

    `y` alone would not catch a fold that dropped the last partial super-chunk's
    deposit: the readout of those tokens is dominated by their own writes. The final
    state is where a dropped or double-counted tail deposit is the answer.
    """
    producer, x, v = _bundle(widths, T, seed=T + 7)
    with torch.no_grad():
        routes = producer(x)
        y_ref, s_ref = _oracle(routes, v, want_state=True)
        y_ker, state = rola_op(routes, v, state=rola.state())
        s_ker = state.materialize()
    assert _relative(y_ker, y_ref) < _BAND
    rel = _relative(s_ker, s_ref)
    assert rel < _BAND, f"ragged T={T} folded the wrong state: rel={rel:.3e}"


@pytest.mark.parametrize("widths", [(64, 64), (8, 8, 8, 8)])
@pytest.mark.parametrize("first,second", [(33, 31), (1, 63), (5, 95), (31, 1)])
def test_a_ragged_prefill_continues_into_the_next_call(widths, first, second):
    """THE CELL THAT CATCHES A WRONG TAIL FOLD: neither leg is on the quantum, and
    the second reads through the state the first's short final super-chunk wrote.

    A tail folded at the wrong token count is invisible in the first leg's `y` and
    arrives in the second leg's readout through the entry state. The comparison is
    against the ONE oracle call over the whole sequence, which knows nothing about
    super-chunks.
    """
    T = first + second
    producer, x, v = _bundle(widths, T, seed=first * 100 + second)
    with torch.no_grad():
        y_ref, s_ref = _oracle(producer(x), v, want_state=True)
        state = rola.state()
        y1, state = rola_op(producer(x[:, :first]), v[:, :first], state=state)
        y2, state = rola_op(producer(x[:, first:]), v[:, first:], state=state)
        s_ker = state.materialize()
    assert _relative(torch.cat([y1, y2], dim=1), y_ref) < _BAND
    rel = _relative(s_ker, s_ref)
    assert rel < _BAND, f"the ragged continuation's state left the band: rel={rel:.3e}"


#: The reverse pass' arm matrix is narrower than the forward's, so the backward cells
#: run the topologies the SCANS carry -- read off the manifest rather than listed, so
#: a widened scan matrix is covered here the day it lands.
_SCAN_TOPOLOGIES = sorted({(depth, width) for _c, _bc, depth, width in manifest.scan_arms()})


@pytest.mark.parametrize("depth,width", _SCAN_TOPOLOGIES)
@pytest.mark.parametrize("T", _RAGGED)
def test_the_reverse_pass_walks_the_tail_too(depth, width, T):
    """A GRAD-BEARING ragged call, both gradients, against the oracle's autograd.

    Both scans walk the same `Ascent`/`Descent` schedule the forward does, so a tail
    the forward learned and the reverse did not would be a silently wrong gradient
    rather than a refusal. This is the cell that says it is not.

    BOTH discrepancies are measured against ONE scale, the reference `d v` norm, and
    that is not a convenience: at `T = 1` the readout is a ratio over a state written
    by that same token, so the routing very nearly cancels out of `y` and the
    reference `d x` is a cancellation RESIDUE three orders below the readout's own
    gradient scale. A relative-to-itself claim there would be measuring the
    cancellation. The common scale states what is actually owed -- that neither
    gradient moves by more than the bf16 family band OF THE GRADIENT -- and it is the
    same statement at every `T`.
    """
    widths = (width,) * depth
    arm = rules.arm(widths, manifest.chunk_arms())
    assert arm is not None, f"the scan manifest carries {widths} but the forward does not"
    producer, x0, v0 = _bundle(widths, T, seed=T * 17 + depth * 7 + width)
    dy = torch.randn(1, T, _HEADS, _DV, device="cuda")

    grads = {}
    for arm_name in ("kernel", "oracle"):
        x = x0.clone().requires_grad_(True)
        v = v0.clone().requires_grad_(True)
        routes = producer(x)
        y = (rola_op(routes, v)[0] if arm_name == "kernel" else _oracle(routes, v)[0])
        (y.to(torch.float64) * dy.to(torch.float64)).sum().backward()
        grads[arm_name] = (x.grad.detach().double(), v.grad.detach().double())

    scale = grads["oracle"][1].norm()
    for i, name in enumerate(("d routing", "d v")):
        moved = float((grads["kernel"][i] - grads["oracle"][i]).norm() / scale)
        assert moved < _BAND, f"{name} at ragged T={T}, {widths}, {arm}: {moved:.3e}"
