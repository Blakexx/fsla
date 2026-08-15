# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""The T=1 decode fork's dispatch, and the autograd gap it must not open.

This is the `layer-dispatch-autograd-gap` gate, generalized to a SECOND kernel. The
failure class it exists to close: a forward-only kernel reachable from a
grad-requiring context does not raise — it returns a tensor with no ``grad_fn`` and the
graph is silently severed at this layer (the fleet-smoke ``x.grad = None`` failure,
fixed at ``fab325ae``). Both RoLA kernels are forward-only, so every row below asserts
**which branch actually ran**, read off ``execution_backend_counts`` — a timing
comparison would be an inference; a counter delta is a fact — or that the call was
REFUSED, which is what everything outside the built envelope now is.

``'decode'`` is a separate counter key rather than a sub-count of ``'cuda'`` precisely
so this table can tell the two CUDA branches apart. A test that could not would pass
while the wrong kernel ran.

**The threshold-adjacent rows are the point.** ``L = 1`` and ``L = 2`` are both
exercised because a mis-stated predicate hides exactly there, and both ``L = 1``
sub-cases (carried state present / absent) because that is the second half of the
predicate: a one-token call with NO carried state is a degenerate prefill, not a decode
step, and a one-token prefill is outside the chunk matrix — a refusal, not a step.

Run: ``pytest tests/layers/test_decode_dispatch.py``
"""

from __future__ import annotations

import pytest
import torch
from conftest import build_layer

import rola
from rola.expert import PlanOverrides
from rola.interface import BackwardNotImplemented

#: `[16, 16]` at `d_v = 64`: a BUILT chunk arm, so the rows that must reach the
#: PREFILL kernel can. Decode itself is indifferent (it is BC-blind and general in the
#: topology, and ran fine on the narrower fixture); the fixture is shaped by the arm
#: the contrast rows need, not by the arm under test.
_HIDDEN, _H, _DV, _BRANCHES = 128, 2, 64, (16, 16)
_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="the decode fork is a CUDA kernel")

#: Old `decay="none"`/`"leaf_mass"` -> the two shipped decay SOURCES `build_layer`
#: takes (`None`/`"learned"`); the exact rate is immaterial to these dispatch-only
#: gates, only whether a decay source is present at all.
_DECAY = {"none": None, "leaf_mass": "learned"}


def _layer(device, *, norm="global", decay="none"):
    return build_layer(
        hidden_size=_HIDDEN, num_heads=_H, d_v=_DV, widths=_BRANCHES,
        normalization=norm, decay=_DECAY[decay], layer_idx=0,
    ).to(device, torch.float32)


def _prime(layer, device, *, prefill=32):
    """Run a prefill so the next call has a carried state — the decode precondition.

    `prefill` is a multiple of the chunk `C = 32`: the prefill itself has to be an
    in-envelope call."""
    state = rola.state()
    x = torch.randn(1, prefill, _HIDDEN, device=device)
    with torch.no_grad():
        _out, state = layer(x, state=state)
    return state


def _step(layer, state, device, *, L=1, requires_grad=False, grad_enabled=True):
    x = torch.randn(1, L, _HIDDEN, device=device, requires_grad=requires_grad)
    with torch.set_grad_enabled(grad_enabled):
        out, _state = layer(x, state=state)
    return x, out


# ---------------------------------------------------------------------------
# the dispatch table
# ---------------------------------------------------------------------------

#: The refusal each out-of-envelope row must carry, by the clause that refused it.
_REFUSALS = {
    "refuses-training": (BackwardNotImplemented, "TRAINING mode"),
    "refuses-grad": (BackwardNotImplemented, "native backward"),
    "refuses-chunk": (NotImplementedError, "multiple of the chunk"),
}


@pytest.mark.cuda
@_CUDA
@pytest.mark.parametrize(
    "training,grad_enabled,requires_grad,L,carried,expected",
    [
        # training mode wins under any wrapper -- including the reentrant checkpointing
        # shape (training=True under no_grad), which is why that is its own row.
        (True, True, True, 1, True, "refuses-training"),
        (True, True, True, 1, False, "refuses-training"),
        (True, False, True, 1, True, "refuses-training"),
        # eval, but a gradient reaches the operation: refused before any fork.
        (False, True, True, 1, True, "refuses-grad"),
        (False, True, False, 1, True, "refuses-grad"),
        # eval mode, no grad: the fork's own predicate decides
        (False, False, False, 1, True, "decode"),
        # ... and the PREFILL rows it is being distinguished from. `T` is a multiple
        # of the chunk `C = 32` wherever the row must reach the kernel; a one-token
        # UNCARRIED call is a degenerate prefill, outside the built matrix, and is
        # refused -- a different statement from `decode` and exactly the distinction
        # being drawn.
        (False, False, False, 1, False, "refuses-chunk"),
        (False, False, False, 32, True, "cuda"),
        (False, False, False, 64, True, "cuda"),
    ],
)
def test_dispatch_table(training, grad_enabled, requires_grad, L, carried, expected):
    device = "cuda"
    layer = _layer(device)
    layer.eval()
    cache = _prime(layer, device) if carried else rola.state()
    layer.train(training)
    before = layer.execution_backend_counts

    if expected.startswith("refuses"):
        error, match = _REFUSALS[expected]
        with pytest.raises(error, match=match):
            _step(layer, cache, device, L=L, requires_grad=requires_grad,
                  grad_enabled=grad_enabled)
        assert layer.execution_backend_counts == before, (
            "the refusal must happen BEFORE a kernel runs; a counter that moved means "
            "some backend ran and then the call unwound")
        return

    _step(layer, cache, device, L=L, requires_grad=requires_grad,
          grad_enabled=grad_enabled)
    after = layer.execution_backend_counts
    moved = {k for k in after if after[k] != before[k]}
    assert moved == {expected}, f"expected the {expected!r} backend, counters moved {moved}"


@pytest.mark.cuda
@_CUDA
def test_a_cpu_decode_shaped_call_is_refused_at_the_prefill():
    """There is no CPU lowering to decline, so a CPU sequence cannot even be primed."""
    layer = _layer("cpu")
    layer.eval()
    with torch.no_grad(), pytest.raises(NotImplementedError, match="CUDA kernel"):
        _prime(layer, "cpu")


@pytest.mark.cuda
@_CUDA
def test_the_op_level_guard_raises_rather_than_redirecting():
    """Defense in depth for a DIRECT caller of the op.

    It raises; it does not silently redirect to the oracle. A silent redirect is the
    same class of failure (two implementations behind one call) that the layer's guard
    exists to prevent, pointed the other way.
    """
    from rola.ops.decode import _decode_step, freeze_decode_config
    from rola.routing.types import IndependentRouting, LevelConfig, SoftmaxActivation, Topology

    device = "cuda"
    widths, d_v = (8, 16), 64
    routing = IndependentRouting(tied=False, read_activation=SoftmaxActivation(),
                                 write_activation=SoftmaxActivation())
    topology = Topology(levels=tuple(LevelConfig(w, routing) for w in widths))
    config = freeze_decode_config(topology, "global", d_v=d_v, decay=False, BH=1, device=device,
                                  mma_precision="bf16")
    levels = tuple(torch.rand(1, 1, 1, w, device=device, requires_grad=True) for w in widths)
    g_write = torch.rand(1, 1, 1, device=device)
    v = torch.randn(1, 1, 1, d_v, device=device)
    state = torch.zeros(1, 1, config.N, config.cols, device=device)
    with pytest.raises(RuntimeError, match="forward-only"):
        _decode_step(v, levels, levels, None, g_write, config, state)


@pytest.mark.cuda
@_CUDA
def test_a_dense_pinned_layer_still_reaches_decode():
    """REPOINTED (P67 S3b): paging is not a layer configuration any more.

    This cell used to assert that a PAGED layer kept the tiled path at ``T = 1``,
    because decode had no page table and would have dropped the layer's cap. There is
    no cap and no layer-level paging flag: the backing is the state's, chosen at bind,
    and decode crosses it through `materialize`/`adopt_dense_contents` at its branch
    (docs/internals/state.md §5). What is still worth asserting is that the expert's
    surviving backing pin does not perturb the DISPATCH: a dense-pinned layer takes
    the decode fork at ``T = 1`` exactly as the default one does. Aligning decode with
    the paged backing proper is P67 S4.
    """
    device = "cuda"
    layer = build_layer(
        hidden_size=_HIDDEN, num_heads=_H, d_v=_DV, widths=_BRANCHES,
        normalization="global", expert=PlanOverrides(paging=False),
        layer_idx=0,
    ).to(device, torch.float32)
    layer.eval()
    with torch.no_grad():
        cache = _prime(layer, device)
        before = layer.execution_backend_counts
        _step(layer, cache, device, L=1, grad_enabled=False)  # noqa: E501
    after = layer.execution_backend_counts
    assert after['decode'] == before['decode'] + 1, "the decode fork did not run"
    assert after['cuda'] == before['cuda']


# ---------------------------------------------------------------------------
# H9 (coverage-closure): the table's rows swept over the decay axis the layer
# default never leaves. THE TWO ENVELOPES SEPARATE HERE: decode implements decay
# (at one token the rule-1 half-mass shift is the identity, so it forms one
# exponential and has no admissibility precondition), while the chunk arm does not
# and refuses by name. A row that moved the wrong way would mean one arm's envelope
# had been asked the other's question.
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@_CUDA
def test_a_decay_step_runs_on_decode_which_has_the_arm():
    """The state is primed by a NON-decay layer of the same shape, because a decay
    prefill has no arm to run on: the state carries a sequence, not a configuration."""
    device = "cuda"
    plain = _layer(device).eval()
    cache = _prime(plain, device)
    layer = _layer(device, decay="leaf_mass").eval()
    before = layer.execution_backend_counts
    _step(layer, cache, device, L=1, grad_enabled=False)
    after = layer.execution_backend_counts
    assert {k for k in after if after[k] != before[k]} == {"decode"}


@pytest.mark.cuda
@_CUDA
@pytest.mark.parametrize("L,carried", [(1, False), (2, True), (16, True)])
def test_every_decay_prefill_row_refuses_by_naming_decay(L, carried):
    device = "cuda"
    plain = _layer(device).eval()
    cache = _prime(plain, device) if carried else rola.state()
    layer = _layer(device, decay="leaf_mass").eval()
    before = layer.execution_backend_counts
    with pytest.raises(NotImplementedError, match="decay"):
        _step(layer, cache, device, L=L, grad_enabled=False)
    assert layer.execution_backend_counts == before


# ---------------------------------------------------------------------------
# the precision pin crosses the boundary (coverage-closure H9 -> terminal cycle)
# ---------------------------------------------------------------------------

@pytest.mark.cuda
@_CUDA
def test_the_frozen_decode_config_carries_the_layers_precision_pin():
    """The L == 1 fork must not drop ``mma_precision`` at the freeze boundary.

    Before this gate the carrier had NO ``mma_precision`` field and the layer's pin
    was silently dropped at ``L == 1`` (coverage-closure H9, report-only). The decode
    GEMV is scalar fp32 for every pin — the field selects nothing today — but the
    boundary must CARRY the pin so nothing downstream can disagree with the layer
    (the feeder-flags lesson, same class).

    REPOINTED (P67 S4): the carrier is cached on the STATE, not on the layer -- it is
    per-sequence launch machinery, like the short-conv rings beside it -- so the pin is
    read off ``state._decode_carrier_cache`` where it used to be read off
    ``layer._decode_config``. The claim is unchanged.

    P26: ``mma_precision`` is a dead axis -- ``PlanOverrides`` and
    ``freeze_decode_config`` both accept only ``"bf16"`` (a second, ``"tf32x3"``,
    pin used to be exercised here to prove the carrier does not re-hardcode a
    default; there is no second value left to try, so
    ``test_freeze_decode_config_refuses_an_unknown_precision`` below is what proves
    the carrier does not silently coerce an off-pin value instead).
    """
    device = "cuda"
    layer = build_layer(
        hidden_size=_HIDDEN, num_heads=_H, d_v=_DV, widths=_BRANCHES,
        normalization="global", layer_idx=0,
    ).to(device, torch.float32)
    layer.eval()
    cache = _prime(layer, device)
    before = layer.execution_backend_counts
    _step(layer, cache, device, grad_enabled=False)
    after = layer.execution_backend_counts
    assert after['decode'] == before['decode'] + 1, "the decode fork did not run"
    frozen = cache._decode_carrier_cache
    assert frozen is not None, "the decode fork ran but froze no carrier"
    assert frozen[1].mma_precision == "bf16", (
        f"the layer's bf16 pin was dropped at the boundary: the frozen carrier says "
        f"{frozen[1].mma_precision!r}")


def test_freeze_decode_config_refuses_an_unknown_precision():
    """The carrier validates the pin at the freeze, not downstream of it."""
    from rola.ops.decode import freeze_decode_config
    from rola.routing.types import IndependentRouting, LevelConfig, SoftmaxActivation, Topology
    routing = IndependentRouting(tied=False, read_activation=SoftmaxActivation(),
                                 write_activation=SoftmaxActivation())
    topology = Topology(levels=tuple(LevelConfig(w, routing) for w in _BRANCHES))
    with pytest.raises(ValueError, match="mma_precision"):
        freeze_decode_config(topology, "global", d_v=_DV, decay=False, BH=1,
                             device="cpu", mma_precision="fp16")
