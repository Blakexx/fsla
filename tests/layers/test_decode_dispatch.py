# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""The T=1 decode fork's dispatch, and the autograd gap it must not open.

This is the `layer-dispatch-autograd-gap` gate, generalized to a THIRD backend. The
failure class it exists to close: a forward-only kernel reachable from a
grad-requiring context does not raise — it returns a tensor with no ``grad_fn`` and the
graph is silently severed at this layer (the fleet-smoke ``x.grad = None`` failure,
fixed at ``fab325ae``). The decode kernels are forward-only, so every row below asserts
BOTH the numeric outcome AND **which branch actually ran**, read off
``execution_backend_counts``. A timing comparison would be an inference; a counter
delta is a fact.

``'decode'`` is a separate counter key rather than a sub-count of ``'cuda'`` precisely
so this table can tell the two CUDA branches apart. A test that could not would pass
while the wrong kernel ran.

**The threshold-adjacent rows are the point.** ``L = 1`` and ``L = 2`` are both
exercised because a mis-stated predicate hides exactly there, and both ``L = 1``
sub-cases (carried state present / absent) because that is the second half of the
predicate: a one-token call with NO carried state is a degenerate prefill, not a decode
step, and must take the tiled path.

Run: ``pytest tests/layers/test_decode_dispatch.py``
"""

from __future__ import annotations

import pytest
import torch
from conftest import build_layer

from rola.expert import PlanOverrides
from rola.state import RoLAState

#: `[8, 16]` at `d_v = 64` so the TILED arm lands on an instantiated `(BC, d_v)` --
#: `BC` defaults to the innermost width, and `BC = 8` is not in the consumer's set.
#: Decode itself is indifferent (it is BC-blind, and ran fine on the narrower fixture),
#: but the rows that must reach the tiled kernel have to be launchable there.
_HIDDEN, _H, _DV, _BRANCHES = 128, 2, 64, (8, 16)
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


def _prime(layer, device, *, prefill=4, requires_grad=False):
    """Run a prefill so the next call has a carried state — the decode precondition."""
    cache = RoLAState()
    x = torch.randn(1, prefill, _HIDDEN, device=device, requires_grad=requires_grad)
    layer(x, past_key_values=cache, use_cache=True)
    return cache


def _step(layer, cache, device, *, L=1, requires_grad=False, grad_enabled=True):
    x = torch.randn(1, L, _HIDDEN, device=device, requires_grad=requires_grad)
    with torch.set_grad_enabled(grad_enabled):
        out, _, _ = layer(x, past_key_values=cache, use_cache=True)
    return x, out


# ---------------------------------------------------------------------------
# the dispatch table
# ---------------------------------------------------------------------------

@pytest.mark.cuda
@_CUDA
@pytest.mark.parametrize(
    "training,grad_enabled,requires_grad,L,carried,expected",
    [
        # A CARRIED STATE UNDER A GRAD CONTEXT RAISES, BEFORE ANY DISPATCH RUNS.
        # This is a pre-existing layer invariant ("RoLA carried-state execution is
        # inference-only") and it is STRONGER than routing such a call to the oracle:
        # a carried state is the decode fork's own precondition, so the fork is
        # MUTUALLY EXCLUSIVE with gradient reachability at the layer boundary and
        # could not sever a graph even if both of its own guards were removed. These
        # are rows rather than an implicit assumption because that exclusivity is
        # load-bearing for the whole autograd argument, and any future relaxation of
        # the raise has to come through here.
        (True, True, True, 1, True, 'raises'),
        (False, True, True, 1, True, 'raises'),
        (False, True, False, 1, True, 'raises'),
        # training mode wins under any wrapper -- including the reentrant checkpointing
        # shape (training=True under no_grad), which is why that is its own row.
        (True, True, True, 1, False, 'oracle'),
        (True, False, True, 1, True, 'oracle'),
        # eval mode, no grad: the fork's own predicate decides
        (False, False, False, 1, True, 'decode'),
        (False, False, False, 1, False, 'cuda'),
        (False, False, False, 2, True, 'cuda'),
        (False, False, False, 16, True, 'cuda'),
    ],
)
def test_dispatch_table(training, grad_enabled, requires_grad, L, carried, expected):
    device = "cuda"
    layer = _layer(device)
    layer.train(training)
    if carried:
        was_training = layer.training
        layer.eval()
        cache = _prime(layer, device)
        layer.train(was_training)
    else:
        cache = RoLAState()
    before = layer.execution_backend_counts

    if expected == 'raises':
        with pytest.raises(RuntimeError, match="inference-only"):
            _step(layer, cache, device, L=L, requires_grad=requires_grad,
                  grad_enabled=grad_enabled)
        assert layer.execution_backend_counts == before, (
            "the refusal must happen BEFORE dispatch; a counter that moved means some "
            "backend ran and then the call unwound")
        return

    x, out = _step(layer, cache, device, L=L, requires_grad=requires_grad,
                   grad_enabled=grad_enabled)
    after = layer.execution_backend_counts
    moved = {k for k in after if after[k] != before[k]}
    assert moved == {expected}, f"expected the {expected!r} backend, counters moved {moved}"

    # THE AUTOGRAD HALF: wherever the graph must survive, it must actually survive.
    if grad_enabled and requires_grad:
        out.sum().backward()
        assert x.grad is not None, (
            "the graph was severed at this layer: a grad-requiring forward returned a "
            "tensor with no usable grad_fn. This is the fab325ae failure class.")


@pytest.mark.cuda
@_CUDA
def test_a_cpu_decode_shaped_call_takes_the_oracle():
    """Device dispatch, not degradation: there is no CPU lowering to decline."""
    layer = _layer("cpu")
    layer.eval()
    with torch.no_grad():
        cache = _prime(layer, "cpu")
        before = layer.execution_backend_counts
        _step(layer, cache, "cpu", L=1, grad_enabled=False)
    after = layer.execution_backend_counts
    assert after['oracle'] == before['oracle'] + 1
    assert after['decode'] == before['decode']


@pytest.mark.cuda
@_CUDA
def test_the_op_level_guard_raises_rather_than_redirecting():
    """Defense in depth for a DIRECT caller of the op.

    It raises; it does not silently redirect to the oracle. A silent redirect is the
    same class of failure (two implementations behind one call) that the layer's guard
    exists to prevent, pointed the other way.
    """
    from rola.ops.decode import decode_forward, freeze_decode_config
    from rola.routing.types import IndependentRouting, LevelConfig, SoftmaxActivation, Topology

    device = "cuda"
    widths, d_v = (8, 16), 64
    routing = IndependentRouting(tied=False, read_activation=SoftmaxActivation(),
                                 write_activation=SoftmaxActivation())
    topology = Topology(levels=tuple(LevelConfig(w, routing) for w in widths), d_v=d_v)
    config = freeze_decode_config(topology, "global", decay=False, BH=1, device=device,
                                  mma_precision="bf16")
    levels = tuple(torch.rand(1, 1, 1, w, device=device, requires_grad=True) for w in widths)
    g_write = torch.rand(1, 1, 1, device=device)
    v = torch.randn(1, 1, 1, d_v, device=device)
    state = torch.zeros(1, 1, config.N, config.cols, device=device)
    with pytest.raises(RuntimeError, match="forward-only"):
        decode_forward(v, levels, levels, None, g_write, config, state)


@pytest.mark.cuda
@_CUDA
def test_a_paged_layer_keeps_the_tiled_path():
    """Decode is unpaged only until §4.5's atom re-key lands.

    A paged layer must therefore keep the TILED path at ``T = 1`` — which is exact
    (paging is bitwise invisible) and keeps the ``max_physical_pages`` cap enforced.
    The alternative, letting a paged layer take an unpaged decode step, would silently
    drop the cap: the same shape of failure the oracle arm's paging refusal names.
    """
    device = "cuda"
    layer = build_layer(
        hidden_size=_HIDDEN, num_heads=_H, d_v=_DV, widths=_BRANCHES,
        normalization="global", expert=PlanOverrides(paging=True),
        layer_idx=0,
    ).to(device, torch.float32)
    layer.eval()
    with torch.no_grad():
        cache = _prime(layer, device)
        before = layer.execution_backend_counts
        _step(layer, cache, device, L=1, grad_enabled=False)
    after = layer.execution_backend_counts
    assert after['cuda'] == before['cuda'] + 1, "a paged layer must keep the tiled path"
    assert after['decode'] == before['decode']


# ---------------------------------------------------------------------------
# H9 (coverage-closure): the four CUDA rows of the table, swept over the decay
# axis the layer default never leaves. `normalization` is no longer a second axis
# here -- `"raw"` is a dead axis (`rola.routing.types.validate_normalization`
# accepts only `"global"`, P26) -- so this collapses from a (norm, decay) cross to
# decay alone. The dispatch predicate does not read decay, so a mis-routed cell
# here means the PREDICATE grew an accidental coupling -- and the decode/tiled
# kernels both branch on decay internally, so 'the row runs' is asserted through
# the counter exactly as the base table does. 4 rows x 1 non-default combo
# (decay="leaf_mass"; decay="none" is the base table's own default combo).
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@_CUDA
@pytest.mark.parametrize("decay", ["leaf_mass"])
@pytest.mark.parametrize("L,carried,expected", [
    (1, True, 'decode'),
    (1, False, 'cuda'),
    (2, True, 'cuda'),
    (16, True, 'cuda'),
])
def test_the_cuda_rows_dispatch_identically_across_decay(
        decay, L, carried, expected):
    device = "cuda"
    layer = _layer(device, decay=decay)
    layer.eval()
    if carried:
        cache = _prime(layer, device)
    else:
        cache = RoLAState()
    before = layer.execution_backend_counts
    _step(layer, cache, device, L=L, grad_enabled=False)
    after = layer.execution_backend_counts
    moved = {k for k in after if after[k] != before[k]}
    assert moved == {expected}, (
        f"(decay={decay}): expected the {expected!r} backend, "
        f"counters moved {moved} -- the dispatch predicate has grown a coupling "
        f"to an axis it must not read")


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
    (the feeder-flags lesson in ``_resolve_decode_config``'s docstring, same class).

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
    frozen = layer._decode_config
    assert frozen is not None, "the decode fork ran but froze no carrier"
    assert frozen.value.mma_precision == "bf16", (
        f"the layer's bf16 pin was dropped at the boundary: the frozen carrier says "
        f"{frozen.value.mma_precision!r}")


def test_freeze_decode_config_refuses_an_unknown_precision():
    """The carrier validates the pin at the freeze, not downstream of it."""
    from rola.ops.decode import freeze_decode_config
    from rola.routing.types import IndependentRouting, LevelConfig, SoftmaxActivation, Topology
    routing = IndependentRouting(tied=False, read_activation=SoftmaxActivation(),
                                 write_activation=SoftmaxActivation())
    topology = Topology(levels=tuple(LevelConfig(w, routing) for w in _BRANCHES), d_v=_DV)
    with pytest.raises(ValueError, match="mma_precision"):
        freeze_decode_config(topology, "global", decay=False, BH=1, device="cpu",
                             mma_precision="fp16")
