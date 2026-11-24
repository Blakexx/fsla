# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE GRADIENT REFERENCE'S OWN GATE (`tests/oracle/stored_vjp.py`): the closed forms it states agree with the fp64
oracle's autograd at unrounded values, level by level and across a routing's packed planes, and the rule that grades a
backward's gradient against it (`ENTMAX_GRADIENTS_FROM_STORED`) passes a bf16 result of the torch lowering and fails the
planted errors. No CUDA: the torch lowering and the oracle run on the host."""
from __future__ import annotations

import pytest
import torch

from fsla.routing.entmax.production import production_routing_levels
from fsla.routing.entmax.reference import entmax
from fsla.routing.reference import routing_factor_levels as oracle_factor_levels
from fsla.routing.types import EntmaxActivation, SoftmaxActivation
from tests.integration.test_production_levels import _routings, _stress_logits
from tests.oracle.fixtures import assert_planted_errors_fail, assert_slots_close
from tests.oracle.stored_vjp import activation_vjp, gradient_from_stored
from tests.oracle.tolerances import ENTMAX_GRADIENTS_FROM_STORED

SHAPES = ("tied", "mixed_per_level", "dense_only")


@pytest.mark.parametrize("activation", (SoftmaxActivation(), EntmaxActivation(1.5), EntmaxActivation(2.0)),
                         ids=("softmax", "entmax-1.5", "entmax-2.0"))
def test_a_closed_form_is_the_oracles_autograd_at_unrounded_values(activation):
    generator = torch.Generator().manual_seed(31)
    z = (3.0 * torch.randn(4, 9, 7, dtype=torch.float64, generator=generator)).requires_grad_()
    dp = torch.randn(4, 9, 7, dtype=torch.float64, generator=generator)
    p = torch.softmax(z, -1) if isinstance(activation, SoftmaxActivation) else entmax(z, activation.alpha, dim=-1)
    (autograd,) = torch.autograd.grad((p * dp).sum(), z)
    torch.testing.assert_close(activation_vjp(p.detach(), dp, activation), autograd)


KINDS = ("random", "near_boundary", "exact_tie", "large_magnitude", "collapsed_support")


@pytest.mark.parametrize("alpha", (1.5, 2.0))
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kind", KINDS)
def test_a_routings_gradient_from_stored_values_is_the_oracles_autograd_on_every_span(alpha, shape, kind):
    """Every level at UNROUNDED fp64 values on the fixtures the
    integration test draws."""
    routing = _routings(alpha)[shape]
    read = _stress_logits(routing, kind, device="cpu", seed=17).double().requires_grad_()
    write = _stress_logits(routing, kind, device="cpu", seed=18).double().requires_grad_()
    reads, writes = oracle_factor_levels(read, write, routing)
    generator = torch.Generator().manual_seed(3)
    cotangents = [torch.randn(level.shape, dtype=torch.float64, generator=generator) for level in (*reads, *writes)]
    autograd = torch.autograd.grad(sum((level * c).sum() for level, c in zip((*reads, *writes), cotangents)),
                                   (read, write))
    d_read, d_write, covered = gradient_from_stored(
        routing, [x.detach() for x in reads], [x.detach() for x in writes], cotangents[:routing.D],
        cotangents[routing.D:])
    assert len(covered) == routing.D
    for span in covered:
        torch.testing.assert_close(d_read[..., span], autograd[0][..., span], rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(d_write[..., span], autograd[1][..., span], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("alpha", (1.5, 2.0))
@pytest.mark.parametrize("shape", SHAPES)
def test_the_gradient_rule_passes_a_bf16_result_and_fails_planted_errors(alpha, shape):
    """The torch lowering's own gradient, rounded to bf16 as the kernel stores it, is the kernel's stand-in on the host."""
    routing = _routings(alpha)[shape]
    read = _stress_logits(routing, "random", device="cpu", seed=17).bfloat16().float().requires_grad_()
    write = _stress_logits(routing, "random", device="cpu", seed=18).bfloat16().float().requires_grad_()
    reads, writes, _ = production_routing_levels(read, write, routing)
    generator = torch.Generator().manual_seed(5)
    cotangents = [torch.randn(level.shape, generator=generator).bfloat16().float() for level in (*reads, *writes)]
    vjp = torch.autograd.grad(sum((level * c).sum() for level, c in zip((*reads, *writes), cotangents)), (read, write))
    d_read, d_write, covered = gradient_from_stored(routing, [r.detach() for r in reads], [w.detach() for w in writes],
                                                    cotangents[:routing.D], cotangents[routing.D:])
    actual = torch.cat([vjp[plane][..., span].bfloat16().double().flatten() for span in covered for plane in (0, 1)])
    want = torch.cat([ref[..., span].flatten() for span in covered for ref in (d_read, d_write)])
    assert_slots_close(actual, want, output=ENTMAX_GRADIENTS_FROM_STORED, what="the bf16 torch gradient")
    assert_planted_errors_fail(actual, want, output=ENTMAX_GRADIENTS_FROM_STORED, what="the bf16 torch gradient")
