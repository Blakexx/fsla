"""What the oracle returns and what it differentiates -- both stated as tests.

`naive_rola` used to add `sum(x) * 0` over every input to both of its outputs, so
that autograd would hand back a ZERO gradient rather than `None` for inputs the
result does not depend on (audit finding F1). Nothing in the mathematics has that
term. It is graph trickery, and it hides the one gradient fact the spec states
outright: `c[t,s] = stop_gradient(prod_l p_write_stored[...])`, the clock is
detached, so the stored levels receive NO gradient at all.

That matters more under the card's backward-conformance plan than it did while
the manufactured edge was harmless: the reference backward IS autograd through
this oracle, so a manufactured graph edge becomes a manufactured gradient in
every backward gate anchored to it.

The contract, in three claims:

1. **THE CLOCK IS NOT DIFFERENTIATED.** `write_levels_stored` gets `None`, with
   decay on and with decay off, and that is the answer -- not a zero.
2. **EVERYTHING THE RESULT DEPENDS ON IS DIFFERENTIATED**, and with a gradient
   that is actually nonzero. Deleting the manufactured edge must not have
   deleted a real one, and a test that only checked `is not None` would have
   passed against the old form no matter what.
3. **THE DETACHMENT IS LOAD-BEARING**, not incidental: the stored levels DO move
   the forward result under decay, so claim 1 is a statement about the gradient
   and not about an unused input.

And one claim about what it RETURNS: `y` used to be cast back to `v.dtype` while
the final state stayed fp64 (audit finding F2), so "the fp64 oracle" handed an
fp32 caller two claims about one run in two precisions -- and an accuracy gate
comparing against it would have been comparing against a reference that had
itself been rounded. Both outputs are fp64 now, always, and the cast belongs to
the caller who knows why they want one.

No device: the oracle is fp64 torch.
"""
from __future__ import annotations

import pytest
import torch

from rola.ops.naive import naive_rola
from rola.routing.types import (
    IndependentRouting,
    LeafMassDecay,
    SoftmaxActivation,
    Topology,
)

_ROUTING = IndependentRouting(
    width=1, tied=False, read=SoftmaxActivation(), write=SoftmaxActivation())

WIDTHS = (4, 4)
B, T, H, D_V = 2, 5, 2, 8


def _case(*, decay_on: bool, normalization: str = "global", seed: int = 7):
    """One small differentiable cell; every operand is a distinct leaf tensor."""
    g = torch.Generator().manual_seed(seed)
    topology = Topology(levels=tuple(_ROUTING.at(w) for w in WIDTHS))

    def simplex(width):
        x = torch.rand(B, T, H, width, generator=g, dtype=torch.float64) + 1e-3
        return (x / x.sum(dim=-1, keepdim=True)).requires_grad_(True)

    operands = {
        "v": torch.randn(B, T, H, D_V, generator=g, dtype=torch.float64).requires_grad_(True),
        "g_write": (torch.rand(B, T, H, generator=g, dtype=torch.float64) + 0.5).requires_grad_(True),
    }
    read = tuple(simplex(w) for w in WIDTHS)
    write = tuple(simplex(w) for w in WIDTHS)
    #: DISTINCT from `write`, and that is the whole point: aliased to the deposit,
    #: the clock's refusal to produce a gradient cannot be observed.
    stored = tuple(level.detach().clone().requires_grad_(True) for level in write)
    for index, (r, w, s) in enumerate(zip(read, write, stored)):
        operands[f"read{index}"], operands[f"write{index}"], operands[f"stored{index}"] = r, w, s
    g_read = None
    if normalization == "raw":
        g_read = (torch.rand(B, T, H, generator=g, dtype=torch.float64) + 0.5).requires_grad_(True)
        operands["g_read"] = g_read
    decay = None
    if decay_on:
        dials = tuple((torch.rand(H, w, generator=g, dtype=torch.float64) * 0.3 + 0.1
                       ).requires_grad_(True) for w in WIDTHS)
        decay = LeafMassDecay(dials=dials)
        for index, dial in enumerate(dials):
            operands[f"dial{index}"] = dial

    def run(stored_override=None):
        return naive_rola(
            operands["v"], read, write, g_read, operands["g_write"], topology, decay,
            normalization, write_levels_stored=stored if stored_override is None else stored_override,
            output_final_state=True)

    return operands, stored, run


def _grads(operands, y, state):
    keys = sorted(operands)
    grads = torch.autograd.grad(
        outputs=(y, state), inputs=[operands[k] for k in keys],
        grad_outputs=(torch.ones_like(y), torch.ones_like(state)), allow_unused=True)
    return dict(zip(keys, grads))


@pytest.mark.parametrize("decay_on", [False, True])
@pytest.mark.parametrize("normalization", ["global"])
def test_the_stored_clock_receives_no_gradient_not_a_zero_one(decay_on, normalization):
    """Claim 1. `None` is autograd's answer and the oracle states nothing else."""
    operands, stored, run = _case(decay_on=decay_on, normalization=normalization)
    y, state = run()
    grads = _grads(operands, y, state)
    for index in range(len(stored)):
        grad = grads[f"stored{index}"]
        assert grad is None, (
            f"write_levels_stored[{index}] received a gradient of "
            f"{None if grad is None else float(grad.abs().max())}; the spec detaches the "
            f"clock, so the answer is None -- a zero here is a manufactured graph edge "
            f"(audit F1)"
        )


@pytest.mark.parametrize("decay_on", [False, True])
@pytest.mark.parametrize("normalization", ["global"])
def test_every_operand_the_result_depends_on_still_gets_a_real_gradient(decay_on, normalization):
    """Claim 2, with the NONZERO half that `is not None` alone would not give."""
    operands, stored, run = _case(decay_on=decay_on, normalization=normalization)
    y, state = run()
    grads = _grads(operands, y, state)
    expected = [k for k in operands if not k.startswith("stored")]
    assert len(expected) >= 6, "the fixture stopped covering the operands it names"
    for key in expected:
        grad = grads[key]
        assert grad is not None, f"{key} lost its gradient"
        assert torch.isfinite(grad).all(), f"{key} has a non-finite gradient"
        assert float(grad.abs().max()) > 0, (
            f"{key}'s gradient is identically zero -- either the operand stopped "
            f"reaching the output or the manufactured edge is back")


def test_the_stored_clock_does_move_the_forward_so_claim_one_is_about_gradients():
    """Claim 3: the detachment is load-bearing, not a statement about an unused input."""
    operands, stored, run = _case(decay_on=True)
    y_base, _ = run()
    perturbed = tuple(level.detach().clone().flip(-1).requires_grad_(True) for level in stored)
    y_moved, _ = run(stored_override=perturbed)
    assert not torch.equal(y_base, y_moved), (
        "perturbing write_levels_stored left the forward unchanged, so the "
        "no-gradient claim above is vacuous for this cell")


def test_the_oracle_declares_the_none_answer_in_its_docstring():
    """The behaviour is a contract, so it is written down where a caller reads it."""
    assert "None" in naive_rola.__doc__ and "write_levels_stored" in naive_rola.__doc__



@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("normalization", ["global"])
def test_both_outputs_are_fp64_whatever_the_inputs_were(dtype, normalization):
    """Claim 4 (audit F2): the fp64 oracle returns fp64, and the two agree."""
    operands, _stored, run = _case(decay_on=True, normalization=normalization)
    lowered = {}
    for key, tensor in operands.items():
        lowered[key] = tensor.detach().to(dtype)
    topology = Topology(levels=tuple(_ROUTING.at(w) for w in WIDTHS))
    read = tuple(lowered[f"read{i}"] for i in range(len(WIDTHS)))
    write = tuple(lowered[f"write{i}"] for i in range(len(WIDTHS)))
    dials = tuple(lowered[f"dial{i}"] for i in range(len(WIDTHS)))
    y, state = naive_rola(
        lowered["v"], read, write,
        lowered.get("g_read"), lowered["g_write"], topology,
        LeafMassDecay(dials=dials), normalization,
        write_levels_stored=write, output_final_state=True)
    assert y.dtype is torch.float64, (
        f"y came back as {y.dtype} for {dtype} inputs; the oracle returns fp64")
    assert state.dtype is torch.float64, f"the final state came back as {state.dtype}"


def test_the_dtype_policy_is_stated_where_a_caller_reads_it():
    """A contract nobody can find is a convention. It is in the docstring."""
    assert "float64" in naive_rola.__doc__


