"""CONFORMANCE — the backward family (ratified design, 2026-08-05).

**The reference backward IS autograd through the decomposed oracle**, in fp64,
with the detachments as LITERAL SPEC: ``naive_rola`` writes ``.detach()`` at
the clock because spec §2 says ``stop_gradient`` there, so the reference VJP
is readable intent, not a derivation to trust. ``None`` is the correct answer
for the clock's input, and gradcheck — which cannot express detachment and
would disagree BY DESIGN — is demoted to one-time finite-difference insurance
on the glue's differentiable parts, env-gated below, never a kernel gate.

Three halves, plus the slot:

* **oracle-VJP over the family regimes** — seeded cotangents through the
  decomposed oracle across representative family cells, every differentiable
  input's gradient present and finite, the stored-clock input ``None``, and
  one SPEC-DERIVED structural zero (the anti regime).
* **producer VJP vs the authors' backward** — the production entmax solve's
  gradient against the pinned DeepSPIN package's own Jacobian (Blondel et al.
  2019), at family-shaped swept-gain logits. Reference code we did not write,
  which is the matching-the-naive rule's strongest available answer.
* **THE KERNEL SLOT** — the CUDA consumer is forward-only; its named refusal
  (`BackwardNotImplemented`) is gated in ``tests/unit/test_interface.py`` and
  the layer's grad-context routing in ``tests/layers/test_layer_dispatch.py``,
  so neither is duplicated here. When the native backward lands
  (``docs/open-work.md`` M-2), its VJP gates plug into THESE families —
  kernel-vs-oracle-VJP under the conditioned-tolerance/seed machinery
  ``tools/oracle_dual_run.py``'s P3 half already runs bit-identically over
  1328 quantities, plus a rerun floor for atomic-nondeterministic grads.
"""
from __future__ import annotations

import os

import pytest
import torch

from rola.ops.naive import naive_rola
from rola.routing.types import LeafMassDecay
from tests.conformance import generators as g
from tests.conformance.families import FAMILY_BY_NAME

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="family fixtures build on CUDA"),
]

#: Representative regime cells. Not the full list: the backward reference is a
#: pure function of the forward fixtures, so the sweep's value is regime
#: DIVERSITY (dense/sparse/anti/singleton/concentrated/ragged/cold/boundary),
#: not instantiation breadth — the instantiation axes do not exist on the
#: oracle, which is the point of a decomposed reference.
_BWD_FAMILIES = [
    "stratified/dense-full",
    "stratified/sparse-partial",
    "stratified/anti",
    "stratified/singleton",
    "stratified/concentrated",
    "stratified/ragged-raw-decay",
    "corner/cold-read-global-decay",
    "corner/chunk-boundary",
]


def _grad_leaves(case):
    """Every differentiable oracle input, as fresh grad-requiring fp64 leaves."""
    leaves = dict(
        v=case.v.detach().clone().requires_grad_(),
        read=tuple(t.detach().clone().requires_grad_() for t in case.read_levels),
        write=tuple(t.detach().clone().requires_grad_() for t in case.write_levels),
        g_write=case.g_write.detach().clone().requires_grad_(),
        g_read=(None if case.g_read is None
                else case.g_read.detach().clone().requires_grad_()),
        m0=(None if case.initial_state is None
            else case.initial_state.detach().clone().requires_grad_()),
        dials=(None if case.decay is None else tuple(
            d.detach().clone().requires_grad_() for d in case.decay.dials)),
    )
    return leaves


def _run_backward(case, leaves, *, stored=None, cy=True, cs=True, seed=1234):
    decay = None if leaves["dials"] is None else LeafMassDecay(dials=leaves["dials"])
    y, state = naive_rola(
        leaves["v"], leaves["read"], leaves["write"], leaves["g_read"],
        leaves["g_write"], case.topology, decay, case.norm,
        write_levels_stored=stored, initial_state=leaves["m0"],
        output_final_state=True)
    gen = torch.Generator(device=y.device).manual_seed(seed)
    loss = y.new_zeros(())
    if cy:
        loss = loss + (y * torch.randn(y.shape, generator=gen, device=y.device,
                                       dtype=y.dtype)).sum()
    if cs:
        loss = loss + (state * torch.randn(state.shape, generator=gen, device=y.device,
                                           dtype=y.dtype)).sum()
    loss.backward()
    return y, state


def _assert_live(name, tensor):
    assert tensor.grad is not None, f"{name} received no gradient"
    assert bool(torch.isfinite(tensor.grad).all()), f"{name}'s gradient is not finite"


@pytest.mark.parametrize("name", _BWD_FAMILIES)
def test_the_oracle_vjp_is_live_on_every_differentiable_input(name):
    family = FAMILY_BY_NAME[name]
    case = g.build_case(family)
    g.assert_nonvacuous(family, case)
    leaves = _grad_leaves(case)
    #: A DISTINCT stored tensor, only where a clock exists to read it: with no
    #: decay the oracle never touches `write_levels_stored` and a `None` grad
    #: there would be vacuous rather than the detachment speaking.
    stored = (tuple(t.detach().clone().requires_grad_() for t in case.write_levels)
              if case.decay is not None else None)
    _run_backward(case, leaves, stored=stored)

    _assert_live("v", leaves["v"])
    _assert_live("g_write", leaves["g_write"])
    for side in ("read", "write"):
        for level, tensor in enumerate(leaves[side]):
            _assert_live(f"{side}[{level}]", tensor)
    if leaves["g_read"] is not None:
        _assert_live("g_read", leaves["g_read"])
    if leaves["m0"] is not None:
        _assert_live("initial_state", leaves["m0"])
    if leaves["dials"] is not None:
        for level, dial in enumerate(leaves["dials"]):
            _assert_live(f"decay dial[{level}]", dial)
        # THE DETACHMENT AS LITERAL SPEC: the clock reads the STORED allocation
        # under stop_gradient (spec §2), so its input gets autograd's ordinary
        # answer for an unused-by-the-graph tensor — None, never a zero filled
        # in to look busy.
        for level, tensor in enumerate(stored):
            assert tensor.grad is None, (
                f"write_levels_stored[{level}] received a gradient; the clock's "
                "stop_gradient is NORMATIVE and the reference must express it")


def test_the_anti_regime_zero_is_structural():
    """Spec-derived, not tolerance-derived: in the anti cell every read lands on
    a never-written leaf, so ``y`` is a function of the carried state alone
    VALUE-wise — and the VJP draws the line exactly where calculus does. The
    deposit at a read leaf is ``g_write * w0[d0] * w1[d1] * v`` with
    ``w0[d0] = 0`` (the anti mask), so:

    * ``v`` and ``g_write`` gradients are EXACTLY ZERO (each is multiplied by
      the vanished product), as is every UNMASKED write level (multiplied by
      the masked one);
    * the MASKED level itself keeps a LIVE gradient off its support — the
      deposit is LINEAR in it, so its derivative does not vanish where its
      value does. A reference that zeroed gradients by support would be wrong
      here, which is precisely the sharpness a decomposed differentiable
      oracle buys.
    """
    family = FAMILY_BY_NAME["stratified/anti"]
    case = g.build_case(family)
    leaves = _grad_leaves(case)
    _run_backward(case, leaves, cy=True, cs=False)

    exact_zero = [("v", leaves["v"]), ("g_write", leaves["g_write"]),
                  *((f"write[{i}]", t) for i, t in enumerate(leaves["write"][1:], start=1))]
    for tag, tensor in exact_zero:
        grad = tensor.grad
        assert grad is None or bool((grad == 0).all()), (
            f"{tag} has a nonzero gradient from a y-only cotangent, but every "
            "term reaching y through it carries the anti mask's exact zero")
    masked = leaves["write"][0].grad
    assert masked is not None and bool((masked != 0).any()), (
        "the masked write level's gradient vanished; the deposit is linear in "
        "it, so a zero here means the VJP is (wrongly) gating gradients by "
        "the realized support")
    _assert_live("initial_state", leaves["m0"])
    assert bool((leaves["m0"].grad != 0).any()), (
        "the initial state's gradient is identically zero; the reads are not "
        "reaching the carried state and the structural zeros above are vacuous")


def test_the_producer_vjp_matches_the_authors_backward_across_the_gain_sweep():
    """The production entmax solve's gradient against the pinned DeepSPIN
    package's own backward, on swept-gain logits (the producer family's shape).
    Tolerances are IMPORTED from the production entmax gate, not restated."""
    from rola.routing.entmax import reference
    from rola.routing.entmax.production import production_factor
    from tests.unit.test_entmax_production import (
        _PRODUCTION_GRAD_ATOL,
        _PRODUCTION_GRAD_RTOL,
    )

    gen = torch.Generator(device="cuda").manual_seed(404)
    base = torch.randn(64, 16, device="cuda", dtype=torch.float64, generator=gen)
    cotangent = torch.randn(64, 16, device="cuda", dtype=torch.float64, generator=gen)
    supports = []
    for gain in (0.25, 1.0, 4.0, 16.0):
        z_ref = (gain * base).detach().clone().requires_grad_()
        p_ref = reference.entmax(z_ref, 1.5)
        (p_ref * cotangent).sum().backward()

        z = (gain * base).float().detach().clone().requires_grad_()
        p, _words = production_factor(z, activation="entmax", alpha=1.5)
        (p * cotangent.float()).sum().backward()

        assert torch.allclose(z.grad.double(), z_ref.grad,
                              rtol=_PRODUCTION_GRAD_RTOL, atol=_PRODUCTION_GRAD_ATOL), (
            f"gain {gain}: the production VJP is "
            f"{float((z.grad.double() - z_ref.grad).abs().max()):.3e} from the "
            "authors' backward")
        supports.append(float((p_ref > 0).double().mean()))
    g.assert_swept(supports)


@pytest.mark.skipif(
    os.environ.get("ROLA_GRADCHECK_INSURANCE") != "1",
    reason="one-time finite-difference insurance on the glue's differentiable "
           "parts (ratified design 2026-08-05: gradcheck cannot express the "
           "spec's detachment and is DEMOTED — never a kernel gate, never in "
           "the lane). Run with ROLA_GRADCHECK_INSURANCE=1 after touching the "
           "oracle's glue.")
def test_gradcheck_insurance_on_the_glue():
    """fp64 gradcheck through softmax -> naive_rola on a tiny cell. The softmax
    head keeps the perturbed inputs on the simplex the oracle validates; what
    is under check is OUR glue (leaf product, gate, ratio readout) composed
    with autograd, against finite differences."""
    widths, d_v, B, T, H = (3, 2), 2, 1, 3, 1
    topology = g._topology(widths)
    gen = torch.Generator().manual_seed(7)

    def leaf(*shape):
        return torch.randn(*shape, dtype=torch.float64, generator=gen,
                           requires_grad=True)

    w0_init, w1_init = leaf(B, T, H, 3), leaf(B, T, H, 2)
    #: THE CLOCK IS HELD CONSTANT, EXPLICITLY. Left to default, the stored
    #: allocation IS the live write, and finite differences would move the
    #: clock while the analytic path (spec §2's stop_gradient) does not —
    #: gradcheck then reports a mismatch that is the DETACHMENT, not a bug.
    #: This is the concrete demonstration of why gradcheck is demoted: it
    #: cannot express the spec, so the harness has to route around it.
    stored = (torch.softmax(w0_init, -1).detach(), torch.softmax(w1_init, -1).detach())

    def run(v, r0, r1, w0, w1, gw, dial0, dial1, m0):
        read = (torch.softmax(r0, -1), torch.softmax(r1, -1))
        write = (torch.softmax(w0, -1), torch.softmax(w1, -1))
        decay = LeafMassDecay(dials=(torch.sigmoid(dial0) * 0.9,
                                     torch.sigmoid(dial1) * 0.9))
        y, state = naive_rola(v, read, write, None, gw, topology, decay, "global",
                              write_levels_stored=stored,
                              initial_state=m0, output_final_state=True)
        return torch.cat([y.reshape(-1), state.reshape(-1)])

    m0 = torch.randn(B, H, 6, d_v + 1, dtype=torch.float64, generator=gen)
    m0[..., d_v] = m0[..., d_v].abs()
    inputs = (leaf(B, T, H, d_v), leaf(B, T, H, 3), leaf(B, T, H, 2),
              w0_init, w1_init,
              leaf(B, T, H).abs().detach().requires_grad_(),
              leaf(H, 3), leaf(H, 2), m0.requires_grad_())
    assert torch.autograd.gradcheck(run, inputs, eps=1e-6, atol=1e-8)
