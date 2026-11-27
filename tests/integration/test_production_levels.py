"""Gates for the dense per-level production routing solve (the producer path).

`fsla.routing.entmax.production.production_routing_levels` is
what `fsla/routing/producer.py` calls. It has two lowerings of one semantics -- a DEVICE
lowering at CUDA bf16 and the identical sorted closed form in torch on CPU/fp64 -- and
both are gated here against the ONE canonical ground truth this branch authorizes for
the routing primitive: the fp64 oracle in `routing/reference.py`
(`entmax/reference.py:entmax`), which no runtime path touches. The oracle is a closed
form at both alphas this file tests (`entmax15` at 1.5, `sparsemax` at 2.0 -- the third
party's own exact forms, inverted algebraically for `tau`) -- there is no bisection
anywhere in this file's ground truth.

**THE DEVICE LOWERING IS NOT ONE KERNEL**, and it is no longer any part Triton. The independent per-side solves and the softmax pair are hand-CUDA + CUB kernels (`csrc/fsla/src/entmax/factor.cu`),
ported off Triton for the closed-world AOT policy and for P65, which trains through
them. "The two lowerings" below means device against torch.

Support is checked for **exact** set equality, never by threshold: entmax support is a
discrete object (sparsity comes from entmax, not from a floor), so "the same zeros" is a hard
assertion and only the amplitudes carry a tolerance. Two scopes are deliberately
excluded from that hard assertion, and neither is a tolerance in disguise:

* **softmax sides.** A dense activation has no support; its "zeros" are fp underflow
  and differ between fp32 and fp64 by construction. Exact support is asserted for
  exactly the zero-capable sides (`Topology.can_zero`).
* **the exact-tie fixture**, which is not gated against the oracle at all. Where a
  logit sits exactly on the threshold, an oracle comparison is a comparison of TWO
  INDEPENDENT closed forms -- production's fp32 sorted solve and the oracle's fp64
  `entmax_tau` -- at an exact equality, each free to land on either side of the same
  strict `z_i > tau` boundary. They are not obligated to agree there, and measured
  directly they do not always (alpha=1.5, level 0 of a mixed-per-level shape disagrees). One
  extra support member at that boundary moves `den` in `s*(dp - num/den)`, so the
  GRADIENT effect of a boundary disagreement is O(1) even though the two sides'
  amplitudes are close -- so nothing about that fixture can be gated on the oracle
  without gating one boundary comparison against a second one that has no obligation
  to agree with it. It is instead gated on the two PRODUCTION lowerings against each
  other -- the SAME closed form computed by two different code paths (the fused
  CUDA/CUB kernels vs the pure-torch lowering), both handed the SAME fp32
  inputs -- which must therefore agree on support exactly
  (`test_the_two_lowerings_agree_on_exact_ties`); its gradient is held, as on every fixture, against the operator's
form at the kernel's own stored values (`ENTMAX_GRADIENTS_FROM_STORED`).
"""

from __future__ import annotations

import pytest
import torch
from conftest import fuse_gain_columns, layout_of, produce_routing, raw_side_gain

import fsla.ops.produce as production_path
from fsla.ops.produce import ProducerWeights, route_logits
from fsla.routing.entmax.production import production_routing_levels
from fsla.routing.producer import Routing
from fsla.routing.projection import RouterLayout
from fsla.routing.reference import routing_factor_levels as oracle_factor_levels
from fsla.routing.types import Topology, entmax, softmax, tied
from tests.oracle.fixtures import assert_planted_errors_fail, assert_slots_close
from tests.oracle.stored_vjp import gradient_from_stored
from tests.oracle.tolerances import (
    ENTMAX_GRADIENTS,
    ENTMAX_GRADIENTS_FROM_STORED,
    ENTMAX_STORED_VALUES,
    ENTMAX_VALUES,
    PROJECTION_LOGITS,
    STORED_LEVELS,
)

#: THE SEAM: a comparison that crosses the STORAGE boundary carries the bf16 operand
#: term and nothing else (`tests/oracle/tolerances.py`'s `STORED_LEVELS`: `BF16_RTOL`,
#: the budget every kernel-vs-oracle gate in the tree uses, as the clause's rtol, and
#: that relative bound applied to a simplex entry's own scale (<= 1) as its atol).
#: Measured worst here: 3.58e-3 relative, 1.91e-3 absolute, both at alpha=1.5. SUPPORT
#: carries no tolerance across this seam at all -- the solve writes exact zeros as
#: zeros, so the set is identical or the storage broke the self-masking theorem's
#: precondition.

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused production solve needs CUDA")


def _layout(widths, write, read, group=1) -> RouterLayout:
    return RouterLayout(Topology(widths=tuple(widths), write=tuple(write), read=tuple(read)), group)


#: the shapes every solve gate sweeps
SHAPES = ("tied", "mixed_per_level", "dense_only", "both_sparse", "grouped")


def _routings(alpha: float) -> dict[str, RouterLayout]:
    """Every structural level shape the producer can hand the solve."""
    return {
        # A second, untied level alongside the tied one, so the read columns are genuinely solved somewhere.
        "tied": _layout((4, 5), (entmax(alpha), softmax()), ((tied(), softmax()),)),
        "mixed_per_level": _layout((3, 5), (softmax(), entmax(alpha)), ((entmax(alpha), softmax()),)),
        "dense_only": _layout((4,), (softmax(),), ((softmax(),),)),
        # A level both of whose sides make exact zeros from TWO solves, each with its own support words
        "both_sparse": _layout((4, 5), (entmax(alpha), softmax()), ((entmax(alpha), softmax()),)),
        # THREE READ HEADS a write head, each by its own tuple, a level tied by one head at most: level 0 tied on head 0
        # (the write solve's second output), softmax on head 1 and its own entmax on head 2; level 1 entmax on head 0
        # (the read support words), softmax on head 1 and tied on head 2
        "grouped": _layout((4, 5), (entmax(alpha), softmax()),
                           ((tied(), entmax(alpha)), (softmax(), softmax()), (entmax(alpha), tied())), group=3),
    }


def _spans(layout: RouterLayout) -> list[slice]:
    """The column span of every solve: each write level's, then each read head's untied entries'."""
    spans = [layout.write_columns(level) for level in range(layout.D)]
    return spans + [span for g in range(layout.group) for level in range(layout.D)
                    if (span := layout.read_columns(g, level)) is not None]


def _stress_logits(layout: RouterLayout, kind: str, *, device, seed: int) -> torch.Tensor:
    """Stress fixtures for the threshold solve, in the projection's ``[S, T, packed_router_width]`` layout."""
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (3, 40, layout.packed_router_width)
    base = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    if kind == "random":
        return base

    if kind == "near_boundary":
        # Quantizing to halves clusters logits onto the candidate thresholds; the jitter
        # then separates them by ~1e-4, which is far inside any tolerance here but still
        # a value the closed-form oracle resolves exactly. This is the case where
        # production's greatest-valid-k rule and the oracle's closed-form tau are most
        # likely to disagree about where the support boundary falls.
        jitter = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
        return torch.round(base * 2.0) / 2.0 + 1.0e-4 * jitter
    if kind == "exact_tie":
        # Logits landing exactly ON a threshold. See the module docstring: support
        # equality is not asserted against the oracle here, because the oracle is the
        # side that cannot answer.
        return torch.round(base * 2.0) / 2.0
    if kind == "collapsed_support":
        # One dominant digit per row: the support collapses to a single element, the
        # smallest support entmax can produce (a genuinely empty support is
        # unreachable -- the argmax is always in it).
        collapsed = torch.full(shape, -20.0, device=device, dtype=torch.float32)
        # ONE dominant digit per SOLVE per row. In the packed layout the dominant index
        # must land inside the solve's own span (`_spans`) -- a single index
        # drawn over the whole packed width would leave every other level a constant row,
        # whose entmax is uniform, i.e. the opposite of the collapsed support this
        # fixture exists to produce.
        for span in _spans(layout):
            index = torch.randint(0, span.stop - span.start, shape[:-1], device=device, generator=generator)
            collapsed[..., span] = collapsed[..., span].scatter(-1, index.unsqueeze(-1), 20.0)
        return collapsed
    if kind == "large_magnitude":
        # 5x the router's natural logit scale: the solve's support collapses hard, so
        # the row-max shift is what keeps it finite. The scale is deliberately not pushed
        # further.
        return base * 5.0
    raise ValueError(kind)


def _assert_levels_match(actual, expected, *, gradient=False, stored=False, extra_atol=0.0):
    """Per slot under the output kind's clauses (`tests.oracle.fixtures.assert_slots_close`). ``extra_atol`` is a
    DERIVED term the caller states, never a loosened constant. A bf16 ``actual`` that is not a ``stored`` level is the
    CUDA solve's own stored form: it is held to the reference's value ROUNDED TO bf16, within one bf16 unit in the last place.
    A bf16 gradient is never held here: `_assert_gradients_from_stored` holds it."""
    if actual.dtype is torch.bfloat16 and not stored:
        assert not gradient, "a bf16 gradient is held by `_assert_gradients_from_stored`"
        output = ENTMAX_STORED_VALUES
        expected = expected.to(torch.bfloat16)
    else:
        output = STORED_LEVELS if stored else ENTMAX_GRADIENTS if gradient else ENTMAX_VALUES
    assert_slots_close(actual, expected, output=output, what=f"the levels ({output.name})", derived_atol=extra_atol)


def _assert_gradients_from_stored(layout, got, stored, cotangents, *, unrounded, label, capsys=None,
                                  teeth=False):
    """THE GRADIENTS INTO THE LOGITS, from the kernel's backward against the reference evaluated at the SAME stored values.

    ``got`` is the backward's gradient into the logits plane; ``stored`` the forward's own ``(reads, writes)``
    (bf16, what the backward reads), and
    ``cotangents`` the incoming gradients, ``(*reads, *writes)``. Every solve (`tests/oracle/stored_vjp.py`: softmax
    and entmax) is held per slot to the gradient the operator's fp64 form gives at those
    stored values (`ENTMAX_GRADIENTS_FROM_STORED`); with ``teeth`` the planted errors must fail on the kernel's own
    gradient, over every slot together (`assert_planted_errors_fail`). The error against ``unrounded``, the gradient at
    the unrounded values, is REPORTED over every span, never gated (nothing when ``unrounded`` is None).
    """
    D = layout.D
    reads, writes = ([level.detach() for level in side] for side in stored)
    reference, covered = gradient_from_stored(layout, reads, writes, cotangents[:D], cotangents[D:])
    for span in covered:
        assert_slots_close(got[..., span], reference[..., span], output=ENTMAX_GRADIENTS_FROM_STORED,
                           what=f"{label}: the logits' gradient over {span}")
    if teeth:
        assert_planted_errors_fail(
            torch.cat([got[..., span].double().flatten() for span in covered]),
            torch.cat([reference[..., span].flatten() for span in covered]),
            output=ENTMAX_GRADIENTS_FROM_STORED, what=f"{label}: the gradients the reference states")
    if unrounded is None:
        return
    worst = float((got.double() - unrounded.double()).abs().max())
    scale = float(unrounded.double().abs().max())
    line = (f"\n  UNROUNDED GRADIENT ERROR {label}: worst {worst:.3e} against the gradient at the unrounded values "
            f"(its largest slot {scale:.3e})")
    if capsys is None:
        print(line)
    else:
        with capsys.disabled():
            print(line)


def _assert_support_identical(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    assert torch.equal(actual != 0, expected != 0), f"{label}: support sets differ"


def _count_exact_ties(logits: torch.Tensor, alpha: float) -> int:
    """Count elements landing exactly on the sorted-solve threshold ``z == tau``.

    Independent reimplementation of `_sorted_solve`'s tau (production.py), not an
    import of it: the point is to positively confirm the "exact_tie" fixture actually
    produces the tied-digit case `test_the_two_lowerings_agree_on_exact_ties` claims to
    exercise, so it must not share a bug with the code it is checking.
    """
    z = logits - logits.amax(dim=-1, keepdim=True)
    width = z.shape[-1]
    sorted_z, _ = torch.sort(z, dim=-1, descending=True)
    prefix = sorted_z.cumsum(dim=-1)
    k = torch.arange(1, width + 1, device=z.device, dtype=z.dtype)
    if alpha == 2.0:
        tau_candidates = (prefix - 1.0) / k
        candidate = sorted_z > tau_candidates
    else:
        prefix_sq = (sorted_z * sorted_z).cumsum(dim=-1)
        discriminant = prefix * prefix - k * (prefix_sq - 4.0)
        tau_candidates = (prefix - discriminant.clamp(min=0.0).sqrt()) / k
        candidate = (discriminant >= 0.0) & (sorted_z > tau_candidates)
    columns = torch.arange(width, device=z.device)
    chosen = torch.where(candidate, columns, torch.full_like(columns, -1)).amax(dim=-1, keepdim=True)
    tau = torch.gather(tau_candidates, -1, chosen.clamp(min=0))
    tau = torch.where(chosen >= 0, tau, torch.zeros_like(tau))
    return int((z == tau).sum().item())


def _check_against_oracle(layout: RouterLayout, logits, *, label: str, exact_support: bool = True, capsys=None):
    cuda = logits.is_cuda
    if cuda:
        #: the CUDA solve takes bf16 logits; the oracle is handed the SAME values, widened (exact)
        logits = logits.bfloat16()
    production = logits.detach().clone().requires_grad_()
    oracle = logits.detach().clone().to(torch.float64).requires_grad_()

    reads, writes, _support = production_routing_levels(production, layout)
    oracle_reads, oracle_writes = oracle_factor_levels(oracle, layout)

    assert len(reads) == len(writes) == layout.D
    topology, S, T = layout.topology, logits.shape[0], logits.shape[1]
    for level in range(layout.D):
        assert reads[level].shape == (S * layout.group, T, layout.widths[level])
        assert writes[level].shape == (S, T, layout.widths[level])
        if exact_support and topology.can_zero("read", level):
            _assert_support_identical(
                reads[level], oracle_reads[level], f"{label} read level {level}")
        if exact_support and topology.can_zero("write", level):
            _assert_support_identical(
                writes[level], oracle_writes[level], f"{label} write level {level}")
        _assert_levels_match(reads[level], oracle_reads[level])
        _assert_levels_match(writes[level], oracle_writes[level])

    generator = torch.Generator(device=logits.device).manual_seed(9109)
    cotangents = [
        torch.randn(level.shape, device=level.device, dtype=torch.float32, generator=generator)
        for level in (*reads, *writes)
    ]
    if cuda:
        #: the cotangents a bf16 level takes are bf16: both sides are handed the same rounded values
        cotangents = [cotangent.bfloat16().float() for cotangent in cotangents]
    loss = sum(
        (level * cotangent).sum()
        for level, cotangent in zip((*reads, *writes), cotangents))
    oracle_loss = sum(
        (level * cotangent.to(torch.float64)).sum()
        for level, cotangent in zip((*oracle_reads, *oracle_writes), cotangents))
    (production_vjp,) = torch.autograd.grad(loss, production)
    (oracle_vjp,) = torch.autograd.grad(oracle_loss, oracle)
    if cuda:
        _assert_gradients_from_stored(layout, production_vjp, (reads, writes),
                                      cotangents, unrounded=oracle_vjp,
                                      label=label, capsys=capsys)
        return
    _assert_levels_match(production_vjp, oracle_vjp, gradient=True)


def _tied(alpha):
    """One tied entmax level: one solve, both sides."""
    return _layout((6,), (entmax(alpha),), ((tied(),),))


@pytest.mark.parametrize("alpha", [1.5, 2.0])
def test_tied_routing_read_factor_is_the_write_factor_on_both_lowerings(alpha):
    """A tied read entry's read factor IS its write factor -- the same tensor, not merely
    numerically equal -- on the fp64 oracle lowering and on the torch production lowering at one read head
    (`_ProductionLevelsFn` always allocates read/write as distinct autograd outputs, even for a tied level, so
    identity does not survive there)."""
    generator = torch.Generator().manual_seed(4242)
    logits = torch.randn(2, 10, 6, dtype=torch.float64, generator=generator)

    oracle_reads, oracle_writes = oracle_factor_levels(logits, _tied(alpha))
    assert oracle_reads[0] is oracle_writes[0]  # the read factor IS the write factor

    prod_reads, prod_writes, _support = production_routing_levels(logits.to(torch.float32), _tied(alpha))
    assert prod_reads[0] is prod_writes[0]


@_CUDA
@pytest.mark.parametrize("alpha", [1.5, 2.0])
def test_cuda_tied_routing_read_factor_is_the_write_factor(alpha):
    """The CUDA-kernel-dispatch path (`_ProductionLevelsFn`, GPU logits) takes a
    standalone tied level through one kernel launch, one solve. Mirrors
    `test_tied_routing_read_factor_is_the_write_factor_on_both_lowerings` (CPU/fp64)
    but forces the CUDA lowering by handing it CUDA bf16 logits."""
    generator = torch.Generator(device="cuda").manual_seed(4242)
    logits = torch.randn(2, 10, 6, device="cuda", dtype=torch.float32, generator=generator).bfloat16()

    tied_reads, tied_writes, _support = production_routing_levels(logits, _tied(alpha))
    torch.testing.assert_close(tied_reads[0], tied_writes[0], rtol=0, atol=0)

    oracle_reads, oracle_writes = oracle_factor_levels(logits.to(torch.float64), _tied(alpha))
    assert oracle_reads[0] is oracle_writes[0]  # the read factor IS the write factor
    _assert_levels_match(tied_reads[0], oracle_reads[0])
    _assert_levels_match(tied_writes[0], oracle_writes[0])


@pytest.mark.parametrize("alpha", [1.5, 2.0])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "kind", ["random", "near_boundary", "collapsed_support", "large_magnitude"])
def test_torch_lowering_matches_the_fp64_oracle(alpha, shape, kind):
    """CPU/fp64 lowering: exact same support, amplitudes and VJP as the closed-form oracle."""
    layout = _routings(alpha)[shape]
    _check_against_oracle(layout, _stress_logits(layout, kind, device="cpu", seed=17), label=f"cpu/{shape}/{kind}/{alpha}")


@_CUDA
@pytest.mark.parametrize("alpha", [1.5, 2.0])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "kind", ["random", "near_boundary", "collapsed_support", "large_magnitude"])
def test_cuda_lowering_matches_the_fp64_oracle(alpha, shape, kind, capsys):
    """CUDA bf16 lowering: exact same support, amplitudes and VJP as the closed-form oracle."""
    layout = _routings(alpha)[shape]
    _check_against_oracle(layout, _stress_logits(layout, kind, device="cuda", seed=17),
                          label=f"cuda/{shape}/{kind}/{alpha}", capsys=capsys)


@_CUDA
@pytest.mark.parametrize("alpha", [1.5, 2.0])
@pytest.mark.parametrize("shape", ["tied", "mixed_per_level", "both_sparse", "grouped"])
def test_the_two_lowerings_agree_on_exact_ties(alpha, shape, capsys):
    """Exact ties, where the oracle cannot arbitrate: the closed forms must still agree.

    Support is asserted as an identical set (a closed form has no convergence slack to
    hide behind) and the amplitudes at the production tolerances; the VJP is held to the
    operator's fp64 form at the kernel's own stored values, like every other fixture.
    """
    layout = _routings(alpha)[shape]
    logits = _stress_logits(layout, "exact_tie", device="cuda", seed=23).bfloat16().requires_grad_()

    # Soundness gate on the fixture itself: this test is the SOLE gate for tie
    # semantics (explicitly excluded from the oracle gate, see module docstring), so it
    # must not go vacuous under a seed/width change without saying so loudly.
    ties = sum(_count_exact_ties(logits.detach()[..., span], alpha) for span in _spans(layout))
    assert ties > 0, (
        f"exact_tie fixture for alpha={alpha} shape={shape!r} contains no exact ties -- "
        "the tie-semantics gate below would be exercising nothing"
    )

    cpu_logits = logits.detach().float().cpu().requires_grad_()
    cuda_reads, cuda_writes, _support = production_routing_levels(logits, layout)
    cpu_reads, cpu_writes, _support = production_routing_levels(cpu_logits, layout)
    for level in range(layout.D):
        if layout.topology.can_zero("read", level):
            _assert_support_identical(cuda_reads[level].cpu(), cpu_reads[level], f"read {level}")
        if layout.topology.can_zero("write", level):
            _assert_support_identical(cuda_writes[level].cpu(), cpu_writes[level], f"write {level}")
        _assert_levels_match(cuda_reads[level].cpu(), cpu_reads[level])
        _assert_levels_match(cuda_writes[level].cpu(), cpu_writes[level])

    generator = torch.Generator().manual_seed(5501)
    cotangents = [
        torch.randn(level.shape, dtype=torch.float32, generator=generator).bfloat16().float()
        for level in (*cpu_reads, *cpu_writes)
    ]
    (cuda_vjp,) = torch.autograd.grad(
        sum((level * cotangent.cuda()).sum()
            for level, cotangent in zip((*cuda_reads, *cuda_writes), cotangents)),
        logits)
    (cpu_vjp,) = torch.autograd.grad(
        sum((level * cotangent).sum()
            for level, cotangent in zip((*cpu_reads, *cpu_writes), cotangents)),
        cpu_logits)
    _assert_gradients_from_stored(layout, cuda_vjp, (cuda_reads, cuda_writes),
                                  [c.cuda() for c in cotangents], unrounded=cpu_vjp.cuda(),
                                  label=f"ties {shape}/{alpha}", capsys=capsys)


@_CUDA
@pytest.mark.parametrize("alpha", [1.5, 2.0])
@pytest.mark.parametrize("shape", SHAPES)
def test_the_gradient_rule_sees_planted_errors_on_the_kernels_own_gradient(alpha, shape, capsys):
    """THE TEETH of `ENTMAX_GRADIENTS_FROM_STORED`: the kernel's own gradient passes the rule, and the median slot wiped, the
    smallest allowances exceeded and each clause moved past its allowance fail it (`assert_planted_errors_fail`), on the
    random draw, over the solves the reference states."""
    layout = _routings(alpha)[shape]
    logits = _stress_logits(layout, "random", device="cuda", seed=17).bfloat16().requires_grad_()
    reads, writes, _support = production_routing_levels(logits, layout)
    generator = torch.Generator().manual_seed(8801)
    cotangents = [torch.randn(level.shape, dtype=torch.float32, generator=generator).bfloat16().float().cuda()
                  for level in (*reads, *writes)]
    (vjp,) = torch.autograd.grad(
        sum((level * cotangent).sum() for level, cotangent in zip((*reads, *writes), cotangents)), logits)
    _assert_gradients_from_stored(layout, vjp, (reads, writes), cotangents, unrounded=None,
                                  label=f"teeth {shape}/{alpha}", capsys=capsys, teeth=True)


def _forbid_every_reference_entry_point(monkeypatch) -> None:
    """Patch all THREE of the reference module's public entry points
    (``__all__ = ["entmax", "entmax_forward", "entmax_tau"]``), not just the one this
    solve happens to call today. Patching only ``entmax_tau`` proves the solve does not
    reach that one function; it says nothing about ``entmax`` or ``entmax_forward``, and
    a caller that reached either of those would have passed the narrower guard."""
    import fsla.routing.entmax.reference as entmax_reference

    def _forbidden(*args, **kwargs):
        raise AssertionError("production routing called the fp64 reference")

    for name in entmax_reference.__all__:
        monkeypatch.setattr(entmax_reference, name, _forbidden)


@_CUDA
def test_the_solve_never_reaches_the_reference_cuda(monkeypatch):
    """No runtime path may fall back to the oracle -- the port's whole point (CUDA half)."""
    _forbid_every_reference_entry_point(monkeypatch)
    layout = _routings(1.5)["mixed_per_level"]
    logits = _stress_logits(layout, "random", device="cuda", seed=31).bfloat16().requires_grad_()
    reads, writes, _support = production_routing_levels(logits, layout)
    sum(level.sum() for level in (*reads, *writes)).backward()
    assert logits.grad is not None


def test_the_solve_never_reaches_the_reference_cpu(monkeypatch):
    """No runtime path may fall back to the oracle -- the port's whole point (CPU half).

    Split from the CUDA half (adversarial review NOTE): the torch lowering needs no GPU,
    so this must run on a CPU-only runner instead of being skipped along with the CUDA
    half under one `@_CUDA`-marked test.
    """
    _forbid_every_reference_entry_point(monkeypatch)
    layout = _routings(1.5)["mixed_per_level"]
    logits = _stress_logits(layout, "random", device="cpu", seed=31).requires_grad_()

    cpu_reads, cpu_writes, _support = production_routing_levels(logits, layout)
    sum(level.sum() for level in (*cpu_reads, *cpu_writes)).backward()
    assert logits.grad is not None


# ---------------------------------------------------------------------------
# End-to-end: the V3 producer on the production path against the V3 producer on
# the fp64 oracle path. Same producer, same conditioning arithmetic, only the
# routing solve swapped -- so any difference is the port's.
# ---------------------------------------------------------------------------


def _producer_topology(alpha: float) -> Topology:
    return Topology(widths=(4, 3), write=(entmax(alpha), softmax()), read=((tied(), softmax()),))


#: the fixtures' value width; the topology carries none, so these gates state theirs -- one the producer kernel's value
#: job takes (32, 64, 128 or 256 channels a head), since these gates run it.
_D_V = 32


def _producer_params(topology: Topology, *, dtype, device, sides, B=2, L=48, H=2, dm=16,
                     seed=404):
    layout = layout_of(topology)
    generator = torch.Generator(device=device).manual_seed(seed)

    def _randn(*shape):
        return torch.randn(shape, device=device, dtype=torch.float32, generator=generator).to(dtype)

    return {
        "x": _randn(B, L, dm),
        "route_W": _randn(H, dm, layout.packed_router_width),
        "gain_W": _randn(H, sides, dm),
        "gain_bias": torch.zeros(H, sides, device=device, dtype=dtype),
        "v": _randn(B, L, H, _D_V),
    }


def _run_producer(params, topology):
    return produce_routing(
        params["x"], params["v"], params["route_W"], None, params["gain_W"],
        params["gain_bias"], topology)


@_CUDA
def test_the_projection_lands_inside_its_bf16_output_envelope():
    """THE PROJECTION half of the seam split, gated numerically against fp64, PER LOGIT.

    cuBLAS's mainloop is opaque, so this does not mirror its arithmetic -- it bounds the
    DECLARED class, which is bf16 operands / fp32 accumulate / bf16 OUT. Two terms of the
    same order, and the fp32 accumulation (`2^-24` per step) is not one of them. On each
    logit slot:

        dz <= 2^-8 * (2 * sum_k |x_k| |W_k|  +  |z|)    (the operands before cancellation, the output)

    The operand sum is the same projection run in fp64 on the parameters' magnitudes, so it
    lands in the logits' own layout (`tests.oracle.fixtures.assert_slots_close`).
    `_logit_floor` is this bound's max over the slots, the `dz` the downstream amplitude
    envelope is built from. MEASURED 2026-09-14: the worst slot sits at 0.40 of its
    allowance. The planted errors (a wiped logit, the smallest allowances exceeded) must
    fail on the production logits themselves.
    """
    topology = _producer_topology(1.5)
    params = _producer_params(topology, dtype=torch.float32, device="cuda", sides=1)
    logits = _production_logits(topology, params)
    assert logits.dtype is torch.bfloat16

    oracle_params = {name: tensor.to(torch.float64) for name, tensor in params.items()}
    logits64 = _production_logits(topology, oracle_params)
    assert logits64.dtype is torch.float64

    magnitudes = {name: tensor.to(torch.float64).abs() for name, tensor in params.items()}
    terms = _production_logits(topology, magnitudes)
    assert_planted_errors_fail(_flat64(logits), logits64, output=PROJECTION_LOGITS, envelope=2.0 * terms + logits64.abs(),
                               what="the routing logits")


@_CUDA
@pytest.mark.parametrize("alpha", [1.5, 2.0])
def test_the_solve_seam_is_exact_on_the_production_logits(alpha):
    """THE SOLVE GATE, at the seam the oracle-precision doctrine puts it at.

    The projection is a bf16-operand tensor-core GEMM whose arithmetic no reference can
    mirror, so it is gated NUMERICALLY elsewhere and the solve is gated EXACTLY here:
    the reference solve is fed the PRODUCTION LOGITS -- the very tensor the kernel read --
    and the support sets must agree BITWISE. bf16 -> fp32 and bf16 -> fp64 are both exact
    widenings, so the two solves see the same real numbers and nothing but solve
    precision can separate them; entmax support is a discrete object, so "the same
    numbers in" has to mean "the same set out".
    """
    topology = _producer_topology(alpha)
    layout = layout_of(topology)
    params = _producer_params(topology, dtype=torch.float32, device="cuda", sides=1)

    logits = _production_logits(topology, params)
    assert logits.dtype is torch.bfloat16, (
        "the shipped projection emits bf16 logits; this gate is meaningless against any "
        "other dtype")

    kernel_read, kernel_write, _support = production_routing_levels(logits, layout)
    reference_read, reference_write = oracle_factor_levels(_flat64(logits), layout)

    for level in range(topology.D):
        if topology.can_zero("read", level):
            _assert_support_identical(
                kernel_read[level], reference_read[level], f"read {level}")
        if topology.can_zero("write", level):
            _assert_support_identical(
                kernel_write[level], reference_write[level], f"write {level}")
        _assert_levels_match(kernel_read[level], reference_read[level], stored=True)
        _assert_levels_match(kernel_write[level], reference_write[level], stored=True)


def _logit_floor(params, logits64) -> float:
    """The declared class's bound on the logit error: operand term + output term."""
    x = params["x"].double().abs()
    weight = fuse_gain_columns(params["route_W"], params["gain_W"]).double().abs()
    operands = float(torch.einsum("bld,hdc->blhc", x, weight).max())
    scale = float(logits64.abs().max())
    return 2.0 * 2.0 ** -8 * operands + 2.0 ** -8 * scale


def _production_logits(topology, params):
    """The routing logits the shipped producer computes for these parameters: its projection step
    (`route_logits`), under the caller's parameter tensors -- reconstructing it is how a gate ends up measuring a
    second implementation."""
    routing = Routing(topology=topology.widths, write=topology.write, read=topology.read)
    weights = ProducerWeights(route_W=fuse_gain_columns(params["route_W"], params["gain_W"]), route_bias=None,
                              gain_bias=params["gain_bias"], v_W=params["x"].new_zeros(0))
    logits, _ = route_logits(params["x"], routing, weights)
    return logits


def _flat64(logits: torch.Tensor) -> torch.Tensor:
    """The token-major plane the kernel reads, as the flat fp64 plane a reference takes.

    Both steps are EXACT: a permute moves no bits, and bf16 -> fp64 is a widening.
    """
    return logits.permute(0, 2, 1, 3).reshape(
        logits.shape[0] * logits.shape[2], logits.shape[1], logits.shape[3]).double()


@_CUDA
@pytest.mark.parametrize("alpha", [1.5, 2.0])
@pytest.mark.requires_kernel("produce")
def test_producer_production_path_matches_its_oracle_path(monkeypatch, alpha, capsys):
    """`produce_routing` end to end: the shipped path against a wholly fp64 one.

    THIS IS PURPOSE-2 ACCOUNTING, and the doctrine says so. The two arms differ in the
    PROJECTION's operand class (bf16 operands on the tensor cores against fp64), so a
    logit near a threshold can land on either side of it and the support sets are not
    the same object. The flip count is therefore REPORTED, not asserted -- the exact
    claim about the solve lives in
    `test_the_solve_seam_is_exact_on_the_production_logits`, where the reference is fed
    the production logits.

    What IS asserted is the amplitudes at the stored tolerance, and that assertion
    bounds the flips it does not count: a member crossing the threshold enters with an
    amplitude of ~0 (entmax's `((z - tau)/2)^2` is continuous at the boundary), so a
    flip large enough to matter is a flip large enough to fail this.
    """
    topology = _producer_topology(alpha)
    layout = layout_of(topology)

    production_params = _producer_params(
        topology, dtype=torch.float32, device="cuda", sides=1)
    oracle_params = {name: tensor.to(torch.float64) for name, tensor in production_params.items()}

    logits = _production_logits(topology, production_params)
    production = _run_producer(production_params, topology)
    #: The solve is patched WHERE THE PRODUCER RESOLVES IT -- `fsla.ops.produce`
    #: binds the name at import -- so the oracle arm really runs the reference solve.
    monkeypatch.setattr(
        production_path, "production_routing_levels", lambda *args, **kwargs: (*oracle_factor_levels(*args, **kwargs), None))
    oracle = _run_producer(oracle_params, topology)

    #: THE ENVELOPE RE-DERIVES with the projection's operand class, it is not widened.
    #: The two arms are handed logits differing by the projection's declared class
    #: (`_logit_floor`); `tau` is a mean of supported logits so it moves by at most
    #: the same, and both entmax specializations have `|dp/dz| <= 1` on support
    #: (`d/dz ((z-tau)/2)^2 = sqrt(p) <= 1` at alpha 1.5, `= 1` at alpha 2). So the
    #: amplitude envelope is `2*dz` on top of the stored form's own tolerance --
    #: ON ROWS WHOSE SUPPORT SET DID NOT MOVE.
    #:
    #: A row whose support DID move is excluded from the envelope and counted instead:
    #: the bound on such a row is a bound on the FLIPPED MASS, which is what the
    #: accounting below reports.
    envelope = 2.0 * _logit_floor(production_params, logits.double())
    #: WRITE LEVEL 0 CARRIES THE GAIN (docs/CONSTITUTION.md#r23), so both arms' level 0 is read in units of the
    #: oracle's own gain `g` (the library's pieces, `raw_side_gain`): the stored tolerance then holds on the
    #: amplitudes' scale, and the gain's own envelope joins theirs -- `softplus` is 1-Lipschitz and the gain logit
    #: a column of the SAME projection, so it is off by `dz`, `dz / g` in those units, on an amplitude <= 1.
    gain = raw_side_gain(oracle_params["x"], oracle_params["route_W"], oracle_params["gain_W"],
                         oracle_params["gain_bias"], layout)[..., None].float()

    flips = elements = 0
    flipped_mass = 0.0
    for level in range(topology.D):
        level_0_write = (production.write_levels[level] / gain, oracle.write_levels[level] / gain,
                         topology.can_zero("write", level), envelope + envelope / (2.0 * gain))
        for got, want, zeroable, extra in (
            (production.read_levels[level], oracle.read_levels[level],
             topology.can_zero("read", level), torch.full_like(gain, envelope)),
            level_0_write if level == 0 else (production.write_levels[level], oracle.write_levels[level],
                                              topology.can_zero("write", level), torch.full_like(gain, envelope)),
        ):
            want, extra = want.to(got.dtype), extra.expand_as(got)
            if not zeroable:
                _assert_levels_match(got, want, stored=True, extra_atol=extra)
                continue
            moved = (got != 0) != (want != 0)
            flips += int(moved.sum())
            elements += got.numel()
            flipped_mass = max(flipped_mass, float((got * moved).abs().max()),
                               float((want * moved).abs().max()))
            settled = ~moved.any(dim=-1, keepdim=True).expand_as(got)
            _assert_levels_match(got[settled], want[settled], stored=True,
                                 extra_atol=extra[settled])

    with capsys.disabled():
        print(f"\n  FLIP ACCOUNTING alpha={alpha}: {flips} support flips in {elements} "
              f"zero-capable elements ({flips / max(elements, 1):.3e}), greatest flipped "
              f"amplitude {flipped_mass:.4f}, between the shipped bf16-operand projection "
              f"and a wholly fp64 one")


# ---------------------------------------------------------------------------
# V9 (compliance-audit) -- gate gaps named by adversarial review, closed here.
# ---------------------------------------------------------------------------


@_CUDA
@pytest.mark.parametrize("alpha", [1.5, 2.0])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "kind", ["random", "near_boundary", "collapsed_support", "large_magnitude"])
def test_the_two_lowerings_agree_directly(alpha, shape, kind, capsys):
    """V9 item 2: torch vs the CUDA kernels, DIRECTLY, on the ordinary fixtures.

    Both lowerings are separately gated against the fp64 oracle above, so
    their mutual drift was only TRANSITIVELY bounded, at twice the oracle tolerance
    (producer-port-review Claim 2). Two lowerings can each sit at the edge of that
    envelope on opposite sides and still pass, which is exactly the drift a port
    needs to be gated on. This asserts the pair directly, from the SAME inputs (bf16 values, widened for torch),
    at the unmodified production tolerances.

    Support is an EXACT set assertion: both are closed forms, so neither has
    convergence slack to hide behind (sparsity comes from entmax, not from a floor -- entmax
    support is a discrete object, not a thresholded one). The exact-TIE fixture is
    excluded because it has its own gate, ``test_the_two_lowerings_agree_on_exact_ties``.
    """
    layout = _routings(alpha)[shape]
    cuda_logits = _stress_logits(layout, kind, device="cuda", seed=17).bfloat16().requires_grad_()
    cpu_logits = cuda_logits.detach().float().cpu().requires_grad_()
    assert torch.equal(cuda_logits.detach().float().cpu(), cpu_logits.detach()), (
        "the two lowerings must be handed bit-identical inputs, or this measures the "
        "fixture rather than the port")

    cuda_reads, cuda_writes, _support = production_routing_levels(cuda_logits, layout)
    cpu_reads, cpu_writes, _support = production_routing_levels(cpu_logits, layout)
    for level in range(layout.D):
        if layout.topology.can_zero("read", level):
            _assert_support_identical(
                cuda_reads[level].cpu(), cpu_reads[level], f"{shape}/{kind} read {level}")
        if layout.topology.can_zero("write", level):
            _assert_support_identical(
                cuda_writes[level].cpu(), cpu_writes[level], f"{shape}/{kind} write {level}")
        _assert_levels_match(cuda_reads[level].cpu(), cpu_reads[level])
        _assert_levels_match(cuda_writes[level].cpu(), cpu_writes[level])

    generator = torch.Generator().manual_seed(6607)
    cotangents = [
        torch.randn(level.shape, dtype=torch.float32, generator=generator).bfloat16().float()
        for level in (*cpu_reads, *cpu_writes)
    ]
    (cuda_vjp,) = torch.autograd.grad(
        sum((level * cotangent.cuda()).sum()
            for level, cotangent in zip((*cuda_reads, *cuda_writes), cotangents)),
        cuda_logits)
    (cpu_vjp,) = torch.autograd.grad(
        sum((level * cotangent).sum()
            for level, cotangent in zip((*cpu_reads, *cpu_writes), cotangents)),
        cpu_logits)
    _assert_gradients_from_stored(layout, cuda_vjp, (cuda_reads, cuda_writes),
                                  [c.cuda() for c in cotangents], unrounded=cpu_vjp.cuda(),
                                  label=f"{shape}/{kind}/{alpha}", capsys=capsys)


def test_production_imports_nothing_from_the_reference_package():
    """The STRONGER, STATIC claim, checked directly rather than inferred from a
    monkeypatch. `fsla/routing/entmax/reference.py` and this module's own docstring say production
    "imports nothing" from the reference package -- a claim about the IMPORT GRAPH,
    stronger than and independent of `test_the_solve_never_reaches_the_reference_*`
    above (those prove the three entry points are never CALLED at runtime; a module
    can import a name and still never call it). This is the trivially checkable form
    the adversarial review found asserted nowhere: read `production.py` as an AST and
    confirm no import statement names `fsla.routing.entmax.reference` or
    `fsla.routing.reference` (the oracle-facing module of the same shape), by module
    or by symbol.
    """
    import ast
    import inspect

    import fsla.routing.entmax.production as production_module

    source = inspect.getsource(production_module)
    tree = ast.parse(source, filename=production_module.__file__)
    forbidden_modules = {"fsla.routing.entmax.reference", "fsla.routing.reference"}
    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in forbidden_modules:
                    offending.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module in forbidden_modules:
            offending.append(node.module)
    assert not offending, (
        f"production.py imports from the reference package: {offending} -- the "
        "producer's own contract (module docstring, fsla/routing/entmax/reference.py) is that it "
        "imports nothing from it")


def test_the_package_imports_no_triton_and_declares_none():
    """P73 the RECEIPT, as a static fact about the tree rather than a claim in a doc.

    `docs/build.md` says the dependencies are "`torch` and `einops`. Not
    `transformers`, not `fla`, not `triton` -- the kernels are CUDA." That sentence
    was FALSE for every level whose two sides were two solves, from the module that owned
    the four Triton kernels this stage ported to `csrc/fsla/src/entmax/factor.cu`.

    Checked two ways, because they can fail independently: the IMPORT GRAPH (an AST
    walk over every module under `fsla/`, so a `triton` reachable at runtime cannot
    hide behind a lazy import inside a function -- `ast.walk` sees those too), and the
    DECLARATION (`pyproject.toml`'s `dependencies`, so a dependency nothing imports
    cannot linger either). Revival of the kernels is git; the ledger row is
    `docs/internals/DELETIONS.md`.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    offending: list[str] = []
    for path in sorted((root / "fsla").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name == "triton" or name.startswith("triton."):
                    offending.append(f"{path.relative_to(root)}:{node.lineno} imports {name}")
    assert not offending, (
        "the package still imports Triton, so `docs/build.md`'s no-triton claim is "
        f"false: {offending}")

    text = (root / "pyproject.toml").read_text()
    declared = text.split("dependencies    = ", 1)[1].split("\n", 1)[0]
    assert "triton" not in declared, (
        f"pyproject.toml still declares a triton dependency: {declared}")


@_CUDA
def test_a_level_several_read_heads_tie_runs_forward_and_refuses_its_gradient():
    """Heads 0 and 2 tie level 0: the forward gives each its write head's rows (the solve's second output, and a copy);
    a call that requires the gradient is refused by name, because the backward adds two cotangents and this level has
    three (docs/open-work.md row A-m)."""
    layout = _layout((4, 5), (entmax(1.5), softmax()), ((tied(), softmax()), (softmax(), softmax()), (tied(), softmax())),
                     group=3)
    torch.manual_seed(0)
    logits = torch.randn(2, 40, layout.packed_router_width, device="cuda").to(torch.bfloat16)
    reads, writes, _support = production_routing_levels(logits, layout)
    by_head = reads[0].view(2, 3, 40, 4)
    assert torch.equal(by_head[:, 0], writes[0]) and torch.equal(by_head[:, 2], writes[0])
    assert not torch.equal(by_head[:, 1], writes[0])
    with pytest.raises(NotImplementedError, match="tied by several read heads"):
        production_routing_levels(logits.clone().requires_grad_(), layout)


@_CUDA
def test_an_empty_call_keeps_the_autograd_contract():
    """Zero tokens: the levels keep their shapes and dtype, and the backward still reaches the logits."""
    layouts = (
        _layout((4, 4), (softmax(), softmax()), ((entmax(1.5), entmax(1.5)),)),
        _routings(1.5)["grouped"],
    )
    for layout in layouts:
        logits = torch.empty(1, 0, layout.packed_router_width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        reads, writes, _support = production_routing_levels(logits, layout)
        for level in range(layout.D):
            assert reads[level].shape == (layout.group, 0, layout.widths[level])
            assert writes[level].shape == (1, 0, layout.widths[level])
            assert reads[level].dtype == writes[level].dtype == torch.bfloat16
        (sum(r.float().sum() for r in reads) + sum(w.float().sum() for w in writes)).backward()
        assert logits.grad is not None
