"""Gates for the dense per-level production routing solve (V3's producer path).

`rola.routing.entmax.production.production_routing_factor_levels` is
what `routing/producer_v3.py` calls. It has two lowerings of one semantics -- a DEVICE
lowering at CUDA fp32 and the identical sorted closed form in torch on CPU/fp64 -- and
both are gated here against the ONE canonical ground truth this branch authorizes for
the routing primitive: the fp64 oracle in `routing/reference.py`
(`entmax/reference.py:entmax`), which no runtime path touches. The oracle is a closed
form at both alphas this file tests (`entmax15` at 1.5, `sparsemax` at 2.0 -- the third
party's own exact forms, inverted algebraically for `tau`) -- there is no bisection
anywhere in this file's ground truth.

**THE DEVICE LOWERING IS NOT ONE KERNEL, and this file's wording used to say "fused
Triton" as though it were** (corrected at the thunder merge, 2026-08-02). Independent
levels are still fused Triton; the UNION solve is a hand-CUDA + CUB kernel
(`csrc/rola/src/entmax.cu`, Track E), and the Triton union kernel was deleted when it
landed -- it was measured at 255 registers with spills and an uncontrollable `tl.sort`
scratch, and a JIT is against the closed-world AOT policy. Everything below that says
"the two lowerings" still means exactly what it meant, device against torch; what
changed is WHICH BINARY the union rows were measured on, and the tie table was
re-measured on the new one (see `_TIE_DIVERGENCE_BOUND`).

Support is checked for **exact** set equality, never by threshold: entmax support is a
discrete object ([[sparsity-comes-from-entmax]]), so "the same zeros" is a hard
assertion and only the amplitudes carry a tolerance. Two scopes are deliberately
excluded from that hard assertion, and neither is a tolerance in disguise:

* **softmax sides.** A dense activation has no support; its "zeros" are fp underflow
  and differ between fp32 and fp64 by construction. Exact support is asserted for
  exactly the zero-capable sides (`ResolvedRouting.read/write_level_can_zero`).
* **the exact-tie fixture**, which is not gated against the oracle at all. Where a
  logit sits exactly on the threshold, an oracle comparison is a comparison of TWO
  INDEPENDENT closed forms -- production's fp32 sorted solve and the oracle's fp64
  `entmax_tau` -- at an exact equality, each free to land on either side of the same
  strict `z_i > tau` boundary. They are not obligated to agree there, and measured
  directly they do not always (alpha=1.5, the `union` shape, level 0 disagrees). One
  extra support member at that boundary moves `den` in `s*(dp - num/den)`, so the
  GRADIENT effect of a boundary disagreement is O(1) even though the two sides'
  amplitudes are close -- so nothing about that fixture can be gated on the oracle
  without gating one boundary comparison against a second one that has no obligation
  to agree with it. It is instead gated on the two PRODUCTION lowerings against each
  other -- the SAME closed form computed by two different code paths (the fused
  CUDA/CUB and Triton kernels vs the pure-torch fallback), both handed the SAME fp32
  inputs -- which must therefore agree on support exactly
  (`test_the_two_lowerings_agree_on_exact_ties`).
"""

from __future__ import annotations

import pytest
import torch

import rola.routing.producer as producer_v3
from rola.routing.entmax.production import production_routing_factor_levels
from rola.routing.producer import (
    BudgetConditioning,
    produce_routing,
    resolved_routing_from_topology,
)
from rola.routing.reference import routing_factor_levels as oracle_factor_levels
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    LevelConfig,
    ResolvedRouting,
    SoftmaxActivation,
    Topology,
    UnionRouting,
)

# Identical to `tests/ops/test_rola_entmax_production.py`'s -- the same solve against
# the same oracle, so the same numbers. Never loosened.
_PRODUCTION_RTOL = 2e-5
_PRODUCTION_ATOL = 2e-6
_PRODUCTION_GRAD_RTOL = 5e-5
_PRODUCTION_GRAD_ATOL = 5e-6

#: The CONDITIONED end-to-end VJP bound, DERIVED 2026-08-03 (perf-campaign batch 2b)
#: from a 200-run measurement, replacing a mis-applied `_PRODUCTION_GRAD_*` pair.
#:
#: WHAT WENT WRONG WITH THE OLD BOUND. `test_producer_conditioned_path_backward_
#: matches_its_oracle_path` compares the fp32 production producer against the SAME
#: producer run END-TO-END IN FP64 with the oracle solve. That comparison does NOT
#: isolate the port: the shared conditioning arithmetic (the 4.5 seam inside a
#: 3-chunk, 48-token feedback recurrence -- each chunk's usage statistics feed the
#: next chunk's logits) runs at DIFFERENT precisions in the two arms, so the
#: compared quantity includes the fp32-vs-fp64 drift of shared code, amplified by
#: the recurrence. Applying the solve-level `5e-5 * max|oracle| + 5e-6` to that
#: quantity was under-derived: swept over 50 fixture seeds x 4 parametrizations on
#: the pristine tree, 37 of 200 runs FAIL it on `usage_bias_b` alone (74 leaf
#: failures over all 6 leaves) -- the committed seed 404 passed by selection.
#:
#: THE MEASUREMENT (sweep + three-arm attribution, batch2.md 2b-§1):
#: * Distribution of `max|prod - oracle| / max|oracle|` per leaf over 200 runs:
#:   median 1e-7..6e-6, worst 5.2e-4 (temp_b, seed 409, alpha=1.5, global); the
#:   worst ABSOLUTE deviation on a small-scale leaf is 5.4e-5 (usage_bias_b,
#:   scale 5.0e-2). The tail is SEED-DEPENDENT AMPLIFICATION, not accumulation
#:   length: over L in {24, 48, 96, 192} the median ratio is flat (~2e-5) and the
#:   worst case is wherever an amplification event lands, so a sqrt(L) model is
#:   wrong and a per-token model is wrong; the distribution itself is the bound's
#:   only honest source.
#: * ATTRIBUTION: a control arm running the fp64-quality BISECTION solve on fp32
#:   params (fp32 conditioning, reference solve) deviates from the fp64 arm by the
#:   SAME magnitude as the Triton path at every swept seed (seed 409, alpha=1.5:
#:   control 2.7e-2 vs production 2.1e-2 on route_W), while production-vs-control
#:   at MATCHED precision stays inside the same envelope. The tail is therefore
#:   the shared recurrence's fp32 drift, not the port's arithmetic -- which is
#:   what makes bounding it a tolerance DERIVATION rather than a defect masked.
#:
#: THE BOUND: `1e-3 * max|oracle| + 1e-4`. Against every one of the 1,200
#: measured (run, leaf) points the worst utilization is 0.52, i.e. >= 1.9x slack
#: over the measured worst case in both the scale-relative term (5.2e-4 -> 1e-3)
#: and the absolute floor (5.4e-5 -> 1e-4). The gate asserts across a SEED SWEEP
#: that includes the two worst measured amplification seeds (409, 451), so the
#: bound is exercised against its own tail rather than at one lucky draw.
#: Port-level precision is NOT gated here -- it is gated where it is derivable, at
#: solve granularity (`test_the_two_lowerings_agree_directly` and the oracle
#: gates above, all at the unchanged `_PRODUCTION_*` tolerances).
_CONDITIONED_VJP_RTOL = 1e-3
_CONDITIONED_VJP_ATOL = 1e-4

#: The fixture seeds the conditioned-backward gate asserts at: the committed
#: historical seed plus four from the measured distribution, two of them the worst
#: amplification events the 50-seed sweep found (409: route_W/temp ratio 4.6e-4;
#: 451: the small-scale usage_bias_b worst case). Chosen from the measurement, not
#: for greenness -- every seed here passes only because the bound is derived from
#: the same distribution they were drawn from.
_CONDITIONED_VJP_SEEDS = (404, 409, 411, 449, 451)

#: Union tie-gradient divergence bound: the measured worst case is 4.44% of
#: elements (write side, alpha=1.5, `union`), and this carries ~1.35x headroom.
#: Tightening is the correct direction; loosening requires a re-measurement in
#: `test_the_two_lowerings_agree_on_exact_ties`'s own table.
#:
#: RE-MEASURED AT THE THUNDER MERGE (2026-08-02), because the thing on the CUDA side of
#: this comparison CHANGED. Track E replaced the union solve's fused-Triton lowering
#: with a hand-CUDA + CUB kernel and DELETED the Triton one, so every number in the
#: table below was a measurement of an implementation that no longer exists -- a stale
#: provenance, not a stale bound. Re-run against the new kernel, the table reproduces to
#: the recorded digits: union 1.5 read 31/720 (4.31%) / write 32/720 (4.44%), union 2.0
#: read 2/720 (0.28%) / write 4/720 (0.56%), mixed_per_level 1.5 28/1800 (1.56%) /
#: 33/1800 (1.83%), mixed_per_level 2.0 6/1800 (0.33%) both sides; complement worst case
#: 2.673e-6 of scale, divergence floor 4.857e-6.
#:
#: That it reproduces is a fact about the divergence's CAUSE rather than a coincidence,
#: and it is the reason these bounds are carried forward unchanged rather than merely
#: re-asserted: the divergence is an ill-conditioned division by a value that is exactly
#: 0 at a tie, so what it measures is fp32 rounding UPSTREAM of that division. Two
#: independent closed forms round independently and land the same distance away.
_TIE_DIVERGENCE_BOUND = 0.06

#: The tie test's complement bound, applied to the elements the production
#: tolerance SELECTS -- deliberately TIGHTER than that selector, which is what makes
#: the assertion a different predicate rather than a restatement of its own
#: selection. Read as a fraction of the tensor's own scale (`max|oracle|`), the same
#: scale form `test_producer_conditioned_path_backward_matches_its_oracle_path`
#: already uses and for the same reason: these are composite gradients with
#: near-zero entries, where a bare elementwise relative bound is not the meaningful
#: quantity.
#:
#: MEASURED, with its headroom stated: across all 8 excluded cases (2 shapes x 2
#: alphas x read/write) the largest complement deviation is 2.67e-6 of scale
#: (union, alpha=1.5, write; the other seven are ~2e-7), while the SMALLEST
#: deviation among the diverging elements is 4.86e-6 of scale. The divergence is
#: therefore separated from the agreement by a real gap, and this bound sits inside
#: it with ~3.7x headroom over the measured worst case.
_TIE_COMPLEMENT_SCALE_BOUND = 1e-5

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused production solve needs CUDA")


def _routings(alpha: float) -> dict[str, ResolvedRouting]:
    """Every structural level shape the V3 producer can hand the solve."""
    return {
        "union": ResolvedRouting((UnionRouting(alpha),), (6,)),
        "shared_solve": ResolvedRouting(
            (
                IndependentRouting(True, EntmaxActivation(alpha), EntmaxActivation(alpha)),
                IndependentRouting(False, SoftmaxActivation(), SoftmaxActivation()),
            ),
            (4, 5),
        ),
        "mixed_per_level": ResolvedRouting(
            (
                IndependentRouting(False, EntmaxActivation(alpha), SoftmaxActivation()),
                IndependentRouting(True, SoftmaxActivation(), EntmaxActivation(alpha)),
                UnionRouting(alpha),
            ),
            (3, 5, 4),
        ),
        "dense_only": ResolvedRouting(
            (IndependentRouting(False, SoftmaxActivation(), SoftmaxActivation()),), (4,)),
    }


def _stress_logits(routing: ResolvedRouting, kind: str, *, device, seed: int) -> torch.Tensor:
    """Stress fixtures for the threshold solve, in the packed ``[BH, T, packed_logit_width]`` layout."""
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (3, 40, routing.packed_logit_width)
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
        # ONE dominant digit per LEVEL per row. In the packed layout the dominant index
        # must land inside the level's own span (`level_logit_slice`) -- a single index
        # drawn over the whole packed width would leave every other level a constant row,
        # whose entmax is uniform, i.e. the opposite of the collapsed support this
        # fixture exists to produce.
        for level, width in enumerate(routing.branches):
            span = routing.level_logit_slice(level)
            index = torch.randint(0, width, shape[:-1], device=device, generator=generator)
            collapsed[..., span] = collapsed[..., span].scatter(-1, index.unsqueeze(-1), 20.0)
        return collapsed
    if kind == "large_magnitude":
        # 5x the router's natural logit scale: the solve's support collapses hard, so
        # the row-max shift is what keeps it finite. The scale is deliberately not pushed
        # further because union's read side is `entmax_value * sigmoid(read - write)`, and
        # once a support member's entmax value falls to ~1e-18 its `log` in the write
        # softmax is fp32-ill-conditioned -- a conditioning property of the fixture, not
        # of the solve (both production lowerings agree with each other to 2e-7 there,
        # and disagree with fp64 together). Union's extreme-magnitude contract (the
        # overflow-free delta) is gated at the fp32 maximum by
        # `test_rola_entmax_production.py::test_production_union_handles_near_max_logits_and_vjp`.
        return base * 5.0
    raise ValueError(kind)


def _assert_levels_match(actual, expected, *, gradient=False):
    rtol = _PRODUCTION_GRAD_RTOL if gradient else _PRODUCTION_RTOL
    atol = _PRODUCTION_GRAD_ATOL if gradient else _PRODUCTION_ATOL
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, check_dtype=False)


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


def _check_against_oracle(
    routing: ResolvedRouting, logits_read, logits_write, *, label: str, exact_support: bool = True,
):
    production_read = logits_read.detach().clone().requires_grad_()
    production_write = logits_write.detach().clone().requires_grad_()
    oracle_read = logits_read.detach().clone().to(torch.float64).requires_grad_()
    oracle_write = logits_write.detach().clone().to(torch.float64).requires_grad_()

    reads, writes = production_routing_factor_levels(
        production_read, production_write, routing)
    oracle_reads, oracle_writes = oracle_factor_levels(oracle_read, oracle_write, routing)

    assert len(reads) == len(writes) == routing.D
    for level in range(routing.D):
        assert reads[level].shape == (
            logits_read.shape[0], logits_read.shape[1], routing.branches[level])
        if exact_support and routing.read_level_can_zero[level]:
            _assert_support_identical(
                reads[level], oracle_reads[level], f"{label} read level {level}")
        if exact_support and routing.write_level_can_zero[level]:
            _assert_support_identical(
                writes[level], oracle_writes[level], f"{label} write level {level}")
        _assert_levels_match(reads[level], oracle_reads[level])
        _assert_levels_match(writes[level], oracle_writes[level])

    generator = torch.Generator(device=logits_read.device).manual_seed(9109)
    cotangents = [
        torch.randn(level.shape, device=level.device, dtype=torch.float32, generator=generator)
        for level in (*reads, *writes)
    ]
    loss = sum(
        (level * cotangent).sum()
        for level, cotangent in zip((*reads, *writes), cotangents))
    oracle_loss = sum(
        (level * cotangent.to(torch.float64)).sum()
        for level, cotangent in zip((*oracle_reads, *oracle_writes), cotangents))
    production_vjp = torch.autograd.grad(loss, (production_read, production_write))
    oracle_vjp = torch.autograd.grad(oracle_loss, (oracle_read, oracle_write))
    _assert_levels_match(production_vjp[0], oracle_vjp[0], gradient=True)
    _assert_levels_match(production_vjp[1], oracle_vjp[1], gradient=True)


@pytest.mark.parametrize("alpha", [1.5, 2.0])
@pytest.mark.parametrize("shape", ["union", "shared_solve", "mixed_per_level", "dense_only"])
@pytest.mark.parametrize(
    "kind", ["random", "near_boundary", "collapsed_support", "large_magnitude"])
def test_torch_lowering_matches_the_fp64_oracle(alpha, shape, kind):
    """CPU/fp64 lowering: exact same support, amplitudes and VJP as the closed-form oracle."""
    routing = _routings(alpha)[shape]
    read = _stress_logits(routing, kind, device="cpu", seed=17)
    write = _stress_logits(routing, kind, device="cpu", seed=18)
    _check_against_oracle(routing, read, write, label=f"cpu/{shape}/{kind}/{alpha}")


@_CUDA
@pytest.mark.parametrize("alpha", [1.5, 2.0])
@pytest.mark.parametrize("shape", ["union", "shared_solve", "mixed_per_level", "dense_only"])
@pytest.mark.parametrize(
    "kind", ["random", "near_boundary", "collapsed_support", "large_magnitude"])
def test_triton_lowering_matches_the_fp64_oracle(alpha, shape, kind):
    """CUDA fp32 lowering: exact same support, amplitudes and VJP as the closed-form oracle."""
    routing = _routings(alpha)[shape]
    read = _stress_logits(routing, kind, device="cuda", seed=17)
    write = _stress_logits(routing, kind, device="cuda", seed=18)
    _check_against_oracle(routing, read, write, label=f"cuda/{shape}/{kind}/{alpha}")


@_CUDA
@pytest.mark.parametrize("alpha", [1.5, 2.0])
@pytest.mark.parametrize("shape", ["union", "shared_solve", "mixed_per_level"])
def test_the_two_lowerings_agree_on_exact_ties(alpha, shape):
    """Exact ties, where the oracle cannot arbitrate: the closed forms must still agree.

    Support is asserted as an identical set (a closed form has no convergence slack to
    hide behind) and the amplitudes and the VJP at the production tolerances.
    """
    routing = _routings(alpha)[shape]
    read = _stress_logits(routing, "exact_tie", device="cuda", seed=23).requires_grad_()
    write = _stress_logits(routing, "exact_tie", device="cuda", seed=24).requires_grad_()

    # Soundness gate on the fixture itself: this test is the SOLE gate for tie
    # semantics (explicitly excluded from the oracle gate, see module docstring), so it
    # must not go vacuous under a seed/width change without saying so loudly.
    ties = sum(
        _count_exact_ties(read.detach()[..., routing.level_logit_slice(index)], alpha)
        + _count_exact_ties(write.detach()[..., routing.level_logit_slice(index)], alpha)
        for index, width in enumerate(routing.branches)
    )
    assert ties > 0, (
        f"exact_tie fixture for alpha={alpha} shape={shape!r} contains no exact ties -- "
        "the tie-semantics gate below would be exercising nothing"
    )

    cpu_read = read.detach().cpu().requires_grad_()
    cpu_write = write.detach().cpu().requires_grad_()
    cuda_reads, cuda_writes = production_routing_factor_levels(read, write, routing)
    cpu_reads, cpu_writes = production_routing_factor_levels(cpu_read, cpu_write, routing)
    for level in range(routing.D):
        if routing.read_level_can_zero[level]:
            _assert_support_identical(cuda_reads[level].cpu(), cpu_reads[level], f"read {level}")
        if routing.write_level_can_zero[level]:
            _assert_support_identical(cuda_writes[level].cpu(), cpu_writes[level], f"write {level}")
        _assert_levels_match(cuda_reads[level].cpu(), cpu_reads[level])
        _assert_levels_match(cuda_writes[level].cpu(), cpu_writes[level])

    generator = torch.Generator().manual_seed(5501)
    cotangents = [
        torch.randn(level.shape, dtype=torch.float32, generator=generator)
        for level in (*cpu_reads, *cpu_writes)
    ]
    cuda_vjp = torch.autograd.grad(
        sum((level * cotangent.cuda()).sum()
            for level, cotangent in zip((*cuda_reads, *cuda_writes), cotangents)),
        (read, write))
    cpu_vjp = torch.autograd.grad(
        sum((level * cotangent).sum()
            for level, cotangent in zip((*cpu_reads, *cpu_writes), cotangents)),
        (cpu_read, cpu_write))
    if any(isinstance(level, UnionRouting) for level in routing.levels):
        # V9 item 3 (compliance-audit): this used to `return`, leaving union tie
        # gradients with NO gate at all. The divergence itself is real and is NOT
        # weakened here -- what is added is a BOUND, so the hole becomes a pinned
        # quantity instead of an early exit.
        #
        # CONFIRMED, not assumed (fix-pass re-check of the adversarial review's
        # NOTE): removing the exclusion and comparing elementwise genuinely fails --
        # union-1.5, union-2.0, mixed_per_level-1.5, mixed_per_level-2.0 all
        # mismatch, up to 0.70 relative / 0.23 absolute on 6/1800 elements, orders
        # of magnitude past the 5e-5/5e-6 production tolerance. Root cause: union's
        # VJP divides the write-softmax adjoint by the entmax value
        # (`d_log_weight / midpoint_values`); at a tie that value is exactly 0 in
        # exact arithmetic, so fp32 rounding upstream of the division (row-max
        # shift, sort, prefix sums -- computed independently by the two lowerings,
        # which are not required to round identically) changes which side of
        # zero-ish the divisor lands on, and the division amplifies that into an
        # O(1) divergence in specific elements. This is not a tolerance gap that a
        # looser bound would close: the two lowerings compute genuinely different
        # fp32 numbers from an ill-conditioned division, not the same number to
        # different precision.
        #
        # What IS gateable, and is gated: the divergence is CONFINED. It touches a
        # small fraction of elements and the rest agree at the unmodified
        # production gradient tolerance. A change that widened it -- more diverging
        # elements, or divergence spreading off the tie -- fails here instead of
        # being invisible behind a `return`.
        #
        # CORRECTION to the number the superseded note carried. It quoted
        # "6/1800 elements" as though that were the divergence; measured across
        # every excluded case it is the alpha=2.0 figure only, and alpha=1.5 is an
        # order of magnitude larger:
        #
        #     shape            alpha  read           write
        #     union             1.5   31/720  4.31%  32/720  4.44%
        #     union             2.0    2/720  0.28%   4/720  0.56%
        #     mixed_per_level   1.5   28/1800 1.56%  33/1800 1.83%
        #     mixed_per_level   2.0    6/1800 0.33%   6/1800 0.33%
        #
        # The bound below is the worst of those (4.44%) with ~1.35x headroom.
        # TIGHTENING it is the correct direction for any future change.
        #
# THE COMPLEMENT ASSERTION IS WHAT MAKES THE BOUND MEAN SOMETHING, and it
        # took three goes to write. Recording the two rejected forms, because each
        # was wrong in a way that reads as right:
        #
        # 1. `_assert_levels_match(cuda_grad[close], cpu_grad[close])` -- selecting
        #    with `torch.isclose` at the production tolerance and then asserting the
        #    SAME predicate on the subset that predicate defines.
        #    `|a-b| <= atol + rtol*|b|` holds by construction on every element of
        #    `close`, so it could not fail for ANY input (adversarial review
        #    2026-07-29 §8). It read as "and the complement agrees" while gating
        #    nothing.
        # 2. Selecting at a COARSE tolerance and asserting at the production one --
        #    i.e. claiming the divergence is BIMODAL. That is a genuinely different
        #    predicate, and it FAILS: measured, the diverging elements run down to
        #    9.8e-4 relative (union, alpha=1.5, read) and 5.1e-4 (mixed_per_level,
        #    alpha=1.5, write), so there is no coarse relative threshold with both
        #    the whole divergence above it and the whole agreement below it. The
        #    claim was false and the test said so. It is recorded rather than
        #    quietly dropped, because "the divergence is bimodal in relative error"
        #    is exactly the kind of thing a future reader would assume.
        #
        # What IS true, and is what this asserts: the separation is in the deviation
        # measured against the TENSOR'S SCALE, not elementwise-relative. Diverging
        # elements deviate by >= 4.86e-6 of scale; agreeing ones by <= 2.67e-6.
        #
        # THE BOUND IS A MARGIN, NOT A CONTAINMENT (corrected 2026-07-30,
        # adversarial review §6). `_TIE_COMPLEMENT_SCALE_BOUND` is 1e-5, which is
        # 2.06x ABOVE the top of that gap (4.86e-6) -- it does not sit inside it,
        # and the commit message, `compaction2.md` §A2 and this comment all said it
        # did. What IS true: it sits 3.7x above the worst MEASURED complement
        # (2.67e-6), so it is a regression bound with headroom, and the injection
        # below is what says it still catches the drift that matters. Tightening to
        # 4e-6 would make the containment claim literally true and is the correct
        # direction for any future change; it is not taken here because the headroom
        # over 2.67e-6 is what absorbs a machine change, and a bound that fails on a
        # new GPU for a reason unrelated to the tie teaches nothing.
        #
        # So the complement is bounded strictly TIGHTER than the selector that chose
        # it -- the reviewer's first suggested form -- and the two predicates are
        # different in the direction that bites: a systematic drift of ~2e-5, small
        # enough to keep most elements inside the production `rtol` and so invisible
        # to the fraction bound above, moves this bound by 2x and fails here. A
        # larger drift (the review's injected 1e-3) fails the fraction bound
        # instead, so between them the two assertions have no blind band.
        for cuda_grad, cpu_grad, side in ((cuda_vjp[0].cpu(), cpu_vjp[0], "read"),
                                          (cuda_vjp[1].cpu(), cpu_vjp[1], "write")):
            close = torch.isclose(cuda_grad, cpu_grad,
                                  rtol=_PRODUCTION_GRAD_RTOL, atol=_PRODUCTION_GRAD_ATOL)
            diverging = int((~close).sum())
            fraction = diverging / close.numel()
            assert fraction <= _TIE_DIVERGENCE_BOUND, (
                f"union tie-gradient divergence WIDENED on the {side} side: "
                f"{diverging}/{close.numel()} = {fraction:.4f} of elements disagree "
                "past the production gradient tolerance, against a measured worst "
                "case of 4.44%. The known divergence is an ill-conditioned division "
                "at a tie; this bound exists so that it staying confined is checked "
                "rather than assumed.")
            scale = max(1e-30, float(cpu_grad.abs().max()))
            complement = float((cuda_grad[close] - cpu_grad[close]).abs().max()) / scale
            assert complement <= _TIE_COMPLEMENT_SCALE_BOUND, (
                f"the {side}-side complement -- the elements the production tolerance "
                f"ACCEPTS -- now deviates by {complement:.3e} of the tensor's scale, past "
                f"the {_TIE_COMPLEMENT_SCALE_BOUND} bound (measured worst case 2.67e-6, "
                "so this bound carries 3.7x MARGIN over the measurement). The known tie "
                "divergence is separated from the agreement by a real gap: agreement "
                "tops out at 2.67e-6 of scale and divergence starts at 4.86e-6. This "
                "bound sits 2.06x ABOVE the top of that gap -- it is a margin, not a "
                "containment -- and it still fails on a systematic drift too small to "
                "move the fraction bound above (verified by injection at 2e-5).")
        return
    _assert_levels_match(cuda_vjp[0].cpu(), cpu_vjp[0], gradient=True)
    _assert_levels_match(cuda_vjp[1].cpu(), cpu_vjp[1], gradient=True)


def _forbid_every_reference_entry_point(monkeypatch) -> None:
    """Patch all THREE of the reference module's public entry points
    (``__all__ = ["entmax", "entmax_forward", "entmax_tau"]``), not just the one this
    solve happens to call today. Patching only ``entmax_tau`` proves the solve does not
    reach that one function; it says nothing about ``entmax`` or ``entmax_forward``, and
    a caller that reached either of those would have passed the narrower guard."""
    import rola.routing.entmax.reference as entmax_reference

    def _forbidden(*args, **kwargs):
        raise AssertionError("production routing called the fp64 reference")

    for name in entmax_reference.__all__:
        monkeypatch.setattr(entmax_reference, name, _forbidden)


@_CUDA
def test_the_solve_never_reaches_the_reference_cuda(monkeypatch):
    """No runtime path may fall back to the oracle -- the port's whole point (CUDA half)."""
    _forbid_every_reference_entry_point(monkeypatch)
    routing = _routings(1.5)["mixed_per_level"]
    read = _stress_logits(routing, "random", device="cuda", seed=31).requires_grad_()
    write = _stress_logits(routing, "random", device="cuda", seed=32).requires_grad_()
    reads, writes = production_routing_factor_levels(read, write, routing)
    sum(level.sum() for level in (*reads, *writes)).backward()
    assert read.grad is not None and write.grad is not None


def test_the_solve_never_reaches_the_reference_cpu(monkeypatch):
    """No runtime path may fall back to the oracle -- the port's whole point (CPU half).

    Split from the CUDA half (adversarial review NOTE): the torch lowering needs no GPU,
    so this must run on a CPU-only runner instead of being skipped along with the CUDA
    half under one `@_CUDA`-marked test.
    """
    _forbid_every_reference_entry_point(monkeypatch)
    routing = _routings(1.5)["mixed_per_level"]
    read = _stress_logits(routing, "random", device="cpu", seed=31).requires_grad_()
    write = _stress_logits(routing, "random", device="cpu", seed=32).requires_grad_()

    cpu_reads, cpu_writes = production_routing_factor_levels(read, write, routing)
    sum(level.sum() for level in (*cpu_reads, *cpu_writes)).backward()
    assert read.grad is not None and write.grad is not None


# ---------------------------------------------------------------------------
# End-to-end: the V3 producer on the production path against the V3 producer on
# the fp64 oracle path. Same producer, same conditioning arithmetic, only the
# routing solve swapped -- so any difference is the port's.
# ---------------------------------------------------------------------------


def _producer_topology(alpha: float) -> Topology:
    return Topology(
        levels=(
            LevelConfig(4, IndependentRouting(True, EntmaxActivation(alpha), EntmaxActivation(alpha))),
            LevelConfig(6, UnionRouting(alpha)),
            LevelConfig(3, IndependentRouting(False, SoftmaxActivation(), SoftmaxActivation())),
        ),
        d_v=5,
    )


def _producer_params(topology: Topology, *, dtype, device, sides, B=2, L=48, H=2, dm=16,
                     seed=404):
    resolved = resolved_routing_from_topology(topology)
    generator = torch.Generator(device=device).manual_seed(seed)

    def _randn(*shape):
        return torch.randn(shape, device=device, dtype=torch.float32, generator=generator).to(dtype)

    return {
        "x": _randn(B, L, dm),
        "route_W": _randn(H, dm, resolved.packed_router_width),
        "gain_W": _randn(H, sides, dm),
        "gain_bias": torch.zeros(H, sides, device=device, dtype=dtype),
        "v": _randn(B, L, H, topology.d_v),
    }


def _run_producer(params, topology, normalization, *, budget=None, r=None, actuators=None):
    kwargs = dict(actuators or {})
    return produce_routing(
        params["x"], None, params["v"], params["route_W"], None, params["gain_W"],
        params["gain_bias"], topology, normalization, budget=budget, r=r, **kwargs)


@_CUDA
@pytest.mark.parametrize("alpha", [1.5, 2.0])
@pytest.mark.parametrize("normalization", ["raw", "global"])
@pytest.mark.parametrize("conditioned", [False, True])
def test_producer_production_path_matches_its_oracle_path(
    monkeypatch, alpha, normalization, conditioned):
    """`produce_routing` end to end: production solve vs fp64 oracle solve.

    Support sets must match EXACTLY on every zero-capable side (entmax support is
    discrete); amplitudes, side gains and the exact support counts carry the production
    tolerance. ``conditioned=True`` runs the architecture 4.5 chunk-recurrent path, so
    the usage/temperature bias seam -- which sits between the logits and the solve --
    is covered on both paths.
    """
    topology = _producer_topology(alpha)
    resolved = resolved_routing_from_topology(topology)
    budget = r = None
    if conditioned:
        budget = BudgetConditioning(
            budgeted_write_levels=(1,), budgeted_read_levels=(0,),
            usage_bias_enabled=True, chunk_size=16)

    def _actuators(dtype):
        if not conditioned:
            return None
        generator = torch.Generator(device="cuda").manual_seed(77)
        pairs = len(budget.budgeted_pairs)

        def _randn(*shape):
            return 0.1 * torch.randn(
                shape, device="cuda", dtype=torch.float32, generator=generator).to(dtype)

        return {
            "temp_W": _randn(2, pairs, 2),
            "temp_b": _randn(2, pairs),
            "usage_bias_W": _randn(2, topology.D, 4),
            "usage_bias_b": _randn(2, topology.D),
        }

    production_params = _producer_params(
        topology, dtype=torch.float32, device="cuda",
        sides=1 if normalization == "global" else 2)
    oracle_params = {name: tensor.to(torch.float64) for name, tensor in production_params.items()}
    if conditioned:
        r = torch.full((production_params["x"].shape[0],), 0.5, device="cuda")

    production = _run_producer(
        production_params, topology, normalization, budget=budget,
        r=None if r is None else r.float(), actuators=_actuators(torch.float32))
    monkeypatch.setattr(
        producer_v3, "production_routing_factor_levels", oracle_factor_levels)
    oracle = _run_producer(
        oracle_params, topology, normalization, budget=budget,
        r=None if r is None else r.double(), actuators=_actuators(torch.float64))

    for level in range(topology.D):
        if resolved.read_level_can_zero[level]:
            _assert_support_identical(
                production.read_levels[level], oracle.read_levels[level], f"read {level}")
        if resolved.write_level_can_zero[level]:
            _assert_support_identical(
                production.write_levels[level], oracle.write_levels[level], f"write {level}")
        _assert_levels_match(production.read_levels[level], oracle.read_levels[level])
        _assert_levels_match(production.write_levels[level], oracle.write_levels[level])
        for side in ("read", "write"):
            torch.testing.assert_close(
                production.exact_support_counts[(level, side)],
                oracle.exact_support_counts[(level, side)], rtol=0, atol=0, check_dtype=False)
        assert torch.equal(
            production.touched_bits[level], oracle.touched_bits[level])

    _assert_levels_match(production.g_write, oracle.g_write)
    if normalization == "raw":
        _assert_levels_match(production.g_read, oracle.g_read)
    else:
        assert production.g_read is None and oracle.g_read is None
    for pair, slack in production.slack_history.items():
        _assert_levels_match(slack, oracle.slack_history[pair])


# ---------------------------------------------------------------------------
# V9 (compliance-audit) -- gate gaps named by adversarial review, closed here.
# ---------------------------------------------------------------------------


@_CUDA
@pytest.mark.parametrize("alpha", [1.5, 2.0])
@pytest.mark.parametrize("shape", ["union", "shared_solve", "mixed_per_level", "dense_only"])
@pytest.mark.parametrize(
    "kind", ["random", "near_boundary", "collapsed_support", "large_magnitude"])
def test_the_two_lowerings_agree_directly(alpha, shape, kind):
    """V9 item 2: torch vs Triton, DIRECTLY, on the ordinary fixtures.

    Both lowerings are separately gated against the fp64 oracle above, so
    their mutual drift was only TRANSITIVELY bounded, at twice the oracle tolerance
    (producer-port-review Claim 2). Two lowerings can each sit at the edge of that
    envelope on opposite sides and still pass, which is exactly the drift a port
    needs to be gated on. This asserts the pair directly, from the SAME fp32 inputs,
    at the unmodified production tolerances.

    Support is an EXACT set assertion: both are closed forms, so neither has
    convergence slack to hide behind ([[sparsity-comes-from-entmax]] -- entmax
    support is a discrete object, not a thresholded one). The exact-TIE fixture is
    excluded because it has its own gate,
    ``test_the_two_lowerings_agree_on_exact_ties``, with its own measured bound.
    """
    routing = _routings(alpha)[shape]
    cuda_read = _stress_logits(routing, kind, device="cuda", seed=17).requires_grad_()
    cuda_write = _stress_logits(routing, kind, device="cuda", seed=18).requires_grad_()
    cpu_read = cuda_read.detach().cpu().requires_grad_()
    cpu_write = cuda_write.detach().cpu().requires_grad_()
    assert torch.equal(cuda_read.detach().cpu(), cpu_read.detach()), (
        "the two lowerings must be handed bit-identical inputs, or this measures the "
        "fixture rather than the port")

    cuda_reads, cuda_writes = production_routing_factor_levels(cuda_read, cuda_write, routing)
    cpu_reads, cpu_writes = production_routing_factor_levels(cpu_read, cpu_write, routing)
    for level in range(routing.D):
        if routing.read_level_can_zero[level]:
            _assert_support_identical(
                cuda_reads[level].cpu(), cpu_reads[level], f"{shape}/{kind} read {level}")
        if routing.write_level_can_zero[level]:
            _assert_support_identical(
                cuda_writes[level].cpu(), cpu_writes[level], f"{shape}/{kind} write {level}")
        _assert_levels_match(cuda_reads[level].cpu(), cpu_reads[level])
        _assert_levels_match(cuda_writes[level].cpu(), cpu_writes[level])

    generator = torch.Generator().manual_seed(6607)
    cotangents = [
        torch.randn(level.shape, dtype=torch.float32, generator=generator)
        for level in (*cpu_reads, *cpu_writes)
    ]
    cuda_vjp = torch.autograd.grad(
        sum((level * cotangent.cuda()).sum()
            for level, cotangent in zip((*cuda_reads, *cuda_writes), cotangents)),
        (cuda_read, cuda_write))
    cpu_vjp = torch.autograd.grad(
        sum((level * cotangent).sum()
            for level, cotangent in zip((*cpu_reads, *cpu_writes), cotangents)),
        (cpu_read, cpu_write))
    _assert_levels_match(cuda_vjp[0].cpu(), cpu_vjp[0], gradient=True)
    _assert_levels_match(cuda_vjp[1].cpu(), cpu_vjp[1], gradient=True)


@_CUDA
@pytest.mark.parametrize("alpha", [1.5, 2.0])
@pytest.mark.parametrize("normalization", ["raw", "global"])
@pytest.mark.parametrize("seed", _CONDITIONED_VJP_SEEDS)
def test_producer_conditioned_path_backward_matches_its_oracle_path(monkeypatch, alpha,
                                                                    normalization, seed):
    """V9 item 1: the §4.5 conditioning seam composed with the TRITON backward.

    ``test_producer_production_path_matches_its_oracle_path`` runs
    ``conditioned=True`` FORWARD ONLY, and the CPU test that does check backward
    (`test_rola_v3_producer.py::test_conditioned_path_matches_oracle_contract_and_
    differentiates_all_actuators`) exercises the TORCH lowering -- so the
    composition of the conditioning seam with the fused Triton VJP had no gate at
    all (producer-port-review Claim 4). It has one here.

    The seam is architecture §4.5's per-digit logit bias `g` and per-side
    temperature `c`, applied BETWEEN the logits and the activation, inside a
    sequential chunk recurrence. Two things are asserted:

    1. Every actuator receives a FINITE, NONZERO gradient. This is the failure this
       gate exists for -- [[layer-dispatch-autograd-gap]]: a path reachable from a
       grad context that silently breaks the graph passes every forward gate.
    2. Every gradient matches the same producer run END-TO-END IN FP64 with the
       oracle solve, within `_CONDITIONED_VJP_RTOL/_ATOL` -- a bound DERIVED from
       the measured distribution of that comparison (see the constant's comment).
       This comparison does NOT isolate the port: the shared conditioning
       recurrence runs at different precisions in the two arms, so the deviation
       is dominated by fp32-vs-fp64 drift of shared code (measured by a
       control arm running the oracle solve on fp32 params, which reproduces
       the deviation without the Triton kernel anywhere in the loop). What this
       assertion gates is the COMPOSITION: a port whose backward is detached,
       support-broken, or grossly wrong fails it; port-level precision is gated at
       solve granularity by the tests above, at the tight `_PRODUCTION_*` bounds.

    Asserted across `_CONDITIONED_VJP_SEEDS` -- including the worst amplification
    seeds the 50-seed derivation sweep found -- because a single-seed assertion
    against a seed-dependent amplification distribution is selection, not a gate
    (the pre-repair bound passed at seed 404 and failed at 19 of 25 neighbours).

    The bound is applied at the TENSOR's scale, not per element: these composite
    gradients contain near-zero entries where a per-element relative bound is
    meaningless (measured pre-repair: per-element relative disagreement up to
    4.5e7 on `usage_bias_b` whose absolute disagreement was 4.7e-5 at scale 59).

    The support sets are asserted identical FIRST, which is what makes a tolerance
    admissible at all: a support disagreement between the two solves would be an
    O(1) gradient difference, not a rounding one, and no tolerance should absorb it.
    """
    topology = _producer_topology(alpha)
    budget = BudgetConditioning(
        budgeted_write_levels=(1,), budgeted_read_levels=(0,),
        usage_bias_enabled=True, chunk_size=16)
    pairs = len(budget.budgeted_pairs)

    def _actuators(dtype):
        generator = torch.Generator(device="cuda").manual_seed(77)

        def _randn(*shape):
            return 0.1 * torch.randn(
                shape, device="cuda", dtype=torch.float32, generator=generator).to(dtype)

        return {
            "temp_W": _randn(2, pairs, 2),
            "temp_b": _randn(2, pairs),
            "usage_bias_W": _randn(2, topology.D, 4),
            "usage_bias_b": _randn(2, topology.D),
        }

    sides = 1 if normalization == "global" else 2
    production_params = _producer_params(
        topology, dtype=torch.float32, device="cuda", sides=sides, seed=seed)
    oracle_params = {name: tensor.to(torch.float64) for name, tensor in production_params.items()}
    r = torch.full((production_params["x"].shape[0],), 0.5, device="cuda")

    def _run(params, actuators, r_arg):
        leaves = {name: tensor.detach().clone().requires_grad_()
                  for name, tensor in params.items() if name in ("route_W", "gain_W")}
        act = {name: tensor.detach().clone().requires_grad_()
               for name, tensor in actuators.items()}
        out = produce_routing(
            params["x"], None, params["v"], leaves["route_W"], None, leaves["gain_W"],
            params["gain_bias"], topology, normalization, budget=budget, r=r_arg, **act)
        # A cotangent that touches every level on both sides and the side gains, so
        # no branch of the seam can be exercised by accident.
        loss = sum(level.square().sum() for level in (*out.read_levels, *out.write_levels))
        loss = loss + out.g_write.square().sum()
        if out.g_read is not None:
            loss = loss + out.g_read.square().sum()
        names = ["route_W", "gain_W", *act]
        tensors = [leaves["route_W"], leaves["gain_W"], *act.values()]
        return out, names, torch.autograd.grad(loss, tensors)

    production, names, production_grads = _run(
        production_params, _actuators(torch.float32), r.float())
    monkeypatch.setattr(
        producer_v3, "production_routing_factor_levels", oracle_factor_levels)
    oracle, _, oracle_grads = _run(oracle_params, _actuators(torch.float64), r.double())

    resolved = resolved_routing_from_topology(topology)
    for level in range(topology.D):
        if resolved.read_level_can_zero[level]:
            _assert_support_identical(
                production.read_levels[level], oracle.read_levels[level],
                f"conditioned read {level}")
        if resolved.write_level_can_zero[level]:
            _assert_support_identical(
                production.write_levels[level], oracle.write_levels[level],
                f"conditioned write {level}")

    for name, production_grad, oracle_grad in zip(names, production_grads, oracle_grads):
        assert production_grad is not None, f"{name} received no gradient through the seam"
        assert torch.isfinite(production_grad).all(), f"{name} gradient is not finite"
        assert float(production_grad.abs().max()) > 0.0, (
            f"{name} gradient is identically zero -- the §4.5 seam is detached on the "
            "Triton path, which a forward-only gate cannot see")
        scale = float(oracle_grad.abs().max())
        limit = _CONDITIONED_VJP_RTOL * scale + _CONDITIONED_VJP_ATOL
        deviation = float((production_grad.double() - oracle_grad).abs().max())
        assert deviation <= limit, (
            f"{name}: the conditioned production VJP deviates from the fp64 "
            f"oracle-path VJP by {deviation:.3e}, past {limit:.3e} "
            f"(= {_CONDITIONED_VJP_RTOL} * {scale:.3e} + {_CONDITIONED_VJP_ATOL}). "
            "The support sets were already asserted identical above, and the bound "
            "carries >= 1.9x slack over the measured 200-run worst case of the "
            "shared recurrence's fp32-vs-fp64 drift -- so this is either a real "
            "port regression or a drift event past everything the derivation "
            "sweep observed; re-run the batch-2b sweep before touching the bound.")


def test_production_imports_nothing_from_the_reference_package():
    """The STRONGER, STATIC claim, checked directly rather than inferred from a
    monkeypatch. `pyproject.toml` and this module's own docstring say production
    "imports nothing" from the reference package -- a claim about the IMPORT GRAPH,
    stronger than and independent of `test_the_solve_never_reaches_the_reference_*`
    above (those prove the three entry points are never CALLED at runtime; a module
    can import a name and still never call it). This is the trivially checkable form
    the adversarial review found asserted nowhere: read `production.py` as an AST and
    confirm no import statement names `rola.routing.entmax.reference` or
    `rola.routing.reference` (the oracle-facing module of the same shape), by module
    or by symbol.
    """
    import ast
    import inspect

    import rola.routing.entmax.production as production_module

    source = inspect.getsource(production_module)
    tree = ast.parse(source, filename=production_module.__file__)
    forbidden_modules = {"rola.routing.entmax.reference", "rola.routing.reference"}
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
        "producer's own contract (module docstring, pyproject.toml) is that it "
        "imports nothing from it")
