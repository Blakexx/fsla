"""DOCKET GATE C -- execution identity: assert WHICH KERNEL RAN, not just the answer.

`development/plans/PERF_CAMPAIGN_DOCKET.md` gate C. Every other gate in this tree
checks a VALUE: the solve's output against the fp64 oracle, the consumer's against
the naive. A value gate cannot see which lowering produced the value, so a route
that quietly lands on a different kernel than the one its config names is invisible
to the entire suite -- it shows up, if at all, as a line in a profile that somebody
happens to read.

That is not hypothetical. `workflows/perf-breakdown/REPORT.md` (iii) found the
freeze-defining parity arm spending 4.9 ms/step in the Triton
`_production_fwd_kernel` where the sparse-both arm spent 1.5 ms in the hand-CUDA
`union_forward_kernel`, and it took a per-kernel profile at the L=16384 cell to
notice. Nothing was WRONG -- both arms matched their oracles -- but nobody had
written down which kernel each routing family is supposed to reach, so there was
nothing to violate.

This file writes it down. `_DISPATCH` is the map of record: routing family ->
(the kernel that must run, the kernels that must NOT). It is a
*characterization* gate -- it pins today's truth so that a future change of
lowering is a deliberate edit to this table with a reason attached, and a
*leak* is a test failure instead of a profiling anecdote.

NOTE the map records that the independent (non-union) families are on TRITON. That
is the current state of the tree, not an endorsement: there is no CUDA kernel for
an independent entmax or a softmax (`csrc/rola/rola_api.cpp` exports only
`entmax_union_forward`/`entmax_union_backward`), so kill-Triton compliance for the
producer is unfinished work, not a regression. When those kernels land, this table
is the thing that changes -- and the equivalence gates next door are what license
the change.

Method: `torch.profiler` with CUDA activities only, so the assertion is on kernels
the GPU actually executed. Deliberately NOT a mock or a call counter on the Python
launcher: a counter proves which `_launch_*` helper Python called, which is the
thing under test, not evidence about it.
"""

import pytest
import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from rola.routing.entmax.production import production_routing_factor_levels
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    ResolvedRouting,
    SoftmaxActivation,
    UnionRouting,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

# The kernels, by the name the profiler reports. Substrings, because the CUDA ones
# are C++ templates (`union_forward_kernel<64, 2, true, float>`) and the Triton ones
# carry a specialization suffix.
UNION_CUDA_FWD = "union_forward_kernel"
UNION_CUDA_BWD = "union_backward_kernel"
TRITON_ENTMAX_FWD = "_production_fwd_kernel"
TRITON_ENTMAX_BWD = "_production_bwd_kernel"
TRITON_SOFTMAX_FWD = "_production_softmax_fwd_kernel"
TRITON_SOFTMAX_BWD = "_production_softmax_bwd_kernel"

_WIDTH = 8
_TOKENS = 64
_STREAMS = 2

# routing family -> (required forward kernels, forbidden forward kernels,
#                    required backward kernels, forbidden backward kernels)
_DISPATCH = {
    "union": (
        ResolvedRouting((UnionRouting(1.5),), (_WIDTH,)),
        [UNION_CUDA_FWD], [TRITON_ENTMAX_FWD, TRITON_SOFTMAX_FWD],
        [UNION_CUDA_BWD], [TRITON_ENTMAX_BWD, TRITON_SOFTMAX_BWD],
    ),
    # THE PARITY ARM (bench_scaling's `v3` row): read dense / write sparse, untied.
    # This is the exact route the breakdown found on Triton.
    "independent_softmax_read_entmax_write": (
        ResolvedRouting(
            (IndependentRouting(False, SoftmaxActivation(), EntmaxActivation(1.5)),), (_WIDTH,)),
        [TRITON_ENTMAX_FWD, TRITON_SOFTMAX_FWD], [UNION_CUDA_FWD],
        [TRITON_ENTMAX_BWD, TRITON_SOFTMAX_BWD], [UNION_CUDA_BWD],
    ),
    "independent_entmax_shared_solve": (
        ResolvedRouting(
            (IndependentRouting(True, EntmaxActivation(1.5), EntmaxActivation(1.5)),), (_WIDTH,)),
        [TRITON_ENTMAX_FWD], [UNION_CUDA_FWD, TRITON_SOFTMAX_FWD],
        [TRITON_ENTMAX_BWD], [UNION_CUDA_BWD, TRITON_SOFTMAX_BWD],
    ),
    "independent_softmax_shared_solve": (
        ResolvedRouting(
            (IndependentRouting(True, SoftmaxActivation(), SoftmaxActivation()),), (_WIDTH,)),
        [TRITON_SOFTMAX_FWD], [UNION_CUDA_FWD, TRITON_ENTMAX_FWD],
        [TRITON_SOFTMAX_BWD], [UNION_CUDA_BWD, TRITON_ENTMAX_BWD],
    ),
}


def _logits(routing, *, requires_grad=False):
    torch.manual_seed(17)
    read = torch.randn(
        _STREAMS, _TOKENS, routing.packed_logit_width, device="cuda", dtype=torch.float32)
    write = torch.randn_like(read)
    return read.requires_grad_(requires_grad), write.requires_grad_(requires_grad)


def observed_cuda_kernels(call) -> set[str]:
    """Every CUDA kernel name ONE call to ``call`` launches.

    Warmed first: a Triton kernel's first call JITs, and the compile-time launches
    would otherwise be attributed to the measured region.
    """
    call()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    return {
        event.key for event in prof.key_averages()
        if event.device_type == DeviceType.CUDA and event.self_device_time_total > 0
    }


def _assert_identity(observed, required, forbidden, *, route, direction):
    for name in required:
        assert any(name in kernel for kernel in observed), (
            f"{route} {direction}: expected kernel {name!r} did NOT run. Observed: "
            f"{sorted(observed)}. Either the dispatch changed or this table is stale -- "
            "if the change is deliberate, edit `_DISPATCH` and say why.")
    for name in forbidden:
        leaked = [kernel for kernel in observed if name in kernel]
        assert not leaked, (
            f"{route} {direction}: forbidden kernel {name!r} ran ({leaked}). This is the "
            "leak class docket gate C exists to catch -- a routing family reached a "
            "lowering its config does not name.")


@pytest.mark.cuda
@pytest.mark.parametrize("route", sorted(_DISPATCH))
def test_forward_dispatch_runs_the_named_kernel(route):
    routing, required, forbidden, _, _ = _DISPATCH[route]
    read, write = _logits(routing)
    observed = observed_cuda_kernels(
        lambda: production_routing_factor_levels(read, write, routing))
    _assert_identity(observed, required, forbidden, route=route, direction="forward")


@pytest.mark.cuda
@pytest.mark.parametrize("route", sorted(_DISPATCH))
def test_backward_dispatch_runs_the_named_kernel(route):
    """The VJP is a separate dispatch and gets its own identity assertion.

    the layer-dispatch autograd gap: a forward-only kernel reachable from a grad
    context is exactly the defect an output-only gate misses, so the backward route
    is pinned by name too, not assumed to follow the forward's.
    """
    routing, _, _, required, forbidden = _DISPATCH[route]
    read, write = _logits(routing, requires_grad=True)

    def backward_call():
        reads, writes = production_routing_factor_levels(read, write, routing)
        total = sum(output.square().sum() for output in (*reads, *writes))
        read.grad = None
        write.grad = None
        total.backward()

    observed = observed_cuda_kernels(backward_call)
    _assert_identity(observed, required, forbidden, route=route, direction="backward")


@pytest.mark.cuda
def test_the_identity_gate_has_teeth():
    """A gate that cannot fail is not a gate.

    Two ways this one could be vacuous: the profiler could report no CUDA kernels at
    all (then every `forbidden` assertion passes for free), or the kernel names could
    be so generic that every route matches every name (then every `required`
    assertion passes for free). Both are checked directly.
    """
    union_routing = _DISPATCH["union"][0]
    independent_routing = _DISPATCH["independent_entmax_shared_solve"][0]

    read_u, write_u = _logits(union_routing)
    read_i, write_i = _logits(independent_routing)
    union_kernels = observed_cuda_kernels(
        lambda: production_routing_factor_levels(read_u, write_u, union_routing))
    independent_kernels = observed_cuda_kernels(
        lambda: production_routing_factor_levels(read_i, write_i, independent_routing))

    # 1. the observation is non-empty -- the profiler really sees kernels
    assert union_kernels, "profiler observed no CUDA kernels; every forbidden-assert is vacuous"
    assert independent_kernels

    # 2. the two routes are DISTINGUISHABLE -- the union kernel is absent from the
    #    independent route and vice versa. If this ever holds trivially (identical
    #    observations), the per-route assertions above prove nothing.
    assert union_kernels != independent_kernels, (
        "the union and independent routes ran indistinguishable kernel sets, so the "
        f"dispatch map cannot discriminate between them: {sorted(union_kernels)}")
    assert not any(TRITON_ENTMAX_FWD in kernel for kernel in union_kernels)
    assert not any(UNION_CUDA_FWD in kernel for kernel in independent_kernels)

    # 3. the negative control: a name that must match NOTHING, proving `_assert_identity`
    #    reports a missing kernel rather than passing on an empty match.
    with pytest.raises(AssertionError, match="did NOT run"):
        _assert_identity(
            union_kernels, ["a_kernel_that_does_not_exist"], [],
            route="teeth", direction="forward")
    with pytest.raises(AssertionError, match="forbidden kernel"):
        _assert_identity(
            union_kernels, [], [UNION_CUDA_FWD], route="teeth", direction="forward")
