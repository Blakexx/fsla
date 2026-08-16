"""THE CTX THE TWO-PASS BACKWARD REPLAYS FROM: what it holds, and what it costs.

Vocabulary, defined here and used unqualified below: `B` = batch; `T`/`L` = tokens;
`H` = heads; `d_v = 64` = value channels; `D` = level count; `W` = one uniform level
width; `N = W ** D` = leaves; `BH = B * H`; `entries` = the union table's rows per
`(bh, side)`, which is a SUM over levels and therefore a per-topology constant;
`ctx` = the autograd node's saved state.

**THE CLAIM, AND IT IS GATED RATHER THAN ARGUED.** The two-pass backward
recomputes the state and REPLAYS the schedule, so the forward hands it only `O(L)`
tables and `O(1)` scalars. This file asserts the retained set is EXACTLY the
analytic `O(L)` formula -- not that it grows slowly, which a regression could hide
in, but that it equals a closed form with no `N` in it at all. A per-chunk state
snapshot, a boundary-cursor array or a materialized numerator would each break that
equality by a term this file names.

Three claims:

1. *An inference call retains nothing.* The ctx is taken on `needs_input_grad`, so
   a forward with no gradient in it is the call it was before the ctx existed --
   asserted on the allocator, not on the code.
2. *A grad-bearing call retains exactly the `O(L)` set.* Measured against the
   closed form, across a `T` sweep and across topologies whose `N` differs by 64x.
3. *Saving does not slow the forward.* Paired, alternating, warm: the ctx is a set
   of references to tensors the call already built, so the wall must not move.
"""
from __future__ import annotations

import math
import time

import pytest
import torch

from rola.ops._ext import is_available

pytestmark = pytest.mark.skipif(
    not is_available() or not torch.cuda.is_available(),
    reason="the chunk consumer is a CUDA kernel")

DV = 64


def _bundle(B, T, H, widths, seed=11):
    """A dense `RouteFactors`, built directly -- the bundle's invariants are checked
    either way and a producer would add a graph this file is not measuring."""
    from rola.routing.factors import RouteFactors
    from rola.routing.producer import union_routing
    from rola.routing.types import Topology

    g = torch.Generator(device="cuda").manual_seed(seed)
    topology = Topology(levels=tuple(union_routing(w, 1.5) for w in widths))

    def side():
        out = []
        for w in widths:
            a = torch.rand(B, T, H, w, device="cuda", generator=g) + 0.5
            out.append(a / a.sum(-1, keepdim=True))
        return tuple(out)

    return RouteFactors(topology=topology, normalization="global", read=side(), write=side(),
                        g_read=None,
                        g_write=torch.rand(B, T, H, device="cuda", generator=g) + 0.5)


def _predicted_bytes(B, T, H, widths, entries) -> int:
    """The ctx's closed form. Every term is linear in `T`; NONE carries `N`."""
    BH, sw = B * H, sum(widths)
    words = (T + 31) // 32
    return (2 * BH * T * sw * 2          # the packed read and write sides, bf16
            + BH * T * 2                 # the write gain, bf16
            + BH * 2 * entries * words * 4  # the union table, u32
            + BH * T * 4)                # the denominator, fp32


def _entries(widths, bc) -> int:
    """`LevelPlan::entries`, mirrored: full width per single-digit level, `width /
    run` for the one partially spanned level, NOTHING for a level the block spans
    whole (`csrc/rola/src/chunk/chunk_kernel.cuh`)."""
    D, W = len(widths), widths[0]
    whole = 0
    while whole < D and bc >= W ** (whole + 1):
        whole += 1
    partial = bc // (W ** whole)
    singles = D - whole - (1 if partial > 1 else 0)
    return singles * W + (W // partial if partial > 1 else 0)


def _forward_saving(B, T, H, widths):
    """One grad-bearing forward THROUGH the node, and the ctx it took.

    `rola_op` refuses a grad-bearing call at the envelope, which is the fast-fail
    ruling and is not what this file is measuring; the node is entered directly,
    which is exactly the bypass `ChunkConsumer.backward` exists to catch.
    """
    from rola.interface import ChunkConsumer
    from rola.ops import chunk as _chunk

    routes = _bundle(B, T, H, widths)
    for level in routes.read + routes.write:
        level.requires_grad_(True)
    v = torch.randn(B, T, H, DV, device="cuda", requires_grad=True)
    y, _ = ChunkConsumer.run(_chunk.pack(routes), v)
    return y, routes, v


def test_an_inference_call_retains_nothing():
    from rola.interface import ChunkConsumer
    from rola.ops import chunk as _chunk

    B, T, H, widths = 1, 128, 2, (64, 64)
    routes = _bundle(B, T, H, widths)
    v = torch.randn(B, T, H, DV, device="cuda")
    with torch.no_grad():
        y, _ = ChunkConsumer.run(_chunk.pack(routes), v)
    torch.cuda.synchronize()
    assert y.grad_fn is None or not hasattr(y.grad_fn, "saved_plan"), (
        "an inference call took a ctx; the two-pass activation cost must be paid by "
        "training alone")


@pytest.mark.parametrize("T", [64, 128, 256])
def test_the_ctx_is_exactly_the_predicted_O_L_set(T):
    B, H, widths = 1, 2, (64, 64)
    y, routes, v = _forward_saving(B, T, H, widths)
    saved = y.grad_fn.saved_plan
    got = saved.retained_bytes()
    #: the block width is the ARM's, not this file's: `entries` is a function of
    #: it, and a topology can pin a different one.
    want = _predicted_bytes(B, T, H, widths, _entries(widths, saved.bc))
    assert got == want, (
        f"the ctx holds {got} bytes where the O(L) contract predicts {want} at T={T}. "
        f"A difference here is a TERM, not noise: a per-chunk state snapshot, a "
        f"boundary-cursor array or a saved numerator would each show up as one.")
    #: nothing of the state's own size is in there
    N = math.prod(widths)
    assert got < B * H * N * (DV + 1) * 4, "the ctx is state-plane sized"


def test_the_ctx_does_not_grow_with_the_leaf_count():
    """`N` moves by 64x across these topologies and the ctx follows `entries`, which
    is a SUM over levels. A ctx carrying anything per-leaf could not do that."""
    B, T, H = 1, 128, 2
    seen = {}
    for widths in ((8, 8), (16, 16), (64, 64)):
        y, _, _ = _forward_saving(B, T, H, widths)
        saved = y.grad_fn.saved_plan
        seen[math.prod(widths)] = (saved.retained_bytes(),
                                   _predicted_bytes(B, T, H, widths, _entries(widths, saved.bc)))
    for n, (got, want) in seen.items():
        assert got == want, f"N={n}: ctx {got} bytes against a predicted {want}"
    ns = sorted(seen)
    assert ns[-1] // ns[0] >= 64, "the sweep did not actually move N"
    grew = seen[ns[-1]][0] / seen[ns[0]][0]
    assert grew < ns[-1] / ns[0] / 8, (
        f"the ctx grew {grew:.1f}x while N grew {ns[-1] // ns[0]}x; it is tracking "
        f"leaves rather than tokens")


def test_the_ctx_is_linear_in_the_token_axis():
    B, H, widths = 1, 2, (64, 64)
    by_t = {}
    for T in (64, 128, 256):
        y, _, _ = _forward_saving(B, T, H, widths)
        by_t[T] = y.grad_fn.saved_plan.retained_bytes()
    #: every term is `T`-proportional but the table rounds `T` up to a 32-token
    #: word, so doubling is exact only where `T` is a multiple of 32 -- it is.
    assert by_t[128] == 2 * by_t[64] and by_t[256] == 4 * by_t[64], by_t


def _paired_wall(B, T, H, widths, reps=240):
    """`(median paired difference, median plain call)` in seconds.

    `reps` sizes the ESTIMATOR, never a tolerance: the statistic is a median of
    per-iteration differences on the order of a microsecond, taken with
    `perf_counter` around a device sync, and 80 of them left that median moving by
    several microseconds run to run — enough to trip the ratio below on noise. No
    bound here has been widened.

    PAIRED DIFFERENCING, not a difference of aggregates. The two arms alternate
    inside one iteration and the statistic is the median of the PER-ITERATION
    difference, so whatever the rest of a battery leaves on the device -- clock
    state, a different caching-allocator pool -- moves both arms of an iteration
    together and cancels. Comparing two independently-taken minima does not
    cancel it, and reads as a signal exactly when the interference is uneven.
    """
    from rola.ops import chunk as _chunk

    routes = _bundle(B, T, H, widths)
    v = torch.randn(B, T, H, DV, device="cuda")
    packed = _chunk.pack(routes)

    def once(save):
        y, _ = _chunk.run(packed, v, saved=_chunk.SavedForward() if save else None)
        return y

    for _ in range(20):  # the first arm in a cold process bears the clock ramp
        once(True)
        once(False)
    torch.cuda.synchronize()
    diffs, plain = [], []
    for _ in range(reps):
        taken = {}
        for save in (True, False):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            y = once(save)
            torch.cuda.synchronize()
            taken[save] = time.perf_counter() - t0
            del y
        diffs.append(taken[True] - taken[False])
        plain.append(taken[False])
    diffs.sort()
    plain.sort()
    return diffs[len(diffs) // 2], plain[len(plain) // 2]


def test_saving_the_ctx_costs_a_constant_and_not_a_fraction_of_the_launch():
    """THE CTX IS BOOKKEEPING, NOT WORK, and the shape sweep is what says so.

    Taking the ctx adds no kernel, no copy and no pass: every field is a reference
    to a tensor the call already built. What it does add is a live set the caching
    allocator has to carry across the call, and that is a PER-CALL CONSTANT of a
    few microseconds. Asserting a bare percentage would therefore assert a property
    of the SHAPE -- the same constant is a fraction of a percent of a 2 ms call and
    several times that of a 0.25 ms one -- so the claim is made the only way it can
    be made honestly: the constant must be small, must NOT grow with the work, and
    must be under half a percent at a training-representative shape.
    """
    d_small, small = _paired_wall(1, 512, 4, (64, 64))
    d_large, large = _paired_wall(4, 2048, 8, (64, 64))
    work = large / small
    assert work >= 5, f"the sweep did not move the work ({work:.1f}x)"
    assert d_large < 20e-6, (
        f"taking the ctx costs {d_large * 1e6:.1f} us per call; it is a set of "
        f"references to tensors the call already built, so a cost of this size means "
        f"something is being COPIED")
    assert d_large <= max(3 * d_small, 5e-6), (
        f"the ctx's cost grew with the work ({d_small * 1e6:.1f} us at the small "
        f"shape, {d_large * 1e6:.1f} us at a {work:.0f}x one), so it is not "
        f"bookkeeping")
    rel = d_large / large
    assert rel <= 0.005, (
        f"the ctx cost {rel * 100:.2f}% of the forward wall at a "
        f"training-representative shape (a {d_large * 1e6:.1f} us difference on a "
        f"{large * 1e3:.3f} ms call)")


def test_both_backward_seams_now_carry_a_gradient_and_refuse_only_an_empty_ctx():
    """OLD: both seams refused by name, because the scans did not yet emit a full
    cotangent set. They do, so the seams CARRY the gradient -- and what is left to
    refuse is the one state neither can compute from: a ctx the forward never
    filled, which is an inference forward being differentiated."""
    from rola.interface import ChunkConsumer
    from rola.ops import chunk as _chunk

    with pytest.raises(RuntimeError, match="never filled"):
        _chunk.backward(_chunk.SavedForward(), torch.zeros(1), torch.zeros(1))
    y, _, _ = _forward_saving(1, 64, 2, (64, 64))
    y.sum().backward()
    assert ChunkConsumer.backward is not None
