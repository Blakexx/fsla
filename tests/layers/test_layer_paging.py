"""Spec §7.1 page commitment, at the LAYER.

`tests/ops/test_rola_v3_paging.py` gates the op. This file gates the wiring, which
is a separate failure surface: [[layer-dispatch-autograd-gap]] is precisely the
lesson that an op-level gate does not cover the layer's mode dispatch, and a flag
honoured on one dispatch arm and ignored on the other is the exact shape of that
bug. So the two things asserted here are (1) the paged layer's output is
bit-identical to the unpaged layer's, and (2) the flag is never silently dropped.

Vocabulary: `N` = leaf count; `BC` = leaves per owner; `BH` = `batch * heads`.
"""

from __future__ import annotations

import pytest
import torch

from rola.layers.rola import RoLA

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="V3 paging is a CUDA feature"),
]


def _reordering_bound(y, layer):
    """See `tests/ops/test_rola_v3_paging.py` -- the same derived fp32
    reassociation bound on the §5 readout's atomic accumulation."""
    n = layer.last_selected_plan.geometry.owners_total
    return n * n * torch.finfo(torch.float32).eps * y.abs().max()


def _layer(*, paging, max_physical_pages=None, seed=0, state_norm="raw",
           mma_precision="tf32x3", **kwargs):
    """The paging fixture, pinned to the EXACT MMA arm.

    Every gate in this file compares two decompositions of the same computation
    against `_reordering_bound`, which is a DERIVED fp32 reassociation envelope
    (`n^2 * eps`) on the readout's atomic accumulation. That derivation is about
    fp32 addition order and nothing else, so it only bounds an arm whose operands
    are fp32-precision -- under the production `bf16` default a continuation that
    tiles the tokens differently rounds its MMA operands differently, which is
    bf16-sized rather than eps-sized and is not what these gates are measuring. The
    default is asserted in `test_rola_v3_layer_dispatch.py`; here the arm is pinned
    so the envelope stays the one that was derived.
    """
    torch.manual_seed(seed)
    # `state_block = BC = 16`: the consumer instantiates `BC in {16, 64}` only, and
    # this layer's default (`b_{D-1} = 8`) is outside that set. Stated explicitly so
    # the fixture's shape is a choice, not an accident.
    return RoLA(hidden_size=64, num_heads=2, head_v_dim=32,
                states_per_level=[4, 8, 8], levels_per_head=3, state_block=16,
                state_norm=state_norm, paging=paging, layer_idx=0,
                mma_precision=mma_precision,
                max_physical_pages=max_physical_pages, **kwargs).cuda().eval()


@pytest.mark.parametrize("state_norm", ["raw", "global"])
@pytest.mark.parametrize("decay", ["none", "leaf_mass"])
def test_a_paged_layer_is_bit_identical_to_an_unpaged_one(state_norm, decay):
    torch.manual_seed(3)
    x = torch.randn(2, 48, 64, device="cuda")
    unpaged = _layer(paging=False, seed=5, state_norm=state_norm, decay=decay)
    paged = _layer(paging=True, seed=5, state_norm=state_norm, decay=decay)
    paged.load_state_dict(unpaged.state_dict())

    with torch.no_grad():
        y0, _, _ = unpaged(x)
        # Null control: the SAME unpaged layer, run twice. The consumer's readout
        # accumulates across owners with `atomicAdd`, so `y` is not bit-reproducible
        # against itself (see `tests/ops/test_rola_v3_paging.py`'s module docstring
        # and findings `paging.md` §2). Paging is allowed no more movement than that.
        y_control, _, _ = unpaged(x)
        y1, _, _ = paged(x)

    assert unpaged.last_execution_backend == paged.last_execution_backend == "cuda"
    bound = _reordering_bound(y0, unpaged)
    assert (y0 - y_control).abs().max() <= bound
    assert (y0 - y1).abs().max() <= bound


def test_committed_state_accounting_is_exposed_on_the_layer():
    """§7.1's reported quantities, on the layer's existing plain-attribute surface."""
    torch.manual_seed(4)
    x = torch.randn(2, 48, 64, device="cuda")
    layer = _layer(paging=True, seed=6)
    with torch.no_grad():
        layer(x)

    stats = layer.last_paging_stats
    assert stats is not None
    assert stats.realized_owners > 0
    assert 0 < stats.resident_pages <= stats.owners_total * 4  # BH = batch * heads = 4
    assert stats.peak_pages >= stats.resident_pages
    assert stats.resident_bytes <= stats.dense_bytes
    # Diagnostic state, deliberately not a buffer: it must not reach the state_dict.
    assert not any("paging" in key for key in layer.state_dict())


def test_an_unpaged_layer_reports_no_paging_stats():
    torch.manual_seed(4)
    x = torch.randn(2, 48, 64, device="cuda")
    layer = _layer(paging=False, seed=6)
    with torch.no_grad():
        layer(x)
    assert layer.last_paging_stats is None


def test_the_paging_flag_is_never_silently_dropped_on_the_oracle_arm():
    """The [[layer-dispatch-autograd-gap]] failure mode, closed at its own seam.

    A grad-enabled forward dispatches to the fp64 oracle, which has no page table.
    Running unpaged there would be numerically exact -- paging is bitwise invisible
    -- and that is exactly what makes it dangerous: the `max_physical_pages` cap
    would be silently unenforced, so the layer would compute a DIFFERENT function
    from the one the user configured, in the direction of appearing to work.
    """
    x = torch.randn(2, 48, 64, device="cuda", requires_grad=True)
    layer = _layer(paging=True, seed=7)
    with pytest.raises(NotImplementedError, match="no page table"):
        layer(x)


def test_a_second_fresh_forward_does_not_silently_continue_the_first():
    """The arena's pages PERSIST -- that is what makes them a cache, and it is also
    the thing that must not leak into a fresh forward.

    Without an explicit reset, step 2 of a `use_cache=False` sequence would read
    step 1's committed pages and compute a continuation, while the unpaged path
    allocates zeros every call. That is a different function, silently. This is a
    real defect that was found by measurement (repeated benchmark steps) rather than
    by the single-forward gates above, so it gets a gate of its own.
    """
    torch.manual_seed(11)
    x = torch.randn(2, 48, 64, device="cuda")
    unpaged = _layer(paging=False, seed=12)
    paged = _layer(paging=True, seed=12)
    paged.load_state_dict(unpaged.state_dict())

    with torch.no_grad():
        y0_unpaged, _, _ = unpaged(x)
        y1_unpaged, _, _ = unpaged(x)
        y0_paged, _, _ = paged(x)
        y1_paged, _, _ = paged(x)

    bound = _reordering_bound(y0_unpaged, unpaged)
    assert (y0_unpaged - y0_paged).abs().max() <= bound
    assert (y1_unpaged - y1_paged).abs().max() <= bound, (
        "the second paged forward diverged from the second unpaged one: the arena "
        "carried step 1's pages into a call that must start from zeros")


@pytest.mark.parametrize("n_steps", [2, 3])
def test_a_paged_use_cache_continuation_matches_the_single_forward(n_steps):
    """R4 fix pass (`workflows/20260730-wiring/paging-review.md`): the paged
    continuation arm (`use_cache=True`, which drives `reset(); adopt_dense(...)`
    in `RoLA.forward`) WORKS -- verified by the reviewer at 2-step and 3-step --
    but had no gate. This is that gate: split one 48-token forward into `n_steps`
    `use_cache=True` calls through the ordinary cache machinery and require the
    concatenated output to match the single forward within the same derived
    reordering envelope every other paging gate in this file uses.
    """
    torch.manual_seed(51)
    x = torch.randn(2, 48, 64, device="cuda")
    unpaged = _layer(paging=False, seed=52)
    paged = _layer(paging=True, seed=52)
    paged.load_state_dict(unpaged.state_dict())

    from rola.state import RoLAState

    with torch.no_grad():
        y_single, _, _ = unpaged(x)
        bound = _reordering_bound(y_single, unpaged)

        boundaries = [0] + [round(48 * (i + 1) / n_steps) for i in range(n_steps)]
        cache = RoLAState()
        chunks = []
        for i in range(n_steps):
            lo, hi = boundaries[i], boundaries[i + 1]
            out, _, _ = paged(x[:, lo:hi], past_key_values=cache, use_cache=True)
            chunks.append(out)
        y_continued = torch.cat(chunks, dim=1)

    assert (y_single - y_continued).abs().max() <= bound, (
        f"the {n_steps}-step paged use_cache continuation diverged from the "
        "single-forward baseline by more than the readout's own reordering envelope")
