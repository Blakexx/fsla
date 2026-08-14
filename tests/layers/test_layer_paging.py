"""Spec §7.1 page commitment, at the LAYER.

`tests/ops/test_rola_v3_paging.py` gates the op. This file gates the wiring, which
is a separate failure surface: [[layer-dispatch-autograd-gap]] is precisely the
lesson that an op-level gate does not cover the layer's mode dispatch, and a flag
honoured on one dispatch arm and ignored on the other is the exact shape of that
bug. So the two things asserted here are (1) the paged layer's output is
bit-identical to the unpaged layer's, and (2) the flag is never silently dropped.

Vocabulary: `N` = leaf count; `BC` = leaves per owner; `BH` = `batch * heads`.

**MIGRATED (API-contract break, 2026-08-05).** `paging=True`/`max_physical_pages=`
moved onto `rola.expert.PlanOverrides` (`PlanOverrides(paging=True,
max_physical_pages=...)`, `paging=True` alone being the uncapped arm).
`layer.last_paging_stats` is GONE with no layer-level replacement -- a paged run's
stats live on the arena itself, `layer._page_arena.last_result` (a
`rola.ops.paging.PagingResult`, the identical fields the old stats object carried).

**A SUSPECTED ROLA/ BUG, surfaced by this migration and NOT fixed here (out of
file scope; see the migration report).** `RoLA.forward`'s shared `else` branch
(`rola/layers/rola.py`, "ONE op for both remaining arms") unconditionally sets
`self.last_selected_plan = None` after calling `rola_op` -- for BOTH the oracle
arm (which has no plan, correctly `None`) AND the ordinary unpaged CUDA arm, which
DOES have one (computed inside `rola_op`'s own `_derive_plan`, but never surfaced
by `rola_op`'s `(y, final_state)` return). So `layer.last_selected_plan` reads
`None` after any plain (unpaged, non-decode) CUDA forward -- only the PAGED arm
(`_paged`, which calls `derive_plan`/`execute` directly rather than through
`rola_op`) still stamps a real `Plan`. Every assertion below that used to read
`unpaged.last_selected_plan` is rewritten to use a fact that does not depend on
it (the state_block PIN, which is deterministic here since every fixture pins one
value) or is dropped with a note, rather than being silently weakened to match
the bug.
"""

from __future__ import annotations

import pytest
import torch
from conftest import build_layer

from rola import dense_routing
from rola.expert import PlanOverrides

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="V3 paging is a CUDA feature"),
]

#: Old `decay="none"`/`"leaf_mass"` -> the two shipped decay SOURCES `build_layer`
#: takes (`None`/`"learned"`).
_DECAY = {"none": None, "leaf_mass": "learned"}
_WIDTHS = (4, 8, 8)


def _reordering_bound(y, layer, *, BC):
    """See `tests/tier2_geometry/test_paging_equivalence.py` -- the same derived
    reassociation bound on the §5 readout's atomic accumulation.

    P26: `mma_precision` is a dead axis (bf16 is the only value `PlanOverrides`
    accepts), so `y` is bf16-cast after the fp32 divide on EVERY arm now (`rola/
    layers/rola.py`'s `v`/`out` casts) -- there is no more fp32-precision arm to
    derive an `eps_fp32`-sized envelope from, and the bound moved to
    `torch.finfo(bfloat16).eps`, the same fix the op-level paging suite made for
    the same reason.

    `owners_total = N / BC` is read off the TOPOLOGY and the caller-known pinned
    `BC`, not off `layer.last_selected_plan` (see the module docstring's BUG
    note) -- every fixture here pins `state_block`, so the launch's realized `BC`
    is deterministic and does not need to be read back from the plan.
    """
    n = layer.topology.N // BC
    return n * n * torch.finfo(torch.bfloat16).eps * y.abs().max()


def _layer(*, paging, max_physical_pages=None, seed=0, state_block=16, decay="none"):
    """The paging fixture. `mma_precision` is not a parameter: `PlanOverrides`'
    default is `"bf16"` and it is the only value the dataclass accepts (P26, dead
    axis). `normalization` is pinned `"global"` for the same reason `"raw"` is
    gone everywhere else (`rola.routing.types.validate_normalization`).
    """
    torch.manual_seed(seed)
    # `state_block = 16` default: pins the historical fixture shape (an expert pin
    # shrinks `feasible_BC` to that one width, so every legacy cell keeps the `BC`
    # it was measured at); the H2 flagship cell overrides it with 64.
    return build_layer(
        hidden_size=64, num_heads=2, d_v=32, widths=_WIDTHS, routing=dense_routing(),
        normalization="global", decay=_DECAY[decay], layer_idx=0,
        expert=PlanOverrides(state_block=state_block,
                             paging=paging, max_physical_pages=max_physical_pages),
    ).cuda().eval()


@pytest.mark.parametrize("decay", ["none", "leaf_mass"])
def test_a_paged_layer_is_bit_identical_to_an_unpaged_one(decay):
    if decay != "none":
        pytest.skip("decay routes to the oracle now (retired from the "
                    "operator), and paging is a CUDA-arm feature -- the "
                    "paged-decay claim returns with the decay reopen "
                    "conditions")
    torch.manual_seed(3)
    x = torch.randn(2, 48, 64, device="cuda")
    unpaged = _layer(paging=False, seed=5, decay=decay)
    paged = _layer(paging=True, seed=5, decay=decay)
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
    bound = _reordering_bound(y0, unpaged, BC=16)
    assert (y0 - y_control).abs().max() <= bound
    assert (y0 - y1).abs().max() <= bound


def test_committed_state_accounting_is_exposed_on_the_layer():
    """§7.1's reported quantities. Old: `layer.last_paging_stats`, a layer
    plain-attribute. New: `layer._page_arena.last_result` -- the arena's own
    `PagingResult`, the SAME fields (see the module docstring)."""
    torch.manual_seed(4)
    x = torch.randn(2, 48, 64, device="cuda")
    layer = _layer(paging=True, seed=6)
    with torch.no_grad():
        layer(x)

    stats = layer._page_arena.last_result
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
    assert layer._page_arena is None


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

    bound = _reordering_bound(y0_unpaged, unpaged, BC=16)
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
        bound = _reordering_bound(y_single, unpaged, BC=16)

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


# ---------------------------------------------------------------------------
# H2 (coverage-closure): the paged layer at the FLAGSHIP `BC = 64`, and a `BC`
# flip over a LIVE arena. The §4.5 atom re-key's claim is that every stored
# object and address is `BC`-invariant, which is why `_resolve_page_arena`'s
# rebuild key deliberately omits `BC` while the arena's extent POLICY freezes at
# the construction `BC` -- these cells are what adjudicates that the omission is
# sound at the layer.
# ---------------------------------------------------------------------------


def test_a_paged_BC64_layer_is_bit_identical_to_an_unpaged_one():
    """`test_a_paged_layer_is_bit_identical_to_an_unpaged_one`, pinned to the
    flagship owner width: `state_block=64` shrinks the feasible set to (64,), so
    the selected plan MUST launch `BC = 64` -- structurally, from the pin, since
    `unpaged.last_selected_plan` itself is unreadable here (module docstring's
    BUG note); `paged.last_selected_plan.BC` is still readable and asserted
    directly."""
    torch.manual_seed(71)
    x = torch.randn(2, 48, 64, device="cuda")
    unpaged = _layer(paging=False, seed=72, state_block=64)
    paged = _layer(paging=True, seed=72, state_block=64)
    paged.load_state_dict(unpaged.state_dict())

    with torch.no_grad():
        y0, _, _ = unpaged(x)
        y_control, _, _ = unpaged(x)
        y1, _, _ = paged(x)

    assert unpaged.feasible_BC == (64,), "the state_block pin did not shrink to BC=64"
    assert paged.last_selected_plan.BC == 64
    assert unpaged.last_execution_backend == paged.last_execution_backend == "cuda"
    bound = _reordering_bound(y0, unpaged, BC=64)
    assert (y0 - y_control).abs().max() <= bound
    assert (y0 - y1).abs().max() <= bound


def test_a_BC_flip_over_a_live_arena_is_invisible(monkeypatch):
    """Two `use_cache=True` chunks through ONE paged layer, with selection FORCED
    to `BC = 16` on chunk 1 and `BC = 64` on chunk 2 -- so chunk 2 launches at a
    different `BC` over the arena chunk 1 populated (and the continuation
    re-adopted). The arena object must be the SAME one (the rebuild key must not
    have rebuilt), its construction `BC` stays the chunk-1 value, and the output
    must match an unpaged layer driven through the IDENTICAL forced flip within
    the readout's own reordering envelope.

    There is no selector seam any more: `BC` is derived by
    `rola.planner.derive.derive_bc`, called from `rola.interface._derive_plan`, which
    both the plain (`rola_op`) and paged (`derive_plan`) arms share. Forcing a
    per-call scripted `BC` therefore means patching `derive_bc` at the name
    `rola.interface` imported it under, which reaches both arms the same way the old
    selector patch did.
    """
    import rola.interface as interface_module
    from rola.state import RoLAState

    torch.manual_seed(81)
    x = torch.randn(2, 48, 64, device="cuda")
    script = [16, 64]
    calls = {"i": 0}

    #: THE FORCING GOES THROUGH THE DERIVATION ITSELF: `derive_bc` is exactly the
    #: function whose job is to pick `BC`, so a test that wants a scripted decision
    #: replaces it outright rather than reaching around it.
    def scripted_derive_bc(topology, machine, *, BH, decay, pinned=None):
        forced = script[min(calls["i"], len(script) - 1)]
        calls["i"] += 1
        return forced

    monkeypatch.setattr(interface_module, "derive_bc", scripted_derive_bc)

    unpaged = _layer(paging=False, seed=82)
    paged = _layer(paging=True, seed=82)
    paged.load_state_dict(unpaged.state_dict())

    def run(layer):
        calls["i"] = 0
        cache = RoLAState()
        chunks = []
        with torch.no_grad():
            for lo, hi in ((0, 24), (24, 48)):
                out, _, _ = layer(x[:, lo:hi], past_key_values=cache, use_cache=True)
                chunks.append(out)
        return torch.cat(chunks, dim=1), cache

    y_unpaged, cache_u = run(unpaged)

    arenas = []
    orig_resolve = type(paged)._resolve_page_arena

    def spy_resolve(self, *a, **k):
        arena = orig_resolve(self, *a, **k)
        arenas.append(arena)
        return arena

    monkeypatch.setattr(type(paged), "_resolve_page_arena", spy_resolve)
    y_paged, cache_p = run(paged)

    assert paged.last_selected_plan.BC == 64
    assert len(arenas) == 2 and arenas[0] is arenas[1], (
        "the arena was rebuilt across the BC flip; the §4.5 rebuild-key omission "
        "is not what actually ran")
    assert arenas[0].BC == 16, "the arena's construction (policy) BC moved"

    bound = _reordering_bound(y_unpaged, unpaged, BC=64)
    assert (y_unpaged - y_paged).abs().max() <= bound, (
        "a BC flip over a live arena moved the output beyond the readout's own "
        "reordering envelope: the atom re-key's BC-invariance does not hold "
        "through the layer's arena reuse")
    assert torch.equal(cache_u[0].recurrent_state, cache_p[0].recurrent_state), (
        "the carried state diverged bitwise across the BC flip under paging")
