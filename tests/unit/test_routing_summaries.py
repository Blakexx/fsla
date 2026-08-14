"""Validation gate for the producer-epilogue summary pass (docs/internals/summaries.md).

Ground truth is an INDEPENDENT SCALAR WALK written from the spec wording with explicit
Python loops over ``bh``/tile/owner/level/digit/token. It shares nothing with
:mod:`rola.routing.summaries` -- not the outer-product factorization of the fixed
levels, not the bit packing, not the chunking. That independence is the point: a gate
built on the implementation's own helpers proves self-consistency and nothing else,
which is exactly how the shared-router defect survived its tests.

Symbols: ``L``/``T`` tokens; ``N`` leaves; ``D`` levels; ``b_l`` widths; ``BC`` leaves
per owner; ``BT`` token tile; ``BH`` = batch x heads; ``QPO = BC / 16`` quanta per owner.

CPU-only except the ``..._on_cuda`` synchronization gate, which is device-specific by
subject.
"""
from __future__ import annotations

import itertools
from math import prod

import pytest
import torch

from rola.planner.calibration.census import exact_routing_summaries
from rola.planner.schedule import StaticDensity, owner_geometry
from rola.routing.summaries import (
    MMA_K_QUANTUM,
    summarize_routing,
    summarize_routing_reference,
    summarize_routing_reference_device,
    unpack_leaf_bitmap,
)

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(),
                           reason="gates the summary pass's device synchronization budget")

#: I1 (the quanta invariant): every quanta-shaped quantity is an expression in `BC`
#: and `MMA_K_QUANTUM`, so the rows sweep the INSTANTIATED `BC` (16, 64) AND one
#: hypothetical `BC = 32` no kernel implements. A `BC`-derived literal -- a mask width,
#: a bin count, a page-slot count -- passes the instantiated rows and fails the fake one.
_BCS = [16, 64, 32]
_BC_IDS = ["BC16-instantiated", "BC64-instantiated", "BC32-FAKE"]

#: Topologies chosen so the OWNER RECTANGLE's boundary level `j` takes every value it
#: can: `j = 0` (no fixed levels), `j = 1`, `j = 2`. The fixed-level product is the one
#: part of the pass whose two derivations (outer product here, index gather in the
#: oracle) could disagree, and it is invisible at `j = 0`.
_TOPOLOGIES = [(8, 16), (4, 4, 8), (2, 4, 16)]
_TOPOLOGY_IDS = ["w8x16", "w4x4x8", "w2x4x16"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _masks(seed, widths, BH, T, kind):
    """Per-level ``[BH, T, b_l]`` fp32 ALLOCATIONS, the shape the producer emits.

    Liveness is EXACT nonzero-ness of the amplitude (spec §1.2) and is computed by the
    pass, not by the fixture: handing it a boolean mask would let the fixture make the
    liveness decision the pass is being gated on.
    """
    generator = torch.Generator().manual_seed(seed)
    out = []
    for level, width in enumerate(widths):
        amps = torch.rand((BH, T, width), generator=generator)
        if kind == "dense":
            live = torch.ones_like(amps, dtype=torch.bool)
        elif kind == "sparse":
            live = amps < 0.15
        elif kind == "very_sparse":
            live = amps < 0.03
        elif kind == "mixed":
            live = torch.ones_like(amps, dtype=torch.bool) if level % 2 == 0 else amps < 0.2
        elif kind == "empty_level":
            live = (torch.zeros_like(amps, dtype=torch.bool) if level == len(widths) - 1
                    else amps < 0.4)
        else:
            raise ValueError(kind)
        out.append(live.to(torch.float32) * (amps + 0.5))
    return tuple(out)


_KINDS = ["dense", "sparse", "very_sparse", "mixed", "empty_level"]


# ---------------------------------------------------------------------------
# The independent scalar walk (spec-derived; explicit loops; no shared helper)
# ---------------------------------------------------------------------------


def _oracle_geometry(widths, BC):
    """(j, c) from §3's wording, derived here rather than read off ``owner_geometry``."""
    D = len(widths)
    for j in range(D - 1, -1, -1):
        inner = prod(widths[j + 1:])
        if inner * widths[j] < BC:
            continue
        assert BC % inner == 0 and widths[j] % (BC // inner) == 0
        return j, BC // inner
    raise AssertionError("no admissible boundary level")


def _oracle_summaries(read, write, widths, BC, BT):
    """Every §4.2 field, by hand. Returns plain Python objects.

    ``K_hat`` is spelled out as §4.2 defines it -- a product over the levels the owner
    spans of the popcount of that level's per-tile UNION word, restricted to the owner's
    digits -- and the fixed levels enter as the 0/1 token-validity factor they are.
    """
    BH, T, _ = read[0].shape
    D = len(widths)
    N = prod(widths)
    j, c = _oracle_geometry(widths, BC)
    n_slices = widths[j] // c
    owners_total = N // BC
    n_tiles = (T + BT - 1) // BT
    QPO = BC // MMA_K_QUANTUM

    # Per-(side, level) occupancy: mean over (bh, token) of popcount / b_l.
    occupancy = []
    for side in (read, write):
        row = []
        for level in range(D):
            live = 0
            for bh in range(BH):
                for t in range(T):
                    for d in range(widths[level]):
                        if bool(side[level][bh, t, d]):
                            live += 1
            row.append(live / float(BH * T * widths[level]))
        occupancy.append(row)

    def tile_bits(side, bh, tile):
        lo, hi = tile * BT, min((tile + 1) * BT, T)
        return [[any(bool(side[level][bh, t, d]) for t in range(lo, hi))
                 for d in range(widths[level])] for level in range(D)]

    hist = [0] * (QPO + 1)
    n_candidate = 0
    for bh in range(BH):
        for tile in range(n_tiles):
            rb, wb = tile_bits(read, bh, tile), tile_bits(write, bh, tile)
            ub = [[rb[level][d] or wb[level][d] for d in range(widths[level])]
                  for level in range(D)]
            owner = 0
            for fixed in itertools.product(*[range(widths[level]) for level in range(j)]):
                for sl in range(n_slices):
                    this = owner
                    owner += 1
                    k_hat = 1
                    for level in range(j):
                        k_hat *= 1 if ub[level][fixed[level]] else 0
                    k_hat *= sum(1 for d in range(sl * c, (sl + 1) * c) if ub[j][d])
                    for level in range(j + 1, D):
                        k_hat *= sum(1 for d in range(widths[level]) if ub[level][d])
                    candidate = False
                    for sb in (rb, wb):
                        ok = all(sb[level][fixed[level]] for level in range(j))
                        ok = ok and any(sb[j][d] for d in range(sl * c, (sl + 1) * c))
                        candidate = candidate or ok
                    assert this < owners_total
                    if candidate:
                        n_candidate += 1
                        hist[-(-k_hat // MMA_K_QUANTUM)] += 1

    # §4.1's canonical per-leaf write bitmap: the union over tokens of the EXACT write
    # support, in canonical (row-major mixed-radix) leaf order.
    written = [[False] * N for _ in range(BH)]
    strides = [prod(widths[level + 1:]) for level in range(D)]
    for bh in range(BH):
        for t in range(T):
            for digits in itertools.product(*[range(w) for w in widths]):
                if all(bool(write[level][bh, t, digits[level]]) for level in range(D)):
                    written[bh][sum(digits[level] * strides[level] for level in range(D))] = True

    written_owners = sum(
        1 for bh in range(BH) for o in range(owners_total)
        if any(written[bh][o * BC + i] for i in range(BC)))

    n_pairs = BH * n_tiles * owners_total
    return dict(
        occupancy=occupancy,
        quanta_hist=[h / n_candidate if n_candidate else 0.0 for h in hist],
        candidate_frac=n_candidate / n_pairs if n_pairs else 0.0,
        active_owner_frac=written_owners / (BH * owners_total) if owners_total else 0.0,
        written=written)


# ---------------------------------------------------------------------------
# 1. The pass against the scalar walk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("widths", _TOPOLOGIES, ids=_TOPOLOGY_IDS)
@pytest.mark.parametrize("BC", _BCS, ids=_BC_IDS)
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("BT", [16, 32], ids=["BT16", "BT32"])
def test_the_summary_pass_equals_an_independent_scalar_walk(widths, BC, kind, BT):
    """§4.2, field by field, against explicit Python loops over the spec's definitions.

    ``T = 40`` is deliberately ragged at both ``BT`` (2.5 and 1.25 tiles), because the
    padded token tail is the one place the packed-word representation could invent
    support: a pad token is exactly zero at every level, so it must contribute no bit
    anywhere, and a fold that got that wrong would only show on a ragged ``L``.
    """
    BH, T = 2, 40
    read = _masks(1, widths, BH, T, kind)
    write = _masks(2, widths, BH, T, kind)
    geometry = owner_geometry(widths, BC)
    got = summarize_routing_reference(read, write, geometry, BT)
    want = _oracle_summaries(read, write, widths, BC, BT)

    assert got.occupancy.shape == (2, len(widths))
    for side in (0, 1):
        assert got.occupancy[side].tolist() == pytest.approx(want["occupancy"][side], abs=1e-6)
    assert got.quanta_hist.shape == (BC // MMA_K_QUANTUM + 1,)
    assert got.quanta_hist.tolist() == pytest.approx(want["quanta_hist"], abs=1e-6)
    assert got.candidate_frac == pytest.approx(want["candidate_frac"], abs=1e-12)
    assert got.active_owner_frac == pytest.approx(want["active_owner_frac"], abs=1e-12)

    bits = unpack_leaf_bitmap(got.written_leaf_bitmap, prod(widths))
    assert bits.tolist() == want["written"], (
        "§4.1's canonical per-leaf write bitmap disagrees with the union over tokens of "
        "the exact write support")


@pytest.mark.parametrize("widths", _TOPOLOGIES, ids=_TOPOLOGY_IDS)
@pytest.mark.parametrize("BC", _BCS, ids=_BC_IDS)
def test_the_published_histogram_bins_are_exhaustive_and_normalized(widths, BC):
    """``K_hat <= BC``, so ``QPO + 1`` bins hold every candidate pair with no clamp.

    The bound is structural, not empirical: the boundary level contributes at most
    ``slice_digits`` and every level below it at most its full width, whose product is
    ``BC`` by the owner rectangle. If a clamp were ever needed the bins would stop
    summing to 1 -- which is what this asserts, over a support built to make ``K_hat``
    as large as it can be (a fully dense routing saturates every factor).
    """
    BH, T = 2, 64
    dense = _masks(3, widths, BH, T, "dense")
    geometry = owner_geometry(widths, BC)
    for BT in (16, 32):
        summaries = summarize_routing_reference(dense, dense, geometry, BT)
        assert float(summaries.quanta_hist.sum()) == pytest.approx(1.0, abs=1e-6)
        # Dense both sides: every pair is a candidate and every K_hat is exactly BC.
        assert summaries.candidate_frac == pytest.approx(1.0)
        assert float(summaries.quanta_hist[-1]) == pytest.approx(1.0, abs=1e-6)
        assert summaries.active_owner_frac == pytest.approx(1.0)


def test_the_atom_condensation_is_derived_from_the_canonical_bitmap():
    """§4.1: atom granularity is COMPUTED, never stored as a second truth."""
    widths, BC, BH, T = (8, 16), 64, 2, 40
    write = _masks(4, widths, BH, T, "sparse")
    geometry = owner_geometry(widths, BC)
    summaries = summarize_routing_reference(write, write, geometry, 16)
    N = prod(widths)
    bits = unpack_leaf_bitmap(summaries.written_leaf_bitmap, N)
    atoms = summaries.atom_bitmap(N)
    assert atoms.shape == (BH, N // MMA_K_QUANTUM)
    assert atoms.tolist() == bits.view(BH, N // MMA_K_QUANTUM,
                                       MMA_K_QUANTUM).any(dim=-1).tolist()
    with pytest.raises(ValueError, match="k-quantum"):
        summaries.atom_bitmap(N - 1)


# ---------------------------------------------------------------------------
# 2. Bound parity with the EXACT census (the auditor's twin)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("widths", _TOPOLOGIES, ids=_TOPOLOGY_IDS)
@pytest.mark.parametrize("BC", _BCS, ids=_BC_IDS)
@pytest.mark.parametrize("kind", _KINDS)
def test_the_factorized_summaries_bound_the_exact_census_and_match_it_elsewhere(
        widths, BC, kind):
    """The one-sided property, plus the four exact agreements.

    ``K_hat = prod_l û_l >= K`` pairwise (union of products SUBSET product of unions),
    so ``ceil(K_hat/16) >= ceil(K/16)`` pairwise and the BOUND histogram's mean sits at
    or above the exact one. That inequality is the whole licence for replacing the
    census with the factorized pass, and its direction is the safe one: a reader can
    only under-state how sparse a launch is.

    Candidacy, occupancy, the active-owner fraction and the per-leaf bitmap are NOT
    bounds -- they are the same quantity by two derivations, so they must agree
    exactly. A cell where the histogram is loose but these four still match is the
    signature of a correct bound; a cell where one of these four moves is a defect in
    one of the two derivations.
    """
    BH, T, BT = 2, 40, 16
    read = _masks(5, widths, BH, T, kind)
    write = _masks(6, widths, BH, T, kind)
    geometry = owner_geometry(widths, BC)
    bound = summarize_routing_reference(read, write, geometry, BT)
    exact = exact_routing_summaries(read, write, geometry, BT)

    assert bound.candidate_frac == pytest.approx(exact.candidate_frac, abs=1e-12)
    assert bound.active_owner_frac == pytest.approx(exact.active_owner_frac, abs=1e-12)
    assert torch.allclose(bound.occupancy, exact.occupancy, atol=1e-6)
    assert torch.equal(bound.written_leaf_bitmap, exact.written_leaf_bitmap)

    bins = torch.arange(BC // MMA_K_QUANTUM + 1, dtype=torch.float32)
    assert float((bins * bound.quanta_hist).sum()) >= float((bins * exact.quanta_hist).sum()) - 1e-6, (
        "the factorized histogram fell BELOW the exact one; K_hat is documented as an "
        "upper bound on K and every reader relies on that direction")
    assert float(bound.quanta_hist.sum()) == pytest.approx(
        float(exact.quanta_hist.sum()), abs=1e-6)


def test_the_bound_is_actually_loose_somewhere_so_the_gate_is_not_vacuous():
    """A cell where ``K_hat > K``, so the inequality above is a real one-sided test.

    At ``D = 1`` the factorization is the identity and the bound is tight; the gate
    would then pass on any implementation at all. This pins a ``D = 2`` cell whose
    per-level unions genuinely over-count the union of products.
    """
    widths, BC, BH, T, BT = (8, 16), 64, 2, 64, 32
    read = _masks(7, widths, BH, T, "sparse")
    write = _masks(8, widths, BH, T, "very_sparse")
    geometry = owner_geometry(widths, BC)
    bins = torch.arange(BC // MMA_K_QUANTUM + 1, dtype=torch.float32)
    bound = float((bins * summarize_routing_reference(read, write, geometry, BT).quanta_hist).sum())
    exact = float((bins * exact_routing_summaries(read, write, geometry, BT).quanta_hist).sum())
    assert bound > exact, (
        "no cell in this file exercises a LOOSE bound, so the one-sided gate above is "
        "vacuous and would pass on an implementation that computed K exactly")


# ---------------------------------------------------------------------------
# 3. The §4.2 underflow assertion
# ---------------------------------------------------------------------------


def _softmax_support(width, T, gain, seed):
    """A softmax routing allocation at a logit GAIN, as the layer's producer emits it.

    Softmax has no exact structural zero -- that is exactly what the activation
    registry's ``exact_zeros`` column says and why the read side is declared dense.
    Float underflow does not care: past some gain an ``exp`` rounds to exactly 0.
    """
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn((1, T, width), generator=generator) * gain
    return torch.softmax(logits, dim=-1) != 0


def test_the_underflow_assertion_fires_only_when_a_declared_dense_cell_has_realized_zeros():
    """§4.2's debug assert, on the mechanism it was MEASURED on (Gate 2 item 3e).

    The measured boundary is a parity-shaped layer at ``logit_gain = 32.0`` (realized
    read-side zero fractions 0.002056 / 0.001883, safe at the bench default 8.0). That
    cell is a bench measurement and this is a source-only gate, so what is asserted
    here is the same MECHANISM at toy scale: a softmax whose gain is large enough to
    underflow breaks a declared-dense cell, and the assertion catches it.
    """
    widths, BC, T = (8, 16), 16, 256
    geometry = owner_geometry(widths, BC)
    declared = StaticDensity(read=(True,) * len(widths), write=(True,) * len(widths))

    calm = tuple(_softmax_support(w, T, 1.0, 11 + i) for i, w in enumerate(widths))
    assert all(bool(mask.all()) for mask in calm), "the calm arm already underflowed"
    summarize_routing_reference(calm, calm, geometry, 16, density=declared, debug=True)

    hot = tuple(_softmax_support(w, T, 64.0, 11 + i) for i, w in enumerate(widths))
    assert not all(bool(mask.all()) for mask in hot), (
        "the hot arm did not underflow, so this gate is not exercising the assertion")
    with pytest.raises(AssertionError, match="FLOAT UNDERFLOW"):
        summarize_routing_reference(hot, hot, geometry, 16, density=declared, debug=True)

    # OFF by default: the check costs a host read and must not lengthen every forward.
    summarize_routing_reference(hot, hot, geometry, 16, density=declared, debug=False)
    summarize_routing_reference(hot, hot, geometry, 16, density=declared)


# ---------------------------------------------------------------------------
# 4. Synchronization budget
# ---------------------------------------------------------------------------


@_CUDA
@pytest.mark.parametrize("BC", _BCS, ids=_BC_IDS)
def test_the_summary_pass_issues_no_host_synchronization_on_cuda(BC):
    """§4.2: device-side reduction, ONE D2H transfer, zero host syncs.

    The transfer is excluded BY CONSTRUCTION -- it lives in
    :func:`summarize_routing`, not in :func:`summarize_routing_device` -- so this can
    run the whole device pass under ``set_sync_debug_mode("error")`` and require it to
    complete clean. It is the same protocol the bit-word census's gate uses, and it is
    why the two functions are split at all rather than being one.

    The parametrization is not decoration: ``QPO`` sets the histogram's bin count, so a
    ``BC``-derived literal in the binning would show up here as well as in the value
    gates above.
    """
    widths, BH, T = (8, 16), 2, 64
    read = tuple(m.cuda() for m in _masks(9, widths, BH, T, "sparse"))
    write = tuple(m.cuda() for m in _masks(10, widths, BH, T, "sparse"))
    geometry = owner_geometry(widths, BC)
    summarize_routing_reference_device(read, write, geometry, 16)      # warm any lazy init
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        out = summarize_routing_reference_device(read, write, geometry, 16)
        assert out.n_candidate.device.type == "cuda"
        assert out.written_leaf_bitmap.device.type == "cuda"
        assert out.quanta_hist.device.type == "cuda"
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_the_pass_refuses_a_BT_its_representation_cannot_carry():
    """One token per BIT of one ``int32``; a wider ``BT`` is a spec change, not a branch."""
    widths, BC = (8, 16), 16
    masks = _masks(12, widths, (BH := 1), 64, "sparse")
    with pytest.raises(ValueError, match="one token bit each"):
        summarize_routing_reference(masks, masks, owner_geometry(widths, BC), 64)
    assert BH == 1


def test_the_pass_refuses_a_BC_that_is_not_a_whole_number_of_atoms():
    """§1b: the page/quantum atom is 16 leaves and every legal ``BC`` is a whole number
    of them by construction, so a ``BC`` that is not is a caller defect."""
    widths = (8, 16)
    geometry = owner_geometry(widths, 8)
    masks = _masks(13, widths, 1, 32, "sparse")
    with pytest.raises(ValueError, match="whole number of MMA_K_QUANTUM"):
        summarize_routing_reference(masks, masks, geometry, 16)


# ---------------------------------------------------------------------------
# 5. THE KERNEL, against the reference above (spec amendment, 2026-08-02)
# ---------------------------------------------------------------------------

def _extension_available():
    try:
        from rola.ops._ext import extension
        extension()
    except Exception:
        return False
    return True


_KERNEL = pytest.mark.skipif(
    not (torch.cuda.is_available() and _extension_available()),
    reason="the summary pass is an AOT CUDA kernel; this gate needs the built extension")


@_KERNEL
@pytest.mark.parametrize("widths", _TOPOLOGIES, ids=_TOPOLOGY_IDS)
@pytest.mark.parametrize("BC", _BCS, ids=_BC_IDS)
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("BT", [16, 32], ids=["BT16", "BT32"])
def test_the_summary_kernel_equals_the_enumerated_reference(widths, BC, kind, BT):
    """The kernel against the torch pass, field by field, on the SAME fp32 allocations.

    The reference is itself gated against an independent scalar walk above, so this is a
    two-link chain down to a from-the-spec definition rather than a self-consistency
    check. ``T = 40`` is ragged at both ``BT``, which is where a token-word pass can
    invent support if its padded tail is not inert.
    """
    BH, T = 2, 40
    read = tuple(t.cuda() for t in _masks(1, widths, BH, T, kind))
    write = tuple(t.cuda() for t in _masks(2, widths, BH, T, kind))
    geometry = owner_geometry(widths, BC)
    got = summarize_routing(read, write, geometry, BT)
    want = summarize_routing_reference(read, write, geometry, BT)

    assert torch.allclose(got.occupancy, want.occupancy, atol=1e-6)
    assert torch.allclose(got.quanta_hist, want.quanta_hist, atol=1e-6)
    assert got.candidate_frac == pytest.approx(want.candidate_frac, abs=1e-12)
    assert got.active_owner_frac == pytest.approx(want.active_owner_frac, abs=1e-12)
    assert torch.equal(got.written_leaf_bitmap, want.written_leaf_bitmap), (
        "§4.1's canonical per-leaf write bitmap differs between the kernel and the "
        "reference; the warp-ballot packing or the write-candidate skip is wrong")


@_KERNEL
def test_the_summary_kernel_issues_no_host_synchronization():
    """One launch, one D2H, zero host syncs -- the D2H excluded BY CONSTRUCTION.

    The transfer lives in :func:`summarize_routing`'s epilogue, so what runs under
    ``set_sync_debug_mode("error")`` here is the output allocation and the launch, which
    must complete clean.
    """
    from rola.ops._ext import extension

    widths, BC, BH, T = (8, 16), 64, 2, 64
    read = [t.cuda() for t in _masks(9, widths, BH, T, "sparse")]
    write = [t.cuda() for t in _masks(10, widths, BH, T, "sparse")]
    geometry = owner_geometry(widths, BC)
    summarize_routing(tuple(read), tuple(write), geometry, 16)   # warm any lazy init
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        N = prod(widths)
        z = lambda shape, dt: torch.zeros(shape, dtype=dt, device="cuda")   # noqa: E731
        extension().rola_routing_summaries(
            read, write, list(widths), 16, BC,
            geometry.first_spanned_level, geometry.slice_digits, geometry.n_slices,
            z(2 * len(widths), torch.int64), z(BC // MMA_K_QUANTUM + 1, torch.int64),
            z(1, torch.int64), z(1, torch.int64),
            z((BH, (N + 31) // 32), torch.int32),
            z((BH, (N // BC + 31) // 32), torch.int32))
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_the_runtime_pass_refuses_to_degrade_to_its_own_reference():
    """NO FALLBACK: an unbuilt extension or a CPU tensor must not silently re-route.

    The reference computes the same five fields, which is exactly why this has to be
    asserted rather than assumed: the cheap thing to do would be to call it, and that
    would make "which code decided this plan" depend on whether somebody ran the build.
    """
    widths, BC = (8, 16), 16
    masks = _masks(21, widths, 1, 32, "sparse")
    with pytest.raises(RuntimeError, match="no CPU path"):
        summarize_routing(masks, masks, owner_geometry(widths, BC), 16)
