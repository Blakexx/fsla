"""Validation gate for the V3 schedule builder (``ROLA_V3_SPEC.md`` §3, §4, §6).

Ground truth throughout is an **independent fp64/exact-nonzero enumeration oracle**
written from the spec wording with explicit Python loops over bh/level/digit/tile/
token. The oracle reuses nothing from ``schedule_v3.py`` except the public
``unpack_bits`` (never ``_pack_bits``/``_tile_live``/``_sort_owners``/
``owner_geometry``'s decomposition helpers, except in the one test group whose
subject IS ``owner_geometry`` itself).

CPU-only, with ONE deliberate exception: the support-counted statistics pass reaches
`Tensor.scatter_reduce_(reduce="amin"/"amax")` on `int16`, `nonzero()` and
`repeat_interleave`, which are backend-divergent surfaces, and the planner is the one
thing in this file that runs on the device the routing lives on. Those two tests are
`@_CUDA`-marked and named `..._on_cuda`; everything else is CPU and builds CPU tensors.
"""

from __future__ import annotations

import inspect
import itertools
import math
from math import prod

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from rola.planner.schedule import (
    _STATS_OWNER_CHUNK_ELEMS,
    CP_ASYNC_ROW_BYTES,
    DEFAULT_SM_COUNT,
    DEFAULT_WRITE_COST_WEIGHT,
    MMA_K_QUANTUM,
    LevelSupportStats,
    Plan,
    StaticDensity,
    TileUnionStats,
    _tile_union_stats_multi,
    _tile_union_stats_multi_enumerated,
    build_clock_stream,
    build_schedule,
    leaf_rate_dials,
    owner_geometry,
    permuted_topology,
    plan_and_schedule,
    plan_cost_per_token,
    plan_waves,
    rank_spans,
    routing_views,
    select_plan,
    select_plan_axes,
    span_score,
    spanned_moduli,
    tile_union_stats,
    unpack_bits,
)
from rola.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    LeafMassDecay,
    LevelConfig,
    SoftmaxActivation,
    Topology,
)

#: The two `..._on_cuda` gates below, and nothing else in this file, run on device --
#: the support-counted statistics pass reaches backend-divergent reductions and this
#: file is where its bitwise gate lives.
_CUDA = pytest.mark.skipif(not torch.cuda.is_available(),
                           reason="gates the planner's device-side reduction path")

# ---------------------------------------------------------------------------
# Shared construction helpers (NOT part of the oracle -- plain fixture plumbing)
# ---------------------------------------------------------------------------


def _routing():
    return IndependentRouting(
        tied=False, read_activation=SoftmaxActivation(), write_activation=SoftmaxActivation())


def _toy_occupancy(blocks=4):
    """A SYNTHETIC occupancy oracle for the CPU-only gates on toy geometries.

    The two gates below plan `(BC, d_v)` the consumer does not instantiate -- `BC`
    in {1..4}, `d_v = 4` -- because what they are about is view sharing and the
    default numeric format, not residency. As of item 6b `resolve_occupancy`
    REFUSES such a geometry instead of diverting it to an analytic mirror (the
    mirror modelled the deleted consumer and was measured inverting the bound it
    claimed), so these gates now supply the machine fact themselves. That is what
    the injection interface is documented to be for, and it makes the dependency
    explicit rather than ambient: a constant oracle is a stated assumption, a
    silent fallback was not.
    """
    def query(BT, BC, d_v, global_norm, decay, mma_precision, compact,
              gram_cols, ring_slots):
        return blocks

    return query


def _topology(widths, d_v=4):
    return Topology(
        levels=tuple(LevelConfig(width=w, routing=_routing()) for w in widths), d_v=d_v)


def _plan(widths, BT, BC, *, d_v=4, decay=False,
          write_cost_weight=DEFAULT_WRITE_COST_WEIGHT, warp_tile=(1, 1)):
    topo = _topology(widths, d_v=d_v)
    return Plan(
        topology=topo, normalization="raw", BT=BT, BC=BC, warp_tile=warp_tile,
        decay=decay, write_cost_weight=write_cost_weight)


def _admissible_bcs(widths):
    """All BC in [1, N] that ``owner_geometry`` accepts -- brute force over divisors."""
    N = math.prod(widths)
    out = []
    for bc in range(1, N + 1):
        if N % bc:
            continue
        try:
            owner_geometry(tuple(widths), bc)
        except ValueError:
            continue
        out.append(bc)
    return out


def _gen_topology_tensors(seed, widths, B, T, H, scenario):
    """Build [B,T,H,width_l] read/write tensors per ``scenario`` (density kind)."""
    g = torch.Generator().manual_seed(seed)
    D = len(widths)
    read_levels, write_levels = [], []

    def _dense(shape):
        return torch.rand(shape, generator=g) + 0.01

    def _sparse(shape, p=0.1):
        mask = torch.rand(shape, generator=g) < p
        vals = torch.rand(shape, generator=g) + 0.01
        return vals * mask

    for lvl, w in enumerate(widths):
        shape = (B, T, H, w)
        if scenario == "dense":
            r, wv = _dense(shape), _dense(shape)
        elif scenario == "sparse":
            r, wv = _sparse(shape), _sparse(shape)
        elif scenario == "union":
            mask = torch.rand(shape, generator=g) < 0.15
            r = (torch.rand(shape, generator=g) + 0.01) * mask
            wv = (torch.rand(shape, generator=g) + 0.01) * mask
        elif scenario == "mixed":
            if lvl % 2 == 0:
                r, wv = _dense(shape), _dense(shape)
            else:
                r, wv = _sparse(shape), _sparse(shape)
        elif scenario == "zero_level":
            if lvl == D - 1:
                r, wv = torch.zeros(shape), torch.zeros(shape)
            else:
                r, wv = _dense(shape), _dense(shape)
        else:
            raise ValueError(scenario)
        read_levels.append(r)
        write_levels.append(wv)
    g_read = torch.rand((B, T, H), generator=g) + 0.01
    g_write = torch.rand((B, T, H), generator=g) + 0.01
    return tuple(read_levels), tuple(write_levels), g_read, g_write


# ---------------------------------------------------------------------------
# Independent oracle primitives (explicit loops; spec-derived, not implementation-derived)
# ---------------------------------------------------------------------------


def _oracle_fold(tensor):
    """``[B,T,H,W] -> [BH,T,W]``, ``bh = b*H+h``, by hand (not ``_fold_bth``)."""
    B, T, H, W = tensor.shape
    out = torch.zeros((B * H, T, W), dtype=tensor.dtype)
    for b in range(B):
        for h in range(H):
            out[b * H + h] = tensor[b, :, h, :]
    return out


def _oracle_tile_live(folded, BT, n_tiles):
    """``[BH,T,W] -> [BH,n_tiles,W]`` bool: exact nonzero-ness anywhere in the tile."""
    BH, T, W = folded.shape
    live = torch.zeros((BH, n_tiles, W), dtype=torch.bool)
    for bh in range(BH):
        for k in range(n_tiles):
            lo, hi = k * BT, min((k + 1) * BT, T)
            for d in range(W):
                any_nonzero = False
                for t in range(lo, hi):
                    if folded[bh, t, d].item() != 0.0:
                        any_nonzero = True
                        break
                live[bh, k, d] = any_nonzero
    return live


def _independent_owner_geometry(widths, BC):
    """(j, c) derived from spec §3 wording, independent of ``owner_geometry``."""
    D = len(widths)
    for j in range(D - 1, -1, -1):
        inner = 1
        for w in widths[j + 1:]:
            inner *= w
        span = inner * widths[j]
        if span < BC:
            continue
        if BC % inner != 0:
            raise AssertionError("misaligned BC reached the owner oracle")
        c = BC // inner
        if widths[j] % c != 0:
            raise AssertionError("misaligned BC reached the owner oracle")
        return j, c
    raise AssertionError("no admissible j found")


def _oracle_slice_union(live, j, c, n_slices):
    """``[BH,n_tiles,width_j] -> [BH,n_slices,n_tiles]`` bool OR over each c-run."""
    BH, n_tiles, width_j = live.shape
    out = torch.zeros((BH, n_slices, n_tiles), dtype=torch.bool)
    for bh in range(BH):
        for sl in range(n_slices):
            for k in range(n_tiles):
                any_live = False
                for d in range(sl * c, (sl + 1) * c):
                    if live[bh, k, d]:
                        any_live = True
                        break
                out[bh, sl, k] = any_live
    return out


def _oracle_owners(read_live, write_live, widths, BC, n_tiles, write_cost_weight):
    """Active (bh, owner) pairs with cost, in emission (pre-sort) order.

    ``read_live``/``write_live``: per-level list of ``[BH,n_tiles,width_l]`` bool
    tensors (already the tile-liveness oracle output). Brute force over every
    (fixed digit tuple, slice index) -- no pruning, matching the spec definition
    exactly (an owner is active iff it has >=1 candidate tile on either side).
    """
    BH = read_live[0].shape[0]
    j, c = _independent_owner_geometry(widths, BC)
    n_slices = widths[j] // c
    results = []  # (bh, owner_id, cost), in emission order
    for bh in range(BH):
        owner_id = 0
        fixed_ranges = [range(widths[level]) for level in range(j)]
        for fixed_digits in itertools.product(*fixed_ranges):
            for slice_index in range(n_slices):
                this_owner = owner_id
                owner_id += 1
                r_count = 0
                w_count = 0
                for k in range(n_tiles):
                    read_ok = True
                    write_ok = True
                    for level, digit in enumerate(fixed_digits):
                        if not bool(read_live[level][bh, k, digit]):
                            read_ok = False
                        if not bool(write_live[level][bh, k, digit]):
                            write_ok = False
                    if read_ok:
                        read_ok = any(
                            bool(read_live[j][bh, k, d]) for d in range(slice_index * c, (slice_index + 1) * c))
                    if write_ok:
                        write_ok = any(
                            bool(write_live[j][bh, k, d]) for d in range(slice_index * c, (slice_index + 1) * c))
                    if read_ok:
                        r_count += 1
                    if write_ok:
                        w_count += 1
                if r_count == 0 and w_count == 0:
                    continue
                cost = r_count + write_cost_weight * w_count
                results.append((bh, this_owner, cost))
    order = sorted(range(len(results)), key=lambda i: -results[i][2])  # stable
    return [results[i] for i in order]


def _oracle_tile_major(folded, BT, n_tiles, pitch):
    """``[BH,T,W] -> [BH,n_tiles,BT,pitch]``, zero-padded, independent construction."""
    BH, T, width = folded.shape
    flat = torch.zeros((BH, n_tiles * BT, pitch), dtype=folded.dtype)
    for bh in range(BH):
        for t in range(T):
            for d in range(width):
                flat[bh, t, d] = folded[bh, t, d]
    return flat.view(BH, n_tiles, BT, pitch)


# ---------------------------------------------------------------------------
# Randomized topology matrix
# ---------------------------------------------------------------------------

_TOPOLOGIES = [
    (5,),           # D=1, non-power-of-two
    (2, 3),         # D=2, mixed
    (3, 5),         # D=2, non-power
    (2, 3, 4),      # D=3, mixed
    (2, 2, 2, 2),   # D=4, power-of-two
]
_DENSITIES = ["dense", "sparse", "union", "mixed", "zero_level"]


def _t_variants(BT):
    return {
        "exact": 2 * BT,
        "plus_one": 2 * BT + 1,
        "minus_one": 2 * BT - 1,
        "small": max(1, BT // 2 - 3),
    }


def _combo_id(widths, BT, t_kind, density):
    return f"w{'x'.join(map(str, widths))}-BT{BT}-{t_kind}-{density}"


def _build_combo(widths, BT, t_kind, density, seed=0):
    T = _t_variants(BT)[t_kind]
    BC = _admissible_bcs(widths)[0]
    B, H = 2, 2
    read_levels, write_levels, g_read, g_write = _gen_topology_tensors(seed, widths, B, T, H, density)
    plan = _plan(widths, BT, BC)
    n_tiles = (T + BT - 1) // BT
    return dict(
        widths=widths, BT=BT, T=T, BC=BC, B=B, H=H, n_tiles=n_tiles,
        read_levels=read_levels, write_levels=write_levels, g_read=g_read, g_write=g_write,
        plan=plan)


_MATRIX = [
    (widths, BT, t_kind, density)
    for widths in _TOPOLOGIES
    for BT in (16, 32)
    for t_kind in ("exact", "plus_one", "minus_one", "small")
    for density in _DENSITIES
]
_MATRIX_IDS = [_combo_id(*c) for c in _MATRIX]


# ---------------------------------------------------------------------------
# 1. Owner geometry vs brute force (spec §3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("widths", _TOPOLOGIES, ids=lambda w: "x".join(map(str, w)))
def test_owner_geometry_matches_brute_force_leaf_enumeration(widths):
    """§3: an aligned block is exactly one rectangle; owner*BC is its base.

    For every admissible BC, independently enumerate the leaf ids of the
    rectangle ``{fixed digits} x {slice's c digits of level j} x {full span
    below}`` two ways -- via ``owner*BC + offset`` and via nested digit loops --
    and check the sets agree, for every owner. Also checks ``owner_digits``/
    ``owner_of`` are mutual inverses over the whole owner space.
    """
    D = len(widths)
    strides = tuple(math.prod(widths[i + 1:]) for i in range(D))

    def leaf_id(digits):
        return sum(d * s for d, s in zip(digits, strides))

    for BC in _admissible_bcs(widths):
        geom = owner_geometry(tuple(widths), BC)
        j, c = geom.first_spanned_level, geom.slice_digits
        assert geom.owners_total == math.prod(widths) // BC

        for owner in range(geom.owners_total):
            fixed_digits, slice_index = geom.owner_digits(owner)
            assert len(fixed_digits) == j

            # Rectangle enumerated from (fixed digits, slice) via nested loops.
            rect_leaves = set()
            below_ranges = [range(widths[level]) for level in range(j + 1, D)]
            for slice_digit in range(slice_index * c, (slice_index + 1) * c):
                for below in itertools.product(*below_ranges):
                    digits = tuple(fixed_digits) + (slice_digit,) + below
                    rect_leaves.add(leaf_id(digits))

            base = geom.leaf_base(owner)
            contiguous_leaves = set(range(base, base + BC))
            assert rect_leaves == contiguous_leaves

            # Inverse round trip both directions.
            assert geom.owner_of(fixed_digits, slice_index) == owner
        for owner in range(min(geom.owners_total, 200)):
            fixed_digits, slice_index = geom.owner_digits(owner)
            round_trip = geom.owner_of(fixed_digits, slice_index)
            assert round_trip == owner


# ---------------------------------------------------------------------------
# 2. Inadmissible BC raises loudly
# ---------------------------------------------------------------------------


def test_bc_not_dividing_n_raises():
    with pytest.raises(ValueError, match="must divide N"):
        owner_geometry((3, 5), 4)


def test_bc_exceeding_capacity_raises():
    """``BC > N`` is checked in ``owner_geometry`` *after* ``N % BC != 0``, but any
    ``BC > N`` already fails the divisibility check first (a divisor of N is never
    larger than N for N >= 1), so the dedicated ">capacity" message is dead code
    reachable only via that same divisibility failure. Recorded in the findings
    doc; not fixed here (see "dead code, not a correctness bug" note) -- this test
    pins down the actually-observed behavior rather than the unreachable message.
    """
    with pytest.raises(ValueError, match="must divide N"):
        owner_geometry((3, 5), 16)


@pytest.mark.parametrize("bc", [0, -1, -5])
def test_bc_below_one_raises(bc):
    with pytest.raises(ValueError, match="BC must be >= 1"):
        owner_geometry((3, 5), bc)


def test_bc_misaligned_inner_extent_raises():
    """widths=(2,3,5): the algorithm settles on j=1 (span=15>=6), inner=5, and
    BC=6 is not a multiple of 5 -- even though 6 divides N=30."""
    widths = (2, 3, 5)
    assert math.prod(widths) % 6 == 0
    with pytest.raises(ValueError, match="not a multiple of the spanned inner extent"):
        owner_geometry(widths, 6)


def test_bc_misaligned_slice_digits_raises():
    """widths=(2,3,5): j=1, inner=5, BC=10 -> c=2, but widths[1]=3 is not a
    multiple of 2 -- BC=10 divides N=30 and is a multiple of inner=5."""
    widths = (2, 3, 5)
    assert math.prod(widths) % 10 == 0
    with pytest.raises(ValueError, match="not an aligned sub-range"):
        owner_geometry(widths, 10)


# ---------------------------------------------------------------------------
# 3. Tile-liveness bitmaps equal the oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("combo", _MATRIX, ids=_MATRIX_IDS)
def test_tile_bitmaps_match_oracle(combo):
    """§4: per-(level,digit) tile-liveness bitmap == exact-nonzero-in-tile oracle."""
    widths, BT, t_kind, density = combo
    c = _build_combo(widths, BT, t_kind, density)
    schedule = build_schedule(c["read_levels"], c["write_levels"], c["g_read"], c["g_write"], c["plan"])

    unpacked_read = unpack_bits(schedule.read_tile_bits, schedule.n_tiles)
    unpacked_write = unpack_bits(schedule.write_tile_bits, schedule.n_tiles)

    for level, width in enumerate(widths):
        off = schedule.level_offsets[level]
        oracle_r = _oracle_tile_live(_oracle_fold(c["read_levels"][level]), BT, c["n_tiles"])
        oracle_w = _oracle_tile_live(_oracle_fold(c["write_levels"][level]), BT, c["n_tiles"])
        got_r = unpacked_read[:, off:off + width, :].permute(0, 2, 1)
        got_w = unpacked_write[:, off:off + width, :].permute(0, 2, 1)
        assert torch.equal(got_r, oracle_r), f"read bitmap mismatch level={level}"
        assert torch.equal(got_w, oracle_w), f"write bitmap mismatch level={level}"


# ---------------------------------------------------------------------------
# 4. Slice unions equal the oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("combo", _MATRIX, ids=_MATRIX_IDS)
def test_slice_union_matches_oracle(combo):
    """§4: the boundary-level slice-union bitmap == OR over each c-run, oracle."""
    widths, BT, t_kind, density = combo
    c = _build_combo(widths, BT, t_kind, density)
    schedule = build_schedule(c["read_levels"], c["write_levels"], c["g_read"], c["g_write"], c["plan"])

    j, cw = _independent_owner_geometry(widths, c["BC"])
    n_slices = widths[j] // cw

    oracle_r_live = _oracle_tile_live(_oracle_fold(c["read_levels"][j]), BT, c["n_tiles"])
    oracle_w_live = _oracle_tile_live(_oracle_fold(c["write_levels"][j]), BT, c["n_tiles"])
    oracle_r_union = _oracle_slice_union(oracle_r_live, j, cw, n_slices)
    oracle_w_union = _oracle_slice_union(oracle_w_live, j, cw, n_slices)

    got_r = unpack_bits(schedule.read_slice_union, c["n_tiles"])
    got_w = unpack_bits(schedule.write_slice_union, c["n_tiles"])
    assert torch.equal(got_r, oracle_r_union)
    assert torch.equal(got_w, oracle_w_union)


# ---------------------------------------------------------------------------
# 5. Owner list and order equal the oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("combo", _MATRIX, ids=_MATRIX_IDS)
def test_owners_match_oracle_set_and_order(combo):
    """§3/§4: active owners, in descending-cost order (stable ties), == oracle."""
    widths, BT, t_kind, density = combo
    c = _build_combo(widths, BT, t_kind, density)
    schedule = build_schedule(c["read_levels"], c["write_levels"], c["g_read"], c["g_write"], c["plan"])

    read_live = [_oracle_tile_live(_oracle_fold(c["read_levels"][lvl]), BT, c["n_tiles"]) for lvl in range(len(widths))]
    write_live = [_oracle_tile_live(_oracle_fold(c["write_levels"][lvl]), BT, c["n_tiles"]) for lvl in range(len(widths))]

    oracle = _oracle_owners(read_live, write_live, widths, c["BC"], c["n_tiles"], c["plan"].write_cost_weight)
    oracle_pairs = [(bh, owner) for bh, owner, _cost in oracle]

    got_pairs = [tuple(row) for row in schedule.owners.tolist()]
    assert got_pairs == oracle_pairs, (got_pairs, oracle_pairs)

    oracle_costs = torch.tensor([cost for _b, _o, cost in oracle], dtype=torch.float32)
    torch.testing.assert_close(schedule.owner_costs, oracle_costs, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. Amplitude stream fidelity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("combo", _MATRIX, ids=_MATRIX_IDS)
def test_amplitude_stream_fidelity(combo):
    """§5 "Staging": amps reproduce the input exactly, zero-padded, cp.async-shaped."""
    widths, BT, t_kind, density = combo
    c = _build_combo(widths, BT, t_kind, density)
    schedule = build_schedule(c["read_levels"], c["write_levels"], c["g_read"], c["g_write"], c["plan"])
    itemsize = torch.empty((), dtype=c["plan"].amp_dtype).element_size()

    for level, width in enumerate(widths):
        for amps, src in ((schedule.read_amps[level], c["read_levels"][level]),
                           (schedule.write_amps[level], c["write_levels"][level])):
            pitch = amps.shape[-1]
            assert (BT * pitch * itemsize) % CP_ASYNC_ROW_BYTES == 0, (BT, pitch, itemsize)
            assert amps.is_contiguous()
            assert amps.stride(2) == pitch  # [BH,n_tiles,BT,pitch]: BT-axis stride == pitch

            folded = _oracle_fold(src)
            expected = _oracle_tile_major(folded, BT, c["n_tiles"], pitch)
            assert torch.equal(amps, expected)
            # Explicit padding check restated (redundant with equality above, but
            # pins the two padding regimes down by name per the task spec).
            assert torch.equal(amps[:, :, :, width:], torch.zeros_like(amps[:, :, :, width:]))
            if c["T"] < c["n_tiles"] * BT:
                last_tile, last_pos = divmod(c["T"], BT)
                assert torch.equal(
                    amps[:, last_tile, last_pos:, :], torch.zeros_like(amps[:, last_tile, last_pos:, :]))


# ---------------------------------------------------------------------------
# 7. P5.20.5 finding 2 (BINDING): no write_levels_stored, ever
# ---------------------------------------------------------------------------


def test_write_liveness_uses_live_deposit_not_a_clock():
    """`GRADIENT_PROOFS_WOLFRAM.md` P5.20.5 finding 2: write liveness must come
    from the live deposit ``write_levels``, never a stored/clock-summary tensor
    that could have been flushed to zero while the deposit itself was nonzero.

    The strongest test of this is structural: ``build_schedule`` must have NO
    parameter that could carry such a clock-keyed summary at all, so the wrong
    tensor is not merely discouraged but unreachable -- confirmed here via
    ``inspect.signature`` rather than by trying to sneak a stored-zero fixture
    past a parameter that doesn't exist. We additionally build a fixture where a
    write allocation is small-but-nonzero at one (t,d) and assert that tile is
    live in ``write_tile_bits`` -- the behavioral half of the same guarantee.

    **The ``views`` parameter (planner increment, 2026-07-30) is the one channel
    through which a clock-keyed tensor could re-enter, and it is closed by
    construction rather than by convention.** A ``RoutingViews`` carries folded
    liveness masks, so one built from ``write_levels_stored`` would be exactly the
    forbidden object -- and ``build_schedule`` calls
    ``views.require_source(read_levels, write_levels)``, which compares storage
    identity against the LIVE tensors it was handed. Asserted below: a view built
    from a stored/flushed clock is REJECTED rather than silently keying the
    bitmaps on it.
    """
    sig = inspect.signature(build_schedule)
    names = list(sig.parameters)
    assert names == ["read_levels", "write_levels", "g_read", "g_write", "plan", "views"]
    assert "write_levels_stored" not in names
    assert not any("stored" in name for name in names)

    widths = (2, 3)
    BT = 16
    T = 20
    B, H = 1, 1
    plan = _plan(widths, BT, _admissible_bcs(widths)[0])
    read_levels = tuple(torch.zeros((B, T, H, w)) for w in widths)
    write_levels = [torch.zeros((B, T, H, w)) for w in widths]
    # A tiny-but-exactly-nonzero deposit at token 5, level 0 digit 1.
    write_levels[0][0, 5, 0, 1] = 1e-30
    write_levels = tuple(write_levels)
    g_read = torch.ones((B, T, H))
    g_write = torch.ones((B, T, H))

    n_tiles = (T + BT - 1) // BT
    schedule = build_schedule(read_levels, write_levels, g_read, g_write, plan)
    unpacked_write = unpack_bits(schedule.write_tile_bits, n_tiles)
    off = schedule.level_offsets[0]
    tile_of_token_5 = 5 // BT
    assert bool(unpacked_write[0, off + 1, tile_of_token_5])

    # The `views` channel: a view built from a STORED clock that flushed the live
    # 1e-30 deposit to an exact zero. Passing it must raise, not silently produce a
    # schedule whose write bitmap has lost the deposit.
    write_stored = tuple(torch.zeros_like(level) for level in write_levels)
    stale = routing_views(read_levels, write_stored, plan.topology)
    with pytest.raises(ValueError, match="different routing tensors"):
        build_schedule(read_levels, write_levels, g_read, g_write, plan, views=stale)
    # ... and the same guard on the selection side, which reads the same masks.
    with pytest.raises(ValueError, match="different routing tensors"):
        select_plan(read_levels, write_levels, plan.topology, plan.BC,
                       normalization="raw", decay=False, views=stale)


# ---------------------------------------------------------------------------
# 8. No O(L*N) / O(N) allocation
# ---------------------------------------------------------------------------


def _flatten_tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _flatten_tensors(item)


class _AllocationGuard(TorchDispatchMode):
    """Records every dispatched tensor result whose numel exceeds ``bound``."""

    def __init__(self, bound):
        super().__init__()
        self.bound = bound
        self.violations = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        for t in _flatten_tensors(result):
            if t.numel() > self.bound:
                self.violations.append((str(func), tuple(t.shape), t.numel()))
        return result


def test_no_o_of_n_allocation_at_large_topology():
    """§4/§8: schedule construction is bounded by routing SUPPORT actually present,
    never by leaf capacity N. widths=(64,64,16) -> N=65536, but only a handful of
    owners are ever active (sparse routing), so no single tensor built along the
    way may approach N.

    Bound: ``max(BH*T*max_width*4, BH*n_tiles*sum_widths, 4096)`` -- generous
    multiples of the legitimately-sized objects (folded per-level tensors, packed
    bitmap rows before packing), strictly below N=65536.
    """
    widths = (64, 64, 16)
    BT = 16
    BC = 256
    T = 64
    B, H = 1, 1
    BH = B * H
    plan = _plan(widths, BT, BC, d_v=1, warp_tile=(1, 1))
    n_tiles = (T + BT - 1) // BT
    max_width = max(widths)
    bound = max(BH * T * max_width * 4, BH * n_tiles * sum(widths), 4096)

    g = torch.Generator().manual_seed(0)
    read_levels, write_levels = [], []
    live_positions = {0: [(3, 0), (3, 1), (40, 2)], 1: [(3, 0), (40, 1)], 2: [(3, 0)]}
    for lvl, w in enumerate(widths):
        r = torch.zeros((B, T, H, w))
        wv = torch.zeros((B, T, H, w))
        for t, d in live_positions[lvl]:
            r[0, t, 0, d] = float(torch.rand((), generator=g)) + 0.01
            wv[0, t, 0, d] = float(torch.rand((), generator=g)) + 0.01
        read_levels.append(r)
        write_levels.append(wv)
    g_read = torch.ones((B, T, H))
    g_write = torch.ones((B, T, H))

    guard = _AllocationGuard(bound)
    with guard:
        schedule = build_schedule(tuple(read_levels), tuple(write_levels), g_read, g_write, plan)
    assert not guard.violations, guard.violations
    assert schedule.owners.shape[0] > 0  # sanity: the fixture actually produced owners
    assert schedule.owners.shape[0] < 200  # far below owners_total = N/BC = 256

    j, c = _independent_owner_geometry(widths, BC)
    n_slices = widths[j] // c
    n_words = (n_tiles + 32 - 1) // 32
    # Tile bitmaps + slice unions, PLUS §4's support word -- one int32 per (side,
    # bh, level-digit row, token TILE) where the tile bitmap is one per token-tile
    # WORD. It is 32x the bitmap and still O(L/BT * sum widths); the quantity this
    # test bounds is dependence on N, and the support word has none.
    support_word_bytes = 2 * BH * n_tiles * 4 * sum(widths)
    expected_bytes = 2 * BH * n_words * 4 * (sum(widths) + n_slices) + support_word_bytes
    assert schedule.schedule_bytes() == expected_bytes
    assert support_word_bytes < 4 * (BC * 256)  # N = 65536 leafs; not O(N)


# ---------------------------------------------------------------------------
# 9. the joint (BT, compact) selection and its cost model
# ---------------------------------------------------------------------------


def test_the_spanned_sum_never_exceeds_BC_and_ties_at_one_spanned_level():
    """``sum_{l>=j} m_l <= prod_{l>=j} m_l = BC``, with equality iff ``D - j == 1``.

    A pure property of the OWNER RECTANGLE (§3), and M-1 (2026-08-01) turned it from
    a bound on a priced extent into the reason a whole term could hide: the EQUALITY
    case is why no calibration cell could ever see `gram_cols` as a variable. All
    four cells of `_CALIBRATION_ARGMIN` span ONE level, so `gram_cols == BC` on every
    row that bound `STAGE_COLUMN_WORK`, and the term was algebraically
    indistinguishable from a per-dense-k-quantum one. The `iso-gc-*` cells exist to
    break exactly this degeneracy (`D - j == 2`), and they measured the per-column
    price as exactly zero. Kept as a gate because the next term priced on a rectangle
    statistic must be isolated on a cell where the statistic is FREE of `BC`."""
    for widths, BC, spans_one in (((8, 16), 16, True), ((8, 16), 64, False),
                                  ((4, 4), 4, True), ((4, 4, 4), 16, False)):
        topology = _topology(widths, d_v=4)
        moduli = spanned_moduli(topology, BC)
        assert prod(moduli) == BC
        assert sum(moduli) <= BC
        assert (sum(moduli) == BC) == spans_one
        assert (len(moduli) == 1) == spans_one


def test_select_plan_axes_prices_BT_and_compact_from_one_model():
    """§6 rev .5 Q1+Q2: `BT` and `compact` come from ONE formula.

    The gates are structural properties of the model, not a re-assertion of the
    constants:

    1. A SATURATED distribution must never compile the gather in. At `K == BC` the
       gather returns the identity, so it can only cost -- that holds at ANY value
       of the calibrated constants and is the one direction that is not a
       measurement.
    2. **At the calibrated constants a collapsing union ALSO compiles it out, and
       that is the measured result, not a modelling failure.** The 48-cell A/B has
       the compiled-out instantiation faster at every reachable cell including
       `K = 28.8/64`, so `GATHER_TILE_WORK` is a lower bound and the break-even
       deficit lies outside the reachable range. Asserting the opposite here would
       be asserting a preference the measurement contradicts.
    3. The axis is nevertheless LIVE, not a constant `False`: lowering
       BOTH halves of the gather price (`gather_candidate_fixed_work` and
       `gather_candidate_quantum_work` -- plan parameters, so this is a probe of the
       model and not a fudge of it) flips the collapsing case to `compact=True` while
       leaving the saturated case dense. A model that had degenerated into "never
       compact" would fail this half.
    4. The returned axes are self-consistent: the reported cost is exactly what
       `plan_cost_per_token` gives for the returned triple, and no other candidate
       prices lower.
    """
    topology = _topology((8, 16), d_v=64)
    BC = 16
    T = 256

    def levels(density: float, seed: int):
        generator = torch.Generator().manual_seed(seed)
        out = []
        for width in topology.widths:
            amps = torch.rand((1, T, 1, width), generator=generator)
            out.append(torch.where(amps < density, amps, torch.zeros_like(amps)))
        return tuple(out)

    dense_r = tuple(torch.ones((1, T, 1, w)) for w in topology.widths)
    saturated = select_plan_axes(dense_r, dense_r, topology, BC, decay=False)
    assert saturated.stats.mean_K_candidate == BC
    assert saturated.compact is False, (
        "a fully saturated routing distribution compiled the gather IN; at K == BC "
        "the gather is the identity and can only cost")

    sparse_r, sparse_w = levels(0.12, 11), levels(0.12, 12)
    sparse = select_plan_axes(sparse_r, sparse_w, topology, BC, decay=False)
    assert sparse.stats.mean_K_candidate < BC
    assert sparse.compact is False, (
        f"the collapsing union (K={sparse.stats.mean_K_candidate:.2f}/{BC}) selected the "
        "gather at the CALIBRATED constants. That contradicts the A/B this model is "
        "calibrated against, where the compiled-out path wins at every reachable cell.")

    # ...but the axis is live: make the gather cheap and the SAME statistics flip.
    # BOTH halves of the gather price are lowered, because the gather is no longer
    # ONE number: `GATHER_TILE_WORK` was replaced (2026-08-01) by an intercept plus a
    # per-`ceil(BC/16)`-quantum slope, measured on an isolating cell PAIR, and
    # lowering only one of them would leave the other carrying the axis -- which
    # would make this probe pass for the wrong reason.
    cheap = dict(gather_candidate_fixed_work=1.0, gather_candidate_quantum_work=1.0)
    flipped = select_plan_axes(sparse_r, sparse_w, topology, BC, decay=False, **cheap)
    assert flipped.compact is True, (
        "with the gather priced at ~0 the collapsing union STILL compiled it out; the "
        "compact axis has degenerated into a constant and is no longer a selection")
    still_dense = select_plan_axes(dense_r, dense_r, topology, BC, decay=False, **cheap)
    assert still_dense.compact is False, (
        "a saturated union selected the gather even at a ~0 gather price; the model is "
        "no longer keyed on the saturation deficit at all")

    for result, reads, writes in ((saturated, dense_r, dense_r), (sparse, sparse_r, sparse_w)):
        assert result.BT in (16, 32)
        # The recorded cost must re-price EXACTLY, and since M-1 retired the staging
        # term there is no depth argument to pass: the model is ring-INVARIANT, which
        # is why `select_plan_axes` states the depth instead of arbitrating it.
        assert result.cost_per_token == plan_cost_per_token(
            topology, BC, result.stats, compact=result.compact, decay=False)
        for BT in (16, 32):
            stats = tile_union_stats(reads, writes, topology, BC, BT)
            for compact in (True, False):
                cost = plan_cost_per_token(topology, BC, stats, compact=compact,
                                           decay=False)
                assert cost >= result.cost_per_token - 1e-9, (
                    f"select_plan_axes returned {result.cost_per_token} but "
                    f"(BT={BT}, compact={compact}) prices {cost}")




def test_select_plan_axes_returns_a_runnable_plan_on_the_decay_arm():
    """The LeafMassDecay leg of the joint selector -- review §4.3's REQUIRED item.

    Before 2026-07-30 this half of the parameter space had NO runnable plan and no
    test said so: under LeafMassDecay ``compact=False`` always wins (the gather term
    dominates), at ``compact=False`` the crossover is evaluated at ``KI = BC``, and
    past the crossover the arm's dense limit is ``panel`` -- which is unimplemented,
    so every input returned a plan the driver refuses to launch. The only joint-
    selector test swept ``decay=False``, so the whole arm was invisible.

    Amended 2026-08-01 (item 6b): the ``intra_lowering`` AXIS is gone. The consumer
    includes ``intra_panel.cuh`` unconditionally and does one intra lowering, so
    "did the selector name a lowering the driver implements" is no longer a
    question anything can answer. What this still gates -- and what was actually
    missing in the first place -- is that the decay arm returns a plan that is
    STAMPABLE and self-consistently priced, at both a saturated and a collapsing
    union and at both ``BC``."""
    topology = _topology((8, 16), d_v=64)
    T = 256

    def levels(density: float, seed: int):
        generator = torch.Generator().manual_seed(seed)
        out = []
        for width in topology.widths:
            amps = torch.rand((1, T, 1, width), generator=generator)
            out.append(torch.where(amps < density, amps, torch.zeros_like(amps)))
        return tuple(out)

    dense_r = tuple(torch.ones((1, T, 1, w)) for w in topology.widths)
    sparse_r, sparse_w = levels(0.12, 11), levels(0.12, 12)

    for BC in (16, 64):
        for reads, writes, label in ((dense_r, dense_r, "saturated"),
                                     (sparse_r, sparse_w, "collapsing")):
            result = select_plan_axes(reads, writes, topology, BC, decay=True)
            # The returned axes must be STAMPABLE: `Plan`'s own arm rules are a
            # second authority, and axes that pass the selector and fail the plan
            # are still not a plan.
            plan = Plan(topology=topology, normalization="raw", BT=result.BT, BC=BC,
                          warp_tile=(1, 2), decay=True, compact=result.compact)
            assert (plan.BT, plan.compact) == (result.BT, result.compact)
            # THE `BT == 32` ASSERTION THAT USED TO SIT HERE IS WITHDRAWN, 2026-08-01,
            # AND ITS STATED REASON WAS THE THING THAT DIED.
            #
            # It read: "the per-tile fixed cost is the only BT term on this arm, so 32
            # is the cost argmin and nothing else may win silently". That premise was
            # true of the PRE-QUANTA model, in which the decay arm's only `BT`-bearing
            # term was `TILE_FIXED_WORK / BT`. It is false of this one, which carries
            # three further `BT`-dependent terms, every one of them measured:
            # `readout_fold_kstep_work * ksteps * (BT/16)`, `decay_kstep_work * ksteps
            # * (BT/16)`, and the causal `BT(BT+1)/272` intra Gram -- plus
            # `candidate_frac`, which is NOT `BT`-invariant (a 32-token tile union
            # catches owners a 16-token one misses; measured 1.846x at
            # `atscale-L2048-BC16`). The claim was therefore verbatim the superseded
            # `_CALIBRATION_BT` table's disposition, re-asserted here as if it were a
            # structural property, and §2.32 measured `BT = 16` FASTER at the flagship
            # on every arm by 89-153 %.
            #
            # WHAT REPLACES IT IS THE PROPERTY THIS TEST CAN ACTUALLY OWN: the `BT`
            # that comes back is the joint argmin's, self-consistently priced. The
            # *value* of the argmin on frozen cells is bound by
            # `_CALIBRATION_ARGMIN`'s 8 measured rows, which is where a disposition
            # claim belongs -- not on a synthetic fixture no measurement covers.
            assert result.BT in (16, 32)
            assert result.cost_per_token == plan_cost_per_token(
                topology, BC, result.stats, compact=result.compact, decay=True), (
                f"at BC={BC} ({label}) the decay-arm axes report a cost that "
                "re-pricing the returned triple does not reproduce: the plan and its "
                "recorded evidence are not the same plan")




#: **`_CALIBRATION_AB` IS DELETED (item 6b, 2026-08-01), ON ITS OWN STATED TERMS.**
#: It was a 48-cell compact/`BT` A/B of the DELETED register-resident consumer
#: (`c3c2f845`)
#: (`workflows/20260729-wave3/compaction2.md` §B2, RTX 3080 Ti, 30 reps median),
#: whose gather is a different algorithm -- a scalar walk's trip count, a
#: dense-panel-to-compacted copy, a staging-ring probe -- and on which compaction
#: LOST at every cell. It was retained UNASSERTED with the note "it is the old
#: consumer's calibration and is deleted with that kernel in item 6b"; that kernel
#: is deleted, so it is. Every constant this model carries was re-measured on the
#: surviving kernel and is bound by `_CALIBRATION_ARGMIN` below.

#: **SUPERSEDED 2026-08-01 -- `_CALIBRATION_BT` IS DELETED, AND ITS KEY WAS WRONG,
#: NOT JUST ITS VALUES** (`rewrite.md` §2.32/§2.34, coordinator ruling item 4).
#:
#: It was `{(BC, decay) -> BT}`, measured on the OLD consumer. In the eight joint
#: argmins of the §2.32 `BT` matrix -- measured on the kernel this model now prices
#: -- NEITHER `BC` NOR `decay` appears in the partition: both `BC` agree with each
#: other on every row and both decay arms agree with each other on every row. What
#: the answer tracks is the ROUTING (`candidate_frac`, `mean_ksteps_candidate`) and
#: the WAVE REGIME, and the table had no column for either. Its values were also
#: hazardous: it selects `BT = 32` at `decay=True` where §2.32 measures `BT = 32`
#: 64-153 % SLOWER at the flagship.
#:
#: What replaces it is `_CALIBRATION_ARGMIN` below -- the JOINT `(BT, compact)`
#: decision `select_plan_axes` actually makes, against the argmin measured for it,
#: rather than 16 per-slice orderings the planner never evaluates.

#: THE §2.32 `BT` x `compact` MATRIX, as the eight decisions the planner makes.
#:
#: Every field is a RECORDED MEASUREMENT of a named, frozen cell in
#: `benchmarks/ops/benchmark_rola_v3_new_consumer.py`, carried here so the gate runs
#: on a CPU-only host and so no constant can drift away from what set it:
#:
#:   * `candidate_frac`, `ksteps`, `kisteps` -- the exact stats-pass reduction over
#:     the realized schedule at each `BT` (`--mode manifest` reproduces them);
#:   * `bpsm` -- `rola_consumer_occupancy`, the DRIVER's
#:     `cudaOccupancyMaxActiveBlocksPerMultiprocessor` on the exact consumer
#:     instantiation, cross-checked against `ncu`'s
#:     `min(launch__occupancy_limit_{registers,shared_mem,warps,blocks})` on all 16
#:     profiled arms with zero disagreement (§2.34);
#:   * `n_cohorts` -- `owners_total * BH`, the requested CTA grid;
#:   * the argmin -- §2.32's four-way table, per cell and decay arm, each entry
#:     with its own instrument (paired interleaved timers where the IQR resolves,
#:     `ncu --clock-control base` durations where it does not).
#:
#: `(cell, widths, d_v, BC, n_cohorts, decay, {BT: (candidate_frac, ksteps, kisteps,
#:   {(decay, compact): bpsm})}, measured_argmin)` -- flattened into rows below.
_ARGMIN_STATS = {
    # cell -> (widths, d_v, BC, n_cohorts, {BT: (candidate_frac, ksteps, kisteps)})
    "atscale-L2048": ((256, 256), 64, 64, 8192, {
        16: (0.19916152954101562, 0.6086546380892184, 0.6086546380892184),
        32: (0.35765838623046875, 0.6364416903090936, 0.6364416903090936)}),
    "atscale-L2048-BC16": ((256, 256), 64, 16, 32768, {
        16: (0.19362306594848633, 0.22103670174384385, 0.22103670174384385),
        32: (0.3573589324951172, 0.23429485639867847, 0.23429485639867847)}),
    "topo-D2-w64x64-BC64": ((64, 64), 64, 64, 128, {
        16: (1.0, 4.0, 4.0), 32: (1.0, 4.0, 4.0)}),
    "topo-D2-w64x16-BC16": ((64, 16), 64, 16, 128, {
        16: (1.0, 1.0, 1.0), 32: (1.0, 1.0, 1.0)}),
}

#: `rola_consumer_occupancy(decay, compact, BC, BT, d_v)` -> CTAs/SM, the rows the
#: eight argmins need. `(BC, BT, decay)` suffices: `compact` does not move the
#: answer at any of them (it adds 136-520 B, inside the same CTA count), and that
#: is asserted rather than assumed by the dump this was read back from.
_ARGMIN_BPSM = {
    (64, 16, False): 3, (64, 16, True): 3, (64, 32, False): 2, (64, 32, True): 2,
    (16, 16, False): 4, (16, 16, True): 3, (16, 32, False): 2, (16, 32, True): 2,
}

#: `(cell, decay) -> measured argmin (BT, compact)`, §2.32's four-way table.
#: Every many-wave cell wants `(16, compact)`; every one-wave cell wants
#: `(32, dense)`.
_CALIBRATION_ARGMIN = {
    ("atscale-L2048", False): (16, True),
    ("atscale-L2048", True): (16, True),
    ("atscale-L2048-BC16", False): (16, True),
    ("atscale-L2048-BC16", True): (16, True),
    ("topo-D2-w64x64-BC64", False): (32, False),
    ("topo-D2-w64x64-BC64", True): (32, False),
    ("topo-D2-w64x16-BC16", False): (32, False),
    ("topo-D2-w64x16-BC16", True): (32, False),
}


def _argmin_stats(cell, BT):
    """The recorded `TileUnionStats` for one calibration row."""
    widths, d_v, BC, n_cohorts, per_bt = _ARGMIN_STATS[cell]
    cf, ks, kis = per_bt[BT]
    return TileUnionStats(
        BC=BC, BT=BT, mean_K=cf * BC, saturated_frac=1.0 if cf == 1.0 else 0.0,
        mean_K_candidate=ks * MMA_K_QUANTUM, mean_KI_candidate=kis * MMA_K_QUANTUM,
        mean_ksteps_candidate=ks, mean_kisteps_candidate=kis,
        candidate_frac=cf, n_pairs=0, n_cohorts=n_cohorts)


def _modelled_argmin(cell, decay, *, sm_count=DEFAULT_SM_COUNT, **overrides):
    """The `(BT, compact)` argmin `select_plan_axes` would return on this row.

    Reproduces the selection's own arithmetic on RECORDED inputs: price every
    `(BT, compact)`, multiply by `plan_waves` from the recorded occupancy and cohort
    count, take the minimum. There is no ring argument any more: M-1 measured the
    staging term's per-column price as EXACTLY zero and retired it, so the model is
    ring-INVARIANT and `select_plan_axes` states the depth rather than arbitrating
    it. This reproduction is exact rather than a simplification.
    """
    widths, d_v, BC, _n_cohorts, _ = _ARGMIN_STATS[cell]
    topology = _topology(widths, d_v=d_v)
    out = {}
    for BT in (16, 32):
        stats = _argmin_stats(cell, BT)
        for compact in (True, False):
            cost = plan_cost_per_token(topology, BC, stats, compact=compact,
                                       decay=decay, **overrides)
            waves = plan_waves(stats.n_cohorts, _ARGMIN_BPSM[(BC, BT, decay)], sm_count)
            out[(BT, compact)] = cost * waves
    return min(out, key=out.get), out


def test_the_cost_model_reproduces_its_calibration():
    """§6's constants must still reproduce the DECISION they were measured from.

    **THE BINDING TARGET CHANGED 2026-08-01 (coordinator ruling, `rewrite.md`
    §2.32-§2.34), and that is the point of this gate rather than a detail of it.**
    It used to assert 16 PER-SLICE orderings -- "which `BT` wins at fixed
    `compact`", "which `compact` wins at fixed `BT`". The planner ranks neither:
    `select_plan_axes` minimizes over the JOINT `(BT, compact)` product and emits
    ONE plan, so a per-slice ordering can be wrong at a combination the planner
    never evaluates. One of them provably IS -- see
    `test_a_saturated_dense_cell_never_argmins_onto_a_compacted_arm` -- and no value
    of any constant can reach it, so the old target could not be met honestly.

    What is asserted now is the EIGHT MEASURED ARGMINS of §2.32's `BT` x `compact`
    matrix: four frozen cells x two decay arms, each cell's four combinations timed
    or counted with its instrument named. That is strictly closer to what the
    planner does, and strictly harder to satisfy by accident -- a wrong constant
    flips the flagship rows' `compact` off or the one-wave rows' `BT` down.

    It remains a CALIBRATION gate, not a performance gate: it asserts the model's
    argmin matches the measurement's, never that any millisecond is reproducible on
    another machine. Its inputs are RECORDED (routing statistics from the exact
    stats pass, CTAs/SM from the driver's occupancy oracle, the cohort count), so it
    runs with no GPU.

    **BINDING IS ASSERTED, NOT CLAIMED (adversarial review 2026-07-30 §4.2).** The
    docstring used to name three constants; mutation showed it bound one. So the
    final sections exhibit, per constant, a value at which these eight rows FAIL --
    with the SIDE it fails from, because several are one-sided -- and then assert
    which two constants this gate does NOT identify, so a non-binding is a checked
    fact rather than an unwritten one."""
    # ---- THE EIGHT JOINT ARGMINS -------------------------------------------
    for (cell, decay), expected in _CALIBRATION_ARGMIN.items():
        got, table = _modelled_argmin(cell, decay)
        assert got == expected, (
            f"the model's joint (BT, compact) argmin at {cell} decay={decay} is "
            f"{got}, but the §2.32 matrix measured {expected}. Effective costs: "
            f"{ {k: round(v, 1) for k, v in table.items()} }")

    # ---- EACH CONSTANT BINDS: a value at which the eight rows FAIL -----------
    def _rows_hold(**overrides) -> bool:
        return all(_modelled_argmin(cell, decay, **overrides)[0] == expected
                   for (cell, decay), expected in _CALIBRATION_ARGMIN.items())

    assert _rows_hold(), "the eight argmins must hold at the calibrated defaults"

    # `DECAY_KSTEP_WORK` is quoted as a MEAN of two measured slopes, so the whole
    # measured range is asserted rather than the midpoint: every ordering must hold
    # across `[959.7, 1106.3]`, which makes the residual spread between the two
    # isolating solves a GATED property instead of a rounding choice.
    # `TILE_FIXED_WORK` is likewise quoted as a MEAN, of the two independent
    # isolating solves M-1 ran (`iso-tile-w16x4x64-BC{64,16}-T{128,256,512}` gives
    # 3507.0; `topo-D2-w64x{64,16}-BC{64,16}` + `iso-tile-w64x{64,16}-BC*-T256`
    # gives 3513.7), so the whole measured range is asserted rather than the
    # midpoint.
    for value in (3507.0, 3513.7):
        assert _rows_hold(tile_fixed_work=value), (
            f"the argmins do not survive TILE_FIXED_WORK={value}, which is an END of "
            "the range the two isolating solves bracket -- the landed value is their "
            "mean and the model may not depend on which solve it sits at")
    for value in (959.7, 1106.3):
        assert _rows_hold(decay_kstep_work=value), (
            f"the argmins do not survive DECAY_KSTEP_WORK={value}, which is an END "
            "of the measured bracket -- the landed value is the mean of the two "
            "isolating solves and the model may not depend on which end it sits at")

    # WHICH CONSTANTS THIS GATE IDENTIFIES, AND FROM WHICH SIDE. Swept rather than
    # asserted: each line below is a value at which the eight rows FAIL, and the
    # side it fails from is stated because several of these are one-sided.
    #
    # `GATHER_CANDIDATE_QUANTUM_WORK`, from ABOVE. This is the headline: at 13000 --
    # the value the DELETED `GATHER_TILE_WORK` carried -- compaction is priced out
    # of every cell and both at-scale rows lose the arm they measured -8.2 % and
    # -24.7 % on. That regression is exactly what the measured split repaired, and
    # it is now a gate rather than a claim.
    assert not _rows_hold(gather_candidate_quantum_work=13000.0), (
        "GATHER_CANDIDATE_QUANTUM_WORK no longer binds from above: at 13000 -- the "
        "deleted GATHER_TILE_WORK's value -- the eight argmins still reproduce, so "
        "nothing stops compaction being priced back out of the plan")
    # `GATHER_CANDIDATE_FIXED_WORK`, from ABOVE, on the `BC = 16` cell first: it is
    # the half that does NOT scale with `ceil(BC/16)`, so it is what tolls the
    # flagship `BC`.
    assert not _rows_hold(gather_candidate_fixed_work=2000.0), (
        "GATHER_CANDIDATE_FIXED_WORK no longer binds from above: the eight argmins "
        "reproduce at 2000 as well as at its measured 177.8")
    # `READOUT_FOLD_KSTEP_WORK`, from BELOW. It carries the `BT`-linear cost and the
    # whole quantum saving; shrink it and the at-scale cells stop seeing compaction
    # pay for itself.
    assert not _rows_hold(readout_fold_kstep_work=100.0), (
        "READOUT_FOLD_KSTEP_WORK no longer binds from below: the eight argmins "
        "reproduce at 100 as well as at its measured 1755")
    # `INTRA_GRAM_KSTEP_WORK`, from ABOVE. It is the only `BT(BT+1)/2` term, so a
    # large value makes `BT = 32` quadratically expensive and the one-wave `BC = 64`
    # cell loses the `BT = 32` it measured 4.2-15.1 % faster.
    assert not _rows_hold(intra_gram_kstep_work=5000.0), (
        "INTRA_GRAM_KSTEP_WORK no longer binds from above: the eight argmins "
        "reproduce at 5000 as well as at its measured 144")
    # `TILE_FIXED_WORK`, from BELOW, and it is IDENTIFIED HERE ONLY SINCE M-1
    # (2026-08-01). It is `BT`-invariant per tile, so per token it divides by `BT`
    # and it is the term that pays for the larger tile on a one-wave cell. Until M-1
    # it was 120 and decision-INERT from 0 to 1e6, because the work it names was
    # being carried by `STAGE_COLUMN_WORK` -- a term priced on `gram_cols`, a
    # variable no calibration cell could see (all four span ONE level, so
    # `gram_cols == BC` on every row). The `iso-gc-*` cells measured the per-column
    # price as EXACTLY zero, the `iso-tile-*` cells measured this constant as
    # 3510.35, and the four one-wave `topo` rows -- which flipped to `(16, dense)`
    # when the staging term was simply zeroed -- are back on the measured
    # `(32, dense)` because the term that carries them is now the one that was
    # measured to carry them.
    assert not _rows_hold(tile_fixed_work=0.0), (
        "TILE_FIXED_WORK no longer binds from below: the eight argmins reproduce at "
        "0 as well as at its measured 3510.35, so nothing in this gate is holding "
        "the per-candidate-tile constant M-1 identified")

    # ---- AND WHAT THIS GATE DOES **NOT** IDENTIFY, RECORDED AS AN ASSERTION --
    # Two constants are measured but decision-INERT at these eight rows, and saying
    # so in a comment is exactly the failure mode the 2026-07-30 review caught
    # (`MMA_ISSUE_WORK` was documented as bound and was not). So it is asserted: if
    # a future change makes either of them identified here, THIS fails, and the
    # correct response is to move the constant into the binding list above -- not to
    # delete the check.
    #
    # `DECAY_KSTEP_WORK` rides the SAME `ksteps` quantum as the readout and enters
    # only the decay arms, where it scales both `(BT, compact)` legs together; on a
    # saturated cell it is identical on the compact and dense arms and cancels
    # outright. It is pinned by its own two-point isolating solve
    # (`topo-D2-w64x64-BC64` against `topo-D2-w64x16-BC16`, slopes 1106.3 and 959.7
    # with intercepts within 10 % of zero), NOT by this table -- and the bracket
    # assertion above still gates that the answer does not depend on which end of
    # the measured range the landed mean sits at.
    assert _rows_hold(decay_kstep_work=0.0) and _rows_hold(decay_kstep_work=1e6), (
        "DECAY_KSTEP_WORK is now IDENTIFIED by the joint argmin, which it was not "
        "when this gate landed -- promote it into the binding sweep above")


#: THE ABSOLUTE `smsp__inst_executed.sum` TOTALS, per profiled arm, at `BT = 16`.
#: `(cell, decay, compact) -> (n_pairs, recorded_total)`.
#:
#: `n_pairs` is `TileUnionStats.n_pairs = owners_total * BH * n_tiles` -- it ALREADY
#: CARRIES `BH`, which is the one construction error worth naming here, because the
#: obvious `candidate_frac * n_pairs * BH * w_tile` form multiplies `BH` twice and
#: inflates every residual by 2.5x-14x. Confirmed against the "1024 candidate pairs
#: each" both `topo` cells report at `BH = 2`.
#:
#: Provenance: the `topo` rows are `rewrite.md` §2.32's `btab` `BT = 16` counters
#: (which §2.33 states reproduce §2.20's to the digit); the `atscale` rows are
#: §2.33's decay-constant solve. Instrument in every case `ncu --clock-control base`,
#: `smsp__inst_executed.sum`, ONE profiled launch per arm. Quoted to the precision
#: the sections record (0.1 M and 1 M respectively), which is far finer than the
#: residuals below and therefore not the limiting term in any of them.
_RECORDED_INSTRUCTION_TOTALS = {
    ("topo-D2-w64x64-BC64", False, False): (1024, 9.410e6),
    ("topo-D2-w64x64-BC64", True, False): (1024, 13.888e6),
    ("topo-D2-w64x64-BC64", False, True): (1024, 11.743e6),
    ("topo-D2-w64x64-BC64", True, True): (1024, 15.786e6),
    ("topo-D2-w64x16-BC16", False, False): (1024, 5.298e6),
    ("topo-D2-w64x16-BC16", True, False): (1024, 6.377e6),
    ("topo-D2-w64x16-BC16", False, True): (1024, 6.017e6),
    ("topo-D2-w64x16-BC16", True, True): (1024, 7.113e6),
    ("atscale-L2048", False, False): (1048576, 1812.1e6),
    ("atscale-L2048", True, False): (1048576, 2719.6e6),
    ("atscale-L2048", False, True): (1048576, 1035.0e6),
    ("atscale-L2048", True, True): (1048576, 1092.6e6),
}


def test_the_absolute_instruction_totals_are_a_RECORDING_not_an_assertion(capsys):
    """THE ABSOLUTE-TOTAL GATE, IN EXPLICIT RECORDING MODE. It computes and commits
    the per-cell residual table and **ASSERTS NO TOLERANCE ON IT, DELIBERATELY.**

    Every other gate in this file binds a DECISION -- an argmin, an ordering, an
    unreachability. This one asks a strictly harder question the model has never
    been held to: does `w_tile` predict the ABSOLUTE instruction count a profiled
    launch retires? The answer is recorded rather than asserted, and both halves of
    that sentence are deliberate.

    **WHY NOTHING IS ASSERTED.** The residuals below run to +45 %. A tolerance wide
    enough to admit that would be FITTED to the rows it must pass -- the standing
    rule forbids exactly that -- and it would certify an absolute accuracy the model
    does not have and does not claim. Asserting a tolerance here would make the gate
    a decoration; recording the table makes it EVIDENCE, and evidence is what the
    missing terms are identified from. **This gate is how M-1 was found and it is
    how the next term will be**, so it stays in recording mode.

    **WHAT THE SHAPE OF THE RESIDUALS SAID, AND WHAT M-1 DID WITH IT (2026-08-01).**
    The table used to be printed in two columns, WITH and WITHOUT the staging term.
    With it the sign FLIPPED ON `BC` (+27..+84 % at `BC = 64`, -26..-35 % at
    `BC = 16`, max 83.9 %, mean 41.9 %) -- not the shape of a model that prices only
    what it measured, and the quantitative half of the finding that
    `STAGE_COLUMN_WORK` priced a staging ring the kernel does not have. Zeroing it
    made every row UNDER-predict (-8..-61 %, mean 27.5 %, signs consistent) and
    flipped four of the eight argmins, which is why it could not simply be deleted.

    M-1 measured the term out and measured the remainder in: the per-column price is
    EXACTLY zero (the `iso-gc-*` cells are byte-identical across a 2.1x range of
    `gram_cols`) and the work was a per-candidate-tile CONSTANT, `TILE_FIXED_WORK`,
    re-measured 120 -> 3510.35 on the `iso-tile-*` cells. The residuals below are
    the result: **max 44.8 %, mean 17.4 %, signs CONSISTENT (+)**, against 83.9 % /
    41.9 % / mixed before -- and the `BC = 16` rows, which were -26..-35 %, are now
    +2.9..+4.6 %.

    **THE REMAINING RESIDUAL IS NAMED, MEASURED AND ONE-SIDED**, which is what makes
    it the next item rather than noise: (1) the model charges 1899 per dense
    k-quantum (`READOUT_FOLD_KSTEP_WORK + INTRA_GRAM_KSTEP_WORK`) where M-1's `BC`
    sweep measures 1033 / 1155 / 1316 at `D = 1 / 2 / 3` -- so every row over-
    predicts in proportion to its k-steps, and the quantum itself grows ~141 per
    level, which no constant in this model expresses; (2) at `atscale-L2048` 39 % of
    candidate pairs gather NOTHING, and such a tile skips per-tile work too, not
    only its k-steps, which is why the two compact at-scale rows are the worst rows
    in the table. Both are MEASUREMENTS to make, on isolating cells, exactly as this
    one was. Nothing here may be closed by choosing a number.

    Runs on CPU: every input is recorded (`_ARGMIN_STATS`,
    `_RECORDED_INSTRUCTION_TOTALS`).
    """
    rows = []
    for (cell, decay, compact), (n_pairs, recorded) in _RECORDED_INSTRUCTION_TOTALS.items():
        widths, d_v, BC, _n_cohorts, _ = _ARGMIN_STATS[cell]
        topology = _topology(widths, d_v=d_v)
        stats = _argmin_stats(cell, 16)
        # `pred = candidate_frac * n_pairs * w_tile`, and
        # `plan_cost_per_token = candidate_frac * w_tile / BT`, so the prediction is
        # `cost * BT * n_pairs` -- the model's OWN arithmetic, with no separate
        # reimplementation of `w_tile` that could drift away from it.
        pred = plan_cost_per_token(topology, BC, stats, compact=compact,
                                   decay=decay) * 16 * n_pairs
        rows.append((cell, decay, compact, recorded, pred, pred / recorded - 1.0))

    hdr = (f"\n{'cell':22s} {'decay':>5s} {'compact':>7s} {'ncu (M)':>10s} "
           f"{'pred (M)':>10s} {'resid':>8s}")
    lines = [hdr, "-" * len(hdr)]
    for cell, decay, compact, rec, pred, resid in rows:
        lines.append(f"{cell:22s} {str(decay):>5s} {str(compact):>7s} {rec / 1e6:10.1f} "
                     f"{pred / 1e6:10.1f} {resid * 100:+7.1f}%")
    lines.append(f"max |resid| {max(abs(r[5]) for r in rows) * 100:.1f} %, "
                 f"mean |resid| "
                 f"{sum(abs(r[5]) for r in rows) / len(rows) * 100:.1f} %; "
                 f"signs {sorted({'+' if r[5] > 0 else '-' for r in rows})} "
                 f"(pre-M-1: 83.9 %, 41.9 %, ['+', '-'])")
    with capsys.disabled():
        print("\n".join(lines))

    # THE ONLY ASSERTION IS THAT THE TABLE WAS PRODUCED -- that every row priced to a
    # finite positive number, i.e. the recording actually ran. It is NOT an accuracy
    # claim and must never become one by having a tolerance added to it.
    assert len(rows) == len(_RECORDED_INSTRUCTION_TOTALS)
    for cell, decay, compact, _rec, pred, _resid in rows:
        assert math.isfinite(pred) and pred > 0, (
            f"the model priced {cell} decay={decay} compact={compact} to a "
            f"non-finite or non-positive absolute total ({pred}) -- the recording "
            "did not run, which is the one thing this gate does assert")


def test_a_saturated_dense_cell_never_argmins_onto_a_compacted_arm():
    """The GUARD on the model's one PROVED blind spot (`COST_MODEL_PROVENANCE`).

    On a SATURATED-DENSE cell (`K == BC` on every candidate pair) compaction
    changes no extent at all, so the per-slice `BT` preference at `decay + compact`
    inverts against this model and no value of any constant can reach it: adding
    compaction is exactly one positive constant `G` on both `BT` arms, and
    `dr/dG < 0` whenever the model prefers `BT = 32` at all, so compaction must
    DEEPEN a `BT = 32` preference rather than invert it. The measurement inverts it
    (`BT = 32` retiring 14 % FEWER instructions in 5.7-27.2 % MORE time: a stall,
    which a work model does not model).

    That row is not a decision -- the joint argmin never selects a compacted arm on
    such a cell, because compaction there is pure toll. **This asserts the
    unreachability rather than leaving it as an observation**, so if a future axis
    ever makes the row reachable this fails loudly instead of the blind spot
    silently becoming load-bearing.

    Swept over BOTH saturated cells, BOTH decay arms, and -- because the claim is
    about the SHAPE and not about today's numbers -- across the whole measured range
    of `DECAY_KSTEP_WORK` and a 10x sweep of the gather constants.
    """
    saturated = [c for c, (_w, _d, _bc, _n, per) in _ARGMIN_STATS.items()
                 if all(cf == 1.0 for cf, _k, _ki in per.values())]
    assert sorted(saturated) == ["topo-D2-w64x16-BC16", "topo-D2-w64x64-BC64"], (
        "the saturated-dense cells are the two `topo` cells; if that set changed, "
        "the guard below is guarding the wrong rows")
    sweeps = [{}]
    for v in (959.7, 1106.3):
        sweeps.append({"decay_kstep_work": v})
    for f in (0.1, 10.0):
        sweeps.append({"gather_candidate_quantum_work": 525.1 * f})
        sweeps.append({"gather_candidate_fixed_work": 177.8 * f})
    for cell in saturated:
        for decay in (False, True):
            for overrides in sweeps:
                (BT, compact), table = _modelled_argmin(cell, decay, **overrides)
                assert compact is False, (
                    f"the joint argmin selected a COMPACTED arm at the "
                    f"saturated-dense cell {cell} (decay={decay}, overrides="
                    f"{overrides}): compaction changes no extent there, so it is "
                    f"pure toll, and the model's proved blind spot -- the inverted "
                    f"per-slice BT preference at decay+compact -- becomes reachable. "
                    f"Effective costs: { {k: round(v, 1) for k, v in table.items()} }")






# ---------------------------------------------------------------------------
# 10. rank_spans / permuted_topology
# ---------------------------------------------------------------------------


def test_span_score_matches_hand_computation():
    stats = LevelSupportStats(s_r=2.0, s_w=3.0, rho_r=0.5, rho_w=0.25)
    width = 4
    expected = (2.0 * 0.5 + 3.0 * 0.25) / 4  # = 1.75/4 = 0.4375
    assert span_score(stats, width) == pytest.approx(expected)
    with pytest.raises(ValueError, match="width must be >= 1"):
        span_score(stats, 0)


def test_rank_spans_orders_descending_with_tie_break():
    widths = (2, 3, 4)
    topology = _topology(widths, d_v=2)
    # level0: score 1.0/2=0.5 ; level1: score 1.0/3~=0.333; level2 ties level? build a tie.
    stats = (
        LevelSupportStats(s_r=1.0, s_w=0.0, rho_r=1.0, rho_w=0.0),  # score = 1/2 = 0.5
        LevelSupportStats(s_r=1.0, s_w=0.0, rho_r=1.0, rho_w=0.0),  # score = 1/3 ~= 0.333
        LevelSupportStats(s_r=2.0, s_w=0.0, rho_r=1.0, rho_w=0.0),  # score = 2/4 = 0.5 (ties level0)
    )
    order = rank_spans(topology, stats)
    scores = [span_score(s, w) for s, w in zip(stats, widths)]
    assert scores[0] == pytest.approx(scores[2])  # constructed tie
    # Tie broken by lower level index first: level0 before level2.
    assert order == (0, 2, 1)
    # Full independent re-derivation: sort by (-score, level) and compare.
    expected = tuple(level for _score, level in sorted(
        ((-s, lvl) for lvl, s in enumerate(scores)), key=lambda item: item))
    assert order == expected


def test_permuted_topology_reorders_and_validates():
    widths = (2, 3, 4)
    topology = _topology(widths, d_v=2)
    order = (2, 0, 1)
    permuted = permuted_topology(topology, order)
    assert permuted.widths == (4, 2, 3)
    assert permuted.d_v == topology.d_v

    with pytest.raises(ValueError, match="must be a permutation"):
        permuted_topology(topology, (0, 0, 1))
    with pytest.raises(ValueError, match="must be a permutation"):
        permuted_topology(topology, (0, 1))


def test_permuted_topology_takes_no_routing_tensors():
    """Docstring warning: callers must permute read/write tensors themselves --
    the helper has no way to, because it accepts none. Verified structurally."""
    sig = inspect.signature(permuted_topology)
    assert list(sig.parameters) == ["topology", "order"]


# ---------------------------------------------------------------------------
# 11. Sort-batching cost order under a NON-DYADIC write_cost_weight (F5)
# ---------------------------------------------------------------------------
#
# `test_owners_match_oracle_set_and_order` above only ever exercises
# `DEFAULT_WRITE_COST_WEIGHT = 0.125`, a dyadic value for which fp64-then-cast
# and fp32 arithmetic agree bit-for-bit (small integer counts times a power of
# two is exact in both). The moment a *measured*, non-dyadic weight replaces
# that placeholder, the batched planner's fp32 cost and the oracle's
# fp64-then-cast cost (`test_rola_v3_schedule.py`'s `_oracle_owners`) can differ
# by 1 ulp and permute a near-tie -- a latent gap noted but not covered by the
# original suite. This mirrors the existing matrix exactly, at a representative
# subset of it (full-matrix cost would roughly double this file's runtime for
# no additional structural coverage), with `write_cost_weight=0.1` in place of
# the default.
_NONDYADIC_WRITE_COST_WEIGHT = 0.1
assert _NONDYADIC_WRITE_COST_WEIGHT != DEFAULT_WRITE_COST_WEIGHT
_NONDYADIC_MATRIX = [
    (widths, 32, "exact", density) for widths in _TOPOLOGIES for density in _DENSITIES
]
_NONDYADIC_MATRIX_IDS = [_combo_id(*c) for c in _NONDYADIC_MATRIX]


@pytest.mark.parametrize("combo", _NONDYADIC_MATRIX, ids=_NONDYADIC_MATRIX_IDS)
def test_owners_match_oracle_set_and_order_nondyadic_weight(combo):
    """F5: same claim as `test_owners_match_oracle_set_and_order`, but with a
    non-dyadic `write_cost_weight` so a bit-identity coincidence of the default
    can no longer be masking an order-sensitivity bug in the batched cost path."""
    widths, BT, t_kind, density = combo
    T = _t_variants(BT)[t_kind]
    BC = _admissible_bcs(widths)[0]
    B, H = 2, 2
    read_levels, write_levels, g_read, g_write = _gen_topology_tensors(0, widths, B, T, H, density)
    plan = _plan(widths, BT, BC, write_cost_weight=_NONDYADIC_WRITE_COST_WEIGHT)
    n_tiles = (T + BT - 1) // BT

    schedule = build_schedule(read_levels, write_levels, g_read, g_write, plan)

    read_live = [_oracle_tile_live(_oracle_fold(read_levels[lvl]), BT, n_tiles) for lvl in range(len(widths))]
    write_live = [_oracle_tile_live(_oracle_fold(write_levels[lvl]), BT, n_tiles) for lvl in range(len(widths))]

    oracle = _oracle_owners(read_live, write_live, widths, BC, n_tiles, plan.write_cost_weight)
    oracle_pairs = [(bh, owner) for bh, owner, _cost in oracle]

    got_pairs = [tuple(row) for row in schedule.owners.tolist()]
    assert got_pairs == oracle_pairs, (got_pairs, oracle_pairs)

    oracle_costs = torch.tensor([cost for _b, _o, cost in oracle], dtype=torch.float32)
    torch.testing.assert_close(schedule.owner_costs, oracle_costs, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 12. LeafMassDecay dial cast-boundary re-validation (review fix pass, B1)
# ---------------------------------------------------------------------------


def test_leaf_rate_dials_rejects_fp64_dial_that_rounds_to_1_0f():
    """B1 cliff, reject side: an fp64 dial in ``[1 - 3e-8, 1)`` passes
    ``LeafMassDecay``'s own-dtype check (strictly ``< 1`` in fp64) but rounds to
    exactly ``1.0f`` under the ``.to(torch.float32)`` cast the kernel consumes,
    which NaNs the kernel (``log1pf(-1.0f) = -inf``, then ``expf(0 * -inf)`` for
    any leaf with a zero clock). ``leaf_rate_dials``'s cast-boundary re-check
    must reject it loudly rather than let it through to the kernel."""
    widths = (4, 4)
    topology = _topology(widths, d_v=4)
    dial_val = 1.0 - 1e-9  # strictly < 1 in fp64
    assert float(torch.tensor(dial_val, dtype=torch.float64).float()) == 1.0, (
        "fixture assumption broken: this fp64 value no longer rounds to 1.0f")
    dials = tuple(torch.full((2, w), dial_val, dtype=torch.float64) for w in widths)
    decay = LeafMassDecay(dials=dials)  # passes the fp64 __post_init__ check
    with pytest.raises(ValueError, match="1\\.0f"):
        leaf_rate_dials(decay, topology)


def test_leaf_rate_dials_accepts_fp64_dial_that_survives_the_cast():
    """B1 cliff, survive side: a dial far enough from 1 (``1 - 1e-6``) rounds to
    a value strictly less than ``1.0f``, so the cast-boundary check must not
    raise, and the flattened dial tensor must be usable by the kernel."""
    widths = (4, 4)
    topology = _topology(widths, d_v=4)
    dial_val = 1.0 - 1e-6
    assert float(torch.tensor(dial_val, dtype=torch.float64).float()) < 1.0, (
        "fixture assumption broken: this fp64 value no longer survives the fp32 cast")
    dials = tuple(torch.full((2, w), dial_val, dtype=torch.float64) for w in widths)
    decay = LeafMassDecay(dials=dials)
    flat = leaf_rate_dials(decay, topology)
    assert torch.all(flat < 1.0)


# ---------------------------------------------------------------------------
# 13. build_clock_stream: stored != 0 => live != 0 (review fix pass, B2)
# ---------------------------------------------------------------------------


def test_build_clock_stream_rejects_stored_nonzero_with_live_zero():
    """B2: the bit-identical tile-skip argument in ``build_clock_stream``'s
    docstring requires ``stored != 0 => live != 0`` (equivalently, ``live == 0
    => stored == 0``). A caller-supplied pair that violates it -- the stored
    clock carries mass at a leaf/token the live deposit does not -- must be
    rejected loudly rather than silently letting a tile-skip drop that leaf's
    survival to an unchecked value."""
    widths = (4, 4)
    plan = _plan(widths, 16, 16, decay=True)
    B, T, H = 1, 16, 2
    live = tuple(torch.zeros(B, T, H, w) for w in widths)
    stored = tuple(torch.zeros(B, T, H, w) for w in widths)
    stored[0][0, 0, 0, 0] = 0.5  # nonzero in stored; live stays exactly zero everywhere
    with pytest.raises(ValueError, match="live"):
        build_clock_stream(stored, live, plan)


def test_build_clock_stream_accepts_stored_zero_with_live_nonzero():
    """Converse of B2, and the actual P5.20.5 finding-2 scenario the shipped
    ``quantize_stored`` fixture cannot reach by bf16 round-tripping a simplex
    (review F1): injected directly rather than via rounding, a leaf with a
    stored clock of exactly zero and a live deposit that is NOT zero is the
    allowed direction (quantization flushes toward zero, never away from it)
    and must NOT be rejected."""
    widths = (4, 4)
    plan = _plan(widths, 16, 16, decay=True)
    B, T, H = 1, 16, 2
    live = tuple(torch.full((B, T, H, w), 0.25) for w in widths)
    stored = tuple(t.clone() for t in live)
    stored[0][0, 0, 0, 0] = 0.0  # flushed to zero in storage; live is untouched, nonzero
    build_clock_stream(stored, live, plan)  # must not raise




def test_panel_exponent_bound_is_an_upper_bound_and_scales_as_derived():
    """``panel``'s admission bound (``intra_panel.cuh``, §12 rules 1 + 3).

    The bound is ``(1/2) * max_tile sum_t prod_l max_d c_l[t,d] * -log1p(-prod_l
    max_d delta_l[d])``. Three properties are gated, because each one is a
    different way for it to stop protecting the kernel:

    1. it is EXACT on a one-hot clock -- there the two maxima are attained by the
       same leaf, so the bound equals the true exponent and the test is checking a
       value rather than an inequality;
    2. it is an UPPER bound on a spread clock (the maxima are attained by
       different leaves, so it over-estimates -- which is the safe direction, and
       the direction has to be asserted or a sign error would pass);
    3. it is proportional to ``BT`` and to ``-log1p(-rate)``, i.e. it responds to
       both halves of the regime it exists to catch. Neither alone reaches the
       threshold; that is why every other decay fixture in the suite is far inside
       the domain."""
    import math

    from rola.planner.schedule import PANEL_MAX_EXPONENT, build_clock_stream, panel_exponent_bound
    from rola.routing.types import LeafMassDecay

    widths, BC, T, H = (4, 4), 16, 32, 1
    topology = _topology(widths, d_v=32)

    def _dials(rate):
        dial = rate ** (1.0 / len(widths))
        return LeafMassDecay(dials=tuple(torch.full((H, w), dial, dtype=torch.float64)
                                         for w in widths))

    def _clock(levels, BT):
        plan = Plan(topology=topology, normalization="raw", BT=BT, BC=BC,
                      warp_tile=(1, 1), decay=True)
        return build_clock_stream(levels, levels, plan)

    one_hot = tuple(
        torch.nn.functional.one_hot(torch.zeros(1, T, H, dtype=torch.long), w).double()
        for w in widths)
    spread = tuple(torch.full((1, T, H, w), 1.0 / w, dtype=torch.float64) for w in widths)

    rate = 0.9
    g = -math.log1p(-rate)
    for BT in (16, 32):
        exact = panel_exponent_bound(_clock(one_hot, BT), _dials(rate), BT)
        # One-hot clock: c == 1 at every token, so the per-tile prefix is exactly
        # BT and the reference exponent is exactly BT/2 * g.
        assert exact == pytest.approx(0.5 * BT * g, rel=1e-5)
        loose = panel_exponent_bound(_clock(spread, BT), _dials(rate), BT)
        true_max = 0.5 * BT * g * (1.0 / widths[0]) * (1.0 / widths[1])
        assert loose == pytest.approx(true_max, rel=1e-5)
        assert loose < exact

    # Both halves of the regime are required: a hot rate with a spread clock, and
    # a one-hot clock with a mild rate, are each far inside the domain.
    assert panel_exponent_bound(_clock(spread, 32), _dials(1 - 1e-6), 32) < PANEL_MAX_EXPONENT
    assert panel_exponent_bound(_clock(one_hot, 32), _dials(0.5), 32) < PANEL_MAX_EXPONENT
    # ...and the two together are not.
    assert panel_exponent_bound(_clock(one_hot, 32), _dials(1 - 1e-6), 32) > PANEL_MAX_EXPONENT


# ---------------------------------------------------------------------------
# 14. The planner's ONE-PASS statistics: exactness of the BT fold and of the
#     shared-view fusion (planner increment, 2026-07-30)
# ---------------------------------------------------------------------------


def _brute_force_tile_union_stats(read_levels, write_levels, topology, BC, BT):
    """CANONICAL definition of :class:`TileUnionStats`, transcribed from the docstring.

    Deliberately a from-the-definition scalar walk over ``(bh, owner, tile, leaf,
    token)`` -- no tiling tricks, no bitmaps, no fold -- so that it is an
    INDEPENDENT ground truth rather than a second spelling of the implementation.
    The `matching-the-naive` failure mode is the whole reason this exists: a fold
    validated against a re-derived fold would only prove self-consistency.

    ``K(owner, tile) = |{i : exists t in tile, live_r[t,i] or live_w[t,i]}|`` with
    ``live_x[t,i] = AND_l x_levels[l][t, digit_l(i)] != 0`` over ALL D levels (the
    spanned levels distinguish leafs, the fixed levels gate token validity).
    ``KI`` adds §5's causal condition: read at or after the first in-tile write.
    Candidacy is the §4 AND of per-level TILE bitmaps, with the boundary level's
    ``c`` digits ORed into one slice entry first.
    """
    geometry = owner_geometry(topology.widths, BC)
    widths = topology.widths
    D = topology.D
    j = geometry.first_spanned_level
    c = geometry.slice_digits
    B, T, H, _ = read_levels[0].shape
    n_tiles = (T + BT - 1) // BT

    r = [level.permute(0, 2, 1, 3).reshape(B * H, T, w) != 0
         for level, w in zip(read_levels, widths)]
    w_ = [level.permute(0, 2, 1, 3).reshape(B * H, T, w) != 0
          for level, w in zip(write_levels, widths)]

    def digit_of(owner, level, leaf):
        fixed_digits, slice_id = geometry.owner_digits(owner)
        if level < j:
            return fixed_digits[level]
        stride = max(1, math.prod(widths[level + 1:]))
        modulus = c if level == j else widths[level]
        base = slice_id * c if level == j else 0
        return base + (leaf // stride) % modulus

    n_pairs = 0
    sum_K = sum_sat = sum_cand = sum_Kc = sum_KIc = 0
    sum_ksc = sum_kisc = 0
    for bh in range(B * H):
        for owner in range(geometry.owners_total):
            _, slice_id = geometry.owner_digits(owner)
            fixed_digits, _ = geometry.owner_digits(owner)
            for tile in range(n_tiles):
                tokens = range(tile * BT, min(T, (tile + 1) * BT))
                K = KI = 0
                for leaf in range(BC):
                    seen_write = False
                    union = intra = False
                    for t in tokens:
                        lr = all(bool(r[l][bh, t, digit_of(owner, l, leaf)]) for l in range(D))
                        lw = all(bool(w_[l][bh, t, digit_of(owner, l, leaf)]) for l in range(D))
                        seen_write = seen_write or lw
                        union = union or lr or lw
                        intra = intra or (lr and seen_write)
                    K += int(union)
                    KI += int(intra)
                cand_r = any(bool(r[j][bh, t, slice_id * c + d])
                             for t in tokens for d in range(c))
                cand_w = any(bool(w_[j][bh, t, slice_id * c + d])
                             for t in tokens for d in range(c))
                for level in range(j):
                    cand_r = cand_r and any(bool(r[level][bh, t, fixed_digits[level]])
                                            for t in tokens)
                    cand_w = cand_w and any(bool(w_[level][bh, t, fixed_digits[level]])
                                            for t in tokens)
                cand = cand_r or cand_w
                n_pairs += 1
                sum_K += K
                sum_sat += int(K == BC)
                sum_cand += int(cand)
                sum_Kc += K * int(cand)
                sum_KIc += KI * int(cand)
                sum_ksc += -(-K // MMA_K_QUANTUM) * int(cand)
                sum_kisc += -(-KI // MMA_K_QUANTUM) * int(cand)
    return TileUnionStats(
        BC=BC, BT=BT,
        mean_K=sum_K / n_pairs, saturated_frac=sum_sat / n_pairs,
        mean_K_candidate=sum_Kc / sum_cand if sum_cand else 0.0,
        mean_KI_candidate=sum_KIc / sum_cand if sum_cand else 0.0,
        mean_ksteps_candidate=sum_ksc / sum_cand if sum_cand else 0.0,
        mean_kisteps_candidate=sum_kisc / sum_cand if sum_cand else 0.0,
        candidate_frac=sum_cand / n_pairs, n_pairs=n_pairs,
        n_cohorts=B * H * geometry.owners_total)


_FOLD_WIDTHS = [(2, 3), (4, 4), (2, 2, 2), (6,)]


@pytest.mark.parametrize("widths", _FOLD_WIDTHS, ids=lambda w: "w" + "x".join(map(str, w)))
@pytest.mark.parametrize("scenario", ["dense", "sparse", "union", "mixed", "zero_level"])
@pytest.mark.parametrize("T", [40, 64])
def test_the_one_pass_bt_fold_reproduces_the_canonical_definition(widths, scenario, T):
    """The fold to a coarser ``BT`` is EXACT, checked against the scalar definition.

    ``_tile_union_stats_multi_enumerated`` measures once at ``BT = 16`` and folds up to
    ``BT = 32`` (union/read-any/write-any bitmaps OR-pair; the causal intra bitmap
    additionally carries ``Rany_B AND Wany_A`` across the half boundary). Both
    candidates are compared, field for field, against
    ``_brute_force_tile_union_stats`` -- a from-the-definition walk that knows
    nothing about tiling halves. ``T = 40`` makes the fine tile count ODD (3 tiles
    at ``BT = 16``), which is the case where the fold pairs the last tile against a
    dead partner.
    """
    B, H = 1, 2
    topology = _topology(widths)
    read_levels, write_levels, _, _ = _gen_topology_tensors(11, widths, B, T, H, scenario)
    views = routing_views(read_levels, write_levels, topology)
    for BC in _admissible_bcs(widths):
        folded = _tile_union_stats_multi_enumerated(views, topology, BC, (16, 32))
        counted = _tile_union_stats_multi(views, topology, BC, (16, 32))
        for BT in (16, 32):
            reference = _brute_force_tile_union_stats(read_levels, write_levels,
                                                      topology, BC, BT)
            assert folded[BT] == reference, (
                f"widths={widths} scenario={scenario} T={T} BC={BC} BT={BT}: "
                f"folded {folded[BT]} != canonical {reference}")
            assert counted[BT] == reference, (
                f"widths={widths} scenario={scenario} T={T} BC={BC} BT={BT}: "
                f"support-counted {counted[BT]} != canonical {reference}")


def test_the_causal_carry_is_load_bearing_in_the_fold():
    """The fold's ``Rany_B AND Wany_A`` term is not decoration: drop it and ``KI`` is wrong.

    Fixture: a leaf written ONLY in the first 16 tokens and read ONLY in the second
    16. At ``BT = 16`` neither half has a causal read-after-write, so ``KI = 0`` in
    both; at ``BT = 32`` the whole tile has one, so ``KI > 0``. A naive
    ``I_A | I_B`` fold reports ``0`` and only the carry term recovers it. Asserted
    against the canonical definition, and the ``BT = 16`` legs are asserted too so
    the fixture provably has the property that makes the mutation visible.
    """
    widths = (1, 4)
    topology = _topology(widths)
    B, T, H = 1, 32, 1
    read_levels = [torch.zeros((B, T, H, w)) for w in widths]
    write_levels = [torch.zeros((B, T, H, w)) for w in widths]
    read_levels[0][:, :, :, 0] = 1.0
    write_levels[0][:, :, :, 0] = 1.0
    write_levels[1][0, 3, 0, 2] = 1.0      # written in the FIRST half only
    read_levels[1][0, 20, 0, 2] = 1.0      # read in the SECOND half only
    read_levels, write_levels = tuple(read_levels), tuple(write_levels)

    BC = 4
    views = routing_views(read_levels, write_levels, topology)
    folded = _tile_union_stats_multi_enumerated(views, topology, BC, (16, 32))
    counted = _tile_union_stats_multi(views, topology, BC, (16, 32))
    fine = _brute_force_tile_union_stats(read_levels, write_levels, topology, BC, 16)
    coarse = _brute_force_tile_union_stats(read_levels, write_levels, topology, BC, 32)

    assert fine.mean_KI_candidate == 0.0, "fixture must have NO in-half causal read"
    assert coarse.mean_KI_candidate > 0.0, "fixture must have a CROSS-half causal read"
    assert folded[16] == fine
    assert folded[32] == coarse
    assert counted[16] == fine, "the support-counted pass lost the in-tile causal read"
    assert counted[32] == coarse, "the support-counted min/max fold lost the cross-half read"


@pytest.mark.parametrize("widths", [(2, 3), (4, 4), (2, 2, 2)],
                         ids=lambda w: "w" + "x".join(map(str, w)))
@pytest.mark.parametrize("scenario", ["dense", "sparse", "mixed"])
def test_the_shared_view_changes_nothing_it_only_stops_recomputing(widths, scenario):
    """``plan_and_schedule`` returns the split path's plan and schedule, exactly.

    The fusion shares the fold, the ``!= 0`` mask and the §4 tile bitmaps between
    the selector and the builder. It must therefore be a pure work deletion: the
    plan compares equal field for field and every schedule tensor compares
    BIT-identical (``torch.equal``, not ``allclose`` -- the bitmaps are integers
    and the amplitudes are copies, so there is nothing for a tolerance to hide).
    """
    B, T, H = 1, 40, 2
    topology = _topology(widths)
    read_levels, write_levels, g_read, g_write = _gen_topology_tensors(
        23, widths, B, T, H, scenario)
    for BC in _admissible_bcs(widths):
        for decay in (False, True):
            for norm in ("raw", "global"):
                occ = _toy_occupancy()
                split_plan, split_axes = select_plan(
                    read_levels, write_levels, topology, BC, normalization=norm,
                    decay=decay, occupancy=occ)
                split_schedule = build_schedule(
                    read_levels, write_levels, g_read, g_write, split_plan)
                fused_plan, fused_axes, fused_schedule = plan_and_schedule(
                    read_levels, write_levels, g_read, g_write, topology, BC,
                    normalization=norm, decay=decay, occupancy=occ)
                assert fused_plan == split_plan
                assert fused_axes == split_axes
                for field in ("read_tile_bits", "write_tile_bits", "read_slice_union",
                              "write_slice_union", "owners", "owner_costs",
                              "g_read", "g_write"):
                    left = getattr(split_schedule, field)
                    right = getattr(fused_schedule, field)
                    if left is None or right is None:
                        assert left is None and right is None
                        continue
                    assert torch.equal(left, right), field
                for field in ("read_amps", "write_amps"):
                    for left, right in zip(getattr(split_schedule, field),
                                           getattr(fused_schedule, field)):
                        assert torch.equal(left, right), field
                assert (split_schedule.BH, split_schedule.T, split_schedule.n_tiles,
                        split_schedule.n_words, split_schedule.level_offsets) == (
                    fused_schedule.BH, fused_schedule.T, fused_schedule.n_tiles,
                    fused_schedule.n_words, fused_schedule.level_offsets)


@pytest.mark.parametrize("widths", _FOLD_WIDTHS + [(8, 16), (3, 5, 2)],
                         ids=lambda w: "w" + "x".join(map(str, w)))
@pytest.mark.parametrize("scenario", ["dense", "sparse", "union", "mixed", "zero_level"])
@pytest.mark.parametrize("T", [40, 64])
def test_the_support_counted_statistics_equal_the_enumerated_reference(widths, scenario, T):
    """The production pass and the demoted enumerator agree FIELD FOR FIELD.

    The two share no arithmetic. The reference asks of every ``(token, owner, leaf)``
    cell whether it is live and ORs the answers; the production pass expands the live
    entries out of the per-level supports and records where they land, then counts
    candidacy in closed form from the tile bitmaps' popcounts and asserts
    ``sum_K_candidate == sum_K``. So this is the increment's exactness gate, not a
    self-consistency check -- and every field is an exact int64 ratio, which is why it
    is compared with ``==`` rather than a tolerance.

    ``T = 40`` makes the finest tile count ODD (3 tiles at ``BT = 16``), the case where
    the coarse tile is half outside ``T``; ``zero_level`` and ``sparse`` are the
    scenarios where the candidate population and the ``K > 0`` population genuinely
    differ, which is what the ``sum_K_candidate == sum_K`` derivation stands on.
    """
    B, H = 1, 2
    topology = _topology(widths)
    read_levels, write_levels, _, _ = _gen_topology_tensors(11, widths, B, T, H, scenario)
    views = routing_views(read_levels, write_levels, topology)
    for BC in _admissible_bcs(widths):
        for candidates in ((16,), (32,), (16, 32)):
            counted = _tile_union_stats_multi(views, topology, BC, candidates)
            reference = _tile_union_stats_multi_enumerated(views, topology, BC, candidates)
            assert counted == reference, (
                f"widths={widths} scenario={scenario} T={T} BC={BC} "
                f"candidates={candidates}: {counted} != {reference}")


@_CUDA
@pytest.mark.parametrize("widths", [(4, 4), (2, 2, 2)],
                         ids=lambda w: "w" + "x".join(map(str, w)))
@pytest.mark.parametrize("scenario", ["sparse", "union", "zero_level"])
@pytest.mark.parametrize("T", [40, 64])
def test_the_support_counted_statistics_equal_the_enumerated_reference_on_cuda(
        widths, scenario, T):
    """The same bitwise gate, ON DEVICE, because the pass's reductions are not portable.

    The CPU grid above cannot see the surfaces production actually runs:
    `scatter_reduce_(reduce="amin"/"amax")` on `int16` (which is where every count in
    this pass comes from -- `first_write` and `last_read` ARE the statistics),
    `nonzero()`, and `repeat_interleave`. Each has a separate CUDA implementation, and
    a CPU-only gate would have declared them equal without ever running them.

    **The grid is a named subset, not a silent cap.** It is 12 parameter sets against
    the CPU grid's 60, chosen to span the axes that reach the scatter path rather than
    to sample uniformly: two MULTI-LEVEL width sets (so the mixed-radix decode runs
    with `D = 2` and `D = 3`, and the fixed-level AND is live at `D = 3`); `sparse`
    (few live entries, so the sentinels survive in most cells), `union` (read and
    write share a mask, so every live leaf has a causal read-after-write and the
    `first_write <= last_read` comparison is exercised on BOTH sides of its answer)
    and `zero_level` (a structurally dead level, i.e. an empty support list, where the
    expansion must produce nothing rather than index into an empty tensor); `T = 40`
    for the odd finest tile count and the padded tail; and BOTH candidate chains plus
    every admissible `BC` inside the body. Measured runtime of the 13 `..._on_cuda`
    tests is 1.03 s of call time against the file's ~18 s, and the largest device
    tensor is a few thousand elements -- VRAM-trivial, so the subset is about
    wall-clock proportion, not memory.
    """
    B, H = 1, 2
    topology = _topology(widths)
    read_levels, write_levels, _, _ = _gen_topology_tensors(11, widths, B, T, H, scenario)
    read_levels = tuple(level.cuda() for level in read_levels)
    write_levels = tuple(level.cuda() for level in write_levels)
    views = routing_views(read_levels, write_levels, topology)
    for BC in _admissible_bcs(widths):
        for candidates in ((16,), (32,), (16, 32)):
            counted = _tile_union_stats_multi(views, topology, BC, candidates)
            reference = _tile_union_stats_multi_enumerated(views, topology, BC, candidates)
            assert counted == reference, (
                f"CUDA widths={widths} scenario={scenario} T={T} BC={BC} "
                f"candidates={candidates}: {counted} != {reference}")


def _gen_sided_density_tensors(seed, widths, B, T, H, read_dense, write_dense):
    """Routing tensors whose REALIZED density matches a per-side declaration.

    A dense side is nowhere zero (the full-support activation's realization); a
    sparse side is a Bernoulli mask. Plain fixture plumbing, like
    :func:`_gen_topology_tensors` -- the declaration under test is stated by the
    caller, and this only makes it TRUE of the data, which is the closed form's
    stated precondition.
    """
    g = torch.Generator().manual_seed(seed)
    levels = {"read": [], "write": []}
    for side, dense in (("read", read_dense), ("write", write_dense)):
        for w in widths:
            shape = (B, T, H, w)
            values = torch.rand(shape, generator=g) + 0.01
            if not dense:
                values = values * (torch.rand(shape, generator=g) < 0.35)
            levels[side].append(values)
    return tuple(levels["read"]), tuple(levels["write"])


_DECLARATIONS = [(True, False), (False, True), (True, True), (False, False)]
_DECLARATION_IDS = ["read-dense", "write-dense", "dense-both", "sparse-both"]


@pytest.mark.parametrize("widths", [(4, 4), (8, 16), (3, 5, 2)],
                         ids=lambda w: "w" + "x".join(map(str, w)))
@pytest.mark.parametrize("declared", _DECLARATIONS, ids=_DECLARATION_IDS)
@pytest.mark.parametrize("T", [40, 64])
def test_the_static_by_config_closed_form_equals_the_enumerated_reference(
        widths, declared, T):
    """A declared-dense side's SUBSTITUTED statistics equal the independent definition.

    `StaticDensity` lets the pass replace a dense side's statistics with a closed
    form (``K = BC``, ``saturated_frac = 1``, ``candidate_frac = 1``,
    ``ksteps = ceil(BC/16)``) instead of measuring a rectangle the configuration
    already determines. This gates that substitution against
    ``_tile_union_stats_multi_enumerated`` -- the from-the-fold-identity walk that
    shares no arithmetic with either the closed form or the production pass -- and
    NOT against the production pass on its own, per
    `[[matching-the-naive-antipattern]]`: agreeing with the implementation the
    closed form is meant to replace would only prove self-consistency. Equality
    with the production pass is asserted too, because bit-identity there is the
    increment's own contract.

    ``T = 40`` keeps the odd finest tile count and the half-outside coarse tile, the
    case where "every tile that holds a token is saturated" has to exclude the
    padded tiles the buffers carry and the ``n_pairs`` denominator has to agree.
    """
    B, H = 1, 2
    read_dense, write_dense = declared
    topology = _topology(widths)
    read_levels, write_levels = _gen_sided_density_tensors(
        17, widths, B, T, H, read_dense, write_dense)
    views = routing_views(read_levels, write_levels, topology)
    density = StaticDensity(read=(read_dense,) * len(widths),
                            write=(write_dense,) * len(widths))
    for BC in _admissible_bcs(widths):
        for candidates in ((16,), (32,), (16, 32)):
            static = _tile_union_stats_multi(views, topology, BC, candidates, density)
            reference = _tile_union_stats_multi_enumerated(views, topology, BC, candidates)
            measured = _tile_union_stats_multi(views, topology, BC, candidates)
            assert static == reference, (
                f"widths={widths} declared={declared} T={T} BC={BC} "
                f"candidates={candidates}: {static} != {reference}")
            assert static == measured, (
                f"declaring a side dense changed the answer: {static} != {measured}")


@_CUDA
@pytest.mark.parametrize("declared", _DECLARATIONS, ids=_DECLARATION_IDS)
def test_the_static_by_config_closed_form_equals_the_reference_on_cuda(declared):
    """The same substitution, ON DEVICE, for the same reason the gate above it is.

    The remaining live side still runs `nonzero`, `repeat_interleave` and the
    `int16` `scatter_reduce_`, and the one-side-dense arm reaches a DIFFERENT
    predicate on the survivor (``first_write < BT_max`` / ``last_read >= 0``) than
    the two-sided arm does, so the device implementations of those reductions are
    on a path the CPU grid never runs.
    """
    B, H, T, widths = 1, 2, 40, (4, 4)
    read_dense, write_dense = declared
    topology = _topology(widths)
    read_levels, write_levels = _gen_sided_density_tensors(
        17, widths, B, T, H, read_dense, write_dense)
    read_levels = tuple(level.cuda() for level in read_levels)
    write_levels = tuple(level.cuda() for level in write_levels)
    views = routing_views(read_levels, write_levels, topology)
    density = StaticDensity(read=(read_dense,) * len(widths),
                            write=(write_dense,) * len(widths))
    for BC in _admissible_bcs(widths):
        for candidates in ((16,), (32,), (16, 32)):
            static = _tile_union_stats_multi(views, topology, BC, candidates, density)
            reference = _tile_union_stats_multi_enumerated(views, topology, BC, candidates)
            assert static == reference, (
                f"CUDA declared={declared} BC={BC} candidates={candidates}: "
                f"{static} != {reference}")


def test_static_density_reads_the_activation_registry_not_a_name_list():
    """`from_topology` is the `exact_zeros` COLUMN, per level and per duty.

    Softmax cannot produce a structural zero (dense by construction); exact entmax
    can (genuinely routed). The mixed topology is the point: the flags are per
    LEVEL, and `side_dense` is the conjunction, because a leaf is live only where
    EVERY level's digit is -- one dense level among sparse ones does not make the
    side's support the full rectangle.
    """
    sparse = IndependentRouting(tied=True, read_activation=EntmaxActivation(1.5),
                                write_activation=EntmaxActivation(1.5))
    mixed = Topology(levels=(LevelConfig(width=4, routing=_routing()),
                             LevelConfig(width=4, routing=sparse)), d_v=4)
    density = StaticDensity.from_topology(mixed)
    assert density.read == (True, False) and density.write == (True, False)
    assert not density.side_dense("read") and not density.side_dense("write")

    dense = StaticDensity.from_topology(_topology((4, 4)))
    assert dense.read == (True, True) and dense.write == (True, True)
    assert dense.side_dense("read") and dense.side_dense("write")
    with pytest.raises(ValueError, match="duty must be"):
        dense.side_dense("both")


def test_a_partly_dense_side_does_not_take_the_closed_form():
    """One dense level among sparse ones must still be MEASURED, not assumed.

    The failure this forbids is the tempting one: the closed form's precondition is
    that the side's support is the whole rectangle, which needs EVERY level dense.
    Gated by equality with the enumerated reference on a side that is dense at one
    level and genuinely sparse at the other -- if `side_dense` were an `any`, `K`
    would come back at `BC` on a cell whose realized union is far below it.
    """
    B, H, T, widths, BC = 1, 2, 40, (4, 4), 4
    topology = _topology(widths)
    g = torch.Generator().manual_seed(23)
    def level(w, dense):
        values = torch.rand((B, T, H, w), generator=g) + 0.01
        # 5 % live: sparse enough that the 16-token tile UNION does not saturate on
        # its own, which is what makes the assertion below discriminating.
        return values if dense else values * (torch.rand((B, T, H, w), generator=g) < 0.05)
    read_levels = (level(4, True), level(4, False))
    write_levels = (level(4, False), level(4, False))
    views = routing_views(read_levels, write_levels, topology)
    density = StaticDensity(read=(True, False), write=(False, False))
    assert not density.side_dense("read")
    static = _tile_union_stats_multi(views, topology, BC, (16, 32), density)
    reference = _tile_union_stats_multi_enumerated(views, topology, BC, (16, 32))
    assert static == reference
    assert static[16].mean_K < BC, "the fixture must not be accidentally saturated"


@_CUDA
def test_the_causal_carry_is_load_bearing_in_the_fold_on_cuda():
    """The §5 causal fixture, on device, against the canonical definition.

    The scenario grid above cannot isolate this: it needs a leaf written ONLY in the
    first half of a coarse tile and read ONLY in the second, so that `KI` is zero at
    `BT = 16` and nonzero at `BT = 32`. That is the one case where the production
    pass's `min`/`max` fold and the reference's `Rany_B AND Wany_A` carry could
    disagree, and on CUDA it is also the case where the `amin` and `amax` scatters
    must land in the same tile cell from two different entries.
    """
    widths = (1, 4)
    topology = _topology(widths)
    B, T, H = 1, 32, 1
    read_levels = [torch.zeros((B, T, H, w), device="cuda") for w in widths]
    write_levels = [torch.zeros((B, T, H, w), device="cuda") for w in widths]
    read_levels[0][:, :, :, 0] = 1.0
    write_levels[0][:, :, :, 0] = 1.0
    write_levels[1][0, 3, 0, 2] = 1.0      # written in the FIRST half only
    read_levels[1][0, 20, 0, 2] = 1.0      # read in the SECOND half only
    read_levels, write_levels = tuple(read_levels), tuple(write_levels)

    BC = 4
    counted = _tile_union_stats_multi(
        routing_views(read_levels, write_levels, topology), topology, BC, (16, 32))
    fine = _brute_force_tile_union_stats(
        tuple(t.cpu() for t in read_levels), tuple(t.cpu() for t in write_levels),
        topology, BC, 16)
    coarse = _brute_force_tile_union_stats(
        tuple(t.cpu() for t in read_levels), tuple(t.cpu() for t in write_levels),
        topology, BC, 32)
    assert fine.mean_KI_candidate == 0.0, "fixture must have NO in-half causal read"
    assert coarse.mean_KI_candidate > 0.0, "fixture must have a CROSS-half causal read"
    assert counted[16] == fine
    assert counted[32] == coarse


@pytest.mark.parametrize("widths", _FOLD_WIDTHS, ids=lambda w: "w" + "x".join(map(str, w)))
@pytest.mark.parametrize("scenario", ["dense", "sparse", "union", "mixed", "zero_level"])
def test_the_quanta_carry_what_the_means_cannot(widths, scenario):
    """§6 prices ``ceil(K/16)``, and this pins the three properties that make it a
    DIFFERENT input rather than a rescaling of ``mean_K_candidate``.

    1. **The quantum brackets the mean, on both sides.** A candidate tile with
       ``K = 1`` pays a whole k-step (so ``mean_ksteps >= mean_K / 16``) and one
       with ``K = 0`` pays none, and no tile can exceed ``ceil(BC/16)``.
    2. **``KI``'s quantum can never exceed the union's**, because ``KI`` is a causal
       sub-list of the same per-tile union.
    3. **``n_cohorts * n_tiles == n_pairs``.** The cohort count is the CTA grid and
       the pair count is the grid times the tile loop; a wave term reading the first
       and a work term reading the second must stay consistent, and this is where
       that is checked rather than assumed.
    """
    B, T, H = 1, 64, 2
    topology = _topology(widths)
    read_levels, write_levels, _, _ = _gen_topology_tensors(29, widths, B, T, H, scenario)
    views = routing_views(read_levels, write_levels, topology)
    for BC in _admissible_bcs(widths):
        for BT, stats in _tile_union_stats_multi(views, topology, BC, (16, 32)).items():
            ceil_bc = -(-BC // MMA_K_QUANTUM)
            assert stats.mean_K_candidate / MMA_K_QUANTUM <= stats.mean_ksteps_candidate + 1e-12
            assert stats.mean_ksteps_candidate <= ceil_bc + 1e-12
            assert stats.mean_kisteps_candidate <= stats.mean_ksteps_candidate + 1e-12, (
                "KI is a causal SUB-list of the union, so its quantum can never "
                "exceed the union's")
            n_tiles = (T + BT - 1) // BT
            assert stats.n_cohorts * n_tiles == stats.n_pairs, (
                f"BC={BC} BT={BT}: {stats.n_cohorts} cohorts x {n_tiles} tiles != "
                f"{stats.n_pairs} pairs")
            assert stats.n_cohorts == B * H * owner_geometry(widths, BC).owners_total


def test_the_quantum_is_not_a_rescaling_of_the_mean():
    """The gap that motivates the field, exhibited on a fixture that has it.

    Every candidate tile here saturates while half the ``(owner, tile)`` pairs are
    dead, so the CANDIDATE-population quantum is ``BC/16`` while a model that
    divided the ALL-pairs ``mean_K`` by 16 would price half of it. That is the same
    shape as the flagship's measured gap (``mean_K_candidate/BC = 0.9724`` against
    ``mean_ksteps_candidate/(BC/16) = 0.6087``, where 39 % of candidate pairs gather
    ``K = 0``) and it is why the quantum is accumulated per pair rather than derived.
    """
    widths = (4, 4)
    BC, BT, T, B, H = 16, 16, 64, 1, 1
    topology = _topology(widths)
    read_levels = [torch.zeros((B, T, H, w)) for w in widths]
    write_levels = [torch.zeros((B, T, H, w)) for w in widths]
    # Every leaf live on the EVEN tiles, nothing on the odd ones.
    for tile in range(0, T // BT, 2):
        for t in range(tile * BT, (tile + 1) * BT):
            for level in range(len(widths)):
                read_levels[level][0, t, 0, :] = 1.0
                write_levels[level][0, t, 0, :] = 1.0
    read_levels, write_levels = tuple(read_levels), tuple(write_levels)
    views = routing_views(read_levels, write_levels, topology)
    stats = _tile_union_stats_multi(views, topology, BC, (BT,))[BT]
    assert stats.candidate_frac == 0.5, "half the pairs must be dead for the gap to exist"
    assert stats.mean_K_candidate == float(BC)
    assert stats.mean_ksteps_candidate == float(BC // MMA_K_QUANTUM)
    assert stats.mean_K == float(BC) / 2.0
    assert stats.mean_K / MMA_K_QUANTUM != stats.mean_ksteps_candidate


def test_the_support_counted_statistics_are_invariant_to_the_chunk_budget(monkeypatch):
    """Chunking is a memory bound, so it may not move a single count.

    The pass chunks over whole ``max(bt_candidates)`` tiles -- the coarsest
    granularity at which every candidate's tile still nests inside one chunk. That
    invariant is exactly what a chunk-boundary bug would break, and it is invisible at
    the default budget (the fixtures below run in ONE chunk). Squeezing the budget to
    a single tile-cell forces the maximum number of chunk boundaries and the answer
    must not move by one count.
    """
    widths = (4, 4)
    B, T, H = 1, 96, 2
    topology = _topology(widths)
    read_levels, write_levels, _, _ = _gen_topology_tensors(7, widths, B, T, H, "mixed")
    views = routing_views(read_levels, write_levels, topology)
    for BC in _admissible_bcs(widths):
        whole = _tile_union_stats_multi(views, topology, BC, (16, 32))
        with monkeypatch.context() as patch:
            patch.setattr(
                "rola.planner.schedule._STATS_TILE_CELL_CHUNK", 1)
            patch.setattr(
                "rola.planner.schedule._STATS_ENTRY_CHUNK", 1)
            split = _tile_union_stats_multi(views, topology, BC, (16, 32))
        assert whole == split, f"BC={BC}: chunking moved the statistics"


def test_the_support_counted_pass_never_materializes_an_o_of_l_times_n_object(monkeypatch):
    """The production pass's two transients are BOUNDED, and neither is ``BH * T * N``.

    It has no owner axis to chunk (owners are a reshape of the leaf axis), so the
    objects to bound are different from the reference's: the live ENTRIES it expands
    and the ``(tile, leaf)`` CELLS it reduces over. The second is the one that scales
    with ``N``, and it is ``BT`` times smaller than the ``(token, leaf)`` rectangle by
    construction -- ``widths=(64,64,16)`` at ``T = 512`` gives ``BH * T * N = 33 M``
    against a tile plane of ``2 M``.

    The mutation check squeezes the cell budget instead of raising it: the guard is
    set to the tile plane the DEFAULT budget produces, and forcing a smaller chunk
    must drop every allocation below it. That direction is the honest one here --
    raising the budget cannot manufacture a violation, because the un-chunked plane
    is already what the default admits at this fixture.
    """
    widths = (64, 64, 16)
    B, T, H, BC = 1, 512, 1, 16
    topology = _topology(widths)
    N = math.prod(widths)
    read_levels, write_levels, _, _ = _gen_topology_tensors(5, widths, B, T, H, "sparse")
    views = routing_views(read_levels, write_levels, topology)

    forbidden = B * H * T * N
    guard = _AllocationGuard(forbidden - 1)
    with guard:
        _tile_union_stats_multi(views, topology, BC, (16, 32))
    assert not guard.violations, (
        f"the support-counted pass built an object at or above BH*T*N={forbidden}: "
        f"{guard.violations[:5]}")

    tile_plane = B * H * ((T + 31) // 32) * 2 * N
    squeezed = _AllocationGuard(tile_plane // 4)
    with monkeypatch.context() as patch:
        patch.setattr("rola.planner.schedule._STATS_TILE_CELL_CHUNK",
                      tile_plane // 8)
        with squeezed:
            _tile_union_stats_multi(views, topology, BC, (16, 32))
    assert not squeezed.violations, (
        "mutation check: squeezing the cell budget must shrink the tile plane it "
        f"allocates, but tensors above {tile_plane // 4} survived: "
        f"{squeezed.violations[:5]}")
    loose = _AllocationGuard(tile_plane // 4)
    with loose:
        _tile_union_stats_multi(views, topology, BC, (16, 32))
    assert loose.violations, (
        "mutation check is broken: at the DEFAULT budget this fixture must allocate "
        f"a tile plane above {tile_plane // 4}, or the squeezed assertion is vacuous")


def test_the_enumerated_reference_never_materializes_an_o_of_l_times_n_object(monkeypatch):
    """The reference's owner axis is CHUNKED, so its transient does not scale with ``N``.

    Gates :func:`_tile_union_stats_multi_enumerated`, which is the TEST-ONLY
    reference since the planner increment; the production pass has no owner axis to
    chunk and is gated by
    ``test_the_support_counted_pass_never_materializes_an_o_of_l_times_n_object``.

    Vectorizing the owner loop is what removed the per-owner synchronization, but
    the whole-owner-axis form would build ``[BH, n_tiles, BT, owners, BC]``, i.e.
    ``BH * T * N`` -- exactly the ``O(L*N)`` object §4 forbids. ``widths=(64,64,16)``
    gives ``N = 65536``; with ``BC = 16`` that is 4096 owners.

    ``T`` must be large enough that ``chunk < owners_total``, i.e. that the code
    actually takes the chunked branch (``chunk = budget // per_owner``, so
    ``chunk * per_owner <= budget`` is an algebraic identity that holds even when
    chunking is deleted and the whole owner axis is built in one shot -- that
    form only exceeds the budget once ``BH*T*N`` itself exceeds it). At ``T=32``
    the un-chunked whole-owner tensor is ``BH*T*N = 2,097,152``, still under the
    16,777,216-element budget, so the guard cannot distinguish the two paths.
    ``T=512`` (``BH*T*N = 33,554,432``) does: the real chunked code stays flat
    against the budget while the un-chunked mutant below does not (measured
    ``planner.md``/`planner-review.md` §4: 0 vs 12 violations).
    """
    widths = (64, 64, 16)
    B, T, H, BC = 1, 512, 1, 16
    topology = _topology(widths)
    N = math.prod(widths)
    owners_total = N // BC
    read_levels, write_levels, _, _ = _gen_topology_tensors(5, widths, B, T, H, "sparse")
    bound = _STATS_OWNER_CHUNK_ELEMS
    views = routing_views(read_levels, write_levels, topology)

    finest_BT = 16
    n_tiles = (T + finest_BT - 1) // finest_BT
    per_owner = B * H * n_tiles * finest_BT * BC
    chunk = max(1, min(owners_total, bound // per_owner))
    assert chunk < owners_total, (
        "fixture does not exercise the chunked path: raise T further "
        f"(chunk={chunk}, owners_total={owners_total})")

    # The measured allocation must respect the fixed per-chunk element budget
    # itself -- not the algebraic identity `chunk * per_owner <= budget`, which
    # holds regardless of whether chunking actually runs.
    guard = _AllocationGuard(bound)
    with guard:
        _tile_union_stats_multi_enumerated(views, topology, BC, (16, 32))
    assert not guard.violations, (
        f"statistics pass allocated tensors above the chunk budget {bound} "
        f"(N={N}, B*H*T*N={B * H * T * N}): {guard.violations[:5]}")

    # Mutation check: deleting chunking (raising the chunk-size constant so the
    # whole owner axis is built in one call) must make the guard fire at this
    # fixture -- otherwise the assertion above would be vacuous.
    monkeypatch.setattr(
        "rola.planner.schedule._STATS_OWNER_CHUNK_ELEMS", 1 << 62)
    mutant_guard = _AllocationGuard(bound)
    with mutant_guard:
        _tile_union_stats_multi_enumerated(views, topology, BC, (16, 32))
    assert mutant_guard.violations, (
        "mutation check is broken: deleting chunking should have produced at "
        "least one tensor above the chunk budget at this fixture, but none was "
        "recorded -- the primary assertion above would be vacuous")




def test_the_production_default_mma_precision_is_bf16_on_every_plan_entry_point():
    """The shipped numeric format is `bf16`, at every door a plan can come through.

    `harvest.md` §30 measured bf16 at -9.5 % at the at-scale cell, -7.3 % mid and
    -6.6 % at the gram crossover, with the 43 bf16 correctness gates green at their
    own declared `BF16_RTOL` (untouched) -- so it ships as the default. The three
    entry points must agree, because a plan built by `select_plan` and one built
    by hand would otherwise instantiate different kernels for the same cell.

    The counterpart of this gate is that every fp64-ORACLE exactness comparison pins
    `tf32x3` explicitly (see `MMA_PRECISIONS` and `v3_readout_mma.cuh`): bf16's ~1e-2
    per-product error would swamp the structural disagreement those gates detect. If
    this assertion is ever flipped back, those pins are what keeps them honest.
    """
    topology = _topology((4, 4))
    B, T, H, BC = 1, 40, 2, 4
    read_levels, write_levels, g_read, g_write = _gen_topology_tensors(
        31, (4, 4), B, T, H, "sparse")

    assert Plan(topology=topology, normalization="global", BT=16, BC=BC,
                  warp_tile=(1, 1)).mma_precision == "bf16"
    selected, _ = select_plan(read_levels, write_levels, topology, BC,
                                 normalization="global", decay=False,
                                 occupancy=_toy_occupancy())
    assert selected.mma_precision == "bf16"
    fused, _, _ = plan_and_schedule(read_levels, write_levels, g_read, g_write,
                                       topology, BC, normalization="global",
                                       decay=False, occupancy=_toy_occupancy())
    assert fused.mma_precision == "bf16"
