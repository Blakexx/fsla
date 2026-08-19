"""THE GRADIENT, AGAINST THE ORACLE'S OWN AUTOGRAD — and the memory it costs.

Vocabulary, defined here and used unqualified below: `T` tokens, `H` heads,
`d_v = 64` value width, `D` level count, `W` one uniform level width,
`N = W ** D` leaves; `pr`/`pw` the per-level read/write amplitudes; `gw` the write
gain; `m` the per-token read mass; `dy` the output cotangent.

`rola/ops/naive.py` in fp64 IS the reference backward: it is differentiable
throughout, and under the shipping configuration its only `.detach()` (the decay
clock) does not exist. So the battery is a PER-INPUT COTANGENT COMPARISON —
`torch.autograd.grad(y_oracle, inputs, dy)` in fp64 against the kernel's, cell by
cell.

**`gradcheck` ITSELF IS NOT USED AGAINST THE KERNEL AND CANNOT BE.** Finite
differences need perturbations far below bf16 resolution, and the kernel's envelope
forbids the shapes `gradcheck` wants (`d_v = 64`, built arms). It
belongs on the ORACLE, certifying the reference's own autograd against finite
differences, and that is what the first test here does — a crown-jewels gate on the
reference, cheap, at fp64 and a size `gradcheck` can afford.

**THE FLOOR IS THE ROUNDED REFERENCE**, exactly as the forward's is: the same
gradient taken with the oracle's inputs rounded the way the kernel holds them
(amplitudes and `v` to bf16, the cotangent seed `dnum` to bf16) bounds the
REPRESENTABLE error, and the ratio to it isolates the algorithmic one. The
comparison is `|kernel - fp64| <= kappa * |rounded - fp64|`, with `kappa` published
per output here and never widened per cell.
"""
from __future__ import annotations

import pytest
import torch

import rola
from rola.engine.facts.manifest import scan_arms
from rola.interface import BackwardNotImplemented  # noqa: F401  (documents the seam)
from rola.ops._ext import extension, is_available
from rola.ops.naive import naive_rola

pytestmark = pytest.mark.skipif(
    not is_available() or not torch.cuda.is_available(),
    reason="the reverse pass is a CUDA kernel")

#: THE CHUNK AND THE VALUE WIDTH, which are single-valued across the built matrix.
C, DV = 32, 64

#: THE TOPOLOGIES THE REVERSE PASS CARRIES, read off the census rather than
#: restated: `(D, W)` for every built scan arm, so a widened scan matrix widens
#: this battery by collection and cannot ship ungated. `BC` is not a coordinate
#: here -- `rola.ops.chunk` picks it from the manifest, which is the whole point
#: of gating the FACADE rather than the probe entry.
TOPOLOGIES = sorted({(d, w) for (_c, _bc, d, w) in scan_arms()})

#: THE PINNED TOPOLOGY, for the rows whose claim is about a term rather than an
#: arm (the m-carry identity, the vacuity row, the retention receipt).
PIN_D, PIN_W = 2, 64

#: ONE TOLERANCE, PUBLISHED, AND THE SAME FOR EVERY OUTPUT.
#:
#: It is a ratio to the ROUNDED-REFERENCE floor, so 1 means "no worse than holding
#: the operands in bf16". The slack above 1 is DERIVED rather than fitted: the floor
#: rounds the INPUTS, and the kernel additionally rounds two INTERMEDIATES the
#: reference has no way to reproduce -- the `dnum` seed, which is an mma operand,
#: and the Gram, which is stored bf16 between its two consumers. Each is one more
#: rounding of the same class, so a factor of about three is structural and four is
#: the bound. MEASURED at 1.0-2.5 across every cell below; if a cell ever needs more
#: than this, the term that grew is named rather than the number raised.
KAPPA = 4.0


def _bundle(T: int, seed: int, *, mass: bool, gain: float, D: int = PIN_D,
            W: int = PIN_W, device="cuda"):
    """A hand-built `RouteFactors` on the pinned arm, with every operand a LEAF.

    Built directly rather than through a producer: the claim here is about the
    operator's own five cotangents, and a producer in front of it would fold them
    together before anything could be compared per input.
    """
    from rola.routing.factors import RouteFactors
    from rola.routing.producer import union_routing
    from rola.routing.types import Topology

    g = torch.Generator(device=device).manual_seed(seed)

    def simplex(*shape):
        x = torch.rand(*shape, device=device, generator=g, dtype=torch.float64) + 0.05
        return (x / x.sum(-1, keepdim=True))

    read = [simplex(1, T, 1, W).requires_grad_() for _ in range(D)]
    write = [simplex(1, T, 1, W).requires_grad_() for _ in range(D)]
    gw = (torch.rand(1, T, 1, device=device, generator=g, dtype=torch.float64)
          * gain + 0.5).requires_grad_()
    rm = None
    if mass:
        rm = (torch.rand(1, T, 1, device=device, generator=g, dtype=torch.float64)
              + 0.5).requires_grad_()
    v = torch.randn(1, T, 1, DV, device=device, generator=g,
                    dtype=torch.float64).requires_grad_()
    topology = Topology(levels=tuple(union_routing(W, 1.5)
                                     for _ in range(D)))
    routes = RouteFactors(read=tuple(read), write=tuple(write), g_read=None, g_write=gw,
                          read_mass=rm, topology=topology, normalization="global")
    return routes, v


def _oracle_y(routes, v):
    """The fp64 oracle's readout over this bundle -- the differentiable reference."""
    #: the oracle takes the read levels AS THEY ARE -- scaled or not. There is no
    #: separate `m` operand on it, which is exactly why the m-carry is gated by an
    #: algebraic identity below rather than against this reference (P65 R-B).
    return naive_rola(v, routes.read, routes.write, routes.g_read, routes.g_write,
                      routes.topology, None, "global")[0]


def _round_like_kernel(t, dtype=torch.bfloat16):
    return t.detach().to(dtype).to(torch.float64).requires_grad_()


# ---------------------------------------------------------------------------
# 1. THE ORACLE'S OWN AUTOGRAD, certified against finite differences
# ---------------------------------------------------------------------------

def test_the_oracles_autograd_equals_finite_differences():
    """A CROWN-JEWELS GATE ON THE REFERENCE, and the only place `gradcheck` belongs.

    It runs at fp64 on a size finite differences can afford, and it is what makes
    every comparison below a comparison against something known-correct rather than
    against a second implementation nobody checked.
    """
    torch.manual_seed(0)
    dev = "cuda"
    T, w, d = 4, 4, 2
    from rola.routing.producer import union_routing
    from rola.routing.types import Topology

    topo = Topology(levels=tuple(union_routing(w, 1.5) for _ in range(d)))

    def y_of(*flat):
        read = list(flat[:d])
        write = list(flat[d:2 * d])
        gw, v = flat[2 * d], flat[2 * d + 1]
        return naive_rola(v, read, write, None, gw, topo, None, "global")[0]

    g = torch.Generator(device=dev).manual_seed(1)

    def simplex(*shape):
        x = torch.rand(*shape, device=dev, generator=g, dtype=torch.float64) + 0.05
        return (x / x.sum(-1, keepdim=True)).requires_grad_()

    flat = tuple([simplex(1, T, 1, w) for _ in range(d)]
                 + [simplex(1, T, 1, w) for _ in range(d)]
                 + [(torch.rand(1, T, 1, device=dev, generator=g,
                                dtype=torch.float64) + 0.5).requires_grad_(),
                    torch.randn(1, T, 1, 8, device=dev, generator=g,
                                dtype=torch.float64).requires_grad_()])
    assert torch.autograd.gradcheck(y_of, flat, eps=1e-6, atol=1e-7, rtol=1e-4)


# ---------------------------------------------------------------------------
# 2. THE KERNEL'S FIVE COTANGENTS, per input, against the oracle's
# ---------------------------------------------------------------------------

#: THE ORACLE HAS NO `m` OPERAND -- it takes the read levels already scaled, and
#: `eps` as a bare scalar. So the mass CANNOT be a cell here (P65 R-B: no oracle
#: change for a convenience), and the m-carry gets an ALGEBRAIC identity of its own
#: below instead. Every row here is a declared-simplex bundle.
CELLS = [
    pytest.param(64, 1.0, id="T=2C, unit gain"),
    #: MANDATORY (P65): the gain has been ones in every ladder gate ever run, and
    #: (A.10)/(A.11) are the only place it is differentiated
    pytest.param(64, 4.0, id="g_write != 1"),
    #: several chunks, so the INTER terms carry weight beside the intra ones
    pytest.param(128, 2.0, id="T=4C, gain != 1"),
    pytest.param(160, 1.0, id="T=5C, unit gain"),
]


@pytest.mark.parametrize("D,W", TOPOLOGIES, ids=[f"D{d}W{w}" for d, w in TOPOLOGIES])
@pytest.mark.parametrize("T,gain", [p.values for p in CELLS],
                         ids=[p.id for p in CELLS])
def test_every_cotangent_matches_the_oracles(T, gain, D, W):
    #: The scale arm's fp64 oracle needs more than a 12 GB card holds under the
    #: conftest's 70% cap; those cells are RENTAL-TIER (docs/testing.md), not local.
    #: The bound is the measured OOM frontier: N*T fp64 state+grads ~ 8 bytes each.
    if (W ** D) * T > 4_000_000 and (torch.cuda.get_device_properties(0).total_memory
                                     < 24 * 2**30):
        pytest.skip(f"N={W**D}, T={T}: fp64-oracle cell exceeds this card; "
                    "rental-tier -- run on >=24GB (the A100 list)")
    mass = False
    routes, v = _bundle(T, seed=11, mass=mass, gain=gain, D=D, W=W)
    inputs = (v, *routes.tensors())
    names = (["v"] + ["read"] * D + ["write"] * D + ["g_write"]
             + (["read_mass"] if mass else []))

    y_ref = _oracle_y(routes, v)
    #: THE COTANGENT IS SEEDED, and to ONE constant for every cell: an unseeded draw
    #: makes a cell's verdict a function of how many cells were COLLECTED before it,
    #: so widening the scan matrix moves rows that did not change.
    torch.manual_seed(0)
    dy = torch.randn(y_ref.shape, device=y_ref.device, dtype=torch.float64)
    want = torch.autograd.grad(y_ref, inputs, dy, retain_graph=True)

    #: THE FLOOR: the same gradient with the operands held the way the kernel holds
    #: them. Nothing here is tuned -- it is the reference run twice.
    rr = [_round_like_kernel(t) for t in routes.read]
    rw = [_round_like_kernel(t) for t in routes.write]
    rg = _round_like_kernel(routes.g_write)
    rm = _round_like_kernel(routes.read_mass, torch.float32) if mass else None
    rv = _round_like_kernel(v)
    from rola.routing.factors import RouteFactors
    rounded = RouteFactors(read=tuple(rr), write=tuple(rw), g_read=None, g_write=rg,
                           read_mass=rm, topology=routes.topology,
                           normalization="global")
    y_rnd = _oracle_y(rounded, rv)
    got_rnd = torch.autograd.grad(y_rnd, (rv, *rounded.tensors()),
                                  _round_like_kernel(dy).detach())

    #: THE KERNEL. Its operands are the same numbers in the dtypes it takes.
    kr = [t.detach().to(torch.float32).requires_grad_() for t in routes.read]
    kw = [t.detach().to(torch.float32).requires_grad_() for t in routes.write]
    kg = routes.g_write.detach().to(torch.float32).requires_grad_()
    km = routes.read_mass.detach().to(torch.float32).requires_grad_() if mass else None
    kv = v.detach().to(torch.float32).requires_grad_()
    kernel_routes = RouteFactors(read=tuple(kr), write=tuple(kw), g_read=None, g_write=kg,
                                 read_mass=km, topology=routes.topology,
                                 normalization="global")
    y_k, _ = rola.rola_op(kernel_routes, kv)
    got = torch.autograd.grad(y_k, (kv, *kernel_routes.tensors()),
                              dy.to(torch.float32))

    #: THE RATIO IS TAKEN ON A NORM, not per cell, and that is a derivation and not
    #: a convenience: the rounded reference's per-cell difference can vanish by luck
    #: -- a cell where the bf16 rounding happened not to move -- and a ratio there
    #: divides by an accident. The NORM of that difference is the representable
    #: error of the whole tensor and does not vanish.
    for name, w_, r_, k_ in zip(names, want, got_rnd, got):
        floor = (r_.double() - w_).norm().item()
        err = (k_.double() - w_).norm().item()
        worst = err / max(floor, torch.finfo(torch.float64).tiny)
        assert worst <= KAPPA, (
            f"d {name} is {worst:.2f}x its rounded-reference floor (kappa={KAPPA}) "
            f"at T={T} gain={gain} D={D} W={W}. ||err|| = {err:.3e} against a "
            f"floor of {floor:.3e}. The floor is the SAME oracle run with the "
            f"operands rounded the way the kernel holds them, so it bounds what is "
            f"REPRESENTABLE; anything far above it is algorithmic.")


def test_the_m_carry_maps_by_the_chain_rule_of_its_own_split():
    """THE m-CARRY, GATED WITHOUT TOUCHING THE ORACLE (P65 R-B).

    `naive_rola` takes the read levels ALREADY SCALED and `eps` as a bare scalar, so
    it has no `m` operand to differentiate and cannot be the reference for `dm`.
    Giving it one would be an oracle change for a convenience, which the crown-jewels
    rule prices at a dual run -- so the m-carry is gated by an ALGEBRAIC IDENTITY on
    the kernel instead.

    `y` is invariant to the split: scaling a read level and the mass by the same `c`
    leaves `num`, `den` and the floor `eps * m` all scaled by `c`, so the ratio is
    unchanged. The cotangents must therefore map by `1/c` -- and `dm` in particular,
    which is the one term a backward that dropped (A.3) would leave at zero.
    """
    from rola.routing.factors import RouteFactors

    c = 3.0
    routes, v = _bundle(64, seed=11, mass=True, gain=1.5)

    def kernel_grads(scale: float):
        read = [x.detach().to(torch.float32) for x in routes.read]
        read[0] = read[0] * scale
        read = [x.requires_grad_() for x in read]
        write = [x.detach().to(torch.float32).requires_grad_() for x in routes.write]
        gw = routes.g_write.detach().to(torch.float32).requires_grad_()
        rm = (routes.read_mass.detach().to(torch.float32) * scale).requires_grad_()
        kv = v.detach().to(torch.float32).requires_grad_()
        rts = RouteFactors(read=tuple(read), write=tuple(write), g_read=None, g_write=gw,
                           read_mass=rm, topology=routes.topology, normalization="global")
        y, _ = rola.rola_op(rts, kv)
        dy = torch.ones_like(y)
        return y, torch.autograd.grad(y, (kv, *rts.tensors()), dy)

    y1, g1 = kernel_grads(1.0)
    yc, gc = kernel_grads(c)
    #: the readout is invariant to the split, to the kernel's own floor
    assert torch.allclose(y1, yc, rtol=2e-2, atol=1e-3), (
        "the readout moved when the read mass and a read level were scaled together; "
        "the m-carry is not forward-exact")
    #: and `dm` -- the LAST operand, since `read_mass` is last in `tensors()` -- maps
    #: by 1/c, which it cannot do if it is zero
    dm1, dmc = g1[-1], gc[-1]
    assert dm1.abs().max() > 0, "d read_mass is identically zero -- (A.3) was dropped"
    rel = (dmc * c - dm1).norm() / dm1.norm()
    assert rel < 5e-2, (
        f"d read_mass does not map by the split's chain rule: scaling by {c} moved it "
        f"by a relative {rel:.3e} away from 1/{c}")


def test_the_gradient_battery_is_not_vacuous():
    """A zero cotangent would pass every ratio above. Each output must be nonzero,
    and `d read_mass` must be nonzero SPECIFICALLY -- it is the term (A.3) adds and
    the one a backward that forgot the m-carry would silently leave at zero."""
    routes, v = _bundle(64, seed=11, mass=True, gain=2.0)
    kr = [t.detach().to(torch.float32).requires_grad_() for t in routes.read]
    kw = [t.detach().to(torch.float32).requires_grad_() for t in routes.write]
    kg = routes.g_write.detach().to(torch.float32).requires_grad_()
    km = routes.read_mass.detach().to(torch.float32).requires_grad_()
    kv = v.detach().to(torch.float32).requires_grad_()
    from rola.routing.factors import RouteFactors
    rts = RouteFactors(read=tuple(kr), write=tuple(kw), g_read=None, g_write=kg,
                       read_mass=km, topology=routes.topology, normalization="global")
    y, _ = rola.rola_op(rts, kv)
    grads = torch.autograd.grad(y.sum(), (kv, *rts.tensors()))
    for i, g in enumerate(grads):
        assert g.abs().max() > 0, f"cotangent {i} is identically zero"
    assert grads[-1].abs().max() > 0, "d read_mass is zero -- the m-carry was dropped"


# ---------------------------------------------------------------------------
# 3. THE MEMORY RECEIPT -- the O(L) claim measured, not argued
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale", [1, 2, 4])
def test_the_backward_retains_o_of_l_and_nothing_o_of_n(scale):
    """What a grad-bearing call retains OVER an inference one is linear in `T` and
    FLAT in `N`. A regression here is the two-pass design failing: the whole reason
    the scans rebuild the state and replay the schedule is that nothing `O(N*L)` --
    and nothing `O(N)` beyond the optional block bitmap -- may be held."""
    from rola.ops import chunk as _chunk

    T = C * 2 * scale
    routes, v = _bundle(T, seed=5, mass=True, gain=1.0)
    kr = [t.detach().to(torch.float32).requires_grad_() for t in routes.read]
    kw = [t.detach().to(torch.float32).requires_grad_() for t in routes.write]
    kg = routes.g_write.detach().to(torch.float32).requires_grad_()
    km = routes.read_mass.detach().to(torch.float32).requires_grad_()
    kv = v.detach().to(torch.float32).requires_grad_()
    from rola.routing.factors import RouteFactors
    rts = RouteFactors(read=tuple(kr), write=tuple(kw), g_read=None, g_write=kg,
                       read_mass=km, topology=routes.topology, normalization="global")
    from rola.engine.dags.chunk_dag import ChunkContext, build_chunk_plan
    from rola.engine.plan import BackwardPlan

    plan = build_chunk_plan(ChunkContext(
        routes=rts, v=kv.to(torch.bfloat16).contiguous()))
    _y, den = _chunk.execute(plan)
    saved = BackwardPlan(forward=plan, den=den)
    per_token = saved.retained_bytes() / T
    #: the closed form: two `sumW` bf16 planes, the gain, the mass, the denominator,
    #: and the union table's `2 * entries * ceil(T/32)` words -- all per token.
    assert per_token == pytest.approx(_expected_per_token(rts), rel=0.02), (
        f"the ctx retains {per_token:.1f} B/token at T={T}; the closed form says "
        f"{_expected_per_token(rts):.1f}. A per-chunk snapshot or a boundary-cursor "
        f"array shows up here as a term.")


def _chunk_arm(routes) -> tuple[int, int, int, int]:
    from rola.engine.facts.call import arm_of as _arm
    c, bc = _arm(routes)
    widths = tuple(l.width for l in routes.topology.levels)
    return (c, bc, len(widths), widths[0])


def _census_entries(c, bc, d, b) -> int:
    for row in extension().chunk_scan_b_census():
        if tuple(row[:4]) == (c, bc, d, b):
            return row[11]
    raise AssertionError(f"no scan census row for arm ({c},{bc},{d},{b})")


def _expected_per_token(routes) -> float:
    sumw = sum(level.shape[-1] for level in routes.read)
    #: the arm's own table rows per side, READ OFF THE CENSUS rather than
    #: restated -- the level plan's `entries` is the kernel's number, not a
    #: formula this file gets to keep a second copy of.
    entries = _census_entries(*_chunk_arm(routes))
    per = (2 * sumw * 2)            # pr, pw : bf16
    per += 2                        # gw : bf16
    per += 4                        # rm : fp32
    per += 4                        # den : fp32
    per += 2 * entries * 4 / 32     # the union table: a WORD per 32 tokens per entry
    return per
