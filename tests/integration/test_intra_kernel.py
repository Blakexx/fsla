"""Gate for the INTRA (csrc/rola/src/intra/): the within-window pairs of the bundle against the declared arithmetic,
per slot.

The reference is `tests/oracle/int_reference.py`'s ``intra_bundle_reference``, itself gated against the fp64 intra of
the bundle's own dequantized levels (`tests/oracle/test_int_reference.py`). ``den`` is a sum of non-negative terms
and is held to its own value within fp32 rounding. ``num`` takes one declared rounding, the value product's weights,
which the reference bounds per slot (``bound``: half a unit a term); the kernel's weights sit within one unit of the
exact ones, so kernel and reference differ by at most three of those bounds, plus fp32 accumulation against the
slot's envelope (docs/internals/intra/intra_kernel.md#value-product).

Symbols: ``L`` tokens, ``H`` heads, ``BH`` batch x heads, a window 512 tokens.
"""
from __future__ import annotations

import pytest
import torch

from rola.engine.facts import liveness as lv
from rola.ops import facts as facts_ops
from rola.ops import intra as intra_ops
from rola.routing.factors import RouteFactors
from rola.routing.types import OpaqueRouting, Topology
from rola.values import Values
from tests.oracle import int_reference as iref

#: the lists' form the facts pass is called in (sm_86's even carry arm); this file reads its words and unions only
_READ_ROW = 64

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the intra is a CUDA kernel")

WINDOW = intra_ops.WINDOW
#: the fp32 accumulation's share of a slot, against its envelope (den: against itself)
FP32 = 2.0 ** -20


def _bundle(L, H, keep, *, widths=(256, 256), B=1, head_major=True, value_digits=2, seed=0):
    """A bundle at ``widths``: u8 rows with ``keep`` of their digits live, scales and gains in [0.1, 1.1), Gaussian
    values in the format; each side one packed plane, head-major (the producer's) or token-major."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    bounds = [sum(widths[:level]) for level in range(len(widths) + 1)]

    def rows():
        shape = (B, H, L, bounds[-1]) if head_major else (B, L, H, bounds[-1])
        d = torch.randint(1, 256, shape, device="cuda", generator=gen).to(torch.uint8)
        d *= (torch.rand(shape, device="cuda", generator=gen) < keep).to(torch.uint8)
        d = d.transpose(1, 2) if head_major else d
        return tuple(d[..., a:b] for a, b in zip(bounds, bounds[1:]))

    def columns():
        return tuple(column() for _ in widths)

    def column():
        return torch.rand((B, L, H), device="cuda", generator=gen) + 0.1

    routes = RouteFactors(topology=Topology(levels=tuple(OpaqueRouting(width=w) for w in widths)),
                          amp_read=rows(), amp_write=rows(), scale_read=columns(), scale_write=columns(),
                          g_write=column())
    values = Values.from_projection(torch.randn(B, L, H, 64, device="cuda", generator=gen), value_digits)
    return routes, values


def _clustered(L, H, seed=0):
    """A bundle the slab gate harvests: at level 0 a reader in window tile ``j`` holds digits only in slab ``j`` and a
    writer only in slabs ``j`` and ``j + 1``, so a tile pair meets on the diagonal and one below it, the second on one
    slab of eight; level 1 is the random draw."""
    routes, values = _bundle(L, H, 0.3, seed=seed)
    tile = (torch.arange(L, device="cuda") % WINDOW) // 64
    slab = torch.arange(256, device="cuda") // 32

    def keep(rows, slabs):
        return rows * torch.stack([slab == s for s in slabs]).any(0)[None, :, None, :].to(torch.uint8)

    read = (keep(routes.amp_read[0], [tile[:, None]]), routes.amp_read[1])
    write = (keep(routes.amp_write[0], [tile[:, None], (tile[:, None] + 1) % 8]), routes.amp_write[1])
    return RouteFactors(topology=routes.topology, amp_read=read, amp_write=write, scale_read=routes.scale_read,
                        scale_write=routes.scale_write, g_write=routes.g_write), values


def _run(routes, values, *, weight_digits=2, num=None, den=None, gated=True):
    """The facts pass and the intra; ``gated=False`` declares every level dense to the intra, so it reads no unions
    and gates nothing."""
    layout = facts_ops.call_layout(routes)
    statics = facts_ops.call_statics(routes, layout)
    unions = facts_ops.facts(routes, values, layout, *statics, _READ_ROW).unions
    if not gated:
        statics = tuple(lv.side_statics(layout, (True,) * layout.D, routes.topology.widths) for _ in range(2))
    BH, L = routes.batch * routes.heads, routes.tokens
    num = torch.zeros(BH, L, 64, device="cuda") if num is None else num
    den = torch.zeros(BH, L, device="cuda") if den is None else den
    intra_ops.intra_forward(routes, values, unions, layout, *statics, num, den, weight_digits=weight_digits)
    return num, den


def _same(a, b):
    """Equal to fp32 reassociation: a block's pieces reduce into one row in either order (a few ulps of the row's
    largest term), where a race or a skipped contribution is off by orders of magnitude."""
    return bool(((a - b).abs() <= 2.0 ** -18 * b.abs().amax() + 1e-12).all())


def _flat(x, routes):
    """``[B, T, H, ...]`` -> ``[BH, T, ...]``, the kernel's planes."""
    B, T, H = routes.batch, routes.tokens, routes.heads
    return x.transpose(1, 2).reshape(B * H, T, *x.shape[3:])


def _check(routes, values, weight_digits=2):
    num, den = _run(routes, values, weight_digits=weight_digits)
    seen: dict = {}
    want_num, want_den = iref.intra_bundle_reference(routes, values, WINDOW, weight_digits=weight_digits,
                                                     telemetry=seen)
    #: the slot's envelope: the unrounded product on |v|
    abs_values = Values.from_projection(values.dequantized().abs(), 2 if values.v_lo is not None else 1)
    abs_num, _ = iref.intra_bundle_reference(routes, abs_values, WINDOW, exact=True)
    want_num, want_den = _flat(want_num, routes), _flat(want_den, routes)
    bound, abs_num = _flat(seen["bound"], routes), _flat(abs_num, routes)
    num_err = (num.double() - want_num).abs()
    assert (num_err <= 3.0 * bound + FP32 * abs_num.abs() + 1e-30).all(), (
        f"{int((num_err > 3.0 * bound + FP32 * abs_num.abs()).sum())} num slots off; worst "
        f"{(num_err / (3.0 * bound + FP32 * abs_num.abs() + 1e-30)).max().item():.3g} of the allowance")
    den_err = (den.double() - want_den).abs()
    assert (den_err <= FP32 * want_den + 1e-30).all(), f"den off by {(den_err / want_den).max().item():.3g}"
    return num, den


@pytest.mark.parametrize(("L", "H", "keep"), [(1024, 2, 0.3), (2048, 1, 1.0), (1536, 3, 0.05)])
def test_the_intra_is_the_declared_arithmetic_per_slot(L, H, keep):
    _check(*_bundle(L, H, keep))


#: THE WIDTHS THE INTRA READS AT RUNTIME: one or two levels, each a whole number of 16-digit granules -- a stage short
#: of its four slabs, a slab half past a level's digits, and levels narrower than the facts layout's (48 in a 64)
@pytest.mark.parametrize("widths", [(256,), (64,), (16,), (48,), (64, 64), (128, 32), (16, 256), (48, 80), (96, 160)])
def test_every_width_is_the_declared_arithmetic(widths):
    _check(*_bundle(1024, 2, 0.3, widths=widths, seed=11))


@pytest.mark.parametrize("widths", [(64, 64), (48, 80)])
def test_the_slab_gate_is_an_exact_certificate_at_any_width(widths):
    """A sparse draw at narrow and uneven widths: the gated run is the ungated one's sums."""
    routes, values = _bundle(1024, 2, 0.02, widths=widths, seed=13)
    assert all(_same(a, b) for a, b in zip(_run(routes, values), _run(routes, values, gated=False), strict=True))


@pytest.mark.parametrize("weight_digits", (1, 2))
@pytest.mark.parametrize("value_digits", (1, 2))
def test_every_digit_count_is_the_declared_arithmetic(weight_digits, value_digits):
    _check(*_bundle(1024, 2, 0.3, value_digits=value_digits), weight_digits=weight_digits)


def test_token_major_rows_read_by_their_strides():
    """The producer stores head-major; a token-major bundle (a hand-built one) is read by its strides all the same."""
    _check(*_bundle(1024, 2, 0.3, head_major=False, seed=3))


def test_the_slab_gate_is_an_exact_certificate():
    """Skipped pairs and slabs contribute exact zeros: the gated run is the declared arithmetic per slot and the
    ungated run's sums (to the pieces' reassociation)."""
    routes, values = _clustered(2048, 2)
    gated = _check(routes, values)
    ungated = _run(routes, values, gated=False)
    assert _same(gated[0], ungated[0]) and _same(gated[1], ungated[1])
    random = _bundle(1024, 2, 0.05, seed=5)
    assert all(_same(a, b) for a, b in zip(_run(*random), _run(*random, gated=False), strict=True))


@pytest.mark.parametrize("draw", ("random", "clustered"))
def test_the_intra_is_reproducible_across_launches(draw):
    """The pipeline's copies are claimed by whichever warp gets there first, so a race would show as run-to-run
    variation: 40 launches over a grid of many CTAs, causal and gated, all within fp32 reassociation of the first (a
    block's pieces reduce into one row in either order; a race is off by orders of magnitude)."""
    routes, values = _bundle(4096, 4, 0.3, seed=7) if draw == "random" else _clustered(4096, 4, seed=7)
    first = _run(routes, values)
    for _ in range(39):
        again = _run(routes, values)
        assert _same(again[0], first[0]) and _same(again[1], first[1])


def test_the_intra_reduces_into_the_planes_it_is_given():
    """A RED, not a store: what the planes held before stays under what the intra adds."""
    routes, values = _bundle(1024, 2, 0.3)
    plain_num, plain_den = _run(routes, values)
    num = torch.full_like(plain_num, 0.5)
    den = torch.full_like(plain_den, 2.0)
    _run(routes, values, num=num, den=den)
    assert _same(num, plain_num + 0.5) and _same(den, plain_den + 2.0)


@pytest.mark.parametrize("widths", [(256, 256), (64, 128)])
def test_the_producers_bundle(widths):
    """A real producer's bundle -- head-major packed rows, the alternating routing's statics -- per slot."""
    import rola

    torch.manual_seed(0)
    routing = rola.Routing([rola.split_routing(widths[0], sparse_duty="write"),
                            rola.split_routing(widths[1], sparse_duty="read")])
    layer = rola.RoLA(routing, hidden_size=128, num_heads=2, d_v=64).cuda().eval()
    with torch.no_grad():
        routes, values = layer._produce(torch.randn(1, 1024, 128, device="cuda"))
    _check(routes, values)


def test_a_bound_ledger_books_every_phase_and_changes_nothing():
    routes, values = _bundle(1024, 2, 0.3)
    plain = _run(routes, values)
    ledger = torch.zeros((intra_ops.LEDGER_WARPS, len(intra_ops.LEDGER_PHASES)), dtype=torch.int64, device="cuda")
    intra_ops.ledger_bind(ledger)
    try:
        booked = _run(routes, values)
    finally:
        intra_ops.ledger_bind(None)
    assert _same(plain[0], booked[0]) and _same(plain[1], booked[1])
    assert (ledger > 0).all(), ledger


def test_the_seam_refuses_what_no_arm_is_built_for():
    routes, values = _bundle(1024, 1, 0.3)
    layout = facts_ops.call_layout(routes)
    statics = facts_ops.call_statics(routes, layout)
    unions = facts_ops.facts(routes, values, layout, *statics, _READ_ROW).unions
    num, den = torch.zeros(1, 1024, 64, device="cuda"), torch.zeros(1, 1024, device="cuda")
    f16 = RouteFactors(topology=routes.topology, amp_read=tuple(r.half() / 255 for r in routes.amp_read),
                       amp_write=tuple(r.half() / 255 for r in routes.amp_write), scale_read=routes.scale_read,
                       scale_write=routes.scale_write, g_write=routes.g_write)
    with pytest.raises(RuntimeError, match="f16 rows' intra arms"):
        intra_ops.intra_forward(f16, values, unions, layout, *statics, num, den)
    with pytest.raises(RuntimeError, match="one or two u8 digits"):
        intra_ops.intra_forward(routes, values, unions, layout, *statics, num, den, weight_digits=3)
    short = RouteFactors(topology=routes.topology, amp_read=tuple(r[:, :1000] for r in routes.amp_read),
                         amp_write=tuple(r[:, :1000] for r in routes.amp_write),
                         scale_read=tuple(s[:, :1000] for s in routes.scale_read),
                         scale_write=tuple(s[:, :1000] for s in routes.scale_write), g_write=routes.g_write[:, :1000])
    with pytest.raises(RuntimeError, match="whole number of 512"):
        intra_ops.intra_forward(short, values, unions, layout, *statics, num, den)
    lo = torch.zeros((*values.v_lo.shape[:3], 128), dtype=torch.uint8, device="cuda")[..., :64]
    lo.copy_(values.v_lo)
    with pytest.raises(RuntimeError, match="v_lo is laid out as v_hat is"):
        intra_ops.intra_forward(routes, Values(v_hat=values.v_hat, v_lo=lo, scale=values.scale), unions, layout,
                                *statics, num, den)


def test_the_seam_refuses_widths_it_does_not_read():
    """A level off the 16-digit granule, and rows wider than the facts layout's level."""
    routes, values = _bundle(1024, 1, 0.3, widths=(24,))
    layout = facts_ops.call_layout(routes)
    statics = facts_ops.call_statics(routes, layout)
    unions = facts_ops.facts(routes, values, layout, *statics, _READ_ROW).unions
    num, den = torch.zeros(1, 1024, 64, device="cuda"), torch.zeros(1, 1024, device="cuda")
    with pytest.raises(RuntimeError, match="whole 16-digit copy granules"):
        intra_ops.intra_forward(routes, values, unions, layout, *statics, num, den)
    routes, values = _bundle(1024, 1, 0.3, widths=(64,))
    narrow = lv.LivenessLayout(D=1, B=(32,), L=1024)
    with pytest.raises(RuntimeError, match="past the layout's 32"):
        intra_ops.intra_forward(routes, values, unions, narrow, *statics, num, den)
    assert not intra_ops.admits((24,), 64) and not intra_ops.admits((64, 64, 64), 64)
    assert not intra_ops.admits((64,), 128) and intra_ops.admits((48, 80), 64)


def test_the_loaded_binary_states_the_footprint_it_carries():
    """The extension-trap check: the device's own fixed tenants are the footprint law's, a reader row of three or more
    128-digit stage slots past them."""
    tenants = intra_ops.build_stamp() // 1000003
    assert intra_ops.footprint((256, 256)) == tenants + 128 * 4 * 128
    assert intra_ops.footprint((64,)) == intra_ops.footprint((128, 32)) == tenants + 128 * 3 * 128
