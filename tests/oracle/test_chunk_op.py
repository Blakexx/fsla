"""The kernel arm is the chunk consumer, proven THROUGH the facade.

`rola_op` over a bundle must agree with the fp64 oracle -- CALLED directly, since the
facade has one arm -- to the bf16 design-floor scale; the difference includes the
kernel arm's bf16 operand casts by design. Everything outside the built envelope must
refuse loudly.
"""
import pytest
import torch

from rola import rola_op
from rola.ops.naive import naive_rola
from tests.conftest import build_routes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")


def _bundle(widths=(64, 64), d_v=64, heads=2, B=1, T=512, seed=0):
    torch.manual_seed(seed)
    rp = build_routes(hidden_size=128, num_heads=heads, widths=widths).cuda()
    x = torch.randn(B, T, 128, device="cuda")
    routes = rp(x)
    v = torch.randn(B, T, heads, d_v, device="cuda")
    return routes, v


def _oracle(routes, v):
    y, _ = naive_rola(v, routes.read_simplex(), routes.write, routes.g_read, routes.g_write,
                      routes.topology, None, routes.normalization)
    return y


def _relative(routes, v):
    with torch.no_grad():
        y_ref = _oracle(routes, v)
        y_ker, _ = rola_op(routes, v)
    return float((y_ker.double() - y_ref).norm() / y_ref.norm())


def test_kernel_arm_matches_the_oracle():
    routes, v = _bundle()
    rel = _relative(routes, v)
    assert rel < 1e-2, f"kernel arm off the oracle: rel={rel:.3e}"
    assert rel < 5e-3  # the expected bf16-floor scale, not just sanity


def test_deep_topology_through_facade():
    routes, v = _bundle(widths=(8, 8, 8, 8), T=256, seed=1)
    rel = _relative(routes, v)
    assert rel < 1e-2, f"deep arm off the oracle: rel={rel:.3e}"


def test_flat_routing_baseline_through_facade():
    # D = 1: the in-kernel flat-routing control (N = B <= 256 by the manifest). The
    # produce's outer chain is a literal 1.0 there, so this also gates the depth guard
    # end to end.
    routes, v = _bundle(widths=(256,), T=256, seed=2)
    rel = _relative(routes, v)
    assert rel < 1e-2, f"flat arm off the oracle: rel={rel:.3e}"


def test_refusals_are_loud():
    with torch.no_grad():
        routes, v = _bundle(widths=(16,), d_v=32, T=256)
        with pytest.raises(NotImplementedError, match="d_v"):
            rola_op(routes, v)
        routes64, v64 = _bundle(T=256)
        with pytest.raises(TypeError, match="rola.RoLAState"):
            rola_op(routes64, v64, state=torch.zeros(1))


def test_the_layer_stateless_eval_path_runs_the_chunk_consumer():
    """The adoption's layer-level proof: a uniform-width d_v=64 layer at stateless
    eval runs the CHUNK consumer (counted as 'cuda'), and the output sits at the
    oracle within the bf16 family bound."""
    from einops import rearrange

    from tests.conftest import build_layer

    torch.manual_seed(11)
    layer = build_layer(hidden_size=128, num_heads=2, d_v=64, widths=(64, 64),
                        normalization="global", decay=None,
                        layer_idx=0).cuda().float().eval()
    x = torch.randn(1, 128, 128, device="cuda")
    before = dict(layer.execution_backend_counts)
    with torch.no_grad():
        y_cuda, _ = layer(x)
        v = rearrange(layer.v_proj(x), "b l (h d) -> b l h d", d=layer.head_v_dim)
        y_ref = layer.o_proj(_oracle(layer.routes(x), v).to(x.dtype).reshape(x.shape[0], x.shape[1], -1))
    assert layer.execution_backend_counts["cuda"] == before["cuda"] + 1
    rel = float((y_cuda.double() - y_ref.double()).norm() / y_ref.double().norm())
    assert rel < 1e-2, f"layer chunk path off the oracle: rel={rel:.3e}"


def test_the_read_mass_rides_to_the_divide_and_is_exact_there():
    """The m-CARRY: the producer stops normalizing the read side, and the divide
    compensates (P73 S4, adjudication 1).

    The readout is a ratio, so scaling every read leaf of a token by ``m`` scales
    numerator and denominator alike; the ONE thing that does not scale with them is
    the floor, so the floor is handed over pre-scaled. The claim is exactness, not
    closeness, and the fixture makes it checkable: every operand is a power of two, so
    the scaled divide is the unscaled one BIT FOR BIT.

    THE TOOTH is the same call with the mass dropped: on a denominator at the floor's
    own scale that is an order-of-magnitude answer, which is what makes the argument
    a compensation rather than a cancellation.
    """
    from rola.ops._ext import extension

    torch.manual_seed(3)
    B, H, T, d_v, eps = 2, 2, 64, 64, 1e-5
    num = torch.randint(-8, 8, (B, H, T, d_v), device="cuda").float()
    den = torch.full((B, H, T), 2.0 ** -18, device="cuda")
    mass = 2.0 ** torch.randint(-10, 0, (B, H, T), device="cuda").float()

    plain = extension().chunk_divide_pass(num, den, None, eps, torch.float32)
    scaled = extension().chunk_divide_pass(
        num * mass.unsqueeze(-1), den * mass, mass, eps, torch.float32)
    assert torch.equal(plain, scaled), "the carried mass is not exact in the divide"

    dropped = extension().chunk_divide_pass(
        num * mass.unsqueeze(-1), den * mass, None, eps, torch.float32)
    moved = (dropped - plain).abs().max() / plain.abs().max()
    assert moved > 0.5, f"dropping the mass must move the readout; it moved {moved:.3e}"


def test_the_readout_is_narrowed_and_transposed_in_the_kernel():
    """The epilogue fold (P73 S4, inventory §B6): the consumer emits the OPERAND's
    layout and dtype, and doing it in the store is bit-for-bit the host cast it
    replaces -- both are round-to-nearest over the same fp32 quotient.
    """
    from rola.ops._ext import extension

    torch.manual_seed(4)
    B, H, T, d_v = 2, 4, 32, 64
    num = torch.randn(B, H, T, d_v, device="cuda")
    den = torch.rand(B, H, T, device="cuda") + 0.5
    wide = extension().chunk_divide_pass(num, den, None, 1e-5, torch.float32)
    narrow = extension().chunk_divide_pass(num, den, None, 1e-5, torch.bfloat16)
    assert tuple(wide.shape) == (B, T, H, d_v) and wide.is_contiguous()
    assert torch.equal(narrow, wide.to(torch.bfloat16))
    torch.testing.assert_close(
        wide.permute(0, 2, 1, 3), num / (den + 1e-5).unsqueeze(-1), rtol=0, atol=0)


def test_only_a_duty_off_the_simplex_carries_a_read_mass_to_the_kernel():
    """WHICH bundles carry it, pinned at the seam the kernel reads (P73 S4).

    A bit-identity row on ``y`` would pass just as well if the column never left the
    producer. This one watches ``read_mass`` arrive at the packing, and pins both
    answers: the union read duty is the one cell the registry does not declare on the
    simplex, so it is the one cell that carries a mass. The carried bundle must still
    satisfy the fp64 oracle's own input contract through :meth:`read_simplex`.
    """
    from rola.engine.dags.chunk_dag import ChunkContext, build_chunk_plan
    from rola.ops._oracle_contract import validate_simplex
    from rola.routing.producer import union_routing

    dense, dense_v = _bundle(T=256)
    dense_plan = build_chunk_plan(ChunkContext(routes=dense, v=dense_v))
    assert dense.read_mass is None and dense_plan.read_mass is None

    torch.manual_seed(7)
    producer = build_routes(hidden_size=128, num_heads=2, widths=(64, 64),
                           routing=union_routing(64, 1.5)).cuda()
    with torch.no_grad():
        routes = producer(torch.randn(1, 256, 128, device="cuda"))
    assert routes.read_mass is not None and float(routes.read_mass.min()) > 0
    plan = build_chunk_plan(ChunkContext(
        routes=routes, v=torch.randn(1, 256, 2, 64, device="cuda")))
    assert plan.read_mass is not None and plan.read_mass.dtype is torch.float32
    torch.testing.assert_close(plan.read_mass, routes.read_mass.permute(0, 2, 1),
                               rtol=0, atol=0)
    validate_simplex("read_levels", routes.read_simplex())
    with pytest.raises(ValueError, match="must sum to 1"):
        validate_simplex("read_levels", routes.read)
