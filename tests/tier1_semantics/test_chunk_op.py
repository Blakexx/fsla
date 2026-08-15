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
    y, _ = naive_rola(v, routes.read, routes.write, routes.g_read, routes.g_write,
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
