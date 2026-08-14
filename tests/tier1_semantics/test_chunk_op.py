"""P69 stage C: the kernel arm is the chunk consumer, proven THROUGH the facade.

`rola_op` with the same bundle must agree between the reference arm (fp64
oracle) and the kernel arm (chunk consumer) to the bf16 design-floor scale --
the difference includes the kernel arm's bf16 operand casts by design.  The
unsupported surface must refuse loudly with the card that adds it.
"""
import pytest
import torch

from rola import rola_op
from tests.conftest import build_routes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")


def _bundle(widths=(64, 64), d_v=64, heads=2, B=1, T=512, seed=0):
    torch.manual_seed(seed)
    rp = build_routes(hidden_size=128, num_heads=heads, d_v=d_v, widths=widths).cuda()
    x = torch.randn(B, T, 128, device="cuda")
    routes = rp(x)
    v = torch.randn(B, T, heads, d_v, device="cuda")
    return routes, v


def test_kernel_arm_matches_reference():
    with torch.no_grad():
        routes, v = _bundle()
        y_ref, _ = rola_op(routes, v, _arm="reference")
        y_ker, _ = rola_op(routes, v, _arm="kernel")
    rel = (y_ker.double() - y_ref).norm() / y_ref.norm()
    assert rel < 1e-2, f"kernel arm off the reference: rel={rel:.3e}"
    assert rel < 5e-3  # the expected bf16-floor scale, not just sanity


def test_deep_topology_through_facade():
    with torch.no_grad():
        routes, v = _bundle(widths=(8, 8, 8, 8), T=256, seed=1)
        y_ref, _ = rola_op(routes, v, _arm="reference")
        y_ker, _ = rola_op(routes, v, _arm="kernel")
    rel = (y_ker.double() - y_ref).norm() / y_ref.norm()
    assert rel < 1e-2, f"deep arm off the reference: rel={rel:.3e}"


def test_flat_routing_baseline_through_facade():
    # D = 1: the in-kernel flat-routing control (N = B <= 256 by the
    # manifest).  The produce's outer chain is a literal 1.0 there, so this
    # also gates the depth guard end to end.
    with torch.no_grad():
        routes, v = _bundle(widths=(256,), T=256, seed=2)
        y_ref, _ = rola_op(routes, v, _arm="reference")
        y_ker, _ = rola_op(routes, v, _arm="kernel")
    rel = (y_ker.double() - y_ref).norm() / y_ref.norm()
    assert rel < 1e-2, f"flat arm off the reference: rel={rel:.3e}"


def test_refusals_are_loud():
    with torch.no_grad():
        routes, v = _bundle(widths=(16,), d_v=32, T=256)
        with pytest.raises(NotImplementedError, match="d_v"):
            rola_op(routes, v, _arm="kernel")
        routes64, v64 = _bundle(T=256)
        with pytest.raises(NotImplementedError, match="P67"):
            rola_op(routes64, v64, _arm="kernel", output_final_state=True)
