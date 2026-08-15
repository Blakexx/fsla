"""THE LAYER'S ENVELOPE GATE — what runs, and what is refused by name.

The built kernel envelope IS the layer's surface (`docs/api.md` §1). There is no
second arm behind it: a call the built matrix cannot run RAISES, and the fp64 oracle
is reached by CALLING IT (`rola.ops.naive.naive_rola`), never by dispatch. So this
file holds two kinds of row — the kernel arm ran and agrees with the oracle, and the
out-of-envelope call refused naming its clause.

**What the refusals are guarding.** A forward-only kernel reachable from a
grad-requiring context does not raise on its own: it returns a tensor with no
`grad_fn`, the graph is severed at this layer, and everything upstream silently gets
`.grad = None` (the recorded `layer-dispatch-autograd-gap` class, the fleet-smoke
failure). Training mode, a gradient reaching the operation, and every unbuilt
configuration are therefore refusals rather than slow paths — the reference-arm
dispatch that used to absorb them was a ~1000x-slower route shipping unnoticed.
"""

from __future__ import annotations

import pytest
import torch
import torch.utils.checkpoint as torch_checkpoint
from conftest import build_layer, build_routes, produce_routing
from einops import rearrange

import rola
from rola import RoLA, dense_routing, union_routing
from rola.expert import PlanOverrides
from rola.interface import BackwardNotImplemented, envelope_refusal
from rola.ops.naive import naive_rola

#: P26: `mma_precision` is a dead axis (bf16 is the only value `PlanOverrides`
#: accepts). Every kernel-vs-oracle comparison in this file therefore runs
#: bf16-vs-fp64-oracle, and the bound is the bf16 family bound
#: (`tests/oracle/tolerances.py:BF16_RTOL`), quoted at its value rather than
#: imported so that loosening it here would be a visible edit to this file.
BF16_RTOL = 1e-2

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="the RoLA arm is a CUDA kernel")

#: `branches=[16, 16]`, `d_v=64`, `H=8` -- a BUILT CHUNK ARM (`(C, BC, D, B) =
#: (32, 64, 2, 16)` in the manifest), at `N = 256`. Every `T` here is a multiple of
#: the chunk `C = 32` for the same reason.
_BRANCHES = (16, 16)
_H, _DV = 8, 64
_HIDDEN = _H * _DV

_DECAY = {"none": None, "leaf_mass": "learned"}


def _routing_for(router: str):
    return dense_routing() if router == "dense" else union_routing(1.5)


def _layer(*, norm: str = "global", decay: str, router: str = "dense", device: str = "cuda",
           logit_gain: float = 1.0, head_v_dim: int = _DV, branches=None) -> RoLA:
    branches = _BRANCHES if branches is None else tuple(branches)
    layer = build_layer(
        hidden_size=_HIDDEN, num_heads=_H, d_v=head_v_dim, widths=branches,
        routing=_routing_for(router), normalization=norm, decay=_DECAY[decay],
        layer_idx=0,
    ).to(device, torch.float32)
    if logit_gain != 1.0:
        # Sharpens an UNTRAINED exact-entmax router into a real sparse support. The
        # support is genuine (exact alpha=1.5, the production path -- never an
        # epsilon floor or a top-k stand-in); only its LEVEL is a synthetic stand-in
        # for training.
        with torch.no_grad():
            layer.routes.producers[0].route_W.mul_(logit_gain)
    return layer.eval()


def _oracle(layer: RoLA, x: torch.Tensor, *, initial_state=None, want_state: bool = False):
    """`(y, final_state)` from the fp64 oracle, CALLED DIRECTLY over this layer's own
    projections and routing. `y` is projected through `o_proj` so it is comparable to
    a layer output; the state is the oracle's own fp64 `[B, H, N, cols]`."""
    with torch.no_grad():
        v = rearrange(layer.v_proj(x), "b l (h d) -> b l h d", d=layer.head_v_dim)
        routes = layer.routes(x)
        decay = None if layer.decay is None else layer.decay()
        y, final = naive_rola(
            v, routes.read_simplex(), routes.write, routes.g_read, routes.g_write, layer.topology,
            decay, layer.normalization, initial_state=initial_state,
            output_final_state=want_state)
        y = layer.o_proj(y.to(x.dtype).reshape(x.shape[0], x.shape[1], -1))
    return y, final


def _sparsity(layer: RoLA, x: torch.Tensor) -> float:
    """Realized exact-zero fraction of the router's write allocations, so a test
    that claims to exercise a sparse fixture can prove the fixture is sparse."""
    producer = layer.routes.producers[0]
    with torch.no_grad():
        out = produce_routing(
            x, None, torch.zeros(x.shape[0], x.shape[1], layer.num_heads, layer.head_v_dim,
                                 device=x.device, dtype=x.dtype),
            producer.route_W, producer.route_bias, layer.routes.side_gain.gain_W,
            layer.routes.side_gain.gain_bias, layer.topology, layer.normalization)
    zeros = sum(float((t == 0).float().mean()) * t.shape[-1] for t in out.write_levels)
    return zeros / sum(t.shape[-1] for t in out.write_levels)


def _support_fraction(layer: RoLA, x: torch.Tensor) -> float:
    """The realized WRITE-side support fraction on the same routing a forward used."""
    routes = layer.routes(x)
    live = sum(int((level != 0).sum()) for level in routes.write)
    total = sum(level.numel() for level in routes.write)
    return live / total


# ---------------------------------------------------------------------------
# 1. Every route to a gradient refuses, and says the backward is coming
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decay", ["none", "leaf_mass"])
@pytest.mark.parametrize("router,gain", [("dense", 1.0), ("sparse", 8.0)])
@_CUDA
def test_a_grad_bearing_forward_refuses_instead_of_severing_the_graph(decay, router, gain):
    """The autograd gap, closed by refusal. What must NOT happen is a forward that
    returns a fine-looking tensor whose graph is gone -- and the reference arm that
    used to absorb this case is exactly what the fast-fail retired."""
    torch.manual_seed(3)
    layer = _layer(decay=decay, router=router, logit_gain=gain)
    x = torch.randn(1, 96, _HIDDEN, device="cuda", requires_grad=True)
    with pytest.raises(BackwardNotImplemented, match="native backward"):
        layer(x)
    assert layer.execution_backend_counts == {"cuda": 0, "decode": 0}


@_CUDA
def test_training_mode_refuses_even_with_nothing_requiring_grad():
    """Training mode is the MODULE's own clause and it is unconditional: reentrant
    checkpointing runs its first pass under `no_grad` with `training` still True, and
    that pass must refuse too rather than hand back a forward no backward can follow."""
    torch.manual_seed(5)
    layer = _layer(decay="none").train()
    for param in layer.parameters():
        param.requires_grad_(False)
    x = torch.randn(1, 96, _HIDDEN, device="cuda")
    with torch.no_grad(), pytest.raises(BackwardNotImplemented, match="TRAINING mode"):
        layer(x)


@_CUDA
def test_the_decay_dials_alone_are_a_grad_bearing_operation_input():
    """A context where ONLY the decay source's `theta` requires grad still carries a
    gradient into the operation -- the dials are handed straight to it -- so it
    refuses. `o_proj` staying trainable is what would have let a severed forward look
    healthy: it sits downstream and re-attaches a graph whether or not the upstream
    one was cut."""
    torch.manual_seed(29)
    layer = _layer(decay="leaf_mass")
    for name, param in layer.named_parameters():
        if not name.startswith(("decay.theta", "o_proj")):
            param.requires_grad_(False)
    x = torch.randn(1, 64, _HIDDEN, device="cuda")  # does not require grad
    assert torch.is_grad_enabled()
    with pytest.raises(BackwardNotImplemented, match="native backward"):
        layer(x)


@pytest.mark.parametrize("use_reentrant", [True, False])
@_CUDA
def test_checkpointing_a_training_mode_layer_refuses_under_both_modes(use_reentrant):
    """Both checkpoint modes, both passes: the layer is in training mode and there is
    no training path. Before the fast-fail this cell asserted the oracle ran twice."""
    torch.manual_seed(31)
    layer = _layer(decay="none").train()
    x = torch.randn(1, 96, _HIDDEN, device="cuda", requires_grad=True)

    def run(inp):
        out, _ = layer(inp)
        return out

    with pytest.raises(BackwardNotImplemented):
        torch_checkpoint.checkpoint(run, x, use_reentrant=use_reentrant)
    assert layer.execution_backend_counts == {"cuda": 0, "decode": 0}


def test_cpu_operands_refuse_because_there_is_no_cpu_lowering():
    """A device clause, refused rather than dispatched: there is no CPU implementation
    to reach. Runs without CUDA so it is part of the CPU gate too."""
    torch.manual_seed(23)
    layer = _layer(decay="none", device="cpu")
    x = torch.randn(1, 32, _HIDDEN)
    with torch.no_grad(), pytest.raises(NotImplementedError, match="CUDA kernel"):
        layer(x)


# ---------------------------------------------------------------------------
# 2. The kernel arm runs, and the counter is what says so
# ---------------------------------------------------------------------------


@_CUDA
def test_a_no_grad_cuda_context_takes_the_kernel_arm():
    """Asserted on the counter -- not on timing, which cannot tell two arms apart."""
    torch.manual_seed(7)
    layer = _layer(decay="none")
    x = torch.randn(1, 96, _HIDDEN, device="cuda")
    v = torch.randn(1, 96, _H, _DV, device="cuda")
    with torch.no_grad():
        layer(x)
        assert envelope_refusal(layer.routes(x), v) is None, (
            "fixture rot: this topology must be a BUILT chunk arm for the row above to "
            "mean anything")
    assert layer.execution_backend_counts == {"cuda": 1, "decode": 0}


@_CUDA
def test_the_kernel_arm_holds_on_a_deeper_built_arm():
    """The same property at `D = 4` (the manifest's `D=4, B=8` arm; `N = 4096`), so a
    dispatch that happened to be right only at `D = 2` cannot pass the file."""
    torch.manual_seed(7)
    layer = _layer(decay="none", branches=(8, 8, 8, 8))
    x = torch.randn(1, 96, _HIDDEN, device="cuda")
    v = torch.randn(1, 96, _H, _DV, device="cuda")
    with torch.no_grad():
        layer(x)
        assert envelope_refusal(layer.routes(x), v) is None, (
            "fixture rot: this topology must be a BUILT chunk arm")
    assert layer.execution_backend_counts == {"cuda": 1, "decode": 0}


@_CUDA
def test_the_routing_is_re_measured_every_step_not_cached():
    """Moving the routing distribution between two forwards on the same shape must
    move the measured routing. A per-shape cache would return the first step's answer
    for the second step's routing."""
    torch.manual_seed(11)
    layer = _layer(decay="none", router="sparse")
    x = torch.randn(1, 128, _HIDDEN, device="cuda")
    with torch.no_grad():
        layer(x)
        dense = _support_fraction(layer, x)
        layer.routes.producers[0].route_W.mul_(16.0)
        layer(x)
        sharp = _support_fraction(layer, x)
    assert layer.execution_backend_counts == {"cuda": 2, "decode": 0}
    assert sharp < 0.5 < dense, (
        "the sharpening did not move the routing it is supposed to move: the realized "
        f"write support went {dense:.3f} -> {sharp:.3f}")


# ---------------------------------------------------------------------------
# 3. The kernel arm agrees with the oracle, which is CALLED, not dispatched to
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("router,gain", [("dense", 1.0), ("sparse", 8.0)])
@_CUDA
def test_the_kernel_arm_matches_the_oracle_on_the_same_inputs(router, gain):
    """ONE layer instance and ONE input, so the only difference between the two runs
    is which implementation executed. The oracle leg calls `naive_rola` directly --
    the fp64 spec is not an arm of the layer."""
    torch.manual_seed(13)
    layer = _layer(decay="none", router=router, logit_gain=gain)
    x = torch.randn(1, 96, _HIDDEN, device="cuda")
    if router == "sparse":
        assert _sparsity(layer, x) > 0.2, "the sparse fixture is vacuously dense"

    reference, _ = _oracle(layer, x)
    with torch.no_grad():
        cuda_out, _ = layer(x)
    assert layer.last_execution_backend == "cuda"

    scale = float(reference.abs().max())
    deviation = float((cuda_out - reference).abs().max()) / scale
    assert deviation < BF16_RTOL, (
        f"the kernel arm is {deviation:.3e} from the oracle (BF16_RTOL={BF16_RTOL:.1e}); "
        f"router={router}")


@_CUDA
def test_the_carried_state_the_kernel_arm_hands_back_matches_the_oracle_s():
    """The state surface, in the shape both write: `[B, H, N, d_v + 1]` (the mass
    column is unconditional). Only the accumulation dtype differs -- fp64 oracle,
    fp32 kernel."""
    torch.manual_seed(17)
    layer = _layer(decay="none")
    x = torch.randn(1, 64, _HIDDEN, device="cuda")

    _y_ref, oracle_state = _oracle(layer, x, want_state=True)
    carried = rola.state()
    with torch.no_grad():
        layer(x, state=(carried, None))
    kernel_state = carried.materialize()

    assert kernel_state.shape == oracle_state.shape
    scale = float(oracle_state.abs().max())
    deviation = float((kernel_state.double() - oracle_state).abs().max()) / scale
    assert deviation < BF16_RTOL, (
        f"carried states disagree at {deviation:.3e} (BF16_RTOL={BF16_RTOL:.1e})")


@_CUDA
def test_a_kernel_produced_state_feeds_an_oracle_step_at_the_ops_level():
    """The two write the same object, so the oracle can CONTINUE from a kernel state.
    Called at the ops level because that is the only place the oracle lives: the
    layer has one arm."""
    from rola.ops import chunk as _chunk

    torch.manual_seed(41)
    layer = _layer(decay="none")
    T = 2 * _chunk.CHUNK_TOKENS
    x = torch.randn(1, T, _HIDDEN, device="cuda")

    carried = rola.state()
    with torch.no_grad():
        layer(x, state=(carried, None))
    kernel_state = carried.materialize()
    assert kernel_state.dtype == torch.float32

    with torch.no_grad():
        v = rearrange(layer.v_proj(x), "b l (h d) -> b l h d", d=_DV)
        routes = layer.routes(x)
    v = v.requires_grad_(True)
    out2, oracle_state2 = naive_rola(
        v, routes.read_simplex(), routes.write, routes.g_read, routes.g_write, layer.topology,
        None, "global", initial_state=kernel_state, output_final_state=True)
    out2.square().mean().backward()
    assert torch.isfinite(out2).all() and v.grad is not None
    assert oracle_state2.dtype == torch.float64
    assert oracle_state2.shape == kernel_state.shape


# ---------------------------------------------------------------------------
# 4. Out of the built matrix: refused by name, at the layer's own boundary
# ---------------------------------------------------------------------------


@_CUDA
def test_a_d_v_outside_the_built_matrix_refuses_and_names_the_clause():
    """`d_v = 48` has no built arm. Under the retired dispatch this ran on the oracle
    with a label; under the fast-fail it RAISES, because a silent ~1000x-slower route
    is the thing the no-fallback rule forbids."""
    torch.manual_seed(19)
    layer = _layer(decay="none", head_v_dim=48)
    x = torch.randn(1, 64, _HIDDEN, device="cuda")
    with torch.no_grad(), pytest.raises(NotImplementedError, match="d_v=48"):
        layer(x)
    assert layer.execution_backend_counts == {"cuda": 0, "decode": 0}
    v = torch.randn(1, 64, _H, 48, device="cuda")
    with torch.no_grad():
        assert "d_v=48" in envelope_refusal(layer.routes(x), v)


@_CUDA
def test_a_decay_configured_layer_refuses_because_the_kernel_has_no_decay():
    """Decay left the operator (`docs/internals/chunk/chunk.md`); the semantics live
    in the oracle, which is not reachable from here."""
    torch.manual_seed(21)
    layer = _layer(decay="leaf_mass")
    for param in layer.parameters():
        param.requires_grad_(False)
    x = torch.randn(1, 64, _HIDDEN, device="cuda")
    with torch.no_grad(), pytest.raises(NotImplementedError, match="decay"):
        layer(x)


@_CUDA
def test_a_jagged_topology_refuses_and_names_its_widths():
    """The built manifest is uniform-width; the flagship `(8, 16)` is not."""
    torch.manual_seed(22)
    layer = _layer(decay="none", branches=(8, 16))
    x = torch.randn(1, 64, _HIDDEN, device="cuda")
    with torch.no_grad(), pytest.raises(NotImplementedError, match="uniform-width"):
        layer(x)


@_CUDA
def test_a_token_count_off_the_chunk_refuses():
    """`T` must be a multiple of the chunk `C`; a ragged tail has no arm."""
    torch.manual_seed(24)
    layer = _layer(decay="none")
    x = torch.randn(1, 48, _HIDDEN, device="cuda")
    with torch.no_grad(), pytest.raises(NotImplementedError, match="multiple of the chunk"):
        layer(x)


@pytest.mark.cuda
@_CUDA
def test_an_uninstantiated_but_aligned_state_block_is_refused_host_side():
    """`state_block = 32` pins an axis the chunk arm does not have. It must be REFUSED
    by a HOST-side check that names the contract -- never a device-side assert, which
    would poison the context for every subsequent test in the process (the
    fleet-freeze failure mode). The context-health probe after the raise is the second
    half of the claim."""
    layer = _layer(decay="none")
    routes_32 = build_routes(hidden_size=_HIDDEN, num_heads=_H, widths=_BRANCHES)
    layer_32 = RoLA(
        routes_32, expert=PlanOverrides(state_block=32), layer_idx=0,
    ).to("cuda", torch.float32).eval()
    x = torch.randn(1, 32, _HIDDEN, device="cuda")
    with pytest.raises(NotImplementedError, match="state_block"), torch.no_grad():
        layer_32(x)
    assert float(torch.ones(4, device="cuda").sum()) == 4.0
    with torch.no_grad():
        layer(x)
    assert layer.execution_backend_counts["cuda"] == 1


# ---------------------------------------------------------------------------
# 5. Paging is invisible on the arm that has two backings
# ---------------------------------------------------------------------------


@_CUDA
def test_the_bf16_arm_is_paging_invisible_at_its_own_tolerance():
    """The backing is the STATE's, so the two arms are two states: the same carried
    state through either backing, `torch.equal`, because the state path has no atomic
    and no precision excuse. `y` is held to the oracle at the bf16 family bound."""
    torch.manual_seed(23)
    kw = dict(hidden_size=_HIDDEN, num_heads=_H, d_v=64, widths=(16, 16),
              routing=dense_routing(), normalization="global", decay=None, layer_idx=0)
    layer = build_layer(**kw).to("cuda", torch.float32).eval()
    x = torch.randn(1, 96, _HIDDEN, device="cuda")
    reference, _ = _oracle(layer, x)

    dense_pin = build_layer(expert=PlanOverrides(paging=False), **kw).to(
        "cuda", torch.float32).eval()
    dense_pin.load_state_dict(layer.state_dict())
    with torch.no_grad():
        state_u, state_p = rola.state(), rola.state()
        y_u, (state_u, _) = dense_pin(x, state=(state_u, None))
        y_p, (state_p, _) = layer(x, state=(state_p, None))

    assert not state_u.paged and state_p.paged, (
        "both arms took the same backing; the cell is vacuous")
    assert torch.equal(state_u.materialize(), state_p.materialize()), (
        "bf16 x paging moved the carried state; the state path has no atomic and "
        "no precision excuse")
    scale = float(reference.abs().max())
    assert float((y_p - reference).abs().max()) / scale < BF16_RTOL, (
        "the paged bf16 output left the bf16 arm's own tier bound vs the oracle")
    assert float((y_u - reference).abs().max()) / scale < BF16_RTOL
