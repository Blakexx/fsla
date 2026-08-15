# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The public op stack (`rola/interface.py`) and the O-3 name contract.

Two things are gated here. The first is THE ENVELOPE -- what `rola_op` runs and what
it refuses -- because the layer-dispatch autograd gap is a failure that produces no
exception and no wrong number, only a `grad_fn` that quietly is not there, and the
answer to it is a refusal rather than a second arm. The second is the NAME contract:
no public spelling says `v3`.

The CUDA arm's own numerics are gated in `tests/oracle/`; nothing here
re-measures them.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch
from conftest import build_routes

import rola
from rola.interface import (
    BackwardNotImplemented,
    ChunkConsumer,
    envelope_refusal,
    operands_of,
    require_envelope,
    requires_backward,
)

# --- the predicate -----------------------------------------------------------

def test_the_predicate_is_false_whenever_grad_is_globally_off():
    leaf = torch.zeros(2, requires_grad=True)
    assert requires_backward(leaf) is True
    with torch.no_grad():
        assert requires_backward(leaf) is False


def test_the_predicate_walks_into_the_per_level_tuples():
    """The routing allocations arrive as one tensor per level. A predicate that only
    looked at its top-level arguments would see a tuple, find no `requires_grad`, and
    send a grad-requiring call to a forward-only kernel."""
    leaf = torch.zeros(2, requires_grad=True)
    assert requires_backward((torch.zeros(2), leaf)) is True
    assert requires_backward([[torch.zeros(2), (leaf,)]]) is True
    assert requires_backward((torch.zeros(2), torch.zeros(2))) is False


def test_the_predicate_ignores_non_tensors_rather_than_guessing():
    assert requires_backward(None, "plan", 3, object()) is False


def test_the_layer_and_the_op_ask_the_same_question():
    """One predicate, or the two surfaces drift into two slightly different answers.
    Asserted structurally -- the layer imports it -- not by comparing behaviours."""
    import rola.layer as layer_module

    assert layer_module.requires_backward is requires_backward
    assert layer_module.operands_of is operands_of


def _built_arm(requires_grad: bool = True):
    """A bundle on a BUILT arm and on the device -- what a gradient test needs.

    `_tiny_routing` is `d_v = 16`, which the chunk matrix has no arm for; a gradient
    claim has to be made where the kernel actually runs.
    """
    torch.manual_seed(5)
    B, T, H, d_v, hidden, widths = 1, 32, 1, 64, 128, (64, 64)
    gateway = build_routes(hidden_size=hidden, num_heads=H, widths=widths,
                           routing=_dense_routing()).to("cuda")
    x = torch.randn(B, T, hidden, device="cuda", requires_grad=requires_grad)
    routes = gateway(x)
    v = torch.randn(B, T, H, d_v, device="cuda", requires_grad=requires_grad)
    from rola.ops import chunk as _chunk
    assert _chunk.unsupported(routes, None, d_v=d_v) is None, "fixture must be a BUILT arm"
    return v, routes, x, gateway


# --- the tripwire ------------------------------------------------------------

#: RETIRED at P67 D2-b: the two `RoLAConsumer` rows (the named refusal, and the
#: absence of `mark_non_differentiable`). Their subject is the TILED arm's autograd
#: node, which D2-c deletes with its consumer. The CLAIM did not retire with it --
#: D2-a's R-10 wrapper gave the chunk arm the same node, and the two rows below are
#: its twins, written against `ChunkConsumer` and asserting the same two things.


def test_the_chunk_node_refuses_a_ctx_the_forward_never_filled():
    """`rola.ops.chunk.run` is a bare extension call, so an arm that reached it from a
    gradient-requiring context would hand back a tensor with no `grad_fn` and no
    error. The node closes that gap, and now that its backward COMPUTES, the loud
    line moves to the one state that cannot be computed from: a backward reached with
    no saved ctx, which is the forward having retained nothing because it carried no
    gradient."""
    class _NoCtx:
        saved_plan = None
    with pytest.raises(BackwardNotImplemented, match="no saved ctx"):
        ChunkConsumer.backward(_NoCtx(), None)


def test_the_chunk_node_does_not_mark_its_output_non_differentiable():
    """Marking it would make the tripwire unreachable -- the same silence the node
    exists to break. Read off the source, since the condition is an absence."""
    source = pathlib.Path(rola.interface.__file__).read_text()
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == "ChunkConsumer")
    calls = {a.attr for n in ast.walk(node) if isinstance(n, ast.Call)
             for a in [n.func] if isinstance(a, ast.Attribute)}
    assert "mark_non_differentiable" not in calls


def test_the_kernel_arm_reaches_the_chunk_kernel_only_through_the_node():
    """`chunk.run` has ONE caller in the package, and it is inside the autograd node.

    A second call site would be a path to the kernel with no node on it, which is the
    gap the node exists to close; asserted structurally because the failure it guards
    against produces no exception to catch."""
    package = pathlib.Path(rola.__path__[0])
    callers = sorted(
        str(path.relative_to(package)) for path in package.rglob("*.py")
        if "_chunk.run(" in path.read_text() or "chunk.run(" in path.read_text())
    assert callers == ["interface.py"], (
        f"the chunk kernel is reached from {callers}; every path to it must go through "
        "ChunkConsumer")
    source = (package / "interface.py").read_text()
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == "ChunkConsumer")
    assert "_chunk.run(" in ast.get_source_segment(source, node)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_kernel_arm_never_returns_a_grad_fn_less_tensor_from_a_grad_context():
    """THE GATE FLIPS. This is the layer-dispatch autograd gap's own claim, and for
    the whole life of the operator it was satisfied by RAISING. The two-pass reverse
    pass is built, so it is now satisfied the other way: the call runs, the output
    carries a `grad_fn`, and a backward through it reaches the operands.

    What it must still never do -- and never could -- is return a good-looking tensor
    whose graph is gone."""
    v, routes, _x, _g = _built_arm()
    y, _ = rola.rola_op(routes, v)
    assert y.grad_fn is not None, "the op returned a tensor with no graph on it"
    y.sum().backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_node_is_what_carries_the_graph_when_the_facade_is_bypassed():
    """The last line, exercised: `ChunkConsumer.run` is the op's facade BYPASSED.

    The kernel runs on grad-requiring operands with no `require_envelope` in front
    of it, and the node is the whole of what stands between a severed graph and a
    working one. `grad_fn` present is the node having attached; the gradient
    arriving is it computing. Without the wrapper both assertions read the other way
    round: no `grad_fn`, no error, and `v.grad is None` several files away."""
    torch.manual_seed(5)
    B, T, H, d_v, hidden, widths = 1, 32, 1, 64, 128, (64, 64)
    gateway = build_routes(
        hidden_size=hidden, num_heads=H, widths=widths,
        routing=_dense_routing()).to("cuda")
    x = torch.randn(B, T, hidden, device="cuda", requires_grad=True)
    routes = gateway(x)
    v = torch.randn(B, T, H, d_v, device="cuda", requires_grad=True)
    from rola.ops import chunk as _chunk

    assert _chunk.unsupported(routes, None, d_v=v.shape[-1]) is None, "fixture must be a BUILT arm"
    y, _ = ChunkConsumer.run(_chunk.pack(routes), v)
    assert y.grad_fn is not None, (
        "the kernel arm returned a grad_fn-less tensor inside a grad context -- the "
        "graph is severed and nothing said so")
    y.sum().backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all(), (
        "the gradient stopped at the routing bundle instead of reaching the producer")


# --- the envelope, end to end ------------------------------------------------

def _dense_routing():
    from rola.routing.types import IndependentRouting

    return IndependentRouting(
        tied=True,
        read_activation=rola.SoftmaxActivation(),
        write_activation=rola.SoftmaxActivation())


def _tiny_routing(requires_grad: bool, device: str = "cpu"):
    """A real `RouteFactors` bundle, on the smallest topology the tests need.

    Old: five loose tensors (one routing level, its gains) plus a `_StubPlan`
    carrying only `topology`/`normalization`. New: `rola_op` takes a
    `RouteFactors`, obtainable only from `RouteGateway`, so the fixture builds
    one -- `requires_grad` on the source stream `x` reaches every tensor the
    gateway derives (read/write levels, gains), exactly as `requires_grad` on the
    five loose tensors used to.
    """
    torch.manual_seed(3)
    B, T, H, d_v, hidden, w = 1, 8, 1, 16, 8, 4
    gateway = build_routes(
        hidden_size=hidden, num_heads=H, widths=(w,), routing=_dense_routing()
    ).to(device)
    x = torch.randn(B, T, hidden, device=device, requires_grad=requires_grad)
    routes = gateway(x)
    v = torch.randn(B, T, H, d_v, device=device, requires_grad=requires_grad)
    return v, routes


def test_a_grad_requiring_call_is_no_longer_refused_for_being_one():
    """OLD: a grad-bearing call was told which milestone brought the backward. That
    milestone landed, so the gradient is not a clause any more -- what refuses this
    CPU bundle is the DEVICE, and the distinction matters because a caller reading
    "unsupported" has to know which wall they hit."""
    v, routes = _tiny_routing(requires_grad=True)
    assert requires_backward(v, *routes.tensors()) is True
    with pytest.raises(NotImplementedError, match="CUDA kernel"):
        rola.rola_op(routes, v)


def test_rola_op_has_no_consumer_only_kwargs_to_silently_ignore():
    """OLD: `rola_func` accepted CUDA-consumer-only options (`paging`, ...) on
    BOTH arms and refused them explicitly with `ValueError` on the reference
    arm -- one function, two arms, one of which had to actively reject what the
    other used. `rola_op` structurally cannot have that failure mode: paging,
    scheduling and every other CUDA-consumer detail live behind
    `rola.expert.execute`'s own seam, reachable only from the CUDA path, and
    `rola_op`'s signature carries no CUDA-consumer-only parameter at all -- so
    passing one is a plain `TypeError` (unknown keyword) regardless of which arm
    the call would otherwise take, rather than a case the op has to notice and
    refuse."""
    v, routes = _tiny_routing(requires_grad=True)
    with pytest.raises(TypeError):
        rola.rola_op(routes, v, paging=object())


def test_an_explicit_no_grad_context_is_how_an_inference_site_calls_the_op():
    """OLD: `rola_forward` was a separate entrypoint that established its own
    no-grad context internally, so an inference caller could not be wrong about
    it by forgetting to wrap the call. `rola_op` merges the two entrypoints into
    one: there is no separate inference spelling any more, and the caller's own
    `torch.no_grad()` is what an inference site now uses -- exactly as any other
    PyTorch op. The surviving half of the old claim: wrapped in `torch.no_grad()`,
    an every-input-requires-grad call is ADMITTED past the gradient clause, because
    the predicate is gradient REACHABILITY and not the `requires_grad` flag.
    """
    v, routes = _tiny_routing(requires_grad=True)
    with torch.no_grad():
        assert requires_backward(v, *routes.tensors()) is False
        #: CPU, so the DEVICE clause is what speaks -- the gradient clause did not.
        assert "CUDA kernel" in envelope_refusal(routes, v)


def test_the_two_refusal_clauses_are_told_apart_by_which_one_speaks():
    """The CONFIGURATION clause speaks whether or not a gradient is attached -- that
    is what "the gradient stopped being an envelope clause" means, and it is a
    stronger statement than the old pair: the same sentence must come back both
    times, so a caller cannot be told two different things about one bundle."""
    v, routes = _tiny_routing(requires_grad=True)
    with pytest.raises(NotImplementedError, match="CUDA kernel"):
        rola.rola_op(routes, v)
    with torch.no_grad(), pytest.raises(NotImplementedError, match="CUDA kernel"):
        rola.rola_op(routes, v)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_training_is_stateless_and_a_carried_state_says_so_by_name():
    """P65 R-D, at the boundary. The state plane is caller-owned and mutated in
    place, and the reverse pass rebuilds from zero -- there is no `d initial_state`
    for a carried state to receive. So `state=` and a gradient are mutually
    exclusive, and the refusal names which of the two to drop rather than producing
    a gradient that silently ignores the carry."""
    v, routes, _x, _g = _built_arm()
    state = rola.state()
    with pytest.raises(BackwardNotImplemented, match="STATELESS ONLY"):
        rola.rola_op(routes, v, state=state)
    #: and with the gradient off it is admitted, so the clause is the GRADIENT's and
    #: not a ban on carrying state
    with torch.no_grad():
        rola.rola_op(routes, v.detach(), state=state)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_backward_returns_gradients_in_the_bundles_own_tensor_order():
    """THE ORDERING IS LOAD-BEARING AND IS ASSERTED, NOT TRUSTED.

    `forward` receives the operands flat, in `RouteFactors.tensors()`' order, so
    `backward` must return them in that same order. Every one of them has a
    DIFFERENT shape from its neighbours only sometimes -- the read and write levels
    of one topology have the same shape -- so a transposed pair would land each
    gradient on the wrong tensor with no error anywhere. The check is that each
    operand's `.grad` matches its own shape and dtype, and that the read and write
    sides are not exchanged: perturbing one read level must move only read-level
    gradients."""
    v, routes, _x, _g = _built_arm()
    operands = routes.tensors()
    y, _ = rola.rola_op(routes, v)
    #: the bundle's tensors are the producer's OUTPUTS, so they are not leaves and
    #: `.grad` would be empty; the cotangents are asked for directly.
    grads = torch.autograd.grad(y.sum(), (v, *operands), allow_unused=False)
    for i, (operand, g) in enumerate(zip((v, *operands), grads)):
        assert g is not None, f"operand {i} received no gradient"
        assert g.shape == operand.shape, f"operand {i} shape mismatch: {g.shape}"
        assert g.dtype == operand.dtype, f"operand {i} dtype mismatch: {g.dtype}"
        assert torch.isfinite(g).all(), f"operand {i} gradient is not finite"


def test_a_cpu_bundle_is_refused_because_there_is_no_cpu_lowering():
    """The arm is a CUDA kernel and there is no CPU implementation to reach."""
    v, routes = _tiny_routing(requires_grad=False)
    with torch.no_grad():
        assert "CUDA kernel" in envelope_refusal(routes, v)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_envelope_admits_a_no_grad_cuda_bundle_on_a_built_arm():
    """The other side of the same decision, which a CPU-only run cannot see: with the
    gradient off and the operands on the device, an in-matrix bundle must actually be
    ADMITTED. Asserting only the refusals would let the arm rot behind them."""
    torch.manual_seed(5)
    gateway = build_routes(hidden_size=32, num_heads=1, widths=(8, 8),
                           routing=_dense_routing()).to("cuda")
    x = torch.randn(1, 32, 32, device="cuda")
    v = torch.randn(1, 32, 1, 64, device="cuda")
    with torch.no_grad():
        routes = gateway(x)
        assert envelope_refusal(routes, v) is None
        require_envelope(routes, v)


# --- the O-3 name contract ---------------------------------------------------

_PUBLIC_V3_EXEMPTIONS = {
    # None. If this set ever gains an entry, the entry must say why the name is
    # public AND must say `v3`, which is a hard thing to justify.
}


def test_no_public_name_in_the_package_says_v3():
    """O-3, checked mechanically over the package's own public surface: every
    module-level `def`/`class`/assignment whose name does not start with `_`, plus
    every `__all__` entry. Comments and docstrings are out of scope -- they cite the
    spec document `ROLA_V3_SPEC.md` and the `rola-v3` source branch, which are
    real proper nouns and not spellings of any function."""
    offenders = []
    for path in sorted(pathlib.Path(rola.__path__[0]).rglob("*.py")):
        tree = ast.parse(path.read_text())
        names = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(node.name)
            elif isinstance(node, ast.Assign):
                names.extend(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.append(node.target.id)
        for name in names:
            if name.startswith("_"):
                continue
            if "v3" in name.lower() and name not in _PUBLIC_V3_EXEMPTIONS:
                offenders.append(f"{path.relative_to(rola.__path__[0])}::{name}")
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
                    and isinstance(node.value, (ast.List, ast.Tuple))):
                for element in node.value.elts:
                    if (isinstance(element, ast.Constant) and isinstance(element.value, str)
                            and "v3" in element.value.lower()):
                        offenders.append(f"{path.relative_to(rola.__path__[0])}::__all__:{element.value}")
    assert not offenders, f"public names still saying v3: {offenders}"


def test_the_top_level_exports_are_exactly_the_declared_api():
    """`__all__` is the contract; a name that is exported but not declared is a name
    nobody promised and everybody will import anyway."""
    for name in rola.__all__:
        assert hasattr(rola, name), f"{name} is in __all__ but not importable"
    assert "rola_op" in rola.__all__
    assert not any(name.startswith("_") and name != "__version__" for name in rola.__all__), (
        "a private name must not be re-exported: the op's public spelling is rola_op")
