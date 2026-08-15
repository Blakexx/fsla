# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The public op stack (`rola/interface.py`) and the O-3 name contract.

Two things are gated here. The first is the DISPATCH -- which arm `rola_op` takes,
made explicit and asserted from both sides, because the layer-dispatch autograd gap
is a failure that produces no exception and no wrong number, only a `grad_fn` that
quietly is not there. The second is the NAME contract: no public spelling says `v3`.

The CUDA arm's own numerics are gated in `tests/tier1_semantics/`; nothing here
re-measures them.

**MIGRATED (API-contract break, 2026-08-05).** `rola_func(v, read, write, g_read,
g_write, plan, ...)` and the separate inference entrypoint `rola_forward(...)` are
both gone, replaced by the ONE op `rola_op(routes: RouteFactors, v, decay=None, *,
...)`. There is no more `plan` argument (the op derives its own) and no more
`_StubPlan` -- the reference arm reads a `RouteFactors` bundle's own `.topology`/
`.normalization`, so the dispatch fixture below builds a real bundle through
`RouteProducer` instead of five loose tensors plus a plan stand-in.
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
    import rola.layers.rola as layer_module
    from rola.interface import select_backend

    assert layer_module.select_backend is select_backend


# --- the tripwire ------------------------------------------------------------

#: RETIRED at P67 D2-b: the two `RoLAConsumer` rows (the named refusal, and the
#: absence of `mark_non_differentiable`). Their subject is the TILED arm's autograd
#: node, which D2-c deletes with its consumer. The CLAIM did not retire with it --
#: D2-a's R-10 wrapper gave the chunk arm the same node, and the two rows below are
#: its twins, written against `ChunkConsumer` and asserting the same two things.


def test_the_chunk_node_refuses_a_gradient_by_name_instead_of_severing_the_graph():
    """`ChunkConsumer` carries `RoLAConsumer`'s claim for the arm that replaced it.

    `rola.ops.chunk.run` is a bare extension call, so an arm that reached it from a
    gradient-requiring context would hand back a tensor with no `grad_fn` and no
    error. The dispatch is what prevents that; this node is what makes a bypassed
    dispatch loud."""
    with pytest.raises(BackwardNotImplemented, match="forward-only"):
        ChunkConsumer.backward(None, None)


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
    """The claim in the form the layer-dispatch autograd gap states it.

    Inside a grad-enabled context with grad-requiring operands, a call must either
    reach the reference arm (the facade's decision) or raise (the node's). What it
    must NOT do is return a good-looking tensor whose graph is gone."""
    v, routes = _tiny_routing(requires_grad=True, device="cuda")
    y, _ = rola.rola_op(routes, v)
    assert y.grad_fn is not None, "the graph was severed at the op boundary"
    y.sum().backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_node_is_what_makes_a_bypassed_dispatch_loud_rather_than_silent():
    """The last line, exercised: `_arm="kernel"` is a caller overruling the dispatch.

    The kernel then runs on grad-requiring operands -- which is exactly the case the
    dispatch exists to prevent -- and the node is the whole of what stands between a
    severed graph and an error. `grad_fn` present is the node having attached; the
    named raise is it refusing. Without the wrapper both assertions read the other
    way round: no `grad_fn`, no error, and `v.grad is None` several files away."""
    torch.manual_seed(5)
    B, T, H, d_v, hidden, widths = 1, 32, 1, 64, 32, (8, 8)
    gateway = build_routes(
        hidden_size=hidden, num_heads=H, widths=widths,
        routing=_dense_routing()).to("cuda")
    x = torch.randn(B, T, hidden, device="cuda", requires_grad=True)
    routes = gateway(x)
    v = torch.randn(B, T, H, d_v, device="cuda", requires_grad=True)
    from rola.ops import chunk as _chunk

    assert _chunk.unsupported(routes, None, d_v=v.shape[-1]) is None, "fixture must be a BUILT arm"
    y, _ = rola.rola_op(routes, v, _arm="kernel")
    assert y.grad_fn is not None, (
        "the kernel arm returned a grad_fn-less tensor inside a grad context -- the "
        "graph is severed and nothing said so")
    with pytest.raises(BackwardNotImplemented, match="forward-only"):
        y.sum().backward()


# --- the dispatch, end to end ------------------------------------------------

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
    `RouteFactors`, obtainable only from `RouteProducer`, so the fixture builds
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


def test_a_grad_requiring_call_lands_on_the_reference_arm_with_a_live_graph():
    """The whole point. The op must come back differentiable, from ONE
    implementation -- not a kernel value with an oracle gradient bolted on."""
    v, routes = _tiny_routing(requires_grad=True)
    assert requires_backward(v, *routes.tensors()) is True
    y, final = rola.rola_op(routes, v)
    assert y.grad_fn is not None, "the graph was severed at the op boundary"
    assert final is None
    y.sum().backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()


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


def test_an_explicit_no_grad_context_around_rola_op_builds_no_graph():
    """OLD: `rola_forward` was a separate entrypoint that established its own
    no-grad context internally, so an inference caller could not be wrong about
    it by forgetting to wrap the call. `rola_op` merges the two entrypoints into
    one: there is no separate inference spelling any more, and the caller's own
    `torch.no_grad()` is what an inference site now uses -- exactly as any other
    PyTorch op. This pins the surviving half of the old claim: wrapped in
    `torch.no_grad()`, `rola_op` produces no graph even though every input
    requires grad.
    """
    v, routes = _tiny_routing(requires_grad=True)
    with torch.no_grad():
        y, _ = rola.rola_op(routes, v)
    assert y.grad_fn is None and not y.requires_grad


def test_turning_the_gradient_off_changes_the_arm_and_nothing_else():
    """Same operands, grad off: no graph. Same operands, grad on: a graph. And the
    two agree exactly, because on CPU both arms ARE the reference -- which is the
    point of asserting it here: the dispatch decision is visible in the `grad_fn`
    and invisible in the numbers."""
    v, routes = _tiny_routing(requires_grad=True)
    with torch.no_grad():
        y_off, _ = rola.rola_op(routes, v)
    y_on, _ = rola.rola_op(routes, v)
    assert y_off.grad_fn is None and y_on.grad_fn is not None
    assert torch.allclose(y_off, y_on.detach())


def test_select_backend_sends_cpu_operands_to_the_reference_arm():
    """Condition 2 of the dispatch: the consumer is a CUDA kernel, so on CPU the
    reference is the only implementation that exists FOR THAT DEVICE. A device
    dispatch, not a degradation -- there is no CPU lowering being declined."""
    from rola.interface import select_backend

    cpu = torch.zeros(2)
    with torch.no_grad():
        assert select_backend(cpu) == "reference"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_select_backend_sends_no_grad_cuda_operands_to_the_consumer_arm():
    """The other side of the same decision, which a CPU-only run cannot see: with a
    gradient off and the operands on the device, the dispatch must actually choose
    the kernel. Asserting only the reference side would let the consumer arm rot."""
    from rola.interface import select_backend

    on_device = torch.zeros(2, device="cuda")
    with torch.no_grad():
        assert select_backend(on_device, (on_device,)) == "cuda"
    leaf = torch.zeros(2, device="cuda", requires_grad=True)
    assert select_backend(on_device, (leaf,)) == "reference"


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
