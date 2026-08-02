# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The public op stack (`rola/interface.py`) and the O-3 name contract.

Two things are gated here. The first is the DISPATCH -- which arm `rola_func` takes,
made explicit and asserted from both sides, because `[[layer-dispatch-autograd-gap]]`
is a failure that produces no exception and no wrong number, only a `grad_fn` that
quietly is not there. The second is the NAME contract: no public spelling says `v3`.

The CUDA arm's own numerics are gated in `tests/tier1_semantics/`; nothing here
re-measures them.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch

import rola
from rola.interface import BackwardNotImplemented, RoLAConsumer, requires_backward

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

def test_the_consumer_node_refuses_a_gradient_by_name_instead_of_severing_the_graph():
    """`RoLAConsumer.backward` is reachable only by bypassing `rola_func`. It must
    raise a named error; the failure being closed is the one where it does NOT."""
    with pytest.raises(BackwardNotImplemented, match="forward-only"):
        RoLAConsumer.backward(None, None)


def test_the_consumer_node_does_not_mark_its_output_non_differentiable():
    """Marking it would make the tripwire unreachable, which is the same silence the
    node exists to break. Read off the source, since the condition is an absence."""
    source = pathlib.Path(rola.interface.__file__).read_text()
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == "RoLAConsumer")
    calls = {a.attr for n in ast.walk(node) if isinstance(n, ast.Call)
             for a in [n.func] if isinstance(a, ast.Attribute)}
    assert "mark_non_differentiable" not in calls


# --- the dispatch, end to end ------------------------------------------------

def _tiny_routing(requires_grad: bool, device: str = "cpu"):
    torch.manual_seed(3)
    B, T, H, d_v, w = 1, 8, 1, 16, 4
    v = torch.randn(B, T, H, d_v, device=device, requires_grad=requires_grad)
    level = torch.softmax(torch.randn(B, T, H, w, device=device), dim=-1)
    level.requires_grad_(requires_grad)
    g_write = torch.rand(B, T, H, device=device) + 0.5
    g_write.requires_grad_(requires_grad)
    g_read = torch.rand(B, T, H, device=device) + 0.5
    g_read.requires_grad_(requires_grad)
    topology = rola.Topology(
        levels=(rola.LevelConfig(width=w, routing=_dense_routing()),), d_v=d_v)
    return v, (level,), (level,), g_read, g_write, topology


def _dense_routing():
    from rola.routing.types import IndependentRouting

    return IndependentRouting(
        tied=True,
        read_activation=rola.SoftmaxActivation(),
        write_activation=rola.SoftmaxActivation())


class _StubPlan:
    """A `Plan` stand-in carrying only what the reference arm reads.

    The reference arm needs `topology` and `normalization` and nothing else; using a
    real `Plan` here would drag in device-dependent selection to test a dispatch
    decision, which is the wrong thing to couple.
    """

    def __init__(self, topology, normalization="raw"):
        self.topology = topology
        self.normalization = normalization


def test_a_grad_requiring_call_lands_on_the_reference_arm_with_a_live_graph():
    """The whole point. The op must come back differentiable, from ONE
    implementation -- not a kernel value with an oracle gradient bolted on."""
    v, read, write, g_read, g_write, topology = _tiny_routing(requires_grad=True)
    assert requires_backward(v, read, write, g_read, g_write) is True
    y, final = rola.rola_func(v, read, write, g_read, g_write, _StubPlan(topology))
    assert y.grad_fn is not None, "the graph was severed at the op boundary"
    assert final is None
    y.sum().backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()


def test_the_reference_arm_refuses_consumer_only_options_rather_than_ignoring_them():
    """A flag honoured on one arm and silently ignored on the other is exactly
    `[[layer-dispatch-autograd-gap]]`'s shape of failure."""
    v, read, write, g_read, g_write, topology = _tiny_routing(requires_grad=True)
    with pytest.raises(ValueError, match="CUDA-consumer options"):
        rola.rola_func(v, read, write, g_read, g_write, _StubPlan(topology), paging=object())


def test_rola_forward_cannot_be_talked_into_building_a_graph():
    """`rola_forward` establishes the no-grad context itself, so an inference caller
    cannot be wrong about it by forgetting one somewhere up the stack. On CPU that
    means the reference arm runs, undifferentiably -- the assertion is about the
    absence of a graph, not about which device ran it."""
    v, read, write, g_read, g_write, topology = _tiny_routing(requires_grad=True)
    y, _ = rola.rola_forward(v, read, write, g_read, g_write, _StubPlan(topology))
    assert y.grad_fn is None and not y.requires_grad


def test_turning_the_gradient_off_changes_the_arm_and_nothing_else():
    """Same operands, grad off: no graph. Same operands, grad on: a graph. And the
    two agree exactly, because on CPU both arms ARE the reference -- which is the
    point of asserting it here: the dispatch decision is visible in the `grad_fn`
    and invisible in the numbers."""
    v, read, write, g_read, g_write, topology = _tiny_routing(requires_grad=True)
    plan = _StubPlan(topology)
    with torch.no_grad():
        y_off, _ = rola.rola_func(v, read, write, g_read, g_write, plan)
    y_on, _ = rola.rola_func(v, read, write, g_read, g_write, plan)
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
    assert "rola_func" in rola.__all__ and "rola_forward" in rola.__all__
    assert not any(name.startswith("_") and name != "__version__" for name in rola.__all__), (
        "a private name must not be re-exported: the op's public spelling is rola_func")
