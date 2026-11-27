"""Focused tests for the routing declaration types: the topology, the tied marker and the packed router layout."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from fsla import Routing, entmax, softmax, tied
from fsla.routing import activations
from fsla.routing.projection import RouterLayout, router_logits
from fsla.routing.types import EntmaxActivation, OpaqueActivation, SoftmaxActivation, Tied, Topology


def test_the_tied_marker_is_a_marker_and_reads_the_write_side():
    assert tied() == Tied() and type(tied()) is Tied
    topology = Topology(widths=(4, 8), write=(entmax(1.5), softmax()), read=((tied(), entmax(2.0)),))
    assert topology.read_activation(0, 0) == EntmaxActivation(1.5)
    assert topology.read_activation(0, 1) == EntmaxActivation(2.0)
    assert topology.can_zero("read", 0) and topology.can_zero("write", 0)
    assert topology.can_zero("read", 1) and not topology.can_zero("write", 1)
    #: a tied softmax level is legal and dense on both sides
    dense = Topology(widths=(3,), write=(softmax(),), read=((tied(),),))
    assert not dense.can_zero("read", 0) and not dense.can_zero("write", 0)


def test_the_topology_derives_its_shape_and_is_frozen():
    topology = Topology(widths=(2, 3, 5, 4), write=(softmax(),) * 4, read=((softmax(),) * 4,))
    assert topology.D == 4 and topology.N == 120
    assert topology.radix_strides == (60, 20, 4, 1)
    assert topology.padded_widths == (16, 16, 16, 16)
    assert topology.padded() == Topology(widths=(16,) * 4, write=topology.write, read=topology.read)
    with pytest.raises(FrozenInstanceError):
        topology.widths = ()


def test_the_read_heads_are_the_declared_tuples_or_one_for_every_head():
    one = Topology(widths=(4,), write=(softmax(),), read=((entmax(1.5),),))
    assert one.read_heads(3) == ((entmax(1.5),),) * 3
    two = Topology(widths=(4,), write=(softmax(),), read=((entmax(1.5),), (tied(),)))
    assert two.read_heads(2) == two.read
    with pytest.raises(ValueError, match="G=3"):
        two.read_heads(3)
    #: a head that may make zeros on a level makes the level's read side one that may
    assert two.can_zero("read", 0)


@pytest.mark.parametrize(
    "factory,error,match",
    [
        (lambda: EntmaxActivation(True), TypeError, "alpha"),
        (lambda: EntmaxActivation(1), ValueError, "ratified"),
        (lambda: EntmaxActivation(1.8), ValueError, "ratified"),
        (lambda: Topology(widths=(4, 4), write=(softmax(),), read=((softmax(), softmax()),)), ValueError, "D=2"),
        (lambda: Topology(widths=(4, 4), write=(softmax(), softmax()), read=((softmax(),),)), ValueError,
         "depth D=2"),
        (lambda: Topology(widths=(4,), write=("softmax",), read=((softmax(),),)), TypeError, "registry"),
        (lambda: Topology(widths=(0,), write=(softmax(),), read=((softmax(),),)), ValueError, ">= 1"),
        (lambda: Topology(widths=(257,), write=(softmax(),), read=((softmax(),),)), ValueError, "<= 256"),
        (lambda: Topology(widths=(4,), write=(softmax(),), read=()), ValueError, "read head"),
    ],
)
def test_invalid_topologies_are_rejected_naming_the_reason(factory, error, match):
    with pytest.raises(error, match=match):
        factory()


def test_an_activation_whose_row_is_not_nonnegative_is_refused(monkeypatch):
    """v1's routing is nonnegative and its readout normalized: a registry row that is not is refused at the
    declaration, on either side, by the column and never by a name."""
    monkeypatch.setitem(activations.ACTIVATION_REGISTRY, ("softmax", None),
                        activations.ActivationProperties(exact_zeros=False, nonnegative=False))
    with pytest.raises(ValueError, match="nonnegative"):
        Routing(topology=(4,), write=(softmax(),), read=((tied(),),))
    with pytest.raises(ValueError, match="nonnegative"):
        Topology(widths=(4,), write=(entmax(1.5),), read=((softmax(),),))


@pytest.mark.parametrize("read_alpha,write_alpha", [(1.5, 1.5), (1.5, 2.0)])
def test_an_untied_sparse_read_over_a_sparse_write_is_refused(read_alpha, write_alpha):
    with pytest.raises(ValueError, match="tied"):
        Routing(topology=(4,), write=(entmax(write_alpha),), read=((entmax(read_alpha),),))
    Routing(topology=(4,), write=(entmax(write_alpha),), read=((tied(),),))


def test_the_routing_refuses_its_malformed_declarations_by_name():
    with pytest.raises(TypeError, match="tuple of ints"):
        Routing(topology=[4], write=(softmax(),), read=((softmax(),),))
    with pytest.raises(TypeError, match="TUPLES"):
        Routing(topology=(4,), write=(softmax(),), read=(softmax(),))
    with pytest.raises(ValueError, match="at least 2"):
        Routing(topology=(1,), write=(softmax(),), read=((softmax(),),))
    with pytest.raises(TypeError, match="registry"):
        Routing(topology=(4,), write=(tied(),), read=((softmax(),),))
    with pytest.raises(TypeError, match="solves"):
        Routing(topology=(4,), write=(OpaqueActivation(),), read=((softmax(),),))
    routing = Routing(topology=(4, 8), write=(softmax(), softmax()), read=((softmax(), softmax()), (tied(), tied())))
    with pytest.raises(ValueError, match="G=3"):
        routing.layout(3)
    assert routing.layout(2).group == 2


def test_the_packed_layout_is_the_write_levels_then_each_read_heads_untied_levels():
    """``route_W`` carries NO column that is not read: the write levels' columns, in level order, then each read head's
    untied levels' (head 0's, then head 1's), each at its level's width; a tied entry has none."""
    topology = Topology(widths=(4, 2, 3), write=(softmax(),) * 3,
                        read=((softmax(), tied(), softmax()), (tied(), softmax(), tied())))
    layout = RouterLayout(topology, 2)
    assert layout.level_offsets == (0, 4, 6, 9)
    assert [layout.write_columns(level) for level in range(3)] == [slice(0, 4), slice(4, 6), slice(6, 9)]
    assert layout.read_columns(0, 0) == slice(9, 13)
    assert layout.read_columns(0, 1) is None
    assert layout.read_columns(0, 2) == slice(13, 16)
    assert layout.read_columns(1, 0) is None
    assert layout.read_columns(1, 1) == slice(16, 18)
    assert layout.read_columns(1, 2) is None
    assert layout.packed_router_width == 18
    assert layout.tied_heads(0) == (1,) and layout.tied_heads(1) == (0,)
    #: one read tuple for every head lays out each head's columns in turn
    assert RouterLayout(Topology(widths=(4,), write=(softmax(),), read=((softmax(),),)), 3).packed_router_width == 16


def test_the_support_words_hold_a_column_for_every_solve_that_makes_zeros():
    topology = Topology(widths=(4, 2, 3), write=(entmax(1.5), softmax(), softmax()),
                        read=((tied(), entmax(2.0), softmax()),))
    layout = RouterLayout(topology, 1)
    assert layout.write_support_offsets == (0, -1, -1) and layout.write_support_width == 4
    assert layout.read_support_offsets == (-1, 0, -1) and layout.read_support_width == 2


def test_the_projection_validates_the_packed_parameter_against_the_layout():
    routing = Routing(topology=(2, 3), write=(softmax(), softmax()), read=((softmax(), softmax()),), gain=False)
    layout = routing.layout(1)
    #: write-0, write-1, read-0, read-1: 2, 3, 2, 3 -- 10 columns, every one consumed
    assert layout.packed_router_width == routing.columns(1) == 10
    x = torch.randn(1, 3, 5)
    logits, gain = router_logits(x, torch.randn(2, 5, 10), layout, route_bias=torch.randn(2, 10),
                                 projection_dtype=torch.float32)
    assert tuple(logits.shape) == (2, 3, 10) and gain is None
    with pytest.raises(ValueError, match="10 routing"):
        router_logits(x, torch.randn(2, 5, 12), layout, projection_dtype=torch.float32)
    with pytest.raises(ValueError, match="route_bias"):
        router_logits(x, torch.randn(2, 5, 10), layout, route_bias=torch.randn(2, 12), projection_dtype=torch.float32)


def test_a_wide_level_is_bounded_without_limiting_total_capacity():
    assert Topology(widths=(256,) * 3, write=(softmax(),) * 3, read=((SoftmaxActivation(),) * 3,)).N == 256 ** 3
