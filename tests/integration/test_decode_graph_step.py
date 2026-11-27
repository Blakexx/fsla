# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""THE GATED DECODE STEP UNDER A CUDA GRAPH: with no host read inside it, the step is capturable, and the
captured step is the eager step -- the same kernels in the same order, the same state bytes and `y` to its sums' order.

Needs a GPU and the built extension.

Run: ``bazel test //tests:integration --test_filter=test_decode_graph_step``
"""
from __future__ import annotations

from dataclasses import fields

import pytest
import torch
from conftest import reference_produce

from fsla import decode_step_gated
from fsla.ops.decode import lattice_of
from fsla.ops.lattice import to_lattice
from fsla.ops.paging import MMA_K_QUANTUM, bytes_equal, encode_pages
from fsla.routing.masks import DECODE
from fsla.routing.quant import VALUES, ScaledBlock
from tests.oracle.fixtures import assert_reordered_close

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="decode is a CUDA kernel"),
]


_HIDDEN, _H, _T, _B = 64, 2, 128, 2
_DV, _COLS = 64, 65
_WIDTHS = (64, 64)


def _sequence():
    """A prefilled dense state, the layer whose producer made it, and a stream of decode tokens.

    The state is SEEDED the facade's own way minus the launch: `_kernel_entry` binds the state, a zeroed
    plane is committed, and the fp64 oracle's `output_final_state` fills it, narrowed to the kernel's fp32,
    crossed into the lattice leaf order the plane's atoms are cut from and encoded in the stored page
    format. What this cell gates (capture, replay) reads the plane, and it does not care which arm deposited
    the numbers.
    """
    import fsla
    from fsla.engine.facts import planes
    from fsla.ops.naive import naive_fsla
    from tests.conftest import alternating_routing, build_layer

    torch.manual_seed(7)
    layer = build_layer(hidden_size=_HIDDEN, num_heads=_H, widths=_WIDTHS, d_v=_DV,
                        routing=alternating_routing(len(_WIDTHS))).cuda().eval()
    torch.manual_seed(507)
    x = torch.randn(_B, _T + 8, _HIDDEN, device="cuda")
    v = torch.randn(_B, _T + 8, _H, _DV, device="cuda")
    with torch.no_grad():
        routes, _values, _masks = reference_produce(layer, x[:, :_T])
        state = fsla.dense_state()
        state._kernel_entry(routes, d_v=_DV)
        state._kernel_commit(planes.state_plane(routes, d_v=_DV, zeros=True))
        _y_ref, oracle_state = naive_fsla(v[:, :_T], routes.dequantized()[0], routes.dequantized()[1], routes.topology, None, output_final_state=True)
        lattice = to_lattice(oracle_state.float(), _WIDTHS, *lattice_of(_WIDTHS))
        state._plane.copy_(encode_pages(lattice.reshape(_B * _H, -1, MMA_K_QUANTUM, _COLS)))
    return layer, x, v, state


def _cloned(val):
    """A field's value over CLONED tensors: a tensor, a scaled block's tensors (its form kept), and tuples of either."""
    if isinstance(val, torch.Tensor):
        return val.clone()
    if isinstance(val, ScaledBlock):
        return ScaledBlock(data=val.data.clone(), scale=val.scale.clone(), form=val.form)
    if isinstance(val, tuple):
        return tuple(_cloned(item) for item in val)
    return val


def _clone_tensors(obj):
    """The same record over CLONED tensors: every tensor field, scaled blocks and tuples of either included."""
    if isinstance(obj, ScaledBlock):
        return _cloned(obj)
    return type(obj)(**{f.name: _cloned(getattr(obj, f.name)) for f in fields(obj)})


def _copied(dst, src):
    if isinstance(dst, torch.Tensor):
        dst.copy_(src)
    elif isinstance(dst, ScaledBlock):
        dst.data.copy_(src.data)
        dst.scale.copy_(src.scale)
    elif isinstance(dst, tuple):
        for buffer, level in zip(dst, src):
            _copied(buffer, level)


def _copy_tensors(static, fresh):
    if isinstance(static, ScaledBlock):
        _copied(static, fresh)
        return
    for f in fields(static):
        _copied(getattr(static, f.name), getattr(fresh, f.name))


def _static(routes, v, masks):
    """The bundles and the decode-layout masks over buffers that never move -- what capture requires of its inputs."""
    return _clone_tensors(routes), _clone_tensors(ScaledBlock.quantize(v, VALUES)), _clone_tensors(masks)


def _fill(static, static_v, static_masks, routes, v, masks):
    _copy_tensors(static, routes)
    _copy_tensors(static_v, ScaledBlock.quantize(v, VALUES))
    _copy_tensors(static_masks, masks)


def _capture(static, static_v, static_masks, state):
    """Warm up on a side stream, then capture ONE gated step. Returns `(graph, y)`."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            decode_step_gated(static, static_v, static_masks, state)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y, _ = decode_step_gated(static, static_v, static_masks, state)
    return graph, y


@torch.no_grad()
def test_the_captured_step_is_the_eager_step():
    """Same kernels, same order, the same state bytes and `y` to its sums' order -- the capture owes exactly this."""
    layer, x, v, state = _sequence()
    token = slice(_T + 4, _T + 5)
    routes, _values, masks = reference_produce(layer, x[:, token], DECODE)
    static, static_v, static_masks = _static(routes, v[:, token], masks)
    _fill(static, static_v, static_masks, routes, v[:, token], masks)
    graph, y_graph = _capture(static, static_v, static_masks, state)

    plane = state._plane
    snapshot = plane.clone()

    graph.replay()
    torch.cuda.synchronize()
    captured_y, captured_plane = y_graph.clone(), plane.clone()
    assert not bytes_equal(captured_plane, snapshot), "the compared step did nothing"

    plane.copy_(snapshot)
    eager_y, _ = decode_step_gated(static, static_v, static_masks, state)
    torch.cuda.synchronize()
    assert_reordered_close(captured_y, eager_y, "the captured step's y")
    assert bytes_equal(captured_plane, plane)
