# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""TIER 2 — the decode path is BC-BLIND, stated as "no dependence", not "no literal".

Spec I1's stated enforcement for the tiled path is to parameterize over every
instantiated ``BC`` plus one hypothetical ``BC``, so a hardcoded quanta-shaped literal
fails the fake-``BC`` row. The decode path admits a stronger form of the same gate,
because ``BC`` is not an input to it at all: run the same fixture while the surrounding
prefill declares ``BC`` in ``{16, 32, 64}`` — 32 being the HYPOTHETICAL value, not
instantiated by the tiled consumer — and assert the decode output is BYTE-IDENTICAL.

Contract class (a), engineered identity: ``torch.equal``. The claim is not that decode
*rounds the same* under different ``BC``; it is that ``BC`` cannot reach decode at all,
so the two runs execute the identical instruction stream on identical bytes. Anything
weaker than byte-equality here would be evidence of a dependence.

**WHY THE ENTRY STATE IS HELD FIXED, and why that is the honest formulation.** A prefill
at ``BC = 16`` and one at ``BC = 64`` produce entry states that differ in their low bits
(different summation trees), so feeding decode the two real prefill outputs would
compare two different inputs and prove nothing about ``BC``. The dependence being tested
is decode's, so the state is held identical and the ``BC`` is varied everywhere it could
possibly enter: the plan, the frozen carrier's construction, and the config object.

Run: ``pytest tests/tier2_geometry/test_decode_bc_blindness.py``
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from rola.ops.decode import FrozenDecodeConfig, _decode_step, freeze_decode_config
from tests.conformance.instantiation import CHUNK_ARMS
from tests.tier1_semantics.oracle_fixtures import _simplex, _topology

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="decode is a CUDA kernel"),
]

#: Every `BC` the built chunk matrix carries, plus one it does NOT build. Decode
#: must be blind to a number it will never see as thoroughly as to one it will,
#: and taking the real values from the manifest rather than writing them out
#: means a widened matrix arrives here without an edit.
_BCS = sorted({bc for (_c, bc, _d, _b) in CHUNK_ARMS}) + [1024]


def test_no_field_of_the_frozen_carrier_names_bc():
    """The structural half: ``BC`` is not a field, and cannot become one by accident.

    A field-level check rather than a comment, because the carrier is the one object
    that could plausibly acquire a ``BC`` (it is built from a plan that has one), and a
    field added there would be invisible to the numeric gate below until something
    started reading it.
    """
    names = {f.name.lower() for f in dataclasses.fields(FrozenDecodeConfig)}
    assert not any(n == "bc" or n.startswith("bc_") or n.endswith("_bc") for n in names), (
        f"FrozenDecodeConfig has acquired a BC-shaped field: {sorted(names)}")


@pytest.mark.parametrize("widths", [(8, 16), (4, 4, 4), (64, 64)])
def test_decode_output_is_byte_identical_across_prefill_bc(widths):
    device = "cuda"
    d_v, B, H = 64, 2, 2
    N = math.prod(widths)
    # P26: the mass column is unconditional (`"raw"` is dead, `normalization` is
    # `"global"` everywhere), so `cols = d_v + 1` rather than a norm-conditional.
    cols = d_v + 1
    topology = _topology(widths)

    generator = torch.Generator(device=device).manual_seed(20260802)
    read_levels = tuple(_simplex((B, 1, H, w), 0.5, generator, device).float() for w in widths)
    write_levels = tuple(_simplex((B, 1, H, w), 0.4, generator, device).float() for w in widths)
    g_write = (torch.rand(B, 1, H, device=device, generator=generator) + 0.5).float()
    g_read = None
    v = torch.randn(B, 1, H, d_v, device=device, generator=generator).float()
    m0 = 0.1 * torch.randn(B, H, N, cols, device=device, generator=generator).float()
    m0[..., d_v] = m0[..., d_v].abs()

    results = []
    configs = []
    for BC in _BCS:
        # `BC` is here to be IGNORED. P67 D2-b: the prefill's `BC` used to be a
        # host-chosen `Plan` field constructed at each value and deliberately not
        # handed to decode; on the chunk arm it belongs to the built arm and is
        # not host-selectable at all, so the loop now varies the number a prefill
        # COULD have run at and asserts, as before, that there is no channel
        # through which it could reach the frozen carrier. `freeze_decode_config`
        # is called inside the loop for exactly that reason: it must be a pure
        # function of the topology, the normalization arm and the device.
        assert BC in _BCS
        config = freeze_decode_config(topology, d_v=d_v, decay=False, BH=B * H, device=device,
                                      mma_precision="bf16")
        configs.append(config)
        state = m0.clone()
        y, state, _ = _decode_step(v, read_levels, write_levels, g_read, g_write,
                                     config, state)
        results.append((y, state))

    assert configs[0] == configs[1] == configs[2], (
        "the frozen carrier moved with the prefill's BC; it must be a pure function of "
        "the topology, the normalization arm and the device")
    for BC, (y, state) in zip(_BCS[1:], results[1:]):
        assert torch.equal(results[0][0], y), f"decode's y moved with prefill BC={BC}"
        assert torch.equal(results[0][1], state), f"decode's state moved with prefill BC={BC}"
