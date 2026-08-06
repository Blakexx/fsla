# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""The amplitude-prefetch derivation is ON where the mechanism is claimed.

`consumer_amp_prefetch` is a per-instantiation FOOTPRINT derivation
(`docs/internals/smem_layout.md#amp-prefetch`): if a regression quietly flipped it
false everywhere, every correctness gate would stay green while measuring the direct
fetch path -- the exact vacuity the matching-the-naive lesson warns about. This file
pins the derivation's measured landing points through the extension's own query
(`rola_consumer_amp_prefetch`, the same `D = 2`/`global`/bf16 query pin as
`rola_consumer_smem_bytes`), and pins the one arithmetic consequence a reader can
check by hand: an enabled arm's footprint grew past its depth-1 carve by the slab.

The pins are MEASURED derivation outputs, not preferences: they were read off the
first prefetch build and any change to them is a real footprint event that must be
re-adjudicated, not absorbed.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.cuda


def _ext():
    from rola.ops import _ext
    return _ext.extension()


def test_the_flagship_class_arm_carries_the_prefetch():
    """BT16 BC64 DV64 bf16 global !decay -- the campaign's measured cell -- is ON."""
    assert _ext().rola_consumer_amp_prefetch(False, 64, 16, 64) is True


def test_an_enabled_arm_pays_exactly_a_slab_no_more_than_the_budget():
    """The enabled flagship arm's smem grew, and by less than the worst-case slab
    plus alignment -- the footprint is the slab and nothing else rode along."""
    smem = _ext().rola_consumer_smem_bytes(False, 64, 16, 64)
    #: depth-1 carve measured pre-slab (ring_depth.md, session record): 27,096 B.
    grown = smem - 27096
    assert 0 < grown <= 6144, smem


@pytest.mark.parametrize("decay,BC,BT,DV,expected", [
    #: The derivation's landing map at the query pin, as measured at the first
    #: prefetch build. OFF rows are arms where the slab costs a CTA; each flip is
    #: a footprint event, not noise.
    (False, 64, 16, 64, True),   # the flagship class
    (False, 16, 16, 64, True),
    (False, 16, 16, 32, True),
    (False, 64, 16, 32, False),  # 4-CTA arm; the slab would cost one
    (False, 64, 32, 32, False),
    (True, 64, 16, 64, False),   # decay fp32 slab does not fit
    (True, 16, 16, 64, True),    # decay arm small enough to fit -- decay is NOT excluded
])
def test_the_derivation_landing_map(decay, BC, BT, DV, expected):
    assert _ext().rola_consumer_amp_prefetch(decay, BC, BT, DV) is expected
