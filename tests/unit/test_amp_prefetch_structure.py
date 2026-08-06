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

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.cuda

ROOT = Path(__file__).resolve().parents[2]

#: BT16 BC64 DV64 D2, (decay, global) in the mangling's trailing bools; prec enum
#: value 1 is bf16, 0 is tf32x3 (both template slots carry the same point).
_FLAGSHIP_BF16 = ("_ZN4rola8consumer23consumer_forward_kernelILi16ELi64ELi64ELi2E"
                  "LNS_7MmaPrecE1ELS2_1ELb0ELb1EEEvNS_14ConsumerParamsE")
_FLAGSHIP_TF32 = ("_ZN4rola8consumer23consumer_forward_kernelILi16ELi64ELi64ELi2E"
                  "LNS_7MmaPrecE0ELS2_0ELb0ELb1EEEvNS_14ConsumerParamsE")


def _sass(fun: str) -> str:
    so = next((ROOT / "rola").glob("_C*.so"))
    return subprocess.run(
        ["cuobjdump", "-arch", "sm_86", "-sass", "-fun", fun, str(so)],
        capture_output=True, text=True, check=True).stdout


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
    #: The derivation's landing map at the query pin, as measured at the ratified
    #: prefetch build. OFF rows are arms the footprint rule declines (the slab
    #: would cost a CTA) or the census scope excludes (`BC = 16`'s 64-register
    #: budget acquired in-loop spill from the issue walk; the P00-precedent
    #: measured scope). Each flip is a footprint event, not noise.
    (False, 64, 16, 64, True),   # the flagship class
    (False, 16, 16, 64, False),  # census scope: 64-register wall spills
    (False, 16, 16, 32, False),  # census scope
    (False, 64, 16, 32, False),  # 4-CTA arm; the slab would cost one
    (False, 64, 32, 32, False),
    (False, 64, 32, 64, True),
    (True, 64, 16, 64, False),   # decay fp32 slab does not fit
    (True, 16, 16, 64, False),   # census scope
])
def test_the_derivation_landing_map(decay, BC, BT, DV, expected):
    assert _ext().rola_consumer_amp_prefetch(decay, BC, BT, DV) is expected


def test_every_issued_copy_has_its_wait_in_the_sass():
    """THE STALE-READ SEAM, made structural. A skipped `cp.async.wait_group`
    (mutant M-PF1) cannot be caught by the value battery on quiet timing -- the
    wait is nearly always settled by a full tile of runway, and racecheck is
    dormant on this kernel (its counter-protocol edges are the tool's blind spot).
    The order is therefore asserted the way the stream-discipline fixture asserts
    host-side order: by PRESENCE in the artifact. The flagship prefetch arm's SASS
    must carry the copies (LDGSTS ... ZFILL), the commit (LDGDEPBAR) and BOTH
    waits (DEPBAR at scoreboard zero: the prologue's and the loop's); a build
    where the copies exist and a wait does not is refused here, deterministically,
    whatever the timing."""
    sass = _sass(_FLAGSHIP_BF16)
    assert "LDGSTS" in sass and "ZFILL" in sass, "the prefetch copies are missing"
    assert sass.count("LDGDEPBAR") >= 2, "the commit_group sites are missing"
    assert sass.count("DEPBAR.LE SB0, 0x0") >= 2, (
        "the cp.async waits are missing: an issued copy with no wait before the "
        "publishing barrier is the planted early-commit this gate exists to refuse")


def test_a_non_prefetch_arm_carries_no_async_copy():
    """The derivation's OFF answer matches codegen: the tf32x3 sibling of the
    flagship (census-scoped out) must contain no LDGSTS at all."""
    assert "LDGSTS" not in _sass(_FLAGSHIP_TF32)
