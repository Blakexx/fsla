# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CODEGEN-EQUIVALENCE INSTRUMENT (`tools/sass_bodies.py`): its BODY HASH can fail both ways -- two different instruction texts
hash apart, the same text laid out with other padding hashes alike, a function emitted twice with two bodies is refused -- and its
RENAME RULE (docs/internals/common/hardware.md#builds-renamed): a kernel's own-image `Build` argument is dropped from its
signature, another image's is refused, and the source stamp is reported, never a verdict."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import sass_bodies  # noqa: E402

#: `fsla::k<Arm<3>>(int)` and `fsla::k<BuildTag<Build{900}>, Arm<3>>(int)`, as nvcc 13.0 mangles them
OLD = "sm_90a:_ZN4fsla1kINS_3ArmILi3EEEEEvi"
NEW = "sm_90a:_ZN4fsla1kIN8denseref4arch8BuildTagIXtlNS2_5BuildELi900EEEEENS_3ArmILi3EEEEEvi"
STAMP = "sm_90a:_ZN4fsla8stampdet17csrc_stamp_kernelEPl"


def test_an_own_image_build_argument_is_dropped():
    assert sass_bodies.by_signature({NEW: "h"}, True) == sass_bodies.by_signature({OLD: "h"}, True)


def test_another_images_build_argument_is_refused():
    with pytest.raises(SystemExit, match="in the sm_100a image"):
        sass_bodies.by_signature({NEW.replace("sm_90a:", "sm_100a:"): "h"}, True)


def test_the_source_stamp_is_never_a_verdict():
    a = sass_bodies.by_signature({OLD: "h", STAMP: "s1"}, True)
    b = sass_bodies.by_signature({NEW: "h", STAMP: "s2"}, True)
    assert sass_bodies.compare(a, b, "a", "b")
    assert not sass_bodies.compare(a, {**b, "sm_90a:fsla::k<fsla::Arm<3> >(int)": "moved"}, "a", "b")


#: three functions in `cuobjdump -sass`'s form: two whose instruction text differs, and the first again with other padding
MOVED = ["arch = sm_90a",
         "                Function : k_a",
         "        /*0000*/ IADD3 R3, R3, 1, RZ ;",
         "                                     /* 0x000fe200078e0203 */",
         "                Function : k_b",
         "        /*0000*/ IADD3 R3, R3, 2, RZ ;",
         "                                     /* 0x000fe200078e0203 */",
         "                Function : k_a_repadded",
         "   /*0000*/     IADD3   R3,  R3, 1, RZ ;",
         "     /* 0x000fe200078e0203 */"]


def test_the_body_hash_separates_changed_code_and_ignores_padding():
    got = sass_bodies.scan(MOVED)
    assert got["sm_90a:k_a"] != got["sm_90a:k_b"]
    assert got["sm_90a:k_a"] == got["sm_90a:k_a_repadded"]


def test_a_function_emitted_twice_with_two_bodies_is_refused():
    with pytest.raises(SystemExit, match="DIFFERENT bodies"):
        sass_bodies.scan(MOVED[:4] + ["                Function : k_a", "        /*0000*/ EXIT ;"])
