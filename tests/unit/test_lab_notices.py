# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The lab's compile-notice checks (tools/lab.py): `STEPS_SERIALIZED`, the toolkit's notices that it serialized a
kernel's asynchronous MMAs (ptxas C7512 for want of registers, C7520 for a tensor step under a branch), read from
notice text in ptxas's form and kept beside a build's binary."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import lab  # noqa: E402

C7512 = ("ptxas info    : (C7512) Potential Performance Loss: wgmma.mma_async instructions are serialized due to "
         "insufficient register resources for the wgmma pipeline in the function '{}'")
C7520 = ("ptxas info    : (C7520) Potential Performance Loss: wgmma.mma_async instructions are serialized due to "
         "program dependence on compiler-inserted WG.AR in divergent path in the function '{}'")
NOISE = "ptxas info    : Used 90 registers, used 1 barriers, 4096 bytes smem in the function '{}'"


def binary_with(tmp_path, *lines):
    binary = tmp_path / "x.sm_90a"
    lab.notices(binary).write_text("\n".join(lines) + "\n")
    return binary


def test_each_serializing_notice_is_counted_a_kernel_by_its_code(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "demangle", lambda names: {n: n.upper() for n in names})
    binary = binary_with(tmp_path, C7512.format("a"), C7520.format("a"), C7520.format("a"), C7520.format("b"),
                         NOISE.format("a"))
    assert lab.serialized(binary) == {"A": Counter({"C7512": 1, "C7520": 2}), "B": Counter({"C7520": 1})}


def test_other_notices_are_not_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "demangle", lambda names: {n: n for n in names})
    assert lab.serialized(binary_with(tmp_path, NOISE.format("a"), "ptxas info    : (C7510) something else in "
                                                                    "the function 'a'")) == {}


def test_a_build_with_no_notices_file_has_none(tmp_path):
    assert lab.serialized(tmp_path / "carried.sm_90a") == {}
