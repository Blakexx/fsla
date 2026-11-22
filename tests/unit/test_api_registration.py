# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""EVERY OP `csrc/fsla/fsla_api.cpp` IMPLEMENTS IS DEFINED, AND EVERY ONE DEFINED IS IMPLEMENTED: a `m.impl` without its `m.def` is silently
dropped by the dispatcher (the op then does not exist), a `m.def` without its `m.impl` is an op that cannot run. No GPU: the source is parsed."""
import re
from pathlib import Path

API = Path(__file__).resolve().parents[2] / "csrc" / "fsla" / "fsla_api.cpp"


def _names():
    text = API.read_text()
    defs = set(re.findall(r'm\.def\(\s*"(\w+)\(', text))
    impls = set(re.findall(r'm\.impl\(\s*"(\w+)"', text))
    return defs, impls


def test_every_implemented_op_has_a_definition_and_the_reverse():
    defs, impls = _names()
    assert defs and impls
    assert not impls - defs, f"implemented but never defined (silently dropped): {sorted(impls - defs)}"
    assert not defs - impls, f"defined but never implemented: {sorted(defs - impls)}"
