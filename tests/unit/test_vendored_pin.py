"""CLOSED-WORLD RULE 5, checkable without a build.

`setup.py::_gate_vendored` refuses a compile whose vendored CUTLASS/CuTe subtree does
not match `csrc/third_party/cutlass/PIN.json`. That gate only runs when somebody
builds; these tests run in CI on a machine with no CUDA, and they check the three
claims the pin makes:

1. the recorded digest describes the subtree that is actually on disk;
2. the redistribution obligations BSD-3-Clause imposes are discharged in-tree;
3. the two places that compile against the vendored headers -- the extension
   (`setup.py`) and the ratification probe (`tools/ratify.py`) -- pass the SAME
   include path. A gate that compiles the probe under a different include set than
   the shipped kernel is not measuring the shipped kernel.

`setup.py` is read as TEXT rather than imported: importing it executes `setup()`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CUTLASS_DIR = ROOT / "csrc" / "third_party" / "cutlass"
PIN_PATH = CUTLASS_DIR / "PIN.json"


def _pin() -> dict:
    if not PIN_PATH.exists():
        pytest.skip(f"no vendored pin at {PIN_PATH}")
    return json.loads(PIN_PATH.read_text())


def _tree_sha256(root: Path) -> tuple[str, int]:
    """The recipe PIN.json states, re-implemented from the RECIPE rather than
    imported from `setup.py` -- a digest checked by the same code that wrote it
    checks nothing."""
    digest = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "PIN.json")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest(), len(files)


def test_pin_records_every_field_the_build_and_the_notice_rely_on():
    pin = _pin()
    for key in ("upstream", "tag", "commit", "license", "tree_sha256", "file_count",
                "modifications"):
        assert pin.get(key), f"PIN.json is missing `{key}`"
    assert len(pin["commit"]) == 40, "the pin must be a full commit sha, not a tag alone"
    assert pin["modifications"].startswith("none"), (
        "a MODIFIED vendored subtree cannot be checked against the upstream tag, which "
        "is the only thing that makes the digest more than self-referential")


def test_vendored_subtree_matches_its_digest():
    pin = _pin()
    here, count = _tree_sha256(CUTLASS_DIR)
    assert count == pin["file_count"], (
        f"vendored file count moved: {count} on disk, {pin['file_count']} pinned")
    assert here == pin["tree_sha256"], (
        "the vendored CUTLASS subtree does not match its pin -- a vendored header is "
        "code `ptxas` assembles, so this is a codegen change and not a packaging one")


def test_redistribution_obligations_are_discharged_in_tree():
    pin = _pin()
    license_text = (CUTLASS_DIR / pin["license_file"]).read_text()
    assert "Redistribution and use in source and binary forms" in license_text
    assert "NVIDIA CORPORATION" in license_text
    notice = (ROOT / "NOTICE").read_text()
    assert pin["commit"] in notice, "NOTICE must name the exact vendored commit"
    assert "BSD-3-Clause" in notice


def test_the_extension_and_the_ratification_probe_share_the_include_path():
    setup_text = (ROOT / "setup.py").read_text()
    ratify_text = (ROOT / "tools" / "ratify.py").read_text()
    needle = '"csrc" / "third_party" / "cutlass"'
    assert needle in setup_text, "setup.py no longer passes the vendored include path"
    assert needle in ratify_text, (
        "tools/ratify.py no longer passes the vendored include path; the manifest "
        "would then be ratified against a compile the extension does not perform")
    assert "_gate_vendored" in setup_text, "closed-world rule 5 gate is gone"


def test_the_built_binary_records_which_cutlass_is_inside_it():
    build_config = pytest.importorskip("rola._build_config")
    vendored = build_config.show().get("vendored")
    assert vendored is not None, (
        "this `.so` was built by a setup.py that did not stamp its vendored codegen "
        "inputs; rebuild it")
    assert vendored["cutlass"]["tree_sha256"] == _pin()["tree_sha256"]
