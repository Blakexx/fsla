# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""ARCH COVERAGE COMPLETENESS -- every ratified arch reaches every topology.

The dispatch degrades GRACEFULLY, and only because something is there to degrade
onto: `rola/engine/rules/arm.py`'s `ArmRule` picks the pinned `(C, BC)` when the
topology's manifest carries it and otherwise falls back to the largest built `BC`,
so a topology whose arch has SOME arm runs slower, and a topology whose arch has
NONE does not run at all. That difference is what this file gates, per
architecture: slower-not-nothing is a property of the built matrix, not of the
rule, and the rule cannot supply what was never measured.

THE FAILURE IT IS AGAINST is an edit to `tools/gen_shards.py`'s arm lists (or a
per-arch re-ratification that measures a subset) which drops the last arm of some
topology class on ONE arch. Nothing else notices: the shard partition still
partitions, the manifest digest still matches, the census is still internally
consistent, and every test that runs on the developer's own card still passes --
the hole is on the arch nobody ran. So the claim here is a CROSS-ARCH one: the
topology classes are collected as the UNION over the ratified manifests, and every
ratified manifest must cover the union.

A topology class is `(D, B)` -- depth and width, the shape a caller asks for.
`(C, BC)` is the answer, and which `(C, BC)` is the fallback's business.

TABULATED BUT NOT RATIFIED is reported, never failed: that state is the product
statement `tools/supported.py` prints (*not ratified -- will refuse to load*), and
an arch with no measurements has no coverage to be incomplete.

Device-free: the subject is `tools/manifests/`, which is what the extension was
gated against, and reading it needs no card. The manifest keys are decoded through
`tools/ratify.py`'s own decoders, so this file has no second spelling of an arm.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO / "tools" / "manifests"
sys.path.insert(0, str(REPO / "tools"))

ratify = pytest.importorskip("ratify")
supported = pytest.importorskip("supported")

#: `(D, B)` -- the topology a caller asks for, and the key both claims are stated in.
Topology = tuple[int, int]


def _arms(blob: dict) -> tuple[set[tuple], set[tuple]]:
    """One manifest's `(C, BC, D, B)` chunk arms and scan arms."""
    chunk, scan = set(), set()
    for key in blob["entries"]:
        mangled = key.split(":", 1)[1]
        shape = ratify.parse_shape(mangled)
        if shape is not None:
            chunk.add(ratify.arm_of_shape(shape)[:4])
            continue
        arm = ratify.parse_scan_arm(mangled)
        if arm is not None:
            scan.add(arm)
    return chunk, scan


def _ratified() -> dict[str, tuple[set[tuple], set[tuple]]]:
    """`sm_XX -> (chunk arms, scan arms)` for every TABULATED arch with a manifest.

    Tabulated is the qualifier that matters: a manifest for an arch the kernel cannot
    size itself for is `tools/supported.py`'s defect row, not a coverage subject.
    """
    out = {}
    for arch in supported.tabulated_archs():
        path = MANIFEST_DIR / f"{arch}.json"
        if path.is_file():
            out[arch] = _arms(json.loads(path.read_text()))
    return out


def _classes(arms: set[tuple]) -> set[Topology]:
    return {(d, b) for (_c, _bc, d, b) in arms}


def _covers(name: str, per_arch: dict[str, set[tuple]]) -> None:
    """Every arch covers the union of the topology classes, or say which is missing."""
    assert per_arch, "no tabulated arch has a ratified manifest; nothing was measured"
    union = set().union(*(_classes(arms) for arms in per_arch.values()))
    assert len(union) > 1, (
        f"the {name} arms name {len(union)} topology class(es), so 'every arch covers "
        "every class' is not a claim about anything -- either the arm list collapsed "
        "or this gate is reading the wrong field")
    for arch, arms in sorted(per_arch.items()):
        missing = sorted(union - _classes(arms))
        assert not missing, (
            f"{arch}'s manifest carries NO {name} arm for topology class(es) {missing}, "
            f"which other ratified arches do carry. A topology with no arm on an arch "
            f"does not run there at all -- ArmRule falls back to another BC, never to "
            f"another (D, B) -- so this is a hole, not a slower path. Restore an arm "
            f"for {missing} in tools/gen_shards.py and re-ratify {arch}, or drop the "
            f"class everywhere so that no arch claims it.")


def test_every_ratified_arch_has_a_chunk_arm_for_every_topology_class():
    """The FORWARD's coverage: what can be inferred at, per arch."""
    _covers("chunk", {arch: chunk for arch, (chunk, _scan) in _ratified().items()})


def test_every_ratified_arch_has_a_scan_arm_for_every_trainable_topology_class():
    """The BACKWARD's coverage, stated separately because the set is narrower.

    The scans carry the TRAINABLE topologies only (`tools/gen_shards.py`'s
    `SCAN_ARMS`), so their union is its own question: folding the two claims together
    would either demand a backward for every inference topology or accept a forward
    hole wherever the backward has none.
    """
    _covers("scan", {arch: scan for arch, (_chunk, scan) in _ratified().items()})


def test_the_scan_topologies_are_a_subset_of_the_chunk_topologies():
    """A backward whose forward is not built could not run a training step.

    `gen_shards.py` asserts this over the DECLARED lists; here it is asserted over
    what each arch actually ratified, which is the thing a per-arch re-measurement
    can break.
    """
    for arch, (chunk, scan) in sorted(_ratified().items()):
        extra = sorted(_classes(scan) - _classes(chunk))
        assert not extra, (
            f"{arch} ratifies scan arms for topology class(es) {extra} with no chunk "
            "arm to run their forward")


def test_tabulated_arches_without_a_manifest_are_the_documented_refusal(capsys):
    """Reported, not failed -- and the report is the row the README already prints."""
    unratified = sorted(set(supported.tabulated_archs()) - set(_ratified()))
    table = supported.table()
    for arch in unratified:
        assert f"| `{arch}` |" in table and "will refuse to load" in table.split(
            f"| `{arch}` |", 1)[1].split("\n", 1)[0], (
            f"{arch} is tabulated with no manifest, which is the 'will refuse to load' "
            "row; the generated support table does not say so")
    with capsys.disabled():
        print(f"\ntabulated, not ratified (no coverage claim): {unratified or 'none'}")
