"""THE FRESHNESS GATE'S OWN GATE.

A checker that cannot FAIL is not a checker, and this one is cheap enough
(hashing a source tree) that its whole risk is silent success: a wrong path, a
renamed manifest key, an exception swallowed into "clean". So the negative
control is the point of this file -- each test below corrupts a real tree copy
in one specific way and demands the complaint.

Provenance: the defect this gate closes was real (K35-E `07c4409` shipped a
manifest ratified against sources one comment-sweep old, corrected at
`7ee9278`). No GPU, no compile.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import manifest_fresh  # noqa: E402
import ratify  # noqa: E402


def test_this_checkout_is_fresh():
    """The repository ships manifests that describe its own sources."""
    assert manifest_fresh.stale_arches(ROOT) == []


@pytest.fixture
def tree(tmp_path):
    """A minimal copy of the real tree: the sources, and the manifests beside them."""
    (tmp_path / "csrc" / "rola").mkdir(parents=True)
    shutil.copytree(ROOT / "csrc" / "rola" / "src", tmp_path / "csrc" / "rola" / "src")
    shutil.copytree(ROOT / "tools" / "manifests", tmp_path / "tools" / "manifests")
    assert manifest_fresh.stale_arches(tmp_path) == []
    return tmp_path


def _manifest(tree, arch):
    return tree / "tools" / "manifests" / f"sm_{arch}.json"


def test_a_source_edit_makes_every_manifest_stale(tree):
    """THE DEFECT ITSELF: the sources move, the manifests do not."""
    victim = next((tree / "csrc" / "rola" / "src").rglob("*.cuh"))
    victim.write_bytes(victim.read_bytes() + b"\n// one more comment\n")
    problems = manifest_fresh.stale_arches(tree)
    assert len(problems) == len(ratify.DEFAULT_ARCHES)
    assert all("ratified against csrc" in p for p in problems)


def test_a_comment_only_edit_is_not_exempt(tree):
    """The edit that slipped through in the field was behaviour-preserving.

    The gate must not try to be clever about which edits 'matter' -- the digest
    is the record of what was measured, and a comment sweep moves it.
    """
    victim = next((tree / "csrc" / "rola" / "src").rglob("*.cu"))
    text = victim.read_text()
    victim.write_text("// a purely editorial line\n" + text)
    assert manifest_fresh.stale_arches(tree) != []


def test_markdown_beside_the_sources_is_not_source_drift(tree):
    """`ratify.csrc_digest` excludes `.md` on purpose; the gate inherits that."""
    (tree / "csrc" / "rola" / "src" / "NOTES.md").write_text("editorial\n")
    assert manifest_fresh.stale_arches(tree) == []


def test_one_stale_arch_is_reported_alone(tree):
    """Per-arch manifests are separate records and fail separately."""
    arch = ratify.DEFAULT_ARCHES[0]
    path = _manifest(tree, arch)
    doc = json.loads(path.read_text())
    doc["source"]["csrc_sha256"] = "0" * 64
    path.write_text(json.dumps(doc))
    problems = manifest_fresh.stale_arches(tree)
    assert len(problems) == 1 and f"sm_{arch}" in problems[0]


def test_a_missing_manifest_is_a_complaint_not_a_pass(tree):
    """An unratified arch is not a fresh one."""
    arch = ratify.DEFAULT_ARCHES[0]
    _manifest(tree, arch).unlink()
    problems = manifest_fresh.stale_arches(tree)
    assert len(problems) == 1 and "no manifest" in problems[0]


def test_an_unreadable_source_record_is_a_complaint(tree):
    """A manifest that cannot answer the question has not answered it."""
    path = _manifest(tree, ratify.DEFAULT_ARCHES[0])
    doc = json.loads(path.read_text())
    del doc["source"]["csrc_sha256"]
    path.write_text(json.dumps(doc))
    assert "unreadable source record" in manifest_fresh.stale_arches(tree)[0]


def test_a_tree_without_sources_is_a_complaint(tmp_path):
    """Pointed somewhere wrong, the gate says so rather than passing."""
    assert manifest_fresh.stale_arches(tmp_path) == [
        f"no kernel sources at {tmp_path / 'csrc' / 'rola' / 'src'}"
    ]
