"""The `docs/internals/` mirror is enforced, so "mirrored" is a fact and not a label.

The comments-to-docs mandate moved the engine's design prose out of `csrc/` and into
a markdown tree that mirrors the sources, leaving one-line hazard stubs behind. Both
halves of that arrangement rot silently on their own: a new source file arrives with
no mirror doc, or a stub cites an anchor that a doc edit renamed. Nothing in a
compiler notices either.

This is what notices. It is deliberately grep-level -- no markdown parser, no
cleverness -- because a check a contributor cannot read is a check they will route
around. It runs in the CPU job: it reads files, not a device.

The rules, and they are the whole contract:

1. Every source file under ``csrc/rola/`` has a mirror doc under
   ``docs/internals/`` at the same relative path, minus the ``src/`` component
   the mirror does not repeat, with its extension replaced by ``.md``. A ``.cu``
   and its ``.cuh`` share one doc, since they are one component.
2. Every ``docs/internals/....md#anchor`` reference in the sources resolves to an
   ``<a id="anchor">`` that the named doc actually defines.
3. A ``//: HAZARD <slug>`` stub cites an anchor whose id is that same slug, so the
   slug in the source and the section in the doc cannot drift apart.
4. No doc defines the same anchor twice, which would make a citation ambiguous.
5. A directory of GENERATED sources is exempt from rule 1 and documented by its own
   README instead -- and that exemption is itself checked: the README must exist and
   the directory must contain nothing the generator does not produce.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "csrc" / "rola"
MIRROR = ROOT / "docs" / "internals"

SOURCE_SUFFIXES = (".cu", ".cuh", ".cpp", ".h", ".hpp")

#: A doc reference as it appears in a source comment, e.g.
#: ``docs/internals/consumer_kernel.md#clock-prefix-fp32``. The anchor is optional: a
#: file-header block may point at a whole doc.
REF = re.compile(r"docs/internals/([A-Za-z0-9_/]+\.md)(?:#([A-Za-z0-9_-]+))?")

#: A hazard stub: ``//: HAZARD <slug> -- docs/internals/<doc>.md#<slug>``.
STUB = re.compile(r"//:\s*HAZARD\s+([A-Za-z0-9_-]+)\b")

#: An anchor definition in a mirror doc.
ANCHOR = re.compile(r'<a id="([A-Za-z0-9_-]+)"')


#: GENERATED SOURCES ARE DOCUMENTED BY THEIR GENERATOR, not one doc each. The eight
#: instantiation shards under ``csrc/rola/src/instantiations/`` are emitted by
#: ``tools/gen_shards.py`` from one matrix definition; eight mirror docs saying the
#: same sentence eight times would be the ceremony this mandate exists to remove, and
#: the next re-partition would silently orphan them. Their reasons live in ONE place
#: that is regenerated with them -- ``csrc/rola/src/instantiations/README.md`` -- and
#: rule 5 below is what keeps that exemption from being a hole: the directory must
#: carry that README, and every file in it must be one the generator produces.
GENERATED_DIRS = ("instantiations",)


def _sources() -> list[Path]:
    return sorted(
        p
        for p in SRC.rglob("*")
        if p.suffix in SOURCE_SUFFIXES
        and "third_party" not in p.parts
        and not set(p.parts) & set(GENERATED_DIRS)
    )


def _mirror_for(source: Path) -> Path:
    """The doc a source file must have.

    ``csrc/rola/src/consumer_kernel.cuh`` -> ``docs/internals/consumer_kernel.md``
    ``csrc/rola/src/paging/vmm_owner.cu`` -> ``docs/internals/paging/vmm_owner.md``
    ``csrc/rola/rola_api.cpp``            -> ``docs/internals/rola_api.md``

    The mirror does not repeat the ``src/`` component: the sources it mirrors are
    all under it, so carrying it would be one directory of pure ceremony.
    """
    relative = source.relative_to(SRC)
    if relative.parts[0] == "src":
        relative = Path(*relative.parts[1:])
    return MIRROR / relative.with_suffix(".md")


def _anchors(doc: Path) -> list[str]:
    return ANCHOR.findall(doc.read_text())


def test_generated_directories_are_documented_by_their_readme():
    """Rule 5. The exemption from rule 1 costs a stricter check, not a weaker one."""
    import subprocess
    import sys

    for name in GENERATED_DIRS:
        directory = SRC / "src" / name
        assert (directory / "README.md").is_file(), (
            f"{directory.relative_to(ROOT)} is exempt from the per-file mirror rule "
            f"because it is generated; that exemption requires its README.md")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "gen_shards.py"), "--check"],
            cwd=str(ROOT), capture_output=True, text=True)
        assert proc.returncode == 0, (
            f"{directory.relative_to(ROOT)} contains something its generator does "
            f"not produce:\n{proc.stdout}{proc.stderr}")


def test_there_are_sources_to_check():
    """A guard on the guard: a broken glob must not make this file vacuously pass."""
    sources = _sources()
    assert len(sources) >= 15, f"only found {len(sources)} sources under {SRC}"


@pytest.mark.parametrize("source", _sources(), ids=lambda p: p.name)
def test_every_source_file_has_a_mirror_doc(source: Path):
    doc = _mirror_for(source)
    assert doc.is_file(), (
        f"{source.relative_to(ROOT)} has no mirror doc.\n"
        f"Expected {doc.relative_to(ROOT)}.\n"
        "Every source file carries its reasons in docs/internals/ at the same "
        "relative path; a .cu and its .cuh share one doc. See "
        "docs/internals/README.md."
    )


def test_every_doc_reference_in_the_sources_resolves():
    broken = []
    for source in _sources():
        text = source.read_text()
        for lineno, line in enumerate(text.splitlines(), 1):
            for doc_name, anchor in REF.findall(line):
                doc = MIRROR / doc_name
                where = f"{source.relative_to(ROOT)}:{lineno}"
                if not doc.is_file():
                    broken.append(f"{where} -> {doc_name} (no such doc)")
                elif anchor and anchor not in _anchors(doc):
                    broken.append(f"{where} -> {doc_name}#{anchor} (no such anchor)")
    assert not broken, "unresolved doc references:\n  " + "\n  ".join(broken)


def test_every_hazard_stub_cites_its_own_slug():
    """A stub's slug IS its anchor id. One name, so the two cannot drift."""
    bad = []
    for source in _sources():
        for lineno, line in enumerate(source.read_text().splitlines(), 1):
            stub = STUB.search(line)
            if stub is None:
                continue
            slug = stub.group(1)
            refs = REF.findall(line)
            where = f"{source.relative_to(ROOT)}:{lineno}"
            if not refs:
                bad.append(f"{where}: HAZARD {slug} cites no docs/internals anchor")
                continue
            anchors = [anchor for _, anchor in refs if anchor]
            if slug not in anchors:
                bad.append(f"{where}: HAZARD {slug} cites {anchors or refs} instead")
    assert not bad, "hazard stubs that do not cite their own slug:\n  " + "\n  ".join(bad)


@pytest.mark.parametrize(
    "doc", sorted(MIRROR.rglob("*.md")), ids=lambda p: p.name
)
def test_no_doc_defines_an_anchor_twice(doc: Path):
    anchors = _anchors(doc)
    duplicates = sorted({a for a in anchors if anchors.count(a) > 1})
    assert not duplicates, (
        f"{doc.relative_to(ROOT)} defines these anchors more than once: "
        f"{duplicates}. A citation to a duplicated anchor is ambiguous."
    )
