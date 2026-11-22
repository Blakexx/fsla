"""`tools/docrefs.py`: what a doc reference is, what its target's digest moves on, and the gate's verdict.

No GPU and no repository: each test plants a small tree under `tmp_path` and points the tool's root at it.
"""
from __future__ import annotations

import sys

import docrefs
import pytest


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(docrefs, "ROOT", tmp_path)

    def plant(files: dict[str, str]) -> set[str]:
        for name, text in files.items():
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / name).write_text(text)
        docrefs._text.cache_clear()  #: the tool reads each file once per run; a test rewrites them within one
        return set(files)
    return plant


def test_a_docstring_a_comment_or_a_reformat_moves_no_pin_and_a_code_change_does(tree):
    files = tree({"fsla/m.py": 'def f(x):\n    """Old words."""\n    return x + 1\n'})
    before = docrefs.read(docrefs.Target("fsla/m.py", "f"), files)
    tree({"fsla/m.py": 'def f(x):  # a comment\n    """New words."""\n\n    return (x + 1)\n'})
    assert docrefs.read(docrefs.Target("fsla/m.py", "f"), files) == before
    tree({"fsla/m.py": "def f(x):\n    return x + 2\n"})
    assert docrefs.read(docrefs.Target("fsla/m.py", "f"), files) not in (before, None)


def test_a_method_a_constant_and_a_re_export_resolve_and_a_missing_name_does_not(tree):
    files = tree({"fsla/__init__.py": "", "fsla/a.py": "LIMIT = 3\n\nclass C:\n    def go(self):\n        return 1\n",
                  "fsla/b.py": "from fsla.a import C\n"})
    assert docrefs.read(docrefs.Target("fsla/a.py", "LIMIT"), files)
    assert docrefs.read(docrefs.Target("fsla/b.py", "C.go"), files) == docrefs.read(
        docrefs.Target("fsla/a.py", "C.go"), files)
    assert docrefs.read(docrefs.Target("fsla/a.py", "C.stop"), files) is None


def test_the_four_citation_forms_are_read_and_a_fenced_block_is_not(tree):
    files = tree({
        "fsla/__init__.py": "", "fsla/ops.py": "def carry():\n    pass\n", "tools/t.py": "def main():\n    pass\n",
        "docs/other.md": "# Title\n\n## Wheels\n\ntext\n",
        "docs/page.md": ("`tools/t.py`'s `main` and `fsla.ops.carry` beside `docs/other.md` and [w](other.md#wheels)\n"
                         "```\n`fsla.ops.fenced`\n```\n`torch.ops.x` and `a phrase`\n"),
    })
    got = {str(t) for t in docrefs.references("docs/page.md", files)}
    assert got == {"tools/t.py::main", "fsla/ops.py::carry", "docs/other.md", "docs/other.md#wheels"}


def test_a_mirror_page_cites_its_sources_names_bare_and_keeps_a_pinned_one_that_left(tree):
    files = tree({"tools/t.py": "def main():\n    pass\n",
                  "docs/internals/tools/t.md": "# `tools/t.py`\n\n`main` calls `helper`, and `gone` went.\n"})
    got = {str(t) for t in docrefs.references("docs/internals/tools/t.md", files)}
    assert got == {"tools/t.py", "tools/t.py::main"}
    kept = docrefs.references("docs/internals/tools/t.md", files, pinned=frozenset({"tools/t.py::gone"}))
    assert docrefs.Target("tools/t.py", "gone") in kept
    assert docrefs.read(docrefs.Target("tools/t.py", "gone"), files) is None


def test_a_section_runs_to_the_next_heading_at_its_level(tree):
    files = tree({"docs/p.md": "# Top\n\n## Wheels\n\none\n\n### Detail\n\ntwo\n\n## Next\n\nthree\n"})
    wheels = docrefs.read(docrefs.Target("docs/p.md", "wheels"), files)
    tree({"docs/p.md": "# Top\n\n## Wheels\n\none\n\n### Detail\n\ntwo\n\n## Next\n\nchanged\n"})
    assert docrefs.read(docrefs.Target("docs/p.md", "wheels"), files) == wheels
    tree({"docs/p.md": "# Top\n\n## Wheels\n\none\n\n### Detail\n\nchanged\n\n## Next\n\nchanged\n"})
    assert docrefs.read(docrefs.Target("docs/p.md", "wheels"), files) != wheels


def test_device_code_pins_a_function_body_or_a_named_region(tree):
    files = tree({"csrc/k.cu": ("static int step(int x) {\n  // why\n  return x + 1;\n}\n"
                                "//: @ref fold\nint y = 2;\n//: @end\n")})
    body, region = (docrefs.read(docrefs.Target("csrc/k.cu", n), files) for n in ("step", "fold"))
    tree({"csrc/k.cu": "static int step(int x) {\n  return x + 1;  /* reworded */\n}\n//: @ref fold\nint y = 3;\n//: @end\n"})
    assert docrefs.read(docrefs.Target("csrc/k.cu", "step"), files) == body
    assert docrefs.read(docrefs.Target("csrc/k.cu", "fold"), files) != region


def test_the_verdict_names_what_is_new_stale_broken_unread_and_pending():
    target = docrefs.Target("fsla/a.py", "f")
    pin = {"doc": "docs/p.md", "target": str(target), "sha256": "aaaa", "pinned_at": "0", "reviewed": True}
    lock = {("docs/p.md", str(target)): pin,
            ("docs/p.md", "fsla/gone.py"): {**pin, "target": "fsla/gone.py"},
            ("docs/p.md", "fsla/broken.py"): {**pin, "target": "fsla/broken.py", "sha256": None},
            ("docs/p.md", "fsla/unread.py"): {**pin, "target": "fsla/unread.py", "reviewed": False}}
    found = {("docs/p.md", str(target)): (target, 3, "bbbb"),
             ("docs/p.md", "fsla/broken.py"): (docrefs.Target("fsla/broken.py"), 4, None),
             ("docs/p.md", "fsla/unread.py"): (docrefs.Target("fsla/unread.py"), 5, "aaaa"),
             ("docs/q.md", str(target)): (target, 1, "bbbb")}
    new, stale, broken, unread, pending = docrefs.verdict(found, lock)
    assert set(new) == {("docs/q.md", str(target))}
    assert stale == [("docs/p.md", "fsla/gone.py")]
    assert broken == [("docs/p.md", "fsla/broken.py")]
    assert unread == [("docs/p.md", "fsla/unread.py")]
    assert set(pending) == {("docs/p.md", str(target))}


#: THE GATE: a doc is clean only when every reference in it is pinned, read and current

def _run(*args: str) -> int:
    """The tool's command line, as `bazel run //:docrefs -- ARGS` gives it."""
    saved, sys.argv = sys.argv, ["docrefs", *args]
    try:
        return docrefs.main()
    finally:
        sys.argv = saved


CODE_V1 = "def f(x):\n    return x + 1\n"
DOC = "# Page\n\nThe function adds one: `fsla/a.py`'s `f`.\n"


@pytest.fixture
def repo(tree, monkeypatch):
    """``plant(files)`` adds tracked files; the tool's git reads are held to what was planted."""
    held: set[str] = set()
    monkeypatch.setattr(docrefs, "tracked", lambda: set(held))
    monkeypatch.setattr(docrefs, "_head_sha", lambda: "0" * 12)

    def plant(files: dict[str, str]) -> None:
        held.update(tree(files))
    plant({"fsla/a.py": CODE_V1, "docs/p.md": DOC, ".github/mirror/declarations.json": '{"private": []}\n'})
    return plant


def test_an_unread_reference_makes_the_gate_red_and_accepting_the_whole_doc_makes_it_green(repo, capsys):
    assert _run("init") == 0
    assert _run() == 1
    assert "docs/p.md:3: `fsla/a.py::f` was never read" in capsys.readouterr().out
    assert _run("accept", "--whole", "docs/p.md") == 0
    assert _run() == 0


def test_a_changed_target_makes_the_gate_red_naming_the_doc_and_the_command_and_an_accept_makes_it_green(repo, capsys):
    _run("init")
    _run("accept", "--whole", "docs/p.md")
    assert _run() == 0
    repo({"fsla/a.py": "def f(x):\n    return x + 2\n"})
    capsys.readouterr()
    assert _run() == 1
    out = capsys.readouterr().out
    assert "docs/p.md:3: `fsla/a.py::f` changed since its pin" in out
    assert "bazel run //:docrefs -- accept docs/p.md" in out
    assert _run("accept", "docs/p.md") == 0
    assert _run() == 0


def test_a_comment_in_the_target_leaves_the_gate_green(repo):
    _run("init")
    _run("accept", "--whole", "docs/p.md")
    repo({"fsla/a.py": "def f(x):  # reworded\n    return x + 1\n"})
    assert _run() == 0


def test_a_vanished_target_is_red_and_a_read_accept_drops_a_bare_name_but_a_path_form_stays_red(repo, capsys):
    repo({"tools/t.py": "def helper():\n    pass\n", "docs/internals/tools/t.md": "# `tools/t.py`\n\n`helper` is called.\n"})
    _run("init")
    _run("accept", "--whole", "docs/p.md", "docs/internals/tools/t.md")
    assert _run() == 0
    repo({"tools/t.py": "def other():\n    pass\n", "fsla/a.py": "def g(x):\n    return x\n"})
    capsys.readouterr()
    assert _run() == 1
    out = capsys.readouterr().out
    assert "docs/internals/tools/t.md:3: `tools/t.py::helper` vanished since its pin" in out
    assert "docs/p.md:3: `fsla/a.py::f` vanished since its pin" in out
    _run("accept", "docs/p.md", "docs/internals/tools/t.md")
    assert "`tools/t.py::helper` names nothing now" in capsys.readouterr().out
    assert _run() == 1  #: the possessive names a function that is gone: it shows as a reference that resolves to nothing
    assert "docs/p.md:3: the reference to `fsla/a.py::f` resolves to nothing" in capsys.readouterr().out
