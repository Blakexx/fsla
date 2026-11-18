#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""EVERY RULE HAS A ROW (KERNEL_STANDARDS, the preamble; Blake, 2026-11-17; gating): `docs/rules.json` names, for each
rule of `docs/CONSTITUTION.md` and `docs/KERNEL_STANDARDS.md`, the mechanism that enforces it -- `gate` (a check of
`bazel test //:gate`), `tool` (a tool refuses when run), `hook` (a git hook tracked in the repo), `review` (the LLM
review, whose block refuses the next commit), `owed` (a named forcing point and how), or `none` (a definition or a
section that states no rule). This lint fails on

1. a row that breaks the schema, a duplicate `id`, a mechanism's required field missing;
2. a site of either document with no row, and a row whose site is in neither document;
3. an enforcer that does not resolve: a path that does not exist, a `file::symbol` the file does not contain, a Bazel
   label its package does not define, a `gate` row's label that is not a member of `//:gate` (read from the build files
   as text; the lint runs inside the gate and cannot run Bazel).

Check 4, OWED ONLY SHRINKS, is the ratchet's: `--owed` prints each `owed` row's id as a finding and
`tools/lint/ratchet.py rules_owed` holds the list against its baseline.

A SITE is derived from the documents here: `<DOC>#<anchor>` for every `<a id>` anchor (every `## ` heading, an
addendum paragraph and each paragraph of the standards' preamble carries one), `<DOC>#<anchor>.<n>` for a numbered
item under it and `.<n>.<letter>` for a lettered item under that, and `CONSTITUTION#principles.<slug>` for each named
principle. A site that has items is their container and needs no row of its own. Docs: docs/internals/tools/lint.md#rules-table.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tools" / "lint" / "fixtures" / "rules_table"
DOCS = {"CONSTITUTION": "docs/CONSTITUTION.md", "KERNEL_STANDARDS": "docs/KERNEL_STANDARDS.md"}
TABLE = "docs/rules.json"
BUILD = "BUILD.bazel"
#: a fixture tree's build files keep another name, so Bazel does not read the fixture's packages as the repository's own
FIXTURE_BUILD = "BUILD.fixture"
GATE = "//:gate"
BASE = ("site", "id", "statement", "mechanism")
#: each mechanism's own fields: the enforcer and its coverage, the reason a reader must judge it, the forcing point it
#: would take, or why no rule is stated
FIELDS = {
    "gate": ("enforcer", "coverage"),
    "tool": ("enforcer", "coverage"),
    "hook": ("enforcer", "coverage"),
    "review": ("why_not_mechanical",),
    "owed": ("would_be",),
    "none": ("why",),
}
STATEMENT_WORDS = 25
ANCHOR = re.compile(r'<a id="([^"]+)"></a>')
ANCHOR_LINE = re.compile(r'^\s*(?:<a id="[^"]+"></a>\s*)+$')
NUMBERED = re.compile(r"^(\d+)\. ")
LETTERED = re.compile(r"^\s+([a-z]{1,2})\. ")
PRINCIPLE = re.compile(r"^- ([A-Z][^:(]*?)(?: \(|:|\.$)")
ADDENDUM = re.compile(r"^\s*(?:§\d+ )?addendum\b", re.I)
LABEL = re.compile(r"^//([^:]*):([\w.+/-]+)$")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def derive(doc: str, text: str) -> tuple[set[str], set[str], list[str]]:
    """`(leaves, every site, structure problems)` of one document: what a row may name, and what needs a row."""
    every, containers, problems, defined = set(), set(), [], {}
    anchor, number, preamble, fenced = "", "", True, False
    lines = text.splitlines()
    for no, line in enumerate(lines, 1):
        found = ANCHOR.findall(line)
        for name in found:
            if name in defined:
                problems.append(f"{doc}: anchor '{name}' is defined at lines {defined[name]} and {no}")
            defined[name] = no
            every.add(f"{doc}#{name}")
        if found:
            anchor, number = found[-1], ""
        if line.startswith("```"):
            fenced = not fenced
        if fenced:
            continue
        bare = ANCHOR.sub("", line)
        anchored = bool(found) or (no > 1 and ANCHOR_LINE.match(lines[no - 2]) is not None)
        if line.startswith("## "):
            preamble = False
        if (re.match(r"^#{1,2} ", line) and no > 1 or ADDENDUM.match(bare)
                or preamble and bare.startswith("**") and not lines[no - 2].strip()) and not anchored:
            problems.append(f"{doc}.md:{no}: carries no `<a id>` anchor, so no row can name it: {bare.strip()[:60]}")
        if doc == "CONSTITUTION" and anchor == "principles" and (m := PRINCIPLE.match(line)):
            every.add(f"{doc}#principles.{slug(m.group(1))}")
            containers.add(f"{doc}#principles")
        if doc == "KERNEL_STANDARDS" and anchor and (m := NUMBERED.match(line)):
            number = m.group(1)
            every.add(f"{doc}#{anchor}.{number}")
            containers.add(f"{doc}#{anchor}")
        if doc == "KERNEL_STANDARDS" and anchor and number and (m := LETTERED.match(line)):
            every.add(f"{doc}#{anchor}.{number}.{m.group(1)}")
            containers.add(f"{doc}#{anchor}.{number}")
    return every - containers, every, problems


def schema(rows: object) -> list[str]:
    """Check 1: every row's keys, its mechanism's required field, the statement's length, the ids' uniqueness."""
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        return [f"{TABLE}: the table is not a JSON list of rows"]
    problems, ids = [], set()
    for position, row in enumerate(rows, 1):
        name = row.get("id", f"#{position}")
        mechanism = row.get("mechanism")
        if mechanism not in FIELDS:
            problems.append(f"row '{name}': mechanism {mechanism!r} is not one of {', '.join(FIELDS)}")
            continue
        for key in BASE + FIELDS[mechanism]:
            if not row.get(key):
                problems.append(f"row '{name}': missing field '{key}' (a {mechanism} row carries it)")
        for key in sorted(set(row) - set(BASE + FIELDS[mechanism])):
            problems.append(f"row '{name}': has key '{key}', which a {mechanism} row does not carry")
        if name in ids:
            problems.append(f"row '{name}': duplicate id")
        ids.add(name)
        site = str(row.get("site", ""))
        if not (name == site or str(name).startswith(site + ":")):
            problems.append(f"row '{name}': the id is its site '{site}' or the site, a colon and a slug")
        if len(str(row.get("statement", "")).split()) > STATEMENT_WORDS:
            problems.append(f"row '{name}': the statement is over {STATEMENT_WORDS} words")
        coverage = row.get("coverage", "full")
        if coverage != "full" and not (isinstance(coverage, str) and coverage.startswith("partial: ")):
            problems.append(f"row '{name}': coverage is `full` or `partial: <what is not checked>`")
        if "enforcer" in row and not (isinstance(row["enforcer"], list) and all(isinstance(e, str) for e in row["enforcer"])):
            problems.append(f"row '{name}': the enforcer is a list of paths, `path::symbol` and Bazel labels")
    return problems


def coverage_of(rows: list[dict], leaves: set[str], every: set[str]) -> list[str]:
    """Check 2: each leaf site has a row, and each row's site is a site of one of the documents."""
    named = {r.get("site") for r in rows}
    problems = [f"site {site} has no row in {TABLE}" for site in sorted(leaves - named)]
    for row in rows:
        if row.get("site") not in every:
            problems.append(f"row '{row.get('id')}': site '{row.get('site')}' is in neither document")
    return problems


class Graph:
    """The build files of one tree, read as text: which targets a package defines and what `//:gate` holds."""

    def __init__(self, root: Path, build: str) -> None:
        self.root = root
        self.build = build
        self.packages: dict[str, dict[str, dict[str, ast.expr]]] = {}

    def targets(self, package: str) -> dict[str, dict[str, ast.expr]]:
        """Each target a package's `BUILD.bazel` defines, by name, with its keyword arguments."""
        if package not in self.packages:
            found: dict[str, dict[str, ast.expr]] = {}
            build = self.root / package / self.build
            tree = ast.parse(build.read_text()) if build.is_file() else ast.parse("")
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    keywords = {k.arg: k.value for k in node.keywords if k.arg}
                    if isinstance(keywords.get("name"), ast.Constant):
                        found[keywords["name"].value] = keywords
                if isinstance(node, ast.ListComp) and isinstance(node.elt, ast.Call):
                    found.update(_expanded(node))
            self.packages[package] = found
        return self.packages[package]

    def defines(self, label: str) -> bool:
        match = LABEL.match(label)
        return bool(match) and match.group(2) in self.targets(match.group(1))

    def members(self, label: str, seen: set[str]) -> set[str]:
        """The label and, through every `test_suite` under it, each test it holds."""
        seen.add(label)
        match = LABEL.match(label)
        suite = self.targets(match.group(1)).get(match.group(2), {}) if match else {}
        for node in ast.walk(suite.get("tests", ast.Constant(value=None))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                held = f"//{match.group(1)}{node.value}" if node.value.startswith(":") else node.value
                if held not in seen:
                    self.members(held, seen)
        return seen


def _expanded(node: ast.ListComp) -> dict[str, dict[str, ast.expr]]:
    """The targets a `[rule(name = name, ...) for name in [...]]` comprehension defines, by the names it iterates."""
    out: dict[str, dict[str, ast.expr]] = {}
    generator, call = node.generators[0], node.elt
    keywords = {k.arg: k.value for k in call.keywords if k.arg}
    variable = keywords.get("name")
    if not (isinstance(variable, ast.Name) and isinstance(generator.iter, ast.List)):
        return out
    names = [generator.target] if isinstance(generator.target, ast.Name) else list(getattr(generator.target, "elts", []))
    for element in generator.iter.elts:
        values = [element] if isinstance(element, ast.Constant) else list(getattr(element, "elts", []))
        env = {n.id: v.value for n, v in zip(names, values) if isinstance(n, ast.Name) and isinstance(v, ast.Constant)}
        if variable.id in env:
            out[env[variable.id]] = keywords
    return out


def resolved(root: Path, graph: Graph, gate: set[str], row: dict) -> list[str]:
    """Check 3: every enforcer a row names resolves, and a `gate` row holds a label of the gate."""
    problems, labels = [], []
    for ref in row.get("enforcer", []):
        if ref.startswith("//"):
            labels.append(ref)
            if not graph.defines(ref):
                problems.append(f"row '{row['id']}': the label {ref} names a target its package's BUILD.bazel does not define")
            elif row["mechanism"] == "gate" and ref not in gate:
                problems.append(f"row '{row['id']}': the label {ref} is not a member of {GATE}")
            continue
        path, _, symbol = ref.partition("::")
        source = root / path
        if not source.is_file():
            problems.append(f"row '{row['id']}': the enforcer path {path} does not exist")
        elif symbol and not re.search(rf"(?<![\w]){re.escape(symbol)}(?![\w])", source.read_text(errors="replace")):
            problems.append(f"row '{row['id']}': the symbol {symbol} does not occur in {path}")
    if row.get("mechanism") == "gate" and not labels:
        problems.append(f"row '{row['id']}': a gate row names the label of a check of {GATE}")
    if row.get("mechanism") == "hook" and not any(r.startswith("tools/git-hooks/") for r in row.get("enforcer", [])):
        problems.append(f"row '{row['id']}': a hook row names a hook under tools/git-hooks/")
    return problems


def derive_both(root: Path) -> tuple[set[str], set[str], list[str]]:
    """The leaves, every site and the structure problems of both documents of a tree."""
    leaves, every, problems = set(), set(), []
    for doc, path in DOCS.items():
        found = derive(doc, (root / path).read_text())
        leaves |= found[0]
        every |= found[1]
        problems += found[2]
    return leaves, every, problems


def judge(root: Path, build: str, rows: object, leaves: set[str], every: set[str]) -> list[str]:
    """Checks 1 to 3 over a table: the schema first, then the sites and the enforcers of rows that hold the schema."""
    problems = schema(rows)
    if problems:
        return problems
    graph = Graph(root, build)
    gate = graph.members(GATE, set())
    problems = coverage_of(rows, leaves, every)
    for row in rows:
        problems += resolved(root, graph, gate, row)
    return problems


def check(root: Path, build: str) -> list[str]:
    """Checks 1 to 3 over one tree: its two documents, its `docs/rules.json` and its build files."""
    leaves, every, problems = derive_both(root)
    try:
        rows = json.loads((root / TABLE).read_text())
    except (OSError, ValueError) as error:
        return problems + [f"{TABLE}: cannot be read: {error}"]
    return problems + judge(root, build, rows, leaves, every)


def owed(root: Path) -> list[str]:
    """Check 4's findings: each `owed` row, as a line the ratchet keys by its id alone (line 0 names no line of the file, so no
    edit of the table moves a finding)."""
    rows = json.loads((root / TABLE).read_text())
    return [f"{TABLE}:0: owed rule {row['id']}" for row in rows if row.get("mechanism") == "owed"]


def mutations() -> dict[str, tuple[object, str]]:
    """The fixture table's ways to fail, each as `(edit of the ok table, the message it must draw)`."""
    def edit(first: str, **changes: object) -> object:
        def apply(rows: list[dict]) -> list[dict]:
            row = next(r for r in rows if r["id"] == first)
            row.update(changes)
            return rows
        return apply

    def drop(first: str) -> object:
        return lambda rows: [r for r in rows if r["id"] != first]

    return {
        "a mechanism that is not one": (edit("KERNEL_STANDARDS#s2", mechanism="hope"), "is not one of"),
        "a field the mechanism requires": (edit("KERNEL_STANDARDS#s1.2", would_be=""), "missing field 'would_be'"),
        "a statement over the cap": (edit("KERNEL_STANDARDS#s2", statement="word " * 26), "over 25 words"),
        "a duplicate id": (lambda rows: rows + [dict(rows[0])], "duplicate id"),
        "a site with no row": (drop("KERNEL_STANDARDS#s1.1"), "site KERNEL_STANDARDS#s1.1 has no row"),
        "a row whose site is in neither document": (edit("CONSTITUTION#r8", site="CONSTITUTION#r99", id="CONSTITUTION#r99"),
                                                   "is in neither document"),
        "an enforcer path that is missing": (edit("KERNEL_STANDARDS#s1.1", enforcer=["tools/git-hooks/gone"]), "does not exist"),
        "an enforcer symbol that is missing": (edit("CONSTITUTION#principles.second-principle",
                                                    enforcer=["tools/check.py::no_such_check"]), "does not occur"),
        "a label no package defines": (edit("KERNEL_STANDARDS#s2", enforcer=["//:ghost"]), "does not define"),
        "a gate row's label outside the gate": (edit("KERNEL_STANDARDS#s2", enforcer=["//:outside"]), "not a member of //:gate"),
    }


def self_test() -> int:
    """Each fixture failure draws its message; the ok table passes and its owed rows are listed."""
    root = FIXTURE / "tree"
    failures = [f"the ok fixture fails: {p}" for p in check(root, FIXTURE_BUILD)]
    owed_rows = owed(root)
    if owed_rows != [f"{TABLE}:0: owed rule CONSTITUTION#r8", f"{TABLE}:0: owed rule KERNEL_STANDARDS#s1.2"]:
        failures.append(f"the ok fixture's owed rows are {owed_rows}")
    leaves, every, _ = derive_both(root)
    for case, (apply, message) in mutations().items():
        found = judge(root, FIXTURE_BUILD, apply(json.loads((root / TABLE).read_text())), leaves, every)
        if not any(message in p for p in found):
            failures.append(f"{case}: no message holds {message!r}: {found}")
    for text, why in (("# T\n\n## Heading\n", "a heading with no anchor"),
                      ('# T\n\n<a id="a"></a>\n## One\n<a id="a"></a>\n', "a twice-defined anchor"),
                      ("# T\n\n**THE CARDS.** no anchor\n", "a preamble paragraph with no anchor")):
        if not derive("KERNEL_STANDARDS", text)[2]:
            failures.append(f"{why} drew no structure problem")
    for failure in failures:
        print(f"SELF-TEST FAILED: {failure}", file=sys.stderr)
    if not failures:
        print("rules_table.py --test-fixtures: PASS")
    return 1 if failures else 0


def main() -> int:
    if "--test-fixtures" in sys.argv:
        return self_test()
    if "--owed" in sys.argv:
        print("\n".join(owed(ROOT)))
        return 0
    problems = check(ROOT, BUILD)
    for problem in problems:
        print(f"  {problem}")
    print(f"rules_table: {len(problems)} problem(s) in {TABLE}" + (" (KERNEL_STANDARDS, the preamble: every rule has a row)" if problems else ""))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
