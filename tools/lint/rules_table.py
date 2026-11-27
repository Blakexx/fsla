#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""EVERY RULE HAS A ROW (KERNEL_STANDARDS, the preamble; Blake, 2026-11-17; gating): `docs/rules.json` names, for each
rule of `docs/CONSTITUTION.md` and `docs/KERNEL_STANDARDS.md`, the mechanism that enforces it -- `gate` (a check of
`bazel test //:gate`), `tool` (a tool refuses when run), `hook` (a git hook tracked under `tools/git-hooks/`, or an action hook listed in
`.claude/action-hooks.json`), `review` (the LLM
review, whose block refuses the next commit), `owed` (a named forcing point and how), or `none` (a definition or a
section that states no rule). This lint fails on

1. a row that breaks the schema, a duplicate `id`, a mechanism's required field missing;
2. a site of either document with no row, and a row whose site is in neither document;
3. an enforcer that does not resolve: a path that does not exist, a `file::symbol` the file does not contain, a Bazel
   label its package does not define, a `gate` row's label that is not a member of `//:gate` (read from the build files
   as text; the lint runs inside the gate and cannot run Bazel).

The table is `{"rules": [the rows], "checker": {the guidance}}`. Check 5, THE CHECKER'S GUIDANCE (Blake, 2026-11-20), reads the `checker`
section, which is not a standard and carries no mechanism: it runs both ways against the rules of the pinned checker
(`warpline.checker.declare.rule_names` over the checker's base ledger with the project's `tools/notice_ledger.json`): every rule has an entry
keyed by its name, no entry names a rule the pin lacks, and every id an entry lists under `rules` (the standards rows it serves, none where
no row states the rule) is a row. An entry carries `guidance` (fsla's one line: the atom or practice, the standards anchor, the lab page with
the evidence; where fsla has nothing to add it says "nothing beyond the generic fix"), `triggers` (the regexes over new kernel source at which
the edit hook injects it, `.claude/hooks/guidance.py`) and `rules`. The refusals of `tools/lab.py` print `guidance` after the checker's fix;
a checker that is not on the path fails the lint, never reads as zero.

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
#: the action hooks a checkout tracks, fired by the user-level dispatcher (`.claude/hooks/repo_hooks.py`)
MANIFEST = ".claude/action-hooks.json"
GIT_HOOKS = "tools/git-hooks/"
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
#: the top-level sections of the table, and the keys of one entry of `checker`: fsla's one line on the checker's rule, the regexes that fire
#: the edit hook, the ids of the standards rows the entry serves
SECTIONS = ("rules", "checker")
GUIDANCE = ("guidance", "triggers", "rules")
#: the rules of the pinned checker a fixture tree stands in for (the lint's own tests)
FIXTURE_RULES = ("alpha_rule", "beta_rule")
LEDGER = "tools/notice_ledger.json"
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


def schema(table: object) -> list[str]:
    """Check 1: the table's two sections, then every row's keys, its mechanism's required field, the statement's length, the ids' uniqueness."""
    if not isinstance(table, dict) or set(table) != set(SECTIONS):
        return [f"{TABLE}: the table is not a JSON object of the sections {', '.join(SECTIONS)}"]
    rows = table["rules"]
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        return [f"{TABLE}: `rules` is not a JSON list of rows"]
    if not isinstance(table["checker"], dict) or not all(isinstance(e, dict) for e in table["checker"].values()):
        return [f"{TABLE}: `checker` is not a JSON object of entries keyed by the checker's rule names"]
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
    for rule, entry in table["checker"].items():
        problems += guidance_problems(rule, entry)
    return problems


def guidance_problems(rule: str, entry: dict) -> list[str]:
    """An entry of the `checker` section carries its one-line guidance, its triggers (each a regex) and the ids it serves."""
    problems = []
    for key in sorted(set(entry) - set(GUIDANCE)):
        problems.append(f"checker '{rule}': has key '{key}', which an entry does not carry")
    line = entry.get("guidance")
    if not isinstance(line, str) or not line.strip() or "\n" in line:
        problems.append(f"checker '{rule}': an entry carries `guidance`, one line")
    served = entry.get("rules")
    if not isinstance(served, list) or not all(isinstance(r, str) and r for r in served):
        problems.append(f"checker '{rule}': an entry carries `rules`, a list of row ids (empty where no row states the rule)")
    triggers = entry.get("triggers")
    if not isinstance(triggers, list) or not all(isinstance(t, str) and t for t in triggers):
        problems.append(f"checker '{rule}': an entry carries `triggers`, a list of regexes (empty where no source text shows the rule)")
        return problems
    for trigger in triggers:
        try:
            re.compile(trigger)
        except re.error as error:
            problems.append(f"checker '{rule}': the trigger {trigger!r} is not a regex ({error})")
    return problems


def guidance_of(table: dict, rules: list[str]) -> list[str]:
    """Check 5: each rule of the pinned checker has an entry in `checker`, each entry is a rule of the pin, each id an entry lists is a row."""
    entries, ids = table["checker"], {row["id"] for row in table["rules"]}
    problems = [f"checker rule {rule} has no entry in the `checker` section of {TABLE}" for rule in rules if rule not in entries]
    for rule, entry in sorted(entries.items()):
        if rule not in rules:
            problems.append(f"checker '{rule}': the entry names a rule the pinned checker does not have")
        for served in entry["rules"]:
            if served not in ids:
                problems.append(f"checker '{rule}': lists {served}, which is no row of {TABLE}")
    return problems


def checker_rules(root: Path) -> list[str] | None:
    """The rule names of the pinned checker over its base ledger with the project's rows (`rule_names`), where its package
    is on the path (`dev_config.warpline_python()`: the graph's hand-in, or the runfiles' `warpline+` of a test); None where it is not."""
    sys.path.insert(0, str(ROOT / "tools"))
    import dev_config
    found = dev_config.warpline_python()
    if not found or not (Path(found) / "warpline" / "checker").is_dir():
        return None
    sys.path.insert(0, found)
    from warpline.checker import manifest, notices
    ledger = notices.merge(notices.base_ledger(), notices.load_ledger((root / LEDGER).read_text()))
    return manifest.rule_names(ledger)


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


def action_hooks(root: Path) -> set[str]:
    """The action hooks a tree lists: each `command` of its `.claude/action-hooks.json`, whatever the event."""
    try:
        manifest = json.loads((root / MANIFEST).read_text())
        return {str(e["command"]) for entries in manifest.values() for e in entries}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return set()


def resolved(root: Path, graph: Graph, gate: set[str], row: dict) -> list[str]:
    """Check 3: every enforcer a row names resolves, a `gate` row holds a label of the gate and a `hook` row a git hook
    of `tools/git-hooks/` or an action hook that is a file AND is listed in `.claude/action-hooks.json`."""
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
    if row.get("mechanism") == "hook":
        listed = action_hooks(root)
        paths = [r.partition("::")[0] for r in row.get("enforcer", [])]
        for path in paths:
            if path.startswith(".claude/hooks/") and (root / path).is_file() and path not in listed:
                problems.append(f"row '{row['id']}': the action hook {path} is not listed in {MANIFEST}")
        if not any(p.startswith(GIT_HOOKS) or p in listed for p in paths):
            problems.append(f"row '{row['id']}': a hook row names a git hook under {GIT_HOOKS} or an action hook listed in {MANIFEST}")
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


def judge(root: Path, build: str, table: object, leaves: set[str], every: set[str], rules: list[str]) -> list[str]:
    """Checks 1 to 3 and 5 over a table: the schema first, then the sites, the enforcers and the `checker` section against the pin's rules."""
    problems = schema(table)
    if problems:
        return problems
    rows = table["rules"]
    graph = Graph(root, build)
    gate = graph.members(GATE, set())
    problems = coverage_of(rows, leaves, every) + guidance_of(table, rules)
    for row in rows:
        problems += resolved(root, graph, gate, row)
    return problems


def check(root: Path, build: str, rules: list[str] | None) -> list[str]:
    """Checks 1 to 3 and 5 over one tree: its two documents, its `docs/rules.json`, its build files and the pinned checker's rules."""
    if rules is None:
        return ["the pinned checker is not on the path (FSLA_WARPLINE_PYTHON, or a test's runfiles): check 5 cannot run, and a lint that cannot run fails"]
    leaves, every, problems = derive_both(root)
    try:
        table = json.loads((root / TABLE).read_text())
    except (OSError, ValueError) as error:
        return problems + [f"{TABLE}: cannot be read: {error}"]
    return problems + judge(root, build, table, leaves, every, rules)


def owed(root: Path) -> list[str]:
    """Check 4's findings: each `owed` row, as a line the ratchet keys by its id alone (line 0 names no line of the file, so no
    edit of the table moves a finding)."""
    rows = json.loads((root / TABLE).read_text())["rules"]
    return [f"{TABLE}:0: owed rule {row['id']}" for row in rows if row.get("mechanism") == "owed"]


def mutations() -> dict[str, tuple[object, str]]:
    """The fixture table's ways to fail, each as `(edit of the ok table, the message it must draw)`."""
    def edit(first: str, **changes: object) -> object:
        def apply(table: dict) -> dict:
            row = next(r for r in table["rules"] if r["id"] == first)
            row.update(changes)
            return table
        return apply

    def entry(rule: str, **changes: object) -> object:
        def apply(table: dict) -> dict:
            table["checker"][rule].update(changes)
            return table
        return apply

    def drop(first: str) -> object:
        def apply(table: dict) -> dict:
            table["rules"] = [r for r in table["rules"] if r["id"] != first]
            return table
        return apply

    def without(rule: str) -> object:
        def apply(table: dict) -> dict:
            del table["checker"][rule]
            return table
        return apply

    def rename(rule: str, name: str) -> object:
        def apply(table: dict) -> dict:
            table["checker"][name] = table["checker"].pop(rule)
            return table
        return apply

    def duplicate(table: dict) -> dict:
        table["rules"].append(dict(table["rules"][0]))
        return table

    return {
        "a mechanism that is not one": (edit("KERNEL_STANDARDS#s2", mechanism="hope"), "is not one of"),
        "a field the mechanism requires": (edit("KERNEL_STANDARDS#s1.2", would_be=""), "missing field 'would_be'"),
        "a statement over the cap": (edit("KERNEL_STANDARDS#s2", statement="word " * 26), "over 25 words"),
        "a duplicate id": (duplicate, "duplicate id"),
        "a guidance key on a row": (edit("KERNEL_STANDARDS#s1.1", guidance="g"), "has key 'guidance'"),
        "a site with no row": (drop("KERNEL_STANDARDS#s1.1"), "site KERNEL_STANDARDS#s1.1 has no row"),
        "a row whose site is in neither document": (edit("CONSTITUTION#r8", site="CONSTITUTION#r99", id="CONSTITUTION#r99"),
                                                   "is in neither document"),
        "an enforcer path that is missing": (edit("KERNEL_STANDARDS#s1.1", enforcer=["tools/git-hooks/gone"]), "does not exist"),
        "an enforcer symbol that is missing": (edit("CONSTITUTION#principles.second-principle",
                                                    enforcer=["tools/check.py::no_such_check"]), "does not occur"),
        "an action hook the manifest does not list": (edit("KERNEL_STANDARDS#s2-addendum", enforcer=[".claude/hooks/unlisted.py"]),
                                                      "is not listed in"),
        "a hook row that names no hook": (edit("KERNEL_STANDARDS#s2-addendum", enforcer=["tools/check.py"]),
                                          "names a git hook under"),
        "a label no package defines": (edit("KERNEL_STANDARDS#s2", enforcer=["//:ghost"]), "does not define"),
        "a gate row's label outside the gate": (edit("KERNEL_STANDARDS#s2", enforcer=["//:outside"]), "not a member of //:gate"),
        "a checker rule with no entry": (without("alpha_rule"), "checker rule alpha_rule has no entry"),
        "an entry naming a rule the pinned checker lacks": (rename("alpha_rule", "ghost_rule"), "names a rule the pinned checker does not have"),
        "an entry listing an id that is no row": (entry("alpha_rule", rules=["KERNEL_STANDARDS#s99"]), "is no row of"),
        "an entry with no guidance": (entry("alpha_rule", guidance=""), "carries `guidance`"),
        "an entry with a key it does not carry": (entry("alpha_rule", mechanism="tool"), "has key 'mechanism'"),
        "a trigger that is no regex": (entry("alpha_rule", triggers=["wg_("]), "is not a regex"),
        "a table with no `checker` section": (lambda table: {"rules": table["rules"]}, "the table is not a JSON object of the sections"),
    }


def self_test() -> int:
    """Each fixture failure draws its message; the ok table passes and its owed rows are listed."""
    root = FIXTURE / "tree"
    failures = [f"the ok fixture fails: {p}" for p in check(root, FIXTURE_BUILD, list(FIXTURE_RULES))]
    if not any("not on the path" in p for p in check(root, FIXTURE_BUILD, None)):
        failures.append("a lint with no checker on the path did not fail")
    owed_rows = owed(root)
    if owed_rows != [f"{TABLE}:0: owed rule CONSTITUTION#r8", f"{TABLE}:0: owed rule KERNEL_STANDARDS#s1.2"]:
        failures.append(f"the ok fixture's owed rows are {owed_rows}")
    leaves, every, _ = derive_both(root)
    for case, (apply, message) in mutations().items():
        found = judge(root, FIXTURE_BUILD, apply(json.loads((root / TABLE).read_text())), leaves, every, list(FIXTURE_RULES))
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
    problems = check(ROOT, BUILD, checker_rules(ROOT))
    for problem in problems:
        print(f"  {problem}")
    print(f"rules_table: {len(problems)} problem(s) in {TABLE}" + (" (KERNEL_STANDARDS, the preamble: every rule has a row)" if problems else ""))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
