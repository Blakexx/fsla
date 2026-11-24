#!/usr/bin/env python3
"""THE FACTS-IN-THE-TEMPLATE RULES, gated by `tools/lint/ratchet.py facts_in_template`.

docs/CONSTITUTION.md#r8, "FACTS IN THE TEMPLATE, DECISIONS IN THE KERNEL": the card's facts reach a kernel as constants
of its build, every quantity the card decides is derived from them at compile time, and a kernel's shared memory is
declared only through `SmemLayout`, whose sizes are expressions and never numbers. Three rules, each a heuristic over
text that says so in its own docstring: a finding is a question for a reviewer, not a proof. This file prints an
INVENTORY and exits 0; the ratchet is the gate, and the findings the tree carries are its baseline.

Usage:

    bazel run //tools/lint:facts_in_template                     # the inventory
    bazel run //tools/lint:facts_in_template -- --rule <name>    # one rule
    bazel test //tools/lint:facts_in_template_fixtures           # each rule fires and stays quiet
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / "csrc" / "fsla"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _walk(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    """Every source under `root`, WITH THE FIXTURES EXCLUDED (a lint that walks its own deliberate violations reports
    them as real ones)."""
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix in suffixes
                  and FIXTURES not in p.parents
                  and "third_party" not in p.parts
                  and "__pycache__" not in p.parts)


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


#: THE FACTS-IN-THE-TEMPLATE RULES' SCOPE (docs/CONSTITUTION.md#r8, "FACTS IN THE TEMPLATE, DECISIONS IN THE KERNEL"): the
#: library's kernel sources and the planner's headers, where the card's rules are written. Not `measure/lab/` (an
#: experiment is where a form is tried before it is a kernel), not generated files, not the host tools beside the
#: planner (`plan_tool.cpp`, `arms_gen.cpp`: no `.cu`/`.cuh`/`.h` suffix).
SMEM_LAYOUT = "common/smem_layout.cuh"


def facts_files() -> list[Path]:
    src = CSRC / "src"
    return sorted({p for p in _walk(src, (".cu", ".cuh"))} | {p for p in _walk(src / "common" / "plan", (".h",))
                                                               if p.parent == src / "common" / "plan"})


def _code_lines(text: str) -> list[str]:
    """The file's lines with `/* */` and `//` comments blanked (line count and columns kept). A `//` inside a string
    literal is read as a comment: a heuristic, and the kernels carry no such string."""
    out = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.DOTALL)
    return [line.split("//", 1)[0] for line in out.split("\n")]


def rule_shared_memory_through_the_layout(files=None) -> list[str]:  #: reserved: the fixtures' self-test and tests/unit/test_drift_facts_rules.py pass their own files
    """A kernel's shared memory is declared through `SmemLayout` (docs/CONSTITUTION.md#r8), nowhere else.

    Heuristic: the token `__shared__` on a line, once `//` and `/* */` comments are blanked, in any file of the scope
    (`facts_files`) but `common/smem_layout.cuh`. One finding a line. It sees the token only: a shared array declared
    by a macro that spells the keyword elsewhere, or by `extern __shared__` through an alias, is not found.
    """
    findings = []
    for path in (facts_files() if files is None else files):
        rel = _rel(path)
        if rel.endswith(SMEM_LAYOUT):
            continue
        for i, line in enumerate(_code_lines(path.read_text())):
            if "__shared__" in line:
                findings.append(f"{rel}:{i + 1}: shared memory declared outside SmemLayout ({SMEM_LAYOUT}): "
                                f"declare a tile")
    return findings


_TILE = re.compile(r"\b(Fixed|Share)\s*<")
_BARE_INTEGER = re.compile(r"(?:0[xX][0-9a-fA-F']+|[0-9][0-9']*)[uUlL]*")


def _tile_arguments(code: str, start: int) -> list[tuple[int, str]]:
    """The arguments of the `<...>` opening just before `start`, as `(offset of its first character, text)`: split at
    the commas of the top nesting level of `<>`, `()`, `[]`, `{}` (angle brackets count only outside the other
    three, so a comparison inside parentheses is not a bracket). `[]` when the list never closes."""
    out, depth, stack, begin = [], 0, [], start
    pairs = {"(": ")", "[": "]", "{": "}"}
    for i in range(start, len(code)):
        c = code[i]
        if stack:
            if c in pairs:
                stack.append(pairs[c])
            elif c == stack[-1]:
                stack.pop()
        elif c in pairs:
            stack.append(pairs[c])
        elif c == "<":
            depth += 1
        elif c == ">" and code[i - 1] != "-":
            if depth == 0:
                out.append((begin, code[begin:i]))
                return out
            depth -= 1
        elif c == "," and depth == 0:
            out.append((begin, code[begin:i]))
            begin = i + 1
    return []


def rule_layout_sizes_are_expressions(files=None) -> list[str]:  #: reserved: the fixtures' self-test and tests/unit/test_drift_facts_rules.py pass their own files
    """A tile's size is an expression of the build's facts, never a number (docs/CONSTITUTION.md#r8).

    Heuristic: the template name `Fixed` or `Share` followed by `<`, in comment-blanked text across line breaks; its
    arguments are split at the top nesting level of `<>`, `()`, `[]`, `{}`, and the third (`COUNT`, or `LEAST`) is a
    finding when it is a bare integer literal: digits or hex, an optional `'` separator, an optional `u`/`l` suffix.
    A `WEIGHT` may be a literal (a ratio). `common/smem_layout.cuh` is exempt (its `smem_proof` namespace states
    literals as the proof's fixture). It misses a number behind a name (`constexpr int kRing = 64;` passed as `kRing`:
    rule `card_quantities_are_derived` reads that declaration), a literal wrapped in a cast or parentheses, and a
    tile built through an alias of `Fixed`/`Share`.
    """
    findings = []
    for path in (facts_files() if files is None else files):
        rel = _rel(path)
        if rel.endswith(SMEM_LAYOUT):
            continue
        code = "\n".join(_code_lines(path.read_text()))
        for m in _TILE.finditer(code):
            args = _tile_arguments(code, m.end())
            if len(args) < 3:
                continue
            offset, text = args[2]
            if _BARE_INTEGER.fullmatch(text.strip()):
                at = offset + len(text) - len(text.lstrip())
                findings.append(f"{rel}:{code.count(chr(10), 0, at) + 1}: a tile's size is the number "
                                f"{text.strip()}: derive it from the build's facts or the arm's parameters")
    return findings


_INTEGER_TYPE = (r"(?:std::int32_t|std::int64_t|std::uint32_t|std::uint64_t|std::size_t|int32_t|int64_t|uint32_t|"
                 r"uint64_t|size_t|long\s+long|unsigned|long|int)")
_NUMBER_DECLARATION = re.compile(rf"^\s*(?:(?:inline|static)\s+)*constexpr\s+{_INTEGER_TYPE}\s+(\w+)\s*=\s*"
                                 rf"((?:0[xX][0-9a-fA-F']+|[0-9][0-9']*)[uUlL]*)\s*;")
_FIXED_BECAUSE = re.compile(r"\bFIXED\s+\S")
_OPENS_A_SCOPE = re.compile(r"\b(?:namespace|struct|class|union)\b|\bextern\s+\"C(?:\+\+)?\"")


def _declaration_scopes(lines: list[str]) -> list[bool]:
    """For each line, whether it begins at namespace or class scope: every brace open at its start was opened by a head
    naming `namespace`/`struct`/`class`/`union` (or `extern "C"`) and containing no parenthesis; any other brace
    (a function body, a lambda, an initialiser, an `enum`) puts the line out of scope. The head is the text since the
    last `;`, `{` or `}`, and preprocessor lines are not part of it."""
    stack: list[bool] = []
    head = ""
    scopes = []
    for line in lines:
        scopes.append(all(stack))
        if line.lstrip().startswith("#"):
            continue
        for c in line:
            if c == "{":
                stack.append(bool(_OPENS_A_SCOPE.search(head)) and "(" not in head and ")" not in head)
                head = ""
            elif c == "}":
                if stack:
                    stack.pop()
                head = ""
            elif c == ";":
                head = ""
            else:
                head += c
        head += " "
    return scopes


def rule_card_quantities_are_derived(files=None) -> list[str]:  #: reserved: the fixtures' self-test and tests/unit/test_drift_facts_rules.py pass their own files
    """A constant at namespace or class scope is derived from the build's facts, or says why the card does not decide it.

    Heuristic: a line (comments blanked) that is `[inline] [static] constexpr <integer type> <name> = <bare integer
    literal>;`, the type one of `int`, `unsigned`, `long`, `long long` and the `std::`-qualified or bare
    `int32_t`/`int64_t`/`uint32_t`/`uint64_t`/`size_t`, and that begins at namespace or class scope
    (`_declaration_scopes`: brace depth, each brace classified by the head that opened it), is a finding UNLESS the
    comment lines directly above it (a contiguous run of lines that are only a `//` comment) contain the word `FIXED`
    in capitals followed by a reason (`//: FIXED by the format: a mask word is 32 tokens`). A comment trailing the
    declaration does not count: the reason is stated in the block above.

    What it misses: a constant inside a function body (scope is brace depth); an initialiser that is not a bare
    literal (`1 << 5`, `-1`, a cast, `true`); a type outside the list (`unsigned int`, `char`, `bool`, `float`,
    `std::ptrdiff_t`, a `using` alias); a declaration split over lines or several declarators on one line; a `const`
    without `constexpr`; an `enum` value; a number passed straight into a template argument (rule
    `layout_sizes_are_expressions` reads tile sizes).
    """
    findings = []
    for path in (facts_files() if files is None else files):
        raw = path.read_text().split("\n")
        code = _code_lines(path.read_text())
        scopes = _declaration_scopes(code)
        for i, line in enumerate(code):
            m = _NUMBER_DECLARATION.match(line)
            if m is None or not scopes[i]:
                continue
            block, j = [], i - 1
            while j >= 0 and raw[j].lstrip().startswith("//"):
                block.append(raw[j].strip())
                j -= 1
            if _FIXED_BECAUSE.search(" ".join(reversed(block))):
                continue
            findings.append(f"{_rel(path)}:{i + 1}: {m.group(1)} is the number {m.group(2)}: derive it from the build's "
                            f"facts, or state in its block why the card does not decide it (FIXED ...)")
    return findings


RULES = {
    "shared_memory_through_the_layout": rule_shared_memory_through_the_layout,
    "layout_sizes_are_expressions": rule_layout_sizes_are_expressions,
    "card_quantities_are_derived": rule_card_quantities_are_derived,
}

#: `{rule -> (the fixture that MUST fire, the fixture that must NOT)}`: a rule never shown to fire is a rule nobody knows
#: is connected, and one never shown to stay quiet is a rule nobody can afford to make gating.
FIXTURE_PAIRS = dict.fromkeys(RULES, ("drift_facts_bad.cuh", "drift_facts_ok.cuh"))


def self_test() -> int:
    bad = []
    for name, (fires, quiet) in sorted(FIXTURE_PAIRS.items()):
        rule = RULES[name]
        if not rule([FIXTURES / fires]):
            bad.append(f"    {name}: did not fire on {fires}")
        cold = rule([FIXTURES / quiet])
        if cold:
            bad.append(f"    {name}: fired on {quiet}: {cold}")
    #: AND THE FIXTURES MUST NOT REACH THE REAL SCAN
    for name, rule in sorted(RULES.items()):
        leaked = [f for f in rule() if "lint/fixtures/" in f]
        if leaked:
            bad.append(f"    {name}: its own fixtures leaked into the real scan: {leaked[:2]}")
    if bad:
        print("SELF-TEST FAILED:")
        print("\n".join(bad))
        return 1
    print(f"facts_in_template.py --test-fixtures: PASS ({len(FIXTURE_PAIRS)} rules fire on their positive fixture and "
          f"stay quiet on their negative; no rule leaks a fixture into the real scan)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rule", action="append", choices=sorted(RULES), help="run only this rule, repeatable")
    ap.add_argument("--test-fixtures", action="store_true",
                    help="prove each rule fires on its positive fixture and stays quiet on its negative")
    a = ap.parse_args()
    if a.test_fixtures:
        return self_test()
    total = 0
    for name in (a.rule or sorted(RULES)):
        findings = RULES[name]()
        total += len(findings)
        print(f"facts_in_template[{name}]: {len(findings)} finding(s)")
        for f in findings:
            print(f"  {f}")
    print(f"facts_in_template: {total} finding(s) over {len(a.rule or RULES)} rule(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
