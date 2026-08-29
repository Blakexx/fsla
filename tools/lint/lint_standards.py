#!/usr/bin/env python3
"""Textual/preprocessor standards checks that ast-grep's AST rules cannot see.

ast-grep (tools/lint/rules/*.yml) covers shapes a parser can express: warp
collectives under a short-circuit, a `__global__` in an anonymous namespace,
a width-shaped `if (kDv == ...)`, a `#include <torch/...>` node. Three checks
in KERNEL_STANDARDS.md are either about the PREPROCESSOR (a `#pragma` has no
AST node next to the statement it modifies in every grammar ast-grep ships)
or about POSITION within one named file (a marker comment, not a syntax
shape) -- those live here as plain line-oriented scans.

Checks:
  1. UNROLL DISCIPLINE (KERNEL_STANDARDS.md §6). A `for`/`while` loop in
     device code (any file under csrc/rola/, excluding csrc/third_party/)
     whose bound is NOT compile-time carries `#pragma unroll 1` immediately
     before it. This is a HEURISTIC, not a compiler: it approximates
     "compile-time bound" by regexing the loop's stop expression, and it can
     only be conservative in one direction --  see HEURISTIC LIMITS below.
  2. NO TORCH INCLUDE IN ARM TUs (belt to the ast-grep rule
     torch_include_in_arm_tu.yml -- K36). A `#include` line naming `torch/`
     under csrc/rola/src/instantiations/, found by TEXT rather than AST so a
     macro-guarded or unusually-formatted include cannot hide from either
     check.
  3. NO __syncthreads AFTER THE DECODE PROLOGUE (P79's law, ledger
     2026-08-27: "11 barriers, all in the prologue ... 0 after it"). The
     marker is decode.cu's own section comment `// ---- THE WALK`; every
     `__syncthreads`/`__syncwarp` after that line is a finding for Blake to
     rule on (some may be the fold/combine barriers the walk's cross-warp
     reduction legitimately needs -- this script does not adjudicate that,
     it only locates them against the law's own stated marker).

HEURISTIC LIMITS (check 1):
  - "Device code" is found by brace-counting from each `__global__`/`__device__`
    qualifier to its function's matching close brace -- a `.cu` mixes host
    launcher code (std::vector loops, TORCH_CHECK, TensorOptions) with kernels,
    and only the latter is in scope. The brace count does not parse string/char
    literals, so a literal containing an unbalanced `{`/`}` (none observed in
    this codebase) would misplace a span's end.
  - No preprocessor expansion and no constant folding: `for (i = 0; i < N; )`
    is judged by whether `N` LOOKS compile-time (an integer literal, `sizeof`,
    or an ALL_CAPS / `k`-prefixed identifier -- the project's constexpr/
    template-param naming convention), not by resolving what `N` actually is.
    A `constexpr int n = ...;` bound one line above with a lowercase name
    reads as "dynamic" and asks for a pragma it does not need (a false
    positive the script cannot avoid without a real preprocessor+compiler
    front end); a `#define BOUND 64` used as `i < BOUND` reads as constexpr
    correctly only because BOUND matches the ALL_CAPS heuristic.
  - Only the closing `#pragma unroll[ 1]` is checked, and only on the lines
    immediately above the loop (comments in between are skipped, anything
    else breaks the search). A pragma placed elsewhere (e.g. hoisted above an
    unrelated statement) is invisible to this check.
  - Loop headers spanning multiple lines are matched as best-effort: the
    stop-expression regex runs on the concatenation of up to 3 lines from the
    `for`/`while` token, which can under- or over-capture on unusual
    formatting.
  - `while` loops with no explicit relational stop expression (e.g. an
    infinite `while (true)` with a `break`) are skipped: there is no "bound"
    to classify, so nothing is asked of them.
This is a lint, not a gate: every finding is reported for Blake to rule on,
never silently fixed or silenced by this script.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSRC_ROLA = ROOT / "csrc" / "rola"
INSTANTIATIONS = CSRC_ROLA / "src" / "instantiations"
DECODE_CU = CSRC_ROLA / "src" / "decode" / "decode.cu"
TESTS_ROOT = ROOT / "tests"

#: docs/testing.md: "there is no fourth tier". A test answers one of exactly three
#: questions (oracle/integration/unit), so it lives in exactly one of these three
#: directories -- never a fourth bucket, campaign-named or otherwise (K44: `tests/k31/`
#: was exactly that, folded into the tier each of its tests actually answered).
TESTS_ALLOWED_TIERS = {"unit", "integration", "oracle"}

DEVICE_SUFFIXES = (".cu", ".cuh")

#: Identifiers that read as compile-time under this project's naming
#: convention: ALL_CAPS (template params / macros: NC, DV, D) or a
#: `k`-prefixed CamelCase constant (kDecodeWarps, kMaxLevels, kKinds).
CONSTEXPR_LOOKING = re.compile(r"^\s*(?:sizeof\s*\(.*\)|\d+|k[A-Z]\w*|[A-Z][A-Z0-9_]*)\s*$")

FOR_RE = re.compile(r"^\s*for\s*\(")
WHILE_RE = re.compile(r"^\s*while\s*\(")
#: The stop clause of a `for (init; STOP; step)` -- second `;`-separated field.
FOR_STOP_RE = re.compile(r"for\s*\([^;]*;\s*([^;]*);")
#: A `while (COND)` condition, relational-shaped: `X < BOUND`, `X <= BOUND`, ...
WHILE_COND_RE = re.compile(r"while\s*\((.*)\)\s*\{?\s*$")
REL_BOUND_RE = re.compile(r"[<>]=?\s*(.+?)\s*$")

PRAGMA_UNROLL_RE = re.compile(r"^\s*#pragma\s+unroll(\s+1)?\s*$")
COMMENT_LINE_RE = re.compile(r"^\s*//")

TORCH_INCLUDE_RE = re.compile(r'#include\s*<\s*torch/')

SYNC_RE = re.compile(r"__syncthreads\s*\(|__syncwarp\s*\(")
WALK_MARKER_RE = re.compile(r"----\s*THE WALK")


def device_files():
    for suf in DEVICE_SUFFIXES:
        for p in CSRC_ROLA.rglob(f"*{suf}"):
            if "third_party" in p.parts:
                continue
            yield p


DEVICE_QUALIFIER_RE = re.compile(r"\b__(global|device)__\b")
#: A `.cu` mixes HOST launcher code (std::vector loops over `widths`, TORCH_CHECK,
#: TensorOptions -- none of it unroll-relevant) with the device kernels §6 is
#: actually about. Scope the check to the textual span of each `__global__`/
#: `__device__` function body, found by brace-counting from the qualifier line
#: to the matching close -- a heuristic itself (see module docstring): it does
#: not parse strings/char literals, so a stray unbalanced brace in a string
#: literal (none observed in this codebase) would throw off the span.
def device_code_line_ranges(lines: list[str]) -> list[tuple[int, int]]:
    ranges = []
    n = len(lines)
    i = 0
    while i < n:
        if DEVICE_QUALIFIER_RE.search(lines[i]) and not COMMENT_LINE_RE.match(lines[i]):
            # Find the function's opening brace, scanning forward (signatures
            # with __launch_bounds__(...) or wrapped parameter lists span
            # multiple lines).
            j = i
            open_line = None
            while j < n:
                for ch in lines[j]:
                    if ch == "{":
                        open_line = j
                        break
                if open_line is not None:
                    break
                j += 1
            if open_line is None:
                i += 1
                continue
            depth = 0
            k = open_line
            close_line = None
            started = False
            while k < n:
                for ch in lines[k]:
                    if ch == "{":
                        depth += 1
                        started = True
                    elif ch == "}":
                        depth -= 1
                        if started and depth == 0:
                            close_line = k
                            break
                if close_line is not None:
                    break
                k += 1
            if close_line is None:
                close_line = n - 1  # unterminated scan -- best effort to EOF
            ranges.append((i, close_line))
            i = close_line + 1
        else:
            i += 1
    return ranges


def in_any_range(idx: int, ranges: list[tuple[int, int]]) -> bool:
    return any(lo <= idx <= hi for lo, hi in ranges)


def bound_looks_constexpr(bound: str) -> bool:
    bound = bound.strip()
    if not bound:
        return True  # nothing to judge (e.g. infinite `while (true)` handled earlier)
    return bool(CONSTEXPR_LOOKING.match(bound))


def preceding_pragma_unroll1(lines: list[str], loop_line_idx: int) -> bool:
    """True iff a `#pragma unroll 1` (explicit 1, not bare) sits directly
    above the loop, skipping over `//` comment lines only."""
    i = loop_line_idx - 1
    while i >= 0 and COMMENT_LINE_RE.match(lines[i]):
        i -= 1
    if i < 0:
        return False
    m = PRAGMA_UNROLL_RE.match(lines[i])
    return bool(m and m.group(1))  # group(1) present => "unroll 1" explicitly


def check_unroll_discipline() -> list[str]:
    findings = []
    for path in device_files():
        rel = path.relative_to(ROOT)
        lines = path.read_text().splitlines()
        ranges = device_code_line_ranges(lines)
        for idx, line in enumerate(lines):
            if not in_any_range(idx, ranges):
                continue  # host launcher code -- §6 is about device code
            header = "\n".join(lines[idx : idx + 3])
            bound = None
            if FOR_RE.match(line):
                m = FOR_STOP_RE.search(header)
                if not m:
                    continue
                stop = m.group(1)
                bm = REL_BOUND_RE.search(stop)
                bound = bm.group(1) if bm else stop
            elif WHILE_RE.match(line):
                m = WHILE_COND_RE.search(line)
                cond = m.group(1) if m else ""
                if not re.search(r"[<>]=?", cond):
                    continue  # no relational stop expression to classify
                bm = REL_BOUND_RE.search(cond)
                bound = bm.group(1) if bm else cond
            else:
                continue

            if bound_looks_constexpr(bound):
                continue
            if preceding_pragma_unroll1(lines, idx):
                continue
            findings.append(
                f"§6: {rel}:{idx + 1}: loop bound `{bound.strip()}` does not read as "
                f"compile-time and no `#pragma unroll 1` precedes it -- "
                f"{line.strip()!r}"
            )
    return findings


def check_no_torch_in_arm_tus() -> list[str]:
    findings = []
    if not INSTANTIATIONS.is_dir():
        return findings
    for path in INSTANTIATIONS.rglob("*"):
        if not path.is_file() or path.suffix not in DEVICE_SUFFIXES:
            continue
        rel = path.relative_to(ROOT)
        for idx, line in enumerate(path.read_text().splitlines(), start=1):
            if TORCH_INCLUDE_RE.search(line):
                findings.append(
                    f"K36 (belt to torch_include_in_arm_tu.yml): {rel}:{idx}: "
                    f"torch include in a per-arm TU -- {line.strip()!r}"
                )
    return findings


def check_no_sync_after_decode_prologue() -> list[str]:
    findings = []
    if not DECODE_CU.is_file():
        return findings
    rel = DECODE_CU.relative_to(ROOT)
    lines = DECODE_CU.read_text().splitlines()
    marker_idx = next((i for i, l in enumerate(lines) if WALK_MARKER_RE.search(l)), None)
    if marker_idx is None:
        findings.append(
            f"P79's law: {rel}: could not locate the `---- THE WALK` prologue-end "
            f"marker this check keys on -- the file was restructured and this check "
            f"needs a new anchor."
        )
        return findings
    for idx in range(marker_idx + 1, len(lines)):
        if SYNC_RE.search(lines[idx]):
            findings.append(
                f"P79's law (ledger 2026-08-27, decode.cu#per-cta-prologue): "
                f"{rel}:{idx + 1}: barrier after the prologue marker "
                f"(line {marker_idx + 1}) -- {lines[idx].strip()!r}"
            )
    return findings


#: K39 (Blake, 2026-08-28): the 2026-08-04 comments-to-docs mandate names "well
#: under 15%" as the line -- a named constant, not a number buried in an f-string,
#: so a future re-derivation of the threshold has exactly one place to change it.
COMMENT_DENSITY_THRESHOLD = 0.15

#: The mandate's own grain: FILE-HEADER and DECL-BLOCK comments are what stays;
#: granular prose belongs in docs/internals/ mirroring csrc/. A comment RUN this
#: long, found OUTSIDE the file's opening header block, is the shape the mandate
#: calls an antipattern -- a decl-block comment names what a declaration IS, it
#: does not walk a reader through a derivation.
MAX_COMMENT_BLOCK_LINES = 6

_CSRC = ROOT / "csrc"
_SOURCE_SUFFIXES = (".cu", ".cuh", ".cpp", ".h")


def _source_files():
    for suf in _SOURCE_SUFFIXES:
        for p in _CSRC.rglob(f"*{suf}"):
            if "third_party" in p.parts:
                continue
            yield p


def _comment_density(lines: list[str]) -> tuple[int, int]:
    """`(comment_lines, total_lines)`, counting `//` line comments and `/* */`
    block comments (a comment line inside an unterminated block still counts,
    even with no leading `//` of its own)."""
    comment = 0
    in_block = False
    for line in lines:
        s = line.strip()
        if in_block:
            comment += 1
            if "*/" in s:
                in_block = False
            continue
        if s.startswith("//"):
            comment += 1
        elif s.startswith("/*"):
            comment += 1
            if "*/" not in s:
                in_block = True
    return comment, len(lines)


def check_comment_density() -> list[str]:
    """Per-file comment-line fraction under `csrc/`, KERNEL_STANDARDS' 2026-08-04
    comments-to-docs mandate. REPORTED, not gated to zero: these are findings for
    Blake to rule on (S5b moves the prose to `docs/internals/`), never silently
    fixed or silenced by this script -- same treatment every other check here
    gets (module docstring)."""
    findings = []
    for p in _source_files():
        lines = p.read_text().splitlines()
        if not lines:
            continue
        comment, total = _comment_density(lines)
        density = comment / total
        if density > COMMENT_DENSITY_THRESHOLD:
            findings.append(
                f"comment density (2026-08-04 comments-to-docs mandate, 'well "
                f"under {COMMENT_DENSITY_THRESHOLD:.0%}'): {p.relative_to(ROOT)}: "
                f"{comment}/{total} lines = {density:.0%} comment -- move the "
                f"prose to docs/internals/, decl-block/file-header comments only"
            )
    return findings


def check_no_long_comment_blocks() -> list[str]:
    """A `//`-comment run longer than `MAX_COMMENT_BLOCK_LINES`, found OUTSIDE
    the file's own opening header block (the copyright/module-docstring run
    starting at line 1), is a NEW prose block the comments-to-docs mandate
    exists to keep out -- reported the same way `check_comment_density` is,
    never auto-fixed."""
    findings = []
    for p in _source_files():
        lines = p.read_text().splitlines()
        n = len(lines)
        header_end = 0
        while header_end < n and (lines[header_end].strip().startswith("//")
                                   or lines[header_end].strip() == ""):
            header_end += 1
        i = header_end
        while i < n:
            if lines[i].strip().startswith("//"):
                start = i
                while i < n and lines[i].strip().startswith("//"):
                    i += 1
                block_len = i - start
                if block_len > MAX_COMMENT_BLOCK_LINES:
                    findings.append(
                        f"comment block (2026-08-04 comments-to-docs mandate, "
                        f"decl-block grain <= {MAX_COMMENT_BLOCK_LINES} lines): "
                        f"{p.relative_to(ROOT)}:{start + 1}-{i}: a {block_len}-line "
                        f"comment run outside the file header -- prose this long "
                        f"belongs in docs/internals/, not a new `//:` block here"
                    )
            else:
                i += 1
    return findings


def check_tests_tier_directories() -> list[str]:
    """docs/testing.md "there is no fourth tier": no directory directly under
    `tests/` other than `unit/`, `integration/` or `oracle/` (a campaign name --
    `tests/k31/` was one -- is exactly the shape of the violation this catches).
    Hidden directories (e.g. `__pycache__`, `.pytest_cache`) are not test tiers
    and are exempt."""
    findings = []
    if not TESTS_ROOT.is_dir():
        return findings
    for child in sorted(TESTS_ROOT.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if child.name == "__pycache__":
            continue
        if child.name not in TESTS_ALLOWED_TIERS:
            findings.append(
                f"docs/testing.md 'there is no fourth tier': tests/{child.name}/ is a "
                f"directory under tests/ that is none of {sorted(TESTS_ALLOWED_TIERS)} -- "
                f"fold its tests into the tier each one answers, by question, not by "
                f"campaign name."
            )
    return findings


def main() -> int:
    #: THESE FOUR are wired as a BLOCKING pre-commit/CI gate
    #: (.pre-commit-config.yaml, .github/workflows/lint.yml) -- a nonzero exit
    #: here fails every commit and every CI run, repo-wide.
    gating_findings: list[str] = []
    gating_findings += check_unroll_discipline()
    gating_findings += check_no_torch_in_arm_tus()
    gating_findings += check_no_sync_after_decode_prologue()
    gating_findings += check_tests_tier_directories()

    #: K39 (Blake, 2026-08-28): comment density and long-comment-block findings
    #: are ADVISORY, NOT gating, deliberately. `csrc/` is ~40 files over the
    #: mandate's 15% line TODAY (S5b is the stage that moves the prose to
    #: docs/internals/ -- "these are findings, not to be fixed by you" is the
    #: instruction this branch received). Wiring them into the SAME exit code
    #: as the three checks above would fail every future commit on PRE-EXISTING
    #: debt this stage did not create, for every contributor, until S5b lands --
    #: a gate that goes red on touching an unrelated line is a gate nobody can
    #: satisfy. They are reported here, every run, so the debt stays visible and
    #: S5b has a checkable before/after, without blocking anything today.
    advisory_findings: list[str] = []
    advisory_findings += check_comment_density()
    advisory_findings += check_no_long_comment_blocks()

    if advisory_findings:
        print(f"lint_standards: {len(advisory_findings)} ADVISORY finding(s) "
              f"(not gating -- comments-to-docs mandate debt, pending S5b)")
        for f in advisory_findings:
            print(f"  {f}")

    if not gating_findings:
        print("lint_standards: no gating findings")
        return 0

    print(f"lint_standards: {len(gating_findings)} gating finding(s)")
    for f in gating_findings:
        print(f"  {f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
