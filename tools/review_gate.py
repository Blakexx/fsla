#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE REVIEW GATE: a commit, a branch's range or the staged diff reviewed by an LLM against the repository's standards
-- docs/CONSTITUTION.md and docs/KERNEL_STANDARDS.md -- every finding citing the rule it breaks and the changed line it
breaks it at. A review runs after its commit and never slows one; a landed review's `block` finding refuses the NEXT
commit until its message answers it (`--unanswered`, run by tools/git-hooks/commit-msg; Blake, 2026-11-17).

    bazel run //tools:review_gate                                    # HEAD
    bazel run //tools:review_gate -- 4d9a3210                        # one commit
    bazel run //tools:review_gate -- --range origin/carry-k5..HEAD   # a branch: the local review in place of a PR
    bazel run //tools:review_gate -- --staged                        # what is staged
    bazel run //tools:review_gate -- --show 5                        # the last five reports
    bazel run //tools:review_gate -- HEAD                            # what tools/git-hooks/post-commit runs, detached

The reviewer is `claude -p` (Opus by default, `--model`) in the checkout, its tools read-only (Read, Grep, Glob), its
output a fixed schema. A report is cached by everything it read (the standards, the message, the diff, the model, the
prompt's version) and kept in the git common dir (`rola-review/`), shared by every worktree. Docs:
docs/internals/tools/review_gate.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STANDARDS = ("docs/CONSTITUTION.md", "docs/KERNEL_STANDARDS.md")
#: generated or recorded bytes a review reads nothing in: measurement records, pinned locks, ratified manifests
EXCLUDED = ("measure/lab/results", "docs/references.lock.json", "tools/manifests")
#: the most diff a prompt carries inline; past it the reviewer reads the files the stat names
DIFF_LIMIT = 200_000
#: bump when the instructions change, so no cached report outlives them
PROMPT_VERSION = 3
SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rule": {"type": "string"},
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    "severity": {"type": "string", "enum": ["block", "warn"]},
                    "claim": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["rule", "file", "line", "severity", "claim", "evidence"],
            },
        },
    },
    "required": ["summary", "findings"],
}
INSTRUCTIONS = """You review a change to the RoLA repository against its standards: the CONSTITUTION (Blake's standing
rulings) and KERNEL_STANDARDS (the laws), both given in full below, then the change's message(s) and diff.

Report ONLY a violation of a specific rule you can cite by its section (e.g. "KERNEL_STANDARDS section 20",
"CONSTITUTION R10") at a specific changed line, with the evidence quoted from the diff or the file. Read the touched
files, their mirror pages under docs/internals/ and anything else in the checkout you need (Read, Grep, Glob) before
asserting; a claim you did not check is not a finding. Severity "block": a clear violation of a constitution rule or a
law as written. "warn": a likely one, or a law's letter met and its stated purpose missed. No style preferences, no
suggestions the standards do not require, no restating what the change does. A change that breaks nothing returns an
empty list. The summary is one or two sentences: what was reviewed and the verdict.

WHAT THE CHANGE MAKES OBSOLETE (the CONSTITUTION's ONE FORM PER MECHANISM and INTERFACE FIRST): for each mechanism the
change adds or replaces, search the checkout for what else does that job or fed the old one, and report what it left
behind -- a second path doing the new mechanism's job (a parallel lookup, a duplicate table or switch); a field, flag,
option or table whose consumers the change removed or replaced, or whose only reader is now a check of itself; a
decision still computed that nothing acts on; a guard or refusal made unreachable; docs, comments or tests
still stating the replaced form. A name with no reader at all is the lints' to catch; this is the semantic case: code
still read, or still compiling, that the change made unneeded. Cite the rule and the changed line that obsoletes it.

THE DOCS' ALTITUDE (the CONSTITUTION's SELF-DOCUMENTING CODE and THE REPO IS ONE ATOMIC DESIGN): a doc passage the change
adds or edits states why a piece exists and how it connects to the rest; one that restates what the code does, step by
step, belongs to the code and is a warn. A passage that points outside the repo for a rule, a design or a fact (a log,
a plan, a tracker) is a block; another project's design cited as the reason ours differs is not.
"""


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def store() -> Path:
    common = Path(git("rev-parse", "--git-common-dir").strip())
    out = (common if common.is_absolute() else ROOT / common) / "rola-review"
    out.mkdir(parents=True, exist_ok=True)
    return out


def target(args) -> tuple[str, str, str]:
    """``(id, messages, diff)`` of what is reviewed."""
    excl = [f":(exclude){p}" for p in EXCLUDED]
    if args.staged:
        diff = git("diff", "--cached", "--find-renames", "--", ".", *excl)
        return "staged-" + hashlib.sha256(diff.encode()).hexdigest()[:12], "(staged, not yet committed)", diff
    if args.range:
        a, b = args.range.split("..", 1)
        a, b = git("rev-parse", a).strip(), git("rev-parse", b or "HEAD").strip()
        return (f"range-{a[:12]}-{b[:12]}", git("log", "--format=%h %s%n%n%b", f"{a}..{b}"),
                git("diff", "--find-renames", a, b, "--", ".", *excl))
    sha = git("rev-parse", args.rev).strip()
    return sha, git("log", "-1", "--format=%B", sha), git("show", "--format=", "--find-renames", sha, "--", ".", *excl)


def prompt(messages: str, diff: str) -> str:
    parts = [INSTRUCTIONS]
    for name in STANDARDS:
        parts.append(f"===== {name} =====\n{(ROOT / name).read_text()}")
    parts.append(f"===== THE CHANGE'S MESSAGE(S) =====\n{messages}")
    if len(diff) > DIFF_LIMIT:
        parts.append(f"===== THE DIFF, its first {DIFF_LIMIT} of {len(diff)} bytes (read the rest from the files) =====\n"
                     + diff[:DIFF_LIMIT])
    else:
        parts.append(f"===== THE DIFF =====\n{diff}")
    return "\n\n".join(parts)


def review(rid: str, text: str, model: str) -> dict:
    key = hashlib.sha256(f"{PROMPT_VERSION}\n{model}\n{text}".encode()).hexdigest()[:20]
    cached = store() / "cache" / f"{key}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    t0 = time.time()
    proc = subprocess.run(
        ["claude", "-p", "--model", model, "--output-format", "json", "--json-schema", json.dumps(SCHEMA),
         "--tools", "Read", "Grep", "Glob", "--allowedTools", "Read", "Grep", "Glob", "--disable-slash-commands",
         "--no-session-persistence", "--strict-mcp-config",
         "Review the change given on standard input, following the instructions at its top."],
        cwd=ROOT, input=text, capture_output=True, text=True, timeout=1800)
    out = json.loads(proc.stdout) if proc.stdout.strip().startswith("{") else {}
    report = {"id": rid, "model": model, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "seconds": round(time.time() - t0, 1), "cost_usd": out.get("total_cost_usd"),
              "ok": proc.returncode == 0 and not out.get("is_error", True) and out.get("structured_output") is not None,
              "error": None if proc.returncode == 0 else (proc.stderr or proc.stdout)[-2000:],
              **(out.get("structured_output") or {"summary": "", "findings": []})}
    if report["ok"]:
        cached.parent.mkdir(exist_ok=True)
        cached.write_text(json.dumps(report, indent=1))
    return report


#: an answer to a block finding, in a commit's message: the finding by its commit and its number among that report's blocks
TRAILER = re.compile(r"^Review-(?:Fix|Override): ([0-9a-f]{7,40}):(\d+) (?:\S+(?: \S+){2,})", re.M)
HOOK = "tools/git-hooks/commit-msg"
WINDOW = 30


def open_blocks(commits: list[tuple[str, str]], reports: dict[str, dict], message: str) -> list[tuple[str, int, dict]]:
    """The block findings nothing answers: `commits` newest first as (sha, message), `reports` the landed reviews by
    sha, `message` the commit being made. A finding is answered by a trailer in any message after its commit."""
    later, out = message, []
    for sha, body in commits:
        found = reports.get(sha) or {}
        answered = {int(n) for s, n in TRAILER.findall(later) if sha.startswith(s)}
        blocks = [f for f in found.get("findings", []) if f["severity"] == "block"] if found.get("ok") else []
        out += [(sha, i, f) for i, f in enumerate(blocks, 1) if i not in answered]
        later += "\n" + body
    return out


def unanswered(message: str) -> list[tuple[str, int, dict]]:
    """`open_blocks` of the branch's last `WINDOW` commits made since the commit hook entered the tree."""
    born = git("log", "--diff-filter=A", "--format=%H", "--", HOOK).split()
    if not born:
        return []
    log = git("log", f"--max-count={WINDOW}", "--format=%H%x00%B%x01", f"{born[-1]}~1..HEAD")
    commits = [tuple(c.strip("\n").split("\x00", 1)) for c in log.split("\x01") if c.strip()]
    reports = {sha: json.loads(p.read_text()) for sha, _ in commits if (p := store() / f"{sha}.json").exists()}
    return open_blocks(commits, reports, "\n".join(ln for ln in message.splitlines() if not ln.startswith("#")))


def render(report: dict) -> str:
    head = (f"review {report['id'][:12]} ({report['model']}, {report['seconds']} s, ${report.get('cost_usd') or 0:.2f})"
            f": {'' if report['ok'] else 'FAILED -- '}{report.get('summary', '')}")
    lines = [head]
    for f in sorted(report.get("findings", []), key=lambda f: f["severity"] != "block"):
        lines.append(f"  {f['severity'].upper():5s} {f['rule']} -- {f['file']}:{f['line']}: {f['claim']}")
        lines.append(f"        evidence: {f['evidence'][:300]}")
    if report.get("error"):
        lines.append(f"  error: {report['error'][-500:]}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rev", nargs="?", default="HEAD")
    ap.add_argument("--range")
    ap.add_argument("--staged", action="store_true")
    ap.add_argument("--model", default="opus")
    ap.add_argument("--show", type=int, metavar="N")
    ap.add_argument("--unanswered", metavar="MESSAGE_FILE", help="refuse while a landed review's block is unanswered")
    args = ap.parse_args()

    if args.unanswered is not None:
        found = unanswered(Path(args.unanswered).read_text())
        for sha, n, f in found:
            print(f"BLOCK {sha[:10]}:{n} {f['rule'][:80]} -- {f['file']}:{f['line']}: {f['claim'][:300]}")
        if found:
            print("review gate: refusing -- answer each block above in this commit's message, one trailer each: "
                  "`Review-Fix: <sha>:<n> <what the fix is>` or `Review-Override: <sha>:<n> <the rule and why the "
                  "finding is wrong>` (docs/internals/tools/review_gate.md#blocking)")
        return 1 if found else 0
    if args.show is not None:
        reports = sorted(store().glob("*.json"), key=lambda p: p.stat().st_mtime)[-args.show:]
        for p in reports:
            print(render(json.loads(p.read_text())))
        return 0
    rid, messages, diff = target(args)
    if not diff.strip():
        print(f"review {rid[:12]}: nothing to review")
        return 0
    report = review(rid, prompt(messages, diff), args.model)
    (store() / f"{rid}.json").write_text(json.dumps(report, indent=1))
    print(render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
