#!/usr/bin/env python3
"""THE RATCHET: a lint the tree does not yet pass gates on what a commit ADDS, and its old findings can only go away.

    bazel test //:ratchet_work_codes                        # the gate: a new finding fails, and so does a fixed one
    bazel run //tools/lint:ratchet -- work_codes --shrink   # drop the fixed entries (it never adds one)
    bazel run //tools/lint:ratchet -- work_codes --init     # record today's findings; refused when a baseline exists
    bazel run //tools/lint:ratchet -- work_codes --rekey OLD NEW   # a finding's message was reworded: rename its entries
    bazel run //tools/lint:ratchet -- work_codes --follow  # files moved or lines reworded since HEAD: entries follow them
    bazel test //:ratchet_baselines                         # every baseline against HEAD's alone, in the checkout

A lint's findings are its `path:line: message` lines. A finding is known by its path, its message's first sentence and
a digest of the line it names, never by the line number, which any edit above it moves. `tools/lint/baselines/<lint>.json`
holds the findings the tree carried when the lint began gating, with their multiplicity: the backlog. The gate fails on

- a finding the baseline does not hold: fix it, or exempt it the lint's own way (vulture's whitelist, a `reserved:`
  marker), with the reason;
- a baseline entry the lint no longer finds: `--shrink` removes it in the commit that fixed it, so the backlog on disk
  is the backlog in the tree;
- a baseline that holds more of an entry than HEAD's: a backlog only shrinks. `--grown` checks it, every baseline
  at once, in the checkout -- the graph's check of the commit; a lint's run reads a copy of the tree, which has no
  HEAD.

A finding's message is part of its key, so rewording it would read as a new finding and a fixed one. A RE-KEY is the stated
rename: `--rekey OLD NEW` rewrites the baseline's entries whose message is OLD to NEW, their counts unchanged, and records
`{"from", "to", "count"}` in the baseline's `rekeys`. `--grown` reads HEAD's entries through the baseline's own re-keys, so a
re-keyed entry is the same entry and a re-key that adds to a count is growth; `--shrink` drops a re-key whose `to` the
baseline no longer holds.

A FOLLOW is the stated move of a file and the reword of a line: a finding is known by the digest of its line, so
`git mv` plus a rewrite of the line reads as a new finding and a fixed one. `--follow` takes each entry's line in HEAD's
file, follows it through git's rename detection against HEAD and a line diff to the working file, and rewrites the entry
to the file and the digest it now has, counts unchanged -- never to an entry it was not: a line that is gone or was
rewritten beyond a one-for-one swap keeps its entry, which then reads NEW and fixed as it should. It records
`{"path", "from_sha256", "to_sha256", "count"}` in the baseline's `follows`, which `--grown` reads HEAD's entries through
(and re-derives from git, so a hand-written follow is refused); `--shrink` drops a follow whose `to_sha256` is not held.

A lint that cannot run (a missing tool, a crash) fails the gate; it never reads as zero findings.
Docs: docs/internals/tools/lint.md#ratchets.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "tools" / "lint" / "baselines"
LINTS = {
    "work_codes": [sys.executable, "tools/lint/work_codes.py"],
    "vulture": ["bash", "tools/lint/run_vulture.sh"],
    "constant_parameters": [sys.executable, "tools/lint/constant_parameters.py"],
    "drift_guards": [sys.executable, "tools/lint/drift_guards.py"],
    "facts_in_template": [sys.executable, "tools/lint/facts_in_template.py"],
    "r9_enforcement": [sys.executable, "tools/lint/r9_enforcement.py"],
    "burst_tier": [sys.executable, "tools/lint/burst_tier.py"],
    "coverage": [sys.executable, "tools/lint/coverage.py"],
    "rules_owed": [sys.executable, "tools/lint/rules_table.py", "--owed"],
}
FINDING = re.compile(r"^\s*([^\s:]+):(\d+): (.+)$")


def findings(lint: str) -> Counter:
    done = subprocess.run(LINTS[lint], cwd=ROOT, capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f"ratchet {lint}: the lint did not run (exit {done.returncode}):\n{(done.stdout + done.stderr)[-1500:]}")
    found: Counter = Counter()
    for line in done.stdout.splitlines():
        match = FINDING.match(line)
        if match:
            path, lineno, message = match.group(1), int(match.group(2)), match.group(3).strip().split(". ")[0]
            source = ROOT / path
            text = source.read_text(errors="replace").splitlines() if source.is_file() else []
            named = text[lineno - 1].strip() if 0 < lineno <= len(text) else ""
            found[(path, message, hashlib.sha256(named.encode()).hexdigest()[:16])] += 1
    return found


def _decode(blob: str) -> Counter:
    return Counter({(e["path"], e["message"], e["line_sha256"]): e["count"] for e in json.loads(blob)["findings"]})


def _rekeys(blob: str) -> list[dict]:
    """A baseline's recorded re-keys: `{"from": the old message, "to": the new one, "count": the entries renamed}`."""
    return json.loads(blob).get("rekeys", [])


def _follows(blob: str) -> list[dict]:
    """A baseline's recorded follows: `{"path": HEAD's, "from_sha256", "to_sha256", "count"}`, the entries a file's move
    and a line's reword carried (`follow`)."""
    return json.loads(blob).get("follows", [])


def _encode(lint: str, entries: Counter, rekeys: list[dict] | None = None, follows: list[dict] | None = None) -> str:
    rows = [{"path": p, "message": m, "line_sha256": d, "count": n} for (p, m, d), n in sorted(entries.items()) if n > 0]
    record = {"lint": lint, "findings": rows}
    if rekeys:
        record["rekeys"] = rekeys
    if follows:
        record["follows"] = follows
    return json.dumps(record, indent=1) + "\n"


def rekeyed(entries: Counter, rekeys: list[dict]) -> Counter:
    """`entries` with each re-key applied: the entries whose message is a re-key's `from` carry its `to`, their counts
    unchanged and added where an entry of the new key exists."""
    renamed = {r["from"]: r["to"] for r in rekeys}
    out: Counter = Counter()
    for (path, message, digest), n in entries.items():
        out[(path, renamed.get(message, message), digest)] += n
    return out


def rekey(entries: Counter, old: str, new: str) -> tuple[Counter, dict]:
    """The entries after the re-key of message `old` to `new`, and its record; refused when no entry has the message `old`."""
    count = sum(n for (_, message, _), n in entries.items() if message == old)
    if not count:
        raise SystemExit(f"ratchet: no baseline entry has the message {old!r}; a re-key renames entries it holds")
    return rekeyed(entries, [{"from": old, "to": new}]), {"from": old, "to": new, "count": count}


def _git(*args: str) -> str:
    done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return done.stdout if done.returncode == 0 else ""


def _moves() -> dict[str, str]:
    """HEAD's path -> the working tree's, for each file git's rename detection finds moved (staged or not)."""
    moved = {}
    for row in _git("diff", "-M", "--name-status", "HEAD").splitlines():
        cells = row.split("\t")
        if cells[0].startswith("R") and len(cells) == 3:
            moved[cells[1]] = cells[2]
    return moved


def follow_line(old_path: str, digest: str, moves: dict[str, str]) -> tuple[str, set[str]] | None:
    """Where the line of HEAD's `old_path` with `digest` is in the working tree: `(its path, the digests of the lines it
    maps to)`, a line mapping one for one through a line diff (an unchanged line, or one swapped for one); None when the
    file is gone or no line of it has the digest, and an empty set when a matching line did not map."""
    new_path = moves.get(old_path, old_path)
    target = ROOT / new_path
    old = _git("show", f"HEAD:{old_path}").splitlines()
    if not old or not target.is_file():
        return None
    new = target.read_text(errors="replace").splitlines()
    hits = [i for i, line in enumerate(old) if hashlib.sha256(line.strip().encode()).hexdigest()[:16] == digest]
    if not hits:
        return None
    mapped: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            mapped.update({i1 + k: j1 + k for k in range(i2 - i1)})
    if any(i not in mapped for i in hits):
        return new_path, set()
    return new_path, {hashlib.sha256(new[mapped[i]].strip().encode()).hexdigest()[:16] for i in hits}


def follow(entries: Counter) -> tuple[Counter, list[dict]]:
    """`entries` (HEAD's keys) at the files and lines they now are, and the record of each digest that changed."""
    moves, out, record = _moves(), Counter(), []
    for (path, message, digest), n in entries.items():
        where = follow_line(path, digest, moves)
        if where is None or len(where[1]) != 1:
            out[(moves.get(path, path), message, digest)] += n
            continue
        new_path, (new_digest,) = where[0], tuple(where[1])
        out[(new_path, message, new_digest)] += n
        if new_digest != digest:
            record.append({"path": path, "from_sha256": digest, "to_sha256": new_digest, "count": n})
    return out, record


def followed(entries: Counter, follows: list[dict]) -> Counter:
    """`entries` (HEAD's) with each follow applied to the entry it names, its digest the one the line now has."""
    renamed = {(f["path"], f["from_sha256"]): f["to_sha256"] for f in follows}
    out: Counter = Counter()
    for (path, message, digest), n in entries.items():
        out[(path, message, renamed.get((path, digest), digest))] += n
    return out


def follows_unproven(head: Counter, follows: list[dict]) -> list[dict]:
    """The follows git does not bear out: a follow of an entry HEAD holds must be what `follow_line` finds in the files."""
    moves, held = _moves(), {(p, d) for (p, _, d) in head}
    bad = []
    for f in follows:
        if (f["path"], f["from_sha256"]) in held:
            where = follow_line(f["path"], f["from_sha256"], moves)
            if where is None or where[1] != {f["to_sha256"]}:
                bad.append(f)
    return bad


def _head(path: Path, rekeys: list[dict] | None = None, follows: list[dict] | None = None) -> Counter | None:
    """HEAD's baseline at `path`, read through `rekeys` and `follows` (the working baseline's own)."""
    shown = subprocess.run(["git", "show", f"HEAD:{path.relative_to(ROOT)}"], cwd=ROOT, capture_output=True, text=True)
    if shown.returncode:
        return None
    return followed(rekeyed(_decode(shown.stdout), rekeys or []), follows or [])


def verdict(found: Counter, baseline: Counter, head: Counter | None) -> tuple[Counter, Counter, Counter]:
    """`(new, fixed, grown)`: findings the baseline lacks, entries no longer found, entries HEAD's baseline lacks.

    A MOVE IS NOT GROWTH: a finding known to HEAD by its message and line digest under a path the baseline no longer
    carries is the same finding after a rename, so a renamed package moves its backlog and does not add to it."""
    if head is None:
        return found - baseline, baseline - found, Counter()
    left_behind = Counter()
    for (path, message, digest), n in (head - baseline).items():
        left_behind[(message, digest)] += n
    grown = Counter()
    for (path, message, digest), n in (baseline - head).items():
        moved = min(n, left_behind[(message, digest)])
        left_behind[(message, digest)] -= moved
        if n - moved:
            grown[(path, message, digest)] = n - moved
    return found - baseline, baseline - found, grown


def _show(title: str, entries: Counter) -> None:
    print(title)
    for (path, message, _digest), n in sorted(entries.items()):
        print(f"  {path}: {message}" + (f"  (x{n})" if n > 1 else ""))


def baselines_grown(lints: list[str]) -> int:
    """No baseline holds an entry HEAD's lacks -- a backlog only shrinks, and a move is not growth (`verdict`). Run
    against the checkout, which knows HEAD; the lints' own findings are the hermetic half. A HEAD git cannot read is
    red: a comparison that compared nothing has not passed."""
    readable = subprocess.run(["git", "rev-parse", "--verify", "--quiet", "HEAD"], cwd=ROOT, capture_output=True)
    if readable.returncode != 0:
        print(f"ratchet baselines: FAILED -- git cannot read HEAD in {ROOT}, so no baseline was compared")
        return 1
    grown_any, compared = False, 0
    for lint in lints:
        path = BASELINES / f"{lint}.json"
        #: a baseline HEAD lacks is new, and one the tree lacks is deleted: neither has grown
        if not path.exists():
            continue
        blob = path.read_text()
        head = _head(path, _rekeys(blob), _follows(blob))
        if head is None:
            continue
        compared += 1
        bad = follows_unproven(_head(path, _rekeys(blob)) or Counter(), _follows(blob))
        if bad:
            print(f"ratchet {lint}: {len(bad)} follow(s) git does not bear out (a follow is derived by `--follow`, never written):")
            grown_any = True
        grown = verdict(Counter(), _decode(path.read_text()), head)[2]
        if grown:
            _show(f"ratchet {lint}: the baseline GREW against HEAD's -- a backlog only shrinks:", grown)
            grown_any = True
    print(f"ratchet baselines: {'GROWN' if grown_any else 'none grown'} ({compared} of {len(lints)} compared against "
          "HEAD's; the rest new or deleted)")
    return 1 if grown_any else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lint", nargs="?", choices=sorted(LINTS))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--grown", action="store_true",
                      help="every baseline against HEAD's, none grown (the commit's half; no lint runs)")
    mode.add_argument("--shrink", action="store_true", help="remove the entries the lint no longer finds")
    mode.add_argument("--init", action="store_true", help="record the current findings as a new lint's baseline")
    mode.add_argument("--follow", action="store_true",
                      help="files moved and lines reworded since HEAD: the entries follow them, counts unchanged, and "
                           "the digests that changed are recorded")
    mode.add_argument("--rekey", nargs=2, metavar=("OLD", "NEW"),
                      help="a finding's message was reworded: rename the baseline's entries OLD to NEW, counts unchanged, "
                           "and record the rename in the baseline")
    a = ap.parse_args()
    if a.grown:
        return baselines_grown(sorted(LINTS) if a.lint is None else [a.lint])
    if a.lint is None:
        ap.error("name a lint, or pass --grown")
    path = BASELINES / f"{a.lint}.json"
    if a.follow:
        if not path.exists():
            raise SystemExit(f"ratchet {a.lint}: no baseline at {path.relative_to(ROOT)}")
        current = _decode(path.read_text())
        moved, record = follow(current)
        path.write_text(_encode(a.lint, moved, _rekeys(path.read_text()), [*_follows(path.read_text()), *record]))
        print(f"ratchet {a.lint}: {sum(current.values())} entr(ies) followed, {len(record)} digest(s) changed, the count unchanged")
        return 0
    found = findings(a.lint)

    if a.init:
        if path.exists():
            raise SystemExit(f"ratchet {a.lint}: {path.relative_to(ROOT)} exists; a baseline is recorded once")
        BASELINES.mkdir(parents=True, exist_ok=True)
        path.write_text(_encode(a.lint, found))
        print(f"ratchet {a.lint}: baseline recorded, {sum(found.values())} finding(s)")
        return 0
    if not path.exists():
        raise SystemExit(f"ratchet {a.lint}: no baseline at {path.relative_to(ROOT)} ("
                         f"bazel run //tools/lint:ratchet -- {a.lint} --init records one)")
    baseline = _decode(path.read_text())

    rekeys = _rekeys(path.read_text())
    if a.rekey:
        renamed, record = rekey(baseline, *a.rekey)
        path.write_text(_encode(a.lint, renamed, [*rekeys, record]))
        print(f"ratchet {a.lint}: {record['count']} entr(ies) re-keyed, the baseline's count {sum(baseline.values())} unchanged")
        return 0
    if a.shrink:
        kept = baseline & found
        path.write_text(_encode(a.lint, kept, [r for r in rekeys if any(m == r["to"] for (_, m, _) in kept)],
                                [f for f in _follows(path.read_text()) if any(d == f["to_sha256"] for (_, _, d) in kept)]))
        print(f"ratchet {a.lint}: baseline {sum(baseline.values())} -> {sum(kept.values())}")
        return 0

    new, fixed, _ = verdict(found, baseline, None)
    if new:
        _show(f"ratchet {a.lint}: {sum(new.values())} NEW finding(s) -- fix each, or exempt it the lint's own way:", new)
    if fixed:
        _show(f"ratchet {a.lint}: {sum(fixed.values())} baseline entr(ies) no longer found -- run "
              f"`bazel run //tools/lint:ratchet -- {a.lint} --shrink` and commit the smaller baseline:", fixed)
    total = sum(baseline.values())
    print(f"ratchet {a.lint}: {'FAIL' if new or fixed else 'OK'} ({total} in the backlog)")
    return 1 if new or fixed else 0


if __name__ == "__main__":
    sys.exit(main())
