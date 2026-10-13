#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE SKILL ROUTER (KERNEL_STANDARDS section 31), a PreToolUse hook. A tool call is routed by the checkout it TOUCHES:
an edit by the edited file's; a command by each of its simple commands (split on `&&`, `||`, `;`, `|` and newlines),
each in the directory a `cd`, `pushd` or `git -C` before it names, else the session's. The checkout's
`.claude/skill-routes.json` matches edits by glob and commands by regex anchored at the command's first token, so a
command that merely mentions a tool routes nothing. The text of each matched skill, read from that same checkout, goes
into the acting agent's context once an agent; `--forget` (the SessionStart hook after a compaction or a resume) drops
the main agent's record, so its next action brings the skill back. Any number of worktrees on any branches route from
any session, and nothing is registered globally. It never blocks a call."""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROUTES = Path(".claude") / "skill-routes.json"
#: a command line's simple commands, and the prefixes a simple command's tool hides behind
SPLIT = re.compile(r"\s*(?:&&|\|\||;|\||\n)\s*")
PREFIX = re.compile(r"^(?:\(\s*|\w+=\S*\s+|timeout\s+(?:-s\s+\S+\s+)?\S+\s+|nice\s+(?:-n\s+-?\d+\s+)?|env\s+)*")
#: a simple command that changes the directory, and `git -C <dir>`
CHDIR = re.compile(r"^(?:cd|pushd)\s+(\S+)$")
GIT_C = re.compile(r"\bgit\s+-C\s+(\S+)")


def checkout(path: Path) -> Path | None:
    """The nearest directory at or above `path` that ships routes."""
    return next((d for d in (path, *path.parents) if (d / ROUTES).is_file()), None)


def steps(call: dict) -> list[tuple[Path, str]]:
    """(directory, simple command) for a command, ('edited path', '') for an edit."""
    args = call.get("tool_input") or {}
    here = Path(call.get("cwd") or os.getcwd())
    if call.get("tool_name") != "Bash":
        path = args.get("file_path") or args.get("notebook_path")
        return [((here / path).resolve(), "")] if path else []
    out = []
    for part in SPLIT.split(args.get("command", "")):
        command = PREFIX.sub("", part.strip()).strip()
        if not command:
            continue
        if m := CHDIR.match(command):
            here = (here / os.path.expanduser(m.group(1).strip("'\""))).resolve()
            continue
        m = GIT_C.search(command)
        out.append(((here / os.path.expanduser(m.group(1).strip("'\""))).resolve() if m else here, command))
    return out


def claim(record: Path, root: Path, skill: str) -> bool:
    """True for exactly one caller an (agent, checkout, skill): the repo's hook and the user's bootstrap run at once on
    one call, so the claim is an exclusive file creation, never a read then a write."""
    record.mkdir(exist_ok=True)
    try:
        os.close(os.open(record / hashlib.sha256(f"{root}\0{skill}".encode()).hexdigest()[:16],
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def procedure(root: Path, skill: str) -> str | None:
    """A skill's text past its frontmatter."""
    try:
        text = (root / ".claude" / "skills" / skill / "SKILL.md").read_text()
    except OSError:
        return None
    return text.split("\n---\n", 1)[1].strip() if text.startswith("---\n") and "\n---\n" in text else text.strip()


def record_of(call: dict) -> Path:
    """An agent's record: a context is an agent's, and subagents share the session's id under their own `agent_id`."""
    name = re.sub(r"[^\w-]", "_", f"{call.get('session_id') or 'none'}-{call.get('agent_id') or 'main'}")
    return Path(tempfile.gettempdir()) / f"claude-skill-routes-{name}"


def main() -> int:
    try:
        call = json.load(sys.stdin)
    except (ValueError, OSError):
        return 0
    if "--forget" in sys.argv:
        shutil.rmtree(record_of(call), ignore_errors=True)
        return 0
    try:
        found = steps(call)
    except (OSError, TypeError, ValueError):
        return 0
    wanted: dict[tuple[Path, str], None] = {}
    for where, command in found:
        root = checkout(where if command or where.is_dir() else where.parent)
        if root is None:
            continue
        try:
            routes = json.loads((root / ROUTES).read_text())["routes"]
        except (OSError, ValueError, KeyError):
            continue
        edited = "" if command else where.relative_to(root).as_posix()
        for r in routes:
            if (edited and any(fnmatch.fnmatch(edited, g) for g in r.get("edit", []))) or (
                    command and any(re.match(p, command) for p in r.get("bash", []))):
                wanted[(root, r["skill"])] = None
    record = record_of(call)
    texts = [(root, f"=== skill `{s}` ===\n{text}") for root, s in wanted
             if (text := procedure(root, s)) and claim(record, root, s)]
    if not texts:
        return 0
    roots = ", ".join(dict.fromkeys(str(root) for root, _ in texts))
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "additionalContext": f"Skill route (KERNEL_STANDARDS section 31): this action in the checkout {roots} is "
                             "covered by the procedure below, from that checkout's own skills. Follow it.\n\n"
                             + "\n\n".join(text for _, text in texts)}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
