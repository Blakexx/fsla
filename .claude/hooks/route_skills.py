#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE SKILL ROUTER (KERNEL_STANDARDS section 31), a PreToolUse hook. A tool call that edits a path or runs a command
is routed by the checkout it TOUCHES -- the edited file's, or the directory a command runs in (a leading `cd <dir>`,
`git -C <dir>`, else the session's) -- through that checkout's `.claude/skill-routes.json`, and the text of each skill
it names, read from that same checkout, goes into the session's context once a session. Any number of worktrees on any
branches work from any session, and nothing is registered globally. Each agent's context is its own: a subagent
gets a skill once in its context as the main agent does in its. It never blocks a call."""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROUTES = Path(".claude") / "skill-routes.json"
#: the directory a command runs in, when it names one: a leading `cd <dir> &&` (or `;`), or `git -C <dir>`
CD = re.compile(r"^\s*cd\s+(\S+)\s*(?:&&|;)")
GIT_C = re.compile(r"\bgit\s+-C\s+(\S+)")


def target(call: dict) -> tuple[Path | None, str]:
    """The path a call acts on, and its command ('' for an edit)."""
    args = call.get("tool_input") or {}
    base = Path(call.get("cwd") or os.getcwd())
    if call.get("tool_name") == "Bash":
        command = args.get("command", "")
        m = CD.match(command) or GIT_C.search(command)
        where = base / os.path.expanduser(m.group(1).strip("'\"")) if m else base
        return where.resolve(), command
    path = args.get("file_path") or args.get("notebook_path")
    return ((base / path).resolve() if path else None), ""


def checkout(path: Path) -> Path | None:
    """The nearest directory at or above `path` that ships routes."""
    return next((d for d in (path, *path.parents) if (d / ROUTES).is_file()), None)


def claim(session: str, root: Path, skill: str) -> bool:
    """True for exactly one caller a (session, checkout, skill): the repo's hook and the user's bootstrap run at once on
    one call, so the claim is an exclusive file creation, never a read then a write."""
    record = Path(tempfile.gettempdir()) / f"claude-skill-routes-{session}"
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


def main() -> int:
    try:
        call = json.load(sys.stdin)
        where, command = target(call)
    except (ValueError, OSError, TypeError):
        return 0
    root = checkout(where) if where else None
    if root is None:
        return 0
    try:
        routes = json.loads((root / ROUTES).read_text())["routes"]
    except (OSError, ValueError, KeyError):
        return 0
    edited = where.relative_to(root).as_posix() if not command and where != root else ""
    wanted = [r["skill"] for r in routes
              if (edited and any(fnmatch.fnmatch(edited, g) for g in r.get("edit", [])))
              or (command and any(s in command for s in r.get("bash", [])))]
    #: a context is an agent's: subagents share the session's id and carry their own `agent_id`
    session = re.sub(r"[^\w-]", "_", f"{call.get('session_id') or 'none'}-{call.get('agent_id') or 'main'}")
    texts = [f"=== skill `{s}` ===\n{text}" for s in dict.fromkeys(wanted)
             if (text := procedure(root, s)) and claim(session, root, s)]
    if not texts:
        return 0
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "additionalContext": f"Skill route (KERNEL_STANDARDS section 31): this action in the checkout {root} is "
                             "covered by the procedure below, from that checkout's own skills. Follow it.\n\n"
                             + "\n\n".join(texts)}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
