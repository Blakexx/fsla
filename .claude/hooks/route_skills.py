#!/usr/bin/env python3
# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE SKILL ROUTER (KERNEL_STANDARDS section 31), a PreToolUse hook: when a tool call edits a path or runs a command
that `.claude/skill-routes.json` maps to a repo skill, it puts that skill's name in the session's context -- once a
session a skill -- so the procedure is loaded by the action, not remembered. It never blocks a call."""
from __future__ import annotations

import fnmatch
import json
import sys
import tempfile
from pathlib import Path

#: the checkout this hook ships in: `<root>/.claude/hooks/route_skills.py`
ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    try:
        call = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    root = ROOT
    try:
        routes = json.loads((root / ".claude" / "skill-routes.json").read_text())["routes"]
    except (OSError, ValueError, KeyError):
        return 0
    tool, args = call.get("tool_name", ""), call.get("tool_input") or {}
    path = args.get("file_path") or args.get("notebook_path") or ""
    if path:
        try:
            path = Path(path).resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            path = ""
    command = args.get("command", "") if tool == "Bash" else ""
    wanted = [r["skill"] for r in routes
              if (path and any(fnmatch.fnmatch(path, g) for g in r.get("edit", [])))
              or (command and any(s in command for s in r.get("bash", [])))]
    if not wanted:
        return 0
    seen_file = Path(tempfile.gettempdir()) / f"claude-skill-routes-{call.get('session_id', 'none')}.json"
    try:
        seen = set(json.loads(seen_file.read_text()))
    except (OSError, ValueError):
        seen = set()
    fresh = [s for s in dict.fromkeys(wanted) if s not in seen]
    if not fresh:
        return 0
    seen_file.write_text(json.dumps(sorted(seen | set(fresh))))
    names = ", ".join(f"`{s}`" for s in fresh)
    which = ("the repo skill", "it") if len(fresh) == 1 else ("the repo skills", "them")
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "additionalContext": f"Skill route (KERNEL_STANDARDS section 31): this action is covered by {which[0]} "
                             f"{names}. Load {which[1]} with the Skill tool and follow {which[1]}, unless already "
                             "loaded."}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
