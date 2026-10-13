#!/usr/bin/env python3
"""The skill-router bootstrap (user-level, generic): find the checkout a tool call touches -- the edited file's, or the
directory a command runs in (a leading `cd <dir>`, `git -C <dir>`, else the session's) -- walk up to the nearest
`.claude/hooks/route_skills.py`, and hand that router the call. Any repo that ships that router is routed from any
session; the repo owns its routes, its router and its skills (RoLA: KERNEL_STANDARDS section 31)."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

CD = re.compile(r"^\s*cd\s+(\S+)\s*(?:&&|;)")
GIT_C = re.compile(r"\bgit\s+-C\s+(\S+)")


def main() -> int:
    raw = sys.stdin.read()
    try:
        call = json.loads(raw)
        args = call.get("tool_input") or {}
        base = Path(call.get("cwd") or os.getcwd())
        if call.get("tool_name") == "Bash":
            m = CD.match(args.get("command", "")) or GIT_C.search(args.get("command", ""))
            where = base / os.path.expanduser(m.group(1).strip("'\"")) if m else base
        elif args.get("file_path") or args.get("notebook_path"):
            where = base / (args.get("file_path") or args.get("notebook_path"))
        else:
            return 0
        where = where.resolve()
        router = next((r for d in (where, *where.parents)
                       if (r := d / ".claude" / "hooks" / "route_skills.py").is_file()), None)
        if router is not None:
            out = subprocess.run([sys.executable, str(router)], input=raw, capture_output=True, text=True, timeout=4)
            sys.stdout.write(out.stdout)
    except Exception:  # a hook never blocks a call
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
