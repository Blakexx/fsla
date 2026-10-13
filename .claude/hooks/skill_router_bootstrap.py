#!/usr/bin/env python3
"""The skill-router bootstrap (user-level, generic): find the checkout a tool call touches -- the edited file's, or the
directory each simple command runs in (a `cd`, `pushd` or `git -C` before it, else the session's) -- walk up to the
nearest `.claude/hooks/route_skills.py`, and hand that router the call. Any repo that ships the router is routed from
any session; the repo owns its routes, its router and its skills (RoLA: KERNEL_STANDARDS section 31). `--forget`, run
at SessionStart after a compaction or a resume, drops the main agent's record under the router's naming."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SPLIT = re.compile(r"\s*(?:&&|\|\||;|\||\n)\s*")
CHDIR = re.compile(r"^(?:cd|pushd)\s+(\S+)$")
GIT_C = re.compile(r"\bgit\s+-C\s+(\S+)")


def places(call: dict) -> list[Path]:
    args = call.get("tool_input") or {}
    here = Path(call.get("cwd") or os.getcwd())
    if call.get("tool_name") != "Bash":
        path = args.get("file_path") or args.get("notebook_path")
        return [here / path] if path else []
    out = [here]
    for part in SPLIT.split(args.get("command", "")):
        part = part.strip().lstrip("( ")
        if m := CHDIR.match(part):
            here = here / os.path.expanduser(m.group(1).strip("'\""))
            out.append(here)
        elif m := GIT_C.search(part):
            out.append(here / os.path.expanduser(m.group(1).strip("'\"")))
    return out


def main() -> int:
    raw = sys.stdin.read()
    try:
        call = json.loads(raw)
        if "--forget" in sys.argv:
            name = re.sub(r"[^\w-]", "_", f"{call.get('session_id') or 'none'}-main")
            shutil.rmtree(Path(tempfile.gettempdir()) / f"claude-skill-routes-{name}", ignore_errors=True)
            return 0
        for place in reversed(places(call)):
            place = place.resolve()
            router = next((r for d in (place, *place.parents)
                           if (r := d / ".claude" / "hooks" / "route_skills.py").is_file()), None)
            if router is not None:
                out = subprocess.run([sys.executable, str(router)], input=raw, capture_output=True, text=True, timeout=4)
                sys.stdout.write(out.stdout)
                return 0
    except Exception:  # a hook never blocks a call
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
