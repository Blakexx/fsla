#!/usr/bin/env python3
"""A checkout's own action hooks fire from any session (Blake, 2026-11-17: "detect the path, find the hooks, engage
them dynamically"). This dispatcher holds no rule. On a tool call it finds each checkout the call touches (an edited
file's path, a command's directory and each `cd` / `git -C` in it), reads that checkout's tracked manifest
`.claude/action-hooks.json` -- `{"PreToolUse": [{"matcher": "<regex of tool names>", "command": "<path in the
checkout>"}]}` -- and runs each matching hook with the call on stdin, in the checkout, `CLAUDE_PROJECT_DIR` set to it.
A hook's refusal is returned at once; hooks' added context is joined. Only checkouts under a root listed in
`~/.claude/repo_hooks_roots.json` (or `--roots FILE`) run code: a cloned stranger's manifest is never executed.
This file is the repo's copy: `bazel run //:init` installs it as the user's `~/.claude/hooks/repo_hooks.py`
(docs/internals/tools/dev.md#action-hooks)."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

MANIFEST = Path(".claude") / "action-hooks.json"
SPLIT = re.compile(r"\s*(?:&&|\|\||;|\||\n)\s*")
CHDIR = re.compile(r"^(?:cd|pushd)\s+(\S+)$")
GIT_C = re.compile(r"\bgit\s+-C\s+(\S+)")


def places(call: dict) -> list[Path]:
    """Where a tool call works: an edited file's path, or a command's directory and each `cd` / `git -C` in it."""
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


def checkouts(call: dict, roots: list[Path]) -> list[Path]:
    found = []
    for place in places(call):
        place = place.resolve()
        top = next((d for d in (place, *place.parents) if (d / MANIFEST).is_file()), None)
        if top and top not in found and any(top == r or r in top.parents for r in roots):
            found.append(top)
    return found


def main() -> int:
    try:
        call = json.loads(sys.stdin.read())
        listed = Path(sys.argv[sys.argv.index("--roots") + 1]) if "--roots" in sys.argv else (
            Path.home() / ".claude" / "repo_hooks_roots.json")
        roots = [Path(r).resolve() for r in json.loads(listed.read_text())]
    except Exception:
        return 0
    event, tool, context = call.get("hook_event_name") or "PreToolUse", str(call.get("tool_name") or ""), []
    for top in checkouts(call, roots):
        try:
            entries = json.loads((top / MANIFEST).read_text()).get(event, [])
        except Exception as ex:
            context.append(f"{top / MANIFEST} does not parse ({ex}): its hooks did not run")
            continue
        for entry in entries:
            if not re.fullmatch(entry.get("matcher") or ".*", tool):
                continue
            script = (top / entry["command"]).resolve()
            if top not in script.parents or not script.is_file():
                context.append(f"{top / MANIFEST}: `{entry['command']}` is not a file of the checkout")
                continue
            try:
                ran = subprocess.run([sys.executable, str(script)], input=json.dumps(call), capture_output=True,
                                     text=True, timeout=5, cwd=top, env={**os.environ, "CLAUDE_PROJECT_DIR": str(top)})
                said = json.loads(ran.stdout) if ran.stdout.strip().startswith("{") else {}
            except Exception as ex:
                context.append(f"{script} failed ({ex})")
                continue
            out = said.get("hookSpecificOutput") or {}
            if out.get("permissionDecision") == "deny":
                print(json.dumps(said))
                return 0
            if out.get("additionalContext"):
                context.append(out["additionalContext"])
    if context:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": event, "additionalContext": "\n".join(context)}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
