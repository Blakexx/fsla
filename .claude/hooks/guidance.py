#!/usr/bin/env python3
"""The checker's guidance, at the edit (Blake, 2026-11-20): when an edit's new text lands in a kernel source (`csrc/**/*.cu*`,
`measure/lab/**/*.cu*`), inject the lines of the `checker` section of `docs/rules.json` whose `triggers` -- regexes over source -- that text
matches, each the rule of warpline's checker (the entry's key) and rola's `guidance` for it: the atom or practice, the standards
anchor, the lab page. Advisory context, never a refusal; a malformed table or an unreadable file injects nothing. The new text
is an Edit's `new_string`, a MultiEdit's every `new_string` and a Write's `content`. Fired by the repo's dispatcher
(`.claude/hooks/repo_hooks.py`); docs: docs/internals/tools/lint.md#action-hooks."""
import fnmatch
import json
import re
import sys
from pathlib import Path

KERNEL_SOURCES = ("csrc/*.cu", "csrc/*.cuh", "measure/lab/*.cu", "measure/lab/*.cuh")


def new_text(args: dict) -> str:
    """What the call writes: an Edit's `new_string`, a MultiEdit's `new_string`s, a Write's `content`."""
    parts = [args.get("new_string"), args.get("content"), *(e.get("new_string") for e in args.get("edits") or [])]
    return "\n".join(p for p in parts if isinstance(p, str))


def kernel_source(path: str, root: Path) -> bool:
    """Whether `path` is a kernel source of the checkout `root` (`fnmatch`'s `*` crosses directories, so `**` is one `*`)."""
    try:
        relative = Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return False
    return any(fnmatch.fnmatchcase(relative, pattern) for pattern in KERNEL_SOURCES)


def fired(entries: dict, text: str) -> list[str]:
    """The guidance lines of the entries one of whose triggers matches `text`, in the section's order; an entry names the rows it serves."""
    lines = []
    for rule, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        for trigger in entry.get("triggers") or []:
            try:
                hit = re.search(trigger, text, re.M)
            except re.error:
                continue
            if hit:
                served = f" ({', '.join(entry.get('rules') or [])})" if entry.get("rules") else ""
                lines.append(f"- {rule}{served}: {entry['guidance']}")
                break
    return lines


def main() -> int:
    try:
        call = json.loads(sys.stdin.read())
        args = call.get("tool_input") or {}
        root = Path(__file__).resolve().parents[2]
        path = str(args.get("file_path") or "")
        if not path or not kernel_source(path, root):
            return 0
        lines = fired(json.loads((root / "docs" / "rules.json").read_text())["checker"], new_text(args))
    except Exception:
        return 0
    if lines:
        context = ("This edit adds kernel code the checker holds to rules; rola's guidance (docs/rules.json, `checker` section):\n"
                   + "\n".join(lines))
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": context}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
