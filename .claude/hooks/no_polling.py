#!/usr/bin/env python3
"""Refuse a Bash command that waits by polling (KERNEL_STANDARDS section 14, `#s14`, and its exit-marker addendum,
`#s14-addendum`): a shell loop (`until`, `while`, `for`) whose text sleeps, lists processes or probes a pid, a `watch`
over such a command, and a bare `sleep` of 30 s or more. The wait the addendum allows is a bounded timer with early
exit, `timeout <=600 tail --pid=$PID -f /dev/null`, after which the agent reads the exit marker. A sleep under 30 s
outside a loop is allowed. Override: `POLL_OVERRIDE=<why>` in the command. Fired by the repo's dispatcher
(`.claude/hooks/repo_hooks.py`); docs: docs/internals/tools/dev.md#action-hooks."""
import json
import re
import sys

#: the addendum's wait: `timeout N tail --pid=PID -f /dev/null`, N seconds (or minutes) at most ten minutes
BOUNDED = re.compile(r"\btimeout\s+(\d+)([sm]?)\s+tail\s+--pid[= ]\S+\s+-f\s+/dev/null")
MAX_SECONDS = 600
#: a loop's keywords, in order, to find each loop's whole text (nested loops included)
LOOP_WORDS = re.compile(r"(?<![\w-])(until|while|for|done)(?![\w-])")
#: what makes a loop a poll: a sleep, a process list or probe, a pid probe
POLLS = re.compile(r"(?<![\w./-])(sleep|pgrep|pidof|ps)(?![\w-])|kill\s+-0|/proc/\d")
WATCH = re.compile(r"(?:^|[;&|(\n])\s*(?:\w+=\S+\s+)*watch\s")
SLEEP = re.compile(r"(?<![\w./-])sleep((?:\s+\d+(?:\.\d+)?[smhd]?)+|\s+infinity)")
UNITS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}
OVERRIDE = re.compile(r"\bPOLL_OVERRIDE=\S")


def sleeps(command: str) -> float:
    """The longest single `sleep` of the command, in seconds (a sleep's arguments add)."""
    longest = 0.0
    for found in SLEEP.finditer(command):
        if "infinity" in found.group(1):
            return float("inf")
        total = sum(float(n) * UNITS[u] for n, u in re.findall(r"(\d+(?:\.\d+)?)([smhd]?)", found.group(1)))
        longest = max(longest, total)
    return longest


def loops(command: str) -> list[str]:
    """The text of each shell loop, from its keyword to its `done`."""
    stack, found = [], []
    for word in LOOP_WORDS.finditer(command):
        if word.group(1) != "done":
            stack.append(word.start())
        elif stack:
            found.append(command[stack.pop():word.end()])
    return found


def problem(command: str) -> str:
    """Why the command is a poll, or an empty string."""
    for seconds, unit in BOUNDED.findall(command):
        if int(seconds) * UNITS[unit or "s"] > MAX_SECONDS:
            return f"`timeout {seconds}{unit} tail --pid` waits longer than the addendum's ten minutes"
    rest = BOUNDED.sub(" ", command)
    for body in loops(rest):
        if POLLS.search(body):
            return "a shell loop that sleeps, lists processes or probes a pid"
    if (m := WATCH.search(rest)) and POLLS.search(rest[m.end():]):
        return "`watch` over a sleep or a process list"
    if sleeps(rest) >= 30:
        return "a bare `sleep` of 30 s or more"
    return ""


try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
command = (data.get("tool_input") or {}).get("command") or ""
if not OVERRIDE.search(command) and (why := problem(command)):
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": (
            f"Never wait by polling (KERNEL_STANDARDS section 14): {why}. A long build runs as a BACKGROUND task "
            "that records its pid and appends an exit marker (`echo BUILD_EXIT=$? >> log`) as its last step; end the "
            "turn and the harness re-invokes you with the exit status. If you must wait inside a turn, use the "
            "bounded timer with early exit, `timeout 600 tail --pid=$PID -f /dev/null` (ten minutes at most), then "
            "read the marker and the build dir. Never `pgrep -f`, never an `until ... sleep` loop. A justified "
            "exception: put `POLL_OVERRIDE=<why>` in the command."),
    }}))
sys.exit(0)
