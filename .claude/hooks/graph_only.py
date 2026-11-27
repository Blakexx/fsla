#!/usr/bin/env python3
"""Refuse a Bash command that runs a toolchain or install command by hand in a checkout; the build graph runs them
(Blake, 2026-11-21, "what runs outside the graph moves into it"): `nvcc`, `ptxas`, `cuobjdump`, `nvdisasm`, `ncu`, `nsys`,
`compute-sanitizer`, `cmake`, `make`, `ninja`, and `pip install` (also `uv pip install`, `python -m pip install`), as a command
word, by name or by a path ending in one; and `git worktree add|remove`, which the machine's worktree tool does under its cap
(Blake, 2026-11-27). A word inside a quoted string, a heredoc body or an argument (`git log | grep nvcc`,
`echo "nvcc is a compiler"`) is text, not a command, and a `bazel` command passes whatever its arguments (`bazel run
//measure/lab:x.run -- sass`). The refusal names the target that does it. Override: `GRAPH_OVERRIDE="<why>"` in the command.
Fired by the repo's dispatcher (`.claude/hooks/repo_hooks.py`); docs: docs/internals/tools/lint.md#action-hooks."""
import json
import re
import sys

#: heredoc bodies are text, not commands
HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2(?=\s|$)", re.S)
OVERRIDE = re.compile(r"\bGRAPH_OVERRIDE=\S")
#: a toolchain command word and the graph target that runs it
TOOLS = {
    "nvcc": "`bazel run //:library` (the shipped build; `--//:archs=<arch>` for one architecture) or, for an experiment, "
            "`bazel run //measure/lab:<name>.run`",
    "ptxas": "`bazel run //measure/lab:<name>.run -- sass` or `bazel build //csrc/fsla:<unit>_sass` (a unit's SASS as text)",
    "cuobjdump": "`bazel run //measure/lab:<name>.run -- sass [REGEX] [--full]`, `bazel build //csrc/fsla:<unit>_sass` (a unit's SASS as text) "
                 "or `bazel build //:static_manifest` (the library's, every rule's count at every kernel)",
    "nvdisasm": "`bazel run //measure/lab:<name>.run -- sass [REGEX] [--full]`, `bazel build //csrc/fsla:<unit>_sass` (a unit's SASS as text) "
                 "or `bazel build //:static_manifest` (the library's, every rule's count at every kernel)",
    "ncu": "`bazel run //measure/lab:<name>.run -- ncu --metric M` (a lab row)",
    "nsys": "`bazel run //measure/lab:<name>.run` (a lab row) or `bazel run //measure:session_entmax` (a declared timing)",
    "compute-sanitizer": "`bazel test //tests:sanitizer` (the sanitizer lane, which passes the pinned toolkit's own)",
    "cmake": "`bazel run //:library` (the graph builds the extension)",
    "make": "`bazel run //:library` (the graph builds the extension)",
    "ninja": "`bazel run //:library` (the graph builds the extension)",
}
PIP = "`bazel run //:dev_env` (this checkout's venv, synced from `requirements.lock`)"
WORKTREE = ("`bazel --worktree add <path> <sha>` or `bazel --worktree remove [<path>]` (fsla-devtools' worktree tool, under the "
            "machine's cap on live worktrees; here `bazel run //:worktree -- <name> <sha>` also sets one up)")
#: words a command may carry before its own word
PREFIX = {"sudo", "env", "time", "nohup", "exec", "command", "nice", "stdbuf", "setsid"}


def commands(text: str) -> list[list[str]]:
    """Each simple command of a shell text as its words: quotes joined, an unquoted `&& || ; | & ( )` or newline ends one."""
    out, words, word, quote, started, i = [], [], "", "", False, 0

    def end_word() -> None:
        nonlocal word, started
        if started:
            words.append(word)
        word, started = "", False

    def end_command() -> None:
        nonlocal words
        end_word()
        if words:
            out.append(words)
        words = []

    while i < len(text):
        c = text[i]
        if quote:
            if c == quote:
                quote = ""
            elif c == "\\" and quote == '"' and i + 1 < len(text):
                i += 1
                word += text[i]
            else:
                word += c
        elif c in "'\"":
            quote, started = c, True
        elif c == "\\" and i + 1 < len(text):
            i += 1
            word, started = word + text[i], True
        elif c in " \t\r":
            end_word()
        elif c in "\n;|&()":
            end_command()
        else:
            word, started = word + c, True
        i += 1
    end_command()
    return out


def base(word: str) -> str:
    return word.rsplit("/", 1)[-1]


def pip_install(words: list[str]) -> bool:
    """`pip install`, `pip3 install`, `uv pip install`, `python -m pip install`, with any options between."""
    name = base(words[0])
    rest = [w for w in words[1:] if not w.startswith("-")]
    if re.fullmatch(r"pip[\d.]*", name):
        return rest[:1] == ["install"]
    if name == "uv":
        return rest[:2] == ["pip", "install"]
    if re.fullmatch(r"python[\d.]*", name) and "-m" in words:
        after = [w for w in words[words.index("-m") + 1:] if not w.startswith("-")]
        return after[:2] == ["pip", "install"]
    return False


def git_worktree(words: list[str]) -> bool:
    """`git worktree add` or `git worktree remove`, with git's own options (`-C DIR`, `-c k=v`) before the subcommand."""
    if base(words[0]) != "git":
        return False
    rest, i = [], 1
    while i < len(words):
        if words[i] in ("-C", "-c"):
            i += 2
            continue
        if not words[i].startswith("-"):
            rest.append(words[i])
        i += 1
    return rest[:1] == ["worktree"] and rest[1:2] in (["add"], ["remove"])


def problem(command: str) -> str:
    """The redirect for the first toolchain or install command the text runs, or an empty string."""
    for words in commands(HEREDOC.sub(" ", command)):
        while words and (re.fullmatch(r"\w+=.*", words[0]) or base(words[0]) in PREFIX or words[0] == "timeout"):
            words = words[2:] if words[0] == "timeout" and len(words) > 1 else words[1:]
        if not words:
            continue
        name = base(words[0])
        if name in ("bazel", "bazelisk"):
            continue
        if name in TOOLS:
            return f"`{name}` runs by hand; run {TOOLS[name]}"
        if pip_install(words):
            return f"an install by hand; run {PIP}"
        if git_worktree(words):
            return f"a worktree made or removed by hand, past the machine's cap; run {WORKTREE}"
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
            f"The graph runs the toolchain, the installs and the worktrees, never a command typed in a checkout: {why}. A justified "
            "exception (a probe the graph has no target for): put `GRAPH_OVERRIDE=\"<why>\"` in the command."),
    }}))
sys.exit(0)
