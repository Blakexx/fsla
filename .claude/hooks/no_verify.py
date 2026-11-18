#!/usr/bin/env python3
"""Refuse a git command that routes around the commit hooks (KERNEL_STANDARDS `#no-verify`: a commit bypasses the hooks
only with an explicit per-commit authorization; red hooks are fixed, never routed around): `git commit` or `git push`
with `--no-verify`, `git commit` with `-n` (alone or in a cluster such as `-nm`), and any `git -c core.hooksPath=...`.
Override: `NO_VERIFY_AUTHORIZED=<the owner's words>` in the command. Fired by the repo's dispatcher
(`.claude/hooks/repo_hooks.py`); docs: docs/internals/tools/dev.md#action-hooks."""
import json
import re
import sys

#: heredoc bodies are text, not commands
HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2(?=\s|$)", re.S)
OVERRIDE = re.compile(r"\bNO_VERIFY_AUTHORIZED=\S")
#: commit's options that take a value, as a letter of a short cluster and as long names
SHORT_VALUE = set("mFCct")
LONG_VALUE = {"--message", "--file", "--author", "--date", "--reuse-message", "--reedit-message", "--fixup", "--squash",
              "--template", "--cleanup", "--trailer"}
#: git's own options that take the next word
GIT_VALUE = {"-C", "--git-dir", "--work-tree", "--namespace", "--exec-path"}


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


def problem(command: str) -> str:
    """Why the command bypasses the hooks, or an empty string."""
    for words in commands(HEREDOC.sub(" ", command)):
        while words and re.fullmatch(r"\w+=.*", words[0]):
            words = words[1:]
        if not words or words[0].rsplit("/", 1)[-1] != "git":
            continue
        rest, i = words[1:], 0
        while i < len(rest) and rest[i].startswith("-"):
            option = rest[i]
            value = rest[i + 1] if option == "-c" and i + 1 < len(rest) else option[2:] if option.startswith("-c") else (
                option.split("=", 1)[1] if option.startswith("--config-env=") else "")
            if value.lower().startswith("core.hookspath") or option.lower().startswith("--config-env=core.hookspath"):
                return "`git -c core.hooksPath=...` swaps the repository's hooks for others"
            i += 2 if option in GIT_VALUE or option == "-c" else 1
        if i >= len(rest) or rest[i] not in ("commit", "push"):
            continue
        sub, flags, skip = rest[i], rest[i + 1:], False
        for flag in flags:
            if skip:
                skip = False
            elif flag == "--no-verify":
                return f"`git {sub} --no-verify` skips the hooks"
            elif flag in LONG_VALUE:
                skip = True
            elif sub == "commit" and re.fullmatch(r"-[A-Za-z]+", flag):
                for letter in flag[1:]:
                    if letter == "n":
                        return "`git commit -n` is `--no-verify`"
                    if letter in SHORT_VALUE:
                        skip = letter == flag[-1]
                        break
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
            f"A commit never bypasses the hooks (KERNEL_STANDARDS section 18 addendum, `#no-verify`): {why}. A red hook "
            "is fixed -- the hook, or the tree -- never routed around; run the failing check (`bazel test //:gate`), read "
            "it and fix what it names. Only the owner's explicit, per-commit authorization naming the pre-existing "
            "failure and the stage that fixes it allows a bypass: put `NO_VERIFY_AUTHORIZED=<the owner's words>` in the "
            "command."),
    }}))
sys.exit(0)
