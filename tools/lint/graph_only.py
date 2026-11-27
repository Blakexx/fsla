#!/usr/bin/env python3
"""THE GRAPH RUNS THE TOOLCHAIN, at the commit gate (KERNEL_STANDARDS#s14-graph; Blake, 2026-11-08: "every operation starts as a
bazel command, no old path survives"): a commit-gate lint refuses these steps in tracked code, its allowlist naming the rule or
the clause of KERNEL_STANDARDS#s14-graph for each.

The scan reads every tracked `*.py`, `*.sh`, `*.bzl`, `BUILD*` and script with a shebang (the hooks), and fails on a subprocess
or shell step that runs

- a toolkit or profiler tool: `nvcc`, `ptxas`, `cuobjdump`, `nvdisasm`, `cu++filt`, `ncu`, `nsys`, `compute-sanitizer`;
- a build tool: `cmake`, `make`, `ninja`;
- a package install: `pip install`, `uv pip install`, `uv sync`, `uv add`;
- `pytest`, as a subprocess (`pytest`, `python -m pytest`);
- or makes a temporary directory by hand (`tempfile.mkdtemp`, `tempfile.TemporaryDirectory`, `mktemp -d`), where a build or a
  compile would stage its files;

unless `ALLOWLIST` holds an entry for it. There is no baseline: an entry is one call site (the file, the function it is in and
the tool), and names what runs it -- the graph's rule whose action it is (`rule`), or the clause of KERNEL_STANDARDS#s14-graph that keeps it outside the
graph (`clause`) -- with the words of that rule or clause quoted. An entry that matches no site fails too, so the list is the sites.

A step is read from the syntax, never from the words: a Python call of `subprocess` (or `os.system`, `os.exec*`, `asyncio`'s
subprocesses, or a runner a tool names `run`, `_run`, `sh`, ...) whose command is a list or a shell string whose first word is
the tool (a name the file gave it, or a getter like `dev_config.cuda_bin("cuobjdump")`); a shell command word of a script, a
`genrule`'s `cmd` or an action's `command` in a build file. A tool named in a comment, a docstring or a string that no call
runs is text and passes.

    bazel test //:graph_only                       # the gate
    bazel run //tools/lint:graph_only              # the findings (and the allowlist's stale entries)
    bazel test //tools/lint:graph_only_fixtures    # a new subprocess `nvcc` fails it; the same in a comment or a test string passes
Docs: docs/internals/tools/lint.md#graph-only.
"""
from __future__ import annotations

import ast
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "graph_only"

TOOLS = ("nvcc", "ptxas", "cuobjdump", "nvdisasm", "cu++filt", "ncu", "nsys", "compute-sanitizer", "cmake", "make", "ninja")
#: a word of source that names a tool as a variable or attribute (`nvcc`, `cuppfilt`), compared as a bare lower-case identifier
_IDENT = {re.sub(r"\W", "", t): t for t in TOOLS} | {"cufilt": "cu++filt", "cuppfilt": "cu++filt", "computesanitizer": "compute-sanitizer",
                                                                         "sanitizer": "compute-sanitizer"}
#: the calls that hand a tool's path back (`dev_config.cuda_bin("cuobjdump")`)
GETTERS = {"tool", "cuda_bin", "_graph_bin", "_tool", "toolkit_tool"}
PREFIX = {"sudo", "env", "time", "nohup", "exec", "command", "nice", "stdbuf", "setsid", "timeout", "xargs"}
SUBPROCESS_ATTRS = {"run", "Popen", "call", "check_call", "check_output", "getoutput", "getstatusoutput", "system", "popen",
                    "execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe", "spawnl", "spawnv",
                    "spawnlp", "spawnvp", "create_subprocess_exec", "create_subprocess_shell", "run_shell"}
SUBPROCESS_ROOTS = {"subprocess", "os", "asyncio", "sp", "ctx", "actions"}
#: a tool's own runner: `run(*argv)`, `_run(cmd, timeout)`, `sh(...)`
RUNNER = re.compile(r"^_?(run|sh|call|capture|execute|exec|popen|check_output|run_tool)\w*$")
TEMP_DIRS = {"mkdtemp", "TemporaryDirectory"}
#: a temporary directory is a build directory by its name, or by the file's running a tool beside it
BUILD_DIR = re.compile(r"build|compile|nvcc|cubin|stag|scratch")
HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2(?=\s|$)", re.S)


@dataclass(frozen=True)
class Site:
    path: str
    where: str
    tool: str
    line: int

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.path, self.where, self.tool)


def base(word: str) -> str:
    return word.rsplit("/", 1)[-1]


def shell_commands(text: str) -> list[list[str]]:
    """Each simple command of a shell text as its words (quotes joined, `&& || ; | ( )` and newlines end one, `#` starts a comment)."""
    out: list[list[str]] = []
    for line in HEREDOC.sub(" ", text).replace("\\\n", " ").splitlines():
        lex = shlex.shlex(line, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        lex.commenters = "#"
        words: list[str] = []
        try:
            for token in lex:
                if token and set(token) <= set("();&|<>") and not token.startswith(("<", ">")):
                    if words:
                        out.append(words)
                    words = []
                elif token and token[0] in "<>" and set(token) <= set("<>&"):
                    words.append("\0redirect")
                else:
                    words.append(token)
        except ValueError:
            words = [w for w in line.split() if not w.startswith("#")]
        if words:
            out.append(words)
    return out


def command_tools(words: list[str], depth: int = 0) -> list[str]:
    """The tools a command (its words, the first the command word) runs: the toolkit and build tools by name or by a path ending in
    one, `pytest`, an install, a shell's `-c` text (read as a shell text), a `bazel` command none."""
    words = [w for w in words if w != "\0redirect"]
    while words and (re.fullmatch(r"\w+=.*", words[0]) or base(words[0]) in PREFIX):
        words = words[2:] if base(words[0]) == "timeout" and len(words) > 1 else words[1:]
    if not words:
        return []
    name = base(words[0])
    rest = [w for w in words[1:] if not w.startswith("-")]
    if name in ("bazel", "bazelisk"):
        return []
    if name in TOOLS:
        return [name]
    if name in ("pytest", "py.test"):
        return ["pytest"]
    if re.fullmatch(r"pip[\d.]*", name) and rest[:1] == ["install"]:
        return ["pip install"]
    if name == "uv" and (rest[:2] == ["pip", "install"] or rest[:1] in (["sync"], ["add"])):
        return ["uv install"]
    if (re.fullmatch(r"python[\d.]*", name) or name.startswith("$") or name == "?") and "-m" in words:
        after = [w for w in words[words.index("-m") + 1:] if not w.startswith("-")]
        if after[:1] == ["pytest"]:
            return ["pytest"]
        if after[:2] == ["pip", "install"]:
            return ["pip install"]
    if name in ("bash", "sh", "zsh") and "-c" in words and depth < 3 and words.index("-c") + 1 < len(words):
        return shell_tools(words[words.index("-c") + 1], depth + 1)
    if name in ("mktemp",) and "-d" in words:
        return ["mktemp -d"]
    return []


def shell_tools(text: str, depth: int = 0) -> list[str]:
    found: list[str] = []
    for words in shell_commands(text):
        found += command_tools(words, depth)
    return found


# ---- Python ----

class Scan(ast.NodeVisitor):
    def __init__(self, rel: str) -> None:
        self.rel = rel
        self.sites: list[Site] = []
        self.scope: list[str] = []
        self.assigned: list[dict[str, ast.AST]] = [{}]
        self.imported_runners: set[str] = set()
        #: temporary directories of a file with a toolchain step beside them: a compile's or a disassembly's staging, found at the end
        self.temp_dirs: list[ast.Call] = []

    def where(self) -> str:
        return ".".join(self.scope) or "<module>"

    @staticmethod
    def enclosing(tree: ast.AST, target: ast.AST) -> str:
        """The dotted names of the functions and classes around `target`."""
        def walk(node: ast.AST, names: list[str]) -> list[str] | None:
            for child in ast.iter_child_nodes(node):
                inner = names + [child.name] if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else names
                if child is target:
                    return inner
                found = walk(child, inner)
                if found is not None:
                    return found
            return None

        return ".".join(walk(tree, []) or []) or "<module>"

    def add(self, tool: str, node: ast.AST) -> None:
        self.sites.append(Site(self.rel, self.where(), tool, node.lineno))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module in ("subprocess", "os", "asyncio"):
            self.imported_runners |= {a.asname or a.name for a in node.names if a.name in SUBPROCESS_ATTRS}

    def _scoped(self, node: ast.AST, name: str) -> None:
        self.scope.append(name)
        self.assigned.append({})
        self.generic_visit(node)
        self.assigned.pop()
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scoped(node, node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scoped(node, node.name)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assigned[-1][target.id] = node.value
        self.generic_visit(node)

    def resolve(self, node: ast.AST) -> ast.AST:
        for _ in range(4):
            if not isinstance(node, ast.Name):
                break
            found = next((frame[node.id] for frame in reversed(self.assigned) if node.id in frame), None)
            if found is None:
                break
            node = found
        return node

    def words(self, node: ast.AST, depth: int = 0) -> list[str]:
        """A command's words as far as the syntax gives them: a literal, an f-string's literal head, a getter's name, an identifier
        (`$nvcc`), a list's elements, a concatenation; a word the syntax does not give is `?`."""
        node = self.resolve(node)
        if isinstance(node, (ast.List, ast.Tuple)):
            out: list[str] = []
            for element in node.elts:
                if isinstance(element, ast.Starred):
                    out += self.words(element.value, depth + 1) if depth < 3 else ["?"]
                else:
                    out += self.one(element)
            return out
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return self.words(node.left, depth + 1) + self.words(node.right, depth + 1)
        return self.one(node)

    def one(self, node: ast.AST) -> list[str]:
        node = self.resolve(node)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]
        if isinstance(node, ast.JoinedStr):
            return ["".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)]
        if isinstance(node, ast.Call):
            callee = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if callee in GETTERS and node.args and isinstance(node.args[0], ast.Constant):
                return [str(node.args[0].value)]
            if callee in ("str", "Path", "fspath", "abspath", "absolute", "resolve") and node.args:
                return self.one(node.args[0])
            if re.sub(r"\W", "", callee.lower()) in _IDENT:
                return ["$" + callee]
            return ["?"]
        if isinstance(node, ast.IfExp):
            return self.one(node.body)
        if isinstance(node, ast.Subscript):
            return self.one(node.value)
        if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
            return self.one(node.generators[0].iter)
        if isinstance(node, ast.Attribute):
            return ["$" + node.attr]
        if isinstance(node, ast.Name):
            return ["$" + node.id]
        return ["?"]

    def classify(self, node: ast.AST, strings_are_shell: bool) -> list[str]:
        node = self.resolve(node)
        if strings_are_shell and isinstance(node, (ast.Constant, ast.JoinedStr)):
            (text,) = self.one(node)
            return shell_tools(text)
        words = [self.ident(w) for w in self.words(node)]
        found = command_tools(words) if words and (words[0] != "?" or "-m" in words) else []
        #: `["bash", "-c", "<text>"]` and a shell string in a list are read as the shell text they are
        return found

    @staticmethod
    def ident(word: str) -> str:
        if word.startswith("$"):
            return _IDENT.get(re.sub(r"\W", "", word[1:].lower()), word)
        return word

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        attr = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        root = func
        while isinstance(root, ast.Attribute):
            root = root.value
        root_name = getattr(root, "id", "")
        runs = (attr in SUBPROCESS_ATTRS and (root_name in SUBPROCESS_ROOTS or attr in self.imported_runners)) or (
            attr in TEMP_DIRS)
        if attr in TEMP_DIRS:
            if BUILD_DIR.search(ast.unparse(node)):
                self.add("temporary directory", node)
            else:
                self.temp_dirs.append(node)
        elif runs and node.args:
            shell = attr in ("system", "popen", "getoutput", "getstatusoutput", "create_subprocess_shell", "run_shell") or any(
                kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value for kw in node.keywords)
            for tool in self.classify(node.args[0], shell):
                self.add(tool, node)
        elif runs and attr in ("run_shell", "run"):
            for kw in node.keywords:
                if kw.arg in ("command", "executable"):
                    for tool in self.classify(kw.value, kw.arg == "command"):
                        self.add(tool, node)
        elif RUNNER.match(attr or "") and node.args:
            #: a tool's own runner: its positional words are the command, or its first is a list of them
            first = self.resolve(node.args[0])
            words = self.words(first) if isinstance(first, (ast.List, ast.Tuple)) else [w for a in node.args for w in self.one(a)]
            words = [self.ident(w) for w in words]
            if words and (words[0] != "?" or "-m" in words):
                for tool in command_tools(words):
                    self.add(tool, node)
        self.generic_visit(node)


def scan_python(text: str, rel: str) -> list[Site]:
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        raise SystemExit(f"graph_only: {rel} does not parse: {error}")
    scan = Scan(rel)
    scan.visit(tree)
    if scan.sites:
        scan.scope = []
        for node in scan.temp_dirs:
            scan.sites.append(Site(rel, scan.enclosing(tree, node), "temporary directory", node.lineno))
    return scan.sites


# ---- shell scripts, hooks, build files ----

def scan_shell(text: str, rel: str) -> list[Site]:
    return [Site(rel, "<script>", tool, 0) for tool in dict.fromkeys(shell_tools(text))]


def scan_build(text: str, rel: str) -> list[Site]:
    """A build file or a `.bzl`: Starlark is Python syntax, so a `cmd`/`command` string of a rule or action is a shell text and a
    `ctx.actions.run`'s `executable` a command, read as a Python step; the file's script strings are read as shell."""
    sites = scan_python(text, rel)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return sites
    seen = {(s.where, s.tool) for s in sites}
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg in ("cmd", "command", "cmd_bash") and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for tool in shell_tools(node.value.value):
                if ("<build>", tool) not in seen:
                    seen.add(("<build>", tool))
                    sites.append(Site(rel, "<build>", tool, node.value.lineno))
    return sites


def kind(path: Path) -> str:
    name = path.name
    if path.suffix == ".py":
        return "python"
    if path.suffix == ".bzl" or name.startswith("BUILD") or name in ("WORKSPACE", "MODULE.bazel"):
        return "build"
    if path.suffix in (".sh", ".bash"):
        return "shell"
    try:
        with path.open("rb") as handle:
            first = handle.readline(200)
    except OSError:
        return ""
    if first.startswith(b"#!"):
        return "python" if b"python" in first else "shell"
    return ""


def scan_file(path: Path, rel: str) -> list[Site]:
    what = kind(path)
    if not what:
        return []
    text = path.read_text(errors="replace")
    return {"python": scan_python, "shell": scan_shell, "build": scan_build}[what](text, rel)


def tracked() -> list[str]:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True, text=True)
    return [p for p in out.stdout.split("\0") if p]


def scan_tree(files: list[str]) -> list[Site]:
    sites: list[Site] = []
    for rel in files:
        path = ROOT / rel
        if path.is_symlink() or not path.is_file() or rel.startswith("tools/lint/fixtures/") or rel == "tools/lint/graph_only.py":
            continue
        sites += scan_file(path, rel)
    return sites


# ---- the allowlist ----

def rule(label: str, quote: str) -> tuple[str, str, str]:
    """An entry kept by the graph's rule: the label of the target whose action runs it, and its words."""
    return ("rule", label, quote)


def clause(letter: str, quote: str) -> tuple[str, str, str]:
    """An entry kept outside the graph by a clause of KERNEL_STANDARDS#s14-graph: its letter, and its words."""
    return ("clause", f"docs/KERNEL_STANDARDS.md#s14-graph ({letter})", quote)


_CUDA_UNIT = rule("bazel/cuda.bzl `cuda_unit`, whose action is this script", "THE GRAPH'S CUDA COMPILE ACTION (`bazel/cuda.bzl`'s `cuda_unit`)")
_LAB = rule("bazel/lab.bzl `lab_experiment`, whose facts action is this script", "THE LAB'S FACTS, read in the graph (bazel/lab.bzl's `lab_experiment`")
_SANITIZER = clause("c", "THE SANITIZER LANE'S PAIR: the lane is the graph's target (`//tests:sanitizer`, which hands the tool the toolkit's own")
_RATIFY = rule("//:ratify (`ratify_run`) and the census tests (`census_test`), on the graph's ptxas (`ratify.cuda_bin`)", "ONE toolchain location for ptxas and the readers, the build's own")
_ORACLE_TIER = rule("//measure:oracle_tier (`instrument`), run on the device by its store target `//measure:store_oracle_tier`",
                    "THE ORACLE TIER ON THE DEVICE (`oracle_tier.py`)")
_PROFILER = clause("a", "THE PROFILER, on the host that grants its counters")

#: ONE ENTRY A CALL SITE -- (file, the function it is in, the tool) -- NAMING WHAT RUNS IT, no baseline: an entry that matches no site fails
ALLOWLIST: dict[tuple[str, str, str], tuple[str, str, str]] = {
    ("bazel/cuda_compile.py", "main", "nvcc"): _CUDA_UNIT,
    ("bazel/cuda_compile.py", "main", "temporary directory"): _CUDA_UNIT,
    ("tools/lab_facts.py", "main", "cuobjdump"): _LAB,
    ("tools/lab_facts.py", "main", "nvdisasm"): _LAB,
    ("tools/lab_facts.py", "main", "nvcc"): _LAB,
    ("tools/lab_facts.py", "main", "temporary directory"): _LAB,
    ("tools/sass_bodies.py", "bodies", "cuobjdump"): rule("//tools:sass_bodies, on the graph's cuobjdump (`dev_config.cuda_bin`)", "PER-INSTANTIATION SASS BODY HASHES -- the codegen-equivalence instrument"),
    ("tools/lint/primitives_census.py", "disassemble", "cuobjdump"): rule("//tools/lint:primitives_census (`tool_test`) over the object `cuda_unit` compiled, with the graph's cuobjdump", "the census is a test over the object's SASS, read with the graph's cuobjdump"),
    ("tools/ratify.py", "ptxas_version", "ptxas"): _RATIFY,
    ("tools/sanitize_oracle.py", "main", "compute-sanitizer"): _SANITIZER,
    ("tools/sanitize_oracle.py", "run_one", "compute-sanitizer"): _SANITIZER,
    ("tools/sanitize_oracle.py", "run_absent", "pytest"): _SANITIZER,
    ("measure/oracle_tier.py", "tier", "pytest"): _ORACLE_TIER,
    ("measure/oracle_tier.py", "tier", "temporary directory"): _ORACLE_TIER,
    ("tools/lab.py", "counters_of", "ncu"): _PROFILER,
    ("tools/lab.py", "ncu_counters", "ncu"): _PROFILER,
    ("tools/lab.py", "ledger", "ncu"): _PROFILER,
    ("tools/lab.py", "ledger_dumps", "ncu"): _PROFILER,
    ("tools/wheels.py", "check", "pip install"): rule("//:wheels (`fsla_wheels`), whose install check this is (docs/build.md#wheels)", "**The wheels package the graph's library.** `bazel run //:wheels` refuses a tree with changes"),
    ("tools/wheels.py", "check", "temporary directory"): rule("//:wheels (`fsla_wheels`), whose install check this is (docs/build.md#wheels)", "**The wheels package the graph's library.** `bazel run //:wheels` refuses a tree with changes"),
}


def judge(sites: list[Site], allowlist) -> list[str]:
    problems = []
    for site in sites:
        if site.key not in allowlist:
            problems.append(f"{site.path}:{site.line}: `{site.tool}` runs in `{site.where}` outside the graph: move it into a graph rule "
                            f"(docs/KERNEL_STANDARDS.md#s14-graph) or, where a rule or a clause keeps it out, add an ALLOWLIST entry "
                            f"citing the rule or the clause of docs/KERNEL_STANDARDS.md#s14-graph that keeps it out (tools/lint/graph_only.py)")
    seen = {site.key for site in sites}
    for key in allowlist:
        if key not in seen:
            problems.append(f"{key[0]}:0: stale ALLOWLIST entry {key}: no such site now; delete it")
    return problems


def self_test() -> int:
    failures = []
    bad = {"bad.py": {("nvcc", "compile"), ("ptxas", "shell_form"), ("uv install", "install"), ("pytest", "child"),
                      ("cuobjdump", "getter"), ("temporary directory", "scratch"), ("ninja", "via_bash")},
           "bad.sh": {("nvdisasm", "<script>"), ("make", "<script>"), ("mktemp -d", "<script>")},
           "BUILD.fixture": {("nvcc", "<build>")}}
    for name, expected in bad.items():
        found = {(s.tool, s.where) for s in scan_file(FIXTURES / name, name)}
        if found != expected:
            failures.append(f"{name}: found {sorted(found)}, expected {sorted(expected)}")
    for name in ("ok.py", "ok.sh"):
        found = scan_file(FIXTURES / name, name)
        if found:
            failures.append(f"{name}: a tool named as text was found: {found}")
    fixture_site = Site("bad.py", "compile", "nvcc", 1)
    if not judge([fixture_site], {}) or judge([fixture_site], {fixture_site.key: ("x", "y", "z")}):
        failures.append("an unlisted site must fail and a listed one pass")
    if not judge([], {("gone.py", "f", "nvcc"): ("x", "y", "z")}):
        failures.append("a stale entry must fail")
    for failure in failures:
        print(f"SELF-TEST FAILED: {failure}", file=sys.stderr)
    if not failures:
        print("graph_only.py --test-fixtures: PASS")
    return 1 if failures else 0


def main() -> int:
    if "--test-fixtures" in sys.argv:
        return self_test()
    sites = scan_tree(tracked())
    if "--sites" in sys.argv:
        for site in sorted({(s.path, s.where, s.tool) for s in sites}):
            print(site)
        return 0
    problems = judge(sites, ALLOWLIST)
    for problem in problems:
        print(f"  {problem}")
    print(f"graph_only: {len(problems)} problem(s); {len(ALLOWLIST)} allowlisted site(s) of {len(sites)} found")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
