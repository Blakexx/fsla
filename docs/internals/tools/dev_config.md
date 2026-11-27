# `tools/dev_config.py` — every machine-dependent value, in one directory

KERNEL_STANDARDS §23. A value that depends on the machine lives in the dev config directory and is read through
this loader. Examples are where the CUDA toolkit is, how many compile slots the host has, how the SM clock is
locked, and where the measurements record is. The tree never reads a setting from the process environment, and
never spells a machine path in a tool.

**The directory** is `~/.config/fsla/`, or the directory the `FSLA_DEV_CONFIG` variable names (a test). It holds one JSON file per section. A missing file or key takes its declared default. An undeclared file
or key is refused, so a typo fails instead of silently taking the default.

**The reader is fsla-devtools'** (`fsla_devtools.config`), and so are the `host` and `clock` sections: the machine's
locks (`fsla_devtools.locks`) read them, for every repository and container on the host. This loader declares fsla's
own sections beside them and, owning the directory, refuses a file no section declares. fsla-devtools is a development
dependency, pinned in `MODULE.bazel` (its module's `git_override`): the graph's environment carries it from the module's
source, and `bazel run //:init` links the `workspace.devtools` checkout into every venv on the machine.

| file | keys |
|---|---|
| `toolchain.json` | `cuda_home` (a CUDA toolkit older checkouts' files name; this tree uses the graph's, `@cuda_tools` pinned by fsla-devtools' `pins/cuda_toolkit_pin.json`, `cuda_home()`), `ncu` (a Nsight Compute CLI older checkouts' files name; this tree finds the pinned one by its pin, `ncu()`, which refuses until init installed it), `sccache`, `sccache_enabled` and `mold` (older checkouts' compiler cache and linker; this tree reads none of them, its graph links with its own mold), `clang_tidy` (the host's clang-tidy, which checkouts before the build graph's unit lints install and read; this tree reads none of it), `bazel` (where the bootstrap links the pinned bazelisk on PATH) |
| `host.json` | `nvcc_threads`, `build_jobs`, `budget_slots`, `lock_dir`, `gpu_lock`, `gpu_shared_slots`, `nice`, `locks_trace`, `scratch`, `build_cache` (the build systems' caches: the graph's under `bazel/`), `store_max_gb` and `store_max_age_days` (the environment store's bound), `tools_dir`, `windows_system32`, `wsl_lib` |
| `clock.json` | `ghz`, `lock`, `unlock` (null `ghz`: the host measures unlocked) |
| `store.json` | `root` (the measurements store: a checkout of the data repository, fsla-results; fsla-devtools' section, with no default, which `init` writes as the sibling `fsla-results`) |
| `workspace.json` | `suite` (the fsla-bench checkout whose root `declare.py` measures this tree), `devtools` (the fsla-devtools checkout, the declared build system, linked into every venv), `worktrees` (the one folder worktrees live in), `hook_roots` (the folders the action-hook dispatcher may run a checkout's hooks under; null: this repository's checkouts and `worktrees`), `base_venv` (a venv the checkouts on the machine whose hooks run from it read; this tree reads none of it) |

<a id="bootstrap"></a>**Before the reader is importable.** The graph's front door (`bazel/front_door.py`, a stub over
`fsla_devtools.graph.front_door`) runs
under the machine's `python3`, before any venv exists. `tools/bootstrap.py` reads one value the standard-library way --
`workspace.devtools`, from `workspace.json` in the directory the reader's own rule names -- and the front door imports
the reader and the host budget from that checkout. Run (`python3 tools/bootstrap.py`), it is the machine's one step
outside the graph: it imports the reader the same way and runs `tools/dev.py`'s `install_bazelisk`, the pinned bazelisk
and the `bazel` link, since nothing can `bazel run` before `bazel` exists. That bazelisk's pin, like the toolkit's and Nsight Compute's, is fsla-devtools' (`pins/`): `dev_config.read_pin` reads a shared pin (`SHARED_PINS`) from the `workspace.devtools` checkout, the one the front door hands the graph as `@fsla_devtools` ([build-graph](../../build-graph.md#front-door)), so the tools and the graph read one file; any other from `tools/`. The sibling checkouts' location (`SIBLINGS`, the default of
`store.root`, `workspace.suite` and `workspace.devtools`) is defined there once, for both. Two read-outs serve a
machine that has nothing yet, CI's runner (`.github/actions/graph`): `--devtools-pin`, the fsla-devtools repository
and commit `MODULE.bazel` pins, which it clones beside the checkout first, and `--bin-dir`, the directory of
`bazel`'s link (`toolchain.bazel`), for its PATH.

Each key's type, default and meaning are declared in `SCHEMA`. `bazel run //:show` prints every
resolved value and the file it came from ([dev.md](dev.md)).

**The API.**
- `get("section.key")` returns a value.
- `cuda_home()` and `cuda_bin(tool)` resolve the toolkit: the graph's (`@cuda_tools`, pinned by
  fsla-devtools' `pins/cuda_toolkit_pin.json`) -- in a test tier the copy the tier hands in (`FSLA_CUDA_HOME`), else the graph's repository
  (`bazel query --output=location @cuda_tools//:bin/nvcc`); there is no second install. Every compile, disassembly and assembler check outside the graph's own rules uses them, so
  the ptxas a gate reads is the one the build ran.
- `scratch(name)` returns a tool's scratch directory, this checkout's own: `<host.scratch>/<digest>/<name>`, where
  `<digest>` is the first 12 hex digits of the SHA-256 of the checkout's absolute root path (`/tmp/fsla/<digest>/lab`
  for the lab's run records, input files and profiler dumps by default). Two checkouts of one machine write
  different directories; no environment variable names it. The digest, not a directory inside the checkout, because
  what a run leaves (a record, a dump, the cubins ratify extracts) is bulky, is read back by no one, and `host.scratch` is
  the one place the machine's config already puts it (`bazel run //:check` prints this checkout's directory). It holds
  no compile: the lab's experiments are the graph's targets (`docs/internals/tools/lab.md#graph`), whose outputs Bazel keeps.
- `windows_tool(rel)` returns a Windows tool under WSL.
- `pinned(name)` returns where a pin's binary is installed (`read_pin(name)`: its folder under `host.tools_dir`,
  the binary in it), the one lookup dev.py and the measurement tools use; `ncu()` returns the pinned Nsight Compute,
  refusing until init installed it.
- `detect()` returns what `init` writes.
- `reload()` rereads the files, here and in the locks' reader, for a test that points `FSLA_DEV_CONFIG` at a
  temporary directory.

**Environment variables that remain.** One list in the loader names them, and nothing else may read one.
`tools/lint/lint_standards.py` enforces this: it checks Python reads through the syntax tree (constants and
f-string prefixes resolved) and shell expansions.
- `HANDOFFS`: values one of the tree's own processes passes to a child it starts.
  - `FSLA_DEV_CONFIG` itself.
  - The graph's copy of the pinned toolkit a test tier hands its tests (`FSLA_CUDA_HOME`).
  - The root of fsla-devtools' module a test tier hands its tests, whose `pins/` they read (`FSLA_DEVTOOLS`).
  - The directory that holds warpline's Python package, which the `lab` run target hands the lab and the lab's graph
    actions carry (`FSLA_WARPLINE_PYTHON`).
  - pytest-xdist's worker id.
  - The agent harness's campaign directory.
  - The login name printed into the sudoers rule `bazel run //:clock` suggests (`USER`).
  - The search path bazelisk hands what it runs, the front door's directory first (`PATH`; `check` finds the user's
    `bazel` past it).
  - `host.nvcc_threads`, which the front door hands the graph's compile flags (`FSLA_NVCC_THREADS`).

  The Bazel release bazelisk hands the front door (`BAZEL_REAL`) and the shared store the front door hands the
  graph's environment (`FSLA_GRAPH_STORE`) are fsla-devtools' to read, not this tree's.

What one build is, is the graph's settings (`--//:archs` and the arm and part settings, `docs/build-graph.md#compile`),
never an environment variable.

**What it replaced**, each variable deleted rather than kept as a fallback:
- `FSLA_NVCC_THREADS`, `MAX_JOBS` as a setting, `FSLA_HOST_BUDGET_SLOTS`, `FSLA_HOST_BUDGET_DIR`,
  `FSLA_GPU_LOCK`, `FSLA_GPU_SHARED_SLOTS`, `FSLA_NO_NICE`, `FSLA_LOCKS_TRACE`.
- `FSLA_SCCACHE`, `FSLA_SCCACHE_BIN` as a setting, `MOLD_PATH`, `CUDA_HOME` as a setting.
- `FSLA_CLOCK_CONFIG`, `FSLA_RESULTS_DIR`, `FSLA_DB`, `FSLA_IMAGE_DIGEST`, `FSLA_FLA_CROSSCHECK`,
  `FSLA_MACHINE_SALT`.
- The shell tools' `FSLA_BASE_VENV`, `FSLA_WORKTREES_DIR`, `FSLA_WORKTREE_BUILD`, `FSLA_BUILD_SLOTS`,
  `FSLA_BUILD_DIR`.
- The `/usr/local/cuda/bin/...` and `/mnt/c/Windows/...` paths spelled in eight tools, and three hardcoded
  `/tmp` scratch directories.
