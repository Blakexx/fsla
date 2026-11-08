# `tools/dev.py` — the dev environment, set up and checked by one command

KERNEL_STANDARDS §23. One command sets up a host or a container for this tree, checks it and describes it. It
replaced tools/build/new_worktree.sh, tools/setup_clock_lock.py and tools/build/build_lock.sh. The
settings it writes and reads are the dev config ([dev_config.md](dev_config.md)).

    python3 tools/bootstrap.py               # the pinned bazelisk and `bazel` (install_bazelisk), before any target
    bazel run //:init [-- --force]
    bazel run //:check
    bazel run //:show
    bazel run //:get -- <section.key>
    bazel run //:clock -- --mhz 1665
    bazel run //:dev_env                     # dev-env, given the checkout and the graph's environment
    bazel run //:worktree -- <name> <sha>
    bazel run //:store -- commit -m <msg>
    python tools/dev.py container compose|build|check|gate

Each target is this file's command of the same name (`bazel/defs.bzl`'s `dev_run`): run in the checkout on the
graph's environment interpreter, its arguments after `--`.

**`init`** writes this machine's config:
- The pinned CUDA toolkit (`TOOLKIT`, `tools/cuda_toolkit_pin.json`), where it is missing or its nvcc does not report
  its pin's version, which every compile, disassembly and assembler check outside the graph's own rules uses
  (`dev_config.cuda_home()`), on no PATH.
- The pinned devbox tools (`DEV_TOOLS`: hyperfine, difftastic, Nsight Compute and Systems), each where it is missing or
  does not report its pin's version; the measurement tools find the pinned Nsight Compute by its pin
  (`dev_config.ncu()`). In the image their commands are linked into the image's venv's bin, first on its PATH.
- `sccache` and `mold` where they are installed.
- The Windows System32 mount under WSL.

A key already in a file is kept unless `--force`. A pinned tool that is missing, or does not report its pin's
version, is downloaded from the pin's release URL (`tools/sccache_pin.json` and `tools/mold_pin.json` by init,
`tools/bazelisk_pin.json` by `install_bazelisk`: the bootstrap on a host, `init --image` in the image), asked again after a server error or a dropped connection, and
then from its `mirror_urls` (the same asset through GitHub's API: github.com's release downloads returned gateway
timeouts from its own edge for hours on 2026-11-08 while the API served the same assets). Its release sha256 is
verified, and so is its binary
sha256 where the pin has one. A toolkit pin names NVIDIA's redistributable manifest by its digest instead and the
components it takes: each is fetched by the digest the manifest lists and unpacked without its archive's own folder
into the pin's one, each component's LICENSE at `licenses/<component>/LICENSE`, then the pin's `links`; the folder is
unpacked beside its place and moved there, so a folder at the pin's path is a whole toolkit, and it is the graph's
copy (`@cuda_tools`) file for file. The whole release -- a tarball, a bare binary under its release's name, or the
toolkit -- lands under `host.tools_dir`, because the layout
is part of the tool: `-fuse-ld=mold` selects the `ld.mold` next to `mold`. sccache and mold are recorded in the
machine's `toolchain.json`; bazelisk is found by its pin (`dev_config.pinned`: under `host.tools_dir`, in its pin's folder, by its release's
name), never recorded there -- every checkout loads the machine's files, and an older loader refuses a key it does not
declare. `install_bazelisk` links `toolchain.bazel` (`~/.local/bin/bazel`) to the pinned bazelisk -- a link it made to another pinned release is replaced, anything else at the path refused. `init` wires this checkout's commit gate (the git-config keys `core.hooksPath` per worktree, whose pre-commit hook runs `bazel test //:gate`, and rola.venv naming the checkout's own `bazel run //:dev_env` venv, which the post-commit review runs from; a checkout with no venv of its own is named and left as it is) in every worktree of the repository that carries the tracked hook, turns the repository's shared hook into a refusal (a checkout reaches it only unwired), names the base venv in the suite's, the store's and the devtools checkouts, whose hooks read it too, installs the agent skill router where Claude Code keeps a user configuration (KERNEL_STANDARDS §31: the repo's `.claude/hooks/skill_router_bootstrap.py` copied to the user's `~/.claude/hooks/skill_router.py` where it differs, and its PreToolUse entry added to the user's `~/.claude/settings.json` where absent, every other setting kept; the image has no `~/.claude` and gets neither), and runs `check`, whose row says whether the router is installed and current. The
clock is not set by `init`, because locking it asks for administrator consent once.

**`check`** prints one row per dependency and exits 1 on any red:
- The config parses.
- The pinned toolkit's nvcc reports its pin's version.
- Each devbox tool reports its pin's version (Nsight Compute among them).
- sccache and mold pass the build's own gates (`sccache_toolchain.gate`, `mold_toolchain.link_flags`), so
  `check` refuses exactly what a build would.
- The commit gate is wired.
- The lock directory, scratch directory and GPU lock directory are writable.
- The clock is set, or the host is marked unlocked.
- The base venv's interpreter exists where the base venv is set, and the worktrees folder exists; `store.root` is a
  rola-results checkout, and `rola_results` and
  `rola_devtools` import from `store.root` and `workspace.devtools` in the base venv and in the checkout's own venv.

**`clock --mhz N`** locks the SM clock, reads the driver's clock under the lock, and releases it: the round trip
the harness makes (`rola_devtools.locks.clock`, [sm_clock.md](../common/sm_clock.md)). Only after the lock is proven
does it write `clock.json`.
- **Under WSL:** it registers two elevated scheduled tasks, `gpu-lock` and `gpu-unlock`. Each runs a hidden
  `wscript` launcher that calls `nvidia-smi.exe` and exits with its code, and registering them asks for administrator
  consent once. The lock and unlock commands start the task and return when it has finished, failing with its result:
  `schtasks /run` returns before the task runs, so a proof read raced the lock, and a driver's refusal read as success.
- **On Linux:** it writes `sudo -n nvidia-smi -lgc` commands and prints the sudoers rule they need.

**`build [--archs 86] [--carry-arms all]`** is the iteration build (`docs/build.md#iteration`): `setup.py build_ext
--inplace` by the checkout's own venv interpreter (`bazel run //:dev_env`). It refuses when the checkout has no venv
of its own, or when that interpreter's `rola` is not this checkout's.

**`dev-env`** is `bazel run //:dev_env`'s implementation, which passes the checkout and the graph's environment
(its store venv, interpreter and uv). It writes the checkout's venv, `venv-<checkout>` beside it, on the environment's
interpreter. Its site holds five `.pth` files, read in name order: the checkout's root (`aa_rola_<name>.pth`), the
co-developed sibling checkouts (`rola_devtools.pth`, `rola_results.pth`, `LINKED`), the environment's site-packages
(`zz_rola_env.pth`, the venv's marker) and the bytecode prefix (`zz_rola_pycache.pth`: `sys.pycache_prefix` under
`host.build_cache`, so no `__pycache__` lands in the checkout). An activated venv is a developer's shell, so its bin
carries every command the devbox has: the environment's (ruff, pytest, ast-grep, clangd, py-spy, ...), run by the
venv's interpreter so they import what its site names, and the devbox's pinned tools (hyperfine, difft, ncu, nsys),
linked from their folders under `host.tools_dir`, one install a machine whichever checkout links it. It refuses
until init has installed them. It replaces only a venv it wrote, wires the checkout's commit gate to run from it, and refuses unless `rola`, `rola_devtools`, `rola_results` and `torch` each import from
where those files say.

**`worktree <name> <sha>`** creates the worktree in `workspace.worktrees`, then runs `bazel run //:dev_env` and
`bazel run //:library` in it: its venv and commit gate, then its extension, which the graph serves from its shared
cache when any checkout built that commit's library. It imports `rola` and the extension from inside the worktree.

**The base venv** is part of the machine: `check` compares `workspace.base_venv` against `requirements.lock`, where a
pin the venv lacks or carries at another version is drift and extras the lock does not name are not. `init` installs
the missing pins with `uv pip install -r requirements.lock` from the host toolchain's `torch_index`, adding without
removing.

**`container`** keeps the dev image a supported environment (§23 (3), `docs/setup.md` §2-3). Its inputs are
`ENV_INPUTS`: the Dockerfile, `.devcontainer/devcontainer.json` and `compose.yaml`, `requirements.lock`, the tool pins,
the toolchain records, and this file with the modules it imports. Their sha256,
path and bytes each, is the ENVIRONMENT KEY.
- `build` copies only those inputs into a context and builds `rola-dev:<key[:12]>` for a toolchain record
  (`--toolchain`, default the one declared), passing its base image, apt pins, torch index and name as build arguments. The image's last step is
  `init --image`, which writes the fixed layout `IMAGE_LAYOUT`, installs the pins and runs `check --image`. A red row
  fails the build. A build that succeeds keeps the `KEPT_IMAGES` (3) `rola-dev` images tagged most recently, the one it
  built always among them, and removes the rest, naming any Docker refuses; `compose` and `check` refuse by name an
  environment whose image is gone.
- `compose` writes `.devcontainer/compose.local.yaml`, binding the host's checkout, lock directories, store and
  worktrees onto that layout, and refuses without an image for the key; so does `check`.
- `check` runs `CONTAINER_PROOFS` inside the image with those mounts: `dev.py check`, the GPU with a call through
  torch's flash backend, a lock held by a host process seen as held, and an iteration build of a copy of the checkout. It
  stores the result through `rola_results` at `environment`, one record per key, every check a sample.
- `gate` is the commit hook. When the staged files touch an input, it computes the key of the staged blobs and
  refuses the commit without a passing sample for that key.

**`store`** keeps the measurements store (`store.root`, a rola-results checkout): `init` links `rola_results` into
every interpreter that runs here -- each checkout's own venv (it names the environment's site-packages as a plain
path, which reads none of its `.pth` files), the base venv the sibling repositories' hooks run from, and in the image
the image's interpreter at the store's mount point -- by a one-line `rola_results.pth`, and `dev-env` links each venv it writes; `commit -m` commits the records (`python -m
rola_results commit`).

`init`, `check`, `show`, `get`, `clock`, `dev-env` and `worktree` run from whichever python starts them (`dev-env`'s is the environment's, through its run target); `build`, `store` and `container` re-execute themselves under the checkout's own venv (`bazel run //:dev_env`), because the build compiles with it and the store and the container read `rola_results`, which it links; the host's own python carries nothing of this project. On a host the base venv is the venv the sibling repositories' commit hooks run from, and in the image it is the image's interpreter: `check` tests it where it is set.

**`get`** prints one value, for a person or a shell (the `rola-dev` skill's way to read one setting).

**Running a CPU-heavy command under the host budget** is `python -m rola_devtools.locks.host [--slots N]
[--exclusive] -- <cmd>`, niced per `host.nice`. `setup.py` takes the budget itself.
