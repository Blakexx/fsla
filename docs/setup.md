# Developer setup — the dev container

This page is the whole story for a DEVELOPER's machine: someone who is going to
compile the kernels, run the battery, and use the reading/searching/profiling
tools the docs and skills use. It is not for a USER of the package — a user
installs the prebuilt wheel (`README.md`), takes the ratified manifest inside it
on faith, and never runs anything in this file.

The container (`Dockerfile`, `.devcontainer/devcontainer.json`) exists because
the closed-world build (`docs/build.md`) is gated on an EXACT `ptxas` string, an
exact `sccache`, and an exact set of the surrounding tools the docs and skills
name (`clangd`, `ast-grep`, `hyperfine`, ...). Reproducing all of that
by hand on a new machine is exactly the kind of silent drift the rest of this
repository refuses; the container makes it one build instead of a checklist.

## 0. The dev environment: `tools/dev.py`

Everything this tree needs from a machine -- the CUDA toolkit, Nsight Compute,
the pinned `sccache` and `mold`, the host's compile slots, the lock files, the SM
clock, the measurements store, the worktrees folder and base venv -- is one
directory of JSON files, the dev config (`~/.config/rola/`,
`docs/internals/tools/dev_config.md`, KERNEL_STANDARDS §23). One command sets a
machine up and checks it (`docs/internals/tools/dev.md`):

Every command is a `bazel run` target but one: `python3 tools/bootstrap.py`, by the machine's own python3, installs the
pinned bazelisk and links `bazel` to it, since nothing can `bazel run` before `bazel` exists. It, and every target, reads
the dev config through rola-devtools (`rola_devtools.config`, which also holds the machine's locks), imported from the
rola-devtools checkout beside this tree, where `workspace.devtools` points by default: clone it there first. A fresh
machine runs on the reader's defaults until `init` writes its config.

```bash
python3 tools/bootstrap.py              # the one step outside the graph: the pinned bazelisk, `bazel` linked to it
bazel run //:init                       # detect, install pinned tools, wire the commit gate and agent router, check
bazel run //:clock -- --mhz 1665        # once per host: lock the SM clock (one administrator prompt under WSL)
bazel run //:check                      # every dependency, one row each; exit 1 on a red
bazel run //:dev_env                    # this checkout's venv, which its commit gate runs from
bazel run //:worktree -- <name> <sha>   # a worktree, its venv and gate, its extension
```

## 1. Host prerequisites

* **An NVIDIA driver for the toolchain's CUDA major.** The declared toolchain is `cu13` (CUDA 13.0,
  `tools/toolchains/cu13.json`), so any driver >= 580 works. This repo's own dev box runs driver `595.95` against an
  RTX 3080 Ti (`sm_86`).
* **Docker**, with the NVIDIA container runtime wired in:
  * **Linux (native):** install `nvidia-container-toolkit`
    (https://github.com/NVIDIA/nvidia-container-toolkit), then
    `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`.
  * **WSL2 (this box's platform):** install the Windows NVIDIA driver (which
    already carries WSL2 CUDA support — do **not** install a Linux NVIDIA
    driver inside WSL2 itself); install Docker Desktop with the WSL2 backend
    enabled, or Docker Engine directly inside the WSL2 distro; then
    `nvidia-container-toolkit` exactly as the native-Linux step above, run
    from inside the WSL2 distro. Verify with:
    ```bash
    docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi
    ```
    A working WSL2 GPU pass-through prints the host's GPU table from inside
    the container. Run it BEFORE anything else on a fresh machine;
    `python tools/dev.py container check` (step 3) is the whole proof.
* **git**, to clone the repo and (for a developer) create worktrees.

## 2. Build the image

```bash
python tools/dev.py container build
```

The image is built from the environment's inputs alone -- `Dockerfile`, `.devcontainer/`, `requirements.lock`, the
tool pins and the `tools/dev.py` files (`ENV_INPUTS`), copied into a throwaway context, never the whole tree -- and
tagged `rola-dev:<first 12 of the environment key>`. The key is sha256 over those inputs; the image records it as
`environment.image_digest`, which `tools/ratify.py` stamps as `source.image_digest` and every record row carries.

Its last step is `tools/dev.py init --image`: the same one init a bare host runs, writing the image's dev config
(`ROLA_DEV_CONFIG=/opt/rola/dev-config`) for its fixed layout, installing the pinned `sccache`, `mold` and
bazelisk from the repo's pins, and running `check --image`. A red check fails the image build -- the image built on 2026-08-28 went two
weeks without `cuobjdump`, the Nsight Compute the pipe timeline needs, or flash-attn, and nothing said so.

The image is built for a toolchain record (`tools/toolchains/<name>.json`; `--toolchain NAME` when more than one is
declared), tagged by its environment key, and a build that succeeds keeps the three `rola-dev` images tagged most
recently, the one it built always among them, and removes the rest (`KEPT_IMAGES`: an older environment's image is a
rebuild away, its layers in Docker's build cache). `container build` passes the record's base image, its exact apt package pins and its torch index as build
arguments, and the Dockerfile fails the build unless `ptxas --version` is the record's string byte for byte -- the same
comparison `setup.py`'s rule 1 makes, so a base tag carrying a newer patch (every CUDA 12.4 tag shipped V12.4.131
against a V12.4.99 pin, measured 2026-08-29) cannot slip through. `cuobjdump` is one of the record's pins and
`nsight-compute-2025.3.0` is pinned beside them. The attention reference needs no package: it is torch's own flash
backend (`docs/measurement.md`).

## 3. Prove it, then open it

```bash
python tools/dev.py container check
```

runs the proofs inside the image, with the host's mounts, and stores the result through `rola_results` at
`environment`, under the environment key: `tools/dev.py check` inside; the GPU and a call through torch's flash backend; a lock
the HOST holds is seen as held inside (so one lock governs host and container, KERNEL_STANDARDS §14 addendum); and an
iteration build of a copy of the checkout, imported. **A commit that changes any environment input needs a passing
record for the key of what it stages -- the commit gate refuses it otherwise (KERNEL_STANDARDS §23 (3)).**

**The mounts come from the host's dev config.** `python3 tools/dev.py container compose` writes
`.devcontainer/compose.local.yaml` (never committed) binding, onto the image's fixed layout: this checkout at
`/workspace/rola`, the host's `host.lock_dir` at `/run/rola/locks` and its GPU lock's directory at `/run/rola/gpu`,
`store.root` at `/workspace/store`, `workspace.suite` at `/workspace/suite`, and `workspace.worktrees` at
`/workspace/worktrees`. It refuses when no image
exists for the current key. Under WSL it also mounts the directory above `host.wsl_lib` (`/usr/lib/wsl`) read-only:
the image's loader path starts at `/usr/lib/wsl/lib`, and without the Windows driver's libraries there CUDA fails to
initialize inside the container with error 500 even though `nvidia-smi` works (measured 2026-09-12, Docker Desktop
20.10.17, driver 595.95 -- the 2026-08-28 image had never been run against the GPU). The whole directory, not `lib/`
alone: that `libcuda` is a loader that opens the driver store beside it.

**VS Code / Claude Code:** "Reopen in Container". `.devcontainer/devcontainer.json` runs `container compose` as its
`initializeCommand`, then the `rola` service of `.devcontainer/compose.yaml` with the generated mounts, and runs
`python tools/dev.py check` once created.

**Without an editor:** `docker compose -f .devcontainer/compose.yaml -f .devcontainer/compose.local.yaml run --rm rola bash`.

Inside, the SM clock is not locked: the lock is a host act (`bazel run //:clock`), and rows measured in the container
record `clock_locked = false`.

The history this replaced, kept because it was measured: the first devcontainer (2026-08-29) mounted worktrees through
a `${localEnv:...}` default nested inside another variable, which `@devcontainers/cli` does not expand, then through a
`-v`-style string the CLI forwards as a malformed `--mount`; it ended as a static empty named volume, so host worktrees
were never visible inside. Generating the mounts from the dev config removes the variable and the question.

## 3b. Building on the bare host (no container) needs `mold`

`mold` is THE extension's linker unconditionally (Blake, 2026-08-28)
-- `setup.py` refuses to build without it, no fallback to `ld.bfd`/`ld.gold`
(`tools/mold_toolchain.py`, `tools/mold_pin.json`). On a bare host with no root
and no apt package for it (this repo's dev box: `apt-cache policy mold` has no
candidate on Ubuntu 22.04), `bazel run //:init` downloads the pinned
release, verifies its sha256 against `tools/mold_pin.json`, unpacks the whole
release (`bin/mold` beside `bin/ld.mold`, which `-fuse-ld=mold` selects) under
`host.tools_dir`, and records it as `toolchain.mold`. The container and the host
link with the byte-identical binary.

## 3c. The checkout's venv: many worktrees on one environment

Each checkout needs its OWN `rola` (its own library, matching its own commit), but installing the whole
`requirements.lock` for every worktree is minutes of identical bytes. The graph builds the environment once into the
shared store, keyed by the lock (docs/build-graph.md#environment), and `bazel run //:dev_env` writes each checkout a
venv over it, `venv-<checkout>` beside the checkout ([`internals/tools/dev.md`](internals/tools/dev.md)). It holds the
invariants that make it safe:

- **It is the environment's interpreter**: the venv is made on the store's Python, so the packages it names are the
  ones that interpreter's ABI built.
- **The `.pth` order puts the checkout first**: `site.py` reads `.pth` files in filename order, so the checkout's
  `rola` and the co-developed sibling checkouts resolve before the environment's copy of anything with the same name.
- **Nothing is installed into it**: its packages are the environment's; the checkout's library is the one
  `bazel run //:library` copies into it.

`bazel run //:worktree -- <name> <sha>` adds the worktree and runs both targets in it. It verifies from inside the
new worktree: `python -c` puts its own working directory first on `sys.path`, so a check launched from another worktree
imports that worktree's `rola`.

## 4. First build of the SHIPPED set

Inside the container (or on the bare host with `mold` vendored per 3b above):

```bash
pip install -e . --no-build-isolation
```

This runs the closed-world gates in order (`docs/build.md`): rule 3 (ratified
archs), rule 1 (toolchain — verified against the SAME manifest string the box
this container was authored on already matches), rule 5 (vendored CUTLASS),
the shard-generation check, the compile itself (`sccache`-cached per
`docs/build.md#sccache` once a cache is warm), the fatbin gate, and the
post-build ratification (`tools/ratify.py`) — which is the gate that answers
"does this container's compile reproduce the committed manifest's SASS
hashes". A clean build reports:

```
rola-standalone $ python tools/ratify.py --arch 80 --arch 86
...
SASS-body gate: all <N> ratified bodies unchanged; PASS
```

`--arch 80` requires an ampere-class-or-later device to actually launch
against, but the RATIFICATION itself (recompiling the shipped TUs and hashing
their SASS) needs only the toolchain, not a matching physical GPU — see
`tools/ratify.py`'s own `--arch` handling.

## 5. Run the battery

```bash
bazel test //:lane     # every test tier: cpu, unit_cuda, oracle, integration
```

This is the battery `docs/testing.md` "Running it" names; the session-scoped
`gpu_lock()` fixture in `tests/conftest.py` serializes GPU-touching runs the
way it does on the bare host — the container adds no scheduling of its own. Expect
~3.6 min total on hardware comparable to this repo's dev box (`docs/testing.md`,
measured 2026-08-06).

## 6. Measure this checkout

```bash
python -m rola_devtools.build run declare.py:all --only 'rola/phases' --only 'rola/sass'
```

rola-devtools' build system runs this checkout's declarations (`declare.py`): the build graph's library
(`bazel run //:library`), SASS, and a timing
session of the carry arms on the cell under the GPU and clock locks, each result stored. A timing entry this binary
cannot run is recorded as that entry's failure in the session's output, not skipped — which is the check this step
stands in for: that the graph's build in the container carries the arms a session measures.

## What's in the image, and why

| Tool | Source | Role |
|---|---|---|
| `ptxas`/`nvcc` | the toolchain record's apt pins (`cu13`: CUDA 13.0, V13.0.48) | the ratified toolchain (`docs/build.md` rule 1) |
| `cuobjdump`, Nsight Compute 2025.3 | apt, pinned versions | cubin extraction (SASS gate, region ledger, composer); PM sampling (pipe timeline) |
| `sccache` 0.17.0 | `tools/dev.py init --image` from `tools/sccache_pin.json`, verified | the compiler cache `docs/build.md#sccache` measures |
| `ninja` | apt | the build's actual compile driver (`docs/build.md` step 5) |
| `mold` 2.42.0 | `tools/dev.py init --image` from `tools/mold_pin.json`, verified | the extension's only linker |
| `graphviz`, `sqlite3` | apt | dependency/DAG rendering, the store's record index inspection |
| `clangd`, `clang-tools-14` (clang-query) | apt | the editor and AST tooling the skills use |
| `universal-ctags` | apt | cross-referencing the `csrc/` tree |
| `jq`, `fd` (via `fd-find`) | apt | the shell one-liners this repo's own docs use |
| `hyperfine` | pinned GitHub release, sha256-verified | ad-hoc wall-clock comparisons outside the declared timing sessions |
| `ast-grep-cli`, `py-spy` | pipx (real PyPI packages) | structural code search; live Python profiling |
| `difftastic` (`difft`) | pinned GitHub release, sha256-verified | structural diffs — **not** pipx: it has no PyPI package, verified at authoring time; installed the same way as `mold`/`hyperfine` instead of being forced through a tool that cannot actually install it |
| Python 3.11, `requirements.lock` | `uv venv` + `uv pip install -r` (lock generated by `uv pip compile`, see `requirements.in`'s header) | replaces the hand-grown dev venv this box iterated in; includes `pytest-timeout` and `hypothesis` per this card |

## Reproducibility as a statement

`tools/ratify.py`'s written manifests carry `source.image_digest`: the image's
`environment.image_digest`, the environment key it was built from (step 2), or the
empty string if a ratification ran on a bare host. It sits beside
`source.csrc_sha256`/`source.date`, not inside `toolchain` — it identifies
WHERE a measurement ran, not WHAT was measured, so (like `sccache`'s version)
it is recorded and reported on drift, never part of `manifest_digest()`'s
ratchet-gated hash. Re-ratifying inside this container and comparing
`source.image_digest` across two runs is how "reproduced under the same
pinned environment" becomes a value you can diff instead of a claim you take
on faith.
