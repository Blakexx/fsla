# Developer setup — the dev container

This page is the whole story for a DEVELOPER's machine: someone who is going to
compile the kernels, run the battery, and use the reading/searching/profiling
tools the queue's briefs assume. It is not for a USER of the package — a user
installs the prebuilt wheel (`README.md`), takes the ratified manifest inside it
on faith, and never runs anything in this file.

The container (`Dockerfile`, `.devcontainer/devcontainer.json`) exists because
the closed-world build (`docs/build.md`) is gated on an EXACT `ptxas` string, an
exact `sccache`, and now an exact set of the surrounding tools every brief
names by name (`clangd`, `ast-grep`, `hyperfine`, ...). Reproducing all of that
by hand on a new machine is exactly the kind of silent drift the rest of this
repository refuses; the container makes it one build instead of a checklist.

## 1. Host prerequisites

* **A CUDA 12.4-capable NVIDIA driver.** This repo's own dev box runs driver
  `595.95` against an RTX 3080 Ti (`sm_86`); the base image's toolkit is
  12.4.1, so any driver new enough for CUDA 12.4 (>= 550.x) works.
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
    docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
    ```
    A working WSL2 GPU pass-through prints the host's GPU table from inside
    the container. **This is the gate this repository's own K37 build could
    not run**: at authoring time (2026-08-28) `docker`/`podman`/`nerdctl` were
    all absent from `PATH` on the box, and `/usr/bin/docker` was a dangling
    symlink into `/mnt/wsl/docker-desktop/cli-tools/...` — Docker Desktop's
    WSL integration was not mounted. Nothing below has been run end-to-end on
    hardware; the Dockerfile, lock file and this page were built to
    completion and reviewed by hand instead. Run the `nvidia-smi` check above
    BEFORE anything else on a fresh machine.
* **git**, to clone the repo and (for a developer) create worktrees.

## 2. Build the image

```bash
cd rola-standalone   # this repository's root — the Dockerfile lives here
docker build \
  --build-arg ROLA_IMAGE_DIGEST="$(sha256sum Dockerfile | cut -d' ' -f1)" \
  -t rola-dev:latest .
```

`ROLA_IMAGE_DIGEST` is stamped into every container built from this image and
recorded by `tools/ratify.py` as `source.image_digest` — it is not part
of the ratified SASS-body digest (`manifest_digest()`), so it can never move a
register/spill number; it exists purely so a manifest can answer "was this
measured inside the pinned image, and which Dockerfile" without a registry
lookup. Passing the flag is optional — an unset `ROLA_IMAGE_DIGEST` records the
empty string, same as a bare-host build.

The build pins its own toolchain the same way the repo's build/ratify gates do
(`docs/build.md` rule 1) — and this took an extra step, measured, not assumed:
the base image `nvidia/cuda:12.4.1-devel-ubuntu22.04` does **not** ship the
`ptxas` build the manifests were ratified under. Both that tag and
`12.4.0-devel-ubuntu22.04` were checked (2026-08-29) and both report
`V12.4.131, cuda_12.4.r12.4/compiler.34097967_0` — a later CUDA 12.4 package
revision than `tools/manifests/sm_86.json`/`sm_80.json`'s pinned
`V12.4.99, cuda_12.4.r12.4/compiler.33961263_0`. The Dockerfile does not use
the base image's toolchain as-is: it downgrades `cuda-nvcc-12-4`,
`cuda-nvvm-12-4` and `cuda-crt-12-4` to the exact `12.4.99-1` package version
from NVIDIA's own apt repo (still available at check time) and asserts the
resulting `ptxas --version` string right there in the same build step — a
build whose apt mirror ever stops carrying `12.4.99-1` fails loudly at THAT
step, not silently later. `setup.py`'s own rule-1 gate is the second,
independent check: even if the image-build assertion above were ever removed,
the first `pip install -e .` inside a mismatched container would still refuse
rather than ship unmeasured codegen.

## 3. Open the devcontainer, or run the image directly

**THE CLONE IS THE WORKSPACE; WORKTREES ARE OPTIONAL.** A plain `git clone` of
this repo, opened with nothing else exported, must start the container —
verified 2026-08-29 with a real fresh clone in a temp directory,
`ROLA_WORKTREES_DIR` unset, and the actual `@devcontainers/cli up` (0.88.0),
not a reasoned-about approximation.

**VS Code / Claude Code, via the devcontainer:**

```bash
code .   # "Reopen in Container" — reads .devcontainer/devcontainer.json;
         # works with NOTHING exported.
```

`/workspace/worktrees` is a statically-declared, auto-created, empty named
Docker volume (`rola-worktrees`) — there to write into from inside the
container, not to read host worktrees from. **Two real bugs were found and
fixed here by actually running the CLI, not by reasoning about Docker
semantics in the abstract (2026-08-29):**
* The original mount nested a `${localWorkspaceFolder}` reference inside a
  `${localEnv:VAR:default}` default value
  (`"source=${localEnv:ROLA_WORKTREES_DIR:${localWorkspaceFolder}/../worktrees},...,type=bind,..."`).
  `@devcontainers/cli`'s substitution pass does not recursively expand a
  variable reference sitting inside another variable's default, so it handed
  Docker the literal, unexpanded text `${localWorkspaceFolder/../worktrees}`
  as a mount source, and Docker refused it outright ("is not an absolute
  path") — unconditionally, whether or not `ROLA_WORKTREES_DIR` was set.
* The first fix attempt replaced that with a plain `-v`-shorthand string
  (`"NAME:/workspace/worktrees"`), reasoning that Docker's classic `-v`
  auto-detects bind-vs-volume from the shape of `SOURCE` (true, and used
  below for the plain `docker run` form) — but `@devcontainers/cli` does NOT
  pass a string mount through as `-v`: it always emits `--mount`, and does
  not reformat a colon-shorthand string into `--mount`'s required
  `key=value,key=value` syntax, so Docker refused THAT too ("must be a
  key=value pair").

The mount that actually works is `.devcontainer/devcontainer.json`'s current
`"type=volume,source=rola-worktrees,target=/workspace/worktrees"` — a STATIC
named volume, unconditionally, because `type=volume` mounts auto-create an
empty volume with no host resource required to pre-exist, so there is no
missing-source question left for the CLI to mishandle. This means the
devcontainer path has NO env-var switch to a host bind mount — a single
static `--mount`-syntax line cannot conditionally pick `type=bind` vs
`type=volume`. A contributor who keeps sibling git worktrees
(`rola-branch-structure`) and wants them reachable from inside the
devcontainer has two options: use the plain `docker run` form below (which
DOES support the env-var switch), or edit that one `mounts` line in their own
local, uncommitted checkout to
`"type=bind,source=/path/to/your/worktrees,target=/workspace/worktrees"`.

**Plain `docker run`, the equivalent without an editor integration, WITH the
env-var worktrees switch:**

```bash
docker run --rm -it \
  --gpus all --ipc=host \
  -v "$(pwd)":/workspace/rola \
  -v "${ROLA_WORKTREES_DIR:-rola-worktrees}":/workspace/worktrees \
  -e ROLA_IMAGE_DIGEST="$(sha256sum Dockerfile | cut -d' ' -f1)" \
  -w /workspace/rola \
  rola-dev:latest
```

Here the raw docker CLI's `-v SOURCE:TARGET` shorthand really does
auto-detect from the shape of `SOURCE` (verified directly, not just for
`@devcontainers/cli`): `ROLA_WORKTREES_DIR` unset falls back to the bare name
`rola-worktrees` (a named volume, matching the devcontainer's default
exactly); set to a host path, Docker bind-mounts that path and creates it if
missing. No path is hard-coded to a home directory either way
(KERNEL_STANDARDS.md §17).

## 3b. Building on the bare host (no container) needs `mold`

K39 (Blake ruling, 2026-08-28): `mold` is THE extension's linker unconditionally
-- `setup.py` refuses to build without it, no fallback to `ld.bfd`/`ld.gold`
(`tools/mold_toolchain.py`, `tools/mold_pin.json`). Inside the container it is
already installed (the `Dockerfile`'s pinned release, below); on a bare host
with no root and no apt package for it (measured on this repo's own dev box:
`apt-cache policy mold` has no candidate on Ubuntu 22.04, and there is no
passwordless sudo to install one even if there were), vendor the SAME pinned
release USER-LOCALLY -- no install step, no root:

```bash
curl -sSL -o mold.tar.gz \
  https://github.com/rui314/mold/releases/download/v2.42.0/mold-2.42.0-x86_64-linux.tar.gz
echo "f5ed2f6e31d1ada4f07fe766fe0de7a73104d1c5cdc59086fcecc16a43720b6d  mold.tar.gz" | sha256sum -c -
tar xzf mold.tar.gz -C /some/persistent/dir
export MOLD_PATH=/some/persistent/dir/mold-2.42.0-x86_64-linux/bin/mold
```

`MOLD_PATH` (or a `mold` already resolvable on `PATH`) is what
`tools/mold_toolchain.py:link_flags()` checks before every build; the sha256
above is the exact pin `tools/mold_pin.json` and the `Dockerfile`'s
`MOLD_SHA256` ARG both carry, so the host and the container link with the
byte-identical binary.

## 3c. The pointer venv, the fast path (many worktrees on one bare host)

K46 (2026-08-29, "agents stop reverse-engineering it from sibling venvs" --
recurring SKILLS FEEDBACK from K42 and K45). Sections 3b/4 are the FIRST build
of a single checkout. This is the DIFFERENT question a bare host with many
concurrent git worktrees (`worktrees-live-in-one-folder`) actually has: each
worktree needs its OWN `rola` editable install (its own compiled `.so`,
matching its own SHA and its own arm subset), but reinstalling `torch` +
`einops` + `entmax` + the rest of `requirements.lock` from scratch for every
worktree is minutes of wasted work repeated dozens of times for bytes that are
identical across all of them. The POINTER VENV is a small venv per worktree
that BORROWS a shared base venv's already-installed packages and adds only
what is actually per-worktree: this checkout's own root, and this checkout's
own editable-installed extension.

`tools/build/new_worktree.sh <name> <sha>` does the whole sequence below in one
call; read it before doing this by hand a second time.

**1. The shared base venv, built ONCE:**

```bash
uv venv --python 3.11 --seed /path/to/shared/base-venv
/path/to/shared/base-venv/bin/python -m pip install -r requirements.lock
```

This is a normal, complete venv — nothing pointer-shaped about it yet. It is
the thing every worktree's pointer venv borrows from, so building it well once
(the full `requirements.lock`, the right CUDA-enabled `torch`) is what makes
every later worktree venv cheap.

**2. Per worktree, a new SEEDED venv:**

```bash
uv venv --python 3.11 --seed /path/to/worktrees/venv-<name>
```

**FOOTGUN 1 — a seedless venv silently falls through to a system python.**
Omitting `--seed` leaves no `pip` binary inside the new venv at all. A later
bare `pip install ...` (as opposed to `<venv>/bin/python -m pip install ...`)
then resolves to whatever `pip` happens to be first on `PATH` OUTSIDE the
venv — installing into a completely different python's site-packages with no
error, no warning, and an import that "worked" moments ago now resolving to
the wrong build. This is the same trap family as the device-side-stamp
extension trap (`extension-trap-device-side-check`): nothing about the venv's
existence proves what actually received the install. `--seed` (or always
spelling the install as `python -m pip install`) is not optional.

**3. Two `.pth` files in the new venv's `site-packages`, pointing OUT:**

```bash
SITE=/path/to/worktrees/venv-<name>/lib/python3.11/site-packages
echo /path/to/worktrees/<name>                              > "$SITE/aa_rola_<name>.pth"
echo /path/to/shared/base-venv/lib/python3.11/site-packages  > "$SITE/zz_rola_base.pth"
```

`site.py` processes `.pth` files in the sorted order of their filenames, so the
`aa_`/`zz_` prefixes are load-bearing, not decoration: this worktree's own root
(`aa_...pth`) and its own editable install (`__editable__...pth`, added by step
4 below, which alphabetically sorts after both) resolve BEFORE the shared base
venv's copy of anything with the same name. A worktree's own compiled `rola`
extension is what gets imported, never the base venv's.

**4. The in-tree editable build, `--no-deps`, through the build lock:**

```bash
cd /path/to/worktrees/<name>
./tools/build/build_lock.sh -- \
  /path/to/worktrees/venv-<name>/bin/python -m pip install -e . --no-deps --no-build-isolation
```

**FOOTGUN 2 — `pip install -e .` without `--no-deps` pulls a fresh,
incompatible torch and yields an ABI-mismatched `.so`.** `pyproject.toml`
declares `torch` as a dependency; without `--no-deps`, `pip`/`uv pip` resolves
and installs its OWN copy of `torch` into the pointer venv — not necessarily
the same build (CPU-only, a different CUDA minor version, a different ABI) as
the one the shared base venv already carries and the extension is about to
compile its C++/CUDA ABI against via `torch.utils.cpp_extension`. The result
imports without error and then crashes or misbehaves at the first real call,
because the compiled `.so` and the `torch` actually resolved at import time
were built against two different ABIs. The shared base venv's `torch` (reached
through the `zz_rola_base.pth` file in step 3) IS the dependency resolution
here; `--no-deps` is not optional either. `./tools/build/build_lock.sh` is
the SAME iteration/gate build-slot discipline every other build in this repo
goes through (`docs/build.md`, `rola-build` skill) — a pointer-venv build is
still a build, and still shares the machine-wide compile budget.

Verify with `python -c "import rola; from rola._build_config import
BUILD_CONFIG; print(BUILD_CONFIG['archs'])"` against the new venv's python —
the device-side stamp, never a path check (`extension-trap-device-side-check`)
-- but run it FROM the new worktree's own directory. **FOOTGUN 3, found
writing `tools/build/new_worktree.sh`: `python -c` puts `''` (the
process's OWN current directory) first in `sys.path`.** A verification launched
from inside a DIFFERENT rola worktree (which also carries a `rola/` package)
resolves `import rola` to THAT worktree instead of the new one — `''` beats
every `.pth`-added path because it is `sys.path[0]`, checked before any of
them. `tools/build/new_worktree.sh` `cd`s into the new worktree before its own
verification step for exactly this reason; do the same by hand.

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
pytest tests/unit tests/oracle -m "not cuda"                       # CPU-only, no device, no build
pytest tests/unit -m cuda                 # tests/unit's CUDA-marked rows
pytest tests/oracle                       # kernels vs the fp64/torch baselines
pytest tests/integration                  # composition, dispatch, two-branches-agree
```

This is the same battery `docs/testing.md` "Running it" names; the session-scoped
`gpu_lock()` fixture in `tests/conftest.py` serializes GPU-touching runs the
way it does on the bare host — the container adds no scheduling of its own. Expect
~3.6 min total on hardware comparable to this repo's dev box (`docs/testing.md`,
measured 2026-08-06).

## 6. Run the probe

```bash
python tools/probe_cells.py \
  --binary "worktree:$(pwd)" --out /tmp/probe.jsonl
```

`tools/probe_cells.py` is the canonical K35 carry-family measurement harness
(`tools/probe_cells.py`'s own module docstring) — it measures the default cell
set (`dense`, `k16_w384`, `k16_w512_record`, `k4_w512_record`) against the
binary named by `--binary`, interleaved A/B rounds under the shared GPU lock,
and prints one row per cell. A cell whose declared arm this binary did not
build is a refusal, not a silent fallback (its pre-flight arm resolution,
same docstring) — which is the check this step is actually standing in for:
that the container's compile produced a binary carrying the arms the probe
expects to measure.

## What's in the image, and why

| Tool | Source | Role |
|---|---|---|
| `ptxas`/`nvcc` (CUDA 12.4.99) | base image | the ratified toolchain (`docs/build.md` rule 1) |
| `sccache` 0.17.0 | `tools/sccache_pin.json`, verified | the compiler cache `docs/build.md#sccache` measures |
| `ninja` | apt | the build's actual compile driver (`docs/build.md` step 5) |
| `mold` | pinned GitHub release, sha256-verified | fast host-side linking for the extension's `.so` |
| `graphviz`, `sqlite3` | apt | dependency/DAG rendering, `perf/ledger.jsonl` inspection |
| `clangd`, `clang-tools-14` (clang-query) | apt | the editor/AST tooling briefs assume is present |
| `universal-ctags` | apt | cross-referencing the `csrc/` tree |
| `jq`, `fd` (via `fd-find`) | apt | the shell one-liners this repo's own docs use |
| `hyperfine` | pinned GitHub release, sha256-verified | ad-hoc wall-clock comparisons outside `probe_cells.py`/`perf/` |
| `ast-grep-cli`, `py-spy` | pipx (real PyPI packages) | structural code search; live Python profiling |
| `difftastic` (`difft`) | pinned GitHub release, sha256-verified | structural diffs — **not** pipx: it has no PyPI package, verified at authoring time; installed the same way as `mold`/`hyperfine` instead of being forced through a tool that cannot actually install it |
| `pre-commit` | pipx | the repo's lint/format hooks |
| Python 3.11, `requirements.lock` | `uv venv` + `uv pip install -r` (lock generated by `uv pip compile`, see `requirements.in`'s header) | replaces the hand-grown dev venv this box iterated in; includes `pytest-timeout` and `hypothesis` per this card |

## Reproducibility as a statement

`tools/ratify.py`'s written manifests now carry `source.image_digest`: the
`ROLA_IMAGE_DIGEST` environment variable the container sets (step 2), or the
empty string if a ratification ran on a bare host. It sits beside
`source.csrc_sha256`/`source.date`, not inside `toolchain` — it identifies
WHERE a measurement ran, not WHAT was measured, so (like `sccache`'s version)
it is recorded and reported on drift, never part of `manifest_digest()`'s
ratchet-gated hash. Re-ratifying inside this container and comparing
`source.image_digest` across two runs is how "reproduced under the same
pinned environment" becomes a value you can diff instead of a claim you take
on faith.
