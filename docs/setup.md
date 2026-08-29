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
recorded by `tools/ratify.py` as `source.image_digest` (K37) — it is not part
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
verified 2026-08-29 with a real fresh clone in a temp directory and
`ROLA_WORKTREES_DIR` unset (`npx @devcontainers/cli up`). `ROLA_WORKTREES_DIR`
exists for a contributor who ALREADY keeps sibling git worktrees next to this
repo (`rola-branch-structure`) and wants them reachable from inside the
container; nobody else needs to set it, or even know it exists.

**VS Code / Claude Code, via the devcontainer:**

```bash
code .   # "Reopen in Container" — reads .devcontainer/devcontainer.json;
         # works with NOTHING exported. `/workspace/worktrees` lands on an
         # auto-created, empty named Docker volume (`rola-worktrees`) in that
         # case -- there to write into from inside the container, not to read
         # host worktrees from.

# OPTIONAL, only if you keep sibling worktrees you want visible inside the
# container (docs on ../worktrees's layout: rola-branch-structure):
export ROLA_WORKTREES_DIR=/path/to/your/worktrees
export ROLA_IMAGE_DIGEST="$(sha256sum Dockerfile | cut -d' ' -f1)"
code .
```

`.devcontainer/devcontainer.json`'s worktrees mount is ONE line,
`"${localEnv:ROLA_WORKTREES_DIR:rola-worktrees}:/workspace/worktrees"` —
Docker's own `-v SOURCE:TARGET` shorthand (not the object `type=bind` form)
decides bind-vs-volume from the shape of `SOURCE` at container-start time:
unset resolves to the bare name `rola-worktrees` (a named volume, Docker
auto-creates it, no host path required to exist — the zero-config case); set
to a host path, Docker bind-mounts THAT path and creates it if missing. No
path is hard-coded to a home directory either way (KERNEL_STANDARDS.md §17).
**This replaces an earlier version of this mount that used the object
`type=bind` form with a `${localWorkspaceFolder}/../worktrees` path default —
that form's pre-flight validation refused to start a fresh clone with
`ROLA_WORKTREES_DIR` unset, since nothing has a `../worktrees` sibling
directory yet (found from a question about exactly this, 2026-08-29).**

**Plain `docker run`, the equivalent without an editor integration:**

```bash
docker run --rm -it \
  --gpus all --ipc=host \
  -v "$(pwd)":/workspace/rola \
  -v "${ROLA_WORKTREES_DIR:-rola-worktrees}":/workspace/worktrees \
  -e ROLA_IMAGE_DIGEST="$(sha256sum Dockerfile | cut -d' ' -f1)" \
  -w /workspace/rola \
  rola-dev:latest
```

Same shape as the devcontainer's mount line, in bash's own `${VAR:-default}`:
`ROLA_WORKTREES_DIR` unset falls back to the bare name `rola-worktrees` (a
named volume); set, it bind-mounts that host path.

## 4. First build of the SHIPPED set

Inside the container:

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
flock /tmp/rola_gpu.lock pytest tests/unit -m cuda                 # tests/unit's CUDA-marked rows
flock /tmp/rola_gpu.lock pytest tests/oracle                       # kernels vs the fp64/torch baselines
flock /tmp/rola_gpu.lock pytest tests/integration                  # composition, dispatch, two-branches-agree
```

This is the same battery `docs/testing.md` "Running it" names; the `flock`
serializes GPU-touching runs the way it does on the bare host — the container
adds no scheduling of its own. Expect ~3.6 min total on hardware comparable to
this repo's dev box (`docs/testing.md`, measured 2026-08-06).

## 6. Run the probe

```bash
flock /tmp/rola_gpu.lock python tools/probe_cells.py \
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
