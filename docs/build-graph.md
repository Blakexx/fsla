# The build graph

The build graph is Bazel, run through bazelisk and the graph's front door. Its workspace is `MODULE.bazel` (the rule
sets and the hermetic CUDA toolkit), `.bazelrc` (the graph's settings), `.bazelversion` and `.bazeliskrc` (Bazel's
release and its digest), `bazel/` (the graph's machinery: its rules, the environment, the platform and the scripts its
actions run) and `bazel/bazel` (the front door); each `BUILD.bazel` sits beside what it describes. The graph builds the
extension (`//:library`, [below](#compile)); `setup.py` builds it too ([`build.md`](build.md)), and runs the gates,
lints, tests and wheels the graph does not hold yet. Measurement is `rola_devtools.build`'s
([`measurement.md`](measurement.md)), whose timing sessions hold the GPU and the clock as machine-wide counting
semaphores, which Bazel cannot do for an action.

## <a id="environment"></a>The environment's layers

| layer | made by | stored |
|---|---|---|
| the host's NVIDIA driver and GPU | the host | -- |
| the machine (base OS libraries, git, `python3`, the host C++ compiler nvcc drives) | the host, or the Docker image | the image |
| bazelisk, Bazel's launcher | `tools/bazelisk_pin.json`, installed by `python tools/dev.py init` (`toolchain.bazelisk`), which links `bazel` on PATH to it (`toolchain.bazel`; `check` has the row) | `host.tools_dir` |
| Bazel | `.bazelversion`, verified against `.bazeliskrc`'s digest | bazelisk's cache |
| the graph's CUDA toolkit | NVIDIA's redistributable components at release 13.0.0 (`MODULE.bazel`'s `cuda.redist_json`: nvcc 13.0.48), only the compiler's (~75 MB of the 4.7 GB set) | the shared repository cache |
| the Python environment | `requirements.lock`, synced with `--require-hashes` by the pinned uv (`tools/uv_pin.json`) into the pinned CPython (3.11.13), by `bazel/environment.bzl`'s `@rola_env` (`bazel/make_env.py`) | `host.build_cache/bazel/store`, one environment a key (the lock, the Python, uv and the torch index), under a file lock; uv's cache there hardlinks the packages |
| the linker | mold at `tools/mold_pin.json`'s release (`@mold`) | the shared repository cache |
| the host C++ compiler's identity | its version and the digests of gcc, g++ and cc1plus (`@host_cc`), an input of every compile | the output base |

## <a id="front-door"></a>The front door

bazelisk runs `bazel/bazel` (`bazel/front_door.py`; the wrapper directory `.bazeliskrc` names, `tools/` by
bazelisk's default) in place of the Bazel it downloaded, so every Bazel command
in a checkout -- typed, or run by a tool -- goes through it. It runs under the machine's `python3` before any venv
exists, reading the dev config through the rola-devtools checkout `tools/bootstrap.py` finds.

- THE MACHINE'S SETTINGS come from the dev config: the shared disk and repository caches under `host.build_cache`
  (`bazel/disk`, `bazel/repository`), in an rc file it regenerates on every run and hands Bazel after the workspace's
  (`--bazelrc`). Nothing in the checkout names a machine.
- EVERY BUILD RUNS UNDER THE HOST BUDGET (`rola_devtools.locks.host`): a command's build phase holds the slots free
  when it starts, at least one, and runs `slots / host.nvcc_threads` jobs (nvcc runs one cicc an architecture). A
  `test` command builds under the budget, then runs its tests holding none; a `run` command writes its
  launch script under the budget (`--script_path`), then runs it holding none. A test or a measurement takes its own
  locks, as outside the graph: a GPU test under a held budget would deadlock against a measurement, which takes the
  GPU before the host's slots. A timing session holds every slot, so no build runs during one.
- Run directly, without bazelisk, it refuses; so does `bazel coverage`, whose instrumented build would differ from the
  budgeted one and run outside the budget (the graph has no coverage target).

## <a id="compile"></a>The extension's compile

`//:library` is `rola_cu13/_C.abi3.so` and its build record `rola_cu13/_build_config.py`; `bazel run //:library` copies
both into the checkout's `rola_cu13/` (a copy: the library is imported from the checkout).

- THE BUILD'S PARAMETERS are the root package's settings: `--//:archs` (default `@rola_env//:flags.bzl`'s
  `RATIFIED_ARCHS`, the toolchain's ratified manifests), `--//:carry_arms` and `--//:decode_arms` (arm indices; empty is
  every arm) and `--//:carry_parts` (`tools/gen_shards.py`'s `carry_parts_spec`). Any value but a default makes an
  iteration build, which its record says.
- THE FLAGS ARE `tools/build_flags.py`'S: the environment repository writes `@rola_env//:flags.bzl` from it
  (`bazel/flags.py`) -- the nvcc codegen flags and torch's extension defines, the host C++ flags -- and the graph states
  none of its own. rules_cuda's own additions are off on every unit (its extended lambdas, compilation-mode flags,
  random seed and host toolchain flags), so nvcc's command line is the build flags, the `-gencode` targets, `-x cu`,
  `-ccbin` and the include roots.
- ONE ARCHITECTURE SPEC A CARD: the library's transition writes `--@rules_cuda//cuda:archs` as
  `compute_80:sm_80;compute_86:sm_86`; a comma list (`sm_80,sm_86`) is one spec, which compiles every card from the
  lowest virtual architecture.
- THE INCLUDE ORDER IS SETUP.PY'S: the repository's headers, then CUTLASS's, then the generated headers and the stamp,
  then torch's, Python's and the toolkit's, so a repository header wins any name collision (`bazel/defs.bzl`'s
  `include_root`).
- THE GENERATED HEADERS are actions over their generators: the arm selections, the carry parts and the carried builds
  (`rola_headers`, `bazel/headers.py`, from the settings) and the device stamp (`tools/csrc_stamp.py` over every file
  under `csrc/rola/src` but its markdown -- the digest ratify computes).
- THE ARM SUBSET IS A UNIT SUBSET: each generated carry arm unit is its own target (`csrc/rola/arms.bzl`, written and
  checked by `tools/gen_shards.py`), and `carry_arm_set` hands the link only the units `--//:carry_arms` names, so an
  unwanted arm is never compiled.
- THE LINK is the pinned mold's, of every unit's objects (each target `alwayslink`, as setup.py links every object)
  and, by path, torch's `libc10.so` and `libtorch_cpu.so`, the CUDA runtime and the driver's stub: each NEEDED by its
  soname, no search path written (no unit carries a dynamic library; the units' CUDA runtime is its headers,
  `//bazel:cuda_headers` -- rules_cuda 0.3.0's own `@cuda//:cuda_headers` names a component CUDA 13 lacks).
- THE RECORD (`bazel/record.py`, `tools/build_record.py`'s writer) runs rule 1 -- the graph's ptxas is the toolchain
  record's byte for byte, the one tie between that record and `MODULE.bazel`'s toolkit pin -- and rule 3, every
  architecture ratified.

## <a id="rules"></a>The rules

Each is a trap a setup like this falls into, and what avoids it (checked against Bazel 9.3.1's own options where it
names one).

1. BZLMOD ONLY, PINNED: `MODULE.bazel` and its `MODULE.bazel.lock` are committed with `.bazelversion`; a gate builds
   with `--lockfile_mode=error`, so a dependency resolved differently from the lock fails rather than drifts.
2. NO MACHINE IN THE CHECKOUT: `.bazelrc` holds the graph's settings and names no path; the machine's come from the dev
   config through the front door. Startup options are never inside a `--config`.
3. THE DISK CACHE IS BOUNDED, AND ITS BOUND RUNS: `--experimental_disk_cache_gc_max_size` caps the shared cache at
   5 GB, the graph's budget on every machine: a full build of `//:library` puts 34 MB in it (measured 2026-11-08), so
   the bound holds ~150 builds' outputs across every worktree of a machine, and the collector runs only after
   the server has idled `--experimental_disk_cache_gc_idle_delay` (5 minutes), so the server's `--max_idle_secs` (15
   minutes) stays above that delay -- below it the cache grows without bound.
   The repository cache's path never moves: Bazel 9 keeps every fetched repository's contents inside it
   (`contents/`), and each output base links into them, so a moved cache breaks every output base until it is
   expunged.
4. EVERY TOOL AN ACTION RUNS IS DECLARED: hermetic (the toolkit), or its identity is an input of the actions that run
   it (the host C++ compiler's version). A tool the graph does not see is one whose change shares stale results between
   machines.
5. A GPU TEST'S RESULT IS THE MACHINE'S: its key carries the device's identity, so a pass is never replayed from
   another machine; it takes the machine's GPU lock itself, since Bazel serializes tests only within one invocation and
   every worktree runs its own server.
6. NO VOLATILE STAMP IN A COMPILE: the commit and other status reach only the final records (the build record, the
   wheels' metadata), never an action whose output is reused.
7. TESTS ARE ONE TARGET A TIER OR AREA, SHARDED: a test process imports torch (seconds), so one target a file costs
   minutes; the runner honours `$TEST_SHARD_STATUS_FILE`, and each target states its timeout.
8. A RUN TARGET READS `$BUILD_WORKSPACE_DIRECTORY`: `bazel run` starts in the runfiles tree, not the checkout.
9. WORKSPACES AND OUTPUT BASES LIVE ON THE LINUX DISK, never under `/mnt` (WSL's Windows mounts: slow, and file
   watching is undefined there).
10. THE SANDBOX STAYS ON: Bazel 9's hermetic `/tmp` keeps nvcc's temporary files apart, so nothing runs
    `--spawn_strategy=local` to dodge it.
11. CUDA THROUGH RULES_CUDA'S OWN SETTINGS: the architectures through `--@rules_cuda//cuda:archs`, one spec a card
    (cubins only, no PTX); a host compile flag reaches nvcc's host pass only through rules_cuda's host options; an external repository's
    headers through the target's include options.
12. AN ACTION'S SCRIPT FINDS ITS SIBLINGS UNRESOLVED (`Path(__file__).absolute()`): in the sandbox it is a link, and
    resolving it imports the checkout's files instead of the action's declared inputs.
13. TWO SCHEDULERS SHARE ONE HOST through the host budget: the front door holds it for every build phase and never for
    a test or run phase; a caller that holds host slots never starts a Bazel build, whose actions run in a server that
    does not inherit the caller's hold.

## <a id="use"></a>Using it

    python tools/dev.py init        # installs the pinned bazelisk and links `bazel` to it (toolchain.bazel)
    bazel run //:library                                            # the shipped build, into rola_cu13/
    bazel run //:library --//:archs=86 --//:carry_arms=3            # an iteration build

Never pass `--jobs` or a cache path: the front door sets both. Nothing the graph builds lands in the checkout but the
library `bazel run //:library` copies in: Bazel's output links are off (`--experimental_convenience_symlinks=clean`).
