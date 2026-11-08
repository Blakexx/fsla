# The build graph

The build graph is Bazel, run through bazelisk and the graph's front door. Its workspace is `MODULE.bazel` (the rule
sets and the hermetic CUDA toolkit), `.bazelrc` (the graph's settings), `.bazelversion` and `.bazeliskrc` (Bazel's
release and its digest), `bazel/` (the graph's machinery: the target platform) and `bazel/bazel` (the front
door). The extension's build is `setup.py`'s ([`build.md`](build.md)); measurement is `rola_devtools.build`'s
([`measurement.md`](measurement.md)), whose timing sessions hold the GPU and the clock as machine-wide counting
semaphores, which Bazel cannot do for an action.

## <a id="environment"></a>The environment's layers

| layer | made by | stored |
|---|---|---|
| the host's NVIDIA driver and GPU | the host | -- |
| the machine (base OS libraries, git, `python3`, the host C++ compiler nvcc drives) | the host, or the Docker image | the image |
| bazelisk, Bazel's launcher | `tools/bazelisk_pin.json`, installed by `python tools/dev.py init` (`toolchain.bazelisk`) | `host.tools_dir` |
| Bazel | `.bazelversion`, verified against `.bazeliskrc`'s digest | bazelisk's cache |
| the graph's CUDA toolkit | NVIDIA's redistributable components at release 13.0.0 (`MODULE.bazel`'s `cuda.redist_json`: nvcc 13.0.48), only the compiler's (~75 MB of the 4.7 GB set) | the shared repository cache |

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
  `test` or `coverage` command builds under the budget, then runs its tests holding none; a `run` command writes its
  launch script under the budget (`--script_path`), then runs it holding none. A test or a measurement takes its own
  locks, as outside the graph: a GPU test under a held budget would deadlock against a measurement, which takes the
  GPU before the host's slots. A timing session holds every slot, so no build runs during one.
- Run directly, without bazelisk, it refuses.

## <a id="rules"></a>The rules

Each is a trap a setup like this falls into, and what avoids it (checked against Bazel 9.3.1's own options where it
names one).

1. BZLMOD ONLY, PINNED: `MODULE.bazel` and its `MODULE.bazel.lock` are committed with `.bazelversion`; a gate builds
   with `--lockfile_mode=error`, so a dependency resolved differently from the lock fails rather than drifts.
2. NO MACHINE IN THE CHECKOUT: `.bazelrc` holds the graph's settings and names no path; the machine's come from the dev
   config through the front door. Startup options are never inside a `--config`.
3. THE DISK CACHE IS BOUNDED, AND ITS BOUND RUNS: `--experimental_disk_cache_gc_max_size` caps the shared cache at
   20 GB (the WSL disk lives on the Windows system drive, shared by every worktree), and the collector runs only after
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
11. CUDA THROUGH RULES_CUDA'S OWN SETTINGS: the architectures through `--@rules_cuda//cuda:archs` (cubins only, no PTX);
    a host compile flag reaches nvcc's host pass only through rules_cuda's host options; an external repository's
    headers through the target's include options.
12. TWO SCHEDULERS SHARE ONE HOST through the host budget: the front door holds it for every build phase and never for
    a test or run phase; a caller that holds host slots never starts a Bazel build, whose actions run in a server that
    does not inherit the caller's hold.

## <a id="use"></a>Using it

    python tools/dev.py init                                        # installs the pinned bazelisk
    ln -s "$(python tools/dev.py get toolchain.bazelisk)" ~/.local/bin/bazel
    bazel build //...

Never pass `--jobs` or a cache path: the front door sets both.
