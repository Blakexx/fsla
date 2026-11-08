# The build graph

ONE MECHANICAL WAY TO RUN EVERYTHING: a Bazel graph builds the extension, generates its headers, runs every lint, test
and gate, and produces the two wheels; the measurement engine (`rola_devtools.build`) runs measurement, started from
the graph. What a graph can hold is not tracked in docs, memory or ad-hoc scripts.

TODAY the graph holds its workspace and the hermetic CUDA toolchain (`MODULE.bazel`, `.bazelrc`, `tools/bazel/`); the
compile, the lints, the tests, the gates and the wheels move into it one stage at a time, each proven against the path
it replaces, and until a stage lands that path is [`build.md`](build.md)'s and [`testing.md`](testing.md)'s.

## <a id="two-systems"></a>Two systems, one seam

| what | where | why there |
|---|---|---|
| the compile, the generated headers, every lint, the tests, the build and binary gates, the wheels | Bazel | content-keyed actions with one cache shared by every worktree; a commit runs what its change reaches |
| measurement: timing sessions, the cross-checkout diffs, the GPU instruments, the stores | `rola_devtools.build` | every run executes and appends every sample to rola-results; the GPU and clock are counting semaphores held machine-wide; workers run in each checkout's own environment; checkouts are compared with each other -- none of which Bazel does (it cannot count a GPU for an action) |

THE SEAM IS THE LIBRARY: the graph builds `_C.abi3.so` and copies it into the checkout (a copy: a diff side refuses a
library imported from outside its checkout), and a `bazel run` target starts a measurement root, whose library node
reads the copy's path and sha256 into its dependents' keys.

## <a id="environment"></a>The environment's layers

| layer | made by | stored |
|---|---|---|
| the host's NVIDIA driver and GPU | the host | -- |
| the machine (base OS libraries, git, bazelisk, the host C++ compiler nvcc drives) | the Docker image | the image |
| the CUDA toolkit | the graph: NVIDIA's redistributable components at the toolchain pin's version (`MODULE.bazel`'s `cuda.redist_json`: release 13.0.0, nvcc 13.0.48), only the compiler's (~75 MB of the 4.7 GB set) | the shared repository cache |
| the Python environment | `requirements.lock` (uv-compiled, hash-pinned) | one copy a lock, shared by every worktree |
| the pinned tools (clang-tidy, mold, uv) | their pins | the shared repository cache |
| Bazel itself | `.bazelversion`, through bazelisk | bazelisk's cache |

## <a id="rules"></a>The rules

Each is a trap a setup like this falls into, and what avoids it (checked against Bazel 9.3.1's own options where it
names one).

1. BZLMOD ONLY, PINNED: `MODULE.bazel` and its `MODULE.bazel.lock` are committed with `.bazelversion`; a gate builds
   with `--lockfile_mode=error`, so a dependency resolved differently from the lock fails rather than drifts.
2. NO MACHINE IN THE SETTINGS: `.bazelrc` holds the graph's settings and names no path; a machine's own (the shared
   disk and repository caches) live in the git-ignored `.bazelrc.user` it imports. Startup options are never inside a
   `--config`.
3. THE DISK CACHE IS BOUNDED, AND ITS BOUND RUNS: `--experimental_disk_cache_gc_max_size` caps the shared cache, and the
   collector runs only after the server has idled `--experimental_disk_cache_gc_idle_delay` (5 minutes), so the
   server's `--max_idle_secs` stays above that delay -- below it the cache grows without bound.
4. EVERY TOOL AN ACTION RUNS IS DECLARED: hermetic (the toolkit, the Python environment, the pinned tools), or its
   identity is an input of the actions that run it (the host C++ compiler's version). A tool the graph does not see is
   one whose change shares stale results between machines.
5. A GPU TEST'S RESULT IS THE MACHINE'S: its key carries the device's identity, so a pass is never replayed from
   another machine; it takes the machine's GPU lock, since Bazel serializes tests only within one invocation and every
   worktree runs its own server.
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
    headers (torch's) through the target's include options.
12. TWO SCHEDULERS SHARE ONE HOST: Bazel never runs while a measurement holds the GPU; its CPU and memory
    (`--local_resources=cpu=...,memory=...`) come from the host's budget; the library is built before any measurement
    holds the GPU.

## <a id="use"></a>Using it

    bazel build //...                 # through bazelisk, which reads .bazelversion
    bazel build //tools/bazel:toolchain_smoke --@rules_cuda//cuda:archs=sm_80,sm_86

The shared caches' paths are in `.bazelrc.user`, beside the checkout and never committed.
