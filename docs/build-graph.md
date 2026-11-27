# The build graph

The build graph is Bazel, run through bazelisk and the graph's front door. Its workspace is `MODULE.bazel` (the rule
sets and the hermetic CUDA toolkit), `.bazelrc` (the graph's settings), `.bazelversion` and `.bazeliskrc` (Bazel's
release and its digest), `bazel/` (the graph's machinery: its rules, the environment, the platform and the scripts its
actions run) and `bazel/bazel` (the front door); each `BUILD.bazel` sits beside what it describes. The graph builds the
extension (`//:library`, [below](#compile)), gates it ([below](#gates)) and runs the checks ([below](#checks)).
The commit gate is `bazel test //:gate` (`tools/git-hooks/pre-commit`, on the commit as git will write it), and the
test tiers are the graph's targets too (`//:lane`, [below](#checks)), and so are the two wheels (`bazel run //:wheels`,
[`build.md`](build.md#wheels)); the census reads the graph's compiles and compiles nothing
([below](#census)). Nothing builds outside the graph: `pip install .` refuses (`tools/no_source_build.py`). CI runs the commit gate's checks through the graph
(`.github/workflows/lint.yml`, on a fresh runner as `.github/actions/graph` sets one up). A timing session is a run
target (`bazel run //measure:session_entmax`, [`measurement.md`](measurement.md#sessions)): the graph builds every
entry's bundle and the session's manifest, and the session program it starts holds the GPU, the host budget and the
clock as machine-wide locks, which Bazel cannot do for an action. An instrument's record is a run target too
(`bazel run //measure:store_oracle_tier`, [`measurement.md`](measurement.md#instruments)): one on the device runs in the
run, never in a build, which holds the host budget; fsla-bench's diffs are `fsla_devtools.build`'s.

- <a id="stamp"></a>THE BUILD IS STAMPED: `.bazelrc` names `bazel/status.sh` as the workspace status command, which
  writes the commit, the branch, the checkout's directory name and the sha256 of the uncommitted tracked diff (`git diff
  HEAD`, none when clean) as stable status keys. A timing entry's bundle manifest is the one action that reads them, so
  a new stamp rebuilds only the manifests; a git that fails fails the build, so no bundle names code it cannot name.

## <a id="environment"></a>The environment's layers

| layer | made by | stored |
|---|---|---|
| the host's NVIDIA driver and GPU | the host | -- |
| the machine (base OS libraries, git, `python3`, the host C++ compiler nvcc drives) | the host, or the devbox image (`//:image`, docs/setup.md#the-devbox-image) | the image |
| bazelisk, Bazel's launcher | fsla-devtools' `pins/bazelisk_pin.json`, installed by `python3 tools/bootstrap.py` under `host.tools_dir` by its pin (never a machine key), which links `bazel` on PATH to it (`toolchain.bazel`; `check` has the row), on a host and in the devbox container alike | `host.tools_dir` |
| Bazel | `.bazelversion`, verified against `.bazeliskrc`'s digest | bazelisk's cache |
| the Python environment | `requirements.lock`, synced with `--require-hashes` by the pinned uv (fsla-devtools' `pins/uv_pin.json`, the graph's `@uv`, which `//:dev_env` writes venvs with too) into the pinned CPython (3.11.13), and fsla-devtools from its module's source, by fsla-devtools' environment extension (`MODULE.bazel`'s `environment.python` tag, `@fsla_env`; `fsla_devtools.graph.make_env`) | `host.build_cache/bazel/store`, one environment a key (the lock, fsla-devtools' content, the Python, uv and the torch index), under a file lock; uv's cache there hardlinks the packages; `@fsla_env//:flags.bzl`'s `VENV` names it, so a command naming it is keyed by the environment; the store is bounded by `host.store_max_gb` and `host.store_max_age_days`: each environment build sweeps the idle or oversized environments and wheels, and a swept environment is rebuilt, never read wrong |
| the linker | mold at fsla-devtools' `pins/mold_pin.json`'s release (`@mold`) | the shared repository cache |
| the lints' analyzer | clang-tidy at fsla-devtools' `pins/clang_tidy_pin.json`'s wheel, its binary's digest checked (`@clang_tidy`) | the shared repository cache |
| the pinned toolkit, the one CUDA toolkit (the graph's compile takes nvcc, its headers and the runtime from it) | fsla-devtools' `pins/cuda_toolkit_pin.json`: NVIDIA's redistributable manifest by its digest, every component it names unpacked into one tree, `@cuda_tools` -- what the compile reads and every tool too, through `dev_config.cuda_home()` (`tests/unit/test_cuda_toolkit_pin.py` holds the manifest and the declared toolchain to the pin): the disassemblers the SASS gate reads with (cuobjdump, nvdisasm), compute-sanitizer (the sanitizer lane's) and the whole toolkit the test tiers hand their tests (`FSLA_CUDA_HOME`); beside it cuRAND's headers, the graph's alone (clang's CUDA front end includes one) | the shared repository cache |
| the host C++ compiler's identity | its version and the digests of gcc, g++ and cc1plus (`@host_cc`), an input of every compile | the output base |
| the devices' identity | each device's name, UUID, driver and compute capability from `nvidia-smi` (`@host_gpu`, read at every fetch; a machine that names no device refuses), an input of every device test tier | the output base |

## <a id="front-door"></a>The front door

bazelisk runs `bazel/bazel` (`bazel/front_door.py`; the wrapper directory `.bazeliskrc` names, in place of
bazelisk's default `tools/`) in place of the Bazel it downloaded, so every Bazel command
in a checkout -- typed, or run by a tool -- goes through it. It runs under the machine's `python3` before any venv
exists: a stub that runs fsla-devtools' front door (`fsla_devtools.graph.front_door`, every FSLA repository's) from the
fsla-devtools checkout `tools/bootstrap.py` finds.

- THE LAB'S IN-KERNEL STAMP AND ITS CHECKER ARE WARPLINE'S MODULE: `MODULE.bazel` pins it at a public commit (`git_override`, the one pin
  of warpline), and `bazel run //tools:lab` (bazel/defs.bzl's `lab_run`) hands `tools/lab.py` the directory of the `warpline` Python package
  (`@warpline//:warpline`) as `FSLA_WARPLINE_PYTHON`, which `tools/lab_trace.py` reads the records with and `tools/lab.py`'s `check` calls the checker
  (`warpline.checker`) through, for every hazard one of its rules computes (the CPU test
  tier and `unit_cuda` carry the same target as data). The stamp's headers (`@warpline//:stamp`, `@warpline//:host`, the writer of the record of
  a traced row) are the lab experiments' compile dependencies, compile-time only: a `lab pack` binary carries
  nothing of warpline, and the box needs none. THE LAB'S EXPERIMENTS ARE THE GRAPH'S TARGETS (`//measure/lab:<name>`,
  `bazel/lab.bzl`, [`internals/tools/lab.md`](internals/tools/lab.md#graph)): each is compiled by the same CUDA rule as the library's
  units (`cuda_unit`) for sm_90a and linked into its binary, read once by the graph's disassemblers (`tools/lab_facts.py`), and checked by
  a validation action (`tools/lab_check.py`) whose refusal fails the build; no tool runs nvcc, cuobjdump or nvdisasm for the lab outside the
  graph's actions. The front door
  overrides only fsla-devtools' pin with a local checkout; warpline is always the pinned commit.
- THE SHARED MACHINERY IS FSLA-DEVTOOLS' MODULE: its environment extension and generic rules (`tree_test`,
  `checkout_test`, `tool_test`, `tool_run`, `tree_part`, `pytest_tier`, `device_lane`; `bazel/defs.bzl` hands each
  fsla's environment). `MODULE.bazel` pins the module at a public commit (`git_override`), the one pin of
  fsla-devtools; the front door overrides that pin with the checkout it found, in an rc file of this workspace's own
  (`--override_module`), so a change made there is the one the graph runs. A machine with no checkout clones one at
  the pin first (`tools/bootstrap.py --devtools-pin`), so every graph runs on an override and `MODULE.bazel.lock` is
  the overridden resolution's (Bazel records one extension digest differently under the bare pin).

- THE MACHINE'S SETTINGS come from the dev config: the shared disk and repository caches under `host.build_cache`
  (`bazel/disk`, `bazel/repository`), in an rc file it regenerates on every run and hands Bazel after the workspace's
  (`--bazelrc`). Nothing in the checkout names a machine.
- EVERY BUILD RUNS UNDER THE HOST BUDGET (`fsla_devtools.locks.host`): a command's build phase holds the slots free
  when it starts, at least one, and runs `slots / host.nvcc_threads` jobs (nvcc runs one cicc an architecture). A
  `test` command builds under the budget, then runs its tests holding none; a `run` command writes its
  launch script under the budget (`--script_path`), then runs it holding none. A test or a measurement takes its own
  locks, as outside the graph: a GPU test under a held budget would deadlock against a measurement, which takes the
  GPU before the host's slots. A timing session holds every slot, so no build runs during one.
- IT HOLDS THE MACHINE'S DISK (`fsla_devtools.disk`, fsla-devtools' README, "The machine's disk"): a `build`, `test`
  or `run` is refused before Bazel starts while the root or, under WSL, the Windows drive WSL's virtual disk lives on
  has less free than `host.free_floor_gb` (30 GiB) -- the refusal names each filesystem, its free space, the floor and
  what removing the clean, merged worktrees would free; the override says why, `DISK_FLOOR_OVERRIDE="<why>"`. Every
  command first deletes the repository cache's downloads unused for `host.downloads_max_age_days` (30), at most once a
  day: each is pinned by its sha256 and refetched when a fetch needs it. `bazel --worktree add|remove|list` is the
  worktree tool: the machine's cap on live worktrees and a removal that expunges the output base first.
- Run directly, without bazelisk, it refuses; so does `bazel coverage`, whose instrumented build would differ from the
  budgeted one and run outside the budget (the graph has no coverage target).

## <a id="compile"></a>The extension's compile

`//:library` is `fsla_cu13/_C.abi3.so` and its build record `fsla_cu13/_build_config.py`; `bazel run //:library` leaves
both where the build wrote them (`.bazel/bin/fsla_cu13/`: a build output is never written into the source tree, and the
checkout's venv imports it from there, `docs/build.md#wheels`), and copies the build's generated headers into `build/generated`. `common/hardware.cuh` is one of the graph's generated headers and is checked in nowhere:
the genrule `//csrc/fsla:hardware_header` writes it from the registry's device parts (`hardware/emit.py --arch-header`) into
the include path every unit reads (`//csrc/fsla:headers`), whatever the build's parameters
([`internals/common/hardware.md`](internals/common/hardware.md)).

- THE BUILD'S PARAMETERS are the root package's settings: `--//:archs` (default `@fsla_env//:flags.bzl`'s
  `BUILT_ARCHS`, the registry's built architectures: those a device part of `hardware/parts/device` states,
  [`architecture/families.md`](architecture/families.md)), and `--//:contract` (default `full`: the contract the build compiles against, or a declared narrowed contract of
  `hardware/kernels/<kernel>/descriptors/dev/`, [`build.md#narrowed`](build.md#narrowed)). Any value but a default makes an iteration build, which its record says; `//:gate`
  refuses a non-default `--//:contract`.
- THE FLAGS ARE `tools/build_flags.py`'S: the environment repository writes `@fsla_env//:flags.bzl` from it
  (`bazel/flags.py`) -- the nvcc codegen flags and torch's extension defines, the host C++ flags -- and the graph states
  none of its own. the compile adds nothing of its own (no extended lambdas, compilation-mode flags, random seed or host
  toolchain flags), so nvcc's command line is the build flags, the `-gencode` targets, `-x cu`, `-ccbin` and the include roots.
- EACH UNIT IS ONE ACTION OF THE GRAPH'S OWN RULE (`bazel/cuda.bzl`'s `cuda_unit`, which runs
  `bazel/cuda_compile.py`): nvcc is the pinned toolkit's (`@cuda_tools`, the one CUDA toolkit: no second declaration), its command line written by the rule, one
  position-independent object a unit, always linked. nvcc writes a dependency file (`-MD -MF`), and every declared
  header the unit did not read becomes the action's `unused_inputs_list`, so the unit's cache key holds what it read:
  a header edit recompiles the units that include it and no other. The file
  lists what every pass read, the host's and each architecture's device pass (nvcc 13.5.37: a header included only
  under `__CUDA_ARCH__ == 900` is listed when sm_90a compiles). Only declared headers are
  prunable: the toolkit, the host compiler and its identity (rule 4) key every compile. The action keeps nvcc's
  diagnostics, ptxas's `-v` report among them, as `<unit>.ptxas.txt` (the `ptxas_reports` output group); the console
  shows them without the report's lines.
- ONE ARCHITECTURE SPEC A CARD: the library's transition writes `//bazel:cuda_archs` as
  `compute_90a:sm_90a`, one spec an architecture; a comma list is one spec, which compiles every card from the
  lowest virtual architecture.
- THE INCLUDE ORDER IS SETUP.PY'S: the repository's headers, then CUTLASS's, then the generated headers (and the stamp,
  for the one unit that reads it), then torch's, Python's and the toolkit's, so a repository header wins any name
  collision (`bazel/defs.bzl`'s `include_root`).
- THE GENERATED HEADERS are actions over their generators: the arm families' selection headers and Python tables
  (`bazel/arms.bzl`, [below](#arms)), the carried builds (`fsla_headers`, `bazel/headers.py`, from the settings) and the
  device stamp (`tools/csrc_stamp.py` over every file under `csrc/fsla/src` but its markdown). Only `build_stamp.cu`'s unit reads the stamp
  and depends on it (`stamp_unit`): a digest of every source, as any unit's input it would recompile every unit on
  any edit.
- THE UNIT LIST: every `.cu` under `csrc/fsla/src` is a hand unit (`//csrc/fsla:units`), the runtime's among them
  (`csrc/fsla/units.bzl`'s `RUNTIME_UNITS`, `//csrc/fsla:runtime`, held to `tools/kernels.py`'s `runtime_units` by
  `tests/unit/test_units_list.py`); a family's arm units are the next section's.
- <a id="arms"></a>A KERNEL'S ARMS are Bazel targets declared in Starlark: `csrc/fsla/src/<kernel>/arms.bzl` holds the arm
  key (`<FAMILY>_KEY`), the kernel's name and its declarations; the arms are literal rows with each field's domain (a family with per-arm
  units: the fixture's), or a rows file a host tool writes (every family's: `arms_gen` evaluates the family's launch function over the contract in
  `hardware/` and fails the build on a contract context no arm covers, [`internals/common/plan/arms_gen.md`](internals/common/plan/arms_gen.md)); the families are decode and the producer's four solves, one for each kernel template (`csrc/fsla/src/entmax/arms.bzl`); a key's field that is a type in the kernel's name is an integer code, and `arm_family`'s `spellings` (`{field: [the type's name for value 0, for value 1, ...]}`) says which type each code spells in the pattern the SASS gate holds the binary to; a kernel
  not built on this line declares no family. `bazel/arms.bzl`'s `arm_family` reads
  that one list and makes the family's pieces (the selection header, table and declarations written by `bazel/arm_emit.py` over the rows file): one unit per arm and built architecture (`<family>_arm_<i>_sm_<arch>`, from
  a generated source that defines the arm under its own selection guard) when the family has per-arm units; the selection
  header of the build (`<family>_selection.inc`: every row of the list, which is the arms the build's contract reaches); and the Python table (`<family>_arms.py`: the key and the rows) that the record, the arm gate and
  `bazel run //:library`'s copy into `build/generated` give the measurement. `<family>_arm_set` hands the link only the
  units the build's architectures name, so an unwanted architecture is never compiled (the arms are the contract's). A malformed
  declaration is refused at analysis, naming the field: a row of the wrong width, a value outside its domain, a duplicate,
  or rows out of canonical order (an arm's index is what the header and the unit names say, so it is the
  declaration's, not where a row was typed). `tests/BUILD.bazel`'s `fixture` family exercises the macro on three rows,
  `tests/unit/test_arm_family.py` reading what it made. `ARM_FAMILIES` is the one list of shipped families: the build's parameters, record and
  gates are made for each name in it, so a family joins by being named there and declared beside its kernel.
- `--threads` IS THE MACHINE'S `host.nvcc_threads`, which the front door hands the graph (`FSLA_NVCC_THREADS`): the
  cicc a job runs is the count its job budget divides by.
- THE LINK is the pinned mold's, of every unit's objects (each target `alwayslink`, so every object is linked)
  and, by path, torch's `libc10.so` and `libtorch_cpu.so`, the CUDA runtime and the driver's stub: each NEEDED by its
  soname, no search path written (no unit carries a dynamic library; the units' CUDA runtime is its headers,
  `//bazel:cuda_headers`, the pinned toolkit's `include/` and `include/cccl`).
- AN INSTRUMENT'S LIBRARY (`bazel/defs.bzl`'s `fsla_instrument`): a measurement's own kernels as
  `.bazel/bin/fsla_cu13/<module>.abi3.so` beside the extension, linked as it is and SASS-gated as it is (its declarations owe
  none of the source's kernels, and its probes keep only the signatures their units' declarations name,
  [`internals/tools/sass_gate.md`](internals/tools/sass_gate.md#probes)), for the architectures `--//:archs` names; never shipped, so no record or arm gate. No instrument's library is declared on this line: a family's part harness (KERNEL_STANDARDS section 19) is
  the rule's user, with the family's first schedule.
- <a id="declarations"></a>DECLARATIONS BESIDE THE KERNELS: a kernel states what its SASS and the toolkit's notices may keep in Starlark, next to its unit (`cuda_unit(declarations = [declaration(...)])`,
  a non-arm unit's) or its arms (`<FAMILY>_KERNEL`, the arm's demangled name with `@field@` for its template arguments, and `<FAMILY>_DECLARATIONS`: `arm_family` generates each arm's exact pattern for each
  built arm and architecture). The library rule writes them, with each arm and each source kernel as a presence, fsla's notes and strict on, as one file a built architecture (`bazel/gate_declarations.py`, `<label>_<arch>.declarations.json`),
  and the SASS gate's check of that architecture reads it with the units' ptxas reports and the project's notice ledger (`tools/notice_ledger.json`): [`internals/tools/sass_gate.md`](internals/tools/sass_gate.md#declarations).
- A UNIT'S SASS AS TEXT: `bazel build //csrc/fsla:<unit>_sass` ([`internals/tools/sass_gate.md`](internals/tools/sass_gate.md#sass-text)).

## <a id="gates"></a>The library's gates

Building `//:library` runs the gates below; their verdicts are its outputs and its runfiles, so neither a build nor
`bazel run` hands over a library one of them has not passed. Each reruns only when its inputs change. The census is not
among them: it is a report, on demand ([below](#census)).

- THE BUILD GATES AND THE RECORD (`bazel/record.py`): rule 1 (`tools/toolchains.py`'s `rule1`) -- the
  graph's ptxas is the toolchain record's byte for byte (the tie between that record and
  `MODULE.bazel`'s toolkit pin), its cuobjdump and nvdisasm are the record's byte for byte too (the tie between the
  pinned toolkit's manifest and the compiler's, which `tests/unit/test_cuda_toolkit_pin.py` holds equal), and the environment's torch is the toolchain's CUDA major --; rule 3, every architecture a built one
  (`tools/build_record.py`'s `rule3`); rule 5, the vendored tree its `PIN.json`'s (`tools/build_record.py`'s
  `vendored_pin`); then the record, `tools/build_record.py`'s writer.
- THE SASS GATE (KERNEL_STANDARDS section 20, `bazel/defs.bzl`'s `_sass_gate`) on the linked library: warpline's check
  (`@warpline//:checker.bzl`'s `warpline_check_actions`) once a built architecture, with the graph's `cuobjdump`, `nvdisasm`
  and `cu++filt` (`@cuda_tools`, the pinned toolkit): the checker's manifest of every function and of the units' ptxas notices
  (a build output, `//:static_manifest`), then its comparison with the declarations ([above](#declarations)); and a PTX image
  refused: closed-world rule 2 on the binary itself ([`internals/tools/sass_gate.md`](internals/tools/sass_gate.md#graph)).
- THE ARM GATE (`bazel/gates.py`, `bazel/arms.py`): the built arm table against the declared arms, and the coverage sweep -- the binary asked
  what it carries, loaded by the environment's interpreter against the toolkit's driver stub, so the answer is the
  binary's on any machine. It runs outside the sandbox, where the environment's packages are, the environment's
  store path on its command line (the path carries the environment's key).

## <a id="census"></a>The census

`bazel run //:ratify -- --census` reports every shipped instantiation's registers and spills, read from the graph's own
compiles ([`ratification.md`](ratification.md)): `tools/ratify.py --graph` in the checkout, `--json PATH` for the raw
record. The census (`//:census`, `bazel/defs.bzl`'s `ratify_census`) is each unit's ptxas report and object (the
`cuda_unit` rule's) with the graph's assembler and disassemblers; the scope is
the build's parameters, and the census's disassemblers run a unit a host-budget slot (the machine's budget under
`bazel run`, a pool of the sandbox's own under a sandboxed test). `bazel test //csrc/fsla:unused_instantiations` reads the census for
the template kernels no unit instantiates.

THE CLOSED WORLD IS THE DECLARED TARGETS AND THE GATES EVERY BUILD RUNS: a kernel's arms (`arms.bzl`), the built
architectures (`BUILT_ARCHS`) and the pinned toolchain are the set; rules 1, 3 and 5, the SASS gate and the arm gate
are what refuses a build outside it. No committed record stands beside them, and a wheel is a clean commit's build
whose record is no iteration build.

## <a id="checks"></a>The checks

`bazel test //:gate` runs every check the graph holds -- `//:lints`, the checks over files (the repo's own and
`//csrc/fsla:lints`); `//tools/lint:lint_fixtures`, each lint firing on its own fixture; `//tests:lab`, the lab's unit
tests; and `//:commit_checks`, the checks of the commit -- and is the commit gate (before it, `tools/git-hooks/pre-commit`
runs the content checks over the staged diff, `bazel run //:tripwire -- --staged`, and `tools/git-hooks/pre-push`
over every commit pushed, `-- --range`: the full history is public, so a home path, another person's email, an IPv4
address, a credential format, a provider id or a ledger event id is refused before it is history):
`tools/git-hooks/pre-commit` runs it with the changes not staged for the commit set aside, and the tree the commit's
index (`GIT_INDEX_FILE`, which `@fsla_tree` reads when the hook hands it in).

- THE TREE is `//:tree`: every tracked file (`git ls-files`, read by the environment extension's `@fsla_tree`, which
  reads again when git's index moves), each package's part its `tracked` filegroup (`bazel/defs.bzl`'s `tree_part`,
  which names its Python files `tracked_python` and the parts a check reads); `//:tree_python` is every Python file.
  A file is in the tree once git tracks it.
- A CHECK OVER FILES IS A TEST OVER THE PART OF THE TREE IT READS (`bazel/defs.bzl`'s `tree_test`): its `data` names
  the parts it reads -- a package's tracked files, the Python files, a named directory, a configuration file -- or the
  whole tree for a check of every tracked file, so an edit reruns only the checks that read the edited file. Its
  inputs are copied into a real tree in the test's own temporary directory, the environment's tools on
  PATH and no machine's dev config; a check that asks git what is tracked gets the copy as a git index of exactly
  those files. Cached on what it reads, it reruns only when one of those files changes.
- THE LINTS live beside what they check, and the linters in `tools/lint` (its package: the scripts, their
  configuration, the modes a person runs by hand and each linter's self-test on its fixtures). `//csrc/fsla:lints` are
  the kernel tree's own: clang-format, ast-grep (`tools/lint/ast_grep_gate.py`), the template kernels no unit instantiates and the
  unit lints below. `//:lints` adds the repo's policies over the whole tree: ruff and its blank-line spacing, the
  standards lint (`tools/lint/lint_standards.py`), the readability spacing, the agent files' portability
  (`tools/lint/portable_claude.py`), the path declarations (`fsla_devtools.mirror --check`), and each ratchet
  (`tools/lint/ratchet.py <lint>`: the lint's findings against its baseline, a new finding and a fixed one each red).
- THE UNIT LINTS (`//csrc/fsla:clang_tidy_device`, `clang_tidy_host`, `host_warnings`; `bazel/defs.bzl`'s `unit_lints`
  aspect and `unit_lint_test`): clang-tidy's two profiles and the host warnings, each run on one unit's own compile
  action -- its command line and its inputs -- so a lint reads what the build compiled
  ([`internals/tools/lint.md`](internals/tools/lint.md#unit-lints)). The units are the library's own (the aspect follows
  its deps), read in one configuration a built architecture (a split transition), whose generated headers are that architecture's build set; each verdict is
  cached on its unit's compile, and states the seconds it took. `bazel build //:compile_commands` makes the editor's
  compile database (`.bazel/bin/compile_commands.json`, which `.clangd` names) from the same commands, for the first built architecture.
- THE TEST TIERS (`//tests:cpu`, `unit_cuda`, `oracle`, `integration`; `//:lane` all five, with `//tests:lab`; `bazel/defs.bzl`'s
  `pytest_tier`; `//tests:lab`, the lab's and the dev config's unit tests on their own, is in the gate, its checker the
  `@warpline` module's): each tier one target, as pytest runs it ([`testing.md`](testing.md#running-it)), over the tree and the
  built library copied into a real tree, outside the sandbox (the machine's locks live in `/tmp`). The CPU tier is
  sharded by Bazel's protocol (`fsla_devtools.graph.pytest_shards`), each shard drawing a host slot, and sees no device; a device
  tier runs alone, takes the machine's GPU lock itself, and has the devices' identity among its inputs (`@host_gpu`).
  `--test_filter` is pytest's `-k`, `--test_arg` passes pytest an argument. Every tier reads the machine's dev config
  (its locks), and hands its tests the graph's copy of the pinned toolkit (`@cuda_tools`, an input of the tier, so a
  new toolkit reruns it) as `FSLA_CUDA_HOME`, which `dev_config.cuda_home()` returns; outside a tier it asks the graph for the same repository (`bazel query --output=location @cuda_tools//:bin/nvcc`). The sanitizer lane (`//tests:sanitizer`,
  `bazel/defs.bzl`'s `device_lane`, `manual`) runs as a device tier does, passing the toolkit's own compute-sanitizer
  by its path ([`testing.md`](testing.md#the-sanitizer-lane)).
- THE CHECKS OF THE COMMIT (`//:commit_checks`, `bazel/defs.bzl`'s `checkout_test` and `docrefs_test`): what git
  knows rather than a file set -- no ratchet's baseline grown against HEAD's (`tools/lint/ratchet.py --grown`), every
  doc reference pinned, read, current and resolving (`//:docrefs`, fsla-devtools' `fsla_devtools.docrefs`: nothing
  pending, unread or broken), the devbox image the one the tree names and checked (`tools/dev.py container
  gate`, which builds `//:image`). Each runs in the
  checkout, outside the sandbox and never cached, its input the repository's state.

## <a id="committed-files"></a>Committed files the graph writes

A file the repository commits that a tool writes (a generated block, a record of a measurement the graph's own build makes, a name) is ONE
pattern, `bazel/defs.bzl`'s `committed_file(name, committed, generated)`: the file is the output of a graph action (`generated`), and the
pattern makes two targets from it. `bazel run //:<name>_update` writes the committed file from that output; the test
`//:<name>` (`tools/committed_file.py`, in `//:commit_checks`, so the gate) fails when the committed file differs from what the graph would
write, showing the difference and the update command. No tool writes a committed file by a mode of its own, and the file carries no
date or host, so the same tree writes the same bytes.

| committed file | the graph writes it | the test | the update |
|---|---|---|---|
| `README.md`'s generated block (the support table) | `//:readme_generated`: `tools/supported.py --readme` over the README, the contract and the device parts (`hardware/`), the toolchain record | `//:readme` | `//:readme_update` |
| `tools/devbox_image.json` | `//:devbox_image_record`: `//:image`'s content digest (`tools/oci_layout.py --record`) | `//:devbox_image` | `//:devbox_image_update` |

`common/hardware.cuh` is not a committed file: it is a graph output the units include, and nothing checks a copy in. A change to the
image's name still carries a passing container check (`tools/dev.py container gate`, `//:dev_container`).

## <a id="packages"></a>The packages

The tree keeps the layout of a Python package with a native extension, as FlashAttention and PyTorch do (ruled
2026-11-08): `fsla/` beside `csrc/`, and the tests that drive the kernels through the op against the oracle in `tests/`,
since the kernels serve the op alone and those tests grade the two as one. A BUILD file sits beside what it describes,
its lints among it, and `bazel/` holds only the graph's own machinery:
- the root package: the build's parameters, the extension, the devbox image, the
  repo-wide checks and the gate;
- `csrc/fsla`: the kernel tree, its units and its own lints; `csrc/third_party`: the vendored trees;
- `tools`: the pins and records the graph reads, the generators it runs, and each tool's run target; `tools/lint`: the
  linters, the modes a person runs by hand and each linter's self-test;
- `tests`: the tiers.

A C++ API of the kernels' own would make a split by language right.

## <a id="rules"></a>The rules

Each is a trap a setup like this falls into, and what avoids it (checked against Bazel 9.3.1's own options where it
names one).

1. BZLMOD ONLY, PINNED: `MODULE.bazel` and its `MODULE.bazel.lock` are committed with `.bazelversion`; the commit gate
   runs with `--lockfile_mode=error` (`tools/git-hooks/pre-commit`), so a dependency resolved differently from the
   lock refuses the commit rather than drifts.
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
   machines. An identity is never a header: a compile drops a declared header it did not read from its key (rules_cc's
   C++ compile by its dependency file, as the CUDA units by theirs: a header nothing includes was edited and nothing
   recompiled, 2026-11-08), so `@host_cc//:identity.txt` is a file each compile takes outright (`host_cc` on a CUDA
   unit, `additional_compiler_inputs` on a C++ one).
5. A GPU TEST'S RESULT IS THE MACHINE'S: its key carries the devices' identity (`@host_gpu`: each one's name, UUID,
   driver and compute capability, read at every fetch), so a pass is never replayed for another device; it takes the machine's GPU lock itself, since Bazel serializes tests only within one invocation and
   every worktree runs its own server.
6. NO VOLATILE STAMP IN A COMPILE: the commit and other status reach only the final records (the build record, the
   wheels' metadata), never an action whose output is reused.
7. TESTS ARE ONE TARGET A TIER OR AREA, SHARDED: a test process imports torch (seconds), so one target a file costs
   minutes; the runner honours `$TEST_SHARD_STATUS_FILE`, and each target states its timeout.
8. A RUN TARGET READS `$BUILD_WORKSPACE_DIRECTORY`: `bazel run` starts in the runfiles tree, not the checkout.
9. WORKSPACES AND OUTPUT BASES LIVE ON THE LINUX DISK, never under `/mnt` (WSL's Windows mounts: slow, and file
   watching is undefined there).
10. THE SANDBOX STAYS ON: Bazel 9's hermetic `/tmp` keeps nvcc's temporary files apart, so nothing runs
    `--spawn_strategy=local` to dodge it. Three kinds of work leave it: the arm gate, because it imports the
    environment's packages from the store, which the sandbox does not hold (the store path on its command line carries
    the environment's key); the checks of the commit, whose input is the checkout itself (rule 15); and the test
    tiers, which take the machine's locks in `/tmp`, a sandbox's own.
11. CUDA THROUGH THE RULE'S OWN SETTINGS: the architectures through `//bazel:cuda_archs`, one spec a card
    (cubins only, no PTX); an external repository's headers through the target's include options.
12. AN ACTION'S SCRIPT FINDS ITS SIBLINGS UNRESOLVED (`Path(__file__).absolute()`): in the sandbox it is a link, and
    resolving it imports the checkout's files instead of the action's declared inputs.
13. TWO SCHEDULERS SHARE ONE HOST through the host budget: the front door holds it for every build phase and never for
    a test or run phase; a caller that holds host slots never starts a Bazel build, whose actions run in a server that
    does not inherit the caller's hold.
14. A CHECK OVER FILES READS A COPY: the runfiles are links into the checkout, so a tool that resolves a path, walks
    a directory or asks git reads the checkout instead of the check's inputs; `tree_test` copies them into a real tree
    (a git index of exactly them where the check asks git), and the test's own script is not among them.
15. A CHECK OF THE COMMIT IS NEVER CACHED: its input is the repository's state (HEAD, the index, the commits a doc
    pins), which no file set names, so it runs in the checkout tagged `external` -- never as a cached test that
    passes on a stale verdict.

## <a id="use"></a>Using it

    python3 tools/bootstrap.py                                      # the pinned bazelisk, `bazel` linked to it
    bazel run //:init                                               # the machine's dev config and hooks
    bazel run //:dev_env                                            # this checkout's venv, which its hooks run from
    bazel test //:gate                                              # every check the graph holds
    bazel build //:compile_commands                                 # the editor's .bazel/bin/compile_commands.json
    bazel test //:lane                                              # every test tier
    bazel run //:library                                            # the shipped build, in .bazel/bin/fsla_cu13/
    bazel run //:library --//:archs=90a --//:contract=decode_d32_l2   # an iteration build
    bazel run //:image_load                                         # the devbox image, into Docker (docs/setup.md)
    bazel run //:image_push                                         # the devbox image, published to Docker Hub
    bazel run //tools:<tool> -- <its arguments>                     # a tree tool, in the checkout on its own venv
    bazel run //tools/lint:ratchet -- <lint> --shrink               # a ratchet's baseline, shrunk with its fixes
    bazel run //:docrefs -- accept <doc>                            # a doc's citations pinned, read against their targets
    bazel run //measure:session_entmax                              # a timing session, measured and stored
    bazel run //measure:store_oracle_tier                           # an instrument on the device, run and recorded
    bazel run //measure:store_static_manifest                       # a build output recorded, unless the store holds it
    bazel run //:results -- show|history|latest|verdict|commit      # the measurements store
    bazel run //:tripwire -- --tree                                 # the content checks over every tracked file of HEAD

Never pass `--jobs` or a cache path: the front door sets both. Nothing the graph builds lands in the checkout but what a
run target writes there -- the library `bazel run //:library` copies in, with the build's generated headers into
`build/generated` (where the tools that compile the tree's units read them: `tools/life_ranges.py`;
a header only when its text changed), and nothing else: the editor's compile database is a build output (`bazel build //:compile_commands`, `.bazel/bin/compile_commands.json`), reached through Bazel's output links, all under one ignored directory (`--symlink_prefix=.bazel/`: `.bazel/bin`, `.bazel/out`, `.bazel/testlogs`; `/.bazel/` ignored), so the root holds no `bazel-*`; a committed generated file is a `committed_file()`. The tools' caches (`.ruff_cache`, `.pytest_cache`, `.hypothesis`, `__pycache__`) are never written there either: `.bazelrc` sets `PYTHONDONTWRITEBYTECODE` and `RUFF_NO_CACHE` for every run and test, pyproject's `addopts` turns pytest's cache provider off, and the tests' conftest turns hypothesis's example database off.
