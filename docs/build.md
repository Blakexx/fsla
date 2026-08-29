# Build — ahead of time, closed world

## The governing constraint

The kernels are compiled **ahead of time** into a fatbin containing
`code=sm_XX` images only. There is no JIT, no PTX for the driver to compile, and no
runtime autotuning. Everything the binary can execute has been assembled by a
`ptxas` whose output was measured and committed
([`ratification.md`](ratification.md)).

That is a deliberate trade. A JIT-and-autotune build (Triton's model) adapts to
hardware nobody tested, at the cost of shipping codegen nobody looked at; the
supported matrix then means "we expect it to work". Here the supported matrix
**is** the ratified set, by construction, and the build refuses to widen it
silently.

## Building

```bash
pip install -e . --no-build-isolation
```

`--no-build-isolation` is required: pip's isolation would fetch a CPU-only torch
from PyPI and build the extension against the wrong ABI. Flash-attention and
mamba-ssm require the same thing for the same reason.

| Environment variable | Effect |
|---|---|
| `ROLA_CUDA_ARCHS` | `;`-separated compute capabilities, default `80;86`. Naming one without a ratified manifest **fails before anything compiles** (rule 3). |
| `MAX_JOBS` | Parallel compile jobs. The default is derived from **available RAM**, not core count: `MAX_JOBS=8` was OOM-killed on a 23 GB / 16-core host after 8m32. The divisor is a measured GB-per-`cicc` constant in `tools/build_flags.py`, so the same source gives 7 jobs on a 23 GB host and 4 on a 16 GB-class one. |
| `ROLA_NVCC_THREADS` | `nvcc --threads`, default 2 = the number of `-gencode` targets. Beyond the target count it buys nothing and multiplies the resident set, since the two parallelism axes multiply. |
| `ROLA_CARRY_ARMS` | A `,`/space list of INDICES into the carry family's arm list (`tools/gen_shards.py`'s `CARRY_ARMS`), or the word `all`. **The subset is a FILE subset** (K36): each selected row contributes its generated `csrc/rola/src/instantiations/carry_arm_<i>.cu` and a `-DROLA_CARRY_BUILD_<i>`, so an arm outside the set is not compiled at all. **The default is the SHIPPED rows** — see [the arm lists](#carry-arm-lists). `all` is the battery's set (shipped + test). A build whose arm set is not the shipped set is **NOT SHIPPABLE**: the build prints a banner and skips the two shippability gates, because the manifest ratifies the shipped set and a binary carrying a different set is not the thing that ships, in either direction. It is self-identifying at run time: `rola.ops.carry.arms()` lists what was built and every entry point refuses an arm the binary does not carry. Pair it with `ROLA_CUDA_ARCHS=86` for iteration. |
| `ROLA_SKIP_POST_BUILD_RATIFY` | Skips the post-build gate. For iterating on non-kernel code only; a shippable binary is one that passed it. |
| `ROLA_STRICT_MANIFEST` | (K43) `0`/unset (default): the two post-build gates (the fatbin's manifest-describes-binary sub-check, `tools/ratify.py`'s own re-run) REPORT a mismatch against the committed `tools/manifests/sm_XX.json` and the install still succeeds — KERNEL_STANDARDS.md §16 makes that manifest stale between milestones by design, so failing the install on it would fail every ordinary stage build. `1`: restores the pre-K43 hard failure. **The milestone/CI gate build sets it** — it is the build that immediately precedes `tools/ratify.py --write`, so it is the one build that must fail loudly if the tree it is about to ratify is not actually shippable yet. Structural checks (the shard partition actually matching the arm list, the dispatch's link closure) are never affected by this variable — those are properties of the build's own source/binary pair, not of the ratified snapshot, and stay fatal unconditionally. |
| `ROLA_SCCACHE` | `auto` (default): use the pinned `sccache` if it resolves and matches; else build uncached. `1`: REQUIRE it -- a missing or mismatched wrapper is a hard failure. `0`: disable. See [below](#sccache). |
| `ROLA_SCCACHE` | `auto` (default, K39): REQUIRE the pinned `sccache` -- a missing or mismatched wrapper is a hard failure, no uncached fallback. `1`: a synonym of the default, kept for existing invocations. `0`: an EXPLICIT, deliberate opt-out that builds uncached. See [below](#sccache). |
| `ROLA_SCCACHE_BIN` | Overrides the `sccache` binary `ROLA_SCCACHE` resolves, the same way `CUDA_HOME` overrides `nvcc`/`ptxas`. |

### Running a build under the shared lock (`tools/build/build_lock.sh`)

An **iteration build** (single-arch, `ROLA_CARRY_ARMS` subset) and a **gate build**
(both arches, the shipped set) both compete for the same host RAM `MAX_JOBS` is
sized against; running one of each at once can OOM the host (K37's own §14
addendum finding, one level up: concurrent GPU-lock work already did this once
for a different resource). K39 (Blake ruling, 2026-08-28) unifies what used to
be two separate mechanisms -- `N` parallel iteration slots and a wholly
separate exclusive lock file -- into `tools/build/build_lock.sh [--gate] --
<cmd>`: an iteration build takes ONE of `ROLA_BUILD_SLOTS` (default 2) slot
locks; a gate build (`--gate`) takes EVERY slot before running, so the two
disciplines share one set of lock files and cannot overlap.

```bash
# iteration: single-arch, arm-subset
tools/build/build_lock.sh -- env ROLA_CUDA_ARCHS=86 ROLA_CARRY_ARMS=all \
  pip install -e . --no-build-isolation

# gate: both arches, the shipped set
tools/build/build_lock.sh --gate -- pip install -e . --no-build-isolation
```

## What the build does, in order

1. **Rule 2 — gencode.** Emits `-gencode arch=compute_XX,code=sm_XX` only. A
   `code=compute_XX` entry would embed PTX and let the driver just-in-time compile
   for an architecture whose register and spill behaviour was never measured.
2. **Rule 3 — ratified archs.** `tools/manifests/sm_XX.json` must exist and be
   non-empty for every arch being built. A file-existence question, asked first,
   because the cheapest check is the one that must never be tempting to skip.
3. **Rule 1 — toolchain.** The host `ptxas --version` must equal the string the
   manifests pin, whitespace-normalized. The whole output, not a parsed release
   number: the build string distinguishes two shipments of one nominal release, and
   it is exactly those that can move an allocation while the version says nothing
   changed.
4. **Shards.** The checked-in instantiation shards must be the ones
   `tools/gen_shards.py` produces. Generated sources are committed so that what is
   reviewed is what compiles; a stale one fails here rather than shipping a matrix
   nobody wrote.
5. **Compile.** `ninja`. Both instantiation matrices are generated translation
   units under `csrc/rola/src/instantiations/`: the facts family in one shard, and
   **one translation unit per carry arm** (K36), so the arms parallelize across
   `MAX_JOBS` processes instead of serializing inside one `cicc`.
6. **The fatbin gate.** `tools/fatbin_entries.py --check` reads the built `.so` and
   requires exactly one device entry per instantiation per architecture, each in
   the shard the matrix assigns it, and every ratified name present. This is the
   only gate that can see a *failed* split: duplicated device code leaves the build
   green, the manifest byte-identical and `nm` reporting one symbol per arm,
   because a `__global__` template's host stub is weak and the linker keeps one.
   Only its "every ratified name present" sub-check is a comparison against the
   committed manifest; the split/duplication check above it is not, and stays a
   hard failure regardless of `ROLA_STRICT_MANIFEST` (K43, below).
7. **Post-build ratification.** `tools/ratify.py --arch …` for exactly the archs
   just built — the pre-build gates prove the *inputs* were ratified; this proves
   the codegen still matches what was measured.
8. **`rola/_build_config.py`.** Generated, carrying the archs, the assembler, the
   torch version and ABI flag, the manifest digest and per-arch entry counts, and
   `ptx_jit_fallback: False`.

**K43 — steps 6 and 7 REPORT between milestones, and are strict at ratify.**
KERNEL_STANDARDS.md §16 rules that the
committed manifest goes stale between milestones by design — a stage lands new
instantiations, moves SASS bodies, and does not re-ratify. Failing an ordinary
`pip install -e .` on that routine drift would make every stage build unusable, which is
what K37 actually hit (`FINDINGS.md` `## K37`): a fresh `pip install -e .` of the SHIPPED
set, on a tip already carrying stage work past the last ratification, hard-failed with
counts like "35 manifest entries absent from the build, 29 new entries not in the
manifest" and "8 of 444 ratified bodies moved" — a true finding about staleness, not a
regression. So: by default (`ROLA_STRICT_MANIFEST` unset or `0`), a manifest mismatch in
either step 6's "every ratified name present" sub-check or step 7's `ratify.py` re-run
prints a `WARNING` banner with the counts and the install **succeeds**; the binary is not
certified shippable, and says so. `ROLA_STRICT_MANIFEST=1` restores the pre-K43 hard
failure — **the milestone/CI gate build sets it**, since that is the one build that
immediately precedes `tools/ratify.py --write` and must fail loudly if the tree is not
actually shippable. This never weakens `tools/ratify.py` itself (`--write`/`--check`
invoked directly are unaffected) and never softens step 6's *structural* sub-checks (the
shard partition actually matching the arm list, the dispatch's link closure) — those are
properties of the build's own source/binary pair, not of the ratified snapshot, and a
failure there is a real defect regardless of any manifest.

## What it costs

MEASURED on a 16-core / 23 GB host, clean build directory, both architectures,
`MAX_JOBS` at the derived default of 7:

| | wall |
|---|---|
| clean rebuild, every gate including the post-build ratification | **7m20s / 7m08s** (two runs) |
| edit inside one shard's arms — one shard, relink, shard and fatbin gates | **49 s** |
| no source change — gates, ratification, `_build_config.py` | **3m47s** |
| one shard, one `nvcc`, both architectures | **43 s** |

## <a id="compile-ledger"></a>The per-TU compile ledger (K36)

**Instantiation cost is the dominant build resource, and after K36 it is not the
kernels that dominate — it is the tensor library's header.** Measured per `cicc`,
sm_86, one `nvcc --threads=1`, cold:

| translation unit | wall | peak RSS |
|---|---|---|
| a bare CUDA TU (`cuda_runtime` + `cuda_bf16`, one trivial kernel) | 0.86 s | 0.19 GB |
| the same, plus `#include <torch/extension.h>` | **33.7 s** | **2.86 GB** |
| one carry arm, on the torch-free ABI | **4.7 s** | **0.33 GB** |
| `carry_arm_6.cu` (the two-row shared TU, channel-split body) | 1.2 s | 0.21 GB |
| `carry/carry.cu` (dispatch only) | 36.2 s | 2.88 GB |

An arm's own codegen is about 6 s; the header is about 34 s. That measurement is why
`csrc/rola/src/carry/carry_arm_abi.cuh` speaks POINTERS and not tensors — with one TU
per arm a tensor-typed seam would re-pay the header 22 times (~12 minutes of redundant
parsing) and pin every arm TU's resident set at 2.9 GB, which is the term `MAX_JOBS` is
derived from. The reasoning is in
[`docs/internals/carry/carry_arm_abi.md`](internals/carry/carry_arm_abi.md#pointers-not-tensors).

**Whole-build walls**, both architectures unless stated, sccache OFF, `MAX_JOBS=3`:

| | before K36 | after K36 |
|---|---|---|
| full build, all 23 carry arms | **15m02s** | **4m01s** |
| the SHIPPED gate build (14 arms) | — (no shipped set) | **3m48s** |
| all 23 arms, sm_86 only, `MAX_JOBS=7` | — | **3m12s** |
| `ratify`, both arches, cold | **935 s** | **62 s** |
| `ratify`, no source change | 935 s (no incrementality) | **28 s** |
| `ratify`, one arm touched | 935 s | **25 s** |
| worst single-TU peak `cicc` RSS | **6.14 GB** (`carry.cu`, 23 arms) | **2.92 GB** (`factor.cu`) |

Before K36 the carry family was one translation unit holding every arm: one `cicc`
compiled them serially at 6.14 GB, with no way to spend a second core on it. `ratify`
recompiled that same TU on every run, which is what made it a 15-minute act and
therefore a milestone act rather than a stage one; it is now per-TU incremental, keyed
on the argv and the content of every input the compiler reported reading.

The two clean rebuilds are byte-identical in every entry point's SASS
(`tools/sass_bodies.py`), so those numbers are the same binary twice.

## <a id="carry-arm-lists"></a>The carry arm lists: shipped and test

`tools/gen_shards.py` owns the carry arm list and tags each row. A row tagged
`TEST` in the generated `carry_arms.inc` is a **conformance cell**: the fp64
oracle materializes `[B,T,H,N]`, so `N = 4096` is where a whole cell is
checkable, and the `W = 64` arms exist so that it is. The two uncarved
`DENSE_BOTH` controls at the flagship cell are test rows for the same reason —
they exist to be compared against, not to be shipped.

| | rows | built by |
|---|---|---|
| **SHIPPED** | the flagship topology: the three containers at both value widths, their park twins, the tied arm, the dense declarations on the carved body, flat ALT at both admitted windows | a default build; the gate build; the ratification manifest |
| **TEST** | every row at `W = 64` or `N = 4096`, plus the two `DENSE_BOTH` controls | `ROLA_CARRY_ARMS=all ROLA_CUDA_ARCHS=86` — the battery's build |

The split has exactly three effects and no others: a test row is out of the
default and gate builds, out of the ratification manifest, and built at one
architecture for the oracle battery. A test that needs one asks for it
(`tests/k31/fixtures.py`'s `require_arm`) and **skips with the arm named** when
the binary does not carry it, rather than failing inside the dispatch.

## <a id="sccache"></a>The nvcc invocations are cached (sccache)

The remaining floor the section above names -- "the ratification compiles the
same translation units the build does" -- is now a compiler-cache hit
instead of a second cold compile. `tools/sccache_pin.json` pins an exact
[sccache](https://github.com/mozilla/sccache) build (version, release asset,
tarball and binary sha256, verified against upstream's published `.sha256`);
`tools/sccache_toolchain.py` resolves and gates it (closed-world rule 1's
extension -- an unpinned or mismatched cache is REFUSED, full stop; K39, Blake
ruling 2026-08-28, "a tool that can run one of two ways must REFUSE, not
choose": the only non-refusing outcome is `ROLA_SCCACHE=0`, an explicit
opt-out, never a silent uncached fallback) before wiring `tools/sccache_nvcc.py` in
as `PYTORCH_NVCC` (`setup.py`) and as the compiler `tools/ratify.py`'s
post-build recompile uses. The wrapper is a pure pass-through -- it never
resolves `nvcc` itself, only `exec`s the exact already-`ptxas`-gated path its
caller hands it through `ROLA_REAL_NVCC` -- so the cache key inherits whatever
compiler identity `sccache`'s own argv hashing derives from that ALREADY-gated
binary, by construction.

**Three mechanical facts, each measured on this tree, not inferred:**

* **The depfile flag torch always adds does not break the cache.**
  `--generate-dependencies-with-compile --dependency-output` (forced by
  `torch.utils.cpp_extension._write_ninja_file` on every `nvcc` call) was
  suspected, from the torch source's own comment, to make `.cu` compiles
  uncacheable. Measured with sccache 0.17.0: a cold compile WITH the flag,
  followed by the same compile WITHOUT it, produced a 100% hit and a
  byte-identical `.o` (`cmp`). The flag is kept (header-dependency correctness
  is not traded away) and the cache still collides across both invocation
  shapes.
* **The cache key includes the invoking process's CWD**, even when every path
  in the command is already absolute. `pip install -e . --no-build-isolation`
  runs its PEP 660 editable-build hook in a FRESH `tempfile.mkdtemp()`
  directory every single invocation, which defeated not just sccache but
  NINJA'S OWN incremental rebuild (a second `pip install -e .` on an unchanged
  tree measured 396 s -- a full rebuild in everything but name, because ninja
  had no `build.ninja` from "last time" to compare against). Fixed by pinning
  `build_temp` to a FIXED, repo-relative `build/temp` directory
  (`setup.py:RoLABuildExtension.finalize_options`), which `tools/ratify.py`'s
  `COMPILE_CWD` shares.
* **The exact compile argv is reused from the build's own record, not
  reconstructed.** `tools/ratify.py`'s `_nvcc_command` reads `nvcc`/
  `cuda_cflags`/`cuda_post_cflags` VERBATIM out of `build/temp/build.ninja`
  when one exists, substituting only the source and object paths -- replacing
  an independent reconstruction that was wrong in three ways, each found only
  by a full build measuring 0% hits: missing the torch-injected pybind11-ABI/
  `TORCH_EXTENSION_NAME` defines, missing `--compiler-options '-fPIC'`, and
  using `-isystem` where `setup.py`'s actual `CUDAExtension`-based build (a
  different code path than `torch.utils.cpp_extension`'s JIT `load()`, which
  DOES use `-isystem`) emits plain `-I` for every include. Falls back to an
  independent reconstruction only when no build has ever run from
  `COMPILE_CWD` (a fresh checkout's first build, which has no cache entry to
  collide with regardless).

**MEASURED, `MAX_JOBS=4` (a lower cap than the `MAX_JOBS=7` table above --
imposed for host RAM safety during this measurement, not a property of
sccache; the two tables are not directly comparable), `nice -n 10`, both
architectures:**

| | wall | sccache hit rate |
|---|---|---|
| cold (empty cache, fresh `build/`) | **3m57.9s** (a second cold run: 4m24.3s) | 41.7% (60/144 -- the post-build ratification's 10 TUs hit the build's own ninja compile, intra-cycle) |
| warm (cache from a prior build, `build/` deleted and recreated) | **1m12.3s** | 100% (144/144) |
| scoped edit (one hand-written `.cu`, comment-only, `ratify.py --write` then rebuild) | 45.0s (the `--write` re-ratification itself: 38s) | 100% |

**THE CORRECTNESS GATE.** A cache-hit build must be bit-identical to a cold
build. Verified: `tools/sass_bodies.py --compare` between an independently
cold-built `.so` and a fully cache-hit (100%) rebuilt `.so` -- **716 entries
compared, 0 differing, 0 absent either way**; the two `.so` files are
byte-identical (`cmp`); `rola/_build_config.py` (including `manifest_sha256`)
is byte-identical between the two builds.

`sccache` is recorded in the manifest's `toolchain` block
(`{"version": ..., "sha256": ...}` or `null` if a ratification ran uncached)
per the input-closure rule, but a version drift there is REPORTED, not FAILED
(`tools/ratify.py`'s `main`, same treatment as register drift): sccache is a
measured-transparent pass-through, so its version cannot move a ratified
register/spill number the way a `ptxas` swap can. What IS gated hard, before
the wrapper is used at all, is that the resolved `sccache` matches the pin.

**A WRITTEN HASH MAP CARRIES THE HASHER'S VERSION AND IS ONLY COMPARABLE WITH ITS
OWN.** What goes into a digest is a property of `tools/sass_bodies.py`, not of the
binary, so widening the capture moves every digest in an unchanged binary. Maps are
stamped with `HASH_VERSION` and `--compare` refuses a mismatched or unstamped pair
rather than reporting the whole matrix as differing.

**THE PATH TRAP IS REMOVED. Worktree builds and cross-path SASS comparison are
ordinary operations.** `decode.cu` and `entmax.cu` used to put their kernels in an
anonymous namespace, whose mangled names embed a discriminator derived from the
SOURCE DIRECTORY — 96 of the ratified names per architecture moved with the checkout
path, so a worktree build failed the post-build gate by construction (*"192 ratified
instantiation(s) name no entry point … the manifest certifies a binary that is not
this one"*) and a cross-path SASS comparison disagreed on all 192 while describing
identical code. Fixed by moving those three blocks into a named internal namespace
(`rola::decode::detail`, `rola::entmax::detail` — P19,
`workflows/refactor/p19_symbols.md`): the ratified names are now a function of source
content and toolchain only, never of where the tree happens to be checked out.
Verified non-vacuously: master built in a fresh worktree fails exactly as described
above; this tree's tip, built in the same worktree, exits 0, ratifies all-green, and
its SASS map matches a canonical-path build of the same tip with 0 differing bodies.

The decode path is now `decode.cu` alone — fold, factor and walk are one kernel, one
TU, one `rola::decode::detail` block — so NO kernel in the binary carries an
anonymous-namespace name: every ratified name is a function of source content and
toolchain only, a worktree build passes the post-build gates, and a same-tip SASS
comparison across paths is 0-differing over the whole matrix.

**`--split-compile` is REJECTED, measured.** It is not a build-speed knob here: it
changes the assembled code of 1,073 of the 1,228 entries then present, which under the
closed-world policy is a kernel change requiring re-ratification, and it is slower
besides -- the parallelism axes multiply, so it collapses `MAX_JOBS` from 7 to 1.
Evidence: `workflows/refactor/p02_split_compile_trial_results.md`.

## <a id="fatbin-compression"></a>The fatbin is compressed explicitly

`-Xfatbin -compress-all` is in the flag list (`tools/build_flags.py`, which both the
build and the ratification import). `fatbinary` compresses by default, but that
default covers PTX and debug images only, and for the cubins it applies a SIZE
HEURISTIC: measured on this tree, every cubin it stored raw was <= 9,076,960 B and
every one it compressed was >= 10,075,232 B. That makes the shipped artifact a
function of the SHARD PARTITION rather than of the code. The measurement below was
taken on the tiled consumer's 256-arm matrix (retired in P67 D2), where at 32 arms
per shard twelve of the sixteen cubins fell under the threshold; the FLAG is what
made the artifact size independent of that partition, which is why the measurement
still governs. Same 256 arms, same sources:

| build | `.so` |
|---|---|
| 8 shards x 32, packer default | 141,919,496 B |
| 4 shards x 64, packer default | 73,743,608 B |
| **8 shards x 32, `-compress-all`** | **61,322,504 B** |

So the flag decouples size from the partition — and beats the coarser partition
outright, because it compresses the four large cubins the heuristic already took AND
the twelve it did not. The partition therefore stays a build-time question decided on
its own terms (`tools/gen_shards.py`: the chunk partition is contiguous blocks of the
declared arm list, which is what keeps a ratchet refusal a one-shard event).

Compression is applied to the image AFTER `ptxas` has assembled it, so it cannot move
codegen — proven rather than asserted, by comparing every entry point's SASS across
the flag change: **716 entries, 0 differing, 0 absent either way**. Taking the
ratification again under the new flag list reproduces every register and spill number,
so the manifests come back byte-identical.

THE COST IS PAID AT MODULE LOAD, AND IT IS MEASURED. Three runs each, fresh process,
CUDA context already up so the number is the module and not the context:

| | before | after |
|---|---|---|
| extension import | 2.01 / 2.04 / 2.13 s | 2.08 / 2.18 / 2.60 s |
| FIRST driver touch of a kernel | 0.0050 s (x3) | 0.0086 / 0.0088 / 0.0088 s |
| a warm query | 3-4 us | 4-6 us |

**+3.7 ms once per process, at the first launch** — that is the decompression, and it
is where it should be. Import is noise-dominated and shows no separable effect.

ONE GAP, RECORDED RATHER THAN CLOSED: the ratification manifest pins the ASSEMBLER
(`ptxas --version`) and the calibration `#define`s, but NOT the flag list, so a
codegen-affecting flag change leaves no fingerprint in the manifest. What catches it
here is that the flag list is single-sourced in `tools/build_flags.py` and both
compilers import it. Pre-existing; recording the flag list in the manifest's
`toolchain` block would close it.

## Verifying a built binary

```bash
python -c "from rola import _build_config; print(_build_config.show())"
cuobjdump --list-ptx rola_cuda.cpython-*.so     # must list NOTHING
python tools/ratify.py --arch 80 --arch 86       # must PASS
python tools/fatbin_entries.py --check rola/_C*.so   # must PASS
```

The shipped `.so` carries 8 ELF images and **0 PTX images**. That is not a
side effect; it is rule 2, checked.

## Dependencies

`torch` and `einops`. Not `transformers` (the HF model classes stay in the fork),
not `fla`, not `triton` — the kernels are CUDA. Optional interop lives behind
extras (`[transformers]`, `[fla]`) and `rola` itself imports neither.

## Standards lint

`KERNEL_STANDARDS.md` is enforced two ways, mirrored between the pre-commit
hooks and `.github/workflows/lint.yml` exactly like clang-format/ruff above:

1. **`ast-grep` rules** (`sgconfig.yml` + `tools/lint/rules/*.yml`, one file
   per rule) — shapes a parser can see: a warp collective inside a
   short-circuit or ternary (§12), a `__global__` in an anonymous namespace
   (anon-namespace-kernels-unratifiable), a width-shaped `if (kDv == ...)`
   test in device code, a `#include <torch/...>` in a per-arm instantiation
   TU (K36), a fence gated behind a single-lane predicate (§12). Each rule's
   `message` names the standards section and the incident it comes from.
   Each has a fixture under `tools/lint/fixtures/` that the rule must fire
   on — its test.
2. **`tools/lint/lint_standards.py`** — the textual/preprocessor checks
   ast-grep cannot express: unroll discipline (§6: a dynamic-bound loop in
   device code carries `#pragma unroll 1`, scoped to `__global__`/`__device__`
   function bodies by brace-counting so host launcher loops are out of
   scope), the torch-include check again as a text belt to the ast-grep
   rule, and `__syncthreads`/`__syncwarp` occurring after decode.cu's
   `---- THE WALK` marker (P79's law: barriers confined to the prologue).
   Every finding prints its section, `file:line`, and the incident; the
   script documents its own heuristic misses in its module docstring.
3. **`tools/lint/run_clang_tidy.sh`** — a conservative `bugprone-*` +
   `readability-*` set (`.clang-tidy`) on this codebase's host-only TUs
   (details and scope below).

Run all three:

```bash
ast-grep scan --config sgconfig.yml csrc/
python3 tools/lint/lint_standards.py
tools/lint/run_clang_tidy.sh
```

`pre-commit install` wires all three in (`.pre-commit-config.yaml`), scoped to
`csrc/`. None is a gate on its own claim of correctness — a hit is a
finding for a human to rule on, same as `ratify`'s backstop philosophy (§9):
these checks should ordinarily catch nothing.

### compile_commands.json (for clangd / clang-query / clang-tidy)

`tools/lint/compdb.sh` reads an ALREADY-BUILT tree's `build/temp/build.ninja`
(the fixed, repo-relative build directory this doc names above) and writes a
compile database:

```bash
pip install -e . --no-build-isolation      # if not already built
tools/lint/compdb.sh                        # writes ./compile_commands.json
```

`.clang-tidy` at the repo root is a conservative `bugprone-*` + small
`readability-*` set. Run it with `tools/lint/run_clang_tidy.sh` (reads
`compile_commands.json`, generating it via `compdb.sh` if missing; SKIPS —
prints why, exits 0 — if no `build/temp` exists yet rather than forcing a
~7-minute AOT build before every commit):

```bash
tools/lint/run_clang_tidy.sh
```

**Scope: HOST TUs only.** The script finds every `.cpp` entry in the compile
database (currently exactly one, `csrc/rola/rola_api.cpp`) and lints it with
the on-disk source, rewriting the compdb's build-tree path to whatever repo
it's run from. `.cu`/`.cuh` device TUs are NOT part of this gate: clang 14's
CUDA front end, run `--cuda-host-only`, DOES reach and diagnose this
codebase's own kernel code (confirmed against `decode.cu` — real
`bugprone-narrowing-conversions` / `bugprone-implicit-widening-of-
multiplication-result` hits on its own lines) but cannot complete a clean
parse: clang 14's bundled CUDA wrapper headers are incompatible with this
host's CUDA toolkit's libcu++ (`<cuda/std/...>`, reached transitively through
torch/c10 — "CUDA device code does not support variadic functions", "no
template named 'texture'", ~99 such hits, all inside toolkit headers), plus a
few genuine device-only-intrinsic gaps (`__ldcg` has no host-mode overload
under `--cuda-host-only`). `tools/lint/try_clang_tidy_device_tu.sh`
reproduces this attempt; it is deliberately NOT wired into pre-commit or CI.

`rola_api.cpp` itself passes with **zero findings** under the current check
set (validated live: the same command flags a deliberately broken scratch
file, so the clean pass is the tool actually running, not a silently inert
config).
