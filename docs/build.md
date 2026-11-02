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

A build's PARAMETERS -- what one build is -- are environment variables set per invocation, because pip passes
nothing else to `setup.py`. Everything about the MACHINE is the dev config (`tools/dev_config.py`,
`docs/internals/tools/dev_config.md`):
- `host.nvcc_threads` is `nvcc --threads`, default 2, the number of `-gencode` targets. Beyond the target count it
  buys nothing and multiplies the resident set, since the two parallelism axes multiply.
- `host.build_jobs` is the jobs a build asks for. Null derives it from **available RAM**, not core count:
  `MAX_JOBS=8` was OOM-killed on a 23 GB / 16-core host after 8m32. The divisor is a measured GB-per-`cicc`
  constant in `tools/build_flags.py`, and the host budget then clamps it to the slots held.
- `toolchain.sccache` is the pinned cache. It must match its pin or the build refuses;
  `toolchain.sccache_enabled: false` is the one deliberate opt-out, which builds uncached ([below](#sccache)).
- `toolchain.mold` is the linker and `toolchain.cuda_home` the toolkit.
- `host.nice` makes every host-budget-guarded compile and every GPU-locked run lower its own CPU niceness and
  I/O class, so it cannot freeze the host (`docs/internals/tools/build_lock.md`; the locks are rola-devtools'
  `rola_devtools.locks`).

| Build parameter | Effect |
|---|---|
| `ROLA_CUDA_ARCHS` | `;`-separated compute capabilities, default `80;86`. Naming one without a ratified manifest **fails before anything compiles** (R3). |
| `ROLA_CARRY_ARMS` | A `,`/space list of INDICES into the carry family's arm list (`tools/gen_shards.py`'s `CARRY_ARMS`), or the word `all`. **The subset is a FILE subset**: each selected row contributes its generated `csrc/rola/src/instantiations/carry_arm_<i>.cu` and a `ROLA_CARRY_BUILD_<i>` line in the generated selection header (`build/generated/carry_selection.inc`), so an arm outside the set is not compiled at all. **The default is every declared row** — see [the arm list](#carry-arm-lists); `all` names the same set. A build whose arm set is not the shipped set is **NOT SHIPPABLE**: the build prints a banner and skips the two shippability gates, because the manifest ratifies the shipped set and a binary carrying a different set is not the thing that ships, in either direction. It is self-identifying at run time: `rola.ops.carry.arms()` lists what was built and every entry point refuses an arm the binary does not carry. Pair it with `ROLA_CUDA_ARCHS=86` for iteration. |
| `ROLA_CARRY_PARTS` <a id="carry-parts"></a> | **Iteration only.** `all` (the default), `none`, or a comma list of the carry kernel's parts (`readout.stream`, `readout.loads`, `readout.drain`, `fold.pool`, `fold.ring`, `fold.loads`) to build REAL; the rest are STUBS that keep their HMMAs and drop their other work, written to the generated `build/generated/carry_parts.inc` on every build and ratify. Any subset makes the build an iteration build (banner, manifest gate skipped). Driven by `tools/compose_ledger.py` (`docs/internals/tools/compose_ledger.md`); with every part real the arm's instructions are identical to a build without the switches. Never shipped. |
| `ROLA_BUILD_PARTS` <a id="parts"></a> | **Iteration only.** `1` also builds `rola_cu13/_C_parts.abi3.so`, the part harness's driver (`measure/harness/bench_carry_calib.py`: the calibration kernels, KERNEL_STANDARDS §22), in the same pass beside the arm with `-lineinfo`. A torch JIT `load()` of the same three sources ran for minutes and was killed by the host's memory watchdog; a gating instrument that cannot run is a defect (§22). Never shipped. |
| `ROLA_DECODE_ARMS` | **Iteration only.** Comma- or space-separated indices into the decode family's declared arm list (`csrc/rola/src/decode/decode.cu`'s `DECODE_ARM_<i>`, the `d_v x D x decay` matrix in that order, 16 rows). Compiles only those arms. Unset = every arm, which is what a gate build must be. Same NOT-SHIPPABLE banner/skip mechanism as `ROLA_CARRY_ARMS` (one `_is_iteration_build` covers both families). See [below](#arm-subset). |
| `ROLA_SKIP_POST_BUILD_RATIFY` | Skips the post-build gate. For iterating on non-kernel code only; a shippable binary is one that passed it. |
| `ROLA_STRICT_MANIFEST` | `1` makes the post-build ratification fatal on a manifest mismatch; unset or `0` reports it ([the rule](#strict-manifest)). **The milestone/CI gate build sets it.** |

### The host budget: every build shares one pool

An **iteration build** (single-arch, `ROLA_CARRY_ARMS` subset) and a **gate build**
(both arches, the shipped set) compete for the same host RAM the job count is
sized against; running one of each at once can OOM the host (the §14
addendum finding, one level up). `setup.py` takes the machine-wide host budget
itself before compiling (`tools/build_lock.py` over `rola_devtools.locks.host`): an
iteration build holds the slots free now, a gate build waits for every slot, and
no invocation path can skip it. Any other CPU-heavy command -- a census compile,
a clang-tidy translation unit -- runs under the same pool with
`python -m rola_devtools.locks.host [--exclusive] -- <cmd>`.

```bash
# iteration: single-arch, arm-subset
ROLA_CUDA_ARCHS=86 ROLA_CARRY_ARMS=all python -m pip install -e . --no-build-isolation

# gate: both arches, the shipped set
python -m pip install -e . --no-build-isolation
```

## What the build does, in order

1. **R2 — gencode.** Emits `-gencode arch=compute_XX,code=sm_XX` only. A
   `code=compute_XX` entry would embed PTX and let the driver just-in-time compile
   for an architecture whose register and spill behaviour was never measured.
2. **R3 — ratified archs.** `tools/manifests/<toolchain>/sm_XX.json` must exist and be
   non-empty for every arch being built. A file-existence question, asked first,
   because the cheapest check is the one that must never be tempting to skip.
3. **R1 — toolchain.** The configured toolkit's `ptxas --version` selects the
   toolchain: the record in `tools/toolchains/` that names it, whitespace-normalized
   ([`internals/tools/toolchains.md`](internals/tools/toolchains.md)). A toolkit no
   record names refuses; that toolchain's manifests must pin the same string; torch
   must be built for the toolchain's CUDA major. The whole output, not a parsed release
   number: the build string distinguishes two shipments of one nominal release, and
   it is exactly those that can move an allocation while the version says nothing
   changed. In the same pass, R5: the vendored subtree under `csrc/third_party/` must hash
   to its `PIN.json` digest.
4. **Shards.** The checked-in instantiation shards must be the ones
   `tools/gen_shards.py` produces. Generated sources are committed so that what is
   reviewed is what compiles; a stale one fails here rather than shipping a matrix
   nobody wrote.
<a id="generated-headers"></a>Before the compile the build writes its GENERATED HEADERS (`build/generated/`: the carry
   selection, the carry parts, the csrc stamp) through `gen_shards.write_if_changed`: a header's file is replaced
   only when its text differs, because a rewrite of the same text moves its mtime and ninja recompiles every unit
   that includes it (an edit to one kernel file rebuilt both carry arms and the carry dispatch unit, 2026-11-01).
   The csrc stamp's text changes with every `csrc/` edit, and only `common/build_stamp.cu` includes it.
5. **Compile.** `ninja`. The carry family's instantiation matrix is generated
   translation units under `csrc/rola/src/instantiations/` — **one translation unit
   per carry arm**, so the arms parallelize across `MAX_JOBS` processes instead
   of serializing inside one `cicc`; `tools/manifests/shipped_set.json` declares the rows
   ([the arm list](#carry-arm-lists)).
6. **The arm table.** `setup.py`'s post-build arm-table check: the generator's table still matches the
   declaration, the selection header names exactly the arms this build selected, and the built `_C` carries exactly
   those ([`ratification.md#arm-switch`](ratification.md#arm-switch)). Always fatal.
7. **The SASS gate.** `tools/sass_gate.py` on the built `.so` (KERNEL_STANDARDS §20): local memory, convergence
   subroutines, predicated HMMA, and a kernel the source defines that no cubin carries. It compares against no
   manifest, so it runs on every build, iteration builds included, and a red is fatal.
8. **Post-build ratification.** `tools/ratify.py --arch …` for exactly the archs just built — the pre-build gates
   prove the *inputs* were ratified; this proves the codegen still matches what was measured.
9. **`rola_cu13/_build_config.py`.** Generated, carrying the archs, the assembler, the torch version and ABI flag,
   the manifest digest and per-arch entry counts, and `ptx_jit_fallback: False`.

<a id="strict-manifest"></a>**Step 8 REPORTS between milestones and is strict at ratify.** The committed manifest goes
stale between milestones by design (KERNEL_STANDARDS §16): a stage lands new instantiations and moves SASS bodies
without re-ratifying, so failing an ordinary install on that drift would make every stage build unusable. By default
(`ROLA_STRICT_MANIFEST` unset or `0`) a manifest mismatch prints a `WARNING` banner with the counts and the install
succeeds, the binary not certified shippable and saying so; `ROLA_STRICT_MANIFEST=1` makes it fatal, and **the
milestone/CI gate build sets it**, since that build immediately precedes `tools/ratify.py --write`. This never weakens
`tools/ratify.py` invoked directly, and never softens steps 6 and 7, which are properties of the build's own
source and binary and fatal regardless of any manifest.

## <a id="arm-subset"></a>An iteration build may compile one decode arm

Instantiation is the dominant build resource, so a developer testing ONE arm may
compile one arm:

```bash
ROLA_DECODE_ARMS=5 ROLA_CUDA_ARCHS=86 pip install -e . --no-build-isolation
```

Three properties make this safe to have:

* **The default is every declared arm.** `ROLA_DECODE_ARMS` unset expands to the whole
  list, so a plain build and every gate build are the closed-world one. There is no
  configuration in which a subset is the default.
* **A subset binary is SELF-IDENTIFYING and refuses what it lacks.**
  `rola.ops.decode.arms()` returns the `(d_v, D, decay)` rows the fatbin actually
  carries, read off the same list the build compiled from; asking the launch for any
  other shape is a `STD_TORCH_CHECK` refusal naming the arm, never a mis-launch.
* **It is NOT SHIPPABLE, and the build says so.** The two post-build gates prove the
  fatbin carries EVERY ratified instantiation in its assigned shard; a subset carries
  fewer by construction, so they are skipped — announced in a banner, in this one
  place and nowhere else. Ratify at the END of a stage, from a full build.

**AND THE RATIFICATION GATE MUST NOT INHERIT IT, WHICH IS THE SUBTLE HALF.**
`tools/ratify.py` reuses the last build's `build.ninja` flags VERBATIM, because
reconstructing them by hand was independently wrong three ways ([sccache](#sccache)). That
makes any flag naming WHAT WAS COMPILED — rather than HOW — inheritable, and this knob is
one. Measured the first time it was used: after `ROLA_DECODE_ARMS=5 setup.py build_ext`,
the next `ratify.py --write` inherited the define and wrote a manifest of 393 entries
carrying ONE decode arm, while the `.so` a full build had linked carried all sixteen — and
it PASSED, because the gate compared its own narrowed output against the manifest it had
just written from it. `-gencode` was excluded from inheritance for exactly this reason
(a single-arch build narrowing the next ratification); `-DROLA_DECODE_ARMS` now is too,
via `ratify.py`'s `_NOT_INHERITED_PREFIXES`. A gate that certifies numbers no shipped
binary has is worse than no gate.

The same rule governs `ROLA_CUDA_ARCHS=86`: iterate single-arch, gate and measure on
the full both-arch build.

## <a id="compile-ledger"></a>Why the arm units are torch-free

Instantiation is the dominant build resource, and with per-arm units it is not the kernels that dominate but the
tensor library's headers: a translation unit that includes them parses longer, and holds a larger resident set, than an
arm's own codegen. So the carry family's per-arm ABI (`csrc/rola/src/carry/carry_arm_abi.cuh`) speaks POINTERS and not
tensors, and only the dispatch unit includes the torch seam: a tensor-typed seam would re-pay the headers once per arm
and pin every arm unit's resident set at theirs, the term the job count is derived from. The ast-grep rule
`tools/lint/rules/torch_include_in_arm_tu.yml` refuses a torch header in a per-arm unit.

Per-arm units are what let the arms compile in parallel across the jobs the host budget grants, and what make `ratify`
per-unit incremental, keyed on the argv and the content of every input the compiler reported reading.

## <a id="carry-arm-lists"></a>The carry arm list

`tools/manifests/shipped_set.json` declares the carry family's arms and `tools/gen_shards.py` reads it. An arm is the
carve's operation-plan pair, keyed `fold_rows, fold_factors, read_rows, read_factors, channels, publish_regs`
([`internals/engine/rules.md#carve`](internals/engine/rules.md#carve)); the box's per-level extents, the depth and the
widths are runtime addressing. The declaration has one `shipped` list: every declared row is built by a default build,
ratified in the manifests and shipped. Each row is one generated `carry_arm_<i>.cu`, `i` its index in the
canonical-sorted
list.

A binary answers which arms it carries itself (`rola.ops.carry.arms()`), never the source tree. A launch whose carve
selects an arm the binary does not carry is refused by name, with the key field that has no built member.

## <a id="sccache"></a>The nvcc invocations are cached (sccache)

The build's and the ratification's compiles of the same translation units are a compiler-cache hit, not a second
cold compile. `tools/sccache_pin.json` pins an exact [sccache](https://github.com/mozilla/sccache) build (version,
release asset, tarball and binary sha256, verified against upstream's published `.sha256`), and
`tools/sccache_toolchain.py` resolves and gates it as an extension of R1: an unpinned or mismatched cache is REFUSED
(Blake, 2026-08-28: "a tool that can run one of two ways must REFUSE, not choose"); the one non-refusing outcome is
`toolchain.sccache_enabled: false` in the dev config, an explicit opt-out, never a silent uncached fallback.

**The wrapper is a pure pass-through.** `tools/sccache_nvcc.py` is `PYTORCH_NVCC` for `setup.py` and the compiler of
`tools/ratify.py`'s post-build recompile; it never resolves `nvcc`, only `exec`s the already-`ptxas`-gated path its
caller hands it (`ROLA_REAL_NVCC`), so the cache key inherits that binary's identity by construction.

**The build directory is fixed.** The cache key includes the invoking process's working directory, and pip's
editable build runs in a fresh temporary directory each time, which defeats both sccache and ninja's own incremental
rebuild. `build_temp` is therefore the fixed, repo-relative `build/temp`
(`setup.py`'s `RoLABuildExtension.finalize_options`), which `tools/ratify.py`'s `COMPILE_CWD` shares.

**The ratification reuses the build's own argv.** `tools/ratify.py`'s `_nvcc_command` reads the compile command
verbatim out of `build/temp/build.ninja`, substituting only the source and object paths: an independent
reconstruction differs in defines, flags and include forms, and every difference is a cache miss. It reconstructs
only when no build has ever run from `COMPILE_CWD`, which has nothing cached to hit anyway. Torch's depfile flag
(`--generate-dependencies-with-compile`) does not break the cache (measured: a 100% hit and a byte-identical object
across both invocation shapes), so it is kept.

**THE CORRECTNESS GATE.** A cache-hit build must be bit-identical to a cold build: `tools/sass_bodies.py --compare`
between a cold-built and a fully cache-hit `.so` reports 0 differing and 0 absent entries, and the two `.so` files and
their `_build_config.py` are byte-identical. `sccache` is recorded in the manifest's `toolchain` block
(`{"version": ..., "sha256": ...}` or `null`), and a version drift there is REPORTED, not FAILED: a measured-transparent
pass-through cannot move a ratified register or spill number the way a `ptxas` swap can.

**A written hash map carries the hasher's version and is comparable only with its own.** What goes into a digest is a
property of `tools/sass_bodies.py`, not of the binary, so maps are stamped with `HASH_VERSION` and `--compare` refuses
a mismatched or unstamped pair rather than reporting the whole matrix as differing.

**Worktree builds and cross-path SASS comparison are ordinary operations.** Every kernel lives in a named namespace,
so a ratified name is a function of source content and toolchain only: an anonymous-namespace kernel's mangled name
embeds a discriminator derived from the source directory, and the ast-grep rule refuses one
([`internals/tools/lint.md`](internals/tools/lint.md)).

**`--split-compile` is REJECTED, measured.** It changes the assembled code of most entries (1,073 of 1,228 when
measured), which under the closed-world policy is a kernel change requiring re-ratification, and it is slower besides:
the two parallelism axes multiply, collapsing the job count.

## <a id="fatbin-compression"></a>The fatbin is compressed explicitly

`-Xfatbin -compress-all` is in the flag list (`tools/build_flags.py`, which both the
build and the ratification import). `fatbinary` compresses by default, but that
default covers PTX and debug images only, and for the cubins it applies a SIZE
HEURISTIC (every cubin it stored raw was <= 9,076,960 B, every one it compressed >= 10,075,232 B when measured),
which would make the shipped artifact a function of how the instantiations are partitioned into translation units
rather than of the code. The explicit flag compresses every cubin, so the artifact's size is the code's: measured on
one 256-instantiation matrix, `-compress-all` beat both partitions the heuristic was given (61.3 MB against 73.7 and
141.9 MB).

Compression is applied to the image AFTER `ptxas` has assembled it, so it cannot move
codegen — proven rather than asserted, by comparing every entry point's SASS across
the flag change: **0 differing, 0 absent either way**. Taking the
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

The manifest's `toolchain` block records the codegen flag list (`nvcc_flags`), so a
codegen-affecting flag change is a fingerprint the ratification gate refuses
(`tools/ratify.py`'s `check_flags`, one of
[`ratification.md`'s closed-world rules](ratification.md#the-four-closed-world-rules)).

## Verifying a built binary

```bash
python -c "from rola_cu13 import _build_config; print(_build_config.show())"
cuobjdump --list-ptx rola_cu13/_C.abi3.so     # must list NOTHING
python tools/ratify.py --arch 80 --arch 86       # must PASS
```

The shipped `.so` carries one ELF image per (translation unit, architecture) and **0 PTX images**. That is not a
side effect; it is R2, checked.

## <a id="stable-abi"></a>The binary targets torch's stable ABI

The extension is built against torch's **stable** C shim and header-only types and against nothing of torch's C++ ABI:
every kernel's host code goes through `csrc/rola/src/common/torch_seam.cuh`, and the registration TU declares operators
(`torch.ops.rola.*`) instead of binding a pybind module ([`internals/rola_api.md`](internals/rola_api.md)). `setup.py`
pins the targeted minimum (`tools/build_flags.py`'s `TORCH_MIN = (2, 13)`, passed as `-DTORCH_TARGET_VERSION`, so any stable call newer than the
minimum fails to compile) and builds the module object against Python's limited API (`py_limited_api=True`, the
`_C.abi3.so` name). One binary therefore loads under every torch from 2.13 on and every CPython from 3.10 on.

2.13 is the minimum because it is the first torch whose stable surface has everything the host code uses; the newest
piece is the native handle of the current CUDA stream. Raising the minimum is a one-line change in `tools/build_flags.py`; lowering
it needs every call above the new minimum replaced first, and the compiler names each one. The gate for a change here is
a build against the minimum torch and the battery on the minimum and on the newest.

## <a id="wheels"></a>Two wheels: `rola` and its binary plugin `rola-cu13`

```bash
pip install "rola[cu13]"
```

installs two distributions at one version. **`rola`** is pure Python (`py3-none-any`): the layer, the routing, the
decay, the refusals. **`rola-cu13`** (`cp310-abi3-linux_x86_64`) is the binary plugin of the cu13 toolchain: the package
`rola_cu13` holding the fatbin `_C.abi3.so` for every ratified architecture, the build record `_build_config.py`, and
its `manifests/` record, the ratification that record's `manifest_sha256` hashes. A future toolchain is a second plugin
beside it, never a rebuild of `rola`.

`rola` loads the plugin when a kernel is first called (`rola/ops/_ext.py`). It reads the plugin's build record BEFORE
its library, and a record of another version refuses without registering an operator: the two are one version. With no
plugin, or a mismatched one, the first kernel call raises, naming `pip install "rola[cu13]==<version>"`; the pure-Python
parts (the oracle, the configuration objects) still import. The extra is declared in `setup.py` and not
`pyproject.toml` because it pins `rola`'s own version.

A checkout builds the same package in place: `pip install -e . --no-build-isolation` puts the binary at
`rola_cu13/_C.abi3.so` (and the part harness's driver at `rola_cu13/_C_parts.abi3.so`), so a checkout and an installed
wheel load the binary by one path. `tools/toolchains.py`'s `built_extension` is how the tools find it.

**The wheels are one gate build, split.** `python tools/wheels.py` refuses a tree with changes, builds the commit with
every iteration knob cleared and `ROLA_STRICT_MANIFEST=1`, and refuses the build unless its record says: this version,
not an iteration build, every ratified arch of the toolchain, and the manifest digest of the files it then packages.
It writes `dist/rola-<v>-py3-none-any.whl` and `dist/rola_cu13-<v>-cp310-abi3-linux_x86_64.whl`, byte-reproducible for
a commit (fixed member order and timestamps). `python tools/wheels.py --check dist` installs them into a scratch venv
over the running interpreter's torch and requires `rola` alone to refuse naming the extra, and both together to load
and name their carry arms ([`internals/tools/wheels.md`](internals/tools/wheels.md)).

The platform tag is the build host's, `linux_x86_64`: the wheels install on a machine whose glibc is at least the build
host's. A `manylinux` tag, which PyPI requires, is not claimed.

## Dependencies

`torch` (2.13 or newer) and `einops` (`pyproject.toml`); nothing else. The kernels are CUDA, so there is no `triton`
or `fla` dependency.

## Standards lint

`KERNEL_STANDARDS.md` is enforced by the lint family under `tools/lint/` — the ast-grep rules, `lint_standards.py`,
the two pinned clang-tidy profiles and the ratchets — run by the commit gate and mirrored in
`.github/workflows/lint.yml`. Its roster, how each gates, the compile database the clang-tidy profiles read and how to
run them by hand are [`internals/tools/lint.md`](internals/tools/lint.md).
