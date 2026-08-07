# Build — ahead of time, closed world

## The governing constraint

The consumer kernel is compiled **ahead of time** into a fatbin containing
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
| `ROLA_SKIP_POST_BUILD_RATIFY` | Skips the post-build gate. For iterating on non-kernel code only; a shippable binary is one that passed it. |
| `ROLA_SCCACHE` | `auto` (default): use the pinned `sccache` if it resolves and matches; else build uncached. `1`: REQUIRE it -- a missing or mismatched wrapper is a hard failure. `0`: disable. See [below](#sccache). |
| `ROLA_SCCACHE_BIN` | Overrides the `sccache` binary `ROLA_SCCACHE` resolves, the same way `CUDA_HOME` overrides `nvcc`/`ptxas`. |

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
5. **Compile.** `ninja`. The consumer's 64 kernel instantiations are spread over
   eight translation units (`csrc/rola/src/instantiations/`), so the matrix
   parallelizes across `MAX_JOBS` processes instead of serializing inside one
   `cicc`.
6. **The fatbin gate.** `tools/fatbin_entries.py --check` reads the built `.so` and
   requires exactly one device entry per instantiation per architecture, each in
   the shard the matrix assigns it, and every ratified name present. This is the
   only gate that can see a *failed* split: duplicated device code leaves the build
   green, the manifest byte-identical and `nm` at 64, because a `__global__`
   template's host stub is weak and the linker keeps one.
7. **Post-build ratification.** `tools/ratify.py --arch …` for exactly the archs
   just built — the pre-build gates prove the *inputs* were ratified; this proves
   the codegen still matches what was measured.
8. **`rola/_build_config.py`.** Generated, carrying the archs, the assembler, the
   torch version and ABI flag, the manifest digest and per-arch entry counts, and
   `ptx_jit_fallback: False`.

## What it costs

MEASURED on a 16-core / 23 GB host, clean build directory, both architectures,
`MAX_JOBS` at the derived default of 7:

| | wall |
|---|---|
| clean rebuild, every gate including the post-build ratification | **7m20s / 7m08s** (two runs) |
| edit inside one shard's arms — one shard, relink, shard and fatbin gates | **49 s** |
| no source change — gates, ratification, `_build_config.py` | **3m47s** |
| one shard, one `nvcc`, both architectures | **43 s** |

The dominant term a single translation unit imposes is measured: one `nvcc` process
holding the whole matrix at 512 arms took 411 s of wall and 4.4 GB of resident set,
with no way to spend a second core on it. A shard is 43 s and 0.61 GB. The remaining floor is the ratification: it compiles
the same ten translation units the build does, which a content-addressed compiler
cache would turn into hits.

The two clean rebuilds are byte-identical in every entry point's SASS
(`tools/sass_bodies.py`), so those numbers are the same binary twice.

## <a id="sccache"></a>The nvcc invocations are cached (sccache)

The remaining floor the section above names -- "the ratification compiles the
same ten translation units the build does" -- is now a compiler-cache hit
instead of a second cold compile. `tools/sccache_pin.json` pins an exact
[sccache](https://github.com/mozilla/sccache) build (version, release asset,
tarball and binary sha256, verified against upstream's published `.sha256`);
`tools/sccache_toolchain.py` resolves and gates it (closed-world rule 1's
extension -- an unpinned or mismatched cache is refused, falling back to an
uncached build, never used silently) before wiring `tools/sccache_nvcc.py` in
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

Three OTHER files' anonymous-namespace kernel blocks are unaffected and still
path-dependent by name (`consumer.cu`'s `scan_kernel`/`divide_kernel`,
`decode_expand.cu`'s `decode_expand_kernel`, `summaries.cu`'s `summaries_kernel`,
5/arch) — P19 was scoped to the two files the ratified manifest actually certifies
through this discriminator. These five were never in the ratified manifest to begin
with (it holds 352 entries/arch; the full SASS scan holds 358, and the 6/arch
difference is these five kernels plus one path-independent CUB entry point), so their residual
path-dependence shows up only in a raw `sass_bodies.py` dump, never in a build or a
ratification gate. A worktree build no longer fails, and a same-tip SASS comparison
across paths is 0-differing over every ratified entry.

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
function of the SHARD PARTITION rather than of the code — at 32 arms per shard
twelve of the sixteen consumer cubins fall under the threshold. Measured, same 256
arms, same sources:

| build | `.so` |
|---|---|
| 8 shards x 32, packer default | 141,919,496 B |
| 4 shards x 64, packer default | 73,743,608 B |
| **8 shards x 32, `-compress-all`** | **61,322,504 B** |

So the flag decouples size from the partition — and beats the coarser partition
outright, because it compresses the four large cubins the heuristic already took AND
the twelve it did not. The partition therefore stays a build-time question decided on
its own terms (`tools/gen_shards.py`: the shard axes are PREC x GLOBAL x BT, which is
what keeps a ratchet refusal a one-shard event).

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
not `fla`, not `triton` — the consumer is CUDA. Optional interop lives behind
extras (`[transformers]`, `[fla]`) and `rola` itself imports neither.
