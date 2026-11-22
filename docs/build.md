# Build — ahead of time, closed world

What the build is and what it guarantees: the closed-world rules it enforces, the iteration builds it allows, the arm
lists, the flags' measured decisions, the stable ABI and the two wheels. The build is the graph's; its targets,
parameters, rules and gates are [`build-graph.md`](build-graph.md). fsla does not build from source: `pip install .`
refuses, naming the targets below (`pyproject.toml`'s backend, `tools/no_source_build.py`).

## The governing constraint

The kernels are compiled **ahead of time** into a fatbin containing
`code=sm_XX` images only. There is no JIT, no PTX for the driver to compile, and no
runtime autotuning. Everything the binary can execute has been assembled by a
`ptxas` the pinned toolchain names, for an architecture the hardware table builds
([`ratification.md`](ratification.md)).

That is a deliberate trade. A JIT-and-autotune build (Triton's model) adapts to
hardware nobody tested, at the cost of shipping codegen nobody looked at; the
supported matrix then means "we expect it to work". Here the supported matrix
**is** the built set, by construction, and the build refuses to widen it
silently.

## <a id="building"></a>Building

```bash
bazel run //:library                                   # every built architecture and arm, left in .bazel/bin/fsla_cu13/
bazel run //:library --//:archs=90a --//:decode_arms=3  # an iteration build (below)
bazel run //:wheels                                    # the two wheels of a clean commit (below)
```

A build's PARAMETERS are the graph's settings -- `--//:archs` and one `--//:<family>_arms` per family
of `ARM_FAMILIES` (`--//:decode_arms`) ([`build-graph.md`](build-graph.md#compile)) -- and everything about the MACHINE is the dev config.
`host.nvcc_threads` is `nvcc --threads`, default 2, the number of `-gencode` targets: beyond the target count it buys
nothing and multiplies the resident set, since the two parallelism axes multiply. Every build runs under the
machine-wide host budget, which the front door takes for it ([`build-graph.md`](build-graph.md#front-door)); any other
CPU-heavy command runs under the same pool with `python -m fsla_devtools.locks.host [--exclusive] -- <cmd>`.

## What the build checks

The library's record action and gates ([`build-graph.md`](build-graph.md#gates)) refuse a build that breaks a
closed-world rule ([`ratification.md`](ratification.md#the-four-closed-world-rules)):

1. **R2 — cubins only.** Each architecture is compiled `code=sm_XX` alone (the library's transition writes rules_cuda's
   spec `compute_XX:sm_XX`), and the SASS gate refuses a library carrying a PTX image. A `code=compute_XX` entry would
   embed PTX and let the driver just-in-time compile for an architecture whose register and spill behaviour was never
   measured.
2. **R3 — built archs.** Every arch being built must be a built one: its row in `common/hardware.json` chooses a window
   (`tools/build_record.py`'s `rule3`). A lookup in a table the kernel reads, the cheapest check, which must never be
   tempting to skip.
3. **R1 — toolchain.** The graph's `ptxas --version` selects the toolchain: the record in `tools/toolchains/` that
   names it, whitespace-normalized ([`internals/tools/toolchains.md`](internals/tools/toolchains.md)). A toolkit no
   record names refuses; the disassemblers the gates read with
   (`cuobjdump`, `nvdisasm`) must report the record's `readers` byte for byte; torch must be built for the toolchain's
   CUDA major (`tools/toolchains.py`'s `rule1`). The whole output, not a parsed release number: the build string
   distinguishes two shipments of one nominal release, and it is exactly those that can move an allocation while the
   version says nothing changed. In the same pass, R5: the vendored subtree under `csrc/third_party/` must hash to its
   `PIN.json` digest.
4. **The arm table.** The selection header names exactly the arms this build selected, of the table the graph generated
   from the declaration, and the built `_C` carries exactly those
   ([`ratification.md#arm-switch`](ratification.md#arm-switch)).
5. **The SASS gate.** `tools/sass_gate.py` on the linked library (KERNEL_STANDARDS §20): local memory, convergence
   subroutines, predicated HMMA, the step lints, a PTX image (R2 on the binary itself), and a kernel the source
   defines that no cubin carries. It runs on every build, iteration builds included.
6. **`fsla_cu13/_build_config.py`**, the record: the archs, the assembler, the torch version and ABI flag, the
   vendored pin, whether it is an iteration build, and
   `ptx_jit_fallback: False`.

The census (every instantiation's registers and spills) is not among them: it is a report on demand,
`bazel run //:ratify -- --census` ([`build-graph.md`](build-graph.md#census)).

<a id="generated-headers"></a>The build's GENERATED HEADERS (the arm selections and tables, the builds, the
csrc stamp) are actions over their generators, and `bazel run //:library` copies them into `build/generated/`, where
a compile outside the graph reads them,. Only `common/build_stamp.cu` reads the csrc stamp, whose text
changes with every `csrc/` edit.

## <a id="iteration"></a><a id="arm-subset"></a>An iteration build compiles a subset

Instantiation is the dominant build resource, so a developer testing ONE arm may compile one arm on one
architecture:

```bash
bazel run //:library --//:archs=90a --//:decode_arms=5
```

Any parameter but a default makes an ITERATION build (`bazel/arms.py`'s `is_iteration`, which the record
states), and three properties make it safe to have:

* **The default is every declared arm and every built architecture.** An empty arm setting expands to the whole
  list, so a plain build and every wheel are the closed-world one. There is no configuration in which a subset is the
  default.
* **A subset binary is SELF-IDENTIFYING and refuses what it lacks.** `fsla.ops.<family>.arms()` (`fsla.ops.decode.arms()`)
  returns the rows the fatbin actually carries, read off the same list the build compiled
  from; asking a launch for any other shape is a `STD_TORCH_CHECK` refusal naming the arm, never a mis-launch.
* **It is NOT SHIPPABLE, and its record says so.** A subset carries fewer instantiations than the declared set by
  construction, so its record says `iteration: True` and `bazel run //:wheels` refuses it.

**AND NOTHING THAT CERTIFIES NUMBERS MAY READ A SCOPE THE BUILD DID NOT HAVE.** The incident: a decode subset that was a
`-D` on a compile's command line reached a record that was not the build's, so the record held 393 entries carrying ONE
decode arm while the library carried all sixteen, and nothing asked the library which arms it carried. A record of
numbers no shipped binary has is worse than none. The census (`bazel run //:ratify -- --census`) reads the units the
graph compiled for the build's own parameters (its record names them), so an iteration build's census is the measurement of its
subset; the arm gate asks the BUILT binary which arms it carries and compares the answer with the declaration
([`build-graph.md#gates`](build-graph.md#gates)); and no arm subset is a compile flag: each family's selection is a
generated header (`<family>_selection.inc`; [`build-graph.md#arms`](build-graph.md#arms)) written for
the build's own scope, so a flag naming WHAT WAS COMPILED, rather than HOW, never reaches a unit's command line. The one
such flag is `-gencode`, the architectures, which the graph takes from the build.

The same rule governs `--//:archs`: iterate on one architecture, gate and measure on the build of every built one.

## <a id="compile-ledger"></a>Why the arm units are torch-free

Instantiation is the dominant build resource, and with per-arm units it is not the kernels that dominate but the
tensor library's headers: a translation unit that includes them parses longer, and holds a larger resident set, than an
arm's own codegen. So a family's per-arm ABI speaks POINTERS and not
tensors, and only the dispatch unit includes the torch seam: a tensor-typed seam would re-pay the headers once per arm
and pin every arm unit's resident set at theirs. The ast-grep rule `tools/lint/rules/torch_include_in_arm_tu.yml`
refuses a torch header in a per-arm unit. MEASURED per `cicc` (sm_86, `--threads=1`):

| translation unit | wall | peak a `cicc` |
|---|---|---|
| a bare CUDA unit (one trivial kernel) | 0.9 s | 0.19 GB |
| the same, plus `torch/extension.h` | 33.7 s | 2.86 GB |
| one carry arm (the torch-free ABI) | 4.7 s | 0.33 GB |
| `carry/carry.cu` (the dispatch alone) | 36.2 s | 2.88 GB |

Over a full both-arch build every unit that parses the extension header lands in one narrow band (`factor.cu` 2.92 GB,
`entmax.cu` 2.89, `decode.cu` 2.81, `carry/carry.cu` 2.75), and no generated arm unit is in the top twelve; the carry
family's single unit, before the per-arm split, peaked at 6.14 GB. A build's peak is the header's footprint times the
`cicc` the host budget runs at once, and it stops being a function of the arm count.

Per-arm units are what let the arms compile in parallel across the jobs the host budget grants, each a Bazel action keyed
on the argv and the content of every input the compiler reported reading.

## <a id="arm-lists"></a>The arm lists

`csrc/fsla/src/<family>/arms.bzl` declares a family's arms and `bazel/arms.bzl` instantiates them
([`build-graph.md#arms`](build-graph.md#arms)); the shipped families are `ARM_FAMILIES` there. The decode's arm is
`d_v × levels × decay`; the box's per-level extents, the depth and the widths are runtime addressing. A declaration has
one list, `<FAMILY>_ARMS`: every declared row is built by a default build and shipped.

A binary answers which arms it carries itself (`fsla.ops.decode.arms()`), never the source tree. A launch whose key
selects an arm the binary does not carry is refused by name.

## <a id="kernel-identity"></a>A kernel's code is a function of its source and toolchain

**The graph's cache is content-addressed.** A unit's object is cached under its command line and the content of every
input its compile read (nvcc's dependency file, [`build-graph.md`](build-graph.md#compile)), shared by every checkout
on the machine, so a hit is the object a cold compile of the same inputs makes. The graph compiles with no other
compiler cache.

**A written hash map carries the hasher's version and is comparable only with its own.** What goes into a digest is a
property of `tools/sass_bodies.py`, not of the binary, so maps are stamped with `HASH_VERSION` and `--compare` refuses
a mismatched or unstamped pair rather than reporting the whole matrix as differing.

**Worktree builds and cross-path SASS comparison are ordinary operations.** Every kernel lives in a named namespace,
so a kernel's name is a function of source content and toolchain only: an anonymous-namespace kernel's mangled name
embeds a discriminator derived from the source directory, and the ast-grep rule refuses one
([`internals/tools/lint.md`](internals/tools/lint.md)).

**`--split-compile` is REJECTED, measured.** It changes the assembled code of most entries (1,073 of 1,228 when
measured), which under the closed-world policy is a kernel change, and it is slower besides:
the two parallelism axes multiply, collapsing the job count.

## <a id="fatbin-compression"></a>The fatbin is compressed explicitly

`-Xfatbin -compress-all` is in the flag list (`tools/build_flags.py`, which the
graph's compile takes). `fatbinary` compresses by default, but that
default covers PTX and debug images only, and for the cubins it applies a SIZE
HEURISTIC (every cubin it stored raw was <= 9,076,960 B, every one it compressed >= 10,075,232 B when measured),
which would make the shipped artifact a function of how the instantiations are partitioned into translation units
rather than of the code. The explicit flag compresses every cubin, so the artifact's size is the code's: measured on
one 256-instantiation matrix, `-compress-all` beat both partitions the heuristic was given (61.3 MB against 73.7 and
141.9 MB).

Compression is applied to the image AFTER `ptxas` has assembled it, so it cannot move
codegen — proven rather than asserted, by comparing every entry point's SASS across
the flag change: **0 differing, 0 absent either way**. The census
under the new flag list reproduces every register and spill number.

THE COST IS PAID AT MODULE LOAD, AND IT IS MEASURED. Three runs each, fresh process,
CUDA context already up so the number is the module and not the context:

| | before | after |
|---|---|---|
| extension import | 2.01 / 2.04 / 2.13 s | 2.08 / 2.18 / 2.60 s |
| FIRST driver touch of a kernel | 0.0050 s (x3) | 0.0086 / 0.0088 / 0.0088 s |
| a warm query | 3-4 us | 4-6 us |

**+3.7 ms once per process, at the first launch** — that is the decompression, and it
is where it should be. Import is noise-dominated and shows no separable effect.

The codegen flag list is `tools/build_flags.py`'s alone, written into the graph's flags (`bazel/flags.py`), so a
codegen-affecting flag change is one edit in one file ([`ratification.md`'s closed-world rules](ratification.md#the-four-closed-world-rules)).

## Verifying a built binary

```bash
python -c "from fsla_cu13 import _build_config; print(_build_config.show())"
bazel run //tools:sass_gate -- .bazel/bin/fsla_cu13/_C.abi3.so    # its signatures, and no PTX image (R2)
bazel run //:ratify -- --census                        # its registers and spills
```

The shipped `.so` carries one ELF image per (translation unit, architecture) and **0 PTX images**. That is not a
side effect; it is R2, checked by the SASS gate on every build (`cuobjdump --list-ptx` lists nothing).

## <a id="stable-abi"></a>The binary targets torch's stable ABI

The extension is built against torch's **stable** C shim and header-only types and against nothing of torch's C++ ABI:
every kernel's host code goes through `csrc/fsla/src/common/torch_seam.cuh`, and the registration TU declares operators
(`torch.ops.fsla.*`) instead of binding a pybind module ([`internals/fsla_api.md`](internals/fsla_api.md)).
`tools/build_flags.py` pins the targeted minimum (`TORCH_MIN = (2, 13)`, passed as `-DTORCH_TARGET_VERSION`, so any
stable call newer than the minimum fails to compile), and every unit compiles against Python's limited API
(`-DPy_LIMITED_API`, torch's minimum CPython; the `_C.abi3.so` name). One binary therefore loads under every torch from 2.13 on and every CPython from 3.10 on.

2.13 is the minimum because it is the first torch whose stable surface has everything the host code uses; the newest
piece is the native handle of the current CUDA stream. Raising the minimum is a one-line change in `tools/build_flags.py`; lowering
it needs every call above the new minimum replaced first, and the compiler names each one. The gate for a change here is
a build against the minimum torch and the battery on the minimum and on the newest.

## <a id="wheels"></a>Two wheels: `fsla` and its binary plugin `fsla-cu13`

```bash
pip install "fsla[cu13]"
```

installs two distributions at one version. **`fsla`** is pure Python (`py3-none-any`): the layer, the routing, the
decay, the refusals. **`fsla-cu13`** (`cp310-abi3-linux_x86_64`) is the binary plugin of the cu13 toolchain: the package
`fsla_cu13` holding the fatbin `_C.abi3.so` for every built architecture and the build record `_build_config.py`. A future toolchain is a second plugin
beside it, never a rebuild of `fsla`.

`fsla` loads the plugin when a kernel is first called (`fsla/ops/_ext.py`). It reads the plugin's build record BEFORE
its library, and a record of another version refuses without registering an operator: the two are one version. With no
plugin, or a mismatched one, the first kernel call raises, naming `pip install "fsla[cu13]==<version>"`; the pure-Python
parts (the oracle, the configuration objects) still import. The `cu13` extra, the wheel's only one, is the wheels tool's, not
`pyproject.toml`'s, because it pins `fsla`'s own version; every other field of the metadata is `pyproject.toml`'s `[project]`.

A checkout's binary stays a build output: `bazel run //:library` writes nothing into the tree (a file nothing commits is never
written into the source tree), and the library is at `.bazel/bin/fsla_cu13/_C.abi3.so`. The checkout's venv (`bazel run //:dev_env`)
has a `.pth` file naming `.bazel/bin`, and the tracked package's `__init__` extends its path to the directories of
that name on `sys.path`, so `import fsla_cu13._C` finds the built library and a wheel's one directory is the same package.
`tools/toolchains.py`'s `built_extension` is how the tools find the file; a machine without Bazel (the rented box) puts a copy of
`.bazel/bin/fsla_cu13/` where its `PYTHONPATH` names a parent of it.

**The wheels package the graph's library.** `bazel run //:wheels` refuses a tree with changes, then refuses the build unless its record names this
version and the toolchain and is no iteration build. It writes
`dist/fsla-<v>-py3-none-any.whl` and `dist/fsla_cu13-<v>-cp310-abi3-linux_x86_64.whl`, byte-reproducible for a commit
(fixed member order and timestamps). `bazel run //tools:wheels -- --check dist` installs them into a scratch venv over
the environment's torch and requires `fsla` alone to refuse naming the extra, and both together to load and name
their decode arms ([`internals/tools/wheels.md`](internals/tools/wheels.md)).

The platform tag is the build host's, `linux_x86_64`: the wheels install on a machine whose glibc is at least the build
host's. A `manylinux` tag, which PyPI requires, is not claimed.

## Dependencies

`torch` (2.13 or newer) and `einops` (`pyproject.toml`); nothing else. The kernels are CUDA, so there is no `triton`
or `fla` dependency.

## Standards lint

`KERNEL_STANDARDS.md` is enforced by the lint family under `tools/lint/` — the ast-grep rules, `lint_standards.py`,
the two pinned clang-tidy profiles and the ratchets — run by the commit gate (`bazel test //:gate`), and by CI through
the graph (`.github/workflows/lint.yml`). Its roster, how each gates, how the clang-tidy
profiles read each unit's own compile and how to run them by hand are
[`internals/tools/lint.md`](internals/tools/lint.md).
