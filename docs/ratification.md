# The closed world, and the census

**The closed world is the declared targets and the gates every build runs.** Only measured binaries run: a kernel's
arms are Bazel targets declared in Starlark (`csrc/rola/src/<kernel>/arms.bzl`, [`build-graph.md`](build-graph.md#arms)),
the build's architectures are the hardware table's built ones (`measure.hardware.built`, [`architecture/families.md`](architecture/families.md)),
the toolchain is the pinned one, and the gates of [the rules below](#the-four-closed-world-rules) run on every build.
There is no committed record of what the assembler did: it is derivable from the pinned hermetic build, and the device
stamp ties a loaded binary to the source it was built from.

## The census: every shipped instantiation's registers and spills, on demand

```bash
bazel run //:ratify -- --census              # the per-kernel, per-architecture table of the build's instantiations
bazel run //:ratify -- --census --json PATH  # the raw per-instantiation record, for arm-versus-arm diffing
```

The census reads the units the graph's CUDA rule compiled (`//:census`, `bazel/defs.bzl`'s `ratify_census`): each unit's
`ptxas -v` report and its object, with the graph's disassemblers -- one compile, read twice, never a recompile with a
flag list of its own. Per kernel and architecture it prints the instantiations, their registers, the register-side CTAs
an SM, the spill stores and loads, and the local accesses inside a loop; the scope is the build's parameters
(`--//:archs`, one `--//:<family>_arms` per family of `ARM_FAMILIES`), so an iteration build's census is the measurement of its subset.
A spill is admissible when it is priced and worse than every alternative in hand, not when it is absent: the census
makes it a number to read, and a change that moves one is read in the census before and after.

It exits 1 on one thing, a launch bound the assembler discarded: `ptxas` reported *"Value of threads per SM … is out of
range"*, so that entry compiled with no launch bound at all and the assembler was free to spend registers as it liked,
whatever its spill count. Its disassemblers are the toolchain record's, checked before the first read.

<a id="arm-switch"></a>
## `arm_switch`, and what the arm gate proves about it

`csrc/rola/src/common/arm_switch.cuh` is the host twin of `uniform_switch`: one
branch over a GENERATED arm set, with the same four layers
([`internals/common/arm_switch.md`](internals/common/arm_switch.md)). The set comes
from the generated selection header, so the set the host dispatches over and the set
the translation units compiled against are one enumeration; `ROLA_<FAMILY>_ARM_COUNT`
and `ROLA_<FAMILY>_ARM_DECLARED` are both emitted, because the refusal has to tell *this
wheel does not carry that arm* from *nothing builds that arm in any configuration*.

`bazel/arms.py`'s `check_built_arm_table` and `sweep_binary` are where the layers become checked facts -- the library's
arm gate runs them (`bazel/gates.py`) -- and they run **after** the link because what they check is what the build
produced:

1. The selection header the build compiled against names exactly the arms the build
   selected — count, declared count, and the set's own members -- of the table the graph generated from the declaration;
2. The coverage sweep: the freshly built `_C`, **loaded by path**, is asked what it
   carries, and the answer must be exactly those rows.

A failure is fatal on every build, iteration builds included.

<a id="the-derivation"></a>
## The derivation's goldens: code hash and golden outputs

The addressing block is host code: one function, `derive_carry_geom`, run once per
launch, whose output every kernel of the family then addresses through. An off-by-one
in a span or a shift is a wrong answer nothing else in this gate would see, because
the SASS it produces belongs to somebody else's kernel. So its outputs are filed in
`tools/manifests/derivation.json` with two independent statements:

* a **code hash** per declared source file (`common/geom.cuh` today) — the wholesale
  trigger, and deliberately *not* `csrc_digest()`: a change to a kernel body is not a
  change to the derivation, and a record that expires on every unrelated edit is a
  record people re-write without reading;
* **golden outputs** — the derived block, byte for byte, on the named cases declared
  in `tools/manifests/derivation_cases.json`, plus the REFUSAL each declared-illegal
  case produces.

The refusals are filed beside the outputs because R13 makes the kernel's strict
shape half a contract whose other half is what it refuses; a derivation that quietly
began admitting a non-power-of-two width would pass a gate that only checked the
shapes it admits.

The record reads the two ways the `sass_sha256` ratchet does: with the code hash
matching, no golden may move; with it moved, the gate names which cases moved and
which did not. A separate `cases_sha256` covers the declaration itself, so a golden
cannot stop failing because somebody deleted the case that failed.

Each geometry case names a real `(D, DV, warps_per_cta)` point through the `BC` that
pair implies — `BC = leaves_per_warp(DV) × warps_per_cta`. The derived block does not
depend on `DV` directly; `DV` enters through `BC`. Two arms whose `(DV,
warps_per_cta)` land on the same `BC` therefore derive the same block, and the
goldens record that rather than hiding it.

The harness compiles the shipped header and links no extension, so this check cannot
be answered by a stale `.so`: there is no binary in the loop to be stale.
The record is the graph's output (`//tools:derivation_record`: the harness the graph built, run over the sources the
cases name): `bazel test //:derivation` fails when the committed `tools/manifests/derivation.json` differs from it, and
`bazel run //:derivation_update` writes it ([`build-graph.md#committed-files`](build-graph.md#committed-files)). The record
is the code hash, the cases' digest and the goldens, and no date, so an edit to a derivation source or a case moves the
record at once and the difference is shown golden by golden; goldens that move with the sources and the cases unchanged are a toolchain
or platform difference, not an edit: investigate before writing the record.

<a id="the-device-stamp"></a>
## The device stamp keys on the csrc content hash

A build stamp is the one check a stale binary cannot pass — but only if its value is
a function of the SOURCE. A stamp folded from struct sizes and launch widths is a
function of a few declarations, so a `csrc/` change that leaves those alone leaves the
stamp alone, and a binary built before the change reads back as fresh.

`rola::csrc_build_stamp()` (`csrc/rola/src/common/build_stamp.cu`) is a kernel that
returns 60 bits of `csrc_digest()`, compiled in through the generated header
`build/generated/csrc_stamp.inc`. `tests/unit/test_build_stamp.py` requires the value
the loaded fatbin returns to EQUAL what this tree hashes to right now — an equality
against a tree fact, not merely a number that changes.

It is a generated header rather than a `-D` for the reason the arm selection is: a
define reaches every translation unit's argv, so every `csrc/` edit would rebuild the
whole extension, while a header included only by the stamp site rebuilds one unit, the
one action that depends on it ([`build-graph.md`](build-graph.md#compile)). (A flag a
build does *not* key on — `NVCC_PREPEND_FLAGS` — is the failure one level over: the
build reports success and the binary silently stays the previous variant.) The graph's
stamp action writes the header before anything compiles, from `tools/csrc_stamp.py`'s function.

The footprint stamps each family already carries are untouched and answer a different
question: whether the loaded binary's struct sizes are the ones this source declares.

## <a id="the-four-closed-world-rules"></a>The closed-world rules

| Rule | Refused | Enforced by |
|---|---|---|
| **R1** | building under an assembler, disassemblers or torch the toolchain record does not name | `tools/toolchains.py::rule1` (the graph's record action) |
| **R2** | emitting `code=compute_XX` — any PTX a driver could JIT | the library's architecture spec (`compute_XX:sm_XX`, `bazel/defs.bzl`'s transition), asserted on the linked library by the SASS gate (`cuobjdump --list-ptx` lists **0 PTX images**) |
| **R3** | an architecture in the fatbin that is not built | `tools/build_record.py::rule3` (the graph's record action, whose refusal hands over no library): the architecture's row in `common/hardware.json` must choose a window ([`internals/common/hardware.md`](internals/common/hardware.md#traits)) |
| **R4** | launching on a device the kernel's arch table does not describe | `common/arch_runtime.cu`'s `check_arch_table()` at run time (first statement of every family's launching entry, carry included), `static_assert` at compile time |
| **R5** | building from a vendored codegen input (the CUTLASS/CuTe headers under `csrc/third_party/`) that does not match its recorded pin | `tools/build_record.py::vendored_pin` (the graph's record action); the pin is stamped into `_build_config.py` |

There is no environment variable that turns any of these off.

## The census's self-test

```bash
bazel run //:ratify -- --self-test
```

requires the readers to **fail** where they must: the loop parser counts an in-loop `LDL` and clears a cold one, the body
hash separates changed code and ignores listing padding, and the derivation check rejects each way its record can be
wrong. The list is `tools/ratify.py`'s, which runs it ([`internals/tools/ratify.md`](internals/tools/ratify.md)). A
reader whose green run is unfalsifiable says nothing, which is why the self-test is part of the tool.

## Where the record lives after the build

`rola_cu13/_build_config.py` is generated by the graph's record action, ships in the binary plugin beside the `.so`
(`rola-cu13`, [build.md#wheels](build.md#wheels)) and answers *"which build is this exact `.so`?"* from an installed
wheel with no repository checkout:

```python
>>> from rola_cu13 import _build_config; _build_config.show()
{'version': ..., 'archs': ['sm_90a'], 'iteration': False, 'toolchain': 'cu13', 'ptxas': ...,
 'ptx_jit_fallback': False, 'vendored': {'cutlass': {...}}}
```
