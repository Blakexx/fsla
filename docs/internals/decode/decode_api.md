# `csrc/fsla/src/decode/decode_api.cuh` — the decode path's host-facing declarations

<a id="note-l3"></a>The ONLY decode header `csrc/fsla/fsla_api.cpp` includes. It declares six
symbols and carries no definitions: the two capacity accessors, the build stamp, the arm list, the phase-ledger bind
and `fsla_decode_forward`. The decode
path itself is documented in
[`decode.md`](decode.md); this file exists for a compilation reason, and the
reason is the whole content of this doc.

---

## <a id="why-split"></a>1. Why it is split from `decode.cuh`

The split looks gratuitous until you hit it.

`fsla_api.cpp` is HOST-ONLY and is compiled by the HOST compiler, not by `nvcc`.
`decode.cuh` carries `__device__` helpers — the bitmap funnel shift, the set-bit
boundary cuts ([`decode.md` §14](decode.md#extract-bits)) — that the host compiler
cannot parse. A binding TU that included `decode.cuh` would therefore fail to
compile.

Guarding those helpers with `#ifdef __CUDACC__` would work, and would be worse:
the header would then mean two different things to two compilers, and which
declarations a caller got would depend on which compiler read it. **One header per
audience instead.**

## <a id="bound-surface"></a>2. The bound surface

<a id="fsla-decode-capacity"></a>`fsla_decode_capacity()` exists so a Python test can assert the C++ level-width
capacity against `fsla.routing.topology.MAX_BRANCH_WIDTH`. A `static_assert`
cannot read a Python constant, so the pair of them is what makes the
producer/decode capacity relation CHECKABLE rather than merely written down.

`fsla_decode_producer_width_mirror()` is the second half of that pair; why it is a
separate claim from the capacity is [`decode.md`
§33](decode.md#producer-width-mirror), and the capacity gate itself is
[`decode.md` §6](decode.md#level-width-capacity).

<a id="arm-subset"></a><a id="fsladecodearms"></a>**THE ARM LIST IS DATA, AND A BUILD COMPILES THE ARMS ITS CONTRACT REACHES.**
the decode family's instantiation matrix is the product `d_v(2) x D(4) x decay(2)`, the arms
`decode_reach` (`common/plan/decode_launch.h`) reaches over the contract (`hardware/`), which `arms_gen` writes as the rows of an
arm family of `bazel/arms.bzl` ([`arms_gen.md`](../common/plan/arms_gen.md), [`../../build-graph.md#arms`](../../build-graph.md#arms)), so a BUILD that names a declared narrowed contract
(`--//:contract=decode_d32_l2`) compiles the arms that contract reaches and nothing else, through the generated
`decode_selection.inc` that `decode.cu` alone includes
([`build.md#arm-subset`](../../build.md#arm-subset)). Instantiation is the dominant build
resource, so this is what makes iterating on a few arms cost a few arms.

**THE LAUNCH IS WRITTEN ONCE.** The MACRO LADDER -- a `#define` per axis whose BODY restates the `<<<...>>>`, so an
`N`-axis dispatch spells the launch `2^N` times inside an expansion the compiler cannot type-check -- is the form refused. Decode selects through
`arm_switch<DecodeArmSet>` ([`../common/arm_switch.md`](../common/arm_switch.md)): the body receives an arm whose three
fields are constants, the launch is a single generic lambda taking three `integral_constant`s, and no `<<<...>>>` is
spelled twice.

**THE BUILT SET IS READ BACK OFF THE SAME SET.** `fsla_decode_arms()` returns `DecodeArmSet::rows()`, the `[d_v, D, decay]`
rows of the arms the build selected, so `fsla.ops.decode.arms()` cannot disagree with what was compiled -- one set, and the
dispatch and the readback both read it. An arm the binary does not carry is the switch's refusal, naming the field that
misses and the built set, never a silent miss.

<a id="fsla-decode-launch"></a>**THE LAUNCH IS THE DESCRIPTOR'S.** `fsla_decode_forward` takes the descriptor's words (`fsla.ops.descriptor.descriptor`, [`../common/plan/environment.md`](../common/plan/environment.md)),
has the planner check them against the contract and the device the call runs on, and launches what `plan::decode_launch` says
([`../common/plan/decode_launch.md`](../common/plan/decode_launch.md)): the arm, the CTA's threads, the shared-memory budget the launch opts in to, the
grid. `v` must be the arm's value width: the seam pads the caller's `d_v` by the launch function's padding and slices `y` back.
`fsla_decode_launch(descriptor, device)` returns that launch for Python (`fsla.ops.decode.decode_launch`).

<a id="fsla-decode-lattice"></a>`fsla_decode_lattice(widths)` returns the lattice the launch function solves for level widths, `[k, m, BC, owners, spans.., grids..]`, and `fsla_decode_lattice_box(widths, k, m)` the box of a given `(k, m)` on them (refused in the law's words when none fits), both pure functions of the widths with no device ([`../common/plan/lattice.md`](../common/plan/lattice.md)); `fsla.ops.decode.lattice_of` and `fsla.ops.lattice.permutation` read them for the oracle and test seam. Widths are held to the decode path's level-width capacity.

**A SUBSET BINARY IS NOT SHIPPABLE** and its record says so: it carries fewer instantiations than the
declaration by construction, so an iteration build is never a wheel ([`build.md#iteration`](../../build.md#iteration)).

<a id="build-stamp"></a>
<a id="fsladecodebuildstamp"></a>`fsla_decode_build_stamp()` is a DEVICE-SIDE FACT a stale `.so` at the right
path cannot fake: it launches a one-thread kernel, compiled with this translation unit,
that folds four facts into one `int64_t` in a positional scheme and reads it back:

    stamp = ((kKinds * kMaxTerms * DecodePlan<B>::kWarps + kUnitBlockWarps + kFieldWidthMax) * 4096
             + sizeof(DecodeParams)) * 1000003
          + sizeof(Unit<4>) * 101
          + kDecodeMaxSpan

A path check or a hash of the sources checks what is ON DISK; this checks what the LOADED
BINARY actually built from, which is the fact a stale extension can silently disagree
about (the standing "extension trap" lesson). It is bound because a Python-side
recomputation of the same quantities would just be a second mirror with its own drift.

THE ENUMERATION-SHAPE FIELD IS THE MOST SIGNIFICANT ONE: a change to the enumeration
([`decode_lattice.md`](decode_lattice.md#the-terms)) can redefine what every unit index in
this path MEANS while moving neither struct's size nor any ceiling, and a stamp without it
would certify the new binary as the old one. A stamp that cannot separate two binaries is
not a stamp. `tests/unit/test_decode_capacity.py` inverts the stamp term by term.

<a id="the-step"></a><a id="note-l43"></a>`fsla_decode_forward` is the single data-plane entry, and it launches
ONE kernel: the fold, the `(k, m)` lattice's factor tables and the walk are phases of it, over
the dense state plane. Its signature is `(read, write, read_scale, write_scale, read_mask,
write_mask, dials, v, v_scale, state, yacc, queue, ctas, eps, descriptor)`.
`write_dense` is the launch function's derived fact that the innermost level's write side is not listed (every digit of it is walked:
the descriptor's innermost `write_listed_k` is 0, which `fsla.ops.produce._decode_descriptor` states from the declaration), which the launch function hands the entry and which lets the write term walk whole atoms (`decode.md#whole-atoms`); the level widths and their row offsets come from it too, so the entry takes neither. Its operands are the
caller's RAW
`[B, 1, H, *]` views — `read`, `write` and their scales, `v` — because the fold is inside the kernel and there is no operand plane
to hand it ([`decode_fold.md`](decode_fold.md)). `heads` is not an argument: `v`'s own
`[B, 1, H, d_v]` shape carries it. The `(k, m)` box the caller's state plane and routing levels are
already written in (`decode_lattice.md#the-refusals`) is the launch function's solve of the level widths, so the entry takes no lattice argument: the leaf order is a function of the widths, never assumed and never passed. Its
argument checks and workspace contracts are documented with the definition, in
[`decode.md`](decode.md#instantiation-matrix).

## <a id="dense-state"></a>3. `state` — the one dense plane

`state` is the dense `[BH, N / 16, page bytes]` uint8 plane, the one backing the entry serves,
and `N` comes from the level widths of the descriptor. The entry checks its extents (`BH` heads, `N / 16` pages a head, the page's bytes for the arm's `d_v`); the
address the kernel builds from an atom is [`decode.md` §4](decode.md#dense-address).

`yacc` is `[BH, d_v + 1]` fp32 and `queue` is `[3 BH + 2]` int32: the work queue's sums and
counters, allocated once per sequence and reset by the kernel itself
([`decode.md#queue`](decode.md#queue)).

## <a id="decode-ledger-bind"></a>`fsla_decode_ledger_bind`

Binds the phase ledger the next decode steps book into (`decode.md#phase-ledger`), or unbinds it with `None`: a
contiguous CUDA int64 tensor `[ctas][warps][kDecodePhases]` (`warps` the step's plan's at its device's build), a row per resident CTA, which the host entry
checks before each launch. Measurement only; nothing on the shipped path binds one.

<a id="decode-ledger-warps"></a>`fsla_decode_ledger_warps()` is that tensor's warps, the step's plan's at the current
card's build (`DecodePlan<B>::kWarps`), so a caller sizes a ledger by asking rather than holding the number.
