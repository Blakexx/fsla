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
arm family of `bazel/arms.bzl` ([`arms_gen.md`](arms_gen.md), [`../../build-graph.md#arms`](../../build-graph.md#arms)), so a BUILD that names a declared narrowed contract
(`--//:contract=decode_d32_l2`) compiles the arms that contract reaches and nothing else, through the generated
`decode_selection.inc` that `decode.cu` alone includes
([`build.md#arm-subset`](../../build.md#arm-subset)). Instantiation is the dominant build
resource, so this is what makes iterating on a few arms cost a few arms.

**THE LAUNCH IS WRITTEN ONCE.** [`dispatch_switch.md`](../dispatch_switch.md) refuses
the MACRO LADDER -- a `#define` per axis whose BODY restates the `<<<...>>>`. Decode selects through
`arm_switch<DecodeArmSet>` ([`../common/arm_switch.md`](../common/arm_switch.md)): the body receives an arm whose three
fields are constants, the launch is a single generic lambda taking three `integral_constant`s, and no `<<<...>>>` is
spelled twice.

**THE BUILT SET IS READ BACK OFF THE SAME SET.** `fsla_decode_arms()` returns `DecodeArmSet::rows()`, the `[d_v, D, decay]`
rows of the arms the build selected, so `fsla.ops.decode.arms()` cannot disagree with what was compiled -- one set, and the
dispatch and the readback both read it. An arm the binary does not carry is the switch's refusal, naming the field that
misses and the built set, never a silent miss.

<a id="fsla-decode-launch"></a>**THE LAUNCH IS THE DESCRIPTOR'S.** `fsla_decode_forward` takes the descriptor's words (`fsla.descriptor`, [`../common/plan/environment.md`](../common/plan/environment.md)),
has the planner check them against the contract and the device the call runs on, and launches what `plan::decode_launch` says
([`../common/plan/decode_launch.md`](../common/plan/decode_launch.md)): the arm, the CTA's threads, the shared-memory budget the launch opts in to, the
grid. `v` must be the arm's value width: the seam pads the caller's `d_v` by the launch function's padding and slices `y` back.
`fsla_decode_launch(descriptor, device)` returns that launch for Python (`fsla.ops.decode.decode_launch`).

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
ONE kernel: the fold, the `(k, m)` lattice's factor tables, the walk and the step's OWN
per-batch-head residency verdict are phases of it. There is no probe launch in front of it:
the write set the walk enumerates is the set the residency test reads, so the test is a pass
over shared memory. The verdict (`growth`, `done`, and the `growth_any` reduction the host
reads) sits beside the write-atom set it was reached from (`atom_bits`, what a host
admission plans from), and under a slack pool a growth step admits itself.
`write_dense` is the launch function's derived fact that the innermost level's write side is not listed (every digit of it is walked:
the descriptor's innermost `write_listed_k` is 0, which `fsla.ops.produce.decode_descriptor` states from the declaration), which the launch function hands the entry and which lets the write term walk whole atoms (`decode.md#whole-atoms`); the level widths and their row offsets come from it too, so the entry takes neither. Its operands are the
caller's RAW
`[B, 1, H, *]` views — `read`, `write` and their scales, `v` — because the fold is inside the kernel and there is no operand plane
to hand it ([`decode_fold.md`](decode_fold.md)). `heads` is not an argument: `v`'s own
`[B, 1, H, d_v]` shape carries it. `lattice_k`/`lattice_m` name the `(k, m)` box
(`decode_lattice.md#the-refusals`) the caller's state plane and routing levels are
already written in — the leaf order travels with the state plane rather than being assumed (the op derives it from the state's format and passes it as an attribute of the state operand). Its
argument checks and workspace contracts are documented with the definition, in
[`decode.md`](decode.md#instantiation-matrix).

## <a id="paged-state"></a>3. `page_table` — which backing the state is

The trailing `std::optional<Tensor> page_table` is the whole ABI difference
between the two backings, and it is what the entry validates differently:

* **absent** — `state` is the dense `[BH, N, d_v + 1]` plane and `N` is its own middle
  extent. This is the pre-P67 signature exactly, defaulted so a caller that never
  pages never mentions paging.
* **present** — `page_table` is `[BH, N / 16]` int32 and NAMES the leaf space, so `N`
  comes from it; `state` is then the arena's `[slots, 16, d_v + 1]` plane, which holds
  only the atoms a plan committed.

Slot VALUES are not bounds-checked. That would be a device-side read of the table, and
the plan owns the bound. The address the kernel builds from a slot, and what a `-1` means, are
[`decode.md` §4](decode.md#paged-address).

## <a id="the-verdict"></a>4. The verdict buffers — required whichever backing this is

`growth`, `growth_any`, `growth_ctr`, `done` and `atom_bits` are the step's OWN
per-batch-head residency answer, written by each head's owner and finisher
([`decode.md#one-writer`](decode.md#one-writer)) and read by nobody's prologue but the
step's own next call. They are validated whichever
backing the call is — a dense step has nothing to admit and simply never writes a `1` into
`growth` — because holding one ABI for both backings is what keeps a dense-vs-paged
disagreement from being expressible in the first place: [`decode.md`
§4c](decode.md#absorbed-verdict).

`atom_bits` is `[BH, N / 16]` bool, this step's own write-atom set condensed on the
device, unconditionally, by every batch-head's owner. It is what a host
admission (`admit_growth`) plans from, so the host derives no support of its own
([`decode.md` §39](decode.md#second-derivation)).

## <a id="the-pool"></a>5. `pool_slots`, `pool_cursor` — the slack pool, both or neither

Present exactly on a paged step whose state reserved slack (`fsla.state(pool_slack=...)`
above `0.0`): the physical slots the host pre-committed and pre-zeroed, and how many of each
batch-head's are spent. The head's owner writes its claim into the page table itself
([`decode.md#queue-claim-pool`](decode.md#queue-claim-pool)), so `page_table` is mutable.
`fsla_decode_forward` checks that both are present together or neither is — the pool is not a mode a caller can half-configure — and
that a pool requires `page_table` (a dense state has no table to admit into). The design and
the device-side claim are [`paging/paging.md`
§the-slack-pool](../paging/paging.md#the-slack-pool).

## <a id="decode-ledger-bind"></a>`fsla_decode_ledger_bind`

Binds the phase ledger the next decode steps book into (`decode.md#phase-ledger`), or unbinds it with `None`: a
contiguous CUDA int64 tensor `[ctas][warps][kDecodePhases]` (`warps` the step's plan's at its device's build), a row per resident CTA, which the host entry
checks before each launch. Measurement only; nothing on the shipped path binds one.

<a id="decode-ledger-warps"></a>`fsla_decode_ledger_warps()` is that tensor's warps, the step's plan's at the current
card's build (`DecodePlan<B>::kWarps`), so a caller sizes a ledger by asking rather than holding the number.
