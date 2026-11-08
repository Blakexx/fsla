# `csrc/rola/src/common/ops.cuh` — the device op helpers

The instruction-level helpers every kernel executes through: the MMA atoms (bf16, f16 and int8) and their fragment
loads, the asynchronous staging and its barriers, the pinned-base addressing discipline the SASS gate patrols, the
lane votes, the reductions and the scheduling controls.

## Layer 2b -- the baseline operations

HOW EACH STEP IS PERFORMED.  Layer 2a (`design.cuh`) says what the design
LOOKS LIKE on this architecture; this file says how each of its steps is
EXECUTED, as operations with contracts.  Every entry is defined by WHAT THE
ALGORITHM NEEDS, never by what an instruction does: "stage this tile and tell
me when it is visible", not "issue a cp.async".  Get that level wrong and the
abstraction leaks on the first new architecture -- TMA cannot implement "issue
a cp.async", but it can implement "stage this tile" ([`atoms/copy.md`](atoms/copy.md)).

A kernel above this header names no architecture. It reads layer 2a for its design and calls this file to execute
it.

This file is also where THE CONSISTENCY BETWEEN 2a AND 2b is checked (see the
section at the foot): a selection is invalid if no available operation can
serve it, and that is what the assertion says -- "selection unsatisfiable
given available operations", never "this architecture lacks instruction X".
Where the capability row offers a better implementation than the one built,
the gap is ANNOUNCED rather than taken silently.

### Operations and contracts

Every operation is defined by WHAT THE ALGORITHM NEEDS. A helper with no caller in a shipped kernel is marked
(no shipped caller); it is kept or removed with the code.

### The atoms

Each concern of the operations is an atom header under `common/atoms/`, its body chosen per family where it is defined (`docs/KERNEL_STANDARDS.md` section 8); `ops.cuh` includes them all and holds the consistency below.

- [`atoms/address.cuh`](atoms/address.md): shared addresses, gathered and pinned bases, the state plane's page translation and the facts' per-atom bits.
- [`atoms/copy.cuh`](atoms/copy.md): global to shared without the issuing thread, their groups and waits, and the 16-byte quantum moved synchronously.
- [`atoms/barrier.cuh`](atoms/barrier.md): the CTA's and named rendezvous, the mbarrier's phases, the warp's edge and the publish fence.
- [`atoms/warp.cuh`](atoms/warp.md): lane votes and their readings, the warp-uniform index, the lane prefix and broadcast, and the controls ptxas cannot see through.
- [`atoms/clock.cuh`](atoms/clock.md): the SM's cycle counter for the phase ledgers.
- [`atoms/mma.cuh`](atoms/mma.md): the MMA's shape and unit, its fragment loads and the register transpose, and the bf16, f16 and int8 atoms.
- [`atoms/math.cuh`](atoms/math.md): the special-function unit's roots and powers, byte and halfword selects and packs, the f16 and bf16 pair arithmetic.
- [`atoms/reduce.cuh`](atoms/reduce.md): contributions into an accumulator others also add into, in global or shared memory.
- [`atoms/memory.cuh`](atoms/memory.md): shared and global accesses addressed rather than indexed, plain and predicated in the instruction.

### Operations that are not here

There is no `warp_reduce`, no `compact` and no publish/acquire pair weaker than a full rendezvous: an operation
invented before it has a caller cannot be defined by what the algorithm needs, because no algorithm needs it yet.

### What is deliberately not a capability

The shared-memory layout (pad-8 against an XOR swizzle) is legal on every
tabulated architecture and is chosen by MEASUREMENT, not by the table.  A
layout knob in the capability table would make every future measurement
unfalsifiable.  It is a free axis, and `design.cuh` says why it stays one.

## <a id="consistency"></a>The consistency between layers 2a and 2b

<a id="static-assert"></a>`design.cuh` claims REGISTERS is a residency this baseline builds: it is built by an `mma`
that reads an operand from the issuing unit's own registers and by the packed bf16 pair (`atoms/math.cuh`), the accumulator-to-operand
conversion that keeps the state out of shared memory. If either went away the claim would be false, and the first
assertion is where that is caught.

<a id="static-assert-2"></a>The second asserts THE CARVE'S UNIT: `mma`, `load_frag` and the carve are written for a
32-thread unit owning a 16-row slice of the state, and a 128-thread warpgroup unit owns 64 and issues a differently
shaped atom. It is a consistency clause between the design's carve and the operations that execute it, and its failure
reads NOT BUILT rather than UNSUPPORTED.

<a id="note-l562"></a>THE ANNOUNCEMENT SITES are where the capability row records a better implementation of a
baseline operation than the one built: the staged copy on a part with bulk asynchronous copy, where TMA satisfies the same
contract with one descriptor issue instead of one a 16 bytes. None is an error -- the built path is correct everywhere
it compiles -- and what is forbidden is taking the slower path silently.
