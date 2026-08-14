# `csrc/rola/src/chunk/ops.cuh` — the device op helpers

The ladder's instruction-level helpers: `mma` (m16n8k16 bf16→fp32),
`load_frag`/ldmatrix wrappers, `cp_async` staging, `gather_ptr`/`pin_address`
(the pinned-base + zero-extended-32-bit-displacement addressing discipline the
SASS gate patrols), ballot/scan primitives, packed bf16 arithmetic. Overlaps
in role with this tree's `mma.cuh`/`cp_async.cuh`; the dedup is the e2e
cleanup step's, after which one helper family serves both consumers.

## Layer 2b -- the baseline operations

HOW EACH STEP IS PERFORMED.  Layer 2a (`design.cuh`) says what the design
LOOKS LIKE on this architecture; this file says how each of its steps is
EXECUTED, as operations with contracts.  Every entry is defined by WHAT THE
ALGORITHM NEEDS, never by what an instruction does: "stage this tile and tell
me when it is visible", not "issue a cp.async".  Get that level wrong and the
abstraction leaks on the first new architecture -- TMA cannot implement "issue
a cp.async", but it can implement `stage_tile`.

The kernel above this header names no architecture and issues no inline PTX.
It reads layer 2a for its design and calls this file to execute it.

This file is also where THE CONSISTENCY BETWEEN 2a AND 2b is checked (see the
section at the foot): a selection is invalid if no available operation can
serve it, and that is what the assertion says -- "selection unsatisfiable
given available operations", never "this architecture lacks instruction X".
Where the capability row offers a better implementation than the one built,
the gap is ANNOUNCED rather than taken silently.

### Operations and contracts

```
  smem_addr(p)              the CTA-local address of a shared object; the
                            address type every operation here consumes.
  stage_tile(dst, src)      copy `kStageBytes` bytes global -> shared without
                            occupying the issuing thread.  Membership of a
                            group is implicit: every stage_tile since the last
                            stage_commit.  Completion is SIGNALLED, not waited.
  stage_commit()            close the current group.
  stage_wait<n>()           all but the newest `n` groups have landed FOR THE
                            ISSUING THREAD.  CTA-wide visibility of another
                            thread's stage requires a following rendezvous();
                            this is a property of the contract, not of the
                            sm_80 implementation, and TMA/mbarrier satisfies
                            it the same way.
  rendezvous()              every thread of the CTA arrives; all prior shared
                            accesses by any thread are ordered before all
                            subsequent ones.  Also the WAR fence that lets a
                            ring slot be overwritten.
  load_frag / load_frag_t   four 8x8 b16 tiles from shared memory, landed in
                            the register order `mma` consumes.  `_t`
                            transposes each 8x8, which is what lets one shared
                            tile serve as an operand of either handedness.
  mma(acc, a, b)            acc += A(16x16) . B(16x8), bf16 operands, fp32
                            accumulator, accumulator PRIVATE to the issuing
                            unit, and BOTH operands read from that unit's
                            registers.  That last property is what serves
                            `STATE_RESIDENCY = REGISTERS`, which is layer 2a's
                            selection and not a flag; which SIDE a given
                            kernel's accumulator lands on is that kernel's
                            clause, asserted against this operation at its own
                            site.
  pack_bf16x2(lo, hi)       two fp32 -> the b32 that `mma` reads as two
                            consecutive k-elements of an operand.  This is
                            the accumulator-to-operand conversion; it is the
                            only reason the state never touches shared memory.
  splat_bf16x2(x)           one bf16 into both halves of a packed pair, so a
                            scalar can be an operand of `mul_bf16x2`.
  mul_bf16x2(a, b)          elementwise product of two packed bf16 pairs.
                            The archetypal SELECTING entry: sm_90 has one
                            instruction for it, sm_80/86/89 must go through
                            fp32 at 1 FMUL + 1/2 pack per element.
  gather_ptr(base, off)     the address of a row selected by an INDEX rather
                            than reached by an induction: a 64-bit base formed
                            once plus a 32-bit byte displacement, ZERO
                            EXTENDED.  The contract is the ARITHMETIC and not
                            the address -- writing `base + i` at element type
                            lets the compiler strength-reduce every access to a
                            widening multiply, which is precisely what the
                            addressing rule forbids, so the widening is stated
                            here once as an extension.  A displacement that
                            does not fit 32 bits is the caller's clause, and
                            the caller that has one states its bound.  Stated
                            by per-side compaction, its first caller: a gather
                            is an indirection and must not become per-element
                            64-bit arithmetic.
  pin_address(p)            a 64-bit address the compiler may not RE-DERIVE.
                            "Form the base once and afterwards only advance
                            it" is a rule about EMITTED code, and a compiler
                            under register pressure satisfies the source while
                            breaking the rule: it drops the live base and
                            recomputes it inside the loop, which is the same
                            widening multiply moved rather than removed.  This
                            is where the rule is stated to the compiler; it
                            costs the two registers the base occupies, which is
                            what keeping it live means.  Stated by per-side
                            compaction, whose gathered bases have no induction
                            to be carried by.
  bit_count(x)              the number of set bits.
  first_set(x)              the index of the lowest set bit, or `kBallotLanes`
                            when there is none.  With `bit_count` this is what
                            turns a lockstep vote into POSITIONS: the count of
                            set lanes below a lane is that lane's rank among
                            the live ones, and the first set lane of a "would
                            overflow" vote is where a run must be cut.  Stated
                            by per-side compaction's cook.
  lane_ballot(pred)         the predicate of every lane of the LOCKSTEP unit,
                            one bit per lane, in lane order.  Nothing is
                            written and nothing is ordered: the result is the
                            unit's own agreement about a property of the data
                            its lanes hold.  The lockstep unit is `kBallotLanes`
                            wide and is NOT the MMA unit -- a warpgroup issues
                            one MMA and still votes in lanes of 32 -- so the
                            two widths are stated separately.  Stated by the
                            ladder's token schedule, its first caller: a group
                            of tokens is live for a block when ANY of the
                            tokens the unit holds is, and a vote is how a unit
                            answers that without a memory round trip.
  fan_in_add(p, x)          add ONE contribution into an accumulator that
                            other units also contribute to.  The issuing
                            thread does not consume a result and must not wait
                            for one -- that is what makes this a reduction
                            rather than an atomic read-modify-write, and it is
                            the whole contract.  No ordering across
                            contributors is defined and no contributor
                            observes another's value, so a caller that needs a
                            reproducible summation order needs a different
                            operation.  Stated by the ladder's capacity carve,
                            which is its first caller: the N/BC blocks of a
                            slot axis each compute a complete partial output.
  store_f32(p, x)           one fp32, at fp32 element addressing.  The
                            counterpart of `store_bf16` for a buffer whose
                            consumer is `fan_in_add` rather than an mma
                            operand: a summand that will be added to N/BC
                            others cannot be rounded to the operand format
                            first without the fan-in's error exceeding the
                            design's own.
  or_run<n>(p)              the bitwise OR of `n` contiguous bf16 in their
                            storage form, at the widest access that covers
                            them.  A magnitude test on the result answers "is
                            ANY of these nonzero" for the whole run at once,
                            because OR cannot clear a bit that was set.
                            Stated by the ladder's membership test, its first
                            caller: a block's span at a routing level is a
                            contiguous run of amplitudes and the test is
                            whether the run is entirely zero.
  store_bf16x2(p, v)        one packed bf16 pair, addressed as bf16 elements.
  store_bf16(p, x)          one fp32 -> one bf16 element, same round-to-
                            nearest-even as `pack_bf16x2` per element.  A
                            TRANSPOSED output accumulator holds two
                            token-adjacent values of ONE d_v column, and in a
                            [token][d_v] buffer those are different rows, so
                            the pair cannot be published as a b32.
  load_vec16 / store_vec16  one naturally aligned 16-byte value, moved in a
                            single access.  These are the SYNCHRONOUS pair of
                            `stage_tile`: 16 B is `kStageBytes`, the widest
                            access this baseline has anywhere, and a row that
                            is COMPUTED rather than staged must be published at
                            the same width the staged rows arrive at, or the
                            two would need different bank analyses of the same
                            buffer.  Stated by the ladder's routed coefficient
                            production, which is their first caller.
  copy_shared_to_global_16b one fully coalesced 128-bit store, shared -> global.
                            The composition of the two above, kept as its own
                            name because it is the whole of the epilogue's
                            contract and reads as such at the call site.
```

### Operations this kernel does not have, and why not

`warp_reduce` and a `publish`/`acquire` pair weaker than a full rendezvous are
absent, because nothing here needs them: the private d_v carve removes every
cross-lane reduction, and every buffer a producer publishes is read by the
whole CTA after a barrier that the ring already required.  There is no
`compact` either, and that is a statement about WHERE the compaction lives: a
caller that gathers tokens builds an INDEX LIST once and then addresses
through `gather_ptr`, so what it needs from this layer is the vote, the two
bit counts and the address -- not a data-moving primitive.  An operation
invented before it has a caller cannot be defined by what the algorithm needs,
because no algorithm needs it yet.

### What is deliberately not a capability

The shared-memory layout (pad-8 against an XOR swizzle) is legal on every
tabulated architecture and is chosen by MEASUREMENT, not by the table.  A
layout knob in the capability table would make every future measurement
unfalsifiable.  It is a free axis, and `design.cuh` says why it stays one.
