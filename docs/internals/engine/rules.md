# The rule layer

Mirror of `fsla/engine/rules/`. This page explains WHY the layer exists and how its folds
connect to the calls; the code is the authoritative statement of what each one decides.

Symbols: `BC` the carry's box in leaves, `D` routing depth, `B` uniform level
width, `d_v` value width, `N` leaf count, `R`/`W`/`S` the liveness byte's read, write
and state bits.

## <a id="facts-and-rules"></a>1. Facts are what is TRUE; rules are what we DECIDE

> A rule is a cheap, pure, host-side FOLD from values to an execution decision. None of
> them owns a value, none of them recomputes one, and none of them may issue a launch.

The layer exists because those two kinds of statement have different costs, different
tests and different failure modes. A fold left inline at the site that needs it — a
preference order in a seam, an envelope split across two modules, a truth table in CUDA
C++ — cannot be read as a set with the others, and cannot be exercised without a device.

| rule | reads | decides |
|---|---|---|
| `envelope` | device | the operand-side refusal (the arm is a CUDA kernel), or None |
| `run_table` | the three state ARGUMENTS' presence | the `uint32` skip table (in no call: kept for the carry port) |
| `admission` | the allocator's cursor, capacity, extent, count | the run's first slot |

## <a id="the-contract"></a>2. The contract is the directory

`fsla/engine/rules/` holds plain functions called directly with values, which is what makes
each testable with no device present (`tests/unit/test_facts_rules.py`). The contract they share
cannot be machine-checked, which is exactly why the directory is the inventory —

> **A rule may read a value's SHAPE, DTYPE, DEVICE and PRESENCE. It may never read a
> value's CONTENTS.**

Reading contents is a device-to-host synchronization, which is the one thing a decode
step must not do and the thing that would break CUDA-graph capture. The `run_table`
fold is the case where that discipline is visible rather than abstract: its three
inputs are the entry-state plane, the exit-state plane and the page table, and it
reads exactly whether each is there.

What a rule folds is ENGINE VOCABULARY, which is what keeps the layer at the bottom of
the import graph: no rule here names the descriptor's type: the engine handles a descriptor as an opaque value, so no module here names
`fsla.interface` ([`engine.md` §6](engine.md#refusals)). The
alternative shape — a rule reaching up into the facade for the type of a value it was
handed — survives an import-graph check only while the import hides in a function body.

## <a id="where-rules-are-called"></a>3. Where the rules are called

* `envelope` is called by `fsla.ops._ext.device_index`, which every descriptor is made through, so a
  non-CUDA call is refused BEFORE any descriptor exists and no refusal can arrive with the operands
  already consumed ([`engine.md` §6](engine.md#refusals)).
* `admission` is called by the page arena (`fsla/ops/paging.py`) where a run is placed.
* `run_table` is called by no call: no kernel takes the skip table as a launch argument
  ([`#run-table`](#run-table)); no carry kernel is built.

## <a id="derive-never-author"></a>4. Derive; never hand-author the derived artefact

`run_table` is the pattern in miniature and the pattern the whole layer copies: write
the RULE once in readable form — the rows of
[`#run-table`](#run-table) — and DERIVE the variant by evaluating
it over all eight bit values. The kernel's predicate is then one shift and one AND with
no call-class knowledge in it.

Generalized: a refusal derives from the binary's own censuses and the contract, and never from a
threshold on `d_v` or `N` — what exists is what was built and measured.

## <a id="admission-splits"></a>5. AdmissionRule splits, and where the seam is

The arena's own docstring already drew half the line: `_admit_pages` is "THE narrow
data-movement boundary. Everything that MOVES BYTES lives here." The other half is
this layer. The PLACEMENT — the extent alignment and the all-or-nothing ceiling — is
`rules.admission`, and `ExtentAllocator.take` is left holding a cursor and no policy.

The reset-vs-union half stays at the arena, and that is a finding rather than a
carve-out: `fresh=True` RESETS THE PAGE TABLE, and the table is what sizes the run the
rule then places. A fold that decided the reset would have to run before the fact it
folds exists.

## <a id="no-new-pass"></a>6. A new decision never adds a GPU pass

A proposed rule either folds facts that already exist, or it NAMES ITS ONE MISSING
FACT explicitly — and that named fact goes through review, as a new
launch and a new byte cost, separately. There is no third path in which a
decision quietly grows a pass. This is the anti-creep clause and it is the reason the
two layers are separated at all.

## <a id="carve"></a>The carve: the carry's CTA's leaves, the prefill launch function's

The carry's CTA owns a CARVE of leaves -- an extent a level, and a channel extent -- and runs two operations on it: the fold over the write
side's lists and the readout over the read side's. The prefill's launch function in `common/plan/` has two halves
(`docs/open-work.md` row A-f): the host's, from the call's descriptor facts -- the level widths, never an operand and never a density -- the leaves of a batch-head, the CTA's leaves (512 at most) and the CTAs a head; and the carve, a function of the level widths and each side's densities alone (the call's own input beside the masks, `density [2, D]`), which the prefill's kernels call. The host's half is a function of launch-static facts (shapes, the binary's arms) and never of a routing statistic: what the host sizes before the launch -- the grid, the ring, shared memory -- cannot depend on the data. The carve is the one layout a measured statistic chooses (CONSTITUTION R9): the densities pick among exact forms once a call, inside the kernels, and change no result while the masks keep their guarantee.

## <a id="run-table"></a>The run table: the skip predicate is a host decision

The liveness facts are three bits per `(bh, block)`: `READ`, `WRITE`, `STATE`. They carry
no verdict, because the verdict is not a property of the routing. What a skipped block would
have cost differs by call class and by backing, and a call class here is the PRESENCE of the
three launch arguments — the entry-state plane, the exit-state plane and the page table —
never whether the call was handed a state, because it is the argument that decides the cost:

| call class | `s_in` | `s_out` | `page_tbl` | rule |
|---|---|---|---|---|
| stateless `y`-only | absent | absent | absent | `R && W` |
| state-in `y`-only, paged | present | absent | present | `R && (W \| S)` |
| state-in `y`-only, dense | present | absent | absent | `R` |
| continued state-out, paged | present | present | present | `(R && (W \| S)) \| W` |
| fresh state-out, paged | absent | present | present | `W` |
| continued state-out, dense | present | present | absent | `R \| W` |
| fresh state-out, dense | absent | present | absent | `W` |

`fsla/engine/rules/run_table.py`'s `run_table(state_in, state_out, page_table)` derives a
`uint32` TRUTH TABLE over the byte's three bits by evaluating that rule for each of the
eight values, so the rule is written once in readable form and the variant is derived, never
hand-authored. The kernel's whole predicate is then one shift and one AND, with no branching
taxonomy and no call-class knowledge in it. The eighth argument triple — a page table with
no state at all — is refused at the binding before the derivation is reached, and derives
the stateless row in any case.

**IT IS A HOST DECISION IN THE PLACE HOST DECISIONS LIVE.** The fold is a rule -- a pure host function from the call's three facts to the table -- and a
kernel that takes it reads the `uint32` as a launch argument beside the bitmap it predicates
over and derives nothing: a second derivation behind the extension boundary could not promise
that the table a call PLANS is the table it RUNS, and the seven rows above could only be
exercised through a launch. They are device-free unit cells over all eight liveness bytes ×
all seven rows (`tests/unit/test_facts_rules.py`), which transcribes the rows above and holds
the doc and the rule to each other.

**The two paged state-out rows differ, and the difference is the entry state.** A CONTINUED
sequence may hold value in a block it does not write this call, and only a read of that
block's carried state can produce it — hence the `S` term. A FRESH one has no entry plane
to read, so a block that writes nothing has nothing any read could observe, and the
predicate collapses to `W`. Both are sound; the fresh row is simply tighter, and a
derivation that gave it the continued row's rule would run blocks that are provably inert.

**Why each row is what it is.** A skipped block's contribution to `y` and to `den` is the
exact additive identity, not a negligible one: its mass accumulator is initialised to `+0.0`
and only ever written inside the write-side gate, and `x + (+0.0)` is the identity in
round-to-nearest for every `x` the zero-initialised accumulator can hold. That is what makes
`R && W` sound for a stateless call — a write-only block folds into the register state and
then discards it, so it is provably inert. A carried entry state breaks that argument, which
is what `S` restores where a page table can witness it; the dense backing has no residency
map, so its predicate degrades to `R` and says so.

**The two dense state-out rows differ for the same reason the paged pair does, and the
carry-through is what separates them.** A dense CONTINUATION updates its plane IN PLACE:
`s_out` IS `s_in`, one plane and no swap (`state.md#the-dense-continuation`). A skipped
block therefore leaves its own atoms holding what it carried in, which is both the paged
unchanged-slot semantics and the exact final state of a block this call never writes into.
What the dense backing still cannot do is WITNESS residency — with no page table there is
no `S` to read — so the `y` term must assume the block carries and degrades to `R`, and the
store term is `W`; together, `R | W`. That is the paged continued row with `S` forced true,
which is precisely what having no residency map means.

**The dense FRESH row skips like its paged twin.** The exit sweep stores the atoms this call
WRITES and no others, in EITHER backing, so no call writes its plane whole, and the fresh
dense plane is ZEROED at bind (`fsla/engine/prefill_call.py`'s `backing`, one fill per
SEQUENCE). An unwritten atom then reads back the same zero an unmapped paged atom does, which
is what lets this row collapse to `W`. The paged backing needs no fill: an unmapped atom is a
no-op in the epilogue, and a mapped slot either was zero-initialised on admission or is
carried from a previous call, which a skipped block must leave UNCHANGED — which is what
skipping does.
