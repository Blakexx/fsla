# The rule layer

Mirror of `rola/rules/`. This page explains WHY the layer exists and how its folds
connect to the DAGs; the code is the authoritative statement of what each one decides.

Symbols: `C` chunk tokens, `BC` owner block, `D` routing depth, `B` uniform level
width, `d_v` value width, `N` leaf count, `R`/`W`/`S` the liveness byte's read, write
and state bits.

## <a id="facts-and-rules"></a>1. Facts are what is TRUE; rules are what we DECIDE

> A rule engine is a cheap, pure, host-side FOLD from facts to execution decisions.
> Rule engines are PLURAL over ONE shared fact set. None of them owns the facts, none
> of them recomputes a fact, and none of them may issue a launch.

The layer exists because those two kinds of statement have different costs, different
tests and different failure modes, and every one of these folds used to live inline at
the site that needed it — a preference order in the chunk seam, an envelope split
across two modules, a truth table in CUDA C++. Inline, none of them could be read as a
set, and none could be exercised without a device.

| rule | reads | decides |
|---|---|---|
| `mask` | `CallClass` | which subgraph — M1 or M2 |
| `arm` | `widths`, the manifest | `Arm = (C, BC)`, or none |
| `arm_envelope` | `widths`, `d_v`, normalization, decay, `T`, `arm` | the arm matrix's refusal, or None |
| `envelope` | unratified activations, device | the operand-side refusal, or None |
| `backward_envelope` | `arm_envelope`, `widths`, `arm`, the scan census | the reverse pass's refusal, or None |
| `training_envelope` | grad, state | the stateless-training refusal, or None |
| `run_table` | the three state ARGUMENTS' presence | the `uint32` skip table |
| `retention` | grad | whether the call keeps a reverse-pass ctx |
| `paging_preference` | the expert overrides | dense or paged backing |
| `admission` | the allocator's cursor, capacity, extent, count | the run's first slot |

## <a id="the-marker"></a>2. The marker is the inventory

`rule(gives=..., needs=...)` decorates a plain function and gives it `.node` — the
band-2 node a DAG walks it as. The function keeps its ordinary signature, and its
PARAMETER NAMES are the fact names, refused at declaration if they disagree, so the
adapter from the fact bag to the fold is a lookup rather than a convention.

Everything in `rola/rules/` carries that marker, and `tests/unit/test_facts_rules.py`
is what makes "everything" a fact: a function landing here unmarked fails the battery.
The contract the marker stands for cannot be machine-checked, which is exactly why it
is declared —

> **A rule may read a fact's SHAPE, DTYPE, DEVICE and PRESENCE. It may never read a
> fact's CONTENTS.**

Reading contents is a device-to-host synchronization, which is the one thing a decode
step must not do and the thing that would break CUDA-graph capture. The `run_table`
fold is the case where that discipline is visible rather than abstract: its three
inputs are the entry-state plane, the exit-state plane and the page table, and it
reads exactly whether each is there.

## <a id="rules-are-nodes"></a>3. Rules are nodes, not a phase

There is no second executor and no rule pass. Band 2 runs INTERLEAVED with band 1 in
the one walk — `paging_preference` at the head, `mask` the moment the call class
exists, `arm` the moment the manifest and the topology do, `decode_mask` the moment the
step's backing does — and every one of them sits on the DAG's parallel front, never on
its critical path. That is why the layer costs nothing.

A rule is capturable by its own contract — it reads no contents, launches nothing and
allocates nothing — so `rule` declares it so, and a rule on a captured recipe needs no
further argument.

Three of the folds are not in a DAG tuple, and each for a stated reason rather than
by oversight:

* `arm_envelope` / `envelope` / `backward_envelope` / `training_envelope` are called by
  `validate_context`, which runs BEFORE the walk's first node so that no refusal can
  arrive with the operands already consumed ([`facts.md` §7](facts.md#refusals)).
  Placing them in the tuple would move a refusal after work.
* `retention` is called at the autograd boundary, which is the only moment the
  retention decision exists: `needs_input_grad` is a property of the node's
  invocation, not of the plan the walk built.
* `run_table`'s consumer is still `chunk.cu`'s own derivation. The fold is written,
  gated against the documented rule over all eight liveness bytes, and moves under the
  ABI in the stage that changes one.

## <a id="derive-never-author"></a>4. Derive; never hand-author the derived artefact

`run_table` is the pattern in miniature and the pattern the whole layer copies: write
the RULE once in readable form — the rows of
[`chunk.md#run-table`](../chunk/chunk.md#run-table) — and DERIVE the mask by evaluating
it over all eight bit values. The kernel's predicate is then one shift and one AND with
no call-class knowledge in it.

Generalized: `arm` derives from the manifest and never from a threshold on `d_v` or
`N`. The rule's inputs do not even include `d_v`, so there is nothing to compare a
dimension against — what exists is what was built and measured, and the pinned
preference order over what exists is the whole of the policy.

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
FACT explicitly — and that named fact goes through the fact-node review, as a new
launch, a new mask and a new byte cost, separately. There is no third path in which a
decision quietly grows a pass. This is the anti-creep clause and it is the reason the
two layers are separated at all.
