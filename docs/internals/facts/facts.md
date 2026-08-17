# The facts engine

Mirror of `rola/facts/`. This page explains WHY the structure is shaped as it is and
how its pieces connect; the code is the authoritative statement of how each body
works. What is DECIDED from facts is [the rule layer's page](rules.md).

Symbols: `B` batch, `T` tokens, `H` heads, `D` routing depth, `N` leaf count, `d_v`
value width, `C` chunk tokens, `BC` owner block, `BH = B*H`.

## <a id="the-three-bands"></a>1. What the engine is

One structure carries every kernel family from a call's raw inputs to its launch
arguments:

```
Context  ->  DAG  ->  Plan  ->  execute(plan)
```

* a **Context** (`ChunkContext`) is a frozen record of the RAW inputs — the routing
  bundle, `v`, the decay dial, the state, the expert overrides. Nothing derived.
* a **DAG** is an ordered tuple of declared nodes in THREE BANDS. Band 1 nodes may
  launch and may allocate, and produce FACTS: what is true about this call. Band 2
  nodes are RULES — pure host folds from facts to decisions, which launch nothing and
  allocate nothing ([`rules.md`](rules.md)). They are interleaved, not phased.
* a **Plan** is band 3: the DAG's terminal fact set — the rules' verdicts plus
  references to the terminal tensor facts, and no third kind of field.
* `execute(plan)` reads the plan and nothing else. A launch site that decides
  anything is the condition this engine exists to remove.

Five conventions used to produce launch configuration and derived device state
independently, agreeing by hand rather than by construction: the facade's longhand
host order, the chunk seam's ad-hoc argument threading, the decode geometry object,
the state's residency entry, and the arena's side-stream/event pattern. The engine is
those five under one declaration.

## <a id="the-split-rule"></a>2. What lives where

Three packages, and one sentence decides which:

> **What crosses the EXTENSION BOUNDARY lives in `rola/ops/`. What a DAG walk produces
> or decides lives in the library.**

* `rola/facts/` — band 1 and the shared value types, BODIES INCLUDED. The packing and
  the write-atom bitmap (`planes.py`), the build's arm censuses (`manifest.py`), the
  operand list (`operands.py`), the adapters that read those facts off a routing
  bundle (`call.py`), the declared nodes over them (`nodes.py`), the value types
  (`types.py`), the plans (`plan.py`) and the runner (`runner.py`). A fact primitive
  is not a wrapper that defers its logic elsewhere; the primitive IS here.
* `rola/rules/` — band 2, one declared fold per decision.
* `rola/dags/` — the per-family compositions: the tuple, its context, its terminal
  node.
* `rola/ops/` — the launches. `chunk.py` is `execute`, the seeds, the two scans and
  the un-packing: the functions whose whole content is an extension call and the
  narrowing around it.

`rola/facts/__init__.py` re-exports the DEVICE-FREE core only, and that is a layering
statement rather than a convenience: `rola.rules` reads `runner` and `types` and must
never acquire a dependency on the extension by importing the package. The primitives
that DO reach the extension are imported by module.

## <a id="a-dag-is-code"></a>3. A DAG is code

There is no graph library and no topological sort. A DAG is a literal tuple and
**the order in the tuple IS the execution order**, chosen by the DAG's author for an
efficient issue order. `needs` exists to make that order CHECKABLE and readable:
`validate_dag` refuses, at import time, a DAG whose node reads a fact no earlier node
or seed produces. A missing edge is therefore an authoring error, never a runtime
surprise.

The runner is the whole execution model:

* a node whose mask does not contain the call's MASK — the MaskRule's own output,
  read as a fact — is SKIPPED, and its facts are
  seeded null — the runtime-null output skip;
* a node whose facts are already in the bag is skipped, which is how a SUPPLIED fact
  short-circuits every node upstream of it;
* otherwise its needs are asserted present and it is called.

## <a id="masks"></a>4. Masks, and why not per-fact bools

Specialization is per CALL CLASS. `CallClass` folds a call into four bits, and the
MaskRule folds those bits into one mask; the chunk family has two (`M1` prefill-stateless, `M2`
prefill-stateful) and the decode family will name the other two. A bool per fact
would be `2^n` specializations of everything downstream, so intra-mask variation is
carried as a null plan FIELD instead — `read_mass is None`, `state_in is None` — which
is free and costs no hot-loop register.

## <a id="the-suppliable-set"></a>5. The suppliable set is closed

Production is the sole pluggable node. A caller may substitute the routing bundle and
the packed amplitude planes it stands for; everything DOWNSTREAM of that boundary —
the tables, the bitmaps, the liveness bits, the residency plan — is one canonical
implementation, un-overridable and gated once. A context that supplies a canonical
fact RAISES rather than forking fact production, because two ungated producers of one
fact is exactly what the two-purposes doctrine forbids.

The round trip through DRAM at that boundary is not a cost to engineer away; it IS
the pluggability seam.

## <a id="streams-and-the-join"></a>6. Streams, the join, and the one host read

Node metadata declares the discipline the arena implements:

* `stream="side"` — issued on the arena's side stream.
* `waits=(...)` — a Join. **There is exactly one Join per DAG** and `validate_dag`
  asserts it, because a second event would be a second synchronization discipline.
* `host_sync=True` — the ONE node permitted a device-to-host read. A capturable
  region asserts no node declares it.

Where a Join sits is a correctness fact, not a scheduling preference: see
[the chunk DAG's page](chunk_dag.md#the-join).

## <a id="memoization"></a>7. Memoization is a node attribute

Not a per-site convention:

* `pure_of=(...)` names the facts a pure host derivation is a function OF, and gives
  the node a plain dict cache. It is for CONFIG derivations — the geometry of a
  shape, the arm of a topology — whose answer is identical for identical keys.
* `per_state=True` gives the node a `WeakKeyDictionary` keyed on the state object,
  for device allocations belonging to ONE live sequence. The weakness is required
  rather than incidental: a dropped sequence must drop its buffers, and two live
  sequences must never share one.

A memo key that is not one of the node's own inputs is refused at declaration.

## <a id="refusals"></a>8. Refusals are one call at the head

Each kernel family has ONE `validate_context(ctx)`, run immediately after the context
is built and before any fact node. It consolidates the operand-type refusals, the
configuration envelope, the suppliable-set closure and the reverse pass's narrower
arm matrix. Scattering preconditions across the DAG would allow a refusal three
launches in, with the operands already consumed; one call at the head makes that
unreachable by construction.

Every refusal names the clause that has no arm, and the message texts are gated
(`tests/unit/test_api_contract.py`).

## <a id="the-plan"></a>9. What a plan may hold

A host field is some rule's verdict; a tensor field is a reference to a terminal
fact. **No plan field may be `O(N * L)`.** The two-pass reverse pass rebuilds the
state and replays the schedule from the union table rather than from per-chunk
snapshots, and that claim is a property of the plan's field list rather than an
accident of what the forward happened to keep.

That rule is load-bearing in a way it was not before the backward's ctx became a
plan REFERENCE: a grad-bearing call now keeps the whole plan alive until the reverse
pass runs, so a field added for the forward's convenience is retained for free.
`tests/integration/test_backward_ctx.py` walks the retained plans field by field and
holds each to a closed form of its own — a field with no term there fails by name.

Two facts are also OPERANDS and the engine does not pretend otherwise: the union
table is read by the consumer kernel as its schedule and by the block bitmap as data,
and the state-out plane is both a launch argument and the call's result.

A plan is also not always a walk's OUTPUT. The backward family's plan is its INPUT —
the forward's, held by reference — because the reverse pass finds no facts of its own
([`backward_dag.md`](backward_dag.md)).

## <a id="neutrality"></a>10. The neutrality gate

`tools/chunk_identity_capture.py` captures `y`, the returned state plane and the
facts themselves — the union table, the block bitmap, the atom bitmap — off a fixed
cell set and compares them with `torch.equal` across two trees. It records what the
extension RETURNED during a `rola_op` call rather than re-deriving anything, so it
is independent of whatever host structure produced the launch arguments. `y` is the
one exception, and only where the topology spans more than one owner block: its
fan-in is an fp32 atomic reduction, so it is held to the reassociation bound instead.
