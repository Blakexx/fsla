# The engine

Mirror of `fsla/engine/` — the umbrella's ROOT (the runner, the value types, the
plans) and its `facts/` domain. This page explains WHY the structure is shaped as it
is and how its pieces connect; the code is the authoritative statement of how each
body works. What is DECIDED from facts is [the rule layer's page](rules.md).

Symbols: `B` batch, `T` tokens, `H` heads, `D` routing depth, `N` leaf count, `d_v`
value width, `BC` the carry's box in leaves (the arena's allocation extent), `BH = B*H`.

## <a id="the-three-bands"></a>1. What the engine is

One structure carries every kernel family from a call's raw inputs to its launch
arguments:

```
Context  ->  DAG  ->  launch
```

* a **Context** (`PrefillContext`, `DecodeContext`) is a frozen record of the RAW
  inputs — the routing bundle, the values, the decay dial, the state, the expert
  overrides. Nothing derived.
* a **DAG** is an ordered tuple of declared nodes in THREE BANDS. Band 1 nodes may
  launch and may allocate, and produce FACTS: what is true about this call. Band 2
  nodes are RULES — pure host folds from facts to decisions, which launch nothing and
  allocate nothing ([`rules.md`](rules.md)). They are interleaved, not phased.
* there is NO PLAN OBJECT: a family's launch node launches from the call's descriptor (a `fsla.ops.descriptor.Descriptor`, its validated
  context) and its workspace, everything derived from the facts being the family's one launch function's in `common/plan/`.
* the launch node reads its descriptor and its facts and decides nothing. A launch site that
  decides anything is the condition this engine exists to remove.

The engine puts under one declaration what would otherwise be five conventions agreeing
by hand: the facade's host order, the prefill seam's argument threading, the decode
geometry object, the state's residency entry, and the arena's side-stream/event pattern.

## <a id="the-split-rule"></a>2. What lives where

ONE umbrella package with three domains under it, and one sentence decides what is in
it at all:

> **What crosses the EXTENSION BOUNDARY lives in `fsla/ops/`. What a DAG walk produces
> or decides lives in `fsla/engine/`.**

* `fsla/engine/` (the ROOT) — the SHARED CORE, which belongs to no domain: the DAG
  framework (`runner.py` — `Node`, `node`, `rule`, `run`, `validate_dag`,
  `validate_capturable`), the value types, the
  refusal every no-backward clause raises —
  `BackwardNotImplemented` — (`types.py`), and the per-family launches (`plan.py`: the internal `PrefillLaunch`/`DecodeLaunch` bundles of `descriptor`, `operands`, `workspace`).
* `fsla/engine/facts/` — band 1, BODIES INCLUDED. The packing, the write-atom
  bitmap and the atom fold of the masks (`planes.py`), the operand list (`operands.py`), the adapters that read those facts off a routing
  bundle and the
  gradient clause they fold into — `refuse_backward` (`call.py`) — and the declared nodes over them (`nodes.py`). A fact primitive is not
  a wrapper that defers its logic elsewhere; the primitive IS here.
* `fsla/engine/rules/` — band 2, one declared fold per decision.
* `fsla/engine/dags/` — the per-family compositions: the tuple, its context, its
  terminal node.
* `fsla/ops/` — the launches and the bodies around them. `decode.py` is the geometry,
  the scratch and `step`, plus `_decode_step` — those same bodies in the step's order
  for a caller holding tensors rather than a bundle; the fold and the factor tables are
  IN the step's one launch and have no separate body here
  ([`decode/decode_fold.md`](../decode/decode_fold.md), [`decode/decode_lattice.md`](../decode/decode_lattice.md)).

**A domain reaches the core through the ROOT, never through a sibling domain.**
`fsla/engine/__init__.py` re-exports the core for exactly that: `from fsla.engine
import rule` is the import a rule module writes, so `rule` is a property of the engine
rather than of whichever domain happened to define it. The core itself imports no
domain, which is what keeps that re-export from being a cycle; `fsla/engine/facts/`
re-exports nothing, because a domain that is also a route to something it does not own
is the mis-homing this layout removes. `fsla.engine.rules` therefore never acquires a
dependency on the extension: the primitives that DO reach it are `fsla.engine.facts`
modules, imported by module.

## <a id="a-dag-is-code"></a>3. A DAG is code

There is no graph library and no topological sort. A DAG is a literal tuple and
**the order in the tuple IS the execution order**, chosen by the DAG's author for an
efficient issue order. `needs` exists to make that order CHECKABLE and readable:
`validate_dag` refuses, at import time, a DAG whose node reads a fact no earlier node
or seed produces. A missing edge is therefore an authoring error, never a runtime
surprise.

The runner is the whole execution model:

* a node whose variants do not contain the call's VARIANT — the VariantRule's own output,
  read as a fact — is SKIPPED, and its facts are
  seeded null — the runtime-null output skip;
* a node whose facts are already in the bag is skipped, which is how a SUPPLIED fact
  short-circuits every node upstream of it;
* otherwise its needs are asserted present and it is called.

## <a id="variants"></a>4. Variants, and why not per-fact bools

Specialization is per CALL CLASS, and the set is CLOSED AT FOUR: `PREFILL_STATELESS`
prefill-stateless and `Variant.PREFILL_STATEFUL` prefill-stateful for the prefill family, `Variant.DECODE_DENSE` decode-dense and
`DECODE_PAGED` decode-paged for the decode family. Each family has its own VariantRule over its own
facts — `variant` folds `CallClass`'s four bits, `decode_variant` folds the one bit a decode
step has — and a variant earns its name only by selecting a DIFFERENT node set. A growth
step does not: it is `DECODE_PAGED` walked twice around an admission
([`decode_dag.md` §1](decode_dag.md#the-two-variants)).

A bool per fact would be `2^n` specializations of everything downstream, so intra-variant
variation is carried as a null plan FIELD instead — `state_in is None`, a dense step's
`page_table is None` — which is free and costs no hot-loop register.

## <a id="the-suppliable-set"></a>5. The suppliable set is closed

Production is the sole pluggable node. A caller may substitute the routing bundle and
the packed amplitude planes it stands for; everything DOWNSTREAM of that boundary —
the tables, the bitmaps, the liveness bits, the residency plan — is one canonical
implementation, un-overridable and gated once. A context that supplies a canonical
fact RAISES rather than forking fact production, because two ungated producers of one
fact is exactly what the two-purposes doctrine forbids.

The round trip through DRAM at that boundary is not a cost to engineer away; it IS
the pluggability seam.

## <a id="streams-and-the-join"></a>6. Streams and capture

Node metadata declares the discipline the arena implements:

* `stream="side"` — issued on the arena's side stream.
* `capturable=True` — this node may stand inside a CUDA-graph capture: it reads no
  device value on the host and allocates nothing per step.
  `validate_capturable` refuses a captured recipe holding a node without it
  ([`decode_dag.md` §4](decode_dag.md#capturable)).
* `reference_only=True` — this node is a GATE's independent second derivation of a
  fact the shipped path derives its own way. `validate_dag` refuses any DAG containing
  one, which is what keeps that independence from being deleted by a well-meaning
  wiring ([`decode_dag.md` §6](decode_dag.md#reference-only)).

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

**A `per_state` cache holds REBUILDABLE TRANSIENTS ONLY**, and the state is its KEY
rather than one of its inputs: every other need is a declared memo key, so what is
cached is a function of facts a rebuild has, and the body never reads the sequence at
all. That is what makes the handle-IS-the-sequence invariant survive caching — two
handles onto one geometry get interchangeable objects, and a fork or a rollback
inherits nothing through a cache (`docs/internals/state.md` §9.1).
`tests/unit/test_facts_memo.py` drives every shipped `per_state` node with a state that
raises on any attribute read, so the property is proven per node rather than declared.

A PRIMITIVE whose answer is a function of the loaded binary rather than of any fact is
not a node and no node memo reaches it: the binary's arm census (`fsla.ops.decode.arms`) is
read by the bench provider, never by a call.

## <a id="refusals"></a>8. Refusals are one call at the head

Each kernel family has ONE `validate_context(ctx)`, run immediately after the context
is built and before any fact node. It consolidates the operand-type refusals, the descriptor's checks against the call, the gradient refusal and the suppliable-set closure (the prefill's device and unbuilt-kernel refusals are made earlier, where its descriptor is). Scattering preconditions across the DAG
would allow a refusal three
launches in, with the operands already consumed; one call at the head makes that
unreachable by construction.

Every refusal names the clause that has no arm, and the message texts are gated
(`tests/unit/test_api_contract.py`).

The gradient half of that call is `facts/call.py`'s `refuse_backward`, and it is
reached DOWNWARD: the engine composes the rule layer's clauses over its own adapters
and the facade calls the engine, never the reverse. The device clause is the rule layer's
`envelope`, called where every descriptor is made (`fsla.ops._ext.device_index`), and the unbuilt-kernel clause is the
owed descriptor's (`docs/internals/hardware/registry.md#the-owed-folder`): a prefill has ONE path to each refusal.

The descriptor is the same statement from the other end. Its facts are read by `common/plan/` and by the checks of its own surface, and the engine handles the descriptor as an opaque value; the
type is `fsla.ops.descriptor.Descriptor` (`docs/api.md` §5). NO MODULE UNDER `fsla/engine/` IMPORTS `fsla.interface` OR
`fsla.expert`, at import time or inside a body: a deferred import in a function body
hides an inversion from the import graph without removing it, so the rule layer holds
none.

## <a id="the-plan"></a>9. What a launch bundle may hold

A host field is some rule's verdict; a tensor field is a reference to a terminal
fact. **No bundle field may be `O(N * L)`**: a launch bundle is held for as long as its call is,
so a field added for one launch's convenience is retained with it.

A fact can also be an OPERAND, and the engine does not pretend otherwise: the
state-out plane is both a launch argument and the call's result.

## <a id="neutrality"></a>10. The neutrality gate

A host refactor that claims neutrality is answerable to what the extension RETURNED —
`y`, the state plane and the facts themselves — off a fixed cell set, compared across
two checkouts. Comparing the returned objects rather than re-deriving them is what
makes the comparison independent of whatever host structure produced the launch
arguments. The mechanism is the diff node
([`docs/measurement.md`](../../measurement.md#the-build-everything-declared-a-run-pruned-by-label)):
two sides, the cells, a rule and an expectation, so "against `HEAD~1`" is a second
checkout the suite's root already composes rather than a second tool.
