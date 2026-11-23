# The engine

Mirror of `fsla/engine/` — the umbrella's ROOT (the value types, the launch bundles, the profiler
range), its two calls (`prefill_call.py`, `decode_step.py`) and its `facts/` and `rules/` domains. This page
explains WHY the structure is shaped as it is and how its pieces connect; the code is the authoritative
statement of how each body works. What is DECIDED from values is [the rule layer's page](rules.md).

Symbols: `B` batch, `T` tokens, `H` heads, `D` routing depth, `N` leaf count, `d_v`
value width, `BC` the carry's box in leaves (the arena's allocation extent), `BH = B*H`.

## <a id="the-three-bands"></a>1. What the engine is

The host side of a call, from its raw inputs to its launch arguments:

```
validate  ->  steps, in order  ->  Launch  ->  execute(launch)
```

* each call is ONE PLAIN FUNCTION: `run_prefill` ([`prefill_call.md`](prefill_call.md)) and
  `decode_step_gated` ([`decode_step.md`](decode_step.md)). Its body is its steps as explicit calls in
  their order, passing values as arguments and locals; the order is the code's.
* a step may launch and may allocate. A RULE ([`rules.md`](rules.md)) is a pure host fold from values to
  a decision, which launches nothing and allocates nothing.
* there is NO PLAN OBJECT: a family's launch step launches from the call's descriptor (a `fsla.ops.descriptor.Descriptor`, its validated
  context, which the call builds itself from its inputs) and its workspace (internal to the call), everything derived from the call's facts being the family's one launch function's in `common/plan/`.
* the launch step reads its descriptor and its operands and decides nothing. A launch site that
  decides anything is the condition this structure exists to remove.

## <a id="the-split-rule"></a>2. What lives where

ONE umbrella package, and one sentence decides what is in it at all:

> **What crosses the EXTENSION BOUNDARY lives in `fsla/ops/`. What a call's host steps produce
> or decide lives in `fsla/engine/`.**

* `fsla/engine/` (the ROOT) — the SHARED CORE: the value types, the
  refusal every no-backward clause raises —
  `BackwardNotImplemented` — (`types.py`), the per-family launches (`plan.py`: the internal `PrefillLaunch`/`DecodeLaunch` bundles of `descriptor`, `operands`, `workspace`) and
  the profiler range (`trace.py`).
* `fsla/engine/prefill_call.py`, `fsla/engine/decode_step.py` — the two calls: the context, the fast-fail, the steps and the one function that runs them.
* `fsla/engine/facts/` — BODIES INCLUDED. The packing, the write-atom
  bitmap and the atom fold of the masks (`planes.py`), the operand list (`operands.py`), the adapters that read those facts off a routing
  bundle and the
  gradient clause they fold into — `refuse_backward` (`call.py`). A fact primitive is not
  a wrapper that defers its logic elsewhere; the primitive IS here.
* `fsla/engine/rules/` — one fold per decision.
* `fsla/ops/` — the launches and the bodies around them. `decode.py` is the geometry,
  the scratch and `step`, plus `_decode_step` — those same bodies in the step's order
  for a caller holding tensors rather than a bundle; the fold and the factor tables are
  IN the step's one launch and have no separate body here
  ([`decode/decode_fold.md`](../decode/decode_fold.md), [`decode/decode_lattice.md`](../decode/decode_lattice.md)).

**A domain reaches the core through its own module, never through a sibling domain.**
The core imports no domain, and `fsla/engine/facts/` re-exports nothing, because a domain
that is also a route to something it does not own is the mis-homing this layout removes.
`fsla.engine.rules` therefore never acquires a dependency on the extension: the primitives
that DO reach it are `fsla.engine.facts` modules, imported by module.

## <a id="a-call-is-a-function"></a>3. A call is a function

There is no graph library, no fact dictionary and no topological sort. The call is a function
whose body is its steps in the order the author chose for an efficient issue order, each
step taking what it reads as arguments. A step that needs a value an earlier step did not
produce is a name error at its first run, never a runtime surprise.

A shipped call selects nothing by call class: stateless and stateful prefill are one function (the
state's absence is a runtime-null operand, `state_in is None`), and dense and paged decode are one
function (a dense step's `page_table is None`). Intra-call variation is a null operand, which is free
and costs no hot-loop register; a bool per fact would be `2^n` specializations of everything downstream.

## <a id="the-profiler-range"></a>4. The profiler range

`fsla.engine.trace.span(name)` opens an NVTX range around each step of a call under the step's own name,
so a new step appears in `nsys` with one `with span(...)` and no registry. It is guarded by
`torch.cuda.is_available()`, so a CPU-only import never raises: the range is a profiling aid and never a
correctness dependency, and its absence never changes behaviour.

## <a id="memoization"></a>5. The held scratch

The one cache a call keeps is the decode step's per-sequence scratch (`decode_step.scratch`): a module-level
`WeakKeyDictionary` keyed on the state object, each value a dict keyed `(descriptor, BH, device)`, for device
allocations belonging to ONE live sequence. The weakness is required
rather than incidental: a dropped sequence must drop its buffers, and two live
sequences must never share one.

**The held scratch is REBUILDABLE TRANSIENTS ONLY**, and the state is its KEY
rather than an input of the build: the build is a function of `(descriptor, BH, device)`, and
never reads the sequence at all. That is what makes the handle-IS-the-sequence invariant survive caching — two
handles onto one geometry get interchangeable objects, and a fork or a rollback
inherits nothing through the cache (`docs/internals/state.md` §9.1).
`tests/unit/test_decode_scratch.py` builds the scratch with a state that
raises on any attribute read, so the property is proven rather than declared.

A PRIMITIVE whose answer is a function of the loaded binary rather than of any call fact is
cached by nothing here: the binary's arm census (`fsla.ops.decode.arms`) is
read by the bench provider, never by a call.

## <a id="refusals"></a>6. Refusals are one call at the head

Each kernel family has ONE `validate_context`, run first in its call, before any step. It consolidates the operand-type refusals, the descriptor's checks against the call and the gradient refusal (the prefill's device and unbuilt-kernel refusals are made earlier, where its descriptor is). Scattering preconditions across the steps
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

## <a id="the-plan"></a>7. What a launch bundle may hold

A host field is some rule's verdict; a tensor field is a reference to a terminal
fact. **No bundle field may be `O(N * L)`**: a launch bundle is held for as long as its call is,
so a field added for one launch's convenience is retained with it.

A value can also be an OPERAND, and the engine does not pretend otherwise: the
state-out plane is both a launch argument and the call's result.

## <a id="neutrality"></a>8. The neutrality gate

A host refactor that claims neutrality is answerable to what the extension RETURNED —
`y`, the state plane and the facts themselves — off a fixed cell set, compared across
two checkouts. Comparing the returned objects rather than re-deriving them is what
makes the comparison independent of whatever host structure produced the launch
arguments. The mechanism is the diff node
([`docs/measurement.md`](../../measurement.md#the-build-everything-declared-a-run-pruned-by-label)):
two sides, the cells, a rule and an expectation, so "against `HEAD~1`" is a second
checkout the suite's root already composes rather than a second tool.
