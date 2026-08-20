# The carried state, and the facade that engages it

Mirror doc for `rola/_state.py` and for `rola_op`'s stateful half in
`rola/interface.py`. It is the contract the repository carries: the design ledger
that produced it is a workflow record, and this file is what a reader ten years from
now is owed.

Vocabulary, defined here and used unqualified below: `N` = leaf count;
`cols` = `d_v` plus one mass column under `normalization='global'`;
`BH` = `batch * heads`; **atom** = 16 consecutive leaves in canonical leaf order,
which is the page granule; **slot** = the physical page an atom is mapped to;
`BC` = leaves one owner block owns.

## The two axes: strategy vs backing

``strategy`` is the write-semantics (aliasing) contract; the backing is
storage mechanics the plan chooses at bind. Orthogonal; all four cells
coherent; dense x cow legal-but-degenerate (full-plane copy per advance).
The dense-vs-paged bit-identity gate is the backing-invisibility claim
made mechanical.

## <a id="one-object"></a>1. A PAGED TENSOR, and one predicate

```python
s = rola.state()                        # unbound
y, s = rola.rola_op(routes, v, state=s) # bound to this call's shape; s is mutated
y, _ = rola.rola_op(routes, v)          # stateless: (y, None)
```

`rola.state` IS THE CONSTRUCTOR, unambiguously: the class and the function live in
`rola/_state.py`, private by its underscore, so no module answers to the public name.

A `RoLAState` is a paged tensor and nothing more:

| it holds | what it is |
|---|---|
| storage | a `PageArena`, or one dense plane |
| the page table | `[BH, N/16]` slot ids, `None` when dense (`state.page_table`) |
| the logical shape | `[B, H, N, cols]` (`state.shape`) |
| device, dtype | `state.device`, `state.dtype` — the STORAGE's |
| `strategy` | storage mechanics: what a write does to this object's own storage |
| `pool_slack` | storage mechanics: the slack pool's size, a FRACTION of full residency |

<a id="the-slack-pool"></a>**`pool_slack`** (`rola.state(pool_slack=0.0)`, default `0.0`)
is the other storage-mechanics knob, and it lives here for the same reason `strategy`
does: it governs what a paged arena PRE-COMMITS, not what a call computes. A fraction
above `0.0` reserves that fraction of each batch-head's full residency as physical slots
a decode step can admit into ON THE DEVICE, with no host round trip — `0.0` is EXACT
COMMITMENT, unchanged from before the pool existed. The mechanics — sizing, the
pre-zeroed ordinary slots, the device-side claim and its shadow row, exhaustion and
refill — are [`paging/paging.md` §the-slack-pool](paging/paging.md#the-slack-pool); what
the pool changes at THIS seam is [§5b](#decode-seam).

**THE MEMBERSHIP TEST is a question, and it is the whole design rule: would a paged
tensor know this?** A tensor knows its layout and its contiguity, so `strategy` stays.
A tensor does not know which routing widths produced it, whether a convolution ran
before it, or what launch scratch some kernel wants — so the widths/normalization
fingerprint, the short-convolution rings, the decode carrier and the decode workspace
are all GONE from here, each to the level that owns it ([§5b](#decode-seam),
[§7](#conv-rings)). Accretion onto a carried state is the failure mode this rule
exists to leave nowhere to land.

**The predicate is `state is not None`**, at the op and at the layer alike. There is
no second flag, because the kernel's store epilogue is skipped by a NULL POINTER: a
stateless call runs a variant with no state I/O compiled into it at all and pays
nothing for the feature.

## <a id="binding"></a>2. Binding is SHAPE ACQUISITION, and there is no interview

A fresh state has a strategy and nothing else. The FIRST call that receives it takes
`[B, H, N, cols]`, the device and the backing kind from the call itself. `cols` needs
`d_v`, which is NOT on the routing — the value width is the value stream's own fact —
so `rola_op` sources it from `v.shape[-1]`.

Later calls are NOT checked against what was bound. A state used against a call it
does not fit fails where the bytes are addressed — a reshape or an index raises —
exactly as it does for a `torch` tensor handed to the wrong op.

**The named accepted hazard**: two states of the same `N` and `cols` built from
DIFFERENT widths have different leaf orders, and interchanging them is silent. That is
the caller's risk, deliberately: a widths/normalization fingerprint on the state is a
second contract to keep in step with the first, and every mechanism that carried one
grew luggage. Resubmitting an older state, sharing one between two sequences, and
continuing after a `clone` are likewise legal and the caller's semantics to own.

`strategy="cow"` is refused AT CONSTRUCTION, by name: copy-on-write branching is
designed (per-branch tables, refcounted slot GC) and not built, and its settled
semantics are [§9.1](#cow). `"mutable"` is the built strategy: the call writes in
place and returns THE SAME OBJECT.

## <a id="backing"></a>3. The backing: dense or paged, and invisible either way

The backing is chosen at BIND time and is not a per-call flag:

* **paged** (the default for every stateful call) — a `PageArena`. The call's exact
  atom bitmap (`rola.engine.facts.planes.atom_bits`, the kernel's own write set) is planned into
  slots, and the kernel resolves one slot per atom through the page table. Resident
  state scales with the atoms the routing REALIZES, not with the `N` the topology
  provisions.
* **dense** — one `[BH, N/16, 16, cols]` plane, byte for byte the atom-major view of
  `[BH, N, cols]`. Selected by `expert=PlanOverrides(paging=False)`, and it is what
  the bit-identity gate compares the paged backing against.

`paging=True` is REFUSED: paging is on by default, so there is nothing to turn on, and
the expert overrides the MAPPING only. Dense and paged are ONE kernel, one ABI, one
accumulation order and one base translation — `torch.equal`, not a tolerance
(`tests/integration/test_chunk_paging_equivalence.py`). A caller cannot observe
which backing a state has except by asking (`state.paged`), and no result depends on
the answer. A state never switches backing mid-life: there is one arm, and it reads
whichever backing the state bound.

`materialize() -> [B, H, N, cols]` gathers the state into a dense tensor. It is for
the oracle, for a cross-arm hand-off and for a test. **Never on a measured path**:
allocating `[B, H, N, cols]` is exactly what paging exists to avoid.

## <a id="backing-policy"></a>4. What the state contributes to a kernel call

The ORDER of a stateful call is the chunk DAG's and is stated there
([`engine/chunk_dag.md` §4](engine/chunk_dag.md#host-order)). What is the STATE's own
is this: `_kernel_entry` plans the call's atoms into slots and hands back
`(s_in, arena)`, and `_kernel_commit` takes the final plane for a state that just BOUND
a dense backing — a paged state owns the arena's plane, and a dense continuation was
launched into its own.

### <a id="the-dense-continuation"></a>The continuation advances IN PLACE, in either backing

`s_out` and `s_in` are ONE plane for a continued sequence, in either backing, so a
stateful chain allocates one plane for the SEQUENCE and not one per call. The design
reason it is worth stating: a second whole-written `[BH, N, d_v + 1]` fp32 plane is
hundreds of megabytes at the parity curve's `N`, it scales with the growth axis, and a
transient whose only job is to be swapped in buys nothing the caller can observe.

**What makes it safe is the CTA structure, not a convention.** The owner blocks tile the
leaf space exactly, so each atom belongs to exactly one CTA and, inside it, to one slot
group: the deleted consumer's `atom_first()` was `blockIdx.y`, the block's BC-aligned leaf
base and the group's offset, and the entry fold and the store epilogue iterate the SAME
`kAtoms` from the SAME base. The entry fold runs ONCE, in the prologue, before the first
super-chunk is staged; the store runs ONCE, in the epilogue, after the last drain. So no
CTA ever reads an atom another CTA may already have stored, and no CTA reads its own
after storing it. Nothing in the walk between them touches the plane — which is why a
short or ragged final super-chunk changes none of this: the state fold sits outside the
chunk walk entirely.

**The cost is that no stateful call is transactional.** A launch that faults part way
through the grid leaves the plane part advanced, in either backing. Recovering a
pre-call state is `clone()` before the call, and a caller who needs that guarantee is
paying for the second plane deliberately rather than on every call.

Two consequences follow and are stated where they belong: a dense continuation can SKIP
blocks, because a skipped block leaves its own atoms carrying what they held
([`engine/rules.md#run-table`](engine/rules.md#run-table)), and a FRESH dense bind cannot —
its plane is uninitialized and is written whole.

`plan_exact`'s single device-to-host size read is the only host synchronization in
the sequence, and it is the read that also SIZES the admission. A fresh state plans
`fresh=True` (call-scoped, the unpaged "allocate zeros every call" semantics); a
continuation plans `fresh=False`, so residency becomes the UNION — which is what
continuing a sequence means.

`BC` arrives FROM THE CALL (`_kernel_entry(..., BC=...)`) and is never stored. It is
allocator policy — it resolves the arena's default extent — and every ADDRESS is
`BC`-invariant by the atom re-key, so a `BC` flip between two calls on one live state
neither rebuilds the arena nor moves a byte. `clone()` reads the widths and `BC` it
needs off the arena it is copying, which is where they already live.

## <a id="arms"></a>5. There is one arm, and no bridge

The fp64 oracle is the executable SPEC, not an arm of the op
([`api.md` §1.1](../api.md#11-the-built-envelope-is-the-surface)): a caller that wants
it calls `naive_rola` and passes raw tensors. So no state ever crosses to it through
the facade, and the dense-contents bridge that existed to carry it (`_dense_entry`,
`adopt_dense_contents`, the re-admission of dense contents into pages) is GONE
(`DELETIONS.md`). What is left is one kernel arm reading the backing its state bound.

## <a id="decode-seam"></a>5b. The `T = 1` seam

`rola.engine.dags.decode_dag.decode_forward(routes, v, state, decay=..., expert=...)` takes
and returns the state exactly as `rola_op` does. Three things live behind that seam:

* **The frozen carrier and the scratch** (`DecodeGeometry`, `DecodeScratch`) are the
  decode DAG's own memoized nodes, not the state's — launch machinery fails the
  membership test. The ORDER and the two memo disciplines are that DAG's page
  ([`engine/decode_dag.md` §5](engine/decode_dag.md#host-carrier)); what the state owes
  them is only its identity, which is what the `per_state` table is keyed on.
* **The backing is invisible.** `_decode_entry` returns `(plane, page_table, arena)`: the
  arena's plane and table (and its slack pool, ridden along for the pool buffers alone —
  a dense state has none), or a dense state's VIEW of its own plane with the other two
  `None`. The view is what makes the kernel's in-place update the state's update. Dense
  and paged compute the same bytes ([`decode/decode.md` §4](decode/decode.md#paged-address)).
* **Growth is CONDITIONAL, and the condition arrives from the device.** `_decode_entry`
  admits nothing itself: it hands back the residency the state already has, plus the
  pool a paged step can admit ITSELF into on the device
  ([`paging/paging.md` §the-slack-pool](paging/paging.md#the-slack-pool)) — a claim the
  pool covers never reaches this seam at all. When the step's own verdict finds a
  batch-head that writes an atom neither the table nor the pool can address,
  `_decode_entry_arena` plans it with `plan_exact(..., fresh=False)` — residency becomes
  the UNION and the launch waits on the one event — and the step is walked again
  ([`decode/decode.md` §4b](decode/decode.md#growth)).

A decode step CONTINUES, so `_require_carrying` refuses a state that carries nothing —
asked once by the decode family's `validate_context` before the step's first launch, and
again at `_decode_entry` for a caller reaching the seam directly.
there is no prefill at one token and no entry state to read. That is a computational
impossibility and not a policy, which is the only kind of refusal this surface has.

## <a id="clone"></a>6. `clone()` is the transition

**THE CONTRACT IS STRATEGY-INVARIANT.** `clone()` returns an independent sequence:
either handle may be passed into any op with no effect on the other, ever. The strategy
chooses the MECHANISM — dense: plane copy; paged: referenced-slot copy; cow, future:
table copy + refcounts — and the contract does not vary. That is what makes COW a
drop-in optimization behind this method ([§9.1](#cow)) rather than a second semantics.

Today's mechanism is an EAGER deep copy: a paged state's copy gets its own arena with
the same residency and its own pages; a dense state's copy gets its own plane. Nothing
is shared, and nothing else is copied — a clone has no rings and no scratch to copy,
because it holds none.

It is also the ONLY sanctioned strategy change (`clone(strategy=...)`). A live mutable
ancestor is a standing alias onto shared slots, so a cheaper transition would hand a
descendant a guarantee its ancestor can break. Strategies never mix within a lineage:
crossing strategies crosses a clone, and the cost is visible at the point it is paid.

## <a id="conv-rings"></a>7. CONTINUATION IS LAYERED

The op's state is the paged tensor. The LAYER returns its own thin bundle:

```python
y, (state, conv_rings) = layer(x, state=(state, conv_rings))   # always the pair
```

`conv_rings` is `None` when no convolution is configured. The rings are genuinely part
of the recurrence — a continuation that dropped them would restart the causal
convolution from zeros and compute a different function for the first `conv_size - 1`
tokens of every step — but they are the LAYER's recurrence, introduced by the layer.

**The membership test, one level up: an object holds exactly the recurrences ITS OWN
level introduced.** Op: the routed state. Layer: the conv rings. Model: the per-layer
bundles. This is not the old pooled state object recreated — that defect was layer
luggage INSIDE the op's object, and this keeps each object pure while giving accretion
nowhere to land. The pair is the contract in BOTH directions, so a bare `RoLAState`
handed to `forward` is refused by name rather than adopted: accepting the op's half
would make the rings silently optional to carry.

## <a id="not-a-kv-cache"></a>8. What is deliberately absent

No `transformers` cache vocabulary (concatenation, eviction, window rolling,
`get_max_cache_shape`): all of it is built for a state that GROWS with the sequence,
and RoLA's does not. No token tally, no per-layer container: one state object belongs
to one layer's sequence, and a stack of layers is a list of them in the caller's hands.
No allocator cap — the footprint is planned exactly, so a ceiling over it is either
never-binding or a self-inflicted refusal.

The `transformers` / `flash-linear-attention` adapters that presented a foreign cache
to the old per-stack state object were DELETED with it (docs/internals/DELETIONS.md).
A shim onto this surface is a thin wrapper at a model boundary if one is ever wanted;
it is not a second stateful surface, and the layer will not import either package.

## <a id="future-work"></a>9. NAMED FUTURE WORK

Two items are designed and NOT BUILT. They are here rather than in a workflow ledger
because the repository has to carry its own contracts: a reader ten years from now gets
the design, not an archaeology exercise. None of them is scheduled.

### <a id="cow"></a>9.1 Copy-on-write state branching — the SEMANTICS, fixed

`strategy="cow"` is refused by name today ([§2](#binding)). What it will mean when it is
built is settled, and the settlement is what must not be re-litigated:

* **Strategy lives on the STATE, stamped at construction** (`rola.state(strategy=...)`),
  and it is a convenience stamp rather than a contract: the PLAN defaults to the state's
  strategy and an expert override wins unconditionally. Cross-branch clobbering that an
  override causes is the caller's business logic, not something this surface polices —
  the same rule that gives the object no step stamps and no lineage checks today.
* **Mutable** (the built strategy): writes in place; the call returns THE SAME object.
* **COW**: writes always to FRESH slots plus an updated table, so every state object ever
  returned stays valid, and resubmitting an older one IS branching — an implicit fork,
  not an error. A COW step therefore returns a NEW state object referencing a
  mostly-shared table. Slot lifetime is refcounted, and refcounts exist for COW's own
  garbage collection and for one data-integrity invariant: **a slot with refcount > 1 is
  never written in place.**
* **Mixing strategies inside a lineage is UNEXPRESSIBLE, not policed.** Changing a
  state's strategy is a full deep copy of its resident data to fresh pages, in BOTH
  directions — i.e. `clone(strategy=...)`, whose cost is visible at the point it is paid.
  A cheaper transition was designed and REJECTED: a live mutable ancestor is a standing
  alias onto shared slots, so its future in-place writes would break every COW
  descendant's validity guarantee, and the mirror-image residual sharing on the way back
  is the same defect reflected.
* **What it buys**: beam and speculative decode at per-branch memory proportional to the
  atoms a branch WRITES rather than to the state's size. The machinery is already here —
  per-branch page tables are the indirection, the stats pass gives the exact write set
  BEFORE the launch (so the copy is planned-exact and never faulted), and the kernel's
  own state store writes through the child's table while reading through the parent's,
  which makes a fork cost zero bytes.
* **The known cost, and its known fix**: at decode grain the per-step TABLE copy
  (`BH * N/16` int32 — about 1 MiB at `N = 65,536`, `BH = 64`) dwarfs the step's own
  state bytes. The answer is a two-level (paged) page table copying only diverged table
  pages. It is decided when COW is scheduled; eager `clone()` ships first and COW is its
  drop-in optimization behind the same contract.
* **Audited against the machinery that exists**, on the two paths a fork or a rollback
  touches that are not the state's own storage. THE PER-STATE CACHES: the decode DAG's
  scratch is keyed weakly on the state OBJECT and holds rebuildable transients only
  ([`engine/engine.md` §7](engine/engine.md#memoization)), so a COW step's new handle
  keys a new entry and builds its own buffers from the same geometry, and a rollback
  that resubmits an older handle finds that handle's own entry. Handle IS sequence in
  both directions; a cache holding a fact ABOUT the sequence would be the thing a fork
  had to copy or invalidate, and there is none. THE CAPTURE PATH: a branch is its own
  page table over the refcounted shared arena, and a captured step reads that table
  through a pointer, so a fork MUTATES THE TABLE IN PLACE and needs no re-capture —
  the growth admission is the precedent (`plan_exact(..., fresh=False)` rewrites the
  same table under the same capture, [`decode/decode.md` §4b](decode/decode.md#growth)).
  THE FORK TEST is an intersection of two sets that both already exist: the step's
  write-atom set, condensed on the device as `atom_bits`, AND the slots whose
  refcount is `> 1`. Those slots are copied before the write; every other write goes in
  place, which is what keeps a fork proportional to divergence.

### 9.2 Lifting the 4 GiB state-plane bound

The chunk consumer's state addressing is a 32-bit byte displacement from a pinned plane
origin, which bounds a plane at 4 GiB (~1,032,192 pages) and is refused loudly above it.
The dense backing hits it first and the paged backing IS the answer to it; when a cell
does reach it, the two backings need different lifts (one per-`bh` 64-bit base for dense,
arena sharding for paged). Stated at
[`chunk/chunk_kernel.md`](chunk/chunk_kernel.md#plane-bound).
