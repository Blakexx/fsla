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
  atom bitmap (`rola.ops.chunk.stats`, the kernel's own write set) is planned into
  slots, and the kernel resolves one slot per atom through the page table. Resident
  state scales with the atoms the routing REALIZES, not with the `N` the topology
  provisions.
* **dense** — one `[BH, N/16, 16, cols]` plane, byte for byte the atom-major view of
  `[BH, N, cols]`. Selected by `expert=PlanOverrides(paging=False)`, and it is what
  the bit-identity gate compares the paged backing against.

`paging=True` is REFUSED: paging is on by default, so there is nothing to turn on, and
the expert overrides the MAPPING only. Dense and paged are ONE kernel, one ABI, one
accumulation order and one base translation — `torch.equal`, not a tolerance
(`tests/tier2_geometry/test_chunk_paging_equivalence.py`). A caller cannot observe
which backing a state has except by asking (`state.paged`), and no result depends on
the answer. A state never switches backing mid-life: there is one arm, and it reads
whichever backing the state bound.

`materialize() -> [B, H, N, cols]` gathers the state into a dense tensor. It is for
the oracle, for a cross-arm hand-off and for a test. **Never on a measured path**:
allocating `[B, H, N, cols]` is exactly what paging exists to avoid.

## <a id="host-order"></a>4. The host order of a stateful kernel call

```
stats(routes)         -> atom bitmap        # BC-free, so it precedes the arm
state._kernel_entry   -> plan_exact on the arena's SIDE STREAM, one size read
chunk.run             -> union table, arena.wait(), launch, divide
state._kernel_commit  -> the final plane is the state's
```

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

`rola.ops.decode.decode_forward(routes, v, state, decay=..., expert=...)` takes and
returns the state exactly as `rola_op` does. Three things live behind that seam:

* **The frozen carrier and the scratch** (`FrozenDecodeConfig`, `DecodeWorkspace`)
  are MEMOIZED IN `rola.ops.decode`, not on the state — launch machinery fails the
  membership test. The carrier is a pure function of its key (widths, `d_v`, the decay
  arm, `BH`, device, the precision pin), so a plain module-level dict is exactly
  right: identical keys are the identical frozen object. The workspace is scratch
  belonging to ONE live state, so it is keyed by the state object in a
  `WeakKeyDictionary` — a dropped sequence drops its scratch, and two concurrent
  sequences can never share a buffer.
* **The backing is invisible.** `_decode_entry` hands back the arena's plane and table,
  or a VIEW of the dense plane — the view is what makes the kernel's in-place update
  the state's update. Dense and paged compute the same bytes
  ([`decode.md` §4](decode.md#paged-address)).
* **Growth is CONDITIONAL, and the condition arrives from the device.** `_decode_entry`
  takes the step's write-atom bitmap or `None`. A bitmap means the device growth probe
  found an atom this step writes that the table does not map, and the entry plans it
  with `plan_exact(..., fresh=False)` — residency becomes the UNION and the launch waits
  on the one event. `None` means there is nothing to admit, and the entry hands back the
  residency the state already has without touching the arena at all
  ([`decode.md` §4b](decode.md#growth)).

A decode step CONTINUES, so `_decode_entry` refuses a state that carries nothing —
there is no prefill at one token and no entry state to read. That is a computational
impossibility and not a policy, which is the only kind of refusal this surface has.

## <a id="clone"></a>6. `clone()` is the transition

`clone()` is an EAGER deep copy: a paged state's copy gets its own arena with the same
residency and its own pages; a dense state's copy gets its own plane. Nothing is
shared, and nothing else is copied — a clone has no rings and no scratch to copy,
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

Three items are designed and NOT BUILT. They are here rather than in a workflow ledger
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

### 9.2 A CUDA-graph-captured decode step

A steady paged decode step is HOST-WAIT-BOUND: the growth probe's `[BH]` int32 read is
0.12–0.16 ms of a 0.22–0.24 ms step, and the floor is the WSL2 host's own
launch-to-visible latency rather than anything tunable. The derivation and the two
structural exits — capture the step, or consume the verdict on the DEVICE — are
[`decode.md` §4b](decode.md#graph-capture).

### 9.3 Lifting the 4 GiB state-plane bound

The chunk consumer's state addressing is a 32-bit byte displacement from a pinned plane
origin, which bounds a plane at 4 GiB (~1,032,192 pages) and is refused loudly above it.
The dense backing hits it first and the paged backing IS the answer to it; when a cell
does reach it, the two backings need different lifts (one per-`bh` 64-bit base for dense,
arena sharding for paged). Stated at
[`chunk/chunk_kernel.md`](chunk/chunk_kernel.md#plane-bound).
