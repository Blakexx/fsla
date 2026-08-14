# The carried state, and the facade that engages it

Mirror doc for `rola/state.py` and for `rola_op`'s stateful half in
`rola/interface.py`. It is the contract the repository carries: the design ledger
that produced it is a workflow record, and this file is what a reader ten years from
now is owed.

Vocabulary, defined here and used unqualified below: `N` = leaf count;
`cols` = `d_v` plus one mass column under `normalization='global'`;
`BH` = `batch * heads`; **atom** = 16 consecutive leaves in canonical leaf order,
which is the page granule; **slot** = the physical page an atom is mapped to;
`BC` = leaves one owner block owns.

## <a id="one-object"></a>1. ONE object, one predicate

```python
s = rola.state()                        # unbound
y, s = rola.rola_op(routes, v, state=s) # bound to this call's keying; s is mutated
y, _ = rola.rola_op(routes, v)          # stateless: (y, None)
```

`rola.state(strategy=...)` builds a fresh UNBOUND state. `RoLAState` holds a strategy
stamp, the keying it bound to, and its backing — and nothing else. There is no handle
tier, no step stamp, no lineage, no staleness check. A state is a buffer with a
contract, exactly as a `torch` tensor is; two callers sharing one is their business.

**The predicate is `state is not None`**, at the op and at the layer alike. There is
no second flag, because the kernel's store epilogue is skipped by a NULL POINTER: a
stateless call runs a variant with no state I/O compiled into it at all and pays
nothing for the feature, and a "give me no state back" flag would only buy a null
return and a second contract to keep in step with the first.

The layer mirrors the op exactly and is a PASS-THROUGH: `forward(x, state=...)`
returns `(y, state)`, unwrapping nothing and checking nothing.

## <a id="binding"></a>2. Binding, and the only refusals

A fresh state has a strategy and nothing else. The FIRST call that receives it binds
its keying (`N`, `cols`, `BH`), its topology fingerprint (the level widths and the
normalization), its device, and its backing. Every later call is checked against
those, and the check is MATH-COMPATIBILITY ONLY:

| refused | because |
|---|---|
| different level widths | a different leaf count addresses different rows |
| different normalization | a different `cols` is a different stride |
| different `(batch, heads)` | a different `BH` is a different keying |
| different device | different memory |

Each of those makes the call literally uncomputable against this state — the
wrong-shaped-tensor class of error. Nothing else is policed. In particular, resubmitting
an older state object, sharing one between two sequences, or continuing after a
`clone` are all legal and are the caller's semantics to own.

`strategy="cow"` is refused AT CONSTRUCTION, by name: copy-on-write branching is
designed (per-branch tables, refcounted slot GC) and not built. `"mutable"` is the
built strategy: the call writes in place and returns THE SAME OBJECT.

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
the answer.

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

## <a id="arms"></a>5. Crossing the arms

The reference arm is the fp64 oracle, and `naive_rola`'s raw-tensor signature is SPEC:
it does not learn about state objects. So the FACADE crosses the boundary in both
directions. An incoming state materializes to `[B, H, N, cols]` fp64; the oracle's
final state is stored back through `adopt_dense_contents`, in the dtype it arrives in,
so a reference-arm chain never rounds through fp32.

A PAGED state that takes the reference arm therefore becomes dense-backed. Returning to
the kernel arm RE-ADMITS: the dense contents' own atom occupancy (a nonzero reduction
at atom grain) is unioned with the call's write bitmap and planned into a fresh arena,
and the contents are scattered into their slots. Occupancy is exact in the safe
direction — an atom holding only zeros reads as zeros whether resident or absent (the
zero-page inertness gate) — so this admits at most a superset of what is carried.

It is deliberately NOT `PageArena.adopt_dense`, which commits the dense limit and
would reintroduce the footprint the arena exists to avoid.

The layer's LEGACY arms (the tiled consumer for topologies the chunk arm has no built
arm for, and the `T = 1` decode kernel) also take raw tensors. They cross the state
object at their branch and nowhere deeper. Aligning decode with the paged state I/O
is P67 S4.

## <a id="clone"></a>6. `clone()` is the transition

`clone()` is an EAGER deep copy: a paged state's copy gets its own arena with the same
residency and its own pages; a dense state's copy gets its own plane. Nothing is
shared.

It is also the ONLY sanctioned strategy change (`clone(strategy=...)`). A live mutable
ancestor is a standing alias onto shared slots, so a cheaper transition would hand a
descendant a guarantee its ancestor can break. Strategies never mix within a lineage:
crossing strategies crosses a clone, and the cost is visible at the point it is paid.

## <a id="conv-rings"></a>7. What rides along: the short-convolution rings

`state.conv_state` carries the layer's causal short-convolution ring buffers. They are
fixed-size and genuinely part of the recurrence — a continuation that dropped them
would restart the convolution from zeros and compute a different function for the
first `conv_size - 1` tokens of each step — but they are not routed state and the op
never reads them. The layer writes them; `clone()` copies them.

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
