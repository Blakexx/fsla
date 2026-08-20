# The stats pass — the exact atom-grain write bitmap

Mirror doc for `atom_bitmap_kernel` and for the atom-grain half of
`chunk_facts_tables` (`csrc/rola/src/facts/facts_kernel.cuh`), their launchers
(`facts_launch.cuh`), the `facts` torch entry (`facts.cu`) and its host callers
(`rola/engine/facts/nodes.py`'s `facts_tables` node, `rola/engine/facts/planes.py`). The page
arena that consumes the bitmap is `paging.md`; the addressing it keys is
`AtomKeying` in `rola/ops/paging.py`.

**The atom rows are not a pass of their own.** They are voted in the same launch
that builds the owner-grain union table, out of the same amplitude row, because the
plane is the byte cost and it grows with `T`
(`docs/internals/facts/facts_kernel.md#the-facts-pass`). What is keyed to the atom
quantum — and therefore free of the launch's `BC` — is the ROW GRAIN and the
reduction below, not the pass.

Vocabulary, defined once: `D` = level count; `B` = one uniform level width;
`N = B^D` = leaves; `L` = tokens; `BH` = batch*heads; `BC` = leaves per owner
block; **atom** = the 16-leaf page granule (`kAtomLeaves`, mirrored by
`MMA_K_QUANTUM`); `atoms = N/16`; `tok_words(L) = ceil(L/32)`.

## What it computes, and why it is EXACT

A leaf is written at a token iff EVERY level's digit for that leaf is live on
the write side; a leaf is written by the call iff that conjunction holds at SOME
token. The union table stores, per `(bh, entry)`, a per-token ballot of "some
digit inside this entry's aligned run is live". Two reductions are available
over it, and they are not equivalent:

* **OR over tokens first, AND over levels second.** Loose in the TOKEN axis:
  level 0's digit live at `t1` and level 1's at `t2` admits a leaf that was
  never live at any single token. A superset.
* **AND over levels per token word, OR over tokens second** — what this pass
  does. For a rectangle of leaves (an aligned digit run per level), the per-token
  AND says "at this token, every level has SOME live digit inside the rectangle".
  Any such combination of live digits IS a leaf of the rectangle and its liveness
  IS that conjunction, so the answer is exact **at the grain of the rectangle it
  is evaluated on**.

The grain is therefore the whole question, and it is why this table is built at
`BC := kAtomLeaves` rather than at the launch's `BC`: a `LevelPlan<D,B,16>`
block IS one atom, so the second reduction lands exactly on the atom bitmap.
Evaluated on the launch `BC` the same expression is exact per OWNER and then
padded to the owner's atoms, which is an overcount at atom grain AND makes
residency a function of `BC` — the property the atom re-key exists to destroy.

Coarser derivations are safe-direction (a superset commits pages that stay zero,
and a zero page reads inert) but they are not shipped: they break
`resident == realized`, which the arena's accounting reports as a fact.

## The identity the kernels are built on

At `BC = 16` the atom index EQUALS the block index of `LevelPlan<D,B,16>`:

    blockBase(blk) / 16 = sum_l off_at(l, blk) * prod_{m>l} B / 16 = blk

because levels spanned WHOLE carry one group (digit zero), and every surviving
level has span product 1 above it. So the consumer's own clause-row resolve,
instantiated at the 16-rectangle, names exactly the atom's rows — one
derivation, two callers, no second expression to drift.

Two corollaries:

* the innermost clause level's digit is the LEAST significant part of `blk`
  (its suffix group count is 1), so the atoms of one upper prefix are the
  contiguous run `blk = prefix * kInner + i` and their innermost rows are the
  contiguous entry run `prefix_entries(last) + i`;
* `clauses == 0` happens exactly when `blocks == 1`, i.e. `N == 16`: one atom,
  unconditionally live, no table to consult.

## The carve, and what it costs

Naive shape — one CTA per `(bh, atom)`, each ANDing `clauses` rows of
`tok_words(L)` words — reads `BH * atoms * clauses * tok_words * 4` bytes, which
at `N = L = 65536`, `BH = 64` is gigabytes. The shipped shape is one CTA per
`(bh, upper prefix)`: the shared upper rows are ANDed into a resident word stream
ONCE and each of the `kInner` innermost rows is ANDed against that stream and
OR-reduced, so reads drop to `BH * prefixes * (uppers + kInner) * tok_words`.
The token axis is walked in tiles of `kAtomTileWords`, so a CTA's shared
footprint does not depend on `L`.

An atom is live as soon as one word survives, and a live atom is never read
again; when every atom of the CTA is live the CTA leaves. The exit is
CTA-UNIFORM (one warp reads the flags behind a barrier and publishes one answer),
so no lane can leave a `rendezvous` alone. This makes the DENSE routing the
cheap case rather than the expensive one.

Rejected alternates, recorded: fusing the REDUCTION into the facts pass (the pass
is token-word parallel, the reduction is atom parallel with a cross-word OR —
fusing needs a grid-wide reduction or atomics into the bitmap, and nothing in this
engine takes a grid barrier), and token-scatter (enumerate each token's live digit
combinations and OR them in: beautiful under sparsity, worst-case dense, and it
makes the cost a routing property rather than a plan property). The row-BUILDING
half is a different question and it did fuse: it shares the facts pass' grid, its
lane-per-token geometry and its side of the residency dependency
(`docs/internals/facts/facts_kernel.md#the-facts-pass`).

## The contract

`facts(pr, pw, chunk, bc, widths, mask, atoms) -> (tbl, bits | None)`, with
`bits` the `[BH, N/16]` uint8 write-side bitmap — write side only, because reads
of never-written atoms are inert. Two launches on the current stream, no host
synchronization, and the atom-grain table is a transient allocated and freed
inside the entry.

The BITMAP carries no `BC` although the entry does: its reduction's arm list is
`(D, B)` alone, derived from the consumer's arm list by projection
(`tools/gen_shards.py`), so the same routing gives the same residency at every
built arm — which is what makes residency an OUTPUT of the plan instead of a
consequence of it. Asking for the bitmap ALONE (`rola/engine/facts/planes.py`'s
`atom_bits`, off the measured paths) therefore names an arm and discards a union
table; the shipped path asks for both at once, which is the point of the pass.

**Who reads it, and what rests on its exactness.** `PageArena.plan_exact` admits
a slot for each of its atoms, and the chunk consumer then resolves every atom
through the resulting table (`docs/internals/chunk/chunk_kernel.md`). An atom the
bitmap omits gets no slot, and the kernel SKIPS it at both ends of the state I/O
— which is sound precisely because this bitmap is the write set exactly: an
omitted atom was never deposited into, so its fragment is zero, and storing zero
or storing nothing are the same state. The safe direction is the SUPERSET one: a
bitmap that overshoots costs footprint and nothing else, which
`tests/integration/test_chunk_paging_equivalence.py` gates by forcing a
never-written atom resident and demanding a bit-identical answer. A bitmap that
UNDERSHOOTS drops state silently, and nothing downstream can detect it — which is
why exactness is gated here, at the source.

The gate is `tests/oracle/test_atom_bitmap.py`: exact equality against
`atom_bitmap_from_schedule` and against an independent fp64 recomputation from
the routing amplitudes, including the two-token cross-product cell that is the
one case separating this derivation from every looser one.
