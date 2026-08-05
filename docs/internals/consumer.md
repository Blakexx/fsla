# `csrc/rola/src/consumer.{cuh,cu}` — the consumer's contract and its host side

Two source files share this doc.

- **Part A — the header (`consumer.cuh`)**: `ConsumerParams`, the owner-geometry
  arithmetic, and `ScanParams`. Only what the kernel and the host binding must
  agree on.
- **Part B — the translation unit (`consumer.cu`)**: the launch surface, the
  dispatch closure, the residency arithmetic, the three-pass driver.

The kernel body itself is [`consumer_kernel.md`](consumer_kernel.md).

Spec references are to `ROLA_V3_SPEC.md`: §3 topology-agnostic kernel structure,
§4 schedule, §5 consumer, §6.4 sequence parallelism, §7.1 the arena.

Symbols: `bh` = batch-head; `BC` = leafs per compute tile; `BT` = tokens per
tile; `DV` = value width; `D` = number of levels; `j` = `first_spanned_level`;
`P` = the sequence-parallel split factor; `M` = state.

---

# Part A — the header

## <a id="topology-agnostic"></a>A1. The kernel is topology-agnostic by construction

It never learns which levels are dense, sparse, inner or outer. Every level is
described by the same four integers (`base`, `stride`, `modulus`, `row_offset`)
and is walked by the same code. §3's *"every apparent special case is a
degeneracy of this path"* is implemented literally:

- a **fixed** level is `stride = BC, modulus = 1`, which makes its digit constant
  over the owner's leafs;
- a **fully spanned** level is `base = 0, modulus = width`.

There is no branch on level kind anywhere in the kernel, and no `LevelDesc`-shaped
type: the per-level fields are flat arrays in `ConsumerParams`.

## <a id="leaf-digit"></a>A2. The leaf-digit decomposition

One level's contribution to the leaf-digit decomposition of a local leaf index
`i` in `[0, BC)`:

```
digit(i) = base + (i / stride) % modulus
```

with, per §3's owner rectangle:

| level | `base` | `stride` | `modulus` |
|---|---|---|---|
| fixed, `l < j` | the owner's fixed digit | `BC` | `1` |
| boundary, `l = j` | `slice * c` | `inner` | `c` |
| spanned, `l > j` | `0` | `prod(widths[l+1:])` | `width_l` |

`row_offset` is the level's first row in the packed tile-bitmap array.

`leaf_digit` is declared in the header rather than in the `.cu`'s anonymous
namespace because `intra_panel.cuh`'s clock prefix needs it: the clock is a
product over ALL levels (it is an exponent, so a fixed level cannot hoist into
the row scale), which is the one operand assembly that walks the leaf digits
outside the kernel translation unit.

## <a id="ring-depth"></a>A3. The staging-ring depth is compile-time

`consumer_ring_depth()` in `consumer_kernel.cuh` resolves it per instantiation
from `ROLA_C_RING_DEPTH` (shipped at 1; depth 2 DECLINED on measurement,
session-9 R2(b)), and the ratification manifest records the macro
(`tools/ratify.py`'s `RATIFIED_DEFINES`).

`ConsumerParams` once carried a runtime `ring_slots`. No kernel line read it and
it was removed (coverage-closure H3). The agreement gate
`tests/unit/test_ring_depth_agreement.py` pins planner depth == compiled depth,
which is the invariant the removed field was pretending to carry.

## <a id="clock-stream"></a>A4. The `LeafMassDecay` clock stream

`c[t,s] = prod_l p_write_stored[t, d_l(s)]` (spec §2), staged by
`build_clock_stream`. That is a **separate builder from the schedule**, so the
STORED tensor can never key a tile bitmap (P5.20.5 finding 2). `nullptr` under
`decay = None`.

The product runs over ALL levels including `l < j`: the clock is an exponent, so
a fixed level's factor does not hoist into the row scale the way the read and
write factors do.

## <a id="support-word"></a>A5. `read_tok_bits` / `write_tok_bits` — the sole support authority

§4's SUPPORT WORD, materialized: `[BH, n_tiles, total_rows]`, bit `t` = token
`tile * BT + t` (`BT <= 32`, spec §6). Same level-digit rows as `read_bits`, one
granularity finer — `read_bits` is literally its tile reduction, built from it —
but TILE-MAJOR, so one visit reads one contiguous slab.

**This is the sole support authority.** §3 step 2's gather, step 1's FIXED-level
token validity, and the staging predicate all read it, and no MEMBERSHIP decision
is derived from an amplitude magnitude any more (harvest.md §23.2 ruling 1;
conformance audit item 3 / S1 — the §3 step 2–3 inversion).

Precisely, because an earlier wording overclaimed (Gate D): amplitude comparisons
still exist in the kernel, at its `if (out != 0.0f)` split-K emission sites.
Those are PERFORMANCE SKIPS ON A VALUE — adding `0.0f` is the identity, so
including the term is a no-op and the test is over-inclusion-safe in the
direction that matters. None of them decides membership, which is what this word
is the authority for. This is the
[self-masking theorem](../architecture/self-masking-theorem.md) in its
"trustless flag" form.

## <a id="dials"></a>A6. `dials` — fp32, never bf16

Flattened `LeafMassDecay` dials, `[H, total_rows]`, level `l` at `row_offset[l]`.
`rate[s] = prod_l delta_l[d_l(s)]` — a plain product of DIALS (spec §2), not V2's
AND over per-branch retentions, so V2's pairwise log-union composition has no V3
counterpart.

Accumulated in fp32 and never bf16: `rate` near 1 makes `1 - rate`
catastrophically cancelling, and `log1pf` cannot repair a cancellation that has
already happened inside the product. See
[fp32 accumulators](../architecture/fp32-accumulators.md).

## <a id="state-layout"></a>A7. `state` — the unpaged and paged layouts, and the atom

Unpaged: `[BH, N, DV (+1 mass column under global)]`; atom `q` of owner `o` of
batch-head `bh` starts at row `bh * N + o * BC + q * MMA_K_QUANTUM`.

Paged (spec §4.5): `[physical_slots, MMA_K_QUANTUM, cols]`; that same atom starts
at row `page_table[bh * atoms_per_bh + o * QPO + q] * MMA_K_QUANTUM`.

**A page is one `MMA_K_QUANTUM`-leaf ATOM and not one owner.** `page_states := BC`
is dead: §1b makes `BC` a re-derivable COMPUTE tile and the page a
MEMORY-RESIDENCY granule, so that a per-launch `BC` never rebuilds the arena. An
owner is therefore `QPO = BC / MMA_K_QUANTUM` independently paged runs, and
`atom_row_base` in `consumer_kernel.cuh` is the one place that arithmetic lives.

## <a id="page-table"></a>A8. `page_table` and why paging is bitwise invisible

`[BH, atoms_per_bh]` int32, or `nullptr` for the unpaged path. `kAbsentPage`
(`-1`) means the atom has no committed slot, which can only happen for an atom
with no write candidate at all — its unpaged state is identically zero.
Admission is TOTAL (the capped mode and its admit mask are deleted), so there is
no second reason for an absent entry.

An absent atom reads as zeros and is never stored back. That is what makes paging
BITWISE INVISIBLE rather than approximately equivalent, and it is the
[self-masking theorem](../architecture/self-masking-theorem.md)'s
identity-store consequence used directly.

## <a id="work-list"></a>A9. §6.4 sequence parallelism — the flat work list

The grid is a FLAT `(owner, segment)` work list: `owners` carries the
`(bh, owner)` pair as always, and `work_seg` / `work_nseg` / `work_slot` carry
the segment coordinates, one entry per work item, in the same order.

Segment `s` of an owner split `nseg` ways owns the contiguous TILE range
`[s*n_tiles/nseg, (s+1)*n_tiles/nseg)`. A `nullptr` here is `P = 1` everywhere:
`seg = 0`, `nseg = 1`, the range is the whole token axis, and every expression
collapses to the pre-`P` kernel WITHOUT reading memory — which is what keeps
`P = 1` bit-identical, and `P = 1` is the primary benchmarked configuration
(spec §6.4).

## <a id="workspace"></a>A10. The partial-state workspace, and why the stash is correctness

LAUNCH-SCOPED and NOT the §7.1 arena. The arena's contract is persistent
cross-chunk state pages only, and mixing a transient lifetime into it would put
launch-scoped garbage under the compactor. The framework's caching allocator owns
this (a plain torch tensor, sized by the plan); what is unified is the planner's
LEDGER, not the allocator.

A split owner owns `nseg` slots starting at `slot[owner]` — §6.4's amended
`P * BC * d_v` peak — laid out so that pass 3's segment `p` reads exactly
`slot[owner] + p`:

```
slot[owner]         : the ENTRY-STATE STASH M_0, dense, written by pass 2 and
                      read by pass 3's segment 0.
slot[owner] + 1 + s : segment s's boundary, s in [0, nseg-1).
```

**The stash is a correctness structure, not an optimization** (fix pass R1).
Without it, pass 3's segment 0 read `p.state`, whose rows the SAME launch's last
segment writes from another CTA — an unsynchronized intra-launch dependence that
only the work list's incidental ordering hid. A permitted permutation of that
list produced `y` with relative error 1.0. With it, every pass-3 read is either a
workspace slot written by an EARLIER LAUNCH or a `p.state` row of an UNSPLIT
owner, which no other CTA touches — disjoint from all same-launch writes by
construction.

That failure is the sharp edge of the schedulability payoff in
[no-delta-rule.md](../architecture/no-delta-rule.md): the affine recurrence makes
the work list a free permutation, and a structure that is only correct under one
permutation is therefore a bug, not a convention.

## <a id="boundary-slot"></a>A11. A boundary slot carries two things in the same bytes

Deliberately, at two different points of the three-pass life:

| after | contents |
|---|---|
| pass 1 | the SPARSE written-leaf summary — `ws_count[slot]` entries, leaf ids in `ws_leaf`, per-leaf `A` in `ws_scale`, per-leaf `B` in the first `ws_count` rows of `ws_state` |
| pass 2 | the DENSE accumulated boundary state, leaf-indexed, written in place over the summary the scan has just consumed |

§6.4 states both: the summary is normatively sparse (its traffic scales with
realized writes), and the accumulated boundary pass 3 reads is dense. The stash
slot uses only `ws_state`; its `ws_scale` / `ws_leaf` / `ws_count` entries are
neither written nor read (the scan's fold indexes `slot[owner] + 1 + s`).

## <a id="pass-mode"></a>A12. `pass_mode` is launch-uniform

Every CTA in a launch has the same value, so the branches it drives are uniform
and are STRUCTURE, not a distributional property — the same status as `is_write`.
This is the branch-on-structure rule
([planner.md](../architecture/planner.md)) applied to a kernel-internal branch.

`PassMode`:

- `kPassRead = 0` — **pass 3**. Reads emitted, entry state from the scan, and the
  LAST segment publishes the owner's final state. At `P = 1` this is the whole
  kernel and is unchanged.
- `kPassSummary = 1` — **pass 1**. Entry state ZERO, no read emitted (a read-only
  tile cannot move the state, and a read emitted here would be double-counted by
  pass 3), publishes the sparse `(A, B)`.

Pass 2 is `scan_kernel`, a separate launch, so only the two consumer passes appear
in the enum. `ScanParams` composes the per-segment summaries into the accumulated
boundary states, in order: one CTA per SPLIT owner, and the fold is sequential
over segments inside it, so there is no inversion anywhere (P5.21's obligation).

## <a id="tracer-retired"></a>A13. The intra-kernel tracer is retired

Item 6b, 2026-08-01. `v3_trace.cuh`, `v3_trace_host.{cu,cuh}` and
`development/measurement/rola_trace{,_capture}.py` were an instrument with 27
`V3T*` emission sites, ALL of them in the REGISTER-RESIDENT consumer — whose
header was also named `consumer_kernel.cuh` (readable at `c3c2f845`, and a
different kernel from today's file of that name).

It is deleted; the surviving consumer carries zero emission sites. An instrument
pointed at no kernel is dead code by the standing rule, so it was retired WITH the
kernel rather than left to look live. Re-instrumenting the surviving consumer is a
new item with its own emission sites, not a revival. See
[`DELETIONS.md`](DELETIONS.md).

---

# Part B — the translation unit

## <a id="what-is-here"></a>B0. What is in `consumer.cu`, and nothing else

- the Torch-level entry `rola_consumer_forward` — argument checking, the
  `ConsumerParams` fill, and §6.4's three-pass sequencing;
- the launch dispatch `launch_consumer`, which resolves the plan's structural
  axes to ONE template instantiation of `consumer::consumer_forward_kernel`;
- the two non-template helper kernels, `scan_kernel` (§6.4 pass 2's ordered
  fold) and `divide_kernel` (`global`'s `num/den` epilogue);
- the driver-backed residency and footprint queries, and the arch-table check;
- the pybind surface.

The kernel itself is [`consumer_kernel.md`](consumer_kernel.md), which is also
where the §3/§5 algorithm lives. This file deliberately does not restate it: two
descriptions of one kernel is one that can go stale.

`rola_consumer_forward` consumes exactly the Stage 3 schedule
(`rola/planner/schedule.py`) plus the value stream and the dense state, and
returns `(y, final_state, num, den)`. Every argument is checked loudly; there is
no silent shape coercion and no CPU fallback.

## <a id="one-consumer"></a>B1. ONE consumer (item 6b, 2026-08-01)

The previous REGISTER-RESIDENT consumer is deleted: its kernel header (also named
`consumer_kernel.cuh`, and a DIFFERENT kernel from the one this file now
includes), plus `intra_gram.cuh`, `consumer_matrix.def` and its eight shard
translation units, its `intra_lowering` template axis, its analytic SMEM mirror,
and its intra-kernel tracer.

There is one kernel family, so there is no second-kernel routing argument, no
`intra_lowering` argument, and no second occupancy oracle: **what the planner
prices and what the driver launches are the same thing by construction**, rather
than by a gate that pins two accountings together.

The deleted consumer is readable in the source repository named by
[`docs/provenance.md`](../provenance.md) — `flash-linear-attention-rola`, branch
`rola-v3`, at commit `c3c2f845` (2026-08-01). That hash resolves in that
repository and **not** in this one, whose history begins at the extraction. Any
bare hash in this tree that is not annotated this way is a citation a reader
cannot follow.

## <a id="three-pass"></a>B2. The three-pass split is DORMANT BY PLAN, not dead

`P > 1` machinery — pass 1's sparse summary, `scan_kernel`, pass 3's stash read,
the `work_seg`/`work_nseg`/`work_slot` triple, and the fourteen optional
sequence-parallel arguments — is fully implemented, oracle-gated, and **not
currently selected by the planner**: `P = 1` is the primary benchmarked
configuration and the plan chooses it everywhere today.

This is recorded explicitly because it looks like dead code and is not. It is
reachable (the driver takes `P` from the plan), it is tested
(`tests/tier2_geometry/`), and its unblock condition is a measured cell where a
split wins — the same standing as `panel` in
[`intra_panel.md`](intra_panel.md#measured-status). Deleting it would delete a
measured-and-kept capability. [`DELETIONS.md`](DELETIONS.md) records it as
dormant rather than removed.

## <a id="all-or-nothing"></a>B3. The §6.4 arguments are all-or-nothing

Fourteen optional tensors, and the check is `n_sp == 0 || n_sp == 14`. A partial
split plan would run some passes and not others, which is **not a degraded answer
but a wrong one** — hence a hard check rather than a filled-in default.

The three passes are sequenced in C++ rather than by three Python calls, because
they are one contract: pass 3 is only correct against the boundary pass 2
published from the summary pass 1 emitted, and interposing a Python frame would
let a caller run them out of order or skip one. They are ordinary stream-ordered
launches — no device synchronization is taken and none is needed.

`launch_pass` is ONE dispatch point for every consumer launch the call makes (the
unsplit one, and passes 1 and 3), written once so that a pass cannot pick a
different kernel from the one its siblings picked. `P` is a WORK-LIST SHAPE, not
a kernel: both consumer passes read `work_seg`/`work_nseg`/`work_slot`/
`pass_mode` out of the same parameter block.

Pass 1's grid is the summary work list, which OMITS every owner's last segment:
that segment's outcome is the owner's final state and pass 3 produces it
directly, so summarizing it would be a `P`th of the work for nothing. The owner's
slot run is nonetheless `P` slots — `P - 1` boundaries plus pass 2's entry-state
stash.

## <a id="ordered-fold"></a>B4. `scan_kernel` — the ordered fold

One CTA per SPLIT owner; the composition is sequential over that owner's segments
inside it. `GRADIENT_PROOFS_WOLFRAM.md` P5.21's obligation is that the fold is
ORDERED and that nothing anywhere inverts `A` — both hold literally: the loop
runs `s = 0 .. nseg-2` ascending and the only operation is `E := A*E + B`. No
inverse of `A` exists anywhere in the engine, which is the
[no-delta-rule](../architecture/no-delta-rule.md) commutativity payoff cashed in.

The boundary is published DENSE, in place over the sparse summary the same slot
just supplied (A11). That is what keeps the per-boundary cost one slot rather
than two.

The owner's entry state is addressed exactly as the consumer addresses it (§4.5:
a page is one `MMA_K_QUANTUM`-leaf ATOM, so an owner is `QPO = BC / 16`
independently paged runs), so a paged and an unpaged fold see the same numbers.
`BC` is a runtime value here, so the per-atom base is resolved per row rather
than in an unrolled loop; this is pass 2 of a split launch, one CTA per SPLIT
owner, and the resolution is an L1-resident int32 load.

`kScanAtomLeaves` is §4.5's page granule, mirrored from
`rola.ops.paging.MMA_K_QUANTUM` and equal by construction to
`consumer_kernel.cuh`'s `kMmaKQuantum` — one number, and
`tests/unit/test_paging_atoms.py` plus the paged equivalence fixtures are what
keep the mirrors honest. It is declared in this TU because `scan_kernel` lives
here and must not include the consumer kernel's header.

## <a id="written-leaf-fold"></a>B5. The fold touches the written-leaf list ONLY

`E := A*E + B` runs over `ws_count[slot]` entries. A leaf absent from the list is
not read, not written, and not arithmetically touched — which is what makes
§6.4's unwritten-leaf carve-out BITWISE rather than merely equal, and is why the
carve-out gate is a genuine detector of fold bugs rather than a concession.

Emitting an identity row for an absent leaf would be numerically exact too, but
it would make the invariant depend on `fmaf(1, x, 0) == x` instead of on the leaf
never being visited. The stronger invariant is the one that fails loudly.

## <a id="entry-stash"></a>B6. The entry-state stash

Slot `slot0` holds `M_0` itself, dense, published in PASS 2 — a launch that
strictly precedes pass 3. Pass 3's segment 0 reads it instead of reading
`p.state`, which is what makes pass 3's reads disjoint from pass 3's writes: the
last segment of the same owner stores those exact `p.state` rows from another CTA
of that launch, with no ordering between them.

The read this replaces was not free before — segment 0 read the same numbers out
of `p.state` — so the added TRAFFIC is one dense store per split owner, and the
added SPACE is the one slot that takes §6.4's bound from `(P-1)` to `P` per split
owner. A non-resident owner stashes zeros, which is exactly what its segment 0
read before, and is the
[self-masking theorem](../architecture/self-masking-theorem.md)'s identity-store
consequence again. See A10 for the failure this fixed.

## <a id="arch-table"></a>B7. The arch table is validated against the driver

`__launch_bounds__` needs compile-time constants, so `sm_resources()` is a table
keyed on `__CUDA_ARCH__` rather than a driver query. A table can be stale or
mis-keyed in a way a query cannot, and the failure mode is SILENT: a bound derived
from the wrong resource numbers is either DISCARDED by `ptxas` (out of range) or
honoured at the wrong residency. So the table is CHECKED against `cudaDeviceProp`
for the device actually running, and a mismatch is a hard error — a
compile-target/hardware mismatch must fail loudly rather than quietly cost
occupancy.

**This does not make the table a planning authority.** Spec §6.5: the driver's
occupancy API remains the sole authority on residency, and
`rola_consumer_occupancy` — which calls it — is what every residency claim is
read from. The table's only job is the launch bound.

The check runs ONCE per process, on the first launch rather than at import: a
`cudaGetDeviceProperties` at module init would create a CUDA context as a side
effect of `import`, which is a worse behaviour than the check is worth. By the
first launch the context exists anyway, and nothing can reach a launch bound
without passing through here.

`rola_sm_shared_bytes` is the same discipline on the other side: §10's ruling
records the SM shared-memory budget in the PLAN rather than reading it at launch,
so that a plan is reproducible and its residency reasoning is auditable off the
machine that made it. The planner's `DEFAULT_SM_SHARED_BYTES` is the CPU-only
fallback and is deliberately a plan field, not a hidden default.

## <a id="instantiation-matrix"></a>B8. The instantiation matrix, and why every axis is a template

```
BT(2) x BC(2) x DV(2) x D(4) x PREC(2) x DECAY(2) x GLOBAL(2) x COMPACT(2)
```

Each is a STRUCTURAL plan fact, fixed before launch, with few legal values — the
standing "branch on structure, not thresholds" rule
([planner.md](../architecture/planner.md)), and the same status `BT` and `BC`
have always had. `D` is the topology's level count (spec §1: `1 <= D <= 4`);
`DECAY` changes which terms exist; `GLOBAL` changes the STATE'S SHAPE (the `+1`
mass column) and so cannot be a runtime test without making `cols` a runtime
value in every state address.

**There is deliberately no runtime fallback on any of them.** An out-of-range
value is a hard `TORCH_CHECK`, never a slower path: one comprehensive path ships.

### The count is measured, not asserted

The number of instantiations is whatever the macro product evaluates to, and the
authority is the ratification manifest, which carries one entry per
instantiation keyed by mangled name. Measured on the current tree
(`tools/manifests/sm_86.json`, identical on `sm_80.json`):

| kernel family | entries |
|---|---|
| `consumer_forward_kernel` | 512 |
| `decode_gemv_kernel` | 32 |
| `union_forward_kernel` | 32 |
| `union_backward_kernel` | 32 |
| **total per architecture** | **608** |

A prose count in a source comment is a number that rots the first time an axis
moves. `python tools/ratify.py` recompiles the closure and compares; see
[`docs/ratification.md`](../ratification.md).

### Compile cost is watched, not assumed

The figure that belongs in the record is one measured on this tree. Whole-extension
`python setup.py build_ext --inplace`, `MAX_JOBS=2`, on the development machine
(`workflows/refactor/p03_format.md`): **23m14s** cold, **19m53s** on the second
build, both including the post-build ratification gate.

A per-translation-unit figure for `consumer.cu` alone is **not currently
measured**; the honest statement is the whole-build one. If a later axis makes
the build cost unacceptable, a `.def`-plus-shard-TU split — the mechanism the
deleted consumer's matrix used (B1) — remains the answer, and the measurement
that would justify it is a per-TU `nvcc` wall-clock under the same `MAX_JOBS`.

## <a id="braces"></a>B9. HAZARD — the macro braces are load-bearing

Every `ROLA_CONSUMER_ONE` is a braced block ending in `return`, so the enclosing
dispatch is a chain of `if (cond) BLOCK`. `ROLA_CONSUMER_CMP` expands to TWO such
blocks, and **without the outer braces the second one escapes its guard and
launches unconditionally.** That is exactly what an earlier revision did: every
call ran the `decay` instantiation, which dereferenced a null `p.dials`.

It was caught by the compact gate's own dense arm — by the instrument the axis was
added with — and not by anything that reads like a check. This is a syntax-level
hazard with a semantic blast radius, which is why it carries a stub.

## <a id="occupancy-axes"></a>B10. Residency is per-instantiation, so the query has axes

`rola_consumer_occupancy` asks the DRIVER, and it takes `decay`, `compact`, `BC`,
`BT` and `DV` because a single reported number would over-promise on one arm and
under-report another. The whole point of deriving `__launch_bounds__` from
`consumer_smem_bytes` is that a decay instantiation which regressed the budget
would drop its OWN bound rather than silently over-promise; this is how that is
checked from Python rather than asserted.

Why each axis is on the query:

| axis | what it moves |
|---|---|
| `DECAY` | three extra `[BC]` vectors |
| `COMPACT` | `c_idx`/`c_slot`/`c_count`; `compact + decay` is the tightest residency cell in the kernel (~140 B of margin at 3 CTAs/SM) |
| `BC` | which term BINDS: at `BC = 64` shared memory binds (3 CTAs), at `BC = 16` the REGISTER FILE does (4) |
| `BT` | `BT = 32` doubles the y-fragment count (register term) and grows `WT`/`Gm` (SMEM term) |
| `DV` | `d_v = 32` halves the state block |

A residency claim about "the kernel" that quoted only `BC = 64, BT = 16,
d_v = 64` would be a claim about an eighth of the matrix.

`D` is NOT a query axis: it changes only the `[D][BC]` digit table (256 B per
level), and the query is fixed at `D = 2`, the at-scale cell's topology depth, so
the number is reported for the shape actually measured rather than for a nominal
worst case.

## <a id="plan-surface"></a>B11. The supported shape surface

`BT in {16, 32}`, `BC in {16, 64}`, `d_v in {32, 64}` — and these are the
planner's own supported sets (`_SUPPORTED_BT`, `_SUPPORTED_BC`, `_SUPPORTED_DV`),
so the kernel instantiates exactly the configurations a plan can name: **no shape
a plan can select is refused, and no shape it cannot select is compiled.**
`launch_consumer`'s `TORCH_CHECK` is the single authority for that surface.

`mma_precision` selects BOTH GEMM sites together (RULING 1 revised): `"bf16"` is
the production point, `"tf32x3"` the exactness-gate point against the fp64
oracle. Both are instantiated. It is a numeric FORMAT axis stamped by the plan
(findings D4.3) and resolved once — not a tolerance knob and not a density
branch. Both decay arms honour it: a read-only tile under `LeafMassDecay` takes
the §5 readout, and its MIXED tiles take the token-sequential walk, which is fp32
and has no operand format to choose, so no stamped field is ignored.

## <a id="state-derivation"></a>B12. Under paging, `N` is derived rather than read

§4.5: paging changes the state's SHAPE, not its addressing rule, and the page
granule is the MMA ATOM rather than the owner. Unpaged the state is
`[BH, N, cols]`; paged it is `[physical_slots, MMA_K_QUANTUM, cols]` and `N` is
then not readable off the state — so it is derived from the owner partition,
which is the definition: `N = owners_total * BC`.

## <a id="support-shape-check"></a>B13. The support word is shape-checked into place

§4's support word is TILE-MAJOR, and it is checked against the tile bitmap it is
the refinement of (same `total_rows` level-digit rows) and against the amplitude
stream's tile axis. A stale or mis-shaped schedule — one built at a different
`BT` — therefore cannot reach the gather. `BT` itself is checked into `[1, 32]`
because the support word carries one bit per token of a tile (spec §6).

## <a id="split-k-emission"></a>B14. The split-K emission sites

The kernel emits its readout and its intra term as separate split-K atomics
guarded by `if (out != 0.0f)`. Two facts about that guard, both load-bearing:

1. It is a PERFORMANCE SKIP, not a membership decision (A5) — licensed by the
   [self-masking theorem](../architecture/self-masking-theorem.md).
2. It is what makes the intra term a second atomic per (token, channel) on a
   mixed tile, which is cause #1 in
   [`intra_panel.md`](intra_panel.md#measured-status) for `panel` measuring
   slower than the walk. Fusing the intra term into the readout epilogue is the
   standing follow-up for BOTH lowerings.
