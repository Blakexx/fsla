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

There is no runtime depth field in `ConsumerParams`: a depth the kernel could
read would be a second accounting of a compile-time fact. The agreement gate
`tests/unit/test_ring_depth_agreement.py` pins planner depth == compiled depth,
and that pin is the whole of the invariant (coverage-closure H3).

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
is derived from an amplitude magnitude (harvest.md §23.2 ruling 1;
conformance audit item 3 / S1 — the §3 step 2–3 inversion).

Precisely (Gate D): amplitude comparisons do exist in the kernel, at its
`if (out != 0.0f)` split-K emission sites. Those are PERFORMANCE SKIPS ON A VALUE — adding `0.0f` is the identity, so
including the term is a no-op and the test is over-inclusion-safe in the
direction that matters. None of them decides membership, which is what this word
is the authority for. This is the
[self-masking theorem](../architecture/self-masking-theorem.md) in its
"trustless flag" form.

## <a id="dials-fp32"></a>A6. `dials` — fp32, never bf16

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

**A page is one `MMA_K_QUANTUM`-leaf ATOM and not one owner.** §1b makes `BC` a
re-derivable COMPUTE tile and the page a MEMORY-RESIDENCY granule, so that a
per-launch `BC` never rebuilds the arena — a `BC`-keyed page would. An
owner is therefore `QPO = BC / MMA_K_QUANTUM` independently paged runs, and
`atom_row_base` in `consumer_kernel.cuh` is the one place that arithmetic lives.

## <a id="page-table"></a>A8. `page_table` and why paging is bitwise invisible

`[BH, atoms_per_bh]` int32, or `nullptr` for the unpaged path. `kAbsentPage`
(`-1`) means the atom has no committed slot, which can only happen for an atom
with no write candidate at all — its unpaged state is identically zero.
Admission is TOTAL — there is no capped mode and no admit mask — so there is no
second reason for an absent entry.

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
collapses to the unsplit kernel WITHOUT reading memory — which is what keeps
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

**The stash is a correctness structure, not an optimization.** Without it, pass
3's segment 0 reads `p.state`, whose rows the SAME launch's last segment writes
from another CTA — an unsynchronized intra-launch dependence that only the work
list's incidental ordering hides. That failure is real and not hypothetical: a
permitted permutation of the work list yields `y` with relative error 1.0
(measured). With it, every pass-3 read is either a
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

## <a id="tracer-retired"></a>A13. There is no intra-kernel tracer

The consumer carries ZERO trace-emission sites, and no tracing header, host
capture translation unit or capture script exists in the tree. An instrument
pointed at no kernel is dead code by the standing rule, so instrumenting the
consumer means adding emission sites to it — a piece of work with its own
sites, not the revival of a switch that is already there. See
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

## <a id="one-consumer"></a>B1. ONE consumer

There is exactly one consumer kernel family. `launch_consumer` resolves the plan
to one instantiation of it, and the file includes one kernel header
(`consumer_kernel.cuh`) and one intra lowering header (`intra_panel.cuh`).

The consequences are the shape of the whole dispatch surface: there is no
second-kernel routing argument, no `intra_lowering` argument, no analytic SMEM
mirror beside the compiled one, and no second occupancy oracle. **What the
planner prices and what the driver launches are the same object by
construction** — not two accountings held together by a gate.

Pre-extraction removals, with the repository and commit they are readable at, are
in [`DELETIONS.md`](DELETIONS.md).

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

§4.5's page granule IS `arch_table.cuh`'s `kMmaKQuantum`
([arch_table.md#mma-quantum](arch_table.md#mma-quantum)) — ONE definition, and that
is exactly what lets `scan_kernel` address the state the way the consumer does while
this TU deliberately does not include the consumer kernel's header.
`rola.ops.paging.MMA_K_QUANTUM` is the Python mirror;
`tests/unit/test_paging_atoms.py` plus the paged equivalence fixtures are what keep
it honest.

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

The read it stands in for is not free either — an unstashed segment 0 reads the
same numbers out of `p.state` — so the added TRAFFIC is one dense store per split
owner, and the added SPACE is the one slot that takes §6.4's bound from `(P-1)`
to `P` per split owner. A non-resident owner stashes zeros, which is exactly what
an unstashed segment-0 read returns, and is the
[self-masking theorem](../architecture/self-masking-theorem.md)'s identity-store
consequence again. See A10 for the failure this prevents.

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

### The 512 arms are compiled in eight shards, not one translation unit

The instantiations do not live in this file. They live in eight generated
translation units under `csrc/rola/src/instantiations/`, partitioned on
`PREC x GLOBAL x BT` at 64 arms each, and `consumer.cu` reaches them through host
wrappers it can DECLARE but not define
([`consumer_launch.md`](consumer_launch.md#shard-only)).

MEASURED, this development host (16 cores, 23 GB), both figures including the
post-build ratification gate:

| | one translation unit | eight shards |
|---|---|---|
| `nvcc` wall for the consumer's arms | 411 s (one process, `--threads=2`) | 8 x ~43 s across `MAX_JOBS` processes |
| peak resident set per `cicc` | 4.4 GB | 0.61 GB |
| `consumer.cu` itself | — | 37 s, zero device entries |

The peak-RSS collapse is what raises `MAX_JOBS` from 2 to 7 on this host
(`tools/build_flags.py` derives it from free RAM), and that multiplies with the
split rather than adding to it. The full-cycle numbers are in
[`docs/build.md`](../build.md).

`consumer.cu` emits exactly two device entry points of its own — `scan_kernel` and
`divide_kernel`, neither a template. If it ever emitted a `consumer_forward_kernel`,
nothing would be sharded and the compile time would simply stop falling; that is
what `tools/fatbin_entries.py` refuses.

## <a id="shard-rows"></a>B9. One table, three enumerations retired

The dispatch used to be a macro ladder, the residency query a second one, the
shared-memory footprint a third, and a probe translation unit under `tools/` a
fourth — four hand-maintained spellings of one 512-element set, each able to drift
from the others without anything failing.

There is now ONE definition, `tools/gen_shards.py`'s matrix, and everything else is
generated from it:

| consumer | what it gets |
|---|---|
| the eight shard translation units | 64 explicit instantiations each |
| `instantiations/dispatch_launch.inc` | the launch ladder `launch_consumer` includes |
| `instantiations/dispatch_rows.inc` | the concatenated `ShardRow` tables |
| `tools/ratify.py` | the translation units to compile, and the manifest's `shards` block |

A `ShardRow` carries an arm's structural coordinates and the three things only its
own shard can answer: `launch`, `occupancy`, and the `constexpr` `smem_bytes`. Both
device-backed queries below resolve a row rather than re-enumerating the matrix,
which is why this file does not include the kernel header at all.

The launch ladder assigns a `LaunchFn` and returns nothing; `launch_consumer` calls
it and checks the CUDA error state once. Taking the address of a specialization this
translation unit cannot define is also what makes the split real: a dispatch arm with
no shard member CANNOT be satisfied by a silent re-instantiation here.

**It is not a link error, and the difference is the reason two gates exist.** This is
a Python extension module, so `ld -shared` leaves the reference undefined for the
loader — correctly; every CPython and torch symbol in the module is resolved that way,
which is also why `--no-undefined` is unavailable. A dropped arm therefore leaves one
unresolved `launch_consumer_arm` symbol and would reach a shipped artifact as an
`ImportError`. The closure is checked instead, at both ends of the compile:

| gate | when | what it refuses |
|---|---|---|
| `tools/gen_shards.py --check` | before the compile (`setup.py`, CI, `tests/unit/test_shard_partition.py`) | a committed shard that disagrees with the matrix definition |
| `tools/fatbin_entries.py --check` | after the link (`setup.py`) | a binary with an unresolved per-arm wrapper, a duplicated or missing device entry, or a ratified name it does not carry |

Measured by removal: dropping one arm from its shard leaves exactly one undefined
symbol, and the post-link gate names its tuple three ways.

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
worst case. `GLOBAL` and `PREC` are likewise pinned to the plan point the planner
prices: `global`, and the production precision.

Both queries resolve a `ShardRow` (B9) and read one field of it —
`occupancy()` calls the driver through the shard's wrapper, because post-shard the
kernel ADDRESS lives in the shard; `smem_bytes` is a `constexpr` the shard computed
at compile time. The wrapper is instantiated for all 512 arms rather than the 32
queried, which is why adding a query axis needs no matching change in the shards
([`consumer_launch.md`](consumer_launch.md#residency)).

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
branch. Both decay arms honour it: under `LeafMassDecay` a tile reduces to the
`decay = None` tile through the panel factorization
([`intra_panel.md`](intra_panel.md)), so the same two GEMM sites take the same
stamped format and no stamped field is ignored on either arm.

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
