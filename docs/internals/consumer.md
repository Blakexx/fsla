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

Symbols: `bh` = batch-head; `BC` = leafs per owner tile; `DV`/`d_v` = value
width; `D` = number of levels; `j` = `first_spanned_level`; `P` = the
sequence-parallel split factor; `M` = state.

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

`row_offset` is the level's first row in the packed token-bit array.

`leaf_digit` is declared in the header rather than in the `.cu`'s anonymous
namespace because `intra_panel.cuh`'s clock prefix needs it: the clock is a
product over ALL levels (it is an exponent, so a fixed level cannot hoist into
the row scale), which is the one operand assembly that walks the leaf digits
outside the kernel translation unit.

## <a id="clock-stream"></a>A3. The `LeafMassDecay` clock stream

`c[t,s] = prod_l p_write_stored[t, d_l(s)]` (spec §2), staged by
`build_clock_stream`. That is a **separate builder from the schedule**, so the
STORED tensor can never key a support decision (P5.20.5 finding 2). `nullptr`
under `decay = None`.

The product runs over ALL levels including `l < j`: the clock is an exponent, so
a fixed level's factor does not hoist into the row scale the way the read and
write factors do.

## <a id="support-word"></a>A4. `read_tok_bits` / `write_tok_bits` — the sole support authority

§4's SUPPORT WORD, materialized: `[BH, n_twords, total_rows]`, bit `t` = token
`word * 32 + t` — the 32-token MACHINE WORD is the metadata granule, so a
membership visit reads one contiguous slab. WORD-MAJOR, not tile-major: the
executor is a streamed scan over the token axis, not a tile-candidacy filter, so
there is no coarser structure for this array to be a reduction of.

**This is the sole support authority.** The membership scan's AND of the owner's
fixed-level rows against the token-granular slice union, and every leaf-axis
consumer downstream of a live group, read it, and no MEMBERSHIP decision is
derived from an amplitude magnitude (harvest.md §23.2 ruling 1; conformance
audit item 3 / S1 — the §3 step 2–3 inversion).

Amplitude comparisons do exist in the kernel, at the split-K emission sites
([B14](#split-k-emission)). Those are PERFORMANCE SKIPS ON A VALUE — adding
`0.0f` is the identity, so including the term is a no-op and the guard is
over-inclusion-safe in the direction that matters. None of them decides
membership, which is what this word is the authority for. This is the
[self-masking theorem](../architecture/self-masking-theorem.md) in its
"trustless flag" form.

## <a id="slice-union-tok"></a>A5. `slice_union_tok` — the token-granular slice union

`[BH, n_slices, n_twords]` uint32 (schedule ABI v4, the card's two union forms,
form (a)): bit `t` of word `[bh, slice, word]` = "some digit of the slice at the
boundary level is read- OR write-live at token `word * 32 + t`". Derived from
the same `read_tok_bits`/`write_tok_bits` authority in `build_schedule`, so it
cannot disagree with it. This is the membership scan's coarse term: the fine
terms it ANDs under are the owner's fixed-level `read_tok_bits`/`write_tok_bits`
rows. The scan streams `(j + 1)` words per 32-token word — one per fixed level
plus this one — at static addresses; there is no tile-candidacy question and no
coarse pre-filter above word granularity, because an intersection does not
survive coarsening.

## <a id="atom-union-bytes"></a>A5a. `read_atom_bits` / `write_atom_bits` — the atom-major contiguous union bitlists

`[BH, n_slices, QPO, n_twords]` int32 (schedule ABI v4, form (b); `QPO = BC /
kMmaKQuantum`, `n_twords = ceil(T/32)`; Blake's confirmed layout, 2026-08-06):
bit `t` of word `[bh, slice, q, w]` = "some leaf of atom `q` of this slice is
live at token `w*32 + t` on this side" (an atom is `kMmaKQuantum` consecutive
leaves). ATOM-MAJOR CONTIGUOUS: one atom's whole token-liveness row is
contiguous — the layout the per-atom fetch era streams without a migration.

These are the walk's per-side masks, consumed WORD-WISE, not by an OR into a
byte: for the taken word of a ballot round, atom `q`'s bit in the group's mask
= `(row q's word at the taken source word) & taken != 0` — one uniform load per
`(side, atom)` per ballot round, `<= 2 * QPO` total, cooked by the schedule
ring's producer warp (`consumer_kernel.cuh`,
[`consumer_kernel.md#producer`](consumer_kernel.md#producer)). The
compact-into-registers atom walk then visits only the atoms either mask sets;
`mask == kFullMask` selects the statically-unrolled dense arm, so dense pays no
mask instructions.

Derived host-side alongside `slice_union_tok`, from the same support authority
(`rola/planner/schedule.py`'s `_atom_union_bits`, one summaries pass,
consistency by construction). The per-(owner, word) pre-OR'd form is FORBIDDEN
by the materialization rule — the per-visit OR/AND is the kernel's job, not the
schedule's.

## <a id="amp-dtype"></a>A6. Amplitude/value storage dtype — per level role

The `amp_read` / `amp_write` / `v` pointers are type-erased (`const void*`)
because their HBM storage dtype is decided per LEVEL ROLE, not per
instantiation — there is no precision or normalization template axis left to
carry it:

| stream | dtype rule |
|---|---|
| SPANNED amplitude levels (`l >= j`) and `v` | bf16, unconditionally — the one precision story |
| FIXED amplitude levels (`l < j`) | always fp32 — they feed the per-token row scales, whose error is uniform over the whole output row and never cancels (Option B) |
| clock (`amp_clock`) | always fp32 — an exponent (see [A7](#dials-fp32) and `consumer_kernel.md#clock-prefix-fp32`) |
| `num` / `den` / `state` | always fp32 — accumulators ([fp32 accumulators](../architecture/fp32-accumulators.md)) |
| `y` | the divide epilogue's bf16 output; the divide itself runs fp32 over the fp32 accumulators |

Each kernel use site types its own pointer from its loop's level role, so a
missed site is a compile error. The cast to storage dtype happens in the
SCHEDULE BUILDER, strictly below support selection; the host binding
(`amp_class_ptr` in `consumer.cu`) hard-checks every bound tensor's dtype
against the role-implied one — a mismatch is a loud error, never a silent cast.

## <a id="amp-alignment"></a>A6b. Spanned amplitude rows are 16-byte aligned, and it is checked

A spanned level's amplitude row is a `cp.async` SOURCE: the consumer prefetches the
owner's digit range out of it in 16-byte chunks
([`consumer_kernel.md#prefetch`](consumer_kernel.md#prefetch)). Every token row must
therefore start on a 16-byte boundary, which for bf16 storage means a pitch that is a
multiple of 8 elements.

`Plan.amplitude_pitch` already pads to exactly this quantum — it has always padded a
token-word block to a whole number of `CP_ASYNC_ROW_BYTES` — so the requirement costs
the builder nothing. What the binding adds is the CHECK, on the pitch and on the
storage pointer, because the kernel's addressing depends on it in a way that would
otherwise fail silently on a stream some other builder produced.

The consequence in the kernel is not merely legality. With a padded pitch, a level's
alignment LEAD (`base_l & 7`) is the same for every token, so the slab's per-level
offset is a compile-time-indexed REGISTER instead of a per-`(level, token)` shared
table — and that table was most of what made the earlier whole-slab prefetch cost more
instructions than it saved (`workflows/perf/p11_cpasync.md`).

## <a id="dials-fp32"></a>A7. `dials` — fp32, never bf16

Flattened `LeafMassDecay` dials, `[H, total_rows]`, level `l` at `row_offset[l]`.
`rate[s] = prod_l delta_l[d_l(s)]` — a plain product of DIALS (spec §2), not V2's
AND over per-branch retentions, so V2's pairwise log-union composition has no V3
counterpart.

Accumulated in fp32 and never bf16: `rate` near 1 makes `1 - rate`
catastrophically cancelling, and `log1pf` cannot repair a cancellation that has
already happened inside the product. See
[fp32 accumulators](../architecture/fp32-accumulators.md).

## <a id="state-layout"></a>A8. `state` — the unpaged and paged layouts, and the atom

Unpaged: `[BH, N, cols]` where `cols = DV + 1` — the mass column is
UNCONDITIONAL now, not a `global`-only shape. Atom `q` of owner `o` of
batch-head `bh` starts at row `bh * N + o * BC + q * MMA_K_QUANTUM`.

Paged (spec §4.5): `[physical_slots, MMA_K_QUANTUM, cols]`; that same atom starts
at row `page_table[bh * atoms_per_bh + o * QPO + q] * MMA_K_QUANTUM`.

**A page is one `MMA_K_QUANTUM`-leaf ATOM and not one owner.** §1b makes `BC` a
re-derivable COMPUTE tile and the page a MEMORY-RESIDENCY granule, so that a
per-launch `BC` never rebuilds the arena — a `BC`-keyed page would. An
owner is therefore `QPO = BC / MMA_K_QUANTUM` independently paged runs, and
`atom_row_base` in `consumer_kernel.cuh` is the one place that arithmetic lives.

## <a id="page-table"></a>A9. `page_table` and why paging is bitwise invisible

`[BH, atoms_per_bh]` int32, or `nullptr` for the unpaged path. `kAbsentPage`
(`-1`) means the atom has no committed slot, which can only happen for an atom
with no write candidate at all — its unpaged state is identically zero.
Admission is TOTAL — there is no capped mode and no admit mask — so there is no
second reason for an absent entry.

An absent atom reads as zeros and is never stored back. That is what makes paging
BITWISE INVISIBLE rather than approximately equivalent, and it is the
[self-masking theorem](../architecture/self-masking-theorem.md)'s
identity-store consequence used directly.

## <a id="work-list"></a>A10. §6.4 sequence parallelism — the flat work list

The grid is a FLAT `(owner, segment)` work list: `owners` carries the
`(bh, owner)` pair as always, and `work_seg` / `work_nseg` / `work_slot` carry
the segment coordinates, one entry per work item, in the same order.

Segment `s` of an owner split `nseg` ways owns the contiguous WORD range
`[s*n_twords/nseg, (s+1)*n_twords/nseg)` over the scan stream — `w_lo`/`w_hi` in
`consumer_kernel.cuh`, computed inline from `p.n_twords`. There is no separate
segment-mask header: a segment's range is arithmetic over the same
32-token-word axis the membership scan already streams, not a materialized
tile bitmap.

A `nullptr` here is `P = 1` everywhere: `seg = 0`, `nseg = 1`, the range is the
whole token axis, and every expression collapses to the unsplit kernel WITHOUT
reading memory — which is what keeps `P = 1` bit-identical, and `P = 1` is the
primary benchmarked configuration (spec §6.4).

## <a id="workspace"></a>A11. The partial-state workspace, and why the stash is correctness

LAUNCH-SCOPED and NOT the §7.1 arena. The arena's contract is persistent
cross-chunk state pages only, and mixing a transient lifetime into it would put
launch-scoped garbage under the compactor. The framework's caching allocator owns
this (a plain torch tensor, sized by the plan); what is unified is the planner's
LEDGER, not the allocator.

A split owner owns `nseg` slots starting at `slot[owner]` — §6.4's amended
`P * BC * cols` peak — laid out so that pass 3's segment `p` reads exactly
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

A boundary slot carries two things in the same bytes, at two different points of
the three-pass life: after pass 1 it holds the SPARSE written-leaf summary
(`ws_count[slot]` entries, leaf ids in `ws_leaf`, per-leaf `A` in `ws_scale`,
per-leaf `B` in the first `ws_count` rows of `ws_state`); after pass 2 it holds
the DENSE accumulated boundary state, leaf-indexed, written in place over the
summary the scan has just consumed. The stash slot uses only `ws_state`; its
`ws_scale` / `ws_leaf` / `ws_count` entries are neither written nor read (the
scan's fold indexes `slot[owner] + 1 + s`).

## <a id="pass-mode"></a>A12. `pass_mode` is launch-uniform

Every CTA in a launch has the same value, so the branches it drives are uniform
and are STRUCTURE, not a distributional property — the same status as
`is_write`. This is the branch-on-structure rule
([planner.md](../architecture/planner.md)) applied to a kernel-internal branch.

`PassMode`:

- `kPassRead = 0` — **pass 3**. Reads emitted, entry state from the scan, and the
  LAST segment publishes the owner's final state. At `P = 1` this is the whole
  kernel and is unchanged.
- `kPassSummary = 1` — **pass 1**. Entry state ZERO, no read emitted (a read-only
  segment cannot move the state, and a read emitted here would be
  double-counted by pass 3), publishes the sparse `(A, B)`.

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
  fold) and `divide_kernel` (the `num`/`den` epilogue);
- the driver-backed residency and footprint queries, and the arch-table check;
- the pybind surface.

The kernel itself is [`consumer_kernel.md`](consumer_kernel.md), which is also
where the §3/§5 algorithm lives. This file deliberately does not restate it: two
descriptions of one kernel is one that can go stale.

`rola_consumer_forward`'s actual signature (`consumer.cu`, `rola_consumer_forward`):

```
rola_consumer_forward(
    amp_read, amp_write, amp_clock?, dials?,
    read_tok_bits, write_tok_bits, slice_union_tok,
    read_atom_bits, write_atom_bits, owners, g_write, v, state, page_table?,
    level_stride, level_modulus, level_row_offset, level_width, owner_stride,
    first_spanned_level, slice_digits, n_slices, owners_total, n_heads, BC, eps,
    # §6.4, all-or-nothing (14 tensors or none):
    work_seg?, work_nseg?, work_slot?,
    summary_owners?, summary_seg?, summary_nseg?, summary_slot?,
    scan_owners?, scan_nseg?, scan_slot?,
    ws_state?, ws_scale?, ws_leaf?, ws_count?,
) -> (y, final_state, num, den)
```

There is no `BT` argument, no `global_norm` argument and no `mma_precision`
argument. `decay` is inferred from `dials.has_value()`; `D` is inferred from
`amp_read.size()`; `BC` is an explicit `int64_t`, checked against the plan
surface ([B11](#plan-surface)) rather than derived. Every argument is checked
loudly; there is no silent shape coercion and no CPU fallback.

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

## <a id="three-pass"></a>B2. The three-pass split is DORMANT, not dead

`P > 1` machinery — pass 1's sparse summary, `scan_kernel`, pass 3's stash read,
the `work_seg`/`work_nseg`/`work_slot` triple, and the fourteen optional
sequence-parallel arguments — is fully implemented and compiled into the binary.
No Python caller constructs a non-null argument set for it: `rola.ops.consumer`
always passes the fourteen arguments null, so every launch takes the single-pass
`P = 1` path.

This is recorded explicitly because it looks like dead code and is not. The
kernel-side mechanism is intact and reachable by any caller of the extension
directly (bypassing `rola.ops.consumer`). [`DELETIONS.md`](DELETIONS.md) records
it as dormant rather than removed. The endgame executor re-keys the split onto
WORD ranges rather than tile ranges; its shape (14 arguments, three passes, the
stash) is otherwise unchanged by that rewrite.

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
a page is one `MMA_K_QUANTUM`-leaf ATOM, so an owner is `QPO = BC / MMA_K_QUANTUM`
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
consequence again. See A11 for the failure this prevents.

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
BC(2) x DV(2) x D(4) x DECAY(2) x GT(2)  =  64 arms
```

`consumer_forward_kernel<BC, DV, D, DECAY, GT>`: `BC` in `{16, 64}`, `DV` in
`{32, 64}`, `D` in `{1, 2, 3, 4}` (spec §1: `1 <= D <= 4`), `DECAY` in
`{false, true}`, `GT` in `{16, 64}` — the WINDOW, the consumer's token quantum:
how many gathered tokens one VISIT stages and consumes against one entry state
before folding back. `GT = 16` is one producer group per visit (`W = 1`); `GT =
64` is four (`W = 4`), amortising every per-visit term `W`-fold against a Gram
that grows quadratically in `W` ([`smem_layout.md#window`](smem_layout.md#window)).
`128` is not instantiated — the Gram term does not fit `sm_86`'s opt-in SMEM
at the flagship shape. There is no `BT` axis, no `PREC` axis and no `GLOBAL`
axis — those three were the plan surface's precision, tiling and normalization
knobs and are dead; the mass column and the `num`/`den` divide are
unconditional now (`cols = DV + 1` always), and the amplitude/value storage
dtype is a function of level role, not a template choice ([A6](#amp-dtype)).
Why they died, and the ratchet evidence: `DELETIONS.md`; the axes that
remain are the ones `tools/gen_shards.py`'s own header names as the survivors.

`GT` is a PLANNER-SELECTED axis rather than a shape-derived one: the other four
axes come from the routing's structural shape (topology, dtype, decay), but the
window is chosen widest-first over `CONSUMER_WINDOWS` by
`rola.planner.schedule.select_window`, which runs its refusals CHEAPEST-FIRST
because it sits on the launch path beside kernels that take a fifth of a
millisecond: the device's SMEM grant (a cached driver fact), then RESIDENCY — a
wider window is admitted only where it costs NO CTA per SM against the group
quantum — then, on the decay arm, `panel_exponent_bound`'s window-scaled
admission bound, then last the routing's member-fill statistic
(`union_member_median`, the one refusal that costs a device-side sweep of the
slice-union bitmap, memoized on the schedule so it is paid at most once).
MEASURED on `sm_86`, the residency refusal is TOTAL: `select_window` returns
the group quantum on every arm measured, because widening `GT` costs CTA
residency (`BC = 16`, 3 -> 2 CTAs; `BC = 64`, 3 -> 1) and the loss tracks the
residency given up, not the window's own extra arithmetic. What that residency
is buying is NOT the panel gather's memory latency: the panels are prefetched
([`consumer_kernel.md#prefetch`](consumer_kernel.md#prefetch)), the prefetch
cuts `long_scoreboard` by 26-46%, and the wide window's penalty does not move
(`workflows/perf/p39_prefetch.md`). The rule is ARCH-GENERAL, not a pin to this
box: `architecture/planner.md#7` states it and its evidence.

Each remaining axis is a STRUCTURAL plan fact, fixed before launch, with few
legal values — the standing "branch on structure, not thresholds" rule
([planner.md](../architecture/planner.md)). `D` is the topology's level count;
`DECAY` changes which terms exist.

**There is deliberately no runtime fallback on any of them.** An out-of-range
value is a hard `TORCH_CHECK`, never a slower path: one comprehensive path ships.

### The count is measured, not asserted

The number of instantiations is whatever the matrix product evaluates to, and
the authority is the ratification manifest, which carries one entry per
instantiation keyed by mangled name. `python tools/ratify.py` recompiles the
closure and compares; see [`docs/ratification.md`](../ratification.md). A prose
count in a source comment is a number that rots the first time an axis moves —
`gen_shards.py`'s `AXES` tuple is the one place the count is defined.

### The 64 arms are compiled in eight shards, not one translation unit

The instantiations do not live in this file. They live in eight generated
translation units under `csrc/rola/src/instantiations/` —
`consumer_gt16_decay_bc16`, `consumer_gt16_decay_bc64`,
`consumer_gt16_nodecay_bc16`, `consumer_gt16_nodecay_bc64`,
`consumer_gt64_decay_bc16`, `consumer_gt64_decay_bc64`,
`consumer_gt64_nodecay_bc16`, `consumer_gt64_nodecay_bc64`, eight members each
— partitioned on `GT` then `DECAY` then `BC` (`tools/gen_shards.py`'s
`SHARD_AXIS_PRIORITY`, the shortest prefix of which reaches
`TARGET_MEMBERS_PER_SHARD = 8`), and `consumer.cu` reaches them through
host wrappers it can DECLARE but not define
([`consumer_launch.md`](consumer_launch.md#shard-only)).

`consumer.cu` emits exactly two device entry points of its own — `scan_kernel` and
`divide_kernel`, neither a template. If it ever emitted a `consumer_forward_kernel`,
nothing would be sharded; that is what `tools/fatbin_entries.py` refuses.

## <a id="shard-rows"></a>B9. One table, and why the matrix is enumerated once

Four hand-maintained spellings of one 64-element set — a dispatch ladder, a
residency ladder, a footprint ladder and a probe translation unit — is a shape that
drifts without anything failing. It is REFUSED here.

There is ONE definition, `tools/gen_shards.py`'s matrix, and everything else is
generated from it:

| consumer | what it gets |
|---|---|
| the eight shard translation units | 8 explicit instantiations each |
| `instantiations/dispatch_table.inc` | the row blocks, the axis key and the row index |
| `tools/ratify.py` | the translation units to compile, and the manifest's `shards` block |

A `ShardRow` carries an arm's structural coordinates, the three things only its own
shard can answer — `launch`, `occupancy`, and the `constexpr` `smem_bytes` — and the
pair that identifies where it came from, `shard` and `src_key`.

**EVERY RESOLUTION IN THE DISPATCH TRANSLATION UNIT IS ONE LOOKUP.** `dispatch_key`
turns a [choice](#dispatch-choice) into a mixed-radix index over the axes' VALUE
POSITIONS, in declaration order (`BC`, `DV`, `D`, `DECAY`, `GT`), and `kRowByKey` turns
that index into a row:

```
key = (((iBC * 2 + iDV) * 4 + iD) * 2 + iDECAY) * 2 + iGT
row = kRowByKey[key]            // nullptr if the binary has no such arm
```

Both the per-axis index functions and `kRowByKey` are emitted from the matrix, so the
axis ORDER exists once. An index formula written by hand beside a generated table is
the drift this file exists to remove — and a search over the rows, which is what the
queries did before, restates nothing but pays for the restatement it avoided.

A value no arm carries makes `dispatch_key` return -1, and an absent arm makes
`kRowByKey` yield `nullptr`; both are REFUSALS naming the choice. Neither resolves to
a neighbouring row, which is the difference between a table and a threshold.

The launch resolves a row and calls `row->launch`; the CUDA error state is checked
once, after it. Taking the address of a specialization this translation unit cannot
define is also what makes the split real: a dispatch arm with no shard member CANNOT
be satisfied by a silent re-instantiation here.

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

## <a id="dispatch-choice"></a>B9.1. What the table consumes: a CHOICE

`DispatchChoice{BC, DV, D, decay, GT}` is one value per matrix axis and nothing else —
no statistics, no thresholds, no device facts. Something upstream DECIDES (`derive_bc`'s
closed form for `BC`/`DV`/`D`/`decay`, `select_window` for `GT`, a test, a debug pin)
and the table RESOLVES.

The separation is the point rather than a tidiness: `derive_bc`
(`rola/planner/derive.py`) reads the table's own arm carve as one of its three
ceilings, but the table itself scores nothing and holds no opinion about which row is
preferred — it is a pure `{axis tuple: row}` map. So there is no scoring here, no
default arm, and no "nearest" anything.

## <a id="row-admission"></a>B9.2. The per-row enable predicate, and why it keys on an INPUT

A row's footprint is `constexpr`; what a device grants a block is not. `row_admitted`
compares the two:

```
row.smem_bytes <= DeviceFacts::max_dynamic_smem_bytes    // cudaDevAttrMaxSharedMemoryPerBlockOptin
```

The device fact is read ONCE per process and the launch refuses a row it excludes,
naming both numbers — where the opt-in would otherwise fail inside the launcher and
surface as a bare CUDA error. `rola_max_dynamic_smem_bytes` exposes the same fact so a
planner can apply the same predicate BEFORE it selects, and
`rola.planner.dispatch_table` mirrors the inequality exactly: a planner that selected
an arm the launch then refused would have priced a plan that cannot run.

**IT IS A PREDICATE ON A RUNTIME INPUT, NEVER ON A BUILD SYMBOL.** flash-attention
gates its arms with `#ifndef FLASHATTENTION_DISABLE_HDIM{N}` around each generated
instantiation (`hopper/generate_kernels.py`, verified in the local clone); that makes
the arm set a property of the build, so the binary cannot say what it can run. Here
the binary carries every arm it was built with and answers the question about THIS
machine.

## <a id="row-staleness"></a>B9.3. The staleness key on every row

Each row carries the name of the generated translation unit that defines it and that
unit's `members-sha256` — the hash of its sorted tuple list, which is exactly what
`tools/manifests/sm_*.json` records under `shards` and what enters the manifest
digest.

Inside one binary a row and its kernel are compiled together and cannot disagree, and
nothing here pretends otherwise. The key exists for the joins ACROSS the compile
boundary: a recorded table, a cached plan, a manifest carried beside a rebuilt `.so`.
`rola.planner.dispatch_table` performs that join against the ratified key set
(`_build_config.py`'s `shards` block, so an installed wheel can make it with no
repository) and FILTERS every row that does not match, recording what it dropped. A
stale row is never returned and never launched — the Inductor `src_hash` discipline:
a cache entry whose source moved is not repaired, it is not used. An empty surviving
set is an error, because it means the binary and the record disagree about
everything.

## <a id="table-census"></a>B9.4. The census: the table and the binary carry one set

`rola_dispatch_table()` returns every row as data and touches no device (the occupancy
function pointer is deliberately not called), so the set the dispatch can reach is
READABLE rather than inferred. `tests/unit/test_dispatch_table.py` joins it against
`tools/fatbin_entries.py`'s device entry points, decoded from their MANGLED NAMES, and
asserts set equality in BOTH directions per architecture: an entry the table cannot
reach is dead payload, and a row with no entry is a dispatch into a launcher the
binary does not define.

## <a id="occupancy-axes"></a>B10. Residency is per-instantiation, so the query has axes

`rola_consumer_occupancy` asks the DRIVER, and it takes `decay`, `BC`, `DV`, `D` and
`GT` -- every template axis -- because a single reported number would over-promise on
one arm and under-report another. The whole point of deriving `__launch_bounds__` from
`consumer_smem_bytes` is that a decay instantiation which regressed the budget
would drop its OWN bound rather than silently over-promise; this is how that is
checked from Python rather than asserted.

Why each axis is on the query:

| axis | what it moves |
|---|---|
| `DECAY` | three extra `[BC]` vectors |
| `BC` | which term BINDS: at `BC = 64` shared memory binds, at `BC = 16` the REGISTER FILE does |
| `DV` | `DV = 32` halves the state block |
| `GT` | the window Gram is quadratic in `GT`: `GT = 64` costs residency directly — `BC = 64, GT = 64` runs at ONE CTA/SM (SMEM-bound), `BC = 16, GT = 64` keeps two |
| `D` | the `[D][BC]` digit table, AND one 16-byte prefetch chunk per token per side for every level beyond the first ([`smem_layout.md#prefetch-slab`](smem_layout.md#prefetch-slab)) |

A residency claim about "the kernel" that quoted only one `(BC, DV, D, GT)` tuple
would be a claim about a fraction of the matrix.

Both queries resolve a `ShardRow` (B9) and read one field of it —
`occupancy()` calls the driver through the shard's wrapper, because post-shard the
kernel ADDRESS lives in the shard; `smem_bytes` is a `constexpr` the shard computed
at compile time. The wrapper is instantiated for all 64 arms rather than the 8
queried, which is why adding a query axis needs no matching change in the shards
([`consumer_launch.md`](consumer_launch.md#residency)).

## <a id="plan-surface"></a>B11. The supported shape surface

`BC in {16, 64}`, `DV in {32, 64}`, `GT in {16, 64}` — and these are the
planner's own supported sets (`GT`'s via `CONSUMER_WINDOWS` and `select_window`,
[`smem_layout.md#window`](smem_layout.md#window)), so the kernel instantiates
exactly the configurations a plan can name: **no shape a plan can select is
refused, and no shape it cannot select is compiled.** `launch_consumer`'s
`TORCH_CHECK` is the single authority for that surface; `p.D` is checked
separately against `[1, 4]`.

## <a id="state-derivation"></a>B12. Under paging, `N` is derived rather than read

§4.5: paging changes the state's SHAPE, not its addressing rule, and the page
granule is the MMA ATOM rather than the owner. Unpaged the state is
`[BH, N, cols]`; paged it is `[physical_slots, MMA_K_QUANTUM, cols]` and `N` is
then not readable off the state — so it is derived from the owner partition,
which is the definition: `N = owners_total * BC`.

## <a id="support-shape-check"></a>B13. The support word is shape-checked into place

§4's support word is WORD-MAJOR, and it is checked against the schedule's row
count (same `total_rows` level-digit rows) and against the amplitude stream's
word axis. A stale or mis-shaped schedule cannot reach the membership scan.

## <a id="split-k-emission"></a>B14. The split-K emission sites

The kernel emits its readout and its intra term as separate split-K atomics
guarded by `if (out != 0.0f)`. Two facts about that guard, both load-bearing:

1. It is a PERFORMANCE SKIP, not a membership decision (A4) — licensed by the
   [self-masking theorem](../architecture/self-masking-theorem.md).
2. It is what makes the intra term a second atomic per (token, channel) on a
   mixed group, which is cause #1 in
   [`intra_panel.md`](intra_panel.md#measured-status) for `panel` measuring
   slower than the walk. Fusing the intra term into the readout epilogue is the
   standing follow-up for BOTH lowerings.
