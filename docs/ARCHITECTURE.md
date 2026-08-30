# The architecture

This is the foundation document: what the kernel family computes, the address space
it computes it over, the format of the state it carries, the grains its work is
partitioned into, which quantities are compiled and which are addressed at runtime,
what each kernel does, and how every structural mechanism maps onto the
architectures the repository ships for.

**Authority.** `docs/KERNEL_STANDARDS.md` is the constitution and the laws; this
document is the architecture those laws describe; `docs/internals/` is the
per-source-file mirror beneath it, and `docs/architecture/` holds the individual
decisions (the self-masking theorem, the affine recurrence, the factored
representation, fp32 accumulation, closed-world codegen) whose arguments this
document assumes rather than repeats. Where the three disagree, the constitution
wins, then this document, then the mirror. Nothing outside the repository is an
authority for anything stated here.

**Status marks.** A mechanism is either **in the tree** (the mirror page describes
running code), **ruled, not yet built** (the law is settled and this document states
it; the mirror page for the mechanism says what is in the tree today), or
**UNSPECIFIED (ruling pending)** — a place where a reader would reasonably expect a
law and there is not one yet. The third mark is deliberate: an invented rule in a
foundation document is worse than an acknowledged hole.

---

## 1. What the kernel computes

The consumer is a **gated linear-attention recurrence over a routed leaf address
space**. Once the routing allocations are materialized it is textbook gated linear
attention — one state row per leaf, a diagonal gate, a rank-one deposit, a linear
readout — and the fp64 oracle (`rola/ops/naive.py`) is written as exactly that. The
oracle is the executable specification; the kernels are optimizations of it, never a
second definition.

**Symbols, defined here and used unqualified below.** `B` batch, `H` heads,
`T` tokens, `D` routing levels, `B_l` the digit width of level `l`,
`N = Π_l B_l` the leaf count, `DV` (`d_v`) the value width,
`cols = DV + 1`, `BH = B·H`, `L` the sequence length of a call, `n` a leaf index,
`W` the window length in tokens, `C` the BATCH TILE — the tokens one fold or readout
batch carries, which is the token dimension of its matrix multiply and one slot of the
value ring, so a window's write list of `nw` entries is `ceil(nw / C)` batches —
`BC` the leaves one owner holds, `s(l)` an owner's span at level `l`, `g(l) = B_l / s(l)` the owner grid at
that level.

Per token `t` and leaf `n`:

    keep[t,n] = (1 - rate[n]) ^ c[t,n]                       the diagonal gate
    S[t,n,:]  = keep[t,n] · S[t-1,n,:] + Wt[t,n] · v~[t,:]   the rank-one deposit
    r[t,:]    = Σ_n R[t,n] · S[t,n,:]                        the readout, read-after-write
    y[t,:]    = r[t, 0:DV] / (r[t, DV] + eps)                the ratio

`R[t,n] = Π_l p_read[t,l,d_l(n)]` and `Wt[t,n] = g_write[t] · Π_l p_write[t,l,d_l(n)]`
are the per-leaf products of the per-level routing allocations — each level's
allocation is a normalized simplex over that level's `B_l` digits, and `d_l(n)` is
leaf `n`'s digit at level `l`. `rate` is the per-leaf decay built the same way out of
per-level dials; the clock `c` is the write product, detached. There is no query, no
key and no feature map anywhere in the family: the routing IS the addressing.

**The mass rides the state.** `v~` is the value row with a column of ones appended,
so the denominator of the ratio is the same recurrence run on ones — one state, one
recurrence, one readout, `cols = DV + 1` logical columns. The ratio readout is the
only normalization the family has; there is no second recurrence and no raw mode.

**Declared arithmetic** (KERNEL_STANDARDS §R17). Operands are bf16 everywhere —
there is no float operand path in any kernel. Accumulation is fp32, and that is
closed (`docs/architecture/fp32-accumulators.md`). The state's logical element is
fp32. **The readout operand is the top half of the fp32 accumulator, taken by
truncation, in every kernel** — the truncation is declared, not incidental: it makes
the hybrid split of §3 exact on rejoin, and the ratio cancels its uniform bias. The
oracle reports the difference in fp64 and the gate asserts against the bf16-mirrored
band (`docs/testing.md`). A round-half-up variant with an invertible rejoin is
designed and not built; it is reached only if a training result ever shows a
state-scale bias.

**The window decomposition.** A call's token axis is cut into fixed windows of `W`
tokens, global to every owner. The carry kernel computes each window's readout
against the state as of that window's start and then folds that window's writes; the
pairs of tokens INSIDE one window are the intra kernel's term. `inter(W) + intra(W)`
is `W`-invariant and equals the oracle — the identity that licenses two kernels
(`docs/internals/carry/carry_kernel.md`, `docs/internals/intra/intra_kernel.md`).

---

## 2. The leaf lattice

**Levels and widths.** The address space is a `D`-level mixed radix. Level `l` has
width `B_l`, a power of two; `N = Π_l B_l`. The widths are per level and are a
property of the state, not of the binary (§5).

**The canonical order is MSB-first mixed radix**: leaf `n` decomposes as
`d_l(n) = (n / stride_l) mod B_l` with `stride_l = Π_{l' > l} B_{l'}`, so level `0`
is the most significant digit and level `D-1` varies fastest. Every derivation in
the repository — the oracle's own strides, the page table, the atom keying, the
kernels' addressing — is this one order, and it is carried as a FIELD of the state's
descriptor rather than assumed, so that a second order would be a refusal and never a
silent reinterpretation of somebody's bytes (`docs/internals/state.md`).

**The page rectangle.** The storage granule is the **atom**: 16 consecutive leaves in
canonical order, aligned to 16. In digit terms it is the rectangle formed by the
trailing four canonical bits — the innermost levels taken whole plus, where the four
bits fall inside a level, an aligned contiguous sub-run of that level's digits, with
every level above fixed. That the atom is a digit rectangle at every admissible span
vector is a theorem of the order, not a constraint on the carve
(`docs/internals/carry/box.md`).

**The owner lattice.** An owner (§4) holds a rectangle of digits per level: span
`s(l)` at level `l`, grid `g(l) = B_l / s(l)`, `BC = Π_l s(l)` leaves. The map from
the canonical index to the lattice index is a fixed BIT PERMUTATION — in the
canonical index level `l`'s field splits as `(o_l | r_l)`, the owner coordinate and
the offset inside its run; in the lattice index the same fields are regrouped as
`[o_0 … o_{D-1}][r_0 … r_{D-1}]`, every field keeping its width. Two consequences the
kernels live on: the owner index is a bit-field extract and never a division chain,
and an owner's leaves are one contiguous `BC`-aligned run by construction. Inside an
owner the order is mixed radix over the runs (`local = Σ_l r_l · Π_{l'>l} s(l')`), so
a bucket keyed on one level's digit owns a contiguous run of leaves.

**Spans are balanced, and the small level takes its width.** `BC` is balanced over
the levels; a level narrower than its balanced share takes its full width
(`s(l) = B_l`, `g(l) = 1`) and the remainder balances over the others; the carve
extends one level deeper when the outer spans' product is smaller than the stream or
warp count it must feed. The only refusal is `Π_l B_l < BC`. Every span satisfies
`s(l) ≤ B_l`, and the atom stays inside one level's run wherever the widths admit it.
There is no uniform per-level span and no capacity multiplier pushing width into the
innermost level: the earlier atom-floored convention is superseded, and what it
protected — the innermost run length that amortizes a resolve and lengthens a vector
load — is a measured efficiency term, never a law. *(Ruled; the derivation moves out
of the templates with the runtime-addressing work, §5.)*

---

## 3. The state

The carried state is a paged tensor of `[B, H, N, cols]` logical shape and nothing
more. Its full contract — the membership test that keeps launch machinery off it,
binding, the backings, clone, the layer's continuation container — is
`docs/internals/state.md`; this section states the architecture that contract
encodes.

### 3.1 The format descriptor, and one admission law

**The state owns its format** (KERNEL_STANDARDS §R13). A `StateFormat` binds at the
state's first call, from the call itself, and carries:

| field | what it is |
|---|---|
| `D`, `B_l` | the depth and the per-level widths, each a power of two |
| `order` | the canonical order of §2, as a field rather than an assumption |
| `page_bits` | the page rectangle: the trailing four canonical bits, which may span levels |
| `DV` | the value width; `cols = DV + 1` is the page's LOGICAL width, not its layout |
| `dtype` | fp32 as split bf16 planes (§3.2) |
| `ids` | the page-table id space, `BH · N / 16` |

**Later calls are CHECKED against it, and a mismatch is a refusal at the seam naming
the field that differs.** The descriptor is fixed for the state's life — a re-layout
is an explicit state operation, never something a call does — and it travels with the
layer's continuation. Each kernel derives its addressing from (descriptor, its own
shape) in its prologue.

**Prefill and decode share ONE admission law: the descriptor.** Decode accepts
exactly what prefill accepts — the same depth, the same widths, the same `N`, the
same value width, bf16 operands, therefore the same topologies. There is no
decode-only shape, no decode-only tolerance and no decode-only admission branch
anywhere, tests included: a second admission law is a second contract to keep in step
with the first, and the two kernels partition ONE paged state.

**There is no ragged state.** `N = Π_l B_l` over powers of two, so a lawful state is
page-aligned by construction; a leaf axis that is not a whole number of pages is
REFUSED — not padded, not carried in a tail page, not served by a second layout.

### 3.2 Split planes, and the page

A page is one atom — 16 leaves — stored as **planes, not rows**:

    [16 × DV hi][16 × DV lo][16 mass hi][16 mass lo]

where `hi || lo` is the fp32 word exactly. The hi plane of a row is `2·DV` bytes,
which at `DV = 64` is one 128 B line; the interleaved fp32 row it replaces was
`(DV+1)·4` bytes and line-aligned only by accident. The mass rides its own blocks
rather than a column of every row. The conversion each way is one integer byte-permute
per pair of elements — no shared-memory traffic and no arithmetic
(`docs/internals/common/state_page.md`).

**The page stride is padded**: `page_stride = round_up(hi + lo + mass, 128 B)`, a pure
function of `DV` (4,224 B at `DV = 64`; 8,320 at 128; 2,176 at 32), so that a hi row
is line-aligned in every slot rather than in even slots only. The mass stays inside
the page; the descriptor's arena sizing and the dense backing use the same stride.
The cost is 1.5% of capacity at `DV = 64`. *(Ruled; the unpadded 4,160 B stride is
what is in the tree today, and the measured price of it is +9.6% of the DRAM traffic
on written atoms.)*

**Hi-only reads.** The readout operand is the hi plane (§1), so an atom a call only
READS has a lo plane no arithmetic can observe, and not moving it is exact rather than
approximate. A read-only atom therefore moves half the bytes; a written atom moves and
stores both planes and rejoins exactly. This is one mechanism in both kernels: prefill
gates its state slice, decode gates its unit.

### 3.3 Two backings, one set of bytes

**Paged** is the default: a page arena, a `[BH, N/16]` page table of slot ids, and
residency planned exactly from the call's own write set. Resident state scales with
the atoms the routing REALIZES, not with the `N` the topology provisions. **Dense** is
the same page sequence with every page resident, in the same split-plane atom-major
layout. Dense and paged are ONE kernel, one ABI, one accumulation order and one base
translation, and the gate between them is BYTE identity — asserted on an integer view
of the storage, never with a float comparison, because a stored word is two
neighbouring elements' halves and a byte-identical pair can read as NaN
(`docs/internals/paging/paging.md`). A caller cannot observe which backing a state
has except by asking.

`materialize()` is the `hi || lo` gather back to the logical `[B, H, N, cols]`
tensor. It is for the oracle, a cross-arm hand-off and a test — never on a measured
path, since allocating that tensor is exactly what paging exists to avoid.

### 3.4 Activity bits: two sets, one grain each

Residency is the arena's fact; **activity is the facts pass's fact**, and only the
second decides what a call TOUCHES (KERNEL_STANDARDS §R14).

* **PRIMARY bits — the kernel's grain.** Per (warp box, side), over the WHOLE CALL:
  one READ bit and one WRITTEN bit, never pre-ORed, emitted at the carve's own grain.
  The state slice is loaded once at the call's entry and stored once at its exit — the
  box lives in registers across every window — so the gate is per call: LOAD on
  `resident ∧ (read ∨ written)`, STORE on `written`. Nothing is kept per window: a
  window with nothing live for a box is a schedule fact, read off the ballots while
  walking, never stored. Because the gate never references the
  atom, prefill needs no page alignment and no row masks: owners splitting an atom
  write disjoint rows. *(Ruled, not yet built: the emission consumes the carve as a
  kernel input, which arrives with the runtime-addressing work of §5.)*
* **DERIVED bits — the page grain.** Per page, host-side: the OR of the primary bits
  over the warp boxes the page intersects, through the box→page map the descriptor
  gives. Exact allocation admits a page iff an intersecting box is WRITTEN this call;
  residency checks read the derived READ set; a decode step consumes this set, its
  unit being the page. *(In the tree.)*

Three properties follow. Dense and paged touch the IDENTICAL atom set, so the byte
gate certifies the skip logic and not merely the addressing. A resident-but-idle atom
costs no state traffic at all, so a call's state traffic is proportional to what IT
touches rather than to what the sequence has ever touched. And the price is the
in-place law: an atom the exit sweep skips must already hold what it carried in, so a
gated continuation's exit plane must BE its entry plane — the seam refuses two
distinct planes when an activity map is supplied.

---

## 4. The grains, and how they nest

Five grains, from the storage outward. **They nest on the page and nowhere else**
(KERNEL_STANDARDS §R14): every other pair of grains is free to cut across.

| grain | what it is | who owns it |
|---|---|---|
| **page** | 16 canonical leaves = one atom = one slot | the state's format |
| **decode unit** | one atom | the decode step |
| **prefill box** | the CTA's `BC` leaves: one contiguous lattice run | the carry launch |
| **stream box** | a stream's rectangle of digits inside the CTA's box | the carve |
| **warp box** | the leaf sub-box one warp holds in registers | the register shape |

**The decode unit is the atom** — not the innermost span. Decode has no boxes; it has
units, which are contiguous runs of consecutive canonical leaves with the outer digits
fixed (one memoized outer product; one absent digit kills the unit). Defining the unit
as the innermost span would shrink it whenever prefill balances its spans, so the unit
is fixed at the atom regardless of what prefill does; when the atom spans two levels
its 16-slot mask is the outer product of the two innermost digit masks, one extra
multiply per slot, once per unit. **Prefill and decode ownership are decoupled**: two
kernels partitioning the same paged state, whose entire shared contract is the atom,
the page table and the activity bits.

**Prefill owns whole ROWS.** A leaf row is a line; two owners splitting an atom write
disjoint rows, so nothing masks and nothing clobbers. Residency stays per page.

**Streams are an axis.** `S = kWarps / WPS`, where `S` is the stream count (the
state matrix of §1 never appears in the same expression), `kWarps` the CTA's warps
and `WPS` the warps consuming one stream. Each stream has a box — the tokens live in
it, from its own ballot — and a private segment; the `WPS` warps consuming a stream partition ITS box among
themselves, each warp taking the leaf sub-box it already has. **`S = 1` is the union
point**: one box (the CTA's), one union list, one shared segment, each warp its own
leaves — the base case, and the control row forever. `S` is a distribution knob, not a
fallback and not a threshold; the same code is correct and at floor at every value up
to one stream per warp. Streams are decoupled from the warp carve: raising `S`
re-partitions the token lists, not the leaf ownership. *(Ruled; `S = 1` is what is in
the tree. `S` is DERIVED: as large as shared memory admits at residency, from the
descriptor and the architecture's shared-memory size — the premise being that a stream
costs close to nothing, which the private-stream measurement is the standing test of;
never a density, never a runtime input.)*

**Per-side carves.** Read and write are carved independently: a stream owns an affine
rectangle of digits per level, per side, and the private slot order puts the carved
(sparse) levels outermost. Liveness is the exact AND over levels of that side's own
amplitude runs — a value test, so a dense level's run is nonzero for every token and
drops out of the conjunction on its own, with no structure fact carried into the
kernel and no branch on a computed statistic. Per level at most one side is sparse, so
there is one stored row set per level and a dense side is absent-means-full; the two
sides' clause sets are different level subsets, which is why the schedule is two-sided
with independent counts. A read carve and a write carve that coincide are a transit
whose row map is the identity — the same code, never a second body
(`docs/internals/carry/carry_kernel.md`).

**The read side has no stream axis: one stream per warp, always.** The write side has
a stream count only because staging `d_v`-wide value rows into private segments
duplicates bytes per stream; the read side duplicates nothing, so every warp reads its
own box's tokens.

---

## 5. The axis law

KERNEL_STANDARDS §R11 and §R12 in architectural form. **A compile-time axis is
legitimate only as a deterministic function of user input that the selector can pick;
everything else is derived, addressed at runtime, or constant.**

| class | quantities | why |
|---|---|---|
| **compiled** | `D`, `DV` | the register shape — the accumulator a warp holds is leaves-per-warp × `DV`, and the level nest is unrolled over `D`. Both are chosen by the user's model. |
| **derived** | `BC`, the spans, `kWarps`, `S` | from residency and the descriptor: leaves-per-warp = state registers / `DV`; `BC` = leaves-per-warp × `kWarps`; spans balance `BC` over the levels (§2); `S = kWarps / WPS`. |
| **runtime addressing** | the per-side modes, `B_l`, the orders, the row maps | CTA-uniform values read from the descriptor in the prologue: a few shifts, spans, per-warp offsets, row-map deltas and swizzle words per level. |
| **constant** | `W`, `C` | kernel internals; a short sequence runs partial windows. |

**Shape is the warp's slot map, and it is compiled.** The quantity that fixes which
register holds which factor in the unrolled expansion — the map from a worker's leaf
slots to per-level digits — is REGISTER SHAPE, not addressing: a runtime span would
index a register array dynamically (a spill) or a switch over every layout in the hot
loop (measured 2× instructions). The derivation is generic in the architecture's state
capacity: `leaves_per_sm` (the register file on sm_80/90, tensor memory on sm_100)
balanced over the D levels is the SM's box; `warps_per_sm` is the constant that carves
it — a warp owns a segment of the box on every architecture; each SIDE's level order
divides the box, sparse levels first, until the warp count is reached, and what remains
is that side's warp segment. The compiled shape is the slot map of the REGISTER TILE a
warp holds per step (today one tile is the whole segment; on sm_100 a warp walks its
tensor-memory segment tile by tile, a runtime trip count over identical bodies). The set
of shapes a side can select is generated from (D, log₂ warps_per_sm, the box's bit
vector) — its count is flat across architectures — and the selection per side is a
`uniform_switch` over that set outside the window loop. A CTA holds a contiguous subset
of the SM's warp segments; how many is a launch parameter and never enters the shape. *(Ruled; landing with the runtime-addressing work.)*

**One geometry block, one derivation.** The modes, the widths, the spans, the row maps
and the decode lattice are derived in exactly ONE place, from the descriptor, and
every consumer reads that block. A second derivation site is the defect the oracle's
own independence rule exists to catch elsewhere — inside the kernel it is simply a
way for two agreeing wrong conventions to pass every gate.

**What the move costs and what it does not.** The immediates become uniform-datapath
registers; the instruction is the same shift or logical operation with a register
operand, at the same throughput. What is lost is the constant folding of span-1 and
shift-0 levels — a few instructions per token per level, a ledger line, not a design
change. The static nest over levels stays static: only the immediates become
registers. The vector register count must not rise beyond noise, and the census is the
gate on that. *(Ruled, not yet built: today the modes and widths are template
arguments and the shipped arm set multiplies with them.)*

**SMEM is launch-sized.** Every shared tenant whose size depends on a derived
quantity — the transit region, the panels, the factor staging, the position bitmaps,
the per-stream value segments — is sized by the host from the descriptor and passed as
DYNAMIC shared memory, with offsets as uniform values. Never max-sized. Above the
48 KB static limit the one-time function attribute is set, on every architecture.
**Residency is ASSERTED at the seam from the launch size, as a refusal** — the launch
does not quietly get fewer blocks than the shape was derived for.

**No runtime trip count in a register-shaped loop.** The spans enter the hot loops only
as shifts and masks; the fold and readout loops iterate leaves-per-warp × `DV`, which
is compile-time. The census — vector registers flat, zero spill — is the proof.

**Structure branches, and the four layers that enforce them** (KERNEL_STANDARDS §R9
and its enforcement clause). A CTA-uniform branch on STRUCTURE is legal — a switch
over the admissible (span, shift) pairs is structure the declaration fixes; a branch
on a measured statistic (a density, a live count, "small L") never is. A legal
structure branch carries all four of: cases GENERATED from the arm's shape with a
static assertion on the count, never hand-enumerated; a compile-time exhaustiveness
assertion, so an arm that could receive an uncovered case does not build; a
launch-time refusal at the seam naming the uncovered case; and a trapping default,
with a post-build sweep of every arm at every admissible case on a tiny cell.

---

## 6. The kernels

Four kernels and one host-side derivation. They share the window grid, the descriptor
and the atom; they share no code path that could make a wrong convention agree with
itself.

### 6.1 Facts — the bits

One pass over the routing amplitude planes emits everything downstream launches need:
the per-side union table of packed liveness bits (`[BH, entries_read + entries_write,
L/32]`, the two blocks consecutive, shape `Σ_l B_l / s(l)` per side and NOT growing
with `N`), the per-level width plan, and the activity bits of §3.4. Rows are voted at
the CARVE grain, per side, per level, per run of THAT side's partition, from that
side's own factors — so a stream's test reads exactly one row per tested level, and a
level a side spans whole emits no rows at all (`docs/internals/facts/facts_kernel.md`).

### 6.2 Carry — the window body

Per window, per owner, against the state as of the window's start:

    Oᵀ[:, t]   += Sᵀ[:, box] · R~ᵀ[box, t]      PHASE 1, the readout
    Sᵀ[:, box] += V~ᵀ[:, t] · W~[t, box]        PHASE 2, the fold

**The ordering law is window-grain read-before-write on the owner's own slots** — a
statement about the window, not about a batch. A structure that read and folded per
batch would let a later batch's readout see an earlier batch's folded writes, and both
are inside one window, so those pairs belong to the intra term and would be counted
twice. The batch line therefore IS the window line, except across windows where the
owner is side-dead, which is the whole-window-skip harvest.

**The schedule.** A window is `W/32` ballot words and a lockstep unit has 32 lanes, so
a lane holds one word and a row costs one 32-bit load per warp — cheap enough that every
warp can hold the owner-grain ballots of both sides. *In the tree:* each warp walks its
own bits over those CTA-identical ballot words and consumes the union list, which is the
`S = 1` point of §4; the whole-window skip is a ballot test needing no shared word, and
the write list is balanced at any distribution because each warp takes a fixed slice of
every word's bits. *Ruled, landing with the pipelined body:* the three-part schedule with
COOPERATIVE staging — the union of live tokens derived ONCE per side per window; a
cooperative fetch of un-expanded factor rows and value rows, by executor share, into ONE
shared staging; the expansion memoized in registers; sends from each stream's own ballot
into private segments; then seal-and-consume. The edge that staging introduces is
absorbed by the covers and the split-phase edges below, which is why the earlier
"cooperative staging forces a barrier the warps wait at" is superseded rather than
weighed again.

**Why the schedule has three parts and not two.** A union over (token, stream) pairs
collapses the fetch and the sends into one walk and re-fetches every token once per
stream; separating them fetches each token once per window and pays the per-stream cost
only on the sends. Consumers are then pure matrix multiply. Ownership of a derivation is
one warp's or one CTA's, never replicated per warp to dodge a barrier the kernel already
pays.

**The hybrid transit at the phase change.** The two sides are carved independently, so
the phase change re-lays the state into the other side's order. The hi plane MOVES: the
writers publish it and DROP it, holding only the lo plane through PHASE 1; the readers
pull it in read layout and it IS their A operand — no second bf16 stack exists; after
PHASE 1 the readers store those same bits back and the writers pull them in write
layout and rejoin, `(hi << 16) | lo`, exactly, with no arithmetic. The register
footprint through PHASE 1 is 1.0× the state, not 1.5×. The publisher places each
leaf's row at the READER's slot, so the reader pulls a contiguous segment with one
transposing matrix load; the region is XOR-swizzled and conflict-free. A copy form —
keeping both planes live — is rejected: the register wall is the binding one.

**Each transit leg covers itself.** *In the tree:* stores, then work that does not
depend on the pulled data, then the CTA edge, then the pull. The order is the whole
mechanism — a cover issued BEFORE the store chain overlaps nothing, which is a defect the
sequence shows and the stall counters do not. A named barrier does not help here: an
arrival and a wait executed by the SAME threads are two phases of one barrier rather than
a split edge, and every warp on this leg is both publisher and puller. *Ruled:* the
split-phase primitive is the asynchronous barrier — arrive, keep issuing, wait at the use
— which lets a reader pull a writer's rows as soon as that writer has arrived, at a finer
grain than the leg.

**The fan-in is a native global reduction out of a staged panel.** The readout
accumulator's fragment is written to a shared panel in TOKEN-ROW layout, and the
reduction is then issued from the panel, so one warp instruction covers one token's 32
contiguous channels — one full 128 B segment. Issuing straight from the accumulator
fragment is fewer instructions and was measured SLOWER: the fragment's layout half-fills
its sectors. The figure of merit is therefore 128 B segments per reduction instruction,
not sector fullness and not instruction count. *Ruled target:* produce the same
token-row layout in REGISTERS with a shuffle transpose and reduce without the shared
round trip, measured against the panel.

**Registers, and what the body is.** The target is a body that hides its own latency by
construction: a software pipeline where the next batch's produce is issued while the
tensor pipe works on the current one, double-buffered panels filled asynchronously,
expansion arithmetic and shared stores interleaved between matrix-multiply issues,
split-phase edges rather than full barriers, and the fan-in interleaved into the fold
rather than phased after it. Exposed latency is a COVERAGE finding, never grounds to
split the CTA (§7). *(Ruled, not yet built: the body in the tree is serial per batch —
wait, rendezvous, produce, load, multiply — and the measured issue rate is what that
structure yields.)*

### 6.3 Decode — one wave of fat CTAs

At `T = 1` every tiling decision is degenerate, so decode is its own kernel: a routed
GEMV per step. Units are spread across CTAs first, then warps, so a small live set
spreads over the machine instead of concentrating on a few multiprocessors. The
admission — candidate atom enumeration, page-table probes, the block scan of demand,
the pool claim by canonical rank — is computed once per CTA, cooperatively, with
barriers CONFINED TO THE PROLOGUE; after the prologue there are none. Each warp then
derives only what its own units need, walks them (resolve the atom, cache the address,
read the row, update-then-read, accumulate in registers), and the combine is an
arrival count: each warp deposits its partial into its own shared row and bumps a
counter, and the last arriver sums the rows in fixed warp order, which is what keeps
the step byte-reproducible. The launch is ONE WAVE, aligned to the multiprocessor
count. **Read-only units pull the hi plane only** (§3.2), which is close to halving
decode's state traffic as `N` grows. `update-then-read` — the deposit lands before the
readout reads the row — IS the semantics, not an optimization
(`docs/internals/decode/decode.md`).

### 6.4 Intra — the within-window term

The causal term inside a window, as its own deterministic side kernel, reducing into
the same output and mass buffers the carry kernel allocated. Per window it walks the
lower-triangular tile grid of `[64 × 64]` output blocks; per tile pair the per-level
Gram matrices accumulate in registers over the level's digit axis, the levels are
multiplied together there, and the result becomes the A operand of the value multiply
IN PLACE — the accumulator-to-A fragment identity, with no shared-memory round trip.
The `[W × W]` matrix is never materialized. Where it touches the paged state it does
so through the same descriptor as every other kernel, and its ownership grain is its
own — the kernel in the tree reads no state at all, since every leaf it touches is
written and read inside the one window. *(**UNSPECIFIED (ruling pending):** what that
grain is once the intra term owns state, and whether the intra family's instantiation
axes fold into the `D × DV` shipped set of §8 — its window and level width are
template axes today.)*

---

## 7. Launch shape and the CTA policy

**The register law.** Eight warps per multiprocessor at 256 registers per lane,
`__launch_bounds__(256, 1)`, the register file filled exactly (8 × 32 × 256 = 65,536).
The census reports the STATE FRACTION of the register file beside the register count on
every stage, because the lever that raises leaves per multiprocessor is registers per
leaf — the working set — and not the block count.

**Registers per leaf is the lever, and the lo plane is its largest term.** The register
allocation is the maximum over the body's moments, and during the readout the lo plane
is idle state: parking it in shared memory for that moment (aliasing the transit region,
which is dead then) lets its registers house the readout's output fragments instead, so
the allocation falls and the freed registers become LEAVES — a bigger box per
multiprocessor, which is the fan-in and DRAM lever. It composes with the hybrid transit
(the writers drop the hi plane at the phase change; parking drops the rest). The
companion rule is that every moment FILLS shared memory to the residency target, spending
it by value per kilobyte: streams first in the fold, the parked lo plane first in the
readout. *(Ruled, not yet built: lands with the pipelined body; per-architecture
declaration where it does not cost residency.)*

**The CTA is the multiprocessor** (KERNEL_STANDARDS §R15). One CTA per multiprocessor
with all the warps the register file admits; streams partition inside it. More CTAs per
multiprocessor duplicate the cooperative fetch and expansion, run the transit once
each, and forbid cross-warp sharing of the readout; streams share all of it. **Latency
hiding is the kernel's job** — covers and split-phase edges — so exposed latency is a
coverage finding and never grounds to split the CTA.

**One CTA per multiprocessor is the GOAL, not a measured choice.** The block count
does not change the leaves resident per multiprocessor (one block of eight warps and two
of four hold the same registers); a two-block launch is only a DIAGNOSTIC — it says how
far the body is from hiding its own latency — and never becomes the law. Every
measurement the repository has was taken at two blocks of four warps, where the second
block was covering every edge; the one-block figure is the number the pipelined body is
graded against.

---

## 8. Shipping

**A user of the Python package never registers an arm** (KERNEL_STANDARDS §R16). The
shipped set is `D × DV` per architecture — a dozen or so binaries — and every other
shape the family serves is served by runtime addressing over the descriptor, checked at
the seam. An unknown descriptor is admitted by refusal-checked derivation, not by
enumeration: it is either lawful (powers of two, `Π_l B_l ≥ BC`, `D` and `DV` matching
a shipped arm, the residency assertion of §5 passing) and it runs, or the refusal names
the field.

**The derivation is ratified as code plus golden outputs.** Closed-world codegen is the
rule for the binaries — pinned assembler, ahead-of-time only, a per-architecture
ratified manifest recording registers, spills, residency, compile time and the
per-arm launch table, and a device-side stamp asserted before any gate
(`docs/ratification.md`, `docs/architecture/closed-world-codegen.md`). A derivation that
now runs at launch instead of at compile time is inside that closed world only if its
OUTPUTS are certified too, which is what the golden outputs are for.

**One body per operation family** (KERNEL_STANDARDS §R18). The channel-split body is
deleted; dense is the `S = 1` point of the one body. A tied carve is a transit with the
identity row map. A single-valued axis is not plumbed as a parameter — it is deleted,
and the ratio readout is simply what the readout is.

---

## 9. Portability

Every structural mechanism states its form on each architecture; a mechanism that
exists on only one is a workaround, not a design (KERNEL_STANDARDS §R8). A primitive
may select different NATIVE instructions per architecture; it may never select a
wider-type emulation.

| mechanism | sm_80 / sm_86 | sm_90 | sm_100 |
|---|---|---|---|
| routing arithmetic, expansion | bf16x2 native fused multiply-add | same | same |
| operand supply to the tensor pipe | register fragments, transposing shared loads | shared-memory-sourced wgmma operands | tensor-memory-sourced tcgen05 operands |
| state in the register file | symmetric warps, in-warp pipelining (a producer warp would waste an eighth of the multiprocessor's state) | producer/consumer specialization with per-warp register repartitioning | same |
| operand staging from DRAM | asynchronous copy into shared memory | tensor-memory-accelerator box loads, descriptors built at runtime | same, plus multicast to a cluster |
| runtime addressing | uniform-datapath registers from the prologue | the same, pairing naturally with runtime descriptors | the same |
| the transit | shared region, XOR-swizzled, transposing loads | the same region; the exchange can ride distributed shared memory across a cluster | the same |
| leg edges | a CTA edge with the cover before it; asynchronous barriers (arrive, issue, wait at use) as the split-phase primitive from sm_80 up | the same object, its completion transaction-counted by the copy engine | the same |
| the fan-in | native global reduction from an aligned layout | the same, vector reduction where the width admits it | the same |
| the box grain | one CTA per multiprocessor | a two-CTA cluster sharing staging through distributed shared memory | the same |
| decode's walk and combine | per-warp gathers, arrival-count combine | bulk copies per warp, an asynchronous-barrier arrival | the same |
| the closed world | ratified manifest per architecture, device stamp | the same | the same |

---

## 10. Index: mechanism → page

| mechanism | page |
|---|---|
| the recurrence and the ratio readout, as the executable spec | `rola/ops/naive.py` |
| the update is affine in the input and never in the state | `docs/architecture/no-delta-rule.md` |
| accumulation precision | `docs/architecture/fp32-accumulators.md` |
| storing factors, expanding transiently | `docs/architecture/factored-representation.md` |
| dead routes are exact zeros | `docs/architecture/self-masking-theorem.md` |
| the state contract, descriptor, backings, activity bits | `docs/internals/state.md` |
| the page's plane geometry and its byte permutes | `docs/internals/common/state_page.md` |
| the arena, the page table, the slack pool | `docs/internals/paging/paging.md` |
| the owner lattice, the bit permutation, the atom rectangle | `docs/internals/carry/box.md` |
| the window body: schedule, carve, transit, fetch, fan-in | `docs/internals/carry/carry_kernel.md` |
| the carry family's arms and their ABI | `docs/internals/carry/carry_arm.md`, `docs/internals/carry/carry_arm_abi.md` |
| the facts pass and its tables | `docs/internals/facts/facts.md`, `docs/internals/facts/facts_kernel.md` |
| the architecture capability table and the closed-world refusal | `docs/internals/facts/arch_caps.md` |
| decode: the step, the walk, growth, capture | `docs/internals/decode/decode.md` |
| decode's lattice and factor tables | `docs/internals/decode/decode_lattice.md` |
| the within-window term | `docs/internals/intra/intra.md`, `docs/internals/intra/intra_kernel.md` |
| the host order of a call | `docs/internals/engine/chunk_dag.md`, `docs/internals/engine/decode_dag.md` |
| the public surface | `docs/api.md`, `docs/internals/rola_api.md` |
| what a manifest is and what it refuses | `docs/ratification.md`, `docs/architecture/closed-world-codegen.md` |
| the tiers, the tolerances, the conformance regimes | `docs/testing.md` |
| the one stopwatch and the measurement ledger | `docs/measurement.md` |
| what is knowingly unfinished | `docs/open-work.md` |
| what was removed, and why it was safe | `docs/internals/DELETIONS.md` |
