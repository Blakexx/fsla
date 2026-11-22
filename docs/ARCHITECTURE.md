# The architecture

This is the foundation document: what the kernel family computes, the address space it computes it over, the format
of the state it carries, the carve that partitions a call's work, which quantities are compiled and which are
addressed at runtime, what each kernel does, and how each structural mechanism maps onto the architectures the
repository ships for. It states what is in the tree; what is knowingly unfinished is
[`open-work.md`](open-work.md).

**Authority.** `docs/CONSTITUTION.md` holds the standing rulings and `docs/KERNEL_STANDARDS.md` the laws; this
document is the architecture they describe; `docs/internals/` is the per-source-file mirror beneath it, and `docs/architecture/` holds the
individual decisions (the self-masking theorem, the affine recurrence, the factored representation, fp32
accumulation, closed-world codegen) whose arguments this document assumes rather than repeats. Where they disagree,
the constitution wins, then this document, then the mirror. Nothing outside the repository is an authority for
anything stated here.

---

## 1. What the kernel computes

The family computes a **gated linear-attention recurrence over a factored sparse leaf address space**. Once the routing
allocations are materialized it is textbook gated linear attention — one state row per leaf, a diagonal gate, a
rank-one deposit, a linear readout — and the fp64 oracle (`fsla/ops/naive.py`) is written as exactly that. The oracle
is the executable specification; the kernels are optimizations of it, never a second definition.

**Symbols, defined here and used unqualified below.** `B` batch, `H` heads, `T` tokens, `D` routing levels, `B_l` the
digit width of level `l`, `N = Π_l B_l` the leaf count, `DV` (`d_v`) the value width, `cols = DV + 1`, `BH = B·H`,
`L` the sequence length of a call, `n` a leaf index, `W` the window length in tokens (512).

Per token `t` and leaf `n`:

    keep[t,n] = (1 - rate[n]) ^ c[t,n]                       the diagonal gate
    S[t,n,:]  = keep[t,n] · S[t-1,n,:] + Wt[t,n] · v~[t,:]   the rank-one deposit
    r[t,:]    = Σ_n R[t,n] · S[t,n,:]                        the readout, read-after-write
    y[t,:]    = r[t, 0:DV] / (r[t, DV] + eps)                the ratio

`R[t,n] = Π_l q[t,l,d_l(n)]` and `Wt[t,n] = Π_l k[t,l,d_l(n)]` are the per-leaf products of the per-level routing
levels — each level a nonnegative row over its `B_l` digits, of any magnitude: no level need sum to 1, and a
producer's write gain is a factor of write level 0 ([`CONSTITUTION.md` R23](CONSTITUTION.md#r23)) — and `d_l(n)` is
leaf `n`'s digit at level `l`. `rate` is the per-leaf decay built the same way out of per-level dials (decode builds it; the prefill does not; outside the first paper's scope); the clock `c`
is the product of the write levels' SHARES (each level's row over its own sum), detached. There is no query, no key
and no feature map: the routing IS the addressing.

**The mass rides the state.** `v~` is the value row with a column of ones appended, so the denominator of the ratio is
the same recurrence run on ones — one state, one recurrence, one readout, `cols = DV + 1` logical columns. The ratio
readout is the only normalization the family has.

**The operands are the bundle's internal format** ([`api.md`
 §2.1](api.md#21-the-bundle-is-the-contract-and-it-is-the-internal-format)): every routing level's rows, dense or
sparse, are u8 digits unit-max against the row's fp32 maximum, a zero a zero, and each token's values are bf16 against
its largest channel with an fp32 scale (CONSTITUTION §R17, §R23). Nothing floating but those values is stored between
the producer and the kernels. THE ORDER OF A PLANE'S BYTES FOLLOWS ITS READERS' MEASURED ACCESS (Blake, 2026-11-13):
the producer writes each plane in the layout its readers' cheapest access wants. On hopper a level's rows are
TILE-MAJOR -- a tile of 64 tokens its 16-digit blocks one after another, each block a token's 16 bytes a line, the
layout an integer step reads -- since the intra stages a tile's level as one copy so and as 16 strided ones
otherwise ([`intra_window`](lab/intra_window.md)), and the carry reads a level in pieces of at most 16 bytes. The
VALUES STAY TOKEN-MAJOR, MEASURED on both readers: the carry's producers read a token's values as one run and lose
most of a window to them tile-major (12,201 cycles a window against 6,590, and 48,903 against 16,385 once-read:
[`carry_window`](lab/carry_window.md)), where the intra's window moves 5% with their order.

**Declared arithmetic, per kernel.** Every kernel's arithmetic is declared and mirrored by its gate's reference: every
accumulation fp32 or an exact integer -- but the output's sum across CTAs, fp32 or f16 as the plan states
(R17) -- the operands one family on every card
([`architecture/precision.md`](architecture/precision.md#components)).

**The window decomposition.** A call's token axis is cut into fixed windows of `W` tokens, global to every box. The
carry computes each window's readout against the state as of that window's start and then folds that window's
writes; the pairs of tokens INSIDE one window are the intra kernel's term. `inter(W) + intra(W)` is `W`-invariant and
equals the oracle, which is the identity that licenses two kernels.

---

## 2. The leaf lattice

**Levels and widths.** The address space is a `D`-level mixed radix. Level `l` has width `B_l`, a power of two;
`N = Π_l B_l`. The widths are per level and are a property of the state, not of the binary (§5).

**The canonical order is MSB-first mixed radix**: leaf `n` decomposes as `d_l(n) = (n / stride_l) mod B_l` with
`stride_l = Π_{l' > l} B_{l'}`, so level `0` is the most significant digit and level `D-1` varies fastest. Every
derivation in the repository — the oracle's own strides, the page table, the atom keying, the kernels' addressing —
is this one order, and it is carried as a FIELD of the state's descriptor rather than assumed, so a second order
would be a refusal and never a silent reinterpretation of somebody's bytes (`docs/internals/state.md`).

**The page rectangle.** The storage granule is the **atom**: 16 consecutive leaves in canonical order, aligned to 16.
In digit terms it is the rectangle formed by the trailing four canonical bits — the innermost levels taken whole plus,
where the four bits fall inside a level, an aligned contiguous sub-run of that level's digits, with every level above
fixed. That the atom is a digit rectangle at every admissible span vector is a theorem of the order, not a constraint
on the carve.

**Boxes.** A box (§4; a hopper CTA owns two, R20) holds a rectangle of digits a level — an extent at level `l` dividing `B_l` — and a
channel extent. Decode's own lattice is the `(k, m)` box: an owner's span `s(l)` at level `l`, grid `g(l) = B_l / s(l)`,
`BC = Π_l s(l)` leaves, and the map from the canonical index to the lattice index a fixed BIT PERMUTATION, so the
owner index is a bit-field extract and never a division chain
([`decode/decode_lattice.md`](internals/decode/decode_lattice.md)).

**Widths have a floor.** Every level is at least 16 wide (`B_l` a power of two ≥ 16), so no narrow-level case exists
in any kernel's addressing.

**Routing is per head, and its shapes are one family.** Every routing weight, kernel and reference carries the head
axis (the user, 2026-06-24: "there is no form of FSLA without per-head routing -- the whole point is it's a per-head
mechanism"); a variant checked against a reference that shares its assumption proves nothing. A routing is a tree of
`D` levels: flat (`D = 1`, width `N`), square (`D = 2`, width `sqrt(N)`) and deeper trees are one family. The capacity
`N` is invariant across the shapes; what the shape tunes is the routing projection's cost, the sum of the widths a side
-- factoring makes the routing cheap, not the state, which is still `N` leaves.

**The routing logits are the head's q/k, and a cost lever is depth or width.** FSLA has no q/k apart from them: each
column of a head's projection scores the hidden state against one digit's weight vector, so a head projects the sum of
both sides' level widths (a tied level counted once) plus `d_v` columns for the product of the widths in leaves. A
cheaper projection is a narrower level or more levels, priced in the model's own declared knobs (level count, widths,
tied levels, the grouped write side), never a special projection form: a codebook or a low-rank factor is today's matrix
under a rank cap (Blake, 2026-10-18). Decode and the producer take depth 1-4 (the intra's shape law is 1-2, open-work C-g): `D` may be a runtime value, and WHICH level is
addressed is compile-time (`uniform_switch` or a guarded `static_for`), never a runtime index (Blake, 2026-10-18). At
matched density the carry's work is the same at every depth, so depth's price is the producer's columns and the model's
quality.

---

## 3. The state

The carried state is a paged tensor of `[B, H, N, cols]` logical shape and nothing more. Its full contract — the
membership test that keeps launch machinery off it, binding, the backings, clone, the layer's continuation
container — is [`internals/state.md`](internals/state.md); this section states the architecture that contract
encodes.

### 3.1 The format descriptor, and one admission law

**The state owns its format** (CONSTITUTION §R13). A `StateFormat` binds at the state's first call, from the call
itself, and carries:

| field | what it is |
|---|---|
| `D`, `B` | the depth and the per-level widths, each a power of two |
| `order` | the canonical order of §2, as a field rather than an assumption |
| `page_bits` | the page rectangle: the trailing four canonical bits, which may span levels |
| `DV` | the value width; `cols = DV + 1` is the page's LOGICAL width, not its layout |
| `dtype` | the stored form, `fsla._state.STATE_DTYPE` (§3.2) |
| `ids` | the page-table id space, `BH · N / 16` |

**Later calls are CHECKED against it, and a mismatch is a refusal at the seam naming the field that differs.** The
descriptor is fixed for the state's life — a re-layout is an explicit state operation, never something a call does —
and it travels with the layer's continuation.

**Prefill and decode share ONE admission law: the descriptor.** Decode accepts exactly what prefill accepts — the
same depth, the same widths, the same `N`, the same value width, the same operand format, therefore the same
topologies. A second admission law would be a second contract to keep in step with the first, and the two kernels
partition ONE paged state.

**There is no ragged state.** `N = Π_l B_l` over powers of two, so a lawful state is page-aligned by construction; a
leaf axis that is not a whole number of pages is REFUSED — not padded, not carried in a tail page.

### 3.2 The page: s24 planes at a group exponent

A page is one atom — 16 leaves — stored as **planes, not rows**
([`internals/common/state_page.md#geometry`](internals/common/state_page.md#geometry)):

    [16 × DV hi s16][16 × DV lo u8][16 × DV/32 exponent s8][16 mass fp32]

A leaf-channel is the s24 mantissa `hi << 8 | lo` at its (leaf, 32-channel group)'s power-of-two exponent; the mass
is fp32. Three properties are the architecture:

- **The planes are aligned and never packed into 3-byte elements**: a hi row is one 128-byte line at `DV = 64`.
- **A hi-only read is the s16 operand.** A leaf a call only READS needs its hi plane, `hi · 2^(e + 8)` — the value
  truncated to 15 bits of its group, a declared operand rather than an approximation — so a read-only atom moves
  about two thirds of its bytes; a written leaf is read in full, updated, rounded once to its group's quantum and
  stored.
- **The zero page is zero bytes.** An all-zero group stores exponent 0, so a freshly zeroed page is the zero state and
  an absent atom reads as zeros without a conversion.

### 3.3 Two backings, one set of bytes

**Paged** is the default: a page arena, a `[BH, N/16]` page table of slot ids, and residency planned exactly from the
call's own write set. Resident state scales with the atoms the routing REALIZES, not with the `N` the topology
provisions. **Dense** is the same page sequence with every page resident. Dense and paged are one kernel, one ABI and
one base translation (`slot · kBytes`, the slot the atom itself under no table), and the gate between them is byte
identity on the storage ([`internals/paging/paging.md`](internals/paging/paging.md)). A caller cannot observe which
backing a state has except by asking.

`materialize()` decodes the pages back to the logical `[B, H, N, cols]` tensor. It is for the oracle and a test —
never on a measured path, since allocating that tensor is exactly what paging exists to avoid.

### 3.4 Activity

Residency is the arena's fact; **activity is the masks' fact**, and only the second decides what a call touches
(CONSTITUTION §R14). Per atom, one READ bit and one WRITTEN bit, never pre-ORed, folded on the host from the masks
(`fsla.engine.facts.planes.atom_bits`): exact allocation admits an atom iff it is WRITTEN this call; a
state sweep LOADS an atom iff it is resident and `(read ∨ written)` and STORES it iff `written`. Dense and paged touch
the identical atom set, so the byte gate certifies the skip logic and not merely the addressing, and a resident but
idle atom costs no state traffic. The price is the in-place law: an atom a sweep skips must already hold what it
carried in, so a continuation's exit plane IS its entry plane.

---

## 4. The carve, and the grains

**The carry's work is partitioned by ONE descriptor, the carve**
([`internals/engine/rules.md#carve`](internals/engine/rules.md#carve); the carve is re-derived by the prefill launch function, `open-work.md` row A-f):
a BOX — an extent a level and a channel extent — and each side's OPERATION PLAN. A side's sparse levels are its LISTS
(their live digits a window, listed by the carry CTA itself, from the token list the gather pass writes -- section 6.2) and its dense levels are its OPERAND (their slices form the MMA
operand, in level order); every side has a dense level, so every side has an operand. The dense level is SUPPORT
SYMMETRY by construction: the read-write overlap factorizes over levels (`Σ_n R[t,n] Wt[s,n]` is the product over `l`
of the levels' inner products), and a full-support factor at every level makes each factor strictly positive, so every
read overlaps every write -- an exact-zero pair, with its exact-zero gradient, cannot occur, while each level keeps the
hard sparsity the kernels compact. A fold warp's tile is its side's lists by its operand rows; a readout warp holds
every leaf for its channels, so where the two sides' carves differ the state reaches the readout through an image
(on hopper where they are equal too, the image then laid along the fold's own lists: MEASURED 1.3% over the divergent cell's window, [`carry_window`](lab/carry_window.md); a readout on the fold's own registers is not built). Every kernel that
depends on the geometry takes the carve — the gather pass groups its stream by the box's list levels, the
carry addresses by it — and validates the one it is handed; none decides a carve. The rule derives it from
launch-static facts (the rows' shapes and the bundle's declared flags), never from a routing statistic, so the
planner can change and every kernel keeps working for whatever it returns.

| grain | what it is | who owns it |
|---|---|---|
| **page** | 16 canonical leaves = one atom = one slot | the state's format |
| **box** | a CTA's leaves × channels: the carry's state, in registers for the whole call | the carve |
| **warp tile** | a side's lists × operand rows: a warp's share of the box | the carve's plan |
| **decode unit** | an owner and every run offset but the innermost: a contiguous run, one atom at the flagship | the decode step |

**The grains nest on the page and nowhere else** (CONSTITUTION §R14). Prefill and decode ownership are
decoupled: two kernels partitioning the same paged state, whose shared contract is the atom, the page table and the
activity bits.

---

## 5. The axis law

CONSTITUTION §R11 and §R12 in architectural form. **A compile-time axis is legitimate only as a deterministic
function of user input that the selector can pick; everything else is derived, addressed at runtime, or constant.**
The compiled key is the family's own:

| family | compiled key | runtime addressing |
|---|---|---|
| carry | not built on this line: no arm is declared | the box's per-level extents, the depth, the widths |
| decode | `DV × D × decay` (`csrc/fsla/src/decode/arms.bzl`) | the `(k, m)` lattice, the widths, the operand storage types |
| intra | the depth, one or two levels | each level's width in 16-digit granules |
| producer, gather pass (designs, not built) | none: one instantiation each | every width, kind and offset, from the plan or the layout |
| entmax solves | the padded width class, alpha, the operand dtypes | the level table |

The window `W`, the warp counts and the register budgets are constants. **Structure is branched on, never a
statistic**: a CTA-uniform branch over a generated, declared set is legal with the four enforcement layers of
KERNEL_STANDARDS R9 ([`internals/common/structure_switch.md`](internals/common/structure_switch.md)); a branch on a
density, a live count or "small L" never is.

**SMEM is launch-sized.** Every shared tenant whose size depends on a derived quantity is sized from the plan or the
descriptor and passed as dynamic shared memory, never max-sized; above the 48 KB static limit the one-time function
attribute is set. Residency is asserted at the seam from the launch size, as a refusal.

---

## 6. The kernels

Five kernel families, in one call's order: the producer writes the bundle; the gather pass reads it once and writes the carry a token
list, from which the carry CTA lays its own lists (section 6.2); the carry and the intra compute the two halves of the recurrence into the same outputs; decode is the
`T = 1` step. They share the window grid, the descriptor and the atom, and no code path that could make a wrong
convention agree with itself.

A buffer is judged by its LIFETIME before its size (Blake, 2026-10-15: two structures -- lifetimes pay for the
primitives, which stay purposeful): the bundle, the facts and the lists are working memory, one layer at a time and
bounded by the segment and the heads in flight, where attention's KV cache is permanent. That pays for per-token bytes
an attention layer does not have; it does not license waste, so a primitive's bytes are cut only where its purpose
survives.

### 6.1 The producer

ONE launch from a stretch of tokens to the bundle: every routing level's projection, its exact entmax or softmax
solve and its unit-max rows, the write gain, and the values, compiled from the layer's declared `Routing` so every
width, kind and offset is a runtime field and the family is one instantiation. No producer kernel is built on this
line (its hopper form: section 9); the solves are their own kernels
([`internals/entmax/entmax.md`](internals/entmax/entmax.md)), and the torch composition behind the same call is the
fp64 reference and the path a gradient takes (`fsla/ops/produce.py`).

### 6.2 The gather pass

On hopper the gather pass reads a sparse side's masks and writes the carry's stream: a sparse block's live tokens in
token order, their values and sparse bytes (RULED 2026-11-13 and 2026-11-15). It runs in its own CTAs beside the
carry's, in one grid, the stream a ring; a carry CTA lays its own lists from the stream and forms its own
coefficients ([`architecture/hopper-gather.md`](architecture/hopper-gather.md),
[`architecture/hopper-carry.md`](architecture/hopper-carry.md)). Sparsity is declared, never detected: the masks are
an input operand the pass uses blindly. No gather kernel is built on this line. Where the factor products and the tile
unions come from, and the masks' format on the surface, are open ([`open-work.md`](open-work.md), row A-e).

### 6.3 The carry — the inter-window term

One CTA a box and batch-head, one CTA an SM; the box's fp32 state lives in the CTA for the whole call. Per window: the
READOUT of the state as of the window's start by the window's readers, each token's partial sums added in fp32 into
`num` and, as the v+1 channel, `den` (carry CTAs join only there, R22; on hopper a carry CTA also reads the ring its
gather CTAs write, a hand-off not designed -- [`architecture/hopper-carry.md`](architecture/hopper-carry.md)); then the FOLD of the window's writes into the state.
No carry kernel is built on this line: its hopper schedule is a design of its own, written before it is built
([`architecture/hopper-carry.md`](architecture/hopper-carry.md), PROPOSED and not locked; section 9's forms;
[`open-work.md`](open-work.md)).

### 6.4 The intra — the within-window term

The causal term inside a window, its own kernel over the bundle, reducing into the output planes the carry reduces
into: per tile pair each level's Gram accumulates EXACTLY over its u8 digits by integer tensor ops, the levels multiply
in fp32, and the value product follows in bf16 against fp32 sums. The `[W × W]` matrix is never materialized, and a
tile pair whose digits meet nowhere is skipped on the gather pass's tile unions (designed, not built), an exact certificate. The kernel reads
no state: every pair it computes is written and read inside the one window. No intra kernel is built on this line
([`open-work.md`](open-work.md)). ON HOPPER it is a kernel of its own BESIDE THE CARRY, not a role in its CTA (Blake, 2026-11-13): its
chain alone is MEASURED at 985 cycles a tile pair of 64 readers by 64 writers at 256 digits a level
([`intra_chain`](lab/intra_chain.md)); its WINDOW as a kernel of its own is MEASURED at 176 SM-cycles a token with
the levels tile-major and the values token-major, and 418 with every plane token-major ([`intra_window`](lab/intra_window.md)); and the carry's CTA has no
register for it. The two are LAUNCHED CONCURRENTLY on two streams (Blake, 2026-11-13): MEASURED WITH THE TABLES CACHED -- the upside, not the
price (R22.2: the call with tables the cache does not hold is unmeasured) -- a call of 65,536 tokens with the carry,
its den and the intra is 0.39 ms a head so and 0.40 one after the other, at one cell
([`call_pair`](lab/call_pair.md)); nothing closer is built on that measurement.
BLACKWELL: NOT STATED -- no blackwell intra is designed and no blackwell card measured.

### 6.5 Decode — the `T = 1` step

At `T = 1` every tiling decision is degenerate, so decode is its own kernel: a sparse gather-GEMV per step over the
`(k, m)` lattice, its operands folded from the bundle by the step itself. One wave of resident CTAs VISITS heads from
a work queue: a visit builds its head's factor tables in a three-phase prologue, the warps claim positions of the
head's terms (the union of the read and write products as a disjoint sum), and each lane group walks a unit:
resolve, update-then-read — the deposit lands before the readout reads the row, which IS the semantics — and
accumulate; a visit adds its partial into the head's sums and the head's last visit writes `y`. A read-only leaf
pulls the hi plane alone (§3.2). The step asks its own residency question in its prologue and, under a slack pool,
admits its own growth ([`internals/decode/decode.md`](internals/decode/decode.md),
[`internals/decode/decode_lattice.md`](internals/decode/decode_lattice.md)).

---

## 7. Launch shape and the CTA policy

**The CTA is the multiprocessor** (CONSTITUTION §R15): every heavy kernel launches one CTA an SM with all the warps
its register plan admits — decode sixteen warps of up to 128 registers a lane; the carry's figures (eight warps of up to 255 registers a lane, 128
of them the box's state) are the design, not built. Latency hiding
is the kernel's job — covers, pipelines and split-phase edges — so exposed
latency is a coverage finding and never grounds to split the CTA. **Registers per leaf is the lever** that raises the
leaves resident per SM, not the block count, and the census reports the state's share of the register file beside the
register count. Units of work are independent and joined by atomics, so a launch degrades smoothly with the work
rather than at a threshold (R22).

---

## 8. Shipping

**A user of the Python package never registers an arm** (CONSTITUTION §R16). The shipped set is each family's
compiled key above, per architecture, and every other shape a family serves is served by runtime addressing, checked
at the seam: a call is either lawful and runs, or the refusal names the field (the descriptor's, `fsla.expert.prefill_descriptor`); no call runs a prefill yet, so every prefill call refuses.

**Closed-world codegen** is the rule for the binaries — pinned assembler, ahead-of-time only, the declared targets and the gates every build runs, and a device-side stamp asserted before any gate
([`ratification.md`](ratification.md), [`architecture/closed-world-codegen.md`](architecture/closed-world-codegen.md)).

**One body per operation family** (CONSTITUTION §R18). A single-valued axis is not plumbed as a parameter — it is
deleted, and the ratio readout is simply what the readout is.

---

## 9. Portability

Every structural mechanism states its hopper and blackwell form; a mechanism that exists on only one is a workaround,
not a design (CONSTITUTION §R8). A primitive may select different NATIVE instructions per architecture; it may never
select a wider-type emulation. Master is hopper only
(CONSTITUTION §R8): the build is sm_90a, blackwell is tabulated and not built
([`internals/common/hardware.md`](internals/common/hardware.md)), and the ampere family is the branch `ampere`, nothing
of it on master.

Each row is the form a kernel's design starts from, ESTIMATED from the PTX specification until its lab runs on the
card; a kernel's schedule is designed, and written into its own page, before it is built.

| mechanism | hopper (sm_90) | blackwell (sm_100) |
|---|---|---|
| the operands | u8 rows expanded to bf16 where they are staged, a scale or a coefficient multiplied in fp32 there -- for the carry by each carry CTA, from the token list the gather pass writes (`architecture/hopper-carry.md`; MEASURED with the fold expanding the entries of a stream laid by digit, and laying its slots from token-ordered views at a quarter over that); bf16 values as stored (`architecture/precision.md`) | the same; the carry's staging NOT STATED |
| the tensor products | `wgmma`: B from shared memory by descriptor, A from shared memory or from registers; fp32 accumulate; int8 for a window's pair products of rows, exact | `tcgen05`, operands and accumulators in tensor memory; the same two products |
| state in the register file | a role a warpgroup, the registers moved between them (`setmaxnreg`, one count a whole warpgroup: the folds at 160 a thread, the readout at 104, the drain warps' warpgroup at 88); the fold's fp32 accumulators the readout's operand, narrowed in place | the accumulator in tensor memory |
| the carry CTA's measured form | THE DESIGN'S FORM ([`architecture/hopper-carry.md`](architecture/hopper-carry.md)): no producer warp -- a fold warpgroup issues its own copies of the stream and makes its steps' operands where they land (MEASURED from a stream laid by digit, the fold expanding its entries' coefficients) -- the readout's coefficients in shared memory and its tokens on the step's columns, a pass 8 digits, and four drain warps; folds at 160 registers a thread, the readout at 104, the drain warps at 88: the least composed window MEASURED at even lists ([`carry_window`](lab/carry_window.md)), its den not measured. THE FORM THE DEN AND THE PRICES BELOW WERE MEASURED IN: two producer warps (one a box, a batch of 32 entries in registers each, two batches in flight) and two drain warps; the readout's A in TWO register sets reused across its chunk's commit groups (depth 2); every barrier wait a suspending poll (`mbarrier.try_wait`), never a plain polled loop; the fold's image K-major, a line one thread's own values; the den's fold half two more steps a slot in the slot's commit group, 8 columns wide, their A the slot's lists' coefficient bricks and their B a constant column of ones, the den's state 4 accumulators a thread; the den read as column 64 of the readout's steps, 72 columns wide, from a ninth brick of the image's tiles, an fp32 a token through the panel and one add a drain warp's lane, the readout at 88 registers and the fourth warpgroup at 104: 5.2% of the composed window with the tables cached and no clock inside a role (5.9% at an aligned cell) -- the fold's steps 64 cycles a window, the image's brick and store 91, the register split 88, the 72 columns 57, the panel's word and the drain's add 44 -- a dense write side unmeasured; the image tile's pitch an odd count of lines ([`carry_window`](lab/carry_window.md): PRICED at 2,014 SM-cycles a token, its tables read once (R22.2), and 832 when the cache holds them (824 with no clock inside a role), at the hardest cell MEASURED, on a card clocked at 1.76 to 1.82 GHz over the session (a memory-bound row's cycles follow the card's clock) -- sparse on opposite levels, 256 x 256 leaves; dense, R10's worst case, is priced higher by `tools/carry_floor.py` and unmeasured composed -- exact). The register-A mechanisms have no `tcgen05` analogue (the blackwell staging is the state's-image row's ESTIMATED form), and the common wait atom's suspending form is unwritten, owed with the first production kernel | NOT STATED: no blackwell carry is designed |
| operand staging from DRAM | contiguous rows by TMA box loads into rings, landed on `mbarrier`s, descriptors built at runtime; a carry's scattered entries, each its own request, by warps' own loads and stores (KERNEL_STANDARDS §33) | the same for contiguous rows; scattered rows not measured on a blackwell card |
| runtime addressing | uniform-datapath registers from the prologue, pairing with runtime descriptors | the same |
| the state's image | where the two sides' carves differ: the fold's fp32 state stored once a window as bf16 tiles in shared memory, one operand of the readout's step by descriptor, the other its readers' coefficients expanded to a tile in shared memory, the readers on the step's columns ([`carry_window`](lab/carry_window.md)); where they are equal, the same image laid along a sparse digit's dense digits (R17) | the image read from tensor memory; the coefficients expanded into shared memory as `tcgen05`'s operand, ESTIMATED from the PTX specification until that family's first lab; no blackwell card measured |
| the output's sum | fp32 inside a CTA; across CTAs fp32 or one f16 line a CTA a token at a coarse state-carried bound, as the plan states (R17). `red.global` a token: a carry's rows are not contiguous | the same |
| shared memory under pressure | a continuum, never a refusal short of the degenerate floor: the buffers' depths give way first (the panel's two to one, the pool's two slots to one), and only when no depth is left is the box shrunk, whose image and pools scale with its leaves; the card's shared-memory size is read where the kernel is entered (R8), never by a planner on the host. A shrunk box leaves fold registers unused: accepted, it is the failure case and no part of the `wgmma` family is short of shared memory (a CTA may take 227 KB of a multiprocessor's 228). The image is never ringed: it is a snapshot of the state at the window's top, which the fold's steps then mutate | the same order; the image is tensor memory's |
| the box grain | one CTA per multiprocessor, two boxes a CTA: a fold warpgroup a box and one readout warpgroup over both boxes' images, whether the two sides' carves differ or are equal (R20); no cluster -- its sum, its state transfer and its multicast each measured as a loss ([`cluster_sum`](lab/cluster_sum.md), [`state_move`](lab/state_move.md), [`value_cast`](lab/value_cast.md)) | one CTA per multiprocessor; its boxes a CTA and its roles are NOT STATED: no blackwell carry is designed and no blackwell card measured (owed with that family's first lab) |
| decode's walk and combine | per-lane-group gathers, a work queue of visits, global atomics into a head's sums | the same |
| the closed world | the declared targets and the per-build gates, device stamp | the same |

---

## 10. Index: mechanism → page

| mechanism | page |
|---|---|
| the recurrence and the ratio readout, as the executable spec | `fsla/ops/naive.py` |
| the update is affine in the input and never in the state | [`architecture/no-delta-rule.md`](architecture/no-delta-rule.md) |
| accumulation precision | [`architecture/precision.md`](architecture/precision.md) |
| a schedule per family, atoms, plans, the one entry | [`architecture/families.md`](architecture/families.md) |
| storing factors, expanding transiently | [`architecture/factored-representation.md`](architecture/factored-representation.md) |
| dead routes are exact zeros | [`architecture/self-masking-theorem.md`](architecture/self-masking-theorem.md) |
| the bundle, the public surface | [`api.md`](api.md), [`internals/fsla_api.md`](internals/fsla_api.md) |
| the state contract, descriptor, backings | [`internals/state.md`](internals/state.md) |
| the page's planes and its codec | [`internals/common/state_page.md`](internals/common/state_page.md) |
| the arena, the page table, the slack pool | [`internals/paging/paging.md`](internals/paging/paging.md) |
| the carve: the box and the operation plans | [`internals/engine/rules.md#carve`](internals/engine/rules.md#carve) |
| the host order of a call | [`internals/engine/prefill_dag.md`](internals/engine/prefill_dag.md), [`internals/engine/decode_dag.md`](internals/engine/decode_dag.md) |
| the digits, the row alignment law and the atom byte | [`internals/facts/liveness_contract.md`](internals/facts/liveness_contract.md) |
| decode: the step, the walk, growth, capture | [`internals/decode/decode.md`](internals/decode/decode.md), [`internals/decode/decode_lattice.md`](internals/decode/decode_lattice.md) |
| the entmax solves | [`internals/entmax/entmax.md`](internals/entmax/entmax.md), [`internals/entmax/factor.md`](internals/entmax/factor.md) |
| the architecture capability table and the per-product rates | [`internals/common/arch_caps.md`](internals/common/arch_caps.md), [`internals/common/hardware.md`](internals/common/hardware.md) |
| the laws and the constitution | [`KERNEL_STANDARDS.md`](KERNEL_STANDARDS.md) |
| the closed world and the census | [`ratification.md`](ratification.md), [`architecture/closed-world-codegen.md`](architecture/closed-world-codegen.md) |
| the tiers, the tolerances, the gates | [`testing.md`](testing.md) |
| the measurement method and record | [`measurement.md`](measurement.md) |
| the machine's measured facts | [`lab/README.md`](lab/README.md), the method [`internals/tools/lab.md`](internals/tools/lab.md) |
| what is knowingly unfinished | [`open-work.md`](open-work.md) |
| what was removed, and why it was safe | [`internals/DELETIONS.md`](internals/DELETIONS.md) |
