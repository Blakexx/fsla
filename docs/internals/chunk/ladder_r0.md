# `csrc/rola/src/chunk/ladder_r0.{cuh,cu}` — the backward ladder, rungs 0–2

The reverse pass, rebuilt by the FORWARD's methodology: start from an efficient dense
chunked linear-attention backward and add one RoLA feature per rung, pricing each in
both currencies. This component is the ladder's base plus two rungs — the mass
column with the self-normalizing readout's adjoint, and the carve with its split-K
fan-in. It is **not the production path**; the two scans in
[`backward.md`](backward.md) are.

Symbols, defined once and used unqualified below: `C` chunk tokens, `BC` slots per CTA
block, `BCg` slots per slot GROUP, `GRP` groups per block, `N` the slot space,
`nblk = N / BC` the block lattice, `d_v = 64` value channels, `T`/`L` tokens, `R[i,s]`/`W[j,s]`
the read/write coefficients, `S` the forward state `[N, d_v]` and `z` its mass column
`[N]`, `dS`/`dz` their adjoints, `dnum`/`dden` the readout's two cotangent seeds,
"visit" one CTA's pass over one chunk, "rendezvous" a `__syncthreads()`.

## <a id="what-it-is"></a>What the rung is, and what it is not

One kernel pair, on RoLA's own quantities, with every remaining RoLA feature absent:

* **dense, per slot**: the slot panel is the identity, so there is no `Producer`, no
  digit arithmetic and no digit gradient. The coefficients arrive PER SLOT,
  `[BH, L, N]`, and the per-slot gradients leave in the same shape;
* **contiguous chunks**: no union table, no `Descent`/`Ascent`, no compaction, no
  causal-threshold walk — chunk `c` is tokens `[cC, (c+1)C)` and `J(i) = i + 1`;
* **no gains**: the write coefficient carries whatever gain its caller folded in;
* **no ragged tail**: `L % C != 0` is REFUSED at the entry rather than masked. The
  mask is rung b6's, and a fallback here would be the special case that rung deletes.

What it carries from the forward, verbatim: `stage_tile`/`stage_commit`/`stage_wait`
with the same derived `STAGES` (`design::stages_for`), the `gather_ptr`/`pin_address`
addressing law, the `reg_floor` declaration mechanism, and the `kUnits = 4` carve in
which one MMA unit owns sixteen of the sixty-four value channels.

## <a id="correspondence"></a>The correspondence table

| dense chunked-LA | the rung's quantity | where |
|---|---|---|
| pass 1: rebuild `S` ascending | `S^T` per slot and `z` per slot, from zero | `rung0_pass1_kernel`'s fold |
| `dq = dO·S^T + (dO V^T ⊙ mask)·K` | `dR[i,s] = <dnum_i, S_a[s,:]> + dden_i z_a[s] + Σ_j [j≤i] W[j,s]·Gd'[i,j]` | pass 1's `dR` block |
| pass 2: carry `dS` descending | `dS^T` per slot and `dz` per slot | `rung0_pass2_kernel`'s fold |
| `dv = K·dS + (mask ⊙ Q K^T)^T dO` | `dv[j,:] = Σ_s W[j,s] dS_a[s,:] + Σ_i [i≥j] Gu[i,j] dnum[i,:]` | pass 2's `dv` block |
| `dk = V·dS^T + …` | `dW[j,s] = <dS_a[s,:], v_j> + dz_a[s] + Σ_i [i≥j] Gd'[i,j] R[i,s]` | pass 2's `dW` block |
| `Q K^T` Gram | `Gu = staircase(R·W^T)`, contracted over SLOTS | pass 2 |
| `dO V^T` Gram | `Gd' = staircase(dnum·v^T + dden⊗1)`, contracted over CHANNELS | both passes |
| state fold `S += k v^T` | `S^T += V^T·W`, `z += 1^T·W` | pass 1 |
| adjoint fold `dS += q dO^T` | `dS^T += dnum^T·R`, `dz += dden^T·R` | pass 2 |

The staircase is one predicate in both passes and in both Grams — rows are read
tokens, columns write tokens, and a write is kept at or before the read. The lists are
the identity here, so it is `tc <= tr` on chunk-local indices.

## <a id="mass"></a>The mass column, and why it costs no barrier and no byte

`W[j,s]` feeds both the state and the mass column, so `dW` differentiates both; the
whole correction is a **ones column appended to the write side** of the `d_v` Gram,
`Gd'[i,j] = <dnum_i, v_j> + dden_i` — one add per Gram entry, inside the store's single
rounding, and one transposed staircase still covers value and mass together
([`backward.md#mass-gram`](backward.md#mass-gram) is the derivation and its
falsification). `dv` is untouched: it is a value-channel quantity and the mass column
does not enter it.

Three placements make the rung absorb it inside the existing two phases:

* **the seed rides in the `dnum` panel's row pad.** Pass 2 carves 50,176 bytes, which
  IS the two-CTA ceiling on sm_86, so the mass column may not claim a buffer. `LdV`
  pads each value row by eight elements for `ldmatrix`'s alignment, and nothing reads
  those columns; `dden` seats there, entry `i` at row `i / 8`, column `d_v + i % 8`,
  written by the same threads that stage the chunk. Zero bytes, and it becomes visible
  at the ring's own rendezvous rather than at a new one;
* **both mass FOLDS ride the tensor pipe.** `z += Σ_j W[j,s]` is an MMA whose `A` is
  all ones, and `dz += Σ_i dden_i R[i,s]` is an MMA whose `A` is the seed broadcast
  down `m`. Both operands are EXACTLY representable in bf16 — one trivially, the other
  because the seed plane IS bf16 — so neither needs the compensated split that a
  general fp32 operand would, and the accumulator lands the totals in precisely the
  lane layout the entry-mass term reads. No reduction, no shuffle, no shared buffer.
  This is a property of the operands, not a general result: an fp32 seed plane would
  put the read-side fold back on the scalar path;
* **the two ENTRY terms stay scalar, because they are rank-1.** `dden_i z_a[s]` is an
  outer product and `dz_a[s]` a per-slot broadcast; both are one FMA or add per
  accumulator element and have no MMA form worth the name. They run inside the
  per-slot block that already exists.

The eight-wide pad groups are load-bearing rather than incidental: they are what makes
`(i, i+1)` adjacent for every even `i`, which is exactly the pair an MMA `A` fragment's
lane wants.

## <a id="phases"></a>Two rendezvous per visit, and what buys them

The forward's per-visit shape is one barrier for "the staged tile is visible and the
last visit is drained" and one for "the produced operands are visible". Rung 0 spends
the same two, in both passes:

```
stage_wait<STAGES-2>(); rendezvous()   # (1) this chunk's tiles and seed visible
stage the chunk STAGES-1 ahead; commit
publish the entry state; form the Gram(s), ones column included
rendezvous()                           # (2) the state and the Grams are visible
the per-slot term with its entry mass; the value term; the state and mass folds
```

The mass column adds no phase to this. Both entry mass columns live in ACCUMULATOR
registers rather than in shared memory, so the only thing that must cross threads is
the seed, and the seed crosses on the ring.

**The publication is what removes the phases.** The per-slot terms (`dR` in pass 1,
`dW` in pass 2) contract over the CHANNEL axis, which is the carve's own — a unit
holds sixteen of sixty-four channels, so no unit can finish one from its registers.
The shipped Scan B answers that by transposing ONE PASS of the slots into shared per
iteration, which costs a rendezvous per pass and is why it spends nine to thirteen.
Rung 0 publishes the WHOLE state once per visit instead — `[d_v][BC]` bf16, written by
four units into disjoint channel rows — and then splits the per-slot output over the
SLOT axis, so every unit runs the whole channel contraction out of shared and the four
outputs are disjoint. One publication, one barrier, no cross-unit sum.

Two consequences make it affordable, and both are geometry rather than luck:

* the published state's rows are **UNPADDED**. It is read four times per unit per
  visit (one `ldmatrix` per channel k-tile) against ~600 MMA, so its bank conflicts are
  unmeasurable, while its pad would be 1,024 bytes and pass 2's carve has none to
  spare: 50,176 bytes IS the largest carve admitting two CTAs per SM on sm_86;
* the value cotangent `dv` needs no publication at all: its contraction runs over
  SLOTS, which the channel carve leaves whole, so each unit emits its own sixteen
  channels complete and there is no `so` staging buffer. What its slot axis does cost
  is the fan-in below.

## <a id="ring"></a>Every consumed stream rides the ring

Four streams in pass 2 (`V`, `dnum`, `R`, `W`) and three in pass 1 (`R` is a write-side
pass's operand only), each a contiguous `[C, cols]` tile of its plane, all staged
through the same `cp.async` ring at the depth `design::stages_for` derives — plus the
`[C]` mass seed, which rides in the `dnum` tile's pad and is therefore the one stream
staged by a plain load/store rather than by `cp.async`: its entries are two bytes and
its destination stride is a row, which is neither of the shapes `cp.async` copies. The
chunks are contiguous, so `TileStage` contributes no gather: a tile's rows are the
plane's own rows and a row's address is one 32-bit displacement off a pinned base.

The ring's cost is what bounds the rung's GROUP width. At `C = 32, BCg = 64` pass 2
carves 50,176 bytes, which is exactly the largest carve that admits two CTAs per SM on
sm_86; that zero margin is why the mass column is placed the way it is — a 256-byte
`[BCg]` fp32 publication buffer would have cost the residency outright — and it is why
a 128-slot group does not exist. See [the group threshold](#the-group-threshold).

## <a id="the-group-threshold"></a>The group threshold, and why it is 128 and not 256

The forward splits a block into two slot GROUPS at `BC >= 256`, where one group's
k-run would exceed its measured shape. The ladder splits at **128**, and the reason is
that the adjoint carries two costs per group's slot width the forward does not: a
SECOND coefficient panel on the ring, and the publication. Held at `C = 32` with the
ring at its derived depth of two:

| buffer | `BCg = 64` | `BCg = 128` |
|---|---:|---:|
| value ring (`V`, `dnum`), CTA-wide | 18,432 | 18,432 |
| coefficient ring (`R`, `W`), per group | 18,432 | 34,816 |
| the Grams | 5,120 | 5,120 |
| the publication `[d_v][BCg]` | 8,192 | 16,384 |
| **pass 2 total** | **50,176** | **74,752** |

The two-CTA ceiling is 50,176. A 128-slot group is 24,576 bytes past it and **no
reordering recovers the difference, because all four buffers are live across the whole
visit**: the publication is irreducible (the object that must cross the units is
`d_v x BCg` bf16 however it is sliced, and publishing it in the contraction's own
k-tiles costs a rendezvous per tile), single-buffering the coefficient panels saves
17,408 and costs a third rendezvous, and killing every `ldmatrix` pad recovers ~4,096
against tiles that carry ~600 MMA a visit.

So `BC = 128` is not a wider group; it is TWO of them. A block of two groups is one
CTA of 256 threads, and what that holds constant is the quantity the ideal regime is
stated in — **warps per SM**: eight, either as two CTAs of four warps at `BC = 64` or
as one CTA of eight at `BC = 128`. The register set per warp is unchanged, because a
group's geometry is unchanged.

Two objects are shared rather than duplicated across the groups, and each is shared
for a stated reason: the **value ring** (`V`, `dnum` carry no slot axis, so both groups
stage one copy — half the global traffic of two separate CTAs) and the **`d_v` Gram**
(`Gd'` contracts over channels and has no slot axis either, so group zero forms it once
for the CTA — half the Gram MMA). The **slot Gram** `Gu` is per group, because it
contracts over the group's own slots; it is therefore a partial of the block's, which
is the same statement as `dv` fanning in.

## <a id="fan-in"></a>The split-K fan-in

A block owns `BC` of the `N` slots and a group `BCg` of the block's. Over that
partition:

* `dR` and `dW` are **slot-disjoint**. Their slot axis is the OUTPUT's, so a block
  writes its own columns of `[BH, L, N]` and writes each exactly once. They are
  allocated and never filled, and nothing about them fans in. (At rung b3 the digit
  gradients replace them and that changes: a digit is shared across blocks, so a digit
  gradient IS a fan-in and inherits the mechanism below.)
* `dv` is **not**. Its contraction `dv[j,:] = Σ_s W[j,s] dS_a[s,:]` runs over the whole
  slot space, and so does the intra term's `Gu`, so every group of every block holds a
  partial and the arena is the only place they meet.

The mechanism is `fan_in_add` — the non-returning `atomicAdd`, which assembles as
`RED` and not `ATOM`, so the issuing warp never waits on a return value. The two
alternatives were priced and rejected on the GROWTH AXIS rather than on today's cell: a
per-block partial arena reduced by a second kernel is `nblk` copies of `[BH, L, d_v]`
fp32 and `nblk = N / BC` is the axis the design scales along, and a slot-ownership
partition cannot apply at all, because `dv`'s sum IS over the partitioned axis.

**The identity is written once, by the allocation, before any contribution.** `dv` is
`torch::zeros` and every other output is `torch::empty`; a fill per contributing block
would be `nblk` fills of the same plane and a race besides. That is the recovery
batch's lesson stated as a rule: a fan-in target needs its identity before the first
contribution, and there is exactly one target here.

At `N = BC` the lattice is one block, `GRP` is one, and the single contribution lands
on the arena's zero — so the one-block cell is BIT-IDENTICAL to the rung that preceded
the carve, which is how the carve is gated.

## <a id="arms"></a>The arm list, and why it is `(C, BC)` alone

`LADDER_R0_ARMS` is GENERATED, from `tools/gen_shards.py` — the same generator that
owns `CHUNK_ARMS` and `SCAN_ARMS` — and each `(C, BC)` is checked at generation time
against a shape the consumer is built at. A topology is not one of the ladder's
COMPILED axes: `N` reaches the kernel as a runtime plane width and the block lattice as
the grid's own x extent, so `(D, B)` has nothing to say to the arm list.

`(32, 64)` is the HEAD-TO-HEAD arm. It is the one place a scan arm's `N` equals its
`BC` — the scan arms `(32, 64, 2, 8)` and `(32, 64, 1, 64)` both have `N = 64` — so it
is the only shape at which the ladder and the shipped scans run the same problem over
the same ONE-BLOCK grid, and therefore where their walls compare most directly.
`(16, 64)` is the second point on the chunk axis. `(32, 128)` is the SHIPPED block
width: the pinned cell `(32, 128, 2, 64)` and the scale cell `(32, 128, 2, 256)` both
resolve to it, it is a two-group block, and it is what rung b2's carve buys.

## <a id="gates"></a>What gates it

`tests/oracle/test_ladder_r0.py`, against three references and no one of them alone:

* the **fp64 autograd oracle** — the eleven-line recurrence in float64, differentiated
  by autograd against the seeded linear form. It carries no chunking and no rounding,
  so it states WHAT the kernel computes and cannot be satisfied by a kernel that is
  merely self-consistent with its own restatement;
* the **fp64 RATIO oracle** — the same recurrence read out as `num / den` and
  differentiated end to end from `dy`, with the two seeds DERIVED from that readout and
  handed to the kernel. Nothing but the mass column and the ones column can carry the
  denominator's half of the gradient, which is what makes this the gate for the
  self-normalizing readout rather than for two unrelated linear functionals;
* the **rounded reference** — the same algebra chunked as the kernel chunks it, with
  every stored operand rounded where the kernel rounds it (the ones column INSIDE the
  Gram's one rounding), which is what makes the budget tight.

The teeth are explicit, and five faults are planted in the reference: the intra
staircase dropped, the staircase made exclusive, the R-b ones column absent, the entry
mass absent, and the per-slot mass flattened to its own mean. The last is what makes
the mass adjoint NON-VACUOUS — a per-slot quantity that never varied over slots would
be indistinguishable from a scalar — and the cells are checked to vary before it is
applied. The mass faults must also leave `dv` UNMOVED, which is the statement that the
mass column stays out of the value channels. Both mass regimes run: `dden = 0` is the
configuration in which every mass term is identically zero, and it is a regression
row rather than a gate.

The fan-in has its own two gates. `test_the_fan_in_has_more_than_one_live_contributor`
asserts of each multi-block cell that EVERY block's partial is a material share of
`dv` — on a cell where one block carried the sum, a kernel that stored instead of
adding would pass everything else — and the one-block cell is checked BIT-IDENTICAL to
the rung before the carve, which is the statement that the addressing changed and the
arithmetic did not.

The census (`chunk_rung0_census`) carries both halves of the floor claim per arm and
per pass — the declared live set and the budget its target admits — and the gate
asserts zero spill, a staging depth of at least two, and the forward's own residency.
That last is stated in **warps per SM** rather than in CTAs, and it became so at rung
b2: `BC = 128` runs one CTA of eight warps where `BC = 64` runs two of four, and the
eight is what the carve holds constant. The twelfth census column is the thread count,
so the gate can say so.
