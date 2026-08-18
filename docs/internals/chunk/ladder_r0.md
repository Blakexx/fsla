# `csrc/rola/src/chunk/ladder_r0.{cuh,cu}` — the backward ladder, rungs 0–3

The reverse pass, rebuilt by the FORWARD's methodology: start from an efficient dense
chunked linear-attention backward and add one RoLA feature per rung, pricing each in
both currencies. This component is the ladder's base plus three rungs — the mass
column with the self-normalizing readout's adjoint, the carve with its split-K fan-in,
and the ROUTED operand with its digit gradients. It is **not the production path**;
the two scans in [`backward.md`](backward.md) are.

Symbols, defined once and used unqualified below: `C` chunk tokens, `BC` slots per CTA
block, `BCg` slots per slot GROUP, `GRP` groups per block, `D` routing levels, `Bw` one
uniform level width, `N = Bw^D` the slot space, `nblk = N / BC` the block lattice,
`d_v = 64` value channels, `T`/`L` tokens, `pr`/`pw` the per-level amplitudes
`[BH, L, D·Bw]`, `R[i,s] = Π_l pr[i, l·Bw + d_l(s)]` / `W[j,s]` the per-slot
coefficients they factor into, `d_pr`/`d_pw` the per-level DIGIT gradients,
`S` the forward state `[N, d_v]` and `z` its mass column `[N]`, `dS`/`dz` their
adjoints, `dnum`/`dden` the readout's two cotangent seeds, `od` a slot's OUTER digit
(level 0), `i1` its INNER digit (level `D−1`), "LOO" the leave-one-out product
`Π_{l'≠l} p_{l'}`, "visit" one CTA's pass over one chunk, "rendezvous" a
`__syncthreads()`.

## <a id="what-it-is"></a>What the rung is, and what it is not

One kernel pair, on RoLA's own quantities, with every remaining RoLA feature absent:

* **routed, per level**: the coefficients arrive as the `D` amplitude factors,
  `[BH, L, D·Bw]`, and the Kronecker product is formed IN KERNEL into the same panel
  rung b2 staged; the gradients leave in the same `[BH, L, D·Bw]` shape as per-level
  digit gradients. Nothing per-slot crosses the kernel boundary in either direction —
  see [the operand](#the-operand) and [the digit reduction](#the-digit-reduction);
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
| `dq = dO·S^T + (dO V^T ⊙ mask)·K` | `dR[i,s] = <dnum_i, S_a[s,:]> + dden_i z_a[s] + Σ_j [j≤i] W[j,s]·Gd'[i,j]`, reduced to `d_pr` (A.9) | pass 1's `dR` block |
| pass 2: carry `dS` descending | `dS^T` per slot and `dz` per slot | `rung0_pass2_kernel`'s fold |
| `dv = K·dS + (mask ⊙ Q K^T)^T dO` | `dv[j,:] = Σ_s W[j,s] dS_a[s,:] + Σ_i [i≥j] Gu[i,j] dnum[i,:]` | pass 2's `dv` block |
| `dk = V·dS^T + …` | `dW[j,s] = <dS_a[s,:], v_j> + dz_a[s] + Σ_i [i≥j] Gd'[i,j] R[i,s]`, reduced to `d_pw` (A.10) | pass 2's `dW` block |
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

Two staged streams in either pass (`V`, `dnum`), each a contiguous `[C, d_v]` tile of
its plane, moved through the `cp.async` ring at the depth `design::stages_for` derives;
the coefficient panels — two in pass 2 (`R`, `W`) and one in pass 1 (`W`; the read side
is a write-side pass's operand only) — occupy the SAME ring slots at the SAME bytes but
are PRODUCED into them rather than copied, because at rung b3 their contents do not
exist in memory. Plus the
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

* `d_pr` and `d_pw` **fan in**, and that is rung b3's own change. Rung b2's per-slot
  `dR`/`dW` were slot-DISJOINT — their slot axis was the output's, so a block wrote its
  own columns and wrote each exactly once, and pass 1 carried zero `RED`. A DIGIT is
  not a slot: every block whose slot run touches a digit contributes to it, so both
  digit gradients meet the same lattice `dv` does and pass 1 acquires its first `RED`.
* `dv` likewise. Its contraction `dv[j,:] = Σ_s W[j,s] dS_a[s,:]` runs over the whole
  slot space, and so does the intra term's `Gu`, so every group of every block holds a
  partial and the arena is the only place they meet.

The mechanism is `fan_in_add` — the non-returning `atomicAdd`, which assembles as
`RED` and not `ATOM`, so the issuing warp never waits on a return value. The two
alternatives were priced and rejected on the GROWTH AXIS rather than on today's cell: a
per-block partial arena reduced by a second kernel is `nblk` copies of `[BH, L, d_v]`
fp32 and `nblk = N / BC` is the axis the design scales along, and a slot-ownership
partition cannot apply at all, because `dv`'s sum IS over the partitioned axis.

**Every arena's identity is written once, by the allocation, before any contribution.**
All three outputs are `torch::zeros` at rung b3 and nothing is `torch::empty`; a fill
per contributing block would be `nblk` fills of the same plane and a race besides. That
is the recovery batch's lesson stated as a rule, and b3 is where it applies to three
targets rather than one.

Unlike `dv`, the digit gradients are non-vacuous fan-ins even at a ONE-BLOCK cell: a
digit is shared across the block's four units and both its groups, so a kernel that
stored instead of adding is caught there too. What ONLY the lattice can catch is a
column formed without the block base, and that is the fault the gate plants — see
[what gates it](#gates).

## <a id="the-operand"></a>The operand: per-level factors, expanded in kernel

A block's slots are a contiguous `BC`-aligned run of the leaf order, a group's a
contiguous `BCg`-aligned one, and a UNIT's a contiguous run of `BCg / kUnits`. Each of
those is therefore a RECTANGLE in digit space, which is what makes the expansion one
splat and one vector multiply rather than a per-slot gather: an eight-slot run is
eight-aligned and `Bw ≥ 8`, so it lies inside ONE outer digit and covers eight
CONSECUTIVE inner ones. `ExpandR0` reads the run's inner amplitudes as one 16-byte load,
splats the outer one, and stores the product where `TileStage` used to land a
`cp.async`. At `D = 1` the outer load does not exist — the level is DELETED, not
weighted by one.

The panel it fills is byte-for-byte the panel rung b2 staged, at the same pitch and in
the same ring slot, so every consumer downstream of it — both Grams, both folds, both
per-slot contractions — is unchanged. The expansion is issued where `stage_chunk` is
issued, into the slot the NEXT visit consumes, so its visibility is established by that
visit's first rendezvous and the barrier count does not move.

## <a id="the-digit-geometry"></a>The digit geometry a unit sees

`DigitR0` carries it, and it is three numbers derived rather than tabulated:

    unit_run = BCg / kUnits          a unit's contiguous slot run
    spo      = min(unit_run, Bw)     slots of that run sharing one outer digit
    opu      = unit_run / spo        outer digits the run covers
    npo      = spo / kMmaN           accumulator tiles under one outer digit

Exactly one of `spo`/`opu` is non-trivial at any legal arm, because `Bw ≥ 8` and the run
is a power of two: at `Bw = 8` a unit's sixteen slots are two whole outer digits, and at
`Bw ≥ 16` they are part of one.

Note what is NOT here: a pass loop. The shipped scans walk `P = BCg / 32` passes of the
slots because a unit reads the adjoint state through one `ldmatrix` per pass; the ladder
PUBLISHES the whole channel axis once per visit, so a unit holds its whole slot run in
accumulators and `P` is 1 by construction. The design's b3 stop — *"the per-pass
rendezvous cannot be hoisted out of the pass loop"* — is therefore vacuous here: there
is no pass loop to hoist it out of, and the barrier count stays at two.

## <a id="the-digit-reduction"></a>The digit reduction, in lane

    d_p_{D−1}[t, i1] = Σ_{s : i1(s)=i1} dX[t,s] · p_0[t, od(s)]        (INNER)
    d_p_0[t, od]     = Σ_{s : od(s)=od}  dX[t,s] · p_{D−1}[t, i1(s)]   (OUTER)

The leave-one-out factor is the OTHER level's amplitude and never `X / p_l`: under exact
entmax both are exactly zero off the support and the division is `0/0` with no epsilon
that does not invent a gradient (`backward.md#digit-reduction`).

**The per-slot gradient never leaves a register**, which is the whole of the rung on the
output side. A lane's accumulator holds two adjacent slots of two tokens per
`(m, n)`, so the INNER digit is the lane's own and needs no reduction at all — one
fan-in per slot, the minimum the destination admits. The OUTER one sums over the slots
of the unit's run that share its digit: the lane's two, then two `__shfl_xor` over
`lane & 3`, then the `npo` tiles the unroll already visits. What remains — across units,
groups and blocks — is the fan-in's, exactly as `dv`'s was at rung b2.

### <a id="the-emit-order"></a>The emit's loop order, and one refutation

Two orderings were built and MEASURED, at the pinned arm:

| order | registers (pass 1) | pass 1 wall |
|---|---:|---:|
| slot tile outer, token inner (SHIPPED) | 143 | 2.087 ms |
| token outer, slot tile inner | 162 | 2.548 ms |

The token-outer form shares one destination row base across every access a
`(token, half)` makes and looks strictly cheaper in address arithmetic; it holds the
whole accumulator column live across the emit instead, and pays 19 registers and 22 %
for it. The slot-outer form retires `acc[·][n]` a tile at a time.

### <a id="the-hoisted-leave-one-out"></a>The hoisted leave-one-out, refuted

The leave-one-out amplitudes depend on the chunk and the lane and on nothing the visit
computes, so hoisting their loads to the top of the visit — where the publication, the
Grams and the per-slot contraction would cover their latency — is the ladder's own
doctrine applied. It was built and it does not pay: **+1.2 % on pass 1, −4.4 % on
pass 2, for 38 registers** at the pinned arm. The emit is not latency-bound; the stall
profile puts it on the LSU/MIO issue queue (`lg_throttle` 1.63/1.46 against the shipped
scans' 0.01/0.00), and a queue is not shortened by issuing into it earlier.

## <a id="arms"></a>The arm list, and why it is `(C, BC, D, B)`

`LADDER_R0_ARMS` is GENERATED, from `tools/gen_shards.py` — the same generator that owns
`CHUNK_ARMS` and `SCAN_ARMS` — and each tuple is checked at generation time against a
geometry the consumer is built at. Rung b3 made the TOPOLOGY a compiled axis: the digit
structure a unit's slot run is carved by is `(D, Bw)`'s, and `N = Bw^D` stopped being a
runtime plane width, so the list carries the whole four-tuple and the census keys on it.

The list is `SCAN_ARMS` — every geometry the shipped reverse pass is built for, so that
every ladder receipt is head to head with a scan receipt at the same one — plus
`(16, 64, 2, 16)`, the second point on the CHUNK axis, which has no scan of its own.
`(32, 64, 2, 8)` and `(32, 64, 1, 64)` are the ONE-BLOCK arms (`N = 64 = BC`), and they
are the control the lattice fault is read against.

## <a id="gates"></a>What gates it

`tests/oracle/test_ladder_r0.py`, against three references and no one of them alone:

* the **fp64 autograd oracle** — the eleven-line recurrence in float64, differentiated
  by autograd against the seeded linear form THROUGH THE FACTORIZED FORM: the autograd
  variables are the per-level amplitudes and the Kronecker product is inside the graph,
  so what comes back is the digit gradient itself, obtained by differentiation rather
  than by restating the kernel's own reduction. It carries no chunking, no lattice and
  no rounding, so it states WHAT the kernel computes and cannot be satisfied by a
  kernel that is merely self-consistent with its own restatement;
* the **fp64 RATIO oracle** — the same recurrence read out as `num / den` and
  differentiated end to end from `dy`, with the two seeds DERIVED from that readout and
  handed to the kernel. Nothing but the mass column and the ones column can carry the
  denominator's half of the gradient, which is what makes this the gate for the
  self-normalizing readout rather than for two unrelated linear functionals;
* the **rounded reference** — the same algebra chunked as the kernel chunks it, with
  every stored operand rounded where the kernel rounds it (the ones column INSIDE the
  Gram's one rounding), which is what makes the budget tight.

The teeth are explicit, and six faults are planted in the reference: the intra
staircase dropped, the staircase made exclusive, the R-b ones column absent, the entry
mass absent, the per-slot mass flattened to its own mean, and — rung b3's own — the
leave-one-out product taken from the WRONG duty's amplitudes, which leaves every
per-slot term exactly right and only the digit reduction wrong. The last is what makes
the mass adjoint NON-VACUOUS — a per-slot quantity that never varied over slots would
be indistinguishable from a scalar — and the cells are checked to vary before it is
applied. The mass faults must also leave `dv` UNMOVED, which is the statement that the
mass column stays out of the value channels. Both mass regimes run: `dden = 0` is the
configuration in which every mass term is identically zero, and it is a regression
row rather than a gate.

The fan-in has its own two gates. `test_the_fan_in_has_more_than_one_live_contributor`
asserts of each multi-block arm that EVERY block's partial is a material share of ALL
THREE fan-in targets — on a cell where one block carried a sum, a kernel that stored
instead of adding would pass everything else. The bar is the EVEN share, `0.3 / nblk`,
and not an absolute: at the 512-block scale arm an even contributor carries 0.2 % of
the total and an absolute floor would call a healthy lattice vacuous.

**There is no bit-identity control at this rung, and that is a statement rather than an
omission.** Rung b2 could be gated `torch.equal` against b1's binary at the one-block
cell, because the carve changed the ADDRESSING and not the arithmetic. b3 changed the
OPERAND FORMAT on both sides: the coefficient is now a product formed in kernel and the
gradient a sum over slots taken in a different order, so a one-block dense cell SHOULD
move, and hunting an identity there would be hunting an impossible one. The control is
the ORACLE, at every arm.

The second gate is the KERNEL-SIDE planted fault, and it is chosen so the LATTICE is
what it tests: the digit column formed from the slot's offset WITHIN the block instead
of from its absolute slot — `sbase % BC` — which is the `off_at` defect
`backward.md#pass-law` records against the shipped code. At a one-block arm the block
base is zero and the fault is invisible; everywhere else every block fans into block
zero's columns.

The census (`chunk_rung0_census`) carries both halves of the floor claim per arm and
per pass — the declared live set and the budget its target admits — and the gate
asserts zero spill, a staging depth of at least two, and the forward's own residency.
That last is stated in **warps per SM** rather than in CTAs, and it became so at rung
b2: `BC = 128` runs one CTA of eight warps where `BC = 64` runs two of four, and the
eight is what the carve holds constant. The thread count is a census column so the gate
can say so, and at rung b3 the row's KEY became the whole `(C, BC, D, B)` arm: two rows
that agree on the block width and differ on the digit geometry are different kernels,
and a census keyed on `(C, BC)` would have collapsed them. The gate also asserts the
peak against the DECLARED floor and not only against the budget — the floor is the
number `cta_target` is derived from, so a peak above it is a residency claim nobody
measured.
