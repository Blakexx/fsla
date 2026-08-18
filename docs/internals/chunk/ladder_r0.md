# `csrc/rola/src/chunk/ladder_r0.{cuh,cu}` — the backward ladder, rung 0

The reverse pass, rebuilt by the FORWARD's methodology: start from an efficient dense
chunked linear-attention backward and add one RoLA feature per rung, pricing each in
both currencies. This component is rung 0 — the base the ladder is measured from. It
is **not the production path**; the two scans in [`backward.md`](backward.md) are.

Symbols, defined once and used unqualified below: `C` chunk tokens, `BC` slots per CTA
block, `N` the slot space, `d_v = 64` value channels, `T`/`L` tokens, `R[i,s]`/`W[j,s]`
the read/write coefficients, `S` the forward state `[N, d_v]`, `dS` its adjoint,
`dnum` the readout's value-channel cotangent seed, "visit" one CTA's pass over one
chunk, "rendezvous" a `__syncthreads()`.

## <a id="what-it-is"></a>What rung 0 is, and what it is not

One kernel pair, on RoLA's own quantities, with every RoLA feature absent:

* **dense**: `N = BC`, one block, no carve, no fan-in — the slot panel is the identity,
  so there is no `Producer`, no digit arithmetic and no digit gradient. The
  coefficients arrive PER SLOT, `[BH, L, BC]`;
* **contiguous chunks**: no union table, no `Descent`/`Ascent`, no compaction, no
  causal-threshold walk — chunk `c` is tokens `[cC, (c+1)C)` and `J(i) = i + 1`;
* **no mass**: no `z`, no `dz`, no ones column — the `d_v` Gram is `Gd = dnum·v^T`;
* **no gains**: the write coefficient carries whatever gain its caller folded in;
* **no ragged tail**: `L % C != 0` is REFUSED at the entry rather than masked. The
  mask is rung b6's, and a fallback here would be the special case that rung deletes.

What it carries from the forward, verbatim: `stage_tile`/`stage_commit`/`stage_wait`
with the same derived `STAGES` (`design::stages_for`), the `gather_ptr`/`pin_address`
addressing law, the `reg_floor` declaration mechanism, and the `kUnits = 4` carve in
which one MMA unit owns sixteen of the sixty-four value channels.

## <a id="correspondence"></a>The correspondence table

| dense chunked-LA | rung 0's quantity | where |
|---|---|---|
| pass 1: rebuild `S` ascending | `S^T` per slot, from zero | `rung0_pass1_kernel`'s fold |
| `dq = dO·S^T + (dO V^T ⊙ mask)·K` | `dR[i,s] = <dnum_i, S_a[s,:]> + Σ_j [j≤i] W[j,s]·Gd[i,j]` | pass 1's `dR` block |
| pass 2: carry `dS` descending | `dS^T` per slot | `rung0_pass2_kernel`'s fold |
| `dv = K·dS + (mask ⊙ Q K^T)^T dO` | `dv[j,:] = Σ_s W[j,s] dS_a[s,:] + Σ_i [i≥j] Gu[i,j] dnum[i,:]` | pass 2's `dv` block |
| `dk = V·dS^T + …` | `dW[j,s] = <dS_a[s,:], v_j> + Σ_i [i≥j] Gd[i,j] R[i,s]` | pass 2's `dW` block |
| `Q K^T` Gram | `Gu = staircase(R·W^T)`, contracted over SLOTS | pass 2 |
| `dO V^T` Gram | `Gd = staircase(dnum·v^T)`, contracted over CHANNELS | both passes |
| state fold `S += k v^T` | `S^T += V^T·W` | pass 1 |
| adjoint fold `dS += q dO^T` | `dS^T += dnum^T·R` | pass 2 |

The staircase is one predicate in both passes and in both Grams — rows are read
tokens, columns write tokens, and a write is kept at or before the read. At rung 0 the
lists are the identity, so it is `tc <= tr` on chunk-local indices.

## <a id="phases"></a>Two rendezvous per visit, and what buys them

The forward's per-visit shape is one barrier for "the staged tile is visible and the
last visit is drained" and one for "the produced operands are visible". Rung 0 spends
the same two, in both passes:

```
stage_wait<STAGES-2>(); rendezvous()   # (1) this chunk's tiles visible, last visit dead
stage the chunk STAGES-1 ahead; commit
publish the entry state; form the Gram(s)
rendezvous()                           # (2) the state and the Grams are visible
the per-slot term; the value term; the fold
```

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
  unmeasurable, while its pad would be 1,024 bytes — exactly the margin that keeps pass
  2's carve at two resident CTAs;
* the value cotangent `dv` needs no publication at all: its contraction runs over
  SLOTS, which the channel carve leaves whole, so each unit emits its own sixteen
  channels complete and stores them straight to global. There is no `so` staging
  buffer and no fan-in, because at `N = BC` one CTA owns the whole slot space.

## <a id="ring"></a>Every consumed stream rides the ring

Four streams in pass 2 (`V`, `dnum`, `R`, `W`) and three in pass 1 (`R` is a write-side
pass's operand only), each a contiguous `[C, cols]` tile of its plane, all staged
through the same `cp.async` ring at the depth `design::stages_for` derives. Rung 0's
chunks are contiguous, so `TileStage` contributes no gather: a tile's rows are the
plane's own rows and a row's address is one 32-bit displacement off a pinned base.

The ring's cost is what bounds the rung's block width. At `C = 32, BC = 64` pass 2
carves 50,176 bytes, which is exactly the largest carve that admits two CTAs per SM on
sm_86; `BC = 128` does not fit at any depth, and widening it is rung b2's carve
question rather than a constant to raise here.

## <a id="arms"></a>The arm list, and why it is `(C, BC)` alone

`LADDER_R0_ARMS` is GENERATED, from `tools/gen_shards.py` — the same generator that
owns `CHUNK_ARMS` and `SCAN_ARMS` — and each `(C, BC)` is checked at generation time
against a shape the consumer is built at. A topology is not one of rung 0's axes: the
slot space IS the block, so `(D, B)` has nothing to say to it.

`(32, 64)` is the HEAD-TO-HEAD arm. It is the one place a scan arm's `N` equals its
`BC` — the scan arms `(32, 64, 2, 8)` and `(32, 64, 1, 64)` both have `N = 64` — so it
is the only shape at which rung 0 and the shipped scans run the same problem over the
same grid, and therefore the only shape at which their walls compare. `(16, 64)` is
the second point on the chunk axis.

## <a id="gates"></a>What gates it

`tests/oracle/test_ladder_r0.py`, against two references and neither alone:

* the **fp64 autograd oracle** — the eleven-line recurrence in float64, differentiated
  by autograd. It carries no chunking and no rounding, so it states WHAT the kernel
  computes and cannot be satisfied by a kernel that is merely self-consistent with its
  own restatement;
* the **rounded reference** — the same algebra chunked as the kernel chunks it, with
  every stored operand rounded where the kernel rounds it, which is what makes the
  budget tight. The teeth are explicit: the budget must REJECT a reference with the
  intra staircase dropped and one with the staircase made exclusive.

The census (`chunk_rung0_census`) carries both halves of the floor claim per arm and
per pass — the declared live set and the budget its CTA target admits — and the gate
asserts zero spill, a staging depth of at least two, and the forward's own residency.
