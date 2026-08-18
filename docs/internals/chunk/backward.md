# `csrc/rola/src/chunk/backward.{cuh,cu}` — Scan B, the reverse adjoint

The chunk consumer's backward is a **uniform two-pass**: Scan A rebuilds the
forward's state from zero and emits the read-side cotangents; Scan B walks the
token axis backwards and emits the write-side ones. This component is Scan B.

Symbols, defined once and used unqualified below: `C` chunk tokens, `BC` slots per
CTA block, `BCg` a slot group's slice (`= BC` at one group), `D` levels, `W` one
uniform level width, `N = W^D` leaves, `d_v = 64` value channels, `T`/`L` tokens,
`s` a slot, `i` a read entry, `j` a write entry, `R`/`W` the read/write Kronecker
coefficients, `dS` the adjoint state `[N, d_v]`, `dz` its mass column,
`dnum`/`dden` the two readout seeds, `dS_a` the adjoint a chunk ENTERS with — the
one arriving from the future.

## <a id="stage-1-scope"></a>What is built here, and what is not

This component is Scan B, and it carries every write-side term. The order below is
the order they were built in, which is the order their residency and their new
primitive had to be measured in:

| carried | the design's term |
|---|---|
| the descending walk and both operand panels | §C.2 |
| the fold `dS += dnum^T · R`, with its mass column `dz += dden^T · R` | (A.16) |
| the readout `dv = W · dS_a` | (A.17), inter |
| the SLOT GRAM `G = staircase(R · W^T)` | the forward's, see [two Grams](#two-grams) |
| `dv`'s intra term `dv^T += dnum^T · G` — so **`dv` is complete** | (A.17), intra |
| the CHANNEL CONTRACTION `⟨dS_a[s,:], v[j,:]⟩ + dz_a[s]`, reduced over slots — **`sum_s dW`, not `d gw`** | (A.18), inter-chunk part |
| **`d gw` COMPLETE**, inter and intra | (A.11) |
| the `d_v` Gram `Gd'` with the R-b ones column | (A.19), repaired |
| the PER-SLOT `dW`, one pass of the slots at a time | (A.18)+(A.19) |
| **`d pw` COMPLETE** — the write side's leave-one-out digit reduction | (A.10) |

Not carried, and named so that nothing reads as an omission: `d pr` (A.9), which
needs the FORWARD state and is therefore [Scan A](#scan-a)'s — Scan B owns `dS`,
Scan A owns `S`, and neither owns both.

**So Scan B is whole**: every gradient the reverse pass owes it — `dv`, `d gw`,
`d pw` — is emitted and gated, at every arm in [the list](#arms).

## <a id="digit-reduction"></a>The write side's digit reduction is in lane

(A.10) weights each slot by the leave-one-out product `prod_{l' != l} pw_{l'}`,
never by `W / pw_l` — under exact entmax `pw_l` is exactly zero off the support and
so is `W`, so the division is `0/0` with no epsilon that does not invent a
gradient. At depth two the leave-one-out factor is simply the OTHER level's
amplitude, and the two digit gradients differ ONLY in that choice, which is what
the gate's non-vacuity row exchanges.

The slots are walked ONE PASS at a time, and a pass is chosen so the digit
structure falls out of the geometry rather than being computed. Every
`(token, inner digit)` cell of a pass is held by **exactly one lane**, so the
inner level's gradient needs no reduction at all — the fan-in is `C * B` adds per
block, the minimum the destination admits — and the outer level's sums over the
lane's slot run (two shuffles) and then over the four units, which the fan-in
does. The law that makes both true at every arm is [the pass law](#pass-law).

**The pass index is a compile-time constant and that is load-bearing.** `u0` and
`u1` are indexed by the pass' own digit class; a runtime index into a register
array is a stack frame, and leaving the loop rolled cost exactly that — 352 bytes
of frame at zero spill, which is local memory rather than register pressure.
Unrolling it returned both arches to a zero-byte frame.

## <a id="pass-law"></a>The pass law

A pass' width is **structural, not `BC`-derived**. Each unit consumes an eight-row
`d_v` tile of the transposed adjoint state through one `ldmatrix`, and there are
`kUnits = 4` of them, so

    kPassSlots = kUnits * kMmaN = 32          slots per pass
    P          = BCg / kPassSlots             passes per block
    unit run   = slots [32p + 8g, 32p + 8g + 8),   p compile-time, g = the unit

`BCg / 4 == 32` **only at `BC = 128`**, which is the whole of the retired "quarter"
language: at `BCg = 64` a pass is half the block, at 32 it is the block.

Slots inside a block are ordered outer-major (`slot = od * kInnerSpan + i`), and
`kInnerSpan` is a multiple of eight at every legal arm (a level's span is either
`B >= 8` or the whole block). A unit's run is eight-aligned, so it lies inside ONE
outer digit — which is what lets the outer amplitude be a scalar over the pass'
accumulator — and covers eight CONSECUTIVE inner digits, one per lane pair.

The pass index then factors, per unit, into an inner-digit class and an outer
digit:

    CLS = min(P, max(1, kInnerSpan / 32))     distinct classes a unit visits
    ODU = P / CLS                             distinct outer digits a unit visits
    cls = p % CLS      (compile-time: the register index)
    od  = (32p + 8g) / kInnerSpan             (runtime: a fan-in COLUMN only)
    i1  = (32*cls + 8g) % kInnerSpan + (lane & 3) * 2

`od` may be runtime because it is never a register index; `cls` is compile-time
because `p` is, which is what keeps `u0`/`u1` out of a stack frame.

**A digit array is carried across passes only where its index recurs.** A class
recurs `ODU` times and an outer digit `CLS` times, so

    u1 : (ODU > 1 ? CLS : 1) * MTOK * 4 registers
    u0 : (CLS > 1 ? ODU : 1) * MTOK * 2 registers   (absent entirely at D = 1)

and where the carry is absent the fan-in issues inside the pass loop, with the same
total atomic count. The arms as built:

| arm | BCg | inner | outer | P | CLS | ODU | u1 | u0 | `pw0` bytes |
|---|---|---|---|---|---|---|---|---|---|
| (32,128,2,64) pinned | 128 | 64 | 2 | 4 | 2 | 2 | 16 | 8 | 256 |
| (32,128,2,256) scale | 128 | 128 | 1 | 4 | 4 | 1 | 8 | 4 | 128 |
| (32,64,2,8) | 64 | 8 | 8 | 2 | 1 | 2 | 8 | 4 | 1,024 |
| (32,64,2,16) | 64 | 16 | 4 | 2 | 1 | 2 | 8 | 4 | 512 |
| (32,64,1,64) | 64 | 64 | 1 | 2 | 2 | — | 8 | 0 | 0 |
| (32,64,1,256) | 64 | 64 | 1 | 2 | 2 | — | 8 | 0 | 0 |

The pinned row is the construction's self-check: `P=4, CLS=2, ODU=2` is the
retired quarter carve exactly, which is why the generalization could be gated on
the pinned arm's numbers not moving.

**The fan-in column carries `off_at`.** A level the block spans WHOLE contributes
no block offset, so `col_base(D-1) + i1` was right at the pinned arm and is wrong
wherever the innermost level is spanned partially (`(32,64,1,256)`,
`(32,128,2,256)`): four blocks would fan into one block's columns. The column is
`col_base(D-1) + off_at(D-1, blk) + i1`, which is the same spelling the coefficient
production already used.

**Depth one deletes the outer level; it does not weight it by one.** At `D = 1`
level zero IS the innermost, `kOuterSpan = 1`, and `spw0` would otherwise read the
inner amplitudes back as an outer factor — a silent squaring. `u0`, `spw0` and the
outer fan-in do not exist there, and the inner level's leave-one-out product is
empty. `GeomB` takes `(D, B)` for exactly this: `pw0_bytes` is a digit-shaped
buffer, and both its SIZE and its place in the carve are the topology's.

## <a id="no-reversed-table"></a>Why the walk needs no reversed table

The chunk partition is an **association of a sum, not a semantic object**: the
adjoint recurrences are exact for any ordered partition of `[0, T)`, and the causal
threshold `J(i)` is defined per chunk from that chunk's own lists. So Scan B does
not traverse the forward's partition backwards — it walks its **own**, applying the
forward's closing rule from the top of the token axis downward. The alternative,
recording each super-chunk's start token, is `[BH, blocks, ~T/C]` integers, which
is the `O(N·L)` class the ctx contract forbids.

Three consequences, and they are the whole of the difference from the forward's
walk:

* the formed ring fills **downward** — `formLow` is the lowest word formed, and
  refilling lowers it; the tail mask lands on the FIRST word the walk touches
  rather than the last;
* the walk consumes each word from its **high** lane, so the vote is read with
  `last_set` where the forward reads `first_set`, and the kept mask is the lanes
  strictly above the cut rather than strictly below it;
* entries are written at **descending buffer positions** and the lists therefore
  come out ASCENDING in token order, which is what keeps produce, the panels, the
  fold and the drain byte-for-byte the forward's code. At close the run is shifted
  down from `[C - n, C)` to `[0, n)`.

The threshold is resolved at the same moment. Walking down, what a read entry can
observe is the count of write entries **above** it; `J(i)`, the count at-or-before
it, is `nw - above` with the chunk's own total `nw`, known at close.

Scan A and Scan B associate their sums differently by construction, so the gate on
this operator is ratio-to-floor and never bit-identity.

## <a id="scan-a"></a>Scan A — the forward rebuild, and the read side

Scan B walks DOWN carrying the adjoint `dS`; Scan A walks UP carrying the forward
state `S`, rebuilt from zero over the SAVED union table, and emits the one gradient
that needs `S` rather than `dS`:

    dR[i,s]     = <dnum[i,:], S[i,s,:]> + dden[i] * z[i,s]                       (A.8)
    d pr_l[t,d] = sum_{s: d_l(s)=d} dR[t,s] * prod_{l' != l} pr_{l'}[t,d_{l'}(s)]  (A.9)

with NO gain anywhere: `g_read` does not exist, so (A.9) carries no gain factor and
there is no `d g_read` output.

**THE GRAM IS THE ONE SCAN B ALREADY FORMS.** Expanding the INCLUSIVE `S`/`z` over
the chunk gives

    dR[i,s] = <dnum[i,:], S_a[s,:]> + dden[i]*z_a[s]
            + sum_j [j <= i] * W[j,s] * Gd'[i,j]

with the same `Gd'`, ones column and all — the design's (A.14) and (A.15) collapsed
into ONE masked contraction. So the correction, the buffer and the code path are
inherited rather than rebuilt. Verified exact over rationals before it was written.

Structurally it is Scan B with the two sides exchanged:

| | Scan B | Scan A |
|---|---|---|
| schedule | `Descent` | `Ascent` |
| state | `dS^T` + `dz` per slot | `S^T` + `z` per slot |
| write panel | UNGAINED (the gain lands at the store) | **GAINED** — the fold deposits `W` and the intra contracts against it |
| Gram | `Gu` then `Gd'` | `Gd'` alone |
| per-slot | `dW`: `m` = write token | `dR`: `m` = READ token |
| intra | `A = Gd'^T` (T map) | `A = Gd'` (F map, NOT transposed) |
| fold | `dS += dnum^T·R` | `S += v^T·W` — the forward's own |
| digit reduction | write side, gained | read side, no gain |

It reuses `GeomB` unchanged: it needs strictly less of the readout window (no `dv`
staging), and sharing the geometry keeps both halves of a training step at one
occupancy. 236 registers, zero spill, zero stack frame.

### <a id="addressing"></a>The addressing rule this scan paid to learn

One rule, four sites, and the gate that enforces it is the hot-loop clause: no
`IMAD.WIDE` inside the chunk loop. A global address is a 64-bit base formed ONCE
outside the loop, pinned, plus a 32-bit byte displacement formed inside it. What
breaks the rule is anything that lets ptxas fold a loop-varying term into the
64-bit add, and it breaks in four distinguishable ways:

* **each side's table row is formed FROM THE TABLE, not from the other side.**
  Writing the write row as `read + entries * tok_words` makes its offset a
  CONSTANT times a register, which ptxas re-derives inside the loop rather than
  keeping. Formed independently the multiply is register times register and it
  stays put. Pinning does not substitute: `pin_address` launders a value, it does
  not make rematerialising it more expensive than keeping it;
* **the head's offset rides the displacement, not the base.** `V` is
  `[B, T, H, d_v]`, so a head is a fixed byte offset INSIDE a row. Folded into the
  gather's displacement, the base carries the batch term alone and the 64-bit
  multiply the head term would otherwise contribute is gone;
* **the gathered `dnum` row is pinned at the gather.** `dnum`'s row stride is a
  compile-time constant, so `token * stride` is exactly the shape ptxas folds into
  the 64-bit add — once per staged row. `V`'s gather needs no such pin: its stride
  is a runtime argument and there is nothing to fold;
* **the digit-gradient plane is a `char*`** on the stride its displacements already
  carry, so the fan-in's address is one 32-bit add off a pinned base rather than a
  typed-pointer scale.

The same rule is why the partition probe forms its three output plane bases once
and pins them: every store inside the walk is a byte displacement off one of them.

## <a id="readout-window"></a>The readout window is two things in sequence

`so` — the fp32 plane `dv`'s partial is staged into before the fan-in — is DEAD
from `drain_dv` to the end of the chunk. `d gw`'s token-major reduction needs the
entry mass column `dz` published per slot, and that is `BCg` floats against the
window's `C · LdO`, so it costs **nothing**: the two phases are separated by a
barrier and share the bytes. `GeomB::win_bytes` is the max of the two and a
`static_assert` holds it there, so a future overlay cannot silently outgrow it.

That is also the room the per-slot `dW` will want. Its adjoint state has to arrive
transposed — `[slot][channel]` rather than the accumulator's `[channel][slot]` —
because (A.18)'s `k` axis is `d_v`, which is the carve's own axis, so no unit can
finish the contraction from its own registers. Writing each unit's 16 channels into
a shared `[slots][d_v]` plane makes the writes DISJOINT (no cross-unit sum at all)
and lets every unit then read all 64 channels for one pass of the slots. Carved a
pass at a time — `[kPassSlots + kMmaN][d_v + kPadE]` bf16 = 5,760 B, a CONSTANT,
since the pass width is the mma carve's and not `BC`'s — it fits inside this window
beside `dz` with room over at every arm, so the whole per-slot `dW` phase costs no
shared memory beyond what `dv` already owns.

The `+ kMmaN` on the ROW count is not slack. `ldmatrix` reads SIXTEEN rows from an
eight-row tile's base, so the last unit of a pass addresses eight rows past the
pass' own slots; without them the transposed state's read runs off the buffer. The
`+ kPadE` on the row STRIDE is the tree's usual bank skew and is a different thing.

## <a id="two-grams"></a>Two Grams, one buffer

`dv[j]` reads the adjoint state the write token `j` **leaves the chunk with** —
the entry state plus every read token above `j` — so its intra term factors over
slots as `sum_i [j < J(i)] G[i,j] dnum[i,:]` with `G = R . W^T`. That is the
**forward's own Gram**, contracted over slots, with the forward's staircase
unchanged.

`dW[j,s]` factors the other way: `sum_i [j < J(i)] R[i,s] Gd[i,j]` with
`Gd = dnum . v^T`, contracted over **`d_v`**. So Scan B needs BOTH Grams — the
design's cost table names only `Gd`, which is right for Scan A and one Gram short
for Scan B.

They cannot both be resident. Measured with the tree's own constants
(`design::cta_target`, sm_86):

| carve | bytes | CTA target |
|---|---|---|
| no Gram | 46,624 | 2 |
| one Gram (`C * LdG` bf16 = 2,560 B) | 49,184 | 2 |
| two Grams | 51,744 | **1** |

The arm's occupancy is not negotiable, so the two share one buffer across disjoint
phases: the slot Gram is formed, consumed by `dv`, and dead before the `d_v` Gram
is formed. (An earlier ruling aliased the Gram into the `so` staging window on the
strength of a 4,608-byte estimate; that figure used the V panel's row stride
(`kD + kPadE`) where the Gram's is `C + kPadE`, and with the real number no alias
into `so` is needed at all.)

## <a id="dw-slotsum"></a>The second output is `sum_s dW`, and that is NOT `d gw`

The probe returns `dv` and a per-token plane that the kernel used to call `dgw`.
It is not `d gw`. Writing `W[t,s] = gw[t] · U[t,s]` with `U` the UNGAINED level
product, (A.11) is

    d gw[t] = sum_s dW[t,s] · U[t,s]

and what the kernel emits is `sum_s dW[t,s]` with no `U` weighting — d `gw`'s
SHAPE, and not its value. The two coincide only when `U ≡ 1`, which is exactly the
configuration every ladder gate has ever run. Checked over exact rationals against
`d/d gw` of the recurrence's loss: the `U`-weighted sum reproduces it, the
unweighted one does not.

The output is therefore named `dwsum` throughout, and the probe's docstring says
so. Nothing downstream may bind it to `gw`'s cotangent. It is kept because it is
what holds (A.18)'s CHANNEL CONTRACTION — the one new primitive, and the only
`MOVM` in this kernel — live and censused until the per-slot `dW` takes it over.

**`d gw` ITSELF IS NOW EMITTED**, beside it, and this is how. `U` is per write
token and per slot, so the weighting cannot be applied after the slot sum. The
clean form is the `(d_v + 1)`-th column identity: with
`ṽ[j,:] = [v[j,:], 1]` and `d̃S = [dS, dz]`, `dW[j,s] = ⟨d̃S_after[j,s,:], ṽ[j,:]⟩`,
so

    d gw[j] = ⟨ sum_s U[j,s] · d̃S_after[j,s,:] , ṽ[j,:] ⟩

— the UNGAINED `dv` readout widened by one column, contracted against `v` with a
trailing one. Two consequences, and both are what the kernel does:

* **the write panel is produced UNGAINED and the gain lands once, at the `so`
  store**, on the `n` axis — `gw[j]` is uniform along the accumulator's token
  index, so one multiply per stored entry is the same number. `dv` is unaffected
  (`sum_s W · dS = gw[j] · sum_s U · dS`), the slot Gram is therefore the UNGAINED
  `Gu` which is what `d gw`'s intra term wants, and the level product is left
  standing — the operand (A.11) needs and the one a decay clock will tap;
* the value channels then need no separate contraction at all: `acc` — the
  ungained readout, inter and intra together — IS `sum_s U[j,s] dS_after[j,s,:]`,
  so `d gw`'s value half is `⟨acc[j,:], v[j,:]⟩`, read straight off the
  accumulator before it is stored.

So `d gw` is three partials that all land in `sgrad` and meet at one fan-in:

| term | what it is | who computes it |
|---|---|---|
| A | `⟨acc[j,:], v[j,:]⟩` | per UNIT, over its own 16 channels |
| B | `sum_s U[j,s] · dz_a[s]` | per QUARTER of the slots |
| H | `sum_i [j<J(i)] · dden[i] · Gu[i,j]` | per QUARTER of the read entries |

and the fan-in sums four of each — which is why A is STORED once per unit (its
four partials are the four channel groups) while B and H are ADDED once (their
four partials are one token's whole sum). The Gram is stored already masked, so H
does not re-apply the staircase.

`Gd'` does not appear: the `⟨dnum,v⟩` half of it is already inside `acc` through
`dv`'s intra term, and only the `dden` column of `Gd'` is left over — which is H.
Building `Gd'` explicitly is the PER-SLOT `dW`'s business, not `d gw`'s.

## <a id="mass-gram"></a>The transposed staircase carries the mass channel too

The design's (A.19) writes the write-side adjoint as

    dW[j,s] = <dS_a[s,:], v[j,:]> + dz_a[s] + sum_i [j < J(i)] Gd[i,j] R[i,s]

with `Gd[i,j] = <dnum[i,:], v[j,:]>`. **That is wrong**, and the error is a whole
term rather than an index: `dW` differentiates BOTH uses of `W[j,s]`, the state and
the mass column, so the adjoint entering `j` is
`dz_after[j,s] = dz_a[s] + sum_i [j < J(i)] R[i,s] dden[i]` — and only the
inter-chunk half `dz_a[s]` survives above. Differentiating the recurrence directly
gives the exact statement

    dW[j,s] = sum_{t >= j} R[t,s] ( <dnum[t,:], v[j,:]> + dden[t] )

which is the same expression with a **ones column appended to the write side**:

    Gd'[i,j] = <dnum[i,:], v[j,:]> + dden[i]

so ONE transposed staircase covers value and mass channels together and no second
pass exists. In this kernel that is not even an extra mma: `Gd'` is `Gd` plus a
row-broadcast of `dden[i]`, and `dden` is already staged per read entry in `sdd`
for the [mass column](#mass-column)'s own fold, so it is one add per Gram entry.

Verified two ways before it was written down: symbolically over exact rationals
(the design's addendum), and independently here by differentiating the loss
`sum_t (<dnum_t, num_t> + dden_t den_t)` against the chunked form on a two-chunk
instance — 15 of 18 entries wrong as written, 0 of 18 with the ones column.

`dv` is untouched by this: it is a value-channel quantity and the mass column does
not enter it, so the term this component ships today is unaffected. The correction
binds the stage that builds `Gd`, the per-slot `dW` and the digit reduction.

## <a id="mass-column"></a>The adjoint mass column is one slot per thread

The forward rides its mass channel on the value mma: the `d_v + 1`-th channel
occupies row 0 of a 16-row `m` quantum and rows 1..15 are structural zeros. That
costs `kMassTiles * 4 = 16` registers per lane to carry one slot's worth of
information, and it is free in the forward because the mma is already issued.

Scan B cannot afford it beside the adjoint state, and does not need it. `dz` is a
per-SLOT quantity, so it is a **scalar** whose fold is a dot product of the chunk's
staged `dden` row against its slot's column of the `R` panel. Fifteen registers
back, and `dz` lands in the layout the digit reduction wants rather than one it
would have to be transposed out of.

**`T = kGroupThreads / BCg` threads share a slot** (asserted: a slot's threads must
be a whole lane group). Thread `(slot, part)` folds the `C / T` read entries of its
segment and the segment totals meet in one `__shfl_xor` butterfly of `log2 T` steps
over ADJACENT lanes — intra-warp by construction, because `slot = gtid / T` puts a
slot's threads in one lane quad. `T = 1` is the degenerate case and reproduces the
one-slot-per-thread form exactly.

The butterfly **collapses** rather than broadcasts: part zero keeps the total and
the others are zeroed. `dz` is an accumulator carried across chunks, so a broadcast
total would be re-summed `T` times at the next fold; with the collapse the
invariant "a slot's value is the sum over its `T` threads" holds at every chunk
boundary, and the warp-wide `dzs` reduction that feeds `d gw` needs no predicate at
all — the four warps still partition `BCg` exactly.

That was not a preference. With the slot Gram and `dv`'s intra term added, the
mma-carried mass column put sm_80's allocator four bytes over its bound; every
smaller perturbation left it there, and this is the change that returned both
arches to zero spill at 255 registers.

## <a id="seeds"></a>The seeds, and the fifth output

`backward_seeds` is the mirror of `divide_pass` and the ONE place the eps floor is
differentiated. With `Dt = den + eps * m` — the exact quantity the divide forms —
the readout `y = num / Dt` gives

    dnum[t,:] = dy[t,:] / Dt
    dden[t]   = -<dy[t,:], y[t,:]> / Dt
    dm[t]     = eps * dden[t]

**The floor is ADDITIVE, not a `max`** (`naive.py:223`, `chunk.cu:203`), so there
is no kink, no subgradient and no `den`-dependent branch anywhere in the backward.
`den >= 0` always — `R >= 0` on a simplex, `W >= 0` with `gw > 0` — so `Dt` is
bounded below by `eps * m > 0` and the readout is smooth in every input.

`dden` uses the **stable form**: `y` is the op's own output, so the denominator is
divided by once instead of squared. That is why `y` is on the ctx at all.

**`dm` IS A FIFTH OUTPUT AND IT IS NOT ZERO.** The m-carry — the producer leaving a
per-token read mass on the read levels instead of dividing it out, with the floor
scaled to match — is forward-exact but **not gradient-neutral**: `m` reaches `y`
only through `Dt`, so `dy/dm = -y * eps / Dt` and the cotangent contracts to
`eps * dden`, one FMA per token. A backward that dropped it would silently zero a
real gradient of every producer that does not declare a simplex. Had the floor been
`max(den, eps)` this same term would have been a subgradient with a `den < eps`
branch; it is not.

`dm` comes back only when a read mass does. There is no `m` to differentiate in a
declared-simplex bundle, and returning a zero for one would be a claim rather than
an absence.

**Dtypes are not a convention here, they are the consumers'.** `dnum` and `dden`
are both mma operands (the fold's `A` operand and the mass column's `A` row), so
both are emitted bf16 — the same class of decision the forward already makes when
it packs its fp32 state into bf16 fragments. `dm` is fp32 for the reason
`read_mass` is: bf16 would quantize the floor itself. And `dm` is taken off the
**fp32** `dden`, not off the bf16 one it was just rounded to.

## <a id="partition-probe"></a>The partition probe — the schedule gated as a schedule

The walk above is the one piece of P65 with no forward counterpart to inherit, and
its correctness is a **schedule** fact. An arithmetic gate can only reach a schedule
through a reference that replays the partition, and that reference stops being
writable the moment two blocks of one head are live on different tokens: each block
then closes its chunks at different tokens, so "the" partition does not exist. The
numeric batteries dodge this by making every token fully live or fully dead on a
side, which makes every block's partition the same one — a real cell, and not the
general case. Where the plan has NO table rows (a block spanning the whole slot
space, e.g. `(32,64,2,8)`), the clause is empty and every token is a member: the
partition is the dense one however sparse the amplitudes are, and the batteries
read `entries` off the census to know which case they are in.

`walk_probe_kernel` runs the walk **alone** and emits what it published: per
`(bh, block, token)`, the ordinal of the chunk that gathered the token and its
position in that chunk's list, one plane per side, plus the causal threshold each
read entry carries. The sentinel is `-1`, so *"gathered exactly once"* is a
checkable claim rather than a covering one. It takes the union table and nothing
else — plus the head batch, which the table cannot carry when the plan builds no
rows at all — because the walk reads no amplitude: a probe that took one would be
gating a wider surface than the schedule it is named for.

The walk itself is a **component**, `Descent`, not a closure inside Scan B: a probe
that re-implemented the closing rule would be gating its own copy. Both kernels
drive the same object, which is what makes the probe's verdict Scan B's verdict.

## <a id="channel-contraction"></a>The one new primitive

Exactly one arithmetic primitive is new in the whole backward: a contraction whose
`k` axis is `d_v` rather than slots or tokens. `dv` (A.17) is the forward's readout
with the write panel; the fold is the forward's fold with `dnum` in `V`'s place and
the read panel in the write panel's; the drains are the forward's drains gathered
by the other list. Only `dW[j,s] = ⟨dS[s,:], v[j,:]⟩` contracts over channels.

Its operands: `A = V` (`m = write token`, `k = d_v`), straight out of the value
panel with a plain `ldmatrix`; `B = dS` (`k = d_v`, `n = slots`), which is the fold
accumulator through one register transpose per 8x8 tile
([`ops.md#reg-transpose`](ops.md#reg-transpose)).

**`d_v` is the carve's own axis**, so this contraction is a CROSS-UNIT partial sum
by construction: a unit owns 16 of the 64 channels and can only contract those.
That is why Scan B has a reduction window the forward has no term for — one row per
MMA unit, `C` columns — and why the sum over units happens exactly where the
forward's denominator drain happens.

## <a id="residency"></a>The register and shared arithmetic (the stage-1 probe)

Measured at the pinned arm `C=32, BC=128, D=2, B=64` on sm_86, nvcc 12.4:

| quantity | value |
|---|---|
| adjoint state `dst[NDV=16][4]` + `dz_st[4][4]` | 80 registers |
| registers used, tiled accumulator | **255**, **0 bytes spill** |
| registers used, whole `C x BCg` accumulator held in registers | 229, 0 bytes spill |
| register DEMAND (bound: forced to a 3-CTA bound, 168 available, 36 bytes spill) | ~177 |
| shared bytes | 46,624 |
| staging depth | 2 |
| CTA target | **2 — the forward's, unmoved** |

The shared arithmetic that decides the carve: holding `kCtaTarget = 2` requires
`round_up(bytes, 128) + 1024 <= smem_per_sm / 2`, i.e. **46,624 <= 50,176**. A
`C x BCg` fp32 plane in shared memory would be 16,384 bytes on top of a carve that
has 12,256 to give, so the accumulator can be register-resident or tiled, but it
cannot be a shared plane at this occupancy. The kernel tiles it and reduces in
place: the tile is produced, reduced over its slots and folded into the per-token
partial, so the live accumulator is 8 registers rather than 128.

**NO OPERAND k-RUN IS HOISTED, anywhere in Scan B, and that is a decision the
forward never has to make.** The forward's mma loops may keep a whole k-run of
fragments live because nothing else of theirs is; Scan B carries the adjoint state
`dst[NDV][4]` through every one of its blocks, so `KD * 4` more live registers is
the difference between 255-with-no-spill and a spill. Three sites pay it and none
of them pays anything for it: the slot Gram re-reads one k-tile of each panel per
step (`MPW` is one at every arm, so nothing is read twice), `dv`'s readout re-packs
the adjoint and re-issues the `ldmatrix` the loop was going to issue anyway, and
the fold re-reads the `R` panel a tile at a time for the same `ldmatrix` count.
The rule to read off this: at this ceiling a live range that merely OUTLIVES its
use is a spill, which is why `dv`'s whole term is written inside its own scope.

## <a id="ctx-contract"></a>The ctx the forward hands over

`rola.engine.plan.BackwardPlan` — the forward's `ChunkPlan` held BY REFERENCE, plus
`den`. The reverse pass finds no facts of its own, so there is no second object to
keep in step with the plan: what the forward configured is what the backward reads
(docs/internals/engine/backward_dag.md).

What it replays from is `O(L)` with a per-topology constant: the packed read and
write sides, the write gain, the read mass, the denominator and the union table.
The state, the per-chunk compaction lists, the causal thresholds and the numerator
are absent by construction — the backward rebuilds the first and replays the rest.

The union table is the ONE byproduct inherited by save rather than by replay:
rebuilding it is a second launch over the whole amplitude plane for a bit-identical
answer, and it is the seam the schedule is replayed from. `entries` is a SUM over
levels, never a product, so the table does not grow with `N`.

`retained_bytes()` counts that replay set, which is what a grad-bearing call costs
over an inference one. The plan reference also pins the forward's `block_bits` —
`[BH, N / BC]` BYTES, carrying no `T` at all, 64 of them at the pinned arm — and
`v`, which is the op's own operand and was pinned before this too. Both are named in
`tests/integration/test_backward_ctx.py`; neither is folded into the closed form,
because a term with no `L` in it would make that a number no closed form matches.

The ctx is taken on `needs_input_grad`, so an inference call retains nothing.
`tests/integration/test_backward_ctx.py` asserts the retained set equals the closed
form exactly — not that it grows slowly, which a regression could hide in — and
asserts it PER FIELD, walked off the plans, so a field arriving on the plan for the
forward's sake cannot be retained silently.

## <a id="reg-floor-scan"></a>The scans declare their own register floor

`reg_floor` is the FORWARD's measured live set and falls steeply with `BC`: 236 at
`BC = 128`, 168 at 64, 136 at 32. The scans' does not. Halving `BC` halves the
adjoint state (`dst[NDV][4]`, 32 registers) and nothing else — every other term is
`C`-shaped — so the reverse pass falls about forty registers where the forward
falls sixty-eight. Declaring one floor for both would put the `BC = 64` backward on
a 168-register budget with a 210-219 live set, which is a SPILL rather than a
refusal: the build would go green and only the census would notice.

So `reg_floor_scan(BC)` is a second declaration — 200 / 224 / 236 at `BC` 32 / 64 /
128 — and the parity clause is restated as `kScanBCtaTarget <= kCtaTarget`: the
scans may run at their forward's occupancy or below it, never above. The intent the
equality carried (nobody moves an arm's residency without deciding to) is enforced
by the CENSUS instead, which carries both the declared floor and the budget that
target admits, per arm, and whose gate asserts the measured live set fits inside
it. The pair is what a review reads.

## <a id="arms"></a>The arm list

`SCAN_B_ARMS_X` is GENERATED, from `tools/gen_shards.py`'s `SCAN_ARMS` — the same
generator that owns the forward's `CHUNK_ARMS`, which is what makes "every scan arm
has a forward arm" a check at generation time rather than a hope. The scans' list is
the narrower one: it carries the campaign's TRAINABLE topologies, and
`rola.engine.rules.envelope.backward_envelope` refuses the difference at the facade rather
than three launches later.

It is not free to widen. This translation unit peaks at 2.75 GB per `cicc`, which is
decode-class, so an arm added here is a re-measurement of `tools/build_flags.py`'s
job formula. It is a translation unit of its own for the same reason `factor.cu` is.
