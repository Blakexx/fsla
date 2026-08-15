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

This component is at **stage one** of the backward's build and carries exactly the
terms whose residency and whose new primitive had to be measured before anything
else was written:

| carried | the design's term |
|---|---|
| the descending walk and both operand panels | §C.2 |
| the fold `dS += dnum^T · R`, with its mass column `dz += dden^T · R` | (A.16) |
| the readout `dv = W · dS_a` | (A.17), inter |
| the SLOT GRAM `G = staircase(R · W^T)` | the forward's, see [two Grams](#two-grams) |
| `dv`'s intra term `dv^T += dnum^T · G` — so **`dv` is complete** | (A.17), intra |
| the CHANNEL CONTRACTION `⟨dS_a[s,:], v[j,:]⟩ + dz_a[s]`, reduced over slots | (A.18), inter-chunk part |

Not carried, and named so that nothing reads as an omission: the `d_v` Gram `Gd`,
the transposed staircase (A.19), the per-slot `dW`, the leave-one-out digit
reduction (A.9)/(A.10), and `d gw`'s own weighting (A.11). So `d gw` is the
inter-chunk part alone and `d pr` / `d pw` are not emitted;
`rola.ops.chunk.backward` refuses by name until they land.

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
per-SLOT quantity and this arm has exactly as many slots per group as threads
(`BCg == kGroupThreads`, asserted), so it is a **scalar** and its fold is a dot
product of the chunk's staged `dden` row against this thread's column of the `R`
panel. Fifteen registers back, and `dz` lands in the layout the digit reduction
wants rather than one it would have to be transposed out of.

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
stage-1 numeric rows dodge this by making every token fully live or fully dead on a
side, which makes every block's partition the same one — a real cell, and not the
general case.

`walk_probe_kernel` runs the walk **alone** and emits what it published: per
`(bh, block, token)`, the ordinal of the chunk that gathered the token and its
position in that chunk's list, one plane per side, plus the causal threshold each
read entry carries. The sentinel is `-1`, so *"gathered exactly once"* is a
checkable claim rather than a covering one. It takes the union table and nothing
else, because the walk reads no amplitude — a probe that took one would be gating a
wider surface than the schedule it is named for.

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

## <a id="ctx-contract"></a>The ctx the forward hands over

`rola.ops.chunk.SavedForward`. Everything in it is `O(L)` with a per-topology
constant: the packed read and write sides, the write gain, the read mass, the
denominator, the union table, the liveness bitmap when the launch used one, and the
plan's scalars. The state, the per-chunk compaction lists, the causal thresholds
and the numerator are absent by construction — the backward rebuilds the first and
replays the rest.

The union table is the ONE byproduct inherited by save rather than by replay:
rebuilding it is a second launch over the whole amplitude plane for a bit-identical
answer, and it is the seam the schedule is replayed from. `entries` is a SUM over
levels, never a product, so the table does not grow with `N`.

The ctx is taken on `needs_input_grad`, so an inference call retains nothing.
`tests/integration/test_backward_ctx.py` asserts the retained set equals the closed
form exactly — not that it grows slowly, which a regression could hide in.

## <a id="arms"></a>The arm list, and why it is one

`SCAN_B_ARMS_X` carries the pinned production arm alone. The forward's arm cross is
not owed until the scan carries every term, and it is not free: this translation
unit already peaks at 2.75 GB per `cicc` at ONE arm, which is decode-class, so
growing it is a re-measurement of `tools/build_flags.py`'s job formula and not a
free axis. It is a translation unit of its own for the same reason `factor.cu` is.
