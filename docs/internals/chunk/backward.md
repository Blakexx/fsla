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
| the readout `dv = W · dS_a` | (A.17) |
| the CHANNEL CONTRACTION `⟨dS_a[s,:], v[j,:]⟩ + dz_a[s]`, reduced over slots | (A.18), inter-chunk part |

Not carried, and named so that nothing reads as an omission: the Gram `Gd`, the
transposed staircase (A.19), the leave-one-out digit reduction (A.9)/(A.10), and
`d gw`'s own weighting (A.11). What the kernel emits today is the INTER-chunk part
of the write side plus `dv`; the intra-chunk terms and the digit reduction are the
next stages'. `rola.ops.chunk.backward` refuses by name until they land.

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
