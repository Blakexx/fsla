# `backward_kernel.cuh` — the carry's two reverse scans

The mirror of `csrc/rola/src/carry/backward_kernel.cuh`. Two `__global__` kernels and
the owner-level bodies they share: Scan A, the ascending replay that emits `d_pr`, and
Scan B, the descending adjoint that emits `d_v`, `d_gw`, `d_pw` and `d_state_in`.

Symbols, each at first use: `D` levels, `B` digits per level, `BC` the leaves one owner
holds, `W` the global window length in tokens, `C` the private batch's token tile,
`d_v = 64` value channels, `cols = d_v + 1`, `s_l` an owner's span at level `l`, `r_l` a
leaf's run offset inside its owner at level `l`.

## Why the state is replayed and not stored

The state is `O(N·cols)` per head and the window count is `T/W`, so a snapshot at any
granularity is `O(N·T/W·cols)` — 33 GB per head at the flagship. With no decay the
recurrence is affine in the state, so the state at a window boundary is the SUM of the
deposits below it: replaying it forward costs one pass over deposits the forward already
read and nothing else, and the window grid's global edges are already the forward's own
fold boundaries. The same argument is applied to the SCHEDULE: neither scan inherits a
list, a threshold or a bucket partition from the forward — both rebuild the two packed
lists from the facts table, in the forward's order, at the forward's cost.

## Why there are two kernels and not one

A fused form holds two fp32 slabs in one thread's registers. At the flagship the slab is
80 registers per thread on the measured forward, so two slabs plus a working set that
cannot be smaller than the forward's 96 exceeds the hard 255-register cap before any of
the backward's own operands are counted. No term couples the replayed state to the
adjoint — the read-side gradients read the state only, the write-side ones read the
adjoint only — so the two scans are data-independent given the seeds. They are launched
concurrently on two streams: two slabs are resident in the MACHINE, and neither thread
holds two.

## Why the slab is leaf-major, and the carve with it

The forward holds the state channel-major because its readout contracts over LEAVES, so
the accumulator's `m` axis is the readout's `m` axis and the accumulator becomes an A
operand by an in-lane pack. Both backward readouts contract over CHANNELS:

    dR[leaf, tok] = sum_c  Sreplay[leaf, c] · dn[tok, c]
    dW[leaf, tok] = sum_c  Lambda[leaf, c] · vt[tok, c]

so `m = leaf` and `k = channel`. Holding the slab leaf-major keeps the same
accumulator-as-operand identity the forward has; holding it channel-major would make
`k` the carve's own axis and turn every readout into a cross-warp partial sum. The carve
follows the `m` axis as it always has, so warps own LEAF slices — the transpose of the
forward's channel slices, and a consequence of the same law rather than an exception to
it.

The price is paid once, on `d_v`: `d_v[tok, :] = sum_leaf U[tok, leaf]·Lambda[leaf, :]`
contracts over the carved axis, so it is the one MMA whose B operand is the slab in the
wrong lane map. It is repaired by an in-register `movmatrix` per 8x8 tile and reduced
across the four warps in the output staging buffer. The deleted P65 design paid that
price on EVERY readout; here it is confined to one MMA on one path.

## The digit reduction's pass geometry

Under the mixed-radix local order a level's digit field is a contiguous bit run of the
owner's local index, and the leaf-split carve slices that index at a bit boundary. This
pass geometry needs every level to fall cleanly on one side of that boundary: either
warp-private (its field covers the whole warp id, so its digits PARTITION across warps)
or warp-replicated (its field lies entirely below the warp id). At the built arms,
`local = (r_0 << 4) | r_1` and warp `u` owns `local in [32u, 32u+32)`:

| level | segment | reduction | cross-warp |
|---|---|---|---|
| `l = 0` (outer, `s_0 = 8`) | a whole MMA m-tile | three lane shuffles | no — the warp owns its two digits |
| `l = 1` (inner, `s_1 = 16`) | one accumulator row | one add | yes — every warp holds every row |

No runtime index into a register array appears at either level, which is the constraint
that governed the deleted design's whole pass law.

### Where the inner level's four warps meet

The inner level is warp-REPLICATED: all four warps hold a partial of the same thirty-two
`(token, digit)` cells, so they have to meet. Three places were built and measured on
this kernel, and the choice is not the obvious one.

| meeting place | what it costs |
|---|---|
| a SHARED-memory atomic | **refused.** `atomicAdd` on a shared `float` has no hardware reduction on sm_80/sm_86; nvcc lowers it to `ATOMS.CAST.SPIN`, a lock-and-retry loop whose body is a five-instruction serial chain. Four warps on the same cells retried **2.08 times per site**, and the block carried **51.6 %** of Scan A's stall samples and **32.1 %** of Scan B's |
| barrier-separated ROUNDS | correct and deterministic, but three rendezvous per batch: measured `short_scoreboard` 3.57 → 2.24 against `barrier` 0.39 → 2.66, and the wall did not move. At four warps per CTA a CTA-wide barrier idles three of them |
| **the GLOBAL plane** | **taken on the read side.** The plane is already the OWNERS' meeting place — two owners sharing a level-1 group reduce into the same cell — so a fourth grain of partial adds no mechanism that was not there. No shared cell, no atomic, no rendezvous |

The global meeting is at the reduction-addressing floor: for one `(nt, e)` a warp's
thirty-two lanes cover FOUR token rows (`lane & 3`) at EIGHT CONTIGUOUS digits each
(`lane >> 2`), so one reducing instruction touches four 32-byte sectors for 128 bytes.
Measured `lg_throttle` stays at 0.00 with the read side's global reductions at 9.4 M.

It is available only where nothing multiplies the COMPLETED sum. `(R7)`'s `gw[j]` does,
so folding it into a warp-grain partial would DISTRIBUTE it over the four warps; the
WRITE side therefore stages its inner level through `sdg`, gains it in the drain, and
meets in barrier-separated rounds. The read side's `(R2)` has no such factor.

### The two conditions, and why only depth two meets them

Whether a level falls cleanly is a property of the SPAN VECTOR, not of the order, so the
refusal is stated as the two conditions rather than as a depth:

* the ROW map, `s_{D-1} == kMmaM` — one accumulator row is one innermost digit;
* the WARP map, `span_bits(0) == kLocalBits - log2 kMmaM` — the bits above the innermost
  level's field are ONE level's, so that level's digits partition across warps and no
  level straddles the warp-id boundary.

Only depth two can meet both: the innermost level owns the four row bits and one other
level must own everything above them. The two deep arms each fail exactly the conditions
that name their obstruction — `s = (2,4,16)` splits the upper bits between levels 0 and
1, so level 1's field straddles the boundary (two warps hold each of its digits, and the
staging would need a per-warp digit base plus a second cross-warp meeting); `s =
(2,2,4,8)` fails the row map as well, since `s_3 = 8` is half an m-tile and one
accumulator row then carries two levels' digits.

Extending either is a per-arm derivation carrying a wrong-gradient risk, so the gap is
NAMED rather than served: `rola.ops.prefill.prefill_autograd` refuses depth by the same
sentence, and the intra's reverse arms stop at the same place.

## What the digit gradients are, and where they are exact

A level's digit gradient is a weighted marginal of the one per-batch gradient tile along
that level's index axis, weighted by the product of the OTHER levels' amplitudes:

    d_p_l[t, e] = sum_{s : r_l(s) = e}  tile[t, s] · prod_{l' != l} p_l'[t, r_l'(s)]

The weight is a prefix/suffix product over the `D` per-level amplitudes the produce
already loads, so it costs `2D` multiplies and NEVER a division — `p_l` is exactly zero
off its support under exact entmax, and the leave-one-out written as `prod_l p_l / p_l`
would be `0/0` with no epsilon that does not invent a gradient.

**The result is exact on each level's own support and partial off it.** The schedule
visits an owner only where EVERY level's support meets that owner's run, while
`d_p_l` drops level `l`'s own factor — so at a digit outside level `l`'s support the
kernel holds the sum over the live owners rather than over all of them. Nothing reads
it: the exact-entmax Jacobian is zero off the support in both indices, and a softmax
level has no off-support entry at all. `tests/k31/test_backward_vs_reference.py` states
that as an algebraic gate, not as a comment.

## The mass channel

`cols = 65` is not a whole number of MMA k-tiles, so the mass channel does not ride the
readout's k axis. It is the slab's last n-tile, one eighth occupied, and the readout adds
its contribution as a broadcast product against the panel's own last column — which is
`d_den` on the seed side and 1 on the value side, and getting those two confused is the
one arithmetic difference between the two scans' readouts.

## The ordering law, mirrored

The forward's law is window-grain read-before-write on the owner's own slots. Scan B's is
its exact mirror: **writes of window `w` execute before `w`'s reads fold.** A batch that
crossed a window line would owe the within-window pairs inside it to neither this scan
(whose readout is against the adjoint at window EXIT) nor the intra backward (which owns
within-window pairs). Inside a window a scan batches freely between its own-side events,
which is the forward's private-quantum freedom iterated downward.

## Register floors are declared per kernel

`cudaFuncGetAttributes` is a property of a KERNEL, so one floor for two kernels is how a
spill goes green: the build honors a bound the heavier scan cannot meet and only the
census notices. `BwdGeom` declares `kRegFloorA` and `kRegFloorB` separately, and each
scan's ring depth and CTA target are derived from its own.
