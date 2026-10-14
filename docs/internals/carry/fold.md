# `carry/fold.cuh` — the fold

Mirrors `csrc/rola/src/carry/fold.cuh`: a warp's write lists over a window's units into the fp32 state, in the fold's
declared arithmetic (`tests/oracle/fold_reference.py`, `fold_state`), gated per slot by
`tests/oracle/test_carry_fold.py`.

<a id="state"></a>
## The state

`State` (in `parts.cuh`) is a warp's accumulators: per fold list (`fold_list`: the box's list `w + 8 j`), its digits'
two m-tiles by the channels' four n-tiles, fp32, each list in its own UNIT (below): 4 x 2 x 4 x 4 = 128 registers a
lane. They are the MMAs' C fragments for the whole call.

<a id="units"></a>
## The units

A record's c is the facts pass's `f16(fp32(b m s) 2^(4 - E))`, `E` its list's window exponent floored at `kUnitMin`
(`docs/internals/facts/facts.md#lists-out`). The state cannot follow `E` window by window: a quiet window would scale
it up past fp32's range. So each list keeps a running unit `u` in a shared-memory book (`UnitBook`: the unit and the
state's bound in units). `next_unit` is the kernel twin of `next_units`: a live list's `E`, held within the bound's
headroom (`kUnitHeadroom` = 100 binades), never below `kUnitMin`. A dead window keeps the unit, as does a side with no
list level. The state is rescaled by `2^(u_old - u_new)`, an exact power of two, zero below fp32's normals, beside the
list's first step.

c is then moved to the unit by an f16 power of two, `2^(E - u)` (`pow2_half2`: normal from 2^-14, subnormal to 2^-24,
zero below), two roundings where a contribution falls past f16's normals, which the reference mirrors. `fold_units`
runs once a window; the window's exponent pairs come in with its lists (`parts.md#lists-in`), so it never waits on a
global load. It posts each list's rescale and c factor to the list's scale (`ListScale`, the `kScale` tenant), which
the window's fold chunks read, the first applying the rescale: so no register holds them through the readout.

<a id="step"></a>
## The step

16 TOKENS WITH AN 8-TOKEN TAIL (ruled 2026-10-12): `steps` takes a list's entries in 16-token steps while more
than 8 remain, then an 8-token tail of m16n8k8 MMAs, half a step's tensor work. `entry_of` turns an entry into a
lane's row offset (its swizzle folded in, `parts.md#swizzle`) and its c paired with the next lane's, moved to the unit;
a pad past the list's count is row 0 with c = 0. `step16`: A is the slots' dense slices by `ldmatrix.trans` (two
m-tiles of the box's 32 digits) times c by HMUL2, B the values (four n-tiles of 32 channels), eight fp32-accumulate
MMAs. `step8`: the same over 8 tokens, one `ldmatrix.trans` each. There are no mass MMAs: the den pass owns the mass.
Each loop's step is one block (KERNEL_STANDARDS §27), guarded by a vote and closed by one.

<a id="chunk"></a>
## The chunk

A CHUNK is a range of the write union the window's plan sets (`schedule.md#chunks`): the whole union where its buffer
fits the area, else the largest `64 m` slots two of whose buffers fit, and the rest. `fold_fill` copies its rows and
its lists' runs into its buffer (`parts.md#entries`); `fold_chunk` takes each of the warp's lists in turn, its run from the buffer, a step's lanes past
it masked (the chunk's first row, c = 0), the rows taken back by the chunk's first slot so an entry's union slot is its
row. A side with no list level steps every slot of the chunk, c the list's digit of the row's coefficient slice. The
list's rescale is applied before its first chunk's steps.

<a id="probe"></a>
## The probe

An instrument (`carry_probe_bind`, `rola.ops.carry.probe_bind`). When bound, the final state is written absolute,
fp32 `[BH, N, d_v]` by global leaf in level order: each accumulator times `2^(u - 4) / 255` (c on the byte scale, the
dense operand at 1/255 of it). Unbound, nothing is written. The fold's oracle gate reads it.
