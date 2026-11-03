# `carry/fold.cuh` — the fold

Mirrors `csrc/rola/src/carry/fold.cuh`: a warp's write lists over a window's units into the fp32 state, in the fold's
declared arithmetic (`tests/oracle/fold_reference.py`, `fold_state`), gated per slot by
`tests/oracle/test_carry_fold.py`.

<a id="state"></a>
## The state

`State` (in `parts.cuh`) is a warp's accumulators: per fold list (`fold_list`: the box's list `w + 8 j`), its digits'
two m-tiles by the channels' four n-tiles, fp32, each list in its own UNIT (below): 4 x 2 x 4 x 4 = 128 registers a
lane. They are the MMAs' C fragments for the whole call. Beside them the DEN'S, the v+1 channel's mass (`#den`), 4
registers a lane, absolute, the box's 32 lists by 32 digits once across the CTA (ruled 2026-11-02: the den as one more
channel of the state, register-resident). A word's place (`den_place`) is the list's owner's on both side forms: lane `4
g + t`'s word `i` list `w + 8 (2 (t & 1) + (i & 1))`, digit `g + 8 (i >> 1) + 16 (t >> 1)`, where `den_add` lands.

<a id="units"></a>
## The units

A record's c is the facts pass's `f16(fp32(b m s) 2^(4 - E))`, `E` its list's window exponent floored at `kUnitMin`
(`docs/internals/facts/facts.md#lists-out`). The state cannot follow `E` window by window: a quiet window would scale
it up past fp32's range. So each list keeps a running unit `u` in a shared-memory book (`UnitBook`: the unit and the
state's bound in units). `next_unit` is the kernel twin of `next_units`: a live list's `E`, held within the bound's
headroom (`kUnitHeadroom` = 100 binades), never below `kUnitMin`. A dead window keeps the unit. A side with no list
level has no records and no list exponents: every list takes the window's BOUND, `dense_exp` = `5 - shift` from the
facts pass's shift of the window's largest factor F (`facts.md#gains`: F under `2^(-3 - shift)`, a dense byte under
2^8), staged with the window's inputs in the exponents' place. A bound and not each list's own maximum: the dense
level's floor (a dense coefficient never under 2^-13 of its row's largest) holds every list's largest within 2^13 of it,
inside f16's 18 normal binades (measured 2026-10-19 against exact per-list exponents: the same state error on
flagship-dense, 2.8x it on corner-concentrated, both within the fold's bound). The state is rescaled by `2^(u_old - u_new)`, an exact power of two, zero below fp32's normals, beside the
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
MMAs. `step8`: the same over 8 tokens, one `ldmatrix.trans` each. A list whose den the box carries takes its den
column's MMAs beside them ([`#den`](#den)).
Each loop's step is one block (KERNEL_STANDARDS §27), guarded by a vote and closed by one. A list's first step's lanes
(`off`, `cpair` for entry `lane & 15`, which serve a tail's lanes 0-7 as well) arrive computed, and each 16-token
step fetches the next step's entries in its own block, after its MMAs in program order, so their loads issue under
them.

<a id="chunk"></a>
## The chunk

A CHUNK is a range of the write union the window's plan sets (`schedule.md#chunks`): the whole union where its buffer
fits the area, else the largest `64 m` slots two of whose buffers fit, and the rest. `fold_fill` copies its rows and
its lists' runs into its buffer (`parts.md#entries`); `fold_chunk` reads every one of the warp's lists' scale, count,
base and first step's entries before any list's MMAs -- four independent chains where they had run one after the other
behind each list's burst (nl64k-alt-k4, sm_86: the fold phase 3.26K -> 2.86K cycles a warp-window, 2026-11-01) -- then
takes each list in turn, its run from the buffer, a step's lanes past it masked (the chunk's first row, c = 0), the
rows taken back by the chunk's first slot so an entry's union slot is its row. A side with no list level steps every slot of the chunk: its buffer holds each slot's token's `(F, m)` in the
entries' place, and c = `f16(fp32(b 255 F) 2^(4 - E))` with `b` the list's digit of the row's coefficient slice (its
second) and `E` the window's bound -- the facts records' arithmetic, rounded once, by the packed conversion (`F2FP`, the
ALU pipe: the scalar one runs on the conversion unit, which a step may not hold). Until 2026-10-19 this branch took c
as `b` alone, without F or the list's unit: the dense-dense carry's state was wrong from its second window, and no gate
read it (the whole-call gate refused read sides with no list level; the probe wrote such a side's lists at one leaf).
The list's rescale is applied before its first chunk's steps, and only where its unit moved: a list whose unit stood
(`f == 1`, the warp's one value) multiplies nothing -- the multiply is a one-trip loop on a warp vote, since 128
multiplies a lane every window (1.0K warp instructions a CTA-window at alt-k4) bought nothing in the steady state.

<a id="laps"></a>
## The laps

An instrument: `fold_chunk`'s `Laps` (default `NoLaps`, whose call is empty) is called at the chunk's marks -- 0 once
every list's scale, count and first step's lanes are read, `1 + 2 J` once list `J`'s rescale is guarded, `2 + 2 J` once
its steps are done. The kernel passes none and its SASS is identical with the hook; the lab's
[`fold_alone`](../../lab/fold_alone.md) passes a recorder.

<a id="den"></a>
## The den column

The den is the v+1 channel -- each token's value 1 at its gain `m` -- so its state is the MASS `sum_t x b m` a leaf
(ruled 2026-11-02, the den in the carry). A box's 32 channels fill the MMAs' tiles, so the den takes one of its own: a
fifth n-tile of one live column (column 0, lanes `0..3`), its A the step's RAW slices (the value tiles' A is `f16(x c)`,
the coefficient on the routing weight, the product order the reference declares), its B the entries' `c_m` pairs
(`entry_of`'s `dpair`: a write record's high half, `f16(b m 2^(4 - Km))` -- the facts pass's, `Km` the list's mass
exponent; one shuffle of the record's word pairs c and c_m both). Two more MMAs a
16-token step (one an 8-token tail), into a `DenColumn` held only through the list's steps: then `den_add` adds each
row's value times `2^(K - 4)` into the DEN'S REGISTERS (`State::den`, `#state`; zero at the call's start), so the
state is in `b m` units across windows with no unit book. B's every column is the c_m pair (no column-0 select), so
every lane holds its quad's four rows: list `J` lies in the lanes `t & 1 == J >> 1`, each adding its two digits (two
selects, no branch) at words `J & 1` and `2 + (J & 1)`. The state left shared memory (4 KB of the area: nl64k-dense's read chunk 192 slots, three a
window, where 256 and two fit without it) and then the call's global workspace (a line of adds a list, a load a
window: 2026-11-02, 8c83835e) -- in registers it costs neither, and the window's tables read it there
(`parts.md#den`: a window's read chunks run before its fold chunks, so they see the state before its writes).

A box carries the den of a warp's list `j` where `j boxes / 4` is its box (`den_lists`, `boxes = dv / 32`): each of
two boxes lists 0-15 or 16-31, of four a quarter -- every list once, the cost shared. The branch is the CTA's, an
`if` on the box's mask (uniform: formed from the block's place and the call's `dv`); the one-trip loop each way it
replaced cost ~12 instructions a list a chunk at nl64k-alt-k4 (the region ledger's 12.6M a launch); `lists` takes the
side's form at compile time, so a side with no list level compiles its lists' steps without the den. MEASURED
(`fold_alone`, sm_86, nl64k-alt-k4's draw, cycles a warp-window, the carry's shape, the first form): no den 2,471, half
the lists 2,747 (+276), every list 3,039 (+568). MEASURED composed (phase ledger, nl64k-alt-k4, sm_86): the fold 3.57K
with its den, 2.86K with the box's den lists forced off in the same build (+0.71K: the den MMAs' fp32 tensor work on a
tensor-bound step, the adds, and the instruction cache's misses on the den steps' second copy).

A SIDE WITH NO LIST LEVEL carries the den the same way (ruled 2026-11-02: the foundation's UNIFORM den, the den riding
every component alike at listed and dense, the better den forms measured against it after): every list's steps over
every slot take its column, `c_m` the dense fetch's record half (`f16(b 255 m 2^(4 - Em))`, the facts records'
rounding), the den's exponent the window's `m` (`dense_exp`). MEASURED (c04b2bf2, the same form into the shared
state): the dense fold 85.0K against 71.8K with no den at nl64k-dense -- the den's one live column of an 8-wide n-tile
on every step of every list, eight times the den's own tensor work. The dense forms it stands in for, kept as candidates:
a tile a warp (digits by a list group, one live MMA a step: the fold 73.3K a warp, balanced; den-as-channel 91c6110c).

<a id="probe"></a>
## The probe

An instrument (`carry_probe_bind`, `rola.ops.carry.probe_bind`). When bound, the final state is written absolute,
fp32 `[BH, N, d_v]` by global leaf in level order: each accumulator times `2^(u - 4) / 255` (c on the byte scale, the
dense operand at 1/255 of it); and the den's registers `[BH, N]` by the box carrying each list, times `1 / 255`. Unbound,
nothing is written. The fold's oracle gate reads both (the den against the reference's mass, `fold_state`).
