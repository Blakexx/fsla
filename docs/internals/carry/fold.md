# `carry/fold.cuh` — the fold

Mirrors `csrc/rola/src/carry/fold.cuh`: a warp's write lists over a window's units into the fp32 state, in the fold's
declared arithmetic (`tests/oracle/fold_reference.py`, `fold_state`), gated per slot by `tests/oracle/test_carry_fold.py`.

<a id="state"></a>
## The state

`State` (in `parts.cuh`) is a warp's accumulators: per fold list, its rows' m-tiles by the channels' four n-tiles, fp32,
each list in its own UNIT (below). They are the MMAs' C fragments for the whole call. At the even box a warp holds 4 lists
x 2 m-tiles (32 operand digits) x 4 n-tiles (32 channels) x 4 = 128 registers.

<a id="units"></a>
## The units

A record's c is the facts pass's `f16(fp32(b m s) 2^(4 - E))`, `E` its list's window exponent floored at `kUnitMin`
(`docs/internals/facts/facts.md#lists-out`). The state cannot follow `E` window by window: a quiet window would scale
it up past fp32's range. So each list keeps a running unit `u` in a shared-memory book (`UnitBook`: the unit and the
state's bound in units). `next_unit` is the kernel twin of `next_units`: a live list's `E`, held within the bound's
headroom (`kUnitHeadroom` = 100 binades), never below `kUnitMin`. A dead window keeps the unit. The state is rescaled by
`2^(u_old - u_new)`, an exact power of two, zero below fp32's normals.

c is then moved to the unit by an f16 power of two, `2^(E - u)` (`pow2_half2`: normal from 2^-14, subnormal to 2^-24,
zero below). That is two roundings where a contribution falls past f16's normals, which the reference mirrors.
`fold_units` runs once a window before the passes. The window's list exponents come in with its lists (`exps_in`, a
shared-memory tenant), so the step never waits on a global load for them.

<a id="step"></a>
## The step

Lab `list_fold` form 5's step, KERNEL_STANDARDS 27's shape: 16 entries a step, held by lanes `16 h ..` as row offsets
and c pairs. Each lane's ldmatrix row offset and its four tokens' c come by shuffles. A is the slot's dense slice by
`ldmatrix.trans` (two m-tiles of 32 digits), times c by HMUL2. B is the values (four n-tiles of 32 channels). That
gives eight fp32-accumulate MMAs. There are no mass MMAs: the den pass owns the mass (ruled 2026-10-06), so `c_m` is
the den pass's.

A row's offset is `(slot & (unit slots - 1))` times the pitch, the swizzle's chunk bit folded in (`row_offset`). The
facts pass orders entries by slot residue, so eight consecutive rows meet eight bank groups at the odd-32 pitch.

<a id="unit"></a>
## The unit

A UNIT is U facts passes (U x 128 slots, `plan.md#units`). A list's entries in it are `[at[U u], at[U u + U])`, read
from the pass starts in shared memory (`list_span`). Two unit slots ring. Slot 0 is clear of the readout's region, so
each window's first unit fills under the publish and the readout (`fill_unit`, issued before the publish). Slot 1 sits
over the readout's region and fills under unit 0's steps. Each lands on its `mbarrier`, one
`cp.async.mbarrier.arrive.noinc` a thread a fill.

Each unit's fill also copies every box list's first 16 records into the slot's HEADS in shared memory (zero past the
list's count, so a pad is row 0 with c = 0). A list holds 1-16 entries a unit almost always at sparse cells, so the
common step reads its row offsets and c pairs in place (`load_head`): no global load, no shuffles. `fold_unit` issues
every list's first step in one straight-line block (an empty list's is pads, adding nothing), then ONE check: a unit
with a list past 16 entries continues those lists from their records in global memory, each step one block (a
clamped load and a select, never a branch). The rescale to the window's unit is applied beside each list's first MMAs.

MEASURED (2026-10-10, sm_86, the part harness `measure/lab/experiments/fold_part.cu` on nl64k-alt-k4's real lists,
`measure/lab/results/fold_part/sm_86.json`; floor 2.07K = the busiest scheduler's two warps' steps x 8 MMAs x 32
cycles): the step steady 1.054x, the MMAs alone 1.056x, a window's units behind their barrier 1.197x; in the composed
kernel's ledger the fold phase 2.92K (1.41x). The forms before: records from global by shuffles 1.25x steady; four
cleanup checks (each a scheduling block) 1.144x. What remains is the burst's edges: at alt-k4 a warp folds ~4 steps
a window between CTA barriers, so the first operand chain after the wait and the barrier are ~15% of the burst. That
cost is structural to sm_86's time-shared phases, one burst a phase a window.

<a id="stub"></a>
## The stub

`fold_stub` (in `parts.cuh`) stands in for the paths the step does not take yet: a side with no list level, and two
operand levels (the dense-dense plan's expansion, step 7a'). It reads each entry's record and one word of its row's
values and sums them into the state.

<a id="probe"></a>
## The probe

An instrument (`carry_probe_bind`, `rola.ops.carry.probe_bind`). When bound, the final state is written absolute,
fp32 `[BH, N, d_v]` by global leaf in level order: each accumulator times `2^(u - 4) / 255` (c on the byte scale, the
dense operand at 1/255 of it). Unbound, nothing is written. The fold's oracle gate reads it.
