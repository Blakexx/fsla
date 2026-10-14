# `carry/schedule.cuh` — the window's plan

Mirrors `csrc/rola/src/carry/schedule.cuh`: each window's PLAN, made by the plan pass before the carry runs from the
facts pass's counts: the readout's form and chunk, the fold's chunk, and where each buffer sits in the AREA, the shared
memory past the tenants (`plan.md#plan`). The carry is one pipeline:
shared memory holds what is active now and what lands next, as full as the window's actual counts allow, and a bigger
card's area takes larger chunks with no other change (ruled 2026-10-14). The executor is `carry_kernel.md#kernel`; the
plan's Python model, which a gate asserts its cell's forms by, is `tests/oracle/carry_plan.py`.

Symbols: `U` a side's union slots in the window, `C` a chunk's slots (a multiple of 64 below the union, or the whole
union), `E` entries, `B` the area's bytes.

<a id="buffers"></a>
## The buffers

A READ BUFFER holds a chunk of the read union: its rows (`C x row`), the dummy panel rows (`kPadBytes`), the panel
(`C x 64`), its entries (`E x 8`), each part 128-byte aligned (`read_bytes`). A FOLD BUFFER holds a chunk of the write
union: its rows (slices then values) and its entries (`fold_bytes`). The IMAGE is the whole state (32 planes, 64 KB) or
a ring of two 8-plane batches (32 KB). An entry's row and panel offsets are the union's, so a chunk from a multiple of
64 takes its pool and panel back by its first slot (`carry.md#lists`); its entries are the lists' runs in list order.

<a id="totals"></a>
## The segment sums

`side_counts`: a side's union slots and `S[j]`, its box lists' entries before union slot `64 (j + 1)` -- the facts
pass's prefix rows' totals row, one 16-byte load (`facts.md#lists-out`). Any chunk's entries are a difference of two of
them, so every candidate is priced exactly; `chunk_most` is the most any chunk of `64 m` slots holds over the union's
chunks, every candidate priced from registers.

<a id="forms"></a>
## The readout's forms

- IMAGE (`kFormImage`): the image whole, stored once, the union read chunk by chunk, two buffers in flight -- one
  barrier a window between the stores and the reads, and a warp folds as soon as it has read out.
- UNION (`kFormUnion`): the union resident in one buffer, the image a ring of four 8-plane batches, each stored, read
  over the union, its slot freed for the batch two on -- a barrier a batch, and the most room left for the fold.
- TILED (`kFormTiled`): both in chunks, the ring's batches stored again for each chunk; on one read buffer when two do
  not fit -- the form every window fits (`least_form_fits`, a static assertion: the ring and one 64-slot chunk of lists
  full on every token).

<a id="planner"></a>
## The planner

`make_plan` prices five READOUT CANDIDATES and takes the first that fits the area, in order: the image whole and the
union resident; the union resident; the image whole and the union in chunks; both in chunks; both in chunks on one
buffer (a chunk the largest `64 m` below the union whose buffers fit, `chunk_of`, every candidate priced so no loop
holds a collective). `CarryParams.prefer` names a candidate taken first where it fits: reserved, for the gates and the
measurements (`rola.ops.carry.READ_CANDIDATES`).

THE FOLD: its whole union in one buffer beside the readout's zone where it fits (it lands at the window's start), else
after the readout, else in chunks over two buffers. Fold buffers sit at the top of the area, buffer 1 below buffer 0;
a buffer below the readout's zone end waits for the readout's release.

THE BOUNDARY: the next window's plan lands with its inputs (mid-window), and its first read chunk lands under this
window's fold when its buffer ends below every fold buffer, else once the fold is done.

This planner takes the first form that fits; the deterministic per-card cost model that chooses among the fitting
candidates, and its check against the best plan over a cell's windows, are the next stage's.

<a id="pass"></a>
## The plan pass

`window_plan_kernel`, launched before the carry on its stream: a thread a (batch-head, box class, window), its sides'
counts (`side_counts`), its plan (`make_plan`) out to `CarryParams.plans` (`[BH][windows][classes]`, `kPlanBytes`
each). A BOX CLASS is the pair of the read and write list levels' groups, which fixes a CTA's counts
(`box_classes`). A window's planning costs the carry nothing: it stages one 48-byte plan with the window's inputs. The
whole sequence of a box's windows is the pass's to see, so the cost model's choice can be the best over the sequence.
Gated field for field against the plans' model on every form cell and preferred candidate
(`tests/oracle/test_carry_plan.py`).

<a id="plan"></a>
## The plan

`WindowPlan`, the CTA's at the window's parity (`plan_of`, read at each use so no field holds a register): the
readout's form and its number of read buffers, whether fold buffer 0 lands beside the readout, each side's union and
chunk, the buffers' places (bytes / 16), a buffer's entries offset, the readout zone's end and the fold zone's start.

Worked, on the even box's rows (64-byte read rows, 128-byte write rows; the area 91,392 bytes on sm_86, 156,928 on
sm_80, 222,464 on sm_90): a window of ~216-slot unions and ~260 entries a side takes the union form on sm_86 (32 KB +
~30 KB) with the whole fold beside (~30 KB), and the image form with both unions resident on sm_80; a dense window of
the dense-dense arm (512 slots, 128- and 192-byte rows) takes the image form in 64-slot chunks on sm_86.

Per architecture: the same plan everywhere, its area the card's (`hardware.json`); on sm_90 the fills become TMA
copies on the same landed barriers. Scaling: the pass a thread a (batch-head, box class, window), O(1) each; nothing
O(L) or O(N) in the carry.
