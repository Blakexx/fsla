# `reg_trade`

Mirrors `measure/lab/experiments/reg_trade.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

Two warpgroups of a CTA trade registers with `setmaxnreg`, one down and one up. When does the compiler honour it, what
does the trade cost, and does state held in registers across it survive?

THE FORM. A CTA of two warpgroups, its launch bounds two CTAs an SM: 128 registers a thread at entry. Both warpgroups
fold one state, an `m64n16` wgmma, into registers. Traded, warpgroup 0 lowers itself to 24 and writes shared memory
while warpgroup 1 raises itself to 232 -- the 104 a thread the other released -- and reads that state out, truncated
into register A, into five accumulator sets (160 fp32 a thread); untraded, both read out into the three sets the entry
count holds. The trade is clocked alone, the readout by its wgmmas, and every accumulator is checked against the
host's.

WHAT THE COMPILER SAYS (ptxas of the pinned toolkit, sm_90a; no device), each row a form that was compiled:

| the kernel | ptxas |
|---|---|
| no `__launch_bounds__`, no register cap | `setmaxnreg` IGNORED: "unable to determine register count at entry" (C7508); the kernel keeps one count, 190 |
| the count at entry stated (`__launch_bounds__(256, 2)`: 128; or the build's `-maxrregcount`), the two roles' work after a join that follows the trade | honoured, but everything after the join is compiled at the LOWER count: the raised warpgroup's five sets landed on one set's registers and its wgmmas were serialized (C7511) |
| the count at entry stated, each role's whole body inside its own branch | honoured: the lowered warpgroup releases down to its count (`USETMAXREG.DEALLOC.CTAPOOL`), the raised one acquires up to 232 (`USETMAXREG.TRY_ALLOC.CTAPOOL`) and its five sets are five register ranges, up to R187 |

A lowered count under what its path holds live is refused ("ignored to maintain minimum register requirements",
C7507): with the roles joined, 56 was refused and 88 taken; with each role in its own branch 24 is taken. So a kernel
that trades registers states its launch bounds, branches on the role once, and keeps each role's entire body, the
`setmaxnreg` first, inside its branch. The trade here is balanced about the entry count; what a raise does when the
pool holds less than it asks is not established (the opcode is a TRY, so the experiment does not ask it).

MEASURED (H100 SXM, 132 SMs, a rented box: vast.ai instance 60164819, driver 595.91.07; 2026-11-11, one session, 7
repetitions, the medians):

| | untraded (128 a thread, three sets) | traded (24 and 232, five sets) |
|---|---|---|
| the trade, cycles a warp | -- | 201 |
| the readout, cycles a wgmma | 77.1 (75.7 to 78.3) | 52.5 |
| the accumulators against the host's | exact | exact |

WHAT IT SHOWS. The trade works on the device as compiled: the state folded before it is read out exactly after it, by
a warpgroup holding 232 registers a thread in a kernel launched at 128, and it costs about 200 cycles a warp, once.
The readout's cycles a wgmma differ between the columns (three sets at 128 registers: 77; five at 232: 52.5), and
the two differ in both the sets a commit group holds and the registers, so these rows do not say which one it is: the
same sets at both counts are owed.

A COMPILE OBSERVATION, 2026-11-22, toolkit 13.5 (ptxas, sm_90a; no device), NOT A ROW OF THE EXPERIMENT (the experiment
must build, and a form ptxas refuses cannot): a kernel that enters at 128 registers a thread is refused one `wgmma
m64n256k16` step, even in a warpgroup raised by `setmaxnreg` to 176, 192 or 232: "ptxas fatal: (C7602) Insufficient
registers (128) to compile instruction ... Try to compile with register target of 154 or higher". It was reproduced in this
experiment's own form (two warpgroups, `__launch_bounds__(256, 2)`, one lowered to 24, one raised to 232, each role's whole
body in its own branch) with the raised group's work replaced by that one step; the committed experiment compiles five
64-column sets there, registers up to R184. So the raise is honoured for placement, and the fit of one instruction is judged
against the ENTRY count: a step of 256 columns cannot be issued by a kernel that enters at 128, whatever its warpgroups
raise themselves to.

Record: `measure/lab/results/reg_trade/sm_90a.json`; its counters are the written exception (no profiler on that
box).
