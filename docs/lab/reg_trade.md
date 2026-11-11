# `reg_trade`

Mirrors `measure/lab/experiments/reg_trade.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

Two warpgroups of a CTA trade registers with `setmaxnreg`, one down and one up. When does the compiler honour it, what
does the trade cost, and does state held in registers across it survive?

THE FORM. A CTA of two warpgroups an SM, compiled under a register cap of 136 a thread (its `BUILD:` line). Both
warpgroups fold one state, an `m64n16` wgmma, into registers. Traded, warpgroup 0 lowers itself to 40 and writes
shared memory while warpgroup 1 raises itself to 232 and reads that state out, truncated into register A, into five
accumulator sets (160 fp32 a thread); untraded, both read out into the three sets the cap holds. The trade is clocked
alone, the readout by its wgmmas, and every accumulator is checked against the host's.

WHAT THE COMPILER SAYS (ptxas of the pinned toolkit, sm_90a; no device) -- three conditions, each found by compiling
the form that breaks it:

| the kernel | ptxas |
|---|---|
| no register cap stated | `setmaxnreg` IGNORED: "unable to determine register count at entry" (C7508); the kernel keeps one count, 190 |
| a cap, and the two roles' work after a join that follows the trade | honoured, but everything after the join is compiled at the LOWER count: the raised warpgroup's five sets landed on one set's registers, its wgmmas serialized (C7511) |
| a cap, each role's whole body inside its own branch | honoured: the launch holds 136 a thread, the lowered warpgroup releases down to 40 (`USETMAXREG.DEALLOC.CTAPOOL`), the raised one acquires up to 232 (`USETMAXREG.TRY_ALLOC.CTAPOOL`) and its five sets are five register ranges, up to R187 |

A lowered count under what its path holds live is refused the same way ("ignored to maintain minimum register
requirements", C7507): with the state's 8 registers live through a join, 56 was refused and 88 taken. So a kernel that
trades registers states its cap at the build, branches on the role once, and keeps each role's entire body, the
`setmaxnreg` first, inside its branch.

UNMEASURED: no hopper card has run it. The trade's cycles, the raised readout's cycles a wgmma beside the untraded
one's, and the state's survival across the trade (the check) are owed to its first run (`bazel run //tools:lab -- pack
reg_trade --arch sm_90a`, then `carried` on the card, under its kill timeout).
