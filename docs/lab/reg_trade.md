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

UNMEASURED: no hopper card has run it. The trade's cycles, the raised readout's cycles a wgmma beside the untraded
one's, and the state's survival across the trade (the check) are owed to its first run (`bazel run //tools:lab -- pack
reg_trade --arch sm_90a`, then `carried` on the card, under its kill timeout).
