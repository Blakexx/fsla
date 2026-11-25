# `carry_supply`

Mirrors `measure/lab/experiments/carry_supply.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

Does the LIBRARY's carry CTA ([`internals/carry/carry_kernel.md`](../internals/carry/carry_kernel.md#supply)), its
fold warpgroups' supply built and every other component a stub, list each of its sets' live tokens from a call's real
operands -- the masks, the mask-preparation pass's prepared words, the densities the carve is derived from -- and land
their value lines in its rows, exactly; and what does a window cost a CTA with that supply running?

THE FORM. This file compiles the library's device code and launches the mask-preparation pass's kernel first. A call
of one head: two levels of 256 digits, 64 value channels, a window 1,024 tokens. Its rows are two calls: THE CELL,
the write side masked on level 0 at 4 to 5 live digits a token and whole on level 1; and the write side masked on
both, level 1 a run of 40 digits a token. The read side's densities are the cell's mirror and its masks are never
read (its role is a stub). The grid is the head's CTAs. CHECKED through the kernel's tracer on a call of 4 windows:
the tracer waits for a round's copies and copies its tokens and rows out, and on the host every round of every set of
every fold is there once, its tokens the set's list in order, its rows their value lines in the step's swizzle order,
a dead place's row zero, on its boundary; the sets' leaves tile the leaf space once. TIMED with the library's tracer
(nothing) on a call of 64 windows.

NOT MEASURED: no run of it is recorded.
