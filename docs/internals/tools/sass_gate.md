# `tools/sass_gate.py` -- the ptxas signatures, on every build

The gate reads every function of every cubin a binary carries and is red on any of KERNEL_STANDARDS
[§20](../../KERNEL_STANDARDS.md#s20)'s ptxas signatures (local memory, a stack frame, convergence calls a converged
warp executes, predicated HMMAs, branches and gaps in an MMA step, step cover under its floor); guarded convergence
fallbacks are counted, cold code, not red. A step loop leaves out a fallback's return: the stub a `BRA.DIV` enters is
read to its unconditional branch back, however long (ptxas copies a step's remaining work into it: stubs of 49-101
lines in the carry's fold steps read as 3,000-instruction MMA loops under the old 32-line read). Every build runs it after the link (`setup.py`'s post-build gate); by hand,
`python tools/sass_gate.py <.so | .cubin> [--kernel RE] [--member S] [--loops] [--json PATH]`. Each function's line
carries its registers (`regs`, off `cuobjdump -res-usage`), so an iteration build that skips the ratify still shows a
register change.

## <a id="step-branches"></a>A branch inside a step

`mma_loop_branches` counts, in each innermost MMA loop (`mma_loops`: a backward branch and the lines from its target),
the convergence barriers (BSSY) and the forward conditional branches: each ends ptxas's scheduling block inside the step
([§20](../../KERNEL_STANDARDS.md#s20), §27). One shape is not counted, the loop's exit laid out as its LATCH: a forward
conditional branch out of the loop on the line just before an UNCONDITIONAL back-edge. ptxas lays a loop out so when its
exit is not the fall-through -- the carry's readout passes, one copy a reduce dtype and accumulate form, three of the
four jumping to the copies' join (2026-11-06) -- and the block it ends is the whole step: no step instruction follows it
in the loop. It costs one issued branch a trip against the folded form (`@P BRA` back, the exit falling through). Any
line between the exit and the back-edge (a register copy ptxas adds there), a branch back into the loop, or a
conditional back-edge still counts (`tests/unit/test_sass_gate.py`).

## <a id="cost"></a>What a read costs

Every lint of a function reads the same two disassemblies of its cubin and the same MMA loops, so each is formed once:
a cubin's `nvdisasm` text (`sass.disassemble`) and its parsed `cuobjdump -sass` (`sass_control._encoded_all`) once a
process for each version of the file (its path, size and mtime), `mma_loops` once a function (`_LOOPS`). A whole read
of the sm_86 extension is ~20 s (event:2026-11-01-p88-the-iteration-build-s-cost-is-the-post-build-sas), which the
verdict cache leaves to the cubins that changed.

## <a id="cache"></a>The verdict cache

A cubin's verdict, report and statistics are recorded under the key of everything that decides them -- the cubin's
bytes, the gate's own source closure (every module of `tools/` it has loaded, so the debt tables and machine models in
its source are covered), `tools/budgets/step_floors.json`, the disassemblers' identity (`nvdisasm` and `cuobjdump` by
their version lines) and the arguments (`--kernel`, `--max-per-hmma`, `--loops`) -- in the dev config's scratch
(`sass_gate_verdicts/`). A cubin whose key matches is replayed: its verdict is the reading of identical bytes under an
identical gate, the argument `tools/ratify.py`'s per-unit cache makes. Any change to the cubin or the gate misses and
re-reads. A no-change build's gate is ~0.1 s; a one-kernel edit re-reads that kernel's cubin and the stamp's.
`--no-cache` reads every cubin and records nothing.
