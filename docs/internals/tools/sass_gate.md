# `tools/sass_gate.py` -- the ptxas signatures, on every build

The gate reads every function of every cubin a binary carries and is red on any of KERNEL_STANDARDS
[§20](../../KERNEL_STANDARDS.md#s20)'s ptxas signatures (a stack frame, convergence calls a converged
warp executes, predicated HMMAs, branches, the conversion unit and global memory in a step, instructions per HMMA over a stated ceiling) and
reports the gaps in a step (the report's gap figures) without failing; guarded convergence
fallbacks are counted, cold code, not red. Every build runs it after the link (the library's gate action, [below](#graph)); by hand,
`bazel run //tools:sass_gate -- <.so | .cubin> [--kernel RE] [--member S] [--loops] [--json PATH]`. Each function's line
carries its registers (`regs`, off `cuobjdump -res-usage`), so an iteration build still shows a
register change. The step cover (KERNEL_STANDARDS §32.4) is a family's machine model's reading, and the gate runs
none until a family's model exists.

## <a id="checker"></a>What the checker computes, and what stays the gate's

Every ptxas signature is warpline's checker's rule (`tools/checked.py` imports it and builds its model of a cubin's kernels;
`tools/lab.py` reads the checker through the same module). Its model reads `mma.sync` (`HMMA`, `IMMA`, `DMMA`) and the
warpgroup's `HGMMA` family alike as tensor steps, so the rules over steps read a production sm_90a cubin's `mma.sync` and `wgmma` steps alike:

| signature | the checker's rule |
|---|---|
| a nonzero stack frame, the `LDL`/`STL` beside it | `stack_memory` |
| a call into a convergence subroutine a converged warp executes (the guarded ones, in cold blocks, are counted in `guarded`, not red) | `convergence_call` |
| a predicated step | `step_predicated` |
| a convergence barrier or forward conditional branch in a loop of steps (the exit laid out as its latch excepted) | `step_branches` |
| the conversion unit in a loop of steps | `step_conversion` |
| global memory other than a staging copy in a loop of steps | `step_global` |
| static instructions per step over the ceiling `--max-per-hmma` states | `step_density` |

The gate holds no declaration of its own for them: undeclared is zero, and its two declarations are translated to the
checker's: `--max-per-hmma N` is `step_density` allowing `N` on every function, and an instrument's declaration
(`step_branches`, `step_xu` = `step_conversion`, `step_global`, [below](#instrument)) is the rule's declaration on the
functions its pattern names, its allowance `_PERMIT` (no bound: the probe keeps the signature whatever its count). The
graph's gate action and `bazel run //tools:sass_gate` hand the tool the checker's package (`ROLA_WARPLINE_PYTHON`, the pin
`MODULE.bazel` names); the verdict cache's key includes the checker's source. A step is one natural loop's body, so a
fallback's return (the stub a `BRA.DIV` enters) and a compare-and-swap spin inside a step are the checker's model's
(`warpline/tests/test_checker_stage3.py`, `test_checker_model.py`), not a gate's lines.

What stays the gate's own: (1) a function's `instr`, the listing's count with the padding to the next 128-byte boundary
(the checker's `step_density` counts the function's instructions without it, and its figure is a whole count, the
quotient rounded up, so `per_hmma` is the checker's); (2) the two gap figures of the report and the JSON (inside, around), the two figures the checker's
`step_gap` takes the larger of (`gap_split`, over the model's loops of steps; the checker's figure is the larger and is what a
declared bound holds, and the gate declares none); (3) a PTX image in a binary (closed-world rule 2); (4) a source kernel no
cubin carries (`tools/kernels.py`); (5) the narrowing that matches nothing is red. The gate reads no build notices (its
action has no ptxas log), so the project's notice ledger is the lab's alone.

## <a id="step-branches"></a>A branch inside a step

The checker's `step_branches` counts, in each innermost natural loop that holds steps, the convergence barriers (BSSY) and
the forward conditional branches: each ends ptxas's scheduling block inside the step
([§20](../../KERNEL_STANDARDS.md#s20), §27). One shape is not counted, the loop's exit laid out as its LATCH: a forward
conditional branch out of the loop followed by nothing but unpredicated register copies (`MOV`, `IMAD.MOV`) and an
UNCONDITIONAL back-edge. ptxas lays a loop out so when its exit is not the fall-through -- the carry's readout passes,
one copy a reduce dtype and accumulate form, three of the four jumping to the copies' join (2026-11-06) -- and the block
it ends is the whole step: no step instruction follows it in the loop. The copies are the loop's carried values that
ptxas did not coalesce, moved on the back path only; a register's pressure is the spill check's (`ldl`) and the
register budget's (`tools/life_ranges.py`), not this lint's. The shape costs one issued branch a trip, and a copy each,
against the folded form (`@P BRA` back, the exit falling through). Other work between the exit and the back-edge, a
predicated copy, a branch back into the loop, or a conditional back-edge still counts (the checker's `latch_exit` fixture
and `test_the_exit_laid_out_as_the_latch_is_not_a_branch_inside_the_step`).

## <a id="cost"></a>What a read costs

Every rule of a function reads the same model of the cubin, built once a process per cubin (`checked.kernels_of`: `cuobjdump`'s
listing and resource report and `nvdisasm`'s graph; the disassembly (`sass.disassemble`) once for each version of the file,
its path, size and mtime, for the instruction count). A whole read
of the sm_86 extension is ~20 s (measured 2026-11-01), which the
verdict cache leaves to the cubins that changed.

## <a id="cache"></a>The verdict cache

A cubin's verdict, report and statistics are recorded under the key of everything that decides them -- the cubin's
bytes, the gate's own source closure (every module of `tools/` it has loaded, and the checker's), the disassemblers' identity (`nvdisasm` and `cuobjdump` by
their version lines) and the arguments (`--kernel`, `--max-per-hmma`, `--loops`, and `--instrument`'s declarations, by
content) -- in the dev config's scratch
(`sass_gate_verdicts/`). A cubin whose key matches is replayed: its verdict is the reading of identical bytes under an
identical gate, the argument the graph's action cache makes for every unit. Any change to the cubin or the gate misses and
re-reads. A no-change build's gate is ~0.1 s; a one-kernel edit re-reads that kernel's cubin and the stamp's.
`--no-cache` reads every cubin and records nothing.

## <a id="instrument"></a>An instrument's library

`--instrument DECLARATIONS` gates an instrument's library (`bazel/defs.bzl`'s `rola_instrument`: a family's part
harness, which lands with the family's first schedule): its kernels are PROBES, so it owes none of the source's
kernels, and a probe keeps a step signature -- branches, the conversion unit or global memory in an MMA step -- only
where its declarations name it (the checker's declarations, above). A calibration that puts a global reduction between its MMAs does so to measure that
very cost, so it declares it, by a pattern over the function's mangled name, with the cost it measures (a JSON file,
`kernels`: `{pattern: {lint: cost}}`, the lints `step_branches`, `step_xu`, `step_global`); `sm_XX <pattern>` declares
a signature one architecture's SASS keeps alone (ptxas schedules each apart). An undeclared
signature is red, as on the extension, and so is a declaration an architecture the binary carries does not show: a
plain one on each such architecture, an `sm_XX` one on its own (a one-card iteration build reads none of another
card's). The declarations say only what is so. Every other
signature (local memory, a stack frame, convergence calls, predicated HMMA, instructions per HMMA over the ceiling) is red on a probe as on any kernel. This
is section 29's lab declaration (STEP_XU, STEP_GLOBAL) for an instrument, beside its source.

## <a id="graph"></a>The graph's gate

The graph's library rule runs the gate as an action with `--cuobjdump` and `--nvdisasm` naming its hermetic toolkit's
(`@cuda_tools`): `sass.use_graph` points every disassembly at those binaries and every scratch directory at one of the
process's own, and the verdict cache is off -- the action is cached by Bazel on its inputs, the library and the gate's
sources ([`../../build-graph.md#gates`](../../build-graph.md#gates)). Run from a checkout, the gate reads the dev
config's toolkit and scratch, and its own cache.
