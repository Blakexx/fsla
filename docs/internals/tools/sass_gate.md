# `tools/sass_gate.py` -- the ptxas signatures, on every build

The gate reads every function of every cubin a binary carries and applies EVERY rule of the checker's registry to each (`sass_gate.rules()`; there is no subset): a stack frame,
convergence calls a converged warp executes, predicated HMMAs, branches, the conversion unit and global memory in a step, instructions per HMMA over a declared ceiling, and
the rules over loops and loads (`load_in_trip`, `loop_convergence`) and over steps (`step_work`, `step_port`, `step_rolled`, `store_reload`, `store_unfenced`, `wait_after_issue`, `clock_uniform`). It is red on any
finding a declaration does not hold, on the build's ptxas
notices its ledger refuses, and on what a declaration over them does not hold; it shows the figures of the bound rules (`function_size`, `step_gap`) without failing;
guarded convergence fallbacks are counted, cold code, not red. Every build runs it after the link (the library's gate action, [below](#graph)); by hand,
`bazel run //tools:sass_gate -- <.so | .cubin> [--kernel RE] [--member S] [--declare FILE] [--notices LOG]... [--loops] [--json PATH]`. Each function's line
carries its registers (`regs`, off `cuobjdump -res-usage`), so an iteration build still shows a
register change. The step cover (KERNEL_STANDARDS §32.4) is a family's machine model's reading, and the gate runs
none until a family's model exists.

## <a id="checker"></a>What the checker computes, and what stays the gate's

Every ptxas signature is warpline's checker's rule (`tools/checked.py` imports it and builds its model of a cubin's kernels, named by their demangled names, which a declaration's regex is over;
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
| static instructions per step over the ceiling a `step_density` declaration states | `step_density` (a bound) |
| a function's instructions, padding excluded (`instr`) | `function_size` (a figure: a bound once a declaration names one) |
| the longest run without a step in a loop of steps (`mma_gap`, the larger of inside a trip and around the back edge) | `step_gap` (a figure) |
| the toolkit's notices (a spill, a serialized step, ...) | the rules of the ledger's rows (`register_spill`, `mma_serialized`, ...) |

THE ONE JUDGMENT. The gate takes the checker's FACTS of a cubin (every rule's count and sites at every function, the build's notices counted at the functions they
name) and COMPARES them with the declarations (`checked.compare`: `facts.collect` and `declare.compare`, the code the lab's check calls too): undeclared is zero, a
declaration's `allow` is the count its kernels may keep (a bound rule's ceiling), the first declaration of a rule whose regex names the function wins, and a count above it is
a refusal printed in the checker's lines -- the kernel, the rule, the count found and declared, the declaration's reason or `undeclared`, the sites and the fix. A refusal is red.
The gate holds no refusal text of its own; a function's report line carries its figures and the refused rules.

What stays the gate's own: (1) a PTX image in a binary (closed-world rule 2); (2) a source kernel no cubin carries (`tools/kernels.py`); (3) the narrowing that
matches nothing is red; (4) the declarations' own check, which the checker does not make across cubins: a declaration that allows a count (not a bound, not a tightening) where
no function it names has one on its architecture is red, and so is an arm whose exact kernel pattern does not match one function of the architecture's SASS
(the pattern is generated from the arm's template arguments, [below](#declarations)); (5) the report and the JSON (`instr`, `mma_gap` are the rules' figures: `instr` leaves out the padding to the next
128-byte boundary, and `mma_gap` is `step_gap`'s one figure, the longest run without a step in a loop of steps, inside a trip or around the back edge; the gate
that kept the two apart is superseded, and what it compared is in `DELETIONS.md`). The notices are the units' ptxas
reports (`--notices`, one a unit), read against the project's ledger; a notice naming a function of a cubin is judged with it, and what no cubin claims (a front end's
warning, a function no cubin carries) once for the build. A warning of ptxas carries no architecture, so a build of several architectures counts each notice for each architecture's cubin.

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
their version lines), the project's ledger and the arguments (`--kernel`, `--loops`, and the architecture's declarations and the notices, by
content) -- in the dev config's scratch
(`sass_gate_verdicts/`). A cubin whose key matches is replayed: its verdict is the reading of identical bytes under an
identical gate, the argument the graph's action cache makes for every unit. Any change to the cubin or the gate misses and
re-reads. A no-change build's gate is ~0.1 s; a one-kernel edit re-reads that kernel's cubin and the stamp's.
`--no-cache` reads every cubin and records nothing.

## <a id="declarations"></a>Declarations, beside the kernels

A kernel states what it keeps beside the unit that compiles it, in Starlark (`bazel/cuda.bzl`'s `declaration(kernel, rule, allow, reason)`, the checker's form, on
`cuda_unit(declarations = [...])`), and a family states its arms' beside the arms: `csrc/rola/src/<family>/arms.bzl` holds `<FAMILY>_KERNEL`, the arm's kernel name up to its parameter
list as the toolkit demangles it with `@<field>@` for each template argument of the key and `@cc@` for the architecture (`decode`'s is
`rola::decode::detail::decode_step_kernel<denseref::arch::BuildTag<denseref::arch::Build{(int)@cc@, (int)0}>, (int)@d_v@, (int)@levels@, (bool)@decay@>`), and
`<FAMILY>_DECLARATIONS`, `arm_declaration(rule, allow, reason, where = {field: [values]})`s over the arms they select. `arm_family(kernel =, declarations =)` generates each arm's EXACT pattern
(`bazel/arms.bzl`'s `arm_kernel_pattern`: the template's text escaped, anchored at `^void` and the `(` of the parameter list) for each arm the build carries (`--//:<family>_arms`) and
each architecture it carries (`--//:archs`), and the declarations on the arms they select; nobody writes a kernel regex by hand for an arm.
The architecture is the attachment's: a declaration carries none, and `arch` on the Starlark struct says which build target it attaches to (none: every one the unit is built for); the
library rule writes one file (`<label>_declarations.json`: per architecture, its declarations and its arms) and the gate reads each cubin's under its own architecture. A
declaration may tighten (`allow` 0 where a ledger row accepts the notice by default) as well as allow; a rule is a rule of the checker or of the project's ledger
(`docs/internals/tools/lab.md#ledger`). `decode`'s and the producer's solves' declarations (`csrc/rola/src/decode/arms.bzl`, `csrc/rola/src/entmax/declarations.bzl`, on the `entmax_units` target) are each kernel's exact count of `load_in_trip` and `loop_convergence`, the two rules the full set found on code that predates them ("predates the rule; fixed at the family's Hopper rewrite"), a cap and never more; the toolkit's notices of production (ptxas's per-function summaries, and `--warn-on-spills`, which finds none) are the base ledger's accepted rows.

## <a id="probes"></a>A library of probes

`--probes` gates an instrument's library (`bazel/defs.bzl`'s `rola_instrument`: a family's part
harness, which lands with the family's first schedule): its kernels are PROBES, so it owes none of the source's
kernels, and a probe keeps a signature -- branches, the conversion unit or global memory in an MMA step -- only
where its units' declarations name it (`cuda_unit(declarations = ...)`, a permit: `allow` of `checked.PERMIT`, the cost the probe measures as the reason; the same constant the lab's presence expectations use). A calibration that puts a global reduction between its MMAs does so to measure that
very cost, so it declares it, by the regex over the function's demangled name. An undeclared
signature is red, as on the extension, and so is a declaration an architecture the binary carries does not show (a declaration attaches to its architecture's target, so a
one-card iteration build reads none of another card's). The declarations say only what is so. Every other
signature (local memory, a stack frame, convergence calls, predicated HMMA, instructions per HMMA over a declared ceiling) is red on a probe as on any kernel. This
is section 29's lab declaration (STEP_XU, STEP_GLOBAL) for an instrument, beside its source.

## <a id="graph"></a>The graph's gate

The graph's library rule runs the gate as an action with `--cuobjdump`, `--nvdisasm` and `--cuxxfilt` naming its hermetic toolkit's
(`@cuda_tools`), `--declare` the file of the declarations stated beside the units and arms it links and `--notices` each unit's ptxas report
(and the project's ledger as an input): `sass.use_graph` points every disassembly at those binaries and every scratch directory at one of the
process's own, and the verdict cache is off -- the action is cached by Bazel on its inputs, the library and the gate's
sources ([`../../build-graph.md#gates`](../../build-graph.md#gates)). Run from a checkout, the gate reads the dev
config's toolkit and scratch, and its own cache.

## <a id="sass-text"></a>A unit's SASS as text

`bazel build //csrc/rola:<unit>_sass` (a unit by its source's name: `decode_sass`) gives that unit's SASS, every architecture the build carries, as text by the graph's own
`cuobjdump -sass` (`bazel/defs.bzl`'s `unit_sass` over the library's units; the file is `_objs/<target>/<unit>.sass` under `.bazel/bin`; `cuda_unit`'s output group `sass` holds
them all, `bazel build //:library --output_groups=sass`), so a SASS diff of two commits is two builds and `diff`, never a disassembler typed by hand.
