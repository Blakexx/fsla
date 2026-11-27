# The SASS gate -- the ptxas signatures, on every build

The library's SASS gate is WARPLINE'S CHECK on the linked library, run by the library rule itself (`bazel/defs.bzl`'s `_sass_gate`, which
calls warpline's `warpline_check_actions` once a built architecture): the checker's MANIFEST of the binary (every rule of its
registry on every function, there is no subset, and the build's ptxas notices counted under the ledger's rules), then its COMPARISON
with fsla's declarations. A refusal fails the comparison's action, so `bazel build //:library` fails, printing it; the comparison's
text is a verdict of the library (`fsla_cu13/sass_gate.<arch>.check.txt`), and so is closed-world rule 2 (beside it, `sass_gate.ptx.txt`,
[below](#graph)). What is fsla's is only what the declarations say ([below](#declarations)), the project's ledger
(`tools/notice_ledger.json`, [`lab.md`](lab.md#ledger)) and the notes (the `checker` section of `docs/rules.json`); the rules, their
counts, the refusals' texts and the one judgment are warpline's. The step cover (KERNEL_STANDARDS §32.4) is a family's machine model's
reading, and the gate runs none until a family's model exists.

## <a id="checker"></a>What the checker computes

Every ptxas signature is a rule of warpline's checker; its model reads `mma.sync` (`HMMA`, `IMMA`, `DMMA`) and the warpgroup's `HGMMA`
family alike as tensor steps, so the rules over steps read a production sm_90a cubin's `mma.sync` and `wgmma` steps alike:

| signature | the checker's rule |
|---|---|
| a nonzero stack frame, the `LDL`/`STL` beside it | `stack_memory` |
| a call into a convergence subroutine a converged warp executes (a guarded one, in a cold block, is not counted) | `convergence_call` |
| a predicated step | `step_predicated` |
| a convergence barrier or forward conditional branch in a loop of steps (the exit laid out as its latch excepted) | `step_branches` |
| the conversion unit in a loop of steps | `step_conversion` |
| global memory other than a staging copy in a loop of steps | `step_global` |
| static instructions per step over the ceiling a `step_density` declaration states | `step_density` (a bound) |
| a function's instructions, padding excluded | `function_size` (a figure: a bound once a declaration names one) |
| the longest run without a step in a loop of steps, inside a trip or around the back edge | `step_gap` (a figure) |
| the toolkit's notices (a spill, a serialized step, ...) | the rules of the ledger's rows (`register_spill`, `mma_serialized`, ...) |

and the rules over loops and loads (`load_in_trip`, `load_behind_load`, `loop_convergence`, `loop_held`) and over steps (`step_work`,
`step_port`, `step_rolled`, `step_serialized`, `accumulator_moved`, `wait_all_with_work`, `store_reload`, `store_unfenced`,
`wait_after_issue`, `clock_uniform`). THE ONE JUDGMENT is warpline's `compare` (`warpline.declare.compare`): undeclared is the rule's
default (zero, or a notice row's allowance), a declaration's `allow` is the count its kernels may keep (a bound rule's ceiling), the
first declaration of a rule whose regex names the function wins, and a count above it is a refusal printed in the checker's lines -- the
kernel, the rule, the count found and declared, the declaration's reason or `undeclared`, the sites, the fix and fsla's `note:`. A bound
rule with no bound declared is a figure, shown and never refused. The lab judges through the same comparison (`tools/checked.py`'s
`check`, [`lab.md`](lab.md#hazards)).

The notices are the units' ptxas reports (`--notices`, one a unit), read as one build's against the project's ledger: a notice
naming a function of the binary is counted at it, any other at its own place (a front end's warning at its `file(line)`, a summary that
names no function at `the build`). A warning of ptxas carries no architecture, so a build of several architectures counts each notice
in each architecture's check.

## <a id="manifest"></a>The static manifest

Each architecture's manifest (`fsla_cu13/sass_gate.<arch>.manifest.json`, warpline's `warpline-manifest/1`) is a build output of the
library, its output group `manifest`, named by `//:static_manifest` (`bazel build //:static_manifest`): every kernel's figures (its
instructions, registers, stack, shared bytes, barriers, steps, and each loop's), every rule's count and sites at every kernel, and the
notices' places. It is data about the build, not a verdict: the comparison above judged it, and it is what a store keeps beside a
run's manifest.

## <a id="step-branches"></a>A branch inside a step

The checker's `step_branches` counts, in each innermost natural loop that holds steps, the convergence barriers (BSSY) and
the forward conditional branches: each ends ptxas's scheduling block inside the step
([§20](../../KERNEL_STANDARDS.md#s20), §27). One shape is not counted, the loop's exit laid out as its LATCH: a forward
conditional branch out of the loop followed by nothing but unpredicated register copies (`MOV`, `IMAD.MOV`) and an
UNCONDITIONAL back-edge. ptxas lays a loop out so when its exit is not the fall-through -- the carry's readout passes,
one copy a reduce dtype and accumulate form, three of the four jumping to the copies' join (2026-11-06) -- and the block
it ends is the whole step: no step instruction follows it in the loop. The copies are the loop's carried values that
ptxas did not coalesce, moved on the back path only; a register's pressure is the spill check's (`stack_memory`) and the
register budget's (`tools/life_ranges.py`), not this lint's. The shape costs one issued branch a trip, and a copy each,
against the folded form (`@P BRA` back, the exit falling through). Other work between the exit and the back-edge, a
predicated copy, a branch back into the loop, or a conditional back-edge still counts (the checker's `latch_exit` fixture
and `test_the_exit_laid_out_as_the_latch_is_not_a_branch_inside_the_step`).

## <a id="declarations"></a>Declarations, beside the kernels

A kernel states what it keeps beside the unit that compiles it, in Starlark (`bazel/cuda.bzl`'s `declaration(kernel, rule, allow, reason)`, the checker's form, on
`cuda_unit(declarations = [...])`), and a family states its arms' beside the arms: `csrc/fsla/src/<family>/arms.bzl` holds `<FAMILY>_KERNEL`, the arm's kernel name up to its parameter
list as the toolkit demangles it with `@<field>@` for each template argument of the key and `@cc@` for the architecture (`decode`'s is
`fsla::decode::detail::decode_step_kernel<denseref::arch::BuildTag<denseref::arch::Build{(int)@cc@}>, (int)@d_v@, (int)@levels@, (bool)@decay@>`), and
`<FAMILY>_DECLARATIONS`, `arm_declaration(rule, allow, reason, where = {field: [values]})`s over the arms they select. `arm_family(kernel =, declarations =)` generates each arm's EXACT pattern
(`bazel/cuda.bzl`'s `kernel_pattern` escapes the template and anchors it at `^void` and the `(` of the parameter list, once; `bazel/arm_emit.py`'s `kernel_pattern` fills each `@<field>@`) for each arm the build carries (the arms its `--//:contract` reaches) and
each architecture it carries (`--//:archs`), and the declarations on the arms they select; nobody writes a kernel regex by hand for an arm.
The architecture is the attachment's: a declaration carries none, and `arch` on the Starlark struct says which build target it attaches to (none: every one the unit is built for).

The library rule writes ONE DECLARATIONS FILE A BUILT ARCHITECTURE (`bazel/gate_declarations.py`, `<label>_<arch>.declarations.json`),
warpline's form, which that architecture's check reads:

- the declarations beside the units, then the arm families', in that order (the first that names a kernel applies);
- each arm as a PRESENCE (`"check": "present", "exactly": 1` on its exact pattern): an arm the binary does not carry once is refused;
- each kernel the source defines (`tools/kernels.py`) as a presence (`"at_least": 1` on its name as the demangler writes it before the
  parameter list, `source_pattern`): a gate that cannot see a kernel has not gated it;
- `"strict": true`: a declaration that allows a count (not a bound, not a tightening) where no function it names has one is refused,
  so the declarations say only what is so;
- the notes, fsla's guidance a rule (the `checker` section of `docs/rules.json`), printed after a refusal's fix.

A declaration may tighten (`allow` 0 where a ledger row accepts the notice by default) as well as allow; a rule is a rule of the checker or of the project's ledger
([`lab.md`](lab.md#ledger)). `decode`'s and the producer's solves' declarations (`csrc/fsla/src/decode/arms.bzl`, `csrc/fsla/src/entmax/arms.bzl`: each family's, over its arms' keys) are each kernel's exact count of `load_in_trip` and `loop_convergence`, the two rules the full set found on code that predates them ("predates the rule; fixed at the family's Hopper rewrite"; the entmax factor backward's are its trip loop's, their cause open, `docs/internals/entmax/factor.md#backward-stages`), a cap and never more; the toolkit's notices of production (ptxas's per-function summaries, and `--warn-on-spills`, which finds none) are the base ledger's accepted rows.

## <a id="probes"></a>A library of probes

An instrument's library (`bazel/defs.bzl`'s `fsla_instrument`: a family's part harness, which lands with the family's first schedule)
is gated as the extension is, but its kernels are PROBES: its declarations owe none of the source's kernels (no source presence), and a
probe keeps a signature -- branches, the conversion unit or global memory in an MMA step -- only where its units' declarations name it
(`cuda_unit(declarations = ...)`, a permit: an `allow` of 10^9, the lab's `checked.PERMIT`, the cost the probe measures as the reason).
A calibration that puts a global reduction between its MMAs does so to measure that very cost, so it declares it, by the regex over the
function's demangled name. An undeclared signature is refused, as on the extension, and so, by strict, is a declaration an architecture
the binary carries does not show (a declaration attaches to its architecture's target, so a one-card iteration build reads none of
another card's). Every other signature (local memory, a stack frame, convergence calls, predicated HMMA, instructions per HMMA over a
declared ceiling) is refused on a probe as on any kernel. This is section 29's lab declaration (STEP_XU, STEP_GLOBAL) for an
instrument, beside its source.

## <a id="graph"></a>The graph's gate

`_sass_gate` runs, for each built architecture: fsla's declarations action (`FslaDeclarations`), warpline's manifest action
(`WarplineManifest`: `warpline manifest <.so> --arch <arch> --toolkit <the graph's> --notices <each unit's ptxas report>... --ledger
tools/notice_ledger.json`) and its comparison action (`WarplineCompare`: `warpline compare <manifest> --declare <the file> --ledger ...`),
on the hermetic toolkit's `cuobjdump`, `nvdisasm` and `cu++filt` (`@cuda_tools`) and warpline's command (`@warpline//:checker_cli`,
the pinned module's); and CLOSED-WORLD RULE 2 on the binary itself, `FslaPtxGate`: `cuobjdump --list-ptx` lists no PTX image
([`../../build.md`](../../build.md#what-the-build-checks)). Each action is cached by Bazel on its inputs
([`../../build-graph.md#gates`](../../build-graph.md#gates)); an unchanged library, declarations and checker re-read nothing.

## <a id="sass-text"></a>A unit's SASS as text

`bazel build //csrc/fsla:<unit>_sass` (a unit by its source's name: `decode_sass`) gives that unit's SASS, every architecture the build carries, as text by the graph's own
`cuobjdump -sass` (`bazel/defs.bzl`'s `unit_sass` over the library's units; the file is `_objs/<target>/<unit>.sass` under `.bazel/bin`; `cuda_unit`'s output group `sass` holds
them all, `bazel build //:library --output_groups=sass`), so a SASS diff of two commits is two builds and `diff`, never a disassembler typed by hand.
