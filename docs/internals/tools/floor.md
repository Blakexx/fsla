# `bazel/floor.bzl` — the floor gate

KERNEL_STANDARDS [§28](../../KERNEL_STANDARDS.md#s28) prices a kernel at the largest of its floors and reds a time above 1.15x
of it, and a phase that idles the binding stream; the CONSTITUTION's performance targets price every kernel against floors.
The floor is warpline's (its own page, the floor, in warpline's docs): a
kernel's declared WORK, by phase, over the RATES its card's probes measured, the stream that binds, and the bound written from
it, judged by warpline's checkers over a run manifest. It is a lower bound and predicts nothing (Blake, 2026-11-28:
"warpline.floor seems fine as the first step of our simple lower bound, and then we can build the predictive timeline model
as its own thing later"). This tree's part is the wiring: which cell, which card, which run, and the line.

## A kernel's gate

    load("//bazel:floor.bzl", "kernel_floor")
    kernel_floor(name = "decode_step", work = "decode_step.work.json", cell = "<a central cell>",
                 rates = "<the H100's rates table>", runs = ["decode_step.run.manifest.json"])

`kernel_floor` makes two targets in `measure/floors/` (every kernel's gate lives there):

- `<name>_bound`: warpline's `warpline_floor_bound`, the build action that runs the floor on the work declaration
  (`warpline-work/1`) at the card's rates (`warpline-rates/1`) with the PARAMETERS of the cell (`cell_parameters`: the
  central cell's numeric fields, `tokens`, `B`, `H`, `dv`, ..., from `@fsla_cells//:cells.bzl`'s `CELLS`; a cell no central
  registry has fails the load), and writes the bound declarations: at most `FACTOR` (1.15) x the launch floor and at least
  the floor, both on `FIGURE` (`launch/grid_ns`, the span of every CTA's launch at once, since the work declared is the
  launch's), the kernel present, and the idle check. Its output group `floor` is the floor's text.
- `<name>_floor`: warpline's `warpline_compare_test`, the bound over the kernel's stamped run manifests on that cell. A
  refusal names the figure, the floor and the binding stream.

`//measure/floors:floor` is every test of the package, and a member of `//:gate`, so a kernel's gate runs in every gate
without a line of its own there.

## <a id="the-idle-check"></a>The idle check

§28.2's phase between barriers that idles the binding stream: the floor writes warpline's `idle` declaration beside the bound,
the phases (`role/phase`, as the run manifest's `roles/<role>/phases/<phase>` names them) in which the binding stream's
family (`hbm`, `l2`, `shared`, `tensor`, `issue`) has declared work or a declared overlap with a phase that has some, and the
`serial` phases, each with the reason no overlapped form exists within the resources. A measured phase in neither is refused,
and so is a declared phase the run does not measure. What it cannot see: whether two roles' phases declared to overlap ran at
once. A run manifest's phases carry their cycles, share, jitter and hold, each in its own role's window, and no start on one
frame; a phase's start on the scope's one window (`roles/<role>/phases/<phase>/start`, the roles' windows aligned on one
anchor) is what the run manifest would need for the check to refuse a declared overlap that did not happen.

## The obligation

`tools/coverage.json`'s `floor` obligation ([`coverage.md`](coverage.md)): a kernel points at its gate's test,
`//measure/floors:<name>_floor`, which `tools/lint/coverage.py` resolves to a `kernel_floor(name = "<name>"` call in
`measure/floors/BUILD.bazel`, or it is `owed` or `exempt` with its reason. Today every kernel owes it: none declares its work,
the H100's rates table is not recorded (warpline's probes, `probes/rates.cu`, on the card), and no kernel's stamped run
manifest on a perf cell is kept. The three stamp and clock kernels are exempt: one tiny launch, no stream to bound.

## <a id="its-own-test"></a>Its own test

`measure/floors/`'s `wiring` is the gate on a kernel of the test's (`wiring.work.json`: 1 MiB from HBM a token of its cell, on
`carry-alt-k4-16x16-T64`'s 64 tokens) at a test card's rates (`wiring.rates.json`: one SM, 10 bytes a nanosecond), judged over
a run inside the line (`wiring.run.manifest.json`); `tests/unit/test_floor_gate.py` evaluates the macro file as Bazel does and
holds its arguments (the cell's fields, the factor, the figure, the test's manifests), and, through warpline's floor and
checkers, that a run over the line (`wiring.over.run.manifest.json`) and a run under the floor are refused.
