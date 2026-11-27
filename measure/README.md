# `measure/` — fsla's reading of the central cells, the benches over them, and the executors of its declarations

The methodology, the instruments and the record are `docs/measurement.md`; the lock rule
is `docs/testing.md`. This file is the map.

**THE CELLS ARE FSLA-DEVTOOLS' (`fsla_devtools.cells`), and there is one registry for every package.** A cell is an
input -- a carry cell's routing amplitudes, gain, values and the state it enters with; a layer cell's hidden states and
values -- with its draw and the regime it proves, so a name means one input everywhere it appears, in fsla and in the
libraries fsla is compared against. A cell names nothing fsla runs it with. `measure/cells/` is fsla's reading:

* `cells/__init__.py` — a central cell's typed record by name (`by_name`), the carry cells by tier (`carry_cells`) and
  a cell's draw (`realize`). What the carry kernel is given for a cell is built where it is timed
  (`subjects.carry_operands`).
* `cells/layer.py` — FSLA's layer CONSTRUCTIONS (a routing template, widths, a gain), each named with the central layer
  inputs it was declared for, from which the amplitudes are PRODUCED. It is the only way to price the producer's own
  solve or a decode step through the layer. A layer arm's fixture builds each part (the layer, the inputs, the produced
  routes) when the arm first reads it.

**`BUILD.bazel` — THE TIMING ENTRIES AND SESSIONS.** Each `timing_entry` is one executor of `executors.py` on central
cells (`@fsla_cells//:<cell>`) in this tree's bundle; each `interleaved_timing` is a session over entries, measured and
stored by `bazel run` (`bazel run //measure:session_entmax`: the entmax solve on the pinned layer cell).

**`measure/executors.py` — WHAT FSLA'S TIMING ENTRIES AND DECLARATIONS RUN.** `timed` (an arm by name) and
`read_clock` are the timing entries' executors. The repository root's `declare.py` declares this checkout's targets for
fsla-devtools' build system (`fsla_devtools.build`): its build, the machine facts, its instruments (the SASS gate and the
realized support; the register walk, phase clock, pipe counters, stall census and timeline are tools a kernel with a schedule takes a row of in `tools/kernel_targets.py`), and,
for fsla-bench's root, which loads it from every checkout it measures, every bench subject as a timing registration on
the central cells it takes and the clock reader. `bazel run //:measure -- run declare.py:all` runs this checkout's
instruments and diffs.

**`measure/lab/` — THE LAB.** Small CUDA experiments on the machine itself (what a pipe costs beside the tensor pipe,
what ptxas does to a pattern), each a `experiments/<name>.cu` on the kit `lab.cuh`, compiled, SASS-gated, timed and
recorded by `tools/lab.py` into the data repository's checkout (`lab/<name>/<sm_XX>.json`); a KERNEL_STANDARDS rule
about the machine cites one by its path in that repository (docs/internals/tools/lab.md). `measure/data.py` finds the pinned data
repository every reader takes the records from, its root handed in by the graph (`FSLA_DATA`).

**`measure/` — the library.** `subjects.py` is the roster: one lean callable per
kernel this line carries, each taking a registry cell and returning the launch to time,
with everything the launch does not pay for built outside the timed callable. The roster
mirrors the oracle roster one for one — a bench roster that does not match the correctness
roster is a roster with kernels nobody measures — and it includes the carry, whose body
does not exist on this line and whose arm therefore refuses by name rather than reporting
a number for something that did not run. `provider.py` is fsla's runner: given a central cell
it offers the subjects this binary runs on it, with their dials
(`carry_forward@schedule=identity`) and on a layer cell their constructions
(`decode_step@layer=chunk-decode-w16`), or refuses the cell by name; arms build in their own
checkout's venv.

**A timing session** interleaves timing entries of fsla checkouts, and of other libraries, call by call under one
stopwatch and the GPU and clock locks (`fsla_devtools.timing.session`, `docs/measurement.md`), and stores the raw
ordered samples; whether a stored difference is a regression is `bazel run //:results -- verdict`.

```bash
bazel run //measure:session_entmax
bazel run //:measure -- run declare.py:all --only 'fsla/support' --only 'fsla/sass'
```

