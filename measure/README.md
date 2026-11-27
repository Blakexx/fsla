# `measure/` — fsla's reading of the central cells, the benches over them, and the timing entries and diff sides over them

The methodology, the instruments and the record are `docs/measurement.md`; the lock rule
is `docs/testing.md`. This file is the map.

**THE CELLS ARE FSLA-DEVTOOLS' (`fsla_devtools.cells`), and there is one registry for every package.** A cell is an
input -- a carry cell's routing amplitudes, gain, values and the state it enters with; a layer cell's hidden states and
values, and the routing and logit gain its layer is built with -- with its draw and the regime it proves, every kind in
one form (`fsla_devtools.cells.form.Cell`) and named by its own fields (`carry-alt-k4-256x256-T1024`,
`layer-alt-entmax1.5-64x64-T2048`), so a name means one input everywhere it appears, in fsla and in the libraries fsla
is compared against. A cell's routing is stated in `fsla.Routing`'s terms: the topology, a write entry a level, a read
tuple a read head. A cell names nothing fsla runs it with. `measure/cells/` is fsla's reading:

* `cells/__init__.py` — a central cell's record by name (`by_name`), the carry cells by tier (`carry_cells`) and a carry
  cell's draw (`realize`). What the carry kernel is given for a cell is built where it is timed (`subjects.bundle_of`).
* `cells/layer.py` — the FSLA layer a layer cell builds (`routing`: the cell's entries as an `fsla.Routing`; the cell's
  write heads, group, value width and gain), from which the amplitudes are PRODUCED. It is the only way to price the
  producer's own solve or a decode step through the layer. A layer arm's fixture builds each part (the layer, the
  inputs, the produced routes) when the arm first reads it.

**`BUILD.bazel`, `entries.bzl` — THE TIMING ENTRIES AND SESSIONS, THE DIFF SIDES, THE INSTRUMENTS AND THE RECORDS'
STORES.** `entries.bzl` declares one `timing_entry` an arm and a cell (`<arm>_<cell>`): every subject of `subjects.py`
on every central cell of its kind it takes (`@fsla_cells//:cells.bzl`), each an arm of `provider.py`'s `timed` in this
tree's bundle (`bundle`: the Python below, the library, the arm tables laid out where `provider.py` reads them), and one
`diff_side` a surface of `tests/oracle/sides.py` (`side_oracle`, `side_producer`, `side_carry-kernel`) in the sides'
bundle. `clock` is the clock reader (`provider.py`'s `read_clock`). Each `interleaved_timing` is a session over entries,
measured and stored by `bazel run` (`bazel run //measure:session_entmax`: the entmax solve on the pinned layer cell),
and `memory` is every entry's memory pass. fsla-bench consumes these entries and sides of two commits as bundles
(docs/measurement.md#bundles). `oracle_tier` is an instrument on the device (`oracle_tier.py`: the oracle battery's
counts, failed cases and margins), and each `results_store` records a build output or an instrument:
`store_static_manifest` the library's static manifests (`//:static_manifest`), `store_oracle_tier` the oracle tier, run
on the device first (docs/measurement.md#instruments).

**`measure/lab/` — THE LAB.** Small CUDA experiments on the machine itself (what a pipe costs beside the tensor pipe,
what ptxas does to a pattern), each a `experiments/<name>.cu` on the kit `lab.cuh`, compiled, SASS-gated, timed and
stored by `tools/lab.py` (each experiment's run target, `//measure/lab:<name>.run`) through the results store into the
data repository's checkout (at the run target's label, a sample of the record its question keys); a KERNEL_STANDARDS rule about the
machine cites a sample's output by its path in that repository (docs/internals/tools/lab.md). `measure/data.py`'s
`lab_record` is the one reader of a lab record: the newest sample. `measure/data.py` finds the pinned data
repository every reader takes the records from, its root handed in by the graph (`FSLA_DATA`).

**`measure/` — the library.** `subjects.py` is the roster: one lean callable per
kernel this line carries, each taking a registry cell and returning the launch to time,
with everything the launch does not pay for built outside the timed callable. The roster
mirrors the oracle roster one for one — a bench roster that does not match the correctness
roster is a roster with kernels nobody measures — and it includes the carry, whose body
does not exist on this line and whose arm therefore refuses by name rather than reporting
a number for something that did not run. `provider.py` is fsla's runner: given a central cell
it offers the subjects this binary runs on it, each at its call counts (`prefill@calls=4`), or refuses the cell by a
fact of the cell or the binary, never by its name; arms build in their own checkout's venv.

**A timing session** interleaves timing entries of fsla checkouts, and of other libraries, call by call under one
stopwatch and the GPU and clock locks (`fsla_devtools.timing.session`, `docs/measurement.md`), and stores the raw
ordered samples; whether a stored difference is a regression is `bazel run //:results -- verdict`.

```bash
bazel run //measure:session_entmax
bazel run //measure:store_oracle_tier
```

