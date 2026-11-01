# `tools/` — the map: what each entry point is, who calls it, and what it records

The tools inventory. Every script under `tools/` is here with one line of what it is, what runs it, and
what it writes; a tool with more to say has its own page beside this one, and a tool without one is fully stated by its
own module docstring. A tool that nothing runs is a tool to delete (`DELETIONS.md`), so the CALLED BY column is the
column that matters.

**No tool writes a measurement record except through a declared target.** A measurement is a target of
`declare.py` (`rola_devtools.build`), its result is stored by a store target beside it, and the tool itself prints or
writes `--json`. The two exceptions are named below: `compose_ledger` is a driver that builds what it measures, and a `--calibrate`
run of `pipe_timeline` writes the plateau it reads back as every later run's scale.

## The build

| tool | what it is | called by | writes |
|---|---|---|---|
| [`dev.py`](dev.md) | the one command that sets up, checks and describes a host or container | a person; the commit gate's §23 check | the `environment` record (its container check) |
| [`dev_config.py`](dev_config.md) | every machine-dependent value, read from one directory of JSON files | every tool here, `setup.py`, the tests | — |
| [`build_flags.py`](../../build.md) | THE ONE SOURCE of the nvcc flag list | `setup.py`, `ratify.py` | — |
| [`build_lock.py`](build_lock.md) | the machine-wide cicc budget | `setup.py`, `ratify.py` | — |
| [`gen_shards.py`](gen_shards.md) | THE ONE ENUMERATION of the carry family's instantiation matrix | `setup.py`, `ratify.py`, the provider | the generated per-arm units and selection header |
| [`toolchains.py`](toolchains.md) | each toolchain as a named record of the assembler a fatbin is built and ratified under | `setup.py`, `ratify.py`, `dev.py` | — |
| [`sccache_toolchain.py`](../../build.md#sccache) | the compiler-cache wrapper pin (closed-world rule 1) | `setup.py`, `ratify.py` | — |
| `sccache_nvcc.py` | the nvcc↔sccache seam: a pass-through, never a resolver | nvcc, as its launcher | — |
| `mold_toolchain.py` | the linker pin — mold is not an option, it is THE linker | `setup.py`, `dev.py`, the Dockerfile | — |
| [`wheels.py`](wheels.md) | one gate build of a clean commit, split into the pure-Python wheel and the binary plugin | a person, at a release | the wheels and their record |
| `supported.py` | the README's supported-configuration table — a REPORT, not a claim | a person, after an arm change | the table |
| [`kernels.py`](kernels.md) | THE KERNEL LIST: every device unit the build compiles and every kernel the source defines, derived from the tree | `setup.py`, `ratify.py`, every kernel gate | — |
| `build/dag_to_dot.py` | a DAG tuple (`rola.engine.dags.*`) as Graphviz DOT and SVG | a person, after a DAG change | the `.svg` beside its page |
| `build/ninjatracing.py` | `.ninja_log` as a Chrome/Perfetto trace, the compile ledger's visual form | a person | the trace |
| `git-hooks/pre-commit` | THE COMMIT GATE, run from the checkout's venv against `.pre-commit-config.yaml` | git, every commit | — |

## Ratification and the assembled code

| tool | what it is | called by | writes |
|---|---|---|---|
| [`ratify.py`](ratify.md) | the spill manifest, the census, the SASS-body gate and the derivation | `setup.py` (post-build), a person at a milestone | `tools/manifests/` |
| `sass.py` | THE DISASSEMBLER SEAM: the one `cuobjdump`/`nvdisasm` invocation and line grammar; a frame map is one function's (`section`: offsets restart in every function) | `ratify.py`, `sass_gate.py`, `region_ledger.py`, `life_ranges.py` | — |
| `sass_gate.py` | the ptxas signatures of KERNEL_STANDARDS §20, red on any of them; guarded convergence fallbacks counted | every iteration build; the `sass` instrument | `--json`, `--loops` (each step loop's reading and the lines its lints count) |
| `sass_bodies.py` | per-instantiation SASS body hashes, the codegen-equivalence instrument | a person, comparing two builds | `--json` |
| `sass_control.py` | the SASS control words (stall count, yield, scoreboards, reuse) and the loop a function compiles to; its `MACHINES` table is the static simulator's per-architecture model | `pipe_sim.py`, `burst_gate.py` | — |
| [`burst_gate.py`](burst_gate.md) | what every burst loop's SASS must satisfy, deterministically (KERNEL_STANDARDS §23) | the `sass` instrument | exit 1 when red |
| [`pipe_sim.py`](pipe_sim.md) | a burst loop's steady-state tensor-pipe occupancy from its SASS alone | a person, before a kernel runs | `--json` |
| `header_selfcheck.py` | every header compiles alone | a person, after moving a declaration | — |

## The instruments (each a target of `declare.py`)

| tool | what it is | called by | writes |
|---|---|---|---|
| [`phase_ledger.py`](phase_ledger.md) | one carry launch with the kernel's phase clock bound | the `phases` target | `--json` |
| [`pipe_counters.py`](pipe_counters.md) | the profiler's pipe and resource counters for one launch | the `counters` target | `--json` |
| [`pipes.py`](pipes.md) | one counter set for any launch and its pricing per card: each pipe's cycles, the binding one | `lab.py`, `pipe_counters.py` | -- (a module) |
| [`carve_model.py`](carve_model.md) | the carry's mechanics for any carve on any card, over a real cell's lists (KERNEL_STANDARDS §32.2) | -- (in build: the carve study) | prints |
| [`overlap_gate.py`](overlap_gate.md) | each kernel's perf rows against the largest of their floors (§28) | -- | exit 1 when red |
| [`stall_census.py`](stall_census.md) | every warp-stall sample of one launch, by source line | the `census` target | `--json` |
| [`pipe_timeline.py`](pipe_timeline.md) | the pipes' activity over one launch, measured on silicon | the `timeline` target | `--json`; a `pipe_timeline.scale` record under `--calibrate`, which it reads back as later runs' scale |
| [`life_ranges.py`](life_ranges.md) | peak LIVE registers per source region, beside ptxas's allocation | the `registers` target | `--json` |
| `region_ledger.py` | executed instructions and stall samples per COMPONENT, against `tools/budgets/` | a person, attributing a phase | `--json` |
| [`compose_ledger.py`](compose_ledger.md) | a phase's time attributed to its parts by composition — a DRIVER: it builds each part | a person, after a component lands | its own `compose_ledger` record |
| `roofline.py` | the intra's wall time on one cell beside the device's own u8 IMMA ceiling — the fraction of the machine, not the wall time | the `roofline` target | `--json` |
| [`budgets/`](budgets.md) | each kernel's instruction budget by class, per MMA (KERNEL_STANDARDS §29) | `region_ledger.py`, the lab | — |
| [`bank_model.py`](bank_model.md) | the shared-memory wavefronts a warp instruction costs, from its lanes' byte addresses (§30) | a person, before a GPU run; the lab | — |
| [`lab.py`](lab.md) | the lab: small SASS-gated CUDA experiments whose measurements become kernel rules ([`docs/lab/`](../../lab/README.md)) | a person | `measure/lab/results/` |
| `nvbit/warp_trace/` | an NVBit tool: every dynamic instruction of one CTA's warps to a file | a person, tracing a schedule | the trace |

## The oracle's own gates

| tool | what it is | called by | writes |
|---|---|---|---|
| `sanitize_oracle.py` | the compute-sanitizer gate on the facts, intra, decode, producer and entmax oracle cells | a person, in the sanitizer lane | — |
| `sanitize_cell.py` | one cell, one launch, no test framework — the racecheck lane's driver | `sanitize_oracle.py` | — |
| [`docrefs.py`](docrefs.md) | every citation a doc makes, pinned to the content it was read against | the commit gate | `docs/references.lock.json` |

`tools/lint/` is its own family with its own page ([`lint.md`](lint.md)): the ratchets, the standards lint and their
baselines, and the coverage declaration `tools/coverage.json` ([`coverage.md`](coverage.md)).
