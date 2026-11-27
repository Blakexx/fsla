# Measurement — the method, the instruments and the record

Every performance number this repository publishes is taken by one method, with one stopwatch per comparison, and
stored with its provenance. This document is what that sentence means.

| layer | where | answers |
|---|---|---|
| cells | fsla-devtools' central registry `fsla_devtools.cells` (fsla's reading: `measure/cells/`) | WHAT is measured: an input in one form, named once for every package by its own fields, with its draw, seed and proven regime |
| subjects | `measure/subjects.py`, `measure/provider.py` (fsla's runner) | WHICH launch is timed on a cell, at which LEVEL, each arm's dials, its untimed reset, and which cells this binary cannot run |
| timing targets | `measure/BUILD.bazel` (fsla-devtools' `bazel/measure.bzl`: `timing_entry`, `interleaved_timing`) | WHAT IS TIMED: each entry an executor on cells in this tree's bundle, each session a run target over entries |
| instruments and stores | `measure/BUILD.bazel` (fsla-devtools' `bazel/measure.bzl`: `instrument`, `results_store`) | WHAT IS RECORDED of a build: a build output (the static manifests) or an instrument's JSON (the oracle tier on the device), each a run target that stores it unless the store holds it |
| declarations | `declare.py` (a checkout's targets), fsla-bench's `declare.py` (checkouts composed) | WHAT fsla-bench RUNS: each checkout's build, machine facts, diff sides and timing registrations as targets of `fsla_devtools.build`, in fsla-bench's sessions and diffs |
| method | fsla-devtools' `fsla_devtools.timing.session` (`interleave`, the session program) | HOW: interleaved call by call, one call at a time, in a fresh random order each rep |
| preconditions | the session program, `measure/provider.py` | what the box and the binary must be first |
| statistics | fsla-devtools' `fsla_devtools.verdict` | whether a difference is real |
| record | `fsla_devtools.results` (the measurements store) over the data repository, fsla-results | where a number lives afterwards, and which stored samples are a baseline |

fsla times itself with `bazel run //measure:session_entmax` and records its instruments with their store targets
(`bazel run //measure:store_oracle_tier`, `bazel run //measure:store_static_manifest`); fsla-bench's root loads
`declare.py` from each checkout it compares,
registers every checkout's timing entries beside its attention reference, and times them in shared sessions of
fsla-devtools' build system. Both interleave with one loop (`fsla_devtools.timing.session.interleave`): a number taken
either way is the same measurement.

## <a id="entries"></a>Timing entries and their bundles

A TIMING ENTRY is a Bazel target (`timing_entry` in `measure/BUILD.bazel`, this tree's wrapper in `bazel/defs.bzl`): an
executor of `measure/executors.py` (`timed`, one of fsla's arms by name; `read_clock`, the clock reader) on central cells
named as targets (`@fsla_cells//:<cell>`, each the cell's description -- its record and the digest of the code that draws
it, never its tensors). Its BUNDLE is everything its worker imports and reads, as the entry's runfiles: the `fsla` and
`measure` packages, the contract registry, the library with its build record, and the arm tables laid out at
`build/generated/`, where `measure/provider.py` reads a tree's. Beside it the build writes the bundle's manifest: the
commit, branch and uncommitted diff the build was stamped with ([`build-graph.md#stamp`](build-graph.md#stamp)), the
digest of every file the bundle holds, and the environment it runs on.

## <a id="sessions"></a>The method

A session is a run target (`interleaved_timing`): `bazel run //measure:session_entmax` builds every bundle and starts the
session program (`fsla_devtools.timing.session`), which measures and stores, and nothing outlives the run. Its entries --
a tree's arm on a central cell, one of fsla's subjects (`entmax_solve`, `layer_prefill`, `layer_op`, `decode_step`) or
fsla-bench's `flash` -- are set up in worker processes, one a bundle, each on the bundle's interpreter with the bundle as
its whole import path, behind a
barrier (nothing is timed until every entry has set up), each warmed past the floor of 10 calls, and then every rep of
every round calls every entry once, in a fresh random order, one call at a time: the next call is sent only when the
last has replied. Each sample is one launch on the entry's own stopwatch, taken inside its worker, so the pipe between
processes is never in a sample; an entry whose launch changes what its next call reads (a carried state, a decode step)
is reset to exactly what its first call saw before every call, untimed, so every call does the same work on the same
data. Reps are odd, so a round's median is one of its samples. The session keeps every sample in the order taken, with
its round, rep and position; which entry is the reference is chosen when the samples are read, never when they are
taken. Each rep's order is drawn from a seed of the run's own identity, stored with the session, so two runs never walk
one order. An entry that cannot set up (a kernel this binary lacks) is recorded in the session and the rest are timed;
two stopwatches in one session fail it, and so does a session in which nothing set up, naming the first refusal.

The session is stored as ONE sample at its own label (`@fsla//measure:session_entmax`): its semantics are what ran -- each entry's executor, parameters,
cells and bundle digest, the rounds, reps and warm-up, whether the clock was locked, and the card -- and its provenance
each bundle's stamped commit. `-- --no-store` measures and keeps `session.json` in the scratch directory; a session that
fails stores nothing and says where its samples were kept.

A NULL GATE (`measure_null_gate`) times one registration's entries against copies of themselves in second workers of the
same checkout: where the per-rep ratios of the two copies put one outside their interquartile range, a worker's bias is
found, and a ratio between two checkouts' workers on that cell is read beside it.

Call by call, because drift on this host is the size of the effects measured: its sustained clock has two states ~17 %
apart with a minutes-long time constant (measured here, no record cited). Blocks let a clock change land between arms, and a fixed order charges drift to one arm. A timing only compares
within the session that interleaved it.

## Preconditions, which the session acquires

Not a checklist: the session program and the runner's setup do each of these, and no flag skips one.

| step | what happens |
|---|---|
| GPU lock | the GPU lock exclusive through `gpu_lock()`, for the whole session, its marker inherited by the entry workers, so a self-locking tool an entry starts does not wait on it. An exclusive holder never lowers its priority. The lock's rules are [`testing.md#running-it`](testing.md#running-it)'s. |
| host budget | every slot of the host budget, after the GPU ([`build-graph.md#front-door`](build-graph.md#front-door)): no build runs beside a session. |
| clock lock | the machine's clock lock (`fsla_devtools.locks.clock`, the dev config's `clock.json`), engaged by its first holder and released by its last, so another process finishing never unlocks it mid-session; proven by this binary's clock read (`//measure:clock`) before the first call and after the last, and a read off the lock fails the session. A host that declares no lock is refused unless the session claims `unlocked` (`//measure:session_entmax` does: the rented card's container declares none); the record says which, and a locked and an unlocked session are never one record. |
| binary identity | building an arm asks the device for the subject's family stamp and refuses the arm without it: a path or a hash passes against a stale extension, a device-side fact does not. |
| extension identity | each worker reports the file `fsla` and the plugin's extension module were loaded from (the entry's `binds`), and a file outside the entry's bundle refuses the member: an import path that names another tree (a checkout's venv, another output base) otherwise answers with that tree's kernel while the record carries this tree's commit. `import fsla` must also resolve inside the runner's own tree, or the arm is refused. |
| tree identity | every stored sample carries each bundle's stamped commit and the sha256 of its tracked diff; a dirty tree is stamped, not refused, and a git that fails fails the build. |

## The host a measurement runs on

The local card runs no GPU test (`docs/CONSTITUTION.md#r8`, `docs/KERNEL_STANDARDS.md#cards`): master is Hopper only, and
every GPU run -- test, gate, probe and measurement -- is taken on a rented H100. The bullets below are the rules a
session on that card follows; sizing a cell to fit is the same arithmetic there.

* **A session's whole member set fits the card, with headroom.** A session sets up every entry before timing and holds
  them all resident at once, each in its checkout's entry worker, so what must fit is every arm on every cell of the
  group plus the reference's. The card is shared with the host's display, and a near-full one pages and corrupts every
  time in the session without a message. Size a local cell at scale smaller (fewer heads, or a group with one FSLA
  cell), estimate each cell's peak footprint on paper before launch, and keep
  `torch.cuda.set_per_process_memory_fraction` as the backstop.
* **A row near the card's cap is discarded, never caveated.** Under WSL2 the driver falls back to host memory over PCIe
  when VRAM is exhausted (Blake: "if we measure above the vram size, it spills into system memory and then it is just
  going to be slower"), so such a row times PCIe, not the kernel. Each entry's peak memory (a memory pass's sample) is read
  beside its times where a memory pass measured it -- fsla-bench's; fsla's own sessions carry none yet
  ([open-work M-f](open-work.md#measurement-and-gates)); to fill a grid, raise the CTA count without raising memory (more heads or batch at a shorter `L`).
* **A foreign consumer of the GPU inflates every number.** Another context time-slices the card and stretches wall
  time and `clock64` cycles alike (the counter runs while ours is preempted); the clock lock proves the SM clock, not
  that our context owns the device, and under WSL the consumer is invisible to `nvidia-smi --query-compute-apps`.
  Before reading an absolute number, and whenever a whole run shifts uniformly, read the idle utilization
  (`nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv` with nothing launched); not ~0%, stop
  measuring and tell the owner, who alone can find the host-side consumer. The lab driver's own refusal above 5% is
  [`internals/tools/lab.md`](internals/tools/lab.md). The 0x4 clock-event flag under sustained dense load (the card's
  boost power limit) and the constant aperture-size reading are not this: rule out the repository's own processes first.

## The three levels, and what a comparison may cross

Blake, 2026-09-15: the LAYER holds the projections and produces the routing (FSLA's analogue of QKV); the OP takes those
bare operands and a state and produces the next state and the readout -- the facts, the paging, the packing and the
kernels; a KERNEL runs its own launches and nothing around them. Every subject states the level it prices
(`Subject.level`) and every sample records it, so a reading pairs entries of ONE level and
`bazel run //:results -- verdict` refuses a pair of two.
A comparison between this library and another is therefore a LAYER comparison: FSLA's
own layer against attention's, because FSLA's number includes the state its capacity buys.

What this line declares today (`measure/subjects.py`): the KERNEL level (`entmax_solve`), the OP level (`prefill`,
`layer_op`, and `decode_step` through the engine's decode step) and the LAYER level (`layer_prefill`). No prefill kernel is
built (the carry, intra and producer are designed and measured in the lab, [`open-work.md`](open-work.md)), so every
subject that reaches the prefill refuses by name and is recorded as that entry's failure; a layer arm's fixture builds
only what the arm reads (`measure/cells/layer.py`'s `Fixture`), so `entmax_solve` sets up with the producer unbuilt and is
the entry `//measure:session_entmax` times. The flash reference is a layer entry, and until a
FSLA layer runs beside it, a session that holds both is two levels measured in one place, not a comparison.

## The reference

Every reported carry number sits beside FlashAttention at the same point, measured in the
same interleaved, clock-locked session: fsla-bench's attention entry
(`fsla_bench/measure/attention.py`) on the group's QKV cell, causal, width `dv`, bf16, through torch's flash
backend: Dao's FlashAttention-2 compiled into torch, or FA3 (Hopper) and FA4 (Blackwell) once
`torch.nn.attention.activate_flash_attention_impl` registers them. The backend is forced, so torch refuses a call
flash cannot take rather than timing its math or memory-efficient backend, and the entry records torch's version and
the implementation.

<a id="incremental"></a>
THE INCREMENTAL CELL is the headline (Blake, 2026-10-18): the time to process `c` NEW tokens given `T` tokens already
absorbed, for a model of state count `N`. FlashAttention's `c` queries attend over a KV cache of `T + c` keys,
bottom-right causal (the entry's `context` parameter); FSLA's layer runs the `c` tokens' producer, carry
from its `N`-leaf state and intra (none of them built; this is the design). FlashAttention's cost grows with `T`; FSLA's does not once its state holds the
history, so the comparison is a grid -- `T` in {0, 4K, 16K, 64K, 256K, 1M} by `c` in {1, 512, 4K, 16K}, the model's
hidden width and heads stated -- never one number. Each side's per-call overhead is in its time: FlashAttention's KV
read and launch; FSLA's state load and store (its HBM round trip, projected from the state's bytes), launch, partial last
window and wave quantization.

AT LAYER LEVEL, WITH BOTH SIDES' PROJECTIONS: FlashAttention's QKV projection of the `c` tokens (the entry's `hidden`:
one cuBLAS GEMM for every head, the library's own form) against FSLA's producer (its routing, gain and values: its
wider per-head outputs against QKV's 192), so a number is what a user pays. The output projection is the same on both sides and
is in neither.

`N = L` -- `T = 0`, `c = L`, the whole sequence from zero tokens -- is the grid's whole-sequence case and the
capacity-fair point a single-length cell states (`N = L`); it averages FlashAttention over contexts from 0 to `L` and
holds one FSLA call's overheads, so it is never quoted alone. Until FSLA's carry starts from a stored state (the
segment boundary), FSLA's side of the grid is projected from the labs, the producer's floor and the state's bytes, each
number labelled projected.

<a id="quoting"></a>
## Quoting a comparison

- THE BASELINE is attention over its KV cache (and wide monolith layers), whose state grows with the context -- not
  single-state linear attention: the method's case is a bounded state against a growing cache, and at `N`
  proportional to `L` the state stays under the cache while its key width is small, so that regime's problem is
  compute, which the activation's sparsity decouples (the user's framing, 2026-07-01).
- CAPACITY-FAIR POINTS (Blake, 2026-08-01): a gap is never quoted from a cell where FSLA holds `N` states while
  FlashAttention attends over `L < N` tokens -- attention is quadratic in `L` and FSLA flat in it at fixed `N`, so the
  cell charges FSLA capacity rent at short-context prices. Headline figures put measured points at capacity-fair
  `(N, L)`, FlashAttention run at `L = N`, and share one x-axis, addressing capacity; an `L < N` cell extrapolates
  per-token latency only.
- EVERY RATIO CARRIES ITS LABELS (Blake, 2026-10-24): the card, the level (op or layer), the depth and the sparsity.
  The headline is end-to-end layer time with the op-level number beside it, at a head at scale, depths 1-4 shown; a
  GeForce op-level ratio is never the headline.
- THE OPERATING POINT `N ~ sqrt(L)` is quoted beside `N = L`, never instead (Blake, 2026-10-31: "sqrt(L) is context,
  not the bar"); `N = L` stays a target the kernel must do well at.
- AT SCALE (Blake, 2026-07-30: "we should be testing the large regime too"): every performance-bearing change is timed
  on at least one at-scale cell (`N >= 65536`, the largest `L` that fits) at both ends of density
  (`KERNEL_STANDARDS.md#s21`). The small cell is the SMOKE cell, never "flagship"; a conclusion drawn on toy cells
  alone is labelled cell-bound. A small state is a depth-1 cell (Blake, 2026-09-14: "If you want small, just use depth
  1"), never a sub-floor depth-2 rung.
- RECORDS FOLLOW THE DECLARATION: when a refactor changes what a record means, the change is re-baselined and the old
  records deleted rather than kept readable through compatibility code (Blake, 2026-09-15: "its ok to just delete old
  records if its too much work to maintain compatability"); what was deleted is said, and anything large is archived
  outside the store first.

## The stopwatch

There is one, it is named, and its name travels with every number.

| instrument | what it measures |
|---|---|
| **`cuda_events`** (committed) | device-side elapsed time between two events straddling the launch |
| `perf_counter` | host wall time around a full device sync on both sides — the launch and sync cost included |
| `ncu` | a counter, or a duration taken under profiling conditions: replayed, caches flushed, base clocks |

**A comparison across two of them is refused, not warned about.** The gap is not
noise; it is the launch and sync cost one instrument includes: roughly constant on a given
box, so nearly the whole measurement on a sub-millisecond cell
and a rounding error at scale. An `ncu` duration reads above the event time (25 % on a
small cell), which is why it is a third instrument and not a cross-check on the first
two.

Every timed entry names its stopwatch (`Timed.instrument`; fsla's arms time with `cuda_events`), and a session refuses
entries that name two.

<a id="comparing-two-builds"></a>
## Comparing two builds

A change is judged against its baseline in ONE session: the baseline checkout (a worktree at the old commit, built,
with its venv) is a reference of fsla-bench's suite, and the run is pruned to the timing sessions and their stores:

```bash
python -m fsla_devtools.build run declare.py:suite \
    --arg target=worktree:<new checkout>,venv:<its venv> --arg references='worktree:<baseline>,venv:<its venv>' \
    --only 'store/session/*'
```

Each session interleaves both checkouts' entries call by call at the locked clock (above), so the ratio between them
is read from one session's samples (`bazel run //:results -- verdict`). Times from two separate runs are never
compared, and an unlocked clock's are never read at all (KERNEL_STANDARDS §32.7): this host's sustained clock has two
states ~17% apart (the method, above), larger than the effects a change is judged on. One launch's cycles each under `ncu`
(`sm__cycles_elapsed.max`) compare two builds without a session where the question is one kernel's cycles.

## The flagging rule: three gates, all must fire

`fsla_devtools.verdict` judges a candidate against a reference timed in the same sessions; which stored sessions and
which reference it is given is the store's query (below). Timing compares only within the session that interleaved it,
so every gate reads within-session quantities: `session(candidate, reference)` takes the two members' samples by round
and gives, per round, each member's median, their RATIO and their DIFFERENCE.

1. **Effect size** — `ratio_limit`: the session's median ratio over `1 + 3·IQR/1.349` of its own per-round ratios.
   `ALARM_SIGMA = 3.0` is the only free choice, and it is stated as a false-alarm rate (~0.1 % one-sided per cell per
   session), never as a percentage of anything.
2. **Significance** — `paired_verdict`: an exact Wilcoxon signed-rank test over the session's per-round differences,
   candidate minus reference, at `ALPHA = 0.01`. **The round floor is derived, not chosen**: the smallest two-sided
   exact p reachable with `n` non-zero differences is `2/2ⁿ`, so below `ceil(log2(2/ALPHA)) = 8` rounds no outcome can
   be significant and the test is undefined rather than underpowered. Sessions run 8; fewer report `insufficient_data`.
3. **Persistence** — `persistence`: a regression needs a trailing run of at least two sessions over their own limits, at
   the same code on both sides. One thermally unlucky session on a box with logged power capping is not a kernel change.
   A run of two that has since healed is `suspicious`, which is not a merge blocker and is not silence either.

`insufficient_data` is a distinct answer from `no_regression` everywhere: the first says the design cannot decide, the
second says it decided.

**THE SPREAD IS THE SPREAD OF THE QUANTITY JUDGED.** Reps taken back to back on one warm fixture spread less than a
comparison that crosses processes (a limit taken from one session's reps flagged most arms of an unchanged binary a
minute later), and a reference's medians across sessions carry the host's drift (two clock states ~17 % apart). What
is judged is the paired ratio inside one session, so the limit is that ratio's own round-to-round spread, drift both
members share cancels in it, and the one thing the ratio crosses that its spread
cannot see -- two worker processes -- is the null gate's.

**AND NEVER A SPREAD SMALLER THAN THE STOPWATCH RESOLVES.** The device-event timer quantizes, so a small kernel's calls
can land on one value, a recorded spread of `0.0000 ms`, and a limit at the median the next tick exceeds. The spread
used is `max(IQR, resolution)`, where the
resolution is the smallest gap between the distinct values one member's samples hold, relative to the reference's
median -- measured on every judgement, because it is a property of the stopwatch and the box -- and the verdict says
when it bound (`floored_at_resolution`).

## Which samples are judged

    bazel run //:results -- verdict --reference LABEL [--candidate LABEL] [--cell C] [--arm A] [--json]

reads the stored timing sessions (the `timing_members` and `timing_samples` views). In every session that timed the
reference label's arm on a cell, each other label's member of that arm on that cell is paired with it and judged inside
the session. A unit is the arm, the cell, the two labels and the code each ran (commit and tracked diff); its sessions,
oldest first, are the runs, and the newest is judged. Nothing is stored: a verdict is a reading of the records, taken
again whenever it is asked for. `bazel run //:results -- dashboard --out FILE --reference LABEL` is one run as a page.

## SASS

`bazel run //:ratify -- --census` and `bazel run //tools:sass_bodies` are the instruments. Two binaries in a
comparison may be built at different paths ([`build.md#kernel-identity`](build.md#kernel-identity)); they must agree
on `ratify.SASS_HASH_VERSION`, which `--compare` checks and refuses on mismatch.

**Measuring a kernel's register DEMAND at a bound it does not ship at** is a
scratch-tree edit — copy the tree, change the `__launch_bounds__` CTA argument at
the instantiation, run `bazel run //:ratify -- --census` there, read the spill
column — and never a flag on the shipped tool, because a tool that can force a
bound is a tool that can report a number nobody ships.

## What is deliberately not here

* **A second method.** One loop times every subject (`fsla_devtools.timing.session.interleave`), whether a Bazel
  session or fsla-bench's declared one runs it: two methods are how a harness comes to disagree with itself.
* **A committed SQLite database.** A binary blob has no diff and cannot be reviewed in a landing. Any query artifact is
  derived and ignored — see the record below, which is exactly this design.
* **A flat percentage threshold.** The IQR-derived line is better than every rule in the surveyed field; replacing it
  would reintroduce the invented number the derivation exists to avoid.

## fsla-bench's declarations: a checkout's targets, a run pruned by label

A checkout's `declare.py` declares, for fsla-bench's root, every target the bench composes of it -- its build (the
build graph's `bazel run //:library`, run every build: Bazel's cache is its key, and the library's digest, its output,
keys every target that reads the binary), its machine facts, its timing arms, its diff sides and its own
kernel-vs-oracle diff -- and takes no selector. Which of it a build runs is pruned at the CLI by LABEL, which knows
nothing of cells or groups (`--only`, `--skip`). A node whose code, binary and inputs have not changed is a cache hit and
does not run.

<a id="the-diffs"></a>THE DIFFS. A SIDE is an executor run over cell nodes in the environment it names, its tensors kept in the build cache;
a DIFF is two sides under one named rule -- `bit-identical`, the per-slot clause rule mirroring the oracle's, or
`support-equal` -- with a claim (`expect` same or different) and a consequence (a build failure or a recorded verdict).
A side proves which checkout's library it imported; a comparison that compared fewer quantities than it declared, or a
cell only one side could produce, refuses rather than passes; a cell both sides refuse is listed and not counted. Inside
Across checkouts fsla-bench pairs each surface a checkout exposes (`SURFACES` in `declare.py`: the fp64 oracle and the
producer reference) under ITS rule for the surface (`RULES`) -- the oracle bit-identical, the producer support-equal --
stored at `diff/<surface>`. Verdicts are stored, tensors never. The carry kernel has no diff surface and no
kernel-vs-oracle diff: its oracle obligation is owed (`tools/coverage.json`).

## <a id="instruments"></a>The instruments

AN INSTRUMENT is a tool's JSON over the files it reads (fsla-devtools' `instrument`, this tree's wrapper in
`bazel/defs.bzl`): the tool runs in a tree of copies of those files, and one that cannot run on what it was handed
writes `{"refused": WHY}`, which is its output and stops nothing else, while one that fails fails. Without a device it
is an action, its JSON a build output; ON THE DEVICE (`device = True`, keyed on the devices' identity) it runs in the
run phase of the store target that records it, never in a build, since a build holds the host budget and the GPU taken
under it deadlocks against a measurement, which takes the GPU first ([`build-graph.md`](build-graph.md#front-door)).

A STORE TARGET (fsla-devtools' `results_store`) records a build output: `bazel run` stores each of its files as a sample
of the record its semantics key -- the source's label, the file's name and the content digest of every input it depends
on (an instrument's tool, files and devices; the `inputs` it names) -- with the stamped commit as its provenance, and asks
the store first: when the record already holds this outcome nothing is stored, so a second run over the same build
stores nothing, into whichever store it is pointed at. A refusal is stored as a failed sample naming why.
`-- --store-root DIR` stores into another records directory, `--no-store` runs a device instrument and keeps its JSON.

| target | records | location | keyed by |
|---|---|---|---|
| `//measure:store_static_manifest` | each built architecture's static manifest (`//:static_manifest`, warpline's, the build output the SASS gate's comparison judged: [`internals/tools/sass_gate.md`](internals/tools/sass_gate.md#manifest)) | `fsla/static-manifest` | the library and warpline's checker |
| `//measure:store_oracle_tier` | the oracle tier on the device (`measure/oracle_tier.py`): the counts, WHICH cases failed (a red case is a stored fact), every per-slot comparison's margins (`--oracle-margins`) and the worst per output kind; a host without a CUDA device refuses | `fsla/oracle-tier` | the tool, the library, the tree and the devices |

The build's SASS signatures (KERNEL_STANDARDS §20) are no instrument: the library's SASS gate judges them on every build.
The realized support (`tools/realized_support.py`) has no target: the producer it reads is not built on this line.

A question about the machine itself, rather than a shipped kernel, is a lab experiment ([`lab/`](lab/README.md)).

## The measurements record: `fsla_devtools.results`

Every measuring tool stores what it measured through one library, fsla-devtools' `fsla_devtools.results`, into the
data repository, fsla-results, which holds data and no code: `store.root` in the dev config (`bazel run //:init` names
the `fsla-results` checkout beside the checkouts), its `records/` the store's backend. A result measured in any
worktree of a machine lands in that one checkout, visible to every other at once; the devbox container has its own
clone (`docs/setup.md#the-devbox-image`), whose records reach another machine's by git (`bazel run //:results --
commit`, then a push and a pull). The store refuses a sample whose stored output or provenance holds a home
directory's path (`/home/<user>`, `/Users/<user>`, `/root`, a Windows user's directory), and a timing sample whose
provenance does not name, for every checkout it ran, that checkout's commit (`checkouts.<owner>.git_sha`).

<a id="the-data"></a>THE DATA THIS TREE READS AND CITES is the data repository at ONE commit, pinned in `MODULE.bazel`
(fsla-devtools' `environment.data` tag, `@fsla_data`, by the commit's archive digest), never the live checkout: a doc
cites a record by a sample's output, `data:records/<workspace>/<package>/<target>/<run>/output.json` (fixed bytes: a
stored sample is never rewritten), which `//:docrefs` resolves inside the pin, and the tools that read a record -- the lab's pages and its
`trace`/`path`, the floor model (`tools/carry_floor.py`), the rates' check (`measure/rates.py`) -- read it there, handed
its root by the graph (`FSLA_DATA`, `measure/data.py`). A LAB RECORD is read through one function, `lab_record(experiment,
arch)` (with `blocks` or `variant` where the question is another's): the newest sample of the run's record its question
keys (`lab_semantics`: the experiment, the architecture, the kind -- `run`, `counters`, `ledger` -- the grid it was held to
and a named condition), its traced rows or a ledger's dumps the sample's parts. `bazel run //measure/lab:NAME.run`
stores a new sample into the store's checkout (at its own label), never this tree, and it becomes citable when a commit of
the data repository holding it is pushed and the pin moves to it. The records the lab kept as plain files before it
stored through the store are BACKFILLED samples, each file's bytes its output unchanged, marked `backfilled` with the
file's old path.

A tool opens a store at the LABEL of the target that produced what it stores (`@<workspace>//<package>:<target>`,
a folder of the data's `records/`: the store is a durable index of what Bazel's targets produced) and hands it the
SEMANTICS of what it measured -- the inputs its numbers depend on, reduced to identities (a commit and its tracked diff,
a binary's manifest or file digest, the cells, the counts, the device's software, the clock lock). A SAMPLE is one file,
named by its run and written once; the KEY is sha256 over its semantics, so the same measurement taken again is another
sample of the same key and a changed input is another key. A sample is the raw output or the
failure, when, how long, and its PROVENANCE (the checkouts by directory name, branch and commit, the stage, the host).
Every sample is kept, failures included. Nothing relational is stored: a ratio against a baseline is the reader's,
and a timing only compares within the session that interleaved it.

| label | writer | the semantics | a sample's output |
|---|---|---|---|
| `@fsla//measure:store_static_manifest` | `bazel run //measure:store_static_manifest` | the source, the manifest's name (its architecture) and the content digests of the library and warpline's checker | the manifest |
| `@fsla//measure:session_entmax` (a session's own) | the session program (`bazel run //measure:session_entmax`) | the session's semantics: every entry's executor, parameters, cells and bundle digest (fsla-bench's: every registration's executor, cells, code and binary), the timing parameters, the clock's state, the card | every sample in order with its round, rep and position, each entry's status, the seed, the clock reads |
| `@fsla//:container_check` | `bazel run //:container_check` | the devbox image's digest | the steps, each timed, and the cold start |
| `@fsla//measure:store_oracle_tier` | `bazel run //measure:store_oracle_tier` | the instrument, its arguments and the content digests of its tool, the library, the tree and the devices' identity | the counts, the failed cases, the margins; or the refusal, as a failed sample |
| `@fsla//measure/lab:<experiment>.run` | `bazel run //measure/lab:<experiment>.run` (`run`, and `-- ncu`) | the experiment, the architecture, the kind, the grid (`blocks`), a variant (`lab_semantics`); the commit, the binary's digest, the toolkit and the device are each sample's provenance | the run's record (the results with their samples and checks, the kernels, the counters) and its traced rows as parts |

Reading across runs is a join, never a parser: `instrument_metrics` is every number every instrument recorded in
long form (run, cell, dotted metric, value), `diff_cells` every stored diff's verdict per cell and quantity, and
`bazel run //:results -- instruments --baseline RUN` is one run's numbers beside another's as deltas; the dashboard
(`--baseline RUN`) renders the same. Which run is the baseline, like which label is the reference, is the reader's.
A result is read from the store (`bazel run //:results -- show`, `history`, `latest`, `instruments`), never from a
terminal print.

A tool run by another tool stores nothing (`--no-record`): the caller keeps the output in its own record.
`bazel run //:results -- sql "..."` queries every location through a derived SQLite index in the machine's cache
(`history` and `latest` for one location; views flatten the stored outputs: `cells`, `diff_cells`, `instrument_cells`,
`instrument_metrics`, `memory_rows`, `null_gates`, `timing_members`, `timing_samples`);
`bazel run //:results -- verdict` is the regression
judgement above; `bazel run //:results -- commit -m <message>` commits the records, after the repository's own check
that every key recomputes from its semantics and every output exists. Pushing is the owner's act.
The record kept before the library -- the per-run JSONL, the perf
ledger, the ledger and composer reports -- left the store for the owner's archive (`docs/internals/DELETIONS.md`).
