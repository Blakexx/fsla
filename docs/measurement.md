# Measurement — the method, the instruments and the record

Every performance number this repository publishes is taken by one method, with one stopwatch per comparison, and
stored with its provenance at the label of the target that produced it. This document is what that sentence means.

| layer | where | answers |
|---|---|---|
| cells | fsla-devtools' central registry `fsla_devtools.cells` (fsla's reading: `measure/cells/`) | WHAT is measured: an input in one form, named once for every package by its own fields, with its draw, seed and proven regime |
| subjects | `measure/subjects.py`, `measure/provider.py` (fsla's runner) | WHICH launch is timed on a cell, at which LEVEL, its untimed reset, and which cells this binary cannot run |
| timing targets | `measure/BUILD.bazel`, `measure/entries.bzl` (fsla-devtools' `bazel/measure.bzl`: `timing_bundle`, `timing_entry`, `interleaved_timing`, `measure_memory`, `diff_side`) | WHAT IS TIMED: each entry an arm on a cell in this tree's bundle, each session or memory pass a run target over entries; each diff side a surface on its cells |
| instruments and stores | `measure/BUILD.bazel` (fsla-devtools' `bazel/measure.bzl`: `instrument`, `results_store`) | WHAT IS RECORDED of a build: a build output (the static manifests) or an instrument's JSON (the oracle tier on the device), each a run target that stores it unless the store holds it |
| bundles | fsla-devtools' `bundle` tag and `exported_entry`, fsla-bench's `measure/` | WHAT fsla-bench RUNS: this package's entries and sides of two commits, each built in its own workspace by its own pins, in the bench's sessions, null gates and diffs |
| method | fsla-devtools' `fsla_devtools.timing.session` (`interleave`, the session program) | HOW: interleaved call by call, one call at a time, in a fresh random order each rep |
| preconditions | the session program, `measure/provider.py` | what the box and the binary must be first |
| statistics | fsla-devtools' `fsla_devtools.results.verdict` | whether a difference is real |
| record | `fsla_devtools.results` (the measurements store) over the data repository, fsla-results | where a number lives afterwards, and which stored samples are a baseline |

fsla times itself with `bazel run //measure:session_entmax` and records its instruments with their store targets
(`bazel run //measure:store_oracle_tier`, `bazel run //measure:store_static_manifest`); fsla-bench consumes this
package's entries of two commits as BUNDLES -- each commit built in its own workspace by its own pins -- and times them
beside its attention reference in sessions of the same rules. Every session runs one program
(`fsla_devtools.timing.session`): a number taken either way is the same measurement.

## <a id="entries"></a>Timing entries and their bundles

A TIMING ENTRY is a Bazel target (`timing_entry`, this tree's wrapper in `bazel/defs.bzl`): an executor of
`measure/provider.py` (`timed`, one of fsla's arms by name; `read_clock`, the clock reader) on central cells named as
targets (`@fsla_cells//:<cell>`, each the cell's description -- its record and the digest of the code that draws it,
never its tensors). `measure/entries.bzl` declares one entry an arm and a cell, `<arm>_<cell>`: every subject of
`measure/subjects.py` on every central cell of its kind, a decode step on the cells that take steps, selected by the
cells' own fields (`@fsla_cells//:cells.bzl`), never by a name's spelling; an arm this line does not build records its
refusal when a session sets it up. The entries share ONE BUNDLE (`//measure:bundle`, `timing_bundle`): everything a
worker imports and reads, as runfiles -- the `fsla` and `measure` packages, the contract registry, the library with its
build record, and the arm tables laid out at `build/generated/`, where `measure/provider.py` reads a tree's -- and its
manifest: the commit, branch, checkout name and uncommitted diff the build was stamped with
([`build-graph.md#stamp`](build-graph.md#stamp)), the digest of every file the bundle holds, and the environment it runs
on. `fsla` and the extension must load from inside the bundle (its `binds`), which the session checks of every member.

## <a id="sessions"></a>The method

A session is a run target (`interleaved_timing`): `bazel run //measure:session_entmax` builds every bundle and starts the
session program (`fsla_devtools.timing.session`), which measures and stores, and nothing outlives the run. Its entries --
a tree's arm on a central cell, one of fsla's subjects (`prefill`, `entmax_solve`, `layer_prefill`, `layer_op`,
`decode_step`) or fsla-bench's `flash` -- are set up in worker processes, one a bundle, each on the bundle's interpreter
with the bundle as its whole import path, behind a barrier: nothing is timed until every entry has set up. A worker that
draws a cell with other code than the cell's description names, or loads a bound module from outside its bundle, fails
that member. Each member is warmed (`warmup`, 10 calls, the program's floor), and then every rep of every round calls
every member once, in a fresh random order, one call at a time: the next call is sent only when the last has replied.
Each sample is one launch on the entry's own stopwatch, taken inside its worker, so the pipe between processes is never
in a sample; an entry whose launch changes what its next call reads (a carried state, a decode step) is reset to exactly
what its first call saw before every call, untimed, so every call does the same work on the same data. A session runs 8
rounds of 11 reps; reps are odd, so a round's median is one of its samples. The session keeps every sample in the order
taken, with its round, rep and position; which entry is the reference is chosen when the samples are read, never when
they are taken. Each rep's order is drawn from a seed of the run's own identity, stored with the session, so two runs
never walk one order. An entry that cannot set up (a kernel this binary lacks), or that fails a call, is recorded in the
session and the rest are timed; two stopwatches in one session fail it, and so does a session in which nothing set up,
naming the first refusal.

The session is stored as ONE sample at its own label (`@fsla//measure:session_entmax`): its semantics are what ran --
each entry's executor, parameters, cells, bundle digest and environment, the rounds, reps and warm-up, the claim, whether
the clock was locked, and the machine's devices -- and its provenance each bundle's stamp. `-- --no-store` measures and
keeps `session.json` in the scratch directory (`--scratch DIR` names it, `--store-root DIR` stores elsewhere); a session
that fails stores nothing and says where its samples were kept.

A NULL GATE (`interleaved_timing(null_gate = True)`: fsla-bench's `null_<group>_<arms>`) times a session's entries as
two copies of each, every copy in a worker of its own (`<owner>#a`, `<owner>#b`), stored at its own label: where the
per-rep ratios of the two copies put one outside their interquartile range, a worker's bias is found, and the verdict
reports such a pair `untrusted` and does not judge it. A MEMORY PASS (`measure_memory`, `bazel run //measure:memory`)
takes each entry on each cell alone: the caching allocator's peak over its calls and what it holds outside it, stored at
its own label (`@fsla//measure:memory`).

Call by call, because the effects measured are the size of this host's clock drift: within one rep every member meets
the same clock, where blocks let a clock change land between arms and a fixed order charges drift to one arm. A timing
only compares within the session that interleaved it.

## Preconditions, which the session acquires

Not a checklist: the session program and the runner's setup do each of these, and no flag skips one.

| step | what happens |
|---|---|
| GPU lock | the GPU lock exclusive through `gpu_lock()`, for the whole session, its marker inherited by the entry workers, so a self-locking tool an entry starts does not wait on it. An exclusive holder never lowers its priority. The lock's rules are [`testing.md#running-it`](testing.md#running-it)'s. |
| host budget | every slot of the host budget, after the GPU ([`build-graph.md#front-door`](build-graph.md#front-door)): no build runs beside a session. |
| clock lock | the machine's clock lock (`fsla_devtools.locks.clock`, the dev config's `clock.json`), engaged by its first holder and released by its last, so another process finishing never unlocks it mid-session; proven by the clock reader (`//measure:clock`, this binary's read of the SM clock) before the members set up and after the last call, and a read off the lock fails the session. A host whose dev config declares no lock is refused unless the session's `claim` is `unlocked`, and a host that declares one refuses a session that names no clock reader. Every session target in fsla and fsla-bench claims `unlocked`, so a host that declares a lock locks and proves it and a host that declares none (the rented card's container) is measured unlocked; the semantics say which (`clock`: `locked` or `unlocked`), so a locked and an unlocked session are never one record. |
| binary identity | building an arm refuses a binary whose build record marks it an iteration build or that lacks an arm its tree declares, and asks the device for the subject's family stamp, refusing the arm without it: a path or a hash passes against a stale extension, a device-side fact does not. |
| extension identity | each worker reports the file every module the entry `binds` (`fsla` and the extension) was loaded from, and a file outside the entry's bundle refuses the member: an import path that names another tree (a checkout's venv, another output base) otherwise answers with that tree's kernel while the record carries this tree's commit. |
| tree identity | every bundle carries its workspace's stamp -- the commit and the sha256 of its uncommitted tracked diff -- and every stored session names each bundle's; a stored number names a commit, so the store refuses a sample whose checkouts carry uncommitted changes (a measured variant is a commit; `--no-store` keeps the run unrecorded, the lab's in its scratch store), a workspace that stamps no commit is refused, and a status command that fails fails the build. |

## The host a measurement runs on

The local card runs no GPU test (`docs/CONSTITUTION.md#r8`, `docs/KERNEL_STANDARDS.md#cards`): master is Hopper only, and
every GPU run -- test, gate, probe and measurement -- is taken on a rented H100. The bullets below are the rules a
session on that card follows; sizing a cell to fit is the same arithmetic there.

* **A session's whole member set fits the card, with headroom.** A session sets up every entry before timing and holds
  them all resident at once, each in its bundle's worker, so what must fit is every arm on every cell of the session.
  The card is shared with the host's display, and a near-full one pages and corrupts every time in the session without
  a message. Size a local cell at scale smaller (fewer heads, or a group with one FSLA cell) and estimate each cell's
  peak footprint on paper before launch: the test tiers cap their process's share of the card (`tests/conftest.py`), a
  session's workers set no cap.
* **A row near the card's cap is discarded, never caveated.** Under WSL2 the driver falls back to host memory over PCIe
  when VRAM is exhausted (Blake: "if we measure above the vram size, it spills into system memory and then it is just
  going to be slower"), so such a row times PCIe, not the kernel. Each entry's peak memory is a memory pass's sample
  (`bazel run //measure:memory`, fsla-bench's `memory_<group>`), read beside its times; to fill a grid, raise the CTA
  count without raising memory (more heads or batch at a shorter `L`).
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
(`Subject.level`) and every member records it, so a reading pairs entries of ONE level and
`bazel run //:results -- verdict` refuses a pair of two.
A comparison between this library and another is therefore a LAYER comparison: FSLA's
own layer against attention's, because FSLA's number includes the state its capacity buys.

What this line declares (`measure/subjects.py`): the KERNEL level (`entmax_solve`), the OP level (`prefill`, `layer_op`,
and `decode_step` through the engine's decode step) and the LAYER level (`layer_prefill`). No prefill kernel is built
(the carry, intra and producer are designed and measured in the lab, [`open-work.md`](open-work.md)), so every subject
that reaches the prefill refuses by name and is recorded as that entry's failure; a layer arm's fixture builds only what
the arm reads (`measure/cells/layer.py`'s `Fixture`), so `entmax_solve` sets up with the producer unbuilt and is the
entry `//measure:session_entmax` times. The flash reference is a layer entry, and until a FSLA layer runs beside it, a
session that holds both is two levels measured in one place, not a comparison.

## The reference

Every reported carry number sits beside FlashAttention at the same point, measured in the same interleaved session:
fsla-bench's attention entry (`fsla_bench/measure/attention.py`) on the group's QKV cell, causal, width `dv`, bf16,
through torch's flash backend: Dao's FlashAttention-2 compiled into torch, or FA3 (Hopper) and FA4 (Blackwell), which
the entry's `impl` activates through `torch.nn.attention.activate_flash_attention_impl`. The backend is forced, so torch
refuses a call flash cannot take rather than timing its math or memory-efficient backend, and the entry records torch's
version and the implementation.

<a id="incremental"></a>
THE INCREMENTAL CELL is the headline (Blake, 2026-10-18): the time to process `c` NEW tokens given `T` tokens already
absorbed, for a model of state count `N`. FlashAttention's `c` queries attend over a KV cache of `T + c` keys,
bottom-right causal (the entry's `context` parameter); FSLA's layer runs the `c` tokens' producer, carry
from its `N`-leaf state and intra (none of them built). FlashAttention's cost grows with `T`; FSLA's does not once its
state holds the history, so the comparison is a grid -- `T` in {0, 4K, 16K, 64K, 256K, 1M} by `c` in {1, 512, 4K, 16K},
the model's hidden width and heads stated -- never one number. Each side's per-call overhead is in its time:
FlashAttention's KV read and launch; FSLA's state load and store (its HBM round trip, projected from the state's bytes),
launch, partial last window and wave quantization.

AT LAYER LEVEL, WITH BOTH SIDES' PROJECTIONS: FlashAttention's QKV projection of the `c` tokens (the entry's `hidden`:
one cuBLAS GEMM for every head, the library's own form) against FSLA's producer (its routing, gain and values: its
wider per-head outputs against QKV's 192), so a number is what a user pays. The output projection is the same on both
sides and is in neither.

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

**`cuda_events` is launch-to-finish time.** A call synchronizes the device, records a start event, makes the launch's
host call, records an end event and synchronizes again, and its sample is the time between the two events
(`measure/provider.py`, fsla-bench's `attention.py`). The device is idle when the start event lands, so every host step
the call takes before its last enqueue is inside the sample: on a kernel of a few hundredths of a millisecond the host's
dispatch can be most of the number, and at scale it is a rounding error.

Every timed entry names its stopwatch (`Timed.instrument`; fsla's arms and the attention entry time with `cuda_events`),
a session refuses entries that name two, and its record names the one it used. A duration under `ncu` (the lab's
`-- ncu`, replayed, caches flushed, at base clocks) reads above the event time and is a counter's instrument, never a
cross-check of a session's.

<a id="comparing-two-builds"></a>
## Comparing two builds

A change is judged against its baseline in ONE session: fsla-bench's `reference` bundle is the baseline commit (its
MODULE.bazel names it, built by its own pins), its `candidate` the checkout the run names, and each group's session
holds both bundles' entries of its arms (below, [bundles](#bundles)):

```bash
bazel run //measure:session_<group>_<arms> --repo_env=FSLA_CANDIDATE=<this checkout>    # in fsla-bench
bazel run //measure:null_<group>_<arms> --repo_env=FSLA_CANDIDATE=<this checkout>       # its null gate
```

Each session interleaves both commits' entries call by call, so the ratio between them is read from one session's
samples (`bazel run //:results -- verdict --reference @reference`). Times from two separate runs are never compared:
the verdict pairs members inside one session only. The verdict reads a session whatever its clock state: the state is
in the session's semantics and its clock reads in `timing_members`, where a reader holds it. Two builds are compared at the
card's own clocks (KERNEL_STANDARDS §32.7): every session target claims `unlocked`, and a locked clock is for isolation
work (the lab, a part harness), never a claim about real-world performance. One launch's cycles each
under `ncu` (`sm__cycles_elapsed.max`) compare two builds without a session where the question is one kernel's cycles.

## The flagging rule: three gates, all must fire

`fsla_devtools.results.verdict` judges a candidate against a reference timed in the same sessions; which stored sessions and
which reference it is given is the store's query (below). Timing compares only within the session that interleaved it,
so every gate reads within-session quantities: `session(candidate, reference)` takes the two members' samples by round
and gives, per round, each member's median, their RATIO and their DIFFERENCE.

1. **Effect size** — `ratio_limit`: the session's median ratio over `1 + 3·IQR/1.349` of its own per-round ratios.
   `ALARM_SIGMA = 3.0` is the only free choice, and it is stated as a false-alarm rate (~0.1 % one-sided per cell per
   session), never as a percentage of anything.
2. **Significance** — `paired_verdict`: an exact Wilcoxon signed-rank test over the session's per-round differences,
   candidate minus reference, at `ALPHA = 0.01`. **The round floor is derived, not chosen**: the smallest two-sided
   exact p reachable with `n` non-zero differences is `2/2ⁿ`, so below `MIN_ROUNDS = ceil(log2(2/ALPHA)) = 8` no outcome
   can be significant and the test is undefined rather than underpowered. Sessions run 8 rounds; fewer, or fewer
   non-zero differences, report `insufficient_data`.
3. **Persistence** — `persistence`: a regression needs a trailing run of at least two sessions over their own limits, at
   the same code on both sides. One thermally unlucky session on a box with logged power capping is not a kernel change.
   A run of two that has since healed is `suspicious`, which is not a merge blocker and is not silence either.

`insufficient_data` is a distinct answer from `no_regression` everywhere: the first says the design cannot decide, the
second says it decided. A violation the other gates do not confirm is `flagged_not_confirmed`.

**THE SPREAD IS THE SPREAD OF THE QUANTITY JUDGED.** Reps taken back to back on one warm fixture spread less than a
comparison that crosses processes, and a reference's medians across sessions carry the host's drift. What is judged is
the paired ratio inside one session, so the limit is that ratio's own round-to-round spread, drift both members share
cancels in it, and the one thing the ratio crosses that its spread cannot see -- two worker processes -- is the null
gate's.

**AND NEVER A SPREAD SMALLER THAN THE STOPWATCH RESOLVES.** The device-event timer quantizes, so a small kernel's calls
can land on one value, a recorded spread of `0.0000 ms`, and a limit at the median the next tick exceeds. The spread
used is `max(IQR, resolution)`, where the resolution is the smallest gap between the distinct values one member's
samples hold, relative to the reference's median -- measured on every judgement, because it is a property of the
stopwatch and the box -- and the verdict says when it bound (`floored_at_resolution`). Where neither member shows a
resolution (each member's calls all on one value), the gap between the two members is the only floor and the pair is
`cannot_resolve`, never judged.

## Which samples are judged

    bazel run //:results -- verdict --reference LABEL [--candidate LABEL] [--cell C] [--arm A] [--json]

reads the stored timing sessions (the `timing_members` and `timing_samples` views). A member's label is its entry's
repository or package (`@reference`, `@candidate`, `//measure`) and its arm the entry's name. In every session that timed
the reference label's arm on a cell, each other label's member of that arm on that cell is paired with it and judged
inside the session. A unit is the arm, the cell, the two labels, the level, the device and the code each ran (commit
and tracked diff); its sessions, oldest first, are the runs, and the newest is judged. A member whose provenance names
no commit is not read; a pair of two levels is a refused row of its own; a pair whose null gate found a worker's bias is
`untrusted`, and every other row says what its gate found (`trusted`, or `none`). Nothing is stored: a verdict is a
reading of the records, taken again whenever it is asked for. `bazel run //:results -- dashboard --out FILE --reference
LABEL` is one run as a page.

## SASS

`bazel run //:ratify -- --census` and `bazel run //tools:sass_bodies` are the instruments. Two binaries in a
comparison may be built at different paths ([`build.md#kernel-identity`](build.md#kernel-identity)); their hash maps
must agree on `sass_bodies.HASH_VERSION`, which `bazel run //tools:sass_bodies -- --compare A B` checks and refuses on
mismatch.

**Measuring a kernel's register DEMAND at a bound it does not ship at** is a
scratch-tree edit — copy the tree, change the `__launch_bounds__` CTA argument at
the instantiation, run `bazel run //:ratify -- --census` there, read the spill
column — and never a flag on the shipped tool, because a tool that can force a
bound is a tool that can report a number nobody ships.

## What is deliberately not here

* **A second method.** One program times every subject (`fsla_devtools.timing.session`), whether this tree's session or
  fsla-bench's over two commits runs it: two methods are how a harness comes to disagree with itself.
* **A committed SQLite database.** A binary blob has no diff and cannot be reviewed in a landing. Any query artifact is
  derived and ignored — see the record below, which is exactly this design.
* **A flat percentage threshold.** The IQR-derived line is better than every rule in the surveyed field; replacing it
  would reintroduce the invented number the derivation exists to avoid.

## <a id="bundles"></a>Two commits in one session: bundles

A timing compares only inside the session that interleaved it, so two commits are entries of ONE run. fsla-bench names
two BUNDLES (fsla-devtools' `bundle` tag): `reference`, a commit of this repository its MODULE.bazel names, and
`candidate`, the checkout a run names (`--repo_env=FSLA_CANDIDATE=PATH`). Each is built in ITS OWN workspace -- a commit
checked out under the machine's store and built by its own pins, the machine's fsla-devtools override dropped; a
checkout in place, with it -- and what `//measure:all` holds of `_timing_entry` targets (entries, the clock reader,
diff sides) is copied into a repository of the bench: `@reference//:entmax_solve_<cell>` is this tree's entry of that
commit, its bundle the commit's own. A record names each member by that label and by its bundle's digest and stamp. A
commit older than the `timing_entry` rule is no bundle, and a commit predating an arm contributes none of it to a
session; the candidate must have every arm the bench's groups name, or the bench's analysis fails.

<a id="the-diffs"></a>THE DIFFS. A SIDE (`diff_side`, `measure/entries.bzl`'s `SURFACES`) is an executor of
`tests/oracle/sides.py` over its cells in the sides' bundle (`//measure:side_bundle`: this tree's bundle and the
oracle's sides): `side_oracle` (the fp64 oracle forward and backward, on the oracle-tier carry cells whose retained
state is under `RETAINED_STATE_BUDGET`), `side_producer` (the routing reference), `side_carry-kernel` (on the device). A
DIFF (`diff_verdict`) is two sides under one named rule -- `bit-identical`, the per-slot clause rule mirroring the
oracle's, or `support-equal` -- with a claim (`expect` same or different); its tool produces each side on its own
bundle's interpreter and compares them, and its gate (`diff_test`) fails a claim not held. A side proves its library
loaded from inside its bundle; two sides of one bundle and one executor are refused; a comparison that compared fewer
quantities than it declared, or a cell only one side could produce, does not hold; a cell both sides refuse is listed
and not counted. Across commits fsla-bench pairs each surface BOTH bundles export under ITS rule for it (`RULES`) --
the oracle bit-identical, the producer support-equal -- as `diff_<surface>`, its gate `diff_<surface>_test` (`bazel
test //measure:diffs`) and its store `store_diff_<surface>`, stored at its label
(`@fsla_bench//measure:store_diff_<surface>`); a reference that exports no side gives no diff. Verdicts are stored,
tensors never. The carry kernel's side has no rule across commits (no commit states its reassociation) and no
comparison against its fp64 inter term inside this tree (`tests/oracle/tolerances.py`'s `CARRY_NUM` is `Owed`).

## <a id="instruments"></a>The instruments

AN INSTRUMENT is a tool's JSON over the files it reads (fsla-devtools' `instrument`, this tree's wrapper in
`bazel/defs.bzl`): the tool runs in a tree of copies of those files, and one that cannot run on what it was handed
writes `{"refused": WHY}`, which is its output and stops nothing else, while one that fails fails. Without a device it
is an action, its JSON a build output; ON THE DEVICE (`device = True`, keyed on the devices' identity) it runs in the
run phase of the store target that records it, never in a build, since a build holds the host budget and the GPU taken
under it deadlocks against a measurement, which takes the GPU first ([`build-graph.md`](build-graph.md#front-door)).

A STORE TARGET (fsla-devtools' `results_store`) records a build output: `bazel run` stores each of its files as a sample
at the store target's own label, of the record its semantics key -- the source's label, the file's name, an
instrument's arguments and the content digest of every input it depends on (an instrument's tool, files and devices;
the `inputs` it names) -- with the run and the stamped commit as its provenance, and asks the store first: when the
record already holds this outcome nothing is stored, so a second run over the same build stores nothing, into
whichever store it is pointed at. A refusal is stored as a failed sample naming why. `-- --store-root DIR` stores into
another records directory, `--no-store` produces the outputs (a device instrument runs) and keeps them, `--scratch DIR`
names where a device instrument writes.

| target | records | keyed by |
|---|---|---|
| `//measure:store_static_manifest` | each built architecture's static manifest (`//:static_manifest`, warpline's, the build output the SASS gate's comparison judged: [`internals/tools/sass_gate.md`](internals/tools/sass_gate.md#manifest)) | the library and warpline's checker |
| `//measure:store_oracle_tier` | the oracle tier on the device (`measure/oracle_tier.py`): the counts, WHICH cases failed (a red case is a stored fact), every per-slot comparison's margins (`--oracle-margins`) and the worst per output kind; a host without a CUDA device refuses | the tool, the library, the tree and the devices |

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
stored sample is never rewritten), which `//:docrefs` resolves inside the pin, and the tools that read a record -- the
lab's pages and its `trace`/`path`, the floor model (`tools/carry_floor.py`), the rates' check (`measure/rates.py`) --
read it there, handed its root by the graph (`FSLA_DATA`, `measure/data.py`). A LAB RECORD is read through one function,
`lab_record(experiment, arch)` (with `blocks` or `variant` where the question is another's): the newest sample of the
run's record its question keys (`lab_semantics`: the experiment, the architecture, the kind -- `run` or `counters` --
the grid it was held to and a named condition), its traced rows the sample's parts. `bazel run //measure/lab:NAME.run`
stores a new sample into the store's checkout (at its own label), never this tree, and it becomes citable when a commit
of the data repository holding it is pushed and the pin moves to it. A sample marked `backfilled` holds a file the data
repository kept as written, its bytes the output unchanged, with the file's path in its mark.

A tool opens a store at the LABEL of the target that produced what it stores (`@<workspace>//<package>:<target>`,
a folder of the data's `records/`: the store is a durable index of what Bazel's targets produced) and hands it the
SEMANTICS of what it measured -- the inputs its numbers depend on, reduced to identities (a commit and its tracked diff,
a binary's manifest or file digest, the cells, the counts, the device's software, the clock lock). A SAMPLE is one file,
named by its run and written once; the KEY is sha256 over its semantics, so the same measurement taken again is another
sample of the same key and a changed input is another key. A sample is the raw output or the
failure, when, how long, and its PROVENANCE (the checkouts by directory name, branch and commit, the run, the host).
Every sample is kept, failures included. Nothing relational is stored: a ratio against a baseline is the reader's,
and a timing only compares within the session that interleaved it.

| label | writer | the semantics | a sample's output |
|---|---|---|---|
| `@fsla//measure:store_static_manifest` | `bazel run //measure:store_static_manifest` | the source, the manifest's name (its architecture) and the content digests of the library and warpline's checker | the manifest |
| `@fsla//measure:session_entmax`, `@fsla_bench//measure:session_<group>_<arms>` (a session's own) | the session program (`bazel run //measure:session_entmax`, fsla-bench's sessions) | every entry's label, executor, parameters, cells, bundle digest and environment, the timing parameters, the claim, the clock's state, the machine's devices | every sample in order with its round, rep and position, each member's status, the seed, the clock reads, each bundle's manifest |
| `@fsla//measure:memory`, `@fsla_bench//measure:memory_<group>` | the session program's memory pass | every entry's label, executor, parameters, cells, bundle digest and environment, the calls, the machine's devices | each entry's peak memory on each cell |
| `@fsla_bench//measure:null_<group>_<arms>` | the session program's null gate | the session's semantics, a null gate | the session of the two copies and, per entry and cell, whether they agree |
| `@fsla//:container_check` | `bazel run //:container_check` | the devbox image's digest | the steps, each timed, and the cold start |
| `@fsla//measure:store_oracle_tier` | `bazel run //measure:store_oracle_tier` | the instrument, its arguments and the content digests of its tool, the library, the tree and the devices' identity | the counts, the failed cases, the margins; or the refusal, as a failed sample |
| `@fsla_bench//measure:store_diff_<surface>` | fsla-bench's `store_diff_<surface>` over a `diff_verdict` | the content digests of both sides' bundles, the rule and its environment | the verdict per cell and quantity, never a tensor |
| `@fsla//measure/lab:<experiment>.run` | `bazel run //measure/lab:<experiment>.run` (`run`, and `-- ncu`) | the experiment, the architecture, the kind, the grid (`blocks`), a variant (`lab_semantics`); the commit, the binary's digest, the toolkit and the device are each sample's provenance | the run's record (the results with their samples and checks, the kernels, the counters) and its traced rows as parts |

Reading across runs is a join, never a parser: `instrument_metrics` is every number every instrument recorded in
long form (run, cell, dotted metric, value), `diff_cells` every stored diff's verdict per cell and quantity, and
`bazel run //:results -- instruments --baseline RUN` is one run's numbers beside another's as deltas; the dashboard
(`--baseline RUN`) renders the same. Which run is the baseline, like which label is the reference, is the reader's.
A result is read from the store (`bazel run //:results -- show`, `history`, `latest`, `instruments`), never from a
terminal print.

`bazel run //:results -- sql "..."` queries every label through a derived SQLite index in the machine's cache
(`history` and `latest` for one label; views flatten the stored outputs: `cells`, `diff_cells`,
`instrument_metrics`, `memory_rows`, `null_gates`, `timing_members`, `timing_samples`);
`bazel run //:results -- verdict` is the regression judgement above; `bazel run //:results -- check` holds every
sample to its label's folder and its key; `bazel run //:results -- commit -m <message>` commits the records, after that
check. Pushing is the owner's act.
