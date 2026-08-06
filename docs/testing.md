# Testing — the two tiers

The suite answers **two different questions**, and conflating them is how a real
bug hides. It is organized into two named tiers plus a units layer.

| Tier | Question | Compares |
|---|---|---|
| **Tier 1 — inter-implementation** | Does the kernel compute the right function? | the kernel against the **fp64 oracle** |
| **Tier 2 — intra-implementation** | Does the kernel compute the *same* function at every runtime geometry the planner may choose? | the kernel against **itself** |

Tier 1 alone is insufficient because a per-geometry tolerance can absorb a geometry
bug. Tier 2 alone is insufficient because two geometries can agree with each other
and both be wrong — this project has a bug that passed exactly that way. Tier 2
therefore **never stands alone**: every geometry it relates is independently
anchored to the oracle in Tier 1.

## Tier 1 — semantics

**Ground truth is `rola.naive_rola`, fp64, and nothing else.** A freshly written
second reference proves self-consistency only. This prohibition is not a
preference; it is the direct lesson of a shared-router bug that agreed with a
sibling implementation and disagreed with the definition.

Every cell of this cross-product is gated against the oracle:

| Axis | Values |
|---|---|
| normalization | `raw`, `global` |
| decay | off, on (including the `1 - rate` catastrophic-cancellation arm) |
| MMA precision | `tf32x3`, `bf16` — **two arms, two tolerances, never one widened number** |
| `P` (sequence-parallel split) | 1, > 1 |
| routing sparsity | dense, sparse, read-only / write-only leaves |
| `(widths, BC, BT, d_v)` | the planner's supported rectangles |
| sequence length `T` | 37 across the structural matrix; 512 / 2048 / 8192 on the accumulation arm |

The length axis is not decoration. An error that ACCUMULATES along the sequence
grows as `sqrt(T)`, so a short fixture under-charges it by `sqrt(8192/37) = 15x` —
the blind spot a resident-state re-quantization walk was measured to exploit.
`test_consumer_vs_oracle.py::test_mass_column_at_sequence_length` closes it, at
the same two tolerances.

### Tolerance discipline

1. `RTOL = 2e-5` for the `tf32x3` arm — a genuine fp32-accumulation-vs-fp64
   tolerance. It was not loosened by a single digit when the MMA landed, which is
   the whole point of gating a novel contraction at near-fp32 precision.
2. `BF16_RTOL` for the `bf16` arm — an **honest format tolerance** derived from 8
   mantissa bits, not a relaxation of the exactness gate. It applies to that arm
   only.
3. The **CPU-only fp64 arm** (`tests/tier1_semantics/test_algorithm_vs_oracle.py`)
   transcribes the kernel's *control structure* into fp64 Python at `~1e-16`. It
   separates "the algorithm is wrong" from "fp32 accumulated differently", and it
   needs no GPU.
4. **Every tolerance is either DECLARED with its derivation in a module comment or
   DERIVED in code.** A number with neither is a defect.
5. **Loosening a tolerance is a semantic change**: changelog entry, derivation,
   reviewer sign-off. Tightening needs none of that.
6. **A tolerance is only half of a claim; the other half is the CHARGE — what the
   error is divided by.** The output is charged PER TOKEN SEGMENT
   (`_output_charge`: the max over eight segments of that segment's max absolute
   error over that segment's max reference magnitude), never as one global max
   over one global max. The readout is a ratio whose denominator is accumulated
   write mass, so `|y|` at the cold start exceeds the rest of the sequence by up
   to `2.4e4` at `T = 8192`: a single normalizer is set by the one token whose
   state has been updated zero times, and the resulting metric was measured
   BIT-IDENTICAL between two arms it was supposed to separate. A charge that
   cannot fail is the same defect as a tolerance that cannot fail, and
   `test_output_charge_sees_a_drift_the_global_max_cannot` holds the mutant that
   proves the current one can.

## Tier 2 — geometry, as data

`tests/tier2_geometry/matrix.py` holds the matrix as a literal table. Every pair of
geometries carries a **declared contract class**, and the class decides the
assertion.

### Class (a) — ENGINEERED IDENTITY

The two geometries are *designed* to produce byte-identical output. Tested with
`torch.equal` — **no tolerance, no `allclose`, ever.** The claim is always
structural and the structure is stated in the row's `basis`.

**Every such row declares a VALIDITY PRECONDITION and the test asserts it.** The
`BC` row, for example, is only meaningful with no initial state carried and with
both `BC` values tiling the same state columns — and it compares STATE, never `y`,
because the split-K `atomicAdd` across concurrent owner CTAs makes `y`
nondeterministic *by construction*, so a bit comparison there would be testing the
scheduler. A bit-identity assertion whose precondition is unstated is a future
flake.

### Class (b) — REASSOCIATION-EQUIVALENT

The two geometries legitimately reorder floating-point accumulation. Tested by
**two assertions, both required**:

1. **Oracle anchor** — *each* geometry is independently within its Tier-1 tolerance
   of the *same* oracle point. This is what stops two wrong geometries from
   agreeing.
2. **Derived mutual bound** — the two are within a bound *derived* from the
   reassociation (`atomic_readout_n2_eps`, `carry_fold_T_eps`), never chosen.

Two prohibitions, both of which have cost this project time:

- **Never bitwise-vs-each-other.** `atomicAdd` order is a random variable; one draw
  agreeing is not a property (`test_split_k_determinism_is_not_assumed`).
- **Never a loosened tolerance in place of a derived bound.** If a pair misses its
  bound, the answer is a bug hunt or a corrected derivation — never a bigger
  number.

### The classes are not a spectrum

A pair is (a) or (b). Where the same axis carries different classes in different
sub-cases, those are **separate rows** with separate preconditions. `P` is the live
example:

| Sub-case | Class | Basis |
|---|---|---|
| `P = 1` vs `P = 2`, zero initial state, **state** | (a) | at `P = 2` pass 1's single summary IS the serial prefix — no re-associated boundary exists |
| `P >= 3`, zero initial state, **state** | (b) | intermediate summaries are folded into the boundaries |
| an owner with **no write anywhere**, every `P` | (a) | the spec's carve-out is **normative and bitwise** |
| written leaves, **state** | (b) | the carry fold re-associates |
| any `P > 1`, **`y`** | (b) | split-K `atomicAdd` across owners |

Note the third row: class (a) not because the arithmetic happens to agree but
because the spec *requires* it to. That is exactly why the class must be declared
data rather than inferred from behaviour.

## The three meta-tests

`tests/tier2_geometry/test_meta.py`. Without these the matrix is a list of tests
with a nicer shape; with them it is a claim about completeness that the suite
enforces.

**M1 — CENSUS.** Decodes the template arguments of every ratified instantiation
from the mangled names in the manifests, and requires every axis to be **declared**
— as a GEOMETRY (which then needs rows), a PROBLEM_SHAPE, or a SEMANTIC arm gated
in Tier 1 — with matching value sets in both directions, and every geometry pair
the census rule requires to have a row. It runs at **module import**, so adding
`BT = 64` to the kernel without declaring its contract class **fails at
collection**. It also pins the manifest it reads to `BUILD_CONFIG["manifest_sha256"]`,
so the census cannot enumerate a product that was never built.

**M2 — NON-VACUITY.** For every class-(a) row, a one-ULP perturbation of the row's
**declared probe input** must move the compared output. A byte-equality that a
broken kernel would also satisfy proves nothing.

**M3 — CLASS DRIFT.** For every class-(b) row, the two geometries must **not** be
byte-equal. This is counter-intuitive and it is the point: if a declared
reassociation starts producing identical output, either the kernel silently changed
(a split-K path disabled, an atomic turned serial, a `P > 1` launch degraded to
`P = 1`) or the fixture stopped exercising it. **A tolerance-based test cannot see
either**, because "identical" passes any bound. The failure message names both
hypotheses, because the correct response differs: a kernel change is a bug, a
fixture change is a re-derivation of the row.

**Each meta-test is verified by construction.** M1 is pointed at a fabricated
product carrying `BT = 64`. M3 is pointed at a `P > 1` launch forced to degrade to
`P = 1`, and at a genuine engineered identity mislabelled as a reassociation. M2 is
pointed at a probe that cannot move its output. *A meta-test that cannot fail is
not a meta-test.*

**A class change is a reviewed edit to `matrix.py`, in a diff, with a new `basis`
string.** It can never happen as a side effect of a kernel edit.

## The build-shape gates

Four checks are about the SHAPE OF THE BUILD rather than about arithmetic, and
they are here because each one guards a property whose failure leaves every other
gate green.

**`python tools/gen_shards.py --check`** regenerates the eight instantiation shards
into memory and diffs them against the committed files. Generated sources are
checked in so that what is reviewed is what compiles; this is what keeps that a
fact. `setup.py` runs it before compiling and CI runs it in the CPU lane.

**`tests/unit/test_shard_partition.py`** checks the partition as a STRUCTURE: every
committed instantiation maps back to its own file under `shard_of`, the sharded
axes are the derived prefix of the priority list rather than a hand-written answer,
each membership hash is the digest of its file's tuple list, and `consumer.cu` does
not include the header that would let it instantiate the kernels locally. CPU only.

**`python tools/fatbin_entries.py --check <so>`** is the one that needs a built
binary, and the only one that can see a FAILED split. It reads device entry points
out of the fatbin — per architecture, per originating object — and requires exactly
one per instantiation, in the shard the matrix assigns it, with every ratified name
present. `nm` cannot answer this: a `__global__` template's host stub is weak, so
the linker keeps one however many objects emitted the device code.

**`python tools/header_selfcheck.py`** compiles every header under `csrc/rola/`
ALONE — one generated translation unit per header, containing that header twice and
nothing else, under the extension's own flag list. A header that uses a name it never
included compiles anyway whenever some other file in the translation unit included
that name's home first; nothing in a normal build says so, and the break surfaces
later, at a new caller or at an unrelated include removal. Including the header twice
is what makes the include guard part of the check. It needs `nvcc` but no device, and
takes about 70 s for the whole tree, so it belongs to the GPU arm's runner image
rather than to the CPU lane.

## The documentation gates

Two of the units-layer tests are about documents rather than arithmetic, and they
are here because a document nothing checks is a document that drifts.

**`tests/unit/test_generated_docs.py`** pins the README's generated support
tables to `tools/supported.py --check`. A hand-maintained support table is a
claim; a generated one is a report, and this is what keeps the difference real.

**`tests/unit/test_selector_interface.py`** holds THE SEAM the card marks binding:
the typed and versioned `StatisticsBundle` (its version refusal, and `expose()`'s exact
key set), the `Selector` protocol's `name`/`status`, `CostModelSelector` declaring
itself `INTERIM`, `FixedChoiceSelector` reading nothing, `select_with`, and the frozen
§4.3 entry points' keyword-only `selector=` — including that a passed selector's choice
is the one that comes back, that cost-model kwargs beside a selector are refused rather
than dropped, that the §6b decode bypass still precedes any selector, and that a plan
records who decided. CPU-only with an injected oracle: selection is a pure function of
its recorded inputs, so a row that needed a device would be testing the driver.

**`tests/unit/test_dispatch_table.py`** is the CENSUS: it reads the dispatch table out
of the built extension and joins it against the fatbin's device entry points, decoded
from their mangled names, asserting set equality in both directions per architecture —
an entry the table cannot reach is dead payload, a row with no entry is a dispatch into
a launcher the binary does not define. It also gives the staleness filter teeth (a
perturbed shard key must drop exactly that shard's rows, and they must stop resolving),
proves the enable predicate is non-vacuous below the largest footprint, and pins the
planner's `SUPPORTED_*` tuples to the table's own axis values.

**`tests/unit/test_docs_mirror.py`** pins the `docs/internals/` mirror. The
engine's design reasons live in a markdown tree that mirrors `csrc/rola/`, with
one-line `//: HAZARD <slug>` stubs left at the sites they explain. Five rules,
all grep-level so a contributor can read the check itself:

1. every source file under `csrc/rola/` has a mirror doc at the same relative
   path (minus the `src/` component), extension `.md`;
2. every `docs/internals/….md#anchor` cited from a source resolves to an anchor
   that doc defines;
3. a hazard stub's slug IS its anchor id, so the two cannot drift apart;
4. no doc defines the same anchor twice, which would make a citation ambiguous;
5. a directory of GENERATED sources is exempt from rule 1 and documented by one
   README instead — and the exemption is checked rather than granted: the README
   must exist, and the directory may contain nothing its generator does not
   produce. `csrc/rola/src/instantiations/` is the only such directory; eight
   mirror docs restating one sentence would be the ceremony this mandate removes,
   and the next re-partition would orphan them.

It carries its own non-vacuity guard: a floor on the number of sources found, so
that a broken glob cannot make the file pass by checking nothing.

Both run in the CPU line. They read files; they never touch a device.

## Running it

```bash
pytest tests/unit tests/tier1_semantics -m "not cuda"                       # CPU: no device, no build
flock /tmp/rola_gpu.lock pytest tests/unit -m cuda                           # tests/unit's CUDA-marked rows
flock /tmp/rola_gpu.lock pytest tests/tier1_semantics -m cuda                # TIER 1
flock /tmp/rola_gpu.lock pytest tests/tier2_geometry                         # TIER 2 (incl. the meta-tests)
flock /tmp/rola_gpu.lock pytest tests/layers                                 # layer <-> op, at every dispatch threshold
flock /tmp/rola_gpu.lock pytest benchmarks/test_latency_regression.py -m bench   # the latency gate
python benchmarks/rola_probe.py duration consumer_forward --ab global:raw    # NO flock -- see below
```

**THE LOCK IS `/tmp/rola_gpu.lock` — that exact path, with an underscore.** `flock`
is advisory and keyed on a NAME, so a second spelling excludes nothing and silently
buys nothing: two commands holding two different paths run on the device at the same
time and neither can tell. Every command in this repository, every benchmark's usage
line and every agent brief uses this one path. If you find another spelling anywhere,
it is a defect — fix it rather than adding a second lock beside it.

**THE ONE EXCEPTION IS `rola-probe`, AND WRAPPING IT IS A DEADLOCK, NOT BELT AND
BRACES.** The probe harness acquires this same lock itself, by design — the
discipline is enforced rather than documented (`docs/measurement.md`). `flock` locks
are held per open file description, so an outer `flock /tmp/rola_gpu.lock` around
`python benchmarks/rola_probe.py …` blocks the harness's own acquisition of the same
file, forever, with no output. The canonical rule is therefore:

> **everything that reaches a device is wrapped in `flock /tmp/rola_gpu.lock`,
> EXCEPT `rola-probe`, which is invoked bare.**

`ROLA_GPU_LOCK` exists to MOVE the harness's lock for a caller that must compose with
a genuinely different outer lock; pointing it at the path the caller is already
holding is the deadlock it exists to avoid.

**EVERY OTHER LINE THAT CAN REACH A DEVICE IS `flock`ed, AND THAT IS NOT THE SAME
QUESTION AS THE `cuda` MARKER.** The marker's job is *collection-time deselection* — it is what
lets the first line avoid initializing a CUDA context at all. It is **not** a device
partition: `tests/tier2_geometry` and `tests/layers` carry 78 rows that launch kernels
without being `cuda`-marked (e.g.
`test_layer_dispatch.py::test_the_bf16_arm_is_paging_invisible_at_its_own_tolerance`),
because in those tiers a device is the premise of the tier rather than a property of a
row. So `-m "not cuda"` may NOT be used to decide whether a command needs the lock;
the tiers it names do, unconditionally, and the recipe says so. This is the same rule
`benchmarks/*.py` state in their usage lines ("GPU work belongs under
`flock /tmp/rola_gpu.lock`"), applied to the test recipe.

`tests/unit -m "not cuda"` above deselects `tests/unit`'s own `cuda`-marked rows (e.g.
`test_interface.py::test_select_backend_sends_no_grad_cuda_operands_to_the_consumer_arm`,
two in `test_sequence_parallel_policy.py`) the same way it deselects `tier1_semantics`'s
-- so the `-m cuda` line above is what makes them reachable at all. Every committed test
must be collected by SOME line here; a test that only a `-m "not cuda"` line ever
selects against is a test nothing runs.

**Never `pytest tests/` unscoped.** The `cuda` marker exists so `-m "not cuda"`
deselects at *collection* time, before torch initializes a CUDA context. The
`bench` marker is opt-in for the same reason in reverse: a latency measurement is
not a correctness gate and must never be able to fail a merge.

## Measurement, and the tests that guard it

Performance work has its own document — **`docs/measurement.md`** — because the
instruments are as much a subject of review as the kernel is: one stopwatch
(`cuda_events`), one harness that acquires its own discipline, and one append-only
record at `perf/ledger.jsonl`.

Three test files hold that machinery to the same non-vacuity standard as everything
else here. Each plants the violation its instrument exists to catch and requires the
refusal; none launches a kernel.

| file | what it plants |
|---|---|
| `tests/unit/test_probe_harness.py` | a lock held by another process (the harness must block), a clock that never settles (the run must abort), a corrupt ncu section file, a subcommand offering to skip the lock |
| `tests/unit/test_perf_ledger.py` | a row missing its provenance, a row from an unknown schema, a shift smaller than its own scatter, a single unlucky run, an effect with no significance behind it |
| `benchmarks/test_latency_regression.py` | a baseline taken with a different stopwatch, and one that does not say which stopwatch took it |

`python tools/ratify.py --self-test` is the same discipline on the codegen side and
is listed arm by arm in `docs/ratification.md`.

### Triaging a suspected stream-ordering bug

There is no runtime detector anywhere for a forgotten cross-stream dependency on
already-allocated memory, and `compute-sanitizer --tool racecheck` does not fill the
gap — it detects on-chip **shared memory** hazards only, and both bugs that escaped
here were host-side Python. Two environment variables classify one before a debugger
is opened, because each removes a different precondition:

```bash
CUDA_LAUNCH_BLOCKING=1            # serializes launches: collapses the window a missing join needs
PYTORCH_NO_CUDA_MEMORY_CACHING=1  # every free becomes a device-synchronizing cudaFree
```

A bug that disappears under either is an allocator-lifetime or stream-ordering bug —
a missing `record_stream` or a missing join — not a device memory hazard.

### The stream-discipline fixture, and what every crossing path owes

`tests/tier2_geometry/stream_discipline.py` is the fixture; it was four bespoke cells
guarding one module, and copying them into each new path is how three of the four end
up missing. **A path that crosses a stream instantiates all five checks.**

| check | makes deterministic |
|---|---|
| `OrderingSpy` | the publication is joined before the consumer reads it — and a join performed *inside* the publication is recorded distinctly, so it cannot satisfy the outer claim |
| `record_stream_spy` | a tensor crossing to another stream declares its lifetime there |
| `allocator_pressure` | the window a lifetime bug needs is actually opened |
| `assert_agrees_under_launch_blocking` | async and serialized runs agree bit-for-bit |
| `assert_capture_rejoins` | a fork that never rejoins **refuses to close a CUDA graph capture** |

The last is the only *mechanical* detector of the missing-join class that exists
anywhere, and it is not ours: `cudaStreamEndCapture` returns
`cudaErrorStreamCaptureUnjoined` when a stream forked inside the captured region never
joins back. Measured here (sm_86, CUDA 12.4): the joined region closes its capture,
the same region with the join deleted raises *"capturing stream has unjoined work"*.
Each attempt runs in its own process, because a failed capture leaves the context in
an error state that later calls inherit.

**Its limits are measured too, and the paged path is outside them.** Capture validates
only the captured region, cannot see an event recorded on the *wrong* stream, and
requires the region to be capture-legal. `PageArena` owns a persistent side stream and
records its readiness event outside any capture, which the capture rejects as a
cross-capture dependency before the body even runs — so the production paged path
cannot be wrapped in one today. The detector is therefore the contract for paths that
*can* be captured, and `test_stream_discipline_fixture.py` proves it reads on this
driver in both directions rather than leaving that an assumption.

Every check in that file is run against a planted violation as well as a correct
path — an unjoined fork, a publication read before its join, an inner join standing in
for an outer one, a tensor crossing without `record_stream`, a burst that allocates
nothing. A fixture that cannot be shown failing converts "we checked" into a sentence
nobody can falsify.

### The sanitizer lane

```bash
python benchmarks/rola_probe.py sanitize consumer_forward --tool memcheck
```

**A scheduled lane over a named subset, never a PR gate** — the shape every surveyed
adopter converged on, because the tools cost one to two orders of magnitude and a gate
that takes an hour gets disabled. `--error-exitcode=1`, a committed suppressions file,
and an explicit skip-list whose entries carry their reason.

Measured on this box, `consumer_forward`: memcheck, initcheck and synccheck are CLEAN
in 4.4–4.6 s each.

**Sold honestly, in both directions.**

* It does **not** own our historical bug class. Both escaped races were host-side
  Python ordering; `racecheck` detects on-chip shared-memory hazards only. The
  stream-discipline fixture above is what covers that class.
* `racecheck` is **skipped for both registered probes** (`consumer_forward` and
  `consumer_forward_deep` — the whole lane, there being no others), and the reason is
  structural rather than a time budget: it reports 4 hazards (131072 each) between the
  `Gm` write and its read, because that edge is published by the item pipeline's own
  release/acquire protocol (`retire_item`'s `__threadfence_block` plus an atomic
  counter, `await_count`'s spin) instead of `__syncthreads`. racecheck orders shared
  memory by standard barriers only and cannot be taught the protocol, so every
  write/read pair across that edge reads as unordered. **racecheck therefore runs on
  NO registered probe today** — the arm is dormant, not narrowed — and stays that way
  until a probe over a kernel without a pipeline-published edge registers; a kernel
  that publishes through a standard barrier WOULD be checked by this lane, but none is
  registered.
* `--leak-check full` is **not** used: torch's caching allocator retains its pools by
  design, so the flag reports 3 "leaks" on a clean run — a lane whose clean state is
  three errors is a lane whose findings nobody reads.
* An overrun that stays **inside** a caching-allocator segment is invisible: tensors
  are suballocated, so that memory is legitimately the allocator's. The lane detects
  accesses that escape the segment, and the planted-violation test plants exactly one.
