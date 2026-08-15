# Testing — the tiers

The suite answers **three different questions**, and conflating them is how a
real bug hides. It is organized into three named tiers plus a units layer.

| Tier | Question | Compares |
|---|---|---|
| **Tier 1 — inter-implementation** | Does the kernel compute the right function? | the kernel against the **fp64 oracle** |
| **Tier 2 — intra-implementation** | Does the kernel compute the *same* function at every geometry the BUILT MATRIX carries, and under both state backings? | the kernel against **itself** |
| **Conformance — the data-regime box** | Is it that function over the whole DISTRIBUTION SPACE the model can put through it? | the kernel against the oracle, at DECLARED regime cells |

Tier 1 alone is insufficient because a per-geometry tolerance can absorb a geometry
bug. Tier 2 alone is insufficient because two geometries can agree with each other
and both be wrong — this project has a bug that passed exactly that way. Tier 2
therefore **never stands alone**: every geometry it relates is independently
anchored to the oracle in Tier 1. The conformance tier exists because routing is a
MOVING distribution (the gain knob alone sweeps candidate fraction ~0.96 → ~0.09
in training), so both other tiers' fixed fixtures answer their questions at a
handful of unmarked points in that space; conformance makes the covered regimes
declared and the uncovered ones listed.

## Tier 1 — semantics

**Ground truth is `rola.naive_rola`, fp64, and nothing else.** A freshly written
second reference proves self-consistency only. This prohibition is not a
preference; it is the direct lesson of a shared-router bug that agreed with a
sibling implementation and disagreed with the definition.

### What the oracle is, and what it promises

The oracle FACTORS, and each factor is trusted for a different reason. The
recurrence is textbook gated linear attention -- one state per leaf, a diagonal
gate, a rank-one deposit, a linear readout -- written as the plain-torch loop a
reader can check against any linear-attention paper, with normalization carried
as a ones column of the value rather than as a second recurrence. The feature
map's hard part, entmax, is the DeepSPIN authors' own package, pinned. What is
OURS is the glue -- the leaf address map, the gate values, the ratio readout --
and it is deliberately small, because it is the one shared blind spot the
factoring does not remove.

Three promises a caller can rely on:

* **Both outputs are `float64`, whatever the inputs were.** Casting back is the
  caller's business; a reference that rounded itself to an operand's precision
  would make an accuracy gate compare against a rounded reference.
* **An input the output does not depend on gets `None`, not a manufactured
  zero.** `write_levels_stored` is the standing case: the spec detaches the
  clock, so it never receives a gradient.
* **The address map is the oracle's own**, derived from the widths rather than
  imported from `Topology`, so a wrong stride convention cannot agree with
  itself in every gate the repository has.

### Changing the oracle: the dual-run protocol

`rola/ops/naive.py` and the routing references are the crown jewels, so a change
to them is not reviewed by reading the diff -- the question is never "is this
code reasonable" but "does it compute the same function". Both versions are RUN:

```bash
TORCHINDUCTOR_COMPILE_THREADS=4 flock /tmp/rola_ram.lock \
  python tools/oracle_dual_run.py --surface oracle --against HEAD~1
```

The old side is the parent commit's real code in a throwaway worktree, run in a
subprocess; the matrix is generated at the NEW commit for both sides; the
comparison is BIT-IDENTITY, forward and backward, with `None` recorded as `None`.
Three declaration forms exist and none of them is an exemption: `--except`
names a quantity whose answer legitimately changed and which must then be
asserted in its own test, `--reassociated` holds a reordered sum to
`REASSOCIATION_RTOL` instead of bit-identity, and `--mutate` injects a known
defect that the run MUST fail on -- a protocol that cannot fail is not a
protocol.

**THE RUN MUST PROVE IT COMPARED.** Every verdict is gated on a per-surface floor
(`MINIMUM_COMPARISONS`: `oracle` 640, `producer` 162) and a run below it fails RED
rather than reporting PASS over whatever survived. This is not hypothetical
bookkeeping: the `oracle` surface was UNRUNNABLE for a window after `raw` died --
the matrix emitted a `raw` row, `naive_rola` refused it, and the run aborted in emit
-- and a shrinking matrix is the same defect wearing a green face. A matrix that
legitimately shrinks moves the floor in the same commit, with the reason.

**`flock /tmp/rola_ram.lock`, and the grid runs in slices.** The backward
retains one autograd graph per token per cell; run whole, the protocol peaked
near 18 GB and froze the host twice. One grid at a time, host-wide.

**`rola` is installed editable, so a tool run from a WORKTREE imports the
canonical checkout unless it is stopped.** The protocol binds and then CHECKS
the library each side imported, and prints both. Test runs are unaffected --
pytest puts the rootdir on the path -- but any new tool that imports `rola` and
compares two trees inherits the trap.

Every cell of this cross-product is gated against the oracle:

| Axis | Values |
|---|---|
| decay | off, on (including the `1 - rate` catastrophic-cancellation arm) |
| routing sparsity | dense, sparse, read-only / write-only leaves |
| `(widths, D, B)` | the built arm list the manifest carries (`chunk_arms()`), projected |
| sequence length `T` | 37 across the structural matrix; 512 / 2048 / 8192 on the accumulation arm |

Three axes this cross-product used to carry are gone (P26, `52a67d8`; see
[`internals/DELETIONS.md`](internals/DELETIONS.md), revival = git). Normalization
is no longer an axis: `raw` is dead, `global` (the mass column and the `num`/`den`
divide, unconditional) is the only value the kernel arm accepts — anything else
is a named refusal. `BC` is likewise no longer a host axis:
it belongs to the built arm (P67 D2), so what varies across cells is the ARM. MMA precision is no
longer an axis: `tf32x3` is dead, bf16 storage with fp32 accumulation is the only
arm the kernel has, and the fp64 oracle stands in as the exactness reference a
`tf32x3` family delta used to provide. `BT` is no longer an axis: the work unit is
always a fixed 16-token group at the fixed `kMmaKQuantum`/word granule, with no
runtime-sized tile left to vary. Cells whose entire purpose was one of those three
axes (a raw-only sweep, a tf32x3-vs-bf16 family delta, a `BT`-boundary
compile-time check) are deleted, with accounting in
`workflows/perf/p26_endgame.md`; cells that gate a semantic property (causality,
the read/write support union, decay pins, `P`-composition) are re-anchored at the
single bf16 family bound below instead.

The length axis is not decoration. An error that ACCUMULATES along the sequence
grows as `sqrt(T)`, so a short fixture under-charges it by `sqrt(8192/37) = 15x` —
the blind spot a resident-state re-quantization walk was measured to exploit.
`tests/tier1_semantics/test_bf16_family_gate.py::test_family_deviation_is_flat_in_T`
closes it, at the same tolerance and over the same three lengths (measured slope
`+0.053` on the chunk arm against the walk's `+0.49`).

**The window, `GT`, was a fifth axis and is not one now.** It was the tiled
consumer's token quantum, chosen per launch by reading the realized routing, and
gated by fixtures built long enough to force the wide arm. Both the axis and its
kernel retired in P67 D2 ([`internals/DELETIONS.md`](internals/DELETIONS.md)); the
chunk arm's token quantum is `CHUNK_TOKENS = 32`, a build fact, and `T` not being a
multiple of it is a named refusal rather than a second formulation to gate.

### Tolerance discipline

1. `BF16_RTOL = 1e-2` — the ONE tolerance the whole file holds to now that there
   is only one storage arm: an **honest format tolerance** derived from 8 mantissa
   bits (`u_bf16 = 2^-8 = 3.9e-3` per requantization), not a relaxation of the
   exactness gate. There is no separate `tf32x3` tolerance to widen it against or
   to protect — the fp64 oracle is the exactness reference this bound is measured
   against directly.
2. The **kernel-vs-oracle arm** is `tests/tier1_semantics/test_chunk_op.py` plus
   the conformance tier's kernel families. The CPU-only fp64 transcription that
   used to sit beside it (`test_algorithm_vs_oracle.py`) transcribed the TILED
   kernel's control structure and retired with that kernel (P67 D2); the
   separation it bought — "the algorithm is wrong" against "fp32 accumulated
   differently" — is carried by the oracle comparison itself, which is fp64 on
   one side, and by `tests/unit/test_oracle_contract.py` on the reference's own
   properties.
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

**The matrix machinery retired with its axes (P67 D2, R-8).** Tier 2 held its
geometry pairs as a literal table in `matrix.py`, over five axes — `BC`, `DV`,
`DECAY`, `P` and `paging`. Four of those are gone: `P` in P28, and `BC`/`DV`/
`DECAY` when the tiled consumer went, leaving a one-axis table that could not earn
the three meta-tests written to keep a matrix honest. The table, its meta-tests
and `test_endgame_executor.py` (the tiled walk's own gate) are in
[`internals/DELETIONS.md`](internals/DELETIONS.md); revival = git.

What Tier 2 is now is the same QUESTION asked of the pairs that still exist, one
file per pair rather than one table over all of them:

| file | the pair it relates |
|---|---|
| `test_chunk_paging_equivalence.py` | paged vs dense backing — bit identity, continuation, absent-atom inertness, residency at every built `BC`, a fragmented non-monotone slot arena, the plan's exact demand through a real launch |
| `test_chunk_final_state.py` | the register-resident state against its stored plane, and the leaf-order law |
| `test_decode_bc_blindness.py` | one decode step across prefill arms that differ in `BC` |
| `test_decode_paging.py`, `test_page_arena_vmm.py`, `test_atom_bitmap.py` | the backing's own structures |
| `test_paging_stream_races.py`, `test_stream_discipline_fixture.py` | the crossing paths (below) |

The two contract classes below still govern every one of those comparisons, and
the prohibitions in them are what the deleted table's meta-tests enforced
mechanically. What is LOST with the table is the mechanical enforcement — a new
built axis no longer fails collection until someone declares its class — and that
is stated rather than papered over: the census that catches an undeclared
instantiation axis now lives on the conformance side
(`test_conformance_census.py`, whose left-hand side is projected from the built
arm list), and the shard partition's own structural checks in
`tests/unit/test_shard_partition.py` carry M1's manifest pin.

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
   reassociation (`split_readout_n2_eps`, `carry_fold_T_eps`), never chosen.

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

## The conformance tier — the data-regime box

`tests/conformance/`. The regime axes and every family are DATA
(`families.py`, pure CPU — the declared-data doctrine the tier-2 table also
followed); generation and the
non-vacuity checkers live in `generators.py`; execution in the `test_*`
files. Three fixture sources cover the box, and each is there for a stated
reason:

* **stratified** — random generation confined to one stratum per axis
  (density, temporal coherence, read-write correlation, mass concentration,
  tail, support cardinality). Axis-DEFINED, never naive random: independent
  per-token sampling manufactures anti-structure exactly where realized
  routing is coincident.
* **producer** — the REAL entmax gateway at swept input gains (realized
  density 1.00 → 0.10, monotone, asserted), because some correlations only
  realized routing produces. (The stale-plan family drove `rola.expert`'s
  derive/execute seam with a plan priced on a distribution 4x dead; that seam no
  longer exists — there is no plan object to make stale — and the family retired
  with it in P67 D2.)
* **adversarial** — named corners FROZEN PERMANENT (cold read, all-read-only,
  chunk boundary): the bug list, kept forever, and enforced by
  `test_family_meta.py::test_the_adversarial_corners_are_permanent`. Two more —
  `k0-fold` and `ragged-tail` — were forced-`P` fixtures and left with the `P`
  axis in P28; the freeze protects a demonstrated bug class, not a deleted
  capability's fixture.

**Self-policing**: every family carries an IN-TEST non-vacuity assertion —
checkers keyed `(axis, value)` so a claim cannot be declared without its
proof running — and a fixture that cannot demonstrate its claimed regime
FAILS before any kernel comparison (the token-0 vacuity lesson generalized).
The checkers themselves are under test: each is fed a violating fixture and
must refuse it, and each accepts an in-regime witness.

**The census join** (`test_conformance_census.py`, pure CPU) is the completeness
claim the suite still enforces mechanically: (instantiation x data-regime),
pairwise, with the left-hand side projected from the BUILT ARM LIST via
`tests/conformance/instantiation.py`, so a new built arm fails here demanding
families or expected-empty rows. Post-D2 it joins 7 off-mid instantiation values
against 10 non-mid regime values — 70 cells, `EXPECTED_EMPTY` 69 — where the
tiled axes gave 6 x 10 and 56. Empty cells are LISTED against a committed
`EXPECTED_EMPTY` rule with named exceptions, asserted with exact equality in
both directions — a newly-empty cell fails, and a stale listing fails too.

**Decode** is its own family (`test_decode_family.py`): the prefill→decode
handoff across the arms from a ragged prefill, a 32-step horizon with
checkpoints ALONG it (the cross-call-compounding blindness the bf16-state
kill named), frozen density declarations proven performance-only under
drifting realized routing, and ONE walk crossing every dispatch boundary
forward and backward with the graph asserted alive on the grad legs.

**Backward** (`test_backward_family.py`): the reference backward is autograd
through the decomposed fp64 oracle with the detachment as literal spec — the
stored-clock input gets `None`, asserted per regime cell — plus one
spec-derived structural-zero test (the anti cell), and the producer VJP
against the pinned DeepSPIN package's own backward. `gradcheck` is DEMOTED to
one-time insurance behind `ROLA_GRADCHECK_INSURANCE=1`, never a lane gate:
it cannot express the spec's detachment, and the insurance harness holds the
clock constant explicitly to demonstrate exactly that. When the native
backward lands (M-2), its kernel-VJP gates plug into these same families.

## The three meta-tests — RETIRED WITH THE MATRIX, and where each claim went

`tests/tier2_geometry/test_meta.py` held them, and it retired with the table they
policed (P67 D2, R-8). They are recorded here because a completeness claim that
quietly stops being enforced is exactly the failure they existed to prevent, and
because the successor of each is a specific file rather than a hope.

| meta-test | what it required | where the claim lives now |
|---|---|---|
| M1 census | every built axis DECLARED, and the manifest pinned to the build's own | `tests/conformance/test_conformance_census.py` (the instantiation side, projected from `chunk_arms()`) and `tests/unit/test_shard_partition.py` (the manifest pin, through `tools/ratify.py`'s own loader) |
| M2 non-vacuity | a class-(a) row's declared probe must move its output | the conformance families' own in-test non-vacuity checkers, and the bf16 gate's M3-mutant rows |
| M3 class drift | a class-(b) pair must NOT be byte-equal | `test_split_k_determinism_is_not_assumed`, and the paging pairs, which are class (a) by construction |

The definitions below are kept because the two contract classes are still the
vocabulary Tier 2 reasons in.

**M1 — CENSUS.** Decodes the template arguments of every ratified instantiation
from the mangled names in the manifests, and requires every axis to be **declared**
— as a GEOMETRY (which then needs rows), a PROBLEM_SHAPE, or a SEMANTIC arm gated
in Tier 1 — with matching value sets in both directions, and every geometry pair
the census rule requires to have a row. It runs at **module import**, so adding
a new value to a live axis (e.g. a fifth `D`) without declaring its contract class
**fails at collection**. It also pins the manifest it reads to `BUILD_CONFIG["manifest_sha256"]`,
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

**Each meta-test was verified by construction**, and the standard outlives them:
M1 was pointed at a fabricated product carrying a new value on an existing axis
(`D = 5`), M3 at a `P > 1` launch forced to degrade to `P = 1` and at a genuine
engineered identity mislabelled as a reassociation, M2 at a probe that cannot move
its output. *A gate that cannot fail is not a gate* — which is why the conformance
tier's own checkers are each fed a violating fixture, and why the bf16 family gate
ships M3-shaped mutant rows that must breach.

**A contract class is declared, never inferred from behaviour**, and changing one
is a reviewed edit with a new stated basis. That was a rule about `matrix.py`; it
is now a rule about the file that holds the pair.

## The build-shape gates

Four checks are about the SHAPE OF THE BUILD rather than about arithmetic, and
they are here because each one guards a property whose failure leaves every other
gate green.

**`python tools/gen_shards.py --check`** regenerates the generated instantiation
sources into memory and diffs them against the committed files — post-D2, **5
generated files, 42 instantiations over 3 shards** (it was 10 files over 4 shards
plus the tiled consumer's `dispatch_table.inc`). Generated sources are
checked in so that what is reviewed is what compiles; this is what keeps that a
fact. `setup.py` runs it before compiling and CI runs it in the CPU lane.

**`tests/unit/test_shard_partition.py`** checks the partition as a STRUCTURE:
membership is a function of the declared ARM LIST and nothing else (a shard is a
contiguous block of `CHUNK_ARMS` crossed with the state axis, and both state
variants of an arm ride the same shard), the committed files are the generated
files, `chunk.cu` does not include the header that would let it instantiate the
kernels locally, and the ratification on disk is the one `_build_config.py`
recorded at build time. The tiled partition derived its sharded AXES from a
priority list; the chunk matrix is an explicit arm list with no axes to
prioritize, so the block rule makes the same statement over the structure that
exists. CPU only.

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

`test_derive.py` (the `BC` derivation, term by term) and `test_dispatch_table.py`
(the tiled dispatch table joined against the fatbin's entry points) were the third
and fourth of these gates. Both retired with their subjects in P67 D2: there is no
derivation left to gate — `BC` belongs to the built arm — and no dispatch table to
census. The binary-side half of the census question survives, in a stronger form
than either: `tools/fatbin_entries.py --check` above, and the DEVICE-SIDE entry
list the build reports (`chunk_arms()`, `chunk_build_stamp`), which answer out of
the running binary rather than out of a host table describing it.

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
pytest tests/conformance -m "not cuda"                                       # conformance meta + census, CPU
flock /tmp/rola_gpu.lock pytest tests/unit -m cuda                           # tests/unit's CUDA-marked rows
flock /tmp/rola_gpu.lock pytest tests/tier1_semantics -m cuda                # TIER 1
flock /tmp/rola_gpu.lock pytest tests/tier2_geometry                         # TIER 2 (incl. the meta-tests)
flock /tmp/rola_gpu.lock pytest tests/layers                                 # layer <-> op, at every dispatch threshold
flock /tmp/rola_gpu.lock pytest tests/conformance                            # CONFORMANCE families (incl. decode + backward)
ROLA_FLA_CROSSCHECK=1 pytest tests/unit/test_oracle_vs_fla.py                # one-time, opt-in -- see below
ROLA_GRADCHECK_INSURANCE=1 pytest tests/conformance/test_backward_family.py  # one-time, opt-in -- see above
```

**The lane, measured** (2026-08-06, sm_86, warm build): CPU 70 s; unit-cuda
2.5 s; tier 1 123 s; tier 2 9 s; layers 6 s; conformance 4.5 s — the WHOLE
battery was ~3.6 min, inside the < 5 min commit-lane goal, so the commit lane
IS the full battery above and no per-commit subset needs choosing. The
same battery, scoped as `tests/tier1_semantics tests/layers tests/unit
tests/tier2_geometry tests/conformance -q`, is **860 passed, 5 skipped, 0
xfailed in ~18 s**, and the tree carries no xfail at all. Three retirements
account for the shape of that number: the tiled arm's gates left with the tiled
arm (P67 D2), the routing-summary surface took 244 rows with it (P71-A), and the
fast-fail retired the oracle-dispatch rows outright (P71-B — every `skip` that
read "decay routes to the oracle now" went with them, which is where 13 skips
became 5) —
`test_docs_mirror.py` and `test_shard_partition.py` each parametrize one row per
mirror doc and per csrc source, so a deleted surface moves the count by more than
its own tests. The
build-shape gates and `tools/ratify.py` run additionally at a batch tip (and
always when csrc, manifests or codegen flags moved); the latency gate and the
sanitizer lane are scheduled, never per-commit. The runtime x
unique-assertion audit that sized this (evidence:
`workflows/refactor/p07_conformance.md`) removed nothing: every tier-1 row
carries a mutant-proven charge, every tier-2 row is census-required, and the
conformance tier is new coverage at 4.5 s.

`tests/unit/test_oracle_vs_fla.py` is SKIPPED BY DEFAULT and that is the design.
It checks the oracle's recurrence against `flash-linear-attention`'s naive GLA
loop under a renaming, once, to catch a transcription error that every in-tree
gate would be blind to -- and then stops, because a fast-moving kernel library
is not something this tree takes a standing dependency on. Measured at adoption
(fla 0.4.2): worst `2.98e-08`, which is float32's resolution, since FLA
accumulates there.

**THE LOCK IS `/tmp/rola_gpu.lock` — that exact path, with an underscore.** `flock`
is advisory and keyed on a NAME, so a second spelling excludes nothing and silently
buys nothing: two commands holding two different paths run on the device at the same
time and neither can tell. Every command in this repository, every benchmark's usage
line and every agent brief uses this one path. If you find another spelling anywhere,
it is a defect — fix it rather than adding a second lock beside it.

**THE `rola-probe` EXCEPTION IS RETIRED.** The probe harness took this lock
itself, so wrapping it deadlocked, and the standing rule carried an exception for
exactly one command. That harness retired with the tiled measurement apparatus in
P67 D2 (`docs/measurement.md`), so the rule has no exception today:

> **everything that reaches a device is wrapped in `flock /tmp/rola_gpu.lock`.**

A successor bench harness for the chunk arm OWES a locking statement of its own —
either it takes the lock itself and is invoked bare, or it does not and is
wrapped — stated in its usage line, because the failure mode of guessing is a
silent shared device on one side and a permanent hang on the other.

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
`test_interface.py`'s dispatch rows) the same way it deselects `tier1_semantics`'s
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
| `tests/unit/test_perf_ledger.py` | a row missing its provenance, a row from an unknown schema, a shift smaller than its own scatter, a single unlucky run, an effect with no significance behind it |

`test_probe_harness.py` (the discipline gates: a lock held by another process, a
clock that never settles, a corrupt ncu section file, a subcommand offering to
skip the lock) and `benchmarks/test_latency_regression.py` (a baseline taken with
a different stopwatch, and one that does not say which) went with the probe
harness in P67 D2. Their claims are part of what the chunk-arm harness card owes
([`measurement.md`](measurement.md)); until it lands, `test_perf_ledger.py` is the
one file holding this machinery.

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

The lane's DRIVER was `rola_probe sanitize`, retired in P67 D2 with the rest of the
harness; the lane's shape and its measured findings are kept here because they are
what the successor has to reproduce, and re-deriving them from memory is how a
finding turns into a belief.

**A scheduled lane over a named subset, never a PR gate** — the shape every surveyed
adopter converged on, because the tools cost one to two orders of magnitude and a gate
that takes an hour gets disabled. `--error-exitcode=1`, a committed suppressions file,
and an explicit skip-list whose entries carry their reason.

Measured on this box, on the TILED consumer's two registered probes
(`consumer_forward`, `consumer_forward_deep`): memcheck, initcheck and synccheck
reported 0 errors; racecheck reported 0 hazards, 0 warnings. **No such measurement
exists for the chunk arm yet** — that is a gap, not a carry-over.

**Sold honestly, in both directions.**

* It does **not** own our historical bug class. Both escaped races were host-side
  Python ordering; `racecheck` detects on-chip shared-memory hazards only. The
  stream-discipline fixture above is what covers that class.
* `racecheck` ran on both registered probes with an EMPTY skip-list. The `Gm`
  write/read edge that once needed an exemption was ordered by the visit's own
  `__syncthreads()` — exactly the barrier class `racecheck` orders shared memory
  by — so no edge was left that a counter protocol published instead. An empty
  skip-list is the standard the successor lane inherits.
* `--leak-check full` is **not** used: torch's caching allocator retains its pools by
  design, so the flag reports 3 "leaks" on a clean run — a lane whose clean state is
  three errors is a lane whose findings nobody reads.
* An overrun that stays **inside** a caching-allocator segment is invisible: tensors
  are suballocated, so that memory is legitimately the allocator's. The lane detects
  accesses that escape the segment, and the planted-violation test plants exactly one.
