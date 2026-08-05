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
```

**EVERY LINE THAT CAN REACH A DEVICE IS `flock`ed, AND THAT IS NOT THE SAME QUESTION
AS THE `cuda` MARKER.** The marker's job is *collection-time deselection* — it is what
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
