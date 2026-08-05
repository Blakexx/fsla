# The planner

> **Dispatch on what the data DOES, not on a threshold over what it IS. Model
> with measured coefficients, never a hand-tuned constant and never a lookup
> table.**

Every other attention library dispatches on a key that is **enumerable and static
per call**: shape, dtype, architecture. For such a key, hand-authored conditions
and lookup tables are sufficient, and that is what the field ships — xformers and
TensorRT-LLM encode "harmful here" as static conditions; nobody benchmarks at
run time (`workflows/research/` dispatch-mechanics report, 2026-08-05).

RoLA's key includes **distribution statistics**: per-level occupancy, candidate
fraction, per-owner write counts. Those are continuous, per step, and never
recur. A lookup table cannot key on them — bucketing destroys the signal, and a
cache that cannot see a distribution will replay a config from the wrong regime.
Hand conditions on them are the banned tuned-threshold antipattern. So the
required form is the third thing: **a parametric cost model, evaluated fresh per
step, with coefficients calibrated offline by measured sweeps, and selection by
argmin.**

The distinction in one line: *the field dispatches on what the problem is; we
dispatch on what the data does* (user ruling, journal 2026-08-05).

## 1. The branching rule

**Branch on a computed structural quantity. Never on a raw dimension or value
threshold.**

- **Allowed**: `if constexpr (ND_V == 1)` — one tile or many, where the tile count
  came from a selected block size. A structural fact, resolved at compile time.
- **Allowed**: a template argument keyed to a *format or topology* axis the file
  already forks on — e.g. the phase-C gate narrowing, scoped by
  `stage_is_narrow<READ_PREC, FOLD_PREC>()`, "a precision/format axis the file
  already forks on in four places, carrying no size or threshold term"
  (journal, 2026-08-04, P01).
- **Forbidden**: `if (dv == 64)`, `if (check_shared_mem(...) == X)`, or any
  hand-coded device- or dimension-specific gate.

The reason is not aesthetic. A magic threshold is a claim about hardware that was
measured once, on one card, at one shape, and it is invisible to every later
measurement. A structural branch is compile-time, so the untaken path costs
nothing, and the *choice* of structure moves to one place — the planner — where
it can be audited.

The same rule sorts admissible constants from inadmissible ones inside the
planner itself. `MMA_K_QUANTUM = 16` is fine: `rola/planner/select.py:60`
annotates it "a SHAPE of the instruction the kernel issues, never a threshold."
A breakeven coefficient is fine *if it is a fitted number with a named producing
sweep*. A number chosen because it worked on the author's machine is not
admissible at all.

## 2. What the runtime planner is

`rola/planner/select.py` — `select_config` chooses `(BC, BT, packing,
feeder_flags)` from exactly three arguments: a static configuration record, a
hardware-facts record, and one `RoutingSummaries` from the producer epilogue. It
reads no tensor, issues no reduction, allocates nothing, and does not touch the
routing. **Nothing in it measures.** Every coefficient is a recorded number with
a named isolating cell, refitted offline and recompiled into a small rule table
(`RULE_TABLE_PROVENANCE`).

Its decision order is acyclic and documented as such (`select.py` module
docstring): feeder flags depend on nothing; packing is a closed form in
`(BC, BT, decay, hist, flags)` and depends on no residency; the `(BC, BT)` argmin
then queries the occupancy oracle **with the arm's actual footprint** and
minimizes `cost_per_token · waves + atomic`. There is no back edge. A per-owner
packing decision would create one; masks are per-tile and packing is per-launch
precisely so that it cannot.

The measured overhead of planning is `+0.15` to `+0.30 ms` on a `3.3 ms` plan
build inside a `73.2 ms` step — `+0.2%` to `+0.4%` (journal, 2026-08-04). The
binding costs of the approach are compile time and wave quantization, not
planning time.

## 3. The calibration machinery

Because the runtime must not measure, the measuring lives beside it:

| Component | Role |
|---|---|
| `rola/planner/calibration/census.py` | the **exact** per-`(owner, tile)` tile-union census, as packed-bit-word arithmetic. The ground truth the closed form approximates. Deliberately free of any dependency on the planner. |
| `rola/planner/calibration/cost_model.py` | the offline cost model: `plan_cost_per_token`, its six measured constants, and the exact argmin. This is the **fit** whose compiled output ships as `select.py`'s rule table. It does not run at run time. |
| `rola/planner/calibration/auditor.py` | runs **both** deciders — the shipped closed form and the exact argmin — on recorded cells and either asserts agreement or writes the disagreement down with its modelled regret **and its margin**. |
| `benchmarks/calibration/` | where the sweeps that produce the coefficients belong. **Today it is a `.gitkeep`.** |

Keeping the measuring machinery outside `rola/planner/select.py` is structural,
not cosmetic, and buys three things: the shipped package carries no
`torch.compile`/inductor dependency; **selection does not audit itself** (a
runtime shadow would keep alive exactly the object it is meant to replace, and
would denominate its regret in that object); and the constants are re-fittable
without touching `rola/`.

The auditor's own limits are stated in its docstring and should be quoted at
anyone who over-reads it: **it compares two models.** It cannot tell you a model
is right about the machine — that is the job of the measured argmin rows, which
are recorded numbers, not model outputs. What it *can* catch is a constant that
moved without a refit, a rule that stopped tracking the model it was derived
from, or a placeholder doing more than it was supposed to.

## 4. The honest state of the coefficients

This is the part a future reader most needs and is least likely to infer from the
code, because the code is complete and the *numbers* are not.

- `C_PACK_BREAKEVEN` (`select.py:98`) is **an empty dict**. With no row, the
  packing predicate declines, so the COMPACT arm is never plan-selected today.
  The docstring states rather than hides the consequence: on read-dense cells the
  default is inert (packing is structurally excluded anyway), but on the sparse
  at-scale cells **four of eight measured argmin rows select a compacted arm**,
  and the empty table declines all four. That regret is real and is quantified by
  a gate that runs the auditor twice — once at the shipped empty table, once at
  the derived breakeven table — and shows the disagreement belongs to the
  placeholder rather than to the rule
  (`tests/unit/test_planner.py::test_the_auditor_reports_the_documented_disagreement_ledger_exactly`).
- `C_SATURATION = 1.0` — exact-dense only. Deliberately the *only* defensible
  value until the occupancy sweep exists: any structural zero at all breaks the
  closed form the flag licenses, and a soft epsilon would re-introduce the
  distributional branch this design forbids.
- The cross-level collision correction `gamma = 1.0` — the bound uncorrected,
  pending a census fit. Its *placement* is load-bearing and already correct: it is
  applied per `(owner, tile)` before quantization, never to the mean, because
  `ceil` is nonlinear and the `K = 0` mass is where the packing saving lives.

**The deliverable that closes this** is named and unambiguous: fill the breakeven
tables with measured cost curves for the executor family — dense / atom-skip /
full compact on the leaf axis — **as a function of occupancy**, so the planner
selects per side per launch by the same argmin that picks `(BC, BT)`. The family
form is required and a point verdict is not admissible: designing against a
single benchmarked configuration is designing against a hole, and a selector with
measured curves absorbs wherever a trained model's routing lands, which a point
verdict does not (user ruling; journal, 2026-08-04, "no point-configuration
verdicts — executor FAMILY").

Until those rows exist, a claim of the form "configuration X is the right default"
is not supported by this repository, and the honest statement is the one made
here.

## 5. Two standing rules for any table this produces

1. **Staleness is a key problem, not a hygiene problem.** The kernel-source hash
   and the toolchain/architecture identity go **in the key** of any tuning or
   breakeven table, or a rebuilt kernel silently inherits a stale winner. This is
   solved well exactly twice in the field — Inductor's `template_hash`-vs-`src_hash`
   filter and XLA's versioned entries — and the project's own manifest digests
   already embody the instinct; extend it to every calibrated constant.
2. **The selector is swappable, by construction.** The pipeline is
   `bundle → selector → choice`; the dispatch table consumes *choices* only;
   today's cost model is one interim-labelled implementation; provenance keys
   mark the tolerated slop; the regret gate is selector-agnostic. The interface is
   what makes replacing the heuristics a swap rather than a rewrite
   (user ruling, journal 2026-08-05).

## 6. Where this sits relative to the field

Recorded so a future reader does not mistake the design for a gap: a
measured-cost-model planner is **ahead of** FlashAttention, xformers and
TensorRT-LLM on this axis, not behind. No mainstream attention library does
measured per-instantiation dispatch. The closest shipping prior art is CUTLASS
4.x wiring an analytic-model heuristic into the *build* (planner output as kernel
filter) and Inductor's lookup-table choices (mechanism shipped, table expected
from your own offline sweep). The nearest relative in kind is
nvMatmulHeuristics: an analytic model calibrated by measured discovery scans —
**their model eats shapes; ours eats distributions.**
