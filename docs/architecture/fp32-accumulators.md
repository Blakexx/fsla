# fp32 accumulators

> **Accumulators are fp32, permanently.** Storage precision is a live,
> measurable axis. Accumulation precision is closed.

The distinction is the whole document, so state it first.

| | What it means | Status |
|---|---|---|
| **ACCUMULATION precision** | the format a value is *added into* across steps: the value state, the mass state, the decay clock, the split-K emission accumulators | **closed at fp32.** Not a default, not a tuning knob — a ruling. |
| **STORAGE precision** | the format a value is *held in* and later read, without being accumulated into: amplitudes, the value tensor, projection GEMM inputs, the `y` output cast | **live.** Each surface gets a precision gate written first, then an oracle A/B, then a performance measurement. |

The two look alike in a dtype table and behave nothing alike. A read-only tensor
that is 1 ULP wrong contributes an error bounded by its own magnitude, once. An
accumulator that is 1 ULP wrong **embeds** that error and carries it forward for
the lifetime of the state.

## What was measured

The proposal was to hold the resident state in bf16, halving its shared-memory
footprint. It was built, gated, and **declined on measurement** (campaign
journal, 2026-08-03, "batch-1 verdict"):

- The re-quantization walk is **real and `sqrt(T)`-exact**: `1.06e-2` at `T=512`
  against a `BF16_RTOL` budget of `1e-2` — already over — and extrapolating to
  `~1.8e-1` at `N ≈ 65K`. The standing design-for-scale rule fired: a `T=37`
  fixture would have passed.
- The composition law was then measured directly rather than assumed
  (150 paired runs, 10 cells × 5 `T` × 3 seeds): the walk composes in
  **quadrature**, log-log slope `+0.490` against theory `+0.5`, with the
  quadrature constant flat in `V` (spread 1.06×) where a linear law spread 1.35×.
- **Decay does not flatten it.** Decay shrinks the signal and the old noise
  together, while fresh rounding always arrives at the current magnitude.
- The fp32-mass argument was **vindicated**: keeping the mass column fp32 left
  results bit-unchanged, because a row-uniform error in the value columns cancels
  in the readout ratio while a per-element error in the denominator has no
  cancelling partner.

The cost of declining was not small and was paid knowingly: the freed shared
memory was the purchase for a larger `BC`, and with it went a designed path from
~18.4 ms to ~9.2 ms on the consumer kernel. The decline was made anyway, because
the failure it avoided is silent.

Three further justifications for bf16 state were raised afterwards and each died
on its own measurement — DRAM traffic (state is 0.36% of it), occupancy (0.85
eligible warps per scheduler; the kernel is not occupancy-starved), and
ring-depth enablement (depth 2 is not shared-memory-blocked; freeing bytes makes
depth 2 available on *fewer* instantiations, 148 → 128). The record of a
proposal surviving three separate rehabilitations and failing each is itself part
of why the ruling is now permanent rather than "not today".

## Why the ruling is permanent, not a tuning default

The final ruling (campaign journal, 2026-08-05 late, "bf16 state accumulation
killed") rests on three arguments that no measurement can reverse, because they
are about what a measurement can *see*:

1. **The governing write count is a lifetime, not a launch.** An fp32 store-back
   does not repair a bf16-embedded error; it preserves it. The cache carries it
   across calls and decode compounds it. (The earlier belief that "the walk resets
   at the fp32 boundary" was wrong in the store direction and is recorded as
   wrong.)
2. **Nothing bounds a burst.** Per-write mass decay gives no guarantee of
   attenuation per event: a burst of small writes re-quantizes repeatedly without
   decaying. The "small writes are cheap" defence required a discount that the
   convexity of the composition law does not permit.
3. **The oracle structurally cannot see it.** The reference is per-call. Cross-call
   compounding is invisible to the gate we would rely on to catch a regression.
   A property no gate can see cannot be an engineering choice; it has to be a
   structural one.

The derived standing conclusion, in the ruling's own words: **normalization
bounds what reads contribute; nothing bounds what accumulation embeds.**

There is a second, independent instance of the same rule already in tree, and it
is worth citing because it is not about the state: the decay **dials** are
accumulated in fp32 and never bf16, because `rate` near 1 makes `1 - rate`
catastrophically cancelling and `log1pf` cannot repair a cancellation that has
already happened inside the product
([`consumer.md#dials`](../internals/consumer.md#dials)).

## What stays open

The bf16 **read-only memory** family is untouched by this ruling, and is
strengthened by contrast — no accumulation, therefore no walk. It is one
decision over five surfaces, all keyed to the MMA precision axis so that the
exact arm stays fp32 end to end (journal, 2026-08-04 night, phase-2 scope):

1. spanned-level amplitudes;
2. the value tensor's HBM storage;
3. `v_proj` / `o_proj` as bf16 GEMMs with fp32 accumulation;
4. `y` cast to bf16 *after* the divide — the divide, the numerator and the
   denominator accumulators all stay fp32;
5. the host-boundary fp32-only check, relaxed.

Note what items 3 and 4 preserve: **bf16 operands feeding fp32 accumulation is
not a violation of this document.** The MMA already works that way. The rule is
about the accumulator, not the operand.

Two conditions govern any of these landing, both learned the expensive way:

- **The precision gate is written before the change**, not after, and it carries a
  `T`-scaling arm (`T ∈ {512, 2048, 8192}`) with a halt on any monotone trend.
  This exists because bf16 state reached a working build before any instrument in
  the suite could see a `sqrt(T)` accumulation.
- **A gate must be shown capable of failing.** The committed output-charged assert
  in that same family was found to be **bit-identical between the fp32 and bf16
  arms in 150/150 runs**, because its maximum is pinned at token 0 — where the
  carried mass is still tiny and the state has been re-quantized zero times. It
  could not have detected a resident-state walk of any size at any `T`. A gate
  that cannot fail is not evidence.

## Provenance

- Batch-1 decline and the walk measurement: campaign journal, 2026-08-03; the
  preserved instruments `i6_walk.py` / `i6_numbers` / `occ32` and the declined
  patch `rung1_bf16_state_DECLINED.patch`.
- Composition law, `V_max` derivations and the gate defect: `vmax_budget.md`
  (150 paired runs), journal 2026-08-04.
- Final kill and the standing conclusion: campaign journal, 2026-08-05 late.
- The dials instance: [`docs/internals/consumer.md#dials`](../internals/consumer.md#dials).
