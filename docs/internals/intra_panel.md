# `csrc/rola/src/intra_panel.cuh` — the `panel` intra lowering

The intra lowering under `LeafMassDecay` (`ROLA_V3_SPEC.md` §7.4.1), and the
decay arm's ONLY lowering. There is no `intra_lowering` template axis and no
alternative path: `panel_read_factor` and `panel_write_factor` are applied on
every decay tile the consumer executes, and `rola/ops/consumer.py` therefore
enforces the admission bound (§6) on every decay launch rather than on a
selected one.

Symbols used throughout: `i` is the owner's leaf; `t` and `j` are tokens of one
tile; `BT` is the token tile, `BC` the leaf tile, `DV` the value width; `R` is
the read amplitude panel, `W` the write amplitude panel, `M` the state,
`rate_i` the per-leaf decay dial, `keep_i = 1 - rate_i`.

---

## <a id="what-it-replaces"></a>1. What it replaces, and why

Under `LeafMassDecay` a MIXED tile (one that both reads and writes) cannot use
the "carry against a fixed tile-entry state plus a causal Gram" split, because
the state evolves inside the tile. The kernel's standing answer is the
TOKEN-SEQUENTIAL WALK: walk the tile's tokens in
order, `M *= keep; M += W*v; y = R*M`. That is exact and `O(BT * BC)` per tile —
but every one of those `BT * BC` FMAs is in the thread→channel map, the walk
carries a loop-carried dependence over `t`, and nothing in it reaches the tensor
cores: neither the carry (which the `decay = None` arm does with the §5 MMA
readout) nor the intra term.

Spec §7.4.1 states the alternative: *`LeafMassDecay` → decay-aware panels:
survival is a per-leaf scalar, so fold cumulative survival into the R/W
factors.*

## <a id="the-factorization"></a>2. The factorization

Exact in exact arithmetic; the fp32 statement is §6. All indices are tile-local,
and the intra term needs `j <= t` (§5's inclusive read-after-write).

```
keep_i           = 1 - rate_i                 log_keep[i] = log1pf(-rate_i) <= 0
P[t,i]           = sum_{u <= t} c[u,i]        (INCLUSIVE prefix of the clock)
survival(j -> t) = prod_{u=j+1..t} keep[u,i]  = keep_i^{P[t,i] - P[j,i]}
```

The exponent splits across the two operands of a Gram because it is a SUM in `t`
and in `j` separately — which is the whole content of "survival is a per-leaf
scalar". With a per-leaf reference exponent `P_ref[i]` (§3):

```
S0[i]   = exp(P_ref[i] * log_keep[i])                          (<= 1)
R~[t,i] = R[t,i] * exp( (P[t,i] - P_ref[i]) * log_keep[i])
W~[j,i] = W[j,i] * exp(-(P[j,i] - P_ref[i]) * log_keep[i])

R~[t,i] * W~[j,i] = R[t,i] * W[j,i] * keep_i^{P[t,i] - P[j,i]}
```

so one mixed tile becomes, verbatim, the `decay = None` tile:

```
carry[t] = sum_i R~[t,i] * (S0[i] * M0[i])            -- the §5 MMA readout
intra    = causal_mask(R~ @ W~^T) @ (v * scale_write) -- the §5 causal Gram
M_end[i] = S0[i] * ( S0[i]*M0[i] + sum_j W~[j,i]*gw[j]*v[j] )
```

**The lowering therefore deletes a path rather than adding one.** `panel` is not
a third tile algorithm: it is the `decay = None` tile algorithm, reached under
`LeafMassDecay` by pre-scaling the entry state into a per-leaf reference frame,
folding the two exponentials into the two panels, and post-scaling out. That is
why it composes with `GLOBAL`, `COMPACT`, `BT`, `BC`, `mma_precision` and §6.4's
`P` with no new interaction: every one of those axes is already exercised by the
path it joins. The one genuinely new object is the pair of exponentials, and
they are where `GLA_MASS_DECAY_SPEC.md` §12 rules 1 and 3 go live — `readout_mma`
records that rule 1 becomes live with exactly this construct and nowhere earlier.

## <a id="rule-1"></a>3. §12 rule 1 — mid-tile clock normalization, and what `P_ref` is

`P_ref[i] := P[BT-1, i] / 2` is the MINIMAX reference. The largest exponent
magnitude either panel forms is `max_t |P[t,i] - P_ref[i]| * |log_keep[i]|`, and
over a non-decreasing `P` that is minimized — at exactly half the tile's total —
by the midpoint of the range. No other single per-leaf reference does better, and
a per-`(t,j)` reference does not exist: a Gram contracts `i`, so the reference
must be constant along the contracted axis.

Rule 1's documented "square-root mitigation" is therefore not an approximation
here, it is the optimal choice of a free parameter. Both scales come out equal —
`P[BT-1] - P_ref == P_ref` — so the tile's entry scale and exit scale are the
same `S0`.

## <a id="rule-2"></a>4. §12 rule 2 — no epsilon floor

There is none, and the asymmetry of the clamp in §5 is what keeps it that way:
`exp` UNDERFLOWING to zero is the correct limit of `keep^{large}` and is left
alone. Flooring it would alter the retention contract, which rule 2 forbids.

## <a id="rule-3"></a>5. §12 rule 3 — the finite clamp, on both panels

`W~[j,i] = W[j,i] * keep_i^{-(P[j,i] - P_ref[i])}` is a DIVISION by cumulative
survival and grows without bound as `P[j,i] -> 0` (late `P_ref`, early `j`);
§7.4's audit table names this as rule 3's reason to exist. But it is not the only
quantity that grows: `R~[t,i] = R[t,i] * keep_i^{+(P[t,i] - P_ref[i])}` grows
equally without bound as `P[t,i] -> 0` (early `t`), by the SAME magnitude. Both
panels' exponents range over `[-E, +E]` with

```
E := max_i P_ref[i] * |log_keep[i]|
```

and `panel-review.md` §1 confirms `max |read exp| == max |write exp| == E`
numerically. `panel_read_factor` carries the identical
`fminf(., kPanelMaxExponent)` clamp for exactly this reason: it is not a
defensive copy of the write side's guard, it is load-bearing on its own.

`kPanelMaxExponent` clamps the exponent ARGUMENT, not the result, at `88` —
`expf(88) = 1.65e38`, finite in fp32 *and* in bf16 (bf16 shares fp32's 8-bit
exponent, so `clamp_bf16_finite` in the readout is a no-op on an already-clamped
panel entry). The clamp is one-sided by construction (§4).

## <a id="admission-bound"></a>6. The clamp is a finiteness guarantee, not an accuracy one

Which is why the lowering has a **plan-time admission bound**. Rules 1 and 3 are
"a pair, not alternatives" (spec §7.4) and the pair is still not sufficient at
unbounded dials. Past `E ~ 88` two things break, in either order depending on the
cell: `W~` clamps DOWN while `R~` underflows toward zero, so their product —
`O(1)` at `t ~ j` — is computed as `~0`; and the deposit sums `BT` terms of size
`exp(88) = 1.65e38`, which overflows fp32 to `inf` and then meets the exit scale
`S0 ~ 0` as `inf * 0 = NaN`.

MEASURED (`panel.md` §2, RTX 3080 Ti, `widths=(4,4) BC=16 BT=32 T=64`, one-hot
write clock so the bound is EXACT, dials swept to move `E` alone; error relative
to the fp64 oracle):

| `E` | 20 | 40 | 60 | 80 | 84 | 88 | 120 | 200 |
|---|---|---|---|---|---|---|---|---|
| `panel` | 1.4e-7 | 1.4e-7 | 1.2e-7 | 1.7e-7 | 1.4e-7 | NaN | NaN | NaN |
| `walk` | 6.1e-8 | 6.3e-8 | 6.0e-8 | 7.2e-8 | 6.5e-8 | 6.4e-8 | 6.8e-8 | 5.9e-8 |

The transition is AT the clamp and it is sharp: exact to `E = 84`, gone at
`E = 88`. `PANEL_MAX_EXPONENT = 40` is therefore a 2.1x margin on a measured
cliff, not a guess. The walk is flat across the whole sweep, which is spec §7.4's
per-lowering verdict made visible: survival as a running product has no range to
lose.

So the domain is CHECKED rather than documented.
`routing/schedule_v3.panel_exponent_bound()` computes a cheap EXACT UPPER BOUND
on `max_{i,tile} P_ref[i] * |log_keep[i]|` from the dials and the clock stream
(no `N`-shaped materialization:
`prod_l c_l[t, d_l(i)] <= prod_l max_d c_l[t,d]` and
`g_i <= -log1p(-prod_l max_d delta_l[d])`), and the driver REFUSES a `panel`
launch above `PANEL_MAX_EXPONENT = 40`. Inside that domain the clamp provably
never fires; outside it the launch is rejected loudly. The clamp remains, because
"provably unreachable" is a property of the admission check, and a kernel should
not be one edit away from producing `inf`.

## <a id="no-log-space"></a>7. Why there is no `-inf * 0` NaN corner here

Architecture §6.2.1's design note describes a LOG-SPACE variant of this lowering
(log the per-level factors, select digits with a binary selector MMA, exponentiate
in the epilogue) whose one numeric corner is `-inf * 0`. That variant is **not**
what this file implements; the decision, with its arithmetic, is recorded in
`workflows/20260730-wiring/panel.md` §1.

In the direct staged-factor form implemented here the amplitude is never logged:
a structurally zero factor stays an exact `0.0f`, is multiplied by a FINITE
(clamped) exponential, and yields an exact `0.0f`. Sparse entmax routing
therefore reaches the oracle's exact zero bit for bit, and §12 rule 4's
branchless write-side annihilation (`a == 0.0f`) still fires on a structurally
dead route — which it would NOT do if a `-inf` had been clamped to `-88` and
exponentiated to `6e-39`. See also
[the self-masking theorem](../architecture/self-masking-theorem.md).

## <a id="measured-status"></a>8. What the lowering costs, and the standing follow-up

The decay arm's per-tile cost has two components that are strictly ADDED relative
to the `decay = None` tile it reduces to, and one epilogue defect that dominates
both:

1. **A mixed tile emits TWO atomics per (token, channel).** The lowering takes
   the `decay = None` path, whose readout and intra term land as separate split-K
   atomics ([`consumer.md#split-k-emission`](consumer.md#split-k-emission)).
   Under `LeafMassDecay` with any write density most tiles are MIXED, so this
   doubles the epilogue traffic on almost every tile. **Fusing the intra term
   into the readout epilogue is the standing follow-up**, and it is the largest
   single item on the decay arm.
2. TWO `expf` panels (`2 * BT * BC`) rather than one, plus the clock PREFIX on
   top of the clock product. MUFU-bound work.
3. Against those, the carry moves to the tensor cores, the intra drops from
   `BT * BC` per THREAD to `BT(BT+1)/2 * KI / DV`, and the serial `t` dependence
   goes.

The measurement that sizes item 1 was taken against a token-sequential
alternative that fused carry and intra into one accumulator before a single
`atomicAdd`. At `B=1 H=8 L=2048 d_v=64 branches=[8,16]`, real producer routing,
30 reps, median, `compact=false`, RTX 3080 Ti, the fused form ran 12% to 106%
faster across 16 cells (two routers, two norms, two `BT`, two `BC`); the
`BC = 16` cells carried the largest deltas and are the least trustworthy in
magnitude, since a byte-identical kernel was measured moving 12–23% between
builds there. **That is the size of the prize for fusing the epilogue**, and it
is the number the fusion has to be measured against. (Provenance:
`workflows/20260730-wiring/panel.md` §4.)

## <a id="cost-model"></a>9. Per-tile cost, in counts

Counts, not a measurement; the measured figures are §8.

```
BT*BC deposit FMAs per thread, the carry on the tensor cores, the intra as
BT(BT+1)/2 * KI Gram entries spread over DV threads, and 2*BT*BC expf for the
two panels.
```

The survival exponential is `BT*BC` per panel and the lowering forms two, so it
is paid twice. Against that, `2 * BT * BC` per-thread FMAs and the serial `t`
dependence do not exist on this path at all: the carry is a tensor-core GEMM and
the intra is a Gram.

## <a id="fp32-prefix"></a>10. HAZARD — the clock prefix accumulates in fp32

`panel_clock_prefix`'s `acc` is fp32 and must stay fp32. This is a requirement,
not a default (spec §7.4's V3-specific rule): the prefix multiplies `log_keep`
and is then exponentiated, so a lost low bit here is an **exponent** error, not a
mantissa one. Narrowing this accumulator is a silent accuracy regression that no
compile-time check catches.

## <a id="clock-prefix"></a>11. `panel_clock_prefix` — the serial prefix and the scatter-fill identity

`clk[t*BC + i] = P[t,i]`, `p_ref[i] = P[BT-1,i]/2`,
`s0[i] = exp(p_ref * log_keep)`. One thread per leaf, serial over the tile's `BT`
tokens, because the prefix IS a serial dependence in `t` and `BT <= 32`;
parallelizing it would cost more barriers than it saves steps.

The clock product runs over ALL levels, unlike the read/write panels, whose loops
start at `first_spanned_level`: a fixed level's factor cannot hoist into the §3
step 3 row scale, because the clock is an EXPONENT.

**Scatter-fill identity**, exactly as the walk's survival panel: an inert leaf
(dropped by §3 step 2's gather) contributes `0` to the prefix, and that is not a
convention — `build_clock_stream` enforces `stored != 0 => live != 0` per
(level, digit), so a leaf the gather dropped has `c == 0` at every token of this
tile and its true prefix IS zero. `keep^0 == 1` then makes every factor this file
computes exactly `1`, which is §2's write-conditioned quiescence.

`s0[i] = expf(pr * log_keep[i])` needs no clamp: `pr >= 0` and `log_keep <= 0`,
so `s0 <= 1`. Underflow to zero is the correct limit (a leaf whose tile-total
survival is below fp32 has genuinely lost its entry state) and is rule 2's
sanctioned underflow, not a floor.
