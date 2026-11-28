# Precision

> **Every accumulator is fp32 or an exact integer, but one: the output's sum across CTAs is fp32 or f16, chosen on the
> device a call at a time.** Storage and operands are one family on every card: u8 rows,
> raw bf16 values, bf16 operands and a bf16 output (`docs/CONSTITUTION.md#r17`).

The distinction is the whole document, so state it first.

| | What it means | Rule |
|---|---|---|
| **ACCUMULATION** | the format a value is *added into* across steps and carried: the fold's state, the mass (the state's v+1 channel), the readout's and the den's sums, the output's sum inside a CTA | fp32 or exact integer, closed. Not a default, not a knob, not a card's. The one exception is the output's sum across CTAs, whose width each call chooses on the device (the table below). |
| **STORAGE and OPERANDS** | the format a value is *held in* and read, without being accumulated into: routing rows, values, the stored state between calls, an MMA's operands, the output | declared per component, each with its quality measured against fp64. |

A read-only value that is 1 ULP wrong contributes an error bounded by its own magnitude, once. An accumulator that is
1 ULP wrong embeds that error and carries it for the state's lifetime: normalization bounds what reads contribute;
nothing bounds what accumulation embeds.

<a id="components"></a>
## The components

| component | storage and operands | accumulation |
|---|---|---|
| the route bundle | every level, either side, dense or sparse: u8 at its row max (255), a zero a zero and no floor under it; an fp32 scale a (token, level). A dense level differs from a sparse one only in having no list | -- |
| <a id="values"></a>the values | raw bf16, `v_proj`'s output as it is: no scale and no normalization, the precision class FlashAttention's values have (NOT BUILT: `fsla/routing/quant.py`'s `VALUES` form still scales them) | -- |
| the stored state | s24 a leaf-channel (an s16 hi plane and a u8 lo plane) at an s8 power-of-two exponent a (leaf, 32-channel group); the mass fp32 | -- |
| the fold | the coefficient of a (token, leaf) -- its levels' bytes with the token's row scales, multiplied in fp32 where the bytes are expanded -- rounded to bf16 once; the values bf16 as stored | fp32 |
| the readout | the state as a bf16 operand; the read coefficient bf16 | fp32 |
| <a id="intra"></a>the intra | a window's token pairs' products exact: the u8 rows' Grams, an integer MMA or a bf16 MMA on the digits widened exactly at staging, the form the card's ([below](#mma-form)); the pair's weight bf16; the values bf16 | fp32, or the integer MMA's exact sum |
| the output's sum | a token's partial sums from every CTA that reads it | fp32 inside a CTA; across CTAs fp32, or one f16 line a CTA a token, scaled by a power of two from a per-channel bound, `max|S_entry, ch|` carried with the state plus the call's growth from its inputs; f16 where the carve's piece count is at most about 150, chosen on the device a call at a time ([`hopper-carry.md`](hopper-carry.md#output-line)) -- 0.083% a token at the median with 512 leaves a CTA (recorded 2026-11-11) |
| decode | the bundle's token, the stored state | each written element updated in fp32 and rounded to its group's unit (2^-23 of the group's largest) |
| the output | `y = num / den`, divided once by the finish step from the sums and written in bf16, the one rounding on the output side; `o_proj`'s bf16 GEMM reads it as it is | -- |

A u8 byte is exact in bf16 (8 significant bits), so a row costs no rounding as an operand; a scale, a coefficient or a
power of two is multiplied in fp32 where the byte is expanded. Nothing in the family's arithmetic depends on the card:
there is no image of the state at a chosen exponent, no pass that bounds a token's sum before it is formed, and no
switch of accumulation by a card's MMA rates or by a side's density: the one width a call chooses is the output's sum
across CTAs.

<a id="mma-form"></a>THE COMPUTE IS NOT TIED TO INTEGER MMAS. The u8 digits are the storage on every card; a kernel's
MMA form for them -- an integer MMA, or a bf16 MMA on the digits widened exactly at staging with fp32 accumulation -- is
chosen per card from the device's facts in its descriptor, by measurement, and selected where the primitive is defined
(the closed world), never by a run-time branch on a measured statistic. Both forms are exact, so the result is the same
on every card: an integer 0 to 255 is exact in bf16, and a width-256 Gram's sums, under 2^24, are exact in fp32; an
exactness lab row gates each card's form. The intra's Grams on hopper and B200 are integer u8 MMAs, at the full INT8
rate. NOT MEASURED: no card's form is chosen by a measurement yet, and no exactness row exists. A card whose INT8 tensor rate is cut against its other MMAs has no form yet (no such card is measured): one
is decided and measured when such a card is a target, and it widens on chip inside the intra's Gram step only -- the
data model stays u8 in memory on every card.

WHY NOT 16-BIT FLOAT OPERANDS WITH 5 EXPONENT BITS. The fp16 family (the branch `ampere`'s) kept an unbounded fp32
state and a token's unbounded sum inside f16's range by machinery: the state's image at an exponent a digit, list
exponents, pre-scaled coefficients, a dense row's floor at 2^-13, a pass that bounded each token's sum, and a switch of
accumulation by card. Its last measurement (2026-11-09, sm_86 (ampere branch, unreproducible here)) closed it: with each (token, digit)'s factor on the
read rows before the MMA, the rows under f16's normal floor 2^-14 were 32% at carry-alt-k4-256x256-T1024, 71% at carry-alt-k16-256x256-T1024, 57% at
carry-cohort-k16-c64-256x256-T1024 and 100% at carry-dense-256x256-T1024. bf16's 8 exponent bits hold every one of those values with no machinery, at 8
significant bits where f16 has 11.

<a id="quality"></a>
## Quality

Per token, `|dy| / |y|` against fp64 on continuous values (`docs/testing.md`).

MEASURED on kernels (2026-10-03 and 2026-10-12, sm_86 (ampere branch, unreproducible here), the carry's own output, median a token): a bf16 family 0.16-0.20% on the flagship cells; the fp16 family 0.018-0.029%. The class is accepted on every
card (Blake, 2026-11-09): it is the bf16 output store's own, the op returning `y` in bf16 as FlashAttention and torch's
SDPA do. The values are raw bf16 for the same reason (Blake, 2026-11-28: "everyone uses bf16 right. lets just be simple
and use raw bf16"): a scale a token would buy accuracy past that class, which has no value here, at a normalize in the
producer, 4 bytes a token and one more rounding.

MODELLED for this family before any kernel of it exists (recorded 2026-11-09: the fp64 recurrence on the formatted operands, rounded only where a kernel of the family rounds; the
median / the 99th percentile a token): carry-alt-k4-256x256-T1024 0.134 / 0.202%, carry-dense-256x256-T1024 0.132 / 0.191%, carry-alt-k4-256x256-T65536
0.115 / 0.174%, carry-dense-256x256-T65536 0.021 / 0.167%. The intra's weight, rounded once a token pair, carries the worst tokens
everywhere; the fold's coefficient, the state as an operand and the read coefficient average over the leaves a token
reads, and each alone costs under 0.02% at the 99th percentile on the dense cells and at most 0.10% on the alt-k4
ones.

THE SUM'S WIDTH IS THE CALL'S because f16 costs by the partials it joins: 29 f16 lines
a token (512 leaves a CTA, added in fp32 first) are 0.083% a token at the median on carry-alt-k4-256x256-T65536, 58 are 0.115%, the
family's own error being 0.115% (recorded 2026-11-11; MODELLED on that one
sparse cell -- a dense read side and wider levels are NOT modelled). So every CTA computes the same piece count from its
carve and the call's live digits and takes f16 where it is at most about 150, fp32 above, the host allocating the sum's
buffer at fp32 size so no launch waits on the choice ([`hopper-carry.md`](hopper-carry.md#output-line)). The
per-channel bound's error is NOT MODELLED here. A sum a fold group with the groups merged in fp32 reads
0.023%, and is no form: its merge costs what the fp32 line does
([`panel_merge`](../lab/panel_merge.md)).

<a id="why"></a>
## Why accumulation stays fp32

A narrowed resident state is refused because the error it embeds walks. Measured on a bf16 state: the re-quantization
walk is `sqrt(T)`-exact (1.06e-2 at T = 512 against a 1e-2 budget, ~1.8e-1 extrapolated to N = 65K), composing in
quadrature (log-log slope +0.490 against theory's +0.5, 150 paired runs); decay does not flatten it, since decay
shrinks the signal and the old noise together while fresh rounding arrives at the current magnitude.

Three arguments make the rule permanent, because they are about what a measurement can see:

1. **The governing write count is a lifetime, not a launch.** A store-back at higher precision preserves an embedded
   error; the state carries it across calls, and decode compounds it.
2. **Nothing bounds a burst.** A burst of small writes re-quantizes repeatedly without decaying.
3. **The oracle cannot see it.** The reference is per call; cross-call compounding is invisible to the gate that
   would have to catch it.

The same rule holds outside the state: decay dials accumulate in fp32, because `1 - rate` near 1 cancels
catastrophically inside a narrow product (`fsla/routing/`).

<a id="gates"></a>
## What a precision change owes

- **The gate is written before the change**, with a length-scaling arm (`T` in {512, 2048, 8192}) and a halt on any
  monotone trend: a gate without it cannot see a `sqrt(T)` accumulation at all.
- **The gate is shown capable of failing.** An output-charged assert whose maximum sits at token 0, where the state has
  been re-quantized zero times, cannot detect a resident-state walk of any size.
