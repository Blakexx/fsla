# `tools/scale_slack.py` — whether a cheaper bound can be the carry's f16 scale

The carry adds each token's partials across CTAs as f16 pairs at one scale byte a token
([`carry_api.md#carry-reduce`](../carry/carry_api.md#carry-reduce)): a power of two bounding |num(t)|, which f16's
exponent absorbs with slack to ~2^27. The den pass's bound is the exact read product against the mass of `F = m s`;
the tool asks whether a bound that reads less can stand in for it, on one bundle at a time.

    python tools/scale_slack.py nl64k-alt-k4 flagship-cohort-k16-L16384 [--json slack.json]   # carry cells
    python tools/scale_slack.py nl64k-alt-h4 nl64k-alt-h4-g16                                 # layer constructions

THE BOUNDS (`bounds`, fp64 in torch over the bundle's dequantized operands, window by window at the device's W; `x` a
read row, `w` a write row, `g` the write gain, `s` a writing token's value scale, each sum over the windows before):

| bound | form | reads |
|---|---|---|
| exact | `sum_a x_S(t, a) sum_r x_R(t, r) M(a, r)`, `M(a, r) = sum w_S(a) w_R(r) g s` | the den pass's own product |
| write | `(sum_r x_R(t, r)) sum_a x_S(t, a) G(a)`, `G(a) = sum w_S(a) max_r w_R(r) g s` | level S's read coefficients and level R's row sum: no level-R read row |
| facts | `(sum_a x_S(t, a)) (sum_r x_R(t, r)) sum max_a w_S(a) max_r w_R(r) g s` | each read row's sum and one sum a window |
| window | `(sum_a x_S(t, a)) (sum_r x_R(t, r)) max_a G(a)` | each read row's sum and `G`'s largest a window: no read coefficients |

Each is taken into the den pass's units (`den_units`: the reader's per-token row scales left out, as
`test_den_vs_oracle.py`'s `in_truth` puts them back) and through `rola.ops.prefill.scale_bytes`.

THE READING, a cell: the den pass's bound against `exact` (log2, a check of the units: 0); each candidate's SLACK,
log2 of it over `exact` (median, p99, max; any token below `exact` counted -- a bound must never be); and its SPREAD,
each token's largest channel difference over its largest channel between the carry at the candidate's scale and at
the den pass's, beside the RUN-TO-RUN spread of two runs at the den pass's scale. f16 is a floating format, so a
power-of-two scale changes no rounding unless a partial leaves f16's normals; but the sum's order across CTAs is the
atomics' arrival, so two runs at one scale already differ by f16's rounding. A candidate is harmless where its spread
is the run-to-run spread and nothing went infinite.

MEASURED (sm_86, 2026-11-02): see the event log's slack measurement. On the synthetic carry cells both candidates sit
within 2^8 at L 16K and 64K; on the layer constructions the facts bound's slack grows as the routing sharpens (to
2^25.7 at gain 16), the write bound's to 2^14.
