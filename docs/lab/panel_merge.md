# `panel_merge`

Mirrors `measure/lab/experiments/panel_merge.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

Were the output's sum f16 across one fold group's CTAs and fp32 across groups, a launch's groups would each leave a
panel of f16 pairs, a row a token, merged into the fp32 output when the launch ends. What does that merge cost a
token, by the groups a launch holds? (The format takes no such form, on this experiment's answer:
[`CONSTITUTION.md#r17`](../CONSTITUTION.md#r17).)

THE FORM. A CTA an SM, 4 warps; each CTA takes its own run of tokens and each warp every fourth of them, so a row
has one writer and nothing is atomic. A token: G loads of a 128-byte panel row (a lane a pair of f16), one of its
output row's 256 bytes, G multiply-adds a half by the group's power of two, two stores. A token's loads run two
tokens ahead of its sums, at fixed registers. G = 2, 8 and 16 over 262,144 tokens -- panels of 67, 268 and 537 MB,
none held by the cache. Every output row is checked exactly.

MEASURED (H100 SXM, 132 SMs, vast.ai instance 60300423, driver 610.57.04; 2026-11-12, the experiment at
db42c477; 7 repetitions, every output row exact; SM-cycles a token, the median, the spread under 0.8%):

| the groups merged | SM-cycles a token |
|---|---|
| 2 | 162 |
| 8 | 213 |
| 16 | 289 |

WHAT IT SHOWS.
- THE MERGE IS 213 SM-CYCLES A TOKEN AT 8 GROUPS: about 9 a group's row beside 143 for the token whatever the
  groups, its output row read and written among them.
- IT IS NOT AT THE MEMORY'S RATE: a token moves 1,536 bytes at 8 groups, and the token's fixed part is several
  times its two rows' bytes. A warp here holds two tokens in flight; whether more in flight brings the fixed part
  down is not in this experiment.

Record: `measure/lab/results/panel_merge/sm_90a.json`.
