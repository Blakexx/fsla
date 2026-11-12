# `panel_merge`

Mirrors `measure/lab/experiments/panel_merge.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The output's sum may be f16 across one fold group's CTAs and is fp32 across groups
([`CONSTITUTION.md#r17`](../CONSTITUTION.md#r17)): a launch's groups each leave a panel of f16 pairs, a row a
token, and the panels are merged into the fp32 output when the launch ends. What does that merge cost a token, by
the groups a launch holds?

THE FORM. A CTA an SM, 4 warps; each CTA takes its own run of tokens and each warp every fourth of them, so a row
has one writer and nothing is atomic. A token: G loads of a 128-byte panel row (a lane a pair of f16), one of its
output row's 256 bytes, G multiply-adds a half by the group's power of two, two stores. A token's loads run two
tokens ahead of its sums, at fixed registers. G = 2, 8 and 16 over 262,144 tokens -- panels of 67, 268 and 537 MB,
none held by the cache. Every output row is checked exactly.

UNMEASURED: no card has run it.
