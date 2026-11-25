# `carry_fetch`

Mirrors `measure/lab/experiments/carry_fetch.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What does a carry warp's own supply of a group's value lines cost -- the LIBRARY's listing of a window's live tokens
([`internals/carry/carry_list.md`](../internals/carry/carry_list.md)) and the copies of their 128-byte value lines by
token index straight into the steps' tiles ([`internals/carry/carry_values.md`](../internals/carry/carry_values.md))
-- by how many tokens are live?

THE FORM. This file compiles the library's device code. A CTA an SM of 16 warps, each warp a group's windows: a lane
loads its mask word a window ahead; the warp lists the window a round of 32 places at a time and issues the round's
copies into its two step tiles, two rounds in flight. The values are one table of 256 windows that every CTA reads.
Rows: 64 live tokens a window; every token live. The first window's first four rounds' tiles of every group are
checked byte for byte against the host's, a dead place's lines zero.

MEASURED (NVIDIA H100 80GB HBM3, 132 SMs, a Nebius instance, driver 580.196.07; 2026-11-25, the experiment at
ded43e54; 7 repetitions; all 34,603,008 checked tile bytes
exact in both rows; cycles a 128-byte line a CTA, the median):

| a group's window | cycles a line |
|---|---:|
| 64 live tokens: listed, and their value lines fetched | 13.30 |
| every token live: listed, and their value lines fetched | 13.86 |

PREDICTED, before the run: 3.0 to 8.0 and 2.6 to 5.0, from the copies alone at 2.70 a line
([`value_fetch`](value_fetch.md)) and the listing riding on them -- BOTH MISSED, by a factor of three to five. THE
LISTING IS NOT IT: it is one to two cycles a token alone ([`carry_list`](carry_list.md)). THE WRONG PRICE IS NOT YET
NAMED: this fetch differs from `value_fetch`'s in two ways -- its copies give their size in a register (the form that
zero-fills a dead place), and a row's 8 pieces land in bricks 17 lines apart -- and `value_fetch` is given a row of
each to part them. Until that row is read, no fold is built on this fetch.

WHAT IT IS NOT: a fetch past the cache. The values are one 32 MB table the whole grid reads, which the card's cache
holds; a table it cannot hold is NOT MEASURED here (the copies alone: 3.79 against 3.69, `value_fetch`).

Record: `measure/lab/results/carry_fetch/sm_90a.json`.
