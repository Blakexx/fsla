# `carry_fetch`

Mirrors `measure/lab/experiments/carry_fetch.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What does a carry warp's own supply of a group's value lines cost -- the LIBRARY's listing of a window's live tokens
([`internals/carry/carry_list.md`](../internals/carry/carry_list.md)) and the copies of their 128-byte value lines by
token index straight into the steps' tiles ([`internals/carry/carry_values.md`](../internals/carry/carry_values.md))
-- by how many tokens are live?

THE FORM. This file compiles the library's device code. A CTA an SM of 16 warps, each warp a group's windows: a lane
loads its mask word a window ahead; the warp lists the window a round of 32 places at a time and issues the round's
copies into the round's 32 rows, two rounds in flight; a warp's tiles start on a 1,024-byte boundary. The values are one table of 256 windows that every CTA reads.
Rows: 64 live tokens a window; every token live. The first window's first four rounds' tiles of every group are
checked byte for byte against the host's, a dead place's lines zero.

MEASURED (NVIDIA H100 80GB HBM3, 132 SMs, a Nebius instance, driver 580.196.07; 2026-11-25, the experiment at
73620f1a; 7 repetitions; all 34,603,008 checked tile bytes exact in both rows and every tile on its boundary; cycles
a 128-byte line a CTA, the median):

| a group's window | cycles a line |
|---|---:|
| 64 live tokens: listed, and their value lines fetched | 2.96 |
| every token live: listed, and their value lines fetched | 2.10 |

PREDICTED, before the run: 3.5 to 9.0 and 2.7 to 5.5, the copies alone at 2.72 a line with rows whole and turned
([`value_fetch`](value_fetch.md)) and the listing at 2.00 and 1.01 a token ([`carry_list`](carry_list.md)) where the
two add -- BOTH MISSED, LOW. THE WRONG PRICE IS THE SUM: the listing costs the warp nothing beside its copies (2.96
against the copies' own 2.72 at 64 live tokens, the last round's dead places filled with zeros), so the warp's time
is its copies'. With every token live a line costs LESS than `value_fetch`'s row, 2.10 against 2.72: its rows are a
window's 1,024 consecutive lines of the table, where `value_fetch` reads rows scattered over it. THAT CONSECUTIVE
SOURCE ROWS ARE WHAT IS CHEAPER IS NOT SHOWN: no row here varies the tokens' spread at one count.

WHAT IT IS NOT: a fetch past the cache. The values are one 32 MB table the whole grid reads, which the card's cache
holds; a table it cannot hold is NOT MEASURED here (the copies alone: 3.79 against 3.69, `value_fetch`).

Record: `measure/lab/results/carry_fetch/sm_90a.json`.

SINCE THE RECORD: the select (`listed_token`) is the lab's own code (`measure/lab/carry_select.cuh`, shared with `carry_list`) and the copies it feeds are this experiment's. The library's supply lays a set's
tokens in a list ([`internals/carry/carry_list.md`](../internals/carry/carry_list.md#lay)), which
[`carry_supply`](carry_supply.md) measures; the rows above are of the select.
