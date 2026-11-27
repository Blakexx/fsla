# `carry_list`

Mirrors `measure/lab/experiments/carry_list.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What does the LIBRARY's listing ([`internals/carry/carry_list.md`](../internals/carry/carry_list.md)) cost -- a warp
turning a group's window of mask words into its live tokens in order, 32 places a round, counting inline -- by how
many tokens are live, and where the words are two levels' ANDed?

THE FORM. This file compiles the library's device code. A CTA an SM of 16 warps, each warp a group's windows: a lane
loads its word of each of two tables a window ahead, ANDs them, and the warp lists the window a round at a time into
its 32 consecutive words of shared memory. Rows: 64 live tokens a window from one table (the other all ones); every
token live; 64 live in both of two tables of a quarter live each. The first four windows' places and totals of every
group are checked against the host's.

MEASURED (NVIDIA H100 80GB HBM3, 132 SMs, a Nebius instance, driver 580.196.07; 2026-11-25, the experiment at
ded43e54; 7 repetitions; every place and total of the
checked windows exact in all three rows -- 539,874, 8,650,752 and 540,301 places; cycles a listed token a CTA, the
median):

| a group's window | cycles a token |
|---|---:|
| 64 live tokens, one level's words | 2.00 |
| every token live | 1.01 |
| 64 live tokens, two levels' words ANDed, each a quarter live | 2.00 |

PREDICTED, before the run, from the page's budget (about 60 instructions a lane a round, 20 a window, 16 warps sharing
an SM's issue): 0.4 to 2.5, 0.2 to 1.5 and 0.4 to 2.6 -- all three HELD.

WHAT IT SHOWS. A full round lists a token a cycle; a window of 64 tokens, whose second round is partly full, two.
A second level's words ANDed in cost nothing measurable: the intersected case is the one-level case. THE TIMED KERNEL
FOLDS EVERY TOKEN IT LISTS INTO A SUNK WORD: a listing whose only output is a shared-memory store no instruction
reads is work the compiler drops.

Record: `data:lab/carry_list/sm_90a.json`.

SINCE THE RECORD: the select (`listed_token`) is the lab's own code (`measure/lab/carry_select.cuh`), shared with `carry_fetch`. The library's supply lays a set's
tokens in a list ([`internals/carry/carry_list.md`](../internals/carry/carry_list.md#lay)), which
[`carry_supply`](carry_supply.md) measures; the rows above are of the select.
