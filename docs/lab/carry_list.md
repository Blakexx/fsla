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

NOT MEASURED: no run of it is recorded.
