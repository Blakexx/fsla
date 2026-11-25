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

NOT MEASURED: no run of it is recorded.
