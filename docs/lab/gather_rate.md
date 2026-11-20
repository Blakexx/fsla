# `gather_rate`

Mirrors `measure/lab/experiments/gather_rate.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The gather fetches a token's rows scattered from a table, stores them into shared memory in the order they will be
read, and sends the image out to global memory by one bulk copy. What does that access pattern cost a row -- a
128-byte row -- when the table is one the card's cache holds (8 MB: [`load_rate`](load_rate.md),
[`state_move`](state_move.md)) and when it is not (2 GB), and when the image goes to a ring the cache holds or to
fresh memory?

THE FORM. A CTA an SM of 16 warps. A round is 64 DEPTH rows of 128 bytes (512 at DEPTH 8, 256 at DEPTH 4): a lane
loads 16 bytes, an instruction four rows, DEPTH instructions a warp in flight, each loaded a round before its use.
- LOAD: every word folded into a chain a lane; nothing stored.
- STORE: a round's rows stored into one of three images in shared memory, a warp's lanes on 32 banks.
- OUT: the image then stored to the CTA's span of global memory by one bulk copy, a round's copy waited two rounds on,
  so one barrier a round orders the images' reuse. The span is a RING of 256 KB a CTA (4 rounds at 512 rows, 8 at 256)
  the cache holds, or FRESH MEMORY, a span as long as the launch.
Both table sizes, 8 MB and 2 GB, take each form. Every form is checked exactly: the chain, the last round's image, or
every image the copies wrote, against the table's words.
The row's cost is the span of a CTA's 16 warps over the measured rounds' rows, in cycles a 128-byte row a CTA.

WHAT IT WILL DECIDE: the gather's floor a token is these rows' cycles times the token's lines, where the bundle fits
the cache and where it does not; which of the load, the store into shared memory and the copy out is the largest term
of the gather, to be split next; and whether a ring the cache holds is cheaper than fresh memory.

MEASURED: no record is taken at this commit (the experiment's results file for sm_90a, which this commit does not have does not exist). This
section is generated from one by `bazel run //tools:lab -- page gather_rate --instance LABEL`.
