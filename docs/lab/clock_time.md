# `clock_time`

Mirrors `measure/lab/experiments/clock_time.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

Is the SM's cycle counter a faithful clock -- does it advance at one rate against the GPU's nanosecond timer whatever
a warp is waiting on -- or does it run slow where an SM's warps wait on memory?

THE FORM. One warpgroup a CTA, a CTA an SM. A body runs many steps between two readings of both clocks (`%clock64` and
`%globaltimer`) by each warp's first lane: eight independent integer operations a step (issue); a chain of four
shared loads (latency inside the SM); a chain of four scattered loads of a 64 MB table (waits on global memory); a
store to global memory a step; and one `m64n64k16` wgmma a step with its commit and wait (the tensor pipe). A body's
rate is its cycles over its microseconds. The lab's unit is the cycle counter: where a body's rate differs from the
others', that body's cycle figures are not its time. The rows are taken at the full grid and at one CTA.

NOT YET MEASURED: its record is taken at its first H100 session.
