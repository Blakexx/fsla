# `stamp_cost`

Mirrors `measure/lab/experiments/stamp_cost.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What does one timeline stamp -- the SM clock's low word beside an event's id, written by a warp -- cost the warp that
takes it, in cycles, in registers and in issued instructions, and which form of it costs least?

THE FORM. One warpgroup a CTA, a CTA an SM. A step is one of three: one `m64n64k16` wgmma with its commit and wait (a
step that mostly waits), four dependent integer operations each with a shared load (a step bound by its chain's
latency), or eight independent integer operations (a step bound by issue). Each runs plain, then with one or two
stamps a step in four forms: the lab's own trace (a stamp held a lane, a global line stored every 32 stamps); a ring
in shared memory written by one lane under a held predicate; the same with the predicate folded into the ring
offset's low bit; and every lane storing its own entry by a plain store the compiler places. A stamp's cost is a
form's cycles a step less the plain step's, over its stamps a step. Every kernel's work is checked against the host's,
and every ring form's stamps are read back: their ids in order, their clocks not decreasing.

AS COMPILED (sm_90a, toolkit 13.0): a ring stamp is about 8 instructions and holds no register these kernels lack;
the lab's trace is 56 instructions and 6 to 8 registers a thread.

NOT YET MEASURED: its record is taken at its first H100 session.
