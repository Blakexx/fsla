# `stamp_cost`

Mirrors `measure/lab/experiments/stamp_cost.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What does one timeline stamp -- the SM clock's low word beside an event's id, written by a warp -- cost the warp that
takes it, in cycles, in registers and in issued instructions, and which form of it costs least?

THE FORM. One warpgroup a CTA, a CTA an SM. A step is one of three: one `m64n64k16` wgmma with its commit and wait (a
step that mostly waits), four dependent integer operations each with a shared load (a step bound by its chain's
latency), or eight independent integer operations (a step bound by issue). Each runs plain, then with one or two
stamps a step: the lab's own trace (a stamp held a lane, a global line stored every 32 stamps); a ring in shared
memory written by one lane under a held predicate; the same with the predicate folded into the ring offset's low bit;
the leanest stamp, every lane storing the same two words to one entry with no predicate and no wrap (a ring sized for
its run); and, on the integer steps, the one-lane stamp's store alone. A stamp's cost is a form's cycles a step less
the plain step's, over its stamps a step. Every kernel's work is checked against the host's, every ring form's stamps
are read back (their ids in order, their clocks not decreasing), and every kernel declares its clock reads -- its
span's own and one a stamp site -- so a loop the compiler unrolled or a read it dropped is refused: ptxas drops a
clock read whose value nothing stores, which is why a clock read alone and a plain every-lane store are not rows.

NOT YET MEASURED: its record is taken at its first H100 session.
