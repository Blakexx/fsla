# `stamp_cost`

Mirrors `measure/lab/experiments/stamp_cost.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What does one timeline stamp -- the SM clock's low word beside an event's id, written by a warp -- cost the warp that
takes it, in cycles, in registers and in issued instructions, and which form of it costs least?

THE FORM. One warpgroup a CTA, a CTA an SM. A step is one of three: one `m64n64k16` wgmma with its commit and wait (a
step that mostly waits), four dependent integer operations each with a shared load (a step bound by its chain's
latency), or eight independent integer operations (a step bound by issue). Each runs plain, then with one or two
stamps a step: the lab's own trace (its first lane's predicated store to the warp's row in global memory); a ring in shared
memory written by one lane under a held predicate; the same with the predicate folded into the ring offset's low bit;
the leanest stamp, every lane storing the same two words to one entry with no predicate and no wrap (a ring sized for
its run); and, on the integer steps, the one-lane stamp's store alone. A stamp's cost is a form's cycles a step less
the plain step's, over its stamps a step. Every kernel's work is checked against the host's, every ring form's stamps
are read back (their ids in order, their clocks not decreasing), and every kernel declares its CS2R -- its span's two
clock reads (on the wgmma step also the 16 that zero its accumulator) and one a stamp site -- so a loop the compiler
unrolled or a read it dropped is refused: ptxas drops a clock read whose value nothing stores, which is why a clock
read alone is not a row.

MEASURED (NVIDIA H100 80GB HBM3, 132 SMs, Nebius preemptible VM 7h1ggrm2, driver 580.173.02; 2026-11-18, the experiment at
fc918f8a; 7 repetitions; every check passed: every kernel's work equal to the host's, every ring
form's stamps read back in order; the median): THE CYCLES A STAMP ADDS TO ITS STEP, one stamp a step and two, each the
form's cycles a step less the plain step's over its stamps a step (the plain step's cycles in the column head; a dash is
a form the step does not run):

| the form | a wgmma step, plain 96.3: one stamp a step | two stamps a step | a chained step, plain 190.6: one stamp a step | two stamps a step | an independent step, plain 29.3: one stamp a step | two stamps a step |
|---|---|---|---|---|---|---|
| the lean stamp: every lane to one entry, no predicate, no wrap | 0.0 | 16.0 | 8.6 | 18.7 | 8.0 | 21.5 |
| one lane, its predicate held | 1.0 | 5.5 | 9.3 | 8.3 | 17.0 | 9.5 |
| one lane, its predicate in the offset | 13.0 | 17.0 | 20.1 | 21.6 | 15.0 | 20.1 |
| the lab's trace | -0.8 | 21.4 | 9.3 | 23.5 | 35.3 | 27.6 |
| the store alone, no clock read | -- | -- | 8.4 | 2.9 | 17.0 | 8.5 |

WHAT IT DECIDES.
- ONE STAMP A STEP COSTS A WAITING STEP NOTHING MEASURABLE (0.0 to 1.0 cycles on 96.3, the lean stamp and the held
  predicate) AND AN INTEGER STEP 8.6 TO 9.3 CYCLES on the chained step in the two good forms (the lean stamp
  8.6, the held predicate 9.3); on the independent step the lean stamp is 8.0 and the held predicate 17.0.
- NO FORM IS CHEAPEST EVERYWHERE. The lean stamp is where stamps are a step apart (one a step: 0.0, 8.6 and 8.0
  against the held predicate's 1.0, 9.3 and 17.0); the held predicate is where two fall close together (two a
  step: 5.5, 8.3 and 9.5 against the lean stamp's 16.0, 18.7 and 21.5). The lean form's second stamp
  costs about twice its first -- its stamp at two a step is 2.2 times its stamp at one on the chained step and
  2.7 times on the independent step. NOT ISOLATED: read as two 32-lane stores a trip.
- THE FOLDED PREDICATE IS NEVER THE CHEAPEST (13.0, 20.1 and 15.0 at one a step): the register it saves costs more than
  it buys. THE LAB'S TRACE -- one lane, its store to GLOBAL memory -- is the held predicate's ring where a step waits or
  is bound by its chain (-0.8 and 9.3 at one a step) and about twice it where the step is bound by issue (35.3
  against 17.0), and it is the dearer of the two when two stamps fall close together (21.4, 23.5 and 27.6 at
  two a step).
- AS COMPILED, a stamp's cost on a step bound by issue is its added instructions at about two cycles each for a warp
  alone: on the independent step (29.3 cycles plain, eight operations) the lean stamp adds 8.0 cycles and the
  store alone 17.0.

Record: `measure/lab/results/stamp_cost/sm_90a.json`.
