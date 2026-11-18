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
its run); and, on the integer steps, the one-lane stamp's store alone. The held-predicate stamp also runs behind a branch every lane takes alike, never taken and taken one step in
eight: what a stamp switched off at run time costs. The caller's stamp runs too: a clock read and a store to a fixed slot whose address the kernel builds from the
step's index, in shared memory and in global memory, the stamp holding no state. A stamp's cost is a form's cycles a step less
the plain step's, over its stamps a step. Every kernel's work is checked against the host's, every ring form's stamps
are read back (their ids in order, their clocks not decreasing), and every kernel declares its CS2R -- its span's two
clock reads (on the wgmma step also the 16 that zero its accumulator) and one a stamp site -- so a loop the compiler
unrolled or a read it dropped is refused: ptxas drops a clock read whose value nothing stores, which is why a clock
read alone is not a row.

MEASURED (NVIDIA H100 80GB HBM3, 132 SMs, Nebius preemptible VM 5w3ffcb7, driver 580.173.02; 2026-11-18, the experiment at
03bbe65a; 7 repetitions; every check passed: every kernel's work equal to the host's, every ring
form's stamps read back in order; the median): THE CYCLES A STAMP ADDS TO ITS STEP, one stamp a step and two, each the
form's cycles a step less the plain step's over its stamps a step (the plain step's cycles in the column head; a dash is
a form the step does not run):

| the form | a wgmma step, plain 96.3: one stamp a step | two stamps a step | a chained step, plain 190.6: one stamp a step | two stamps a step | an independent step, plain 29.3: one stamp a step | two stamps a step |
|---|---|---|---|---|---|---|
| the lean stamp: every lane to one entry, no predicate, no wrap | 1.0 | 16.5 | 8.0 | 19.1 | 8.0 | 21.5 |
| one lane, its predicate held | 1.0 | 5.5 | 8.9 | 8.3 | 17.0 | 9.5 |
| one lane, its predicate in the offset | 14.0 | 17.0 | 19.8 | 21.5 | 15.0 | 20.1 |
| the lab's trace | -0.8 | 21.4 | 9.3 | 23.4 | 35.3 | 27.6 |
| the store alone, no clock read | -- | -- | 8.2 | 2.8 | 17.0 | 8.5 |
| the held-predicate stamp behind a branch never taken | 22.1 | 11.0 | 22.2 | 16.0 | 25.0 | 16.5 |
| the same, its branch taken one step in eight | 25.3 | 14.7 | 29.1 | 20.1 | 26.2 | 19.3 |
| the caller's stamp: a fixed slot in shared memory | 1.0 | 16.5 | 8.0 | 19.1 | 8.0 | 17.5 |
| the caller's stamp: a fixed slot in global memory | -4.8 | 10.1 | 6.9 | 17.2 | 9.2 | 24.1 |

WHAT IT DECIDES.
- ONE STAMP A STEP COSTS A WAITING STEP NOTHING MEASURABLE (1.0 to 1.0 cycles on 96.3, the lean stamp and the held
  predicate) AND AN INTEGER STEP 8.0 TO 8.9 CYCLES on the chained step in the two good forms (the lean stamp
  8.0, the held predicate 8.9); on the independent step the lean stamp is 8.0 and the held predicate 17.0.
- NO FORM IS CHEAPEST EVERYWHERE. The lean stamp is where stamps are a step apart (one a step: 1.0, 8.0 and 8.0
  against the held predicate's 1.0, 8.9 and 17.0); the held predicate is where two fall close together (two a
  step: 5.5, 8.3 and 9.5 against the lean stamp's 16.5, 19.1 and 21.5). The lean form's second stamp
  costs about twice its first -- its stamp at two a step is 2.4 times its stamp at one on the chained step and
  2.7 times on the independent step. NOT ISOLATED: read as two 32-lane stores a trip.
- THE FOLDED PREDICATE IS NEVER THE CHEAPEST (14.0, 19.8 and 15.0 at one a step): the register it saves costs more than
  it buys. THE LAB'S TRACE -- one lane, its store to GLOBAL memory -- is the held predicate's ring where a step waits or
  is bound by its chain (-0.8 and 9.3 at one a step) and about twice it where the step is bound by issue (35.3
  against 17.0), and it is the dearer of the two when two stamps fall close together (21.4, 23.4 and 27.6 at
  two a step).
- A STAMP SWITCHED OFF BY A BRANCH COSTS MORE THAN THE STAMP: behind a branch every lane takes alike and never takes, a
  stamp site costs 22.1, 22.2 and 25.0 cycles at one a step where the stamp itself costs 1.0, 8.9 and 17.0; taken one step
  in eight, 25.3, 29.1 and 26.2. AS COMPILED the skipping path is two instructions, a test and a branch, and the toolkit
  stalls both: a block ends with what its pending instructions are owed (KERNEL_STANDARDS section 27's fence, on this
  card; the stall counts are read with `lab sass stamp_cost`). A STAMP IS SWITCHED AT COMPILE TIME, never by a branch at run time.
- THE CALLER'S STAMP -- a clock read and a store to an address the kernel builds from an index it already holds, no
  state of the stamp's own -- is the lean stamp in shared memory (1.0, 8.0 and 8.0 at one a step against the lean
  stamp's 1.0, 8.0 and 8.0; 16.5, 19.1 and 17.5 at two): the toolkit compiles both to the same four
  instructions at one a step. At two a step on the independent step the two differ (17.5 against 21.5): NOT
  ISOLATED. THE SAME STAMP TO GLOBAL MEMORY is -4.8, 6.9 and 9.2 at one a step and 10.1, 17.2 and 24.1 at two:
  the destination alone, every other instruction the same.
- AS COMPILED, a stamp's cost on a step bound by issue is its added instructions at about two cycles each for a warp
  alone: on the independent step (29.3 cycles plain, eight operations) the lean stamp adds 8.0 cycles and the
  store alone 17.0.

Record: `measure/lab/results/stamp_cost/sm_90a.json`.
