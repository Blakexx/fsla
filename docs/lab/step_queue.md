# `step_queue`

Mirrors `measure/lab/experiments/step_queue.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The carry's fold issues its steps as a chain: a step's two operands loaded from a shared table, moved to the uniform
registers, the step. Beside the read side's warps a step costs some 58 cycles; the fold alone, 48; its operands held,
15 ([`carry_window`](carry_window.md)), and the profile puts the chain's waits on the operands' shared load. Changes
to the fold's own code have not moved the composed window. What does each kind of traffic other warps put on the SM
cost a step, and how does that grow with its rate? The answer is the model the carry's next forms are predicted by.

THE FORM. A CTA an SM. One fold warpgroup issues trips of 16 m64n16k16 steps into 16 accumulators (the carry fold's
128 registers a thread), a step's operands one 16-byte line of a shared table loaded at its own issue behind a
compiler edge (LOADED, as the carry's fold) or one line held in registers (HELD). Beside it 0, 1, 2 or 4 co-runner
warps issue one kind of traffic back to back until the fold is done:
- SHARED LOADS: 16 bytes a lane, 4 wavefronts an instruction.
- SHARED STORES: 16 bytes a lane.
- GLOBAL ADDS: an f16 pair a lane, a line an instruction, as a drain warp adds.
- GLOBAL LOADS: 16 bytes a lane, of lines the cache holds.

Recorded: the fold's cycles a step, and the co-runners' instructions a hundred cycles a CTA. The fold's sums are
checked exactly (16 a step in every value); the co-runners count themselves.

PREDICTED, before the run: alone LOADED some 45-50 cycles a step and HELD some 15-20 (the carry's fold's own figures).
Beside shared loads and stores a LOADED step grows with their rate -- toward the composed carry's 58 at the read
side's rate (some 1,300 shared wavefronts a window beside the fold's) -- and a HELD step barely: what the co-runners
take is the operands' load, not the step. Beside global adds a LOADED step grows as well (the carry's adds-out row
took some 310 cycles a window off a fold that never waits on them), and a HELD step not.

MEASURED (H100 SXM, 132 SMs, a Nebius preemptible VM, driver 580; 2026-11-16, the experiment at ff7a954d; 7
repetitions, every fold sum exact; cycles a step, the median, and the co-runners' instructions a hundred cycles a CTA
at 1, 2 and 4 warps, the LOADED fold's beside them):

| beside the fold | co-runners' rate | LOADED operands | HELD operands | THE HELD TABLE |
|---|---|---|---|---|
| nothing | | 34.0 | 23.2 | 24.8 |
| shared loads, 16 bytes a lane | 2.7 / 5.1 / 8.6 | 37.6 / 39.2 / 44.1 | 24.7 / 25.0 / 25.1 | - / - / 27.5 |
| shared stores, 16 bytes a lane | 3.1 / 6.0 / 9.0 | 40.2 / 44.5 / 46.8 | 24.5 / 24.8 / 25.0 | - / - / 27.3 |
| global adds, a line an instruction | 8.1 / 9.2 / 10.5 | 37.2 / 57.2 / 79.8 | 24.8 / 25.0 / 25.0 | - / 27.4 / 27.3 |
| global loads, 16 bytes a lane | 6.2 / 12.5 / 24.4 | 35.2 / 34.8 / 34.9 | 24.3 / 24.4 / 24.5 | - / - / - |

WHAT IT SHOWS.
- A STEP WHOSE ONE SET OF OPERANDS IS HELD IS IMMUNE: some 23-25 cycles a step whatever the co-runners put on the SM.
  Its binary holds no instruction between its steps -- the toolkit moved the one set to the uniform registers before the
  loop -- so this is a step with no load and no move of its own.
- A LOADED STEP IS SLOWED BY THE OTHER WARPS' MEMORY TRAFFIC, AND MOST BY GLOBAL ADDS: four warps adding back to back
  -- a drain warp's burst -- take a step from 34 to some 80 cycles; shared stores cost a step more than shared loads
  at a like rate; global loads of lines the cache holds cost it nothing. A loaded step's binary holds its load and
  some seven moves to the uniform registers a step. The prediction had shared loads and stores the larger cost and the
  alone figures some 45-50 and 15-20; the alone figures measured are 34 and 23, the difference from the carry fold's
  own NOT ISOLATED.
- A HELD TABLE IS NEARLY IMMUNE TOO: every step its own operands in registers, its binary a move to the uniform
  registers a step and no load, is some 25 cycles a step alone and some 27 beside four warps of shared loads, of shared
  stores and of global adds. Beside it those co-runners run slower than beside a loaded fold (5.7, 5.4 and 9.5
  instructions a hundred cycles against 8.6, 9.0 and 10.5); at matched rates a loaded step is 39 (shared loads at 5.1),
  45 (shared stores at 6.0) and 57 (global adds at 9.2) against the held table's 27. What the other warps' traffic costs
  a carry fold's step is then its operands' shared load: a step that holds its operands escapes it. (The carry's counted
  and uniform rows, near that form, lost: what else in them grew is not isolated.)

THE HELD-TABLE ROWS (every step its own operands' low words in registers, loaded once where the toolkit cannot know
them one warp's: its binary's steps hold their moves to the uniform registers and no load), alone and beside four
warps of shared loads, four of shared stores, and two and four of global adds. PREDICTED: if a step's move to the
uniform registers is not what the traffic slows, some 24-27 cycles a step alone and beside each -- the held one set's
immunity, and the form for the carry; if the move is, the loaded steps' growth. Measured: some 25 alone and 27 beside
each, as predicted for a move the traffic does not slow.

Record: `data:lab/step_queue/sm_90a.json`.
