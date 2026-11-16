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

MEASURED (H100 SXM, 132 SMs, a Nebius preemptible VM, driver 580; 2026-11-16, the experiment at fb9058be; 7
repetitions, every fold sum exact; cycles a step, the median, and the co-runners' instructions a hundred cycles a CTA
at 1, 2 and 4 warps):

| beside the fold | co-runners' rate | LOADED operands | HELD operands |
|---|---|---|---|
| nothing | | 34.0 | 23.2 |
| shared loads, 16 bytes a lane | 2.7 / 5.1 / 8.6 | 37.7 / 39.1 / 44.2 | 24.7 / 25.0 / 25.1 |
| shared stores, 16 bytes a lane | 3.1 / 6.5 / 8.9 | 40.3 / 44.9 / 46.5 | 24.5 / 24.8 / 25.0 |
| global adds, a line an instruction | 8.0 / 9.7 / 10.8 | 38.0 / 55.6 / 77.0 | 24.8 / 25.0 / 25.0 |
| global loads, 16 bytes a lane | 6.2 / 12.5 / 24.3 | 35.2 / 34.8 / 34.9 | 24.3 / 24.4 / 24.5 |

WHAT IT SHOWS.
- A STEP WHOSE OPERANDS ARE HELD IS IMMUNE: some 23-25 cycles a step whatever the co-runners put on the SM. What
  other warps' traffic costs a step is its operands' shared load alone.
- A LOADED STEP IS SLOWED BY THE OTHER WARPS' MEMORY TRAFFIC, AND MOST BY GLOBAL ADDS: four warps adding back to back
  -- a drain warp's burst -- take a step from 34 to 77 cycles; shared stores cost a step more than shared loads at a
  like rate; global loads of lines the cache holds cost it nothing. The prediction had shared loads and stores the
  larger cost and the alone figures some 45-50 and 15-20: the probe's alone step is its chain alone (34, 23), the
  carry fold's alone figures carry its other phases.
- FOR THE CARRY: its fold's steps should hold their operands, not load them at their issue -- made once a slot,
  outside the step chain, and held (the counted row made them beside each step and lost).

Record: `measure/lab/results/step_queue/sm_90a.json`.
