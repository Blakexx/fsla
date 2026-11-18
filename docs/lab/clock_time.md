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

MEASURED (NVIDIA H100 80GB HBM3, 132 SMs, Nebius preemptible VM db7wwmvj, driver 580.173.02; the locked records 2026-11-18, the unlocked 2026-11-18, the experiment at
fbba665c; 7 repetitions; every check passed in all four records: every body's clocks advanced in
every warp and the launch's rate was a GPU clock's; the median): THE CYCLE COUNTER'S RATE, in cycles a microsecond, a body's
cycles over its nanoseconds, one CTA and the full grid, the clock as the card left it (unlocked) and locked:

| a body | unlocked, one CTA | unlocked, full grid | locked, one CTA | locked, full grid |
|---|---|---|---|---|
| eight independent integer operations a step | 1995.0 | 1980.0 | 1830.0 | 1830.0 |
| a chain of four shared loads a step | 1995.0 | 1980.0 | 1830.0 | 1830.0 |
| a chain of four scattered loads of a 64 MB table a step | 1995.0 | 1980.0 | 1830.0 | 1830.0 |
| a store to global memory a step | 1995.0 | 1980.0 | 1830.0 | 1829.8 |
| one wgmma a step, its commit and wait | 1814.9 | 1814.2 | 1829.4 | 1826.0 |

A STEP'S TIME, in nanoseconds by the nanosecond timer, the same four columns:

| a body | unlocked, one CTA | unlocked, full grid | locked, one CTA | locked, full grid |
|---|---|---|---|---|
| eight independent integer operations a step | 14.5 | 14.6 | 15.8 | 15.8 |
| a chain of four shared loads a step | 94.4 | 95.0 | 102.9 | 102.8 |
| a chain of four scattered loads of a 64 MB table a step | 2034.2 | 2411.4 | 2069.3 | 2436.3 |
| a store to global memory a step | 128.3 | 268.2 | 139.9 | 268.8 |
| one wgmma a step, its commit and wait | 52.9 | 52.5 | 52.5 | 52.6 |

WHAT IT DECIDES.
- THE COUNTER KEEPS ONE RATE THROUGH WAITS ON MEMORY (at one CTA the global-load chain's 1995.0 cycles a microsecond is the same as the integer body's 1995.0; at the full grid the global-load chain's 1980.0 cycles a microsecond is the same as the integer body's 1980.0; two rates are called the same within 0.5%).
- UNLOCKED, THE WGMMA BODY'S RATE IS OFF THE OTHERS': at one CTA the wgmma body's 1814.9 is 9.9% under the other four bodies' mean 1995.0; at the full grid the wgmma body's 1814.2 is 9.1% under the other four bodies' mean 1980.0.
- LOCKED, EVERY BODY IS ONE RATE: at one CTA the five bodies' rates run 1829.4 to 1830.0 cycles a microsecond, within 0.0% of one another (mean 1829.9); at the full grid the five bodies' rates run 1826.0 to 1830.0 cycles a microsecond, within 0.2% of one another (mean 1829.2). What the lock asked for and the clock the card then reported are not in the
  record (the box's own account of the lock is beside its output, not in it); the record gives only the measured rate,
  1829.2 cycles a microsecond at the full grid across the five bodies.
- A STEP IN REAL TIME, locked against unlocked: a wgmma step (one CTA: 52.5 ns locked is shorter than 52.9 ns unlocked by 0.8%; full grid: 52.6 ns locked is the same as 52.5 ns unlocked); an integer step (one CTA: 15.8 ns locked is longer than 14.5 ns unlocked by 9.0%; full grid: 15.8 ns locked is longer than 14.6 ns unlocked by 8.2%).

Record: `measure/lab/results/clock_time/sm_90a.json` and `sm_90a.blocks1.json` (the clock locked) and `sm_90a.unlocked.json`
and `sm_90a.unlocked.blocks1.json` (unlocked).
