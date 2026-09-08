# The SM's effective clock, read off the device

Mirrors `csrc/rola/src/common/sm_clock.cu` (`sm_clock_ghz`) and its use in
`tools/probe_cells.py`.

## Why it exists

The driver reports one SM clock on this host whatever the silicon runs at: through WSL,
`nvidia-smi` says 1950 MHz while the carry kernel measures at 1.965 GHz in one stretch and
1.665 GHz in another (`clock64` against `%globaltimer` inside the kernel, journal section
37), and clocks cannot be locked from WSL. On 2026-09-08 the state was found to be the
BOOST GOVERNOR'S: it moves the clock with power. A light all-SM spin lifted the clock and
the same kernel, the same commit, measured 6.45 ms and 5.92 ms either side of it; a heavy
dense matmul before the reps did not lift it (the card meets its power cap). Every harness
number before that day was taken in whichever state the preceding milliseconds left the
governor in, which is what the "two clock states" were.

## The standard, and the setup here

Reproducible kernel numbers come from a LOCKED clock: `nvidia-smi -lgc <MHz>` at a clock
the card holds under the kernel's power, which is what `ncu` does by default
(`--clock-control base`) and why its cycle counts were stable across the days the harness
was not. On this host the lock is set from the Windows side, in an administrator shell:
`nvidia-smi -q -d SUPPORTED_CLOCKS` to pick the clock, `nvidia-smi -lgc 1665,1665` to lock
it, `nvidia-smi -rgc` to release. Timing itself is the field's: CUDA events on the stream,
a warm-up, a median over reps and rounds, binaries interleaved in one run.

## What the probe does

`sm_clock_ghz(spin_cycles)` launches one CTA per SM, each spinning a dependent FMA chain
for `spin_cycles` of its own cycles; CTA 0 reports its cycles against the global timer.
The ratio is the effective clock during the spin, in GHz.

`tools/probe_cells.py` reads the clock after the warm-up, right before the timed reps,
where the binary carries the probe. Each round's row records `sm_ghz`; the printed row
shows the rounds' range; the ledger row keeps `round_sm_ghz`. A binary from before the
probe reports none and prints `GHz n/a`. Under a lock the read is the proof the lock held
through the reps; without one it is the label that makes a millisecond readable.

## How to read a number

A harness millisecond is comparable across days only with its clock: convert to cycles
(`ms * GHz`) or compare interleaved pairs of one run. Locked-clock `ncu` durations are the
other cross-day instrument, with one caveat: ncu locks the memory clock too, and this
latency-bound kernel spends more SM cycles under it than at full speed (13.24M under ncu
against ~11.1M real for the deferral commit), so ncu cycles compare only with ncu cycles.
