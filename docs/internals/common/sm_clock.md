# The SM's effective clock, read off the device

Mirrors `csrc/rola/src/common/sm_clock.cu` (`sm_clock_ghz`) and its use in
`tools/probe_cells.py`.

## Why it exists

The driver reports one SM clock on this host whatever the silicon runs at: through WSL,
`nvidia-smi` says 1950 MHz while the carry kernel measures at 1.965 GHz in one stretch and
1.665 GHz in another (`clock64` against `%globaltimer` inside the kernel, journal section
37), and clocks cannot be locked from this side. On 2026-09-08 the state was found to be
LOAD-DEPENDENT: under a sustained load the SM sits at its top clock; under a bursty
warm-up (five calls, then timed reps) it sits ~15% lower. The same kernel, the same
commit, measured 6.45 ms and 5.92 ms either side of a ten-millisecond all-SM spin. Every
harness number before that day was taken in whichever state the preceding seconds left
the GPU in, which is what the "two clock states" were.

## What the probe does

`sm_clock_ghz(spin_cycles)` launches one CTA per SM, each spinning a dependent FMA chain
for `spin_cycles` of its own cycles; CTA 0 reports its cycles against the global timer.
The ratio is the effective clock during the spin, in GHz.

`tools/probe_cells.py` runs, for every binary alike and after the warm-up: a sustained
dense load for 30 ms (`_sustain_load`, a matmul loop, independent of the binary), then
the clock read where the binary carries it, then the timed reps. Each round's row records
`sm_ghz`; the printed row shows the rounds' range; the ledger row keeps `round_sm_ghz`. A
binary from before the probe reports none and prints `GHz n/a`.

## How to read a number

A harness millisecond is comparable across days only with its clock: convert to cycles
(`ms * GHz`) or compare interleaved pairs of one run. Locked-clock `ncu` durations are the
other cross-day instrument, with one caveat: ncu locks the memory clock too, and this
latency-bound kernel spends more SM cycles under it than at full speed (13.24M under ncu
against ~11.1M real for the deferral commit), so ncu cycles compare only with ncu cycles.
