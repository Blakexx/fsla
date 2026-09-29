# `common/phase_clock.cuh` — the phase ledger and trace

Mirrors `csrc/rola/src/common/phase_clock.cuh`.

`PhaseClock<Phases>` is one warp's measurement of itself, shared by every kernel family that
measures its phases: lane 0 accumulates the cycles between laps into the warp's shared row, one
slot a phase, and `publish` adds the row into the launch's ledger, `[cta][warp][Phases]` int64,
at the kernel's end. The row is `kRow = Phases + 1` slots: the last holds the previous lap's
clock, so the instrument keeps no register live across a phase -- a held 64-bit mark spilled
the decode step's deep arms, whose walk fills the register file. When a trace is bound (`bind_trace`), each lap and each `stamp` records
`cycle << 8 | event` into the warp's trace row up to its capacity.

The carry aliases it at its phase count (`carry_kernel.md#phase-ledger`). The struct is
inlined wherever it is used.
