# `prepass/prepass_api.cuh` — the pre-pass's host entries

Mirrors `csrc/rola/src/prepass/prepass_api.cuh`; the bodies are `prepass.cu` (`prepass.md`).

`prepass_forward` takes the bundle's u8 rows and per-level maxima on both sides, the write gain,
the value scale, the two output planes to reduce into, the widths, the mode word and the
liveness words, and launches the depth this line builds. `prepass_ledger_bind` binds or unbinds
the phase ledger the next launches add into, `[ctas][8][kPpPhases]` int64 on the device.
`prepass_plan` reports the compile-time plan and the kernel's registers and local bytes, which
is how a test holds the skeleton to zero local memory.
