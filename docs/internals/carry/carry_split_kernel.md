# `carry_split_kernel.cuh` — the channel-split carry body

The mirror of `csrc/rola/src/carry/carry_split_kernel.cuh`. One of the carry family's
TWO bodies; [`carry_kernel.md`](carry_kernel.md) holds the other and states how the
per-level MODE DECLARATION chooses between them.

**Symbols** are `carry_kernel.md`'s.

## What it is for

A declaration with NO sparse level on the carve axis leaves the leaf carve nothing to
exploit: at DENSE routing every read entry is live at every digit of every level, so a
warp owning one digit group would drain every entry anyway and the only thing the carve
could still do is multiply the fan-in by `kWarps`. This body is the answer — every warp
owns `kMmaM` of the `d_v` CHANNELS across ALL `BC` leaves, so the warps' readout rows
are DISJOINT and the kernel's only fan-in is the RED at the top.

It is not a fallback. The declaration resolves at the seam and the axis never reaches
run time, so a dense declaration runs THIS kernel and a sparse one runs the other; a
declaration the family has no carved body for is served this one, which is always
correct and only ever slower.

## What differs from the carve body

| | channel-split (here) | leaf carve |
|---|---|---|
| a warp owns | `kMmaM` channels x all `BC` leaves | all `kCols` channels x `BC/kWarps` leaves |
| the state's n-tiles | `NDV = BC/kMmaN`, the box's own | `NL = BC/(kWarps*kMmaN)`, the warp's own |
| the mass channel | split along the LEAVES (`kMassUnits`), one channel being unsplittable along channels | a channel like any other — the warp holds its own leaves' partial |
| the panel | ONE CTA-shared panel per ring slot, produced cooperatively (`kRowThr = kThreads/C` threads per row) | one PRIVATE region per warp per slot, one lane per row |
| the ballot rows | the OWNER's, one per (level, owner group) | the CARVE grain, `kWarps` group rows at the carve level |
| the readout's fan-in | the warps hold different CHANNELS of one token, so they MEET: the `so` transpose, the `sden` cross-warp mass sum and two rendezvous per read batch | warp-private and DIRECT: no meeting, no `sden` |
| CTA barriers per window | ~13 | 1 outside the fold |

## Why it is a separate file rather than a branch inside the other

The two bodies differ in the state's register map, the panel's ownership, the schedule's
list count, the mass channel's split and the fan-in's shape — five things, not one — and
a single template carrying both would express that as five `if constexpr` blocks whose
joint reading is neither body. Two files each state one kernel, and the seam states
which. The cost is the text they share (the page sweep, the run loader, the window
grid); the benefit is that either body can be read end to end.

## The identity this file carries

It is `perf/pipeline` 7825d31's carry kernel, and the gate is that it still IS: the SASS
of every arm is byte-identical to that commit's, which is what makes "a dense
declaration costs nothing" a measurement rather than a claim. See
`rola-scratch/prototypes/k31_formb_build/FINDINGS.md`.
