# `readout_alone`

Mirrors `measure/lab/experiments/readout_alone.cu`, a part harness of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

THE QUESTION. What does the carry's readout phase cost a warp-window alone -- production's `read_chunk`
([`parts.md`](../internals/carry/parts.md#readout); fp16 accumulate, digits in pairs, sm_86's form), nothing else in
the window -- against its floors and the composed kernel's readout? Asked 2026-11-01 under the ruling that each phase is
at its own floor before composition.

THE METHOD. One CTA an SM with arm 0's plan tenants and the fold's 128 accumulators held live (empty `asm` uses: the
readout's register pressure, no instructions); the image's half (16 planes) as nl64k-alt-k4 reads it; each of 64
windows a box's 32 read digits of Binomial(16, 1/2) entries over a 216-slot union (the read side's ~256 entries a
window); a warp writes its own digits' entries, then each half (batches 0-1, 2-3) behind a rendezvous as the carry
runs it, each half's `read_chunk` timed. The tensor floor is the host's from the same draw: each scheduler's two warps'
digit-pair tiles (the pair's longer run in 8-token tiles) x 8 HMMA.16816.F16 x 16 cycles.

THE MEASUREMENT (sm_86, cycles a warp-window, median of 7, spread under 0.1%):

| variant | cycles |
|---|---|
| as the carry runs it: a rendezvous a half | 2,818 |
| back to back | 1,991 |
| the tensor floor (mean scheduler / busiest) | 839 / 957 |
| the composed kernel's readout phase (`phase_ledger.py`, nl64k-alt-k4) | 3,214 |

THE SHARED PIPE (ncu, a CTA-window, 210 MMAs): 1,966 wavefronts -- loads 1,161 (bank conflicts 255), the panel's
atomic adds 385 (conflicts 175; an atomic takes at least two cycles: [`smem_rates`](smem_rates.md)), shuffles and the
rest ~381, the harness's own stores ~40.

WHAT IT SETTLES.
- The readout is bound by the SM's shared-memory pipe, not the tensor pipe: ~1.95K cycles of shared work a window, and
  the readout alone back to back runs ~1.0x that. The tensor floor (839) is not its floor.
- In the carry's shape (a rendezvous a half) it is 2,818: the warps in lockstep queue their shared work together
  (~0.83K, as the fold's lanes do: [`fold_alone`](fold_alone.md)); composition adds ~0.4K more.
- Its own waste is in the shared pipe: the bank conflicts (~430 wavefronts a window, 22% of the work -- the slots'
  residues, the coloring's question: [`facts.md`](../internals/facts/facts.md#ranks)) and the shuffles not needed.

Record: `measure/lab/results/readout_alone/sm_86.json`, `sm_86.counters.json`.
