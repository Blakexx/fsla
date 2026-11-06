# `read_alone`

Mirrors `measure/lab/experiments/read_alone.cu`, a part harness of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

THE QUESTION. What does the value-carve readout cost a warp-window alone -- production's `read_factors` and
`read_chunk` ([`readout.md`](../internals/carry/readout.md)), nothing else in the window -- against its tensor and
shared-memory floors and against the composed kernel's readout phase? The readout component's instrument for the
iteration Blake ruled 2026-11-06 (the readout to its floor without changing the ruled design).

THE METHOD. One CTA an SM with arm 0's plan tenants; each of 64 windows a 216-slot union whose 32 digits hold ~8
entries each (slot `s` in digit `d`'s list with probability 10 / 256; or COLORED, the coloring's best case, a list's
entry `k` on residue `k mod 8`; two denser draws, 18 / 256 -- alt-k16's density, 2.2 reads a token -- and 40 / 256
past it, the listed regime's other end), the image in halves as sm_86 runs it: the factors, a rendezvous, both halves'
readouts
with a rendezvous between. The draw is stored to shared memory before two barriers and read inside the span: after one
barrier the empty span read 19.4K cycles a window (the warps not lined up by it, measured), after two 38. The floors are
computed on the host from the same draw: TENSOR each scheduler's two warps' MMAs (4 a tile at the first form's den on
every warp, 16 cycles at fp16 accumulate on sm_86, 32 at fp32, 8 on sm_80), SHARED the CTA's wavefront-cycles at the
first form's traffic (a load one, a store two). `--blocks 1` runs it at one CTA for racecheck.

THE MEASUREMENT (sm_86, cycles a warp-window, median of 7, the f16 panel at fp16 accumulate; random / colored slots):

| form | random | colored |
|---|---|---|
| as first built (the den on every warp, each token's factor a channel group) | 23,860 | 23,530 |
| the den on its own channel group | 21,412 | 22,002 |
| the entries' factors once a chunk, the tiles a tile ahead | 14,478 | 13,890 |
| the two tokens' adds overlapped | 13,482 | 13,157 |
| a pass's next digit a lane's, the den in its digits' num tiles, the panel's word one XOR of the row key, a tile its run's remainder | 10,746 | 10,387 |
| two halves a tile (rows 0-7 one digit's up to 8, rows 8-15 the next's), the den block a channel group's | 8,200 | 7,157 |
| the empty span (the harness's zero) | 38 | 38 |

At the denser draws, the one-digit tile against two halves a tile: 18 / 256 11,602 -> 11,139; 40 / 256 17,893 ->
18,766 (a run of 16 or more is a tile of one digit's two halves, its MMAs and slices twice). A table of the pass's
halves in place of the successor shuffle: 9,614 / 8,510 / 20,478 at 10 / 256 random and colored and 40 / 256 (its
serial build). Floors at the first form's traffic: tensor 4,023, shared 8,091. Bank conflicts are not what binds: the colored draw
has none (104K conflicted wavefronts against 6.4M random, ncu) and costs the same. RACECHECK at one CTA over every
variant: 0 hazards.

WHAT IT DECIDES. The readout's own form was 3x its shared floor from instructions and its tile's dependent chain, not
from conflicts; each change above was taken into the kernel and gated (the per-slot oracle, the SASS gate), the
composed readout phase following it (24.9K -> 13.1K -> 11.0K at nl64k-alt-k4). Halves a tile win at the headline
densities and lose 5% at twice alt-k16's: a pass of whole tiles for runs of 16 or more is the dense end's lever.
