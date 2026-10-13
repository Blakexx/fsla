# `facts_color`

Mirrors `measure/lab/experiments/facts_color.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The facts pass's option B (ruled 2026-10-13): the coloring and the list writing leave the CTA that holds the window and
run as a COLORER, a warp a list unit in small CTAs, from what the STREAM HALF hands it -- the unit's bit tile and its
STASH (each live entry in digit-then-token order, its token and digit and its final coefficient). The question: on the
`facts_block` cell's 8,192 list units, does the colorer fit beside the stream half within the line (1.15x the real
floor, 516 / 665 µs at 4 / 16 live digits a token), and what does a union cost by phase?

THE COLORER, a warp a union (4 warps a CTA, no CTA barrier; 4 CTAs an SM on the write side, 5 on the read side, whose
2 passes' tables are smaller):
1. The tile in registers (lane `d` digit `d`'s 16 words); each token's digits by sixteen 32 x 32 bit transposes.
2. THE PASSES, cut from the union in token order: pass `q` holds the union's tokens of union rank `q P .. q P + P - 1`
   (`P` the carry's pass for the side: 128 write, 256 read), so each pass is colored alone and a list's pass sees its
   digit's tokens of that pass only. A slot is `q P + 8 x (its color's tokens before it in the pass) + its color`.
3. The claims of every pass at once, the facts pass's batched form on BIASED byte counts (a (pass, digit, color)
   field `32 - capacity + count`, bit 5 marking a full digit; the capacity `ceil(n / 8)` of the digit's tokens in the
   pass; the pass's cap `P / 8` a color): each batch shares its 32 lanes among the passes still queued, so a pass's
   tokens meet few of their own.
4. The singles, pass by pass, one a digit a round: each digit takes the first color in the rotation from `d` with room
   under its capacity and in the pass (else any with room in the pass), the digits choosing one color granted in digit
   order (a match on the color); then dealt to their colors over the pass's words.
5. The lists: a counting walk over the stash (a chunk ahead) into each (list, pass)'s color counts; each pass's start
   and its colors' tokens before; a walk a lane an entry (a chunk ahead) placing it at its pass's start + 8 x (its rank -
   its color's tokens before) + its color, each list's whole run first written with its pads as
   the walk reaches it (coalesced, so every sector is whole before the entries land over it).

Every list is checked exactly against the mirror given the kernel's coloring (entries, starts, heads), the coloring by
its invariants (every union token one slot, each pass's tokens its union ranks, each color dense in token order within
its pass, the extent).

THE PASS LAYOUT, chosen on the cell's tiles by simulation (list places over live entries, 300 units a cell):

| layout | k 4 | k 16 |
|---|---|---|
| the window colored whole (the facts pass's today; the serial reference / the kernel's batched claims) | 1.90 / 1.96 | 1.91 / 2.13 |
| passes cut from the union in token order, each colored alone (serial) | 1.71 | 1.37 |
| the same, the claims batched 8 a pass (this kernel) | 1.75 | 1.44 |
| passes of the window's tokens (not the union's) | 2.75 | 1.32 |
| 32 (pass, color) bins, each token the bin opening its digits the fewest new steps (serial) | 1.48 | 1.36 |
| the same in batches of 32 | 2.00 | 1.51 |
| the bound: each digit in one pass | 1.41 | 1.11 |

Measured (RTX 3080 Ti at its boost clock, ~1.96 GHz; medians):

| | k 4 | k 16 |
|---|---|---|
| the colorer, both sides' 8,192 unions | 299 µs | 571 µs |
| list places over live entries | 1.76 | 1.45 |
| the window-colored form (`a044e932`), both sides | 228-258 µs | 580 µs |
| the stream half's model alone (a persistent CTA an SM, two windows) | 393-420 µs | 447 µs |
| tokens over a pass capacity / gap slots, a union | 1.6 / 23.6 | 15.1 / 23.1 |

What a place costs: the carry steps every place of a list in every box that holds the list -- 16 CTAs a list at this
cell (8 boxes along the other level x 2 channel groups at d_v 64), ~3.7 cycles a place a CTA in the readout
(`read_image`) -- so the places are the colorer's real output, and the per-pass layout's fewer places (-10% at k 4,
-31% at k 16) outweigh its slower coloring.

Findings on the way (each a form the source keeps):
- A data-dependent term in a vote loop's bottom test, a counted loop holding a collective, or vote-bottomed loops nested
  in fully unrolled loops each left ptxas unable to prove convergence anywhere after: every collective got a fallback
  call. The whole test inside the vote, and vote-bottomed runtime loops, fix it.
- Entries and pads stored 8 bytes at a time to scattered places wrote 198 MB to DRAM for ~105 MB of lists at k 16
  (partial sectors evicted before they were whole) and stalled on the stash's loads; writing each list's run whole as
  the walk reaches it, and the stash prefetched to L2, halved the k 16 time.
- Per-pass capacities are small (often 1 a color at k 16), so 32 claims of one pass collide: 20 batches and 116 losers
  a union at k 16's write side, against ~11 and 35 when the batch's lanes are shared among the passes.
- The colorer and the stream half's model added rather than overlapped (k 16: 1,020 µs together against 447 + 580).

What it decides: the lists' passes are cut from the union in token order and colored alone (the carry's steps fall
with the places); the colorer's own speed, still above the window-colored form's, is the next item.

Record: none yet (over its stated instruction budget; the stated lines are kept in the source).
