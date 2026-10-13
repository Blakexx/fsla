# `facts_color`

Mirrors `measure/lab/experiments/facts_color.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

The facts pass's option B (ruled 2026-10-13): the coloring and the list writing leave the CTA that holds the window and
run as a COLORER, a warp a list unit in small CTAs, from what the STREAM HALF hands it -- the unit's bit tile and its
STASH (each live entry in digit-then-token order, its token and digit and its final coefficient). The question: on the
`facts_block` cell's 8,192 list units, does the colorer fit beside the stream half within the line (1.15x the real
floor, 516 / 665 µs at 4 / 16 live digits a token), and what does a union cost by phase?

THE COLORER, a warp a union (4 warps a CTA, no CTA barrier):
1. The tile in registers (lane `d` digit `d`'s 16 words); each token's digits by sixteen 32 x 32 bit transposes.
2. The claims (the facts pass's batched form, 32 tokens a batch, three-or-more first) on BIASED counts: a digit's color
   field holds `64 - capacity + count`, so its bit 6 marks a full digit and the sums order as the counts do.
3. The singles and the slots, the facts pass's (residue layout); a single's color from its digit's color ends by a packed
   subtraction.
4. The lists: a counting walk over the stash into each (list, pass)'s color counts; each (list, pass, color)'s place
   base; then a walk a lane an entry placing it at `base + 8 x rank`, each list's whole run first written with its
   pads as the walk reaches it (coalesced, so every sector is whole before the entries land over it).

Every list is checked exactly against the mirror given the kernel's coloring (entries, starts, heads), the coloring by
its invariants (every union token one slot, the residue layout, the union's cap, the extent).

Measured (RTX 3080 Ti at its boost clock, ~1.96 GHz; medians):

| | k 4 | k 16 |
|---|---|---|
| the colorer, both sides' 8,192 unions | 228-258 µs | 580 µs |
| instructions a union (read / write side) | 7.9K / 8.3K | 15K / 16.7K |
| the stream half's model alone (a persistent CTA an SM, two windows) | 393-420 µs | 447 µs |
| the model and the colorer on two streams | 597-608 µs | 1,020 µs |
| tokens over a digit's capacity / gap slots, a union | 0.002 / 12.4 | 0.58 / 16.1 |

Where a union's instructions go at k 16 (write side, `tools/lab.py ledger`): the lists' walk with its pads 6.6K, the
claims 4.2K, the singles 2.3K, the slots 1.25K, the counting walk 0.9K, the inputs 1.0K. At k 4: the walk 2.3K, the
singles 1.8K, the slots 1.25K, the claims 0.9K. The colorer is now issue-bound (45-53% of issue slots, ~14 warps an SM).

Findings on the way (each a form the source keeps):
- A data-dependent term in a vote loop's bottom test (`__any_sync(..) & (level <= cap)`) left ptxas unable to prove
  convergence anywhere after it: every collective in the kernel got a fallback call. The whole test inside the vote
  fixes it.
- Vote-bottomed loops nested in fully unrolled loops did the same; a plain lane loop with no collective inside does not.
- The first list form, entries and pads stored 8 bytes at a time to scattered places, wrote 198 MB to DRAM for ~105 MB
  of lists at k 16 (partial sectors evicted before they were whole) and stalled on its stash loads (long scoreboard 10
  of 19 cycles an instruction); writing each list whole halved the k 16 time.

What it decides: the colorer is correct at a warp a union, but it does not hide beside the stream half -- the two add
(the model leaves the colorer about 8 warps an SM). Projected at the locked clock with the real stream half (371 µs):
~640 / ~1,190 µs, 1.24x / 1.79x the line (the fused kernel: 907 / 1,720 µs). Which form the facts pass takes next is
Blake's decision, in the coordinator's user queue.

Record: none yet (over its stated instruction budget; the stated lines are kept in the source).
