# `carry/carry_list.cuh` — the carry's listing

Mirrors `csrc/fsla/src/carry/carry_list.cuh`: a carry CTA's own list of a group's live tokens of a window
([`../../architecture/hopper-carry.md`](../../architecture/hopper-carry.md#what-a-cta-makes)), the first of the
skeleton's components ([`carry_kernel.md`](carry_kernel.md)). The fold's supply calls it ([`carry_kernel.md`](carry_kernel.md#supply)); the lab checks and times it alone
([`../../lab/carry_list.md`](../../lab/carry_list.md)).

WHY THIS FORM. A side's group has a window of mask words (the prepared group words of each of the side's masked
levels, ANDed); a warp takes a word a lane. The owner asked for the count inline and for one form whether a side has
one masked level or several (25 November 2026: "the counts are useless anyway? We need to find a way to do the
counting inline anwyay"), and shared memory is conflict-free by construction
([`../../KERNEL_STANDARDS.md` section 30](../../KERNEL_STANDARDS.md#s30)). So the list is not compacted by each lane
storing its tokens where the counts before it say (the lab's listing: the data picks the address, and two lanes meet
on a bank by chance). It is SELECTED BY PLACE: lane `p` of round `r` computes the token at place `32 r + p` and stores
it at its own word, so a round's 32 stores are 32 consecutive words.

- <a id="listing"></a>`listing_of<WORDS>(word)`: the window as the warp holds it -- a lane's word, the live tokens
  before it and through it (a popcount and a running sum across the lanes, 5 shuffles), and the window's total, the
  last lane's, broadcast. `WORDS` is the window's mask words; it is a warp's lanes or the header does not compile (a
  card whose window is another size needs a word of another width: NOT BUILT).
- <a id="select"></a>`listed_token<WORDS>(listing, round)`: the token at the lane's place of the round. The lane that
  holds the place is the count of lanes that end at or before it -- a binary search of the running sums, 5 shuffles
  whose source lane is the lane's own probe; then that lane's word and start (2 shuffles) and the place's bit of the
  word (`nth_set_bit`). Past the total it is -1. Every lane runs the same instructions whatever is live; the rounds a
  window takes are its total over 32, a count every lane holds alike.

THE BUDGET, stated before the code ([`../../KERNEL_STANDARDS.md` section 19](../../KERNEL_STANDARDS.md#s19)): a
window, a warp -- one global load a lane a masked level; about 20 instructions a lane to the running sums; a round of
32 places about 60 a lane (7 shuffles, the select's 5 steps of a popcount and three selects, one store). At 64 live
tokens a window that is 2 rounds, some 140 instructions a lane; at every token live 32 rounds. Shared memory: a round's
32 words. Registers: the listing's four words and the next window's.
