# `carry/carry_list.cuh` — the carry's listing

Mirrors `csrc/fsla/src/carry/carry_list.cuh`: a carry CTA's own list of a group's live tokens of a window
([`../../architecture/hopper-carry.md`](../../architecture/hopper-carry.md#what-a-cta-makes)), the first of the
skeleton's components ([`carry_kernel.md`](carry_kernel.md)). The fold's supply calls it ([`carry_kernel.md`](carry_kernel.md#supply)); the lab checks and times it alone
([`../../lab/carry_list.md`](../../lab/carry_list.md)).

WHY THIS FORM. A side's group has a window of mask words (the prepared group words of each of the side's masked
levels, ANDed); a warp takes a word a lane. The owner asked for the count inline and for one form whether a side has
one masked level or several (25 November 2026: "the counts are useless anyway? We need to find a way to do the
counting inline anwyay"), and shared memory is conflict-free by construction
([`../../KERNEL_STANDARDS.md` section 30](../../KERNEL_STANDARDS.md#s30)). THE LIST IS LAID, NOT SELECTED. It was
first selected by place -- lane `p` of round `r` computing the token at place `32 r + p`, a binary search of the
running sums and the place's bit of a word, about 65 instructions a lane a round, with no store and so no conflict.
That select is now the code of the lab's `carry_list` and `carry_fetch` alone, the form their records measured. The
library LAYS the list: each lane stores its word's tokens at their ranks' places, once a set, and a round reads its 32
places. THE LAY'S STORES ARE NOT CONFLICT-FREE BY CONSTRUCTION, which section 30 asks of every shared buffer: their
places are ranks, the data's. The access table below states them. The form is held against that law on a measurement
that is NOT RECORDED (an unrecorded session's rounds, the select the larger half of a round's instructions): A RECORDED
RUN OF [`carry_supply`](../../lab/carry_supply.md) IN THIS FORM IS OWED, and the law's text names no such exception
until it is there.

- <a id="listing"></a>`listing_of<WORDS>(word)`: the window as the warp holds it -- a lane's word, the live tokens
  before it and through it (a popcount and a running sum across the lanes, 5 shuffles), and the window's total, the
  last lane's, broadcast. `WORDS` is the window's mask words; it is a warp's lanes or the header does not compile (a
  card whose window is another size needs a word of another width: NOT BUILT).
- <a id="share"></a>`shared_part(split_bits, turn)`: THE PART OF A WORD a warp takes of a set it shares. A fold's
  warps share a set where the fold has fewer than four ([`carry_kernel.md`](carry_kernel.md#supply)); they split it
  BY BIT -- `1 << split_bits` warps, the warp their `turn`th, each `32 >> split_bits` of a word's 32 tokens -- and each
  lists its own part and runs its own rounds. WHY BY BIT and not by round or by word: every lane of every sharer then
  holds tokens of a full word, so a sharer of four lays a full word in 8 trips, where a split by rounds was 32 trips
  with three lanes in four storing nothing. The parts tile the word whatever is live.
- <a id="lay"></a>`lay_tokens<WORDS>(listing, list)`: THE SET'S TOKENS IN THE WARP'S LIST, the token of rank `r` at
  place `r`. A place is a `uint16`, a token's index in the window (`kNoToken`, 0xFFFF, where there is none); a list is
  a window's tokens and a spare place a lane (`kListPlaces`). A TRIP stores one token a lane, its lowest left, at its
  rank; the trips are the fullest word's tokens, one maximum over the lanes a set (5 shuffles), so no trip votes; a
  lane with no token left stores at its spare place, so no trip branches. After the trips the places from the total
  to the end of its round are set to no token: a round's fetch reads 32 places whatever the total. A `__syncwarp`
  stands between the stores and the fetch's loads.
  THE ACCESS TABLE ([`../../KERNEL_STANDARDS.md` section 30](../../KERNEL_STANDARDS.md#s30)): a place is 2 bytes, two
  a word, and 32 banks span 64 places. (i) A TRIP'S STORE: 32 lanes, a place each -- a rank, or the lane's spare
  place. Two lanes meet on a bank where their places are 64 apart, or where a rank's word and a spare's share one: NOT
  conflict-free. At worst every word is full: a trip's ranks are `32 j + k`, 16 lanes a bank (4 lanes a bank for a
  sharer of four, whose ranks are `8 j + k`). A set of at most two rounds lies in 64 places and its ranks never meet.
  NOT MEASURED: the store's wavefronts. (ii) THE TAIL'S STORE: the last round's dead places and the other lanes'
  spares, two runs of consecutive places that never share a bank. (iii) THE FETCH'S LOADS
  ([`carry_values.md`](carry_values.md#fetch)): an instruction's 32 lanes read 4 consecutive places, two words:
  conflict-free.

THE BUDGET, stated before the code ([`../../KERNEL_STANDARDS.md` section 19](../../KERNEL_STANDARDS.md#s19)): a
set, a warp -- one global load a lane a masked level; about 20 instructions a lane to the running sums; the lay about
11 to the fullest word's count, about 15 a trip, and 8 for the tail. Shared memory: a list a warp, 1,056 places of 2
bytes on the hopper card. Registers: the listing's four words and the next window's.
