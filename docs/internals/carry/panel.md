# `carry/panel.cuh` — the panel and the drain

Mirrors `csrc/rola/src/carry/panel.cuh`. A read chunk's PANEL is its slots' `num` words, the readout's adds landing
there (`readout.md#chunk`), and its DEN WORDS beside it; the DRAIN takes both to HBM once every warp has read the chunk.
The read buffer holds them after the chunk's rows (`read_panel`, `read_dens`; `schedule.md#buffers`), and the chunk's
fill clears them (`parts.md#fill`).

<a id="words"></a>
## The words

A slot's row holds the box's 32 channels at the reduce dtype (`panel_row_bytes`): f16 pairs, 16 words (64 bytes), word
`k` channels `2 k` and `2 k + 1`; or fp32, 32 words (128 bytes, a word a channel). A readout add touches a tile's 8
tokens (rows `g`) at a channel group's words: f16 its 4 pair words, one 4-byte load and store a lane; fp32 its 8
channels, one 8-byte load and store a lane. The row's words are ROTATED by the slot as an XOR of its 16-byte units
(`panel_word`): f16 `(64 s + 16 ((s >> 1) & 3)) ^ 4 w` -- two slots share a 128-byte line, and the rotation takes a
pair's two slots to its halves and four pairs to distinct words of them -- fp32 `(128 s + 16 (s & 7)) ^ 4 w`, a slot a
line. So 8 slots of distinct residues (a dense chunk's consecutive slots) meet 32 banks an add on either dtype; a list's
rank slots meet them where their ranks run consecutively, as the pool rows do (`parts.md#swizzle`). The f16 row's base
`64 s + 16 ((s >> 1) & 3)` is the slot's 64-byte pool row's key in 16-byte units times 16 (`readout.md#factors`), so
a readout add's word is one XOR of the key it already holds; the XOR leaves bits 0-3 alone, so an fp32 lane's two
channels stay adjacent.

THE DEN WORDS are `kDenWords` fp32 a slot, one a readout channel group (the group that reads digit `d` out adds its den
at word `d mod 4`), 16 bytes: no two warps add into one word, so the den needs no atomics either; the drain sums them.

<a id="drain"></a>
## The drain

`drain<P, F32>`, after every warp's adds (the caller's rendezvous): warp `w` takes slots `w + 8 i`. F16 PAIRS: two
slots a step, a lane a word (its half of the warp a slot), each added by `red.global.add.noftz.f16x2` at the slot's row
of `num` (`token_row`: the slot's query head and window token) -- the word's channels in their natural order, so
`num`'s f16 pairs read as its f16 values (`rola.ops.prefill.unpack_num`). FP32: a slot a step, a lane a channel,
`red.global.add.f32`. Then THE DEN: a thread a slot, its four words summed, one `red.global.add.f32` into `den`. Each
address is a 32-bit word offset from the head's base (`word_at`, one widening multiply-add; the host holds a write
head's rows under 2^32 words). A GAP slot (the union's tail past its tokens, its token's bit 15) adds nothing. The
CTAs join only here (R22: independent units, joined by atomics): the cross-CTA adds (`red_add_h2`, `red_add_f32`) are
unpredicated, each one `red.global` instruction. The f16 pairs meet across CTAs at one scale, the
token's byte, which every CTA reads the same.

ITS FORM AGAINST ITS FLOOR: the gap test is a branch around each add (ptxas wraps a predicated `red.global` in a
convergence region: the SASS gate counts them, cold, as guarded), a lane re-derives its slot's row, and the den's adds
are a slot a thread. The drain's floor is the SM's global request path, ~5 cycles a line request with every SM draining
(lab [`drain`](../../lab/drain.md)); its own budget, full-line requests and the clears in the fill, is open.
