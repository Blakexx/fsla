# The prefill's mask-preparation pass (`csrc/fsla/src/mask_prepare/mask_prepare.cu`, `mask_prepare.cuh`, `mask_prepare_kernel.cuh`)

The pass the op runs before its carry ([`../../architecture/mask-preparation.md`](../../architecture/mask-preparation.md#the-mask-preparation-pass)): one launch over
both sides of the call, writing each side's LEAF GROUPS' mask words and counts into the op's workspace at the layout of
[`../common/plan/mask_prepare_layout.md`](../common/plan/mask_prepare_layout.md). That is all it makes (the owner, 2026-11-25: "we can concer mask prep pass into just
materializing these group masks"). The device code is `mask_prepare_kernel.cuh` ([`mask_prepare_kernel.md`](mask_prepare_kernel.md)), the host half (the op's checks, the
grid, the place in the workspace, the launch) `mask_prepare.cu`. Hand CUDA: global loads, each listed level's group words in shared memory behind CTA barriers, global
stores; no tensor core. The Python face is `fsla/ops/mask_prepare.py` (`prepare_masks`); its exactness is graded by `tests/oracle/test_mask_prepare.py` against the host
reference `tests/oracle/mask_prepare_reference.py` (every line and count, bit for bit), and its host checks run without a GPU in `tests/unit/test_mask_prepare.py`; the
registry of its contract is `hardware/kernels/mask_prepare/` ([`../hardware/registry.md`](../hardware/registry.md)).

STATUS: COMPILED for the hopper card, the checker's rules held (64 registers a thread, no spill, 132,416 bytes of shared memory). EXACT on an H100
(2026-11-26: `tests/oracle/test_mask_prepare.py`'s 108 cases against the host reference, the library built from this source); its time's
record ([`../../lab/mask_prepare_time.md`](../../lab/mask_prepare_time.md)) is owed.

Symbols: `T` tokens a head, a window is the card's (`plan::window_of_cc`, [`../common/plan/levels.md`](../common/plan/levels.md); tokens), `words` mask words of 32
tokens (the hopper card's: 1,024 tokens, 32 words); a SIDE is 0, the read side, or 1, the write side; a level-side is a side and a level, `slot = side * 4 + level`;
`width` a level's digits (a power of two, 2 to 256), `width_bits` its log2; `g` a side's group of a level in digits (a power of two from 1 to 256), `GB` its log2; a
LEAF GROUP is one group on each of a side's levels ([`../common/plan/leaf_groups.md`](../common/plan/leaf_groups.md)).

## <a id="the-kernel"></a>The kernel

`mask_prepare_kernel<Bt>` has ONE template argument, the build tag: the pass has one arm a card, and THE WINDOW IS THE CARD'S, a constant of its build
(`kWindowWords<Bt>`); the host takes the same from `card_of(device).cc`. A side's groups are no host choice, because the host never reads the density
([`../common/plan/prefill_launch.md`](../common/plan/prefill_launch.md)): the kernel derives them on the device.

- THE GRID IS A CTA A (SIDE, BATCH-HEAD, WINDOW), 1,024 threads, the read side's CTAs first (`Args::first_cta`, the last entry the grid). That is what the host can know:
  a side's batch-heads and the call's windows. A CTA finds its side, its batch-head and its window from its index.
- THE LEAF GROUPS ARE THE CARVE'S, DERIVED THERE. A CTA's first warp computes the call's carve (`hand_leaf_groups`: `prefill_carve` of every level's width and the two
  sides' density bytes, a byte `b` being `b + 1` 256ths and a level-side with no mask (`Args::masked`, a bit a level a side) taken at the whole whatever its byte
  ([the layout](../common/plan/mask_prepare_layout.md#what-a-leaf-group-holds)), then `leaf_groups` of the CTA's side; its own function, not inlined, so that the carve's values are allocated
  apart from the walk's, which holds the kernel's 64 registers) and hands the rest, in shared words (`Handed`) behind a CTA barrier: the side's leaf groups' count as
  log2, and a value a level of its group and of each field of a leaf group's index.
- EACH LISTED LEVEL'S GROUP WORDS OF THE WINDOW ARE FORMED IN SHARED MEMORY, all the listed levels at once: a level has a quarter of the CTA's threads (256: a thread a
  digit of the widest level at a group of one digit). A level's warps enter the body compiled for its group through `uniform_switch` over the group's log2 (8 members, 1
  to 128 digits: a group of 256 is a level of 256 whole), `form<GB, WORDS>`. A level the side walks whole forms nothing. THE WARP'S PLACE AND ITS LEVEL'S GROUP ARE
  VALUES EVERY LANE HOLDS, lane 0's by shuffle (`lane_broadcast`): the branches that pick a warp's level and body stand on them, so the toolkit keeps the walk's
  shuffles whole and puts no convergence instruction in its loops (derived from the thread index alone, it put nine there, and the lab's checker refused the kernel).
- THEN, BEHIND A SECOND CTA BARRIER, THE LEAF GROUPS' LINES (`lay_lines`): a thread a leaf group a trip, 1,024 leaf groups a trip.
- THE HOST (`mask_prepare`) checks what the kernel assumes and nothing else: the density, the workspace and each mask a contiguous array of its dtype on the density's
  card (`check_operand`); a slot a mask, distinct and inside the density's depth, a call with no mask at all being a call of none; `widths`, the digits of each of the
  call's levels, masked or not, each a power of two from 2 to 256; each side's batch-heads (`heads`, one or more) and the call's tokens a head (`tokens`, one or more);
  each mask int32 `[B, H, ceil(T / 32), width]` with its level's width and its side's batch-heads; the mask's memory 16-byte aligned (a group's digits are loaded 16 bytes
  at once; a width under 4 has no such load) and the workspace's (a line is stored 16 bytes at once); the workspace as large as the layout; the grid at most 2^31 - 1
  CTAs. It takes the call's descriptor, `plan::descriptor_of<plan::mask_prepare::Kernel>` at the widest level's width and the density's depth, so a call outside the
  contract (`hardware/kernels/mask_prepare/parts/call.json`) is refused by name before a launch.
- THE PASS ALWAYS WRITES, A DENSE SIDE AS ONE LEAF GROUP OF THE CALL'S OWN TOKENS. The owner (2026-11-26), asked whether a fully dense call should do nothing in the pass and
  the carry read no mask and take it as dense, was told what is built: the pass writes both sides whatever the masks, a side that lists no level being one leaf group a window
  whose line is the call's own tokens, so the fold always reads a line, has one path and keeps no bookkeeping of the call's end. He answered: "ok thats fine makes sense".
  The host passes each kernel a bit a level a side for the masks that are present (`Args::masked`), and a level-side with no mask is taken at the whole density in the
  carve whatever its density byte, so a caller's inconsistent byte cannot ask for more leaf groups than the workspace, sized from the masks present, holds. A dense side's
  cost, predicted before its measurement: 1.2 to 2.5 SM-cycles a token; NOT MEASURED.

## <a id="entry"></a>The entry

`mask_prepare.cuh` declares two functions.

- `mask_prepare(masks, slots, widths, heads, tokens, density, workspace)` runs the pass over a call. `masks[i]` is the mask of level-side `slots[i]` (`side * 4 + level`, the read
  side first); a level-side with no mask has none, so `masks` and `slots` list the masks that are present and no others. `widths` are the digits of each of the call's levels,
  `heads` each side's batch-heads (the read side's first), `tokens` the call's tokens a head, and `density` the call's density, uint8 `[2, D]`. It writes each side's leaf
  groups' mask words and counts into `workspace` at the layout of [`../common/plan/mask_prepare_layout.md`](../common/plan/mask_prepare_layout.md).
- `mask_prepare_bytes(widths, masked, heads, tokens, device)` is the size of that layout in bytes, from the call's shape alone: its levels' digits (`widths`), each side's masked
  levels as a bit a level (`masked`, the read side's first), each side's batch-heads (`heads`) and its tokens a head (`tokens`), on the card `device` (which gives the window).
  It reads no density.

## <a id="smem"></a>Its shared memory

`Smem<Bt>`, two tiles of the pass's `SmemLayout`: `Handed`, the 21 handed integers, and `Formed`, each level's group words of the CTA's window,
`formed[(level * words + word) * 256 + group]` -- 4 levels x 32 words x 256 groups x 4 bytes, 131,072 bytes on the hopper card.

BOTH ACCESSES ARE A WAVEFRONT AN INSTRUCTION ([`../../KERNEL_STANDARDS.md` section 30](../../KERNEL_STANDARDS.md#s30)), by the layout. The walk stores WORD-MAJOR: a
warp's storing lanes hold consecutive groups, so their stores of one mask word are consecutive words. The lines' loads are of one mask word of one level across a warp's
32 consecutive leaf groups, whose groups of that level differ in their low five bits at most (the leaf group's index puts a level's inner field lowest and its outer
field directly above the inner fields, [`../common/plan/leaf_groups.md`](../common/plan/leaf_groups.md#order)): the lanes read 32 consecutive words at most, or the same
word. NOT MEASURED: the count the profiler gives.

## <a id="the-walk"></a>The walk

`form<GB, WORDS>` is a level-side of one window at a group of `1 << GB` digits. A thread's UNIT is the digits it loads of one word, `min(GB, 4)` bits of them:

- a thread a group at groups of 1 to 16 digits: its unit is the group, `Raw<U>` holding them as loaded (a word at one digit, a pair at 2, 16-byte lines from 4);
- a thread 16 digits at groups of 32 to 128: its unit is 16 digits and the group's `1 << (GB - 4)` lanes (2 to 8, adjacent threads of a warp) OR their words by warp
  shuffle, in `finish`.

A thread walks the window's `WORDS` words in order with ITS LOADS HELD AHEAD: `kHeld` words' loads are issued before any is read -- 8 at up to four digits a thread, 4
at eight, 2 at more -- and each trip finishes `kHeld` words, issuing as it finishes one the load `kHeld` words on into the registers that word left (`#pragma unroll 1`
over the trips). Per word it stores the group's word at its place in the level's formed words.

- A word past the mask's last (the window is the mask's last and ragged) is read as zero: its load re-reads the thread's first word (an address inside the mask) and the
  word's `keep` is zero, so there is no branch and no read past the mask.
- ONE THREAD WRITES EACH ADDRESS (`store`). A level has `width / unit` units and 256 threads: a warp whose first unit is past them returns at once; a lane past them in the level's last
  warp walks the first unit and stores nothing; and of a group's lanes only the first stores. Every lane of a walking warp loads and shuffles, so a warp's shuffles are
  whole.

## <a id="finish"></a>A word finished

`finish<U, LANES>(raw, keep, store, word)` is the one place a group's word is finished: it ORs the group's digits' words (by shuffle across the group's lanes) and stores
it. Two things in it are constraints and not choices: the shuffle runs on every lane, storing or not (a lane that sat out would leave `__shfl_xor_sync`'s full mask naming
a lane that is not there), and the store is under `store`, so that no two threads write one address (KERNEL_STANDARDS, the sanitizer lane: a benign race is a race).

## <a id="lines"></a>The lines

`lay_lines<WORDS>` writes the CTA's block ([the layout](../common/plan/mask_prepare_layout.md#the-layout)). A trip is 1,024 leaf groups, a thread one of them: its group on
each level from the handed fields (`leaf_group_levels`' arithmetic); then, a mask word at a time, the call's own tokens of the word ANDed with each used level's formed
word of the thread's group; four words are one 16-byte store of the line, and the words' popcounts sum to the count, stored last. A thread past the side's last leaf group
forms its first again and stores nothing.

- A LEVEL IS USED where the side lists it. A side that lists no level has one leaf group, and thread 0 writes the call's own tokens of the window as its line.
- THE CALL'S OWN TOKENS of a word: every bit where the word is below the call's last word, the low `T % 32` bits at it, none past it. So a line never holds a token past
  the call's last, whatever the masks hold there, and a reader masks nothing.

## Its time

NO RECORD in this form: the experiment ([`docs/lab/mask_prepare_time.md`](../../lab/mask_prepare_time.md)) is written for it (one masked level at each group size, two
masked levels at groups of 4 digits, no masked level) and its recorded run is owed. WHAT TO ISOLATE WHEN IT RUNS, none of it measured: a line is stored 16 bytes an
instruction by 32 lanes into 32 different lines, so no instruction writes contiguous memory; each CTA derives the call's carve itself; and the walk of a leaf group's
words is one dependent trip after another.
