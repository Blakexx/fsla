# The prefill's mask-preparation pass (`csrc/fsla/src/mask_prepare/mask_prepare.cu`, `mask_prepare.cuh`, `mask_prepare_kernel.cuh`)

The pass the op runs before its carry ([`../../architecture/mask-preparation.md`](../../architecture/mask-preparation.md#the-mask-preparation-pass)): one launch over
every level-side of the call that has a mask, writing each level-side's prepared group words and totals into the op's workspace at the layout of
[`../common/plan/mask_prepare_layout.md`](../common/plan/mask_prepare_layout.md). The device code is `mask_prepare_kernel.cuh` ([`mask_prepare_kernel.md`](mask_prepare_kernel.md)), the host half (the op's checks, the grid, the place in the workspace, the launch) `mask_prepare.cu`. Hand CUDA, global loads and stores, and for the group one shared word behind one CTA barrier: no tensor core. The Python face is
`fsla/ops/mask_prepare.py` (`prepare_masks`, which passes the masks' device to `mask_prepare_bytes`; the op derives the card's window); its exactness is graded by `tests/oracle/test_mask_prepare.py` against the host reference `tests/oracle/mask_prepare_reference.py` (the words and totals, bit for bit), and its host checks run without a GPU in `tests/unit/test_mask_prepare.py`; the registry of its contract is `hardware/kernels/mask_prepare/` ([`../hardware/registry.md`](../hardware/registry.md)).

Symbols: `T` tokens, a window is the card's (`plan::window_of_cc`, [`../common/plan/levels.md`](../common/plan/levels.md); tokens), `words` mask words of 32 tokens (the hopper card's: 1,024 tokens, 32 words); a level-side is a side (0 reads, 1 writes) and a level, `slot = side * 4 + level`; `width` its digits (a power of two, 2 to 256), `width_bits` its log2; `g` its
group in digits (a power of two from 1 to 256), `GB` its log2, `groups = width / g`.

## <a id="the-kernel"></a>The kernel

`mask_prepare_kernel<Bt>` has ONE template argument, the build tag: the pass has one arm a card, and THE WINDOW IS THE CARD'S: the kernel takes its `words` from `plan::window_of_cc(Bt::build.cc) / 32`, a constant of its build (`prepare<GB, WORDS>`, `mask_prepare_cta<WORDS>`), and the host takes the same from `card_of(device).cc`. A level-side's group is no host choice, because the host never reads the density
([`../common/plan/prefill_launch.md`](../common/plan/prefill_launch.md)): the kernel derives it on the device.

- THE GRID IS SIZED FOR A GROUP OF ONE DIGIT, which is what the host can know: a thread a (window, digit), `ctas_of` CTAs of 1,024 threads for each level-side, the level-sides' CTAs one after the other
  (`Args::first_cta`, the last entry the grid). A CTA finds its level-side from its index.
- THE GROUP IS THE CARVE'S, DERIVED THERE. A level-side's SIZE is the group its own density byte gives (`prefill_group_bits`, `cta_carve.h`; a byte `b` is `b + 1` 256ths), and the call's carve raises a group over its size where a set would hold fewer than 16 leaves
  ([`../common/plan/prefill_launch.md`](../common/plan/prefill_launch.md#least)); the carry loads one word a level a set, so the pass prepares at the carve's group. A CTA first takes the size, which costs it one byte: where the size is the level's the level-side is WHOLE (the carve raises nothing over a level), its mask is never read and the CTA leaves
  with its region untouched; and a CTA that is surplus at the size (below) is surplus at any group raised from it, and leaves. Then its first warp computes the call's carve (`carved_group_bits`: `prefill_carve` of every level's width, `Args::level_bits`, and the two sides' density bytes) and hands the level-side's group to the CTA's other warps in one shared word (`Handed`, the one tile of the pass's `SmemLayout`, which the kernel passes to `mask_prepare_cta`) behind a CTA barrier.
  A group raised to the level's size is whole and the CTA leaves; otherwise `uniform_switch` over the group's log2 (8 members, 1 to 128 digits: a group of 256 is a level of 256 whole) enters the body compiled for it, `prepare<GB, WORDS>`.
- A SURPLUS CTA LEAVES AT ONCE. At a group of 2 digits or more a window needs fewer threads than the grid holds: a CTA whose first thread is past the level-side's last window returns before it loads.
- THE HOST (`mask_prepare`) checks what the kernel assumes and nothing else: the density, the workspace and each mask a contiguous array of its dtype on the density's card (`check_operand`); a slot a mask, distinct and inside the density's depth; `widths`, the digits of each of the call's levels, masked or not, one a level of the density's depth, each a power of two from 2 to 256, and a mask's width its level's; each mask int32 `[B, H, ceil(T / 32), width]` with `width` a power of two from 2 to 256; the mask's memory
  16-byte aligned (a group's digits are loaded 16 bytes at once; a width under 4 has no such load); the workspace as large as the layout; the grid at most 2^31 - 1 CTAs. It takes the call's descriptor, `plan::descriptor_of<plan::mask_prepare::Kernel>`
  at the widest mask's width and the density's depth, so a call outside the contract (`hardware/kernels/mask_prepare/parts/call.json`) is refused by name before a launch, and `mask_prepare_bytes(width_bits, heads, words, device)` is the layout's size from the call's shape and the card's window (tokens, whole mask words) alone.

## <a id="the-walk"></a>The walk

`prepare<GB, WORDS>` is a level-side at a group of `1 << GB` digits. A thread's UNIT is the digits it loads of one word, `min(GB, 4)` bits of them:

- a thread a (window, group) at groups of 1 to 16 digits: its unit is the group, `Raw<U>` holding them as loaded (a word at one digit, a pair at 2, 16-byte lines from 4);
- a thread a (window, 16 digits) at groups of 32 to 256: its unit is 16 digits and the group's `1 << (GB - 4)` lanes (2 to 16, adjacent threads of a warp) OR their words by warp shuffle, in `finish`.

A thread walks its window's `WORDS` words in order with ITS LOADS HELD AHEAD: `kHeld` words' loads are issued before any is read -- 8 at up to four digits a thread, 4 at eight, 2 at more -- and each trip finishes `kHeld` words, issuing as it finishes one the load `kHeld` words on into the registers that word left, so
`kHeld` loads are in flight through the walk (`#pragma unroll 1` over the trips). Per word it stores, at its group's place in the block, the group's word (where the group is more than one digit) and adds the word's popcount to its running total (`live`); at the end it stores the window's total, the group's live tokens of the window.

- A word past the mask's last (the window is the mask's last and ragged, `have < WORDS`) is read as zero: its load re-reads the thread's first word (an address inside the mask) and the word's `keep` is zero, so there is no branch and no read past the mask.
- ONE THREAD WRITES EACH ADDRESS (`store`). A thread past the level-side's last window (in the level-side's last CTA) walks the first window instead and stores nothing, and of a group's lanes only the first stores. Every lane still loads and shuffles, so a warp's shuffles are whole.

## <a id="finish"></a>A word finished

`finish<U, LANES>(raw, keep, store, word, live)` is the one place a word is finished: it ORs the group's digits' words (by shuffle across the group's lanes), stores the word where the group is more than one digit, and adds its popcount to `live`. Two things in it are constraints and not choices: the shuffle runs on every lane, storing or not (a lane that sat out would leave `__shfl_xor_sync`'s full mask naming a lane that is not there), and the stores are under `store`, so that no two threads write one address (KERNEL_STANDARDS, the sanitizer lane: a benign race is a race).

## Its time

Measured on an H100 at each of the eight group sizes it prepares, the mask read once: [`docs/lab/mask_prepare_time.md`](../../lab/mask_prepare_time.md).
NOT MEASURED: a level narrower than 256 digits, several level-sides in one launch, and a call short enough that the launch itself is most of its time.

