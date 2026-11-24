# The prefill's mask-preparation pass (`csrc/fsla/src/mask_prepare/mask_prepare.cu`, `mask_prepare.cuh`)

The pass the op runs before its gather and its carry ([`../../architecture/hopper-gather.md`](../../architecture/hopper-gather.md#the-mask-preparation-pass)): one launch over
every level-side of the call that has a mask, writing each level-side's prepared words, counts and totals into the op's workspace at the layout of
[`../common/plan/mask_prepare_layout.md`](../common/plan/mask_prepare_layout.md). Hand CUDA, global loads and stores alone: no shared memory, no barrier, no tensor core. The Python face is
`fsla/ops/mask_prepare.py` (`prepare_masks`); the registry of its contract is `hardware/kernels/mask_prepare/` ([`../hardware/registry.md`](../hardware/registry.md)).

Symbols: `T` tokens, a window is 1,024 tokens, 32 mask words of 32 tokens; a level-side is a side (0 reads, 1 writes) and a level, `slot = side * 4 + level`; `width` its digits (a power of two, 2 to 256), `width_bits` its log2; `g` its
group in digits (1, 4, 16, 32, 64, 128 or 256), `GB` its log2, `groups = width / g`.

## <a id="the-kernel"></a>The kernel

`mask_prepare_kernel<Bt>` has ONE template argument, the build tag: the pass has one arm a card. A level-side's group is no host choice, because the host never reads the density
([`../common/plan/prefill_launch.md`](../common/plan/prefill_launch.md)): the kernel derives it on the device.

- THE GRID IS SIZED FOR A GROUP OF ONE DIGIT, which is what the host can know: a thread a (window, digit), `ctas_of` CTAs of 1,024 threads for each level-side, the level-sides' CTAs one after the other
  (`Args::first_cta`, the last entry the grid). A CTA finds its level-side from its index.
- THE GROUP IS DERIVED THERE. The CTA reads its level-side's density byte `density[side * levels + level]`; a byte `b` is `b + 1` 256ths, and at 256 (byte 255) the level-side is dense, its mask is never read and the CTA leaves
  with its region untouched. Otherwise `prefill_group_bits(density, width_bits)` (`cta_carve.h`, the carve's one rule) gives the group and `uniform_switch` over `kGroupSizeBits` (7 members, `group_index` the place of a group's bits among them) enters
  the body compiled for it, `prepare<GB>`.
- A SURPLUS CTA LEAVES AT ONCE. At a group of 4 digits or more a window needs fewer threads than the grid holds: a CTA whose first thread is past the level-side's last window returns before it loads.
- THE HOST (`mask_prepare`) checks what the kernel assumes and nothing else: a slot a mask, distinct and inside the density's depth; each mask int32 `[B, H, ceil(T / 32), width]` with `width` a power of two from 2 to 256; the mask's memory
  16-byte aligned (a group's digits are loaded 16 bytes at once; a width under 4 has no such load); the workspace as large as the layout; the grid at most 2^31 - 1 CTAs. It takes the call's descriptor, `plan::descriptor_of<plan::mask_prepare::Kernel>`
  at the widest mask's width and the density's depth, so a call outside the contract (`hardware/kernels/mask_prepare/parts/call.json`) is refused by name before a launch, and `mask_prepare_bytes` is the layout's size from the call's shape alone.

## <a id="the-walk"></a>The walk

`prepare<GB>` is a level-side at a group of `1 << GB` digits. A thread's UNIT is the digits it loads of one word, `min(GB, 4)` bits of them:

- a thread a (window, group) at groups of 1, 4 and 16 digits: its unit is the group, 1, 4 or 16 digits, `Raw<U>` holding them as loaded (a word at one digit, 16-byte lines from 4);
- a thread a (window, 16 digits) at groups of 32 to 256: its unit is 16 digits and the group's `1 << (GB - 4)` lanes (2 to 16, adjacent threads of a warp) OR their words by warp shuffle, in `finish`.

A thread walks its window's 32 words in order, TWO WORDS' LOADS HELD: the first two words' loads are issued before either is read, and each trip finishes a word and issues the load two words on into the registers that word left, so
two loads are in flight through the walk (`#pragma unroll 1` over word pairs). Per word it stores, at its group's place in the block, the group's word (where the group is more than one digit), the count of the group's live tokens of the window before the
word (`before`) and adds the word's popcount; at the end it stores the window's total.

- A word past the mask's last (the window is the mask's last and ragged, `have < 32`) is read as zero: its load re-reads the thread's first word (an address inside the mask) and the word's `keep` is zero, so there is no branch and no read past the mask.
- A thread past the level-side's last window (the last CTA of the level-side) walks the first window instead, so EVERY LANE OF A GROUP STORES THE VALUES ITS WINDOW'S OWN THREAD STORES, at the same addresses: the walk has no branch on who the thread is.
- The group's lanes all store the same word, count and total; the stores are idempotent.

## <a id="finish"></a>A word finished

`finish<U, LANES>` takes the thread's loaded digits of a word: `word_of` ORs the unit's digits' words, the `keep` mask zeroes a word past the mask's last, the lanes OR across the warp by `__shfl_xor_sync`
(`LANES` is 1 up to a group of 16 digits), the group's word is stored (not at a group of one digit, where the word is the mask's own), the count stored is `before`, and `before` grows by the word's popcount.

## What is NOT measured

- THE LIBRARY KERNEL ITSELF. The lab ([`../../lab/mask_prep.md`](../../lab/mask_prep.md)) measured groups of 1, 4 and 16 digits in a lab kernel with its last two words peeled; the library's `mask_prepare_kernel` is not that kernel, and no measurement of it has run.
- GROUPS OF 32 TO 256 DIGITS, at any kernel: the lab measured none.
- NO DEVICE TEST HAS RUN: `tests/oracle/test_mask_prepare.py` grades the kernel against `tests/oracle/mask_prepare_reference.py` exactly and is written for a GPU; the host's checks run without one (`tests/unit/test_mask_prepare.py`).
