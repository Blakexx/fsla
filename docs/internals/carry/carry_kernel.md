# `carry/carry_kernel.cuh` — the carry's device code

Mirrors `csrc/fsla/src/carry/carry_kernel.cuh`. THE SKELETON
([`KERNEL_STANDARDS.md` section 19](../../KERNEL_STANDARDS.md#s19)) of the design
[`../../architecture/hopper-carry.md`](../../architecture/hopper-carry.md): every role with its registers, its
hand-offs and its order of work, and ONE COMPONENT BUILT IN IT, the fold warpgroups' supply ([below](#supply)); every
other component is a stub named where it goes. It has no host half and no op: the lab runs it, on a call's real
operands ([`../../lab/carry_supply.md`](../../lab/carry_supply.md)) and on a call of no tokens, where a window is its
hand-offs ([`../../lab/carry_skeleton.md`](../../lab/carry_skeleton.md)). The supply is the listing
([`carry_list.md`](carry_list.md)) and the value lines' fetch ([`carry_values.md`](carry_values.md)) on the call's
groups ([`carry_groups.md`](carry_groups.md)).

- <a id="roles"></a>THE ROLES. A CTA is 512 threads, four warpgroups of 128 (a warpgroup is the tensor step's unit):
  two FOLD warpgroups (`kFolds`), the READOUT's (`kReadAt`), and the DRAIN's four warps (`kDrainAt`).
- <a id="registers"></a>THE REGISTERS. A fold thread's and a readout thread's counts are the arm's (`FOLD`, `READ`,
  template parameters of `carry_cta`): a split the climb tunes, so the kernel states no number for them. A drain
  thread takes what the multiprocessor's register file leaves (`Registers<Bt, FOLD, READ>::kDrain`, from `regs_per_sm`
  of the build's caps; a split that leaves it under the instruction's least does not compile). A role's count is set
  once, at the head of its branch: two counts in one warpgroup hang the CTA.
- <a id="handoff"></a>A HAND-OFF between roles is a NAMED BARRIER of the two roles' threads: the producer's warps
  arrive and go on (`hand`), the waiter's sync (`take`). There are twelve -- a tile group's "published" (the folds to
  the readout) and "read" (the readout to the folds), a panel's "filled" (the readout to the drain) and "drained" --
  beside the roles' own three (a fold warpgroup's and the readout's, `kFoldId`, `kReadId`): fifteen of a CTA's sixteen,
  which a static assertion holds. WHY NAMED: on the idle skeleton a window's 16 hand-offs cost 528 cycles so, against
  849 as shared barriers with suspending waits ([`../../lab/carry_skeleton.md`](../../lab/carry_skeleton.md)), and the
  owner keeps one form (25 November 2026: "I don't want to maintain two forms, so we need to pick one after the barrier
  result comes in"). A WAIT THAT SPINS was not tried there ("the spinning will look free until it contends with real
  work ... just keep spin in mind as something we can try if we see barriers as a problem in the final kernel"). A
  copy's landing is not a hand-off of this kind: it stays the copy's own wait.
- <a id="trace"></a>THE TRACER (`Tr`, a template parameter of every role). A role tells its tracer each wait's start
  and end, each arrival, and its phases (`Site`: a role's entry and exit, a window's start). The library's is
  `NoTrace`, whose calls are nothing, so the library takes no dependency and pays nothing; the lab passes warpline's
  stamps, and a warp's timeline is then the kernel's own account of where it waited
  ([`../../lab/carry_skeleton.md`](../../lab/carry_skeleton.md)). A component adds its phases to `Site` as it is
  built; the supply's are a set's start, its tokens listed, the next set's words asked for, and a round's fetch begun,
  issued and landed. `round(fold, set, window, round, token, rows)` is the supply's: a fold warp calls it when it has issued a
  round's copies, which is how the lab copies a round's tokens and rows out with no test code in the library
  ([`../../lab/carry_supply.md`](../../lab/carry_supply.md)).
- <a id="fold"></a>`fold_role`. THE IMAGE GOES OUT A TILE GROUP AT A TIME so that the readout starts on the first
  group while the folds store the rest, and each group waits on the readout's read of the window before because the
  image is overwritten in place: the state has one image, not a ring of them
  ([`../../architecture/hopper-carry.md`](../../architecture/hopper-carry.md#a-window)). The warpgroup meets at its own
  barrier before its steps (every thread's operands are stored) and behind them (every step has read its tile, so the
  next window's copies may land in it: R21's release edge).
- <a id="readout"></a>`read_role`. TWO PANELS so that a pass's sums are stored while the drain adds the pass before:
  a panel's turn waits on the drain's arrival of its turn before, which is what holds the readout back when the
  drain's adds are the slower (R22: no cliff, the readout waits).
- <a id="drain"></a>`drain_role` follows the readout a panel behind and frees each panel as it has added it.
- <a id="channels"></a>A CTA'S VALUE CHANNELS are the tensor step's 64 (`kChannels`), a SLICE of a token's line: a
  wider value is more CTAs over the slices of the same leaves
  ([`../../architecture/hopper-carry.md`](../../architecture/hopper-carry.md)), and a CTA's place carries its slice
  ([`carry_groups.md`](carry_groups.md#place)).
- <a id="smem"></a>THE CTA'S SHARED MEMORY is one layout (`Smem`, [`../common/smem_layout.md`](../common/smem_layout.md)).
  Its one tile so far is the fold warps' rows (`FoldRows`): a warp two rounds of 32 rows (`kHeld`), 8 KB, the eight
  fold warps 64 KB, on a 1,024-byte boundary ([`../common/smem_layout.md`](../common/smem_layout.md#aligned)), which
  is where a step's operand of rows starts ([`../common/atoms/tensor.md`](../common/atoms/tensor.md#rows)). ITS EDGES
  (R21): a round's copies are one committed group, and the warp waits for the group before it, so two are in flight;
  a slot's rows are overwritten by the round issued two later, and that issue stands behind the tracer's call for the
  round whose rows they were -- where the wait of that round's steps will stand (NOT BUILT: the steps).
- <a id="supply"></a>THE FOLD'S SUPPLY, built: a fold warp lists its sets' live tokens and fetches their value lines.
  A SET is one leaf group of the fold's box, a group a level ([`carry_groups.md`](carry_groups.md#sets)); a fold holds
  `fold_sets` of them, a power of two. THE WARPS' SHARES ARE THE CARVE'S, NOT THE DATA'S: with four sets or more a
  warp takes every fourth set, whole; with two, two warps take a set and alternate its rounds; with one, the four
  warps take its rounds in turn -- so a fold's four warps issue copies together whatever its sets. A SET'S WORD a lane
  is the call's own tokens ANDed with the set's group's word on each level the side lists; its parts are formed
  outside the loop over sets -- a level's `base`, `shift` and `mask` once a fold, a level's row once and then a step
  on a window ([`carry_groups.md`](carry_groups.md#word)). THE WORDS ARE LOADED A SET AHEAD AND ANDed AT THEIR USE:
  a warp issues in order, so a word ANDed where it is loaded holds the warp there until it lands, whatever was meant
  to run under it. The listing ([`carry_list.md`](carry_list.md)) gives a round's 32 tokens and the fetch
  ([`carry_values.md`](carry_values.md)) lands their lines in the warp's rows; a set with no live token issues
  nothing. The loops are bounded by values every lane holds (a shuffle from lane 0, a vote) and the levels the side
  lists are decided once a call, outside them ([`../../lab/collective_branch.md`](../../lab/collective_branch.md)).
  NOT BUILT: sides of different head counts (one head index reads both); the coefficients and the steps that consume
  a round. THE BUDGET, stated before the code
  ([`../../KERNEL_STANDARDS.md` section 19](../../KERNEL_STANDARDS.md#s19)): a set a window, a warp -- a load a lane
  for each level the side lists and the listing's running sums; a round -- the select and 8 copy instructions a lane.
  A ROUND IS PAID WHOLE whatever number of its 32 places are live. What a warp's window is made of, by its stamps:
  [`../../lab/carry_supply.md`](../../lab/carry_supply.md).
- WHAT IS NOT BUILT is named in the code where it goes: the image, the readout's lists, the coefficients, the steps,
  the sums' stores and the adds, and the tiles they add to the CTA's shared memory.
- <a id="the-cta"></a>`carry_cta<Bt, FOLD, READ, Tr>(operands, tracer)`: each role's whole body runs in its own branch,
  its registers set at the branch's head. A fold thread derives the call's carve and its CTA's place there, once
  ([`carry_groups.md`](carry_groups.md#carve)).
