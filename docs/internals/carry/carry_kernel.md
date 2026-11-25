# `carry/carry_kernel.cuh` — the carry's device code

Mirrors `csrc/fsla/src/carry/carry_kernel.cuh`. THE SKELETON
([`KERNEL_STANDARDS.md` section 19](../../KERNEL_STANDARDS.md#s19)) of the design
[`../../architecture/hopper-carry.md`](../../architecture/hopper-carry.md): every role with its registers, its
barriers and its order of work, each component a stub named where it goes. It has no host half, no op and no data:
the lab runs it ([`../../lab/carry_skeleton.md`](../../lab/carry_skeleton.md)).

- <a id="roles"></a>THE ROLES. A CTA is 512 threads, four warpgroups of 128 (a warpgroup is the tensor step's unit):
  two FOLD warpgroups (`kFolds`), the READOUT's (`kReadAt`), and the DRAIN's four warps (`kDrainAt`).
- <a id="registers"></a>THE REGISTERS. A fold thread's and a readout thread's counts are the arm's (`FOLD`, `READ`,
  template parameters of `carry_cta`): a split the climb tunes, so the kernel states no number for them. A drain
  thread takes what the multiprocessor's register file leaves (`Registers<Bt, FOLD, READ>::kDrain`, from `regs_per_sm`
  of the build's caps; a split that leaves it under the instruction's least does not compile). A role's count is set
  once, at the head of its branch: two counts in one warpgroup hang the CTA.
- <a id="barriers"></a>THE BARRIERS (`Bars`, a shared word each; an arrival is a warp's first lane's, a wait
  suspends): `tiles_ready[k]`, the folds' eight warps' arrival when tile group `k` of the window's image is
  published; `tiles_read[k]`, the readout's four warps' when its steps have read it; `panel_ready[p]`, the readout's
  when panel `p` holds a pass's sums; `panel_freed[p]`, the drain's when it has added them. A fold warpgroup and the
  readout each have a named barrier of their own 128 threads (`kFoldId`, `kReadId`).
- <a id="smem"></a>SHARED MEMORY is the one layout (`CarrySmem<Bt>`,
  [`../common/smem_layout.md`](../common/smem_layout.md)), one CTA a multiprocessor: in the skeleton the barriers
  alone.
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
- WHAT IS NOT BUILT is named in the code where it goes: the image, the lists, the value lines' fetch, the
  coefficients, the steps, the sums' stores and the adds.
- <a id="handoff"></a>A HAND-OFF's form is the arm's while it is measured (`hand<WAIT>`, `take<WAIT>`): the shared
  barrier with a wait that suspends (0), or a named barrier of the two roles' threads, the producer arriving and the
  waiter syncing (a tile group's two and a panel's two take twelve of the CTA's sixteen with the roles' own three).
  The lab prices both on the idle skeleton ([`../../lab/carry_skeleton.md`](../../lab/carry_skeleton.md)); the form not
  kept goes. A WAIT THAT SPINS is not tried there (the owner, 25 November 2026: "the spinning will look free until it
  contends with real work ... just keep spin in mind as something we can try if we see barriers as a problem in the
  final kernel").
- <a id="the-cta"></a>`carry_cta<Bt, FOLD, READ, WAIT>(args)`: the first thread initializes the barriers, the CTA meets, and each
  role's whole body runs in its own branch.
