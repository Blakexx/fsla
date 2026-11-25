# `carry/carry_kernel.cuh` — the carry's device code

Mirrors `csrc/fsla/src/carry/carry_kernel.cuh`. THE SKELETON
([`KERNEL_STANDARDS.md` section 19](../../KERNEL_STANDARDS.md#s19)) of the design
[`../../architecture/hopper-carry.md`](../../architecture/hopper-carry.md): every role with its registers, its
barriers and its order of work, each component a stub named where it goes. It has no host half, no op and no data:
the lab runs it ([`../../lab/carry_skeleton.md`](../../lab/carry_skeleton.md)).

- <a id="roles"></a>THE ROLES. A CTA is 512 threads, four warpgroups of 128 (a warpgroup is the tensor step's unit):
  two FOLD warpgroups (`kFolds`), the READOUT's (`kReadAt`), and the DRAIN's four warps (`kDrainAt`).
- <a id="registers"></a>THE REGISTERS (`Registers<Bt>`): a fold thread 176 and a readout thread 104, the lab's split
  ([`../../lab/carry_token.md`](../../lab/carry_token.md)) and the climb's to change; a drain thread what the SM's
  register file leaves (`regs_per_sm` of the build's caps), 56 on hopper. A role's count is set once, at the head of
  its branch (`registers_up`, `registers_down`).
- <a id="barriers"></a>THE BARRIERS (`Bars`, a shared word each; an arrival is a warp's first lane's, a wait
  suspends): `tiles_ready[k]`, the folds' eight warps' arrival when tile group `k` of the window's image is
  published; `tiles_read[k]`, the readout's four warps' when its steps have read it; `panel_ready[p]`, the readout's
  when panel `p` holds a pass's sums; `panel_freed[p]`, the drain's when it has added them. A fold warpgroup and the
  readout each have a named barrier of their own 128 threads (`kFoldId`, `kReadId`).
- <a id="smem"></a>SHARED MEMORY is the one layout (`CarrySmem<Bt>`,
  [`../common/smem_layout.md`](../common/smem_layout.md)), one CTA a multiprocessor: in the skeleton the barriers
  alone.
- <a id="fold"></a>`fold_role`: a window is, for each tile group, the wait on the readout's read of the window
  before, the group published, and its arrival; then the warpgroup's barrier before its steps and behind them. NOT
  BUILT, named in the code where each goes: the image; the next window's lists; the value lines' landing; the entries
  expanded; the steps; the next lines' copies.
- <a id="readout"></a>`read_role`: a window is two passes of two tile groups: the waits on them, the warpgroup's
  barrier, the arrivals that they are read, then each of the two panels in its turn behind the drain's arrival of
  the turn before. NOT BUILT: the readers' coefficients; the steps; the sums' stores.
- <a id="drain"></a>`drain_role`: a window is four turns, a panel each alternately: the wait on it, and the arrival
  that frees it. NOT BUILT: the adds.
- <a id="the-cta"></a>`carry_cta<Bt>(args)`: the first thread initializes the barriers, the CTA meets, and each
  role's whole body runs in its own branch.
