# `carry/carry_kernel.cuh` — the carry's device code

Mirrors `csrc/fsla/src/carry/carry_kernel.cuh`. THE SKELETON
([`KERNEL_STANDARDS.md` section 19](../../KERNEL_STANDARDS.md#s19)) of the design
[`../../architecture/hopper-carry.md`](../../architecture/hopper-carry.md): every role with its registers, its
hand-offs and its order of work, each component a stub named where it goes. It has no host half, no op and no data:
the lab runs it ([`../../lab/carry_skeleton.md`](../../lab/carry_skeleton.md)). Its first component, built and not
yet called by a role, is the listing ([`carry_list.md`](carry_list.md)); its second, the value lines' fetch
([`carry_values.md`](carry_values.md)).

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
  built.
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
- WHAT IS NOT BUILT is named in the code where it goes: the image, the roles' use of the lists and of the value lines'
  fetch, the coefficients, the steps, the sums' stores and the adds; with them, the CTA's shared memory through the one layout
  ([`../common/smem_layout.md`](../common/smem_layout.md)).
- <a id="the-cta"></a>`carry_cta<Bt, FOLD, READ, Tr>(args, tracer)`: each role's whole body runs in its own branch,
  its registers set at the branch's head.
