# The lab's experiments

One page per experiment, mirroring `measure/lab/experiments/<name>.cu`: the question it asks of the machine, its
method, its record (a sample's output, `data:records/fsla/measure/lab/<name>.run/<run>/output.json`) and what it settled. How an experiment is written, run
and recorded, and what guards a measurement, is [`docs/internals/tools/lab.md`](../internals/tools/lab.md); a law
about the machine cites the experiment it rests on ([`docs/KERNEL_STANDARDS.md#s26`](../KERNEL_STANDARDS.md#s26)).

| experiment | kind |
|---|---|
| [`collective_branch`](collective_branch.md) | experiment |
| [`smem_rates`](smem_rates.md) | experiment |
| [`row_span`](row_span.md) | experiment |
| [`vote_stash`](vote_stash.md) | experiment |
| [`facts_block`](facts_block.md) | experiment |
| [`facts_two`](facts_two.md) | experiment |
| [`wgmma_regs`](wgmma_regs.md) | experiment |
| [`bulk_drain`](bulk_drain.md) | experiment |
| [`reg_trade`](reg_trade.md) | experiment |
| [`reg_five`](reg_five.md) | experiment |
| [`row_ring`](row_ring.md) | experiment |
| [`wgmma_rate`](wgmma_rate.md) | experiment |
| [`stamp_cost`](stamp_cost.md) | experiment |
| [`clock_time`](clock_time.md) | experiment |
| [`device_probe`](device_probe.md) | experiment |
| [`byte_expand`](byte_expand.md) | experiment |
| [`cluster_sum`](cluster_sum.md) | experiment |
| [`state_move`](state_move.md) | experiment |
| [`fold_stream`](fold_stream.md) | experiment |
| [`drain_rate`](drain_rate.md) | experiment |
| [`drain_role`](drain_role.md) | experiment |
| [`state_image`](state_image.md) | experiment |
| [`panel_merge`](panel_merge.md) | experiment |
| [`wgmma_width`](wgmma_width.md) | experiment |
| [`intra_chain`](intra_chain.md) | experiment |
| [`intra_window`](intra_window.md) | experiment |
| [`call_pair`](call_pair.md) | experiment |
| [`facts_lists`](facts_lists.md) | experiment |
| [`mask_prep`](mask_prep.md) | experiment |
| [`mask_prepare_time`](mask_prepare_time.md) | experiment |
| [`gather_pass`](gather_pass.md) | experiment |
| [`cast_cluster`](cast_cluster.md) | experiment |
| [`value_gather`](value_gather.md) | experiment |
| [`value_fetch`](value_fetch.md) | experiment |
| [`carry_skeleton`](carry_skeleton.md) | experiment |
| [`carry_list`](carry_list.md) | experiment |
| [`carry_fetch`](carry_fetch.md) | experiment |
| [`carry_supply`](carry_supply.md) | experiment |
| [`step_swizzle`](step_swizzle.md) | experiment |
| [`readout_role`](readout_role.md) | experiment |
| [`value_cast`](value_cast.md) | experiment |
| [`pool_fill`](pool_fill.md) | experiment |
| [`load_rate`](load_rate.md) | experiment |
| [`carry_window`](carry_window.md) | experiment |
| [`carry_token`](carry_token.md) | experiment |
| [`add_rate`](add_rate.md) | experiment |
| [`slot_sum`](slot_sum.md) | experiment |
| [`drain`](drain.md) | experiment |
| [`step_queue`](step_queue.md) | experiment |
| [`fold_forms`](fold_forms.md) | experiment |
| [`gather_rate`](gather_rate.md) | experiment |
| [`gather_group`](gather_group.md) | experiment |
| [`step_rolled`](step_rolled.md) | experiment |
| [`stream_rate`](stream_rate.md) | experiment |
| [`newton_row`](newton_row.md) | experiment |
| [`projection_tile`](projection_tile.md) | experiment |
| [`mask_pack`](mask_pack.md) | experiment |
| [`tma_box`](tma_box.md) | experiment |
| [`l2_policy`](l2_policy.md) | experiment |
| [`dependent_launch`](dependent_launch.md) | experiment |

## The toolkit's move to 13.5: records that carry the 13.0 build

Every record under `data:records/fsla/measure/lab/` but `carry_token`'s carries the 13.0 build (`Build cuda_13.0.r13.0/compiler.36260728_0`;
`carry_token`'s two records, of 2026-11-23, carry 13.5); the tree's toolkit
is 13.5 ([`docs/KERNEL_STANDARDS.md#s26`](../KERNEL_STANDARDS.md#s26): a new toolkit re-runs the experiments its rules cite). A
`[COMPILER]` or `[MACHINE]` rule's records, by what each needs:

- RE-READ UNDER 13.5, no GPU (`bazel build //measure/lab:<name>`, its validation, and `bazel run //measure/lab:<name>.run -- sass` for the registers: the build's declared SASS counts hold, no
  `setmaxnreg` notice, and each kernel's registers are the record's): `collective_branch` (sm_90a: no `CALL` in any of the five
  forms, one `WARPSYNC.COLLECTIVE` before each of the 8 shuffles in the two role forms, none in the others -- the form its page
  declares), `reg_trade` (no `setmaxnreg` notice, registers 128 and 126, [§5](../KERNEL_STANDARDS.md#s5)'s `[COMPILER]` form),
  `wgmma_regs` (the narrowing's `F2FP.BF16.F32.PACK_AB` per bf16 pair, its seven forms' registers 34 to 70,
  [§2](../KERNEL_STANDARDS.md#s2)), `intra_chain` (the cast form `I2FP.F32.S32` declared as `STEP_XU`, its twelve kernels'
  registers, [§29](../KERNEL_STANDARDS.md#s29) item 6).
- OWED TO THE NEXT H100 SESSION (cycles or rates: a GPU run; the record is replaced by the run, its rule stands on the 13.0
  record until then): `wgmma_regs` sm_90a (§2's 110-cycle narrowed step); `reg_trade` sm_90a (§5's trade of about 200 cycles a
  warp); `state_move`, `load_rate` and `value_gather` sm_90a ([§33](../KERNEL_STANDARDS.md#s33)'s scattered 128-byte row: the bulk
  copy 32.8 cycles, one warp's loads 33.0, four warps' 8.5); `intra_chain` sm_90a (§29 item 6's 985 cycles against 1,336);
  `smem_rates` sm_90a, with its sm_86 record (the shared-memory rates §29's price list is checked against); `facts_block` sm_90a,
  its record sm_86 (§26.3 `ee`: the floor counts the writes); `collective_branch` sm_90a, its record sm_86 (§26.3 `b`: what the
  inline `WARPSYNC.COLLECTIVE` form costs on an H100 is unmeasured).
- OWED, NOT RE-READABLE HERE: `drain` (sm_86; §26.3 `q`'s re-check on sm_90a is owed, and its experiment lives on the branch
  `ampere`, no source in this tree); `reg_trade`'s two refused forms (compiled once, recorded on [`reg_trade`](reg_trade.md), their
  sources not kept: their 13.5 compile is owed with the next `reg_trade` run).
