# The lab's experiments

One page per experiment, mirroring `measure/lab/experiments/<name>.cu`: the question it asks of the machine, its
method, its record (`measure/lab/results/<name>/<sm_XX>.json`) and what it settled. How an experiment is written, run
and recorded, and what guards a measurement, is [`docs/internals/tools/lab.md`](../internals/tools/lab.md); a law
about the machine cites the experiment it rests on ([`docs/KERNEL_STANDARDS.md#s26`](../KERNEL_STANDARDS.md#s26)).

| experiment | kind |
|---|---|
| [`tensor_tax`](tensor_tax.md) | experiment |
| [`collective_branch`](collective_branch.md) | experiment |
| [`smem_rates`](smem_rates.md) | experiment |
| [`row_span`](row_span.md) | experiment |
| [`vote_stash`](vote_stash.md) | experiment |
| [`facts_block`](facts_block.md) | experiment |
| [`facts_two`](facts_two.md) | experiment |
| [`issue_latency`](issue_latency.md) | experiment |
| [`wgmma_regs`](wgmma_regs.md) | experiment |
| [`bulk_drain`](bulk_drain.md) | experiment |
| [`reg_trade`](reg_trade.md) | experiment |
| [`reg_five`](reg_five.md) | experiment |
| [`row_ring`](row_ring.md) | experiment |
| [`wgmma_rate`](wgmma_rate.md) | experiment |
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
| [`gather_pass`](gather_pass.md) | experiment |
| [`cast_cluster`](cast_cluster.md) | experiment |
| [`value_gather`](value_gather.md) | experiment |
| [`readout_role`](readout_role.md) | experiment |
| [`value_cast`](value_cast.md) | experiment |
| [`pool_fill`](pool_fill.md) | experiment |
| [`load_rate`](load_rate.md) | experiment |
| [`carry_window`](carry_window.md) | experiment |
| [`add_rate`](add_rate.md) | experiment |
| [`slot_sum`](slot_sum.md) | experiment |
| [`drain`](drain.md) | experiment |
