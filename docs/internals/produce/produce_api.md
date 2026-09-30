# `csrc/rola/src/produce/produce_api.cuh` — the producer's host entry

## <a id="rola-produce-forward"></a>`rola_produce_forward`

The one launch of the producer (`produce.md`). The caller -- `rola.ops.produce.run_kernel` -- hands it the operands
as the kernel reads them: `x` as `[B * T, hidden]` bf16; `route_W` as `[H, hidden, C]` bf16 with its hidden axis
contiguous (`hidden_contiguous`; the head and column axes read through their strides); `v_W` as `[H * d_v, hidden]`
bf16, a linear's weight as stored; the plan's job table on the device; every output, allocated by the caller so a gate
can poison them; the arrival counters, zero between calls; and the optional logit probe `[jobs, B * T, kMaxColumns]`
fp32, which receives every job's projection for its gate and is null in production. It validates each against the
others -- one hidden width, a multiple of 8 with 16-byte-aligned weights and strides (every hidden run is read in
16-byte chunks), whole heads of value channels, the rows `[..., H, rows]` u8 or f16 on both sides, the fp32 per-level
scales and masses, the counters covering every (token tile, head), at most 65535 jobs -- and launches token tiles by
the job table's rows, with the logit rows' dynamic shared memory opted into once.
