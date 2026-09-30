# `csrc/rola/src/produce/produce_api.cuh` — the producer's host entry

## <a id="rola-produce-forward"></a>`rola_produce_forward`

The one launch of the producer (`produce.md`). The caller -- `rola.ops.produce.run_kernel` -- hands it the operands
as the kernel reads them: `x` as `[B * T, hidden]` bf16; `route_W` as `[H, hidden, C]` bf16 with its columns
contiguous (either layout of the head and hidden axes, read through their strides); `v_W` as `[H * d_v, hidden]` bf16
with its channels contiguous (`channel_major`); the plan's job table on the device; every output, allocated by the
caller so a gate can poison them; and the arrival counters, zero between calls. It validates each against the others
-- one hidden width, whole heads of value channels, the rows `[..., H, rows]` u8 or f16 on both sides, the fp32
per-level scales and masses, the counters covering every (token tile, head) -- and launches the job table's rows by
token tiles.
