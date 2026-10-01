# `csrc/rola/src/produce/produce_api.cuh` — the producer's host entry

## <a id="rola-produce-forward"></a>`rola_produce_forward`

The one launch of the producer (`produce.md`). The caller -- `rola.ops.produce.run_kernel` -- hands it the operands
as the kernel reads them: `x` as `[B * T, hidden]` bf16; `route_W` as `[H, hidden, C]` bf16 with its hidden axis
contiguous (`hidden_contiguous`; the head and column axes read through their strides); `v_W` as `[H * d_v, hidden]`
bf16, a linear's weight as stored; the plan's job table on the device; every output, allocated by the caller so a gate
can poison them -- each side's dense levels as one packed plane `[B, H, T, rows]`, its sparse levels as words `[B, H, rows, nw]` and
terms `[B, H, rows, windows, slot]` (none at one token, `produce.md#sparse`); the arrival counters and the look-back flags (two counters a (job, side,
sequence, window)), zero between calls; the optional logit and level probes
`[jobs, B * T, kMaxColumns]` fp32, which receive every job's projection and every route job's solved levels (read at
`[0, w)`, write at `[w, 2w)`) for their gates and are null in production; and the plan's shape words, a host int16
tensor a job, copied into the launch's parameters. It validates each against the
others -- one hidden width, a multiple of 8 with 16-byte-aligned weights and strides (every hidden run is read in
16-byte chunks), whole heads of value channels, the dense rows head-major `[B, H, T, rows]` u8 or f16 (the kernel splits a flat token
into its batch and step, `produce.md#bundle`), the sparse buffers in the rows' dtype at the call's form and window
count, the fp32 per-level scales and masses, the counters covering every (token tile, head) and every (job, side,
sequence, window), at most 65535 jobs -- and launches token tiles by the job table's rows (a sequence's tiles over more
than one token, the flat batch's at one: `produce.md#jobs`), with the logit rows' dynamic shared memory opted into
once.

## <a id="rola-produce-ledger-bind"></a>`rola_produce_ledger_bind`

Binds the phase ledger the next launches book into (`produce.md#phase-ledger`), a contiguous CUDA int64
`[jobs, 8, 9]`, or unbinds it with `None`. A launch with more jobs than the bound ledger holds is refused. It issues no
launch.
