# `smem_rates`

Mirrors `measure/lab/experiments/smem_rates.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What an SM's shared-memory pipe sustains, for the hardware configuration (`common/hardware.json`): one CTA an SM, 8
or 16 warps, each issuing
independent operations at addresses that change every iteration (volatile asm; an address that repeats each iteration
lets ptxas hoist a plain `ld.shared` out of the loop -- the first build did, and timed a loop without its loads). The
block's span over the wavefronts the SM issued, the wavefronts confirmed by ncu
(`l1tex__data_pipe_lsu_wavefronts_mem_shared`). sm_86, cycles a wavefront (8 warps; 16 the same):

| operation | wavefronts an instruction | cycles a wavefront | cycles an instruction |
|---|---|---|---|
| `ld.shared.v4` (LDS.128) | 4 | 1.01 | 4.0 |
| `ldmatrix.x4` (LDSM) | 4 | 1.00 | 4.0 |
| `red.shared.add.u32` (ATOMS.ADD), a lane a bank | 1 | 2.00 | 2.0 |
| the same, two lanes a bank | 2 | 1.00 | 2.0 |
| the same, four lanes a bank | 4 | 1.00 | 4.0 |

Loads run at the documented bank rate, a wavefront a cycle. A shared atomic takes the larger of two cycles an
instruction and a cycle a wavefront: a two-way conflict between the warp's halves is free. Records:
`measure/lab/results/smem_rates/sm_86.json`, `sm_86.counters.json`.
