# `smem_rates`

Mirrors `measure/lab/experiments/smem_rates.cu`, an experiment of the lab
([`docs/internals/tools/lab.md`](../internals/tools/lab.md)).

What an SM's shared-memory pipe sustains and which global traffic takes its cycles, for the hardware configuration
(`common/hardware.json`): one CTA an SM, 8 or 16 warps, each issuing independent operations at addresses that change
every iteration (volatile asm; an address that repeats each iteration lets ptxas hoist a plain `ld.shared` out of the
loop -- the first build did, and timed a loop without its loads). The block's span over the wavefronts the SM issued,
the wavefronts confirmed by ncu (`l1tex__data_pipe_lsu_wavefronts_mem_shared`). sm_86, cycles a wavefront (8 warps;
16 the same):

| operation | wavefronts an instruction | cycles a wavefront | cycles an instruction |
|---|---|---|---|
| `ld.shared.v4` (LDS.128) | 4 | 1.01 | 4.0 |
| `ldmatrix.x4` (LDSM) | 4 | 1.00 | 4.0 |
| `red.shared.add.u32` (ATOMS.ADD), a lane a bank | 1 | 2.00 | 2.0 |
| the same, two lanes a bank | 2 | 1.00 | 2.0 |
| the same, four lanes a bank | 4 | 1.00 | 4.0 |

Loads run at the documented bank rate, a wavefront a cycle. A shared atomic takes the larger of two cycles an
instruction and a cycle a wavefront: a two-way conflict between the warp's halves is free.

WHAT GLOBAL TRAFFIC TAKES FROM THE PIPE: 16 warps, 8 issuing `ldmatrix.x4` (alone 1.011 cycles a wavefront) while 8
stream global data, 512 bytes a warp-instruction, in one form -- from a 16 KB table (L1 hits) or a 4 MB one (L2 hits)
-- the stream's own rate beside them, and the pipe cycles it took: `1 - 1.011 / (ldmatrix's cycles a wavefront)` over
the stream's 128-byte units a cycle. Bank reads and writes a 128 bytes streamed are ncu's (`l1tex__data_bank_reads`,
`_writes`, the stream alone; 32 a full wavefront).

| stream | alone, B/cycle | beside, B/cycle | ldmatrix beside, cycles/wf | pipe cycles/128 B | bank R/W per 128 B |
|---|---|---|---|---|---|
| `ld.global.nc.v4` (`__ldg`), L1 hits | 119.8 | 53.7 | 2.028 | 1.19 | 32 / 0 |
| `ld.global.nc.v4`, L2 hits (allocated in L1) | 21.3 | 16.3 | 1.304 | 1.76 | 32 / 25 |
| `ld.global.cg.v4`, L2 hits | 15.5 | 16.3 | 1.273 | 1.62 | 32 / 32 |
| `cp.async.cg` into shared memory, L2 hits | 18.0 | 17.9 | 1.087 | 0.50 | 0 / 32 |
| `cp.async.ca` into shared memory, L2 hits | 20.9 | 18.4 | 4.669 | 5.44 | 35 / 58 |

What it settles:

- A load from L2 into registers passes through the shared-memory banks: its data is written into them and read out (32
  and 32 a 128 bytes), ~1.6 pipe cycles a 128 bytes -- as much as a `cp.async.cg` copy (0.5, its write) and the
  `ldmatrix` that reads it (1.0). Loading straight into MMA fragments saves no pipe cycles over staging; it saves the
  copier's instructions and the ring, and holds registers for the loads in flight.
- `cp.async.cg` is the cheapest way in: its write into shared memory takes half a pipe cycle a 128 bytes.
- `cp.async.ca` fills L1, reads the line back and writes shared memory, and holds the pipe beyond its bank passes: 5.4
  cycles a 128 bytes, the `ldmatrix` beside it 4.6x slower. Never where the pipe binds.

Records: `measure/lab/results/smem_rates/sm_86.json`, `sm_86.counters.json`.
