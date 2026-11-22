# `tools/overlap_gate.py` — a kernel's time against the largest of its floors

KERNEL_STANDARDS §28. A ROW is a kernel on a perf cell (`ROWS`: the kernel, the launch that calls it, the cell, its
heads, its MMA kind if it issues any). One `ncu` launch of the row, after `warmup` launches of the same call, reads:
the elapsed time and SM clock, instructions executed, shared wavefronts, HMMA, IMMA and wgmma instructions, and DRAM's share
of its sustained peak. The FLOORS, each a time:
- issue: instructions over the SMs' schedulers at one a clock (`common/hardware.json`: `schedulers`, `issue`);
- tensor: the row's `mma.sync` kind's instructions at the product's calibrated cycles each, over the SMs' schedulers --
  never the profiler's pipe ratio (§22); a `wgmma` is counted apart (`sm__inst_executed_pipe_tensor_op_gmma`) and
  `hardware.json` declares no rate for it, so a row that issues one is REFUSED BY NAME (OWED: an H100 lab measures the rate);
- shared: wavefronts at the product's `wavefront` cycles each, over the SMs;
- DRAM: the bytes at the profiler's sustained DRAM peak, `dram__throughput` x elapsed.

The ratio is elapsed / max(floors), red above `LINE` = 1.15. A kernel at its binding floor reads near 1.0; a kernel
whose phases take turns, one idling a stream while another runs, reads near the sum of its floors over their max.

    bazel run //tools:overlap_gate                 # every row; exit 1 when any is red
    bazel run //tools:overlap_gate -- --row ROW

THE INSTRUMENT'S CHECK is owed on an H100: the sm_86 probes (a pure read stream at 1.04x its DRAM floor; torch's cuBLAS fp16
GEMM, priced by `mma.sync` shape) are the branch `ampere`'s, and the GEMM probe row and its shape-cost table are deleted --
a Hopper GEMM issues `wgmma`, which no rate prices. Its Hopper probe rows land with the first measured `wgmma` rate.

THE ROWS: none on this line. The two rows it had (`facts-alt-k4`, `facts-dense`) were the facts pass's, which is deleted (`docs/internals/DELETIONS.md`); with no row the gate is red, never vacuously green, and a kernel with a schedule adds its row and its launch with it.

`tools/lint/coverage.py` reads `ROWS` by AST: every kernel's `overlap` obligation in `tools/coverage.json` names a row
gating it, or is exempt or owed with its reason.
