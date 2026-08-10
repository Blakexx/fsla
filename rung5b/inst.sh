#!/usr/bin/env bash
# The instruction floor, rung 5A against rung 5B, arm for arm, one instrument.
#
# The cell is B=1 H=2 L=4096: `inst/mma` is a ratio of two exact,
# occupancy-independent counters and does not move with the grid, so the small
# cell is the cheap one to profile and the honest one to quote.
#
# THE LAUNCH IS SELECTED BY KERNEL NAME, not by an index: a launch index counts
# every kernel the process issues, torch's own initialisation included.
#
# A TARGETED METRIC LIST RATHER THAN `--set full`. p59's note that
# `smsp__inst_executed.sum` comes back MULTIPLEXED is a property of the full
# set's pass scheduling; a list this short fits one pass, so the counter is
# exact and no SASS source page has to be summed to recover it. The tensor-pipe
# utilisation is collected beside it because whether the arm is TENSOR-BOUND is
# what decides how much of an instruction delta reaches the wall.
#
#   bash rung5b/inst.sh <outdir>
set -euo pipefail
OUT=${1:?outdir}
PY=/home/blake/rola311/bin/python
M=smsp__inst_executed.sum,smsp__inst_executed_pipe_tensor_op_hmma.sum,\
smsp__inst_executed_pipe_alu.sum,smsp__inst_executed_pipe_fma.sum,\
smsp__inst_executed_pipe_lsu.sum,smsp__inst_executed_pipe_cbu.sum,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__issue_active.avg.pct_of_peak_sustained_active,\
gpu__time_duration.sum
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
REG=${2:-dense}
for a in 32_0_64_64_64 32_0_32_64_64 16_0_128_64_64 64_0_64_64_64; do
  for r in r4 r5; do
    nice ncu --metrics "$M" -k regex:'routed_normalized_kernel|compacted_kernel' --launch-count 1 --csv \
      "$PY" rung5b/profile_cell.py "${r}${a}_${REG}" 4096 1 2 \
      >"$OUT/${r}${a}_${REG}.csv" 2>"$OUT/${r}${a}_${REG}.err"
  done
done
nice ncu --metrics "$M" -k regex:union_table_kernel --launch-count 1 --csv \
  "$PY" rung5b/profile_cell.py "r5t32_0_64_64_64_${REG}" 4096 1 2 \
  >"$OUT/r5t32_0_64_64_64_${REG}.csv" 2>"$OUT/r5t32_0_64_64_64_${REG}.err"
echo "captured into $OUT"
