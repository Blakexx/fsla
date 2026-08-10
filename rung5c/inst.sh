#!/usr/bin/env bash
# THE ABLATION LADDER'S INSTRUCTION FLOOR: rung 5A and the four stages, arm for
# arm, one instrument.
#
# The cell is B=1 H=2 L=4096: `inst/mma` is a ratio of two exact,
# occupancy-independent counters and does not move with the grid, so the small
# cell is the cheap one to profile and the honest one to quote.
#
# THE LAUNCH IS SELECTED BY KERNEL NAME, not by an index: a launch index counts
# every kernel the process issues, torch's own initialisation included.
#
# A TARGETED METRIC LIST RATHER THAN `--set full`, so the list fits ONE PASS and
# `smsp__inst_executed.sum` comes back exact rather than multiplexed.
#
#   bash rung5c/inst.sh <outdir> [regime]
set -euo pipefail
OUT=${1:?outdir}
REG=${2:-dense}
PY=/home/blake/rola311/bin/python
M=smsp__inst_executed.sum,smsp__inst_executed_pipe_tensor_op_hmma.sum,\
smsp__inst_executed_pipe_alu.sum,smsp__inst_executed_pipe_fma.sum,\
smsp__inst_executed_pipe_lsu.sum,smsp__inst_executed_pipe_cbu.sum,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__issue_active.avg.pct_of_peak_sustained_active,\
gpu__time_duration.sum
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
for a in 16_0_128_64_64 32_0_128_64_64 64_0_128_64_64 32_0_64_64_64 64_0_64_64_64 32_0_32_64_64; do
  for r in r4 a1 a2 a3 a4 a5; do
    nice ncu --metrics "$M" -k regex:'routed_normalized_kernel|compacted_kernel' \
      --launch-count 1 --csv \
      "$PY" rung5c/profile_cell.py "${r}${a}_${REG}" 4096 1 2 \
      >"$OUT/${r}${a}_${REG}.csv" 2>"$OUT/${r}${a}_${REG}.err"
  done
done
echo "captured into $OUT"
