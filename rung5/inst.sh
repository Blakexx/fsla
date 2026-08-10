#!/usr/bin/env bash
# The instruction floor, rung 4 against rung 5A, arm for arm, one instrument.
#
# The cell is B=1 H=2 L=4096: `inst/mma` is a ratio of two exact,
# occupancy-independent counters and does not move with the grid, so the small
# cell is the cheap one to profile and the honest one to quote.
#
#   bash rung5/inst.sh <outdir>
set -euo pipefail
OUT=${1:?outdir}
PY=/home/blake/rola311/bin/python
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
ARMS="16_0_64_64_64 32_0_64_64_64 64_0_64_64_64 32_0_32_64_64 64_0_32_64_64 32_0_128_64_64"
for a in $ARMS; do
  for r in r4 r5; do
    rep="$OUT/${r}${a}_dense"
    [ -f "$rep.ncu-rep" ] && continue
    nice ncu --set full -f -o "$rep" \
      "$PY" rung5/profile_cell.py "${r}${a}_dense" 4096 1 2 >"$rep.log" 2>&1
  done
done
rep="$OUT/r5t32_0_64_64_64_dense"
[ -f "$rep.ncu-rep" ] || nice ncu --set full -f -o "$rep" \
  "$PY" rung5/profile_cell.py "r5t32_0_64_64_64_dense" 4096 1 2 >"$rep.log" 2>&1
echo "captured into $OUT"
