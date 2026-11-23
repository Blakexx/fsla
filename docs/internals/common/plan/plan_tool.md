# `plan_tool` (`csrc/fsla/src/common/plan/plan_tool.cpp`)

The planner ([`planner.md`](planner.md)) and decode's launch function ([`decode_launch.md`](decode_launch.md)) on a synthetic environment: the
device's facts and the call's given on the command line, no GPU. It prints the descriptor, the arm and its arguments (and the grid, given a
unit ceiling, the batch times heads and a forced count), or the refusal on stderr and exit 1. `//csrc/fsla:plan_tool` is its target;
`tests/unit/test_planner.py` drives it with devices of other SM counts in range, a count and facts out of range, and calls outside the contract.

    plan_tool --device cc=900,regs_per_sm=65536,smem_per_sm=233472,smem_per_cta_optin=232448,max_threads_per_sm=2048,sm_count=120 \
              --call d_v=48,levels=3,decay=1 [--model <text>] [--grid unit_ceiling=256,bh=64,forced=-1] [--carve json]

With `--carve json` a prefill call's carve is printed ([`prefill_launch.md`](prefill_launch.md)) as JSON instead of the decode launch: `carved` false for a call that has none, else the box bits,
each side's list bits and list level, the pair and split levels, the fold (`fold_lists`, `fold_columns`, `fold_outer`), the counts, `window` and `windows`, and `kernels`: `owed` (with the kernels it waits on) while the prefill is
an owed descriptor, `built`, or `outside the contract`. The carve needs no contract check, so it is printed from the call's facts either way; a fact out of its range aborts.
