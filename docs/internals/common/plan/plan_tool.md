# `plan_tool` (`csrc/fsla/src/common/plan/plan_tool.cpp`)

The planner ([`planner.md`](planner.md)) and a kernel's launch function on a synthetic environment: the kernel (`--kernel decode` or `--kernel prefill`), the
device's facts and the call's given on the command line, no GPU. For decode ([`decode_launch.md`](decode_launch.md)) it prints the descriptor, the arm and its arguments (and the grid, given a
unit ceiling, the batch times heads and a forced count); for the prefill ([`prefill_launch.md`](prefill_launch.md)) the descriptor, or with `--carve json` the carve. A refusal goes to stderr, exit 1. `//csrc/fsla:plan_tool` is its target;
`tests/unit/test_planner.py` drives it with devices of other SM counts in range, a count and facts out of range, and calls outside the contract.

    plan_tool --kernel decode --device cc=900,regs_per_sm=65536,smem_per_sm=233472,smem_per_cta_optin=232448,max_threads_per_sm=2048,sm_count=120 \
              --call d_v=48,levels=3,decay=1,batch_heads=1,width_bits_1=4,... [--model <text>] [--grid unit_ceiling=256,bh=64,forced=-1]
    plan_tool --kernel prefill --device ... --call d_v=64,levels=2,decay=0,tokens=4096,batch_heads=1,stateful=0,... [--carve json]

`--carve json` is the prefill's: the carve of a call is printed ([`prefill_launch.md`](prefill_launch.md)) as JSON: `carved` false for a call that has none, else the box bits,
each side's list bits and list level, the pair and split levels, the fold (`fold_lists`, `fold_columns`, `fold_outer`), the counts, `window` and `windows`, and `kernels`: `owed` (with the kernels it waits on) while the prefill is
an owed descriptor, `built`, or `outside the contract`. The carve needs no contract check, so it is printed from the call's facts either way; a fact out of its range aborts. A call of one kernel is refused naming a fact only the other states.
