# `plan_tool` (`csrc/fsla/src/common/plan/plan_tool.cpp`)

The planner ([`planner.md`](planner.md)) and a kernel's launch function on a synthetic environment: the kernel (`--kernel decode`, `--kernel prefill` or one of the four entmax kernels), the
device's facts and the call's given on the command line, no GPU. For decode ([`decode_launch.md`](decode_launch.md)) it prints the descriptor, the arm and its arguments, the launch's layout (the `lattice:` line, `k m bc owners spans grids`, then `unit:`, `tables:` and `step:` lines: the unit's bits, the tables' sizes, the leaves, rows, unit ceiling, shared-memory bytes and whether they fit the budget) (and the grid, given a
unit ceiling, the batch times heads and a forced count); for the prefill ([`prefill_launch.md`](prefill_launch.md)) the descriptor, or with `--launch json` the launch's host half. For an entmax kernel ([`entmax_launch.md`](entmax_launch.md)) it prints the descriptor, the arm (`lw`, `ipt`, `alpha15` where the kernel has one, `logit`, `value`) and the launch (`klass`, `levels`); `--call` takes several calls, `;` between them, each printed in turn, and `--widths w,...` holds the launch's levels to its one width class (`class: one`, or the refusal). A refusal goes to stderr, exit 1. `//csrc/fsla:plan_tool` is its target;
`tests/unit/test_entmax_matcher.py` drives the entmax kernels with every width from 2, both alphas and bf16 planes, and each refusal (a width of 1, an fp32 plane); `tests/unit/test_planner.py` drives it with devices of other SM counts in range, a count and facts out of range, and calls outside the contract.

    plan_tool --kernel decode --device cc=900,regs_per_sm=65536,smem_per_sm=233472,smem_per_cta_optin=232448,max_threads_per_sm=2048,sm_count=120 \
              --call d_v=48,levels=3,decay=1,batch_heads=1,width_bits_1=4,... [--model <text>] [--grid unit_ceiling=256,bh=64,forced=-1]
    plan_tool --kernel prefill --device ... --call d_v=64,levels=2,decay=0,tokens=4096,batch_heads=1,stateful=0,... [--launch json]
    plan_tool --lattice 64,64

`--launch json` is the prefill's: the launch's host half of a call is printed ([`prefill_launch.md`](prefill_launch.md)) as JSON: `levels`, `width_bits`, `leaves` (a batch-head's), `cta_leaves`, `ctas_per_head`, `window` and `windows`, and `kernels`: `owed` (with the kernels it waits on) while the prefill is
an owed descriptor, `built`, or `outside the contract`. The launch needs no contract check, so it is printed from the call's facts either way; a fact out of its range aborts. A call of one kernel is refused naming a fact only the other states.

`--lattice w1,w2,...` stands alone and needs no device: the lattice law's solve ([`lattice.md`](lattice.md)) of the level widths, printed as the same `lattice:` line (`k`, `m`, `bc`, `owners`, `spans`, `grids`) and a
`constants:` line (`atom_leaves`, `cap_bc`, `max_span`, `max_span_k`). `tests/unit/test_lattice.py` ties its answers to the extension's (`fsla_decode_lattice`) and the atom's leaves it prints to Python's `MMA_K_QUANTUM`.

`--cta-carve w,... --write-density n,... --read-density n,...` stands alone and needs no device: the carve ([`cta_carve.md`](cta_carve.md)) of levels of width `2^w` (log2 values) and each side's densities, one number a level, printed as one line of JSON:
the carve's every field (`write_density` and `read_density` as clamped into 1..256, the derived groups, `carve_bits`, `cta_leaves`, `split_level`, `folds`, `fold_bits`, `fold_leaves`, the sets, `unit_ctas`), then the launch's `ctas_per_head` of those widths
and `units_per_head`, their quotient (the carve holds no count of a head: the tool joins the two). The carve is total, so no density is refused. Each flag takes several calls, `;` between them, one line each.
