# Decode's launch function (`csrc/fsla/src/common/plan/decode_launch.h`)

`decode_launch(const Descriptor&)` returns `{DecodeArm arm; DecodeArgs args;}`: the arm to launch and the arguments to launch it with. The two
parts are composed from two functions and stay structurally separate:

- `decode_arms(const Descriptor&)` is the ARM half, written in the rule language ([`rules.md`](rules.md)) so it is evaluable over the
  contract's RANGES. It returns every arm the descriptor can reach: over a point exactly one, over a context no arm covers none. The Hopper
  build (`cc` 900) at `d_v` 32 or 64, one to four `levels`, `decay` 0 or 1: sixteen arms. `arms_gen`
  ([`../../decode/arms_gen.md`](../../decode/arms_gen.md)) evaluates it over every context of the contract to collect the arms and to fail the
  build on a gap.
- `decode_args(const Descriptor&, const DecodeArm&)` is the ARGS half, computed from EXACT values only at run time and never evaluated over a
  range (a range aborts). Today it holds what decode's launcher derives from facts the schema has: the CTA's threads (the register file over
  128 registers a thread, `decode.cuh`'s `kThreads`), the shared memory a CTA may opt in to (the value `smem_budget_of` reads as a stray
  attribute today) and the padding of the call's `d_v` to the arm's. The grid (`grid_of`: SM count and the call's batch times heads) joins it
  with the SM-count fact, when the device probe measures one.

THE DESCRIPTOR IS READ HERE ONLY. `Facts` (`Descriptor`) is generated from the schema with private members; `Reader`, its only friend, is the
accessor of the functions of `common/plan/`. A launcher or a kernel holds a `DecodeLaunch`, never a raw fact.

WHERE THE LAUNCHER WILL CALL IT (the run-time planner, a later step; nothing here changes `select.cu`): `decode.cu`'s `launch_decode_step`
takes `{arm, args}` from `decode_launch(environment)` in place of `arm_switch<DecodeArmSet>({d_v, p.D, decay}, ...)` (its key), `smem_budget_of`,
the `kThreads` read in `decode.cuh` and `grid_of`'s SM-count read; `fsla/ops/decode.py`'s `_shipped_decode_dv` padding becomes `args.pad`.
