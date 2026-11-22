# Decode's launch function (`csrc/fsla/src/common/plan/decode_launch.h`)

`decode_launch(const Descriptor&)` returns `{DecodeArm arm; DecodeArgs args;}`: the arm to launch and the arguments to launch it with. The two
parts are composed from two functions and stay structurally separate:

- `decode_arms(const Descriptor&)` is the ARM half, written in the rule language ([`rules.md`](rules.md)) so it is evaluable over the
  contract's RANGES. It returns every arm the descriptor can reach: over a point exactly one, over a context no arm covers none. The Hopper
  build (`cc` 900) serves a call's `d_v` with the narrowest arm that holds it (`kDecodeArmWidths`, 32 and 64: `d_v` 1 to 32 reach the 32 arm,
  33 to 64 the 64 arm), one to four `levels`, `decay` 0 or 1: sixteen arms. `arms_gen` ([`../../decode/arms_gen.md`](../../decode/arms_gen.md))
  evaluates it over every context of the contract to collect the arms and to fail the build on a gap.
- `decode_args(const Descriptor&, const DecodeArm&)` is the ARGS half, computed from EXACT values only and never evaluated over a range (a range
  aborts): the compute capability of the build to launch, the CTA's threads (the register file over `kDecodeRegsPerThread`, the one definition
  of the walk's registers a thread: `decode.cuh`'s `DecodePlan` derives its compiled width from it, and the launch refuses a descriptor whose
  threads differ from the compiled plan's), the shared memory a CTA may opt in to (`smem_per_cta_optin`: the launch opts the kernel in above the
  48 KB every launch has), the padding of the call's `d_v` to the arm's (`arm.d_v - d_v`: the value width is a contract range and the arm
  holds it, so the padding is the launch function's, not a second law beside it) and the device's SMs.
- `decode_grid(args, unit_ceiling, bh, forced)` is the grid, from exact values right before the launch: one wave of the declared residency
  (`kDecodeBlocksPerSm` CTAs an SM) at most `bh` heads' unit supply (`ceil(unit_ceiling / kDecodeUnitsPerCta)`, at least one); a gate's forced
  count (at least 1) replaces it, -1 leaves it ([`../../decode/decode.md#the-split-policy`](../../decode/decode.md#the-split-policy)).
- `decode_workspace(const Descriptor&)` is the scratch a call needs beside its operands: none (decode's buffers belong to the sequence and its
  state, never to the call), at the alignment every workspace carries.

THE DESCRIPTOR IS READ HERE ONLY. `Facts` (`Descriptor`) is generated from the schema with private members; `Reader`, its only friend, is the
accessor of the functions of `common/plan/` ([`planner.md#the-reader`](planner.md#the-reader)). A launcher or a kernel holds a `DecodeLaunch`,
never a raw fact.

`decode.cu`'s `fsla_decode_forward` takes the descriptor's words, has the planner check them against the contract and this device, and
launches the arm `decode_launch` names with its arguments; the compiled arms are the build's selection (`arm_switch<DecodeArmSet>`), so an
arm the contract reaches that an iteration build left out is refused by name. The build selection stays beside the launch function because it
answers a different question: which arms THIS binary compiled.
