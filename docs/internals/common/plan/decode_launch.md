# Decode's launch function (`csrc/fsla/src/common/plan/decode_launch.h`)

`decode_launch(const DecodeDescriptor&)` returns `{DecodeArm arm; DecodeArgs args; DecodeLayout layout;}`: the arm to launch, the arguments to launch it with and the layout the step's
parameters carry. The three parts are composed from three functions and stay structurally separate:

- `decode_reach(const DecodeDescriptor&)` is the ARM half, a decision tree in the rule language ([`rules.md`](rules.md)): `cc` 900, then ladders on the value width, `levels`
  and `decay`, so it is evaluable and exact over the contract's RANGES ([`rules.md#exact`](rules.md#exact)): the arms the descriptor can reach, and `gap` when some point
  of it reaches no arm. `decode_arm(const DecodeDescriptor&)` is the arm of a point, the same walk: over a point exactly one leaf (compile-time facts in `rules.h`). The Hopper
  build (`cc` 900) serves a call's `d_v` with the narrowest arm that holds it (`kDecodeArmWidths`, 32 and 64: `d_v` 1 to 32 reach the 32 arm,
  33 to 64 the 64 arm), one to four `levels`, `decay` 0 or 1: sixteen arms. `arms_gen` ([`arms_gen.md`](arms_gen.md))
  walks it over every context of the contract to collect the arms and to fail the build on a gap.
- `decode_args(const DecodeDescriptor&, const DecodeArm&)` is the ARGS half, computed from EXACT values only and never evaluated over a range (a range
  aborts): the compute capability of the build to launch, the CTA's threads (`decode_threads(regs_per_sm)`, the register file over `kDecodeRegsPerThread`, the one
  function of the walk's registers a thread: `decode_args` calls it on the device's register file and `decode.cuh`'s `DecodePlan<B>::kThreads` on the build's, and the launch refuses a descriptor whose
  threads differ from the compiled plan's), the padding of the call's `d_v` to the arm's (`arm.d_v - d_v`: the value width is a contract range and the arm
  holds it, so the padding is the launch function's, not a second law beside it), the device's SMs, and what the call states of its levels:
  each level's width (`widths`, `1 << width_bits_k` for a level the call has, 0 beyond its `levels`), its offset into the flattened per-level rows
  (`row_offsets`, the prefix sums of the widths) and `write_dense` (whether the step deposits whole atoms: DERIVED here, 1 when the innermost level's `write_listed_k` fact is 0).
- `decode_layout(const DecodeArm&, const DecodeArgs&)` is the LAYOUT half, and it is the matcher's because the lattice's law is launch logic: how the level widths fall into the
  box the step walks (`lattice_solve`, [`lattice.md`](lattice.md): C++ owns the lattice) is a distillation of the call's facts into what to launch, which is what a matcher holds
  ("anything that has any kind of logic for that mapping should be in a matcher", the owner, 2026-11-23). So the matcher returns the layout whole and `decode.cu`'s `fill_lattice` copies it into
  the step's parameters and derives nothing; the fields, and what each is the step's parameter of, are `DecodeLayout`'s in the header. The refusals the layout implies are this header's,
  in one set of words: `decode_smem_refusal` (a call that needs more of the step's run-time tile, `DecodeLayout::step_words`, than the kernel's layout holds on the card: a declared capacity refused loudly rather than taken as a slower path) and `decode_paged_refusal` (a
  paged backing under an owner that holds less than one atom); the host throws them, the matcher words them.
- `decode_grid(args, unit_ceiling, bh, forced)` is the grid, from exact values right before the launch: one wave of the declared residency
  (`kDecodeBlocksPerSm` CTAs an SM) at most `bh` heads' unit supply (`ceil(unit_ceiling / kDecodeUnitsPerCta)`, at least one); a gate's forced
  count (at least 1) replaces it, -1 leaves it ([`../../decode/decode.md#the-split-policy`](../../decode/decode.md#the-split-policy)). The host passes the layout's `unit_ceiling`.

EVERY CONTEXT OF DECODE'S CONTRACT IS A DECODE STEP, so the matcher tests no call kind. THERE IS NO DECODE PLAN. The op takes its data (`op(data...)`) and builds the descriptor itself: the descriptor is every device fact and every call fact,
immutable and hashable, and everything derived from those facts (the widths and offsets above, the arm, the threads, the lattice, the layout, the grid) is computed by this one launch
function right before the launch and never stored. The level widths and the write side's walk of each level are CALL FACTS because the caller's declaration decides them
(`fsla.ops.produce._decode_descriptor` states them: a width as its log2 -- a level's width is a power of two, `fsla/_state.py`'s `StateFormat` -- and
the `write_listed_k` of the producer's rule over the declaration; `write_dense` is derived from the innermost level's); a width the contract does not hold is refused when the descriptor is made, naming the fact
([`../../hardware/registry.md`](../../hardware/registry.md#parts)). The state plane's leaf order is no fact of the call: it is the lattice of the call's level widths, which this function solves
(`fsla.ops.decode.lattice_of` reads the same solve through the extension for the oracle and test seam); the
readout's `eps` is data, and the grid is the launch function's own (a gate's forced count is a private keyword of the internal step).

THE DESCRIPTOR IS READ HERE ONLY. `decode::Facts` (`DecodeDescriptor`) is generated from the device table and decode's schema with private members; `Reader`, its only friend, is the
accessor of the functions of `common/plan/` ([`planner.md#the-reader`](planner.md#the-reader)). A launcher or a kernel holds a `DecodeLaunch`,
never a raw fact.

`decode.cu`'s `fsla_decode_forward` takes the descriptor's words (and the readout's `eps` and a gate's `ctas`), has the planner check them
against the contract and this device, and launches the arm `decode_launch` names with its arguments and layout, the level widths, the row offsets, `write_dense` and the lattice's fields
among them. A call outside a narrowed contract is refused by name at that check
([`planner.md`](planner.md)), before any launch; the compiled arms (`arm_switch<DecodeArmSet>`) are by construction exactly the arms the
build's contract reaches, so the launch finds the arm the rules named, and its refusal of an arm the build did not compile is the
backstop for a rule that names one the contract does not reach.
