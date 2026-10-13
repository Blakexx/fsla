# `csrc/rola/src/common/design.cuh` — the design derivations

The ladder's derivation layer: staging depth from leftover SMEM
(`stages_for`), the CTA target from the register economy (`cta_target`),
residency clauses asserted against the capability row. Every derived quantity
is a pure function of `Caps` + the arm's compile-time geometry — nothing
queries the machine ambiently, which is what makes the census replayable.

## Layer 2a -- the derived selections

LAYER 2a -- THE DERIVED SELECTIONS.

WHAT THE DESIGN LOOKS LIKE ON THIS ARCHITECTURE.  Every entry here is
COMPUTED from `arch_caps.cuh`'s facts; none is looked up per architecture,
because a per-arch table of selections is an architecture branch in a costume
and the standing rule is to branch on the flag, never on the number.

The kernel reads THIS layer for its design and calls `ops.cuh` to execute it.
It never reads a capability fact, and it never sees an architecture number.

WHAT IS HERE AND WHAT IS NOT
----------------------------
  STATE_RESIDENCY   where the fold's accumulator lives while it is also the
                    readout's operand.  DERIVED from the residency facts.
  STAGES            a ring's depth.  DERIVED from the shared-memory budget:
                    the shallowest ring that reaches the arm's maximum
                    residency.
  the CTA target    the second launch bound: the residency an arm can
                    actually reach, shared-memory ceiling CAPPED BY THE
                    REGISTER CEILING.  The cap is not optional -- without it
                    the shared-memory ceiling runs past the register file and
                    an arm spills.

NOT here, deliberately: the SHARED-MEMORY LAYOUT.  It is legal on every
tabulated architecture and its measured optimum MOVES between kernels (a
swizzle that saves instructions on one body costs them on another), so it
stays a FREE AXIS, chosen by measurement; a selection here would make every
future measurement of it unfalsifiable.

THE ONE OBLIGATION.  A selection that is knowingly suboptimal -- one where the
table says a better implementation exists on this part and the baseline does
not build it -- must ANNOUNCE ITSELF at compile time.  A documented gap a
reader sees is a stated engineering decision; a SILENT slower path is the
forbidden thing.

<a id="announce"></a>**`Announce`** is that mechanism: a compile-time diagnostic that does NOT fail the build.
Instantiating `Announce<true>` calls a deprecated constructor, which every supported toolchain reports with the
message attached. A gap that would produce WRONG or UNBUILDABLE code is a `static_assert` instead, and reads NOT
BUILT. `ops.cuh` wires the sites, because whether an implementation is built is a fact about operations.

**Consumers today:** `ops.cuh`'s consistency asserts read the residency selection and announce the unbuilt bulk copy,
and the carry's build stamp reads `kCapsDigest`. No shipped kernel calls the ring-depth or CTA-target derivations below.

## <a id="note-l36"></a>Where the state lives

The algorithm's requirement is not an instruction: it is that THE FOLD'S ACCUMULATOR IS REUSABLE AS THE READOUT'S
OPERAND WITHOUT A MEMORY ROUND TRIP. Ampere and Hopper satisfy it in the register file (Hopper via the `_RS` atom
family); a `tcgen05` part satisfies it in tensor memory, with no register-resident state at all. Naming the requirement
after the Ampere instruction would REFUSE that part for having the wrong storage class while meeting the requirement.

- <a id="satisfiableresidencies"></a>`satisfiable_residencies` is the mask of residencies that meet the requirement on
  this architecture. SHARED never appears: a state that goes through shared memory HAS taken the round trip, which is
  a different algorithm.
- <a id="near-line-51"></a>The preference order puts TMEM first where it exists, because the accumulator is already
  there and the register file stays free.
- <a id="kbuiltresidencies"></a>`kBuiltResidencies` is the one statement layer 2a needs from layer 2b, a statement
  about OPERATIONS: `ops.cuh`'s consistency section asserts that every residency named there is served by an operation
  it implements, so the two cannot drift. REGISTERS ONLY: a TMEM path is not built and a SHARED path never will be.
  Selecting an unbuilt residency is a compile error reading NOT BUILT -- the architecture is admissible in structure
  and the path is unwritten -- and a speculative path for hardware we do not have would be an operation invented
  before it has a caller.
- <a id="selectresidency"></a>`select_residency` is the best residency that both meets the requirement and has an
  implementation; it returns SHARED to mean "none", the value that can never be selected.
- <a id="residencysuboptimal"></a><a id="kresidencysuboptimal"></a>`residency_suboptimal` is true when a better
  residency meets the requirement on this part and is not built; `kResidencySuboptimal` ANNOUNCES it rather than
  refusing, since the built path is correct there, merely not the best available.

## <a id="kspecializationpays"></a>Whether a unit may be dedicated to production

`kSpecializationPays` is derived, and the derivation is about REGISTERS rather than warps. Allocation is uniform per
thread across a kernel and sized by the MAXIMUM live set over a thread's paths, so a unit that produces and then
consumes reuses the producer's registers once they are dead -- it pays `max(produce, consume)`. Dedicating a unit to
production pays the same `max()`, because a producer unit cannot be given a small budget while a consumer unit keeps a
large one: specialization buys nothing and costs the unit's threads, which in a latency-bound regime are the hiding
mechanism. What changes that is a run-time reallocation instruction (`dyn_reg_realloc`), which is why specialization
is a derivation rather than a preference.

<a id="kbuiltspecialization"></a>`kBuiltSpecialization` is NOT BUILT, announced rather than refused: where the
reallocation exists a specialized producer would be the better implementation; the unspecialized path is correct
everywhere it compiles, each unit running its own production pipelined ahead of its own consumption.

## <a id="kstagesmin"></a>The ring's depth and the CTA target

One clause: TAKE THE SHALLOWEST RING THAT REACHES THE ARM'S MAXIMUM RESIDENCY. Depth is bought, never taken. Depth
does not move the optimum (measured across depths 2 to 6: every arm within 1.2%, the shallowest best or tied on eleven
of twelve arm-cell pairs), and a ring deeper than residency needs is not free: shared memory is carved out of the
unified L1, so it pays for reads the L1 would otherwise absorb (+3.2% and +5.8% for the deepest affordable ring at one
resident CTA). In the other direction the clause pays: a shallower ring can fit a second CTA.

- <a id="note-l217"></a>The residency reachable at ANY depth is the shallowest legal ring's: the ring can only shrink
  it. An arm whose shallowest ring does not fit is refused by `smem_fits` at its own site, never silently deepened.
- <a id="ctasbyregfloor"></a>`ctas_by_reg_floor` inverts the per-lane register count: the most CTAs whose budget
  still holds a live set of `reg_floor`.
- <a id="ceilingwithoutsmem"></a>`ceiling_without_smem` is the residency ceiling that is not a function of the arm's
  shared memory: registers, threads and the hard CTA cap.
- <a id="ctatarget"></a>`cta_target` is THE SECOND LAUNCH BOUND, the CTA count the arm can reach: shared memory,
  registers, threads and the CTA cap, whichever binds first.
- <a id="stagesbyresidency"></a>`stages_by_residency` searches upward from `kStagesMin` for the shallowest ring that
  reaches `want` CTAs. On every tabulated architecture it evaluates to `kStagesMin`, because the residency ceiling
  only falls as the ring deepens; it is written in the general form so it states the rule rather than its answer.
- <a id="stagesfor"></a>`stages_for` gives the compiling target's answers, so a kernel writes a geometry and never a
  capability row.
- <a id="smemfits"></a>`smem_fits` is THE ONE HARD RESOURCE REFUSAL, from the table rather than a literal: an arm fits
  the architecture's per-CTA shared-memory maximum or is refused. No arm is silently shrunk to fit.
- <a id="kcapsdigest"></a>`kCapsDigest` is the digest of the capability row the device code was compiled against,
  carried here so a kernel's build stamp names no layer-1 symbol.
