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
  STAGES            the ring's depth.  DERIVED from the work one round trip
                    must be covered with and from the shared-memory budget,
                    capped by this kernel's live register set.  It was a free
                    axis; what ended it is that the optimum turned out not to
                    move in DEPTH but in TOKENS IN FLIGHT, which is a
                    derivation from the chunk size rather than a search.
  the CTA target    the second launch bound: the residency this arm can
                    actually reach, shared-memory ceiling CAPPED BY THE
                    REGISTER CEILING.  The cap is not optional -- without it
                    the shared-memory ceiling ran past the register file and
                    three arms spilled (measured).

NOT here, deliberately: the SHARED-MEMORY LAYOUT (pad-8 against an XOR
swizzle) and the CHUNK SIZE.  Both are legal on every tabulated architecture
and both have a measured optimum that MOVES -- the swizzle saved the dense
base 0.545 inst/mma at C=16 and cost the routed kernel 1.260 at the same arm,
a genuine sign flip.  They stay FREE AXES, chosen by measurement, and a
selection here would make every future measurement of them unfalsifiable.

THE ONE OBLIGATION.  A selection that is knowingly suboptimal -- one where the
table says a better implementation exists on this part and the baseline does
not build it -- must ANNOUNCE ITSELF at compile time.  A documented gap a
reader sees is a stated engineering decision; a SILENT slower path is the
forbidden thing.  `Announce` below is that mechanism, and `ops.cuh` wires the
sites, because whether an implementation is built is a fact about operations.
