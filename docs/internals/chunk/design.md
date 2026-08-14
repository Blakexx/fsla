# `csrc/rola/src/chunk/design.cuh` — the design derivations

The ladder's derivation layer: staging depth from leftover SMEM
(`stages_for`), the CTA target from the register economy (`cta_target`),
residency clauses asserted against the capability row. Every derived quantity
is a pure function of `Caps` + the arm's compile-time geometry — nothing
queries the machine ambiently, which is what makes the census replayable.
