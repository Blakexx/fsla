# `geom.cuh` — the addressing block

The mirror of `csrc/rola/src/carry/geom.cuh`. One derivation, computed once per launch
on the host, consumed by every kernel of the family.

**Symbols.** `D` levels; `B_l` level `l`'s digit width; `BC` the leaves one owner (one
CTA's box) holds; `MO` the per-level mode word (two bits a level, read then write);
`s(l)` the OWNER span; `g(l) = B_l / s(l)` the owner grid; `NS` a side's stream count;
`sv(l)` the STREAM span (one warp's box); `rank(l)` a level's position in a side's
private order (0 innermost); `shift(l)` the private slot index's bit offset for level
`l`; `kAtomLeaves = 16` the page granule.

## What is compile-time and what is not (K50's axis law)

SHAPE is compile-time because it sizes registers and shared memory: the level count `D`,
the value width `DV`, the window `W`, the batch tile `C`, the CTA's warps and the warps
per write stream. ADDRESSING is runtime because it does not: level orders, spans, grids,
column bases, id weights, the transit's row map. Every runtime value here is CTA-UNIFORM,
so it lives in the uniform datapath and the vector register count does not move — which
is the census G2 gates on.

The consequence is the ship principle: a user of the Python package never registers an
arm. One binary per shape serves every topology and every declaration of its depth.

<a id="span-rule"></a>
## The span rule, and why the atom is a constraint and not an afterthought

K50 (Blake, 2026-08-29): "balance BC over levels; a level narrower than its balanced
share takes its full width and the remainder balances over the others ... Refusal only if
`Π_l B_l < BC`." The same card's amended ownership ruling names the invariant the balance
runs under: **a box is a union of whole atoms**.

Balance ALONE does not keep that invariant. An atom is 16 CONSECUTIVE CANONICAL leaves
and it is the page granule — the state's entry load and exit store address the plane by
canonical atom id, at page grain. At `D = 3, B = 16` pure balance gives `[4, 4, 8]` and at
`D = 4, B = 8` it gives `[2, 4, 4, 4]`; in both, an atom straddles TWO owners, and each of
them stores the whole page. That is measured, not argued: the unconstrained form failed
nine fp64 cells, STATE PLANE only, on BOTH bodies, at exactly those two topologies.

So the rule is balance UNDER the atom invariant:

1. reserve the atom innermost-outward — the levels a 16-leaf granule spans are held WHOLE;
2. balance `log2 BC` over the levels, remainder INNERMOST-FIRST;
3. raise any level to its atom floor and cap it at its own width;
4. take the surplus back from the WIDEST level above its floor, ties to the OUTERMOST, so
   the levels above the atom stay balanced among themselves; give any shortfall to the
   innermost level with room.

This reproduces the compile-time `BoxPlan` it replaces at every built topology, which is
what makes the model test a PARITY PROOF rather than a re-declaration:

| topology | derived | the compile-time law |
|---|---|---|
| `D = 2, B = 256` | `[8, 16]` | `[8, 16]` |
| `D = 2, B = 64` | `[8, 16]` | `[8, 16]` |
| `D = 3, B = 16` | `[2, 4, 16]` | `[2, 4, 16]` |
| `D = 4, B = 8` | `[2, 2, 4, 8]` | `[2, 2, 4, 8]` |

K50's CORRECTION rules the eventual relaxation — "prefill owns whole ROWS ... the fix is
ROW-GRAIN activity masks (16 bits per atom) from the facts pass" — and that machinery is
the page I/O and the facts bit emission. Until it exists, the atom invariant is what the
page-grain store requires, and the unconstrained balance is a follow-on gated on it.

<a id="refusals"></a>
## The refusal set, and what is a census fact instead

Refused at the seam, each by name: a width that is not a power of two; `Π_l B_l < BC`; a
topology that cannot hold one page atom; a packed amplitude row that is not a whole number
of sixteen-byte granules; a level whose column base is not a whole number of the runs read
out of it (the wide loads' alignment); and a level whose `(span, shift)` field is outside
the body's admissible set.

`min_l B_l < 16` is **not** refused. The `B_l >= 16` floor in K50's per-level-width ruling
is guidance about where the geometry is efficient — that ruling's own text says "no refusal
path", and the later span rule states the refusal set as `Π_l B_l < BC` alone. A narrow
level takes its full width and the remainder re-balances. `min_width` is reported as a
census fact so a caller can see it.

<a id="row-map"></a>
## The transit's row map

A publisher's write slot `P` names a lattice digit per level; splitting that digit at the
reader's own span sends its high part to the READER INDEX and its low part to that
reader's own slot. `geom_xch_row` is that composition. The model test proves it is a
bijection of `[0, BC)`, that it is GF(2)-linear in `P`'s bits (which is what lets the
kernel pay one base per warp and a step per fragment), and that it is the IDENTITY exactly
when the two sides' orders and span vectors coincide.

That identity case is the former `kTied`. K50 DELETED the elided body — "one transit,
always; the tied case is a transit whose row map is the identity, same code, same register
plan" — so `identity_map` is a reported FACT that makes a wasted transit visible in the
ledger, and never a second code path.
