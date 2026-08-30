# `geom.cuh` — the addressing block

The mirror of `csrc/rola/src/carry/geom.cuh`. One derivation, computed once per launch
on the host, consumed by every kernel of the family.

**Symbols.** `D` levels; `B_l` level `l`'s digit width; `BC` the leaves one owner (one
CTA's box) holds; `MO` the per-level mode word (two bits a level, read then write);
`s(l)` the OWNER span; `g(l) = B_l / s(l)` the owner grid; `NS` a side's stream count;
`sv(l)` the WARP SUB-BOX's span; `rank(l)` a level's position in a side's private order
(0 innermost); `shift(l)` the slot index's bit offset for level `l`; `kAtomLeaves = 16`
the page granule.

## What is compile-time and what is not (K50's axis law)

SHAPE is compile-time because it sizes registers and shared memory. G2 read that too
narrowly and G2b corrects it (K50's "PROPOSED CORRECTION TO R12", ruled 2026-08-29): the
warp's leaf-slot → digit map is baked into the unrolled expansion, so **the sub-box's
spans and slot layout ARE register shape**, exactly like leaves-per-warp and `DV`. Made
runtime they force either a dynamically indexed local array (a spill) or a case
enumeration — G2 measured the second at 2x instructions, 5.9x I-cache stalls and a 63 →
190 KB kernel.

So: compile-time is `D`, `DV`, `W`, `C`, the CTA's warps, and the two SPAN LAWS below
(the warp sub-box from `(D, DV)`, the owner box from `(D, BC)`). Runtime is the widths
`B_l` — strides, column bases, id weights, the owner grid — and the per-side ORDER, which
reaches the kernel only as CARVE OFFSETS: which stream index bit fixes which level's
digits. Every runtime value here is CTA-UNIFORM, so it lives in the uniform datapath and
the vector register count does not move — which is the census this gates on.

The consequence is the ship principle: a user of the Python package never registers an
arm. One binary per shape serves every topology and every declaration of its depth.

<a id="arch-constants"></a>
## The arch's primary constants, in derivation order

1. `leaves_per_sm(dv)` — what the SM's STATE CAPACITY admits at this value width (the
   register file on sm_80/sm_90, TMEM on sm_100).
2. `warps_per_sm` — the sub-box owners are WARPS on every architecture.
3. The SM box is `leaves_per_sm` balanced over `D`; the carve divides it, in the side's
   own order, into `warps_per_sm` SEGMENTS.
4. `leaves_per_warp = leaves_per_sm / warps_per_sm` is the warp's SEGMENT, derived.
5. The COMPILED SHAPE is the MMA ACCUMULATOR TILE's leaf layout — the slot → digit map
   that fixes which leaf row of the tile each expanded coefficient (the panel row) lands
   on. On sm_80/sm_90 that tile is the warp's register fragment (32 leaves at `DV = 64`);
   on sm_100 it is the TMEM tile the MMA addresses in place (`tcgen05.mma` accumulates
   into TMEM and reads A from TMEM, so the state never passes through registers to
   compute). The map is the same object either way.
6. The warp's SEGMENT is its tiles, and TILES PER SEGMENT is derived: it is `1` here, one
   tile being the whole segment. A warp that walks several tiles does so as a runtime trip
   count over identical tile bodies — not a shape, and not in this table.

The SHAPE COUNT is flat across architectures: it depends on `D` and on how many bits the
carve takes (`log2 warps_per_sm`), not on the capacity. What scales with
`leaves_per_warp` is each generated body's SIZE.

<a id="warp-box-set"></a>
## The generated set of sub-box shapes, and why it is a SET

The carve is TOP-DOWN (Blake, 2026-08-29): the box's spans are divided level by level in
the SIDE'S OWN order — its sparse levels outermost — until the box count reaches the
warp count; what remains is that side's sub-box. `[4,4,4,4]` at eight warps goes
`[1,4,4,4]` (four boxes) then `[1,2,4,4]` (eight), and `[1,2,4,4]` is the shape.

Because the result depends on the order, and the order comes from the MODE WORD, there is
not one shape but a SET: `sub_boxes(D, box_leaves, warps)` runs the carve over EVERY
level order and de-duplicates. It is a generated constexpr table, never hand-written, and
`sb_index` is the question "is this carved shape a member" asked of the generator itself.
At `D = 2` the set has two members, at `D = 3` four, at `D = 4` seven.

Each side selects its member at launch from its own order; the two sides therefore no
longer share a slot layout, which is what restores the two-sided carve's liveness
selectivity (the `## G2b` finding) and what makes the transit's row map a per-side
scatter rather than a per-warp base.

**The grain this branch compiles.** `warps_per_sm` is eight — one CTA per SM, R15 — but
these arms compile `kWarps = 4`, so the CTA box is HALF the SM box (`BC = 128` of the SM's
256 leaves) and the set is the same law run on the half. The two sets are NOT the same
shapes: at `D = 2` the SM box `[16,16]` yields `[2,16]` and `[16,2]`, while the half box
`[8,16]` yields `[2,16]` and `[8,4]` — and `[16,2]` does not fit a box that is 8 wide at
level 0. A four-warp CTA cannot be a contiguous subset of the SM's eight sub-boxes with
BOTH sides covering the same leaves, because the two sides carve different levels. The set
is generated from the box the CTA actually owns for exactly that reason.

<a id="warp-box"></a>
## The warp sub-box, and the canonical slot layout

`leaves_per_warp(DV) = 2048 / DV` is the register law: 32 leaves of state at `DV = 64`,
16 at `DV = 128`. Its `log2` is split BALANCED over the `D` levels, remainder
innermost-first — `warp_span_bits` — and that is the whole of `sv(l)`. At `DV = 64`:

| `D` | `sv` | `shift` |
|---|---|---|
| 1 | `[32]` | `[0]` |
| 2 | `[4, 8]` | `[3, 0]` |
| 3 | `[2, 4, 4]` | `[4, 2, 0]` |
| 4 | `[2, 2, 2, 4]` | `[4, 3, 2, 0]` |

<a id="canonical-slots"></a>
The slot layout is CANONICAL and the SAME ON BOTH SIDES: a slot index is the sub-box's
digits in canonical order, innermost lowest, so `shift(l) = Σ_{l' > l} log2 sv(l')`. The
per-side order therefore decides only which stream holds which sub-box — the carve
offsets — and never how a warp lays its own leaves out. Three things follow, and they are
why the correction is worth its stage:

* the produce's span and shift are IMMEDIATES on both sides, so the expansion is the
  parent's code with the parent's instruction count;
* the transit's row map is a permutation of leaves BETWEEN warps with the slot bits
  FIXED (see below), so its addresses are immediates plus one base word per warp;
* `BC = leaves-per-warp × kWarps` is DERIVED, never an axis: the carve stacks warp
  sub-boxes along the levels where the owner box is wider than the sub-box.

<a id="carve"></a>
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
| `D = 4, B = 16` | `[2, 2, 2, 16]` | — (the parent's `B = 8` topology is unlawful under the floor) |

Under K50's restored floor (`B_l >= 16`, power of two) no level width can bind the
balance, so this law is compile-time `f(D, BC)` — which is what lets the channel-split
body's slice map be immediates too. A level narrower than its span is REFUSED, by name;
there is no narrower slot layout to fall back to.

K50's CORRECTION rules the eventual relaxation — "prefill owns whole ROWS ... the fix is
ROW-GRAIN activity masks (16 bits per atom) from the facts pass" — and that machinery is
the page I/O and the facts bit emission. Until it exists, the atom invariant is what the
page-grain store requires, and the unconstrained balance is a follow-on gated on it.

<a id="refusals"></a>
## The refusal set, and what is a census fact instead

Refused at the seam, each by name: a width that is not a power of two; **a level
narrower than the owner box's span there** (which subsumes `Π_l B_l < BC` and names the
level that cannot hold its share); a value width that is not a multiple of sixteen; a
packed amplitude row that is not a whole number of sixteen-byte granules; a level whose
column base is not a whole number of the runs read out of it (the wide loads' alignment);
and a stream count the grid cannot partition.

`min_l B_l` is REPORTED, not refused, and it stays a census fact: the width law above is
the one the shape actually owns, and under K50's floor (`B_l >= 16`) it never fires for a
lawful state. The one topology it removed at G2b is `D = 4, B = 8`, whose conformance
cells moved to the lawful `(16, 16, 16, 16)`.

<a id="row-map"></a>
## The transit's row map

A publisher's write slot `P` names a lattice digit per level; splitting that digit at the
reader's own span sends its high part to the READER INDEX and its low part to that
reader's own slot. Under the canonical layout the two sides tile the same box with the
SAME sub-box, so the low part is `P` itself: the map is

    row(w, P) = reader(w) * leaves-per-warp + P

with `reader(w) = Σ_l ((w >> w.grid_div_bits[l]) & (grid[l]-1)) << r.grid_div_bits[l]`.
`geom_xch_reader` is that base word — the only runtime part — and the slot bits are the
identity. The model test still COMPOSES the two sides' own derivations rather than
assuming this, then asserts the composition is a bijection of `[0, BC)`, that every slot
keeps its offset, and that each publisher's base is warp-aligned.

Because every row a single access touches is now an aligned run apart, and
`LdX / 2 = 4 × odd`, the row stride alone puts eight rows in eight different four-bank
blocks: **the region needs no swizzle**. `geom_swz`/`geom_msw` are deleted with G2b, and
with them the deep-3 return-pull bank conflict (`KNOWN_RETURN_PULL_CONFLICTS`) — the
conflict is gone, not tolerated. The proof is a compile-time enumeration over all four
accesses at every declaration, in `tests/unit/test_exchange_layout.py`.

That identity case is the former `kTied`. K50 DELETED the elided body — "one transit,
always; the tied case is a transit whose row map is the identity, same code, same register
plan" — so `identity_map` is a reported FACT that makes a wasted transit visible in the
ledger, and never a second code path.
