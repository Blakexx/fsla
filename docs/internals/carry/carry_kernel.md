# `carry/carry_kernel.cuh` — the carry body's frame

Mirrors `csrc/rola/src/carry/carry_kernel.cuh`, `carry/box.cuh` and `carry/params.cuh`.
The shared-memory tenants have their own page, `smem_ledger.md`; the host side has
`carry.md`.

Symbols, each at first use. `D` = routing depth (levels). `B_l` = level `l`'s padded digit
count. `N = prod_l B_l` = leaf capacity. `DV` = the padded value width. `BC` = the CTA
box's leaf count. `W` = the window, 512 tokens. `M`/`N_mma`/`K` = the MMA's 16/8/16.
`kSeg` = a segment, 16 tokens. The three shared strides — the transit region's, the state
page image's and the V tile's — are named and sized in
[`smem_ledger.md`](smem_ledger.md).

## The frame, in one picture

```
prologue: the parameter block -> uniform registers; the box's two call-level bits ->
          die fast; the page slots resolved once; the per-side warp sub-box selected
          once from the generated member set; the state slice loaded
for each window:
  [out leg]    publish hi(state) -> COVER SLOT A -> edge -> pull as read-side A fragments
  PHASE 1:     per read batch [ stage(b+1) | produce(b+1) | mma(b) | fan-in(b-1) ]
  [return leg] store back lo(state) -> COVER SLOT B -> edge -> rejoin
  PHASE 2:     per write batch [ stage(b+1) | expand(b+1) | fold-mma(b) ]
  [window line]
exit: the state slice stored, iff this call writes it
```

Every bracketed slot is a NAMED function in the source, whether or not it has a body yet.
The frame is what a schedule is built into; a stage that arrived later and had to invent
its own edges would be a schedule retrofitted, which is the shape this arrangement exists
to prevent.

<a id="edges"></a>
## The three edges, and why there are exactly three

`Edge::kOutLeg`, `Edge::kReturnLeg`, `Edge::kWindowLine`. Each is one `mbarrier` word, and
each is used SPLIT: a unit arrives when its own contribution is published, runs a cover
slot that depends on nobody, and waits only when it needs what the others published.

* the OUT LEG separates publishing the accumulator's high halves into the transit region
  from reading them back in the other side's carve;
* the RETURN LEG separates writing the low halves back out of the region from reusing the
  region's bytes as PHASE 2's segment buffers;
* the WINDOW LINE separates a window's fold from the next window's publish — the fold
  writes the accumulator that the publish reads.

Nothing else in the loop is CTA-wide. A segment's producer/consumer handshake belongs to
its STREAM and is a warp-local wait (`cp.async` groups today) or, where the producer and
the consumer are different warps, that stream's own barrier — not the frame's.

The cover slots are load-bearing rather than opportunistic: at one CTA per SM there is no
second CTA to run while this one waits, so every leg cycle is either covered by
construction or exposed. LEG EXPOSURE — uncovered cycles per leg — is a ledger number from
this frame on.

<a id="accumulator"></a>
## The accumulator's orientation is forced, not chosen

`C[m = channel][n = leaf]`, `kMCh` m-tiles by `kNL` n-tiles per warp, `kMCh * kNL * 4`
fp32 registers per lane.

The readout's A fragments ARE the fold accumulator's top halves (KERNEL_STANDARDS §2), and
only this orientation admits that. Written out:

```
fold      C[c][leaf] += sum_t  V^T[c][t]      . Wcoef[t][leaf]     -> C = (m = c, n = leaf)
readout   out^T[c][t] = sum_l  S^T[c][leaf]   . R^T[leaf][t]       -> A = (m = c, k = leaf)
```

The `m16n8k16` accumulator's thread-value map is `lane -> m = lane>>2 (+8), n = 2*(lane&3)
(+1)`; the A operand's is `lane -> m = lane>>2 (+8), k = 2*(lane&3) (+1) (+8)`. They agree
on `m`, so two adjacent n-tiles compose one k-tile and the conversion is a RELAYOUT with no
data movement. The transposed orientation puts the leaf on `lane&3` in `C` and on
`lane>>2` in the B operand: no relayout exists, and the accumulator would have to make a
round trip through memory to become an operand — which `common/design.cuh` already
static_asserts is a different algorithm rather than a slower arm of this one.

The cost of the orientation is that the channel count is padded to a multiple of `M = 16`:
at `DV = 64` the 65 channels (64 values and the mass) occupy 5 m-tiles of 16, so 80 of the
lane's registers hold 65 elements of state. The mass is the deposit value's trailing one,
so it is not a special case anywhere except that its A fragment is a CONSTANT rather than a
loaded tile.

<a id="slot-map"></a>
## From a warp's slot to a leaf, a page and a factor column

A side's warp sub-box is a member of the generated set `sub_boxes(D, BC, streams)`. The
member fixes, per level, how many of the box's span bits this warp keeps (`sub_bits`) and
where they sit in the warp's own slot index (`sub_shift`). `slot_local` composes the
warp's stream offset with the slot's digits into the OWNER-LOCAL CANONICAL INDEX — the same
quantity `geom_xch_row` swizzles for the transit region, so the two are one law and not two.

From that index:

* the PAGE and the ROW split at the page rectangle: `page = local >> 4`, `row = local & 15`.
  This holds because the owner span law gives the innermost level at least the page's four
  bits and no shift below it — asserted in the kernel, not assumed.
* the FACTOR COLUMN of level `l` is `col_base[l] + geom_digit(l)`, the owner's grid digit
  above the box's span and the slot's run below it. It depends on the lane's leaf and not
  on the token, so it is computed once in the prologue and never inside a window.

<a id="state-io"></a>
## The state slice

The box's pages are resolved to slots ONCE, at the prologue, into a shared array of
`BC/16` int32 — so the page table is read `atoms per box` times per launch and never per
fragment. A null table is the dense backing, where the slot IS the page (`ops::page_slot`).

The sweep runs in rounds sized by how many page images the transit region's bytes hold. A
round stages its pages with `state_page::load` — a coalesced 32-bit gather with a `PRMT`
interleave and no other arithmetic — issues one edge, and then every warp lifts its own
fragments out of the images by the slot map above. The exit sweep is the mirror. A page
this call only READS moves its high plane alone; one it will WRITE moves both. A page the
box does not touch at all moves nothing, and an absent slot moves nothing.

The die-fast is the same fact at the box's grain: the OR of the box's page bytes, read
before any shared-memory touch. Both bits clear means this CTA has no work in this call and
returns.

<a id="fold"></a>
## PHASE 2: the fold

The fold is WINDOW-BLIND. A deposit never reads the accumulator, so a whole window's
segment MMAs chain back to back with no accumulator dependency, and the segment buffer's
refill is covered by the previous segment's chain rather than by a barrier. This is the
structural difference from an attention mainloop, which must read-modify-write its
accumulator on every tile.

Per segment: the value tile is drained out of the shared slot into A fragments by a
transposing tile load — the segment is stored token major and the operand is channel major,
so the transpose is the load's, not a pass — the slot is then refilled for the next segment,
and the coefficients are expanded in registers. A lane holds one leaf per n-tile and four of
the segment's tokens, so its coefficient is the gain times the product of that leaf's `D`
factor amplitudes, rounded to bf16 as the declared operand arithmetic. A token past the
window's end contributes a zero coefficient and its value quantum zero-fills, so a partial
window is the same code and not a tail case.

The segment slot count is arch-derived: where the MMA sources both operands from the
issuing unit's registers, ONE slot suffices and the registers are the second buffer
(§4's two-stages-in-flight, in its register form); where it does not, the ring is two slots.

<a id="forms"></a>
## The two forms per side, and where each switch sits

Each phase has two bodies, `Form::kUnion` and `Form::kSparse`, and the launch picks one per
side. The switches are ADDITIVE — "two plus two bodies", four in all, not four combinations
— so each sits at its own phase: inside the window loop, outside the batch loop that the
in-kernel-switch ruling calls the hot loop. One uniform branch per phase per window.

The two per-side MEMBER switches are different: they fix the addressing every stage on that
side uses, so they sit in the prologue, outside every loop, nested. That nesting is
MULTIPLICATIVE in the two sides' member counts. At the flagship arm the generated set has
two members per side and the product is four; a deep topology can generate up to twelve per
side, and the product is then the compile budget's problem rather than the schedule's.

<a id="ledger"></a>
## The primitive ledger: where each quantity lives

| class | quantity | where it lives |
|---|---|---|
| 1 arm axes (compile-time free) | `D`, `DV`, `warps_per_cta` | the arm key; template parameters of `carry_kernel`, declared in `tools/manifests/shipped_set.json` |
| 2 the problem (runtime, untuned) | `B_l`, the level modes, `N` | `CarryParams::g`, derived on the host by `derive_carry_geom` |
| 2 the problem (runtime, untuned) | `BH`, `L` | the grid's `y`, `CarryParams::L` |
| 3 runtime dials (the only tuned class) | the form per phase | `CarryParams::readout_form` / `fold_form` |
| 3 runtime dials | the stream grain, the active leaf count, the carve order, the producer:consumer split | not yet consumed; they arrive with the stages that read them |
| 4 arch-deterministic | `M`, `N_mma`, `K` | `common/ops.cuh` |
| 4 arch-deterministic | `warps_per_sm`, the launch table, the cluster size, the state's share of the register file | `common/geom.cuh`, `common/constants.cuh` |
| 4 arch-deterministic | `W`, `kSeg`, the segment slot count | `common/constants.cuh` |
| 5 derived quotients | `leaves_per_sm`, `leaves_per_warp`, `BC`, the atoms per box, the accumulator's tile counts, the shared ledger's rows | `BoxPlan` in `carry/box.cuh` |

A class-4 or class-5 quantity appearing as a knob anywhere is a bug, and that is what makes
the table worth writing down rather than inferring.

<a id="box-plan"></a>
## `BoxPlan`

One `struct` per arm holding every class-5 quantity above as a `constexpr`, with the
static_asserts that make the derivation refuse rather than round: the box is a whole number
of pages, a warp's leaves are a whole number of n-tiles, the window is a whole number of
segments, the page images fit the region, the ledger fits the architecture's per-CTA
maximum. Nothing in it is a literal that a reader has to trust.
