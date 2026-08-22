# `csrc/rola/src/decode/decode_lattice.cuh` — the `(k, m)` box lattice and the T=1 factor tables

The leaf order for `T = 1` decode IS the `(k, m)` box lattice, in the ONE order
`csrc/rola/src/carry/box.cuh` defines and the carry family writes: a leaf is
named by its owner coordinate `o_l` at every level and by its per-level RUN
OFFSET `r_l = d_l - o_l * s_l`, composed in MIXED RADIX most significant level
first,

```
lambda = O * BC + sum_l r_l * prod_{l' > l} s_l'
```

and this file is the device object the step's walk reaches a leaf THROUGH — by
composing those factors — rather than by consulting a map over the leaf space.
Nothing here is sized in `N`: one bit per DIGIT (`sum_l ceil(B_l/32)` words) and
one entry per live OWNER COORDINATE (`sum_l g_l` at most) is the whole
apparatus, for the ONE SIDE the calling pass walks.

THE INTERLEAVE IS DEAD. The R0 spec's LATTICE-1 §2.2 named a second intra-box
order — the `[u_0..u_{D-1} | c_0..c_{D-1}]` interleave, in which a sub-box `k^D`
was a contiguous run — and an earlier draft of this walk enumerated
`(owner, sub-box)` units because of it. Mixed radix is the shipped order (the
one-order ruling; `box.cuh`'s `run_of`), a sub-box is a scattered rectangle
under it, and the walk's unit moved accordingly (§3). The measured reason is
recorded in `docs/internals/DELETIONS.md`'s wave-2c rows and in the batch's
findings: under the interleave a token's run over a level that is DENSE for its
side straddles atoms at 25% occupancy at EITHER span vector; under mixed radix
with `s_{D-1} >= kAtomLeaves` it FILLS them; where the level is narrower than
an atom the rectangle spans two levels and the fill is partial
(`carry/box.md`).

Symbols, restated from the spec at first use here: `D` the number of routing
levels; `B_l` level `l`'s width; `k = 2^kappa` the uniform per-level sub-box
span; `m = prod_l m_l` the capacity multiplier, distributed innermost-first to
the atom floor, capped by the level's own width and balanced over the rest
(spec §2.5 as the atom-rectangle law generalizes it, `carry/box.md`);
`s_l = k * m_l` an owner's span at level `l`; `g_l = B_l / s_l` the owner-grid
extent; `BC = prod_l s_l` the leaves one owner holds; `atom` the 16-leaf page
granule (`rola::carry::kAtomLeaves`).

`decode.cuh` carries the box's fields on `DecodeParams` under `BoxPlan`'s own
names (`lat_k`, `lat_s[]`, `lat_g[]`, `lat_gsuf[]`, `lat_run_shift[]`,
`lat_local_bits`, and the unit's own `lat_inner_bits`, `lat_outer_bits`,
`lat_oshift[]`) because decode takes `(k, m)` and the widths as DATA rather than
as template arms — one built binary serves every topology the producer can emit,
so `BoxPlan`'s constexpr fields become runtime scalars derived by `BoxPlan`'s own
formulas. This doc is their authority; `decode.cuh` merely carries them.

---

## <a id="the-digit-mask"></a>1. `build_digit_mask` — one bit per digit, race-free without atomics

A side's DIGIT MASK is `sum_l ceil(B_l/32)` words, one bit per digit of every
level, packed level by level (`mask_off[l]` / `mask_words[l]` locate a level's
slice, `mask_total` the whole plane). One thread owns one destination word and
reads only the amplitudes that word covers — the exact-nonzero test spec
LATTICE-1 assumes throughout, never a threshold — so the pass needs no atomic
and no second pass to merge partial words.

`level_slice` reads `count <= kDecodeMaxSpan` digits of one level as a single
word through `extract_bits` (`decode.md#extract-bits` — the funnel shift is
written once and this file is one more caller of it, not a second copy of it).
An owner's span and a sub-box's span are each one such extract, which is what
`kDecodeMaxSpan` is a capacity FOR.

## <a id="the-owner-lists"></a>2. `build_owner_list` — ONE side, and why one

Per level, the ascending owner coordinates at which THIS SIDE has amplitude —
`olist[level]`, `n_o` the realized count — built from the one digit mask the
caller passes, with `units` its own product `prod_l |live_l|` and never a
union's.

**ONE SIDE IS THE WHOLE POINT, because a union over the sides is the DENSE
PRODUCT.** A list holding "a leaf either side reaches" enumerates
`prod_l (R_l u W_l)`, and under an alternating declaration (`carry/box.md`)
every level is dense for one side, so `R_l u W_l` is that whole level and the
product is the entire lattice — however sparse each side is on its own. Each
side's own product prunes; their union does not. So the object here takes one
mask and produces one list, and the step's two passes call it once each with
their own side's (`decode.md#per-side-enumeration`).

Building a list is one `cub::BlockScan` exclusive-sum per level —
`lat_g[l] <= kDecodeThreads` follows from the level-width capacity
(`decode.md#level-width-capacity`), so one block covers a whole level's owner
grid.

**Each level's live count is padded to the next power of two.** `bits[level]`
is `ceil_log2(n_o[level])`, `shift[level]` the running sum of the levels
below it, and a unit's owner rank decomposes into per-level fields by SHIFTS
AND MASKS alone — never a division. A rank in the padded-but-unrealized part
of a level's field is `i >= n_o[level]`, marked `invalid` and skipped rather
than wrapped: the padding costs some dead ranks in exchange for never issuing
a device integer division in the walk's innermost enumeration, which — unlike
the divisions the deleted mixed-radix decomposition paid per bitmap word — a
lattice unit would otherwise pay once per candidate atom per CTA.

`units` is the whole enumeration for that side: the padded owner-rank field's
width plus `lat_outer_bits` low bits for the fixed run offsets, i.e.
`2^(sum_l bits[l] + lat_outer_bits)` — zero wherever any level's list is empty,
so an absent owner on that side kills its enumeration outright rather than
walking zero-width ranges.

## <a id="the-unit"></a>3. `Unit` / `resolve_unit` — one unit is an `(owner, r_0 .. r_{D-2})` pair

A UNIT is what the grid partitions: one owner coordinate `O` together with
EVERY RUN OFFSET BUT THE INNERMOST. Under the mixed-radix intra-box order the
innermost level's run is the CONTIGUOUS TAIL of the owner-local index, so a
unit's leaves are `s_{D-1}` consecutive lattice indices at an `s_{D-1}`-aligned
base. At the flagship spans `(8, 16)` that run IS one 16-leaf atom, at full
occupancy, and the unit's 16 slots are level 1's digits — which is the whole
reason the unit is this shape rather than the interleave's `(owner, sub-box)`.

`resolve_unit` decomposes a unit index by shifts alone: the OWNER RANK occupies
the high bits, one padded field per level (§2), and the low `lat_outer_bits`
are the fixed run offsets in the SAME mixed radix the owner-local index uses, so
`outer << lat_inner_bits` is the unit's base inside its owner with no multiply.

**A UNIT IS ONE SIDE'S.** `resolve_unit<D, DECAY>` takes ONE digit mask, ONE
staged amplitude row and ONE owner list — always the same side's, the side whose
rank space the unit index runs over (§2) — so what it hands back carries that
side's amplitude and that side's membership and has no field for the other's.

The result is a `Unit<D>`: `base` (the unit's first leaf as a lattice index), a
`valid` flag (false where the padded rank space overshoots a level's realized
count, §2), `live` (false where some FIXED level's digit is absent on this side
— the pruning the product structure `supp(R) = prod_l supp_l(R)` buys: one
absent level kills the whole unit, in ONE test rather than a per-leaf one), the
innermost level's staging base `dlast`, this side's `s_{D-1}`-wide slice `c`,
and the FIXED LEVELS' AMPLITUDE PRODUCT `a` (with `rate`, the decay dials'
product, on the `DECAY` arm). The owner coordinate itself is a local of the
decomposition: `base` is what the walk addresses through, so a unit that carried
the owner as well would hold a second name for the same place across the walk's
tightest scope.

Those products are the unit's second point. Because a unit fixes every level but
the innermost, their amplitudes are resolved ONCE HERE and a slot below costs a
single innermost amplitude rather than `D` of them — where the interleaved
walk's unit cut every level and had to re-form the whole product per leaf.

THERE IS NO SECOND UNIT SHAPE. A topology whose innermost span is thinner than
an atom does not get a deeper walk: it gets SEVERAL UNITS PER ATOM, which the
CANDIDATE enumeration absorbs (`lat_units_per_atom_shift`,
`decode.md#candidate-atoms`) and the walk never sees. An earlier draft made the
unit's depth a runtime quantity so that a unit was always at least an atom; it
was measured to SPILL the step's register budget at `D = 2` and `D = 3` — the
extra loop level held every live value of the walk's tightest scope across a
trip count of one at every shipped topology — and the batch's findings carry the
census rows. This is the "no branch to dodge edge cases" rule applied: one
strategy, uniformly, and the degenerate config pays at the candidate enumeration
instead of in the walk.

`Unit` is the object the step's walk (`decode.md#the-unit-partition`) and the
candidate-atom enumeration (`decode.md#candidate-atoms`) both resolve a unit
into; neither reads the mask or owner-list arrays through any other route.

## <a id="capacities"></a>4. The lattice's declared capacity

`kDecodeMaxSpan = 32` (`decode.cuh`), and it is now the ONLY one. An owner's
span `s_l` is read as a SINGLE 32-bit slice — both when a level's owner grid is
built (§1) and when a unit's innermost run is resolved (§3) — so the span is a
gate the host refuses past, never a slower fallback, at
[`fill_lattice`](#the-refusals). `rola.ops.lattice` mirrors it (`_MAX_SPAN`) so
a topology the device would refuse is refused on the host, once, before a
launch, rather than negotiated at one; the decode build stamp carries it, so a
host that outgrew the device's box fails a device-side check
(`decode_api.md#build-stamp`).

The interleave's other two ceilings died with it: `kDecodeMaxCap` bounded the
capacity radix a unit's low bits carried, and `kDecodeMaxSubBox` bounded the
intra-sub-box scan. Under mixed radix a unit is a contiguous run, there is no
capacity radix separate from the run offsets, and there is no sub-box scan to
bound.

## <a id="the-refusals"></a>5. `fill_lattice` — the box's constraints, as runtime checks

Decode takes its lattice as DATA, not as a template arm, so every constraint
`box.cuh`'s `BoxPlan` states as a `static_assert` is a `TORCH_CHECK` in
`decode.cu::fill_lattice`:

- `k` and `m` are exact powers of two (K1);
- `k <= kDecodeMaxSpan`;
- per level, `s_l = k * m_l <= kDecodeMaxSpan` and `B_l % s_l == 0` (K6: an
  owner sits inside a level — `m_l` is spec §2.5 as the atom-rectangle law
  generalizes it, innermost first to the atom floor, capped by the level's own
  width and balanced over the rest, restated in `fill_lattice`'s loop and in
  `rola.ops.lattice.box_shape`);
- `g_l <= kDecodeThreads` — a level's owner grid is built in one block pass;
- `owners * BC == N` — the lattice PARTITIONS the leaf space exactly, with no
  remainder leaf and no double-covered one.

THE ATOM FLOOR IS A PREFERENCE, THE WIDTH CEILING IS THE LAW. A topology whose
widths cannot reach `s_{D-1} >= kAtomLeaves` (an odd level, a level narrower
than an atom) still has a lattice and still decodes: the floor is taken as far
as `log2 m` and the level's width allow, the capacity spills outward, and the
atom then spans more than one level. What IS refused is a capacity no span
vector can carry — `sum_l a_l != log2 m`.

A PAGED backing adds ONE more, checked only where `page_table` is present
(`rola_decode_forward`, cross-referenced from `decode.md#candidate-atoms`):
`BC >= kAtomLeaves`, i.e. an owner holds a whole number of atoms. That is what
makes an atom and a unit nest one way or the other, which is what makes the
candidate enumeration a PARTITION. The interleave needed two clauses here
(`k^D % kAtomLeaves == 0` and `4 % log2(k) == 0`); mixed radix discharges both
by the order itself — an atom is 16 consecutive `local` values at a multiple of
16, hence an aligned digit rectangle for EVERY admissible span vector
(`box.cuh`, K3/K4). A DENSE backing never asks an atom a liveness question and
carries no constraint.

`rola.ops.lattice.derive_lattice` and `box_shape` are the SAME arithmetic,
host-side and static, and are what choose `(k, m)` for a topology before any of
this runs — `fill_lattice` re-derives nothing; it validates the pair the host
already committed to and fills the suffix products and shift widths
`resolve_unit` reads (§3). `rola.ops.carry` reads its box from the same module,
so the two families cannot disagree about where a leaf is.

## <a id="the-seam"></a>6. `rola.ops.lattice` — the oracle and test seam, and nowhere else

`permutation(widths, k, m)` builds `pi`, an `[N]` int64 tensor with
`pi[canonical_leaf] = lattice_leaf` — spec §2.4's theorem that `pi` is a FIXED
BIT PERMUTATION, computed here as the explicit mixed-radix formula (spec
§2.3's forward map) rather than assumed. `to_lattice`/`to_canonical` are the
one legal place a state plane crosses between the oracle's canonical leaf
order and the lattice order the kernel is BORN in: `to_lattice` via
`index_copy_(1, pi, canonical)`, the inverse via `index_select`.

No shipped kernel path ever calls `permutation` or converts a plane — a
kernel never HOLDS a plane in canonical order to begin with, so there is
nothing on the production path to convert. The seam exists for the oracle
(which is canonical forever, `docs/internals/DELETIONS.md`'s standing rule)
and for a test that must compare the two: `_write_atom_bitmap`
(`decode.md#second-derivation`) and the fp64 oracle battery both cross through
here exactly once, on the way into or out of the comparison, and never inside
a measured step.

`LATTICE_CAP_BC = 128` is the one number in `derive_lattice` that is a CHOICE
rather than a consequence: the leaves ONE OWNER holds, which is what the carry
family's register budget bounds, and the ceiling the solver searches down from.
It is stated as `BC` rather than as a capacity multiplier because `BC = k^D * m`
grows with DEPTH — a ceiling on `m` bounds the box at one `D` only, and `m = 8`
is a box of 128 at `D = 2` and 2048 at `D = 4`. The objective is spec §7.3's in
its order: maximize `BC` under the cap, then the SHALLOWEST atom rectangle, then
the largest `k`. That reproduces the built arms (`(64,64)` and `(256,256)` both
`(4, 8)`) and selects `k = 2` at `(16,16,16)`, whose one-level rectangle the
`k = 4` box cannot reach.
