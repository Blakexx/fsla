# `csrc/rola/src/decode/decode_lattice.cuh` — the `(k, m)` box lattice and the T=1 factor tables

The leaf order for `T = 1` decode IS the `(k, m)` box lattice derived by
`rola.ops.lattice` (spec `K31_R0_ABI_RIPPLE_SPEC.md` sections 2.1–2.5, LATTICE-1):
a leaf is named by its owner coordinate `o_l`, its sub-box coordinate `u_l` and
its intra-sub-box digit `c_l` at every level `l`, and this file is the device
object the step's walk reaches a leaf THROUGH — by composing those three
factors — rather than by consulting a map over the leaf space. Nothing here is
sized in `N`: one bit per DIGIT (`sum_l ceil(B_l/32)` words per side) and one
entry per live OWNER COORDINATE (`sum_l g_l` at most, per list) is the whole
apparatus.

Symbols, restated from the spec at first use here: `D` the number of routing
levels; `B_l` level `l`'s width; `k = 2^kappa` the uniform per-level sub-box
span; `m = prod_l m_l` the capacity multiplier, distributed across levels
round-robin (spec §2.5); `s_l = k * m_l` an owner's span at level `l`;
`g_l = B_l / s_l` the owner-grid extent; `BC = prod_l s_l` the leaves one owner
holds; `k^D` the leaves one sub-box holds. `owner`/`O`, `cap`/`U` and `C` below
are the mixed-radix numbers spec §2.2 calls `O(o)`, `U(u)` and `C(c)`.

`decode.cuh` carries the `(k, m)` fields on `DecodeParams` (`lat_k`, `lat_m`,
`lat_m_l[]`, `lat_s[]`, `lat_g[]`, `lat_gsuf[]`, `lat_ushift[]`, `lat_cshift[]`
— the suffix products and shift widths spec §2.2–2.4 name, held rather than
recomputed because the walk forms them once per unit) because the decode TUs
share no other header with the fields' consumer; this doc is their authority,
`decode.cuh` merely carries them.

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

## <a id="the-owner-lists"></a>2. `build_owner_lists` — two lists, and why two

Per (list, level), the ascending owner coordinates at which that list's side
has amplitude — `olist[list][level]`, `n_o` the realized count. TWO lists:
`kListUnion` (a leaf either side reaches, live-a-side-or-the-other) drives the
walk itself; `kListWrite` (an atom the step deposits into) drives the
candidate-atom enumeration alone. One list would make the atom enumeration the
union's size, and a page-table probe per entry is what that would cost on a
batch-head whose read support is far wider than its write support.

Building both is one `cub::BlockScan` exclusive-sum per (list, level) —
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

`units[list]` is the whole enumeration for that list: the padded owner-rank
field's width plus `lat_msb = log2(m)` low bits for the sub-box index, i.e.
`2^(sum_l bits[l] + log2 m)` — zero wherever any level's list is empty, so an
absent owner or write kills that list's enumeration outright rather than
walking zero-width ranges.

## <a id="the-unit"></a>3. `SubBox` / `resolve_unit` — one unit is an `(owner, sub-box)` pair

A UNIT is what the grid partitions: one owner coordinate `O` and one sub-box
coordinate `U` inside it. `resolve_unit` decomposes a unit index by shifts
alone — the OWNER RANK occupies the high bits, one padded field per level
(§2), and the low `lat_msb` bits are the CAPACITY RADIX, i.e. `U`'s own
mixed-radix digits `u_l = (cap >> lat_ushift[l]) & (m_l - 1)`. Both are
`decode.cuh`'s own fields, restated as the shift arithmetic that reads them.

The result is a `SubBox<D>`: `owner`, `cap`, a `valid` flag (false where the
padded rank space overshoots a level's realized count, §2), `live_r`/`live_w`
(false where some level's slice is entirely absent on that side — the pruning
the product structure `supp(R) = prod_l supp_l(R)` buys, spec §2 restated:
one absent level's slice kills the whole sub-box for that side, in ONE
register test rather than a per-leaf one) and, per level, the digit base
`dbase[l]` and both sides' `k`-wide intra-sub-box slices `cr[l]`/`cw[l]`. Every
array is indexed by a COMPILE-TIME level, which is what keeps the whole object
in registers rather than in local memory.

`SubBox` is the object the step's walk (`decode.md#the-unit-partition`) and
the candidate-atom enumeration (`decode.md#candidate-atoms`) both resolve a
unit into; neither reads the mask or owner-list arrays through any other
route.

## <a id="capacities"></a>4. The lattice's declared capacities

`kDecodeMaxSpan = 32`, `kDecodeMaxCap = 32`, `kDecodeMaxSubBox = 64`
(`decode.cuh`). An owner's span `s_l` and the sub-box span `k^D` are each read
as a SINGLE 32-bit slice (§1), and the intra-sub-box scan (`decode.md#update-then-read`'s
inner loop) is a fixed cost per live sub-box bounded by `kDecodeMaxSubBox`, so
both capacities are gates the host refuses past — never a slower fallback —
at [`fill_lattice`](#the-refusals). `rola.ops.lattice` mirrors both
(`_MAX_SPAN`, `_MAX_SUB_BOX`) so a topology the device would refuse is refused
on the host, once, before a launch, rather than negotiated at one.

## <a id="the-refusals"></a>5. `fill_lattice` — the spec's constraints, as runtime checks

Decode takes its lattice as DATA, not as a template arm, so every constraint
the R0 spec's LATTICE-1 states algebraically (§§2.1–2.5, 3.1–3.3) is a
`TORCH_CHECK` in `decode.cu::fill_lattice` rather than a `static_assert`:

- `k` and `m` are exact powers of two (spec K1);
- `k <= kDecodeMaxSpan`, `m <= kDecodeMaxCap`;
- per level, `s_l = k * m_l <= kDecodeMaxSpan` and `B_l % s_l == 0` (spec K6:
  an owner sits inside a level — `m_l` is the balanced round-robin split of
  `log2(m)` across levels, spec §2.5, restated verbatim in `fill_lattice`'s
  loop and in `rola.ops.lattice.box_shape`);
- `k^D <= kDecodeMaxSubBox`;
- `owners * BC == N` — the lattice PARTITIONS the leaf space exactly, with no
  remainder leaf and no double-covered one.

A PAGED backing adds two more, checked only where `page_table` is present
(`rola_decode_forward`, cross-referenced from `decode.md#candidate-atoms`):
`k^D % kAtomLeaves == 0` (an atom must not straddle two sub-boxes, spec §3.2)
and `4 % log2(k) == 0` (spec §3.2's digit-rectangle claim: the atom's low
4 bits must be whole trailing `c_l` fields, or the candidate-atom enumeration's
per-level bit test would cut a level's digit in half). A DENSE backing never
asks an atom a liveness question and carries neither constraint.

`rola.ops.lattice.derive_lattice` and `box_shape` are the SAME arithmetic,
host-side and static, and are what choose `(k, m)` for a topology before any
of this runs — `fill_lattice` re-derives nothing; it validates the pair the
host already committed to and fills the suffix products and shift widths
`resolve_unit` reads (§3).

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

`LATTICE_CAP_M = 8` is the one number in `derive_lattice` that is a CHOICE
rather than a consequence: the capacity multiplier the flagship carry arm is
solved at, and the ceiling the solver searches down from — chosen so the
flagship rows agree with the carry family's built arms, restated from neither
family's own doc because the two share no other object.
