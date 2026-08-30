"""THE EXCHANGE'S LAYOUT, PROVED ON THE MODEL (K35 F1).

The transit is the one place in the carry where two INDEPENDENT derivations have to
meet: the publisher names a row from its own WRITE order and the consumer addresses
that same row as a slot of its own READ order, and the two share no code.  What has to
be proved is therefore that the COMPOSITE IS THE IDENTITY ON LATTICE LEAVES -- and,
separately, that neither leg of the traffic conflicts in the banks.

This file is the HOST leg and it is pure arithmetic: no CUDA, no built binary.  The
device leg is the fp64 oracle at every carved arm (`tests/oracle/test_carry_vs_oracle.py`'s
`test_f1_*`), where `num`/`den` ARE the fold->readout round trip and the state plane is
the accumulator in the write carve's order.  Together they are section 3's "round-trip
test proving bit-exactness and conflict-freedom" for this layout.

THE MODEL NO LONGER MIRRORS THE HEADER -- IT CONSUMES THE SHIPPED DERIVATION.  Before
K50 this file rebuilt the carve in Python from `tools/gen_shards.py`'s `_BoxPlan`/
`_ModeOrder`/`_StreamSpans`, which made it the FOURTH place a carve was derived
(`## F3` SKILLS FEEDBACK item 3 named all four).  The mode word and the digit widths are
DATA now, so there is ONE derivation -- `derive_carry_geom` in `csrc/rola/src/carry/geom.cuh`
-- and this file reads its block through `rola.ops.carry.geometry`.  What it still does
independently is COMPOSE the two sides' orders into a row and check the properties, so a
wrong derivation shows up as a broken permutation here rather than as noise on a device.
"""

import pytest

from rola.ops import carry as carry_ops

K_PAD_E = 8
K_MMA_M, K_MMA_N, K_MMA_K = 16, 8, 16

def _ilog2c(x):
    """the bit index of a power of two -- the model's own, matching `geom_ilog2`."""
    b = 0
    while (1 << b) < x:
        b += 1
    return b


#: the field layout of the flat block `carry_ops.geometry` returns.
_LVL = 4
_BLK_ARRAYS = ("width", "span", "grid", "col_base", "run_shift", "owner_div", "weight")
_BLK_SCALARS = ("wtot", "owners", "leaves", "local_bits", "min_width", "identity_map")
_SIDE_ARRAYS = ("level_at", "rank", "row_span", "grid", "grid_div", "row_g", "row_prefix")
_SIDE_SCALARS = ("carves", "rows", "leaf", "streams")


def _block(widths, modes, ns):
    g = list(carry_ops.geometry(widths, modes, carry_ops.BC, ns, ns))
    D, i = g[0], 2
    out = {"D": D, "bc": g[1]}
    for name in _BLK_ARRAYS:
        out[name] = g[i:i + D]
        i += _LVL
    for name in _BLK_SCALARS:
        out[name] = g[i]
        i += 1
    for side in ("r", "w"):
        for name in _SIDE_ARRAYS:
            out[f"{side}_{name}"] = g[i:i + D]
            i += _LVL
        for name in _SIDE_SCALARS:
            out[f"{side}_{name}"] = g[i]
            i += 1
    out["map"] = g[i:]
    return out


class _Side:
    """One side's derived carve, READ OFF THE BLOCK -- never re-derived here."""

    def __init__(self, blk, side, ns):
        self.D, self.ns, self.blk = blk["D"], ns, blk
        self._p = side  # "r" or "w"
        self.bc = blk["bc"]
        self.leaf = blk[f"{side}_leaf"]

    def s(self, l):
        return self.blk[f"{self._p}_row_span"][l]

    def g(self, l):
        return self.blk[f"{self._p}_grid"][l]

    def rank(self, l):
        return self.blk[f"{self._p}_rank"][l]

    def shift(self, l):
        """THE CANONICAL SLOT LAYOUT (G2b): the sub-box's digits, innermost lowest.  It is
        the same expression on both sides, which is the claim this file exists to check."""
        return sum(_ilog2c(self.s(i)) for i in range(l + 1, self.D))

    def grid_div(self, l):
        return self.blk[f"{self._p}_grid_div"][l]

    def off(self, l, stream):
        return ((stream // self.grid_div(l)) % self.g(l)) * self.s(l)


#: THE DECLARATIONS THE TRANSIT IS PROVED OVER.  They are no longer arms -- one binary
#: serves every one of them (K50) -- so the parametrization names the geometry directly,
#: and it is a SUPERSET of what the former arm list covered: the flagship's ALTERNATING
#: word and its TIED word, the small window, both deep topologies, and the three flat
#: declarations the conformance cells drive.
CASES = [
    ("flag-alt", (256, 256), 6),
    ("flag-tied", (256, 256), 3),
    ("flag-dense", (256, 256), 0),
    ("flag-read-one", (256, 256), 1),
    ("flag-write-both", (256, 256), 10),
    ("small-alt", (64, 64), 6),
    ("deep3-alt", (16, 16, 16), 38),
    ("deep4-alt", (16, 16, 16, 16), 102),
]
NS = 4


def _sides(case):
    _, widths, modes = case
    blk = _block(widths, modes, NS)
    return _Side(blk, "w", NS), _Side(blk, "r", NS), 64


def _row_of(w_side, r_side, stream, slot):
    """publisher `stream`'s write slot -> the exchange's row, reader-major.  Under the
    canonical layout the two sides tile the same box with the same sub-box, so a leaf
    keeps its SLOT and only changes WARP -- but this composes the two sides' own
    derivations rather than assuming that, so a broken derivation shows up here."""
    reader, q = 0, 0
    for l in range(w_side.D):
        d = ((slot >> w_side.shift(l)) & (w_side.s(l) - 1)) + w_side.off(l, stream)
        reader += (d // r_side.s(l)) * r_side.grid_div(l)
        q |= (d % r_side.s(l)) << r_side.shift(l)
    return reader * r_side.leaf + q


def _canonical(w_side, r_side):
    """THE LAYOUT IS THE SAME ON BOTH SIDES -- the register-shape claim itself."""
    return all(w_side.s(l) == r_side.s(l) and w_side.shift(l) == r_side.shift(l)
               for l in range(w_side.D))


def _tied(w_side, r_side):
    """THE IDENTITY-MAP FACT, and it is a fact and no longer a body: K50 deleted the
    elided tied path ("one transit, always"), so what a tie buys is a row map that
    happens to be the identity -- reported so a wasted transit is visible in the ledger."""
    return all(w_side.g(l) == 1 or w_side.grid_div(l) == r_side.grid_div(l)
               for l in range(w_side.D))


CARVED = [(c[0], c) for c in CASES]
UNTIED = [(i, c) for i, c in CARVED if not _tied(*_sides(c)[:2])]
IDS = [c[0] for c in CASES]


def test_at_least_one_declaration_is_the_identity_map_and_one_is_not():
    """Neither end of the fact may go untested: the identity map needs a declaration
    that reaches it, and the transit needs a permuted one to prove it moves anything."""
    tied = [i for i, c in CARVED if _tied(*_sides(c)[:2])]
    assert tied, "no declaration reaches the identity row map; the fact is unproved"
    assert UNTIED, "no declaration carries a permuted transit"


@pytest.mark.parametrize("idx,row", CARVED, ids=IDS)
def test_the_exchange_relays_the_state_leaf_for_leaf(idx, row):
    """THE COMPOSITE IS THE IDENTITY.  Every (publisher, write slot) claims exactly one
    row, the rows are a permutation of `[0, BC)`, and the row a consumer pulls at its
    own slot carries the leaf that slot names."""
    w, r, _ = _sides(row)
    assert w.leaf == r.leaf and w.bc == r.bc
    claimed = {}
    for stream in range(w.ns):
        for slot in range(w.leaf):
            row_i = _row_of(w, r, stream, slot)
            assert row_i not in claimed, f"arm {idx}: two publishers claim row {row_i}"
            claimed[row_i] = (stream, slot)
    assert sorted(claimed) == list(range(w.bc)), f"arm {idx}: the rows are not a permutation"

    # and the two derivations agree on WHICH leaf: both sides name the same lattice leaf.
    for (stream, slot), row_i in ((v, k) for k, v in claimed.items()):
        reader, q = divmod(row_i, r.leaf)
        for l in range(w.D):
            d_w = ((slot >> w.shift(l)) & (w.s(l) - 1)) + w.off(l, stream)
            d_r = ((q >> r.shift(l)) & (r.s(l) - 1)) + r.off(l, reader)
            assert d_w == d_r, (
                f"arm {idx}: publisher ({stream}, {slot}) and consumer ({reader}, {q}) "
                f"disagree on level {l}'s digit: {d_w} != {d_r}")


@pytest.mark.parametrize("idx,row", CARVED, ids=IDS)
def test_the_exchange_row_map_is_affine_in_the_slots_bits(idx, row):
    """`row(w, P) = row(w, 0) ^ delta(P)`, every step ONE bit and independent of the
    publisher, and the base's bits disjoint from every step's -- which is what lets the
    kernel pay one base per warp and an IMMEDIATE per fragment, and what makes the XOR
    an ADD in the address arithmetic (`XchMap`'s own static_assert)."""
    w, r, _ = _sides(row)
    nb = _ilog2c(w.leaf)
    ref_step = [_row_of(w, r, 0, 1 << b) ^ _row_of(w, r, 0, 0) for b in range(nb)]
    for b, st in enumerate(ref_step):
        assert bin(st).count("1") == 1, f"arm {idx}: slot bit {b} is not one row bit"
    delta_mask = 0
    for st in ref_step:
        delta_mask |= st
    base_mask = 0
    for stream in range(w.ns):
        base = _row_of(w, r, stream, 0)
        base_mask |= base
        for b in range(nb):
            assert _row_of(w, r, stream, 1 << b) ^ base == ref_step[b], (
                f"arm {idx}: stream {stream}'s step for bit {b} differs from stream 0's")
        for slot in range(w.leaf):
            acc = base
            for b in range(nb):
                if slot >> b & 1:
                    acc ^= ref_step[b]
            assert acc == _row_of(w, r, stream, slot)
    assert delta_mask & base_mask == 0, (
        f"arm {idx}: a slot bit shares a row bit with a publisher base, so `^` is not `+`")


@pytest.mark.parametrize("idx,row", CARVED, ids=IDS)
def test_the_slot_layout_is_the_same_on_both_sides_and_the_map_moves_whole_warps(idx, row):
    """THE REGISTER-SHAPE CLAIM (G2b), on the shipped derivation.

    The warp sub-box is compile-time `f(D, DV)` and CANONICAL: both sides tile the owner's
    box with the same spans in the same slot order, so the per-side ORDER can only decide
    which stream holds which sub-box.  The transit is therefore a permutation of leaves
    BETWEEN warps with the slot bits FIXED -- one runtime base word per warp and the
    identity on the slot -- which is what lets the kernel address it with immediates."""
    w, r, _ = _sides(row)
    assert _canonical(w, r), (
        f"arm {idx}: the two sides do not share a slot layout, so the row map is not a "
        f"warp permutation and the produce's shifts are not immediates")
    for slot in range(w.leaf):
        assert _delta(w, r, slot) == slot, (
            f"arm {idx}: slot {slot} does not keep its offset in the row map")
    for stream in range(w.ns):
        base = _row_of(w, r, stream, 0)
        assert base % r.leaf == 0, f"arm {idx}: publisher {stream}'s base is not a warp base"
        for slot in range(w.leaf):
            assert _row_of(w, r, stream, slot) == base + slot


def _bank_free(rows, ldx, gran, word_off=0, words_per_gran=4):
    """the region's bank model: a 16-byte granule is four consecutive banks, and a row
    is `ldx / 2` words on from the one before it."""
    words, banks = ldx // 2, set()
    for row_i in rows:
        for word in range(words_per_gran):
            b = (words * row_i + 4 * gran + word_off + word) % 32
            if b in banks:
                return False
        for word in range(words_per_gran):
            banks.add((words * row_i + 4 * gran + word_off + word) % 32)
    return True


@pytest.mark.parametrize("idx,row", CARVED, ids=IDS)
def test_the_region_needs_no_swizzle_because_the_row_stride_separates_the_banks(idx, row):
    """THE SWIZZLE SEARCH, AS A COMPILE-TIME ENUMERATION -- and its answer is NONE.

    Before G2b a leaf changed both its warp AND its slot in the transit, so a lane quad's
    four rows sat 16 or 32 apart, `ldx / 2 * 16 = 0 (mod 32)` put them in one four-bank
    block, and a shift-XOR swizzle on bits 3 and above had to separate them -- which it
    could not do for the deep-3 carve, whose eight writer rows differed in the LOW bits
    (the former `KNOWN_RETURN_PULL_CONFLICTS`).  Under the canonical slot layout every
    access covers rows an ALIGNED RUN apart, and `ldx / 2 = 4 x odd` maps a run of eight
    onto eight different four-bank blocks by itself.  This asserts that lemma AND walks
    all four accesses unswizzled at every declaration, which is what deletes the swizzle
    rather than re-tuning it."""
    w, r, dv = _sides(row)
    ldx, n_gran, leaf = dv + K_PAD_E, dv // K_MMA_M_CH, w.leaf
    assert (ldx // 2) % 4 == 0 and (ldx // 2) // 4 % 2 == 1, (
        f"arm {idx}: half a row is {ldx // 2} words, which is not four times an odd "
        f"number -- an aligned eight of rows would share a four-bank block")

    for stream in range(w.ns):
        base = _row_of(w, r, stream, 0)
        #: the publish's lane quad: four rows two apart at eight consecutive channels.
        for j in range(leaf // K_MMA_N):
            for e1 in (0, 1):
                rows_p = [base + j * K_MMA_N + 2 * q + e1 for q in range(4)]
                for g in range(n_gran):
                    assert _bank_free(rows_p, ldx, g), f"arm {idx}: publish conflicts"
                assert _bank_free(rows_p, ldx, n_gran, words_per_gran=1), (
                    f"arm {idx}: the mass publish conflicts")
        #: the writer's return pull: eight rows of an aligned eight, one granule.
        rows_r = [base + q for q in range(8)]
        for g in range(n_gran):
            assert _bank_free(rows_r, ldx, g), f"arm {idx}: the return pull conflicts"

    for reader in range(r.ns):
        for kk in range(leaf // K_MMA_K):
            for half in (0, 8):
                rows_l = [reader * leaf + kk * K_MMA_K + half + i for i in range(8)]
                for g in range(n_gran):
                    assert _bank_free(rows_l, ldx, g), f"arm {idx}: the pull conflicts"
                for e1 in (0, 1):
                    rows_m = [reader * leaf + kk * K_MMA_K + half + 2 * q + e1
                              for q in range(4)]
                    assert _bank_free(rows_m, ldx, n_gran, words_per_gran=1), (
                        f"arm {idx}: the mass pull conflicts")


@pytest.mark.parametrize("idx,row", CARVED, ids=IDS)
def test_the_transit_is_sized_by_the_disorder_between_the_two_carves(idx, row):
    """The transit is ZERO exactly when no leaf crosses a warp, and one whole
    `[consumer][leaf][channel]` region otherwise -- never something in between (K35
    S1b addendum 3(b): every group is live at the same time, so the union of the
    groups' rows is all `BC` of them)."""
    w, r, dv = _sides(row)
    crossings = sum(1 for stream in range(w.ns) for slot in range(w.leaf)
                    if _row_of(w, r, stream, slot) != stream * r.leaf + slot)
    expected = 0 if crossings == 0 else w.bc * (dv + K_PAD_E) * 2
    assert (crossings == 0) == _tied(w, r), (
        f"arm {idx}: `kTied` and the permutation disagree about whether a leaf moves")
    assert expected == (0 if _tied(w, r) else w.bc * (dv + K_PAD_E) * 2)


# ---------------------------------------------------------------- the hybrid's two legs
#
# F1b: the hi plane MOVES.  The writers publish it and drop it; the readers pull it as
# their A operand; after the readout the readers store those same bits BACK and the
# writers pull them in their OWN layout and rejoin them with the lo plane they kept.
# What has to be proved is that the round trip returns each publisher's OWN element to
# the register the rejoin expects -- so this models all four accesses at BYTE addresses,
# with the kernel's own arithmetic, and runs the whole transit as a simulation.

K_MMA_M_CH = 8  #: channels per 16-byte granule


def _addr_chan(row, col, ldx, n_gran):
    """the region's byte address of one channel element -- a linear tile, no swizzle."""
    gran, within = divmod(col, K_MMA_M_CH)
    return row * ldx * 2 + gran * 16 + within * 2


def _addr_mass(row, ldx, dv):
    return row * ldx * 2 + dv * 2


def _delta(w, r, slot):
    """the row map's step per slot bit: the IDENTITY under the canonical layout, and
    this reads it off the composed derivation rather than assuming it."""
    return _row_of(w, r, 0, slot) ^ _row_of(w, r, 0, 0)


def _ldm_trans(addr_of_lane, mem):
    """`ldmatrix.m8n8.x4.trans`: lane L supplies row (L & 7) of matrix (L >> 3); lane L's
    register i is matrix i's column (L >> 2) at rows 2*(L & 3) and 2*(L & 3) + 1."""
    a = [addr_of_lane(L) for L in range(32)]
    out = []
    for L in range(32):
        regs = []
        for i in range(4):
            lo = mem[a[8 * i + 2 * (L & 3)] + 2 * (L >> 2)]
            hi = mem[a[8 * i + 2 * (L & 3) + 1] + 2 * (L >> 2)]
            regs.append((lo, hi))
        out.append(regs)
    return out


@pytest.mark.parametrize("idx,row", UNTIED, ids=[f"arm{i}" for i, _ in UNTIED])
def test_the_hybrid_returns_every_hi_half_to_the_register_that_published_it(idx, row):
    """THE WHOLE TRANSIT, SIMULATED AT BYTE ADDRESSES.  Publish -> pull -> store back ->
    pull in write layout, with the kernel's own address arithmetic at every step; the
    assertion is that the writer's rejoin register holds the hi halves of ITS OWN two
    accumulator elements, and that the store-back leg is the identity on the region."""
    w, r, dv = _sides(row)
    ldx, n_gran, leaf = dv + K_PAD_E, dv // K_MMA_M_CH, w.leaf
    mch, knl, kdw = dv // K_MMA_M, leaf // K_MMA_N, leaf // K_MMA_K
    d1, d2, d4 = (_delta(w, r, 1), _delta(w, r, 2), _delta(w, r, 4))

    mem, published = {}, {}
    for stream in range(w.ns):
        base = _row_of(w, r, stream, 0)
        for lane in range(32):
            lane_row = base ^ (d2 if lane & 1 else 0) ^ (d4 if lane & 2 else 0)
            for j in range(knl):
                for e in range(4):
                    dl = _delta(w, r, j * K_MMA_N + (e & 1))
                    for m in range(mch):
                        gr = 2 * m + (e >> 1)
                        a = (lane_row * ldx * 2 + (lane >> 2) * 2
                             + dl * ldx * 2 + gr * 16)
                        slot = j * K_MMA_N + 2 * (lane & 3) + (e & 1)
                        col = m * K_MMA_M + (lane >> 2) + 8 * (e >> 1)
                        assert a == _addr_chan(_row_of(w, r, stream, slot), col, ldx, n_gran), (
                            f"arm {idx}: the publish's address arithmetic is not the region's map")
                        assert a not in mem, f"arm {idx}: two publishers claim byte {a}"
                        mem[a] = (stream, lane, j, m, e)
                        published[(stream, lane, j, m, e)] = a
                if (lane >> 2) == 0:
                    for e in range(2):
                        dl = _delta(w, r, j * K_MMA_N + e)
                        a = lane_row * ldx * 2 + dv * 2 + dl * ldx * 2
                        assert a == _addr_mass(_row_of(w, r, stream, j * K_MMA_N
                                                       + 2 * (lane & 3) + e), ldx, dv)
                        mem[a] = ("mass", stream, lane, j, e)

    before = dict(mem)

    #: THE PULL, and the STORE-BACK that must be its exact inverse.
    frags = {}
    for rs in range(r.ns):
        for kk in range(kdw):
            for m in range(mch):
                def addr_of(L, rs=rs, kk=kk, m=m):
                    row_i = rs * leaf + ((L >> 3) >> 1) * 8 + (L & 7)
                    return ((row_i + kk * K_MMA_K) * ldx * 2
                            + ((2 * m) | ((L >> 3) & 1)) * 16)
                out = _ldm_trans(addr_of, mem)
                for L in range(32):
                    frags[(rs, L, kk, m)] = out[L]
                # the store-back: the kernel's own address, and it must land the same bits
                for L in range(32):
                    sb_c = (rs * leaf + 2 * (L & 3)) * ldx * 2 + (L >> 2) * 2
                    for i in range(4):
                        rr = kk * K_MMA_K + (i >> 1) * 8
                        a = sb_c + rr * ldx * 2 + (2 * m + (i & 1)) * 16
                        assert mem[a] == out[L][i][0], (
                            f"arm {idx}: the store-back's low half misses its own address")
                        assert mem[a + ldx * 2] == out[L][i][1], (
                            f"arm {idx}: the store-back's high half misses its own address")
                        mem[a], mem[a + ldx * 2] = out[L][i]
    assert mem == before, f"arm {idx}: the two legs are not the identity on the region"

    #: THE WRITER'S PULL, in the WRITE layout, and the rejoin's register map.
    for ws in range(w.ns):
        base = _row_of(w, r, ws, 0)
        for j in range(knl):
            dj = _delta(w, r, j * K_MMA_N)
            for t in range(mch // 2):
                def addr_of(L, base=base, dj=dj, t=t):
                    ret_row = (base ^ (d1 if L & 1 else 0) ^ (d2 if L & 2 else 0)
                               ^ (d4 if L & 4 else 0))
                    return (ret_row + dj) * ldx * 2 + (4 * t + (L >> 3)) * 16
                out = _ldm_trans(addr_of, mem)
                for L in range(32):
                    for i in range(4):
                        g = 4 * t + i
                        m, h = g >> 1, g & 1
                        want = (mem[published[(ws, L, j, m, 2 * h)]],
                                mem[published[(ws, L, j, m, 2 * h + 1)]])
                        assert out[L][i] == want, (
                            f"arm {idx}: writer {ws} lane {L} n-tile {j} granule {g} did not "
                            f"get back its own (m={m}, e={2 * h}, {2 * h + 1})")
