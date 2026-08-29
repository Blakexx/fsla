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

The model mirrors `csrc/rola/src/carry/box.cuh` through `tools/gen_shards.py`'s own
`_BoxPlan`/`_ModeOrder`/`_StreamSpans` -- the SAME mirror the facts rows are derived
from, so a drift between the mirror and the header shows up here and at the device
import, not silently.
"""
import itertools

import pytest

from tools.gen_shards import _K_SIDE_READ, _K_SIDE_WRITE, CARRY_ARMS, _BoxPlan, _ilog2c, _ModeOrder, _StreamSpans

#: the swizzles the kernel ships (`XchMap::swz` / `XchMap::msw` in carry_kernel.cuh).
CHAN_SHIFTS = (3, 4, 6)
MASS_SHIFTS = (3, 5)
K_PAD_E = 8
K_MMA_M, K_MMA_N, K_MMA_K = 16, 8, 16


class _Side:
    """One side's derived carve, with the private order's shifts and the stream grid."""

    def __init__(self, D, B, K, M, MO, side, ns):
        self.D, self.ns = D, ns
        self.bp = _BoxPlan(D, B, K, M)
        self.sv = _StreamSpans(D, B, K, M, MO, side, ns)
        self.ord = _ModeOrder(D, MO, side)
        self.bc = 1
        for l in range(D):
            self.bc *= self.bp.s(l)
        self.leaf = self.bc // ns

    def s(self, l):
        return self.sv.s(l)

    def g(self, l):
        return self.bp.s(l) // self.s(l)

    def shift(self, l):
        return sum(_ilog2c(self.s(i)) for i in range(self.D)
                   if self.ord.rank(i) < self.ord.rank(l))

    def grid_div(self, l):
        d = 1
        for i in range(l + 1, self.D):
            d *= self.g(i)
        return d

    def off(self, l, stream):
        return ((stream // self.grid_div(l)) % self.g(l)) * self.s(l)


def _sides(row):
    D, B, K, M, DV, W, C, CV, MO, WPS, NSW, NSR, PK = row
    return (_Side(D, B, K, M, MO, _K_SIDE_WRITE, NSW),
            _Side(D, B, K, M, MO, _K_SIDE_READ, NSR), DV)


def _row_of(w_side, r_side, stream, slot):
    """publisher `stream`'s write slot -> the exchange's row, reader-major."""
    reader, q = 0, 0
    for l in range(w_side.D):
        d = ((slot >> w_side.shift(l)) & (w_side.s(l) - 1)) + w_side.off(l, stream)
        reader += (d // r_side.s(l)) * r_side.grid_div(l)
        q |= (d % r_side.s(l)) << r_side.shift(l)
    return reader * r_side.leaf + q


def _tied(w_side, r_side):
    return all(w_side.s(l) == r_side.s(l) and w_side.ord.rank(l) == r_side.ord.rank(l)
               for l in range(w_side.D))


def _swz(row, shifts, mask):
    v = 0
    for sh in shifts:
        v ^= row >> sh
    return v & mask


CARVED = [(i, r) for i, r in enumerate(CARRY_ARMS) if r[10] > 1 or r[11] > 1]
UNTIED = [(i, r) for i, r in CARVED if not _tied(*_sides(r)[:2])]
IDS = [f"arm{i}" for i, _ in CARVED]


def test_at_least_one_built_arm_is_tied_and_one_is_not():
    """Neither branch of `kTied` may go untested: the elision needs a tied arm to prove
    its zero, and the transit needs an untied one to prove it moves anything at all."""
    tied = [i for i, r in CARVED if _tied(*_sides(r)[:2])]
    assert tied, "no built arm is tied; the elided exchange is unproved"
    assert UNTIED, "no built arm carries a real transit"


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


def _publish_row_sets(w, r):
    """One `st.shared.b16` covers the lane quad's FOUR rows at EIGHT consecutive
    channels: the rows vary with `lane & 3`, the channels with `lane >> 2`."""
    return [[_row_of(w, r, stream, j * K_MMA_N + 2 * q + e1) for q in range(4)]
            for stream in range(w.ns) for j in range(w.leaf // K_MMA_N) for e1 in (0, 1)]


@pytest.mark.parametrize("idx,row", UNTIED, ids=[f"arm{i}" for i, _ in UNTIED])
def test_the_exchange_swizzle_is_a_bank_permutation(idx, row):
    """MAX-WAY ONE on BOTH legs of the transit, channels and mass.

    `LdX / 2` is `4 (mod 32)` at the built widths, so consecutive rows sit exactly ONE
    four-bank block apart and a bank is `(4*row + 4*granule + word) mod 32`.  The
    publish needs its four rows in four DIFFERENT blocks (they are identical mod 8
    without a swizzle -- `{r, r+16, r+32, r+48}` at the flagship); the pull needs only
    that the swizzle be CONSTANT over an aligned eight, which every shift >= 3 gives."""
    w, r, dv = _sides(row)
    ldx = dv + K_PAD_E
    words = ldx // 2
    n_gran = dv // 8

    for rows in _publish_row_sets(w, r):
        for g in range(n_gran):
            banks = set()
            for row_i in rows:
                gp = g ^ _swz(row_i, CHAN_SHIFTS, n_gran - 1)
                for word in range(4):
                    b = (words * row_i + 4 * gp + word) % 32
                    assert b not in banks, (
                        f"arm {idx}: publish granule {g} conflicts in bank {b} over rows {rows}")
                    banks.add(b)
        banks = set()
        for row_i in rows:
            b = (words * row_i + 2 * n_gran + _swz(row_i, MASS_SHIFTS, 3)) % 32
            assert b not in banks, f"arm {idx}: mass publish conflicts in bank {b}"
            banks.add(b)

    for reader in range(r.ns):
        for kk in range(r.leaf // K_MMA_K):
            for half in (0, 8):
                rows = [reader * r.leaf + kk * K_MMA_K + half + i for i in range(8)]
                assert len({_swz(x, CHAN_SHIFTS, n_gran - 1) for x in rows}) == 1, (
                    f"arm {idx}: the pull's swizzle is not constant over an aligned eight")
                for g in range(n_gran):
                    banks = set()
                    for row_i in rows:
                        gp = g ^ _swz(row_i, CHAN_SHIFTS, n_gran - 1)
                        for word in range(4):
                            b = (words * row_i + 4 * gp + word) % 32
                            assert b not in banks, f"arm {idx}: pull conflicts in bank {b}"
                            banks.add(b)
                for e1 in (0, 1):
                    mrows = [reader * r.leaf + kk * K_MMA_K + half + 2 * q + e1
                             for q in range(4)]
                    banks = set()
                    for row_i in mrows:
                        b = (words * row_i + 2 * n_gran + _swz(row_i, MASS_SHIFTS, 3)) % 32
                        assert b not in banks, f"arm {idx}: mass pull conflicts in bank {b}"
                        banks.add(b)


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


def test_the_shipped_swizzles_are_the_search_result_and_not_a_guess():
    """Both maps were found by SEARCH over shift-XOR forms against the built arms' real
    row sets.  This re-runs that search and asserts the shipped form is in the passing
    set -- so a future arm that breaks it fails HERE, with the alternatives listed,
    rather than as a silent bank stall."""
    sets = [rows for _, row in UNTIED for rows in _publish_row_sets(*_sides(row)[:2])]
    assert sets, "no untied arm to search against"

    def ok(shifts):
        for rows in sets:
            for g in range(8):
                seen = set()
                for row_i in rows:
                    v = (row_i + (g ^ _swz(row_i, shifts, 7))) % 8
                    if v in seen:
                        return False
                    seen.add(v)
        return True

    forms = [c for n in (1, 2, 3) for c in itertools.combinations(range(3, 8), n) if ok(c)]
    assert CHAN_SHIFTS in forms, f"the shipped channel swizzle fails; passing forms: {forms}"
    assert not ok(()), "an unswizzled layout must CONFLICT, or the swizzle is unnecessary"


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
    """the region's byte address of one channel element, swizzle included."""
    gran, within = divmod(col, K_MMA_M_CH)
    return row * ldx * 2 + ((gran ^ _swz(row, CHAN_SHIFTS, n_gran - 1)) * 16) + within * 2


def _addr_mass(row, ldx, dv):
    return row * ldx * 2 + dv * 2 + _swz(row, MASS_SHIFTS, 3) * 4


def _delta(w, r, slot):
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
            swz_lane = _swz(lane_row, CHAN_SHIFTS, n_gran - 1)
            msw_lane = _swz(lane_row, MASS_SHIFTS, 3)
            for j in range(knl):
                for e in range(4):
                    dl = _delta(w, r, j * K_MMA_N + (e & 1))
                    for m in range(mch):
                        gr = (2 * m + (e >> 1)) ^ _swz(dl, CHAN_SHIFTS, n_gran - 1)
                        a = (lane_row * ldx * 2 + (lane >> 2) * 2
                             + dl * ldx * 2 + ((gr ^ swz_lane) * 16))
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
                        a = (lane_row * ldx * 2 + dv * 2 + dl * ldx * 2
                             + ((_swz(dl, MASS_SHIFTS, 3) ^ msw_lane) * 4))
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
                    swz_k = _swz(row_i + kk * K_MMA_K, CHAN_SHIFTS, n_gran - 1)
                    return ((row_i + kk * K_MMA_K) * ldx * 2
                            + (((2 * m) | ((L >> 3) & 1)) ^ swz_k) * 16)
                out = _ldm_trans(addr_of, mem)
                for L in range(32):
                    frags[(rs, L, kk, m)] = out[L]
                # the store-back: the kernel's own address, and it must land the same bits
                swz_sb = _swz(rs * leaf, CHAN_SHIFTS, n_gran - 1)
                for L in range(32):
                    sb_c = (rs * leaf + 2 * (L & 3)) * ldx * 2 + (L >> 2) * 2
                    for i in range(4):
                        rr = kk * K_MMA_K + (i >> 1) * 8
                        gr = (2 * m + (i & 1)) ^ _swz(rr, CHAN_SHIFTS, n_gran - 1)
                        a = sb_c + rr * ldx * 2 + ((gr ^ swz_sb) * 16)
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
                    swz_ret = _swz(ret_row, CHAN_SHIFTS, n_gran - 1)
                    return (ret_row + dj) * ldx * 2 + (
                        ((4 * t + (L >> 3)) ^ swz_ret
                         ^ _swz(dj, CHAN_SHIFTS, n_gran - 1)) * 16)
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


@pytest.mark.parametrize("idx,row", UNTIED, ids=[f"arm{i}" for i, _ in UNTIED])
def test_the_return_legs_two_accesses_are_max_way_one(idx, row):
    """The two NEW accesses, in banks.  The store-back covers four rows two apart inside
    an aligned eight (one swizzle) at eight consecutive channels; the writer's pull hands
    `ldmatrix` EIGHT SCATTERED rows at one granule, and it is the swizzle that has to put
    them in eight different four-bank blocks."""
    w, r, dv = _sides(row)
    n_gran, leaf, words = dv // K_MMA_M_CH, w.leaf, (dv + K_PAD_E) // 2
    mch, knl, kdw = dv // K_MMA_M, leaf // K_MMA_N, leaf // K_MMA_K

    for rs in range(r.ns):
        for kk in range(kdw):
            for i in range(4):
                rows = [rs * leaf + kk * K_MMA_K + (i >> 1) * 8 + 2 * q for q in range(4)]
                for m in range(mch):
                    banks = set()
                    for row_i in rows:
                        gp = (2 * m + (i & 1)) ^ _swz(row_i, CHAN_SHIFTS, n_gran - 1)
                        for word in range(4):
                            b = (words * row_i + 4 * gp + word) % 32
                            assert b not in banks, (
                                f"arm {idx}: the store-back conflicts in bank {b}")
                            banks.add(b)

    for ws in range(w.ns):
        for j in range(knl):
            rows = [_row_of(w, r, ws, j * K_MMA_N + q) for q in range(8)]
            for g in range(n_gran):
                banks = set()
                for row_i in rows:
                    gp = g ^ _swz(row_i, CHAN_SHIFTS, n_gran - 1)
                    for word in range(4):
                        b = (words * row_i + 4 * gp + word) % 32
                        assert b not in banks, (
                            f"arm {idx}: the writer's return pull conflicts in bank {b} "
                            f"over rows {rows}")
                        banks.add(b)
