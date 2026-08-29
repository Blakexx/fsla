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
