"""THE ONE DERIVATION, against the PARENT KERNEL'S OWN COMPILE-TIME CENSUS.

`csrc/rola/src/carry/geom.cuh` computes at runtime what the shipped carry bodies compute
in their templates (`box.cuh`'s `BoxPlan`/`ModeOrder`/`StreamSpans`). No kernel on this
branch consumes the runtime block -- the pipelined body is what will
(`docs/internals/carry/geom.md#on-the-tip`) -- so THIS FILE IS THE PROOF that the block is
the same law: for every arm the parent builds, the derivation's own output is checked
against the census the parent binary reports off the device.

**WHY A GOLDEN TABLE AND A LIVE READ.** The golden table below is the parent's census,
transcribed, so the assertion survives the parent's deletion (`## G1` SKILLS FEEDBACK 3:
the A-side of any parity claim is a pinned tip that will not always exist). While the
parent's arms still build, `test_the_golden_table_is_the_binarys_own` re-reads them off
THIS binary and refuses a golden row that has drifted -- a golden table nobody re-derives
is the stale-expectation antipattern.

**WHAT IS ASSERTED AS THE LAW, NOT AS THE TABLE.** The parity rows are 11 of the parent's
13 arms. The other two are the `D = 4, B = 8` rows, which K50's RESTORED WIDTH FLOOR
(`B_l >= 16`, power of two) makes unlawful: a level narrower than the owner box's span
there is REFUSED BY NAME, and the deep-4 conformance topology moves to the lawful
`(16, 16, 16, 16)` at the same state size (`## G2b`, "THE ONE SCOPE ITEM"). The test names
that ruling in its own message so a reader meets the citation, not a surprise.

Symbols, each at first use: `D` routing levels; `B_l` level `l`'s digit width; `BC` the
leaves one owner (one CTA's box) holds; `DV` the value width; `MO` the per-level mode word
(two bits a level, read then write); `s(l)` the OWNER span; `g(l) = B_l/s(l)` the owner
grid; `sv(l)` a side's WARP SUB-BOX span; `NS` a side's stream count.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rola.ops import carry as carry_ops

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import gen_shards  # noqa: E402 -- the path insert must precede it

#: THE PARENT'S CENSUS, ARM BY ARM, read off `carry_build_stamp` on the all-arms binary at
#: `k31/k35-final` 335c2b9: `{arm index: (BC, owners, write spans, read spans, kXchBytes)}`.
#: The spans are `StreamSpans::s(l)` -- what ONE STREAM of that side holds -- padded past
#: `D` with `B` the way the stamp reports them; the runtime block's `row_span` is the same
#: quantity. `kXchBytes == 0` is the parent's compile-time TIED case.
PARENT_CENSUS = {
    0: (128, 512, [8, 16], [8, 16], 0),
    1: (128, 512, [2, 16], [8, 4], 18432),
    2: (128, 512, [8, 16], [8, 16], 0),
    3: (128, 512, [2, 16], [8, 4], 18432),
    4: (128, 32, [8, 16], [8, 16], 0),
    5: (128, 32, [2, 16], [8, 4], 18432),
    6: (128, 32, [2, 4, 16], [2, 4, 16], 0),
    7: (128, 32, [2, 4, 16], [2, 4, 16], 0),
    8: (128, 32, [2, 2, 4, 8], [2, 2, 4, 8], 0),
    9: (128, 32, [1, 2, 2, 8], [2, 1, 4, 4], 18432),
    10: (128, 512, [8, 16], [8, 16], 0),
    11: (128, 512, [8, 16], [8, 16], 0),
    12: (128, 512, [2, 16], [2, 16], 0),
}

#: the rows K50's restored floor removes: `B = 8` is narrower than the owner box's
#: innermost span (16, the page atom held whole), so the shape refuses them by name.
UNLAWFUL_UNDER_THE_FLOOR = (8, 9)

#: THE ARCH CONSTANTS, mirrored from `geom.cuh` -- three scalars, and the only thing in
#: this file that is a mirror rather than a call. The SET they feed is the shipped
#: generator's, read through `carry_ops.sub_boxes`. -- docs/internals/carry/geom.md#arch-constants
WARPS_PER_SM = 8


def leaves_per_sm(dv: int) -> int:
    return 16384 // dv


def leaves_per_warp(dv: int) -> int:
    return leaves_per_sm(dv) // WARPS_PER_SM


def _arm(i):
    (D, B, K, M, DV, W, C, CV, MO, WPS, NSW, NSR, PK) = gen_shards.CARRY_ARMS[i]
    return dict(D=D, widths=[B] * D, k=K, m=M, dv=DV, window=W, chunk=C, carve=CV, mo=MO,
                nsw=NSW, nsr=NSR)


def _block(i):
    a = _arm(i)
    return carry_ops.geometry(a["widths"], level_modes=a["mo"], bc=PARENT_CENSUS[i][0],
                              nsr=a["nsr"], nsw=a["nsw"], d_v=a["dv"])


LAWFUL = [i for i in PARENT_CENSUS if i not in UNLAWFUL_UNDER_THE_FLOOR]


@pytest.mark.parametrize("arm", LAWFUL)
def test_the_derivation_reproduces_the_parents_census(arm):
    """PARITY, arm by arm: the runtime block IS the compile-time law it replaces."""
    bc, owners, w_spans, r_spans, xch = PARENT_CENSUS[arm]
    g = _block(arm)
    assert g["bc"] == bc
    assert g["owners"] == owners
    assert g["w"]["row_span"] == w_spans, "the WRITE side's stream span vector moved"
    assert g["r"]["row_span"] == r_spans, "the READ side's stream span vector moved"
    assert g["w"]["streams"] == _arm(arm)["nsw"]
    assert g["r"]["streams"] == _arm(arm)["nsr"]
    #: the parent ELIDES the transit at compile time exactly when the two carves
    #: coincide; the block reports the same fact instead of a second code path (K50
    #: deleted the elided body). An uncarved arm has no transit at all, so the parent's
    #: zero says nothing there and the claim is made only where it carves.
    if _arm(arm)["nsw"] > 1:
        assert (xch == 0) == bool(g["identity_map"])


@pytest.mark.parametrize("arm", UNLAWFUL_UNDER_THE_FLOOR)
def test_the_deep_four_rows_are_refused_by_the_restored_width_floor(arm):
    """RULED, not a regression: K50's restored floor makes `B_l = 8` unlawful.

    The parent builds these two rows and carves them; the shape this block owns has no
    narrower slot layout to fall back to, so it REFUSES rather than rounding, and the
    deep-4 conformance topology is the lawful `(16, 16, 16, 16)` at the same state size
    (`## G2b`, "THE ONE SCOPE ITEM I HAD TO SETTLE")."""
    with pytest.raises(RuntimeError, match="narrower than the owner box's span"):
        _block(arm)


def test_the_lawful_deep_four_topology_is_admitted_where_the_unlawful_one_is_not():
    g = carry_ops.geometry([16, 16, 16, 16], level_modes=102, bc=128, nsr=4, nsw=4, d_v=64)
    assert g["span"] == [2, 2, 2, 16], "the atom is held WHOLE innermost"
    assert g["leaves"] == 65536


@pytest.mark.parametrize("arm", LAWFUL)
def test_the_block_obeys_its_own_laws_not_just_its_table(arm):
    """The laws, restated as assertions: spans, grids, bases, weights, slots, streams."""
    g = _block(arm)
    D, bc = g["D"], g["bc"]
    prod = 1
    for s in g["span"]:
        prod *= s
    assert prod == bc, "prod_l s(l) = BC"
    for lvl in range(D):
        assert g["width"][lvl] % g["span"][lvl] == 0, "s(l) divides B_l"
        assert g["grid"][lvl] == g["width"][lvl] // g["span"][lvl], "g(l) = B_l / s(l)"
    assert g["col_base"] == [sum(g["width"][:lvl]) for lvl in range(D)], "prefix sum of B_l"
    weight = [1] * D
    for lvl in range(D - 2, -1, -1):
        weight[lvl] = weight[lvl + 1] * g["width"][lvl + 1]
    assert g["weight"] == weight, "weight(l) = prod_{l' > l} B_l'"
    assert g["wtot"] == sum(g["width"])
    for side in ("r", "w"):
        s = g[side]
        leaf, streams = 1, 1
        for lvl in range(D):
            leaf *= 1 << s["sub_bits"][lvl]
            streams *= s["grid"][lvl]
            assert s["row_span"][lvl] == (1 << s["sub_bits"][lvl]), "row_span IS sv(l)"
            assert s["grid"][lvl] * s["row_span"][lvl] == g["span"][lvl]
        assert leaf == s["leaf"], "prod_l sv(l) = the leaves one stream owns"
        assert streams == s["streams"], "prod_l s(l)/sv(l) = NS"
        assert leaf * streams == bc, "the streams partition the owner's box"
        #: the slot layout is a BIJECTION of the slot's bits: each level's digits occupy
        #: `sub_bits[l]` bits at `sub_shift[l]`, disjointly, covering `log2 leaf`.
        used = 0
        for lvl in range(D):
            mask = ((1 << s["sub_bits"][lvl]) - 1) << s["sub_shift"][lvl]
            assert used & mask == 0, "two levels claim the same slot bit"
            used |= mask
        assert used == leaf - 1, "the slot bits are exactly covered"
        assert sorted(s["rank"]) == list(range(D))
        assert [s["level_at"][s["rank"][lvl]] for lvl in range(D)] == list(range(D))


@pytest.mark.parametrize("arm", LAWFUL)
def test_each_sides_transit_row_map_is_a_bijection_of_the_region(arm):
    """THE ROW MAP. Every leaf of `[0, BC)` is reached exactly once, by each side."""
    g = _block(arm)
    for side in ("r", "w"):
        rows = [row for stream in g["xch"][side] for row in stream]
        assert sorted(rows) == list(range(g["bc"])), f"the {side} map is not a bijection"
    same = g["xch"]["r"] == g["xch"]["w"]
    assert same == bool(g["identity_map"]), (
        "`identity_map` is the FACT that the two sides' maps coincide, and it disagreed "
        "with the maps themselves")


def test_a_width_that_is_not_a_power_of_two_is_refused():
    with pytest.raises(RuntimeError, match="power of two"):
        carry_ops.geometry([24, 256], level_modes=6, bc=128, nsr=4, nsw=4, d_v=64)


def test_a_topology_too_small_for_the_owner_box_is_refused():
    """`Pi_l B_l < BC` cannot hold a box; the refusal names the level that cannot."""
    with pytest.raises(RuntimeError, match="narrower than the owner box's span"):
        carry_ops.geometry([16, 4], level_modes=0, bc=128, nsr=1, nsw=1, d_v=64)


def test_a_stream_count_that_is_not_a_power_of_two_is_refused():
    with pytest.raises(RuntimeError, match="power of two"):
        carry_ops.geometry([256, 256], level_modes=6, bc=128, nsr=3, nsw=4, d_v=64)


#: THE GENERATED SET -- R9's layer 1. `sub_boxes(D, box_leaves, workers)` runs the
#: top-down carve over EVERY level order and de-duplicates; these are the counts the
#: switch's `NA` is, and they are asked of the generator, never of a written list.
@pytest.mark.parametrize("D,count", [(1, 1), (2, 2), (3, 4), (4, 7)])
def test_the_sub_box_set_is_generated_and_deduplicated(D, count):
    dv = 64
    box = leaves_per_warp(dv) * 4  # the CTA box these arms compile: four warps
    members = carry_ops.sub_boxes(D, box, 4)
    assert len(members) == count, f"the generated set at D={D} is not {count} shapes"
    assert len(set(members)) == len(members), "the generator returned a duplicate shape"
    for shape in members:
        prod = 1
        for s in shape:
            prod *= s
        assert prod * 4 == box, "a member times the worker count IS the box"


def test_the_shape_set_is_the_arch_law_at_one_cta_per_sm():
    """`sb_matches_arch_law`, asserted where it is EVALUABLE (G2c's own design point).

    The set is a property of `(D, leaves_per_warp, warps_per_cta)` with THE CTA'S BOX as
    the unit, and it equals the SM-grain set exactly when the CTA owns the SM
    (`warps_per_cta == warps_per_sm`, R15's one CTA per SM). At the `kWarps = 4` these
    arms compile it is the same law run on HALF the SM box, which is a different set --
    asserted here too, so the two grains cannot be confused for each other."""
    dv = 64
    sm_box = leaves_per_sm(dv)
    for D in (1, 2, 3, 4):
        at_sm = carry_ops.sub_boxes(D, sm_box, WARPS_PER_SM)
        by_cta = carry_ops.sub_boxes(D, leaves_per_warp(dv) * WARPS_PER_SM, WARPS_PER_SM)
        assert at_sm == by_cta, (
            "at one CTA per SM the CTA-box law and the SM-box law are the same set")
    half = carry_ops.sub_boxes(2, leaves_per_warp(dv) * 4, 4)
    assert half != carry_ops.sub_boxes(2, sm_box, WARPS_PER_SM), (
        "the four-warp CTA's set is the same law on a DIFFERENT box, and is not the "
        "SM-grain set (G2c: `[16,2]` does not fit a box eight wide at level 0)")


def test_the_golden_table_is_the_binarys_own():
    """The golden rows, RE-DERIVED off this binary while the parent's arms still build.

    A checked-in expectation nobody re-derives is the stale-expectation antipattern; this
    is the re-derivation, and it skips (never passes vacuously) for an arm the build does
    not carry."""
    from tests.oracle.fixtures import assert_fresh_binary, built_arms
    assert_fresh_binary()
    built = built_arms()
    checked = 0
    for i, (bc, owners, w_spans, r_spans, xch) in PARENT_CENSUS.items():
        a = _arm(i)
        key = (a["D"], a["widths"][0], a["k"], a["m"], a["dv"], a["window"], a["chunk"],
               a["carve"], a["mo"])
        if key not in built:
            continue
        st = carry_ops.build_stamp(a["widths"], a["k"], a["m"], a["window"], a["chunk"],
                                   a["carve"], a["mo"], a["dv"])
        assert (st[9], st[10]) == (bc, owners), f"arm {i}: BC/owners drifted from the golden"
        assert list(st[16:16 + a["D"]]) == w_spans, f"arm {i}: the golden write spans drifted"
        assert list(st[20:20 + a["D"]]) == r_spans, f"arm {i}: the golden read spans drifted"
        assert st[24] == xch, f"arm {i}: the golden transit bytes drifted"
        checked += 1
    assert checked, "no declared arm was built: the golden table was not re-derived at all"
