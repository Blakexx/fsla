# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""K31 R1 -- the carry kernel against the fp64 reference.

THE GATE IS THE fp64 ORACLE, NOT THE SHIPPED KERNEL.  There is no bit-identity
target across this redesign: the summation orders differ by construction (a global
window grid instead of per-CTA super-chunks, and a two-MMA loop instead of the
fused drain), so the only defensible reference is the arithmetic itself.  The band
is the repository's one bf16 family bound.

Milestones, each a named test:
  M1  the lattice state I/O and the fold          `test_m1_*`
  M2  + the readout and the RED fan-in            `test_m2_*`
  M3  + random-k_tok, TWO-SIDED schedule          `test_m3_*`
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

from rola.ops import carry as carry_ops
from tests.k31 import reference
from tests.k31.fixtures import Cell, assert_fresh_binary, canonical_from_lattice, relative, require_arm
from tests.oracle.tolerances import BF16_RTOL

#: THE DECLARED ARM TABLE, IMPORTED AND NEVER RESTATED (K36).  The rows live in
#: `tools/gen_shards.py`, which generates both the dispatch's X-macro and the per-arm
#: translation units; `tools/` is not a package, so it goes on the path the way
#: `tests/oracle/instantiation.py` puts it there.
_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import gen_shards  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

#: (widths, k, m, W) -- the built arms this suite drives.
SMALL = ((64, 64), 4, 8, carry_ops.WINDOW_SMALL)  # N = 4096,  BC = 128, owners = 32
FLAG = ((256, 256), 4, 8, carry_ops.WINDOW)       # N = 65536, BC = 128, owners = 512
#: THE DEEP ARMS.  Both hold `BC = 128` at `N = 4096`; they differ from the flat
#: arms only in how those 128 leaves cut into digits, and `DEEP4` is the topology
#: whose atom rectangle cannot be one level -- `B = 8` is narrower than an atom, so
#: an atom is level 3's whole span times two of level 2's digits.
DEEP3 = ((16, 16, 16), 2, 16, carry_ops.WINDOW_SMALL)   # s = (2, 4, 16),   owners = 32
DEEP4 = ((8, 8, 8, 8), 2, 8, carry_ops.WINDOW_SMALL)    # s = (2, 2, 4, 8), owners = 32
#: THE MULTI-LEVEL CARVE ARM (K35 S1).  The owner box is the WHOLE topology --
#: `s = (4, 4, 4, 4)`, one owner -- and each SIDE carves TWO levels of it: write spans
#: `[1, 4, 2, 4]`, read spans `[4, 1, 4, 2]`, eight streams of 32 leaves either way and
#: one warp per stream.  Under `ALT4` each side carves exactly where it is sparse.
CARVE4 = ((4, 4, 4, 4), 4, 1, carry_ops.WINDOW_SMALL)
CARVE4_KEY = (4, 4, 4, 1, 64, carry_ops.WINDOW_SMALL, 32, 1)
#: THE 1x8 CONTAINER ARMS (K35 S3a).  The SAME cell as `flagship ALT` -- box `[8, 16]`,
#: `W = 384`, the ALT declaration -- reached with EIGHT executors instead of four, by
#: the two routes the card leaves to measurement: four write streams column-split two
#: ways (`SW = [2,16]`, WPS = 2) and eight write streams one warp each (`SW = [1,16]`,
#: WPS = 1).  Both read-carve at eight streams (`SR = [8,2]`), because the read side is
#: declared at ONE STREAM PER WARP.
CONT_SPLIT = carry_ops.CONTAINER_1X8_SPLIT
CONT_STREAMS = carry_ops.CONTAINER_1X8_STREAMS
D_V = carry_ops.D_V
WIDE = carry_ops.D_V_WIDE
CONT_SPLIT_KEY = (2, 256, 4, 8, D_V, carry_ops.WINDOW, 32, CONT_SPLIT)
CONT_STREAMS_KEY = (2, 256, 4, 8, D_V, carry_ops.WINDOW, 32, CONT_STREAMS)
#: THE `d_v = 128` ARMS (K35 S3b).  `d_v` is the third dial and part of an arm's key, so
#: the two container declarations reappear at the wide value plane -- SAME cell, SAME
#: box `[8, 16]`, sixteen leaves per warp either route -- and a THIRD container joins
#: them: 2x4 at `BC = 64`, which halves the CTA's box instead of doubling its warps and
#: reaches sixteen leaves per warp under the ruled one-warp-per-stream map.  `BC = 64`
#: is `(k, m) = (4, 4)` on the flagship topology: spans `[4, 16]`, 1024 owners.
WIDE_SPLIT_KEY = (2, 256, 4, 8, WIDE, carry_ops.WINDOW, 32, CONT_SPLIT)
WIDE_STREAMS_KEY = (2, 256, 4, 8, WIDE, carry_ops.WINDOW, 32, CONT_STREAMS)
WIDE_2X4_KEY = (2, 256, 4, 4, WIDE, carry_ops.WINDOW, 32, carry_ops.LEAF_SPLIT)
#: THE PARK'S TWINS (K35 S5).  `PARKED` is BIT 2 of the declaration scalar, so a parked
#: arm is the same cell, box and executor map as the row it twins -- what differs is
#: that the write-view `lo` spends the readout in SMEM instead of in registers (P76 F5).
#: The three wide containers are the arms the park exists for and the flagship `d_v = 64`
#: ALT arm is the control.
PARK = carry_ops.PARKED
PARK_KEYS = {(2, 256, 4, 8, WIDE, carry_ops.WINDOW, 32, CONT_SPLIT + PARK),
             (2, 256, 4, 8, WIDE, carry_ops.WINDOW, 32, CONT_STREAMS + PARK),
             (2, 256, 4, 4, WIDE, carry_ops.WINDOW, 32, carry_ops.LEAF_SPLIT + PARK),
             (2, 256, 4, 8, D_V, carry_ops.WINDOW, 32, carry_ops.LEAF_SPLIT + PARK)}
#: THE DENSE DECLARATIONS ON THE CARVED BODY (K35 S1c).  Rows 11 and 13 again -- the
#: flagship cell, the 1x8 column split -- declared DENSE_BOTH instead of ALT, at both
#: widths.  They are the only rows in the list that are dense AND carved, which is what
#: makes the carved body measurable at a dense cell; the seam still serves a production
#: dense declaration from the CHANNEL-SPLIT body and these rows change no dispatch.
DENSE_CARVED_KEYS = {CONT_SPLIT_KEY, WIDE_SPLIT_KEY}
#: (widths, k, m, W) for the `BC = 64` box, in the shape `run`/`ref` read.
FLAG_HALF = ((256, 256), 4, carry_ops.CAP_BC64, carry_ops.WINDOW)
#: the ALTERNATING declaration packed as the seam's word, at depth four.
ALT4 = 102

CARVE = carry_ops.LEAF_SPLIT


def run(cell, arm, carve, chunk=32, state_in=None):
    widths, k, m, W = arm
    require_arm(widths, k, m, W, chunk, carve, 0, d_v=cell.d_v)
    read, write, g, v = cell.ref_levels()
    num, den, plane = carry_ops.carry_inter(
        cell.read_bf, cell.write_bf, cell.g_bf, cell.v_bf, widths,
        k=k, m=m, window=W, chunk=chunk, carve=carve, state_in=state_in)
    return num, den, plane


def ref(cell, arm, state_in=None):
    widths, k, m, W = arm
    read, write, g, v = cell.ref_levels()
    return reference.inter_reference(read, write, g, v, widths, W, state_in=state_in)


# ---------------------------------------------------------------- the binary

def test_build_stamp_is_this_source():
    stamp = assert_fresh_binary()
    assert stamp[3] > 0, "the arm census must report a register count"


def test_arms_are_the_declared_solver_rows():
    """The TOPOLOGY rows, and the declarations each is built for.

    WHAT THE BINARY REPORTS IS A SUBSET (K36): a default build compiles the SHIPPED
    rows only, so `arms()` is asserted to be declared rows in the declared order, and
    the table's own properties are asked of `gen_shards.CARRY_ARMS`.

    An arm is `(topology, d_v, declaration)`: the declaration chooses the BODY, so the
    same topology appears once per declaration built for it.  DENSE_BOTH is built
    for every topology it is the seam's answer FOR -- which is every one but the
    multi-level carve arm, whose box is the whole topology and whose carve is the
    point of the arm.  A declaration with no body built for it is REFUSED at every
    topology (K35 R-B): DENSE_BOTH is served when it is DECLARED and never as a
    substitute for something else.  The DECLARATION SCALAR (`a[6]`) names the arm's (carve, executor map),
    so the two 1x8 container rows share the flagship's topology and window and differ
    from it and from each other only there (`a[7]`).  `d_v` (`a[4]`) is part of the key too
    (K35 S3b), so the two container declarations appear once per built width and the
    2x4 container is a row of its own -- same topology, a different capacity.  BIT 2 of
    the declaration is the PARK (K35 S5), so each parked twin is a row whose key differs
    from its own twin's in that bit alone.
    """
    declared = [tuple(row[:9]) for row in gen_shards.CARRY_ARMS]
    reported = [tuple(a[:9]) for a in carry_ops.arms()]
    unknown = [row for row in reported if row not in declared]
    assert not unknown, f"the binary reports rows that are not declared: {unknown}"
    assert reported == [row for row in declared if row in set(reported)], (
        "the built rows must be the declared rows, in the declared order")
    #: THE PROPERTIES BELOW ARE THE TABLE'S, so they are asked of the DECLARED rows and
    #: not of the subset a build carries: a default build compiles the SHIPPED rows only
    #: (K36), and "every other topology carries the DENSE_BOTH body" is a statement about
    #: the list rather than about which of its arms a given `.so` was asked for.
    keys = {row[:8] for row in declared}
    carved_only = {CARVE4_KEY, CONT_SPLIT_KEY, CONT_STREAMS_KEY,
                   WIDE_SPLIT_KEY, WIDE_STREAMS_KEY, WIDE_2X4_KEY} | PARK_KEYS
    dense = {row[:8] for row in declared if row[8] == 0}
    assert keys - carved_only <= dense, (
        "every other topology must carry the DENSE_BOTH body")
    #: AND TWO CARVED KEYS ARE DENSE TOO (K35 S1c): a declaration being DENSE_BOTH no
    #: longer implies the channel-split body.  The dense rows are now every uncarved key
    #: PLUS exactly the two the compare exists for, named rather than counted so a third
    #: appearing is a failure and not a silently widened set.
    assert dense - (keys - carved_only) == DENSE_CARVED_KEYS, (
        "the only DENSE_BOTH rows on the carved body are S1c's compare arms")
    #: THE WIDE PLANE IS THE CHANNEL-SPLIT BODY'S REFUSAL, by that body's own arithmetic:
    #: it gives each of a fixed four warps exactly one MMA m-tile of channels, so its
    #: width IS its executor count and 128 is not a declaration it can take.  The law is
    #: about the BODY, so it is asked of the body -- `a[12] == -1` is the arm whose read
    #: side carves nothing -- and not of the declaration, which no longer selects it.
    assert all(a[4] == D_V for a in carry_ops.arms() if a[12] == -1)
    for arm in carry_ops.arms():
        D, B, k, m, d_v, W, C, carve, modes, BC, owners, smem, level = arm
        assert d_v in (D_V, WIDE)
        _, spans_host, _, bc_host, owners_host = carry_ops.box_shape([B] * D, k, m)
        assert (BC, owners) == (bc_host, owners_host), (
            "the device box and the host solver disagree about the owner's shape")
        #: the carve level is the innermost level the READ side carves, or -1 for the
        #: channel-split body.  IT IS READ OFF THE DERIVED VECTOR (K35 S1b) and no longer
        #: assumed to be the innermost level the read side is SPARSE at: under ORDER BY
        #: MODE the read side carves the OUTERMOST levels of its own order, which is a
        #: different level whenever a BOTH_SPARSE level sits outside a dense one -- the
        #: tied arm carves level 0 while being dense at level D-1.  Comparing the seam's
        #: reported level against the census's reported vector also makes this a
        #: cross-check of two derivations rather than a restatement of one.
        sr = list(carry_ops.build_stamp([B] * D, k, m, W, C, carve, modes, d_v))[20:24]
        carved = [l for l in range(D) if sr[l] < int(spans_host[l])]
        assert level == (max(carved) if carved else -1), (
            f"the seam reports carve level {level}, but the derived read vector "
            f"{sr[:D]} against the owner box {list(spans_host)} carves {carved}")


def test_s3b_the_wide_arms_are_sixteen_leaves_per_warp():
    """THE CARD'S CONTAINERS CLAUSE, asserted off the device rather than commented.

    At ``d_v = 128`` sixteen leaves per warp is REQUIRED (P76 CONTAINERS), and the three
    built containers are the three ways to get there: 1x8 at ``BC = 128`` by a column
    split, 1x8 at ``BC = 128`` by eight write streams, and 2x4 at ``BC = 64`` by halving
    the CTA's box.  ``stamp[7] // 32`` is the executor budget, ``stamp[14]`` the WRITE
    stream count and ``stamp[15]`` the arm's ``d_v``, so all three dials are read back
    from the binary and the leaves-per-warp claim is arithmetic on them.
    """
    assert_fresh_binary()
    want = {(FLAG, CONT_SPLIT): (8, 4, 2), (FLAG, CONT_STREAMS): (8, 8, 1),
            (FLAG_HALF, CARVE): (4, 4, 1)}
    for (arm, declaration), (warps, streams, wps) in want.items():
        widths, k, m, W = arm
        _, _, _, bc, _ = carry_ops.box_shape(list(widths), k, m)
        stamp = carry_ops.build_stamp(widths, k, m, W, 32, declaration, 6, d_v=WIDE)
        assert stamp[15] == WIDE, f"{arm} {declaration}: the arm's d_v"
        assert stamp[7] // 32 == warps and stamp[14] == streams
        assert stamp[7] // 32 == stamp[14] * wps
        assert bc // warps == 16, "the wide plane's executor holds sixteen leaves"
    #: and the SAME cell at the narrow plane still holds thirty-two, which is what says
    #: the width moved the box rather than the arm list gaining an unrelated row.
    stamp = carry_ops.build_stamp(FLAG[0], FLAG[1], FLAG[2], FLAG[3], 32, CONT_SPLIT, 6)
    assert stamp[15] == D_V and 128 // (stamp[7] // 32) == 16


# ---------------------------------------------------------------- the lattice

def test_lattice_is_a_permutation():
    for widths, k, m in ((( 64, 64), 4, 8), ((64, 64), 4, 4), ((256, 256), 4, 8)):
        pi = carry_ops.lattice_of_canonical(widths, k, m)
        assert torch.equal(torch.sort(pi).values,
                           torch.arange(math.prod(widths), device=pi.device))
        #: an owner's leaves are ONE contiguous BC-aligned run, BY CONSTRUCTION
        #: (LATTICE-1) rather than by static_assert.
        _, s, g, bc, owners = carry_ops.box_shape(widths, k, m)
        owner_of = pi // bc
        assert int(owner_of.max()) == owners - 1


# ------------------------------------------------------- M1: state I/O + fold

def test_m1_state_io_and_fold_dense(carve=CARVE):
    """The final state alone, on the tiny cell.  A leaf-order error surfaces here
    as a PERMUTATION rather than as noise, which is why the state is gated first."""
    assert_fresh_binary()
    cell = Cell(SMALL[0], T=256, k_tok=None, seed=1)
    _, _, plane = run(cell, SMALL, carve)
    _, _, s_ref = ref(cell, SMALL)
    got = canonical_from_lattice(plane, SMALL[0], SMALL[1], SMALL[2])
    want = s_ref.reshape(cell.B * cell.H, cell.N, cell.d_v + 1)
    assert relative(got, want) < BF16_RTOL


def test_m1_entry_state_is_carried(carve=CARVE):
    assert_fresh_binary()
    cell = Cell(SMALL[0], T=128, k_tok=None, seed=2, state_in=True)
    s_lat = cell.state_in_lattice(SMALL[1], SMALL[2])
    _, _, plane = run(cell, SMALL, carve, state_in=s_lat)
    s0 = cell.state_in.reshape(cell.B, cell.H, cell.N, cell.d_v + 1)
    _, _, s_ref = ref(cell, SMALL, state_in=s0)
    got = canonical_from_lattice(plane, SMALL[0], SMALL[1], SMALL[2])
    want = s_ref.reshape(cell.B * cell.H, cell.N, cell.d_v + 1)
    assert relative(got, want) < BF16_RTOL


# ------------------------------------------ M2: readout + fan-in, dense cells

def test_m2_inter_dense_small(carve=CARVE):
    assert_fresh_binary()
    cell = Cell(SMALL[0], T=256, k_tok=None, seed=3)
    num, den, plane = run(cell, SMALL, carve)
    n_ref, d_ref, s_ref = ref(cell, SMALL)
    assert relative(num.reshape(cell.B, cell.H, cell.T, cell.d_v).permute(0, 2, 1, 3),
                    n_ref) < BF16_RTOL
    assert relative(den.reshape(cell.B, cell.H, cell.T).permute(0, 2, 1), d_ref) < BF16_RTOL
    got = canonical_from_lattice(plane, SMALL[0], SMALL[1], SMALL[2])
    assert relative(got, s_ref.reshape(cell.B * cell.H, cell.N, cell.d_v + 1)) < BF16_RTOL


def test_m2_decomposition_is_exact():
    """`inter(W) + intra(W)` is the SAME full recurrence for every `W`, and equals
    the canonical oracle.  This is the study's no-double-count claim, and it is the
    reason the inter-only kernel can be gated at all."""
    cell = Cell(SMALL[0], T=128, k_tok=None, seed=4)
    read, write, g, v = cell.ref_levels()
    widths = SMALL[0]
    tot = None
    for W in (1, 8, 64):
        ni, di, _ = reference.inter_reference(read, write, g, v, widths, W)
        na, da = reference.intra_reference(read, write, g, v, widths, W)
        cur = (ni + na, di + da)
        if tot is None:
            tot = cur
        else:
            assert relative(cur[0], tot[0]) < 1e-12
            assert relative(cur[1], tot[1]) < 1e-12
    #: and against the repository's canonical fp64 oracle, through its ratio.
    from rola.ops.constants import READOUT_EPS
    from rola.ops.naive import naive_rola
    from rola.routing.types import IndependentRouting, SoftmaxActivation, Topology
    routing = IndependentRouting(width=1, tied=False, read=SoftmaxActivation(),
                                 write=SoftmaxActivation())
    topo = Topology(levels=tuple(routing.at(w) for w in widths))
    y, _ = naive_rola(v, read, write, None, g, topo, None, "global",
                      output_final_state=False)
    mine = tot[0] / (tot[1][..., None] + READOUT_EPS)
    assert relative(mine, y) < 1e-10


def test_m2_inter_dense_flagship(carve=CARVE):
    """The flagship topology at `L = 1024`.  `N = 65536` is the largest cell whose
    fp64 reference fits: `[B,T,H,N]` fp64 is 537 MB at this `L`, and the reference
    holds one window's worth at a time."""
    assert_fresh_binary()
    cell = Cell(FLAG[0], T=1024, k_tok=None, seed=5)
    num, den, _ = run(cell, FLAG, carve)
    n_ref, d_ref, _ = ref(cell, FLAG)
    assert relative(num.reshape(cell.B, cell.H, cell.T, cell.d_v).permute(0, 2, 1, 3),
                    n_ref) < BF16_RTOL
    assert relative(den.reshape(cell.B, cell.H, cell.T).permute(0, 2, 1), d_ref) < BF16_RTOL


# ------------------------------------------------ M3: the TWO-SIDED schedule

def clustered(shape, k_tok, cohort, gen, device):
    """A simplex draw whose support is ONE contiguous digit window per run of
    ``cohort`` consecutive tokens -- the structure the whole-window skip exists to
    exploit, and the one that leaves an owner with a handful of live tokens per
    window instead of a full batch."""
    B, T, H, width = shape
    x = torch.rand(shape, device=device, dtype=torch.float64, generator=gen)
    starts = torch.randint(0, width, (B, T // cohort, 1, 1), device=device, generator=gen)
    idx = (starts.expand(B, T // cohort, cohort, 1).reshape(B, T, 1, 1)
           + torch.arange(k_tok, device=device).view(1, 1, 1, k_tok)) % width
    keep = torch.zeros(shape, device=device, dtype=torch.float64)
    keep.scatter_(-1, idx.expand(B, T, H, k_tok), 1.0)
    x = x * keep
    return x / x.sum(-1, keepdim=True)


def alt_cell(widths, T, k_tok, seed, cohort=None, d_v=carry_ops.D_V):
    """THE PRIMARY SPARSE STRUCTURE (Blake ruling 2026-08-19): per level at most ONE
    sparse side, and the sparse side ALTERNATES -- EVEN levels write-sparse/read-dense,
    ODD levels read-sparse/write-dense.  The two sides' clause sets are therefore
    DIFFERENT level subsets, which is exactly what the two-sided schedule exists for
    and what a union-routed cell would never exercise.

    The alternation is stated over ``D`` levels rather than over two, so a deep arm
    exercises it at every depth; at ``D = 2`` it is the ruling verbatim."""
    from tests.k31.fixtures import simplex
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)
    cell = Cell(widths, T=T, k_tok=None, seed=seed, d_v=d_v)
    shape = (cell.B, T, cell.H)

    def draw(width, kt):
        if kt is None or kt >= width:
            return simplex(shape + (width,), None, gen, device)
        if cohort is None:
            return simplex(shape + (width,), kt, gen, device)
        return clustered(shape + (width,), kt, cohort, gen, device)

    read = tuple(draw(w, k_tok if l % 2 else None) for l, w in enumerate(widths))
    write = tuple(draw(w, None if l % 2 else k_tok) for l, w in enumerate(widths))
    cell.read, cell.write = read, write
    cell.read_bf = tuple(t.to(torch.bfloat16) for t in read)
    cell.write_bf = tuple(t.to(torch.bfloat16) for t in write)
    return cell


@pytest.mark.parametrize("k_tok", [4, 16])
def test_m3_two_sided_alt_small(k_tok, carve=CARVE):
    assert_fresh_binary()
    cell = alt_cell(SMALL[0], T=256, k_tok=k_tok, seed=10 + k_tok)
    num, den, plane = run(cell, SMALL, carve)
    n_ref, d_ref, s_ref = ref(cell, SMALL)
    assert relative(num.reshape(cell.B, cell.H, cell.T, cell.d_v).permute(0, 2, 1, 3),
                    n_ref) < BF16_RTOL
    assert relative(den.reshape(cell.B, cell.H, cell.T).permute(0, 2, 1), d_ref) < BF16_RTOL
    got = canonical_from_lattice(plane, SMALL[0], SMALL[1], SMALL[2])
    assert relative(got, s_ref.reshape(cell.B * cell.H, cell.N, cell.d_v + 1)) < BF16_RTOL


@pytest.mark.parametrize("k_tok", [4, 16])
def test_m3_two_sided_alt_flagship(k_tok):
    assert_fresh_binary()
    cell = alt_cell(FLAG[0], T=1024, k_tok=k_tok, seed=20 + k_tok)
    num, den, _ = run(cell, FLAG, CARVE)
    n_ref, d_ref, _ = ref(cell, FLAG)
    assert relative(num.reshape(cell.B, cell.H, cell.T, cell.d_v).permute(0, 2, 1, 3),
                    n_ref) < BF16_RTOL
    assert relative(den.reshape(cell.B, cell.H, cell.T).permute(0, 2, 1), d_ref) < BF16_RTOL


@pytest.mark.parametrize("k_tok", [4, 16])
def test_m3_two_sided_alt_clustered(k_tok):
    """The CLUSTERED cell: the support is one contiguous digit window per run of 64
    consecutive tokens, so most (owner, window) pairs are side-dead and the live ones
    carry a PARTIAL batch.  That is the whole-window skip and the dead-row zeroing law
    exercised together, and neither is reached by the scattered cells."""
    assert_fresh_binary()
    cell = alt_cell(FLAG[0], T=1024, k_tok=k_tok, seed=40 + k_tok, cohort=64)
    num, den, _ = run(cell, FLAG, CARVE)
    n_ref, d_ref, _ = ref(cell, FLAG)
    assert relative(num.reshape(cell.B, cell.H, cell.T, cell.d_v).permute(0, 2, 1, 3),
                    n_ref) < BF16_RTOL
    assert relative(den.reshape(cell.B, cell.H, cell.T).permute(0, 2, 1), d_ref) < BF16_RTOL


def test_m3_lists_differ_when_the_sides_do():
    """NON-VACUITY: the ALT cell must actually drive `nr != nw` somewhere, or the
    two-sided gate above is a dense gate wearing a sparse name."""
    cell = alt_cell(SMALL[0], T=256, k_tok=4, seed=99)
    widths, k, m, W = SMALL
    _, s, g, bc, owners = carry_ops.box_shape(widths, k, m)
    read = tuple(t.double() for t in cell.read_bf)
    write = tuple(t.double() for t in cell.write_bf)
    diff = 0
    for owner in range(owners):
        o = [(owner // math.prod(g[l + 1:])) % g[l] for l in range(len(widths))]
        liveR = torch.ones(cell.T, dtype=torch.bool, device="cuda")
        liveW = torch.ones(cell.T, dtype=torch.bool, device="cuda")
        for l in range(len(widths)):
            sl = slice(o[l] * s[l], (o[l] + 1) * s[l])
            liveR &= (read[l][0, :, 0, sl] != 0).any(-1)
            liveW &= (write[l][0, :, 0, sl] != 0).any(-1)
        diff += int((liveR != liveW).sum())
    assert diff > 0, "the ALT cell did not separate the two sides' schedules"


# ---------------------------------------------- the DEEP arms, D = 3 and D = 4

@pytest.mark.parametrize("arm", [DEEP3, DEEP4], ids=["d3", "d4"])
def test_m2_inter_dense_deep(arm, carve=CARVE):
    """Dense both sides at depth three and four: the state plane, the readout and
    the mass column, against the fp64 oracle in CANONICAL order -- so a deep arm's
    mixed-radix leaf order surfaces here as a permutation, not as noise."""
    assert_fresh_binary()
    cell = Cell(arm[0], T=256, k_tok=None, seed=6)
    num, den, plane = run(cell, arm, carve)
    n_ref, d_ref, s_ref = ref(cell, arm)
    assert relative(num.reshape(cell.B, cell.H, cell.T, cell.d_v).permute(0, 2, 1, 3),
                    n_ref) < BF16_RTOL
    assert relative(den.reshape(cell.B, cell.H, cell.T).permute(0, 2, 1), d_ref) < BF16_RTOL
    got = canonical_from_lattice(plane, arm[0], arm[1], arm[2])
    assert relative(got, s_ref.reshape(cell.B * cell.H, cell.N, cell.d_v + 1)) < BF16_RTOL


@pytest.mark.parametrize("arm,k_tok", [(DEEP3, 4), (DEEP3, 8), (DEEP4, 2), (DEEP4, 4)],
                         ids=["d3-k4", "d3-k8", "d4-k2", "d4-k4"])
def test_m3_two_sided_alt_deep(arm, k_tok, carve=CARVE):
    """The ALTERNATING declaration at depth: `D` clause levels, the two sides'
    subsets disjoint, scattered `k_tok` cells."""
    assert_fresh_binary()
    cell = alt_cell(arm[0], T=256, k_tok=k_tok, seed=30 + k_tok + len(arm[0]))
    num, den, plane = run(cell, arm, carve)
    n_ref, d_ref, s_ref = ref(cell, arm)
    assert relative(num.reshape(cell.B, cell.H, cell.T, cell.d_v).permute(0, 2, 1, 3),
                    n_ref) < BF16_RTOL
    assert relative(den.reshape(cell.B, cell.H, cell.T).permute(0, 2, 1), d_ref) < BF16_RTOL
    got = canonical_from_lattice(plane, arm[0], arm[1], arm[2])
    assert relative(got, s_ref.reshape(cell.B * cell.H, cell.N, cell.d_v + 1)) < BF16_RTOL


@pytest.mark.parametrize("arm", [DEEP3, DEEP4], ids=["d3", "d4"])
def test_m3_deep_lists_differ_when_the_sides_do(arm):
    """NON-VACUITY at depth, the small arm's gate restated: the deep ALT cell must
    drive `nr != nw` somewhere, or the two-sided gate above is a dense gate."""
    widths, k, m, W = arm
    cell = alt_cell(widths, T=256, k_tok=2, seed=77 + len(widths))
    _, s, g, bc, owners = carry_ops.box_shape(widths, k, m)
    read = tuple(t.double() for t in cell.read_bf)
    write = tuple(t.double() for t in cell.write_bf)
    diff = 0
    for owner in range(owners):
        o = [(owner // math.prod(g[l + 1:])) % g[l] for l in range(len(widths))]
        liveR = torch.ones(cell.T, dtype=torch.bool, device="cuda")
        liveW = torch.ones(cell.T, dtype=torch.bool, device="cuda")
        for l in range(len(widths)):
            sl = slice(o[l] * s[l], (o[l] + 1) * s[l])
            liveR &= (read[l][0, :, 0, sl] != 0).any(-1)
            liveW &= (write[l][0, :, 0, sl] != 0).any(-1)
        diff += int((liveR != liveW).sum())
    assert diff > 0, "the deep ALT cell did not separate the two sides' schedules"


@pytest.mark.parametrize("arm,spans,atom", [(DEEP3, (2, 4, 16), (1, 1)),
                                            (DEEP4, (2, 2, 4, 8), (1, 2))],
                         ids=["d3", "d4"])
def test_deep_box_is_the_derived_one(arm, spans, atom):
    """The M1 solve, asserted against the host derivation: the spans, and the atom
    rectangle they induce.  `DEEP4`'s rectangle spans TWO levels and no `(k, m)` at
    this box can make it one, because `B = 8` is narrower than an atom."""
    from rola.ops.lattice import atom_rectangle, walk_unit
    widths, k, m, W = arm
    _, s, g, bc, owners = carry_ops.box_shape(widths, k, m)
    assert tuple(s) == spans and bc == 128
    assert atom_rectangle(s) == atom
    base, leaves, atoms = walk_unit(s)
    assert leaves == 16 * atoms and atoms >= 1


# ------------------------------------------- the MULTI-LEVEL carve, D = 4

@pytest.mark.parametrize("k_tok", [None, 2], ids=["dense", "k2"])
def test_m4_the_multi_level_carve_matches_the_fp64_oracle(k_tok, carve=CARVE):
    """TWO CARVED LEVELS PER SIDE, against the fp64 oracle in CANONICAL order.

    This is the arm the single-level carve could not state: the write side owns
    `[1, 4, 2, 4]` of the box `[4, 4, 4, 4]` and the read side `[4, 1, 4, 2]`, so
    each side's stream box is an affine RECTANGLE that is not a contiguous run of
    leaves, and each side's private order has TWO levels moved outermost.  A leaf-map
    or ownership error surfaces here as a permutation of the state plane, not noise.
    """
    assert_fresh_binary()
    widths, k, m, W = CARVE4
    require_arm(widths, k, m, W, 32, carve, ALT4)
    cell = (Cell(widths, T=256, k_tok=None, seed=8) if k_tok is None
            else alt_cell(widths, T=256, k_tok=k_tok, seed=88))
    num, den, plane = carry_ops.carry_inter(
        cell.read_bf, cell.write_bf, cell.g_bf, cell.v_bf, widths, k=k, m=m, window=W,
        chunk=32, carve=carve, level_modes=ALT4)
    n_ref, d_ref, s_ref = ref(cell, CARVE4)
    assert relative(num.reshape(cell.B, cell.H, cell.T, cell.d_v).permute(0, 2, 1, 3),
                    n_ref) < BF16_RTOL
    assert relative(den.reshape(cell.B, cell.H, cell.T).permute(0, 2, 1), d_ref) < BF16_RTOL
    got = canonical_from_lattice(plane, widths, k, m)
    assert relative(got, s_ref.reshape(cell.B * cell.H, cell.N, cell.d_v + 1)) < BF16_RTOL


def test_the_multi_level_carve_arm_is_eight_streams_of_thirty_two_leaves():
    """The declaration's arithmetic, asserted rather than commented: both sides cut
    the same box into the SAME number of streams (this stage's 1:1 executor map) and
    each carves two levels."""
    widths, k, m, _ = CARVE4
    _, spans, groups, bc, owners = carry_ops.box_shape(widths, k, m)
    assert tuple(spans) == (4, 4, 4, 4) and bc == 256 and owners == 1
    write, read = (1, 4, 2, 4), (4, 1, 4, 2)
    for side in (write, read):
        assert all(s <= w and w % s == 0 for s, w in zip(side, spans))
        assert math.prod(w // s for s, w in zip(side, spans)) == 8
        assert math.prod(side) == 32
        assert sum(1 for s, w in zip(side, spans) if s < w) == 2
        assert side[-1] >= 2, "the pair level must hold at least two adjacent leaves"


# ------------------------------------------- the 1x8 CONTAINER, D = 2 (K35 S3a)

@pytest.mark.parametrize("k_tok", [None, 4], ids=["dense", "k4"])
@pytest.mark.parametrize("declaration", [CONT_SPLIT, CONT_STREAMS], ids=["wps2", "streams8"])
def test_s3a_the_1x8_container_matches_the_fp64_oracle(declaration, k_tok):
    """EIGHT EXECUTORS on the flagship cell, both ways, against the fp64 oracle.

    Under the column split (``WPS = 2``) the two warps of a write stream read the SAME
    segment rows and fold DISJOINT halves of that stream's n-tiles, so a mis-folded
    column half, a mis-addressed mask field or a dead-row clear that crossed into the
    other half surfaces here as a wrong state plane rather than as noise.  The plane is
    compared in CANONICAL order, which is what makes a leaf-map error a permutation.
    """
    assert_fresh_binary()
    widths, k, m, W = FLAG
    cell = (Cell(widths, T=768, k_tok=None, seed=5) if k_tok is None
            else alt_cell(widths, T=768, k_tok=k_tok, seed=21))
    num, den, plane = carry_ops.carry_inter(
        cell.read_bf, cell.write_bf, cell.g_bf, cell.v_bf, widths, k=k, m=m, window=W,
        chunk=32, carve=declaration, level_modes=6)
    n_ref, d_ref, s_ref = ref(cell, FLAG)
    assert relative(num.reshape(cell.B, cell.H, cell.T, cell.d_v).permute(0, 2, 1, 3),
                    n_ref) < BF16_RTOL
    assert relative(den.reshape(cell.B, cell.H, cell.T).permute(0, 2, 1), d_ref) < BF16_RTOL
    got = canonical_from_lattice(plane, widths, k, m)
    assert relative(got, s_ref.reshape(cell.B * cell.H, cell.N, cell.d_v + 1)) < BF16_RTOL


def test_s3a_the_container_arms_reach_eight_executors_two_ways():
    """THE EXECUTOR MAP, asserted off the device rather than commented.

    ``threads / 32`` is the executor budget and the census's last field the WRITE
    stream count, so their ratio IS the arm's declared ``WPS``.  Both container arms
    spend the SAME eight executors on the same box; what differs is the ownership grain
    under them, which is the card's two dials stated as one table.  The READ carve is
    one stream per WARP by declaration, so a warp's accumulator holds ``BC / warps``
    leaves whichever route reached those warps.
    """
    assert_fresh_binary()
    widths, k, m, W = FLAG
    _, _, _, bc, _ = carry_ops.box_shape(list(widths), k, m)
    want = {carry_ops.LEAF_SPLIT: (4, 4, 1), CONT_SPLIT: (8, 4, 2), CONT_STREAMS: (8, 8, 1)}
    for declaration, (warps, streams, wps) in want.items():
        stamp = carry_ops.build_stamp(widths, k, m, W, 32, declaration, 6, d_v=D_V)
        assert stamp[7] // 32 == warps, f"declaration {declaration}: executors"
        assert stamp[14] == streams, f"declaration {declaration}: write streams"
        assert stamp[7] // 32 == stamp[14] * wps, f"declaration {declaration}: warps per stream"
        assert bc % warps == 0 and bc // warps == (bc // streams) // wps


# ------------------------------------- the d_v = 128 CONTAINERS, D = 2 (K35 S3b)

@pytest.mark.parametrize("k_tok", [None, 4], ids=["dense", "k4"])
@pytest.mark.parametrize(
    "arm,declaration",
    [(FLAG, CONT_SPLIT), (FLAG, CONT_STREAMS), (FLAG_HALF, CARVE)],
    ids=["1x8-wps2", "1x8-streams8", "2x4-bc64"])
def test_s3b_the_wide_containers_match_the_fp64_oracle(arm, declaration, k_tok):
    """THE WIDE VALUE PLANE, all three containers, against the fp64 oracle.

    ``d_v = 128`` doubles the channel m-tiles the accumulator holds (``MCH``), the
    value row's asynchronous quanta, the transit plane's rows and the state page's
    columns -- and it must move NOTHING else.  So the gate is the same one the narrow
    arms take: ``num``, ``den`` AND the state plane in CANONICAL order, where a leaf-map
    or a channel-map error is a permutation rather than noise.  The ``BC = 64`` row is
    the 2x4 container and reaches the same sixteen leaves per warp by halving the box,
    so it also gates the page sweep at a state row WIDER than the CTA (129 columns
    against 128 threads), which no narrow arm exercises.
    """
    assert_fresh_binary()
    widths, k, m, W = arm
    cell = (Cell(widths, T=768, k_tok=None, seed=5, d_v=WIDE) if k_tok is None
            else alt_cell(widths, T=768, k_tok=k_tok, seed=21, d_v=WIDE))
    num, den, plane = carry_ops.carry_inter(
        cell.read_bf, cell.write_bf, cell.g_bf, cell.v_bf, widths, k=k, m=m, window=W,
        chunk=32, carve=declaration, level_modes=6)
    assert num.shape[-1] == WIDE and plane.shape[-1] == WIDE + 1
    n_ref, d_ref, s_ref = ref(cell, arm)
    assert relative(num.reshape(cell.B, cell.H, cell.T, cell.d_v).permute(0, 2, 1, 3),
                    n_ref) < BF16_RTOL
    assert relative(den.reshape(cell.B, cell.H, cell.T).permute(0, 2, 1), d_ref) < BF16_RTOL
    got = canonical_from_lattice(plane, widths, k, m)
    assert relative(got, s_ref.reshape(cell.B * cell.H, cell.N, cell.d_v + 1)) < BF16_RTOL


# ------------------------------------------- THE PARKED `lo`, both widths (K35 S5)

#: THE FOUR PARK TWINS.  `PARKED` is a BIT of the declaration scalar, so each row here
#: names one cell twice -- once with the write-view `lo` in registers across the readout
#: and once with it parked in the warp's own dead exchange segment (P76 F5).  The three
#: `d_v = 128` containers are the arms the park exists for (they sit AT the register
#: wall) and the flagship `d_v = 64` ALT arm is the CONTROL.
PARK_TWINS = [
    pytest.param(FLAG, WIDE, CONT_SPLIT, id="1x8-wps2-dv128"),
    pytest.param(FLAG, WIDE, CONT_STREAMS, id="1x8-streams8-dv128"),
    pytest.param(FLAG_HALF, WIDE, CARVE, id="2x4-bc64-dv128"),
    pytest.param(FLAG, D_V, CARVE, id="flagship-alt-dv64"),
]


@pytest.mark.parametrize("k_tok", [None, 4], ids=["dense", "k4"])
@pytest.mark.parametrize("arm,d_v,declaration", PARK_TWINS)
def test_s5_the_park_matches_the_oracle_and_its_own_twin(arm, d_v, declaration, k_tok):
    """THE PARK IS A REGISTER DECISION AND NOTHING ELSE, gated as two statements.

    The park moves the accumulator's LOW halves out of registers and into the warp's
    own exchange segment for the length of the readout, then takes them back before
    any warp publishes into that segment again.  So it is BOTH: an arithmetic identity
    -- the fp64 oracle, `num`, `den` and the state plane in canonical order -- and a
    LOSSLESS ROUND TRIP, which is the stronger of the two and is what gates the park's
    own SMEM layout (KERNEL_STANDARDS section 3): the parked arm's state plane must be
    the unparked twin's BIT FOR BIT, because the two run identical arithmetic in an
    identical order.  A dropped word, an aliased byte or a swizzle applied to the wrong
    map cannot survive that.  (`num`/`den` are NOT compared: the readout's fan-in is
    `red.shared` float atomics, whose summation order is arrival order.)
    """
    assert_fresh_binary()
    widths, k, m, W = arm
    cell = (Cell(widths, T=768, k_tok=None, seed=5, d_v=d_v) if k_tok is None
            else alt_cell(widths, T=768, k_tok=k_tok, seed=21, d_v=d_v))
    got = {}
    for parked in (False, True):
        got[parked] = carry_ops.carry_inter(
            cell.read_bf, cell.write_bf, cell.g_bf, cell.v_bf, widths, k=k, m=m, window=W,
            chunk=32, carve=declaration + (carry_ops.PARKED if parked else 0), level_modes=6)
    num, den, plane = got[True]
    n_ref, d_ref, s_ref = ref(cell, arm)
    assert relative(num.reshape(cell.B, cell.H, cell.T, cell.d_v).permute(0, 2, 1, 3),
                    n_ref) < BF16_RTOL
    assert relative(den.reshape(cell.B, cell.H, cell.T).permute(0, 2, 1), d_ref) < BF16_RTOL
    assert relative(canonical_from_lattice(plane, widths, k, m),
                    s_ref.reshape(cell.B * cell.H, cell.N, cell.d_v + 1)) < BF16_RTOL
    assert torch.equal(plane, got[False][2]), (
        "the park is a lossless round trip: the state plane must be its twin's, bit for bit")


# ------------------------ THE DENSE DECLARATIONS ON THE CARVED BODY (K35 S1c)

@pytest.mark.parametrize("k_tok", [None, 4], ids=["dense", "k4"])
@pytest.mark.parametrize("d_v", [D_V, WIDE], ids=["dv64", "dv128"])
def test_s1c_the_carved_body_serves_a_dense_declaration(d_v, k_tok):
    """`MO = 0` ON THE CARVED BODY, against the fp64 oracle at both widths.

    The declaration reaches this body at ONE place -- `ballot_of`'s per-level
    conjunction, which a side does not enter where it is dense -- so `DENSE_BOTH`
    leaves that conjunction with no level at all.  Its identity is `valid`, the
    window's own in-range mask, so the ballot comes out VACUOUSLY FULL: every token of
    the window is offered to every stream and the schedule prunes nothing.  That is the
    degenerate case the carve had never been asked for, and the two ways it could have
    been wrong are both caught here -- an empty conjunction reading as EMPTY would drop
    every contribution, and a table row addressed anyway would read a row the facts pass
    never voted.  The oracle sees either as a wrong state plane in canonical order.

    Run at a SPARSE cell too, because `DENSE_BOTH` is the declaration that must be
    correct at any cell -- it is the one whose schedule prunes nothing anywhere -- so a
    cell whose ballots would have pruned is the case where an over-broad schedule has to
    reduce to the same arithmetic rather than to more of it.
    """
    assert_fresh_binary()
    widths, k, m, W = FLAG
    cell = (Cell(widths, T=768, k_tok=None, seed=5, d_v=d_v) if k_tok is None
            else alt_cell(widths, T=768, k_tok=k_tok, seed=21, d_v=d_v))
    num, den, plane = carry_ops.carry_inter(
        cell.read_bf, cell.write_bf, cell.g_bf, cell.v_bf, widths, k=k, m=m, window=W,
        chunk=32, carve=CONT_SPLIT, level_modes=0)
    assert num.shape[-1] == d_v and plane.shape[-1] == d_v + 1
    n_ref, d_ref, s_ref = ref(cell, FLAG)
    assert relative(num.reshape(cell.B, cell.H, cell.T, cell.d_v).permute(0, 2, 1, 3),
                    n_ref) < BF16_RTOL
    assert relative(den.reshape(cell.B, cell.H, cell.T).permute(0, 2, 1), d_ref) < BF16_RTOL
    got = canonical_from_lattice(plane, widths, k, m)
    assert relative(got, s_ref.reshape(cell.B * cell.H, cell.N, cell.d_v + 1)) < BF16_RTOL


@pytest.mark.parametrize("d_v", [D_V, WIDE], ids=["dv64", "dv128"])
def test_s1c_the_dense_carve_follows_the_stream_counts_and_is_not_tied(d_v):
    """THE DERIVATION AT `MO = 0`, read back off the device.

    Both sides class every level DENSE, so the two private ORDERS coincide -- which is
    the condition the tied arm (row 20) needs.  It is not sufficient, and this is the
    row that says why: the tie also needs the two derived VECTORS to agree, and under
    the 1x8 column split the read side has `NSR = NSW * WPS` streams to the write side's
    `NSW`.  The walk carves the outer level of a shared order by two different counts,
    so the vectors part (`[2, 16]` against `[1, 16]`), the leaf-to-reader permutation is
    not the identity, and the exchange is FULL.  Asserted rather than commented because
    `xch = 0` would otherwise be an easy thing to assume from "the orders agree".
    """
    assert_fresh_binary()
    widths, k, m, W = FLAG
    stamp = carry_ops.build_stamp(widths, k, m, W, 32, CONT_SPLIT, 0, d_v=d_v)
    #: a level past `D` carries `B`, which is the identity the vector is padded with.
    assert list(stamp)[16:20] == [2, 16, 256, 256], "the write side keeps two digits of level 0"
    assert list(stamp)[20:24] == [1, 16, 256, 256], "the read side keeps one, at twice the streams"
    assert stamp[24] > 0 and stamp[25] == 1, (
        "the orders agree but the counts do not, so the exchange is full and not elided")
    #: the SAME cell, box and executor map as the ALT row it is compared against, so the
    #: only thing that differs between the two measurements is the declaration.
    alt = carry_ops.build_stamp(widths, k, m, W, 32, CONT_SPLIT, 6, d_v=d_v)
    assert (stamp[7], stamp[9], stamp[14]) == (alt[7], alt[9], alt[14]), (
        "threads, box and write-stream count must match the ALT row this arm is measured against")


def test_an_unbuilt_declaration_is_refused_and_not_served_dense(): 
    """THE SEAM REFUSES WHAT IT HAS NO BODY FOR (K35 R-B, P76).

    `carry.cu`'s law-3 retry used to re-dispatch an unbuilt declaration with `mo`
    forced to zero AT THE SAME CARVE, so a caller asking for a declaration nothing
    was built for silently got the DENSE_BOTH body and no diagnostic -- which is how
    a 5x loss stayed hidden from a harness for nine stages (`## BISECT (corrected)`).
    The flagship topology at `carve = LEAF_SPLIT` is built for DENSE_BOTH (row 0) and
    for the ALT declaration (row 1) and for nothing else, so the depth-four ALT word
    is a declaration with no body at this topology AND a DENSE_BOTH arm sharing its
    carve -- exactly the pair the retry used to resolve.  It must raise, and the
    message must name the key that was asked for, or the refusal is not readable.
    """
    assert_fresh_binary()
    widths, k, m, W = FLAG
    cell = Cell(widths, T=256, k_tok=None, seed=3)
    with pytest.raises(RuntimeError, match="unbuilt carry arm"):
        carry_ops.carry_inter(cell.read_bf, cell.write_bf, cell.g_bf, cell.v_bf, widths,
                              k=k, m=m, window=W, chunk=32, carve=CARVE, level_modes=ALT4)
