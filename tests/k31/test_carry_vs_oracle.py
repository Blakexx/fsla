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
from tests.k31.fixtures import Cell, assert_fresh_binary, canonical_from_plane, relative, require_arm
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
D_V = carry_ops.D_V
#: the ALTERNATING declaration packed as the seam's word, at depth four.
ALT4 = 102

CARVE = carry_ops.LEAF_SPLIT


def run(cell, arm, carve, chunk=32, state_in=None):
    widths, k, m, W = arm
    require_arm(widths, k, m, W, chunk, carve, 0, d_v=cell.d_v)
    read, write, g, v = cell.ref_levels()
    #: THE STATE PLANE IS THE CALLER'S (state.md section 4). This cell binds a FRESH
    #: dense backing, so it allocates one; the paged and continued forms are
    #: `tests/integration/test_chunk_paging_equivalence.py`'s.
    plane = carry_ops.state_plane(widths, cell.v_bf.shape[0] * cell.v_bf.shape[2],
                                  cell.v_bf.shape[-1])
    num, den, plane = carry_ops.carry_inter(
        cell.read_bf, cell.write_bf, cell.g_bf, cell.v_bf, widths,
        k=k, m=m, window=W, chunk=chunk, carve=carve, state_in=state_in, state_out=plane)
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
    same topology appears once per declaration built for it.  A declaration with no
    body built for it is REFUSED at every topology (K35 R-B): DENSE_BOTH is served when
    it is DECLARED and never as a substitute for something else.

    SINCE M1P THE LIST IS ONE SHAPE.  The transplanted M1a body is four warps, ONE warp
    per stream, one order and `d_v = 64`, so every carved row carries that executor map
    and every row carries that width; the containers, the wide plane, the park, the
    tied arm and the dense declarations on the carved body left with the mechanisms
    that served them, and each returns with the stage that rebuilds one.  What survives
    as a property of the table is therefore stronger than before: EVERY topology
    carries the DENSE_BOTH body, with no carved exception at all.
    """
    declared = [tuple(row[:9]) for row in gen_shards.CARRY_ARMS]
    reported = [tuple(a[:9]) for a in carry_ops.arms()]
    unknown = [row for row in reported if row not in declared]
    assert not unknown, f"the binary reports rows that are not declared: {unknown}"
    assert reported == [row for row in declared if row in set(reported)], (
        "the built rows must be the declared rows, in the declared order")
    #: THE PROPERTIES BELOW ARE THE TABLE'S, so they are asked of the DECLARED rows and
    #: not of the subset a build carries.
    keys = {row[:8] for row in declared}
    dense = {row[:8] for row in declared if row[8] == 0}
    assert keys == dense, "every topology must carry the DENSE_BOTH body"
    #: ONE EXECUTOR MAP AND ONE WIDTH (K35 M1P), read off the declared row: `WPS` is the
    #: row's tenth field, the two stream counts its eleventh and twelfth, and the park
    #: its last.  A row that breaks any of these has no body in this tree.
    for row in gen_shards.CARRY_ARMS:
        D, B, k, m, d_v, W, C, CV, MO, WPS, NSW, NSR, PK = row
        assert d_v == D_V, f"{row}: M1P builds one value width"
        assert (WPS, PK, CV) == (1, 0, carry_ops.LEAF_SPLIT), (
            f"{row}: one warp per stream, no park, the ruled carve")
        assert NSW == NSR, f"{row}: M1P carves the box once, in the read side's order"
    for arm in carry_ops.arms():
        D, B, k, m, d_v, W, C, carve, modes, BC, owners, smem, level = arm
        assert d_v == D_V
        _, spans_host, _, bc_host, owners_host = carry_ops.box_shape([B] * D, k, m)
        assert (BC, owners) == (bc_host, owners_host), (
            "the device box and the host solver disagree about the owner's shape")
        #: the carve level is the innermost level the READ side carves, or -1 for the
        #: channel-split body.  IT IS READ OFF THE DERIVED VECTOR (K35 S1b) and no longer
        #: assumed to be the innermost level the read side is SPARSE at: under ORDER BY
        #: MODE the read side carves the OUTERMOST levels of its own order.  Comparing
        #: the seam's reported level against the census's reported vector makes this a
        #: cross-check of two derivations rather than a restatement of one.
        sr = list(carry_ops.build_stamp([B] * D, k, m, W, C, carve, modes, d_v))[20:24]
        carved = [l for l in range(D) if sr[l] < int(spans_host[l])]
        assert level == (max(carved) if carved else -1), (
            f"the seam reports carve level {level}, but the derived read vector "
            f"{sr[:D]} against the owner box {list(spans_host)} carves {carved}")
        #: AND THE TWO SIDES' VECTORS AGREE, because M1P carves once: the census reports
        #: the write spans at [16:20] and the read spans at [20:24], and the exchange's
        #: transit and state (the last two fields) are ZERO because there is none.
        st = list(carry_ops.build_stamp([B] * D, k, m, W, C, carve, modes, d_v))
        assert st[16:20] == st[20:24], "M1P walks ONE span vector on both sides"
        assert st[24] == 0 and st[25] == 0, "M1P has no exchange to size or to state"


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
    got = canonical_from_plane(plane)
    want = s_ref.reshape(cell.B * cell.H, cell.N, cell.d_v + 1)
    assert relative(got, want) < BF16_RTOL


def test_m1_entry_state_is_carried(carve=CARVE):
    assert_fresh_binary()
    cell = Cell(SMALL[0], T=128, k_tok=None, seed=2, state_in=True)
    s_in = cell.state_in_plane()
    _, _, plane = run(cell, SMALL, carve, state_in=s_in)
    s0 = cell.state_in.reshape(cell.B, cell.H, cell.N, cell.d_v + 1)
    _, _, s_ref = ref(cell, SMALL, state_in=s0)
    got = canonical_from_plane(plane)
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
    got = canonical_from_plane(plane)
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
    got = canonical_from_plane(plane)
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
    got = canonical_from_plane(plane)
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
    got = canonical_from_plane(plane)
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


# ------------------------------------------------------------------ the refusal

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
