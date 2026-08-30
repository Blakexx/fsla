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

from rola.engine.facts import planes
from rola.ops import carry as carry_ops
from rola.ops.paging import bytes_equal
from tests.oracle import reference
from tests.oracle.fixtures import Cell, assert_fresh_binary, canonical_from_plane, relative, require_arm
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


def _facts_plane(cell, levels):
    """One side as the facts pass' `[B, H, L, sum_l w_l]` plane.

    `pack_side` already lays the bytes out this way for the kernel seam, which names
    the leading axes `[BH, L, sum]`; the facts entry names them apart because its own
    grid is per `(bh)`. Same bytes, one view.
    """
    return carry_ops.pack_side(levels).view(cell.B, cell.H, cell.T, -1)


def run(cell, arm, carve, chunk=32, state_in=None, modes=0, body="split"):
    """`modes` IS THE DECLARATION, and it is what selects the BODY -- so `run` ASSERTS
    WHICH BODY IT GOT, off the device, before it measures anything.

    `level_modes = 0` (DENSE_BOTH) reaches the channel-split body; a declaration with a
    sparse level AND a stream count above one reaches the CARVED body, which is the only
    body carrying the two-sided carve and its exchange (K35 F1).  Before F1 every cell
    here ran with `modes = 0`, so the fp64 truth gated ONE of the two shipped bodies and
    nothing said so -- the suite looked like full coverage because no cell named the body
    it was exercising.  The stamp's `kXchState` distinguishes them off the built binary
    (`-1` = no exchange at all = the channel-split body; `0` = tied; `1` = a real
    transit), so a cell that silently lands on the wrong body now FAILS here rather than
    reporting green coverage of a body it never touched.
    """
    widths, k, m, W = arm
    require_arm(widths, k, m, W, chunk, carve, modes, d_v=cell.d_v)
    assert body in ("split", "carved")
    xch_state = list(carry_ops.build_stamp(widths, k, m, W, chunk, carve, modes,
                                           cell.d_v))[25]
    if body == "carved":
        assert xch_state >= 0, (
            f"this cell asked for the CARVED body (level_modes={modes}) and the binary "
            f"dispatched the channel-split one (kXchState={xch_state}): the exchange and "
            f"the two-sided carve would go ungated")
    else:
        assert xch_state == -1, (
            f"this cell asked for the channel-split body (level_modes={modes}) and got a "
            f"carved one (kXchState={xch_state})")
    read, write, g, v = cell.ref_levels()
    #: THE STATE PLANE IS THE CALLER'S (state.md section 4). A fresh cell binds a
    #: dense backing and allocates one; a carried cell ADVANCES ITS OWN PLANE IN PLACE,
    #: which is what the activity gate requires of every continuation (P80b) and what
    #: the facade's dense backing does. The paged forms are
    #: `tests/integration/test_chunk_paging_equivalence.py`'s.
    plane = state_in if state_in is not None else carry_ops.state_plane(
        widths, cell.v_bf.shape[0] * cell.v_bf.shape[2], cell.v_bf.shape[-1])
    #: THE SHIPPED PREDICATE IS GATED HERE TOO: every oracle cell runs with the facts
    #: pass' own activity byte, so the fp64 comparison certifies the arithmetic the
    #: kernel does with its sweeps SKIPPED, not a fuller one no shipped call makes.
    bits = planes.atom_bits(_facts_plane(cell, cell.write_bf), widths,
                            read_plane=_facts_plane(cell, cell.read_bf))
    num, den, plane = carry_ops.carry_inter(
        cell.read_bf, cell.write_bf, cell.g_bf, cell.v_bf, widths,
        k=k, m=m, window=W, chunk=chunk, carve=carve, level_modes=modes,
        state_in=state_in, state_out=plane, atom_bits=bits)
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

    SINCE F1 THE CARVED ROWS ARE TWO-SIDED.  The body is still four warps, ONE warp per
    stream and `d_v = 64`, but each side now derives its OWN order and span vector, so a
    carved row's two vectors are no longer required to agree -- and the ones that DO
    agree are the tied rows, whose exchange is elided.  What the table is asked for is
    therefore the LAW relating the three reported quantities rather than a remembered
    shape: the two vectors, the transit's size, and the transit's STATE.
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
        #: THE EXCHANGE'S LAW, read off the census (K35 F1).  The write spans are at
        #: [16:20], the read spans at [20:24], the transit's bytes at [24] and its STATE
        #: at [25]: `-1` no exchange at all (the channel-split body), `0` the tie
        #: (identical vectors AND identical orders, so publish and pull are elided), `1`
        #: a real transit.  The three are not independent, and asserting the RELATION is
        #: what makes this a cross-check rather than a restatement: a transit exists
        #: exactly when the two derived vectors differ, and it is sized to one whole
        #: `[consumer][leaf][channel]` region when it does.
        st = list(carry_ops.build_stamp([B] * D, k, m, W, C, carve, modes, d_v))
        sw_v, sr_v, xch_bytes, xch_state = st[16:20], st[20:24], st[24], st[25]
        same = sw_v == sr_v
        if xch_state == -1:
            assert same and xch_bytes == 0, (
                f"{arm}: the channel-split body carves neither side, so its two vectors "
                f"are the owner's box and it has no transit to size")
        elif xch_state == 0:
            assert same and xch_bytes == 0, (
                f"{arm}: a TIED row's two carves coincide, so the permutation is the "
                f"identity and the transit is elided, not merely small")
        else:
            assert xch_state == 1 and not same, (
                f"{arm}: a row reporting a real transit must have two DIFFERENT derived "
                f"vectors, or it pays for a permutation that is the identity "
                f"({sw_v} vs {sr_v})")
            assert xch_bytes == BC * (d_v + 8) * 2, (
                f"{arm}: a full transit is one whole [consumer][leaf][channel] region, "
                f"`BC * LdX * 2` with `LdX = d_v + kPadE`")


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


def test_m1_a_continuation_that_touches_a_strict_subset(carve=CARVE):
    """THE IDLE-RESIDENT CELL (P80b): call two's routing is call one's, truncated.

    Two structured-support cells over one plane, advanced IN PLACE. Every atom call one
    reaches is committed; call two reaches a STRICT SUBSET of them, so the rest are
    resident and IDLE -- and the kernel must neither load nor store them. The fp64
    reference chains the same two calls and knows nothing about any of it, which is what
    makes this a statement about the arithmetic rather than about the skip.

    THE CENSUS IS ASSERTED HERE, not inferred: an idle atom's bytes must be BIT-identical
    across call two (nothing stored), and perturbing one must not move `y` (nothing
    read). Equality on the written half is the fp64 comparison itself.
    """
    assert_fresh_binary()
    first = Cell(SMALL[0], T=256, k_tok=None, seed=71, support=0.5)
    second = Cell(SMALL[0], T=256, k_tok=None, seed=72, support=0.25)

    _, _, plane = run(first, SMALL, carve)
    _, _, s1_ref = ref(first, SMALL)

    bits = planes.atom_bits(_facts_plane(second, second.write_bf), SMALL[0],
                            read_plane=_facts_plane(second, second.read_bf))
    written = planes.written_atoms(
        planes.atom_bits(_facts_plane(first, first.write_bf), SMALL[0],
                         read_plane=_facts_plane(first, first.read_bf)))
    idle = written & (bits == 0)
    assert bool(idle.any()), "the truncated cell reaches every atom the first one did"

    before = plane.clone()
    num, den, plane = run(second, SMALL, carve, state_in=plane)
    _, _, s2_ref = ref(second, SMALL, state_in=s1_ref)
    got = canonical_from_plane(plane)
    assert relative(got, s2_ref.reshape(second.B * second.H, second.N,
                                        second.d_v + 1)) < BF16_RTOL
    assert bytes_equal(plane[idle], before[idle]), (
        "an idle-resident atom moved: the exit sweep stored an atom this call does "
        "not write")

    #: THE READ HALF OF THE CENSUS: the same call over a plane whose idle atoms hold
    #: something else entirely must produce the same `y`, because nothing reads them.
    perturbed = before.clone()
    perturbed[idle] = -1234.5
    num2, den2, _ = run(second, SMALL, carve, state_in=perturbed)
    assert torch.equal(num, num2) and torch.equal(den, den2), (
        "perturbing an idle-resident atom moved the readout: the entry sweep is "
        "reading an atom this call does not read")


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
    y, _ = naive_rola(v, read, write, g, topo, None,
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
    from tests.oracle.fixtures import simplex
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


#: the ALTERNATING declaration at depth two, and the BOTH-SPARSE one -- the seam's
#: mode word, two bits per level (bit 0 read, bit 1 write), level `l` at bits `2l`.
ALT2 = 6    # level 0 write-sparse, level 1 read-sparse
BOTH2 = 3   # level 0 sparse on BOTH sides, level 1 dense on both -- the TIED arm


def both_cell(widths, T, k_tok, seed, d_v=carry_ops.D_V):
    """SPARSE ON BOTH SIDES AT LEVEL ZERO, dense above it -- the declaration whose two
    derived carves COINCIDE, and therefore the one that proves the elided exchange.

    "Both-sparse" is a statement about the DECLARATION, not the supports: the two sides
    are drawn independently here on purpose, because the kernel never assumes they
    agree (P76, "SHARED means sparse on both sides, nothing more").  The dense pass is
    drawn first and the supports redrawn off a fresh same-seeded generator, so this
    consumes the RNG stream exactly as `alt_cell` does."""
    from tests.oracle.fixtures import simplex
    device = "cuda"
    cell = Cell(widths, T=T, k_tok=None, seed=seed, d_v=d_v)
    shape = (cell.B, T, cell.H)
    gen = torch.Generator(device=device).manual_seed(seed)

    def draw(width, sparse):
        kt = k_tok if sparse else None
        return simplex(shape + (width,), None if kt is None or kt >= width else kt, gen, device)

    read = tuple(draw(w, l == 0) for l, w in enumerate(widths))
    write = tuple(draw(w, l == 0) for l, w in enumerate(widths))
    cell.read, cell.write = read, write
    cell.read_bf = tuple(t.to(torch.bfloat16) for t in read)
    cell.write_bf = tuple(t.to(torch.bfloat16) for t in write)
    return cell


# ------------------------------------------- the CARVED body, against the fp64 truth
#
# Every cell above declares `level_modes = 0` and therefore runs the CHANNEL-SPLIT
# body.  These drive the CARVED body -- the two-sided carve, the two private orders and
# the EXCHANGE between them (K35 F1) -- at the three shapes the transit can take:
# a full transit at `D = 2`, a full transit at a MULTI-LEVEL carve (`D = 4`), and the
# ELIDED transit at the tied arm.  `num`/`den` are the fold->readout round trip and the
# state plane is the accumulator in the WRITE carve's order, so a wrong permutation,
# a wrong swizzle or a dropped mass column cannot pass any of them.


@pytest.mark.parametrize("k_tok", [4, 16])
def test_f1_carved_alt_small(k_tok, carve=CARVE):
    assert_fresh_binary()
    cell = alt_cell(SMALL[0], T=256, k_tok=k_tok, seed=60 + k_tok)
    num, den, plane = run(cell, SMALL, carve, modes=ALT2, body="carved")
    n_ref, d_ref, s_ref = ref(cell, SMALL)
    assert relative(num.reshape(cell.B, cell.H, cell.T, cell.d_v).permute(0, 2, 1, 3),
                    n_ref) < BF16_RTOL
    assert relative(den.reshape(cell.B, cell.H, cell.T).permute(0, 2, 1), d_ref) < BF16_RTOL
    got = canonical_from_plane(plane)
    assert relative(got, s_ref.reshape(cell.B * cell.H, cell.N, cell.d_v + 1)) < BF16_RTOL


@pytest.mark.parametrize("k_tok", [4, 16])
def test_f1_carved_alt_flagship(k_tok):
    assert_fresh_binary()
    cell = alt_cell(FLAG[0], T=1024, k_tok=k_tok, seed=70 + k_tok)
    num, den, _ = run(cell, FLAG, CARVE, modes=ALT2, body="carved")
    n_ref, d_ref, _ = ref(cell, FLAG)
    assert relative(num.reshape(cell.B, cell.H, cell.T, cell.d_v).permute(0, 2, 1, 3),
                    n_ref) < BF16_RTOL
    assert relative(den.reshape(cell.B, cell.H, cell.T).permute(0, 2, 1), d_ref) < BF16_RTOL


@pytest.mark.parametrize("k_tok", [4, 16])
def test_f1_carved_multi_level_deep4(k_tok, carve=CARVE):
    """THE MULTI-LEVEL CARVE.  Row 9's derived vectors are write `[1, 2, 2, 8]` and
    read `[2, 1, 4, 4]` -- four levels, both sides carving more than one of them, and
    the two orders differing at every level.  It is the widest disorder any built arm
    hands the exchange, so it is the arm the row map's generality is proved on."""
    assert_fresh_binary()
    cell = alt_cell(DEEP4[0], T=256, k_tok=k_tok, seed=80 + k_tok)
    num, den, plane = run(cell, DEEP4, carve, modes=ALT4, body="carved")
    n_ref, d_ref, s_ref = ref(cell, DEEP4)
    assert relative(num.reshape(cell.B, cell.H, cell.T, cell.d_v).permute(0, 2, 1, 3),
                    n_ref) < BF16_RTOL
    assert relative(den.reshape(cell.B, cell.H, cell.T).permute(0, 2, 1), d_ref) < BF16_RTOL
    got = canonical_from_plane(plane)
    assert relative(got, s_ref.reshape(cell.B * cell.H, cell.N, cell.d_v + 1)) < BF16_RTOL


@pytest.mark.parametrize("k_tok", [4, 16])
def test_f1_tied_carve_elides_the_exchange(k_tok):
    """THE TIED ARM: both sides carve the same levels into the same vector, so the
    permutation is the identity and the transit is compiled away.  The arithmetic must
    be the fp64 truth exactly as it is when the transit runs -- an elision that changed
    a number would mean the two paths were never the same map."""
    assert_fresh_binary()
    cell = both_cell(FLAG[0], T=1024, k_tok=k_tok, seed=90 + k_tok)
    num, den, _ = run(cell, FLAG, CARVE, modes=BOTH2, body="carved")
    n_ref, d_ref, _ = ref(cell, FLAG)
    assert relative(num.reshape(cell.B, cell.H, cell.T, cell.d_v).permute(0, 2, 1, 3),
                    n_ref) < BF16_RTOL
    assert relative(den.reshape(cell.B, cell.H, cell.T).permute(0, 2, 1), d_ref) < BF16_RTOL


@pytest.mark.parametrize("k_tok", [4, 16])
def test_f1b_the_rejoin_survives_the_window_lines_at_the_flagship(k_tok):
    """THE HYBRID'S OWN CELL (K35 F1b).  The transit no longer COPIES the hi plane: the
    writers drop it for the whole readout and rebuild `st` after it from the halves that
    came back plus the lo plane they kept.  That rejoin happens ONCE PER WINDOW, so what
    has to be exercised is a run of window lines -- and the STATE PLANE, which the
    flagship cell above never checked, is the only output that carries a rejoin error
    that the readout of the SAME window would hide.  `T` is chosen to be four windows
    with the last one PARTIAL, so a short window's rejoin is in the gate too."""
    assert_fresh_binary()
    widths, k, m, W = FLAG
    T = 3 * W + W // 2
    assert (T + W - 1) // W >= 3, "the cell must cross at least three window lines"
    cell = alt_cell(FLAG[0], T=T, k_tok=k_tok, seed=110 + k_tok)
    num, den, plane = run(cell, FLAG, CARVE, modes=ALT2, body="carved")
    n_ref, d_ref, s_ref = ref(cell, FLAG)
    assert relative(num.reshape(cell.B, cell.H, cell.T, cell.d_v).permute(0, 2, 1, 3),
                    n_ref) < BF16_RTOL
    assert relative(den.reshape(cell.B, cell.H, cell.T).permute(0, 2, 1), d_ref) < BF16_RTOL
    got = canonical_from_plane(plane)
    assert relative(got, s_ref.reshape(cell.B * cell.H, cell.N, cell.d_v + 1)) < BF16_RTOL


def test_f1_the_census_reports_the_exchange_off_the_device():
    """`kXchBytes`/`kXchState`, read back off the built binary rather than modelled.

    The tied row must report ZERO bytes and `kXchZero`; every other carved row must
    report one whole `[consumer][leaf][channel]` region and `kXchFull`.  A size alone
    could not say it -- `kXchFull` and `kXchFullReorder` allocate the same bytes for
    opposite reasons (K35 S1b) -- so the STATE is what is asserted."""
    assert_fresh_binary()
    seen = {}
    for arm in carry_ops.arms():
        D, B, k, m, d_v, W, C, carve, modes, BC, owners, smem, level = arm[:13]
        st = list(carry_ops.build_stamp([B] * D, k, m, W, C, carve, modes, d_v))
        xch_bytes, xch_state = st[24], st[25]
        seen[(D, B, W, modes)] = (xch_bytes, xch_state)
        if xch_state == -1:
            assert xch_bytes == 0, f"{arm}: the split body has no exchange to size"
        elif xch_state == 0:
            assert xch_bytes == 0, f"{arm}: a tied arm's transit is elided, not sized"
        else:
            assert xch_bytes == BC * (d_v + 8) * 2, (
                f"{arm}: a full transit is one [consumer][leaf][channel] region")
    tied = [v for kk, v in seen.items() if kk[3] == BOTH2]
    assert tied and all(v == (0, 0) for v in tied), (
        "the tied arm must be BUILT and must report the zero, or the elision is untested")


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
