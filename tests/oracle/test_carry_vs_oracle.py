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

#: (widths, W) -- THE CELLS this suite drives, and they are no longer arms.  `(k, m)`
#: left the vocabulary at K50: the owner's box is a REGISTER shape the arm carries
#: (`BC = 128`) and the spans that fill it are DERIVED from the topology, so a cell
#: names a topology and a window and the seam derives the rest.
SMALL = ((64, 64), carry_ops.WINDOW_SMALL)         # N = 4096,  BC = 128, owners = 32
FLAG = ((256, 256), carry_ops.WINDOW)              # N = 65536, BC = 128, owners = 512
#: THE DEEP CELLS.  Both hold `BC = 128` at `N = 4096`; they differ from the flat cells
#: only in how those 128 leaves cut into digits.  THE SPANS ARE UNCHANGED by K50: the
#: runtime rule (balance `BC` over the levels, SUBJECT TO the box being a union of whole
#: atoms) reproduces the compile-time `BoxPlan`'s answer at every built topology, which
#: is the parity proof `test_the_derived_box_reproduces_the_compile_time_law` asserts.
DEEP3 = ((16, 16, 16), carry_ops.WINDOW_SMALL)     # s = (2, 4, 16),   owners = 32
DEEP4 = ((16, 16, 16, 16), carry_ops.WINDOW_SMALL)  # s = (2, 2, 2, 16), owners = 2048
D_V = carry_ops.D_V
#: the ALTERNATING declaration packed as the seam's word, at depth four.
ALT4 = 102

#: THE BODY, which is all this seam still declares (K50: the carve is derived).
CARVED = carry_ops.CARVED
SPLIT = carry_ops.CHANNEL_SPLIT


#: THE ADDRESSING BLOCK, DECODED.  `carry_ops.geometry` returns the SHIPPED derivation
#: flat (one `int64` vector, so the seam needs no schema); this names its fields once so
#: the law tests read the block instead of re-deriving it in Python -- which would only
#: prove that two derivations agree with each other (`matching-the-naive-antipattern`).
_SIDE_ARRAYS = ("level_at", "rank", "row_span", "grid", "grid_div", "row_g", "row_prefix",
                "sub_bits", "sub_shift")
_SIDE_SCALARS = ("carves", "rows", "leaf", "streams", "assign")
_BLK_ARRAYS = ("width", "span", "grid", "col_base", "run_shift", "owner_div", "weight")
_BLK_SCALARS = ("wtot", "owners", "leaves", "local_bits", "min_width", "identity_map")
_LVL = 4


def block(widths, modes=0, ns=4, bc=carry_ops.BC):
    g = list(carry_ops.geometry(widths, modes, bc, ns, ns))
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


def _facts_plane(cell, levels):
    """One side as the facts pass' `[B, H, L, sum_l w_l]` plane.

    `pack_side` already lays the bytes out this way for the kernel seam, which names
    the leading axes `[BH, L, sum]`; the facts entry names them apart because its own
    grid is per `(bh)`. Same bytes, one view.
    """
    return carry_ops.pack_side(levels).view(cell.B, cell.H, cell.T, -1)


def run(cell, arm, split, chunk=32, state_in=None, modes=0, body="split"):
    """THE BODY IS DECLARED, NOT INFERRED (K50) -- and `run` still ASSERTS WHICH BODY IT
    GOT, off the device, before it measures anything.

    Before K50 the declaration `level_modes` picked the body through the arm list, which
    is why this assertion existed at all (`## F1`: every cell ran at `modes = 0`, so the
    fp64 truth gated ONE of the two shipped bodies and nothing said so).  The mode word
    is DATA now and one binary serves every declaration, so the body is the caller's
    `split` flag -- and the census still answers off the device, because `kXchBytes` is
    ZERO exactly on the channel-split body and one whole transit region on the carved
    one.  `kXchState` is GONE with the deleted tied body (K50); what the stamp reports in
    its place is whether the transit's ROW MAP IS THE IDENTITY, which is a FACT about the
    geometry and never a second code path.
    """
    widths, W = arm
    require_arm(widths, W, chunk, split, d_v=cell.d_v)
    assert body in ("split", "carved")
    st = list(carry_ops.build_stamp(widths, W, chunk, split, modes, cell.d_v))
    xch_bytes = st[24]
    if body == "carved":
        assert not split and xch_bytes > 0, (
            f"this cell asked for the CARVED body and the binary dispatched the "
            f"channel-split one (transit bytes {xch_bytes}): the transit and the "
            f"two-sided carve would go ungated")
    else:
        assert split and xch_bytes == 0, (
            f"this cell asked for the channel-split body and got a carved one "
            f"(transit bytes {xch_bytes})")
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
        window=W, chunk=chunk, split=split, level_modes=modes,
        state_in=state_in, state_out=plane, atom_bits=bits)
    return num, den, plane


def ref(cell, arm, state_in=None):
    widths, W = arm
    read, write, g, v = cell.ref_levels()
    return reference.inter_reference(read, write, g, v, widths, W, state_in=state_in)


# ---------------------------------------------------------------- the binary

def test_build_stamp_is_this_source():
    stamp = assert_fresh_binary()
    assert stamp[3] > 0, "the arm census must report a register count"


def test_arms_are_the_declared_shape_rows():
    """AN ARM IS A SHAPE (K50), and the table is asked for the law relating its fields.

    WHAT THE BINARY REPORTS IS A SUBSET (K36): a default build compiles the SHIPPED rows
    only, so `arms()` is asserted to be declared rows in the declared order.

    THE KEY LOST ITS TOPOLOGY AND ITS DECLARATION.  Before K50 a row was
    `(D, B, k, m, d_v, W, C, carve, level_modes)` and every declaration needed its own
    compiled arm -- the reason the shipped set could not travel ("it's not reasonable for
    people using a Python library to register their arms in a kernel library when it
    doesn't compile for their thing").  A row is now
    `(D, d_v, W, C, warps, WPS, split)`: the REGISTER AND SMEM PLAN, plus which body.
    So the properties to assert are that the plan is internally consistent and that the
    ADDRESSING is a function of the call, not of the row.
    """
    declared = [tuple(row[:7]) for row in gen_shards.CARRY_ARMS]
    reported = [tuple(a[:7]) for a in carry_ops.arms()]
    unknown = [row for row in reported if row not in declared]
    assert not unknown, f"the binary reports rows that are not declared: {unknown}"
    assert reported == [row for row in declared if row in set(reported)], (
        "the built rows must be the declared rows, in the declared order")
    #: THE ROW'S OWN LAW, asked of the DECLARED rows and not of the subset a build
    #: carries.  `PK`, `NSW` and `NSR` are DELETED axes (P76 "AXIS PRUNING"): the read
    #: stream count IS the warp count and the write count is `kWarps / WPS`, so there is
    #: nothing left to declare about them and nothing left to disagree.
    for row in gen_shards.CARRY_ARMS:
        D, d_v, W, C, warps, WPS, split = row
        assert d_v == D_V, f"{row}: M1P builds one value width"
        assert (warps, WPS) == (4, 1), f"{row}: four warps, one warp per stream"
        assert split in (0, 1), f"{row}: the body flag is a body"
        assert 1 <= D <= 4, f"{row}: the block is built to depth four"
    #: EVERY BODY MUST BE REACHABLE AT ITS DEPTH: the channel-split control exists so a
    #: dense measurement has a body that carves nothing, and G4 deletes it.
    depths = {row[0] for row in gen_shards.CARRY_ARMS}
    for D in depths:
        bodies = {row[6] for row in gen_shards.CARRY_ARMS if row[0] == D}
        assert 1 in bodies, f"depth {D} carries no channel-split control"
    #: AND THE ROW'S PLAN AGAINST THE DEVICE'S: `BC` is the register law's
    #: (leaves-per-warp times warps) and the SMEM the arm's own geometry, both read back.
    for arm in carry_ops.arms():
        D, d_v, W, C, warps, WPS, split, BC, smem, cta = arm[:10]
        assert 32 * warps == BC, f"{arm}: BC is the register law's, not a declaration"
        assert cta >= 1 and smem > 0, f"{arm}: an arm that reaches no CTA is not an arm"


def test_the_addressing_is_a_function_of_the_call_and_not_of_the_arm():
    """THE SHIP PRINCIPLE, as a test: ONE binary, EVERY former mode word.

    K50's ruling is that a user of the Python package never registers an arm.  The
    former arm list carried seven distinct mode words across four topologies; every one
    of them must now resolve on the SAME row, with the derivation -- not the row --
    telling the spans apart.  A binary that quietly served one declaration's geometry to
    another would pass every value test in this file, so what is asserted is that the
    census's derived vectors actually DIFFER between declarations that should differ and
    COINCIDE where the mode word makes the two sides agree.
    """
    assert_fresh_binary()
    seen = {}
    for widths, modes in (((256, 256), 0), ((256, 256), 1), ((256, 256), 3),
                          ((256, 256), 6), ((256, 256), 10), ((16, 16, 16), 38),
                          ((16, 16, 16, 16), 102)):
        b = block(widths, modes, ns=4)
        seen[(widths, modes)] = (tuple(b["r_grid"]), tuple(b["w_grid"]),
                                 b["identity_map"], tuple(b["r_grid_div"]),
                                 tuple(b["w_grid_div"]))
    #: THE SUB-BOX IS PER SIDE (G2c): each side selects a member of the generated shape
    #: set from its OWN order, so the mode word decides WHICH LEVEL a side carves -- not
    #: only which stream holds which sub-box.  The flagship's ALTERNATING word therefore
    #: gives the read side the level-1 carve and the write side the level-0 carve (the
    #: parent's two-sided carve, restored); its TIED word (`MO = 3`, both sides sparse at
    #: level 0) gives both the same member with the same offsets, the identity map.
    assert seen[((256, 256), 6)][:2] == ((1, 4), (4, 1)), (
        "the alternating word must carve the two sides at OPPOSITE levels")
    assert seen[((256, 256), 3)][:2] == ((4, 1), (4, 1)), (
        "the tied word must give both sides the same carve")
    assert seen[((256, 256), 6)][2] == 0 and seen[((256, 256), 3)][2] == 1
    assert len({v[:2] for v in seen.values()}) > 1, (
        "every declaration derived the same carve: the mode word is not reaching the "
        "derivation and mode-routed ownership is not being tested")


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

def test_m1_state_io_and_fold_dense():
    """The final state alone, on the tiny cell.  A leaf-order error surfaces here
    as a PERMUTATION rather than as noise, which is why the state is gated first."""
    assert_fresh_binary()
    cell = Cell(SMALL[0], T=256, k_tok=None, seed=1)
    _, _, plane = run(cell, SMALL, SPLIT)
    _, _, s_ref = ref(cell, SMALL)
    got = canonical_from_plane(plane)
    want = s_ref.reshape(cell.B * cell.H, cell.N, cell.d_v + 1)
    assert relative(got, want) < BF16_RTOL


def test_m1_entry_state_is_carried():
    assert_fresh_binary()
    cell = Cell(SMALL[0], T=128, k_tok=None, seed=2, state_in=True)
    s_in = cell.state_in_plane()
    _, _, plane = run(cell, SMALL, SPLIT, state_in=s_in)
    s0 = cell.state_in.reshape(cell.B, cell.H, cell.N, cell.d_v + 1)
    _, _, s_ref = ref(cell, SMALL, state_in=s0)
    got = canonical_from_plane(plane)
    want = s_ref.reshape(cell.B * cell.H, cell.N, cell.d_v + 1)
    assert relative(got, want) < BF16_RTOL


def test_m1_a_continuation_that_touches_a_strict_subset():
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

    _, _, plane = run(first, SMALL, SPLIT)
    _, _, s1_ref = ref(first, SMALL)

    bits = planes.atom_bits(_facts_plane(second, second.write_bf), SMALL[0],
                            read_plane=_facts_plane(second, second.read_bf))
    written = planes.written_atoms(
        planes.atom_bits(_facts_plane(first, first.write_bf), SMALL[0],
                         read_plane=_facts_plane(first, first.read_bf)))
    idle = written & (bits == 0)
    assert bool(idle.any()), "the truncated cell reaches every atom the first one did"

    before = plane.clone()
    num, den, plane = run(second, SMALL, SPLIT, state_in=plane)
    _, _, s2_ref = ref(second, SMALL, state_in=s1_ref)
    got = canonical_from_plane(plane)
    assert relative(got, s2_ref.reshape(second.B * second.H, second.N,
                                        second.d_v + 1)) < BF16_RTOL
    assert torch.equal(plane[idle], before[idle]), (
        "an idle-resident atom moved: the exit sweep stored an atom this call does "
        "not write")

    #: THE READ HALF OF THE CENSUS: the same call over a plane whose idle atoms hold
    #: something else entirely must produce the same `y`, because nothing reads them.
    perturbed = before.clone()
    perturbed[idle] = -1234.5
    num2, den2, _ = run(second, SMALL, SPLIT, state_in=perturbed)
    assert torch.equal(num, num2) and torch.equal(den, den2), (
        "perturbing an idle-resident atom moved the readout: the entry sweep is "
        "reading an atom this call does not read")


# ------------------------------------------ M2: readout + fan-in, dense cells

def test_m2_inter_dense_small():
    assert_fresh_binary()
    cell = Cell(SMALL[0], T=256, k_tok=None, seed=3)
    num, den, plane = run(cell, SMALL, SPLIT)
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


def test_m2_inter_dense_flagship():
    """The flagship topology at `L = 1024`.  `N = 65536` is the largest cell whose
    fp64 reference fits: `[B,T,H,N]` fp64 is 537 MB at this `L`, and the reference
    holds one window's worth at a time."""
    assert_fresh_binary()
    cell = Cell(FLAG[0], T=1024, k_tok=None, seed=5)
    num, den, _ = run(cell, FLAG, SPLIT)
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
def test_m3_two_sided_alt_small(k_tok):
    assert_fresh_binary()
    cell = alt_cell(SMALL[0], T=256, k_tok=k_tok, seed=10 + k_tok)
    num, den, plane = run(cell, SMALL, SPLIT)
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
    num, den, _ = run(cell, FLAG, SPLIT)
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
    num, den, _ = run(cell, FLAG, SPLIT)
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
def test_f1_carved_alt_small(k_tok):
    assert_fresh_binary()
    cell = alt_cell(SMALL[0], T=256, k_tok=k_tok, seed=60 + k_tok)
    num, den, plane = run(cell, SMALL, CARVED, modes=ALT2, body="carved")
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
    num, den, _ = run(cell, FLAG, CARVED, modes=ALT2, body="carved")
    n_ref, d_ref, _ = ref(cell, FLAG)
    assert relative(num.reshape(cell.B, cell.H, cell.T, cell.d_v).permute(0, 2, 1, 3),
                    n_ref) < BF16_RTOL
    assert relative(den.reshape(cell.B, cell.H, cell.T).permute(0, 2, 1), d_ref) < BF16_RTOL


@pytest.mark.parametrize("k_tok", [4, 16])
def test_f1_carved_multi_level_deep4(k_tok):
    """THE MULTI-LEVEL CARVE.  At depth four the derived vectors are write
    `[1, 4, 2, 4]` and read `[2, 1, 4, 4]` -- four levels, both sides carving more than
    one of them, and the two orders differing at every level.  It is the widest disorder
    any declaration hands the transit, so it is where the row map's generality is proved.
    (The WRITE vector moved with the balanced box: at the old `[2, 2, 4, 8]` owner spans
    it read `[1, 2, 2, 8]`.  Same carve rule, a different box under it.)"""
    assert_fresh_binary()
    cell = alt_cell(DEEP4[0], T=256, k_tok=k_tok, seed=80 + k_tok)
    num, den, plane = run(cell, DEEP4, CARVED, modes=ALT4, body="carved")
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
    num, den, _ = run(cell, FLAG, CARVED, modes=BOTH2, body="carved")
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
    widths, W = FLAG
    T = 3 * W + W // 2
    assert (T + W - 1) // W >= 3, "the cell must cross at least three window lines"
    cell = alt_cell(FLAG[0], T=T, k_tok=k_tok, seed=110 + k_tok)
    num, den, plane = run(cell, FLAG, CARVED, modes=ALT2, body="carved")
    n_ref, d_ref, s_ref = ref(cell, FLAG)
    assert relative(num.reshape(cell.B, cell.H, cell.T, cell.d_v).permute(0, 2, 1, 3),
                    n_ref) < BF16_RTOL
    assert relative(den.reshape(cell.B, cell.H, cell.T).permute(0, 2, 1), d_ref) < BF16_RTOL
    got = canonical_from_plane(plane)
    assert relative(got, s_ref.reshape(cell.B * cell.H, cell.N, cell.d_v + 1)) < BF16_RTOL


def test_the_census_reports_the_transit_and_its_row_map_off_the_device():
    """`kXchBytes` and the IDENTITY-MAP fact, read back off the built binary.

    K50 DELETED THE TIED BODY ("one transit, always; the tied case is a transit whose row
    map is the identity, same code, same register plan").  So the census can no longer
    report an ELISION, and asserting one would be asserting a body that does not exist.
    What it reports instead is the FACT -- is this call's row map the identity -- which is
    exactly what makes a wasted transit visible in the ledger, and the LAW is that the
    carved body always sizes one whole `[consumer][leaf][channel]` region while the
    channel-split body has no transit at all.
    """
    assert_fresh_binary()
    identities = []
    for arm in carry_ops.arms():
        D, d_v, W, C, warps, WPS, split, BC, smem, cta = arm[:10]
        widths = {2: (256, 256), 3: (16, 16, 16), 4: (16, 16, 16, 16)}[D]
        for modes in (0, 3, 6):
            st = list(carry_ops.build_stamp(widths, W, C, bool(split), modes, d_v))
            xch_bytes, identity = st[24], st[25]
            if split:
                assert xch_bytes == 0, f"{arm}: the channel-split body has no transit"
            else:
                assert xch_bytes == BC * (d_v + 8) * 2, (
                    f"{arm} at modes={modes}: the carved body always runs ONE transit, "
                    f"sized to a whole [consumer][leaf][channel] region -- an elided one "
                    f"would be the deleted tied body")
                identities.append((D, modes, identity))
    #: NON-VACUITY: the identity map must actually be REACHED by some declaration, or the
    #: fact is untested and a wasted transit would never be visible.
    assert any(i for _, _, i in identities), (
        "no declaration reached the identity row map: the fact that replaced `kTied` is "
        "not being exercised")
    assert any(not i for _, _, i in identities), (
        "every declaration reached the identity map: the permuted transit is untested")


def test_m3_lists_differ_when_the_sides_do():
    """NON-VACUITY: the ALT cell must actually drive `nr != nw` somewhere, or the
    two-sided gate above is a dense gate wearing a sparse name."""
    cell = alt_cell(SMALL[0], T=256, k_tok=4, seed=99)
    widths, W = SMALL
    b = block(widths, ns=1)
    s, g, owners = b["span"], b["grid"], b["owners"]
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
def test_m2_inter_dense_deep(arm):
    """Dense both sides at depth three and four: the state plane, the readout and
    the mass column, against the fp64 oracle in CANONICAL order -- so a deep arm's
    mixed-radix leaf order surfaces here as a permutation, not as noise."""
    assert_fresh_binary()
    cell = Cell(arm[0], T=256, k_tok=None, seed=6)
    num, den, plane = run(cell, arm, SPLIT)
    n_ref, d_ref, s_ref = ref(cell, arm)
    assert relative(num.reshape(cell.B, cell.H, cell.T, cell.d_v).permute(0, 2, 1, 3),
                    n_ref) < BF16_RTOL
    assert relative(den.reshape(cell.B, cell.H, cell.T).permute(0, 2, 1), d_ref) < BF16_RTOL
    got = canonical_from_plane(plane)
    assert relative(got, s_ref.reshape(cell.B * cell.H, cell.N, cell.d_v + 1)) < BF16_RTOL


@pytest.mark.parametrize("arm,k_tok", [(DEEP3, 4), (DEEP3, 8), (DEEP4, 2), (DEEP4, 4)],
                         ids=["d3-k4", "d3-k8", "d4-k2", "d4-k4"])
def test_m3_two_sided_alt_deep(arm, k_tok):
    """The ALTERNATING declaration at depth: `D` clause levels, the two sides'
    subsets disjoint, scattered `k_tok` cells."""
    assert_fresh_binary()
    cell = alt_cell(arm[0], T=256, k_tok=k_tok, seed=30 + k_tok + len(arm[0]))
    num, den, plane = run(cell, arm, SPLIT)
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
    widths, W = arm
    cell = alt_cell(widths, T=256, k_tok=2, seed=77 + len(widths))
    b = block(widths, ns=1)
    s, g, owners = b["span"], b["grid"], b["owners"]
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


@pytest.mark.parametrize("arm,spans", [(SMALL, (8, 16)), (FLAG, (8, 16)),
                                       (DEEP3, (2, 4, 16)), (DEEP4, (2, 2, 2, 16))],
                         ids=["small", "flag", "d3", "d4"])
def test_the_derived_box_reproduces_the_compile_time_law(arm, spans):
    """THE PARITY PROOF: the RUNTIME span rule against the constants `BoxPlan` produced.

    K50's span rule is "balance BC over levels; a level narrower than its balanced share
    takes its full width and the remainder balances over the others", and its amended
    ownership ruling names the invariant the balance runs under: "a box is a union of
    whole atoms".  Balance ALONE does not keep that invariant -- at `D = 3, B = 16` it
    gives `[4, 4, 8]` and at `D = 4, B = 8` it gives `[2, 4, 4, 4]`, and in both an atom
    (16 CONSECUTIVE CANONICAL leaves) straddles two owners, so two owners store the same
    page at page grain.  That is measured, not argued: the unconstrained form failed nine
    fp64 cells, STATE PLANE only, on BOTH bodies, at exactly those two topologies.

    Balance UNDER the atom invariant reproduces the compile-time law's answer at every
    built topology -- which is what makes this a parity proof and not a re-declaration.
    K50's CORRECTION rules the eventual relaxation ("prefill owns whole ROWS ... the fix
    is ROW-GRAIN activity masks from the facts pass"); that machinery is the page I/O and
    the facts bit emission, and it is not this stage's to write.
    """
    widths, W = arm
    b = block(widths, ns=1)
    assert tuple(b["span"]) == spans, (
        f"the derived box for {widths} is {tuple(b['span'])}, not the law's {spans}")
    prod = 1
    for x in b["span"]:
        prod *= x
    assert prod == b["bc"] == 128, "the spans must fill the owner's box exactly"
    for l, (sp, wd) in enumerate(zip(b["span"], b["width"])):
        assert wd % sp == 0 and b["grid"][l] == wd // sp, (
            f"level {l}: the span must be an aligned run inside its level")
    assert b["min_width"] == min(widths), "min_l B_l is reported, not refused"
    #: THE ATOM INVARIANT ITSELF, asserted directly: walking innermost outward, the box
    #: must hold WHOLE whatever share of a 16-leaf atom each level contributes.
    need, l = 16, len(widths) - 1
    while need > 1 and l >= 0:
        take = min(widths[l], need)
        assert b["span"][l] >= take, (
            f"level {l}: the box spans {b['span'][l]} of the {take} digits a page atom "
            f"needs there, so an atom straddles two owners")
        need //= take
        l -= 1


def test_the_derivation_refuses_only_what_the_card_refuses():
    """THE REFUSAL SET, exactly as K50 states it, and nothing beyond it.

    A shipped kernel that refused more than this would put the arm-registration problem
    back: the whole point is that a topology nobody compiled still runs.
    """
    #: a width that is not a power of two
    with pytest.raises(RuntimeError, match="power of two"):
        block((100, 100), ns=1)
    #: a topology that cannot hold ONE owner's box -- named at the LEVEL that cannot
    #: hold its share, which is the field a caller can act on.
    with pytest.raises(RuntimeError, match="narrower than the owner box's span"):
        block((4, 4), ns=1)
    #: THE ONE WIDTH LAW THE SHAPE OWNS (G2b): the box's spans are compile-time
    #: `f(D, BC)`, so a level narrower than its span is refused rather than served by a
    #: narrower slot layout -- there is no narrower layout to fall back to.  Under K50's
    #: restored floor (`B_l >= 16`) no lawful state reaches it; `min_l B_l` stays a
    #: CENSUS FACT so an unlawful one is visible in the ledger rather than silently run.
    with pytest.raises(RuntimeError, match="narrower than the owner box's span"):
        block((8, 8, 8, 8), ns=1)
    b = block((16, 16, 16, 16), ns=1)
    assert tuple(b["span"]) == (2, 2, 2, 16) and tuple(b["grid"]) == (8, 8, 8, 1)
    assert b["min_width"] == 16
    #: THE ALIGNMENT REFUSAL IS REAL AND IS NAMED: a level whose column base is not a
    #: whole number of the runs read out of it would issue an unaligned wide load, so the
    #: seam refuses it rather than the kernel faulting on it.
    with pytest.raises(RuntimeError, match="not a whole number of the runs"):
        block((8, 16), ns=1)


# ------------------------------------------------------------------ the refusal

def test_an_unbuilt_shape_is_refused_and_no_declaration_ever_is():
    """THE SEAM REFUSES WHAT IT HAS NO BODY FOR -- AND AFTER K50 THAT IS A SHAPE.

    `carry.cu`'s law-3 retry used to re-dispatch an unbuilt DECLARATION with `mo` forced
    to zero, so a caller asking for a declaration nothing was built for silently got the
    DENSE_BOTH body and no diagnostic -- how a 5x loss stayed hidden for nine stages
    (`## BISECT (corrected)`).  K50 removes the CLASS: a declaration cannot be unbuilt,
    because the mode word never reaches a template.  So the two halves of this test are
    (1) every former mode word resolves, at every depth, and (2) an unbuilt SHAPE -- a
    window no row carries -- still refuses by name.
    """
    assert_fresh_binary()
    widths, W = FLAG
    cell = Cell(widths, T=256, k_tok=None, seed=3)
    #: (1) the depth-four ALTERNATING word at the FLAGSHIP topology: before K50 this was
    #: a declaration with no body at this topology, and the retry served it dense.
    num, den, _ = carry_ops.carry_inter(
        cell.read_bf, cell.write_bf, cell.g_bf, cell.v_bf, widths,
        window=W, chunk=32, split=CARVED, level_modes=ALT4)
    assert torch.isfinite(num).all() and torch.isfinite(den).all()
    #: (2) a window no arm carries is a SHAPE the binary does not have, and it must name
    #: the key rather than degrade to one it does.
    with pytest.raises(RuntimeError, match="unbuilt carry arm"):
        carry_ops.carry_inter(cell.read_bf, cell.write_bf, cell.g_bf, cell.v_bf, widths,
                              window=256, chunk=32, split=CARVED, level_modes=ALT4)
