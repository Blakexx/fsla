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

import pytest
import torch

from rola.ops import carry as carry_ops
from tests.k31 import reference
from tests.k31.fixtures import Cell, assert_fresh_binary, canonical_from_lattice, relative
from tests.oracle.tolerances import BF16_RTOL

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

#: (widths, k, m, W) -- the two built arms this suite drives.
SMALL = ((64, 64), 4, 8, carry_ops.WINDOW_SMALL)  # N = 4096,  BC = 128, owners = 32
FLAG = ((256, 256), 4, 8, carry_ops.WINDOW)       # N = 65536, BC = 128, owners = 512

CHANNEL = carry_ops.CHANNEL_SPLIT


def run(cell, arm, carve, chunk=32, state_in=None):
    widths, k, m, W = arm
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
    got = {tuple(a[:7]) for a in carry_ops.arms()}
    assert got == {(2, 256, 4, 8, carry_ops.WINDOW, 32, 1),
                   (2, 256, 4, 8, carry_ops.WINDOW_WIDE, 32, 1),
                   (2, 64, 4, 8, carry_ops.WINDOW_SMALL, 32, 1)}, got
    for arm in carry_ops.arms():
        D, B, k, m, W, C, carve, BC, owners, smem = arm
        _, _, _, bc_host, owners_host = carry_ops.box_shape([B] * D, k, m)
        assert (BC, owners) == (bc_host, owners_host), (
            "the device box and the host solver disagree about the owner's shape")


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

def test_m1_state_io_and_fold_dense(carve=CHANNEL):
    """The final state alone, on the tiny cell.  A leaf-order error surfaces here
    as a PERMUTATION rather than as noise, which is why the state is gated first."""
    assert_fresh_binary()
    cell = Cell(SMALL[0], T=256, k_tok=None, seed=1)
    _, _, plane = run(cell, SMALL, carve)
    _, _, s_ref = ref(cell, SMALL)
    got = canonical_from_lattice(plane, SMALL[0], SMALL[1], SMALL[2])
    want = s_ref.reshape(cell.B * cell.H, cell.N, cell.d_v + 1)
    assert relative(got, want) < BF16_RTOL


def test_m1_entry_state_is_carried(carve=CHANNEL):
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

def test_m2_inter_dense_small(carve=CHANNEL):
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


def test_m2_inter_dense_flagship(carve=CHANNEL):
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

def alt_cell(widths, T, k_tok, seed):
    """THE PRIMARY SPARSE STRUCTURE (Blake ruling 2026-08-19): per level at most ONE
    sparse side, and the sparse side ALTERNATES -- level 0 write-sparse/read-dense,
    level 1 read-sparse/write-dense.  The two sides' clause sets are therefore
    DIFFERENT level subsets, which is exactly what the two-sided schedule exists for
    and what a union-routed cell would never exercise."""
    from tests.k31.fixtures import simplex
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(seed)
    cell = Cell(widths, T=T, k_tok=None, seed=seed)
    shape = (cell.B, T, cell.H)
    read = (simplex(shape + (widths[0],), None, gen, device),
            simplex(shape + (widths[1],), k_tok, gen, device))
    write = (simplex(shape + (widths[0],), k_tok, gen, device),
             simplex(shape + (widths[1],), None, gen, device))
    cell.read, cell.write = read, write
    cell.read_bf = tuple(t.to(torch.bfloat16) for t in read)
    cell.write_bf = tuple(t.to(torch.bfloat16) for t in write)
    return cell


@pytest.mark.parametrize("k_tok", [4, 16])
def test_m3_two_sided_alt_small(k_tok, carve=CHANNEL):
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
    num, den, _ = run(cell, FLAG, CHANNEL)
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
