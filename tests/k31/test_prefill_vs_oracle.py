# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""K31 integration — the COMBINED operator against the canonical fp64 oracle.

THE REFERENCE IS `rola.ops.naive.naive_rola` DIRECTLY, and that is the point of
this file. The rung's earlier gates compared the carry against a windowed
reference and subtracted the intra term on the host; nothing then checked that
the two kernels' window grids were the SAME grid, only that each matched its own
statement of one. Here the two kernels run as one call and the whole answer is
compared against the recurrence itself — so a window-grid disagreement, a
double-counted diagonal or a dropped tile-pair has nowhere to hide.

The oracle returns the NORMALIZED readout; the operator returns the numerator and
the mass undivided, so the comparison is through the ratio the operator's own
docstring names. The band is the repository's one bf16 family bound.
"""
from __future__ import annotations

import zlib

import pytest
import torch

from rola.ops import carry as carry_ops
from rola.ops import intra as intra_ops
from rola.ops.constants import READOUT_EPS
from rola.ops.naive import naive_rola
from rola.ops.prefill import prefill
from rola.routing.types import IndependentRouting, SoftmaxActivation, Topology
from tests.k31.fixtures import assert_fresh_binary, canonical_from_lattice, relative
from tests.k31.intra_reference import _simplex
from tests.oracle.tolerances import BF16_RTOL

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required")

#: the flagship topology, and the two windows it is built at.  `W = 512` is the
#: one that divides a power-of-two length, so it carries the long cells;
#: `W = 384` -- the architecture doc's nominal sizing -- carries a length that is
#: a whole number of ITS windows, which is the only difference between them here.
WIDTHS = (256, 256)
K, M = 4, 8

DENSE = (intra_ops.DENSE_BOTH, intra_ops.DENSE_BOTH)
#: THE PRIMARY SPARSE STRUCTURE: per level at most one sparse side, and the
#: sparse side ALTERNATES, so the two sides' clause sets are different level
#: subsets -- what the two-sided schedule exists for.
ALT = (intra_ops.WRITE_SPARSE, intra_ops.READ_SPARSE)
#: ALL-WRITE-SPARSE: entmax on BOTH write levels, a softmax (dense) read side.
#: The read side then has NO sparse level -- which under K33 is the trivial
#: one-bucket readout -- and the write side is sparse at two levels at once, the
#: multi-level distribution the exactness claim has to survive.
WRITE_BOTH = (intra_ops.WRITE_SPARSE, intra_ops.WRITE_SPARSE)
#: ONE-SPARSE-READ: a single entmax read level and nothing else sparse.
READ_ONE = (intra_ops.READ_SPARSE, intra_ops.DENSE_BOTH)


def cell(T, modes, k_tok=None, clustered=False, seed=0, d_v=64, device="cuda"):
    """One realized cell as ``[B=1, T, H=1, width]`` per-level tuples.

    ``clustered`` is the COHORT form: a run of that many consecutive tokens draws
    ONE contiguous digit window, which is the structure the intra's slab gate and
    the carry's whole-window skip both exist to exploit.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    read, write = [], []
    for level, mode in enumerate(modes):
        width = WIDTHS[level]
        read_k = k_tok if mode in (intra_ops.READ_SPARSE, intra_ops.SHARED_SPARSE) else None
        write_k = k_tok if mode in (intra_ops.WRITE_SPARSE, intra_ops.SHARED_SPARSE) else None
        read.append(_simplex((1, T, width), read_k, gen, device, clustered))
        write.append(_simplex((1, T, width), write_k, gen, device, clustered))
    read = tuple(x.reshape(1, T, 1, WIDTHS[i]) for i, x in enumerate(read))
    write = tuple(x.reshape(1, T, 1, WIDTHS[i]) for i, x in enumerate(write))
    g = torch.rand(1, T, 1, device=device, dtype=torch.float64, generator=gen) + 0.5
    v = torch.randn(1, T, 1, d_v, device=device, dtype=torch.float64, generator=gen)
    #: THE OPERAND PLANES ARE bf16 AND THE ORACLE READS THE SAME BYTES: comparing
    #: an fp64 oracle of the fp64 draws against a kernel fed their bf16 images
    #: would measure the CAST, which is not what is under test.
    return (tuple(x.to(torch.bfloat16) for x in read),
            tuple(x.to(torch.bfloat16) for x in write),
            g.to(torch.bfloat16), v.to(torch.bfloat16))


def oracle(read_bf, write_bf, g_bf, v_bf, state_in=None):
    routing = IndependentRouting(width=1, tied=False, read=SoftmaxActivation(),
                                 write=SoftmaxActivation())
    topo = Topology(levels=tuple(routing.at(w) for w in WIDTHS))
    return naive_rola(v_bf.double(), tuple(x.double() for x in read_bf),
                      tuple(x.double() for x in write_bf), None, g_bf.double(), topo, None,
                      "global", initial_state=state_in, output_final_state=True)


def readout(num, den, T, d_v=64):
    """``[BH, T, d_v]`` / ``[BH, T]`` seen as the oracle's ``[B, T, H, d_v]``."""
    y = num / (den[..., None].double() + READOUT_EPS)
    return y.reshape(1, 1, T, d_v).permute(0, 2, 1, 3)


CELLS = [
    ("dense-W512", 2048, 512, DENSE, None, False),
    ("dense-W384", 1920, 384, DENSE, None, False),
    ("alt-k4-W512", 2048, 512, ALT, 4, False),
    ("alt-k16-W512", 2048, 512, ALT, 16, False),
    ("alt-k4-cohort64-W512", 2048, 512, ALT, 4, 64),
    ("alt-k16-cohort64-W384", 1920, 384, ALT, 16, 64),
    ("write-both-k4-W512", 2048, 512, WRITE_BOTH, 4, False),
    ("read-one-k4-W512", 2048, 512, READ_ONE, 4, False),
    #: THE MULTI-LIVE-DIGIT CELL.  `clustered` draws a CONTIGUOUS digit window, so
    #: with `k_tok = 16` against the lattice spans `(16, 8)` a token routinely has
    #: several live digits inside ONE owner's run at the keyed level -- which is the
    #: distribution K33's exactness claim rests on and which
    #: `test_a_multi_live_digit_entry_appears_once_per_digit` counts before it is
    #: relied on here.
    ("alt-k16-cohort1-W512", 2048, 512, ALT, 16, 1),
]


def test_a_multi_live_digit_entry_appears_once_per_digit():
    """K33's exactness is BY DISTRIBUTION over the key level's live digits.

    A bucket entry is a (token, key digit) pair, so a token with several live
    digits inside one owner's run at the keyed level is scheduled SEVERAL TIMES,
    each with that digit's own coefficient, and the sum over those entries is the
    contraction the one-bucket form would have computed in a single panel. This
    cell (a) counts the multi-digit tokens, so the claim is not tested against a
    draw that has none, and (b) checks the result against the fp64 oracle.

    The count is a property of the DRAW and is computed here on the host. The
    kernel is never told it, and nothing in the kernel branches on it.
    """
    assert_fresh_binary()
    T, window, k_tok = 2048, 512, 16
    read, write, g, v = cell(T, ALT, k_tok=k_tok, clustered=1, seed=4242)
    #: ALT is (WRITE_SPARSE, READ_SPARSE), so the READ side keys level 1 and its
    #: owner run is `s_1 = 8` digits wide; the WRITE side keys level 0 at `s_0 = 16`.
    for side, level, span in ((read, 1, 8), (write, 0, 16)):
        live = (side[level][0, :, 0].reshape(T, -1, span) != 0).sum(-1)
        multi = int((live > 1).sum())
        assert multi > 0, (
            f"side keyed at level {level}: this draw has no token with two live "
            f"digits in one owner's {span}-digit run, so it cannot test the claim")
        assert int(live.max()) >= 2

    num, den, plane = prefill(read, write, g, v, WIDTHS, k=K, m=M, window=window,
                              modes=ALT)
    y_ref, s_ref = oracle(read, write, g, v)
    assert relative(readout(num, den, T), y_ref) < BF16_RTOL
    got = canonical_from_lattice(plane, WIDTHS, K, M)
    assert relative(got, s_ref.reshape(1, 65536, 65)) < BF16_RTOL
    del num, den, plane, y_ref, s_ref, got
    torch.cuda.empty_cache()


@pytest.mark.parametrize("name,T,window,modes,k_tok,clustered", CELLS,
                         ids=[c[0] for c in CELLS])
def test_prefill_matches_the_fp64_oracle(name, T, window, modes, k_tok, clustered):
    assert_fresh_binary()
    #: A SEED THAT IS A PROPERTY OF THE CELL: `hash(str)` is salted per process,
    #: so a cell seeded from it is a different draw on every run and a failure
    #: cannot be reproduced from its own name.
    read, write, g, v = cell(T, modes, k_tok=k_tok, clustered=clustered,
                             seed=zlib.crc32(name.encode()) & 0xFFFF)
    num, den, plane = prefill(read, write, g, v, WIDTHS, k=K, m=M, window=window,
                              modes=modes)
    y_ref, s_ref = oracle(read, write, g, v)
    err = relative(readout(num, den, T), y_ref)
    assert err < BF16_RTOL, f"{name}: the readout leaves the bf16 band at {err:.3e}"
    got = canonical_from_lattice(plane, WIDTHS, K, M)
    s_err = relative(got, s_ref.reshape(1, 65536, 65))
    assert s_err < BF16_RTOL, f"{name}: the final state leaves the band at {s_err:.3e}"
    del num, den, plane, y_ref, s_ref, got
    torch.cuda.empty_cache()


def test_a_chained_call_equals_one_long_call():
    """THE STATE IS THE WHOLE CONTRACT of a chunked prefill: two calls over the
    halves, the first's `state_out` seeding the second, must be the one long
    call. The split is placed on a WINDOW line, which is where a chunk boundary
    is legal at all -- the window grid is global, so a split inside a window
    would ask the second call to carry pairs neither term owns."""
    assert_fresh_binary()
    T, window = 2048, 512
    read, write, g, v = cell(T, DENSE, seed=7)
    whole_num, whole_den, whole_state = prefill(read, write, g, v, WIDTHS, k=K, m=M,
                                                window=window, modes=DENSE)
    cut = 1024  # a whole number of windows
    first = tuple(x[:, :cut] for x in read), tuple(x[:, :cut] for x in write)
    second = tuple(x[:, cut:] for x in read), tuple(x[:, cut:] for x in write)
    n0, d0, s0 = prefill(first[0], first[1], g[:, :cut], v[:, :cut], WIDTHS, k=K, m=M,
                         window=window, modes=DENSE)
    n1, d1, s1 = prefill(second[0], second[1], g[:, cut:], v[:, cut:], WIDTHS, k=K, m=M,
                         window=window, modes=DENSE, state_in=s0)
    #: THE STATE IS BIT-IDENTICAL, and it is the claim that matters: an owner
    #: writes its own leaves and nothing reduces across CTAs there, so the state
    #: plane is deterministic, and the chain's only difference is where its entry
    #: state came from -- an fp32 plane written and read back exactly.
    assert torch.equal(s1, whole_state)
    #: THE OUTPUT IS NOT, AND CANNOT BE.  `o`/`den` are accumulated by a RED at
    #: the top (`fan_in_add`), so the order the owners' contributions land in is
    #: the scheduler's and differs run to run -- for the WHOLE call as much as
    #: for the chain.  The claim here is therefore agreement to fp32 rounding,
    #: which is far tighter than the bf16 conformance band and is what a
    #: reordered fp32 sum can actually promise.
    chained_num = torch.cat((n0, n1), dim=1)
    chained_den = torch.cat((d0, d1), dim=1)
    assert relative(chained_num, whole_num) < 1e-5
    assert relative(chained_den, whole_den) < 1e-5
    #: and the chain is the oracle's, through the same ratio the cells use.
    y_ref, _ = oracle(read, write, g, v)
    err = relative(readout(chained_num, chained_den, T), y_ref)
    assert err < BF16_RTOL, f"the chained readout leaves the bf16 band at {err:.3e}"


def test_the_combined_operator_refuses_what_it_has_no_kernel_for():
    """The two refusals are STRUCTURAL, and stating them is how the gap stays a
    gap rather than becoming a host-side second code path."""
    read, write, g, v = cell(512, DENSE, seed=1)
    #: a partial trailing window: the intra's tile grid is a compile-time count.
    with pytest.raises(ValueError, match="whole number"):
        prefill(read, write, g, v, WIDTHS, k=K, m=M, window=384, modes=DENSE)
    #: a narrower topology: the intra kernel's level width is 256 by
    #: instantiation, so the carry's `(64, 64)` arm is an INTER-ONLY cell.
    with pytest.raises(ValueError, match="level width"):
        prefill(read, write, g, v, (64, 64), k=K, m=M, window=64, modes=DENSE)


def test_the_intra_term_is_actually_present():
    """NON-VACUITY. The carry alone must NOT pass the oracle gate: if it did,
    every cell above would be testing the carry and calling it the operator."""
    T, window = 1024, 512
    read, write, g, v = cell(T, DENSE, seed=3)
    inter_num, inter_den, _ = carry_ops.carry_inter(read, write, g, v, WIDTHS, k=K, m=M,
                                                    window=window)
    y_ref, _ = oracle(read, write, g, v)
    err = relative(readout(inter_num, inter_den, T), y_ref)
    assert err > 10 * BF16_RTOL, (
        f"the INTER term alone is within {err:.3e} of the whole recurrence; the "
        f"intra term this operator adds is not being tested by the cells above")
