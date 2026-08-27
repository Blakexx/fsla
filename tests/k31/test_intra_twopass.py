# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""P78 v2 Step A: the two-GEMM composition against the SAME fp64 references.

`rola.ops.intra_twopass` is a reference module, not a shipped path, so this file
gates it exactly as `test_intra_kernel.py` and `test_intra_backward.py` gate the
one-pass kernel: the forward against `tests/k31/intra_reference.py` (which
`test_intra_reference.py` pins to the oracle) and the reverse against
`tests/k31/backward_reference.py`'s intra half. One reference, two implementations
-- which is the only arrangement in which a comparison between them means
anything.

THE CELLS ARE THE KERNEL'S OWN, arm for arm: the three built windows, the three
level widths, and the support laws (dense, ALT, SHARED). The composition is dense
by construction -- P78 v2 gives up G's own sparsity -- so the modes change the
DRAW here, not the arithmetic, and a mode-blind implementation must still land in
the band on every one of them.
"""

from __future__ import annotations

import pytest
import torch

from rola.ops import intra_twopass as twopass
from rola.ops.intra import LEVEL_WIDTH, LEVEL_WIDTH_DEEP3, LEVEL_WIDTH_DEEP4
from tests.k31.intra_reference import (
    DENSE_BOTH,
    READ_SPARSE,
    SHARED_SPARSE,
    WINDOW,
    WRITE_SPARSE,
    intra_reference,
    make_cell,
)
from tests.oracle.tolerances import BF16_RTOL

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the composition is CUDA")

HEADS = 2
WINDOW_WIDE = 512
WINDOW_SMALL = 64

#: `(name, levels, level_width, window, modes, k_tok, windows)`.
CELLS = [
    ("dense-W384", 2, LEVEL_WIDTH, WINDOW, (DENSE_BOTH,) * 2, None, 8),
    ("alt-k4-W384", 2, LEVEL_WIDTH, WINDOW, (WRITE_SPARSE, READ_SPARSE), 4, 8),
    ("alt-k16-W384", 2, LEVEL_WIDTH, WINDOW, (WRITE_SPARSE, READ_SPARSE), 16, 8),
    ("shared-k16-W384", 2, LEVEL_WIDTH, WINDOW, (SHARED_SPARSE,) * 2, 16, 2),
    ("shared-k4-W384", 2, LEVEL_WIDTH, WINDOW, (SHARED_SPARSE,) * 2, 4, 2),
    ("dense-W512", 2, LEVEL_WIDTH, WINDOW_WIDE, (DENSE_BOTH,) * 2, None, 4),
    ("alt-k4-W512", 2, LEVEL_WIDTH, WINDOW_WIDE, (WRITE_SPARSE, READ_SPARSE), 4, 4),
    ("shared-k16-W512", 2, LEVEL_WIDTH, WINDOW_WIDE, (SHARED_SPARSE,) * 2, 16, 2),
    ("dense-W64", 2, LEVEL_WIDTH, WINDOW_SMALL, (DENSE_BOTH,) * 2, None, 16),
    ("alt-k4-W64", 2, LEVEL_WIDTH, WINDOW_SMALL, (WRITE_SPARSE, READ_SPARSE), 4, 16),
    ("d3-dense", 3, LEVEL_WIDTH_DEEP3, WINDOW_SMALL, (DENSE_BOTH,) * 3, None, 16),
    ("d3-alt-k4", 3, LEVEL_WIDTH_DEEP3, WINDOW_SMALL,
     (WRITE_SPARSE, READ_SPARSE, WRITE_SPARSE), 4, 16),
    ("d3-shared-k4", 3, LEVEL_WIDTH_DEEP3, WINDOW_SMALL, (SHARED_SPARSE,) * 3, 4, 16),
    ("d4-dense", 4, LEVEL_WIDTH_DEEP4, WINDOW_SMALL, (DENSE_BOTH,) * 4, None, 16),
    ("d4-alt-k4", 4, LEVEL_WIDTH_DEEP4, WINDOW_SMALL,
     (WRITE_SPARSE, READ_SPARSE) * 2, 4, 16),
    ("d4-shared-k2", 4, LEVEL_WIDTH_DEEP4, WINDOW_SMALL, (SHARED_SPARSE,) * 4, 2, 16),
]


def _relative(got, want):
    got, want = got.double(), want.double()
    return ((got - want).abs().max() / want.abs().max()).item()


def _draw(levels, width, window, modes, k_tok, windows, seed=11):
    return make_cell(2 * HEADS, windows * window, levels, modes, k_tok=k_tok or 4, seed=seed,
                     level_width=width)


@pytest.mark.parametrize("name,levels,width,window,modes,k_tok,windows", CELLS,
                         ids=[c[0] for c in CELLS])
def test_the_composition_matches_the_fp64_reference(name, levels, width, window, modes,
                                                    k_tok, windows):
    pread, pwrite, gwrite, v_bh = _draw(levels, width, window, modes, k_tok, windows)
    o, den = twopass.intra_terms(pread, pwrite, gwrite, v_bh, levels, width, window)
    o_ref, den_ref = intra_reference(pread, pwrite, gwrite, v_bh, levels, window=window,
                                     level_width=width)
    e_o, e_den = _relative(o, o_ref), _relative(den, den_ref)
    assert e_o < BF16_RTOL, f"{name}: the output leaves the bf16 band at {e_o:.3e}"
    assert e_den < BF16_RTOL, f"{name}: the mass leaves the bf16 band at {e_den:.3e}"


def test_the_bf16_chain_is_why_the_chain_dtype_is_declared():
    """TEETH on the ONE numeric decision the composition makes.

    P78 v2 stores the running product; carrying it in bf16 rounds it once per
    level, and at the D = 4 shared-support arm that is enough to leave the band.
    The declaration is therefore load-bearing, and this states the failure it
    prevents rather than leaving `chain_dtype=fp32` looking like a default.
    """
    name, levels, width, window, modes, k_tok, windows = CELLS[-1]
    pread, pwrite, gwrite, v_bh = _draw(levels, width, window, modes, k_tok, windows)
    o_ref, _ = intra_reference(pread, pwrite, gwrite, v_bh, levels, window=window,
                               level_width=width)
    o_bf16, _ = twopass.intra_terms(pread, pwrite, gwrite, v_bh, levels, width, window,
                                    chain_dtype=torch.bfloat16)
    o_fp32, _ = twopass.intra_terms(pread, pwrite, gwrite, v_bh, levels, width, window)
    assert _relative(o_bf16, o_ref) > BF16_RTOL > _relative(o_fp32, o_ref)


def test_the_group_is_a_residency_dial_and_not_an_arithmetic_one():
    """Windows are independent, so the L2 group carries no term of the answer.

    It does not carry a BIT-IDENTITY either, and the reason is worth stating: the
    group is the GEMM's batch count, and cuBLAS picks its tiling (and therefore
    its k-order) by batch, so the fp32 accumulation reassociates and the bf16 G
    rounds differently. The dial is numerically inert at the band's scale --
    two orders below it -- and a form that ever needed bit-identity across groups
    would have to pin the tiling, which is a reason the shipped form states its
    own GEMM rather than inheriting the library's choice.
    """
    name, levels, width, window, modes, k_tok, windows = CELLS[5]
    pread, pwrite, gwrite, v_bh = _draw(levels, width, window, modes, k_tok, windows)
    whole = twopass.intra_terms(pread, pwrite, gwrite, v_bh, levels, width, window,
                                group=10 ** 6)
    single = twopass.intra_terms(pread, pwrite, gwrite, v_bh, levels, width, window, group=1)
    for grouped, alone in zip(whole, single):
        drift = float((grouped - alone).abs().max() / alone.abs().max())
        assert drift < BF16_RTOL / 10, f"the group moved the answer by {drift:.3e}"


def test_the_write_back_reduces_rather_than_stores():
    """`fan_in_add` semantics, as the shipped op has them: the caller owns the buffers."""
    levels, width, window = 2, LEVEL_WIDTH, WINDOW
    modes = (DENSE_BOTH, DENSE_BOTH)
    pread, pwrite, gwrite, v_bh = _draw(levels, width, window, modes, None, 2, seed=3)
    once = twopass.intra_forward(pread, pwrite, gwrite, v_bh, levels, width, window)
    twice = twopass.intra_forward(pread, pwrite, gwrite, v_bh, levels, width, window)
    twopass.intra_forward(pread, pwrite, gwrite, v_bh, levels, width, window,
                          o=twice[0], den=twice[1])
    assert torch.equal(twice[0], once[0] * 2) and torch.equal(twice[1], once[1] * 2)


#: the reverse pass's cells -- `test_intra_backward.py`'s, which are stated at
#: D = 2, width 256, the two windows the reverse families are built at.
BACKWARD_CELLS = [
    (WINDOW_SMALL, 128, None, (DENSE_BOTH, DENSE_BOTH), 1, "W=64 dense"),
    (WINDOW_SMALL, 128, 8, (SHARED_SPARSE, SHARED_SPARSE), 3, "W=64 k_tok=8"),
    (WINDOW_WIDE, 512, None, (DENSE_BOTH, DENSE_BOTH), 4, "W=512 dense"),
    (WINDOW_WIDE, 1024, 16, (SHARED_SPARSE, SHARED_SPARSE), 5, "W=512 k_tok=16"),
    (WINDOW_SMALL, 128, 8, (WRITE_SPARSE, READ_SPARSE), 6, "W=64 ALT k_tok=8"),
    (WINDOW_WIDE, 1024, 16, (WRITE_SPARSE, READ_SPARSE), 7, "W=512 ALT k_tok=16"),
]

#: the reverse band, `test_intra_backward.py`'s: `KAPPA * 2 ** -9`.
KAPPA = 8.0


@pytest.mark.parametrize("window,tokens,k_tok,modes,seed,label", BACKWARD_CELLS,
                         ids=[c[-1] for c in BACKWARD_CELLS])
def test_the_composition_cotangents_match_the_reference(window, tokens, k_tok, modes, seed,
                                                        label):
    from rola.ops import carry as carry_ops
    from tests.k31.backward_reference import intra_backward_reference
    from tests.k31.fixtures import simplex

    widths = (LEVEL_WIDTH, LEVEL_WIDTH)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shape = (1, tokens, 1)
    read = tuple(simplex(shape + (w,), k_tok, gen, "cuda") for w in widths)
    write = tuple(simplex(shape + (w,), k_tok, gen, "cuda") for w in widths)
    gain = torch.rand(*shape, device="cuda", dtype=torch.float64, generator=gen) + 0.5
    value = torch.randn(*shape, 64, device="cuda", dtype=torch.float64, generator=gen)
    d_num = torch.randn(1, tokens, 1, 64, device="cuda", dtype=torch.float64, generator=gen)
    d_den = torch.randn(1, tokens, 1, device="cuda", dtype=torch.float64, generator=gen)

    read_bf = tuple(x.to(torch.bfloat16) for x in read)
    write_bf = tuple(x.to(torch.bfloat16) for x in write)
    gain_bf, value_bf = gain.to(torch.bfloat16), value.to(torch.bfloat16)
    dnum_bf = d_num.to(torch.bfloat16)

    got = twopass.intra_backward(
        carry_ops.pack_side(read_bf), carry_ops.pack_side(write_bf),
        gain_bf.permute(0, 2, 1).contiguous().reshape(1, tokens),
        value_bf.permute(0, 2, 1, 3).reshape(1, tokens, 64).contiguous(),
        dnum_bf.reshape(1, tokens, 64).contiguous(),
        d_den.to(torch.float32).reshape(1, tokens).contiguous(),
        len(widths), LEVEL_WIDTH, window, modes)
    want = intra_backward_reference(
        tuple(x.double() for x in read_bf), tuple(x.double() for x in write_bf),
        gain_bf.double(), value_bf.double(), widths, window, dnum_bf.double(),
        d_den.to(torch.float32).double())

    def flat(levels):
        return torch.cat([x.permute(0, 2, 1, 3).reshape(1, tokens, x.shape[-1])
                          for x in levels], -1)

    def restriction(sides, bit):
        live = []
        for side, width, mode in zip(sides, widths, modes):
            plane = side.permute(0, 2, 1, 3).reshape(1, tokens, width)
            live.append((plane != 0) if (mode & bit) else torch.ones_like(plane,
                                                                         dtype=torch.bool))
        return torch.cat(live, -1)

    rows = {
        "d_pr": (got[0], flat(want[0]), restriction(read_bf, READ_SPARSE)),
        "d_pw": (got[1], flat(want[1]), restriction(write_bf, WRITE_SPARSE)),
        "d_gw": (got[2], want[2].permute(0, 2, 1).reshape(1, tokens), None),
        "d_v": (got[3], want[3].permute(0, 2, 1, 3).reshape(1, tokens, 64), None),
    }
    for name, (value_got, ref, live) in rows.items():
        scale = max(1e-30, float(ref.abs().max()))
        if live is None:
            live = torch.ones_like(value_got, dtype=torch.bool)
        else:
            dead = int((value_got[~live] != 0).sum())
            assert dead == 0, f"{name}: {dead} nonzero cells off the declared support"
        err = float((value_got.double() - ref)[live].abs().max()) / scale
        assert err < KAPPA * 2 ** -9, f"{label}/{name}: {err:.3e}"
