"""ONE-TIME TRANSCRIPTION CHECK: our textbook recurrence against FLA's naive GLA.

The oracle's recurrence claims to BE the gated linear-attention loop. That claim
is the load-bearing half of the reduction theorem -- it is why the correctness
reference gets to be "the textbook" rather than "another implementation of ours"
-- and everything else in the repository is anchored to it. A transcription
error in it (an index off by one, a gate applied after the deposit instead of
before, a readout that is exclusive where the spec is inclusive) would be
invisible to every gate, because every gate compares against this file.

So it is checked once, at adoption, against a naive GLA loop written by someone
else: `fla.ops.gla.naive.naive_recurrent_gla` (flash-linear-attention 0.4.2).
Under plain-GLA settings the two are the same function under a renaming:

    FLA          ours
    K            N          key dim  <->  leaf address space
    q_i * scale  R[t]       read allocation x read gain
    k_i          W[t]       write allocation x write gain
    exp(gk_i)    keep[t]    the diagonal gate
    h            state      one row per key / per leaf

FLA scales `q` by `K ** -0.5` internally, so `q = R * sqrt(N)` is what makes the
two readouts the same expression rather than the same shape.

**FLA IS NOT A STANDING DEPENDENCY, AND THIS TEST IS SKIPPED BY DEFAULT.** The
decomposition's rule is naive torch references only, from third parties, and one
adopted at a version -- adding an unpinned import of a fast-moving kernel library
to the routine suite would be taking on exactly the coupling the pin machinery
exists to prevent. Its job is to catch a transcription error ONCE. Set
`ROLA_FLA_CROSSCHECK=1` to run it.

MEASURED AT ADOPTION (2026-08-05, fla 0.4.2), over decay on/off x widths
`(8,8)` and `(4,4,4)`: worst `2.98e-08` on the output and `5.96e-08` on the
final state, against outputs of magnitude ~0.2 -- i.e. agreement at float32's
own resolution. The floor is FLA's, not ours: `naive_recurrent_gla` casts its
operands to float32 and accumulates there. Our own oracle-vs-oracle contracts
are bit-identical in fp64; nothing here loosens them.
"""
from __future__ import annotations

import os

import pytest
import torch

from rola.ops.naive import naive_rola
from rola.routing.types import (
    IndependentRouting,
    LeafMassDecay,
    LevelConfig,
    SoftmaxActivation,
    Topology,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("ROLA_FLA_CROSSCHECK"),
    reason="one-time transcription check against flash-linear-attention; FLA is "
           "deliberately NOT a standing dependency (see the module docstring). "
           "Set ROLA_FLA_CROSSCHECK=1 to run it.")

_ROUTING = IndependentRouting(
    tied=False, read_activation=SoftmaxActivation(), write_activation=SoftmaxActivation())

#: The bound, and it is float32's rather than a fitted number: FLA accumulates in
#: float32, whose unit roundoff is 6e-8, over a sum of `N <= 64` leaves. The
#: measured worst is 2.98e-08 against outputs of magnitude ~0.2, so `1e-5`
#: relative leaves two decades of headroom while still being far tighter than any
#: transcription error could hide in -- an off-by-one index or a gate applied on
#: the wrong side of the deposit moves the result by O(1), not by 1e-8.
FLOAT32_FLOOR = 1e-5


@pytest.mark.parametrize("widths", [(8, 8), (4, 4, 4)])
@pytest.mark.parametrize("decay_on", [False, True])
def test_the_oracle_recurrence_is_flas_naive_gla_under_a_renaming(widths, decay_on):
    fla_naive = pytest.importorskip("fla.ops.gla.naive")

    B, T, H, d_v = 2, 8, 2, 16
    N = 1
    for width in widths:
        N *= width
    g = torch.Generator().manual_seed(31)
    topology = Topology(levels=tuple(LevelConfig(w, _ROUTING) for w in widths))

    def simplex(width):
        x = torch.rand(B, T, H, width, generator=g, dtype=torch.float64) + 1e-3
        return x / x.sum(dim=-1, keepdim=True)

    read = tuple(simplex(w) for w in widths)
    write = tuple(simplex(w) for w in widths)
    v = torch.randn(B, T, H, d_v, generator=g, dtype=torch.float64)
    g_read = torch.rand(B, T, H, generator=g, dtype=torch.float64) + 0.5
    g_write = torch.rand(B, T, H, generator=g, dtype=torch.float64) + 0.5
    decay = None
    if decay_on:
        decay = LeafMassDecay(dials=tuple(
            torch.rand(H, w, generator=g, dtype=torch.float64) * 0.3 + 0.1 for w in widths))

    #: `raw` and never `global`: the ones column is OURS, and the point of this
    #: test is the part that is not.
    y, state = naive_rola(v, read, write, g_read, g_write, topology, decay, "raw",
                          output_final_state=True)

    #: The renaming, built from the same inputs by the same rules the oracle uses
    #: -- deliberately re-derived here rather than reached for inside the oracle,
    #: since a shared helper would make the two sides agree by construction.
    def leaves(levels):
        strides, out = [], None
        for index in range(len(widths)):
            stride = 1
            for width in widths[index + 1:]:
                stride *= width
            strides.append(stride)
        ids = torch.arange(N)
        for level, stride in zip(levels, strides):
            digit = (ids // stride) % level.shape[-1]
            picked = level.index_select(-1, digit)
            out = picked if out is None else out * picked
        return out

    R = g_read[..., None] * leaves(read)
    W = g_write[..., None] * leaves(write)
    if decay is None:
        keep = torch.ones(B, T, H, N, dtype=torch.float64)
    else:
        rate = leaves(tuple(decay.dials))
        keep = torch.pow(1.0 - rate, leaves(write))

    o, h = fla_naive.naive_recurrent_gla(
        q=(R * (N ** 0.5)).to(torch.float32),
        k=W.to(torch.float32),
        v=v.to(torch.float32),
        gk=torch.log(keep).to(torch.float32),
        output_final_state=True)

    #: NON-VACUITY: a comparison against an all-zero reference passes for free.
    assert float(o.abs().max()) > 1e-3, "FLA's output is ~0; the fixture is degenerate"

    scale = max(float(y.abs().max()), 1.0)
    delta_y = float((y.to(torch.float32) - o).abs().max())
    delta_h = float((state.to(torch.float32) - h).abs().max())
    assert delta_y <= FLOAT32_FLOOR * scale, (
        f"the oracle's recurrence and FLA's naive GLA disagree by {delta_y:.3e} on "
        f"the output, past float32's floor -- that is a transcription error, not a "
        f"rounding difference")
    assert delta_h <= FLOAT32_FLOOR * max(float(state.abs().max()), 1.0), (
        f"the final states disagree by {delta_h:.3e}")
