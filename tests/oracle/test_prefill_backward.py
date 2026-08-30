# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE COMBINED OPERATOR'S GRADIENT, against the fp64 reference for the whole recurrence.

The mirror of `test_prefill_vs_oracle.py`: each half's own reverse gate would pass on its
own statement of the window split, so here ONE autograd node runs all three reverse
kernels and the whole cotangent set is compared against the reference the decomposition
gate already pinned to `torch.autograd` -- a grid disagreement, a double-counted diagonal
or a dropped tile pair in the GRADIENT has nowhere to hide either.

RED BY DESIGN (card `development/queue/C_CLEAN_SLATE.md`, "TEST-DRIVEN"): the carry's
reverse pass is C4's, after C3 builds the forward. The cells are written for the FINAL
surface -- no window, no ``(k, m)``, no carve, the descriptor and the launch shape
instead -- so they are the spec the reverse pass is built against and not a record of
what the deleted one took.

Symbols, each at first use: ``D`` = routing depth, ``B_l`` = level ``l``'s padded digit
count, ``N = prod_l B_l`` = leaf capacity, ``DV`` = the padded value width, ``L`` =
tokens, ``BH`` = batch times heads, ``kappa`` = the published ratio of the allowed error
to the rounded-reference floor, one per output.
"""
from __future__ import annotations

import pytest
import torch

from benchmarks.cells import realize
from rola.ops import carry as carry_ops
from rola.ops import intra as intra_ops
from rola.ops._ext import is_available
from rola.ops.paging import to_split_planes
from rola.ops.prefill import prefill_autograd
from tests.oracle.backward_reference import carry_backward_reference, intra_backward_reference
from tests.oracle.fixtures import canonical_from_plane
from tests.oracle.test_prefill_vs_oracle import CELLS, modes_of

pytestmark = pytest.mark.skipif(not is_available() or not torch.cuda.is_available(),
                               reason="the reverse pass is a CUDA kernel")

#: ONE KAPPA PER OUTPUT, published: a ratio to the rounded-reference floor.
KAPPA = 6.0

#: The reverse families are built at depth two only, so the differentiable entry is
#: narrower than the forward one by exactly that gap and says so rather than
#: half-serving it. A cell here is a forward cell that clears the reverse gap.
REVERSE_CELLS = tuple(c for c in CELLS if c.D == 2 and c.tokens <= 1024)
IDS = [c.name for c in REVERSE_CELLS]


def _leaves(drawn):
    """The routing operands as fp64 leaves that carry gradients.

    They are the bf16 IMAGES cast back up, never the raw draws: the reference must read
    what the kernel actually received, or the comparison measures the cast.
    """
    return [x.double().clone().requires_grad_(True) for x in drawn.read + drawn.write]


@pytest.mark.parametrize("spec", REVERSE_CELLS, ids=IDS)
@pytest.mark.parametrize("chained", [False, True], ids=["fresh", "chained"])
def test_every_cotangent_of_the_combined_operator(spec, chained):
    #: THE fp64 REFERENCE MATERIALIZES `[L, N]` PER WINDOW; a card that cannot hold it
    #: skips with the reason and the class rather than narrowing the claim.
    if torch.cuda.get_device_properties(0).total_memory < 10 * 2 ** 30:
        pytest.skip(f"N={spec.N}, L={spec.tokens}: the fp64 reverse reference exceeds "
                    f"this card; rental-tier")
    drawn = realize(spec)
    desc = spec.descriptor()
    gen = torch.Generator(device="cuda").manual_seed(spec.seed)

    #: THE PLANES ARE HANDED TO THE KERNEL IN THEIR STORED FORM (split bf16 planes); the
    #: references read them back through `canonical_from_plane`'s gather.
    state_in = to_split_planes(
        0.02 * torch.randn(1, desc.N // carry_ops.PAGE_LEAVES, carry_ops.PAGE_LEAVES,
                           desc.cols, device="cuda", dtype=torch.float32,
                           generator=gen)) if chained else None

    leaves = _leaves(drawn)
    gain = drawn.gain.double().clone().requires_grad_(True)
    value = drawn.v.double().clone().requires_grad_(True)
    num, den, state_out = prefill_autograd(
        tuple(leaves[:spec.D]), tuple(leaves[spec.D:]), gain, value, spec.widths,
        modes=modes_of(spec), sread=intra_ops.pack_support(drawn.read),
        swrite=intra_ops.pack_support(drawn.write), state_in=state_in)

    d_num = torch.randn(num.shape, device="cuda", dtype=torch.float32, generator=gen)
    d_den = torch.randn(den.shape, device="cuda", dtype=torch.float32, generator=gen)
    d_state = to_split_planes(torch.randn(state_out.shape, device="cuda",
                                          dtype=torch.float32, generator=gen))
    got = torch.autograd.grad((num, den, state_out), leaves + [gain, value],
                              (d_num, d_den, d_state))
    del num, den, state_out
    torch.cuda.empty_cache()

    read, write, gain64, value64 = drawn.doubles()
    dn = d_num.double().reshape(1, 1, spec.tokens, spec.dv).permute(0, 2, 1, 3)
    dd = d_den.double().reshape(1, 1, spec.tokens).permute(0, 2, 1)
    sin = (canonical_from_plane(state_in).double().reshape(1, 1, spec.N, desc.cols)
           if chained else None)
    dso = canonical_from_plane(d_state).double().reshape(1, 1, spec.N, desc.cols)
    c_read, c_write, c_gain, c_v, _ = carry_backward_reference(
        read, write, gain64, value64, spec.widths, carry_ops.WINDOW, dn, dd, sin, dso)
    torch.cuda.empty_cache()
    i_read, i_write, i_gain, i_v = intra_backward_reference(
        read, write, gain64, value64, spec.widths, carry_ops.WINDOW, dn, dd)
    torch.cuda.empty_cache()

    want = ([c_read[l] + i_read[l] for l in range(spec.D)]
            + [c_write[l] + i_write[l] for l in range(spec.D)]
            + [c_gain + i_gain, c_v + i_v])
    names = ([f"d_read{l}" for l in range(spec.D)] + [f"d_write{l}" for l in range(spec.D)]
             + ["d_gain", "d_v"])
    #: THE DIGIT GRADIENTS ARE EXACT ON EACH LEVEL'S OWN SUPPORT and partial off it: the
    #: carry's schedule visits an owner only where every level's support meets it, and
    #: the exact-entmax Jacobian is zero exactly where the partial sums live.
    masks = [x != 0 for x in read] + [x != 0 for x in write] + [None, None]
    for label, mine, theirs, mask in zip(names, got, want, masks):
        scale = max(1e-30, float(theirs.abs().max()))
        sel = (slice(None),) if mask is None else (mask,)
        err = float((mine.double() - theirs)[sel].abs().max()) / scale
        assert err < KAPPA * 2 ** -9, f"{spec.name}/{label}: {err:.3e}"


def test_the_reverse_surface_takes_the_same_contract_as_the_forward():
    """ONE DESCRIPTOR, ONE GEOMETRY BLOCK, ONE SET OF LIVENESS WORDS.

    A reverse pass that re-derived any of them would be a second derivation of the
    schedule, which is the drift shape this family has already paid for once. The
    signatures are asserted to agree rather than the derivations compared, because at C1
    there is no body on either side to compare."""
    import inspect

    forward = set(inspect.signature(carry_ops.carry_forward).parameters)
    reverse = set(inspect.signature(carry_ops.carry_backward).parameters)
    shared = {"routes", "v", "descriptor", "geometry", "liveness", "activity", "launch",
              "state_in", "page_table"}
    assert shared <= forward and shared <= reverse
    assert reverse - forward == {"d_num", "d_den", "d_state_out"}
    assert forward - reverse == {"state_out"}
