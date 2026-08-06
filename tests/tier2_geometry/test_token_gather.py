"""TIER 2 — the token-gather walk (P14, incremental form).

The gathered walk ballots each owner's live tokens (fixed-level token rows AND
the token-granular slice union) into dense groups of ``BT`` and runs the SAME
visit body on each group, ascending. Its correctness criteria are
``SUPPORT_GRANULARITY_DESIGN.md`` §6.2, and the validation target is the
canonical fp64 oracle ``naive_rola`` — never a fresh reference (spec §2.1):

1. **Degeneracy, bit-exact.** On a full-support fixture membership is all-ones,
   groups coincide with aligned tiles, and the gathered walk must be
   ``torch.equal`` to the aligned walk on ``y`` AND ``state`` — single-owner
   rectangles, because split-K atomic ORDER across concurrent owner CTAs is
   nondeterministic by construction.
2. **Sparse, against the oracle, with owners straddling many source tiles.**
   This is the causality gate: the intra Gram's inclusive mask is
   ORDER-RELATIVE, so an unstable (non-ascending) or permuted gather produces a
   *plausible* wrong answer that only the oracle comparison catches; a
   dropped-token gather produces a missing deposit the state comparison
   catches. Sparse results are NOT claimed bit-identical to the aligned walk —
   grouping tokens from different tiles re-associates the intra term.
3. **Ragged everything:** a global token tail (``T % BT != 0``), the final
   ragged flush group (``fill < BT``), and ultra-sparse owners whose membership
   is empty on most candidate tiles.
4. **Structural scope:** with entmax on both sides the gather is THE path (no
   override), and it composes with ``P > 1`` (pass 1 stays aligned, pass 3
   gathers).

``token_gather=True/False`` overrides are GATE MACHINERY (pinning one walk
against the other on identical routing); production derives the scope from the
activation registry.
"""

from __future__ import annotations

import pytest
import torch

from rola.routing.types import EntmaxActivation, IndependentRouting
from tests.tier1_semantics.test_consumer_vs_oracle import (
    RTOL,
    _decay_dials,
    _run_case,
)
from tests.tier2_geometry.test_p_composition import _Fixture, _relative

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="V3 consumer is a CUDA kernel"),
]

#: `tied=True`: the registry rule permits two exact-zero sides only over one
#: shared solve (untied sparse/sparse is UnionRouting's job). The fixture
#: tensors are drawn, not solved, so tying changes nothing they express.
_ENTMAX_BOTH = IndependentRouting(
    tied=True, read_activation=EntmaxActivation(1.5), write_activation=EntmaxActivation(1.5))


# ---------------------------------------------------------------------------
# 1. Degeneracy: full support => the gather is the identity, bit for bit
# ---------------------------------------------------------------------------

#: Single-owner rectangles only (one CTA per bh => deterministic emission order,
#: the `test_compact_equivalence` precedent): (16,) BC=16 has no fixed level and
#: no slice partition; (8,8) BC=64 spans everything.
@pytest.mark.parametrize("widths,BC", [((16,), 16), ((8, 8), 64)])
@pytest.mark.parametrize("BT", [16, 32])
@pytest.mark.parametrize("norm", ["raw", "global"])
@pytest.mark.parametrize("decay_on", [False, True])
def test_full_support_gather_is_bit_exact(widths, BC, BT, norm, decay_on):
    dials = None
    if decay_on:
        gen = torch.Generator(device="cuda").manual_seed(7)
        dials = _decay_dials(widths, 2, gen, "cuda")
    kwargs = dict(nonzero_m0=True, seed=11, decay_dials=dials, return_outputs=True)
    y_al, s_al = _run_case(widths, BC, BT, 64, 1, 64, 2, norm, 1.0, 1.0,
                           token_gather=False, **kwargs)
    y_ga, s_ga = _run_case(widths, BC, BT, 64, 1, 64, 2, norm, 1.0, 1.0,
                           token_gather=True, **kwargs)
    assert torch.equal(s_ga, s_al), "full-support gathered state must be BIT-identical"
    assert torch.equal(y_ga, y_al), "full-support gathered output must be BIT-identical"


# ---------------------------------------------------------------------------
# 2. Sparse, against the canonical oracle (stability/causality teeth)
# ---------------------------------------------------------------------------

#: The structural grid: every owner rectangle class, including the fixed-level
#: AND chain the membership scan walks (j=1 one fixed row, j=2 two) and the
#: multi-slice partitions the token-granular union discriminates.
_GRID = [
    ((16,), 16),      # D=1, one owner, membership = the union word alone
    ((8, 8), 16),     # j=0, c=2, 4 slices
    ((8, 8), 64),     # BC == N, one owner
    ((4, 4, 4), 16),  # j=1, ONE fixed token row in the AND
    ((2, 3, 4, 4), 16),  # j=2, TWO fixed rows, non-power-of-two widths
]


@pytest.mark.parametrize("widths,BC", _GRID)
@pytest.mark.parametrize("BT", [16, 32])
@pytest.mark.parametrize("norm", ["raw", "global"])
@pytest.mark.parametrize("p_read,p_write", [(0.4, 0.3), (0.12, 0.1)])
def test_sparse_gather_matches_the_canonical_oracle(widths, BC, BT, norm, p_read, p_write):
    # T=96 at density ~0.1: an owner's live tokens straddle ~all source tiles and
    # groups genuinely mix tiles (the mean gap at benchmark density is 12-20).
    _run_case(widths, BC, BT, 64, 1, 96, 2, norm, p_read, p_write,
              nonzero_m0=True, seed=23, token_gather=True)


# ---------------------------------------------------------------------------
# 3. Ragged: global tail, final flush, near-empty membership
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("widths,BC", [((4, 4, 4), 16), ((8, 8), 64)])
@pytest.mark.parametrize("BT", [16, 32])
@pytest.mark.parametrize("T", [17, 33, 65])
@pytest.mark.parametrize("p_read,p_write", [(0.4, 0.3), (0.06, 0.05)])
def test_ragged_tail_and_final_flush(widths, BC, BT, T, p_read, p_write):
    _run_case(widths, BC, BT, 64, 1, T, 2, "global", p_read, p_write,
              nonzero_m0=False, seed=31, token_gather=True)


# ---------------------------------------------------------------------------
# 4. The decay arm: gathered clock prefix + the padded slots' zero prefix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("widths,BC", [((8, 8), 16), ((4, 4, 4), 16)])
@pytest.mark.parametrize("BT", [16, 32])
@pytest.mark.parametrize("hot", [False, True])
def test_decay_gathered_clock_matches_the_oracle(widths, BC, BT, hot):
    gen = torch.Generator(device="cuda").manual_seed(41)
    dials = _decay_dials(widths, 2, gen, "cuda", hot=hot)
    _run_case(widths, BC, BT, 64, 1, 65, 2, "global", 0.4, 0.3,
              nonzero_m0=True, seed=43, decay_dials=dials, token_gather=True)


# ---------------------------------------------------------------------------
# 5. The bf16 arm: gathered staging on the narrow-stage (ring-2 class) kernel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("BT", [16, 32])
@pytest.mark.parametrize("p_read,p_write", [(0.4, 0.3), (0.12, 0.1)])
def test_bf16_arm_gathers(BT, p_read, p_write):
    _run_case((8, 8), 64, BT, 64, 1, 96, 2, "global", p_read, p_write,
              nonzero_m0=False, seed=53, mma_precision="bf16", token_gather=True)


# ---------------------------------------------------------------------------
# 6. Structural scope + composition with P
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("norm", ["raw", "global"])
def test_entmax_both_sides_gathers_by_structure(norm):
    """No override: an entmax-both topology must take the gathered walk and agree
    with the oracle. (That the flag derives True is what routes it; a False
    derivation would silently retest the aligned walk, so the derivation itself
    is asserted too.)"""
    from rola.ops.consumer import _consumer_forward  # noqa: F401  (import parity)
    fx = _Fixture((4, 4, 4), 16, 64, 1, 96, 2, norm, 0.4, 0.3, 61, routing=_ENTMAX_BOTH)
    assert all(
        fx.topology.activation_properties(level, duty).exact_zeros
        for level in range(fx.topology.D) for duty in ("read", "write"))
    y_ref, state_ref = fx.oracle()
    y, state, _ = fx.run()
    assert _relative(y, y_ref) < RTOL
    assert _relative(state.view_as(state_ref), state_ref) < RTOL


@pytest.mark.parametrize("P", [2, 4])
@pytest.mark.parametrize("decay_on", [False, True])
def test_gather_composes_with_P(P, decay_on):
    """P > 1 under structural gather scope: pass 1 (summary) stays aligned this
    increment, pass 3 gathers within its segments; every segment's ragged flush
    is exercised. Held to the P=1 tolerance, per the §6.4 rule."""
    fx = _Fixture((8, 8), 16, 64, 1, 96, 2, "global", 0.4, 0.3, 71,
                  decay_on=decay_on, nonzero_m0=True, routing=_ENTMAX_BOTH)
    y_ref, state_ref = fx.oracle()
    y, state, _ = fx.run(P=P)
    assert _relative(y, y_ref) < RTOL
    assert _relative(state.view_as(state_ref), state_ref) < RTOL
