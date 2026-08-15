# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""The DEVICE-SIDE growth trigger: does this step reach an atom the table does not map?

A paged decode step's exact admission preamble -- the write-atom derivation,
`plan_exact`, the cross-stream event -- costs roughly ten times the kernel it precedes
(`workflows/p67-paging/s4_build.md` §6b). It is also usually unnecessary: a sequence
whose routing has settled writes atoms that are already resident, and there is nothing
to admit. The trigger is what lets the step find that out for one kernel and one
four-byte read, and the whole of its correctness is one claim:

  **THE TRIGGER FIRES IFF THE PREAMBLE WOULD ADMIT SOMETHING.**

Both halves matter and this file sweeps both. A trigger that MISSED would launch a step
whose deposit lands in an absent atom -- silently dropped, since a `slot < 0` row is
never formed. A trigger that fired SPURIOUSLY would only cost the preamble it exists to
skip, but it would also mean the two derivations disagree, and a disagreement that is
harmless in one direction is not evidence about the other.

The claim is cheap to make exactly because the two sides SHARE a definition rather than
mirror one: `rola_decode_growth_probe` calls `decode_expand.cu`'s own `expand_side` over
the same `amp_write` pointers the step itself expands, and `_write_atom_bitmap` is the
torch derivation the exact preamble plans from. So this file checks an EQUALITY the call
graph already forces, which is the only kind of agreement worth gating: the failure it
can catch is a condensation or a residency test that drifted, not a second copy of the
support rule.

Vocabulary, defined here and used unqualified below: `N` = leaf count; `d_v = 64`;
`cols = d_v + 1`; `BH = batch * heads`; **atom** = 16 consecutive leaves in canonical
leaf order = the page granule; **slot** = the physical page an atom is mapped to, `-1`
for an atom no plan committed.

Run: ``pytest tests/tier2_geometry/test_decode_growth_trigger.py``
"""
from __future__ import annotations

import pytest
import torch

from rola.ops.decode import DecodeWorkspace, _growth_probe, _write_atom_bitmap
from rola.ops.paging import PageArena
from tests.tier2_geometry.test_decode_paging import (
    _ARMS,
    _config,
    _f32,
    _fixture,
    _leaf_support,
)

_SUPPORT = {"sparse": 0.25, "half": 0.5, "all": 1.0}

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="decode is a CUDA kernel"),
]


def _probe(fixture, config, arena):
    """The trigger's answer, and the workspace it ran on."""
    _v, _read, write, _g = _f32(fixture)
    BH = fixture["B"] * fixture["H"]
    workspace = DecodeWorkspace.build(config, BH, torch.device("cuda"))
    fired = _growth_probe(write, config, arena.page_table, workspace)
    return fired, workspace


def _would_admit(fixture, config, arena) -> int:
    """What the EXACT preamble would admit, run here and nowhere near the shipped path."""
    _v, _read, write, _g = _f32(fixture)
    bitmap = _write_atom_bitmap(write, config)
    return int((bitmap & (arena.page_table < 0)).sum())


def _arena(widths, BC, BH, **kwargs):
    return PageArena(widths=widths, BC=BC, cols=65, BH=BH, device=torch.device("cuda"),
                     **kwargs)


# ---------------------------------------------------------------------------
# 1. the two directions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: "x".join(map(str, a[0])))
@pytest.mark.parametrize("support", ("sparse", "half", "all"))
def test_the_trigger_fires_exactly_when_the_preamble_would_admit(arm, support):
    """Empty table -> must fire; then the step's own atoms admitted -> must not.

    The two states are the two ends of a real sequence: the first step after a prefill
    that never reached these atoms, and every settled step after it. The assertion is
    the same in both, stated against what the preamble WOULD do rather than against a
    second expectation of what the routing looks like.
    """
    widths, BC = arm
    fixture = _fixture(widths, seed=hash((widths, support)) % 10_000, support=_SUPPORT[support])
    config = _config(widths, fixture)
    BH = fixture["B"] * fixture["H"]
    _v, _read, write, _g = _f32(fixture)

    #: EMPTY: nothing is mapped, so every atom this step writes is new.
    arena = _arena(widths, BC, BH)
    admits_before = _would_admit(fixture, config, arena)
    assert admits_before > 0, "this cell needs a step that writes something"
    fired, _ = _probe(fixture, config, arena)
    assert fired, (
        f"the trigger missed a step the preamble would admit {admits_before} atoms for; "
        "the step would have deposited into an absent atom and lost it")

    #: SETTLED: this exact write set is now resident, so there is nothing to admit.
    arena.plan_exact(_write_atom_bitmap(write, config), fresh=True)
    arena.wait()
    torch.cuda.synchronize()
    assert _would_admit(fixture, config, arena) == 0, "the fixture's own precondition"
    fired, _ = _probe(fixture, config, arena)
    assert not fired, (
        "the trigger fired on a step with nothing to admit; the preamble it exists to "
        "skip would be paid on every settled step of the sequence")


@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: "x".join(map(str, a[0])))
def test_one_absent_atom_is_enough_to_fire(arm):
    """The trigger is an ANY, and this is its tightest cell.

    The step's whole write set is admitted and then EXACTLY ONE of its atoms is evicted
    from the table. Nothing else moves. A trigger that reduced over the wrong axis, or
    that condensed 32 leaves to an atom instead of 16, or that tested `>= 0` on the wrong
    row of the table, passes the coarse cells above and fails here.
    """
    widths, BC = arm
    fixture = _fixture(widths, seed=907 + len(widths), support=0.5)
    config = _config(widths, fixture)
    BH = fixture["B"] * fixture["H"]
    _v, _read, write, _g = _f32(fixture)
    bitmap = _write_atom_bitmap(write, config)

    arena = _arena(widths, BC, BH)
    arena.plan_exact(bitmap, fresh=True)
    arena.wait()
    torch.cuda.synchronize()
    assert not _probe(fixture, config, arena)[0], "the fixture's own precondition"

    written = bitmap.nonzero()
    for pick in (0, written.shape[0] // 2, written.shape[0] - 1):
        bh, atom = (int(x) for x in written[pick])
        slot = int(arena.page_table[bh, atom])
        arena.page_table[bh, atom] = -1
        fired, _ = _probe(fixture, config, arena)
        arena.page_table[bh, atom] = slot
        assert fired, (
            f"evicting the single atom ({bh}, {atom}) this step writes did not fire the "
            "trigger")
    assert not _probe(fixture, config, arena)[0], "restoring the slot did not settle it"


@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: "x".join(map(str, a[0])))
def test_an_absent_atom_the_step_does_not_write_is_not_growth(arm):
    """Residency is not the question; the step's OWN write set against residency is.

    A sparsely-resident arena is the normal case -- that is what paging is for -- so a
    trigger keyed on "is anything absent" rather than "is anything I WRITE absent" would
    fire on every step forever and quietly restore the preamble this stage removed.
    """
    widths, BC = arm
    fixture = _fixture(widths, seed=311 + len(widths), support=0.25)
    config = _config(widths, fixture)
    BH = fixture["B"] * fixture["H"]
    _v, _read, write, _g = _f32(fixture)
    bitmap = _write_atom_bitmap(write, config)

    arena = _arena(widths, BC, BH)
    arena.plan_exact(bitmap, fresh=True)
    arena.wait()
    torch.cuda.synchronize()
    absent = int((arena.page_table < 0).sum())
    assert absent > 0, "this cell needs a routing that leaves atoms absent"
    assert not _probe(fixture, config, arena)[0], (
        f"{absent} atoms are absent and the trigger fired, but the step writes none of "
        "them")


# ---------------------------------------------------------------------------
# 2. the probe's own support is the step's support
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: "x".join(map(str, a[0])))
@pytest.mark.parametrize("support", ("sparse", "half", "all"))
def test_the_probe_expands_the_same_write_support_the_preamble_derives(arm, support):
    """The bitmap the probe leaves behind, against the definition -- not against itself.

    The probe publishes its expansion into the step's own `bits_w` workspace, so the
    leaf-level object it ANDed against the table is READABLE, and it is checked against
    `_leaf_support` -- `schedule.py`'s `write_mask` taken to leaf granularity in torch.
    This is what makes the two cells above a claim about the TRIGGER rather than about
    two derivations that happen to agree on these fixtures.
    """
    widths, BC = arm
    fixture = _fixture(widths, seed=57 + len(widths), support=_SUPPORT[support])
    config = _config(widths, fixture)
    BH = fixture["B"] * fixture["H"]
    _v, _read, write, _g = _f32(fixture)

    arena = _arena(widths, BC, BH)
    _fired, workspace = _probe(fixture, config, arena)
    torch.cuda.synchronize()

    n_words = config.n_words
    published = workspace.bits[BH * n_words:2 * BH * n_words].view(BH, n_words)
    bit = torch.arange(config.N, device="cuda")
    expanded = (published[:, bit // 32] >> (bit % 32).to(torch.int32)) & 1
    assert torch.equal(expanded.bool(), _leaf_support(write, widths)), (
        "the probe's expansion is not the write support's own definition")
    assert torch.equal(
        expanded.bool().view(BH, config.N // 16, 16).any(-1),
        _write_atom_bitmap(write, config)), (
        "the probe's expansion and the preamble's bitmap condense differently")


@pytest.mark.parametrize("arm", _ARMS, ids=lambda a: "x".join(map(str, a[0])))
@pytest.mark.parametrize("support", ("sparse", "half", "all"))
def test_the_published_condensation_is_what_a_growth_step_admits(arm, support):
    """The probe's `atom_bits` IS the write-atom set, against an independent derivation.

    A growth step plans from this buffer and derives nothing on the host, so the buffer
    is not an internal of the probe -- it is the admission's input, and a wrong bit in it
    is either a page committed for nothing (a bit set too many) or a deposit silently
    dropped into an absent atom (a bit missing). `_write_atom_bitmap` is the ENUMERATED
    REFERENCE the comparison needs: the same set derived from the Cartesian-product
    definition in torch, with no share of the kernel's expansion or its 16-deep
    condensation, and with this gate as its only caller.

    The arena is EMPTY here, which is the growth step's own condition -- every atom the
    step writes is new -- so the equality is asserted exactly where the shipped path
    consumes it.
    """
    widths, BC = arm
    fixture = _fixture(widths, seed=419 + len(widths), support=_SUPPORT[support])
    config = _config(widths, fixture)
    BH = fixture["B"] * fixture["H"]
    _v, _read, write, _g = _f32(fixture)

    arena = _arena(widths, BC, BH)
    fired, workspace = _probe(fixture, config, arena)
    torch.cuda.synchronize()
    reference = _write_atom_bitmap(write, config)
    assert fired, "an empty table and a step that writes: this cell IS a growth step"
    assert reference.any(), "a vacuous cell would compare two empty bitmaps"
    assert torch.equal(workspace.atom_bits, reference), (
        "the set the probe published is not the set the definition derives, so a growth "
        "step would admit the wrong atoms")

    #: and the admission agrees with the verdict: planning from the published set leaves
    #: nothing for the next probe to find.
    arena.plan_exact(workspace.atom_bits, fresh=False)
    arena.wait()
    torch.cuda.synchronize()
    assert not _probe(fixture, config, arena)[0], (
        "admitting the probe's own published set did not settle the trigger")
