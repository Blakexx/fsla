# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CARRY LAUNCH SURFACE'S CONTRACT, tested by calling the surface directly.

C1c of card `development/queue/C_CLEAN_SLATE.md`: the interface before the body. Two
things are asserted here and nothing else.

1. THE KERNEL-BOUNDARY REFUSALS (KERNEL_STANDARDS R13 addendum, "TWO CONTRACTS, ONE
   TRANSLATOR"): an unpadded shape, a non-bf16 operand and a descriptor mismatch are
   each refused BY THE SURFACE, by name, before any launch. The API above translates by
   padding; this boundary never does.
2. THE HONEST NO-IMPLEMENTATION FAILURE: a call whose every shape is lawful reaches the
   extension and is refused there for the ONE reason that is true on this line -- the
   carry kernel is not built. Not a skip, not a missing attribute, not a shape excuse.

The gate is deliberately shapeless: it draws the smallest lawful cell it can, because
what is under test is the contract and not the arithmetic (that is the oracle tier's,
and it is red by design until C3).
"""
from __future__ import annotations

import re

import pytest
import torch

from benchmarks.cells import conservative_activity
from rola._state import CANONICAL_ORDER, PAGE_RECTANGLE_BITS, SPLIT_PLANE_DTYPE, StateFormat
from rola.engine.facts.liveness import LivenessLayout
from rola.ops import carry as carry_ops
from rola.ops._ext import is_available

pytestmark = pytest.mark.skipif(not is_available() or not torch.cuda.is_available(),
                                reason="the launch surface is the extension's boundary")

#: THE ONE REASON, as `csrc/rola/rola_api.cpp`'s `kCarryUnimplemented` spells it. It is
#: matched rather than restated so a message that drifts fails the meta-test below
#: instead of silently widening what "the one reason" means.
NO_IMPLEMENTATION = re.compile(r"no implementation on this line")

#: The smallest lawful descriptor: two levels at the width floor, a shipped DV.
SMALL = (16, 16)
DV = 64
TOKENS = 64
BH = 1


def descriptor(widths=SMALL, dv=DV, bh=BH) -> StateFormat:
    n = 1
    for width in widths:
        n *= width
    return StateFormat(D=len(widths), B=tuple(widths), order=CANONICAL_ORDER,
                       page_bits=PAGE_RECTANGLE_BITS, DV=dv, dtype=SPLIT_PLANE_DTYPE,
                       ids=bh * (n >> PAGE_RECTANGLE_BITS))


def operands(desc: StateFormat, length=TOKENS, bh=BH, device="cuda"):
    """A lawful call's every operand, at this descriptor. Values are irrelevant here."""
    wtot = sum(desc.B)
    routes = carry_ops.RoutePlanes(
        read=torch.zeros((bh, length, wtot), dtype=torch.bfloat16, device=device),
        write=torch.zeros((bh, length, wtot), dtype=torch.bfloat16, device=device),
        gain=torch.zeros((bh, length), dtype=torch.bfloat16, device=device))
    v = torch.zeros((bh, length, desc.DV), dtype=torch.bfloat16, device=device)
    layout = LivenessLayout.of(desc, length)
    liveness = carry_ops.LivenessWords(
        words=torch.zeros((bh, 2, layout.rows, layout.words), dtype=torch.int32,
                          device=device))
    activity = conservative_activity(desc, bh, device)
    return routes, v, liveness, activity


def call(desc: StateFormat, launch=None, **overrides):
    """One lawful forward call, with named parts swapped out by the tests."""
    launch = launch or carry_ops.LaunchShape()
    routes, v, liveness, activity = operands(desc)
    kwargs = dict(descriptor=desc, geometry=carry_ops.geometry_block(desc, launch),
                  liveness=liveness, activity=activity, launch=launch)
    args = {"routes": routes, "v": v}
    for key, value in overrides.items():
        (args if key in args else kwargs)[key] = value
    return carry_ops.carry_forward(args["routes"], args["v"], **kwargs)


# ------------------------------------------------------- the surface's shape

def test_the_forward_signature_carries_no_deleted_axis():
    """K50's axis law, read off the signature: the window, the token tile, the stream
    count and the ``(k, m)`` box are NOT arguments.

    `W` is a kernel constant, `C` a file constant, `S` derived and `BC` a launch
    consequence, so a parameter naming any of them would be a dead axis's residue --
    which is a stop condition, never a compatibility shim.
    """
    import inspect

    for entry in (carry_ops.carry_forward, carry_ops.carry_backward):
        names = set(inspect.signature(entry).parameters)
        gone = names & {"window", "chunk", "k", "m", "carve", "level_modes", "widths",
                        "d_v", "bc", "nsr", "nsw", "streams", "W", "C"}
        assert not gone, (
            f"{entry.__name__} still names {sorted(gone)}: every shape a kernel entry "
            f"reads comes from the descriptor and the launch shape (K50's axis law).")
        assert {"descriptor", "geometry", "liveness", "activity", "launch"} <= names, (
            f"{entry.__name__} must take the descriptor, the geometry block, the "
            f"liveness words, the activity bits and the launch shape.")


def test_the_launch_shape_is_the_two_built_ones():
    for warps in carry_ops.WARPS_PER_CTA:
        assert carry_ops.LaunchShape(warps).threads == warps * 32
    assert carry_ops.SHIPPED_WARPS_PER_CTA == 8, (
        "one CTA per SM with the warps the register file admits is the shipped shape "
        "(KERNEL_STANDARDS R15)")
    with pytest.raises(carry_ops.CarryRefusal, match="not a launch shape"):
        carry_ops.LaunchShape(16)


def test_the_box_is_derived_and_halves_with_the_value_width():
    """``BC = leaves_per_warp * warps_per_cta``, and ``leaves_per_warp`` is the register
    file's share divided by DV -- 32 leaves at DV = 64, half that at DV = 128 (K50)."""
    assert carry_ops.leaves_per_warp(64) == 32
    assert carry_ops.leaves_per_warp(128) == 16
    assert carry_ops.box_leaves(64, 8) == 256
    assert carry_ops.box_leaves(128, 8) == 128


# ------------------------------------------ REFUSAL 1: unpadded / unlawful shapes

@pytest.mark.parametrize("widths,reason", [
    ((31, 16), "power of two"),
    ((8, 16), "power of two"),
    ((16, 12), "power of two"),
], ids=["odd", "below-the-floor", "not-a-power-of-two"])
def test_an_unpadded_level_width_is_refused(widths, reason):
    """THE KERNEL DOES NOT PAD. A logical `b_l` of 31 is served by padding the
    producer's logits with -inf to 32, above this boundary; reaching the kernel with 31
    means the translator did not run, and the refusal says so."""
    with pytest.raises(ValueError, match=reason):
        descriptor(widths)


def test_an_unshipped_value_width_is_refused():
    """The refusal lands on the DERIVATION too: an unshipped DV has no register shape,
    so the geometry block cannot be built for it either."""
    with pytest.raises(carry_ops.CarryRefusal, match="not a shipped value width"):
        carry_ops.geometry_block(descriptor(dv=48), carry_ops.LaunchShape())


def test_the_operand_widths_must_be_the_descriptor_s():
    desc = descriptor()
    routes, v, liveness, activity = operands(desc)
    narrow = carry_ops.RoutePlanes(read=routes.read[:, :, :-8], write=routes.write,
                                   gain=routes.gain)
    with pytest.raises(carry_ops.CarryRefusal, match="packed routing plane"):
        call(desc, routes=narrow)
    with pytest.raises(carry_ops.CarryRefusal, match=r"\[BH, L, DV\]"):
        call(desc, v=v[:, :, :-1])


def test_the_liveness_block_is_one_bit_per_digit():
    desc = descriptor()
    routes, v, liveness, activity = operands(desc)
    wrong = carry_ops.LivenessWords(words=liveness.words[:, :, :-1])
    with pytest.raises(carry_ops.CarryRefusal, match="class-1 block"):
        call(desc, liveness=wrong)


def test_the_surface_takes_the_pass_s_words_unchanged():
    """THE TWO CONTRACTS ARE ONE: what the pass emits is what the surface accepts.

    The refusal here is the extension's `carry_forward` by name -- the body is not on
    this line -- which is reached only AFTER every operand refusal has passed, so
    getting that far IS the statement that the words were accepted unchanged.
    """
    from rola.engine.facts.liveness import side_statics
    from rola.ops.liveness import liveness_words

    desc = descriptor()
    routes, v, _, activity = operands(desc)
    layout = LivenessLayout.of(desc, TOKENS)
    statics = side_statics(layout, (False,) * layout.D, layout.B)
    words = liveness_words(routes.read, routes.write, layout, statics, statics)
    assert tuple(words.shape) == (BH, 2, layout.rows, layout.words)
    with pytest.raises((AttributeError, RuntimeError, NotImplementedError)):
        call(desc, liveness=carry_ops.LivenessWords(words=words))


def test_the_activity_byte_is_one_per_page():
    desc = descriptor()
    with pytest.raises(carry_ops.CarryRefusal, match="per-page byte"):
        call(desc, activity=torch.zeros((BH, 3), dtype=torch.uint8, device="cuda"))


def test_a_state_plane_is_page_blocked():
    desc = descriptor()
    flat = torch.zeros((BH, desc.N, desc.cols), dtype=torch.float32, device="cuda")
    with pytest.raises(carry_ops.CarryRefusal, match="state plane is"):
        call(desc, state_out=flat)


# ------------------------------------------------- REFUSAL 2: non-bf16 operands

@pytest.mark.parametrize("field", ["read", "write", "gain", "v"])
def test_a_non_bf16_operand_is_refused(field):
    """BF16 EVERYTHING (R17): one form, the float operand paths deleted. A cast belongs
    above this boundary, where it is visible."""
    desc = descriptor()
    routes, v, liveness, activity = operands(desc)
    if field == "v":
        override = {"v": v.float()}
    else:
        parts = {"read": routes.read, "write": routes.write, "gain": routes.gain}
        parts[field] = parts[field].float()
        override = {"routes": carry_ops.RoutePlanes(**parts)}
    with pytest.raises(carry_ops.CarryRefusal, match="operands are bf16"):
        call(desc, **override)


def test_the_reverse_seeds_are_fp32():
    desc = descriptor()
    launch = carry_ops.LaunchShape()
    routes, v, liveness, activity = operands(desc)
    d_num = torch.zeros((BH, TOKENS, desc.DV), dtype=torch.bfloat16, device="cuda")
    d_den = torch.zeros((BH, TOKENS), dtype=torch.float32, device="cuda")
    with pytest.raises(carry_ops.CarryRefusal, match="reverse seeds"):
        carry_ops.carry_backward(routes, v, d_num, d_den, descriptor=desc,
                                 geometry=carry_ops.geometry_block(desc, launch),
                                 liveness=liveness, activity=activity, launch=launch)


# ------------------------------------------------- REFUSAL 3: descriptor mismatch

def test_a_geometry_block_from_another_descriptor_is_refused():
    """The kernel derives the SAME block at its prologue from the descriptor it is
    handed (K50), so a block derived elsewhere would address somebody else's leaves."""
    desc = descriptor()
    other = carry_ops.geometry_block(descriptor((32, 32)), carry_ops.LaunchShape())
    with pytest.raises(carry_ops.CarryRefusal, match="geometry block was derived for"):
        call(desc, geometry=other)


def test_a_geometry_block_from_another_launch_shape_is_refused():
    desc = descriptor()
    four = carry_ops.geometry_block(desc, carry_ops.LaunchShape(4))
    with pytest.raises(carry_ops.CarryRefusal, match="geometry block was derived for"):
        call(desc, geometry=four, launch=carry_ops.LaunchShape(8))


def test_a_bare_shape_is_not_a_descriptor():
    desc = descriptor()
    launch = carry_ops.LaunchShape()
    routes, v, liveness, activity = operands(desc)
    with pytest.raises(carry_ops.CarryRefusal, match="StateFormat"):
        carry_ops.carry_forward(routes, v, descriptor=SMALL,
                                geometry=carry_ops.geometry_block(desc, launch),
                                liveness=liveness, activity=activity, launch=launch)


def test_the_state_owns_its_dtype():
    desc = descriptor()
    fp32 = StateFormat(D=desc.D, B=desc.B, order=desc.order, page_bits=desc.page_bits,
                       DV=desc.DV, dtype="fp32", ids=desc.ids)
    with pytest.raises(carry_ops.CarryRefusal, match="split bf16 planes"):
        call(fp32)


# -------------------------------------------- the honest no-implementation failure

def test_a_lawful_forward_call_reaches_the_one_reason():
    with pytest.raises(RuntimeError, match=NO_IMPLEMENTATION):
        call(descriptor())


def test_a_lawful_backward_call_reaches_the_one_reason():
    desc = descriptor()
    launch = carry_ops.LaunchShape()
    routes, v, liveness, activity = operands(desc)
    d_num = torch.zeros((BH, TOKENS, desc.DV), dtype=torch.float32, device="cuda")
    d_den = torch.zeros((BH, TOKENS), dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match=NO_IMPLEMENTATION):
        carry_ops.carry_backward(routes, v, d_num, d_den, descriptor=desc,
                                 geometry=carry_ops.geometry_block(desc, launch),
                                 liveness=liveness, activity=activity, launch=launch)


def test_the_device_build_stamp_reaches_the_one_reason():
    """There is NO device-side fact for this family on this line, and the surface says
    so rather than substituting a path or a hash for it."""
    with pytest.raises(RuntimeError, match=NO_IMPLEMENTATION):
        carry_ops.build_stamp()


def test_the_census_is_empty_and_says_so_without_raising():
    """The census is a fact about the BINARY, and "no arm" is a fact it can state. It is
    the one carry entry that is not a stub: an empty list is the truth of a line whose
    table G4 refills and whose bodies C3 builds."""
    assert carry_ops.arms() == []


def test_the_stubs_are_the_only_ones():
    """THE ONLY REFUSING STUBS IN THIS TREE are the three carry entries. A fourth would
    mean some other family had been half-deleted, which the no-fallback rule forbids."""
    from rola.ops._ext import extension

    stubs = []
    for name in dir(extension()):
        if not name.startswith("carry_"):
            continue
        doc = getattr(extension(), name).__doc__ or ""
        if "CONTRACT ONLY" in doc or "only a BUILT kernel" in doc:
            stubs.append(name)
    assert sorted(stubs) == ["carry_backward", "carry_build_stamp", "carry_forward"]
