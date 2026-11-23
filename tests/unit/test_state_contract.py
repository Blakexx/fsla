"""THE STATE OWNS ITS FORMAT: the descriptor's field list, pinned without a kernel.

`fsla/_state.py::StateFormat` is what every kernel derives its addressing from and what a
later call is checked against (`docs/internals/state.md#format`). This file pins that
contract at the Python object -- no CUDA extension, no kernel body, no GPU required -- so
the pipelined body inherits a descriptor whose shape cannot silently drift underneath it.

Symbols, each at first use: `D` routing levels; `B_l` level `l`'s digit width (the
descriptor's `B` tuple); `DV` the value width; `cols = DV + 1` the page's logical width
(the mass column); an **atom** is 16 consecutive canonical leaves, the page granule.

**THE ONE ADMISSION LAW.** K50: "prefill and decode share ONE admission law" -- there is no
decode-only shape, tolerance or topology. `FSLAState._kernel_entry` (prefill) and
`FSLAState._decode_entry` (decode) both build the call's presented format through the SAME
free function, `_format_of`, and check it against the bound one through the SAME method,
`StateFormat.check` -- never a second comparison written at either seam. That is asserted
here structurally (each entry point's own source names both calls), which is what makes a
future edit that reintroduces a decode-only shape check a self-evident diff instead of a
silent divergence.
"""

from __future__ import annotations

import inspect

import pytest

from fsla._state import (
    CANONICAL_ORDER,
    MIN_LEVEL_WIDTH,
    PAGE_RECTANGLE_BITS,
    SPLIT_PLANE_DTYPE,
    STATE_DTYPE,
    FSLAState,
    StateFormat,
    _format_of,
)

#: A LAWFUL DESCRIPTOR, D = 2, at the flagship's own widths -- the fixture every test below
#: mutates one field of at a time.
_FIELDS = dict(D=2, B=(256, 256), order=CANONICAL_ORDER, page_bits=PAGE_RECTANGLE_BITS,
               DV=64, dtype=STATE_DTYPE, ids=8 * (65536 >> PAGE_RECTANGLE_BITS))


def _format(**over):
    return StateFormat(**{**_FIELDS, **over})


# -- THE FIELD LIST, AS ASSERTED -----------------------------------------------------------

def test_the_descriptor_is_exactly_these_seven_fields():
    """The kernel-facing surface: D, per-level B, order, page_bits, DV, dtype, ids.

    A field added or removed here is a contract change, not a refactor -- `check()`
    iterates a hand-named tuple of them (below), so an eighth field silently unchecked is
    exactly the drift this test exists to catch.
    """
    assert [f.name for f in StateFormat.__dataclass_fields__.values()] == [
        "D", "B", "order", "page_bits", "DV", "dtype", "ids"]


def test_the_descriptor_is_frozen():
    """A bound state's format never mutates in place -- rebinding is a fresh object."""
    fmt = _format()
    with pytest.raises((AttributeError, TypeError)):
        fmt.D = 3  # type: ignore[misc]


def test_derived_properties_read_off_the_fields_and_nothing_else():
    fmt = _format(D=2, B=(256, 256), DV=64)
    assert fmt.N == 256 * 256, "N is the mixed-radix product, never configured directly"
    assert fmt.cols == 65, "cols = DV + 1, the mass column"
    assert fmt.atoms_per_bh == fmt.N >> PAGE_RECTANGLE_BITS
    assert fmt.page_bytes == 16 * 64 * 2 + 16 * 64 + 16 * 2 + 16 * 4 == 3168, (
        "[16 x DV hi s16][16 x DV lo u8][16 x DV/32 exponent s8][16 mass fp32] -- one page, in bytes")
    #: the unported carry's own descriptor keeps its split fp32 page
    assert _format(dtype=SPLIT_PLANE_DTYPE).page_bytes == (1 << PAGE_RECTANGLE_BITS) * fmt.cols * 4


# -- THE FIELDS' OWN LAWS, EACH A REFUSAL --------------------------------------------------

def test_b_must_agree_with_d_in_length():
    with pytest.raises(ValueError, match="D=2 levels and 3 widths"):
        _format(D=2, B=(256, 256, 256))


@pytest.mark.parametrize("width", [15, 24, 17, 0, -16])
def test_every_level_width_is_a_power_of_two_at_or_above_the_floor(width):
    """the restored floor: B_l >= 16 (= K_max), power of two -- no narrower level exists."""
    with pytest.raises(ValueError, match=f"level 0 is {width}"):
        _format(D=2, B=(width, 256))


def test_min_level_width_constant_is_the_floor_every_field_test_assumes():
    assert MIN_LEVEL_WIDTH == 16


def test_the_only_leaf_order_is_the_canonical_one():
    with pytest.raises(ValueError, match="only leaf order this build addresses"):
        _format(order="mixed-radix-lsb-first")


def test_page_bits_is_the_page_rectangle_and_nothing_else():
    with pytest.raises(ValueError, match="trailing .* canonical bits"):
        _format(page_bits=PAGE_RECTANGLE_BITS + 1)


def test_dv_is_whole_channel_groups():
    """A channel group is 32 channels, each with its own exponent a leaf: DV is a multiple of 32 (the carry's split
    page needs only an even DV)."""
    with pytest.raises(ValueError, match="multiple of 32"):
        _format(D=1, B=(65536,), DV=48, ids=8 * (65536 >> PAGE_RECTANGLE_BITS))
    with pytest.raises(ValueError, match="DV is even"):
        _format(D=1, B=(65536,), DV=63, dtype=SPLIT_PLANE_DTYPE, ids=8 * (65536 >> PAGE_RECTANGLE_BITS))


def test_n_is_a_whole_number_of_pages_by_construction_not_by_padding():
    """RAGGED N IS REFUSED, NEVER PADDED: under the width floor every lawful `B`
    already makes `N` a multiple of the page, so the refusal is a standing invariant of
    the OTHER fields' laws, not a fourth, independent thing to satisfy."""
    fmt = _format(D=3, B=(16, 16, 16), ids=8 * ((16 ** 3) >> PAGE_RECTANGLE_BITS))
    assert fmt.N % (1 << PAGE_RECTANGLE_BITS) == 0


def test_dtype_names_the_int8_state_format():
    """THE STATE FORMAT (ruled 2026-09-28): s24 hi/lo planes at an s8 group exponent, the mass fp32 -- `dtype`
    records the layout the kernels read (`state_page.md#geometry`)."""
    assert _FIELDS["dtype"] == STATE_DTYPE == "s24 hi/lo planes at an s8 group exponent, fp32 mass"


# -- THE ONE ADMISSION LAW, PREFILL AND DECODE IDENTICAL -----------------------------------

def test_check_accepts_an_identical_presented_format():
    bound = _format()
    bound.check(_format())  # no raise: same fields, different object


@pytest.mark.parametrize("field,over", [
    ("D", dict(D=3, B=(256, 256, 256), ids=8 * (256 ** 3 >> PAGE_RECTANGLE_BITS))),
    ("B", dict(B=(64, 1024))),
    ("DV", dict(DV=128)),
    ("dtype", dict(dtype="fp32")),
])
def test_check_refuses_a_call_that_disagrees_naming_the_field(field, over):
    bound = _format()
    with pytest.raises(ValueError, match=f"this state was bound with {field}="):
        bound.check(_format(**over))


def test_kernel_entry_and_decode_entry_both_call_the_one_admission_law():
    """STRUCTURAL PIN: `_format_of` builds the CALL's presented descriptor and
    `StateFormat.check` is the ONE comparison -- both entry points name both, so a future
    decode-only shape/tolerance path (K50: refused by name) shows up as a source diff here
    rather than as a silent divergence between the two seams."""
    prefill_src = inspect.getsource(FSLAState._kernel_entry)
    decode_src = inspect.getsource(FSLAState._decode_entry)
    for src, name in ((prefill_src, "_kernel_entry"), (decode_src, "_decode_entry")):
        assert "_format_of(" in src, f"{name} does not build its presented format via _format_of"
        assert ".check(" in src or "self._format = presented" in src, (
            f"{name} does not route through StateFormat.check (or bind it, on first call)")
    # decode NEVER binds -- a step continues, so its only legal path is the check.
    assert "self._format = presented" not in decode_src
    assert "self._format.check(" in decode_src


def test_format_of_sources_dv_from_the_value_stream_not_the_routing():
    """`cols` needs `d_v`, which the routing does not carry (`state.md#binding`)."""
    assert "d_v" in inspect.signature(_format_of).parameters


# -- DECODE'S OWN OP ENTRY: THE WIDTHS ARE FACTS OF ITS DESCRIPTOR, AND THE CONTRACT IS THE LAW -----

def _widths_topology(widths, *, write=None):
    from fsla.routing.types import IndependentRouting, OpaqueRouting, Topology

    if write is None:
        return Topology(levels=tuple(OpaqueRouting(width=w) for w in widths))
    return Topology(levels=tuple(IndependentRouting(width=w, read=write, write=write) for w in widths))


@pytest.fixture()
def captured(monkeypatch):
    """`decode_descriptor` with the planner replaced by a recorder of the call it states: the facts are the producer's, the check is the contract's."""
    from fsla.ops import produce

    seen = []
    monkeypatch.setattr(produce, "descriptor", lambda **call: seen.append(call) or call)
    produce._decode_descriptor.cache_clear()
    produce._prefill_descriptor.cache_clear()
    yield seen
    produce._decode_descriptor.cache_clear()
    produce._prefill_descriptor.cache_clear()


@pytest.mark.parametrize("widths, bits", [((64, 64), (6, 6, 0, 0)), ((16, 16, 16), (4, 4, 4, 0)), ((16, 32, 64, 256), (4, 5, 6, 8)),
                                          ((8, 8, 8, 8), (3, 3, 3, 3))])
def test_decode_descriptor_states_each_levels_width_as_its_log2_and_zero_beyond_the_depth(captured, widths, bits):
    """THE WIDTHS ARE CALL FACTS: `width_bits_k` is the log2 of level k's width, 0 for a level the call does not have; the depth is `levels`. A width
    below the floor (8) is stated as it is: the contract refuses it, naming `width_bits_k` (`tests/unit/test_planner.py`)."""
    from fsla.ops.produce import decode_descriptor

    decode_descriptor(_widths_topology(widths), d_v=48, decay=True, batch_heads=4, device=0)
    (call,) = captured
    assert (call["width_bits_1"], call["width_bits_2"], call["width_bits_3"], call["width_bits_4"]) == bits
    assert (call["device"], call["d_v"], call["levels"], call["decay"], call["batch_heads"]) == (0, 48, len(widths), 1, 4)
    assert call["kernel"] == "decode" and not {"prefill", "tokens", "stateful"} & set(call) and not any(name.startswith("read_listed_") for name in call)


def test_decode_descriptor_states_each_sides_walk_of_each_level_from_the_producers_rule_over_the_declaration(captured):
    """THE PRODUCER'S RULE: a level a side walks by lists (`read_listed_k` of the prefill, `write_listed_k` of both kernels, 1) is one whose activation has exact zeros -- an opaque level
    may have them; a softmax has none. A level beyond the depth is 0. The planner's launch function derives the whole-atom deposit from the innermost
    level's write flag."""
    from fsla.ops.produce import decode_descriptor
    from fsla.routing.types import SoftmaxActivation

    decode_descriptor(_widths_topology((16, 16)), d_v=64, decay=False, batch_heads=1, device=0)
    decode_descriptor(_widths_topology((16, 16), write=SoftmaxActivation()), d_v=64, decay=False, batch_heads=1, device=0)
    opaque, softmax = captured
    assert [opaque[f"write_listed_{k}"] for k in (1, 2, 3, 4)] == [1, 1, 0, 0]
    assert [softmax[f"write_listed_{k}"] for k in (1, 2, 3, 4)] == [0, 0, 0, 0]


@pytest.mark.parametrize("widths, level, width", [((24, 16), 1, 24), ((16, 31), 2, 31)])
def test_decode_descriptor_refuses_a_width_that_is_no_power_of_two_naming_the_level(captured, widths, level, width):
    """A width the contract states by its log2 must be one: the refusal names the level and the fact, before the planner is asked."""
    from fsla.ops.produce import decode_descriptor

    with pytest.raises(ValueError, match=rf"level {level}'s width is {width}: a level's width is a power of two .*width_bits_{level}"):
        decode_descriptor(_widths_topology(widths), d_v=64, decay=False, batch_heads=1, device=0)
    assert not captured


def test_prefill_descriptor_states_the_call_kind_the_extents_and_the_declarations_walks_and_padded_widths(captured):
    """THE PREFILL'S DESCRIPTOR, `fsla.ops.produce.prefill_descriptor`: the kernel (`prefill`), the caller's extents and declarations, and from the declaration
    each level's PADDED width as its log2 and each side's walk; a declared budget of `None` is the fact 0."""
    from fsla.ops.produce import prefill_descriptor

    prefill_descriptor(_widths_topology((24, 16)), d_v=64, decay=False, tokens=4096, batch_heads=16, stateful=True, device=0)
    prefill_descriptor(_widths_topology((64, 64)), d_v=48, decay=True, tokens=1, batch_heads=1, stateful=False, device=0)
    first, second = captured
    assert (first["kernel"], first["tokens"], first["batch_heads"], first["stateful"]) == ("prefill", 4096, 16, 1)
    assert (first["levels"], first["width_bits_1"], first["width_bits_2"], first["width_bits_3"]) == (2, 5, 4, 0), "24 pads to 32"
    assert [first[f"read_listed_{k}"] for k in (1, 2, 3, 4)] == [1, 1, 0, 0] and [first[f"write_listed_{k}"] for k in (1, 2, 3, 4)] == [1, 1, 0, 0]
    assert (second["decay"], second["stateful"], second["d_v"]) == (1, 0, 48)


def test_a_state_owns_its_backing_paged_by_default_and_dense_when_made_so():
    """THE STATE'S BACKING is chosen when the state is made (R13: the state owns its format), the one way to choose it: paged by default, one
    dense plane for `backing="dense"`; anything else is refused by name."""
    import fsla

    assert fsla.state()._wants_pages() is True
    assert fsla.state(backing="dense")._wants_pages() is False
    with pytest.raises(ValueError, match="backing must be 'paged' or 'dense'"):
        fsla.state(backing="sparse")
