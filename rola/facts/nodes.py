# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The primitive library: one declared node each, typed on the FACTS.

Nodes are typed on facts and not on any one family's context, which is what lets a
DAG share them (docs/internals/facts/facts.md). Each body is the smallest thing that
produces its fact; the derivations it calls stay where they are implemented.
"""
from __future__ import annotations

from rola.facts.runner import node
from rola.facts.types import Arm, CallClass, Mask


@node(gives=("paging",), needs=("expert",))
def paging_preference(f):
    from rola.interface import _paging_preference

    return _paging_preference(f["expert"])


@node(gives=("widths", "d_v"), needs=("routes", "v"))
def topology(f):
    return (tuple(level.width for level in f["routes"].topology.levels),
            f["v"].shape[-1])


@node(gives=("call_class",),
      needs=("routes", "v", "decay", "state", "given_state_in", "given_arena",
             "want_state", "paging"))
def classify_call(f):
    from rola.interface import operands_of, requires_backward

    state, given_arena = f["state"], f["given_arena"]
    return CallClass(
        state_in=f["given_state_in"] is not None or state is not None,
        state_out=bool(f["want_state"]),
        paged=given_arena is not None or (state is not None
                                          and state._wants_pages(f["paging"])),
        grad=requires_backward(*operands_of(f["routes"], f["v"], f["decay"])))


@node(gives=("arm",), needs=("routes",))
def select_arm(f):
    from rola.ops.chunk import arm

    return Arm(*arm(f["routes"]))


@node(gives=("read_plane", "write_plane", "g_write", "read_mass"), needs=("routes",),
      stream="main")
def pack_planes(f):
    import torch

    from rola.ops.chunk import pack_side, packed_copy

    routes = f["routes"]
    mass = routes.read_mass
    return (pack_side(routes.read),
            pack_side(routes.write),
            packed_copy(routes.g_write.permute(0, 2, 1)),
            #: fp32, because it scales the readout's floor and `eps` is an fp32
            #: definition constant; bf16 here would quantize the floor.
            None if mass is None else packed_copy(mass.permute(0, 2, 1), torch.float32))


@node(gives=("atom_bits",), needs=("write_plane", "widths", "state", "paging"),
      stream="main", masks=(Mask.M2,))
def atom_bits(f):
    """THE ABSENT `arm` EDGE IS THE POINT: the bitmap is keyed to the 16-leaf
    quantum, so residency is an OUTPUT of the plan rather than a consequence of it
    and the admission may run concurrently with the arm's own work."""
    from rola.ops.chunk import atom_bits as build

    state = f["state"]
    if state is None or not state._wants_pages(f["paging"]):
        return None
    return build(f["write_plane"], f["widths"])


@node(gives=("state_in", "arena"),
      needs=("state", "given_state_in", "given_arena", "routes", "d_v", "atom_bits",
             "paging", "arm"),
      stream="side", masks=(Mask.M2,), host_sync=True)
def residency(f):
    state = f["state"]
    if state is None:
        return f["given_state_in"], f["given_arena"]
    return state._kernel_entry(f["routes"], f["atom_bits"], d_v=f["d_v"],
                               paging=f["paging"], BC=f["arm"].bc)


@node(gives=("v_packed",), needs=("v",), stream="main")
def pack_v(f):
    """THE KERNEL TAKES V AS THE CALLER HOLDS IT: token-major, and bf16 because that
    is the operand dtype the consumer reads."""
    import torch

    from rola.ops.chunk import packed_copy

    v = f["v"]
    return v if v.dtype is torch.bfloat16 and v.is_contiguous() else packed_copy(v)


@node(needs=("arena",), stream="main", masks=(Mask.M2,), waits=("residency",))
def join(f):
    """The admission's table write and slot zeroing ran on the arena's side stream;
    this wait puts them before the BLOCK-BITS pass, which READS the page table, and
    not merely before the consumer launch."""
    arena = f["arena"]
    if arena is not None:
        arena.wait()


@node(gives=("page_table", "state_out"), needs=("arena", "want_state", "routes", "d_v"))
def backing(f):
    from rola.ops.chunk import state_plane

    arena, want_state = f["arena"], f["want_state"]
    if arena is None:
        return None, (state_plane(f["routes"], d_v=f["d_v"]) if want_state else None)
    return arena.page_table, (arena.state if want_state else None)


@node(gives=("union_table",),
      needs=("read_plane", "write_plane", "v_packed", "arm", "widths"), stream="main")
def union_table(f):
    from rola.ops._ext import extension

    return extension().chunk_union_table(f["read_plane"], f["write_plane"],
                                         f["v_packed"], *f["arm"], list(f["widths"]))


@node(gives=("block_bits",),
      needs=("union_table", "arm", "widths", "routes", "page_table"), stream="main")
def block_bits(f):
    from rola.ops._ext import extension

    return extension().chunk_block_bits(f["union_table"], f["arm"].bc, list(f["widths"]),
                                        f["routes"].tokens, f["page_table"])
