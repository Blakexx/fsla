# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The DAG framework's structure and its two caches, WITHOUT a device.

Memoization is a node attribute rather than a per-site convention, and the structural
refusals fire at DAG declaration rather than at launch. Both are host-only properties
and this battery holds them that way: no CUDA, no extension, no kernel. That is a new
capability -- the conventions these replace were only reachable through a launch.
"""
from __future__ import annotations

import gc

import pytest

from rola.facts.runner import Node, node, run, validate_dag
from rola.facts.types import Mask


def _counting(**kwargs):
    """A node whose body records every call, so a cache HIT is observable."""
    calls: list[tuple] = []

    @node(**kwargs)
    def probe(f):
        calls.append(tuple(f[name] for name in kwargs.get("needs", ())))
        return len(calls)

    return probe, calls


class _Sequence:
    """A weak-referenceable stand-in for the state object a sequence is keyed by."""


def test_a_pure_of_node_runs_once_per_key():
    probe, calls = _counting(gives=("geometry",), needs=("widths",), pure_of=("widths",))
    dag = (probe,)

    first = run(dag, {"widths": (64, 64)})["geometry"]
    again = run(dag, {"widths": (64, 64)})["geometry"]
    other = run(dag, {"widths": (16, 16)})["geometry"]

    assert first == again, "an identical key rebuilt the fact"
    assert other != first, "a different key returned the cached fact"
    assert len(calls) == 2, f"the body ran {len(calls)} times over two distinct keys"


def test_a_per_state_node_never_shares_between_two_live_sequences():
    probe, calls = _counting(gives=("scratch",), needs=("state", "widths"),
                             pure_of=("widths",), per_state=True)
    dag = (probe,)
    one, two = _Sequence(), _Sequence()

    a = run(dag, {"state": one, "widths": (64, 64)})["scratch"]
    a_again = run(dag, {"state": one, "widths": (64, 64)})["scratch"]
    b = run(dag, {"state": two, "widths": (64, 64)})["scratch"]

    assert a == a_again, "one sequence rebuilt its own scratch"
    assert b != a, "two concurrent sequences shared one buffer"
    assert len(calls) == 2


def test_a_dropped_sequence_drops_its_scratch():
    """The weakness is required, not incidental: a live table keyed on dead sequences
    is a leak of exactly the device allocations paging exists to bound."""
    @node(gives=("scratch",), needs=("state",), per_state=True)
    def probe(f):
        #: deliberately not closing over the state: a probe that recorded its own
        #: argument would pin the sequence and the table would look strong.
        return object()

    dag = (probe,)

    sequence = _Sequence()
    run(dag, {"state": sequence})
    assert len(probe.memo) == 1
    del sequence
    gc.collect()
    assert len(probe.memo) == 0, "the scratch outlived the sequence that owned it"


def test_a_memo_key_that_is_not_an_input_is_refused():
    with pytest.raises(ValueError, match="does not read"):
        node(gives=("geometry",), needs=("widths",), pure_of=("d_v",))(lambda f: None)


def test_a_per_state_node_must_read_the_state():
    with pytest.raises(ValueError, match="must read `state`"):
        node(gives=("scratch",), needs=("widths",), per_state=True)(lambda f: None)


def test_a_need_that_arrives_later_is_a_declaration_error():
    late, _ = _counting(gives=("arm",))
    early, _ = _counting(gives=("plan",), needs=("arm",))
    with pytest.raises(ValueError, match="reads"):
        validate_dag((early, late), seeds=())


def test_a_dag_declares_exactly_one_join_and_at_most_one_host_read():
    first, _ = _counting(gives=("a",), waits=("residency",))
    second, _ = _counting(gives=("b",), waits=("residency",))
    with pytest.raises(ValueError, match="exactly one Join"):
        validate_dag((first, second), seeds=())

    reader, _ = _counting(gives=("a",), host_sync=True)
    other, _ = _counting(gives=("b",), host_sync=True)
    with pytest.raises(ValueError, match="one device-to-host read"):
        validate_dag((reader, other), seeds=())


def test_a_masked_out_node_leaves_its_facts_null():
    """The runtime-null OUTPUT skip: intra-mask variation is a null plan field, never
    a mask split, so a skipped band's facts must be PRESENT and null rather than
    absent -- a downstream node reads them either way."""

    class _Class:
        mask = Mask.M1

    skipped, calls = _counting(gives=("arena",), masks=(Mask.M2,))
    facts = run((skipped,), {"call_class": _Class()})
    assert facts["arena"] is None and not calls


def test_a_supplied_fact_short_circuits_its_producer():
    probe, calls = _counting(gives=("read_plane",))
    facts = run((probe,), {"read_plane": "supplied"})
    assert facts["read_plane"] == "supplied" and not calls


def test_the_chunk_dag_is_walkable_and_declares_its_discipline():
    """The shipped DAG is validated at import; this is that check made visible, plus
    the two properties a reader would otherwise have to trust."""
    from rola.facts.chunk_dag import CHUNK_DAG

    assert all(isinstance(step, Node) for step in CHUNK_DAG)
    joins = [step for step in CHUNK_DAG if step.waits]
    reads = [step for step in CHUNK_DAG if step.host_sync]
    assert len(joins) == 1 and len(reads) == 1
    atom = next(step for step in CHUNK_DAG if step.name == "atom_bits")
    assert "arm" not in atom.needs, (
        "the atom bitmap acquired an arm edge; residency would stop being an output "
        "of the plan and the admission's overlap would go silently")
    order = [step.name for step in CHUNK_DAG]
    assert order.index("join") < order.index("block_bits"), (
        "the Join moved past the block bitmap, which reads the page table the "
        "admission writes")
