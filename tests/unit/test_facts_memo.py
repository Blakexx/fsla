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

    skipped, calls = _counting(gives=("arena",), masks=(Mask.M2,))
    facts = run((skipped,), {"mask": Mask.M1})
    assert facts["arena"] is None and not calls


def test_a_supplied_fact_short_circuits_its_producer():
    probe, calls = _counting(gives=("read_plane",))
    facts = run((probe,), {"read_plane": "supplied"})
    assert facts["read_plane"] == "supplied" and not calls


def test_the_chunk_dag_is_walkable_and_declares_its_discipline():
    """The shipped DAG is validated at import; this is that check made visible, plus
    the two properties a reader would otherwise have to trust."""
    from rola.dags.chunk_dag import CHUNK_DAG

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


# ---------------------------------------------------------------------------
# the decode family's two structural declarations
# ---------------------------------------------------------------------------

def test_a_capturable_node_may_not_read_the_device_on_the_host():
    """The two attributes are one claim, so declaring both is refused at declaration."""
    with pytest.raises(ValueError, match="cannot be capturable"):
        node(gives=("grow",), host_sync=True, capturable=True)(lambda f: True)


def test_the_decode_step_is_declared_capturable_end_to_end():
    """THE CAPTURABILITY CLAIM, structurally: no host read, and every node says so.

    `tests/integration/test_decode_graph_step.py` proves a real capture replays; this
    proves the claim is a property of the DECLARATION, so a node added to the step
    without the attribute fails here rather than at someone's first `graph.replay()`.
    """
    from rola.dags.decode_dag import DECODE_STEP, DECODE_VERDICT, read_verdict

    assert all(step.capturable for step in DECODE_STEP)
    assert not any(step.host_sync for step in DECODE_STEP)
    assert [step for step in DECODE_VERDICT if step.host_sync] == [read_verdict]
    #: and the verdict read is the LAST node: it is enqueued behind the launches it
    #: reads the answer of, which is the whole of P67-a's host order.
    assert DECODE_VERDICT[-1] is read_verdict


def test_a_reference_only_node_is_refused_by_every_shipped_dag():
    """§H.7's hand-written warning, made mechanical.

    `write_atom_bitmap` is the gate's independent second derivation of the write-atom
    set. Splicing it into a shipped DAG would make the gate compare the probe against
    itself, and `validate_dag` is where that is refused.
    """
    from rola.dags.backward_dag import BACKWARD_DAG
    from rola.dags.chunk_dag import CHUNK_DAG
    from rola.dags.decode_dag import DECODE_STEP, DECODE_VERDICT, write_atom_bitmap

    assert write_atom_bitmap.reference_only
    for dag in (CHUNK_DAG, BACKWARD_DAG, DECODE_STEP, DECODE_VERDICT):
        assert not any(step.reference_only for step in dag)
    with pytest.raises(ValueError, match="second derivation"):
        validate_dag((write_atom_bitmap,), ("routes", "geometry"))


def test_the_decode_dag_is_walkable_and_masks_the_growth_band_off_the_dense_step():
    from rola.dags.decode_dag import DECODE_STEP, growth_probe

    assert all(isinstance(step, Node) for step in DECODE_STEP)
    assert growth_probe.masks == (Mask.M4,), (
        "the growth probe is the paged backing's own stats pass; asking it of a dense "
        "step would launch a probe against a page table that does not exist")
    order = [step.name for step in DECODE_STEP]
    assert order.index("growth_probe") < order.index("execute"), (
        "the step reads the probe's verdict on the device, so the probe must be "
        "enqueued ahead of it")
