# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The DAG framework: a declared node, the structural check, and the walk.

A DAG is a literal ordered tuple of nodes and **the order in the tuple IS the
execution order**, chosen by the DAG's author. `needs` exists to make that order
CHECKABLE and readable, not to compute it: :func:`validate_dag` refuses a DAG whose
node reads a fact no earlier node produced, at import time rather than at launch
(docs/internals/facts/facts.md).

Memoization is a node ATTRIBUTE here rather than a per-site convention: `pure_of`
names the facts a pure host derivation is a function OF, and `per_state` binds a
device allocation to the lifetime of the one sequence that owns it.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from weakref import WeakKeyDictionary

from rola.facts.types import Mask

_STREAMS = ("host", "main", "side")


@dataclass(frozen=True, slots=True)
class Node:
    """One declared unit of the walk: its facts, its edges and its discipline."""

    fn: Callable[[dict], Any]
    name: str
    gives: tuple[str, ...]
    needs: tuple[str, ...]
    stream: str
    masks: tuple[Mask, ...]
    waits: tuple[str, ...]
    host_sync: bool
    pure_of: tuple[str, ...]
    per_state: bool
    memo: Any = field(compare=False)

    def __call__(self, facts: dict) -> Any:
        return self.fn(facts)

    def table(self, facts: dict) -> dict | None:
        """The cache this call's output belongs in, or None when the node is uncached.

        A `per_state` node's table is WEAK-keyed on the state object, and the weakness
        is required rather than incidental: a dropped sequence must drop its buffers,
        and two live sequences must never share one.
        """
        if self.per_state:
            return self.memo.setdefault(facts["state"], {})
        return self.memo if self.pure_of else None


def node(*, gives: tuple[str, ...] = (), needs: tuple[str, ...] = (),
         stream: str = "host", masks: tuple[Mask, ...] = (Mask.M1, Mask.M2),
         waits: tuple[str, ...] = (), host_sync: bool = False,
         pure_of: tuple[str, ...] = (), per_state: bool = False,
         ) -> Callable[[Callable], Node]:
    """Declare a node. `host_sync` marks the ONE node permitted a device-to-host read."""

    def declare(fn: Callable[[dict], Any]) -> Node:
        if stream not in _STREAMS:
            raise ValueError(f"{fn.__name__}: stream={stream!r} is not one of {_STREAMS}")
        if per_state and "state" not in needs:
            raise ValueError(f"{fn.__name__}: a per_state node must read `state`")
        if set(pure_of) - set(needs):
            raise ValueError(
                f"{fn.__name__}: pure_of names {sorted(set(pure_of) - set(needs))} it "
                f"does not read; a memo key that is not an input is not a key")
        return Node(fn=fn, name=fn.__name__, gives=tuple(gives), needs=tuple(needs),
                    stream=stream, masks=tuple(masks), waits=tuple(waits),
                    host_sync=host_sync, pure_of=tuple(pure_of), per_state=per_state,
                    memo=WeakKeyDictionary() if per_state else {})

    return declare


def validate_dag(dag: tuple[Node, ...], seeds: tuple[str, ...]) -> None:
    """Refuse a DAG that is not walkable in its own order. Import-time, per DAG."""
    available = set(seeds)
    for step in dag:
        missing = [name for name in step.needs if name not in available]
        if missing:
            raise ValueError(
                f"{step.name} reads {missing} but no earlier node or seed produces "
                f"them: the tuple's order is the execution order, so a need that "
                f"arrives later is a DAG authoring error")
        available.update(step.gives)

    joins = [step.name for step in dag if step.waits]
    if len(joins) > 1:
        raise ValueError(
            f"{joins}: exactly one Join per DAG. A second event is a second "
            f"synchronization discipline.")
    syncs = [step.name for step in dag if step.host_sync]
    if len(syncs) > 1:
        raise ValueError(f"{syncs}: at most one device-to-host read per DAG")


def run(dag: tuple[Node, ...], facts: dict) -> dict:
    """Walk `dag` in order over the seeded fact bag, and return it."""
    for step in dag:
        call_class = facts.get("call_class")
        if call_class is not None and call_class.mask not in step.masks:
            facts.update({name: None for name in step.gives if name not in facts})
            continue
        if step.gives and all(name in facts for name in step.gives):
            continue
        absent = [name for name in step.needs if name not in facts]
        if absent:
            raise RuntimeError(f"{step.name} was reached without {absent}")
        table = step.table(facts)
        key = tuple(facts[name] for name in step.pure_of)
        if table is not None and key in table:
            produced = table[key]
        else:
            produced = step(facts)
            if table is not None:
                table[key] = produced
        if len(step.gives) == 1:
            facts[step.gives[0]] = produced
        elif step.gives:
            facts.update(dict(zip(step.gives, produced, strict=True)))
    return facts
