# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The backward family's DAG, read top to bottom.

THE BACKWARD FINDS NO FACTS. Every fact its nodes read is one the forward already
produced, so this DAG's plan is an INPUT rather than its terminal output: it is
handed the :class:`~rola.engine.plan.BackwardPlan` the grad-bearing forward retained,
and what its walk produces is the operand-layout gradient list
(docs/internals/engine/backward_dag.md).

The two scans are the reason the DAG exists in this shape: they read the SAME facts
and neither reads the other's, which the `needs` edges now state. They are still
issued in tuple order on one stream -- putting them on two is a measurement, and this
stage claims none.

Nodes are declared over both masks because GRAD IS NOT A MASK: it is a retention
decision the forward already made, and a call that reaches here retained a plan
(docs/internals/engine/engine.md §4).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rola.engine import Seeds, node, rules, run, validate_dag

if TYPE_CHECKING:
    import torch

    from rola.engine import BackwardPlan

_SEEDS = ("plan", "dy", "y", "v", "call_class")


@node(gives=("seeds",), needs=("plan", "dy", "y"), stream="main")
def backward_seeds(f):
    from rola.ops.chunk import backward_seeds as build

    return Seeds(*build(f["plan"], f["dy"], f["y"]))


@node(gives=("d_read",), needs=("plan", "seeds"), stream="main")
def scan_a(f):
    from rola.ops.chunk import scan_a as run_scan

    return run_scan(f["plan"], f["seeds"])


@node(gives=("dv", "d_gain", "d_write"), needs=("plan", "seeds"), stream="main")
def scan_b(f):
    """INDEPENDENT OF `scan_a`, and the absent edge is what says so: it reads
    `seeds` and the plan, and nothing scan A produced."""
    from rola.ops.chunk import scan_b as run_scan

    return run_scan(f["plan"], f["seeds"])


@node(gives=("grads",),
      needs=("plan", "seeds", "dv", "d_read", "d_write", "d_gain", "v"),
      stream="main")
def unpack(f):
    """The narrowing and the un-permute are COPIES, which is why this is `main` and
    not `host`: what the host decides here is only the slicing."""
    from rola.ops.chunk import unpack_grads

    return unpack_grads(f["plan"], f["dv"], f["d_read"], f["d_write"], f["d_gain"],
                        f["seeds"].dm, v_dtype=f["v"].dtype)


BACKWARD_DAG = (
    rules.mask.node,
    backward_seeds,
    scan_a,
    scan_b,
    unpack,
)

validate_dag(BACKWARD_DAG, _SEEDS)

if set(scan_a.gives) & set(scan_b.needs) or set(scan_b.gives) & set(scan_a.needs):
    raise ValueError(
        "the two scans acquired an edge between them. They walk the same schedule in "
        "opposite directions off the same retained tables, and their independence is "
        "what a later stage would stream; an edge here would silently remove that.")


def run_backward(plan: BackwardPlan, dy: torch.Tensor, y: torch.Tensor,
                 v: torch.Tensor) -> list[torch.Tensor]:
    """Walk the backward DAG over one retained plan; return the gradients, ordered.

    `dy`, `y` and `v` are SEEDS and not plan fields: `dy` arrives with the call, and
    `y` and `v` ride `save_for_backward` because they are the node's own output and
    operand and autograd owns their versioning. `v` is read for its DTYPE alone --
    the scans consume the plan's packed copy.
    """
    return run(BACKWARD_DAG, {"plan": plan, "dy": dy, "y": y, "v": v,
                              "call_class": plan.forward.call_class})["grads"]
