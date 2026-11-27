# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""FSLA'S DECLARATIONS: this checkout's targets for fsla-devtools' declared build system (`fsla_devtools.build`), for
the root that composes checkouts (fsla-bench's).

`declare(g, env, timing=...)` declares one checkout's targets under the graph's scope and returns them (the timing
registrations by arm name), so a root elsewhere (fsla-bench's) loads this file from each checkout it measures, calls it
per checkout with one shared timing server, and composes its own sessions. fsla times itself, and records its
instruments, with Bazel's own targets: `bazel run //measure:session_entmax`, `bazel run //measure:store_oracle_tier`
(`measure/BUILD.bazel`, docs/measurement.md).

Per checkout: `binary` (the build graph's library, `bazel run //:library`, every run), `environment` (the
machine facts, every run), the diff sides and the kernel-vs-oracle diff, and the timing registrations: `entmax_solve`,
`layer_prefill`, `decode_step` and the op on the layer's own bundle (`layer_op`), each on every central layer cell it
takes (a decode step on the cells that take steps: a fact of the cell), and the clock reader. No carry, intra or
producer kernel is built on this line, so the registrations that reach one record its refusal. Every target that reads
the binary depends on the binary and the environment, whose outputs reach its key.
Every cell is a NODE (`fsla_devtools.cells.declare`) whose output is its record and the digest of the code that drew it;
a target that runs on cells takes those nodes as its data inputs, so nothing here or in the build system resolves a cell
by name at run time. This file imports nothing of fsla's: what it reads of a cell is the central record
(`fsla_devtools.cells.cell`).
"""
from __future__ import annotations

from pathlib import Path

from fsla_devtools.build.declare import Env, load
from fsla_devtools.cells import cell, central
from fsla_devtools.cells.declare import cells as cell_nodes
from fsla_devtools.timing.declare import register_clock_reader, register_timing

EXECUTORS = "measure.executors:"
#: where this checkout's repository-local imports resolve, for code digests
IMPORT_ROOTS = ["tools", "."]
LAYER_ARMS = ("entmax_solve", "layer_prefill", "decode_step", "layer_op")


def checkout(path, *, python: str, label: str) -> Env:
    path = Path(path).resolve()
    return Env(label, python, str(path), {"PYTHONPATH": f"{path}:{path}/tools"})


def _code(entry: str, data=()) -> dict:
    return {"entry": entry, "roots": IMPORT_ROOTS, "data": list(data)}


def declare(g, env: Env, *, timing) -> dict:
    """EVERY target this checkout has, its timing entries registered with `timing` (a timing server). A node declares
    the cells it runs on; nothing here takes a cell list, and a build that wants fewer prunes by label."""
    layer = [c for c in sorted(central().cells) if cell(c).kind == "layer"]
    #: THE LIBRARY IS THE BUILD GRAPH'S (docs/build-graph.md#use): the node runs every build (uncached) -- Bazel's cache
    #: is the build's key, and the node's output, the library's digest, keys every target that reads the binary -- and
    #: holds no host slot, since the graph's front door takes the host budget itself (a holder never starts a Bazel
    #: build, build-graph.md rule 13)
    binary = g.node("binary", executor=EXECUTORS + "compile_kernel", env=env, cache=False,
                    code=_code("measure/executors.py"))
    environment = g.node("environment", executor=EXECUTORS + "probe_environment", env=env, cache=False)
    facts = {"binary": binary, "environment": environment}
    out = {"binary": binary, "environment": environment, "entries": {}, "clock": None, "sides": _sides(g, env, facts),
           "diffs": {}}
    verdict = _kernel_vs_oracle(g, env, out["sides"])
    if verdict is not None:
        out["diffs"]["carry-vs-oracle"] = verdict
    timed = _code("measure/executors.py", ["measure/bench"])
    for arm in LAYER_ARMS:
        wanted = [c for c in layer if arm != "decode_step" or cell(c).decode_steps > 0]
        out["entries"][arm] = register_timing(g, arm, server=timing, env=env, executor=EXECUTORS + "timed",
                                              cells=cell_nodes(g, wanted), params={"arm": arm}, deps=facts, code=timed)
    out["clock"] = register_clock_reader(g, "clock", server=timing, env=env, executor=EXECUTORS + "read_clock", deps=facts,
                                         code=_code("measure/executors.py"))
    return out


#: THE DIFF SURFACES (`tests/oracle/sides.py`): what each IS -- the side executor, the cells it runs on (a cell kind
#: and, where it has one, a tier), and what the side holds while it runs. Exposed for fsla-bench, which pairs a surface
#: across checkouts and OWNS the rule it compares them under (its `RULES`), as it owns the timing methodology; the
#: only diff this file declares itself is the kernel against its own oracle, below. What fsla states about a surface's
#: numerics is a FACT, in `NUMERICS`; it states nothing exact of the carry: a token's partial sums from its CTAs land
#: in any order and reassociate, so its ``num`` is compared under its clauses, never bit for bit.
SURFACES = {
    "oracle": ("tests.oracle.sides:oracle", "carry", "oracle", {"host_cpu": "all"}),
    "producer": ("tests.oracle.sides:producer", "producer", None, {"host_cpu": "all"}),
    "carry-kernel": ("tests.oracle.sides:carry_kernel", "carry", "oracle", {"gpu": "all"}),
}
NUMERICS: dict = {}
#: the kernel-vs-oracle diff INSIDE one checkout: the carry's slots against the fp64 inter term's, PER SLOT under each
#: output kind's clauses and envelope (`tests/oracle/tolerances.py`) -- the numeric half of the oracle tier
KERNEL_VS_ORACLE = ("carry-kernel", "tests.oracle.sides:carry_reference", {"num": "CARRY_NUM"})


#: THE BACKWARD'S BUDGET. The oracle side runs `naive_fsla` forward AND backward in fp64, and autograd retains the
#: recurrence's state at every token: `B * H * T * N * (DV + 1) * 8` bytes of retained state, and MEASURED (2026-09-16)
#: a peak above nine times that -- a (64, 64) dense cell of 256 tokens, 1.0 GiB of retained state, could not run under
#: a 9 GiB cap, and the run over every oracle-tier cell was killed by the host. A surface that runs the backward therefore admits only
#: the cells whose retained state is under this budget, stated here so the excluded cells are a consequence a reader can
#: compute and not a run that died.
RETAINED_STATE_BUDGET = 512 * 2**20
BACKWARD_SURFACES = ("oracle",)


def retained_state_bytes(name: str, batch: int = 1, heads: int = 2) -> int:
    c = cell(name)
    return batch * heads * c.tokens * c.routing.N * (c.dv + 1) * 8


def surface_cells(surface: str) -> list:
    """The cells one surface runs on: its kind, at its tier where it has one, and under the backward's budget where the
    surface runs a backward."""
    _executor, kind, tier, _holds = SURFACES[surface]
    return sorted(name for name in central().cells
                  if cell(name).kind == kind and (tier is None or cell(name).tier == tier)
                  and (surface not in BACKWARD_SURFACES or retained_state_bytes(name) < RETAINED_STATE_BUDGET))


SIDE_CODE = ("tests/oracle/sides.py", ["tests/oracle", "fsla", "measure"])


def _sides(g, env: Env, facts: dict) -> dict:
    """One side TARGET per surface, in this checkout, over the cells the surface runs on. A side that launches the
    kernel reads the binary; a pure-torch reference does not, so a rebuilt binary re-keys the one and not the other."""
    from fsla_devtools.diff import side

    return {name: side(g, f"side/{name}", env=env, executor=executor, cells=cell_nodes(g, surface_cells(name)),
                       code=_code(*SIDE_CODE), binds="fsla", holds=holds, deps=facts if "gpu" in holds else {})
            for name, (executor, _kind, _tier, holds) in SURFACES.items()}


def _kernel_vs_oracle(g, env: Env, sides: dict):
    """The carry against the fp64 inter term, in this checkout: the clause rule per output kind; ``None`` while the
    kind's clauses are owed."""
    from fsla_devtools.diff import diff, side

    kernel_surface, reference_executor, kinds = KERNEL_VS_ORACLE
    outputs = load(Path(env.cwd) / "tests" / "oracle" / "tolerances.py")
    #: NO VERDICT WITHOUT A BUILT KERNEL: a kind whose clauses its kernel's measurement still owes (`tolerances.Owed`)
    #: declares no diff, so none is stored until the kernel lands with its measured clauses
    if any(hasattr(outputs[k], "kernel") for k in kinds.values()):
        return None
    nodes = cell_nodes(g, surface_cells(kernel_surface))
    reference = side(g, "side/carry-reference", env=env, executor=reference_executor, cells=nodes,
                     code=_code(*SIDE_CODE), binds="fsla", holds={"gpu": "all"})
    per_quantity = {q: {"clauses": [list(c) for c in outputs[k].clauses], "rtol": outputs[k].rtol,
                        "envelope_from": f"{q}_envelope"} for q, k in kinds.items()}
    #: RECORDED, not a build failure: the verdict is stored per cell, red cells included (a carve the binary has no arm
    #: for is red by decision until kernel work ships it), and the rest of the build measures on
    return diff(g, "diff/carry-vs-oracle", left=sides[kernel_surface], right=reference, strategy="per-slot",
                params={"per_quantity": per_quantity}, minimum=len(nodes), on_difference="record")
