"""THE CENSUS'S LEFT-HAND SIDE — the chunk consumer's instantiation axes, as DATA.

The conformance census (`test_conformance_census.py`) joins the INSTANTIATION
axes with the data-regime axes (`families.REGIME_AXES`). Until P67 D2 the
left-hand side was the retired geometry matrix's `AXIS_DECLARATIONS`, the TILED
consumer's template parameter list `<BC, DV, D, DECAY>` crossed with the runtime
`paging` flag. That consumer retires, and four of those five axes retire with
it — `BC` is a slot tile inside a built chunk arm and is not host-selectable,
`DV` is single-valued (`d_v = 64` is the whole built matrix), `DECAY` is
single-valued (decay is retired from the operator, its semantics kept by the
fp64 oracle). So the census keeps its shape and changes its axes, which is
exactly the move P28 made when an axis left before.

**The surviving axes are the ones a CALLER can move.** A topology is a level
count `D` and, since the built manifest is uniform-width, one width `B`; the
state's backing is dense or paged. Those three are what a family's config picks
and what the kernel envelope is stated in, so those three are the census's
left-hand side.

**Declared here, CHECKED against the shipped arm list.** `tools/gen_shards.py`'s
`CHUNK_ARMS` is THE enumeration of built `(C, BC, D, B)` arms — the same tuple
list the generated instantiation TUs and the `chunk_arms.inc` X-macro come from,
so it cannot disagree with the binary the way a hand-kept list can. The value
sets below are asserted equal to its projections (`test_conformance_census.py`),
which keeps the old M1 property — *a new instantiation axis value fails at
collection until it is declared and rowed* — on the axes that survived.

**Pure data and pure CPU.** No torch, no device, no extension: the census runs
in the CPU CI job, and so does the kernel-envelope predicate below.
"""
from __future__ import annotations

import pathlib
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO / "tools") not in sys.path:
    sys.path.insert(0, str(_REPO / "tools"))

import gen_shards  # noqa: E402

#: `(C, BC, D, B)` -- every built chunk arm, read from the generator that emits
#: the instantiation TUs rather than transcribed.
CHUNK_ARMS: tuple[tuple[int, int, int, int], ...] = tuple(gen_shards.CHUNK_ARMS)

#: The `(D, B)` topologies the dispatch can reach. `rola.ops.chunk` pins
#: `C = CHUNK_TOKENS` and picks `BC` from the manifest, so an arm at any other
#: `C` is built but unreachable from the facade, and the census must be stated
#: over what the DISPATCH reaches or it would claim coverage nobody can run.
CHUNK_TOKENS = 32
BUILT_TOPOLOGIES: frozenset[tuple[int, int]] = frozenset(
    (D, B) for (C, _BC, D, B) in CHUNK_ARMS if C == CHUNK_TOKENS)

#: The value width the whole built matrix is compiled at (`rola.rules.envelope.arm_envelope`).
BUILT_DV = 64


@dataclass(frozen=True)
class AxisDeclaration:
    name: str
    #: The values the shipped product carries.
    values: tuple[Any, ...]
    #: "manifest" -- projected from `CHUNK_ARMS` and checked against it;
    #: "runtime" -- a host-side choice that leaves no symbol behind, so the
    #: census cannot discover it and an undeclared one would be invisible.
    source: str
    basis: str

    def __post_init__(self):
        assert self.basis, f"axis {self.name!r} declares no basis"
        assert self.source in ("manifest", "runtime")


AXIS_DECLARATIONS: tuple[AxisDeclaration, ...] = (
    AxisDeclaration(
        "D", tuple(sorted({D for (D, _B) in BUILT_TOPOLOGIES})), "manifest",
        basis="the number of routed levels. A different `D` is a different "
              "topology -- a different function of different inputs -- so it is "
              "swept rather than compared, and every built value must be "
              "reached by some family or the census is claiming a shape nobody "
              "ran."),
    AxisDeclaration(
        "B", tuple(sorted({B for (_D, B) in BUILT_TOPOLOGIES})), "manifest",
        basis="the uniform level width. The built manifest is uniform-width, so "
              "one number fixes the whole topology beside `D`; it also fixes "
              "`N = B ** D`, which is what the owner blocking and the page "
              "granule are stated in."),
    AxisDeclaration(
        "paging", (False, True), "runtime",
        basis="the state's backing. A page table changes the ADDRESS of an "
              "atom's rows and nothing else, so a residency decision that moved "
              "a number would be a defect. It leaves no symbol in the binary, "
              "so it is declared here or it is invisible."),
)

AXIS_BY_NAME: Mapping[str, AxisDeclaration] = {a.name: a for a in AXIS_DECLARATIONS}


def instantiation_of(config: Mapping[str, Any]) -> dict[str, Any]:
    """A family config's coordinates on the axes above."""
    widths = tuple(config["widths"])
    return {"D": len(widths), "B": widths[0], "paging": bool(config["paging"])}


#: The clauses `rola.rules.envelope.arm_envelope` refuses on, each paired with the
#: substring its production message carries. The mirror below answers with a
#: clause NAME rather than with prose, so the two can be compared without one
#: file having to restate the other's wording -- a comparison on wording would
#: fail on a punctuation edit and pass on a swapped clause, which is exactly
#: backwards.
REFUSAL_CLAUSES: Mapping[str, str] = {
    "d_v": "d_v=",
    "normalization": "normalization=",
    "decay": "decay",
    "widths": "widths=",
    "topology": "topology (",
    "T": "T=",
}


def kernel_refusal(config: Mapping[str, Any]) -> str | None:
    """The clause on which the chunk arm refuses this config, or None if it runs.

    A pure-CPU MIRROR of `rola.rules.envelope.arm_envelope`, clause for clause and in
    the same order, so `families.py` can stay device-free while still knowing
    which arm each family executes on. A mirror can drift, so it is not trusted:
    a CUDA row in `test_families.py` asserts this clause is the one production
    actually refuses on, for every declared family -- the same "a declaration
    cannot lie" discipline the regime cells are already held to.
    """
    widths = tuple(config["widths"])
    if config["d_v"] != BUILT_DV:
        return "d_v"
    if config["norm"] != "global":
        return "normalization"
    if config["decay"]:
        return "decay"
    if len(set(widths)) != 1:
        return "widths"
    if (len(widths), widths[0]) not in BUILT_TOPOLOGIES:
        return "topology"
    if config["T"] % CHUNK_TOKENS:
        return "T"
    return None
