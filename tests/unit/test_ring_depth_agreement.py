"""H3 (coverage-closure, 2026-08-03): the RING-DEPTH AGREEMENT gate.

Three parties state a staging-ring depth and nothing pinned them to each other:

* the PLANNER -- ``Plan.ring_slots`` (``rola/planner/schedule.py``), the value a
  plan STATES it was reasoned for;
* the ORACLE QUERY -- the ``ring_slots`` argument ``select_p`` hands the injected
  occupancy oracle (``rola/planner/select.py``), i.e. the value selection reasons
  with;
* the KERNEL -- ``ROLA_C_RING_DEPTH``, the compile-time macro
  ``consumer_ring_depth()`` resolves the staging ring from
  (``csrc/rola/src/consumer_kernel.cuh``). The shipped binary compiles depth 1
  (session-9 R2(b) DECLINED depth 2 on measurement), and per the coverage-closure
  ruling the COMPILED value is the truth the other two must state.

The compiled value is read from the RATIFICATION MANIFEST's provenance blob
(``tools/manifests/sm_86.json``, audit item 10) -- the record produced from the
measured build -- never from a header parse in this test: a header parse would
re-derive the claim instead of checking the recorded one.

The audit found the three in three-way DISAGREEMENT (planner stated 2, kernel
compiled 1, and the runtime ``p.ring_slots`` plumbing was read by no kernel line).
This gate is what makes that class of silent drift a test failure.
"""

from __future__ import annotations

import json
import pathlib

from rola.planner.schedule import _RING_DEPTHS, Plan
from rola.planner.select import HardwareFacts, select_p
from rola.routing.types import IndependentRouting, LevelConfig, SoftmaxActivation, Topology

_MANIFEST = (pathlib.Path(__file__).resolve().parents[2]
             / "tools" / "manifests" / "sm_86.json")


def _compiled_define(name: str) -> int:
    """The macro value the ratified binary was measured under, off the manifest."""
    manifest = json.loads(_MANIFEST.read_text())
    defines = manifest["toolchain"].get("defines")
    assert defines is not None and name in defines, (
        f"the ratification manifest records no {name!r} in its toolchain/provenance "
        f"blob -- the compiled ring/window macros are unverifiable (audit item 10); "
        f"re-run tools/ratify.py --write on a fresh build")
    return int(defines[name])


def _plan() -> Plan:
    routing = IndependentRouting(tied=False, read_activation=SoftmaxActivation(),
                                 write_activation=SoftmaxActivation())
    topo = Topology(levels=(LevelConfig(width=8, routing=routing),
                            LevelConfig(width=8, routing=routing)), d_v=64)
    return Plan(topology=topo, normalization="global", BT=16, BC=16,
                warp_tile=(1, 2), decay=False)


def test_planner_oracle_query_and_compiled_kernel_agree_on_the_ring_depth(monkeypatch):
    compiled = _compiled_define("ROLA_C_RING_DEPTH")

    # Leg 1 -- the kernel implements what it compiled.
    assert compiled in _RING_DEPTHS, (
        f"the binary compiled ring depth {compiled}, which the planner does not even "
        f"advertise ({_RING_DEPTHS})")

    # Leg 2 -- the planner's stated depth is the compiled one.
    plan = _plan()
    assert plan.ring_slots == compiled, (
        f"Plan states ring_slots={plan.ring_slots} but the ratified binary compiled "
        f"ROLA_C_RING_DEPTH={compiled}; the plan is recording a depth no launch runs")

    # Leg 3 -- the value selection hands the occupancy oracle is the same one.
    seen = {}

    def oracle(BT, BC, d_v, global_norm, decay, mma_precision, compact,
               gram_cols, ring_slots):
        seen["ring_slots"] = ring_slots
        return 2

    import rola.routing.sequence_parallel as sp
    monkeypatch.setattr(sp, "select_sequence_parallelism",
                        lambda *a, **k: "SP")
    assert select_p(plan, object(), HardwareFacts(sm_count=1, occupancy=oracle)) == "SP"
    assert seen["ring_slots"] == compiled, (
        f"select_p queried the occupancy oracle at ring_slots={seen['ring_slots']} "
        f"but the binary compiled {compiled}")


def test_the_wt_window_macro_is_recorded_in_the_manifest():
    """Audit item 10's second macro: recorded so drift is checkable, value-pinned to
    the shipped default (0 = full-QPO window; session-9 R2(c) declined 1 as default)."""
    assert _compiled_define("ROLA_C_WT_WINDOW") == 0
