"""fsla's declarations (`declare.py`): the root declares this checkout's build, machine facts and instruments, and a
store for each result, and no timing (fsla times itself with `//measure:session_entmax`); with a timing server, as
fsla-bench declares it, a layer arm is registered per construction on the cells declared for it, and a decode step only
on a cell that decodes, beside the clock reader; every declared code digest names files that exist; loading the file
imports neither fsla nor torch. No GPU: nothing is built."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def declared() -> dict:
    from fsla_devtools.build.declare import Graph, load

    g = Graph()
    public = load(ROOT / "declare.py")["root"](g, python=sys.executable)
    return {"public": public, "targets": g.targets}


def test_the_root_declares_the_checkout_its_instruments_and_their_stores_and_no_timing():
    out = declared()
    labels = set(out["targets"])
    assert {"fsla/binary", "fsla/environment", "fsla/sass", "fsla/support", "store/sass"} <= labels
    assert not {"session", "memory", "store/session", "store/memory", "timing-server", "timing-server-stop",
                "fsla/clock"} & labels
    #: no kernel has a pipe-counter or stall-census row on this line, so neither instrument is declared
    assert not {"fsla/counters", "fsla/census", "fsla/facts_pass"} & labels
    #: the graph's front door takes the host budget for the build: a holder never starts a Bazel build
    assert out["targets"]["fsla/binary"].holds == {} and not out["targets"]["fsla/binary"].cache
    assert set(out["public"]) == {"all"}


def test_with_a_timing_server_a_layer_arm_is_registered_per_construction_on_its_cells():
    from fsla_devtools.build.declare import Graph, load
    from fsla_devtools.timing.declare import start_timing_server

    d = load(ROOT / "declare.py")
    g = Graph()
    d["declare"](g.scoped("fsla"), d["checkout"](ROOT, python=sys.executable, label="fsla"), timing=start_timing_server(g))
    labels = set(g.targets)
    assert {"fsla/entmax_solve@layer=chunk-sparse-gain8", "fsla/entmax_solve@layer=chunk-decode-w16",
            "fsla/decode_step@layer=chunk-decode-w16", "fsla/clock"} <= labels
    assert "fsla/decode_step@layer=chunk-sparse-gain8" not in labels
    assert g.targets["cells/layer-B2-T128-H2-h128-dv64-fp32-s4-dec8"].params == {"cell": "layer-B2-T128-H2-h128-dv64-fp32-s4-dec8"}


def test_every_declared_code_digest_names_files_that_exist():
    for target in declared()["targets"].values():
        if target.code is None:
            continue
        cwd = Path(target.env.cwd)
        paths = target.code.get("files") or [target.code["entry"], *target.code.get("data", ())]
        missing = [p for p in paths if not (cwd / p).exists()]
        assert not missing, f"{target.label} declares code it cannot digest: {missing}"


def test_every_node_declares_the_cells_it_runs_on_and_the_root_takes_no_cell_list():
    """Selection is the build CLI's, by label; a node's cells are its own declaration. The sides are declared beside
    the instruments, and a side that holds no GPU does not read the binary."""
    import inspect

    from fsla_devtools.build.declare import load

    declared = load(ROOT / "declare.py")
    assert "cells" not in inspect.signature(declared["root"]).parameters
    from fsla_devtools.build.declare import Graph

    g = Graph()
    declared["root"](g, python=sys.executable)
    assert "binary" not in g.targets["fsla/side/oracle"].deps and "binary" not in g.targets["fsla/side/producer"].deps
    assert {t.label for t in g.targets["fsla/side/producer"].inputs} == {
        f"cells/{n}" for n in declared["surface_cells"]("producer")}


def test_a_surface_states_what_it_is_but_chooses_no_comparison():
    """fsla-bench owns the rule a surface is compared under across checkouts, as it owns the timing methodology; fsla
    states what a surface IS."""
    from fsla_devtools.build.declare import load

    d = load(ROOT / "declare.py")
    for name, (executor, kind_, tier, holds) in d["SURFACES"].items():
        assert executor.startswith("tests.oracle.sides:") and kind_ in ("carry", "producer") and holds


def test_the_oracle_tier_is_one_target_keyed_on_its_code_and_the_binary_and_stored():
    """The whole pytest tier as a node: a red cell becomes a stored fact, and an unchanged tree is a cache hit."""
    from fsla_devtools.build.declare import Graph, load

    g = Graph()
    load(ROOT / "declare.py")["root"](g, python=sys.executable)
    tier = g.targets["fsla/oracle-tier"]
    assert tier.executor.endswith(":pytest_tier") and tier.cache and tier.holds == {"gpu": 1}
    assert tier.params["paths"] == ["tests/oracle"] and {"binary", "environment"} <= set(tier.deps)
    assert "tests/oracle" in tier.code["files"] and "fsla" in tier.code["files"]
    assert g.targets["store/oracle-tier"].deps["source"] is tier


def test_a_backward_surface_admits_only_the_cells_under_the_retained_state_budget():
    """The rule is STATED, so the excluded cells are a consequence a reader can compute and not a run that died: the
    oracle side retains the fp64 state at every token and its peak is measured at over nine times that."""
    from fsla_devtools.build.declare import load
    from fsla_devtools.cells import central

    d = load(ROOT / "declare.py")
    admitted = d["surface_cells"]("oracle")
    assert admitted and "one-box-short" in admitted and "flat-small-dense" not in admitted
    for name, record in central().cells.items():
        if d["kind"](name) == "carry" and record["params"].get("tier") == "oracle":
            assert (name in admitted) == (d["retained_state_bytes"](record) < d["RETAINED_STATE_BUDGET"])


def test_loading_the_declarations_imports_neither_fsla_nor_torch():
    probe = (f"import json, sys; from fsla_devtools.build.declare import Graph, load; "
             f"load({str(ROOT / 'declare.py')!r})['root'](Graph()); "
             f"print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in ('fsla', 'torch'))))")
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd="/", timeout=120, check=True)
    assert json.loads(done.stdout.strip().splitlines()[-1]) == []
