"""fsla's declarations (`declare.py`), as fsla-bench's root declares them with a timing server: this checkout's build,
machine facts, diff sides and timing registrations, and no instrument (fsla records its instruments with Bazel's own
targets, `//measure:store_oracle_tier`); a layer arm is registered per construction on the cells declared for it, and a
decode step only on a cell that decodes, beside the clock reader; every declared code digest names files that exist;
loading the file imports neither fsla nor torch. No GPU: nothing is built."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def declared() -> dict:
    from fsla_devtools.build.declare import Graph, load
    from fsla_devtools.timing.declare import start_timing_server

    d = load(ROOT / "declare.py")
    g = Graph()
    out = d["declare"](g.scoped("fsla"), d["checkout"](ROOT, python=sys.executable, label="fsla"),
                       timing=start_timing_server(g))
    return {"declarations": d, "out": out, "targets": g.targets}


def test_the_checkout_declares_its_build_facts_sides_and_timing_and_no_instrument():
    got = declared()
    labels = set(got["targets"])
    assert {"fsla/binary", "fsla/environment", "fsla/side/oracle", "fsla/clock"} <= labels
    assert "instruments" not in got["out"]
    assert not {"fsla/support", "fsla/oracle-tier", "fsla/sass", "fsla/counters", "fsla/census"} & labels
    #: the graph's front door takes the host budget for the build: a holder never starts a Bazel build
    assert got["targets"]["fsla/binary"].holds == {} and not got["targets"]["fsla/binary"].cache


def test_a_layer_arm_is_registered_per_construction_on_its_cells():
    targets = declared()["targets"]
    assert {"fsla/entmax_solve@layer=chunk-p73-pinned", "fsla/entmax_solve@layer=chunk-decode-w16",
            "fsla/decode_step@layer=chunk-decode-w16", "fsla/clock"} <= set(targets)
    assert "fsla/decode_step@layer=chunk-p73-pinned" not in targets
    assert targets["cells/layer-B2-T128-H2-h128-dv64-fp32-s4-dec8"].params == {
        "cell": "layer-B2-T128-H2-h128-dv64-fp32-s4-dec8"}


def test_every_declared_code_digest_names_files_that_exist():
    for target in declared()["targets"].values():
        if target.code is None:
            continue
        cwd = Path(target.env.cwd)
        paths = target.code.get("files") or [target.code["entry"], *target.code.get("data", ())]
        missing = [p for p in paths if not (cwd / p).exists()]
        assert not missing, f"{target.label} declares code it cannot digest: {missing}"


def test_every_node_declares_the_cells_it_runs_on_and_declare_takes_no_cell_list():
    """Selection is the build CLI's, by label; a node's cells are its own declaration. A side that holds no GPU does not
    read the binary."""
    import inspect

    got = declared()
    assert "cells" not in inspect.signature(got["declarations"]["declare"]).parameters
    targets = got["targets"]
    assert "binary" not in targets["fsla/side/oracle"].deps and "binary" not in targets["fsla/side/producer"].deps
    assert {t.label for t in targets["fsla/side/producer"].inputs} == {
        f"cells/{n}" for n in got["declarations"]["surface_cells"]("producer")}


def test_a_surface_states_what_it_is_but_chooses_no_comparison():
    """fsla-bench owns the rule a surface is compared under across checkouts, as it owns the timing methodology; fsla
    states what a surface IS."""
    from fsla_devtools.build.declare import load

    d = load(ROOT / "declare.py")
    for name, (executor, kind_, tier, holds) in d["SURFACES"].items():
        assert executor.startswith("tests.oracle.sides:") and kind_ in ("carry", "producer") and holds


def test_a_backward_surface_admits_only_the_cells_under_the_retained_state_budget():
    """The rule is STATED, so the excluded cells are a consequence a reader can compute and not a run that died: the
    oracle side retains the fp64 state at every token and its peak is measured at over nine times that."""
    from fsla_devtools.build.declare import load
    from fsla_devtools.cells import central

    d = load(ROOT / "declare.py")
    admitted = d["surface_cells"]("oracle")
    assert admitted and "one-box-short" in admitted and "flagship-dense" not in admitted
    for name, record in central().cells.items():
        if d["kind"](name) == "carry" and record["params"].get("tier") == "oracle":
            assert (name in admitted) == (d["retained_state_bytes"](record) < d["RETAINED_STATE_BUDGET"])


def test_loading_the_declarations_imports_neither_fsla_nor_torch():
    probe = (f"import json, sys; from fsla_devtools.build.declare import Graph, load; "
             f"from fsla_devtools.timing.declare import start_timing_server; "
             f"d = load({str(ROOT / 'declare.py')!r}); g = Graph(); "
             f"d['declare'](g, d['checkout']({str(ROOT)!r}, python=sys.executable, label='fsla'), "
             f"timing=start_timing_server(g)); "
             f"print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in ('fsla', 'torch'))))")
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd="/", timeout=120, check=True)
    assert json.loads(done.stdout.strip().splitlines()[-1]) == []
