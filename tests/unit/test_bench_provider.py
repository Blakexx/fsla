"""fsla's runner (measure/provider.py): it reads central cells, names an arm by its subject and its call count, offers
the arms this binary carries at the cell's shape, and refuses a cell by a fact -- an iteration build of a declared arm
family, a cell of another kind -- while the unbuilt carry leaves `prefill` unoffered; a layer arm's fixture builds only
the parts the arm reads. No GPU: nothing here builds an arm, and the binary's
arm tables are planted (the decode family's, and the declared tables the graph generates)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "measure", ROOT / "tools"):
    sys.path.insert(0, str(path))

from measure import provider, subjects  # noqa: E402
from measure.cells import by_name  # noqa: E402
from measure.subjects import SUBJECTS  # noqa: E402

DECODE_CELL = "layer-alt-entmax1.5-16x16-T128-h2-dec8-fp32"


@pytest.fixture
def binary(monkeypatch):
    """A binary carrying every decode arm (the intra reads its widths by rule)."""
    import fsla.ops.decode as decode

    tables = SimpleNamespace(decode=[(dv, d, decay) for dv in (32, 64) for d in (1, 2, 3, 4) for decay in (False, True)])
    declared_decode = tuple(tables.decode)
    monkeypatch.setattr(decode, "arms", lambda: tuple(tables.decode))
    monkeypatch.setattr(provider, "_carried", lambda family: tuple(tables.decode))
    monkeypatch.setattr(provider, "_declared_families", lambda: ["decode"])
    monkeypatch.setattr(provider, "_declared", lambda family: declared_decode)
    #: a full-contract build's record, planted as the tables are
    monkeypatch.setattr(provider, "_build_config", lambda: {"iteration": False, "ptxas": "x"})
    #: the window an sm_90a card runs, planted as the tables are: no device answers it here
    monkeypatch.setattr(subjects, "_shared_window", lambda: 1024)
    return tables


def test_an_arm_is_named_by_its_subject_and_its_call_count():
    assert provider.arm_name("prefill") == "prefill"
    assert provider.arm_name("prefill", calls=4) == "prefill@calls=4"


def test_a_cells_arms_are_the_subjects_it_takes(binary):
    names = set(provider.arms(by_name("carry-alt-k4-256x256-T1024")))
    assert all(name.split("@")[0] in SUBJECTS for name in names)
    #: the carry is not built on this line: its op's prefill is offered on no cell
    assert "prefill" not in names | set(provider.arms(by_name("carry-alt-k16-256x256-T1024")))


def test_a_layer_cells_arms_are_the_layer_subjects_its_facts_take(binary):
    names = set(provider.arms(by_name(DECODE_CELL)))
    assert {"entmax_solve", "decode_step"} <= names <= {"entmax_solve", "decode_step", "layer_prefill", "layer_op"}
    #: a prefill-only layer cell takes no decode step: its decode steps are none (a fact of the cell)
    assert "decode_step" not in provider.arms(by_name("layer-alt-entmax1.5-64x64-T2048"))


def test_an_arm_is_offered_only_where_the_binary_carries_its_kernel(binary):
    assert "decode_step" in provider.arms(by_name(DECODE_CELL))
    binary.decode = []
    with pytest.raises(RuntimeError, match="lacks the declared arms.*rebuild it"):
        provider.arms(by_name(DECODE_CELL))


def test_a_binary_built_under_a_narrowed_contract_is_refused_though_its_own_table_is_complete(binary, monkeypatch):
    #: a narrowed build's generated table is its own subset, so declared == built: only the record names it
    monkeypatch.setattr(provider, "_build_config", lambda: {"iteration": True})
    with pytest.raises(RuntimeError, match="iteration build.*--//:contract"):
        provider.arms(by_name(DECODE_CELL))
    monkeypatch.setattr(provider, "_build_config", lambda: {})
    with pytest.raises(RuntimeError, match="no build record"):
        provider.arms(by_name(DECODE_CELL))


def test_a_subject_applies_by_the_cells_facts_never_its_name():
    import dataclasses

    decode = by_name(DECODE_CELL)
    assert "decode_step" in subjects.applicable(decode)
    assert "decode_step" not in subjects.applicable(dataclasses.replace(decode, decode_steps=0))
    assert not subjects.applicable(dataclasses.replace(by_name("carry-alt-k4-256x256-T1024"), backing="paged"))
    #: the intra reads one or two levels of whole 16-digit granules: a depth-three cell takes no prefill
    assert "prefill" in subjects.applicable(by_name("carry-alt-k4-256x256-T1024"))
    assert "prefill" not in subjects.applicable(by_name("carry-alt-k4-16x16x16-T256"))


def test_a_cell_is_refused_by_its_facts(binary):
    assert "prefill" not in provider.arms(by_name("carry-dense-16x16x16-T256"))
    with pytest.raises(TypeError, match="fsla's runner takes a central carry cell or layer cell, got a qkv cell"):
        provider.arms(by_name("qkv-L1024-dv64"))
    binary.decode = []
    with pytest.raises(RuntimeError, match="lacks the declared arms"):
        provider.arms(by_name("carry-alt-k4-256x256-T1024"))


def test_the_declared_arms_are_the_graphs_generated_table_and_its_absence_names_the_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(provider, "CHECKOUT", tmp_path)
    with pytest.raises(RuntimeError, match="a timing bundle lays out"):
        provider._declared("decode")
    generated = tmp_path / "build" / "generated"
    generated.mkdir(parents=True)
    (generated / "decode_arms.py").write_text('KEY = ("a", "b")\nARMS = (\n    (1, 2),\n)\n')
    assert provider._declared("decode") == ((1, 2),)


def test_the_declared_families_are_the_registry_not_a_directory(tmp_path, monkeypatch):
    (tmp_path / "bazel").mkdir()
    (tmp_path / "bazel" / "arms.bzl").write_text('ARM_FAMILIES = ["decode"]\n')
    (tmp_path / "build" / "generated").mkdir(parents=True)
    (tmp_path / "build" / "generated" / "stale_arms.py").write_text("ARMS = ()\n")
    monkeypatch.setattr(provider, "CHECKOUT", tmp_path)
    assert provider._declared_families() == ["decode"]


def test_a_layer_fixture_builds_only_the_parts_an_arm_reads_so_the_solve_never_runs_the_producer(monkeypatch):
    """The solve reads the cell and the layer's routing; only an arm that reads the produced routes runs the producer,
    whose kernel this line does not build. The parts are planted: nothing here touches a device."""
    from measure.cells import layer

    built = []

    def part(name, value):
        def make(fx):
            built.append(name)
            return value(fx)
        return make

    monkeypatch.setitem(layer.PARTS, "layer", part("layer", lambda fx: SimpleNamespace(routing="R")))
    monkeypatch.setitem(layer.PARTS, "drawn", part("drawn", lambda fx: SimpleNamespace(x="X", v="V")))
    monkeypatch.setitem(layer.PARTS, "routes", part("routes", lambda fx: (fx["layer"], fx["x"])))
    cell = by_name("layer-alt-entmax1.5-64x64-T2048")
    fx = layer.build(cell)
    assert built == []
    assert (fx["cell"], fx["layer"].routing) == (cell, "R")
    assert built == ["layer"]
    fx["layer"]
    assert built == ["layer"], "a part is built once"
    fx["routes"]
    assert built == ["layer", "routes", "drawn"]
    with pytest.raises(KeyError):
        fx["v_prefill"]
    with pytest.raises(TypeError, match="a layer is built on a layer cell"):
        layer.build(by_name("carry-alt-k4-256x256-T1024"))


def test_a_layer_cells_routing_is_the_fsla_routing_its_entries_name():
    """The cell states its routing in FSLA's own terms; the layer is built with exactly that declaration."""
    from fsla import Routing, entmax, softmax
    from measure.cells.layer import routing

    assert routing(by_name("layer-alt-entmax1.5-64x64-T2048")) == Routing(
        topology=(64, 64), write=(entmax(1.5), softmax()), read=((softmax(), entmax(1.5)),))
    assert routing(by_name("layer-dense-softmax-256-T128-h2-dec8-fp32")) == Routing(
        topology=(256,), write=(softmax(),), read=((softmax(),),))
