"""fsla's runner (measure/provider.py): it reads central cells, names an arm by its subject, its call count
and a layer arm's construction, offers the arms this binary carries at the cell's shape, and refuses a cell by name --
an iteration build of a declared arm family, a cell of another kind -- while the unbuilt carry leaves `prefill` unoffered. No GPU: nothing here builds an arm, and the binary's
arm tables are planted (the decode family's, and the declared tables the graph generates)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "measure", ROOT / "tools"):
    sys.path.insert(0, str(path))

from measure import hardware, provider, subjects  # noqa: E402
from measure.cells import by_name  # noqa: E402
from measure.cells.layer import CONSTRUCTIONS  # noqa: E402
from measure.subjects import SUBJECTS  # noqa: E402

DECODE_CELL = "layer-B2-T128-H2-h128-dv64-fp32-s4-dec8"


@pytest.fixture
def binary(monkeypatch):
    """A binary carrying every decode arm (the intra reads its widths by rule)."""
    import fsla.ops.decode as decode

    tables = SimpleNamespace(decode=[(dv, d, decay) for dv in (32, 64) for d in (1, 2, 3, 4) for decay in (False, True)])
    declared_decode = tuple(tables.decode)
    monkeypatch.setattr(decode, "arms", lambda: tuple(tables.decode))
    monkeypatch.setattr(provider, "_declared_families", lambda: ["decode"])
    monkeypatch.setattr(provider, "_declared", lambda family: declared_decode)
    #: the window an sm_90a card's build runs (hardware.json), planted as the tables are: no device answers it here
    monkeypatch.setattr(subjects, "_shared_window", lambda: hardware.load()["arch"]["sm_90a"]["build"]["window"])
    return tables


def test_an_arm_is_named_by_its_subject_its_call_count_and_its_construction():
    assert provider.arm_name("prefill") == "prefill"
    assert provider.arm_name("prefill", calls=4) == "prefill@calls=4"
    assert provider.arm_name("decode_step", construction="chunk-decode-w16") == "decode_step@layer=chunk-decode-w16"


def test_a_cells_arms_are_the_subjects_it_takes(binary):
    names = set(provider.arms(by_name("flagship-alt-k4")))
    assert all(name.split("@")[0] in SUBJECTS for name in names)
    #: the carry is not built on this line: its op's prefill is offered on no cell
    assert "prefill" not in names | set(provider.arms(by_name("flat-small-alt-k16")))


def test_a_layer_cells_arms_are_its_declared_constructions(binary):
    names = set(provider.arms(by_name(DECODE_CELL)))
    declared = sorted(c.name for c in CONSTRUCTIONS.values() if DECODE_CELL in c.cells)
    assert declared == ["chunk-decode-w16"]
    assert {"entmax_solve@layer=chunk-decode-w16", "decode_step@layer=chunk-decode-w16"} <= names
    assert all(name.endswith("@layer=chunk-decode-w16") for name in names)


def test_an_arm_is_offered_only_where_the_binary_carries_its_kernel(binary):
    assert "decode_step@layer=chunk-decode-w16" in provider.arms(by_name(DECODE_CELL))
    binary.decode = []
    with pytest.raises(RuntimeError, match="lacks the declared arms.*--//:contract"):
        provider.arms(by_name(DECODE_CELL))


def test_a_cell_is_refused_by_name(binary):
    assert "prefill" not in provider.arms(by_name("deep3-dense"))
    with pytest.raises(TypeError, match="fsla's runner takes a central carry cell or layer cell"):
        provider.arms(by_name("qkv-L1024-dv64"))
    binary.decode = []
    with pytest.raises(RuntimeError, match="iteration build"):
        provider.arms(by_name("flagship-alt-k4"))


def test_the_declared_arms_are_the_graphs_generated_table_and_its_absence_names_the_build(tmp_path, monkeypatch):
    monkeypatch.setattr(provider, "CHECKOUT", tmp_path)
    with pytest.raises(RuntimeError, match="bazel run //:library"):
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
