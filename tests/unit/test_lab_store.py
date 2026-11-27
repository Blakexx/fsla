# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The lab's run stores what it measured through the results store (docs/internals/tools/lab.md#the-loop): a sample of the
record its question keys (measure/data.py's `lab_semantics`) at the run target's label, its body the output and its traced
rows the sample's parts, its provenance the stamp the run target hands; and the one reader (`lab_record`) returns the
newest sample. The run target's command line becomes the driver's, and a command that reads a build is refused where none
was handed. A FIXTURE: a synthetic body and stamp, no device -- what a card's run proves is the measurement."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import lab  # noqa: E402
from fsla_devtools.results import Store, key  # noqa: E402

from measure.data import lab_record, lab_semantics  # noqa: E402

STAMP = {"checkout": "fsla", "branch": "master", "git_sha": "a" * 40, "diff_sha256": None}


class Config:
    """The dev config's scratch space, under the test's directory."""

    def __init__(self, root: Path):
        self.root = root

    def scratch(self, name: str) -> Path:
        return self.root / "scratch" / name


@pytest.fixture
def handed(tmp_path, monkeypatch):
    (tmp_path / "stamp.json").write_text(json.dumps(STAMP))
    monkeypatch.setattr(lab, "_dev_config", lambda: Config(tmp_path))
    monkeypatch.setattr(lab, "HANDED", {"built": tmp_path / "built", "stamp": tmp_path / "stamp.json",
                                        "label": "@fsla//measure/lab:add_rate.run"})
    return tmp_path


def body(median: float) -> dict:
    return {"experiment": "add_rate", "arch": "sm_90a", "device": "NVIDIA H100 80GB HBM3", "git": lab.stamped_git(),
            "results": [{"name": "an add", "unit": "cycles", "median": median}], "checks": []}


def test_a_run_is_stored_as_a_sample_and_read_back_as_the_newest(handed):
    data = handed / "data"
    provenance = {"checkouts": {"fsla": lab.workspace_stamp()}, "binary": "b" * 64}
    traces = ['{"format":"warpline/0"}\n', '{"format":"warpline/0","row":1}\n']
    cited = lab.store_record("add_rate", "sm_90a", body(4.0), kind="run", provenance=provenance, traces=traces,
                      store_root=data / "records")
    first = lab_record("add_rate", "sm_90a", data=data)
    assert cited == first.cited and cited.startswith("data:records/fsla/measure/lab/add_rate.run/")
    assert cited.endswith(f"{first.sample['run']}/output.json")
    assert first.body()["results"][0]["median"] == 4.0 and first.body()["git"] == {"sha": "a" * 40, "dirty": False}
    assert (first.part("trace") / "row1.json").read_text() == traces[1]
    assert first.sample["provenance"]["checkouts"]["fsla"] == STAMP and first.sample["provenance"]["run"]
    lab.store_record("add_rate", "sm_90a", body(5.0), kind="run", provenance=provenance, store_root=data / "records")
    newest = lab_record("add_rate", "sm_90a", data=data)
    assert newest.sample["run"] > first.sample["run"] and newest.body()["results"][0]["median"] == 5.0
    assert not (handed / "scratch" / "lab" / "staged").exists() or not any((handed / "scratch" / "lab" / "staged").iterdir())


def test_a_grid_a_variant_and_a_kind_are_records_of_their_own(handed):
    data = handed / "data"
    for kind, blocks in (("run", 1), ("counters", 0)):
        lab.store_record("add_rate", "sm_90a", body(float(blocks)), kind=kind, blocks=blocks, provenance={},
                  store_root=data / "records")
    assert lab_record("add_rate", "sm_90a", blocks=1, data=data).body()["results"][0]["median"] == 1.0
    counters = Store("@fsla//measure/lab:add_rate.run", data / "records").samples(
        key(lab_semantics("add_rate", "sm_90a", kind="counters")))
    assert len(counters) == 1
    with pytest.raises(SystemExit, match="no lab record"):
        lab_record("add_rate", "sm_90a", data=data)
    with pytest.raises(SystemExit, match="no lab record"):
        lab_record("add_rate", "sm_90a", variant="unlocked", data=data)
    assert lab_semantics("add_rate", "sm_90a", blocks=1, variant="unlocked") == {
        "experiment": "add_rate", "arch": "sm_90a", "kind": "run", "blocks": 1, "variant": "unlocked"}
    with pytest.raises(ValueError, match="a lab record is of"):
        lab_semantics("add_rate", "sm_90a", kind="trace")


def test_no_store_keeps_the_record_in_the_scratch_store(handed):
    lab.store_record("add_rate", "sm_90a", body(4.0), kind="run", provenance={}, no_store=True)
    assert lab_record("add_rate", "sm_90a", data=lab.scratch_store()).body()["results"][0]["median"] == 4.0


def test_the_run_targets_command_line_is_the_drivers(monkeypatch):
    monkeypatch.setattr(lab, "HANDED", {})
    handed = ["--built", "/b", "--stamp", "/s.json", "--label", "@fsla//measure/lab:add_rate.run", "add_rate"]
    assert lab.handed(handed) == ["run", "add_rate"]
    assert {"built": Path("/b"), "stamp": Path("/s.json"), "label": "@fsla//measure/lab:add_rate.run"} == lab.HANDED
    assert lab.handed([*handed, "sass", "k", "--full"]) == ["sass", "add_rate", "k", "--full"]
    assert lab.handed(["trace", "add_rate"]) == ["trace", "add_rate"]
    with pytest.raises(SystemExit, match="runs `trace`"):
        lab.handed([*handed, "trace"])


def test_a_command_that_reads_a_build_is_refused_without_one(monkeypatch):
    monkeypatch.setattr(lab, "HANDED", {})
    with pytest.raises(SystemExit, match=r"bazel run //measure/lab:add_rate.run -- sass"):
        lab.build("add_rate", "sm_90a", "sass")
