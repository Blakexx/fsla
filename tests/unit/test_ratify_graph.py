"""The census on the graph's build (tools/ratify.py's `--graph`): the census record's paths resolve against the runfiles and its
tools and warpline become this process's; the census reads each architecture's static manifest (warpline's form, written here by
warpline's own writer), names every kernel the library carries, and names the launch bounds the assembler discarded. No CUDA, no
GPU."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import checked  # noqa: E402
import ratify  # noqa: E402

PARAMETERS = {"archs": "90a", "contract": "full"}


def manifest(discarded: int = 0, spills: bool = True) -> str:
    """A static manifest of one kernel, as warpline writes one (`manifest.dumps`)."""
    formats = checked.checker().formats
    kernel = {"instructions": 12, "registers": 40, "stack_bytes": 0, "shared_bytes": 0, "barriers": 0, "tensor_steps": 0,
              "local_loads": 3, "local_stores": 2, "local_in_loops": 4, **({"spill_stores": 8, "spill_loads": 4} if spills else {})}
    rules = {"launch_bound_discarded": formats.RuleFindings(discarded, (formats.Finding("k(int *)", discarded, "x"),) if discarded
                                                            else (), "a bound discarded", "lower it")}
    place = formats.Place("k(int *)", True, rules, "_Z1kPi", {"kernel": kernel, "loops_keyed_by": "nesting path", "loops": {}})
    return formats.dumps(formats.Manifest((place,), (), 4))


@pytest.fixture()
def graph(tmp_path, monkeypatch):
    """A census record of one architecture's manifest under a runfiles root; restores this module's graph state after."""
    monkeypatch.setattr(ratify, "_GRAPH", {})
    monkeypatch.setattr(ratify, "cuda_bin", ratify.cuda_bin)
    monkeypatch.setenv("FSLA_WARPLINE_PYTHON", str(checked.checker().root))
    runfiles = tmp_path / "runfiles" / "_main"
    runfiles.mkdir(parents=True)

    def write(text: str = manifest()):
        (runfiles / "lib.sm_90a.manifest.json").write_text(text)
        census = runfiles / "census.json"
        census.write_text(json.dumps({
            "manifests": {"sm_90a": "lib.sm_90a.manifest.json"},
            "tools": {"ptxas": "../cuda+/bin/ptxas", "cuobjdump": "../cuda_tools+/bin/cuobjdump",
                      "nvdisasm": "../cuda_tools+/bin/nvdisasm"},
            "warpline": str(checked.checker().root),
            "parameters": PARAMETERS}))
        ratify.use_graph(census, runfiles)
        return runfiles

    return write


def test_the_census_paths_resolve_against_the_runfiles_and_its_tools_are_this_process_s(graph):
    runfiles = graph()
    assert ratify.cuda_bin("ptxas") == str(runfiles.parent / "cuda+" / "bin" / "ptxas")
    assert ratify._GRAPH["manifests"]["sm_90a"] == runfiles / "lib.sm_90a.manifest.json"


def test_the_compiled_entries_are_every_manifest_s_kernels(graph):
    graph()
    assert ratify.compiled_entries() == ["_Z1kPi"]


def test_the_census_reads_each_kernel_s_figures(graph):
    graph()
    entries, unhonored = ratify.measure_built()
    assert entries == {"sm_90a:_Z1kPi": {"regs": 40, "spill_stores": 8, "spill_loads": 4, "hot_locals": 4, "local_ops": 5}}
    assert unhonored == []


def test_a_discarded_launch_bound_is_named(graph):
    graph(manifest(discarded=1))
    assert ratify.measure_built()[1] == ["sm_90a:_Z1kPi"]


def test_a_kernel_with_no_spill_bytes_is_refused(graph):
    graph(manifest(spills=False))
    with pytest.raises(SystemExit, match="no spill bytes"):
        ratify.measure_built()
