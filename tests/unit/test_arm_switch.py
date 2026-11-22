"""The HOST dispatch primitive, exercised against a synthetic arm set.

`csrc/rola/src/common/arm_switch.cuh` dispatches over the arm set the generated
selection header carries; this file never depends on which arms are actually shipped
(a family's arms are declared in `csrc/rola/src/<family>/arms.bzl`).
This file runs a probe the graph compiled against the SHIPPED HEADER with
a synthetic three-arm selection header in front of it -- rendered by the one renderer
the graph's selection rule uses (`bazel/arms.bzl`'s `arm_selection`), so what is tested is what a build would emit --
and checks every layer of the contract:

* the two counts are the generated ones, and the set names its members;
* a BUILT arm dispatches, and its body reads its own compile-time constants;
* an arm the DECLARATION declares but this build did not carry is refused, and the
  refusal says so with both counts;
* an arm outside the declaration is refused with the FIELD that misses named -- one
  case per field, because a refusal that names the wrong field is worse than one that
  names none;
* the EMPTY set refuses everything, exercised here as a synthetic case (the shipped
  binary's own set carries one arm, not zero).
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
#: the probe, compiled by the graph against a synthetic selection header (tests/BUILD.bazel's `arm_switch_probe_<variant>`,
#: the header `arm_switch_selection.py` renders: three rows, a build carrying the first and the last, or none)
PROBES = REPO / "tests"


def _run(variant):
    import torch

    out = PROBES / f"arm_switch_probe_{variant}"
    #: c10's library, which the probe's `TORCH_CHECK` resolves into
    env = {"LD_LIBRARY_PATH": str(pathlib.Path(torch.__file__).parent / "lib")}
    run = subprocess.run([str(out)], capture_output=True, text=True, timeout=120, env=env)
    assert run.returncode == 0, run.stdout + run.stderr
    return {line.split("\t")[0]: line.split("\t")[1:]
            for line in run.stdout.splitlines() if line}


@pytest.fixture(scope="module")
def built():
    return _run("built")


@pytest.fixture(scope="module")
def empty():
    return _run("empty")


def test_the_counts_and_the_members_are_the_generated_ones(built):
    assert built["count"] == ["2", "3"]
    assert built["members"] == ["(32, 1, 0), (64, 2, 1)"]
    assert built["rows"] == ["2"]


@pytest.mark.parametrize("label,want", [
    ("built_first", ["0", "32", "1", "0"]),
    ("built_last", ["2", "64", "2", "1"]),
])
def test_a_built_arm_dispatches_with_its_own_constants(built, label, want):
    """The body runs under ITS member: a body carrying another arm's constants is
    the failure this makes visible, and it is invisible to a switch that only
    reports whether it matched."""
    assert built[label][0] == "dispatched"
    assert built[label][1:] == want


def test_a_declared_arm_this_build_did_not_carry_is_refused_with_both_counts(built):
    assert built["declared_unbuilt"][0] == "refused"
    message = built["declared_unbuilt"][1]
    assert "this build carries no fixture arm" in message
    assert "2 of 3 declared arms" in message


@pytest.mark.parametrize("label,field", [
    ("undeclared_d_v", "d_v"),
    ("undeclared_levels", "levels"),
    ("undeclared_decay", "decay"),
])
def test_the_refusal_names_the_field_that_misses(built, label, field):
    assert built[label][0] == "refused"
    assert f"the field that misses is {field}." in built[label][1]


def test_an_empty_set_refuses_every_call_and_says_it_carries_none(empty):
    """The shipped binary's state: a family with no built arm refuses by name."""
    assert empty["count"] == ["0", "3"]
    assert empty["members"] == ["none"]
    for label in ("built_first", "built_last", "declared_unbuilt"):
        assert empty[label][0] == "refused"
        assert "0 of 3 declared arms" in empty[label][1]
        assert "none" in empty[label][1]
