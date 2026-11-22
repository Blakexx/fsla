"""The arm macro (`bazel/arms.bzl`'s `arm_family`) on the fixture family (tests/unit/fixtures/fixture_arms.bzl), which is not a
shipped family: the graph made the Python table, the selection header and one unit an arm for each built architecture, each
with the kernel its arm names and no other. The generated files are the graph's (tests/BUILD.bazel's `fixture`), read
here as data; the objects are the units' own compiles."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "bazel"))

import arms  # noqa: E402
import ratify  # noqa: E402
import sass  # noqa: E402

from measure import hardware  # noqa: E402

GENERATED = ROOT / "tests" / "generated"
#: the rows of tests/unit/fixtures/fixture_arms.bzl, as the table must carry them
EXPECTED_ROWS = ((32, 1, 0), (32, 1, 1), (64, 2, 1))


@pytest.fixture(scope="module")
def table():
    return arms.Table(GENERATED / "fixture_arms.py")


def test_the_table_is_the_declared_rows_on_the_decode_key(table):
    assert table.key == ("d_v", "levels", "decay")
    assert table.arms == EXPECTED_ROWS


def test_the_selection_header_carries_every_arm_when_the_build_names_none(table):
    text = (GENERATED / "fixture_selection.inc").read_text()
    assert arms.check_built_arm_table(text, arms.family_arms(table, "", "--//tests:fixture_arms"), table, "fixture") == []
    assert "#define FSLA_FIXTURE_ARM_COUNT 3" in text and "#define FSLA_FIXTURE_ARM_DECLARED 3" in text
    assert "  F(2, 64, 2, 1)" in text


def test_a_scope_names_arm_indices_or_all_and_refuses_the_rest(table):
    assert arms.family_arms(table, "2,0", "knob") == [0, 2]
    assert arms.family_arms(table, "all", "knob") == [0, 1, 2]
    with pytest.raises(SystemExit, match="no such arm 3"):
        arms.family_arms(table, "3", "knob")
    with pytest.raises(SystemExit, match="not an arm index or 'all'"):
        arms.family_arms(table, "x", "knob")


@pytest.mark.parametrize("arch", [a.removeprefix("sm_") for a in hardware.built()])
@pytest.mark.parametrize("arm", range(len(EXPECTED_ROWS)))
def test_each_arm_has_a_unit_for_each_built_architecture_defining_its_own_kernel(arch, arm):
    obj = GENERATED.parent / "_objs" / f"fixture_arm_{arm}_sm_{arch}" / f"fixture_arm_{arm}.pic.o"
    with sass.sass_lines(obj) as lines:
        found = ratify.sass_scan(lines)
    wanted = f"arm_kernelILi{arm}E"
    assert [key for key in found if wanted in key] == [key for key in found], sorted(found)
    assert found and all(key.startswith(f"sm_{arch}:") for key in found), sorted(found)
    assert re.search(rf"arm_kernelILi{arm}E(?:Li\d+E){{3}}E", next(iter(found)))
