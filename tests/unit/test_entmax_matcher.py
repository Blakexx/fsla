# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE ENTMAX KERNELS' MATCHER AND THEIR CONTRACTS (`csrc/fsla/src/common/plan/entmax_launch.h`, `hardware/kernels/<kernel>/`,
docs/internals/common/plan/entmax_launch.md): `plan_tool --kernel <kernel>` is the matcher on a call given by hand, with no GPU. Every width, both alphas and the four
dtype pairs reach the arm the old switches picked (the padded width's lane width and items a thread); a call outside a contract is refused naming the fact; a
launch of two width classes is refused in the words the solves' level table always used; each family's arm list is the product of its contract's ranges. The Python
copies of the matcher's laws read the extension's answer, and the public surface's own limits are tied to the contracts' ranges."""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hardware import registry  # noqa: E402

TOOL = ROOT / "csrc" / "fsla" / "plan_tool"
PLAN = ROOT / "csrc" / "fsla"

ENTMAX = ("union_forward", "union_backward", "factor_forward", "factor_backward")
SOFTMAX = ("softmax_forward", "softmax_backward")
KERNELS = ENTMAX + SOFTMAX

HOPPER = registry.device_point()

#: the code of each alpha the contracts hold, and the dtypes' codes
ALPHAS = {15: 1, 20: 0}
DTYPES = (0, 1)


def _old_class(width: int) -> int:
    """What `padded_width` computed: the next power of two from 2."""
    w = 1
    while w < width:
        w <<= 1
    return max(w, 2)


def _old_arm(width: int):
    """What the six `int_switch` sites derived from the class: the lane width `min(class, 32)` and the items a thread `class / lane`."""
    klass = _old_class(width)
    lane = min(klass, 32)
    return klass, lane, klass // lane


def _facts(kernel: str, width: int, levels: int = 1, alpha: int = 15, logit: int = 0, value: int = 0) -> str:
    cells = {"width": width, "levels": levels, **({"alpha_tenths": alpha} if kernel in ENTMAX else {}), "logit_dtype": logit, "value_dtype": value}
    return ",".join(f"{k}={v}" for k, v in cells.items())


def _run(kernel: str, call: str, *extra: str):
    return subprocess.run([str(TOOL), "--kernel", kernel, "--device", ",".join(f"{k}={v}" for k, v in HOPPER.items()), "--call", call, *extra],
                          capture_output=True, text=True, timeout=120)


def _blocks(run) -> list[dict]:
    """The printed `descriptor`, `arm` and `launch` lines of each call of a run."""
    assert run.returncode == 0, run.stderr
    lines = run.stdout.splitlines()
    assert len(lines) % 3 == 0
    return [{line.split(": ", 1)[0]: line.split(": ", 1)[1] for line in lines[i:i + 3]} for i in range(0, len(lines), 3)]


def _arm_text(kernel: str, lane: int, items: int, alpha15: int, logit: int, value: int) -> str:
    alpha = f"alpha15={alpha15} " if kernel in ENTMAX else ""
    return f"lw={lane} ipt={items} {alpha}logit={logit} value={value}"


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("logit, value", list(itertools.product(DTYPES, DTYPES)))
def test_every_width_alpha_and_dtype_pair_reaches_the_arm_the_switches_picked(kernel, logit, value):
    for alpha, alpha15 in (ALPHAS.items() if kernel in ENTMAX else [(15, 0)]):
        widths = range(1, 257)
        run = _run(kernel, ";".join(_facts(kernel, w, 1, alpha, logit, value) for w in widths))
        for width, out in zip(widths, _blocks(run), strict=True):
            klass, lane, items = _old_arm(width)
            assert out["arm"] == _arm_text(kernel, lane, items, alpha15, logit, value), (kernel, width, alpha)
            assert out["launch"] == f"klass={klass} levels=1", (kernel, width)


@pytest.mark.parametrize("kernel", KERNELS)
def test_a_launch_of_up_to_sixteen_levels_keeps_its_levels(kernel):
    run = _run(kernel, ";".join(_facts(kernel, 40, levels) for levels in range(1, 17)))
    assert [out["launch"] for out in _blocks(run)] == [f"klass=64 levels={levels}" for levels in range(1, 17)]


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("call, named", [
    ({"width": 257}, "width=257 (the contract: 1..256)"),
    ({"width": 0}, "width=0 (the contract: 1..256)"),
    ({"levels": 0}, "levels=0 (the contract: 1..16)"),
    ({"levels": 17}, "levels=17 (the contract: 1..16)"),
    ({"logit": 2}, "logit_dtype=2 (the contract: 0..1)"),
    ({"value": -1}, "value_dtype=-1 (the contract: 0..1)"),
])
def test_a_call_outside_a_contract_is_refused_naming_the_fact(kernel, call, named):
    facts = {"width": 64, "levels": 1, "logit": 0, "value": 0, **call}
    run = _run(kernel, _facts(kernel, facts["width"], facts["levels"], 15, facts["logit"], facts["value"]))
    assert run.returncode == 1
    assert f"outside FSLA's {kernel} contract" in run.stderr and named in run.stderr


@pytest.mark.parametrize("kernel", ENTMAX)
@pytest.mark.parametrize("alpha", [0, 10, 17, 25])
def test_an_alpha_outside_the_two_the_contract_holds_is_refused(kernel, alpha):
    run = _run(kernel, _facts(kernel, 64, alpha=alpha))
    assert run.returncode == 1
    assert f"outside FSLA's {kernel} contract" in run.stderr and f"alpha_tenths={alpha}" in run.stderr


@pytest.mark.parametrize("kernel", SOFTMAX)
def test_the_softmax_kernels_have_no_alpha(kernel):
    run = _run(kernel, _facts(kernel, 64) + ",alpha_tenths=15")
    assert run.returncode == 1 and "alpha_tenths" in run.stderr and "no fact a " + kernel + " call supplies" in run.stderr


@pytest.mark.parametrize("kernel", KERNELS)
def test_levels_of_one_width_class_launch_together_and_two_classes_are_refused_in_the_level_tables_words(kernel):
    one = _run(kernel, _facts(kernel, 33), "--widths", "33,64,40")
    assert one.returncode == 0, one.stderr
    assert "launch: klass=64 levels=1" in one.stdout and "class: one" in one.stdout
    two = _run(kernel, _facts(kernel, 64), "--widths", "64,16")
    assert two.returncode == 1
    assert "a batched solve must be one width class; level 1 has width 16 against class 64" in two.stderr


def _contract_arms(kernel: str) -> int:
    """The arm count the kernel's contract's ranges state: the width classes its widths fall in, the alphas, and each dtype's range."""
    contexts = registry.contexts(kernel)
    arms = set()
    for c in contexts:
        lo, hi = c.facts["width"]
        classes = {_old_class(w) for w in range(lo, hi + 1)}
        alpha = c.facts.get("alpha_tenths", (0, 0))
        for klass, a, logit, value in itertools.product(
                classes, range(alpha[0], alpha[1] + 1), range(c.facts["logit_dtype"][0], c.facts["logit_dtype"][1] + 1),
                range(c.facts["value_dtype"][0], c.facts["value_dtype"][1] + 1)):
            arms.add((klass, a, logit, value))
    return len(arms)


@pytest.mark.parametrize("kernel", KERNELS)
def test_each_family_has_as_many_arms_as_its_contract_states(kernel):
    rows = json.loads((PLAN / f"{kernel}_plan" / f"{kernel}_arm_rows.json").read_text())
    assert rows["family"] == kernel
    assert len(rows["arms"]) == _contract_arms(kernel) == (64 if kernel in ENTMAX else 32)
    assert rows["key"] == (["lw", "ipt", "alpha15", "logit", "value"] if kernel in ENTMAX else ["lw", "ipt", "logit", "value"])
    assert rows["arms"] == sorted(rows["arms"])


@pytest.mark.parametrize("kernel", KERNELS)
def test_a_contract_states_every_width_both_alphas_the_four_dtype_pairs_and_sixteen_levels(kernel):
    contexts = registry.contexts(kernel)
    assert {c.facts["width"] for c in contexts} == {(1, 256)}
    assert {c.facts["levels"] for c in contexts} == {(1, 16)}
    assert {c.facts["logit_dtype"] for c in contexts} == {(0, 1)} and {c.facts["value_dtype"] for c in contexts} == {(0, 1)}
    if kernel in ENTMAX:
        assert sorted(c.facts["alpha_tenths"] for c in contexts) == [(15, 15), (20, 20)]
    else:
        assert all("alpha_tenths" not in c.facts for c in contexts)


def test_the_public_surfaces_limits_are_the_contracts_ranges():
    """The activation registry's alpha rows and the topology's branch width limit are the PUBLIC surface's own validation; they stay, and each is the contracts' range."""
    from fsla.routing.activations import ratified_entmax_alphas
    from fsla.routing.topology import MAX_BRANCH_WIDTH

    for kernel in KERNELS:
        contexts = registry.contexts(kernel)
        assert max(c.facts["width"][1] for c in contexts) == MAX_BRANCH_WIDTH
        if kernel in ENTMAX:
            assert sorted(c.facts["alpha_tenths"][0] for c in contexts) == sorted(round(a * 10) for a in ratified_entmax_alphas())


def test_the_planners_width_class_and_level_bound_are_the_extensions_answer():
    """Python states neither law: `_width_class` and the level bound of a launch's table are the matcher's, read from the extension."""
    from fsla.ops._ext import extension
    from fsla.routing.entmax import production

    for width in range(1, 257):
        assert production._width_class(width) == _old_class(width) == int(extension().entmax_width_class(width))
    assert production._max_batched_levels() == 16 == max(c.facts["levels"][1] for k in KERNELS for c in registry.contexts(k))
