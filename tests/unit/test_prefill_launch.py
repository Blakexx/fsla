# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PREFILL LAUNCH FUNCTION'S HOST HALF (`csrc/fsla/src/common/plan/prefill_launch.h`, docs/internals/common/plan/prefill_launch.md): `plan_tool --launch json` prints what the host knows of a prefill call from its
descriptor -- the levels' widths, the leaves of a batch-head, the CTA's leaves, the CTAs a head, the window and the windows -- with no GPU. The weights are no fact of the descriptor: the host never reads them, and the carve
(`tests/unit/test_prefill_carve.py`) is a function of widths and weights alone."""
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

HOPPER = registry.device_point()


def _launch(widths: list[int], tokens: int = 65536, **override):
    """The tool's run on a prefill call of the given levels' width bits, every other fact lawful."""
    levels = len(widths)
    facts = {"d_v": 64, "decay": 0, "tokens": tokens, "batch_heads": 1, "stateful": 0, "levels": levels}
    for k in range(1, 5):
        facts[f"width_bits_{k}"] = widths[k - 1] if k <= levels else 0
    facts.update(override)
    return subprocess.run([str(TOOL), "--kernel", "prefill", "--device", ",".join(f"{k}={v}" for k, v in HOPPER.items()),
                           "--call", ",".join(f"{k}={v}" for k, v in facts.items()), "--launch", "json"],
                          capture_output=True, text=True, timeout=60)


def _out(run) -> dict:
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def _carves(widths: list[int], calls: list[tuple[list[int], list[int]]]) -> list[dict]:
    """The tool's carves of the widths under each (write weights, read weights), a thousand a process (an argument has a length limit)."""
    join = lambda v: ",".join(map(str, v))  # noqa: E731
    out = []
    for at in range(0, len(calls), 1000):
        chunk = calls[at:at + 1000]
        run = subprocess.run([str(TOOL), "--cta-carve", ";".join(join(widths) for _ in chunk), "--write-weight", ";".join(join(w) for w, _ in chunk),
                              "--read-weight", ";".join(join(r) for _, r in chunk)], capture_output=True, text=True, timeout=300)
        assert run.returncode == 0, run.stderr
        out += [json.loads(line) for line in run.stdout.splitlines()]
    return out


def test_two_levels_of_256_digits_are_65536_leaves_a_cta_of_512_and_128_ctas_a_head():
    out = _out(_launch([8, 8]))
    assert (out["levels"], out["width_bits"], out["leaves"], out["cta_leaves"], out["ctas_per_head"]) == (2, [8, 8], 65536, 512, 128)
    assert (out["window"], out["windows"]) == (1024, 64)


def test_one_level_of_256_digits_is_one_cta_of_256_leaves_and_one_level_of_16_a_cta_of_16():
    out = _out(_launch([8]))
    assert (out["leaves"], out["cta_leaves"], out["ctas_per_head"]) == (256, 256, 1)
    out = _out(_launch([4]))
    assert (out["leaves"], out["cta_leaves"], out["ctas_per_head"]) == (16, 16, 1)


def test_the_calls_kernels_are_owed_and_the_launch_is_printed_all_the_same():
    out = _out(_launch([8, 8]))
    assert out["kernels"] == "owed" and out["owed"] == ["prefill"]


def test_a_call_is_cut_into_the_matchers_windows():
    for tokens in (1, 1024, 1025, 65536, 65537):
        out = _out(_launch([8, 8], tokens))
        assert out["window"] == 1024 and out["windows"] == -(-tokens // 1024)


def test_the_ctas_a_head_do_not_depend_on_how_the_carve_falls_across_the_levels():
    # ctas_per_head is the call's leaves over the CTA's leaves; every weight pattern's carve, whichever level takes the doublings, leaves the same count
    for widths in ([8, 8], [4, 6, 8], [8, 8, 8, 8], [3, 2], [8]):
        launch = _out(_launch(widths))
        assert launch["ctas_per_head"] == launch["leaves"] // launch["cta_leaves"] == 1 << (sum(widths) - min(9, sum(widths)))
        weights = [list(w) for w in itertools.product((1, 5, 256), repeat=len(widths))]
        calls = [(w, r) for w in weights for r in weights]
        for call, carve in zip(calls, _carves(widths, calls)):
            assert carve["cta_leaves"] == launch["cta_leaves"] and 1 << (sum(widths) - sum(carve["carve_bits"])) == launch["ctas_per_head"], (widths, call)


@pytest.mark.parametrize("override", [dict(tokens=0), dict(levels=5), dict(write_weight_1=256), dict(read_listed_1=1)])
def test_a_call_outside_the_contract_and_a_weight_as_a_fact_are_refused(override):
    run = _launch([8, 8], **override)
    assert run.returncode != 0 and run.stdout == ""
