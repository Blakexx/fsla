# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE HARDWARE CONFIGURATION (`measure/hardware.py` over `csrc/fsla/src/common/hardware.json`): every source holds and a tampered rate is
caught; a product's override wins over its architecture's default; and each product's build variant follows its traits."""
from __future__ import annotations

import json

from measure import hardware


def test_the_configuration_checks_clean():
    assert hardware.check() == []


def test_a_measured_rate_off_its_record_is_a_finding():
    cfg = hardware.load()
    cfg["arch"]["sm_90a"]["rates"]["atomic"]["cycles"] = 1.2
    assert any("atomic" in f for f in hardware.check(cfg))


def test_the_header_is_the_tables_text_and_no_file_of_the_tree_holds_it():
    assert "constexpr bool tabulated(int cc)" in hardware.header()
    assert not list(hardware.COMMON.glob("hardware.cuh"))


def test_a_product_resolves_its_architectures_rates_and_a_device_name_its_product():
    assert hardware.resolve("sm_90a")["mma.hmma_bf16_f32"].cycles == 4
    assert hardware.resolve(product="h10080gbhbm3")["atomic"].cycles == hardware.resolve("sm_90a")["atomic"].cycles
    assert hardware.product_of("NVIDIA H100 80GB HBM3") == "h10080gbhbm3"
    assert json.loads(hardware.CONFIG.read_text()) == hardware.load()


def test_every_architecture_names_its_family_and_the_built_ones_are_those_choosing_a_window():
    cfg = hardware.load()
    assert {a: r["build"]["family"] for a, r in cfg["arch"].items()} == {"sm_90a": "hopper", "sm_100a": "blackwell"}
    assert hardware.built(cfg) == ["sm_90a"]
    cfg["arch"]["sm_90a"]["build"].pop("window")
    assert hardware.built(cfg) == []


def test_a_built_architecture_must_resolve_its_pipes_rates():
    cfg = hardware.load()
    del cfg["arch"]["sm_90a"]["rates"]["schedulers"]
    assert any("sm_90a is built but its defaults lack schedulers" in f for f in hardware.check(cfg))


def test_a_device_runs_its_architectures_one_build():
    cfg = hardware.load()
    assert [(a, t["family"]) for a, t in hardware.builds(cfg)] == [("sm_90a", "hopper"), ("sm_100a", "blackwell")]
    assert "if (b.cc == 900)" in hardware.header(cfg)
    assert "int variant" not in hardware.header(cfg)


def test_a_product_holds_no_build_of_its_own():
    cfg = hardware.load()
    cfg["product"]["h10080gbhbm3"]["build"] = {"window": 512}
    assert any("a product holds no build" in f for f in hardware.check(cfg))
