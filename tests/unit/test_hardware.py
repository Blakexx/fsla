# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE HARDWARE CONFIGURATION (`measure/hardware.py` over `csrc/rola/src/common/hardware.json`): the header is what the
file generates and every source holds; a tampered rate or a stale header is caught; a product's override wins over its
architecture's default; and each product's build variant follows its traits."""
from __future__ import annotations

import json

from measure import hardware


def test_the_configuration_checks_clean():
    assert hardware.check() == []


def test_a_measured_rate_off_its_record_is_a_finding():
    cfg = hardware.load()
    cfg["arch"]["sm_90a"]["rates"]["atomic"]["cycles"] = 1.2
    assert any("atomic" in f for f in hardware.check(cfg))


def test_a_stale_header_is_a_finding():
    assert any("hardware.cuh" in f for f in hardware.check(header_text=hardware.HEADER.read_text() + "\n"))


def test_a_product_resolves_its_architectures_rates_and_a_device_name_its_product():
    assert hardware.resolve("sm_90a")["mma.hmma_bf16_f32"].cycles == 4
    assert hardware.resolve(product="h10080gbhbm3")["atomic"].cycles == hardware.resolve("sm_90a")["atomic"].cycles
    assert hardware.product_of("NVIDIA H100 80GB HBM3") == "h10080gbhbm3"
    assert json.loads(hardware.CONFIG.read_text()) == hardware.load()


def test_every_architecture_names_its_family_and_each_built_one_its_window():
    cfg = hardware.load()
    assert {a: r["build"]["family"] for a, r in cfg["arch"].items()} == {"sm_90a": "hopper", "sm_100a": "blackwell"}
    cfg["arch"]["sm_90a"]["build"].pop("window")
    assert any("sm_90a is built but its build chooses no window" in f for f in hardware.check(cfg))


def test_products_equal_in_build_traits_share_a_variant_and_an_override_is_the_next():
    cfg = hardware.load()
    assert {p: hardware.variant_of(p, cfg) for p in cfg["product"]} == {"h10080gbhbm3": 0}
    cfg["product"]["h10080gbhbm3"]["build"] = {"window": 512}
    cfg["product"]["other90"] = {"arch": "sm_90a", "device": "NVIDIA other90", "build": {"window": 512}}
    assert hardware.variant_of("h10080gbhbm3", cfg) == hardware.variant_of("other90", cfg) == 1
    assert [(a, v) for a, v, _ in hardware.builds(cfg) if a == "sm_90a"] == [("sm_90a", 0), ("sm_90a", 1)]
    assert "if (b.cc == 900 && b.variant == 1)" in hardware.header(cfg)


def test_a_product_build_override_outside_the_traits_is_a_finding():
    cfg = hardware.load()
    cfg["product"]["h10080gbhbm3"]["build"] = {"family": "blackwell"}
    assert any("overrides a trait outside window" in f for f in hardware.check(cfg))


def test_a_product_build_override_is_refused_until_a_variant_is_gated():
    cfg = hardware.load()
    cfg["product"]["h10080gbhbm3"]["build"] = {"window": 512}
    assert any("a build variant is not gated yet" in f for f in hardware.check(cfg))
