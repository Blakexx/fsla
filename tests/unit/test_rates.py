# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE RATES (`measure/rates.py` over `measure/rates.json`): every source holds and a tampered rate is caught; a product's override wins over its
architecture's default; and an architecture priced there is one the registry builds."""
from __future__ import annotations

import json

from measure import rates


def test_the_rates_check_clean():
    assert rates.check() == []


def test_a_measured_rate_off_its_record_is_a_finding():
    cfg = rates.load()
    cfg["arch"]["sm_90a"]["rates"]["atomic"]["cycles"] = 1.2
    assert any("atomic" in f for f in rates.check(cfg))


def test_a_product_resolves_its_architectures_rates_and_a_device_name_its_product():
    assert rates.resolve("sm_90a")["mma.hmma_bf16_f32"].value == 4
    assert rates.resolve(product="h10080gbhbm3")["atomic"].value == rates.resolve("sm_90a")["atomic"].value
    assert rates.product_of("NVIDIA H100 80GB HBM3") == "h10080gbhbm3"
    assert json.loads(rates.CONFIG.read_text()) == rates.load()


def test_a_built_architecture_must_resolve_its_pipes_rates():
    cfg = rates.load()
    del cfg["arch"]["sm_90a"]["rates"]["schedulers"]
    assert any("sm_90a is built but its defaults lack schedulers" in f for f in rates.check(cfg))


def test_an_architecture_priced_here_is_one_the_registry_builds_and_a_built_one_is_priced():
    cfg = rates.load()
    cfg["arch"]["sm_100a"] = {"rates": {}}
    assert any("sm_100a: named here but no device part of the registry states it" in f for f in rates.check(cfg))
    del cfg["arch"]["sm_90a"]
    assert any("sm_90a is built but rates.json names it not" in f for f in rates.check(cfg))
