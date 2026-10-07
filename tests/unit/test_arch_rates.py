# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE MACHINE'S RATES (`measure/arch_rates.py` over `csrc/rola/src/common/arch_rates.cuh`): the one table parses, its
sources hold, a tampered value is caught, and the Python cost is the header's on the header's own parity case."""
from __future__ import annotations

from measure import arch_rates


def test_every_built_arch_has_every_rate_from_a_source_that_holds():
    assert arch_rates.check() == []


def test_a_measured_value_off_its_record_is_a_finding():
    text = arch_rates.HEADER.read_text()
    tampered = text.replace('measured(32.132, "tensor_tax", "hmma f16->f32 / alone")',
                            'measured(30.000, "tensor_tax", "hmma f16->f32 / alone")')
    assert tampered != text
    assert any("mma[kHmmaF16F32]" in f for f in arch_rates.check(tampered))


def test_the_python_cost_is_the_headers_on_its_parity_case():
    text = arch_rates.HEADER.read_text()
    assert "== 133204);" in text
    q = arch_rates.quanta(860)
    assert arch_rates.cost(q, [16, 0, 0, 0], [8, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0], 40, 64, 16, 24) == 133204
