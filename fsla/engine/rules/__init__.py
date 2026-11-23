# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The rule layer: the pure host folds from values to execution decisions.

A rule may read a value's SHAPE, DTYPE, DEVICE and PRESENCE, and never its CONTENTS. It issues no
launch and allocates nothing, so each is called directly with values and is testable with no device
present (docs/internals/engine/rules.md).
"""

from fsla.engine.rules.envelope import envelope
from fsla.engine.rules.paging import admission
from fsla.engine.rules.run_table import run_table

__all__ = [
    "admission",
    "envelope",
    "run_table",
]
