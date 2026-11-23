# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The facts: what is TRUE about one call, and the primitives that find it.

The packing and the write-atom bitmap (:mod:`fsla.engine.facts.planes`), the operand list
(:mod:`fsla.engine.facts.operands`) and the bundle adapters (:mod:`fsla.engine.facts.call`).

Nothing is re-exported here: the value types and the plans are the ENGINE's core and are imported
from :mod:`fsla.engine`, which is what keeps a domain from being the route to something it does
not own (docs/internals/engine/engine.md).
"""
