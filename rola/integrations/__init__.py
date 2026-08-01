# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""Optional adapters onto other people's abstractions.

Nothing in this subpackage is imported by `rola` itself, and nothing in it may be
imported at `rola` import time. Each module here needs a package that is an OPTIONAL
EXTRA (`pip install rola[transformers]`, `rola[fla]`); the hard dependency set stays
`torch` + `einops` (extraction design §3.9, ratified decision O-6).

The native surface is `rola.RoLAState`. These modules exist so that a model built on
`transformers` or on `flash-linear-attention` can hand a `RoLA` layer *its* state
object -- they are conversions at the boundary, not a second stateful surface.
"""
