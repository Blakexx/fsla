# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The planner: the measured cost model that chooses the runtime geometry.

Promoted out of `routing/` into its own tier at extraction. It is not routing —
routing decides *which* leaves a token reads and writes; the planner decides
`(BT, BC, D_V, P, compact)`, the shape of the work the kernel does to serve that
routing, from committed measured cost constants. See docs/planner.md.
"""
