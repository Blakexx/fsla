# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The planner: the closed form that fixes the runtime geometry.

It is not routing — routing decides *which* leaves a token reads and writes; the
planner decides the shape of the work the kernel does to serve that routing. That
shape is one number, `BC`, and it is DERIVED from launch-static facts rather than
selected from a model (`derive.py`). `schedule.py` builds the plan object and the §4
schedule; `dispatch_table.py` is the binary's own arm set. See
docs/architecture/planner.md.
"""
