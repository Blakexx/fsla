# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""BAZEL'S TEST SHARDS FOR PYTEST (`bazel/defs.bzl`'s `pytest_tier`): `--bazel-shard INDEX/TOTAL`, which the tier's
script reads off Bazel's sharding protocol, runs every collected test whose position is INDEX modulo TOTAL. A tier's
shards split its collection, never its imports: each shard collects the tier whole. Docs: docs/build-graph.md#checks."""
from __future__ import annotations


def pytest_addoption(parser):
    parser.addoption("--bazel-shard", metavar="INDEX/TOTAL", default=None,
                     help="run this shard's tests: every TOTAL-th collected test from INDEX")


def pytest_collection_modifyitems(config, items):
    shard = config.getoption("--bazel-shard")
    if shard is None:
        return
    index, total = (int(part) for part in shard.split("/"))
    kept = [item for position, item in enumerate(items) if position % total == index]
    config.hook.pytest_deselected(items=[item for position, item in enumerate(items) if position % total != index])
    items[:] = kept
