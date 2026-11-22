# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE CARRY, as a caller reaches it. No carry kernel is built on this line: :func:`carry` raises
:class:`fsla.ops._ext.KernelNotBuilt`.
"""
from __future__ import annotations

from fsla.ops._ext import extension, not_built

__all__ = ["carry", "sm_clock_ghz"]


def carry(launch, *args, **kwargs):
    """THE LAUNCH: not built on this line (:func:`fsla.ops._ext.not_built`)."""
    not_built("carry")


def sm_clock_ghz() -> float | None:
    """The SM's effective clock under load, read off the device beside a measurement; None from a binary without the
    probe."""
    fn = getattr(extension(), "sm_clock_ghz", None)
    return None if fn is None else float(fn())
