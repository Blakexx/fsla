# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""Access to the compiled CUDA extension. **This module imports; it does not build.**

That sentence is the whole point of the extraction. In the fork this file called
``torch.utils.cpp_extension.load(...)`` at first use, which meant the binary a user
ran was compiled by *their* toolchain, from source, at import time — the exact
thing the closed-world codegen policy exists to forbid. The census's register and spill
counts describe a binary that was *measured*; a JIT build
produces a different binary every time the environment moves, and no manifest can
speak for it.

So the extension is built ahead of time by the build graph (``bazel run //:library``,
which runs the binary gates on what it links; the wheels package that library) and is
merely *imported* here. No build-environment knob lives in this file: a build's
parameters are the graph's settings, where a build's environment belongs.

There is deliberately **no pure-torch fallback**. ``fsla.ops.naive`` is an fp64
oracle, not a fast path: silently routing to it when the wheel is missing would
turn an installation problem into a ~100x slowdown that no test would catch.
The import failure is loud and names the remedy instead.
"""

from __future__ import annotations

import importlib
from functools import lru_cache

from fsla import __version__

#: THE BINARY PLUGIN (wheel naming D, docs/build.md#wheels): `fsla` is pure Python and the build of each toolchain is
#: its own distribution, `fsla-cu13`, whose package holds the extension and its build record. A
#: checkout's in-place build lays out the same package, so an installed wheel and a checkout take this one import.
PLUGIN, EXTRA = "fsla_cu13", "cu13"


def _load():
    """The extension module and None, or None and why not. The plugin's build record is read BEFORE its library
    loads: a binary built for another `fsla` never registers its operators in this process."""
    try:
        built = importlib.import_module(f"{PLUGIN}._build_config").BUILD_CONFIG
    except ImportError as e:
        return None, f"no `{PLUGIN}` binary plugin is installed ({e})"
    if built["version"] != __version__:
        return None, (f"`{PLUGIN}` {built['version']} is not this `fsla` ({__version__}); the two are one version, "
                      f"installed together")
    try:
        return importlib.import_module(f"{PLUGIN}._C"), None  # loading the library registers torch.ops.fsla
    except ImportError as e:
        return None, f"`{PLUGIN}`'s extension did not load ({e})"


_C, _WHY = _load()


class _Extension:
    """fsla's operators (``torch.ops.fsla``) under the names the package calls them by."""

    def __init__(self, ops) -> None:
        self._ops = ops

    def __getattr__(self, name: str):
        return getattr(self._ops, name)


_fsla_cuda: _Extension | None = None
if _C is not None:
    import torch

    _fsla_cuda = _Extension(torch.ops.fsla)


_MISSING = """\
No FSLA kernel can run: {why}.

Install the binary for your CUDA toolchain at this version:

    pip install "fsla[{extra}]=={version}"

It carries the kernels for every built architecture. A checkout builds
the same package in place (`bazel run //:library`, docs/build.md), and refuses an architecture that
is not built (docs/bringup.md).

There is no CPU or pure-torch fallback by design: `fsla.ops.naive.naive_fsla` is a float64 reference for checking the
kernel, not a substitute for it.\
"""


@lru_cache(maxsize=1)
def extension():
    """Return the compiled extension module, or raise with an actionable message."""
    if _fsla_cuda is None:
        raise ImportError(_MISSING.format(why=_WHY, extra=EXTRA, version=__version__))
    return _fsla_cuda


def is_available() -> bool:
    """True when the compiled extension imported successfully.

    For probing (tests, capability reporting). Never branch the math on this —
    see the module docstring on why there is no fallback path.
    """
    return _fsla_cuda is not None


def device_index(device) -> int:
    """A CUDA device's index for an entry's device argument: an index as given, else the device's, the current one for
    ``None`` or an unindexed ``cuda``. A device that is not CUDA is refused by the envelope's own clause
    (:func:`fsla.engine.rules.envelope.envelope`): THE ONE PLACE a non-CUDA call is refused, since every descriptor is made through it."""
    import torch

    if isinstance(device, int):
        return device
    if device is not None:
        from fsla.engine.rules.envelope import envelope

        reason = envelope(torch.device(device))
        if reason is not None:
            raise NotImplementedError(reason)
    index = None if device is None else torch.device(device).index
    return torch.cuda.current_device() if index is None else index


class KernelNotBuilt(NotImplementedError):
    """An op this line has no kernel for: its fp64 definition stands (`fsla.ops.naive`), its launch raises this."""


def not_built(*kernels: str):
    """Raise :class:`KernelNotBuilt` naming each of ``kernels``."""
    names = " and ".join(kernels)
    raise KernelNotBuilt(
        f"the {names} {'kernel is' if len(kernels) == 1 else 'kernels are'} not built on this line: no hopper or blackwell schedule of "
        f"{'it is' if len(kernels) == 1 else 'them is'} written yet, and the ampere family's is the branch `ampere`'s. The op's fp64 definition is "
        "fsla.ops.naive.naive_fsla.")


__all__ = ["KernelNotBuilt", "device_index", "extension", "is_available", "not_built"]
