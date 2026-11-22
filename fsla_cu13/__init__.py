# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""fsla's binary plugin for the cu13 toolchain: the extension ``_C``, the record of its build ``_build_config``.

``fsla`` is pure Python and loads this package when a kernel is first called (``fsla.ops._ext``); nothing else imports
it. It installs as ``pip install "fsla[cu13]"``, at exactly ``fsla``'s version. A checkout's in-place build lays out the
same package, so an installed wheel and a checkout load the binary by one path (docs/build.md#wheels).
"""
