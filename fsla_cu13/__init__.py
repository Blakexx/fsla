# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""fsla's binary plugin for the cu13 toolchain: the extension ``_C``, the record of its build ``_build_config``.

``fsla`` is pure Python and loads this package when a kernel is first called (``fsla.ops._ext``); nothing else imports
it. It installs as ``pip install "fsla[cu13]"``, at exactly ``fsla``'s version. A checkout's in-place build lays out the
same package, so an installed wheel and a checkout load the binary by one path (docs/build.md#wheels).
"""

from pkgutil import extend_path

#: the package is one directory of the source tree (this file) and one of the build's output (`_C.abi3.so`, the record
#: `_build_config.py`): the output is never written into the tree, so the wheel's single directory and a checkout's two
#: are the one package
__path__ = extend_path(__path__, __name__)
