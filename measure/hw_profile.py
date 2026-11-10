"""A DEVICE NAME'S PRODUCT SLUG, never a hostname: `NVIDIA GeForce RTX 3080 Ti` is `rtx3080ti`, `NVIDIA A100-SXM4-80GB`
`a100` -- the memory-config suffix after the first `-` is a fact about the SKU, not about which kernel binary applies.
"""
from __future__ import annotations

import re

#: Vendor words carry no comparability information; stripping them is what turns
#: `torch.cuda.get_device_properties(0).name` into a slug instead of a sentence.
_VENDOR_WORDS = ("NVIDIA", "GeForce", "Tesla", "Quadro")


def slugify_gpu(name: str) -> str:
    """`NVIDIA GeForce RTX 3080 Ti` -> `rtx3080ti`; `NVIDIA A100-SXM4-80GB` -> `a100`.

    The cut at the first `-` drops a memory/interconnect suffix (`SXM4-80GB`,
    `PCIE-40GB`) that does not change which kernel binary applies -- an A100-40GB and
    an A100-80GB run the identical SASS.
    """
    stripped = name
    for word in _VENDOR_WORDS:
        stripped = stripped.replace(word, "")
    stripped = stripped.split("-", 1)[0]
    return re.sub(r"[^A-Za-z0-9]", "", stripped).lower()
