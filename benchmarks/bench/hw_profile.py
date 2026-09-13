"""The results key: a HARDWARE PROFILE, never a hostname.

`benchmarks/results/<hw-profile>/<yyyy-mm>/<run_id>.jsonl` is keyed on the thing that
makes two runs COMPARABLE -- the GPU model and its SM architecture -- not on the box
that happened to run them. `NVIDIA GeForce RTX 3080 Ti` + `sm_86` becomes
`rtx3080ti-sm86`; `NVIDIA A100-SXM4-80GB` + `sm_80` becomes `a100-sm80`, because the
memory-config suffix after the first `-` is a fact about the SKU, not about which
run this is. Two boxes that share a profile and are ever told apart use an opaque
salted hash (`machine_tag`), never the hostname itself -- see `machine_tag` below.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

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


def hw_profile(gpu_name: str, sm: str) -> str:
    """The results-tree key: `<gpu-slug>-<sm>`, e.g. `rtx3080ti-sm86`."""
    return f"{slugify_gpu(gpu_name)}-{sm.replace('_', '')}"


def machine_tag() -> str | None:
    """An OPAQUE, salted short hash distinguishing two same-profile machines.

    Per the RULED section: nothing identifiable is checked in, ever -- not a
    hostname, not a username, not a machine id. This function exists only for the
    day two boxes of one profile need telling apart; it hashes the hostname together
    with an operator-supplied salt (never committed, never derived from the box
    alone) and returns `None` when no salt is configured, which is the default and
    the common case. A hash of the hostname WITHOUT a secret salt would be
    reversible by any reader with a GPU-model list and is not used.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    import dev_config

    salt = dev_config.get("environment.machine_salt")
    if not salt:
        return None
    import socket

    material = f"{salt}:{socket.gethostname()}".encode()
    return hashlib.sha256(material).hexdigest()[:12]
