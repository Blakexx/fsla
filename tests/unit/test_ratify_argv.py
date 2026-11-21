"""THE RATIFICATION HAS NO COMPILE OF ITS OWN.

`tools/ratify.py` reads the units the graph's CUDA rule compiled (docs/internals/tools/ratify.md#graph); its nvcc
command, its per-unit cache and its `--arch`/`--shard`/`--arm` scopes are gone (docs/internals/DELETIONS.md), and a
tool that can force a launch bound is a tool that can ratify one.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import ratify  # noqa: E402


def test_ratify_runs_no_compiler_and_keeps_no_unit_cache():
    src = (ROOT / "tools" / "ratify.py").read_text()
    for gone in ("_nvcc_command", "measure_units", "_cache_read", "_compile_dir", '"-gencode"', "\"-arch=sm_90a\"",
                 "tempfile", '"--arch"', '"--shard"', '"--arm"'):
        assert gone not in src, gone
    assert not hasattr(ratify, "measure_units") and not hasattr(ratify, "_nvcc_command")


def test_the_forcing_census_is_gone():
    """`--force-ctas` and its `ROLA_FORCE_CTAS` define were deleted (P65 S2,
    `docs/internals/DELETIONS.md`) once nothing in `csrc/` read the macro. A
    tool that can force a launch bound is a tool that can ratify one; the
    measurement lives in `docs/measurement.md` as a scratch-tree edit."""
    assert "force_ctas" not in ratify.__dict__
    src = (ROOT / "tools" / "ratify.py").read_text()
    assert "ROLA_FORCE_CTAS" not in src and "--force-ctas" not in src
    for path in (ROOT / "csrc").rglob("*"):
        if path.is_file() and path.suffix in (".cu", ".cuh", ".cpp", ".h"):
            assert "ROLA_FORCE_CTAS" not in path.read_text(errors="ignore"), path
