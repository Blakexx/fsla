"""UNIT BENCH: the reverse pass' kernels -- the two shipped scans and the ladder's rung 0.

ONE DRIVER FOR THE FOUR, and that is the point rather than a shortcut: the ladder's
verdict is a COMPARISON (`rung0_pass1` against `scan_a`, `rung0_pass2` against `scan_b`
on one cell), and `--ab` may only pair two specs that are both on the driver's roster.
Four drivers would make the one measurement the ladder exists to take unexpressible.

Invoked WRAPPED (this harness checks that it was):

    flock /tmp/rola_gpu.lock python benchmarks/unit/bench_backward.py --tier landing
    flock /tmp/rola_gpu.lock python benchmarks/unit/bench_backward.py \\
        --ab rung0_pass1@chunk-bwd-dense-N64:scan_a@chunk-bwd-dense-N64 --tier landing
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bench.driver import main as drive

    return drive({"scan_a", "scan_b", "rung0_pass1", "rung0_pass2"}, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
