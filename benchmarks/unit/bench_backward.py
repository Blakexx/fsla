"""UNIT BENCH: the reverse pass' scan kernels -- Scan A and Scan B.

ONE DRIVER FOR BOTH, because the shipped reverse pass is two independent launches:
a row containing both could be attributed to neither, and `--ab` may only pair two
specs that are both on the driver's roster.

Invoked WRAPPED (this harness checks that it was):

    flock /tmp/rola_gpu.lock python benchmarks/unit/bench_backward.py --tier landing
    flock /tmp/rola_gpu.lock python benchmarks/unit/bench_backward.py \\
        --ab scan_a@chunk-bwd-smalln:scan_b@chunk-bwd-smalln --tier landing
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bench.driver import main as drive

    return drive({"scan_a", "scan_b"}, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
