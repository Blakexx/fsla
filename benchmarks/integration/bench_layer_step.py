"""INTEGRATION BENCH: the whole layer forward — projections, producer, kernel, readout, output projection.

Invoked WRAPPED (this harness checks that it was):

    flock /tmp/rola_gpu.lock python benchmarks/integration/bench_layer_step.py --tier landing
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bench.driver import main as drive

    return drive({"layer_step"}, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
