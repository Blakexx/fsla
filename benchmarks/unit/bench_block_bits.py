"""UNIT BENCH: the block bits — the block-grain liveness reduction the sparse grid launches over.

Invoked WRAPPED (this harness checks that it was):

    flock /tmp/rola_gpu.lock python benchmarks/unit/bench_block_bits.py --tier landing
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bench.driver import main as drive

    return drive({"chunk_block_bits"}, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
