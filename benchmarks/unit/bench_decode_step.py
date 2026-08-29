"""UNIT BENCH: one carried single-token decode step.

Invoked BARE (K46: `bench.discipline.disciplined` takes `gpu_lock()` itself):

    python benchmarks/unit/bench_decode_step.py --tier landing
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bench.driver import main as drive

    return drive({"decode_step"}, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
