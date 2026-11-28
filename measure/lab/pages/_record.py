# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The page generators' one lookup: a record's newest sample in the pinned data (measure/data.py's `lab_record`, on the lab's
architecture), and its rows by name -- a row a page names and the record lacks is refused by name."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from measure.data import LabSample, lab_record  # noqa: E402 -- the path insert above must precede it

#: the architecture every page's records are of
ARCH = "sm_90a"


def read(data: pathlib.Path, experiment: str, blocks: int = 0, variant: str | None = None, kind: str = "run") -> LabSample:
    """The newest sample of `experiment`'s record of `kind` (a timed run, or the profiler's counters) in the pinned data
    `data` (the grid `blocks`, the condition `variant`), refused by name where none is stored."""
    return lab_record(experiment, ARCH, kind=kind, blocks=blocks, variant=variant, data=data)


class Record(dict):
    def __missing__(self, name):
        raise SystemExit(
            f"lab page: the record holds no row {name!r} -- the page names a row of the experiment that has not been "
            "run and recorded"
        )


#: two rows within this share of each other (of the lesser) are said to be the same
SAME = 0.02


def against(a: float, b: float) -> str:
    """How `a` stands to `b`: `the same as` within `SAME` of the lesser, else `below` or `above`."""
    if abs(a - b) <= SAME * min(a, b):
        return "the same as"
    return "below" if a < b else "above"


def card(device: str, instance: str) -> str:
    """A record's card line: its device and the rented box's label (`nb...` Nebius, `hs...` Hyperstack, digits vast.ai)."""
    if instance.startswith("hs"):
        return f"{device}, Hyperstack spot VM {instance[2:]}"
    if instance.startswith("nb"):
        return f"{device}, Nebius preemptible VM {instance[2:]}"
    return f"{device}, vast.ai instance {instance}"
