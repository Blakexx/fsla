# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""A record's rows by name, for the page generators: a row a page names and the record lacks is refused by name."""


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
