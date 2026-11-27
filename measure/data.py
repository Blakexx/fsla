# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PINNED DATA AND THE LAB'S RECORDS: the data repository (fsla-results) at the ONE commit this tree pins
(MODULE.bazel's `environment.data`, the repository `@fsla_data`), where every record a doc cites (`data:records/...`)
and a tool reads lives. The graph hands its root to what reads it as `FSLA_DATA`: the lab's run targets (bazel/defs.bzl's
`lab_run`), and the checks and tiers that read a record (`repos`). A lab record is read through ONE function,
`lab_record`: the newest sample of its question (`lab_semantics`) that the experiment's run target wrote through the
results store (`fsla_devtools.results`) at its own label, `@fsla//measure/lab:<experiment>.run`.
docs/measurement.md#the-data.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from fsla_devtools.results import Store, key, label_path

#: the handoff that names the pinned data's root (tools/dev_config.py's HANDOFFS)
VARIABLE = "FSLA_DATA"
#: how a doc names a file of the pinned data: `data:records/<label path>/<run>/output.json`
PREFIX = "data:"
#: what a lab record is of: a timed run (`lab run`) and the profiler's counters (`lab ncu`), at the experiment's run
#: target's label; a carried ledger's dumps, at `@fsla//tools:lab` (the carried path's, which `lab_record` does not read)
KINDS = ("run", "counters", "ledger")


def root() -> Path:
    """The pinned data's root, as the graph handed it."""
    value = os.environ.get(VARIABLE)
    if not value:
        raise SystemExit(f"{VARIABLE} is unset: run this through its Bazel target, which hands it the pinned data "
                         "(@fsla_data, MODULE.bazel's `environment.data`)")
    return Path(value)


def lab_semantics(experiment: str, arch: str, *, kind: str = "run", blocks: int = 0, variant: str | None = None) -> dict:
    """WHAT A LAB RECORD IS OF, its key: the experiment, the architecture, the kind, the grid it was held to (`--blocks`)
    and a named condition (`variant`); the code it ran is each sample's provenance, so a new run is a new sample of the
    same record."""
    if kind not in KINDS:
        raise ValueError(f"a lab record is of {KINDS}, not {kind!r}")
    return {"experiment": experiment, "arch": arch, "kind": kind, **({"blocks": blocks} if blocks else {}),
            **({"variant": variant} if variant else {})}


@dataclass(frozen=True)
class LabSample:
    """One sample of a lab record: its output (the record body the lab wrote) and its parts (a traced row)."""

    store: Store
    sample: dict

    @property
    def output(self) -> Path:
        return self.store.output(self.sample)

    def body(self) -> dict:
        return json.loads(self.output.read_text())

    def part(self, name: str) -> Path:
        """A part of the sample by its name (`trace/row0.json`), or the directory of several (`trace`)."""
        return self.store.parts(self.sample) / name

    @property
    def cited(self) -> str:
        """The output as a doc cites it: `data:records/<label path>/<run>/output.json`."""
        return f"{PREFIX}records/{label_path(self.store.label).as_posix()}/{self.sample['output']}"


def lab_record(experiment: str, arch: str, *, blocks: int = 0, variant: str | None = None,
               data: Path | None = None) -> LabSample:
    """THE NEWEST SAMPLE of `experiment`'s run on `arch` (`lab_semantics`, the kind `run`) at its run target's label, in
    the pinned data or in `data` (a data repository's root: a store's checkout, a run's scratch store); refused by name
    where none is stored."""
    semantics = lab_semantics(experiment, arch, blocks=blocks, variant=variant)
    store = Store(f"@fsla//measure/lab:{experiment}.run", (data or root()) / "records")
    ok = [s for s in store.samples(key(semantics)) if s["ok"]]
    if not ok:
        raise SystemExit(f"no lab record of {json.dumps(semantics, sort_keys=True)} at {store.label} -- the "
                         f"experiment has not been run and stored (`bazel run //measure/lab:{experiment}.run`)")
    return LabSample(store, ok[0])
