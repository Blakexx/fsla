# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The measured section of `docs/lab/device_probe.md` and THE CANDIDATE DEVICE PART, generated from `measure/lab/results/device_probe/sm_90a.json`.

`write(root, instance)` reads that record under `root`, writes `hardware/parts/device/<id>.candidate.json` (`candidate`: every device fact the
schema declares, each with its value and the record row that measured it) and rewrites `docs/lab/device_probe.md` under `root` from its `MEASURED (`
line (the first run: from its `NOT YET MEASURED` line) through its `Record:` line. `instance` is the rented box's label (`nb...` Nebius, `hs...`
Hyperstack, digits vast.ai), which only words the card's line. It returns the files it wrote (`lab page device_probe --instance LABEL`,
docs/internals/tools/lab.md). A reviewer reads the candidate and renames it to a part (docs/internals/hardware/registry.md#probe).
"""

from __future__ import annotations

import json
import pathlib
import re

from _record import Record, card

RECORD = "measure/lab/results/device_probe/sm_90a.json"
#: the schema's device facts the probe reads, each the record's row of the same name
READ = ("cc", "regs_per_sm", "smem_per_sm", "smem_per_cta_optin", "max_threads_per_sm", "sm_count", "l2_bytes")
#: the facts only an experiment or a document establishes, which the probe cannot read: taken from the architecture table's row of the same compute capability
TABLE = "csrc/fsla/src/common/hardware.json"
ESTABLISHED = ("tensor", "instructions")


def model_id(cc: int, name: str) -> str:
    """The candidate's id: the compute capability and the driver's name, lowercased to words (`cc900_nvidia_h100_80gb_hbm3`). The PCI slot the driver
    reports is the machine's, not the model's, so it is no part of the key."""
    words = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"cc{cc}_{words}"


def arch_row(root: pathlib.Path, cc: int) -> tuple[str, dict] | None:
    """The architecture table's row whose compute capability is `cc` (`sm_90a` is 900), or None where the table has none."""
    table = json.loads((root / TABLE).read_text())
    for arch, row in table["arch"].items():
        if 10 * int(re.match(r"sm_(\d+)", arch).group(1)) == cc:
            return arch, row
    return None


def candidate(root: pathlib.Path, record: dict) -> tuple[str, dict]:
    """The record's candidate part: `(id, part)`. Each device fact the probe read is `{"value", "measured_by"}` naming its row of the record; `tensor` and
    `instructions` come from the table's row of the device's compute capability (`documented_by`, never re-measured here) and are left out,
    named in `unestablished`, where the table has no such row. A record with a failed check is refused."""
    assert all(c["ok"] for c in record["checks"]), "lab page: the record holds a check that failed"
    rows = Record({r["name"]: r["median"] for r in record["results"]})
    cc = int(rows["cc"])
    ident = model_id(cc, record["device"])
    by = f"{RECORD}: the row {{row}} (experiment device_probe on {record['device']}, driver {record['driver']}, {record['date'][:10]}, the experiment at {record['git']['sha'][:8]})"
    facts: dict = {n: {"value": int(rows[n]), "measured_by": by.format(row=n)} for n in READ}
    unestablished: list[str] = []
    found = arch_row(root, cc)
    if found is None:
        unestablished = list(ESTABLISHED)
    else:
        arch, row = found
        facts["tensor"] = {"value": {"set": sorted(row["rates"]["mma"])},
                           "documented_by": f"{TABLE} arch.{arch}.rates.mma: the tensor-core instruction kinds of the table's row for compute capability {cc}"}
        facts["instructions"] = {"value": {"set": sorted(c for c, v in row["caps"].items() if v is True)},
                                 "documented_by": f"{TABLE} arch.{arch}.caps: the instruction capabilities of the table's row for compute capability {cc}, each cap's page in docs/internals/common/arch_caps.md"}
    part = {
        "name": f"device/{ident}",
        "doc": f"CANDIDATE written by `//measure/lab:device_probe` for {record['device']}: a reviewer reads it, completes what it names in `unestablished`, and renames the file to "
               f"`{ident}.json` to admit it (docs/internals/hardware/registry.md#probe)",
        "model": {"cc": cc, "name": record["device"]},
        "facts": facts,
        "unestablished": unestablished,
    }
    return ident, part


def write(root: pathlib.Path, instance: str) -> list[pathlib.Path]:
    d = json.loads((root / RECORD).read_text())
    ident, part = candidate(root, d)
    out = root / "hardware/parts/device" / f"{ident}.candidate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(part, indent=2) + "\n")

    R = Record({r["name"]: r["median"] for r in d["results"]})
    rows = "\n".join(f"| `{n}` | {int(R[n])} |" for n in (*READ, "nsmid_register", "total_smem_size_register", "pci_device_id_slot"))
    text = f"""MEASURED ({card(d["device"], instance)}, driver {d["driver"]}; {d["date"][:10]}, the experiment at
{d["git"]["sha"][:8]}; {d["reps"]} repetitions; every check passed: the properties and the attributes agree, the registers were read, the SM count is within
the register's identifiers): WHAT THE DEVICE REPORTS, each a row of the record:

| the row | the value |
|---|---|
{rows}

WHAT IT DECIDES.
- THE CANDIDATE PART `hardware/parts/device/{ident}.candidate.json` states each fact the schema reads from the device with the row that measured it; `tensor` and `instructions`
  come from the architecture table's row of compute capability {int(R["cc"])}{"" if part["unestablished"] == [] else " -- which has none, so they are named in `unestablished` for the reviewer"}.
- THE MODEL KEY is the compute capability and the driver's name; the PCI slot the driver reports (`pci_device_id_slot`) is the machine's, not the model's.

Record: `{RECORD}`.
"""
    p = root / "docs/lab/device_probe.md"
    s = p.read_text()
    i = s.index("MEASURED (") if "MEASURED (" in s else s.index("NOT YET MEASURED")
    rec = f"Record: `{RECORD}`.\n"
    j = s.index(rec, i) + len(rec) if rec in s[i:] else len(s)
    p.write_text(s[:i] + text + s[j:])
    return [out, p]
