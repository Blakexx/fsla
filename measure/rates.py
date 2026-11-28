# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE MEASURED AND DOCUMENTED RATES, READ FROM THEIR ONE FILE: `measure/rates.json` holds every architecture's rates and every product's overrides,
each rate with its source, and the rate-name schema. This module resolves a product's rates (its overrides over its architecture's defaults, for run
time) and checks that every source holds -- a measured rate matches the lab record it names, on the product it names -- and that every architecture
named there is one the registry builds (`hardware/`: the device facts and the architectures live there once). No `fsla` and no GPU (the lab's
records through `measure/data.py`, the results store's reader), so the lint runs it. docs/internals/common/hardware.md.

    python -m measure.rates [--product h10080gbhbm3 | --arch sm_90a]   # the resolved rates
    python -m measure.rates --check                                   # the gate
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]

from hardware import registry  # noqa: E402
from measure import data as pinned  # noqa: E402

CONFIG = ROOT / "measure" / "rates.json"
#: the rates a built architecture must resolve: its pipes' (the schedulers, the issue, a shared wavefront and the
#: shared atomics), which the phase floors (`tools/overlap_gate.py`), the pipe model and the budgets' forms read
PIPE_RATES = ("schedulers", "issue", "wavefront", "atomic", "atomic_wavefront")
#: the streams' rates a card may state, in bytes a cycle an SM (`bytes_a_cycle`, not `cycles`): HBM's read, write and copy
#: (`docs/lab/stream_rate.md`); optional, and no tool but this check reads them yet
STREAM_RATES = ("stream.hbm_read", "stream.hbm_write", "stream.hbm_copy")
#: an entry's figure: cycles an operation, or a stream's bytes a cycle an SM
UNITS = ("cycles", "bytes_a_cycle")
SOURCES = ("measured", "documented", "announced")
#: a measured rate may differ from its record's median by its own rounding and no more
TOLERANCE = 0.002

#: Vendor words carry no comparability information; stripping them is what turns
#: `torch.cuda.get_device_properties(0).name` into a slug instead of a sentence.
_VENDOR_WORDS = ("NVIDIA", "GeForce", "Tesla", "Quadro")


def slugify_gpu(name: str) -> str:
    """`NVIDIA H100 80GB HBM3` -> `h10080gbhbm3`.

    The cut at the first `-` drops a dashed suffix that does not change
    which kernel binary applies.
    """
    stripped = name
    for word in _VENDOR_WORDS:
        stripped = stripped.replace(word, "")
    stripped = stripped.split("-", 1)[0]
    return re.sub(r"[^A-Za-z0-9]", "", stripped).lower()


def load() -> dict:
    return json.loads(CONFIG.read_text())


@dataclass(frozen=True)
class Rate:
    #: the figure, in `unit`: cycles an operation, or a stream's bytes a cycle an SM
    value: float
    kind: str
    where: str
    row: str
    on: str
    unit: str = "cycles"


def _flatten(rates: dict, prefix: str = "") -> dict[str, dict]:
    out = {}
    for key, value in rates.items():
        name = f"{prefix}{key}"
        if any(u in value for u in UNITS):
            out[name] = value
        else:
            out.update(_flatten(value, f"{name}."))
    return out


def _rate(entry: dict, on: str) -> Rate:
    kind = next(k for k in SOURCES if k in entry)
    unit = next(u for u in UNITS if u in entry)
    return Rate(float(entry[unit]), kind, entry[kind], entry.get("row", ""), entry.get("on", on), unit)


def resolve(arch: str | None = None, product: str | None = None, cfg: dict | None = None) -> dict[str, Rate]:
    """A product's rates (its architecture's defaults under its own overrides), or an architecture's defaults alone:
    `{field: Rate}` with fields `issue`, `mma.hmma_f16_f32`, ..."""
    cfg = cfg or load()
    if product is not None:
        arch = cfg["product"][product]["arch"]
    rates = {k: _rate(v, "") for k, v in _flatten(cfg["arch"][arch].get("rates", {})).items()}
    if product is not None:
        rates.update({k: _rate(v, product) for k, v in _flatten(cfg["product"][product].get("rates", {})).items()})
    return rates


def product_of(device: str) -> str | None:
    """The product row for a device name (`torch.cuda.get_device_properties(...).name`), if the configuration has one."""
    slug = slugify_gpu(device)
    return slug if slug in load()["product"] else None


def _source_findings(where: str, arch: str, rate: Rate) -> list[str]:
    if rate.kind != "measured":
        return [] if rate.where else [f"{where}: a {rate.kind} rate names no source"]
    if not rate.on:
        return [f"{where}: a measured rate names no product it was measured on (`on`)"]
    try:
        record = pinned.lab_record(rate.where, arch)
    except SystemExit as missing:
        return [f"{where}: {missing}"]
    data, cited = record.body(), record.cited
    medians = {r["name"]: r["median"] for r in data["results"]}
    findings = []
    if slugify_gpu(data.get("device", "")) != rate.on:
        findings.append(f"{where}: {cited} was recorded on `{data.get('device')}`, not `{rate.on}`")
    if rate.row not in medians:
        findings.append(f"{where}: {cited} has no result `{rate.row}`")
    elif abs(rate.value - medians[rate.row]) > TOLERANCE * max(1.0, medians[rate.row]):
        findings.append(f"{where}: {rate.value} in {CONFIG.name}, {medians[rate.row]:.4f} in {cited}")
    return findings


def check(cfg: dict | None = None) -> list[str]:
    """Every architecture named here one the registry builds, every built architecture priced for its pipes and MMAs, every product on an architecture
    named here, every rate a known field with one source that holds."""
    cfg = cfg or load()
    findings = []
    kinds = cfg["schema"]["mma"]
    known = {*PIPE_RATES, *STREAM_RATES, *(f"mma.{k}" for k in kinds)}
    built = registry.built()
    for arch in cfg["arch"]:
        if arch not in built:
            findings.append(f"{arch}: named here but no device part of the registry states it (built: {', '.join(built)})")
    for arch in built:
        if arch not in cfg["arch"]:
            findings.append(f"{arch} is built but rates.json names it not")
            continue
        rates = resolve(arch, cfg=cfg)
        missing = [f for f in (*PIPE_RATES, *(f"mma.{k}" for k in kinds)) if f not in rates]
        if missing:
            findings.append(f"{arch} is built but its defaults lack {', '.join(missing)}")
    for product, row in cfg["product"].items():
        if row["arch"] not in cfg["arch"]:
            findings.append(f"product {product} names {row['arch']}, which has no row")
            continue
        if slugify_gpu(row.get("device", "")) != product:
            findings.append(f"product {product}: its device `{row.get('device')}` is not that product")

    scopes = [(arch, f"{arch}", row.get("rates", {}), "") for arch, row in cfg["arch"].items()]
    scopes += [(row["arch"], f"product {p}", row.get("rates", {}), p) for p, row in cfg["product"].items()]
    for arch, scope, rates, on in scopes:
        for field, entry in _flatten(rates).items():
            where = f"{scope} {field}"
            if field not in known:
                findings.append(f"{where}: not a rate the schema names")
            if sum(k in entry for k in SOURCES) != 1:
                findings.append(f"{where}: needs exactly one of {', '.join(SOURCES)}")
                continue
            findings += _source_findings(where, arch, _rate(entry, on))
    return findings


def main() -> int:
    args = sys.argv[1:]
    if "--check" in args:
        findings = check()
        for f in findings:
            print(f)
        print(f"rates: {len(findings)} finding(s)")
        return 1 if findings else 0
    cfg = load()
    product = args[args.index("--product") + 1] if "--product" in args else None
    arch = args[args.index("--arch") + 1] if "--arch" in args else None
    targets = ([(None, product)] if product else [(arch, None)] if arch
               else [(None, p) for p in cfg["product"]] + [(a, None) for a in cfg["arch"]])
    for a, p in targets:
        rates = resolve(a, p, cfg)
        print(p or a)
        for field, r in rates.items():
            source = r.where + (f" / {r.row}" if r.row else "") + (f" (on {r.on})" if r.on else "")
            print(f"  {field:28s} {r.value:8.3f} {r.unit:13s} {r.kind:10s} {source}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
