# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE DEVICE PROBE'S CANDIDATE WRITER (`measure/lab/pages/device_probe.py`, docs/internals/hardware/registry.md#probe): a record of the probe becomes a
candidate part naming each value's row, a known H100 SXM, an H100 PCIe and an unknown model each get their own id, a record with a failed check or a missing row is refused,
and the registry refuses the candidate file until it is renamed to a part. The records are SYNTHETIC -- the shape the lab writes, the figures a card's published ones --
so what is tested is the writer, never a measurement."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "measure/lab/pages"))

from hardware import registry  # noqa: E402

RATES = "measure/rates.json"
H100_SXM = dict(device="NVIDIA H100 80GB HBM3", cc=900, regs=65536, smem_sm=233472, optin=232448, threads=2048, sms=132, l2=52428800)
H100_PCIE = dict(H100_SXM, device="NVIDIA H100 PCIe", sms=114)
UNKNOWN = dict(device="NVIDIA GeForce RTX 3090", cc=860, regs=65536, smem_sm=102400, optin=101376, threads=1536, sms=82, l2=6291456)


def generator():
    spec = importlib.util.spec_from_file_location("device_probe_page", ROOT / "measure/lab/pages/device_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(card, checks=(True,)):
    rows = {"cc": card["cc"], "regs_per_sm": card["regs"], "smem_per_sm": card["smem_sm"], "smem_per_cta_optin": card["optin"],
            "max_threads_per_sm": card["threads"], "sm_count": card["sms"], "l2_bytes": card["l2"], "pci_device_id_slot": 7,
            "nsmid_register": card["sms"], "total_smem_size_register": card["optin"]}
    return {"device": card["device"], "driver": "580.0", "date": "2026-01-01T00:00:00", "reps": 7, "git": {"sha": "0123456789abcdef"},
            "results": [{"name": n, "unit": "u", "median": v, "min": v, "max": v} for n, v in rows.items()],
            "checks": [{"lab": "check", "variant": f"c{i}", "ok": ok} for i, ok in enumerate(checks)]}


def tree(tmp_path, card):
    """A root holding what `write` reads (the rates, the record, the page) and the registry to refuse into."""
    (tmp_path / "measure").mkdir(parents=True)
    shutil.copy(ROOT / RATES, tmp_path / RATES)
    d = tmp_path / "lab/device_probe"
    d.mkdir(parents=True)
    (d / "sm_90a.json").write_text(json.dumps(record(card)))
    (tmp_path / "docs/lab").mkdir(parents=True)
    shutil.copy(ROOT / "docs/lab/device_probe.md", tmp_path / "docs/lab/device_probe.md")
    shutil.copytree(registry.ROOT, tmp_path / "hardware", ignore=shutil.ignore_patterns("__pycache__", "*.py", "BUILD.bazel"))
    return tmp_path


@pytest.mark.parametrize("card, ident", [
    (H100_SXM, "cc900_nvidia_h100_80gb_hbm3"),
    (H100_PCIE, "cc900_nvidia_h100_pcie"),
    (UNKNOWN, "cc860_nvidia_geforce_rtx_3090"),
])
def test_each_model_gets_its_own_candidate_id_and_every_device_fact_with_its_row(card, ident):
    got, part = generator().candidate(ROOT, record(card))
    assert got == ident and part["name"] == f"device/{ident}" and part["model"] == {"cc": card["cc"], "name": card["device"]}
    #: the schema's device facts today, with the two the probe reads for the schema's coming entries (`sm_count`, `l2_bytes`)
    device_facts = {n for n, f in registry.schema().items() if f.read == "DEVICE"} | {"sm_count", "l2_bytes"}
    expected = {"cc": card["cc"], "regs_per_sm": card["regs"], "smem_per_sm": card["smem_sm"], "smem_per_cta_optin": card["optin"],
                "max_threads_per_sm": card["threads"], "sm_count": card["sms"], "l2_bytes": card["l2"]}
    for fact, value in expected.items():
        assert part["facts"][fact]["value"] == value
        assert fact in part["facts"][fact]["measured_by"] and "lab/device_probe/sm_90a.json" in part["facts"][fact]["measured_by"]
    assert set(part["facts"]) | set(part["unestablished"]) == device_facts
    for fact, record_ in part["facts"].items():
        assert sum(k in record_ for k in registry.PROVENANCE) == 1


def test_a_known_model_takes_its_sets_from_the_rates_and_the_registry_and_an_unknown_one_names_them_unestablished():
    _, known = generator().candidate(ROOT, record(H100_SXM))
    hopper = registry.part("device/hopper_sm90a")["facts"]
    assert known["unestablished"] == []
    for fact in ("tensor", "instructions"):
        assert sorted(known["facts"][fact]["value"]["set"]) == sorted(hopper[fact]["value"]["set"])
    assert "documented_by" in known["facts"]["tensor"]
    _, unknown = generator().candidate(ROOT, record(UNKNOWN))
    assert unknown["unestablished"] == ["tensor", "instructions"] and "tensor" not in unknown["facts"] and "instructions" not in unknown["facts"]


def test_the_models_sharing_a_compute_capability_differ_in_sm_count_and_id():
    (a, pa), (b, pb) = (generator().candidate(ROOT, record(c)) for c in (H100_SXM, H100_PCIE))
    assert a != b and pa["facts"]["sm_count"]["value"] == 132 and pb["facts"]["sm_count"]["value"] == 114


def test_a_failed_check_and_an_absent_row_are_refused():
    with pytest.raises(AssertionError, match="a check that failed"):
        generator().candidate(ROOT, record(H100_SXM, checks=(True, False)))
    bad = record(H100_SXM)
    bad["results"] = [r for r in bad["results"] if r["name"] != "l2_bytes"]
    with pytest.raises(SystemExit, match="l2_bytes"):
        generator().candidate(ROOT, bad)


def test_write_puts_the_candidate_beside_the_parts_and_the_registry_refuses_it_until_renamed(tmp_path):
    root = tree(tmp_path, UNKNOWN)
    written = generator().write(root, root, "nb123")
    candidate = root / "hardware/parts/device/cc860_nvidia_geforce_rtx_3090.candidate.json"
    assert candidate in written and json.loads(candidate.read_text())["facts"]["sm_count"]["value"] == 82
    assert "MEASURED (" in (root / "docs/lab/device_probe.md").read_text() and "cc860_nvidia_geforce_rtx_3090" in (root / "docs/lab/device_probe.md").read_text()
    for call in (registry.check, lambda hardware: registry.contexts("decode", hardware)):
        with pytest.raises(registry.RegistryError, match="CANDIDATE part"):
            call(root / "hardware")
    candidate.rename(candidate.with_name("cc860_nvidia_geforce_rtx_3090.json"))
    #: renamed, it is a part: its name, its facts and its provenance are the registry's to check (the hopper descriptor still lacks the probe's facts, which is its own refusal)
    part = registry.part("device/cc860_nvidia_geforce_rtx_3090", root / "hardware")
    assert part["facts"]["l2_bytes"]["value"] == 6291456
    assert "device/cc860_nvidia_geforce_rtx_3090" in registry.parts(root / "hardware")
