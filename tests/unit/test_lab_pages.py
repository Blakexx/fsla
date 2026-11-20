# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The lab's page generators for `gather_rate` and `gather_group` (measure/lab/pages/): each rewrites its page's
measured section from a record of the experiment's rows, and refuses, by name, an absent record, a row the record
lacks and a failed check. The records here are SYNTHETIC -- the shape the lab writes, figures chosen only to be
distinct -- so what is tested is the generator, never a measurement."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "measure/lab/pages"))


def generator(name):
    spec = importlib.util.spec_from_file_location(f"{name}_page", ROOT / "measure/lab/pages" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(name, median):
    return {"name": name, "unit": "u", "median": median, "min": median * 0.99, "max": median * 1.01}


def record(root, name, results, checked):
    d = root / "measure/lab/results" / name
    d.mkdir(parents=True)
    (d / "sm_90a.json").write_text(json.dumps({
        "device": "NVIDIA H100 80GB HBM3", "driver": "580.0", "date": "2026-01-01T00:00:00", "reps": 7,
        "git": {"sha": "0123456789abcdef"},
        "results": results,
        "checks": [{"lab": "check", "variant": v, "ok": ok} for v, ok in checked],
    }))


def page(root, name):
    shutil.copy(ROOT / "docs/lab" / f"{name}.md", root / "docs/lab" / f"{name}.md")
    return root / "docs/lab" / f"{name}.md"


def tree(tmp_path, name):
    (tmp_path / "docs/lab").mkdir(parents=True)
    return page(tmp_path, name)


def rate_results(gen):
    return [
        result(gen.row_name(t, what, rows), 1.0 + 0.37 * k)
        for k, (t, (what, rows, _)) in enumerate((t, f) for t, _ in gen.TABLES for f in gen.FORMS)
    ]


def group_results(gen):
    names = [name for name, _ in gen.FORMS]
    out = [result(n, 3000.0 + 100 * i) for i, n in enumerate(names)]
    out += [result(n + ": a token a side", 50.0 + i) for i, n in enumerate(names)]
    out += [result(n + ": an entry", 10.0 + i) for i, n in enumerate(names)]
    out += [result(f"{gen.CELL}, loads and stores, 8 lanes a row, PROBE {p}", 2000.0 + 50 * i) for i, (p, _) in enumerate(gen.PROBES)]
    out += [result(f"{gen.PHASED}: thread 0, {p}", 500.0 + 10 * i) for i, p in enumerate(gen.PHASES)]
    return out


CASES = {"gather_rate": rate_results, "gather_group": group_results}


@pytest.mark.parametrize("name", CASES)
def test_the_page_before_a_record_says_none_is_taken(name):
    text = (ROOT / "docs/lab" / f"{name}.md").read_text()
    assert "MEASURED: no record is taken at this commit" in text


@pytest.mark.parametrize("name", CASES)
def test_the_measured_section_is_written_from_a_record_and_rewritten_alike(tmp_path, name):
    gen = generator(name)
    p = tree(tmp_path, name)
    before = p.read_text()
    results = CASES[name](gen)
    record(tmp_path, name, results, [(r["name"], True) for r in results[:3]])
    assert gen.write(tmp_path, "nbabc") == [p]
    once = p.read_text()
    assert once.startswith(before[: before.index("MEASURED")])
    assert "MEASURED (NVIDIA H100 80GB HBM3, Nebius preemptible VM abc, driver 580.0; 2026-01-01" in once
    assert "no record is taken" not in once and "the experiment at\n01234567" in once
    gen.write(tmp_path, "nbabc")
    assert p.read_text() == once


@pytest.mark.parametrize("name", CASES)
def test_an_absent_record_is_refused(tmp_path, name):
    tree(tmp_path, name)
    with pytest.raises(SystemExit, match="no record"):
        generator(name).write(tmp_path, "nbabc")


@pytest.mark.parametrize("name", CASES)
def test_a_row_the_record_lacks_is_refused_by_name(tmp_path, name):
    gen = generator(name)
    tree(tmp_path, name)
    results = CASES[name](gen)
    lost = results.pop(0)["name"]
    record(tmp_path, name, results, [])
    with pytest.raises(SystemExit, match="holds no row"):
        gen.write(tmp_path, "nbabc")
    assert lost


@pytest.mark.parametrize("name", CASES)
def test_a_failed_check_is_refused(tmp_path, name):
    gen = generator(name)
    tree(tmp_path, name)
    results = CASES[name](gen)
    record(tmp_path, name, results, [(results[0]["name"], False)])
    with pytest.raises(SystemExit, match="failed checks"):
        gen.write(tmp_path, "nbabc")
