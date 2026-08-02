"""The README's GENERATED tables are gated, so "generated" is a fact and not a label.

Design §7.2: *a hand-maintained support table is a claim; a generated one is a
report.* That distinction only survives if something fails when the table is edited
by hand, or when the artifacts it is derived from move and the table does not. This
is that something, and it runs in the CPU job — it reads files, not a device.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SUPPORTED = ROOT / "tools" / "supported.py"


def _check() -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SUPPORTED), "--check"],
                          capture_output=True, text=True, cwd=str(ROOT))


def test_the_readmes_generated_tables_have_not_drifted():
    proc = _check()
    assert proc.returncode == 0, (
        f"{proc.stdout}{proc.stderr}\n"
        "Regenerate with `python tools/supported.py --write`.")


def test_the_supported_table_lists_every_tabulated_arch_and_marks_the_unratified():
    """The unratified rows are the PRODUCT STATEMENT, not an omission.

    An architecture the kernel can size itself for but nobody measured must appear,
    and must say it will refuse to load. A table that silently dropped those rows
    would imply coverage that was never measured, which is the exact failure the
    first README claim is about.
    """
    sys.path.insert(0, str(ROOT / "tools"))
    import supported

    table = supported.table()
    ratified = supported.ratified()
    for arch in supported.tabulated_archs():
        assert f"`{arch}`" in table, f"{arch} is tabulated in the kernel but absent from the table"
        row = next(line for line in table.splitlines() if f"`{arch}`" in line)
        if arch in ratified:
            assert "ratified |" in row and "not ratified" not in row
        else:
            assert "**not ratified — will refuse to load**" in row, (
                f"{arch} has no manifest but the table does not say so: {row}")


def test_the_drift_check_can_fail(tmp_path, monkeypatch):
    """NON-VACUITY. A drift check that passes on a hand-edited table is decoration."""
    sys.path.insert(0, str(ROOT / "tools"))
    import supported

    tampered = tmp_path / "README.md"
    text = supported.README.read_text().replace(
        "**not ratified — will refuse to load**", "ratified", 1)
    assert text != supported.README.read_text(), (
        "the tamper had no effect; this test is not exercising the check")
    tampered.write_text(text)
    monkeypatch.setattr(supported, "README", tampered)
    monkeypatch.setattr(sys, "argv", ["supported.py", "--check"])
    with pytest.raises(SystemExit) as exc:
        raise SystemExit(supported.main())
    assert exc.value.code == 1
