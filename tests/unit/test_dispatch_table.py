# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""THE CENSUS: the dispatch table and the binary carry the same arms.

`tools/gen_shards.py` emits the shards, the row table and the key index from ONE
matrix definition, so they cannot disagree with each other. What that argument does not
cover is whether the BINARY agrees: a shard that failed to instantiate, an arm dropped
by a linker, a row pointing at a launcher that is not there. `tools/fatbin_entries.py`
reads the built `.so` and answers that from the device side; this file joins the
RUNTIME table against it, so the set the dispatch can reach, the set the generator
wrote, and the set the fatbin carries are checked to be one set.

The join is on the template tuple, decoded from each device entry's MANGLED NAME --
what the assembler saw, which cannot disagree with the binary the way a recorded list
can.

Also here: the STALENESS FILTER's teeth (a row whose shard membership is not the
ratified one must be dropped, not used), and the AGREEMENT between the table and the
`SUPPORTED_*` tuples the planner still states in Python -- which is the drift this
card exists to kill.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cuda

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))


def _table():
    from rola.planner.dispatch_table import DispatchTable
    return DispatchTable.from_extension()


def _built_so() -> Path:
    hits = sorted(ROOT.glob("rola/_C*.so"))
    if not hits:
        pytest.skip("no built extension to census")
    return hits[0]


# ---------------------------------------------------------------------------
# 1. THE CENSUS
# ---------------------------------------------------------------------------


def test_the_table_and_the_binary_carry_exactly_the_same_arms():
    """Table rows == device entry points, joined on the template tuple.

    Both directions matter and both are asserted: an entry the table cannot reach is an
    arm that ships and cannot be selected (dead payload), and a row with no entry is a
    dispatch that resolves to a launcher the binary does not define (an ImportError
    waiting for the first launch, since a Python extension leaves the symbol to the
    loader).
    """
    import fatbin_entries
    import gen_shards

    so = _built_so()
    if subprocess.run(["which", "cuobjdump"], capture_output=True).returncode != 0:
        try:
            fatbin_entries.ratify.cuda_bin("cuobjdump")
        except SystemExit:
            pytest.skip("cuobjdump is not available to read the fatbin")

    entries = fatbin_entries.device_entries(so)
    per_arch: dict[str, set[str]] = {}
    for arch, _source, mangled in entries:
        key = fatbin_entries.shape_key(mangled)
        if key is not None:
            per_arch.setdefault(arch, set()).add(key)
    assert per_arch, "the fatbin reports no consumer entry points at all"

    table = _table()
    from_table = {gen_shards.canonical(
        dict(BT=r.BT, BC=r.BC, DV=r.DV, D=r.D, PREC=r.prec, DECAY=r.decay,
             GLOBAL=r.global_norm)) for r in table.rows}

    for arch, in_binary in sorted(per_arch.items()):
        assert from_table == in_binary, (
            f"sm_{arch}: the table and the binary disagree.\n"
            f"  in the table, not the binary: {sorted(from_table - in_binary)[:4]}\n"
            f"  in the binary, not the table: {sorted(in_binary - from_table)[:4]}")


def test_every_table_row_names_the_shard_that_defines_it():
    """`row.shard` is the membership function's answer, not a label.

    The shard a row claims must be the shard `gen_shards.shard_of` assigns its tuple --
    otherwise the staleness key joins against the wrong record and a stale row could
    survive by naming a shard that did not move.
    """
    import gen_shards

    for row in _table().rows:
        want = gen_shards.shard_of(dict(BT=row.BT, BC=row.BC, DV=row.DV, D=row.D,
                                        PREC=row.prec, DECAY=row.decay,
                                        GLOBAL=row.global_norm))
        assert row.shard == want, f"{row.key} claims shard {row.shard}, membership says {want}"


def test_the_table_is_the_whole_matrix_and_nothing_else():
    """Row count and axis values are the generator's matrix, read back off the binary."""
    import gen_shards

    table = _table()
    assert len(table.rows) == len(gen_shards.matrix())
    assert len({r.key for r in table.rows}) == len(table.rows), "a tuple appears twice"
    for axis, name in (("BT", "BT"), ("BC", "BC"), ("DV", "DV"), ("D", "D")):
        assert table.values(axis) == tuple(sorted(gen_shards.AXIS[name].values))
    assert set(table.values("prec")) == set(gen_shards.AXIS["PREC"].values)


# ---------------------------------------------------------------------------
# 2. THE STALENESS FILTER
# ---------------------------------------------------------------------------


def test_a_row_whose_shard_membership_moved_is_filtered_not_used():
    """The Inductor `src_hash` discipline, with teeth.

    The ratified key set is perturbed for ONE shard; every row of that shard must leave
    the table with a reason, every other row must survive, and the filtered rows must
    not be reachable through `resolve`.
    """
    from rola.ops._ext import extension
    from rola.planner.dispatch_table import DispatchTable

    live = _table()
    ratified = {shard: next(r.src_key for r in live.rows if r.shard == shard)
                for shard in sorted({r.shard for r in live.rows})}
    victim = sorted(ratified)[0]
    perturbed = dict(ratified, **{victim: "0" * 64})

    table = DispatchTable.from_extension(ext=extension(), ratified_keys=perturbed)
    assert {f.row.shard for f in table.filtered} == {victim}
    assert len(table.filtered) == sum(1 for r in live.rows if r.shard == victim)
    assert all(r.shard != victim for r in table.rows)
    assert all(victim in f.reason for f in table.filtered)

    gone = next(r for r in live.rows if r.shard == victim)
    with pytest.raises(KeyError):
        table.resolve(gone.BT, gone.BC, gone.DV, gone.D, gone.prec, gone.decay,
                      gone.global_norm)


def test_an_unrecorded_shard_is_filtered_with_its_own_reason():
    """A shard the record does not mention at all is not silently trusted."""
    from rola.ops._ext import extension
    from rola.planner.dispatch_table import DispatchTable

    live = _table()
    keep = sorted({r.shard for r in live.rows})[:1]
    ratified = {s: next(r.src_key for r in live.rows if r.shard == s) for s in keep}
    table = DispatchTable.from_extension(ext=extension(), ratified_keys=ratified)
    assert {r.shard for r in table.rows} == set(keep)
    assert all("no ratified record" in f.reason for f in table.filtered)


def test_an_all_stale_table_is_an_error_rather_than_an_empty_candidate_set():
    """Everything filtered is a broken install, not a plan with no candidates."""
    from rola.ops._ext import extension
    from rola.planner.dispatch_table import DispatchTable

    with pytest.raises(ValueError, match="empty after the staleness join"):
        DispatchTable.from_extension(ext=extension(), ratified_keys={})


def test_the_shipped_build_config_records_the_ratified_shard_keys():
    """The join must be makeable from an installed wheel, which has no manifests."""
    from rola._build_config import BUILD_CONFIG

    shards = BUILD_CONFIG.get("shards")
    assert shards, "_build_config.py carries no shard membership to join against"
    table = _table()
    assert set(shards) == {r.shard for r in table.rows}
    for row in table.rows:
        assert shards[row.shard] == row.src_key


# ---------------------------------------------------------------------------
# 3. THE ENABLE PREDICATE, AND THE PYTHON RESTATEMENTS IT REPLACES
# ---------------------------------------------------------------------------


def test_every_shipped_arm_is_admitted_by_this_device():
    """The predicate keys on a device fact, and on this device it admits everything.

    Asserted rather than assumed because it is what makes the refusal path unreachable
    HERE while remaining live: a machine with a smaller opt-in maximum gets a named
    refusal instead of a bare CUDA error.
    """
    from rola.ops._ext import extension

    optin = extension().rola_max_dynamic_smem_bytes()
    assert optin > 0
    table = _table()
    assert len(table.selectable(optin)) == len(table.rows)
    #: NON-VACUITY: below the largest footprint the predicate must exclude something.
    largest = max(r.smem_bytes for r in table.rows)
    assert len(table.selectable(largest - 1)) < len(table.rows)


def test_the_planners_supported_sets_are_the_tables_own_axes():
    """The four Python restatements of the instantiation set, pinned to the table.

    `schedule._SUPPORTED_BT/_BC/_DV` and `select.SUPPORTED_BT/BC` are declared in Python
    so that CPU-only planning works with no extension present. That makes them a MIRROR,
    and a mirror needs a gate: this one. The table is the authority.
    """
    from rola.planner import schedule, select

    table = _table()
    assert table.values("BT") == schedule._SUPPORTED_BT
    assert table.values("BC") == schedule._SUPPORTED_BC
    assert table.values("DV") == schedule._SUPPORTED_DV
    assert table.values("BT") == select.SUPPORTED_BT
    assert table.values("BC") == select.SUPPORTED_BC
