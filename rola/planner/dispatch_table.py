# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0

"""THE DISPATCH TABLE, AS THE PLANNER SEES IT.

The table is generated into the extension by ``tools/gen_shards.py`` and lives in the
binary: one row per compiled arm, carrying the arm's structural coordinates, its
``constexpr`` shared-memory footprint, and the generated translation unit that owns it
together with that unit's ``members-sha256``. This module reads those rows and answers
two questions the planner has to ask before it selects anything:

* **which arms exist** -- so the derivation's candidate set is the binary's own arm
  set rather than a tuple written down beside it;
* **which of them this machine can run** -- the per-row ENABLE PREDICATE, which keys on
  a runtime device fact (``cudaDevAttrMaxSharedMemoryPerBlockOptin``) and never on a
  build symbol.

**THE TABLE CONSUMES CHOICES; IT DOES NOT MAKE THEM.** Nothing here scores, ranks or
prefers a row. :func:`DispatchTable.resolve` takes a fully-determined choice and returns
the row or raises; what fixed the choice, and from which facts, is
:mod:`rola.planner.derive`'s business.

---- THE STALENESS KEY, AND WHY IT IS FILTERED RATHER THAN TRUSTED ----------------

Every row carries ``src_key``, the ``members-sha256`` of the shard that defines it --
the same hash ``tools/manifests/sm_*.json`` records under ``shards`` and that enters the
manifest digest. A row whose key is not the one the RATIFICATION measured describes an
arm nobody has register/spill numbers for.

That cannot happen inside one binary, and this module does not pretend otherwise: rows
and kernels are compiled together. It happens ACROSS the compile boundary -- a recorded
table, a cached plan, a manifest carried beside a rebuilt ``.so``. So the filter runs at
the join: :meth:`DispatchTable.from_extension` takes the ratified key set as an
argument, drops every row that does not match, and RECORDS what it dropped. A stale row
is never returned to a caller and never launched, which is the Inductor ``src_hash``
discipline: a cache entry whose source moved is not repaired, it is not used.

An EMPTY surviving set is an error rather than an empty table: it means the binary and
the record disagree about everything, which is a broken install, not a plan with no
candidates.
"""
from __future__ import annotations

from dataclasses import dataclass

#: The axes a row is keyed by, in the generated table's own order. Restating this
#: tuple is what the table exists to make unnecessary -- it is here as the NAMES of
#: the fields, not as their values.
AXES = ("BC", "DV", "D", "decay")


@dataclass(frozen=True, slots=True)
class DispatchRow:
    """One compiled arm, exactly as the binary carries it."""

    BC: int
    DV: int
    D: int
    decay: bool
    smem_bytes: int
    shard: str
    src_key: str

    @property
    def key(self) -> tuple:
        """The arm's coordinates, which is what a choice is compared against."""
        return (self.BC, self.DV, self.D, self.decay)

    def admitted_by(self, max_dynamic_smem_bytes: int) -> bool:
        """THE ENABLE PREDICATE, on an INPUT: does this machine grant a block enough?

        The mirror of `row_admitted` in `csrc/rola/src/consumer.cuh`, and deliberately
        the same inequality: a planner that would select an arm the launch then refuses
        has priced a plan that cannot run.
        """
        return self.smem_bytes <= max_dynamic_smem_bytes


@dataclass(frozen=True, slots=True)
class FilteredRow:
    """A row that did not survive the staleness join, and the reason."""

    row: DispatchRow
    reason: str


class DispatchTable:
    """The binary's arm set, joined against the ratification that measured it."""

    def __init__(self, rows, filtered=()):
        self._rows = tuple(rows)
        self.filtered = tuple(filtered)
        if not self._rows:
            raise ValueError(
                "the dispatch table is empty after the staleness join: every row the "
                "extension carries names a shard membership the ratification does not "
                "record. The binary and the manifest describe different matrices -- "
                "rebuild, or re-ratify, but do not plan against this.")

    # -- construction --------------------------------------------------------
    @classmethod
    def from_extension(cls, ext=None, ratified_keys=None) -> DispatchTable:
        """Read the table out of the built extension and apply the staleness filter.

        ``ratified_keys`` is ``{shard: members_sha256}``, normally
        ``rola.internal.manifest.ratified_shard_keys()``. Passing ``None`` means the
        caller has no record to join against and the rows are taken as they are -- an
        explicit act, not a default that hides a missing record.
        """
        if ext is None:
            from rola.ops._ext import extension
            ext = extension()
        rows = [DispatchRow(*record) for record in ext.rola_dispatch_table()]
        if ratified_keys is None:
            return cls(rows)
        kept, dropped = [], []
        for row in rows:
            want = ratified_keys.get(row.shard)
            if want is None:
                dropped.append(FilteredRow(row, f"no ratified record for shard {row.shard!r}"))
            elif want != row.src_key:
                dropped.append(FilteredRow(
                    row, f"shard {row.shard!r} is ratified at {want[:12]}... and the "
                         f"binary carries {row.src_key[:12]}..."))
            else:
                kept.append(row)
        return cls(kept, dropped)

    # -- what exists ---------------------------------------------------------
    @property
    def rows(self) -> tuple[DispatchRow, ...]:
        return self._rows

    def values(self, axis: str) -> tuple:
        """Every value the surviving rows carry on one axis, ascending.

        This is what replaces a hand-kept `SUPPORTED_*` tuple: the answer is the
        binary's, so a matrix change cannot leave a stale set behind it.
        """
        if axis not in AXES:
            raise ValueError(f"{axis!r} is not a table axis; the axes are {AXES}")
        return tuple(sorted({getattr(row, axis) for row in self._rows}))

    # -- what this machine can run -------------------------------------------
    def selectable(self, max_dynamic_smem_bytes: int) -> tuple[DispatchRow, ...]:
        """The rows whose footprint this device will grant a block."""
        return tuple(r for r in self._rows if r.admitted_by(max_dynamic_smem_bytes))

    # -- resolution ----------------------------------------------------------
    def resolve(self, BC: int, DV: int, D: int, decay: bool) -> DispatchRow:
        """The row for a fully-determined choice, or a refusal naming the choice.

        No nearest match and no default: a choice this binary has no arm for is an
        error, exactly as it is at the C++ lookup.
        """
        want = (BC, DV, D, bool(decay))
        for row in self._rows:
            if row.key == want:
                return row
        raise KeyError(
            f"no compiled arm for BC={BC}, d_v={DV}, D={D}, decay={decay}. The table "
            f"carries {len(self._rows)} arms over "
            + ", ".join(f"{a}={self.values(a)}" for a in AXES))


__all__ = ["AXES", "DispatchRow", "DispatchTable", "FilteredRow"]
