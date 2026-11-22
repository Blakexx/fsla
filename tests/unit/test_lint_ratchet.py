"""The ratchet gates on what a commit adds: a new finding fails, a fixed finding must leave the baseline, and a baseline
never gains an entry over HEAD's (tools/lint/ratchet.py). No GPU."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "lint"))

import ratchet  # noqa: E402

A, B = ("a.py", "unused function 'f'", "0" * 16), ("b.py", "names the work code 'P1'", "1" * 16)


def test_the_tree_at_its_baseline_passes():
    assert ratchet.verdict(Counter({A: 2}), Counter({A: 2}), Counter({A: 2})) == (Counter(), Counter(), Counter())


def test_a_new_finding_and_a_second_copy_of_an_old_one_are_new():
    new, fixed, grown = ratchet.verdict(Counter({A: 3, B: 1}), Counter({A: 2}), Counter({A: 2}))
    assert new == Counter({A: 1, B: 1}) and not fixed and not grown


def test_a_fixed_finding_must_leave_the_baseline():
    new, fixed, grown = ratchet.verdict(Counter({A: 1}), Counter({A: 2}), Counter({A: 2}))
    assert fixed == Counter({A: 1}) and not new and not grown


def test_a_baseline_that_gains_over_head_fails_even_when_it_matches_the_tree():
    new, fixed, grown = ratchet.verdict(Counter({A: 1, B: 1}), Counter({A: 1, B: 1}), Counter({A: 1}))
    assert grown == Counter({B: 1}) and not new and not fixed


def test_a_renamed_path_moves_a_finding_and_does_not_grow_the_backlog():
    """A finding HEAD knew by message and line digest, under a path the baseline no longer carries, is the same
    finding after a rename: a renamed package moves its backlog. A genuinely new entry still grows it."""
    from collections import Counter

    from ratchet import verdict

    head = Counter({("old/a.py", "knob x", "d1"): 1, ("old/a.py", "knob y", "d2"): 1})
    moved = Counter({("new/a.py", "knob x", "d1"): 1, ("new/a.py", "knob y", "d2"): 1})
    assert verdict(moved, moved, head)[2] == Counter()
    grown = moved + Counter({("new/a.py", "knob z", "d3"): 1})
    assert verdict(grown, grown, head)[2] == Counter({("new/a.py", "knob z", "d3"): 1})


def test_the_baseline_round_trips():
    entries = Counter({A: 2, B: 1})
    assert ratchet._decode(ratchet._encode("x", entries)) == entries


OLD, NEW = "'_ARMS' is a literal arm table outside tools/gen_shards.py", "'_ARMS' is a literal arm table outside an `arms.bzl`"
OLD_ENTRY, NEW_ENTRY = ("t.py", OLD, "2" * 16), ("t.py", NEW, "2" * 16)


def test_a_rekey_renames_the_entries_counts_unchanged_and_records_itself():
    renamed, record = ratchet.rekey(Counter({OLD_ENTRY: 2, A: 1}), OLD, NEW)
    assert renamed == Counter({NEW_ENTRY: 2, A: 1}) and record == {"from": OLD, "to": NEW, "count": 2}
    assert ratchet._rekeys(ratchet._encode("x", renamed, [record])) == [record]
    assert ratchet._rekeys(ratchet._encode("x", renamed)) == []


def test_a_rekey_of_a_message_no_entry_has_is_refused():
    import pytest

    with pytest.raises(SystemExit, match="no baseline entry has the message"):
        ratchet.rekey(Counter({A: 1}), OLD, NEW)


def test_head_is_read_through_the_rekeys_so_a_rename_is_not_growth_and_an_added_count_is():
    rekeys = [{"from": OLD, "to": NEW, "count": 1}]
    head = ratchet.rekeyed(Counter({OLD_ENTRY: 1}), rekeys)
    assert ratchet.verdict(Counter(), Counter({NEW_ENTRY: 1}), head)[2] == Counter()
    assert ratchet.verdict(Counter(), Counter({NEW_ENTRY: 2}), head)[2] == Counter({NEW_ENTRY: 1})
    assert ratchet.verdict(Counter(), Counter({NEW_ENTRY: 1}), Counter({OLD_ENTRY: 1}))[2] == Counter({NEW_ENTRY: 1})


def test_a_follow_carries_an_entry_to_its_new_digest_and_head_is_read_through_it():
    """A reworded line changes its digest; the recorded follow maps HEAD's entry to the new key, so the move is not
    growth, while a follow cannot add to a count."""
    old, new = ("a.py", "m", "1" * 16), ("b.py", "m", "2" * 16)
    follows = [{"path": "a.py", "from_sha256": "1" * 16, "to_sha256": "2" * 16, "count": 2}]
    head = ratchet.followed(Counter({old: 2}), follows)
    assert head == Counter({("a.py", "m", "2" * 16): 2})
    assert ratchet.verdict(Counter(), Counter({new: 2}), head)[2] == Counter()
    assert ratchet.verdict(Counter(), Counter({new: 3}), head)[2] == Counter({new: 1})
