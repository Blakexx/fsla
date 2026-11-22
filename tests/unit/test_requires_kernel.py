"""THE OWED MARKER on a CPU-runnable case (`tests/conftest.py`: `requires_kernel`): the carry's launch refuses by name, so this
test is a strict xfail (`carry not built`), counted in the owed line; were the carry built, it would pass and the marker would turn it
into a failure. No device."""
from __future__ import annotations

import pytest

from fsla.ops import carry


@pytest.mark.requires_kernel("carry")
def test_a_call_of_the_unbuilt_carry_is_owed():
    carry.carry(None, None, None)
