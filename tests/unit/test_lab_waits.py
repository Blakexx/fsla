# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The lab's advisory report of what could precede each tensor wait (tools/lab.py `waits_report`, KERNEL_STANDARDS 29.16), read from
synthetic SASS text in `cuobjdump -sass`'s form. The hazards -- a wait directly after its group's steps (`WAIT_AFTER_ISSUE`), steps in
a loop that holds no wait (`STEP_ROLLED`) -- are warpline's checker rules `wait_after_issue` and `step_rolled`, which its own tests cover
(tests/unit/test_lab_checker.py: how the lab calls them)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import checked  # noqa: E402
import lab  # noqa: E402

STEP = "HGMMA.64x16x16.F32.BF16 R24, gdesc[UR24], R24"
LAST = STEP + ", gsb0"
WAIT = "WARPGROUP.DEPBAR.LE gsb0, 0x0"


def body(*instructions):
    """A kernel's (address, opcode, line) as `lab.disassembled` returns them, from instruction texts."""
    lines = [f"        /*{16 * i:04x}*/                   {text} ;      /* 0x0 */" for i, text in enumerate(instructions)]
    return [(int(m.group(1), 16), m.group(3), line) for line in lines if (m := checked.checker().readers.INSTRUCTION.match(line))]


def test_the_report_counts_what_could_precede_a_wait_by_class():
    code = body(LAST, WAIT,
                "LDS.128 R4, [R2]", "IADD3 R8, R8, 0x1, RZ", "STS [R9], R7", "LDGSTS.E [R10], [R12.64]",
                "FADD R30, R24, R4",          # reads an accumulator: not a candidate
                "FADD R31, R30, R4",          # reads what that wrote: not a candidate
                "BAR.SYNC.DEFER_BLOCKING 0x0", "IADD3 R9, R9, 0x1, RZ")
    classes, stop = lab.wait_candidates(code, 1)
    assert dict(classes) == {"loads": 1, "other": 1, "stores": 1, "async": 1}
    assert stop == "BAR"


def test_the_report_widens_a_vector_register_to_the_run_it_reaches():
    clear = body(LAST, WAIT, "LDS.128 R20, [R2]", "IADD3 R9, R9, 0x1, RZ")      # R20..R23: short of R24..R31
    assert dict(lab.wait_candidates(clear, 1)[0]) == {"loads": 1, "other": 1}
    reaching = body(LAST, WAIT, "LDS.128 R22, [R2]", "IADD3 R9, R9, 0x1, RZ")   # R22..R25: into the accumulators
    assert dict(lab.wait_candidates(reaching, 1)[0]) == {"other": 1}
