# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The lab's tensor-wait checks (tools/lab.py): `WAIT_AFTER_ISSUE`, a wait on a commit group with no work of substance
between its last step and itself, and the advisory report of what could precede each wait (KERNEL_STANDARDS 29.16),
read from synthetic SASS text in `cuobjdump -sass`'s form."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import lab  # noqa: E402
import sass  # noqa: E402

STEP = "HGMMA.64x16x16.F32.BF16 R24, gdesc[UR24], R24"
LAST = STEP + ", gsb0"
WAIT = "WARPGROUP.DEPBAR.LE gsb0, 0x0"


def body(*instructions):
    """A kernel's (address, opcode, line) as `lab.disassembled` returns them, from instruction texts."""
    lines = [f"        /*{16 * i:04x}*/                   {text} ;      /* 0x0 */" for i, text in enumerate(instructions)]
    return [(int(m.group(1), 16), m.group(3), line) for line in lines if (m := sass.INSTRUCTION.match(line))]


def test_a_wait_straight_after_its_steps_is_counted():
    assert lab.wait_after_issue(body(STEP, LAST, WAIT)) == 1
    assert lab.wait_after_issue(body(STEP, LAST, "UIADD3 UR11, UR9, 0x3, URZ", "ISETP.NE.AND P2, PT, RZ, UR4, PT", WAIT)) == 1


def test_a_load_between_is_work_and_the_wait_is_not_counted():
    assert lab.wait_after_issue(body(LAST, "LDS.128 R4, [R2+0x10]", WAIT)) == 0
    assert lab.wait_after_issue(body(LAST, "UBLKCP.S.G [UR4], [UR6], UR8", WAIT)) == 0
    assert lab.wait_after_issue(body(LAST, "STG.E [R2.64], R6", WAIT)) == 0
    assert lab.wait_after_issue(body(LAST, "SYNCS.ARRIVE.TRANS64.A1T0 RZ, [UR5+0x34a20], RZ", WAIT)) == 0


def test_register_moves_between_are_not_work():
    assert lab.wait_after_issue(body(LAST, "IMAD.MOV.U32 R2, RZ, RZ, UR11", "MOV R3, R2", "LDC R5, c[0x0][0x160]", WAIT)) == 1


def test_two_groups_one_of_each():
    assert lab.wait_after_issue(body(LAST, WAIT, STEP, LAST, "LDS R4, [R2]", WAIT)) == 1
    assert lab.wait_after_issue(body(LAST, "LDG.E R4, [R2.64]", WAIT, STEP, LAST, WAIT)) == 1


def test_a_wait_that_leaves_a_group_in_flight_or_across_a_branch_is_not_counted():
    assert lab.wait_after_issue(body(LAST, "WARPGROUP.DEPBAR.LE gsb0, 0x1")) == 0
    assert lab.wait_after_issue(body(LAST, "BRA 0x100", WAIT)) == 0
    assert lab.wait_after_issue(body(WAIT)) == 0


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
