# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The lab's step reports (tools/lab.py): `STEP_PORT`, the bytes one tensor step puts through the shared-memory port,
`STEP_WORK`, the instructions a thread executes for it, and the `REGS` bound, read from synthetic SASS text in
`cuobjdump -sass`'s form."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import lab  # noqa: E402
import sass  # noqa: E402

SS = "HGMMA.64x16x16.F32.BF16 R24, gdesc[UR24], R24"               # A and B descriptors: 2,048 + 32 N bytes
RS = "HGMMA.64x64x16.F32.BF16 R24, R8, gdesc[UR24], R24"           # A in registers: B alone, 16 x N elements
FP8 = "HGMMA.64x128x32.F32.E4M3.E4M3 R24, gdesc[UR24], R24"
BACK = "BRA 0x0"


def body(*instructions, at=0):
    """A kernel's (address, opcode, line) as `lab.disassembled` returns them, from instruction texts."""
    lines = [f"        /*{16 * (at + i):04x}*/                   {text} ;      /* 0x0 */" for i, text in enumerate(instructions)]
    return [(int(m.group(1), 16), m.group(3), line) for line in lines if (m := sass.INSTRUCTION.match(line))]


def figures(code):
    """The report's rows for `code`: (what, steps, operand bytes, shared bytes, total bytes, work)."""
    return [(what, steps, *lab.region_figures(code, steps, members)) for what, steps, members in lab.step_regions(code)]


def test_a_descriptor_step_reads_both_operands_from_shared_memory():
    assert lab.step_operand_bytes(f"/*0000*/ {SS} ; /* 0x0 */") == 2048 + 32 * 16
    assert lab.step_operand_bytes("/*0000*/ HGMMA.64x256x16.F32.BF16 R24, gdesc[UR4], R24, gsb0 ; /* 0x0 */") == 2048 + 32 * 256


def test_a_register_operand_is_zero_and_only_b_is_shared():
    assert lab.step_operand_bytes(f"/*0000*/ {RS} ; /* 0x0 */") == 16 * 64 * 2
    assert lab.step_operand_bytes("/*0000*/ HGMMA.64x64x16.F32.BF16 R24, R8, gdesc[UR4], RZ, !UPT ; /* 0x0 */") == 2048


def test_the_element_size_is_the_types():
    assert lab.step_operand_bytes(f"/*0000*/ {FP8} ; /* 0x0 */") == (64 * 32 + 32 * 128) * 1
    assert lab.step_operand_bytes("/*0000*/ HGMMA.64x32x8.F32.TF32 R24, gdesc[UR4], R24 ; /* 0x0 */") == (64 * 8 + 8 * 32) * 4
    assert lab.step_operand_bytes("/*0000*/ HGMMA.64x16x16.F32.BF16 R24 ; /* 0x0 */") is None


def test_a_shared_access_moves_its_width_a_lane():
    assert [lab.shared_bytes(op) for op in ("LDS", "LDS.U16", "LDS.U8", "LDS.64", "LDS.128", "STS.128", "STS")] == \
        [128, 64, 32, 256, 512, 512, 128]
    assert [lab.shared_bytes(op) for op in ("LDSM.16.M88.4", "STSM.16.MT88.2", "LDG.E", "LDGSTS.E", "ATOMS.ADD")] == \
        [512, 256, 0, 0, 0]


def test_a_loop_with_one_step_and_known_shared_loads():
    code = body(SS, "LDS.128 R4, [R2]", "LDS.U16 R8, [R2+0x10]", "STS [R9], R7", "IADD3 R9, R9, 0x1, RZ", BACK)
    (what, steps, own, shared, total, work), = figures(code)
    assert (what.split()[0], steps, own, shared, total) == ("loop", 1, 2560, 512 + 64 + 128, 2560 + 704)
    assert -(-total // 128) == 26


def test_a_loop_of_steps_is_a_mean_a_step():
    code = body(SS, "LDS.128 R4, [R2]", SS, "LDS.128 R8, [R2]", BACK)
    (_, steps, own, shared, total, _), = figures(code)
    assert (steps, own, shared, total) == (2, 2560, 512, 3072)


def test_a_register_step_in_a_loop_counts_b_alone():
    code = body(RS, BACK)
    assert figures(code)[0][2:5] == (2048, 0, 2048)


def test_unrolled_steps_each_take_the_work_since_the_one_before():
    code = body("LDS.128 R4, [R2]", SS, "IADD3 R9, R9, 0x1, RZ", "STS [R9], R7", SS, SS)
    regions = figures(code)
    assert [(what, steps, total) for what, steps, _, _, total, _ in regions] == \
        [("an unrolled step", 1, 2560 + 512), ("an unrolled step", 1, 2560 + 128), ("an unrolled step", 1, 2560)]


def test_a_back_edge_ends_the_scan_of_an_unrolled_step():
    code = body("LDS.128 R4, [R2]", "IADD3 R9, R9, 0x1, RZ", BACK, SS)
    assert [row[4] for row in figures(code)] == [2560]


def test_the_instruction_classes():
    ops = ["HGMMA.64", "F2I.TRUNC", "MUFU.EX2", "IADD3", "IMAD", "ISETP.NE.AND", "UIADD3", "FFMA", "FADD", "LDG.E", "STG.E",
           "LDGSTS.E", "LDS.128", "STSM.16.MT88.4", "MOV", "IMAD.MOV.U32", "PRMT", "SHFL.IDX", "BRA", "BSSY", "EXIT",
           "WARPGROUP.DEPBAR.LE", "BAR.SYNC", "LDC"]
    assert [lab.work_class(op) for op in ops] == \
        ["tensor", "conv", "conv", "int", "int", "int", "int", "float", "float", "global", "global", "global", "shared",
         "shared", "move", "move", "move", "move", "branch", "branch", "branch", "other", "other", "int"]


def test_the_work_of_a_step_is_its_regions_instructions_by_class():
    code = body("F2I.TRUNC R4, R5", "IADD3 R9, R9, 0x1, RZ", "IADD3 R10, R9, 0x1, RZ", "LDG.E R6, [R2.64]", SS, BACK)
    (_, steps, _, _, _, work), = figures(code)
    assert (steps, dict(work)) == (1, {"conv": 1, "int": 2, "global": 1, "tensor": 1, "branch": 1})


def exceeded(rows, **bound):
    """`step_bounds` over a stub of the report: the refusals for the declared `bound`s."""
    expected = [{"kernel": "k", "op": op, "cmp": "<=", "count": n} for op, n in bound.items()]
    table = {"k": {"regs": 200}}
    real = lab.step_report
    lab.step_report = lambda _: {"k": rows}
    try:
        return lab.step_bounds(None, table, expected)
    finally:
        lab.step_report = real


ROW = ("loop 0x0-0x40", 4, 2048, 512, 2560, Counter({"tensor": 1, "int": 5, "branch": 1}))


def test_a_declared_bound_is_kept_or_refused():
    assert exceeded([ROW], STEP_PORT=2560, STEP_WORK=7, REGS=200) == []
    assert any("2,560 bytes" in f and "STEP_PORT" in f for f in exceeded([ROW], STEP_PORT=2559))
    assert any("7 instructions" in f and "STEP_WORK" in f for f in exceeded([ROW], STEP_WORK=6))


def test_a_bound_is_refused_in_any_one_region():
    small = ("an unrolled step", 1, 2048, 0, 2048, Counter({"tensor": 1}))
    assert len(exceeded([small, ROW], STEP_PORT=2559)) == 1


def test_a_register_count_over_the_roles_budget_is_refused_naming_it():
    (failure,) = exceeded([ROW], REGS=168)
    assert "200 registers" in failure and "budget of 168" in failure


def test_a_bound_on_nothing_or_not_a_bound_is_refused():
    expected = [{"kernel": "nothing", "op": "STEP_PORT", "cmp": "<=", "count": 1},
                {"kernel": "k", "op": "STEP_WORK", "cmp": ">=", "count": 1}]
    real, lab.step_report = lab.step_report, lambda _: {"k": [ROW]}
    try:
        failures = lab.step_bounds(None, {"k": {"regs": 1}}, expected)
    finally:
        lab.step_report = real
    assert len(failures) == 3 and "matches 'nothing'" in failures[0] and "declared \"<=\"" in failures[1]


def test_an_unread_step_is_refused_where_port_is_declared():
    unread = ("loop 0x0-0x40", 1, -1, 0, -1, Counter({"tensor": 1}))
    assert any("does not read" in f for f in exceeded([unread], STEP_PORT=10**6))
