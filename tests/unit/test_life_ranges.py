"""The life-range report's registers (`tools/life_ranges.py`, docs/internals/tools/life_ranges.md#liveness): warpline's pressure over
the life-range marks (`flow.pressure`), read through the seam (`tools/checked.py`). A value held across a shuffle's slow-path call
and one held across a loop's exit through a call to a label both count -- the two shapes nvdisasm's own count drops -- proven on a
compiled probe and on a listing that has the exit.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import checked  # noqa: E402
import dev_config  # noqa: E402
import life_ranges  # noqa: E402

#: the probe, compiled by the graph's CUDA rule (tests/BUILD.bazel's `life_ranges_probe`): its one cubin
PROBE = ROOT / "tests" / "_objs" / "life_ranges_probe" / "life_ranges_probe.pic.o"


@pytest.fixture(scope="module")
def cubin(tmp_path_factory):
    found = checked.checker().runner.cubins(PROBE, tmp_path_factory.mktemp("cubins"), dev_config.cuda_home())
    assert len(found) == 1, found
    return found[0]


def test_values_held_across_a_slow_path_call_count_before_the_call(cubin):
    """nvdisasm's marks have the call write registers the probe holds (its own count drops 16 of them at the row
    before the call): the call's registers are its subroutine's."""
    rows = life_ranges.live_counts(cubin, "division_probe")
    before = [rows[k - 1][2] for k, (_, text, _) in enumerate(rows) if text.startswith("CALL") and "0x" in text]
    assert before, "the probe's division has no slow-path call: the probe no longer has the shape it tests"
    assert min(before) >= 32, rows


#: THE LOOP THAT LEAVES THROUGH A CALL TO A LABEL, as the toolkit prints one: R2 written, a loop that leaves through
#: `CALL.REL.NOINC` to a label of the function (marked, as every call is, as writing R2..R9), then R2 read; the graph takes the call
#: for one that returns, so no edge reaches the label. Each row: (label, instruction, registers read, registers written)
LOOP = [(None, "IMAD.MOV.U32 R2, RZ, RZ, 0x1", (), (2,)),
        (".L_x_0", "IADD3 R3, R3, 0x1, RZ", (3,), (3,)),
        (None, "@!P0 CALL.REL.NOINC 0x40", (), tuple(range(2, 10))),
        (None, "BRA 0x10", (), ()),
        (".L_x_1", "STG.E [R4.64], R2", (2, 4, 5), ()),
        (None, "EXIT", (), ())]


def texts(rows) -> tuple[str, str, str, str]:
    """The loop's `cuobjdump -sass`, `nvdisasm -bbcfg -poff`, `cuobjdump -res-usage` and `nvdisasm --print-life-ranges` texts."""
    sass = "\t\tFunction : _Z4loopPf\n" + "".join(f"        /*{16 * k:04x}*/                   {line} ;\n"
                                                  for k, (_, line, _, _) in enumerate(rows)) + "        /*0060*/       BRA 0x60 ;\n"

    def node(name, offsets):
        return f'"{name}"\n[label="{{' + "|".join(f"<i{o}>{o:04x}:\\ X\\ ;\\l" for o in offsets) + '}"]\n'

    cfg = ('digraph f {\nsubgraph "cluster__Z4loopPf" {\n' + node("_Z4loopPf", [0x0]) + node(".L_x_0", [0x10, 0x20, 0x30])
           + node(".L_x_1", [0x40, 0x50]) + '"_Z4loopPf":exit0:s -> ".L_x_0":entry:n [style=solid];\n'
           + '".L_x_0":exit0:s -> ".L_x_0":entry:n [style=solid];\n}\n}\n')
    res = " Function _Z4loopPf:\n  REG:10 STACK:0 SHARED:0 LOCAL:0\n"
    head = "                    // |   # " + " ".join(str(r) for r in range(12)) + " |"
    base = head.index("// |") + 4
    at = [m.end() - 1 - base for m in re.finditer(r"\b\d+\b", head)]
    life = ['\t.section\t.text._Z4loopPf,"ax",@progbits', "                    // |   GPR   |", head]
    for k, (label, line, reads, writes) in enumerate(rows):
        if label:
            life.append(f"{label}:")
        marks = [" "] * (max(at) + 2)
        for r in set(reads) | set(writes):
            marks[at[r]] = "x" if r in reads and r in writes else "v" if r in reads else "^"
        life.append(f"        /*{16 * k:04x}*/                   {line} ;   // |{0:4d}{''.join(marks)[4:]}")
    return sass, cfg, res, "\n".join(life) + "\n"


def test_a_value_held_across_a_loop_exit_through_a_call_counts_in_the_loop():
    c = checked.checker()
    sass, cfg, res, life = texts(LOOP)
    (k,) = c.model.kernels_from_text(sass, cfg, res, life_text=life).values()
    held = {at: sum(1 for r in regs if re.fullmatch(r"R\d+", r)) for at, regs in c.flow.pressure(k).items()}
    #: in the loop: R2 (held to the store) and R3 (written each trip); R4 and R5 are never written, so never count
    assert [held[at] for at in (0x10, 0x20, 0x30, 0x40)] == [2, 2, 2, 1], held
