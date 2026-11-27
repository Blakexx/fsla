# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""The lab reads the graph's built experiments (bazel/lab.bzl, docs/internals/tools/lab.md#graph): the texts its disassemblers wrote, never a
tool of its own. A built experiment's kernels are read off those texts; a pack holds the same files beside its binary and reads the same way."""
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
import lab  # noqa: E402

SASS = """\
\t\tFunction : _Z1kv
\t.headerflags\t@"EF_CUDA_SM90"
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   IADD3 R2, R2, 0x1, RZ ;
        /*0020*/                   IADD3 R3, R3, 0x1, RZ ;
        /*0030*/                   EXIT ;
"""
USAGE = "Resource usage:\n Function _Z1kv:\n\tREG:12 STACK:0 SHARED:0 LOCAL:8 CONSTANT[0]:896\n"


def built(tmp_path) -> "lab.Built":
    exp = lab.Built(tmp_path, "x", "sm_90a")
    exp.part("sass").write_text(SASS)
    exp.part("usage").write_text(USAGE)
    exp.part("names").write_text("_Z1kv\tk()\n")
    exp.part("toolkit").write_text("Cuda compilation tools, release 13.5, V13.5.37\nBuild cuda_13.5.r13.5/compiler.0\n")
    return exp


def test_a_built_experiments_kernels_are_read_off_its_texts(tmp_path):
    table = lab.kernels(built(tmp_path))
    assert table == {"k()": {"ops": {"MOV": 1, "IADD3": 2, "EXIT": 1}, "regs": 12, "local": 8}}


def test_a_pack_is_read_as_the_target_is(tmp_path):
    exp = built(tmp_path)
    packed = lab.Built(exp.binary.parent, "x", "sm_90a")
    assert packed.functions() == ["_Z1kv"] and packed.toolkit() == "Build cuda_13.5.r13.5/compiler.0"
    assert packed.demangle(["_Z1kv"]) == {"_Z1kv": "k()"}


def test_a_name_the_build_did_not_write_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="names no _Z2kv"):
        built(tmp_path).demangle(["_Z2kv"])


def test_a_missing_part_is_named(tmp_path):
    with pytest.raises(SystemExit, match="x.sm_90a.lines does not exist"):
        built(tmp_path).text("lines")


def test_the_lab_runs_no_compiler_and_no_disassembler():
    """The experiments are the graph's targets: no lab command runs nvcc, cuobjdump, nvdisasm or cu++filt itself (the graph's actions do)."""
    text = (TOOLS / "lab.py").read_text()
    for call in ("cuda_bin(", "sass.tool(", "sass.cubins(", "sass.disassemble(", "sass.sass_lines(", "checked.demangle(",
                 "build_flags", '"nvcc"', '"cuobjdump"', '"nvdisasm"'):
        assert call not in text, call


def test_the_graphs_lab_architecture_is_the_labs_and_the_built_one():
    """`bazel/lab.bzl` hand-writes the lab's architecture and its compute capability: they are `lab.ARCH`, `hardware.registry`'s
    built architecture and its `cc_of`, one fact in three places."""
    import re

    sys.path.insert(0, str(TOOLS.parent))
    from hardware import registry

    rule = (TOOLS.parent / "bazel" / "lab.bzl").read_text()
    arch = re.search(r'^_ARCH = "(\w+)"$', rule, re.M).group(1)
    cc = re.search(r'^_CC = "(\d+)"$', rule, re.M).group(1)
    assert arch == lab.ARCH and registry.built() == [arch] and int(cc) == registry.cc_of(arch)
