"""R4 -- the closed-world RUNTIME arch refusal, checked where it can be checked.

``setup.py``'s rule R4 forbids *a launch on a device whose arch was never
measured*. Unlike R1/R2/R3 it cannot be enforced at build time, because the
device is not present then; it is enforced in C++ by a pair of clauses, and
this file gates that the pair is intact.

**THE PAIR.** The compile-time half is
``csrc/rola/src/chunk/arch_caps.cuh``'s ``static_assert(tabulated(kArch))``,
which fixes the capability row every arm's staging depth, CTA target and
shared-memory fit was DERIVED from. The runtime half is
``chunk.cu``'s ``check_arch_table()``, which refuses -- on the first launch,
per device -- an architecture the table has no row for, or a row the driver
disagrees with. Neither half can substitute for the other: a derivation is only
as good as the facts it was closed over, and only a fact read off the DEVICE can
establish that this is the device they were read for
(the extension trap: only a device-side fact catches a stale binary).

**WHY THIS SHAPE OF TEST.** The refusal's subject is an architecture this
machine is not, so the refusal itself cannot be provoked on real hardware: an
untabulated device is exactly the thing that is unavailable. What CAN be checked
is the plumbing -- that the clause exists, that no entry which launches can be
reached without it, and that on the device that IS here the check passes and a
launch follows. The same discipline the build-time half already gets from
``tools/supported.py``, whose reader would fail if ``sm_arch_tabulated`` stopped
being a table.

CPU-lane except the one live row, which needs the device it is about.
"""

from __future__ import annotations

import pathlib
import re

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
CHUNK_CU = REPO / "csrc" / "rola" / "src" / "chunk" / "chunk.cu"
ARCH_CAPS = REPO / "csrc" / "rola" / "src" / "chunk" / "arch_caps.cuh"
CHUNK_API = REPO / "csrc" / "rola" / "src" / "chunk" / "chunk_api.cuh"

#: The entries of the chunk surface that issue no launch of THEIR OWN. ``arms``
#: reads the built matrix out of compile-time constants and returns it;
#: ``forward`` is a composition of three entries that are each gated, and
#: :func:`test_the_composed_entry_is_only_gated_entries` is what keeps it one.
#: Every other declared entry reaches a kernel and must pass the gate first.
_LAUNCHLESS_ENTRIES = {"arms", "forward"}


def _entry_names() -> set[str]:
    """The host entries the pybind TU is offered, read off the declaration header."""
    decls = re.findall(r"^\s*(?:[\w:<>,\s]+?)\s(\w+)\(", CHUNK_API.read_text(), re.M)
    names = {name for name in decls if name not in {"if", "for", "while"}}
    assert names, "cannot read the chunk entry declarations out of chunk_api.cuh"
    return names


def test_the_compile_time_half_is_a_table_and_a_static_assert():
    """``tabulated()`` gates the compiling target, and nothing softens it."""
    text = ARCH_CAPS.read_text()
    assert "constexpr bool tabulated(int cc)" in text
    assert re.search(r"static_assert\(\s*tabulated\(kArch\)", text), (
        "arch_caps.cuh no longer fails the BUILD on an untabulated compile target; "
        "that is half of R4 and it is not optional")


def test_the_runtime_half_refuses_an_untabulated_architecture():
    """``check_arch_table()`` keys on ``tabulated`` and on the driver's own numbers."""
    text = CHUNK_CU.read_text()
    body = text[text.index("static void check_arch_table()"):]
    body = body[:body.index("\n}\n") + 3]
    assert "denseref::arch::tabulated(cc)" in body, (
        "the runtime half must refuse an arch the TABLE does not carry -- an unbuilt "
        "ARM and an untabulated ARCHITECTURE are different refusals")
    assert "cudaGetDeviceProperties" in body and "cudaDeviceGetAttribute" in body, (
        "the check must read the DEVICE; a path or a hash cannot establish which "
        "device a compiled derivation was derived for")
    for field in ("smem_per_sm", "smem_per_cta_max", "regs_per_sm", "max_threads_per_sm"):
        assert f"want.{field}" in body, (
            f"the table's {field} is an input to a compiled derivation and must be "
            "compared against the driver")


@pytest.mark.parametrize("entry", sorted(_entry_names() - _LAUNCHLESS_ENTRIES))
def test_every_launching_entry_passes_the_gate_first(entry: str):
    """No entry that reaches a kernel can be called without the refusal running.

    There is no single launcher in this TU to hang the check on -- the table pass,
    the stats pass, the kernel and the divide are four independent entries -- so the
    property is per entry, and this is what keeps a fifth from being added without it.
    """
    text = CHUNK_CU.read_text()
    match = re.search(rf"^[\w:<>,\s&*]+?\b{entry}\(.*?\)\s*\{{\n(.*?)\n^\}}",
                      text, re.M | re.S)
    assert match, f"cannot find the definition of {entry} in chunk.cu"
    first = next(line.strip() for line in match.group(1).splitlines() if line.strip())
    assert first == "check_arch_table();", (
        f"{entry} launches, so the closed-world arch refusal must be its FIRST "
        f"statement; found {first!r}")


def test_the_composed_entry_is_only_gated_entries():
    """``forward`` needs no gate of its own only for as long as it issues no launch."""
    text = CHUNK_CU.read_text()
    body = re.search(r"^torch::Tensor forward\(.*?\)\s*\{\n(.*?)\n^\}", text, re.M | re.S)
    assert body, "cannot find the definition of forward in chunk.cu"
    called = set(re.findall(r"(\w+)\(", body.group(1)))
    assert called <= {"union_table", "forward_raw", "divide_pass"}, (
        "forward has grown a call of its own; either it launches, in which case it "
        f"needs the gate as its first statement, or say which of {sorted(called)} is "
        "gated")


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the live half needs the device")
def test_this_device_is_tabulated_and_the_gate_admits_it():
    """The positive direction: the check ran, said yes, and a launch followed.

    A gate that only ever refuses is indistinguishable from a gate that is never
    reached, so the admitted case is asserted too -- and asserted through a real
    entry, because the entry is where the call site is.
    """
    from rola.ops._ext import extension

    major, minor = torch.cuda.get_device_capability()
    cc = major * 100 + minor * 10
    rows = re.search(r"constexpr bool tabulated\(int cc\) \{\s*return ([^;]+);",
                     ARCH_CAPS.read_text(), re.S).group(1)
    assert f"cc == {cc}" in rows, (
        f"this device is compute capability {major}.{minor}, which the chunk table "
        "does not carry -- the refusal below is then the correct behaviour, and this "
        "row is the one that must be added")
    arms = extension().chunk_arms()
    assert arms, "no built arms to launch through"
    c, bc, d, b = (int(x) for x in arms[0][:4])
    bh, tokens = 1, c
    pw = torch.zeros(1, 1, tokens, d * b, dtype=torch.bfloat16, device="cuda")
    bits = extension().chunk_stats(pw, [b] * d)
    assert bits.shape[0] == bh


def test_the_dynamic_shared_memory_OPT_IN_is_load_bearing_for_the_built_matrix():
    """At least one built arm needs more shared memory than a CTA gets by default.

    The launcher raises every arm's dynamic-shared-memory limit with
    `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, ...)`
    (`csrc/rola/src/chunk/chunk_launch.cuh`). That call is the difference between a
    working arm and `cudaErrorInvalidValue` at launch, and a matrix whose every arm
    happened to fit under the 48 KB default would let the opt-in be deleted with
    every gate staying green. So the NON-VACUITY is asserted: the shipped footprints
    straddle the default limit.

    P67 D2-b re-anchored this from `tests/unit/test_dispatch_table.py::
    test_every_shipped_arm_is_admitted_by_this_device`, whose authority was
    `rola_max_dynamic_smem_bytes` -- a tiled binding D2-c deletes. The ADMISSION
    half moved with R-7: `check_arch_table` compares the arch row's
    `smem_per_cta_max` against the driver's on every launching entry, which refuses
    a device the table does not describe rather than one derived number.
    """
    from rola.ops._ext import extension

    #: `chunk_arms()` rows are `(C, BC, D, B, ...)` with the arm's shared-memory
    #: footprint at index 8 -- the number `chunk_launch.cuh` opts in to.
    footprints = [row[8] for row in extension().chunk_arms()]
    assert footprints, "the manifest reports no built arms"
    default_limit = 48 * 1024
    assert max(footprints) > default_limit, (
        f"every built arm fits in the {default_limit}-byte default ({max(footprints)} "
        "is the largest), so the launcher's opt-in could be deleted and nothing here "
        "would notice -- the claim is vacuous and the matrix or this row must change")
    assert min(footprints) > 0
