# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE PLANNER ON ENVIRONMENTS THAT ARE NOT THIS MACHINE'S (`csrc/fsla/src/common/plan/planner.h`, `decode_launch.h`, docs/internals/common/plan/planner.md):
`plan_tool` is the library's own planner on device facts and a call given by hand, with no GPU. A device in the contract's ranges plans; one
outside it is refused naming the facts, as is a call outside it; the launch function maps a call's `d_v` into its arm's; the grid is the launch
function's and a forced count replaces it."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hardware import registry  # noqa: E402

TOOL = ROOT / "csrc" / "fsla" / "plan_tool"

#: the Hopper device the registry's part describes, every integer fact a point (a range at its upper end)
HOPPER = registry.device_point()


def _full(call: str, kernel: str = "decode") -> str:
    """The call with the facts it does not name given a lawful value: a call over one batch-head, 16-wide levels
    (`width_bits_k` 4) for the levels it has and no level written by lists (a dense innermost write); the prefill's call also has one token and a state."""
    named = dict(cell.split("=") for cell in call.split(",") if "=" in cell)
    levels = int(named.get("levels", 0))
    rest = {"batch_heads": 1,
            **({"tokens": 1, "stateful": 1} if kernel == "prefill" else {f"write_listed_{k}": 0 for k in (1, 2, 3, 4)}),
            **{f"width_bits_{k}": 4 if k <= levels else 0 for k in (1, 2, 3, 4)}}
    return ",".join([call, *(f"{k}={v}" for k, v in rest.items() if k not in named)])


def _plan(device: dict | None = None, call: str = "d_v=64,levels=2,decay=1", *extra: str, kernel: str = "decode"):
    facts = {**HOPPER, **(device or {})}
    run = subprocess.run([str(TOOL), "--kernel", kernel, "--device", ",".join(f"{k}={v}" for k, v in facts.items()), "--call", _full(call, kernel), *extra],
                         capture_output=True, text=True, timeout=60)
    return run


def _lines(run) -> dict:
    assert run.returncode == 0, run.stderr
    return {line.split(": ", 1)[0]: line.split(": ", 1)[1] for line in run.stdout.splitlines()}


def test_a_device_of_other_facts_in_the_ranges_plans_with_its_own_numbers():
    for sms in (114, 120, 132):
        out = _lines(_plan({"sm_count": sms}))
        assert out["args"] == f"cc=900 threads=512 smem_budget=232448 pad=0 sms={sms}"
        assert out["arm"] == "d_v=64 levels=2 decay=1"


@pytest.mark.parametrize("call, widths, offsets, dense", [
    ("d_v=64,levels=1,decay=0,width_bits_1=8", "256", "0", 1),
    ("d_v=64,levels=2,decay=1,width_bits_1=5,width_bits_2=4,write_listed_2=1", "32,16", "0,32", 0),
    ("d_v=64,levels=2,decay=1,width_bits_1=5,width_bits_2=4,write_listed_1=1", "32,16", "0,32", 1),
    ("d_v=64,levels=4,decay=0,width_bits_1=4,width_bits_2=5,width_bits_3=6,width_bits_4=7", "16,32,64,128", "0,16,48,112", 1),
])
def test_the_launch_function_derives_the_whole_atom_deposit_and_takes_each_levels_width_and_row_offset_from_the_calls_facts(call, widths, offsets, dense):
    out = _lines(_plan(None, call))
    assert out["levels"] == f"widths={widths} row_offsets={offsets} write_dense={dense}"


@pytest.mark.parametrize("call, named", [
    ("d_v=64,levels=2,decay=1,width_bits_1=3", "width_bits_1=3 (the contract: 4..8)"),
    ("d_v=64,levels=2,decay=1,width_bits_2=9", "width_bits_2=9 (the contract: 4..8)"),
    ("d_v=64,levels=2,decay=1,width_bits_3=4", "width_bits_3=4 (the contract: 0)"),
    ("d_v=64,levels=2,decay=1,write_listed_1=2", "write_listed_1=2 (the contract: 0..1)"),
])
def test_a_width_or_an_action_outside_the_contract_is_refused_naming_the_fact(call, named):
    run = _plan(None, call)
    assert run.returncode == 1 and "the call lies outside FSLA's decode contract" in run.stderr and named in run.stderr
    assert run.stdout == ""


def test_a_device_out_of_a_range_is_refused_naming_the_fact_and_its_range():
    for sms in (113, 133):
        run = _plan({"sm_count": sms})
        assert run.returncode == 1
        assert f"sm_count={sms} (the contract: 114..132)" in run.stderr and "the device lies outside FSLA's decode contract" in run.stderr


@pytest.mark.parametrize("fact, value", [
    ("cc", 800), ("regs_per_sm", 32768), ("smem_per_sm", 166912), ("smem_per_cta_optin", 166912), ("max_threads_per_sm", 1536),
])
def test_a_device_whose_point_fact_differs_is_refused_naming_it(fact, value):
    run = _plan({fact: value})
    assert run.returncode == 1 and f"{fact}={value} (the contract: {HOPPER[fact]})" in run.stderr


def test_the_model_key_names_the_device_in_a_refusal_and_is_never_the_match():
    model = "cc=900 pci=0x2330 name=NVIDIA H100 80GB HBM3"
    run = _plan({"sm_count": 100}, "d_v=64,levels=2,decay=1", "--model", model)
    assert run.returncode == 1 and f"the device: {model}" in run.stderr
    assert _plan(None, "d_v=64,levels=2,decay=1", "--model", "a name that matches nothing").returncode == 0


@pytest.mark.parametrize("d_v, arm, pad", [(2, 32, 30), (31, 32, 1), (32, 32, 0), (33, 64, 31), (48, 64, 16), (64, 64, 0)])
def test_the_launch_function_maps_a_value_width_to_the_arm_that_holds_it(d_v, arm, pad):
    out = _lines(_plan(None, f"d_v={d_v},levels=3,decay=0"))
    assert out["arm"] == f"d_v={arm} levels=3 decay=0"
    assert f"pad={pad} " in out["args"]


@pytest.mark.parametrize("call, named", [
    ("d_v=65,levels=2,decay=1", "d_v=65 (the contract: 2..64)"),
    ("d_v=1,levels=2,decay=1", "d_v=1 (the contract: 2..64)"),
    ("d_v=128,levels=2,decay=1", "d_v=128 (the contract: 2..64)"),
    ("d_v=64,levels=5,decay=1", "levels=5 (the contract: "),
    ("d_v=64,levels=2,decay=2", "decay=2 (the contract: "),
])
def test_a_call_outside_the_contract_is_refused_before_any_launch_naming_the_fact(call, named):
    run = _plan(None, call)
    assert run.returncode == 1 and "the call lies outside FSLA's decode contract" in run.stderr and named in run.stderr
    assert run.stdout == ""


@pytest.mark.parametrize("call, why", [
    ("d_v=64,levels=2", "the call states no value for 'decay'"),
    ("d_v=64,d_v=32,levels=2,decay=1", "the call supplies 'd_v' twice"),
    ("d_v=64,levels=2,decay=1,warps=4", "the call names 'warps', which is no fact a decode call supplies"),
    ("d_v=64,levels=2,decay=1,cc=900", "the call names 'cc', which is no fact a decode call supplies"),
    ("d_v=64,levels=2,decay=1,tokens=1", "the call names 'tokens', which is no fact a decode call supplies"),
    ("d_v=64,levels=2,decay=1,stateful=1", "the call names 'stateful', which is no fact a decode call supplies"),
    ("d_v=64,levels=2,decay=1,read_listed_1=1", "the call names 'read_listed_1', which is no fact a decode call supplies"),
    ("d_v=64,levels=2,decay=1,prefill=0", "the call names 'prefill', which is no fact a decode call supplies"),
])
def test_a_call_that_misstates_its_facts_is_refused_by_name(call, why):
    run = _plan(None, call)
    assert run.returncode == 1 and why in run.stderr


def test_a_device_read_that_misstates_its_facts_is_refused_by_name():
    run = subprocess.run([str(TOOL), "--kernel", "decode", "--device", "cc=900,regs_per_sm=65536", "--call", "d_v=64,levels=2,decay=1"], capture_output=True, text=True)
    assert run.returncode == 1 and "the device read states no value for the schema's fact" in run.stderr
    run = subprocess.run([str(TOOL), "--kernel", "decode", "--device", ",".join(f"{k}={v}" for k, v in {**HOPPER, "d_v": 64}.items()), "--call", "d_v=64,levels=2,decay=1"],
                         capture_output=True, text=True)
    assert run.returncode == 1 and "the device read names 'd_v', which is no integer fact the device supplies" in run.stderr


def _grid(device: dict | None, ceiling: int, bh: int, forced: int) -> int:
    run = _plan(device, "d_v=64,levels=2,decay=1", "--grid", f"unit_ceiling={ceiling},bh={bh},forced={forced}")
    return int(_lines(run)["grid"])


def test_the_grid_is_one_wave_of_the_devices_sms_at_most_the_heads_unit_supply():
    assert _grid({"sm_count": 120}, ceiling=256, bh=64, forced=-1) == 120
    assert _grid({"sm_count": 132}, ceiling=256, bh=64, forced=-1) == 132
    assert _grid({"sm_count": 132}, ceiling=256, bh=2, forced=-1) == 32  # 2 heads of 16 CTAs' supply each
    assert _grid({"sm_count": 132}, ceiling=16, bh=3, forced=-1) == 3  # one CTA's share a head
    assert _grid({"sm_count": 132}, ceiling=1, bh=1, forced=-1) == 1


def test_a_forced_grid_replaces_the_planned_one_at_every_count_the_device_could_plan():
    """The forced-plan law's CPU half: a gate's count reaches the launch as it was given, whatever the device and the data; the device half
    (the same data at each forced count matches the oracle) is the gated decode tests'."""
    for sms in (114, 132):
        planned = _grid({"sm_count": sms}, ceiling=256, bh=64, forced=-1)
        for forced in (1, 2, planned - 1, planned, planned + 1, 4 * sms):
            assert _grid({"sm_count": sms}, ceiling=256, bh=64, forced=forced) == forced


def test_the_depth_is_one_number_the_launch_header_and_the_width_facts_agree_on():
    """Decode's depth is stated by `kDecodeMaxLevels` (the launch header, which sizes the per-level arrays of `DecodeArgs` and, through `decode.cuh`, the kernel's)
    and by the call's `width_bits_k` facts (which `fsla.ops.produce._decode_descriptor` reads from the planner): adding a fact without raising the constant, or the
    reverse, fails here; a call of the deepest level plans, one level deeper is outside the contract."""
    from fsla.ops._ext import extension

    constant = int(re.search(r"kDecodeMaxLevels = (\d+);", (ROOT / "csrc" / "fsla" / "src" / "common" / "plan" / "decode_launch.h").read_text()).group(1))
    facts = [name for name in extension().fsla_call_facts("decode").split(",") if name.startswith("width_bits_")]
    assert facts == [f"width_bits_{k}" for k in range(1, constant + 1)]
    deepest = _lines(_plan(None, f"d_v=64,levels={constant},decay=0"))
    assert deepest["levels"].startswith(f"widths={','.join(['16'] * constant)} ")
    assert _plan(None, f"d_v=64,levels={constant + 1},decay=0").returncode == 1


#: the supported B times H of a decode step (docs/api.md#batch-heads): the largest its work queue of 4 B H + 2 int32 counters indexes
BATCH_HEADS_MAX = 536870911


def test_the_batch_heads_the_contract_states_plan_at_both_ends_and_one_beyond_either_is_refused():
    """`batch_heads` is a range of the contract: its two ends plan, the point below and the point above are refused naming the fact and the range."""
    for bh in (1, BATCH_HEADS_MAX):
        out = _lines(_plan(None, f"d_v=64,levels=2,decay=1,batch_heads={bh}"))
        assert out["arm"] == "d_v=64 levels=2 decay=1"
    for bh in (0, BATCH_HEADS_MAX + 1):
        run = _plan(None, f"d_v=64,levels=2,decay=1,batch_heads={bh}")
        assert run.returncode == 1 and f"batch_heads={bh} (the contract: 1..{BATCH_HEADS_MAX})" in run.stderr
        assert run.stdout == ""


def test_the_documented_batch_heads_bound_is_the_contracts():
    """The sentence of docs/api.md#batch-heads states the bound the contract's range holds."""
    text = (ROOT / "docs" / "api.md").read_text()
    assert f"{BATCH_HEADS_MAX}" in text[text.index('<a id="batch-heads">'):][:1200]


PREFILL = "d_v=64,levels=2,decay=0,tokens=4096,batch_heads=8,stateful=0"


def test_a_prefill_call_inside_an_owed_descriptor_is_refused_naming_each_owed_kernel():
    """The prefill's descriptors are OWED (hardware/kernels/prefill/descriptors/owed): a call inside one is held by no context of its contract, and the planner says
    which kernels it waits on rather than that it lies outside the contract."""
    run = _plan(None, PREFILL, kernel="prefill")
    assert run.returncode == 1 and run.stdout == ""
    assert "OWED descriptors ('hopper_prefill_d2" in run.stderr and "not built: prefill." in run.stderr
    assert "lies outside FSLA's prefill contract" not in run.stderr


@pytest.mark.parametrize("call, named", [
    ("d_v=1,levels=2,decay=0,tokens=4096,batch_heads=8,stateful=0", "d_v=1 (the contract: "),
    ("d_v=64,levels=2,decay=0,tokens=0,batch_heads=8,stateful=0", "tokens=0 (the contract: "),
])
def test_a_prefill_call_in_no_owed_descriptor_is_outside_the_contract_naming_the_fact(call, named):
    run = _plan(None, call, kernel="prefill")
    assert run.returncode == 1 and "the call lies outside FSLA's prefill contract" in run.stderr and named in run.stderr


def _cell_prefill_calls():
    """Every call a `requires_kernel("prefill")` or `requires_kernel("carry")` test builds: each central carry cell's prefill, as the planner's `--call` states it (the facts
    are the call's extents, its levels' widths and its kind: no weight is a fact)."""
    from measure.cells import carry_cells

    for cell in carry_cells():
        levels = len(cell.widths)
        facts = [f"d_v={cell.dv}", f"levels={levels}", "decay=0", f"tokens={cell.tokens}", f"batch_heads={cell.heads * cell.group}",
                 f"stateful={int(cell.state != 'none')}"]
        facts += [f"width_bits_{k}={width.bit_length() - 1}" for k, width in enumerate(cell.widths, 1)]
        yield pytest.param(",".join(facts), id=cell.name)


@pytest.mark.parametrize("call", _cell_prefill_calls())
def test_every_cell_a_prefill_test_builds_is_refused_as_owed_naming_the_prefill_kernel_never_as_outside_the_contract(call):
    """THE CLASS OF FAILURE THE CPU TIER MUST SEE: a prefill test marked `requires_kernel("prefill")` is a strict xfail only if the call's descriptor is
    refused as `KernelNotBuilt`. A cell no owed descriptor covers would be refused as outside the contract instead (a RuntimeError), and the xfail
    would fail on a device. Every central carry cell is planned here against the synthetic Hopper device."""
    run = _plan(None, call, kernel="prefill")
    assert run.returncode == 1 and run.stdout == "", run.stdout
    assert "OWED descriptors" in run.stderr and "not built: prefill." in run.stderr, run.stderr
    assert "lies outside FSLA's prefill contract" not in run.stderr
