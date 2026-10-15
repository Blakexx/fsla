# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE WINDOW LAB'S INPUTS: a real carry call captured, for `carry_window.cu` (docs/lab/carry_window.md).

Each cell runs the pipeline the measurements run (`measure.subjects`: the bundle, the facts pass, the carve, zero units)
over its first `TOKENS` tokens -- a window's cost is flat in L -- with `rola.ops.carry.capture_bind` bound for one call.
A cell's directory holds `capture.bin` (the call's `CarryCapture`: its byte count, boxes and batch-heads, then the
operand block), every buffer an operand-block word points into as `buffer_<k>.bin`, and `buffers.txt`, a line a buffer:
`k role base bytes` -- the role `num` (the readout's output, its file the call's result: the reference), `check`, or
`in`; base the buffer's device address in the capturing process, so the experiment moves each pointer into its own
copy. `arm.txt` is the carve's arm key, the experiment's to match. Cells: `alt/` nl64k-alt-k4, `thin/` the same with
each (digit, token) of both list levels kept with probability 1/16 (lists averaging under one write a window, as at
~1M leaves).
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

CELL, TOKENS, SEED = "nl64k-alt-k4", 16384, 7
CELLS = (("alt", 1.0), ("thin", 1 / 16))


def _tensors(x, out: list) -> None:
    """Every tensor reachable from `x` through sequences, mappings and object fields."""
    import torch

    if isinstance(x, torch.Tensor):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for y in x:
            _tensors(y, out)
    elif isinstance(x, dict):
        for y in x.values():
            _tensors(y, out)
    elif hasattr(x, "__dict__") or hasattr(x, "__slots__"):
        for name in getattr(x, "__slots__", ()) or vars(x):
            _tensors(getattr(x, name, None), out)


def _cell(directory: Path, keep: float) -> list[str]:
    import torch

    from measure.cells import by_name, realize
    from measure.subjects import bundle_of, cell_carve, read_row_of
    from rola.ops import carry as c
    from rola.ops import facts as facts_ops

    spec = by_name(CELL)
    drawn = realize(spec)
    routes, values, layout, statics = bundle_of(spec, drawn, slice(0, TOKENS))
    if keep < 1.0:
        gen = torch.Generator(device="cuda").manual_seed(SEED)
        for side, plane in ((0, routes.amp_read), (1, routes.amp_write)):
            level = facts_ops.list_level(statics[side])
            mask = torch.rand(plane[level].shape, generator=gen, device="cuda") < keep
            plane[level].mul_(mask.to(plane[level].dtype))
    carve = cell_carve(spec)
    facts = facts_ops.facts(routes, values, layout, *statics, read_row_of(spec))
    unit = torch.zeros((routes.batch * routes.heads, routes.tokens), dtype=torch.int8, device="cuda")
    plans = c.plans_out(routes, carve)
    capture = torch.zeros(c.capture_bytes(), dtype=torch.uint8)
    c.capture_bind(capture)
    try:
        num, check = c.carry(routes, values, facts, carve, unit, plans=plans)
        torch.cuda.synchronize()
    finally:
        c.capture_bind(None)

    words = set(struct.unpack(f"<{capture.numel() // 8}Q", capture.numpy().tobytes()))
    found: list = []
    _tensors((routes, values, facts, unit, plans), found)
    storages = {}
    for t in found + [num, check]:
        if t.is_cuda:
            s = t.untyped_storage()
            storages.setdefault(s.data_ptr(), s)
    roles = {num.untyped_storage().data_ptr(): "num", check.untyped_storage().data_ptr(): "check"}
    directory.mkdir(parents=True, exist_ok=True)
    files, lines = ["capture.bin", "buffers.txt", "arm.txt"], []
    for base, s in sorted(storages.items()):
        if not any(base <= w < base + s.nbytes() for w in words):
            continue
        k = len(lines)
        data = torch.empty(0, dtype=torch.uint8, device="cuda").set_(s, 0, (s.nbytes(),), (1,))
        (directory / f"buffer_{k}.bin").write_bytes(data.cpu().numpy().tobytes())
        files.append(f"buffer_{k}.bin")
        lines.append(f"{k} {roles.get(base, 'in')} {base} {s.nbytes()}")
    (directory / "capture.bin").write_bytes(capture.numpy().tobytes())
    (directory / "buffers.txt").write_text("\n".join(lines) + "\n")
    (directory / "arm.txt").write_text(" ".join(str(x) for x in c.arm_key(carve)) + "\n")
    return [f"{directory.name}/{f}" for f in files]


def write(directory: Path) -> list[str]:
    out: list[str] = []
    for name, keep in CELLS:
        out += _cell(directory / name, keep)
    return out
