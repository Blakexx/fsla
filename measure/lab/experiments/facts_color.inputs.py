# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE COLORER LAB'S INPUTS for `facts_color.cu` (docs/internals/tools/lab.md#facts-color): every list unit of the
`facts_block` cell (`measure/harness/bench_facts_block.py`: 65,536 tokens, 4 heads, two levels of 256 digits a side,
each token-head live in k random digits of each) as the facts pass's stream half would hand it to a colorer that holds
no window -- the unit's bit TILE (its 32 digits' 16 words, the pass's words) and its STASH, each live entry in
(digit, token) order as its token and digit (`token | digit << 9`, u16) and its final coefficient (u32: the write
side's `c_v | c_m << 16` at the list's exponents, the read side's f16 `byte / 256`).

Two cells, `k4/` and `k16/`. Each holds `meta.bin` (int32: 0x434F4C46, units, the write side's units, the write and
read passes' slots, the read pool row's bytes), `tiles.bin` (`[units][32][16]` u32, the write side's units first, each
side's in (batch-head, window, group) order), `offsets.bin` (`[units + 1]` u32, each unit's first stash entry),
`tokens.bin` (u16) and `coefs.bin` (u32), the stash."""
from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "measure" / "harness"))

CELLS = (("k4", 4), ("k16", 16))
#: the sm_86 even arm's form: a 256-slot readout pass, a 128-slot fold unit, 64-byte read rows
READ_PASS, WRITE_PASS, READ_ROW = 256, 128, 64


def _side(rows, words, level: int, side: int, coef):
    """``(tiles, offsets, tokens, coefs)`` of one side's list units: the tile `[units, 32, 16]` from the words, and each
    live entry in (unit, digit, token) order with its coefficient (``coef(bytes, b, t, h, window, digit)``)."""
    import bench_facts_block as B
    import torch

    BH, windows, groups = B.HEADS, B.TOKENS // B.WINDOW, B.WIDTH // 32
    base = level * B.WIDTH
    w = words[:, side, base:base + B.WIDTH].reshape(BH, groups, 32, windows, 16).permute(0, 3, 1, 2, 4)
    tiles = w.reshape(-1, 32, 16).contiguous()
    # the rows [1, T, H, 256] as [bh, window, group, digit, token]
    r = rows[0].reshape(windows, B.WINDOW, BH, groups, 32).permute(2, 0, 3, 4, 1)
    live = r != 0
    counts = live.reshape(BH * windows * groups, -1).sum(1)
    offsets = torch.zeros(counts.numel() + 1, dtype=torch.int64, device=r.device)
    offsets[1:] = counts.cumsum(0)
    bh, win, g, d, t = live.nonzero(as_tuple=True)
    tokens = (t | (d << 9)).to(torch.int32)
    coefs = coef(r[bh, win, g, d, t], bh, win * B.WINDOW + t, win, g * 32 + d)
    return tiles, offsets, tokens, coefs


def write(directory: Path) -> list[str]:
    import bench_facts_block as B
    import torch

    from rola.ops.carry import Form

    files = []
    form = Form(2, READ_PASS, WRITE_PASS, READ_ROW, 128)
    for name, k in CELLS:
        out = directory / name
        out.mkdir(parents=True, exist_ok=True)
        read, write_rows, scales, g, vs = B.cell(k)
        outs = B.outputs()
        B.launch(read, write_rows, scales, g, vs, form, outs)
        words, list_exp, fm = outs[0], outs[2].long(), outs[3]

        def write_coef(byte, bh, t, win, digit):
            m, F = fm[bh, t, 1], fm[bh, t, 0]
            em = list_exp[bh, win, digit, 0].clamp_min(4 - 127)
            ev = list_exp[bh, win, digit, 1].clamp_min(4 - 127)
            b = byte.float()
            cm = ((b * m) * torch.exp2((4 - em).float())).half().view(torch.int16).int() & 0xFFFF
            cv = ((b * F) * torch.exp2((4 - ev).float())).half().view(torch.int16).int() & 0xFFFF
            return cv | (cm << 16)

        def read_coef(byte, bh, t, win, digit):
            return (byte.float() / 256).half().view(torch.int16).int() & 0xFFFF

        sides = [_side(write_rows[0], words, 0, 1, write_coef), _side(read[1], words, 1, 0, read_coef)]
        tiles = torch.cat([s[0] for s in sides]).cpu().numpy().astype("uint32")
        offsets = torch.cat([sides[0][1], sides[1][1][1:] + sides[0][1][-1]]).cpu().numpy().astype("uint32")
        tokens = torch.cat([s[2] for s in sides]).cpu().numpy().astype("uint16")
        coefs = torch.cat([s[3] for s in sides]).cpu().numpy().astype("uint32")
        (out / "meta.bin").write_bytes(struct.pack("<6i", 0x434F4C46, len(tiles), len(sides[0][0]), WRITE_PASS,
                                                   READ_PASS, READ_ROW))
        (out / "tiles.bin").write_bytes(tiles.tobytes())
        (out / "offsets.bin").write_bytes(offsets.tobytes())
        (out / "tokens.bin").write_bytes(tokens.tobytes())
        (out / "coefs.bin").write_bytes(coefs.tobytes())
        print(f"facts_color {name}: {len(tiles)} units, {len(tokens)} live entries")
        files += [f"{name}/{f}" for f in ("meta.bin", "tiles.bin", "offsets.bin", "tokens.bin", "coefs.bin")]
        del read, write_rows, outs, sides
        torch.cuda.empty_cache()
    return files
