# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""WHAT THE COLORING BUYS THE CARRY: one carry launch on a cell's operands with the facts pass's colored lists, and again
with UNCOLORED lists in the same final form -- each union's slots its tokens' union ranks, each list's pass its tokens in
slot order 8 a step, a pass's last step padded -- timed by CUDA events, the two launches' fold states compared by their
sum (the launch runs without the den pass's units, so its readout partials are not a reading), each side's list places
counted (docs/internals/facts/facts.md#coloring: the coloring buys conflict-free gathers and pays pads).

    python measure/harness/bench_carry_lists.py [--cells flagship-alt-k4-L16384 ...] [--reps 20]
"""
from __future__ import annotations

import argparse
import statistics
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

WINDOW, GROUP, COLORS, LIST_AT, HEAD_ENTRIES, WARPS = 512, 32, 8, 8, 16, 8
GAP = -0x8000
PAD_BYTES = 8 * 64
UNIT_MIN, COEF_TOP = 4 - 127, 4
CELLS = ("flagship-alt-k4-L16384", "flagship-alt-k16-L16384", "nl64k-alt-k4")


def i16(x: torch.Tensor) -> torch.Tensor:
    """Low 16 bits as int16 (the lists' words are u16 stored as int16)."""
    x = x.to(torch.int32) & 0xFFFF
    return torch.where(x >= 0x8000, x - 0x10000, x).to(torch.int16)


def f16_bits(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF


def panel_row(s: torch.Tensor) -> torch.Tensor:
    return 128 * (s >> 1) + 32 * ((2 * (s & 1)) ^ ((s >> 1) & 3))


def pool_chunk0(row: int, s: torch.Tensor) -> torch.Tensor:
    return s & 7 if row == 128 else (s >> 1) & 3


def uncolored(routes, facts, level: int, side: str, P: int, row: int):
    """One side's lists at ``level`` in the final form, uncolored (see the module docstring)."""
    from rola.engine.facts.lists import Lists

    rows = routes.rows(level, side)                                   # [B, T, H, width] u8
    B, T, H, width = rows.shape
    BH, W, G, Q = B * H, T // WINDOW, width // GROUP, WINDOW // P
    byts = rows.permute(0, 2, 3, 1).reshape(BH, G, GROUP, W, WINDOW).permute(0, 3, 1, 2, 4)  # [BH, W, G, 32, 512]
    live = byts != 0
    union = live.any(3)                                               # [BH, W, G, 512]
    rank = union.to(torch.int32).cumsum(-1) - 1
    slot = rank.unsqueeze(3).expand_as(live)
    q = torch.div(slot.clamp(min=0), P, rounding_mode="floor")
    n = torch.stack([(live & (q == k)).sum(-1) for k in range(Q)], -1).to(torch.int32)   # [BH, W, G, 32, Q]
    steps = (n + 7) // 8
    starts = torch.cumsum(8 * steps, -1) - 8 * steps
    before = torch.cumsum(n, -1) - n
    idx = live.to(torch.int32).cumsum(-1) - 1 - before.gather(-1, q)
    pos = starts.gather(-1, q) + idx

    if side == "write":
        m, F = facts.fm[..., 1].float(), facts.fm[..., 0].float()   # [BH, L]
        exp = facts.list_exp.long().clamp(min=UNIT_MIN)               # [BH, W, width, 2] (K_m, K_v)
        exp = exp.reshape(BH, W, G, GROUP, 1, 2)
        tok = (torch.arange(W, device=rows.device).reshape(1, W, 1, 1, 1) * WINDOW
               + torch.arange(WINDOW, device=rows.device).reshape(1, 1, 1, 1, WINDOW))
        mt = m[:, None, None, None, :].expand(BH, W, 1, 1, T).gather(-1, tok.expand(BH, W, 1, 1, WINDOW))
        Ft = F[:, None, None, None, :].expand(BH, W, 1, 1, T).gather(-1, tok.expand(BH, W, 1, 1, WINDOW))
        b = byts.float()
        cm = f16_bits((b * mt) * torch.exp2((COEF_TOP - exp[..., 0]).float()))
        cv = f16_bits((b * Ft) * torch.exp2((COEF_TOP - exp[..., 1]).float()))
        words = torch.stack([slot, torch.zeros_like(slot), cv, cm], -1)
    else:
        loc = slot - q * P
        coef = f16_bits(byts.float() / 256)
        words = torch.stack([loc * row + 16 * pool_chunk0(row, loc), torch.zeros_like(loc), coef,
                             PAD_BYTES + panel_row(loc)], -1)

    entries = torch.zeros((BH, W, G, GROUP, WINDOW, 4), dtype=torch.int16, device=rows.device)
    e_idx = live.nonzero(as_tuple=True)
    entries[e_idx[0], e_idx[1], e_idx[2], e_idx[3], pos[live].long()] = i16(words[live])
    # the pads: a pass's last step filled
    k = torch.arange(COLORS, device=rows.device)
    pad_pos = starts[..., None] + n[..., None] + k                    # [BH, W, G, 32, Q, 8]
    pad_live = (n[..., None] + k) < 8 * steps[..., None]
    c = pad_pos & 7
    qq = torch.arange(Q, device=rows.device).reshape(1, 1, 1, 1, Q, 1).expand_as(pad_pos)
    if side == "write":
        pad = torch.stack([qq * P + c, torch.zeros_like(c), torch.zeros_like(c), torch.zeros_like(c)], -1)
    else:
        pad = torch.stack([c * row + 16 * pool_chunk0(row, c), torch.zeros_like(c), torch.zeros_like(c),
                           panel_row(c)], -1)
    p_idx = pad_live.nonzero(as_tuple=True)
    entries[p_idx[0], p_idx[1], p_idx[2], p_idx[3], pad_pos[pad_live].long()] = i16(pad[pad_live])
    entries = entries.reshape(BH, W, width, WINDOW, 4)

    end = (8 * steps).sum(-1, keepdim=True)
    at = torch.cat([starts, end.expand(*end.shape[:-1], LIST_AT - Q)], -1).reshape(BH, W, width, LIST_AT)
    heads = entries.reshape(BH, W, G, 4, WARPS, WINDOW, 4)[..., :HEAD_ENTRIES, :].permute(0, 1, 2, 4, 3, 5, 6)
    heads = heads.reshape(BH, W, G, WARPS, 4 * HEAD_ENTRIES, 4).contiguous()
    utok = torch.full((BH, W, G, WINDOW), GAP, dtype=torch.int16, device=rows.device)
    u_idx = union.nonzero(as_tuple=True)
    utok[u_idx[0], u_idx[1], u_idx[2], rank[union].long()] = u_idx[3].to(torch.int16)
    count = union.sum(-1).to(torch.int32)
    return Lists(entries, at.to(torch.int16), heads, utok, count), int((8 * steps).sum())


def time_it(fn, reps: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(1000.0 * a.elapsed_time(b))
    return statistics.median(times)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    from rola_devtools.locks.gpu import gpu_lock

    from measure.cells import by_name
    from measure.subjects import carry_operands, facts_form
    from rola.ops import carry as carry_ops

    with gpu_lock():
        for name in args.cells:
            spec = by_name(name)
            routes, values, facts, carve, unit = carry_operands(spec)
            form = facts_form(spec)
            places_c = {s: int(getattr(facts, s).at[..., -1].long().sum()) for s in ("write", "read")}
            w_lists, w_places = uncolored(routes, facts, carve.fold.lists[0], "write", form.unit_slots, form.read_row)
            r_lists, r_places = uncolored(routes, facts, carve.readout.lists[0], "read", form.pass_slots,
                                          form.read_row)
            for side, lst in (("write", w_lists), ("read", r_lists)):
                col = getattr(facts, side)
                same = bool(torch.equal((col.union != GAP).sum(-1).to(torch.int32), lst.count))
                print(f"{name} {side}: union sizes {'agree' if same else 'DIFFER'} with the colored lists'")
            plain = facts._replace(write=w_lists, read=r_lists)
            run_c = partial(carry_ops.carry, routes, values, facts, carve, unit)
            run_u = partial(carry_ops.carry, routes, values, plain, carve, unit)
            chk_c, chk_u = float(run_c()[1]), float(run_u()[1])
            t_c, t_u = time_it(run_c, args.reps), time_it(run_u, args.reps)
            print(f"{name}: carry {t_c:.1f} us colored, {t_u:.1f} us uncolored ({t_u / t_c:.3f}x); list places write "
                  f"{places_c['write']:,} -> {w_places:,}, read {places_c['read']:,} -> {r_places:,}; the fold's final "
                  f"state sum {chk_c:.7g} vs {chk_u:.7g}")
            del routes, values, facts, plain, w_lists, r_lists, run_c, run_u
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
