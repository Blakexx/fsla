# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""THE OWNER CARRY'S SKELETON in the part harness (KERNEL_STANDARDS §19): each component on a cell's real words and
rows, checked against a pure model, and timed for the carve's pricing (`docs/internals/owner/owner.md#skeleton`). The
driver is `owner/owner_skeleton.cu`, built ahead of time into the parts module (`ROLA_BUILD_PARTS=1`).

    python measure/harness/bench_owner_skeleton.py --check lists --cell flagship-alt-k4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

WINDOW = 512
WINDOW_WORDS = WINDOW // 32
WARPS = 8
LIST_CAPACITY = WINDOW
CARVES = ((16, 64), (32, 32))
STEP_CHOSEN, STEP_PER_LIST, STEP_MERGED = 0, 1, 2
#: the lists' cost model, `owner_lists.cuh`'s
COST_A, COST_B, COST_IMMAS, COST_ENTRY = 4, 12, 8 * 16, 6


def ops():
    try:
        from rola_cu13 import _C_parts  # noqa: F401 -- registers torch.ops.rola_owner
    except ImportError as ex:
        raise SystemExit("the part harness's driver is not built: ROLA_BUILD_PARTS=1 ROLA_CUDA_ARCHS=86 "
                         "python -m pip install -e . --no-build-isolation") from ex
    return torch.ops.rola_owner


def cell_words(name: str):
    """``(spec, words, level_s, sparse_write, sparse_read)``: the cell's facts words and its carve's sides."""
    from measure.cells import by_name, realize
    from measure.subjects import bundle_of
    from rola.ops import facts as facts_ops

    spec = by_name(name)
    routes, values, layout, statics = bundle_of(spec, realize(spec))
    words = facts_ops.facts(routes, values, layout, *statics).words
    declared = spec.declared_sparsity()
    level_s = 1 if declared[1][0] or not declared[0][0] else 0
    level_r = 1 - level_s
    return spec, words, level_s, int(declared[level_r][1]), int(declared[level_s][0])


def _popc(x: int) -> int:
    return bin(x).count("1")


def lists_model(w, widths, level_s, sparse_write, sparse_read, rs, rd, step, bh, sblk, rblk, win):
    """One CTA-window's dump, `owner_skeleton.cu`'s `dump_lists` layout, from the words."""
    level_r = 1 - level_s
    row_base = (0, widths[0])
    cols = slice(WINDOW_WORDS * win, WINDOW_WORDS * (win + 1))
    full = (1 << 32) - 1
    rows = []
    for j in range(rd):
        rows.append([int(x) & full for x in w[bh, 1, row_base[level_r] + rblk * rd + j, cols]] if sparse_write else None)
    for i in range(rs):
        rows.append([int(x) & full for x in w[bh, 0, row_base[level_s] + sblk * rs + i, cols]] if sparse_read else None)
    sparse = (sparse_write, sparse_read)
    unions, bases = [], []
    for side, members in ((0, rows[:rd]), (1, rows[rd:])):
        u = [full] * WINDOW_WORDS if not sparse[side] else [0] * WINDOW_WORDS
        if sparse[side]:
            for r in members:
                u = [a | b for a, b in zip(u, r, strict=True)]
        unions.append(u)
        acc, b = 0, []
        for x in u:
            b.append(acc)
            acc += _popc(x)
        bases.append(b)
    counts = [sum(_popc(x) for x in r) if r is not None else WINDOW for r in rows]
    forms, totals, entries = [[0] * WARPS, [0] * WARPS], [[0] * WARPS, [0] * WARPS], [[], []]

    def slots(side, word_list):
        out = []
        for k, bits in enumerate(word_list):
            for b in range(32):
                if bits >> b & 1:
                    out.append(bases[side][k] + _popc(unions[side][k] & ((1 << b) - 1)))
        return out

    for side, lists, row0f in ((0, rd // WARPS, lambda wp: wp * (rd // WARPS)),
                               (1, rs // WARPS, lambda wp: rd + wp * (rs // WARPS))):
        for wp in range(WARPS):
            if not sparse[side]:
                forms[side][wp], totals[side][wp] = STEP_MERGED, WINDOW
                entries[side].append([])
                continue
            mine = [rows[row0f(wp) + j] for j in range(lists)]
            u = [0] * WINDOW_WORDS
            for r in mine:
                u = [a | b for a, b in zip(u, r, strict=True)]
            merged_n = sum(_popc(x) for x in u)
            ns = [counts[row0f(wp) + j] for j in range(lists)]
            listed, steps = sum(ns), sum((n + 31) >> 5 for n in ns)
            msteps = (merged_n + 31) >> 5
            per_list = steps * (COST_A + COST_B + COST_IMMAS) + listed * COST_ENTRY
            merged = msteps * (COST_B + lists * (COST_A + COST_IMMAS)) + merged_n * COST_ENTRY
            form = step if step != STEP_CHOSEN else (STEP_PER_LIST if per_list <= merged else STEP_MERGED)
            if listed > LIST_CAPACITY:
                form = STEP_MERGED
            forms[side][wp] = form
            if form == STEP_MERGED:
                totals[side][wp] = merged_n
                entries[side].append(slots(side, u))
            else:
                totals[side][wp] = listed
                entries[side].append([s for r in mine for s in slots(side, r)])
    dump = [x for u in unions for x in u] + counts + forms[0] + forms[1] + totals[0] + totals[1]
    for side in (0, 1):
        for wp in range(WARPS):
            e = entries[side][wp]
            dump += e + [full] * (LIST_CAPACITY - len(e))
    return np.array(dump, dtype=np.uint64)


def check_lists(name: str, steps=(STEP_CHOSEN, STEP_PER_LIST, STEP_MERGED)) -> bool:
    spec, words, level_s, sw, sr = cell_words(name)
    widths = tuple(spec.widths)
    w = words.cpu().numpy().view(np.uint32)
    ok = True
    for rs, rd in CARVES:
        per = int(ops().lists_dump_words(rs, rd))
        sblocks, rblocks = widths[level_s] // rs, widths[1 - level_s] // rd
        for step in steps:
            out = ops().lists_part(words, list(widths), level_s, sw, sr, rs, rd, step, 1)
            torch.cuda.synchronize()
            got = out.cpu().numpy().view(np.uint32).astype(np.uint64).reshape(-1, per)
            bad = 0
            windows = spec.tokens // WINDOW
            for bh in range(words.shape[0]):
                for rb in range(rblocks):
                    for sb in range(sblocks):
                        for win in range(windows):
                            cta = (bh * rblocks + rb) * sblocks + sb
                            want = lists_model(w, widths, level_s, sw, sr, rs, rd, step, bh, sb, rb, win)
                            if not np.array_equal(got[cta * windows + win], want):
                                bad += 1
            total = words.shape[0] * rblocks * sblocks * windows
            print(f"lists {name} carve {rs}x{rd} step {step}: {total - bad}/{total} CTA-windows match")
            ok &= bad == 0
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", choices=("lists",), required=True)
    ap.add_argument("--cell", action="append", required=True)
    args = ap.parse_args()
    ok = all(check_lists(cell) for cell in args.cell)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
