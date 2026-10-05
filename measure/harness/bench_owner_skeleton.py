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
#: the read side's cost model, `owner_lists.cuh`'s (the fold side is merged)
COST_A, COST_B, COST_IMMAS, COST_ENTRY = 4, 12, 8 * 16, 6


def ops():
    try:
        from rola_cu13 import _C_parts  # noqa: F401 -- registers torch.ops.rola_owner
    except ImportError as ex:
        raise SystemExit("the part harness's driver is not built: ROLA_BUILD_PARTS=1 ROLA_CUDA_ARCHS=86 "
                         "python -m pip install -e . --no-build-isolation") from ex
    return torch.ops.rola_owner


def cell_bundle(name: str, heads: int = 1):
    """``(spec, routes, values, words, level_s, sparse_write, sparse_read)``: the cell's bundle at ``heads`` heads, its
    facts words and its carve's sides."""
    from measure.cells import by_name, realize
    from measure.subjects import bundle_of
    from rola.ops import facts as facts_ops

    spec = by_name(name)
    routes, values, layout, statics = bundle_of(spec, realize(spec, H=heads))
    words = facts_ops.facts(routes, values, layout, *statics).words
    #: the floating draw is gone; at many heads its cached blocks would push the bundle out of device memory
    torch.cuda.empty_cache()
    declared = spec.declared_sparsity()
    level_r = facts_ops.list_level(statics[0])
    level_s = 1 - level_r
    return spec, routes, values, words, level_s, int(declared[level_r][1]), int(declared[level_s][0])


def fold_bundle(name: str, heads: int = 1):
    """``(spec, routes, values, facts, level_s, sparse_write, sparse_read, floor_s)``: `cell_bundle`'s, with the facts
    pass's whole output (the list exponents beside the words) and whether level S's write side is dense (its floor)."""
    from measure.cells import by_name, realize
    from measure.subjects import bundle_of
    from rola.ops import facts as facts_ops

    spec = by_name(name)
    routes, values, layout, statics = bundle_of(spec, realize(spec, H=heads))
    out = facts_ops.facts(routes, values, layout, *statics)
    torch.cuda.empty_cache()
    declared = spec.declared_sparsity()
    level_r = facts_ops.list_level(statics[0])
    level_s = 1 - level_r
    return (spec, routes, values, out, level_s, int(declared[level_r][1]), int(declared[level_s][0]),
            not declared[level_s][1])


def _fold(routes, values, facts, level_s, sw, sr, floor_s, ledger=None):
    return ops().fold_part(list(routes.amp_read), list(routes.amp_write), list(routes.scale_write), routes.g_write,
                           values.v, values.scale, facts.words, facts.list_exp, level_s, sw, sr, int(floor_s),
                           16, 64, STEP_CHOSEN, ledger)


#: the fold part's ledger phases, `owner_skeleton.cu`'s `FoldPartPhase`
FOLD_PHASES = ("lists", "fill", "stage", "head", "first", "steps", "tail", "sync")


def ledger_fold(name: str, part: str, heads: int) -> None:
    """The fold part's phase ledger on the cell's call: each phase's cycles a warp a CTA-window and its share."""
    spec, routes, values, facts, level_s, sw, sr, floor_s = fold_bundle(name, heads)
    _fold(routes, values, facts, level_s, sw, sr, floor_s)
    ledger = torch.zeros(WARPS, len(FOLD_PHASES), dtype=torch.int64, device="cuda")
    _fold(routes, values, facts, level_s, sw, sr, floor_s, ledger)
    torch.cuda.synchronize()
    widths = tuple(spec.widths)
    ctas = facts.words.shape[0] * 2 * (widths[level_s] // 16) * (widths[1 - level_s] // 64)
    per = ledger.sum(0).double() / (WARPS * ctas * (spec.tokens // WINDOW))
    total = float(per.sum())
    print(f"ledger {part} {name}: cycles a warp a CTA-window " + ", ".join(
        f"{ph} {float(c):.0f} ({float(c) / total:.0%})" for ph, c in zip(FOLD_PHASES, per)))


#: THE FOLD GATE'S BOUND, per slot against the declared arithmetic's envelope: fp32 accumulate, an HMMA a 16-token
#: step, a few dozen a slot
FOLD_RTOL = 2.0 ** -14


def check_fold(name: str) -> bool:
    """The fold part on the cell, per slot against `fold_reference.fold_state`; its units exactly."""
    from tests.oracle.fold_reference import COEF_TOP, UNIT_SCALE, fold_state

    spec, routes, values, facts, level_s, sw, sr, floor_s = fold_bundle(name)
    widths = tuple(spec.widths)
    w_s, w_r = widths[level_s], widths[1 - level_s]
    sblocks, rblocks = w_s // 16, w_r // 64
    BH = facts.words.shape[0]
    state, env, mass, units = fold_state(routes, values, facts.list_exp, level_s, floor_s, not sw, WINDOW)
    want_units = torch.tensor(units).view(BH, rblocks, 1, 64, 2)
    st, ms, un = _fold(routes, values, facts, level_s, sw, sr, floor_s)
    torch.cuda.synchronize()
    un = un.view(BH, 2, rblocks, sblocks, 64, 2).cpu()
    units_ok = bool((un == want_units[:, None]).all())
    scale = torch.exp2(un.double() - COEF_TOP) * UNIT_SCALE
    got = (st.view(BH, 2, rblocks, sblocks, 16, 64, 32).double().cpu() * scale[..., 0][:, :, :, :, None, :, None])
    got = got.permute(0, 3, 4, 2, 5, 1, 6).reshape(BH, w_s, w_r, 64)
    diff, e = (got - state.cpu()).abs(), env.cpu()
    ratio = (diff / e.clamp_min(1e-300))[e > 0]
    state_ok = bool((diff <= FOLD_RTOL * e).all())
    gm = ms.view(BH, 2, rblocks, sblocks, 16, 64).double().cpu() * scale[..., 1][:, :, :, :, None, :]
    gm = gm.permute(0, 1, 3, 4, 2, 5).reshape(BH, 2, w_s, w_r)
    mdiff, mw = (gm - mass.cpu()[:, None]).abs(), mass.cpu()[:, None].expand_as(gm)
    mratio = (mdiff / mw.clamp_min(1e-300))[mw > 0]
    mass_ok = bool((mdiff <= FOLD_RTOL * mw).all())
    print(f"fold {name}: units {'ok' if units_ok else 'DIFFER'}, "
          f"state |d|/env median {ratio.median():.1e} max {ratio.max():.1e} "
          f"{'ok' if state_ok else 'OVER'} (<= {FOLD_RTOL:.0e}), mass median {mratio.median():.1e} "
          f"max {mratio.max():.1e} {'ok' if mass_ok else 'OVER'}")
    return units_ok and state_ok and mass_ok


def cell_words(name: str):
    """``(spec, words, level_s, sparse_write, sparse_read)``."""
    spec, _routes, _values, words, level_s, sw, sr = cell_bundle(name)
    return spec, words, level_s, sw, sr


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
            if listed > LIST_CAPACITY or side == 0:
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


def _bits(words_row) -> np.ndarray:
    """A window's 16 words as 512 booleans, token order."""
    w = np.asarray(words_row, dtype=np.uint32)
    return ((w[:, None] >> np.arange(32, dtype=np.uint32)) & 1).astype(bool).reshape(-1)


def fill_model(routes, values, w, level_s, sw, sr, rs, rd, bh, cg, sb, rb, win):
    """One CTA-window's write and read pool rows, in slot order, `owner_fill.cuh`'s layout."""
    level_r = 1 - level_s
    H = routes.heads
    b, h = divmod(bh, H)
    row_base = (0, routes.topology.widths[0])
    cols = slice(WINDOW_WORDS * win, WINDOW_WORDS * (win + 1))
    s0, r0 = sb * rs, rb * rd
    if sw:
        wrows = np.stack([_bits(w[bh, 1, row_base[level_r] + r0 + j, cols]) for j in range(rd)])
    else:
        wrows = np.ones((rd, WINDOW), dtype=bool)
    if sr:
        rrows = np.stack([_bits(w[bh, 0, row_base[level_s] + s0 + i, cols]) for i in range(rs)])
    else:
        rrows = np.ones((rs, WINDOW), dtype=bool)
    out = []
    t0 = win * WINDOW

    def take(x, tokens, lo, hi):
        return x[b, t0 + tokens, h, lo:hi].contiguous().view(torch.uint8).cpu().numpy()

    wt = np.nonzero(wrows.any(0))[0]
    dense = take(routes.amp_write[level_s], wt, s0, s0 + rs)
    sparse = take(routes.amp_write[level_r], wt, r0, r0 + rd).copy()
    for c in range(rd // 32):
        live = wrows[32 * c:32 * c + 32][:, wt].any(0)
        sparse[~live, 32 * c:32 * c + 32] = 0
    v = take(values.v, wt, cg * 32, cg * 32 + 32)
    cols4 = [routes.g_write, values.scale, routes.scale_write[0], routes.scale_write[1]]
    factors = np.stack([c[b, t0 + wt, h].float().cpu().numpy() for c in cols4], 1).astype(np.float32).view(np.uint8)
    run = sparse if rs == 16 else np.concatenate([dense, sparse], 1)
    tail = np.concatenate([dense, factors], 1) if rs == 16 else factors
    out.append(np.concatenate([run, v, tail], 1))
    rt = np.nonzero(rrows.any(0))[0]
    out.append(np.concatenate([take(routes.amp_read[level_r], rt, r0, r0 + rd),
                               take(routes.amp_read[level_s], rt, s0, s0 + rs)], 1))
    return out


def pool_rows(rs):
    """``(write row bytes, read row bytes)``, `owner_fill.cuh`'s pitches."""
    return 160, (96 if rs == 16 else 64)


def check_fill(name: str) -> bool:
    spec, routes, values, words, level_s, sw, sr = cell_bundle(name)
    w = words.cpu().numpy().view(np.uint32)
    widths = tuple(spec.widths)
    ok = True
    BH, windows = words.shape[0], spec.tokens // WINDOW
    for rs, rd in CARVES:
        sblocks, rblocks = widths[level_s] // rs, widths[1 - level_s] // rd
        wd, rdump = ops().fill_part(list(routes.amp_read), list(routes.amp_write), list(routes.scale_write),
                                    routes.g_write, values.v, values.scale, words, level_s, sw, sr,
                                    rs, rd, 1)
        torch.cuda.synchronize()
        wrow, rrow = pool_rows(rs)
        wd = wd.cpu().numpy().reshape(-1, WINDOW, wrow)
        rdump = rdump.cpu().numpy().reshape(-1, WINDOW, rrow)
        bad = total = 0
        for z in range(BH * 2):
            bh, cg = divmod(z, 2)
            for rb in range(rblocks):
                for sb in range(sblocks):
                    cta = (z * rblocks + rb) * sblocks + sb
                    for win in range(windows):
                        want_w, want_r = fill_model(routes, values, w, level_s, sw, sr, rs, rd, bh, cg, sb, rb, win)
                        i = cta * windows + win
                        total += 1
                        if not (np.array_equal(wd[i, :len(want_w), :want_w.shape[1]], want_w)
                                and np.array_equal(rdump[i, :len(want_r), :want_r.shape[1]], want_r)):
                            bad += 1
        print(f"fill {name} carve {rs}x{rd}: {total - bad}/{total} CTA-windows match")
        ok &= bad == 0
    return ok


#: launches a carve under `--run`: ncu reads one, the rest show its spread
RUN_REPS = 3


def run_part(name: str, part: str, heads: int) -> None:
    """The part on the cell's whole call at ``heads`` heads, no dump, each carve `RUN_REPS` times: the launches an
    ncu run reads."""
    if part == "fold":
        spec, routes, values, facts, level_s, sw, sr, floor_s = fold_bundle(name, heads)
        for _ in range(RUN_REPS):
            _fold(routes, values, facts, level_s, sw, sr, floor_s)
        torch.cuda.synchronize()
        print(f"ran {part} on {name}: carve (16, 64), {RUN_REPS} times, BH {facts.words.shape[0]}, L {spec.tokens}")
        return
    spec, routes, values, words, level_s, sw, sr = cell_bundle(name, heads)
    for rs, rd in CARVES:
        for _ in range(RUN_REPS):
            if part == "lists":
                ops().lists_part(words, list(spec.widths), level_s, sw, sr, rs, rd, STEP_CHOSEN, 0)
            elif part.startswith("readout"):
                step = {"readout": STEP_CHOSEN, "readout-list": STEP_PER_LIST, "readout-merged": STEP_MERGED}[part]
                ops().readout_part(list(routes.amp_read), list(routes.amp_write), list(routes.scale_write),
                                   routes.g_write, values.v, values.scale, words, level_s, sw, sr,
                                   rs, rd, step)
            elif part == "drain":
                num = torch.zeros(words.shape[0], spec.tokens, 64, device="cuda")
                den = torch.zeros(words.shape[0], spec.tokens, device="cuda")
                ops().drain_part(words, list(spec.widths), level_s, sw, sr, heads, rs, rd, num, den)
            else:
                ops().fill_part(list(routes.amp_read), list(routes.amp_write), list(routes.scale_write),
                                routes.g_write, values.v, values.scale, words, level_s, sw, sr,
                                rs, rd, 0)
    torch.cuda.synchronize()
    print(f"ran {part} on {name}: carves {CARVES}, {RUN_REPS} each, BH {words.shape[0]}, L {spec.tokens}")


def check_drain(name: str) -> bool:
    """A panel of ones drained: each token's `num` channels and `den` are the count of boxes it is read-live in."""
    spec, words, level_s, sw, sr = cell_words(name)
    widths, L, BH = tuple(spec.widths), spec.tokens, words.shape[0]
    bits = _all_bits(words)
    level_r = 1 - level_s
    ok = True
    for rs, rd in CARVES:
        num = torch.zeros(BH, L, 64, device="cuda")
        den = torch.zeros(BH, L, device="cuda")
        ops().drain_part(words, list(widths), level_s, sw, sr, BH, rs, rd, num, den)
        torch.cuda.synchronize()
        rows = bits[:, 0, widths[0] * level_s: widths[0] * level_s + widths[level_s]] if sr else None
        boxes = np.zeros((BH, L))
        for sb in range(widths[level_s] // rs):
            live = rows[:, sb * rs:(sb + 1) * rs].any(1) if sr else np.ones((BH, L), bool)
            boxes += live * (widths[level_r] // rd)
        want = torch.tensor(boxes, dtype=torch.float32)
        good = (torch.equal(num.cpu(), want[..., None].expand(-1, -1, 64).contiguous())
                and torch.equal(den.cpu(), want))
        print(f"drain {name} carve {rs}x{rd}: {'match' if good else 'MISMATCH'}")
        ok &= good
    return ok


def _all_bits(words):
    """The words as booleans, ``[BH, 2, rows, L]``."""
    w = words.cpu().numpy().view(np.uint32)
    return ((w[..., None] >> np.arange(32, dtype=np.uint32)) & 1).astype(bool).reshape(*w.shape[:3], -1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", choices=("lists", "fill", "drain", "fold"))
    ap.add_argument("--run", choices=("lists", "fill", "drain", "readout", "readout-list", "readout-merged", "fold"),
                    help="the part's launches alone, for ncu")
    ap.add_argument("--ledger", choices=("fold",), help="the fold part's phase ledger")
    ap.add_argument("--cell", action="append", required=True)
    ap.add_argument("--heads", type=int, default=1, help="heads under --run: enough to fill the machine")
    args = ap.parse_args()
    if args.ledger:
        for cell in args.cell:
            ledger_fold(cell, args.ledger, args.heads)
        return 0
    if args.run:
        for cell in args.cell:
            run_part(cell, args.run, args.heads)
        return 0
    check = {"lists": check_lists, "fill": check_fill, "drain": check_drain, "fold": check_fold}[args.check]
    ok = all(check(cell) for cell in args.cell)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
