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


def cell_bundle(name: str, heads: int = 1, thin: bool = False):
    """``(spec, routes, values, facts, level_s, sparse_write, sparse_read)``: the cell's bundle at ``heads`` heads, the
    facts pass's output (words, list exponents, factor products) and its carve's sides. ``thin`` (a sparse level R)
    zeroes level R's write rows in every third window and all but every eighth token in the next, so the box windows
    hold no pass and one pass -- what no cell declares."""
    from measure.cells import by_name, realize
    from measure.subjects import bundle_of, lists_form_of
    from rola.ops import facts as facts_ops

    spec = by_name(name)
    routes, values, layout, statics = bundle_of(spec, realize(spec, H=heads))
    if thin:
        rows = routes.amp_write[1 - facts_ops.list_level(statics[1])]
        for win in range(rows.shape[1] // WINDOW):
            span = rows[:, win * WINDOW:(win + 1) * WINDOW]
            if win % 3 == 1:
                span.zero_()
            elif win % 3 == 2:
                keep = torch.arange(WINDOW, device=rows.device) % 8 == 0
                span[:, ~keep] = 0
    out = facts_ops.facts(routes, values, layout, *statics, lists_form_of(spec))
    #: the floating draw is gone; at many heads its cached blocks would push the bundle out of device memory
    torch.cuda.empty_cache()
    declared = spec.declared_sparsity()
    level_r = facts_ops.list_level(statics[1])
    level_s = 1 - level_r
    return spec, routes, values, out, level_s, int(declared[level_r][1]), int(declared[level_s][0])


def _fold(routes, values, facts, level_s, sw, sr, ledger=None):
    return ops().fold_part(list(routes.amp_read), list(routes.amp_write), values.v, facts.fm, facts.words,
                           facts.list_exp, level_s, sw, sr, 16, 64, STEP_CHOSEN, ledger)


#: the fold part's ledger phases, `owner_skeleton.cu`'s `FoldPartPhase`
FOLD_PHASES = ("lists", "prologue", "wait", "rows", "loads", "steps", "middle", "tail")


def ledger_fold(name: str, part: str, heads: int) -> None:
    """The fold part's phase ledger on the cell's call: each phase's cycles a warp a CTA-window and its share."""
    spec, routes, values, facts, level_s, sw, sr = cell_bundle(name, heads)
    _fold(routes, values, facts, level_s, sw, sr)
    ledger = torch.zeros(WARPS, len(FOLD_PHASES), dtype=torch.int64, device="cuda")
    _fold(routes, values, facts, level_s, sw, sr, ledger)
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
    """The fold part on the cell, per slot against `fold_reference.fold_state`, its units exactly; on a sparse level R
    the cell thinned too (`cell_bundle`'s ``thin``)."""
    ok = _check_fold(name, thin=False)
    if cell_bundle(name)[5]:
        ok &= _check_fold(name, thin=True)
    return ok


def _check_fold(name: str, thin: bool) -> bool:
    from tests.oracle.fold_reference import COEF_TOP, fold_state, unit_scale

    spec, routes, values, facts, level_s, sw, sr = cell_bundle(name, thin=thin)
    widths = tuple(spec.widths)
    w_s, w_r = widths[level_s], widths[1 - level_s]
    sblocks, rblocks = w_s // 16, w_r // 64
    BH = facts.words.shape[0]
    state, env, mass, units = fold_state(routes, values, facts.list_exp, level_s, WINDOW)
    want_units = torch.tensor(units).view(BH, rblocks, 1, 64, 2)
    st, ms, un = _fold(routes, values, facts, level_s, sw, sr)
    torch.cuda.synchronize()
    un = un.view(BH, 2, rblocks, sblocks, 64, 2).cpu()
    units_ok = bool((un == want_units[:, None]).all())
    scale = torch.exp2(un.double() - COEF_TOP) * unit_scale(routes, level_s)
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
    print(f"fold {name}{' thinned' if thin else ''}: units {'ok' if units_ok else 'DIFFER'}, "
          f"state |d|/env median {ratio.median():.1e} max {ratio.max():.1e} "
          f"{'ok' if state_ok else 'OVER'} (<= {FOLD_RTOL:.0e}), mass median {mratio.median():.1e} "
          f"max {mratio.max():.1e} {'ok' if mass_ok else 'OVER'}")
    return units_ok and state_ok and mass_ok


def cell_words(name: str):
    """``(spec, words, level_s, sparse_write, sparse_read)``."""
    spec, _routes, _values, facts, level_s, sw, sr = cell_bundle(name)
    return spec, facts.words, level_s, sw, sr


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
    forms, totals, entries = [0] * WARPS, [0] * WARPS, []

    def slots(word_list):
        out = []
        for k, bits in enumerate(word_list):
            for b in range(32):
                if bits >> b & 1:
                    out.append(bases[1][k] + _popc(unions[1][k] & ((1 << b) - 1)))
        return out

    lists = rs // WARPS
    for wp in range(WARPS):
        row0 = rd + wp * lists
        if not sparse_read:
            forms[wp], totals[wp] = STEP_MERGED, WINDOW
            entries.append([])
            continue
        mine = [rows[row0 + j] for j in range(lists)]
        u = [0] * WINDOW_WORDS
        for r in mine:
            u = [a | b for a, b in zip(u, r, strict=True)]
        merged_n = sum(_popc(x) for x in u)
        ns = [counts[row0 + j] for j in range(lists)]
        listed, steps = sum(ns), sum((n + 31) >> 5 for n in ns)
        msteps = (merged_n + 31) >> 5
        per_list = steps * (COST_A + COST_B + COST_IMMAS) + listed * COST_ENTRY
        merged = msteps * (COST_B + lists * (COST_A + COST_IMMAS)) + merged_n * COST_ENTRY
        form = step if step != STEP_CHOSEN else (STEP_PER_LIST if per_list <= merged else STEP_MERGED)
        if listed > LIST_CAPACITY:
            form = STEP_MERGED
        forms[wp] = form
        if form == STEP_MERGED:
            totals[wp] = merged_n
            entries.append(slots(u))
        else:
            totals[wp] = listed
            entries.append([s for r in mine for s in slots(r)])
    dump = [x for u in unions for x in u] + counts + forms + totals
    for e in entries:
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


def fill_model(routes, values, facts, w, level_s, sw, sr, rs, rd, bh, cg, sb, rb, win):
    """One CTA-window's write and read pool rows, in slot order, `owner_fill.cuh`'s layout: a write row is level R's
    run, the values, level S's slice and ``(F, m)`` at 16 x 64, level S's run, level R's, the values and ``(F, m)`` at
    32 x 32 (each level's digits in its stored type)."""
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
    s_run = take(routes.amp_write[level_s], wt, s0, s0 + rs)
    r_run = take(routes.amp_write[level_r], wt, r0, r0 + rd).copy()
    width = r_run.shape[1] // rd
    for c in range(rd // 32):
        live = wrows[32 * c:32 * c + 32][:, wt].any(0)
        r_run[~live, width * 32 * c:width * 32 * (c + 1)] = 0
    v = take(values.v, wt, cg * 32, cg * 32 + 32)
    fm = facts.fm[bh, t0 + wt].contiguous().view(torch.uint8).cpu().numpy()
    parts = (r_run, v, s_run, fm) if rs == 16 else (s_run, r_run, v, fm)
    out.append(np.concatenate(parts, 1))
    rt = np.nonzero(rrows.any(0))[0]
    out.append(np.concatenate([take(routes.amp_read[level_r], rt, r0, r0 + rd),
                               take(routes.amp_read[level_s], rt, s0, s0 + rs)], 1))
    return out


def write_row(routes, level_s, rs, rd):
    """A write row as `owner_skeleton.cu` dumps it (`kDumpRow`): level S's and R's digits in the bundle's write types,
    the values, then ``(F, m)``."""
    size = [routes.amp_write[lv].element_size() for lv in (level_s, 1 - level_s)]
    return rs * size[0] + rd * size[1] + 64 + 8


def read_row(rs):
    """`owner_fill.cuh`'s `ReadRows::kRow`."""
    return 96 if rs == 16 else 64


def check_fill(name: str) -> bool:
    spec, routes, values, facts, level_s, sw, sr = cell_bundle(name)
    words = facts.words
    w = words.cpu().numpy().view(np.uint32)
    widths = tuple(spec.widths)
    ok = True
    BH, windows = words.shape[0], spec.tokens // WINDOW
    for rs, rd in CARVES:
        sblocks, rblocks = widths[level_s] // rs, widths[1 - level_s] // rd
        wd, rdump = ops().fill_part(list(routes.amp_read), list(routes.amp_write), values.v, facts.fm, words,
                                    level_s, sw, sr, rs, rd, 1)
        torch.cuda.synchronize()
        wrow, rrow = write_row(routes, level_s, rs, rd), read_row(rs)
        wd = wd.cpu().numpy().reshape(-1, WINDOW, wrow)
        rdump = rdump.cpu().numpy().reshape(-1, WINDOW, rrow)
        bad = total = 0
        for z in range(BH * 2):
            bh, cg = divmod(z, 2)
            for rb in range(rblocks):
                for sb in range(sblocks):
                    cta = (z * rblocks + rb) * sblocks + sb
                    for win in range(windows):
                        want_w, want_r = fill_model(routes, values, facts, w, level_s, sw, sr, rs, rd, bh, cg, sb,
                                                    rb, win)
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
    spec, routes, values, facts, level_s, sw, sr = cell_bundle(name, heads)
    words = facts.words
    if part == "fold":
        for _ in range(RUN_REPS):
            _fold(routes, values, facts, level_s, sw, sr)
        torch.cuda.synchronize()
        print(f"ran {part} on {name}: carve (16, 64), {RUN_REPS} times, BH {words.shape[0]}, L {spec.tokens}")
        return
    for rs, rd in CARVES:
        for _ in range(RUN_REPS):
            if part == "lists":
                ops().lists_part(words, list(spec.widths), level_s, sw, sr, rs, rd, STEP_CHOSEN, 0)
            elif part.startswith("readout"):
                step = {"readout": STEP_CHOSEN, "readout-list": STEP_PER_LIST, "readout-merged": STEP_MERGED}[part]
                ops().readout_part(list(routes.amp_read), list(routes.amp_write), values.v, words, level_s, sw, sr,
                                   rs, rd, step)
            elif part == "drain":
                num = torch.zeros(words.shape[0], spec.tokens, 64, device="cuda")
                den = torch.zeros(words.shape[0], spec.tokens, device="cuda")
                ops().drain_part(words, list(spec.widths), level_s, sw, sr, heads, rs, rd, num, den)
            else:
                ops().fill_part(list(routes.amp_read), list(routes.amp_write), values.v, facts.fm, words, level_s,
                                sw, sr, rs, rd, 0)
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
