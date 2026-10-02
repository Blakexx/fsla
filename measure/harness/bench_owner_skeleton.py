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
    level_s = 1 if declared[1][0] or not declared[0][0] else 0
    level_r = 1 - level_s
    return spec, routes, values, words, level_s, int(declared[level_r][1]), int(declared[level_s][0])


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
    v_hat = take(values.v_hat, wt, cg * 32, cg * 32 + 32)
    v_lo = take(values.v_lo, wt, cg * 32, cg * 32 + 32) if values.v_lo is not None else np.zeros_like(v_hat)
    cols4 = [routes.g_write, values.scale, routes.scale_write[0], routes.scale_write[1]]
    factors = np.stack([c[b, t0 + wt, h].float().cpu().numpy() for c in cols4], 1).astype(np.float32).view(np.uint8)
    run = sparse if rs == 16 else np.concatenate([dense, sparse], 1)
    tail = np.concatenate([dense, factors], 1) if rs == 16 else factors
    out.append(np.concatenate([run, v_hat, v_lo, tail], 1))
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
                                    routes.g_write, values.v_hat, values.v_lo, values.scale, words, level_s, sw, sr,
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
    spec, routes, values, words, level_s, sw, sr = cell_bundle(name, heads)
    for rs, rd in CARVES:
        for _ in range(RUN_REPS):
            if part == "lists":
                ops().lists_part(words, list(spec.widths), level_s, sw, sr, rs, rd, STEP_CHOSEN, 0)
            elif part.startswith("fold"):
                if (rs, rd) != FOLD_CARVE:
                    continue
                step = {"fold": STEP_CHOSEN, "fold-list": STEP_PER_LIST, "fold-merged": STEP_MERGED}[part]
                ops().fold_part(list(routes.amp_read), list(routes.amp_write), list(routes.scale_write),
                                routes.g_write, values.v_hat, values.v_lo, values.scale, words, level_s, sw, sr,
                                rs, rd, step, 0)
            elif part.startswith("readout"):
                step = {"readout": STEP_CHOSEN, "readout-list": STEP_PER_LIST, "readout-merged": STEP_MERGED}[part]
                ops().readout_part(list(routes.amp_read), list(routes.amp_write), list(routes.scale_write),
                                   routes.g_write, values.v_hat, values.v_lo, values.scale, words, level_s, sw, sr,
                                   rs, rd, step)
            elif part == "drain":
                num = torch.zeros(words.shape[0], spec.tokens, 64, device="cuda")
                den = torch.zeros(words.shape[0], spec.tokens, device="cuda")
                ops().drain_part(words, list(spec.widths), level_s, sw, sr, heads, rs, rd, num, den)
            else:
                ops().fill_part(list(routes.amp_read), list(routes.amp_write), list(routes.scale_write),
                                routes.g_write, values.v_hat, values.v_lo, values.scale, words, level_s, sw, sr,
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


#: THE FOLD'S DECLARED ARITHMETIC (the plan card's fold line): two coefficient digits at a power-of-two scale a (list, leaf, pass),
#: each rounded once, the pass's digit sums into the fp32 state -- `owner_fold.cuh`'s, bit for bit.
WRITE_PASS = 128
FOLD_CARVE = (16, 64)
FACTOR_TOP = 32


def _bitlen(x: np.ndarray) -> np.ndarray:
    """``floor(log2 x) + 1``, and 0 at 0, elementwise on non-negative integers below 2^53."""
    out = np.zeros(x.shape, dtype=np.int64)
    nz = x > 0
    out[nz] = np.frexp(x[nz].astype(np.float64))[1]
    return out


def token_factor(g, sw_s, sw_r, vs):
    """``(m, e)``: the token factor ``((g sw_S) sw_R) vs`` in fp32 as a 16-bit mantissa and an exponent in
    ``[-79, 32]``, ``F ~ m 2^e``; below ``2^-64`` (and NaN) zero, above the top saturated."""
    f = ((g.astype(np.float32) * sw_s.astype(np.float32)).astype(np.float32) * sw_r.astype(np.float32)).astype(np.float32)
    f = (f * vs.astype(np.float32)).astype(np.float32)
    bits = f.view(np.uint32).astype(np.int64)
    m = (((bits & 0x7FFFFF) | 0x800000) + 128) >> 8
    e = (bits >> 23) - 142
    carry = m == 65536
    m, e = np.where(carry, 32768, m), np.where(carry, e + 1, e)
    top = e > FACTOR_TOP
    m, e = np.where(top, 65535, m), np.where(top, FACTOR_TOP, e)
    zero = ~(f >= np.float32(2.0 ** -64))
    return np.where(zero, 0, m), np.where(zero, 0, e)


def _rne_shift(p: np.ndarray, sh: np.ndarray) -> np.ndarray:
    """``RNE(p / 2^sh)`` for ``sh >= 1``, elementwise int64."""
    sh = np.minimum(sh, 40)
    q = p >> sh
    r = p - (q << sh)
    half = np.left_shift(1, sh - 1)
    return q + ((r > half) | ((r == half) & ((q & 1) == 1)))


def fold_model(routes, values, w, level_s, sw, bh, cg, sb, rb):
    """One CTA's fp32 fold state after every window, ``[lists, leaves, 32]``, passes of `WRITE_PASS` slots of each
    window's write union."""
    rs, rd = FOLD_CARVE
    level_r = 1 - level_s
    b, h = divmod(bh, routes.heads)
    s0, r0 = sb * rs, rb * rd
    u_wd = routes.amp_write[level_s][b, :, h, s0:s0 + rs].cpu().numpy().astype(np.int64)          # [T, 16]
    u_ws = routes.amp_write[level_r][b, :, h, r0:r0 + rd].cpu().numpy().astype(np.int64)          # [T, 64]

    def col(x):
        return x[b, :, h].float().cpu().numpy()

    m_f, e_f = token_factor(col(routes.g_write), col(routes.scale_write[level_s]), col(routes.scale_write[level_r]),
                            col(values.scale))
    v_hat = values.v_hat[b, :, h, cg * 32:cg * 32 + 32].cpu().numpy().astype(np.int64)            # [T, 32]
    v_lo = (values.v_lo[b, :, h, cg * 32:cg * 32 + 32].cpu().numpy().astype(np.int64) if values.v_lo is not None
            else np.zeros_like(v_hat))
    X = np.zeros((rd, rs, 32), dtype=np.float32)
    row_base = (0, routes.topology.widths[0])

    for win in range(u_wd.shape[0] // WINDOW):
        cols = slice(WINDOW_WORDS * win, WINDOW_WORDS * (win + 1))
        if sw:
            live = np.zeros(WINDOW, dtype=bool)
            for j in range(rd):
                live |= _bits(w[bh, 1, row_base[level_r] + r0 + j, cols])
            tokens = np.nonzero(live)[0] + win * WINDOW
        else:
            tokens = np.arange(win * WINDOW, (win + 1) * WINDOW)
        for first in range(0, len(tokens), WRITE_PASS):
            t = tokens[first:first + WRITE_PASS]
            c = u_ws[t] * m_f[t, None]                                                   # [n, 64]
            p = u_wd[t][:, None, :] * c[:, :, None]                                      # [n, 64, 16]
            ok = p > 0
            L = _bitlen(u_wd[t])[:, None, :] + (_bitlen(c) + e_f[t, None])[:, :, None]
            K = np.where(ok, L, np.iinfo(np.int64).min).max(0)
            K = np.where(ok.any(0), K, 0)                                                # [64, 16]
            sh = K[None] - 16 - e_f[t, None, None]
            assert (sh[ok] >= 1).all()
            A = np.where(ok, np.minimum(_rne_shift(p, np.where(ok, sh, 1)), 65535), 0)
            hi, lo = A >> 8, A & 255

            vh, vl = v_hat[t], v_lo[t]
            hh = np.einsum("nsd,nc->sdc", hi, vh)
            mid = np.einsum("nsd,nc->sdc", hi, vl) + np.einsum("nsd,nc->sdc", lo, vh)
            ll = np.einsum("nsd,nc->sdc", lo, vl)
            hm = hh * 256 + mid
            assert np.abs(hm).max(initial=0) < 2 ** 31 and ll.max(initial=0) < 2 ** 24
            low = np.ldexp(np.float32(1), K - 16).astype(np.float32)[..., None]
            high = np.ldexp(np.float32(1), K - 8).astype(np.float32)[..., None]
            X = (X + ll.astype(np.float32) * low).astype(np.float32)
            X = (X + hm.astype(np.float32) * high).astype(np.float32)
    return X


def check_fold(name: str, steps=(STEP_CHOSEN, STEP_PER_LIST, STEP_MERGED)) -> bool:
    """Every CTA's state after the call, against `fold_model`, bit for bit, under each step form."""
    spec, routes, values, words, level_s, sw, sr = cell_bundle(name)
    w = words.cpu().numpy().view(np.uint32)
    widths = tuple(spec.widths)
    rs, rd = FOLD_CARVE
    sblocks, rblocks = widths[level_s] // rs, widths[1 - level_s] // rd
    BH = words.shape[0]
    want = {}
    for z in range(BH * 2):
        bh, cg = divmod(z, 2)
        for rb in range(rblocks):
            for sb in range(sblocks):
                want[z, rb, sb] = fold_model(routes, values, w, level_s, sw, bh, cg, sb, rb)
    ok = True
    for step in steps:
        got = ops().fold_part(list(routes.amp_read), list(routes.amp_write), list(routes.scale_write),
                              routes.g_write, values.v_hat, values.v_lo, values.scale, words, level_s, sw, sr,
                              rs, rd, step, 1)
        torch.cuda.synchronize()
        got = got.cpu().numpy().reshape(-1, rd, rs, 32)
        bad = worst = 0
        for (z, rb, sb), x in want.items():
            y = got[(z * rblocks + rb) * sblocks + sb]
            if not np.array_equal(y.view(np.uint32), x.view(np.uint32)):
                bad += 1
                worst = max(worst, int((y.view(np.uint32) != x.view(np.uint32)).sum()))
        print(f"fold {name} step {step}: {len(want) - bad}/{len(want)} CTAs match bit for bit"
              + (f" (worst CTA: {worst} elements differ)" if bad else ""))
        ok &= bad == 0
    return ok


#: the fold part's phase ledger, `owner_skeleton.cu`'s order: the fold's own phases, then the part's
FOLD_PHASES = ("head", "scale", "first step", "bursts", "state add", "lists", "fill", "token factors", "pass sync")


def ledger_fold(name: str, heads: int) -> None:
    """The fold part's phase ledger on the cell: each phase's cycles a warp a window, and its share of the warp's."""
    spec, routes, values, words, level_s, sw, sr = cell_bundle(name, heads)
    rs, rd = FOLD_CARVE
    led = ops().fold_part(list(routes.amp_read), list(routes.amp_write), list(routes.scale_write), routes.g_write,
                          values.v_hat, values.v_lo, values.scale, words, level_s, sw, sr, rs, rd, STEP_CHOSEN, 2)
    torch.cuda.synchronize()
    led = led.cpu().numpy().astype(np.float64)                         # [ctas, warps, phases]
    windows = spec.tokens // WINDOW
    per = led.sum(axis=(0, 1)) / (led.shape[0] * led.shape[1] * windows)
    total = per.sum()
    print(f"fold part phase ledger, {name} x {heads} heads: {total:,.0f} cycles a warp a window")
    for ph, c in zip(FOLD_PHASES, per):
        print(f"  {ph:14s} {c:10,.0f}  {100 * c / total:5.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", choices=("lists", "fill", "drain", "fold"))
    ap.add_argument("--run", choices=("lists", "fill", "drain", "fold", "fold-list", "fold-merged", "readout", "readout-list",
                                         "readout-merged"), help="the part's launches alone, for ncu")
    ap.add_argument("--cell", action="append", required=True)
    ap.add_argument("--heads", type=int, default=1, help="heads under --run: enough to fill the machine")
    ap.add_argument("--ledger", choices=("fold",), help="the part's phase ledger on each cell (at --heads)")
    args = ap.parse_args()
    if args.ledger:
        for cell in args.cell:
            ledger_fold(cell, args.heads)
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
