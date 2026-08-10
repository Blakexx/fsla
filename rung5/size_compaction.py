"""What per-side compaction would buy, SIZED FROM THE TABLE THIS RUNG BUILT.

    python size_compaction.py [L] [B] [H]

The union table is the structure compaction consumes: per SIDE, per token, one
bit saying whether this block holds work for that token on that side. So the
saving can be measured before the kernel exists, and measured EXACTLY rather than
estimated -- this walks the same tokens in the same order the cook would.

WHAT IS COUNTED. Today a block visits a whole GROUP of `C` tokens whenever any
one of them is live on either side, and both panels are `C` tokens wide. Under
per-side compaction it walks tokens in order, keeps a read list and a write list,
and closes a SUPER-CHUNK when either list reaches `C` -- so the visit count falls
from `live groups` to `super-chunks` and every mma term falls with it, because
every term's token axis is one of the two lists.

  * `read live` / `write live`   the per-side token density inside the block
  * `idle`                       the fraction of a live group's token slots that
                                 carry no work on the side that reads them --
                                 rung 3 measured 76-98% and this is the same
                                 quantity per side
  * `visits`                     live groups today, super-chunks after
  * `mma x`                      the ratio of the two, which is the factor every
                                 mma term is multiplied by
  * `staircase`                  the fraction of the diagonal tile the monotone
                                 mask keeps, against today's triangle at 0.5 --
                                 the intra term's own correction

THE BUFFERS ARE HELD EQUAL AT `C` HERE. The design sizes them ASYMMETRICALLY (a
read entry is panel rows plus one byte, a write entry carries a whole value row),
so a longer read buffer would close fewer super-chunks than this reports. Equal
buffers are therefore the CONSERVATIVE reading of the saving.
"""
from __future__ import annotations

import sys

import torch

ROOT = __file__.rsplit("/", 2)[0]
sys.path.insert(0, ROOT)
from rung5 import build, reference  # noqa: E402

L = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
B = int(sys.argv[2]) if len(sys.argv) > 2 else 1
H = int(sys.argv[3]) if len(sys.argv) > 3 else 4
FLAG = (64, 64)
dev = "cuda"


def side_bits(pr, pw, widths, bc):
    """(read, write) [BH, blocks, L] bool, straight out of the device table."""
    v = torch.empty(pr.shape[0], pr.shape[1], pr.shape[2], 64,
                    device=pr.device, dtype=torch.bfloat16)
    tbl = build.union_table(pr, pw, v, 32, bc, *widths)
    bits = reference.union_unpack(tbl, pr.shape[2])
    plan, lead_span, tail_span = block_of(widths, bc)
    w0, w1 = widths
    lead_entries = 0 if lead_span == w0 else w0 // lead_span
    tail_groups = w1 // tail_span
    out = []
    for s in range(2):
        rows = []
        for k in range(len(plan)):
            if bits.shape[2] == 0:
                rows.append(torch.ones(bits.shape[0], L, dtype=torch.bool, device=dev))
                continue
            r = bits[:, s, k // tail_groups]
            if tail_span < w1:
                r = r & bits[:, s, lead_entries + (k % tail_groups)]
            rows.append(r)
        out.append(torch.stack(rows, 1))
    return out


def block_of(widths, bc):
    return reference.block_plan(widths, bc)


def super_chunks(rl, wl, C):
    """Greedy super-chunk count per row, exactly as the cook would close them.

    Vectorised over rows: every row carries its own cursor and they step
    together, so the loop runs once per super-chunk rather than once per token.
    """
    n, T = rl.shape
    cr = torch.cumsum(rl.int(), 1)
    cw = torch.cumsum(wl.int(), 1)
    pos = torch.zeros(n, dtype=torch.long, device=rl.device)
    base_r = torch.zeros(n, dtype=torch.int32, device=rl.device)
    base_w = torch.zeros(n, dtype=torch.int32, device=rl.device)
    count = torch.zeros(n, dtype=torch.long, device=rl.device)
    idx = torch.arange(n, device=rl.device)
    alive = pos < T
    while bool(alive.any()):
        # the first token at which either list would exceed C
        tr = torch.searchsorted(cr, (base_r + C).unsqueeze(1), right=True).squeeze(1)
        tw = torch.searchsorted(cw, (base_w + C).unsqueeze(1), right=True).squeeze(1)
        nxt = torch.maximum(torch.minimum(tr, tw), pos + 1).clamp(max=T)
        count = count + alive.long()
        j = (nxt - 1).clamp(min=0)
        base_r = torch.where(alive, cr[idx, j], base_r)
        base_w = torch.where(alive, cw[idx, j], base_w)
        pos = torch.where(alive, nxt, pos)
        alive = pos < T
    return count


def staircase_kept(rl, wl, C):
    """Mean kept fraction of a super-chunk's diagonal tile, over a sample."""
    keep, tot = 0, 0
    for row in range(min(rl.shape[0], 8)):
        r = rl[row].nonzero().squeeze(1)
        w = wl[row].nonzero().squeeze(1)
        n = min(len(r), len(w), 4 * C)
        if n < 2:
            continue
        r, w = r[:n], w[:n]
        j = torch.searchsorted(w, r, right=True)          # J(i), the threshold
        keep += int(j.sum())
        tot += n * n
    return keep / tot if tot else float("nan")


def main():
    torch.manual_seed(0)
    print(f"cell B={B} H={H} L={L}, N=4096, buffers held equal at C\n")
    print(f"{'regime':>8s}{'BC':>5s}{'C':>4s}{'read live':>11s}{'write live':>12s}"
          f"{'groups live':>13s}{'idle r':>8s}{'idle w':>8s}"
          f"{'visits now':>12s}{'compacted':>11s}{'mma x':>8s}{'stair':>8s}")
    for regime in ("dense", "lead4", "split", "d4"):
        pr, pw = reference.regime_amplitudes(B, H, L, regime, dev, seed=5)
        for bc in (32, 64, 128):
            rd, wr = side_bits(pr, pw, FLAG, bc)
            rl = rd.reshape(-1, L)
            wl = wr.reshape(-1, L)
            for C in (16, 32, 64):
                grp = (rl | wl).reshape(-1, L // C, C).any(-1)
                live = int(grp.sum())
                sc = int(super_chunks(rl, wl, C).sum())
                fr = rl.float().mean().item()
                fw = wl.float().mean().item()
                # idle: of the token slots a LIVE group presents, how many carry
                # no work on that side
                slots = live * C
                idle_r = 1 - int(rl.sum()) / slots if slots else float("nan")
                idle_w = 1 - int(wl.sum()) / slots if slots else float("nan")
                st = staircase_kept(rl, wl, C)
                print(f"{regime:>8s}{bc:5d}{C:4d}{100*fr:10.2f}%{100*fw:11.2f}%"
                      f"{100*live/(rl.shape[0]*L/C):12.2f}%{100*idle_r:7.1f}%"
                      f"{100*idle_w:7.1f}%{live:12d}{sc:11d}{live/max(sc,1):8.2f}"
                      f"{st:8.3f}", flush=True)
        print()


if __name__ == "__main__":
    main()
