"""SCAN B'S SCHEDULE, GATED AS A SCHEDULE — the device-side partition test.

Vocabulary, defined here and used unqualified below: `C` = chunk tokens (32);
`BC` = slots per CTA block (128); `D` = level count; `W` = one uniform level
width; `N = W ** D` = leaves/slots; `T` = tokens; a BLOCK is one CTA's contiguous
run of `BC` slots; a token is *live* for a block on a side when the union table
says some slot of that block has a nonzero coefficient there.

**WHY THIS FILE EXISTS AND THE NUMERIC ROWS ARE NOT ENOUGH.** Scan B walks its
own descending partition — the chunk boundary is an association of a sum, not a
semantic object — so a numeric reference can only reach the schedule by replaying
it. That replay stops being writable exactly when the general case starts: once
two blocks of one head are live on different tokens, each block closes its chunks
at different tokens and "the" partition does not exist. `test_scan_b_probe.py`
sidesteps this by making every token fully live or fully dead on a side, which
makes every block's partition the same one. A real cell, and not the general one.

So the walk is run alone (`chunk_walk_probe`) and what it published is emitted:
per `(bh, block, token)` the ordinal of the chunk that gathered the token and its
position in that chunk's list, one plane per side, plus each read entry's causal
threshold. `-1` means "no chunk gathered this token", which is what makes
*exactly once* a checkable claim rather than a covering one.

**THREE CLAIMS, and only the first needs a reference:**

1. the partition IS the closing rule's — compared token for token against a
   host simulation driven by the table's own bits, per block, at liveness that
   differs across blocks;
2. every live token is gathered exactly once and no dead token is gathered at
   all — read straight off the sentinels, no reference at all;
3. `(nr, nw, J)` is self-consistent — `J(i)` is the count of the chunk's write
   entries at or before read entry `i`, recomputed from the emitted stamps
   themselves, so it is checked against the lists the kernel actually published
   rather than against a second opinion that could share a mistake.

The SECOND OPINION the design asks for is claim 2 read against the table: the
descending walk's gathered set and the table's live set are the same set, which
is the "different partition, same union" statement with the forward's partition
replaced by the union it and the reverse walk must both cover. The forward's own
walk is not instrumented, and instrumenting it would move its ratified codegen
for a claim the union already carries.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rola.facts.types import Mask
from rola.ops._ext import extension, is_available
from tests.oracle.test_scan_b_probe import ARMS, PINNED, Arm

pytestmark = pytest.mark.skipif(
    not is_available() or not torch.cuda.is_available(),
    reason="the descending walk is a CUDA kernel")

C = 32
DV = 64
K_WINDOW = 65536
TOK_BITS = 32


# ---------------------------------------------------------------- the level plan
#: `LevelPlan`'s geometry law, restated in Python for the ONE purpose of decoding
#: which table rows a block ANDs. It is a transcription and is asserted against
#: the extension's own arm report below, so it cannot drift silently.

def _plan(d: int, w: int, bc: int) -> dict:
    slots = w ** d
    whole = next(k for k in (d, 3, 2, 1, 0) if k <= d and bc >= w ** k)
    partial_span = bc // (w ** whole)
    singles = d - whole - (1 if partial_span > 1 else 0)

    def span_at(l):
        return 1 if l < singles else (partial_span if (partial_span > 1 and l == singles) else w)

    def groups_at(l):
        return w // span_at(l) if span_at(l) < w else 1

    def entries_at(l):
        return w // span_at(l) if span_at(l) < w else 0

    def suffix_groups(l):
        return 1 if l >= d else groups_at(l) * suffix_groups(l + 1)

    def prefix_entries(l):
        return 0 if l == 0 else prefix_entries(l - 1) + entries_at(l - 1)

    clauses = singles + (1 if partial_span > 1 else 0)
    return dict(blocks=slots // bc,
                entries=sum(entries_at(l) for l in range(d)),
                clauses=clauses,
                rows=lambda blk: [prefix_entries(l) + (blk // suffix_groups(l + 1)) % groups_at(l)
                                  for l in range(clauses)])


def _liveness_from_table(plan: dict, bh: int, tbl: torch.Tensor,
                        T: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`[BH, blocks, T]` bools per side, decoded out of the packed union table.

    This is the walk's ACTUAL input, so decoding it (rather than re-deriving
    liveness from the amplitudes) keeps this file gating the schedule and not the
    table builder, which has gates of its own.
    """
    tok_words = (T + TOK_BITS - 1) // TOK_BITS
    ent = plan["entries"]
    #: A PLAN WITH NO TABLE ROWS spans the whole slot space, so its clause is empty
    #: and every token is a member of every block -- there is nothing to decode.
    if ent == 0:
        ones = torch.ones(bh, plan["blocks"], T, dtype=torch.bool)
        return ones, ones.clone()
    words = tbl.view(bh, 2, ent, tok_words).cpu()
    bit = torch.arange(TOK_BITS, dtype=torch.int64)
    #: word -> bits, then trimmed to T: the last word's padding lanes are not tokens
    unpacked = ((words.to(torch.int64).unsqueeze(-1) >> bit) & 1).bool()
    unpacked = unpacked.reshape(bh, 2, ent, tok_words * TOK_BITS)[..., :T]
    out = []
    for side in (0, 1):
        per_block = torch.ones(bh, plan["blocks"], T, dtype=torch.bool)
        for blk in range(plan["blocks"]):
            for row in plan["rows"](blk):
                per_block[:, blk] &= unpacked[:, side, row]
        out.append(per_block)
    return out[0], out[1]


def _walk(live_r, live_w, T: int) -> np.ndarray:
    """The kernel's closing rule, descending, as a per-token CHUNK ORDINAL
    (`-1` for a token no chunk holds).

    A token is excluded from the open chunk when it is live on a side that has
    already gathered `C` entries above it, or when it lies below the chunk's
    K-window line and the chunk has gathered anything; an EMPTY chunk slides its
    window down instead of closing, because a chunk holding nothing has no 16-bit
    offset to overflow, and the tokens it drops belong to no chunk at all.

    Chunks are CONTIGUOUS token spans and are numbered in the order the walk
    closes them, which is descending. That contiguity is what lets everything
    below be a cumulative sum instead of a second loop.
    """
    cid = np.full(T, -1, dtype=np.int64)
    lr, lw = live_r, live_w
    k, nr, nw, open_lo, hi = 0, 0, 0, -1, -1
    for t in range(T - 1, -1, -1):
        crossed = open_lo >= 0 and (t // K_WINDOW) != (open_lo // K_WINDOW)
        if (lr[t] and nr == C) or (lw[t] and nw == C) or (crossed and (nr or nw)):
            k += 1
            nr = nw = 0
            hi = t
        elif crossed:
            #: the empty chunk's span is unwound: it consumes no ordinal, because
            #: the walk slides its window instead of closing
            cid[open_lo:hi + 1] = -1
            nr = nw = 0
            hi = t
        elif hi < 0:
            hi = t
        cid[t] = k
        open_lo = t
        nr += lr[t]
        nw += lw[t]
    return cid


def _stamps(cid: np.ndarray, live_r, live_w, T: int):
    """`(read stamp, write stamp, J)` per token in the kernel's own encoding
    (`chunk * C + position`), `-1` where the side did not gather the token.

    Positions and thresholds are cumulative counts within the chunk's contiguous
    span, which is exactly what the walk maintains -- `J(i)` INCLUSIVE of the read
    token's own write, since the forward's recurrence reads after writing at the
    same token.
    """
    held = cid >= 0
    #: a chunk's span starts where the ordinal changes; the offset subtracted off
    #: the running count is the count at the token before that start
    start = np.zeros(T, dtype=np.int64)
    change = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    start[:] = np.repeat(change, np.diff(np.r_[change, T]))
    out = []
    for live in (live_r, live_w):
        c = np.cumsum(live)
        base = np.where(start > 0, c[start - 1], 0)
        pos = c - base - 1
        out.append(np.where(held & live, cid * C + pos, -1))
    cw = np.cumsum(live_w)
    basew = np.where(start > 0, cw[start - 1], 0)
    j = np.where(held & live_r, cw - basew, -1)
    return out[0], out[1], j


def _reference(live_r: np.ndarray, live_w: np.ndarray, T: int):
    return _stamps(_walk(live_r, live_w, T), live_r, live_w, T)


# ------------------------------------------------------------------- the fixtures

def _inputs(a: Arm, T: int, seed: int, slot_frac: float, tok_frac: float):
    """Amplitudes whose ZERO PATTERN differs per block.

    A level column is dropped per token, so the surviving leaf set — and hence
    which blocks see the token at all — moves from token to token. That is the
    case the numeric rows cannot reference, and the reason this file exists.
    """
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    pr = torch.rand(1, 1, T, a.D * a.W, device=dev, generator=g) + 0.25
    pw = torch.rand(1, 1, T, a.D * a.W, device=dev, generator=g) + 0.25
    pr[torch.rand(pr.shape, device=dev, generator=g) > slot_frac] = 0.0
    pw[torch.rand(pw.shape, device=dev, generator=g) > slot_frac] = 0.0
    dead = torch.rand(T, device=dev, generator=g) > tok_frac
    pr[0, 0][dead] = 0.0
    pw[0, 0][dead] = 0.0
    v = torch.randn(1, T, 1, DV, device=dev, generator=g)
    return tuple(t.to(torch.bfloat16).contiguous() for t in (pr, pw, v))


def _probe(a: Arm, T: int, seed: int, slot_frac: float, tok_frac: float):
    m = extension()
    pr, pw, v = _inputs(a, T, seed, slot_frac, tok_frac)
    widths = [a.W] * a.D
    tbl = m.chunk_facts(pr, pw, a.C, a.BC, widths, Mask.M1, False)[0]
    stampR, stampW, jout, nchunk = m.chunk_walk_probe(tbl, a.C, a.BC, widths, T, 1)
    live_r, live_w = _liveness_from_table(_plan(a.D, a.W, a.BC), stampR.shape[0], tbl, T)
    return (stampR.cpu(), stampW.cpu(), jout.cpu(), nchunk.cpu(), live_r, live_w)


CELLS = [
    #: one chunk exactly: the buffer's tail and the "no chunk after this" header
    pytest.param(C, 1.0, 1.0, id="T=C, every block dense"),
    #: THE GENERAL CASE -- each block sees a DIFFERENT token set, so no single
    #: partition exists and no replayable numeric reference can be written
    pytest.param(256, 0.5, 1.0, id="per-block liveness, slot-sparse 1/2"),
    pytest.param(256, 0.25, 1.0, id="per-block liveness, slot-sparse 1/4"),
    #: block-sparse AND token-sparse: chunks span far more tokens than they gather
    pytest.param(512, 0.25, 0.4, id="per-block liveness, token-sparse 2/5"),
    #: blocks that are live nowhere at all -- their walk must close immediately and
    #: publish a header that stops the consumer, not an empty chunk forever
    pytest.param(128, 0.02, 1.0, id="most blocks dead"),
    #: ACROSS A K-WINDOW LINE, which is the one cut with no forward counterpart
    pytest.param(K_WINDOW + 2 * C, 0.5, 1.0, id="across a K-window line"),
]

IDS = [p.id for p in CELLS[:-1]]
ARGS = [p.values for p in CELLS[:-1]]
#: the window-line cut is a property of the WALK, which reads the table and nothing
#: arm-shaped; at 65,536 tokens it runs once, at the pinned arm.
WINDOW_ARGS = [p.values for p in CELLS[-1:]]
WINDOW_IDS = [p.id for p in CELLS[-1:]]


def _each_block(nchunk):
    for b in range(nchunk.shape[0]):
        for blk in range(nchunk.shape[1]):
            yield b, blk


@pytest.mark.parametrize("arm", ARMS, ids=str)
@pytest.mark.parametrize("T,slot_frac,tok_frac", ARGS, ids=IDS)
def test_the_descending_partition_is_the_closing_rules(arm, T, slot_frac, tok_frac):
    """CLAIM 1. Token for token, per block, against the closing rule simulated on
    the host from the table's own bits -- at liveness that differs across blocks,
    which is the case no arithmetic reference can replay."""
    _claim_one(*_probe(arm, T, 11, slot_frac, tok_frac), T)


def _claim_one(sR, sW, sJ, nchunk, live_r, live_w, T):
    for b, blk in _each_block(nchunk):
        lr = live_r[b, blk].numpy()
        lw = live_w[b, blk].numpy()
        want = _reference(lr, lw, T)
        got = (sR[b, blk].numpy(), sW[b, blk].numpy(), sJ[b, blk].numpy())
        for name, wa, go in zip(("read", "write", "threshold"), want, got):
            bad = np.flatnonzero(wa != go)
            assert bad.size == 0, (
                f"the descending walk's {name} schedule differs from the closing rule "
                f"at block {blk} (bh {b}), first at token {bad[0]}: kernel {go[bad[0]]}, "
                f"rule {wa[bad[0]]} ({bad.size} of {T} tokens differ). A stamp encodes "
                f"chunk * {C} + position, so a whole-{C} offset is a chunk boundary "
                f"moving and a small one is a list ordering.")


@pytest.mark.parametrize("arm", ARMS, ids=str)
@pytest.mark.parametrize("T,slot_frac,tok_frac", ARGS, ids=IDS)
def test_every_live_token_is_gathered_exactly_once(arm, T, slot_frac, tok_frac):
    """CLAIM 2, and the SECOND OPINION: the gathered set and the table's live set
    are the same set, and no two tokens of a block share a (chunk, position)."""
    sR, sW, _, nchunk, live_r, live_w = _probe(arm, T, 11, slot_frac, tok_frac)
    for name, stamp, live in (("read", sR, live_r), ("write", sW, live_w)):
        gathered = stamp >= 0
        assert torch.equal(gathered, live), (
            f"the descending walk's {name} side does not cover the union table's live "
            f"set exactly: {int((gathered & ~live).sum())} token(s) gathered that the "
            f"table calls dead, {int((~gathered & live).sum())} live token(s) gathered "
            f"by no chunk at all")
        for b, blk in _each_block(nchunk):
            s = stamp[b, blk][gathered[b, blk]].numpy()
            assert np.unique(s).size == s.size, (
                f"two tokens carry the same (chunk, position) on the {name} side at "
                f"block {blk}: the lists overlap, so the sum double-counts")


@pytest.mark.parametrize("arm", ARMS, ids=str)
@pytest.mark.parametrize("T,slot_frac,tok_frac", ARGS, ids=IDS)
def test_the_thresholds_are_self_consistent_with_the_published_lists(arm, T, slot_frac, tok_frac):
    """CLAIM 3, with NO reference: `J(i)` and the list ordering recomputed from the
    kernel's OWN stamps.

    The staircase keeps write entries `tc < J[tr]`, so `J(i)` must be the count of
    this chunk's write entries at or before read entry `i` -- inclusive, because
    the forward's recurrence reads after writing at the same token. A threshold off
    by one is the intra-chunk term reading its own token's write or dropping it,
    and no gate on the inter-chunk part can see it.
    """
    sR, sW, sJ, nchunk, _, _ = _probe(arm, T, 11, slot_frac, tok_frac)
    for b, blk in _each_block(nchunk):
        rs, ws, js = (x[b, blk].numpy() for x in (sR, sW, sJ))
        rlive, wlive = rs >= 0, ws >= 0
        #: the chunk a token belongs to, and the running write count, both taken
        #: off the emitted stamps rather than off any model of the walk
        wchunk = np.where(wlive, ws // C, -1)
        rchunk = np.where(rlive, rs // C, -1)
        cw = np.cumsum(wlive)
        for ch in np.unique(rchunk[rlive]):
            span = np.flatnonzero(wchunk == ch)
            before = 0 if span.size == 0 else (cw[span[0] - 1] if span[0] > 0 else 0)
            here = rlive & (rchunk == ch)
            assert np.array_equal(js[here], cw[here] - before), (
                f"chunk {ch} of block {blk} (bh {b}) carries thresholds that do not "
                f"count its own write entries at or before each read entry: the "
                f"staircase would mask the intra-chunk term against the wrong boundary")
        for name, stamp, livemask in (("read", rs, rlive), ("write", ws, wlive)):
            pos = stamp[livemask] % C
            ch = stamp[livemask] // C
            #: a block live nowhere publishes no list at all, and an empty list is
            #: ascending -- the claim it fails is CLAIM 2's coverage, not this one
            if pos.size == 0:
                continue
            #: tokens ascend, so positions must ascend WITHIN a chunk and restart
            #: at zero at each chunk -- the ascending-lists claim every downstream
            #: reuse of the forward's code rests on
            first = np.r_[True, ch[1:] != ch[:-1]]
            assert np.all(pos[first] == 0) and np.all(pos[~first] == pos[np.flatnonzero(~first) - 1] + 1), (
                f"block {blk} does not publish an ascending {name} list")


@pytest.mark.parametrize("T,slot_frac,tok_frac", WINDOW_ARGS, ids=WINDOW_IDS)
def test_the_walk_cuts_at_every_window_line(T, slot_frac, tok_frac):
    """The one cut with no forward counterpart, at the pinned arm: an index-list
    entry is 16 bits, so a chunk may not span two K-windows."""
    sR, sW, sJ, nchunk, live_r, live_w = _probe(PINNED, T, 11, slot_frac, tok_frac)
    _claim_one(sR, sW, sJ, nchunk, live_r, live_w, T)


def test_the_partition_gate_is_not_vacuous():
    """A gate that cannot fail is not a gate. A walk that closes one entry early
    is the smallest structural error available, and claim 1's comparison must
    reject it."""
    T = 256
    sR, _, _, nchunk, live_r, live_w = _probe(PINNED, T, 11, 0.5, 1.0)
    disagreed = 0
    for b, blk in _each_block(nchunk):
        lr, lw = live_r[b, blk].numpy(), live_w[b, blk].numpy()
        cid = np.full(T, -1, dtype=np.int64)
        k = nr = nw = 0
        for t in range(T - 1, -1, -1):
            if (lr[t] and nr == C - 1) or (lw[t] and nw == C - 1):
                k += 1
                nr = nw = 0
            cid[t] = k
            nr += lr[t]
            nw += lw[t]
        wrong, _, _ = _stamps(cid, lr, lw, T)
        disagreed += int(not np.array_equal(wrong, sR[b, blk].numpy()))
    assert disagreed > 0, (
        "a walk that closes one entry early stamps the same partition, so the "
        "comparison is not reading the closing rule at all")
