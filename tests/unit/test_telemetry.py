"""Tests for rola/diagnostics/telemetry.py (remaining-tasks-spec.md TASK #2).

Synthetic pr/pw streams with hand-planted structure (a dead write state + a
read-of-never-written) -> assert the JSONL record matches the planted ground truth;
plus disabled-mode no-op/speed, throttling, env activation, and grad-path safety.
"""

import inspect
import json
import math
import time

import pytest
import torch

from rola.diagnostics import telemetry as telemetry_module
from rola.diagnostics.telemetry import RolaTelemetry
from rola.routing.types import (
    READ_SUPPORT,
    ROUTE_TILE_SIZE,
    WRITE_SUPPORT,
    ResolvedRouting,
    RouteDescriptor,
    UnionRouting,
)


# ------------------------------------------------------------------ planted streams
# D=2, b=2 -> nc=4, leaves MSB-first: 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1). BH=1, L=4.
#
# WRITE: level-0 factor is always [1,0] (digit0 = 0) -> only leaves {0,1} ever get
# write mass; leaves {2,3} are the PLANTED DEAD STATES (dead bitset 0b1100 = 0xc).
# READ: token 0 routes level-0 -> [0,1], level-1 -> [1,0] => all its read mass on
# leaf 2 = a PLANTED READ-OF-NEVER-WRITTEN. Tokens 1..3 read leaves {0,1} (written).
# Total read mass = 4.0, never-written read mass = 1.0 -> frac = 0.25 exactly.
def _planted_streams():
    D, b, L = 2, 2, 4
    pw = torch.zeros(1, L, D, b)
    pw[..., 0, 0] = 1.0                                   # lvl0 = [1,0] every token
    pw[0, 0, 1] = torch.tensor([1.0, 0.0])
    pw[0, 1, 1] = torch.tensor([0.0, 1.0])
    pw[0, 2, 1] = torch.tensor([0.5, 0.5])
    pw[0, 3, 1] = torch.tensor([1.0, 0.0])

    pr = torch.zeros(1, L, D, b)
    pr[0, 0, 0] = torch.tensor([0.0, 1.0])                # token 0 -> digit0 = 1
    pr[0, 0, 1] = torch.tensor([1.0, 0.0])                # token 0 -> leaf 2 (never written)
    for t in (1, 2, 3):
        pr[0, t, 0] = torch.tensor([1.0, 0.0])
        pr[0, t, 1] = torch.tensor([0.5, 0.5])            # leaves 0,1
    routing = ResolvedRouting((UnionRouting(), UnionRouting()), (b, b))
    return pr, pw, routing


def _level_views(values, branches):
    return tuple(values[..., level, :width] for level, width in enumerate(branches))


def _radix_indices(branches, device):
    state_ids = torch.arange(math.prod(branches), device=device, dtype=torch.long)
    return tuple(
        (state_ids // math.prod(branches[level + 1:])) % width
        for level, width in enumerate(branches)
    )


def _fold_levels(levels, indices, state_lo=0, state_hi=None):
    state_hi = len(indices[0]) if state_hi is None else state_hi
    folded = None
    for factors, index in zip(levels, indices):
        selected = factors.index_select(-1, index[state_lo:state_hi])
        folded = selected if folded is None else folded * selected
    return folded


def _fold_level_chunks(levels, indices, state_count, chunk_size=16):
    for state_lo in range(0, state_count, chunk_size):
        state_hi = min(state_lo + chunk_size, state_count)
        yield state_lo, state_hi, _fold_levels(levels, indices, state_lo, state_hi)


def _pack_support_words(values):
    batch_heads, tokens, width = values.shape
    tiles = (tokens + ROUTE_TILE_SIZE - 1) // ROUTE_TILE_SIZE
    padded = torch.zeros(
        batch_heads, tiles * ROUTE_TILE_SIZE, width,
        device=values.device, dtype=torch.bool)
    padded[:, :tokens] = values
    bits = (1 << torch.arange(
        ROUTE_TILE_SIZE, device=values.device, dtype=torch.int64)).view(1, 1, ROUTE_TILE_SIZE, 1)
    return (padded.reshape(batch_heads, tiles, ROUTE_TILE_SIZE, width).to(torch.int64) * bits).sum(
        dim=2).to(torch.int32).contiguous()


def _route_descriptor(read, write, routing):
    branches = routing.branches
    batch_heads, tokens = read.shape[:2]
    tiles = (tokens + ROUTE_TILE_SIZE - 1) // ROUTE_TILE_SIZE

    def arena(values):
        compact = torch.cat(
            [values[..., level, :width] for level, width in enumerate(branches)], dim=-1
        ).to(torch.bfloat16)
        padded = torch.cat((compact, compact.new_zeros(
            batch_heads, tiles * ROUTE_TILE_SIZE - tokens, compact.shape[-1])), dim=1)
        return padded.reshape(batch_heads, tiles, ROUTE_TILE_SIZE, compact.shape[-1]).permute(
            0, 1, 3, 2).contiguous()

    chunks = []
    for level, (width, flags) in enumerate(zip(branches, routing.support_side_flags)):
        if flags == 0:
            continue
        level_support = torch.zeros(
            batch_heads, tokens, width, device=read.device, dtype=torch.bool)
        if flags & READ_SUPPORT:
            level_support |= read[..., level, :width].to(torch.bfloat16) != 0
        if flags & WRITE_SUPPORT:
            level_support |= write[..., level, :width].to(torch.bfloat16) != 0
        chunks.append(_pack_support_words(level_support))
    support_words = torch.cat(chunks, dim=-1).contiguous() if chunks else torch.empty(
        batch_heads, tiles, 0, device=read.device, dtype=torch.int32)
    return RouteDescriptor(
        arena(read), arena(write), support_words,
        routing.level_offsets, routing.support_offsets, routing.branches,
        routing.radix_strides, routing.support_side_flags,
        0, tokens, routing.topology_stamp,
    )


def _planted_descriptor(pr, pw, routing):
    descriptor = _route_descriptor(pr, pw, routing)
    read_logits = torch.zeros_like(pr)
    write_logits = torch.ones_like(pw)
    return descriptor, read_logits, write_logits


def _read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ------------------------------------------------------------------------------ tests
def test_fold_matches_canonical_leaf_order():
    # local fold == the canonical MSB-first Kronecker fold semantics (spot-check by hand)
    f = torch.tensor([[0.25, 0.75], [0.4, 0.6]]).view(1, 1, 2, 2)
    levels = _level_views(f, (2, 2))
    g = _fold_levels(levels, _radix_indices((2, 2), f.device)).reshape(-1)
    expect = torch.tensor([0.25 * 0.4, 0.25 * 0.6, 0.75 * 0.4, 0.75 * 0.6])
    assert torch.allclose(g, expect)


def test_chunked_fold_matches_canonical_without_full_leaf_plane():
    factors = torch.tensor(
        [[[[0.2, 0.8, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]],
          [[0.1, 0.9, 0.0, 0.0], [0.25, 0.75, 0.0, 0.0]]]]
    )
    levels = _level_views(factors, (2, 2))
    indices = _radix_indices((2, 2), factors.device)
    expected = _fold_levels(levels, indices)
    pieces = [piece for _lo, _hi, piece in _fold_level_chunks(levels, indices, 4, chunk_size=3)]
    assert torch.equal(torch.cat(pieces, dim=-1), expected)


def test_mixed_radix_tile_support_reports_both_dead_tail():
    # D=2, branches=(4,5), nc=20. Only leaf 0 is live, so the 16-state tail tile is
    # read-dead, write-dead, and both-dead. The first tile remains live on both sides.
    pr = torch.zeros(1, 3, 2, 5)
    pw = torch.zeros_like(pr)
    pr[..., 0, 0] = 1.0
    pr[..., 1, 0] = 1.0
    pw[..., 0, 0] = 1.0
    pw[..., 1, 0] = 1.0
    levels_r = _level_views(pr, (4, 5))
    levels_w = _level_views(pw, (4, 5))
    indices = _radix_indices((4, 5), pr.device)
    stats = RolaTelemetry()._tile_stats(levels_r, levels_w, indices, 20)
    assert stats['cohort_width'] == 16
    assert stats['batch']['count'] == 2
    assert stats['batch']['read_dead_count'] == 1
    assert stats['batch']['write_dead_count'] == 1
    assert stats['batch']['both_dead_count'] == 1
    assert stats['batch']['both_dead_frac'] == pytest.approx(0.5)


def test_tile_and_leaf_support_remain_per_bh():
    # BH0 has a contiguous 32-state dead range; BH1 is fully live. Collapsing BH would
    # erase every dead tile and dead leaf because BH1 supplies support for all states.
    pr = torch.ones(2, 32, 2, 8)
    pw = torch.ones_like(pr)
    pr[0, :, 0, 2:6] = 0.0
    pw[0, :, 0, 2:6] = 0.0
    levels_r = _level_views(pr, (8, 8))
    levels_w = _level_views(pw, (8, 8))
    indices = _radix_indices((8, 8), pr.device)
    stats = RolaTelemetry()._tile_stats(levels_r, levels_w, indices, 64)
    first, second = stats['per_bh']
    assert first['both_dead_count'] > 0
    assert second['both_dead_count'] == 0
    assert first['read_dead_count'] == first['both_dead_count']
    assert first['write_dead_count'] == first['both_dead_count']
    assert stats['batch']['both_dead_count'] == first['both_dead_count']
    assert stats['batch']['count'] == 2 * stats['tile_count_per_bh']

    leaf = RolaTelemetry()._leaf_stats(levels_r, levels_w, indices, layer_idx=4, nc=64)
    assert leaf['per_bh'][0]['dead_write_count'] == 32
    assert leaf['per_bh'][1]['dead_write_count'] == 0
    assert leaf['batch']['dead_write_count'] == 32


def test_each_reducer_has_one_post_reduction_host_boundary():
    source = inspect.getsource(RolaTelemetry._tile_stats)
    assert 'bool(' not in source
    assert '.item(' not in source
    assert 'cuda.synchronize' not in source
    assert source.count('.cpu()') == 1

    leaf_source = inspect.getsource(RolaTelemetry._leaf_stats)
    assert 'cuda.synchronize' not in leaf_source
    assert leaf_source.count('.cpu()') == 1


def test_missing_cumulative_identity_is_explicitly_unavailable(tmp_path):
    pr, pw, routing = _planted_streams()
    descriptor, read_logits, write_logits = _planted_descriptor(pr, pw, routing)
    out = tmp_path / 't.jsonl'
    tel = RolaTelemetry().enable(str(out), every=1)
    tel.record_descriptor(descriptor, read_logits=read_logits, write_logits=write_logits,
                         union_levels=(0, 1))
    leaf = _read_jsonl(out)[0]['leaf']
    assert leaf['ever_written_scope'] == 'unavailable'
    assert leaf['batch']['never_written_count'] is None
    assert leaf['batch']['read_never_written_frac'] is None


def test_route_parameter_identity_scopes_cumulative_state(tmp_path):
    pr, pw, routing = _planted_streams()
    descriptor, read_logits, write_logits = _planted_descriptor(pr, pw, routing)
    out = tmp_path / 't.jsonl'
    tel = RolaTelemetry().enable(str(out), every=1)
    route_a = torch.empty(1)
    route_b = torch.empty(1)
    tel.record_descriptor(descriptor, read_logits=read_logits, write_logits=write_logits,
                          union_levels=(0, 1), _source=route_a)
    tel.record_descriptor(descriptor, read_logits=read_logits, write_logits=write_logits,
                          union_levels=(0, 1), _source=route_b)
    recs = _read_jsonl(out)
    assert recs[0]['leaf']['ever_written_scope'] == 'route_parameter'
    assert recs[1]['leaf']['ever_written_scope'] == 'route_parameter'
    assert recs[0]['leaf']['batch']['never_written_count'] == 2
    assert recs[1]['leaf']['batch']['never_written_count'] == 2


# NOTE (rola-v3 purge): `test_producer_forwards_route_parameter_identity` was
# deleted here, not moved. It pinned that the V2 producer
# (`routing/producer.py`, deleted) forwards its route-parameter identity into
# `record_route_descriptor`. The V3 producer (`routing/producer_v3.py`) is not
# yet wired to telemetry, so there is nothing equivalent to re-point it at; when
# it is, this contract needs a replacement test rather than a rename.
#
# RE-EXAMINED 2026-07-29 (compliance-audit V9 item 4) and deliberately still not
# replaced. The lost contract is a PRODUCER-TO-TELEMETRY FORWARDING contract, so
# closing it requires a call site, not a test: with no telemetry call in
# `producer_v3.py` there is nothing to assert forwarding FROM. A test driving
# `RolaTelemetry` directly would not restore it -- it would duplicate
# `test_route_parameter_identity_scopes_cumulative_state` above, which already
# pins the scoping mechanism the deleted test relied on. Classified in
# `cuda/src/V3_TODO.md`'s V9 section as a WIRING task rather than a gate task, so
# that it is not closed with a hollow test and is not mistaken for "no known gap".


def test_planted_ground_truth(tmp_path):
    pr, pw, routing = _planted_streams()
    descriptor, read_logits, write_logits = _planted_descriptor(pr, pw, routing)
    D, branches = routing.D, routing.branches
    out = tmp_path / 'telemetry.jsonl'
    tel = RolaTelemetry().enable(str(out), every=1)
    tel.record_descriptor(
        descriptor, layer_idx=3, step=17, union_levels=(0, 1),
        read_logits=read_logits, write_logits=write_logits,
    )

    recs = _read_jsonl(out)
    assert len(recs) == 1
    r = recs[0]
    assert r['step'] == 17 and r['layer_idx'] == 3
    assert r['D'] == D and r['branch_counts'] == list(branches) and r['nc'] == 4
    assert set(r['union']) == {'0', '1'}

    # ---- write-side zero fraction: lvl0 has 4 zeros / 8 entries, lvl1 has 3 / 8
    assert r['w']['zero_frac'] == pytest.approx(7 / 16)
    assert r['w']['zero_frac_per_level'][0] == pytest.approx(0.5)
    assert r['w']['zero_frac_per_level'][1] == pytest.approx(3 / 8)

    # ---- write support histogram: lvl0 all support-1; lvl1 sizes [1,1,2,1]
    assert r['w']['support_hist_per_level'][0] == [0, 4, 0]
    assert r['w']['support_hist_per_level'][1] == [0, 3, 1]

    # ---- collapse metrics: write lvl0 mean dist = [1,0] -> both entropies 0, p_max 1
    assert r['w']['entropy_mean_dist_per_level'][0] == pytest.approx(0.0, abs=1e-6)
    assert r['w']['entropy_per_token_per_level'][0] == pytest.approx(0.0, abs=1e-6)
    assert r['w']['p_max_per_level'][0] == pytest.approx(1.0)
    assert r['w']['max_entropy_per_level'] == pytest.approx([math.log(2), math.log(2)])
    # read lvl1 per-token entropy: token0 one-hot (0) + 3 tokens uniform (ln 2) -> 3/4 ln 2
    assert r['r']['entropy_per_token_per_level'][1] == pytest.approx(0.75 * math.log(2), rel=1e-5)

    # ---- planted dead write states: leaves {2,3} -> count 2, bitset 0b1100
    leaf = r['leaf']['per_bh'][0]
    assert leaf['dead_write_count'] == 2
    assert leaf['dead_write_frac'] == pytest.approx(0.5)
    assert int(leaf['dead_write_bitset_hex'], 16) == 0b1100
    assert leaf['never_written_count'] == 2
    assert int(leaf['never_written_bitset_hex'], 16) == 0b1100

    # ---- planted read-of-never-written: 1.0 of 4.0 total read mass -> 0.25 exactly
    assert leaf['read_never_written_frac'] == pytest.approx(0.25)
    assert r['tile']['scope'] == 'logical production forward-cohort geometry'
    assert r['tile']['batch']['count'] == 1
    assert r['tile']['batch']['both_dead_count'] == 0


def test_ever_written_accumulates_across_firings(tmp_path):
    pr, pw, routing = _planted_streams()
    descriptor, read_logits, write_logits = _planted_descriptor(pr, pw, routing)
    out = tmp_path / 't.jsonl'
    tel = RolaTelemetry().enable(str(out), every=1)
    tel.record_descriptor(descriptor, layer_idx=0,
                          read_logits=read_logits, write_logits=write_logits,
                          union_levels=(0, 1))
    # second window: write touches leaf 2 (lvl0=[0,1], lvl1=[1,0]) -> never_written shrinks to {3}
    pw2 = torch.zeros_like(pw)
    pw2[..., 0, :] = torch.tensor([0.0, 1.0])
    pw2[..., 1, :] = torch.tensor([1.0, 0.0])
    descriptor2, read_logits2, write_logits2 = _planted_descriptor(pr, pw2, routing)
    tel.record_descriptor(descriptor2, layer_idx=0,
                          read_logits=read_logits2, write_logits=write_logits2,
                          union_levels=(0, 1))
    recs = _read_jsonl(out)
    first = recs[0]['leaf']['per_bh'][0]
    second = recs[1]['leaf']['per_bh'][0]
    assert first['never_written_count'] == 2
    assert second['never_written_count'] == 1
    assert int(second['never_written_bitset_hex'], 16) == 0b1000
    # dead-THIS-WINDOW in rec 2 is {0,1,3} (window bitset, not cumulative)
    assert second['dead_write_count'] == 3


def test_disabled_is_noop_and_fast(tmp_path, monkeypatch):
    pr, pw, routing = _planted_streams()
    descriptor, read_logits, write_logits = _planted_descriptor(pr, pw, routing)
    tel = RolaTelemetry()                                  # no env, no enable
    assert not tel.enabled

    def no_work(*args, **kwargs):
        raise AssertionError('descriptor unpack/fold ran while telemetry was disabled')

    for name in ('_unpack_descriptor_values', '_radix_indices', '_fold_state_block'):
        monkeypatch.setattr(telemetry_module, name, no_work)
    t0 = time.perf_counter()
    for _ in range(20000):
        tel.record_descriptor(descriptor, layer_idx=0, step=0,
                              read_logits=read_logits, write_logits=write_logits,
                              union_levels=(0, 1))
    dt = time.perf_counter() - t0
    assert tel.calls == 0 and tel.fired == 0               # counter untouched when disabled
    assert dt < 0.5                                        # single-guard fast path
    assert not list(tmp_path.iterdir())                    # nothing written anywhere in tmp


def test_throttling(tmp_path, monkeypatch):
    pr, pw, routing = _planted_streams()
    descriptor, read_logits, write_logits = _planted_descriptor(pr, pw, routing)
    out = tmp_path / 't.jsonl'
    tel = RolaTelemetry().enable(str(out), every=3)
    tel.record_descriptor(descriptor, step=0, union_levels=(0, 1),
                          read_logits=read_logits, write_logits=write_logits)

    def no_work(*args, **kwargs):
        raise AssertionError('descriptor unpack/fold ran on a throttled call')

    names = ('_unpack_descriptor_values', '_radix_indices', '_fold_state_block')
    originals = {name: getattr(telemetry_module, name) for name in names}
    for name in names:
        monkeypatch.setattr(telemetry_module, name, no_work)
    for i in (1, 2):
        tel.record_descriptor(descriptor, step=i, union_levels=(0, 1),
                              read_logits=read_logits, write_logits=write_logits)
    for name, original in originals.items():
        monkeypatch.setattr(telemetry_module, name, original)
    tel.record_descriptor(descriptor, step=3, union_levels=(0, 1),
                          read_logits=read_logits, write_logits=write_logits)
    for name in names:
        monkeypatch.setattr(telemetry_module, name, no_work)
    for i in (4, 5):
        tel.record_descriptor(descriptor, step=i, union_levels=(0, 1),
                              read_logits=read_logits, write_logits=write_logits)
    for name, original in originals.items():
        monkeypatch.setattr(telemetry_module, name, original)
    tel.record_descriptor(descriptor, step=6, union_levels=(0, 1),
                          read_logits=read_logits, write_logits=write_logits)
    recs = _read_jsonl(out)
    assert len(recs) == 3                                  # fires on calls 1, 4, 7
    assert [r['step'] for r in recs] == [0, 3, 6]
    assert tel.fired == 3 and tel.calls == 7


def test_env_activation(tmp_path, monkeypatch):
    out = tmp_path / 'env.jsonl'
    monkeypatch.setenv('ROLA_TELEMETRY', str(out))
    monkeypatch.setenv('ROLA_TELEMETRY_EVERY', '2')
    tel = RolaTelemetry()
    assert tel.enabled and tel.every == 2
    pr, pw, routing = _planted_streams()
    descriptor, read_logits, write_logits = _planted_descriptor(pr, pw, routing)
    for i in range(4):
        tel.record_descriptor(descriptor, step=i, union_levels=(0, 1),
                              read_logits=read_logits, write_logits=write_logits)
    assert len(_read_jsonl(out)) == 2


def test_grad_safety(tmp_path):
    # producer is called with grad ENABLED; telemetry must not retain graph or mutate inputs
    pr, pw, routing = _planted_streams()
    pr = (pr + 1e-3).requires_grad_(True)                  # strictly positive, differentiable
    pw = (pw + 1e-3).requires_grad_(True)
    descriptor, read_logits, write_logits = _planted_descriptor(pr, pw, routing)
    out = tmp_path / 't.jsonl'
    tel = RolaTelemetry().enable(str(out), every=1)
    pr_c, pw_c = pr.detach().clone(), pw.detach().clone()
    tel.record_descriptor(descriptor, layer_idx=0, step=0, union_levels=(0, 1),
                          read_logits=read_logits, write_logits=write_logits)
    assert torch.equal(pr.detach(), pr_c) and torch.equal(pw.detach(), pw_c)
    (pr.sum() + pw.sum()).backward()                       # graph intact after telemetry
    assert pr.grad is not None and pw.grad is not None
    with torch.no_grad():                                  # and safe under no_grad
        tel.record_descriptor(descriptor, layer_idx=0, step=1, union_levels=(0, 1),
                              read_logits=read_logits, write_logits=write_logits)
    assert len(_read_jsonl(out)) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no GPU')
def test_gpu_streams(tmp_path):
    pr, pw, routing = _planted_streams()
    descriptor, read_logits, write_logits = _planted_descriptor(
        pr.cuda(), pw.cuda(), routing)
    out = tmp_path / 't.jsonl'
    tel = RolaTelemetry().enable(str(out), every=1)
    tel.record_descriptor(descriptor, layer_idx=1, step=5, union_levels=(0, 1),
                          read_logits=read_logits, write_logits=write_logits)
    recs = _read_jsonl(out)
    leaf = recs[0]['leaf']['per_bh'][0]
    assert leaf['dead_write_count'] == 2
    assert leaf['read_never_written_frac'] == pytest.approx(0.25)
