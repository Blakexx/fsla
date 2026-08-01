"""Budget-machinery unit tests (architecture §4.5; task item 4: "counts exact,
tax fires only on first touches, slack arithmetic")."""

from __future__ import annotations

import torch

from rola.ops.losses import (
    DualAscentLambda,
    TouchedPrefixState,
    tier1_support_loss,
    tier2_prefix_novelty_tax,
    tsallis_participation_surrogate,
)
from rola.routing.budget import (
    chunk_boundaries,
    chunk_slack,
    chunk_target_count,
    exact_support_count,
    normalized_slack,
    touched_bitset_update,
    update_soft_usage,
)


def test_exact_support_count_counts_nonzero_entries():
    values = torch.tensor([[1.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    counts = exact_support_count(values)
    assert torch.equal(counts, torch.tensor([2.0, 0.0, 4.0]))


def test_chunk_target_count_scales_with_r_width_and_tokens():
    r = torch.tensor([0.25, 1.0])
    target = chunk_target_count(r, width=8, token_count=10)
    assert torch.allclose(target, torch.tensor([0.25 * 8 * 10, 1.0 * 8 * 10]))


def test_chunk_target_count_rejects_out_of_range_r():
    import pytest

    with pytest.raises(ValueError):
        chunk_target_count(torch.tensor([0.0]), width=4, token_count=1)
    with pytest.raises(ValueError):
        chunk_target_count(torch.tensor([1.5]), width=4, token_count=1)


def test_chunk_slack_and_normalized_slack_arithmetic():
    target = torch.tensor([10.0, 10.0, 10.0])
    usage = torch.tensor([10.0, 5.0, 15.0])
    slack = chunk_slack(target, usage)
    assert torch.allclose(slack, torch.tensor([0.0, 5.0, -5.0]))
    norm = normalized_slack(target, usage)
    assert torch.allclose(norm, torch.tensor([0.0, 0.5, -0.5]))
    # exactly-on-target slack (usage == target) must normalize to exactly 0 -- this is the
    # regime spec §4.5 wants the model to have trained on ("slack -> 0" as a seen regime,
    # not a cliff), so it must not merely be "small" but exactly zero when target == usage.
    assert norm[0].item() == 0.0


def test_normalized_slack_is_clamped_to_unit_range():
    target = torch.tensor([1.0])
    usage = torch.tensor([100.0])  # wildly over-target
    norm = normalized_slack(target, usage)
    assert norm.item() == -1.0


def test_update_soft_usage_accumulates_route_mass():
    running = torch.zeros(2, 3)
    chunk = torch.ones(2, 4, 3)  # 4 tokens, each contributing 1.0 per digit
    updated = update_soft_usage(running, chunk)
    assert torch.equal(updated, torch.full((2, 3), 4.0))
    updated2 = update_soft_usage(updated, chunk)
    assert torch.equal(updated2, torch.full((2, 3), 8.0))  # genuinely cumulative


def test_touched_bitset_update_is_or_reduction_and_detached():
    touched = torch.zeros(1, 3, dtype=torch.bool)
    chunk = torch.zeros(1, 2, 3, requires_grad=True)
    with torch.no_grad():
        chunk[0, 0, 0] = 1.0
        chunk[0, 1, 1] = 1.0
    updated = touched_bitset_update(touched, chunk)
    assert torch.equal(updated, torch.tensor([[True, True, False]]))
    assert not updated.requires_grad
    # idempotent: touching the same digit again does not un-touch anything
    updated2 = touched_bitset_update(updated, chunk)
    assert torch.equal(updated2, updated)


def test_chunk_boundaries_covers_the_sequence_exactly():
    boundaries = chunk_boundaries(10, chunk_size=4)
    assert boundaries == [(0, 4), (4, 8), (8, 10)]
    covered = []
    for start, end in boundaries:
        covered.extend(range(start, end))
    assert covered == list(range(10))


def test_dual_ascent_lambda_increases_above_target_decreases_below_and_floors_at_zero():
    lam = DualAscentLambda(torch.tensor(0.5))
    above = lam.step(torch.tensor(10.0), target=5.0, lr=0.1)
    assert above.value.item() > lam.value.item()
    below = lam.step(torch.tensor(0.0), target=5.0, lr=1.0)
    assert below.value.item() == 0.0  # relu floor: cannot go negative


def test_tsallis_participation_surrogate_bounds():
    onehot = torch.zeros(6)
    onehot[2] = 1.0
    uniform = torch.full((6,), 1.0 / 6)
    assert abs(tsallis_participation_surrogate(onehot).item() - 1.0) < 1e-6
    assert abs(tsallis_participation_surrogate(uniform).item() - 6.0) < 1e-4


def test_tier1_support_loss_reports_exact_mean_matching_manual_count():
    values = torch.tensor([[[1.0, 0.0, 0.0, 0.0]], [[0.5, 0.5, 0.0, 0.0]]])  # [T=2,H=1,width=4]
    lam = DualAscentLambda(torch.tensor(1.0))
    _, exact_mean, updated = tier1_support_loss(values, target_fraction=0.5, lambda_state=lam, lr=0.1)
    assert exact_mean.item() == 1.5  # (1 + 2) / 2
    # target = 0.5 * 4 = 2.0; exact_mean (1.5) < target -> lambda should decrease
    assert updated.value.item() < lam.value.item()


def test_tier2_tax_fires_only_on_first_touch_of_each_prefix():
    # Two levels, widths (3, 4). Token 0 writes a NEW prefix (both levels first-touch);
    # token 1 repeats the exact same leaf (no tax); token 2 reuses level 0's digit but
    # touches a NEW level-1 digit (only the level-1 weight taxes); token 3 repeats token 2.
    w0 = torch.zeros(1, 4, 1, 3)
    w0[0, :, 0, 0] = 1.0
    w1 = torch.zeros(1, 4, 1, 4)
    w1[0, 0, 0, 1] = 1.0
    w1[0, 1, 0, 1] = 1.0
    w1[0, 2, 0, 2] = 1.0
    w1[0, 3, 0, 2] = 1.0
    touched0 = TouchedPrefixState.init((3, 4), B=1, H=1, device=w0.device)
    tax, touched1 = tier2_prefix_novelty_tax((w0, w1), touched0, (1.0, 1.0))
    assert torch.allclose(tax, torch.tensor([[3.0]]))  # 2 (token0) + 0 + 1 (token2) + 0
    # touched state persists correctly for a second chunk: repeating everything now taxes nothing
    tax2, touched2 = tier2_prefix_novelty_tax((w0, w1), touched1, (1.0, 1.0))
    assert torch.allclose(tax2, torch.zeros(1, 1))


def test_tier2_tax_weights_scale_per_level():
    w0 = torch.zeros(1, 1, 1, 2)
    w0[0, 0, 0, 0] = 1.0
    w1 = torch.zeros(1, 1, 1, 2)
    w1[0, 0, 0, 0] = 1.0
    touched0 = TouchedPrefixState.init((2, 2), B=1, H=1, device=w0.device)
    tax, _ = tier2_prefix_novelty_tax((w0, w1), touched0, (2.0, 3.0))
    assert torch.allclose(tax, torch.tensor([[5.0]]))  # 2.0 (level0 first touch) + 3.0 (level1 first touch)
