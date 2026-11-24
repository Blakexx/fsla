"""Focused tests for the production entmax solve: its width limit's refusal, the packed-projection read, the dense
levels' complete write, the batched solve against the per-level loop and the plan's launch count. The CUDA solve takes
bf16 logits and writes bf16 route values."""

import pytest
import torch

from fsla.routing.types import (
    EntmaxActivation,
    IndependentRouting,
    ResolvedRouting,
    SoftmaxActivation,
    TiedRouting,
)


def test_production_routing_width_limit_rejects_at_the_level():
    with pytest.raises(ValueError, match="width must be <= 256"):
        IndependentRouting(width=257, read=SoftmaxActivation(), write=SoftmaxActivation())


@pytest.mark.cuda
@pytest.mark.parametrize("alpha", [1.5, 2.0])
def test_the_solve_reads_a_packed_projections_half_slice_where_it_lies(alpha):
    """Every solve entry takes an explicit stride triple, so the logits need not be
    contiguous (P73 S4, inventory §B3).

    An untied packed projection returns each side as a half-slice of one GEMM output,
    and the claim is that the entries address it AS IT LIES: the levels off the slice
    must be the levels off a materialized copy of it BIT FOR BIT, forward and backward
    alike. A row that only checked closeness would pass on a stride bug that read the
    wrong columns of a symmetric fixture.
    """
    from fsla.routing.entmax.production import production_routing_levels

    torch.manual_seed(5)
    routing = ResolvedRouting(levels=(
        IndependentRouting(width=8, read=EntmaxActivation(alpha), write=SoftmaxActivation()),
        IndependentRouting(width=8, read=SoftmaxActivation(), write=SoftmaxActivation())))
    both = torch.randn(4, 64, 2 * routing.packed_logit_width, device="cuda", dtype=torch.bfloat16,
                       requires_grad=True)
    sides = (both[:, :, routing.packed_logit_width:], both[:, :, :routing.packed_logit_width])
    assert not sides[0].is_contiguous()

    def run(read, write):
        levels = production_routing_levels(read, write, routing)[:2]
        out = sum(level.square().sum() for side in levels for level in side)
        return levels, torch.autograd.grad(out, both, retain_graph=True)[0]

    strided, d_strided = run(*sides)
    packed, d_packed = run(*(side.contiguous() for side in sides))
    for side_a, side_b in zip(strided, packed):
        for level_a, level_b in zip(side_a, side_b):
            assert torch.equal(level_a, level_b)
    assert torch.equal(d_strided, d_packed)


@pytest.mark.cuda
@pytest.mark.parametrize("width", [3, 8, 64])
def test_every_solve_writes_every_element_of_a_dense_level(width):
    """The licence for allocating a dense level UNINITIALIZED (P73 S4, inventory §B5).

    A dense level is exactly `[BH, T, width]` -- no tile padding, unlike the
    tile-major arena whose tail lanes past `T` no kernel touches -- so nothing may
    survive a solve. Each shipped solve writes into a buffer poisoned two different
    ways; if any element were left alone the two results would differ there. Widths
    that are not powers of two are the ones a masked lane could hide in.
    """
    from fsla.routing.entmax.production import _Plan

    torch.manual_seed(6)
    BH, T = 4, 96
    logits = torch.randn(BH, T, width, device="cuda", dtype=torch.bfloat16)
    words = torch.empty((BH, (T + 31) // 32, width), dtype=torch.int32, device="cuda")

    def buffers(fill, count, dtype=torch.bfloat16):
        return [torch.full((BH, T, width), fill, device="cuda", dtype=dtype) for _ in range(count)]

    def solves(fill, alpha):
        factor, soft = buffers(fill, 2), buffers(fill, 2)
        plan = _Plan()
        plan.add("entmax", alpha, (logits, factor[0], factor[1], words), width, logit_off=0)
        plan.add("softmax", 0.0, (logits, soft[0], soft[1]), width, logit_off=0)
        plan.issue()
        return factor + soft

    for alpha in (1.5, 2.0):
        for cold, hot in zip(solves(0.0, alpha), solves(7.0, alpha)):
            assert torch.equal(cold, hot), (
                "a solve left an element of its dense output alone")


@pytest.mark.cuda
@pytest.mark.parametrize("alpha", [1.5, 2.0])
@pytest.mark.parametrize("kind", ["entmax", "softmax"])
def test_the_batched_solve_is_the_per_level_loop_bit_for_bit(alpha, kind):
    """P73 the STOP CLAUSE, as a gate: batching may change the launch SHAPE and nothing
    else.

    One launch over four levels of one width class, against four launches of one level
    each, on the same inputs and into the same layout -- asserted with `torch.equal`,
    because the claim is that each row performs the arithmetic it performed before, not
    that it lands nearby. A tolerance here would pass on the failure this gate exists to
    catch: a level silently run in a WIDER arm, where `LW`/`IPT` change how the sort
    permutes and how the prefix scan associates, moving `tau` by an ulp and with it a
    support boundary.

    Widths differ INSIDE the class (3 and 4 both pad to 4) because a table that carried
    only equal widths would not exercise the per-level `width` the kernel reads.
    """
    from fsla.routing.entmax.production import _Plan

    torch.manual_seed(11)
    BH, T = 3, 70
    widths = [3, 4, 4, 3]
    offsets = [0, 3, 7, 11]
    total = 14
    read = torch.randn(BH, T, total, device="cuda", dtype=torch.bfloat16)
    n_words = (T + 31) // 32

    def run(batched):
        values = torch.zeros(BH, T, total, device="cuda", dtype=torch.bfloat16)
        second = torch.zeros_like(values)
        words = torch.zeros((BH, n_words, total), dtype=torch.int32, device="cuda")
        plans = [_Plan()] if batched else [_Plan() for _ in widths]
        for index, (width, offset) in enumerate(zip(widths, offsets)):
            plan = plans[0] if batched else plans[index]
            if kind == "entmax":
                plan.add("entmax", alpha, (read, values, second, words), width,
                         logit_off=offset, value_off=offset, support_off=offset)
            else:
                plan.add("softmax", 0.0, (read, values, second), width,
                         logit_off=offset, value_off=offset)
        launches = sum(plan.issue() for plan in plans)
        return launches, values, second, words

    batched_count, *batched = run(True)
    loop_count, *loop = run(False)
    assert batched_count == 1 and loop_count == len(widths), (
        f"the shapes under comparison are wrong: batched issued {batched_count} launch(es) "
        f"and the loop {loop_count}; the point of the gate is 1 against {len(widths)}")
    for name, got, want in zip(("values", "second", "support"), batched, loop):
        assert torch.equal(got, want), (
            f"the batched solve's {name} differs from the per-level loop's in "
            f"{int((got != want).sum())} elements -- batching changed the arithmetic, "
            "not just the launch shape")

    # THE TOOTH. The equality above is only worth reading if the class rule it rests on
    # is ENFORCED rather than assumed, and the enforcement is not in this file: a table
    # carrying two width classes is refused at the entry, so the failure the `torch.equal`
    # could never see (a narrow level silently padded into a wider arm) cannot be built
    # in the first place. `_Plan` is what keeps such a table from being assembled; this
    # asserts the layer under it refuses one anyway.
    from fsla.ops._ext import extension

    values = torch.zeros(BH, T, 68, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="one width class"):
        extension().routing_softmax_forward(
            torch.randn(BH, T, 68, device="cuda", dtype=torch.bfloat16), values, None, [0, 4], [0, 4], [4, 64], 1)


@pytest.mark.cuda
def test_a_uniform_plan_solves_all_its_levels_in_one_launch():
    """`producer.py` has claimed "one packed projection and one batched solve" since it
    was written, and until P73 S5 the second half was false -- a python `for` over
    `routing.levels`, one launch and three allocations per level. This is the claim made
    checkable.

    A level joins a launch only when it selects the same template arm, so the gate also
    pins the NEGATIVE: two levels of different width CLASSES (4 and 64) stay two
    launches, because running the narrow one in the wide arm would not be the loop it
    replaces.
    """
    from fsla.routing.entmax.production import _Plan, _ProductionLevelsFn

    def launches(routing, widths):
        counted = []
        original = _Plan.issue

        def counting(self):
            counted.append(len(self._issued))
            return original(self)

        _Plan.issue = counting
        try:
            read = torch.randn(2, 64, routing.packed_logit_width, device="cuda", dtype=torch.bfloat16)
            _ProductionLevelsFn.apply(read, torch.randn_like(read), routing)
        finally:
            _Plan.issue = original
        return counted[0]

    level = TiedRouting(width=4, op=EntmaxActivation(1.5))
    uniform = ResolvedRouting((level, level, level))
    assert launches(uniform, (4, 4, 4)) == 1, (
        "three entmax levels of one width class must solve in ONE launch")

    mixed = ResolvedRouting((level, TiedRouting(width=64, op=EntmaxActivation(1.5))))
    assert launches(mixed, (4, 64)) == 2, (
        "levels of different width classes must NOT share a launch: the sort and the "
        "prefix scan associate differently per class")
