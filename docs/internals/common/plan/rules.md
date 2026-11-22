# The rule language (`csrc/fsla/src/common/plan/rules.h`)

What a matcher (an arm half of a launch function, [`decode_launch.md`](decode_launch.md)) is written in: header-only and constexpr, so the build evaluates a matcher over
the contract's ranges and the run time will evaluate it over a device's points with the same code, and a matcher means one thing in both.

- `Interval {lo, hi}`: an int fact's value; a point is `lo == hi`. `EnumSet`: a set fact's value, a bit a member of the fact's enum.
- `Truth {No, Yes, Maybe}`: the result of a test over a range. `lt`, `le`, `ge`, `eq` and `in` are Yes when every value of the range
  passes, No when none does and Maybe when the range straddles; `has` asks a set (one value, so Yes or No).
- `all(tests...)` is No when any test is No, Yes when every test is Yes, else Maybe; `any(tests...)` is Yes when any is Yes, No when every
  one is No, else Maybe.
- `Reach<Arm, Cap>`: what a walk reached: its arms, `gap` (the NONE leaf was visited) and the count of leaves visited. `fork(test, then, otherwise)` runs `then` when the
  test is not No and `otherwise` when it is not Yes: both on Maybe. `ladder(x, first, upto, n, hole, leaf)` is the one-fact tree decode's matcher is built of:
  `leaf(i)` for the values up to `upto[i]` (from `first`, then from the rung before), `hole()` below `first` or above the last rung.

A matcher is a decision tree on Truth: forks with arm and NONE leaves, walked once. Over a RANGE it returns every arm the range can reach and `gap` when some point reaches none;
over a POINT exactly one leaf, which `Reach::one()` returns (the arm at a point is the same walk).

<a id="exact"></a>THE WALK IS EXACT where each fact is tested along a path only by a ladder of `le` thresholds (and its one `ge` floor), and each test sees one fact: the Truth of a test over
the fact's whole range is Maybe exactly when a point of the range can go either way along this path (the rungs above are tested first, so a rung is reached
exactly when `hi` is above the rung before it, and its leaf when `lo` is at or below it: the values the range holds in the rung). A test on one fact is independent of
the tests on the others (the ranges are a box). It is CONSERVATIVE where one path tests the same fact twice other than as a ladder: nothing narrows a range
along a path, so `eq(x, 1)` then `eq(x, 2)` ... `hole()` over `x` in 1:2 reaches the hole (the last else is taken as Maybe though no point of 1:2 is left), and `in(x, 1, 32)` then
`in(x, 33, 64)` over 2:40 reaches the hole for the same reason. A conservative walk reaches an arm (or a hole) no point reaches: the build compiles an arm too many
(or refuses a contract that is covered), never one too few. Decode's tree is ladders on `d_v`, `levels` and `decay` and one `eq` on `cc`, so it is exact; `tests/unit/test_decode_arms_gen.py`
checks it against every point of seeded ranges. THE POINT GUARANTEE is a `static_assert` in `rules.h`: every primitive on a point Interval or EnumSet is Yes or No, so `all`/`any` of
them are, a fork takes one branch, and a ladder over a point reaches one leaf.

`struct Facts` (one member per fact) is not here: it is generated from the schema ([`../../hardware/registry.md`](../../hardware/registry.md#schema)).
