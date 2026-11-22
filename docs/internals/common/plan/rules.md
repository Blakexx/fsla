# The rule language (`csrc/fsla/src/common/plan/rules.h`)

What a matcher (an arm half of a launch function, [`decode_launch.md`](decode_launch.md)) is written in: header-only and constexpr, so the build evaluates a matcher over
the contract's ranges and the run time will evaluate it over a device's points with the same code, and a matcher means one thing in both.

- `Interval {lo, hi}`: an int fact's value; a point is `lo == hi`. `EnumSet`: a set fact's value, a bit a member of the fact's enum.
- `Truth {No, Yes, Maybe}`: the result of a test over a range. `lt`, `le`, `ge`, `eq` and `in` are Yes when every value of the range
  passes, No when none does and Maybe when the range straddles; `has` asks a set (one value, so Yes or No).
- `all(tests...)` is No when any test is No, Yes when every test is Yes, else Maybe; `any(tests...)` is Yes when any is Yes, No when every
  one is No, else Maybe.
- `ArmSet<Arm, Cap>`: the arms a matcher returns, each once; empty is NONE.

A matcher keeps an arm unless its test is No. Over a range it so returns every arm the range can reach; over points it must return exactly
one, which `arms_gen` checks at the corners of every range of the contract.

`struct Facts` (one member per fact) is not here: it is generated from the schema ([`../../hardware/registry.md`](../../hardware/registry.md#schema)).
