# The per-level facts (`csrc/fsla/src/common/plan/levels.h`)

`level_width_bits(d, l)`, `level_read_listed(d, l)` and `level_write_listed(d, l)` read level `l`'s (0-based) width fact and each side's walk of it from a `Facts`
through `Reader`, one lookup a fact for every launch function ([`decode_launch.md`](decode_launch.md), [`prefill_launch.md`](prefill_launch.md)); a level beyond the
fourth reads the fourth. They live in a hand-written header of their own because `rules.h` is the rule language and cannot include `facts.h` (which includes it),
and `Reader` is in the generated `facts.h`.
