# The per-level facts (`csrc/fsla/src/common/plan/levels.h`)

`level_width_bits(d, l)` and `level_write_listed(d, l)` read level `l`'s (0-based) width fact and decode's write-side walk of it from a kernel's `Facts`
through its `Reader` (`F::Read`), one lookup a fact for the launch functions ([`decode_launch.md`](decode_launch.md), [`prefill_launch.md`](prefill_launch.md)); a level beyond the
fourth reads the fourth. They are templates over the facts type, each instantiated by the kernel whose registry has its fact: the width by both, the write flag by decode alone (the prefill's descriptor states no walk and no weight: the host never reads them). They live in a hand-written header of their own because `rules.h` is the rule language and cannot include a kernel's `facts.h` (which includes it),
and `Reader` is in the generated `facts.h`.

`levels.h` also holds the cards' constants and the rule that maps a card to its window: `kHopperCc` (900), `kHopperWindow` (1,024 tokens, provisional until the Hopper carry's design picks it) and `window_of_cc(cc)`, THE WINDOW OF A CARD in tokens (`kHopperWindow` at `kHopperCc`, 0 where no window is derived for the card), the one statement of it ([`prefill_launch.md`](prefill_launch.md#window)).
