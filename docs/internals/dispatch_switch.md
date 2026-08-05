# `csrc/rola/src/dispatch_switch.cuh` — host-side compile-time dispatch

A kernel with `N` structural template axes is selected at runtime from `N` runtime
values. The shape that selection takes is a real engineering choice, and this header
is the one form this engine uses for it.

The consumer's own dispatch does not use it: that ladder is GENERATED from the
instantiation matrix by `tools/gen_shards.py` into
`instantiations/dispatch_launch.inc`, where each leaf is a single function-pointer
assignment and the enumeration is single-sourced with the shards themselves
([`consumer.md#instantiation-matrix`](consumer.md#instantiation-matrix)). This header
is for the hand-written dispatches: `decode.cu`'s `launch_decode_gemv` and
`entmax.cu`'s forward and backward launchers.

## <a id="macro-ladder"></a>1. What it replaced, and why that form is refused

A macro ladder: one `#define` per axis whose body restates the launch, so an
`N`-axis dispatch spells the launch `2^N` times inside a preprocessor expansion.

```cpp
#define ROLA_DECODE_DISPATCH_DECAY(DV_, GLOBAL_)      \
  if (decay) { ROLA_DECODE_DISPATCH_D(DV_, true,  GLOBAL_) } \
  else       { ROLA_DECODE_DISPATCH_D(DV_, false, GLOBAL_) }
```

Three properties make it the wrong shape, and none of them is a matter of taste:

1. **The launch configuration is written once per arm.** `entmax.cu`'s forward ladder
   spelled a fourteen-argument `<<<...>>>` twice inside one macro, once per `alpha`
   arm, and eight more times through the width `switch`. Nothing checks that the
   copies agree; a drift between two arms is a silent wrong-result bug on the arm the
   suite happens not to reach first.
2. **The compiler cannot help.** A macro argument is a token, so a swapped `DECAY_`
   and `GLOBAL_` is not a type error — it is a different kernel.
3. **Neither a debugger nor a reader can step it.** The expansion has no source line
   of its own.

## <a id="one-launch"></a>2. The form used instead — the launch is written ONCE

flash-attention's `BOOL_SWITCH` idiom with the macro removed. The axis walker is a
function template; the launch is a generic lambda written once; each axis arrives as
an `integral_constant` whose `::value` is a compile-time constant at the point of use.

```cpp
int_switch<32, 64>(d_v, "decode d_v", [&](auto DV) {
  bool_switch(global_norm, [&](auto GLOBAL) {
    bool_switch(decay, [&](auto DECAY) {
      int_switch<1, 2, 3, 4>(p.D, "D (spec §1)", [&](auto LEVELS) {
        decode_gemv_kernel<decltype(DV)::value, decltype(LEVELS)::value,
                           decltype(DECAY)::value, decltype(GLOBAL)::value>
            <<<dim3(p.n_split, p.BH), kDecodeThreads, smem, stream>>>(p);
      });
    });
  });
});
```

Two primitives, and deliberately only two:

| | axis | refusal |
|---|---|---|
| `bool_switch(value, body)` | one boolean | none possible — both arms exist |
| `int_switch<VALUES...>(value, what, body)` | one integer over a CLOSED set | `TORCH_CHECK` naming the axis and the value |

`int_switch`'s candidate set is part of the CALL, which is what makes an unmatched
runtime value a refusal rather than a silently missing arm. The sets in this engine
are two to six values wide, so the expansion is a comparison chain and does not need
to be a jump table.

**Derived arm parameters stay derived.** `entmax.cu`'s width axis selects a
`(lane width, items per thread)` pair, and the ladder tabulated that pair per case —
`case 64: ROLA_ENTMAX_FWD(32, 2)`. It is now computed from the padded width inside the
lambda:

```cpp
constexpr int kLaneWidth      = kWPad < 32 ? kWPad : 32;
constexpr int kItemsPerThread = kWPad / kLaneWidth;
```

so the two cannot disagree with the width they were chosen for.

## <a id="what-is-gated"></a>3. What a gate can and cannot say about this change

**The set of DEVICE instantiations is identical.** The same template arguments are
named, so the fatbin carries the same entry points and the ratification manifest is
unmoved — 608 entries per arch, `tools/ratify.py` and `tools/fatbin_entries.py`. An
arm that stopped being instantiated fails those gates.

**The HOST selection code is genuinely different**, it is not device code, and no SASS
comparison can speak for it: `tools/sass_bodies.py` hashes device entry points, and
the host `.text` of the extension is not in that hash. What covers the host side is
the CUDA suites, which exercise every arm of both dispatches — the decode tier and the
entmax gates. A claim of "no change" for this file rests on those, and stating that
plainly is the point of this section.
