# The entmax kernels' matcher (`csrc/fsla/src/common/plan/entmax_launch.h`)

One matcher for the four kernels of `csrc/fsla/src/entmax/` (`factor_forward`, `factor_backward`, `softmax_forward`, `softmax_backward`), written as templates
over the kernel's generated definition `K` the way [`planner.md`](planner.md) is, so one source serves four registries (`hardware/kernels/<kernel>/`, one for each kernel:
[`../../entmax/entmax.md#the-contract`](../../entmax/entmax.md#the-contract)). The owner's words, 2026-11-23: "the descriptor is runtime context derived from hardware facts and user input. the matchers
contain our logic that distills that into an arm. anything that has any kind of logic for that mapping should be in a matcher". The facts it reads are the call's, in the caller's terms:
`width` (the first level's width, a real width), `levels`, `alpha_tenths` (the two kernels with an alpha), `logit_dtype` and `value_dtype`.

THE LAW IT OWNS, each stated once:

- `entmax_padded_width`: the width class, the next power of two from 2; the classes are the powers of two up to `kEntmaxMaxWidth` (256, the branch width limit), derived, not listed.
- the class's lane width and items a thread: `min(class, 32)` and `class / lane` (`entmax_reach`'s leaf), which is the pair `LW`, `IPT` of the kernels' template arguments.
- the level bound, `kEntmaxMaxLevels` (16, the level table's size; `entmax.cuh`'s `MAX_BATCHED_LEVELS` is this constant, which the table's arrays read).
- the alpha set: `entmax_alpha_code` maps exactly 1.5 to 15 and 2.0 to 20 and anything else to the code no context holds; alpha 1.5 is the arm's `alpha15` of 1, 2.0 its 0.
- the dtype codes (`kEntmaxFp32` 0, `kEntmaxBf16` 1); the host maps a tensor's dtype to its code (`dtype_code`) and the code back to the type inside the arm's body.
- the one-width-class rule, `entmax_class_refusal`: the levels of a launch share the class of the first, or the refusal names the level, its width and the class ("a batched solve must be one
  width class; level i has width w against class k"); a test pins the words.

`entmax_reach<K>` is the ARM half, a decision tree on `Truth` in the rule language ([`rules.md`](rules.md)): the levels within the bound, a ladder over the width's classes, a fork on alpha
where the kernel has one, a ladder over each dtype (the contracts hold bf16 alone, so each ladder reaches one rung over a contract's range; the arm key keeps both dtypes). It is exact over ranges ([`rules.md#exact`](rules.md#exact)), so the arm generator ([`arms_gen.md`](arms_gen.md)) asks it what a contract reaches; the arm
of a point is the same walk (`entmax_arm`). `entmax_launch<K>` is `{arm, klass, levels}` for a point: the arm, the padded width the launch's levels share (what `entmax_class_refusal`
holds every level to) and the level count (the grid's depth). `entmax_call<K>` states the call's facts in the form `describe` takes, so the fact names are written beside the reader.

THE GRID STAYS IN THE LAUNCH FUNCTIONS: `ceil(tokens / WORD_TOKENS)` reads the support word's 32 tokens, which the kernels read too (`entmax.cuh`'s `WORD_TOKENS`, FROZEN), so a matcher that
computed it would state the constant twice.

The arm has two forms because the softmax kernels have no alpha: `EntmaxAlphaArm` (`lw`, `ipt`, `alpha15`, `logit`, `value`, named as the kernels' template parameters are: 16 arms from 8 classes and 2 alphas, the dtypes bf16 alone) and
`EntmaxPlainArm` (`lw`, `ipt`, `logit`, `value`: 8). `EntmaxArmOf<K>` picks by the kernel's facts (`EntmaxHasAlpha<K>`: whether its reader has an alpha).

THE HOST USES IT as `decode.cu` uses decode's: an entry builds its kernel's facts from its own arguments, `plan::descriptor_of<K>` checks them against the contract (the refusal is the contract's,
naming the fact and the nearest context's range), `entmax_launch<K>` is the launch, and `arm_switch` over the family's generated arm set selects the instantiation. `tests/unit/test_entmax_matcher.py`
holds the arm of every width from 2, both alphas and bf16 planes for each kernel (through `plan_tool --kernel <kernel>`), each refusal, and the families' row counts.
