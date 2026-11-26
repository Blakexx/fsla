# `segments.h`: the segments of a call

How a long prefill is cut into op calls. The op runs one segment and knows nothing of segmentation; the layer loops over the steps of a
PLAN carrying the state ([`../../../api.md`](../../../api.md) section 4.2). The plan is the planner's: it needs the card's window, a
hardware fact, and Python shapes no call from one. `csrc/fsla/src/common/plan/segments.h` is pure and `constexpr`, with no device; its
cases are proved at compile time (`segments_proof`).

## The plan

`segment_plan(length, maximum, window)` returns `SegmentPlan{steps, tokens}`: `steps` op calls of `tokens` tokens each, the last holding
the rest.

- A call that fits (`length <= maximum`, or `maximum == kNoMaximum`, 0, no limit) is ONE step of `length` tokens, a partial window
  included: the kernels run a partial last window and the caller pads nothing.
- A longer call is split on window boundaries only: tokens of one window interact, and the window grid is global, so a boundary on a
  window boundary moves no pair. `tokens` is the most whole windows under the maximum, `maximum / window * window`, and
  `steps = ceil(length / tokens)`.
- Where a split is needed and the maximum is under one window there is no plan: `steps == 0`. The owner (2026-11-26): "sure we can not
  allow such a case".

| length | maximum | window | plan |
|---|---|---|---|
| 300 | none | 1,024 | 1 step of 300 |
| 300 | 500 | 1,024 | 1 step of 300 |
| 10,000 | 4,000 | 1,024 | 4 steps of 3,072, the last 784 |
| 2,049 | 2,048 | 1,024 | 2 steps of 2,048, the last 1 |
| 10,000 | 500 | 1,024 | none |
| 800 | 500 | 1,024 | none |

## <a id="step"></a>A step's bytes

The memory limit is a PREFILL question -- what does one step of this call hold? -- so one function answers it, summing a step's parts,
each from the function that sizes the real thing (the owner, 2026-11-26, on counting a step's inputs: "why wouldnt it? thats also part
of the memory usage"; and on the form: "why isnt it like a prefill function that calls the actual same underlying thigns so drift is
impossilbe"). A step's shape is `StepShape`: its levels' widths as log2, each side's masked levels (a bit a level) and batch-heads (the
read side first), and the values' channels. `step_shape(widths, levels, masked, heads, value_width)` is the one place a shape is
made from a call's numbers and judged (`levels == 0` where a part is outside the op's: no level or more than four, a width that is no
power of two from 2 to 256, no batch-head on a side, no value channel); the library's entries and `plan_tool --step` both call it.

| part | function | what sizes the real thing |
|---|---|---|
| the op's workspace, at the most leaf groups the shape allows | `workspace_bytes(shape, windows, window)` | `prepared_layout` ([`mask_prepare_layout.md`](mask_prepare_layout.md)), the function `mask_prepare_bytes` allocates the workspace by: the same code |
| the projected operands | `operand_bytes(shape, windows, window)` | the op's operand formats: a route digit a byte and a scale an fp32 a level a side, a mask word an int32 of 32 tokens of one digit on each masked level-side, a value channel a bf16 and the values' scale an fp32 on the write side. PyTorch allocates these from the producer's formats, so this figure is a statement of them, tied to a real bundle's tensors by `tests/unit/test_segmentation.py` |

`step_bytes` is their sum and `step_tokens_within(shape, limit, window)` the most tokens, whole windows, under `limit` bytes:
`limit / step_bytes(shape, 1, window) * window`, 0 where one window does not fit. It is exact because every part is a window's bytes a
window (`segments_proof` asserts it). A part the prefill gains later -- the carry's workspace -- is a term added here. OUTSIDE THE SUM:
the state, the same however the call is cut, and the call's output, assembled whole.

## <a id="entries"></a>The entries

All three are the planner's (`environment.cu`, declared in `plan_api.cuh`), on the card `device`, whose window is `window_of_cc` of its
compute capability ([`levels.md`](levels.md)).

- `fsla_segment_plan(length, maximum, device)` returns `[steps, tokens]`. A call with no plan is refused, naming the window.
  `fsla.segment_plan(length, maximum, device=None)` is its Python face (`fsla/segmentation.py`); with no maximum it is one step and
  asks no card.
- `fsla_step_bytes(widths, masked, heads, value_width, tokens, device)` returns `[workspace, operands]` of a step of `tokens` tokens,
  in whole windows (`fsla.segmentation.step_bytes`).
- `fsla_step_tokens(widths, masked, heads, value_width, limit, device)` returns the most tokens of a step under `limit` bytes; where
  one window does not fit it is refused, naming the window's bytes and both parts. `fsla.max_memory(bytes)` is the handle that asks it.
- `plan_tool --segments length,maximum,window;...` and `plan_tool --step widths/masked/heads/value_width/windows/window`
  ([`plan_tool.md`](plan_tool.md)) print the plan and a step's bytes as JSON, with no device: `tests/unit/test_segmentation.py` runs
  the cases above and a sweep through the first, and through the second checks the operand bytes against the tensors a producer makes;
  on a card it checks the library's plan, and the workspace against `mask_prepare_bytes`.
