# The window W

A call's token axis is cut into windows of W tokens, global to every box and every kernel: the carry reads each
window's state out to the window's readers and folds its writes, the intra takes the pairs inside one window, the facts
pass emits each window's lists. W is not a data size. A window's resident rows
follow its read and write unions (the plan's chunks, multiples of 64 union slots); W sets how many tokens share one
window's fixed work -- the hand-off of the state into the image, the staging of the next window's inputs, the barriers.
A card with more shared memory amortizes that work over more tokens (ruled 2026-10-31: W and the other structural
constants are per-card configuration).

<a id="window-for"></a>
## Where W comes from

W is the matcher's: the prefill launch function derives it from the device's facts, as it derives the arm
(`prefill_window` in `csrc/fsla/src/common/plan/prefill_launch.h`, [`plan/prefill_launch.md`](plan/prefill_launch.md#window)). The owner's
ruling (2026-11-23): "why? shouldnt the matchers be mapping the hardware facts etc to the arm with the correct window. where else would we derive window".
It is no fact of the registry and no trait of a build. The device card reports it to Python (`device_card_op` in `fsla_api.cpp`,
`fsla.ops.window.window_of`) from the same function over the card's compute capability, so a host shape and the launch it describes cannot disagree. Python
reads no window to shape a call: the layer's segment plan is the planner's (`segment_plan`, [`plan/segments.md`](plan/segments.md)), which reads the card's
window through that same function.

The hopper (sm_90a, compute capability 900) is 1,024, provisional until the sm_90 carry's design picks it. A call is a whole number of the device's
windows (the intra's check), so a call is a multiple of 1,024 tokens until the intra takes a partial last window.
